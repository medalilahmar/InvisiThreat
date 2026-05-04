import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from jira_service import jira_service, JiraService
from pydantic import BaseModel
from pydantic import BaseModel, Field, field_validator
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

try:
    from llm_service import explain_finding as _llm_explain
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    def _llm_explain(*args, **kwargs):
        return None

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("invisithreat.api")


API_VERSION = "3.2.0"
APP_START   = datetime.now(timezone.utc)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline_latest.pkl"))
META_PATH  = Path(os.getenv("META_PATH",  "models/pipeline_latest_meta.json"))

DEFECTDOJO_URL     = os.getenv("DEFECTDOJO_URL", "http://localhost:8080")
DEFECTDOJO_API_KEY = os.getenv("DEFECTDOJO_API_KEY")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   "60"))

CACHE_TTL_PRODUCTS    = 300
CACHE_TTL_TESTS       = 300
CACHE_TTL_FINDINGS    = 60
SCORES_CACHE_TTL      = 60

CSV_FINDINGS_PATH = Path(os.getenv("CSV_FINDINGS_PATH", "data/processed/findings_clean.csv"))

RISK_LEVELS: Dict[int, str]  = {0: "low", 1: "medium", 2: "high", 3: "critical"}
CLASS_LABELS: Dict[int, str] = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
CLASS_COLORS: Dict[str, str] = {
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE_COLS — IDENTIQUE à train.py FEATURE_COLS
# NE PAS MODIFIER sans réentraîner le modèle
# Colonnes interdites supprimées : severity_num, exploit_risk, days_open_high,
# epss_score_norm, score_composite_raw, score_composite_adj,
# cvss_severity_gap, severity_x_active, cvss_x_severity, severity_x_urgent
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS: List[str] = [
    "cvss_score",
    "cvss_score_norm",
    "has_cve",
    "has_cwe",
    "epss_score",
    "epss_percentile",
    "has_high_epss",
    "epss_x_cvss",
    "age_days",
    "age_days_norm",
    "delay_norm",
    "tag_urgent",
    "tag_in_production",
    "tag_sensitive",
    "tag_external",
    "tags_count",
    "tags_count_norm",
    "context_score",
    "exposure_norm",
    "product_fp_rate",
    "cvss_x_has_cve",
    "age_x_cvss",
]


# ══════════════════════════════════════════════════════════════════════════════
# Jira models
# ══════════════════════════════════════════════════════════════════════════════

class JiraCreateIssueRequest(BaseModel):
    finding_id: int
    title: str
    severity: str
    cvss_score: float
    description: Optional[str] = None
    cve: Optional[str] = None
    cwe: Optional[str] = None
    file_path: Optional[str] = None
    line: Optional[int] = None
    tags: Optional[List[str]] = None
    risk_level: Optional[str] = None
    ai_score: Optional[float] = None
    ai_confidence: Optional[float] = None
    engagement_id: Optional[int] = None
    product_name: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "finding_id": 42,
                "title": "SQL Injection in login form",
                "severity": "CRITICAL",
                "cvss_score": 9.8,
            }
        }


class JiraIssueResponse(BaseModel):
    key: str
    id: str
    self: str
    url: Optional[str] = None
    already_exists: bool = False
    message: Optional[str] = None


class JiraHealthResponse(BaseModel):
    status: str
    jira_server: Optional[str] = None
    project_key: Optional[str] = None
    connected: bool
    message: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════════

_cache_store: Dict[str, Tuple[Any, float]] = {}
_scores_cache_memory: Dict[str, Any]       = {}
_scores_cache_loaded_at: float             = 0.0


def get_cached_or_fetch(cache_key: str, fetch_func, ttl: int = CACHE_TTL_PRODUCTS) -> Any:
    now   = time.time()
    entry = _cache_store.get(cache_key)
    if entry is not None:
        data, ts = entry
        if now - ts < ttl:
            return data
    data = fetch_func()
    _cache_store[cache_key] = (data, now)
    return data


def get_scores_cache() -> Dict[str, Any]:
    global _scores_cache_memory, _scores_cache_loaded_at
    now = time.time()
    if now - _scores_cache_loaded_at < SCORES_CACHE_TTL and _scores_cache_memory:
        return _scores_cache_memory
    cache_file = Path("data/ai_scores_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                _scores_cache_memory = json.load(f)
            _scores_cache_loaded_at = now
            logger.debug(f"Scores cache rechargé : {len(_scores_cache_memory)} entrées")
        except Exception as e:
            logger.warning(f"Impossible de recharger le cache scores : {e}")
    return _scores_cache_memory


def invalidate_cache(prefix: str = "") -> int:
    keys = [k for k in list(_cache_store.keys()) if k.startswith(prefix)]
    for k in keys:
        del _cache_store[k]
    return len(keys)


# ══════════════════════════════════════════════════════════════════════════════
# LocalDataLoader
# ══════════════════════════════════════════════════════════════════════════════

class LocalDataLoader:

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df_findings: Optional[pd.DataFrame] = None
        self.products: Dict[int, Dict] = {}
        self.engagements: Dict[int, Dict] = {}
        self.tests: Dict[int, Dict] = {}
        self.findings_by_id: Dict[int, Dict] = {}
        self._loaded_at: Optional[datetime] = None
        self._ready = False

    def load(self) -> bool:
        if not self.csv_path.exists():
            logger.warning(f"CSV introuvable : {self.csv_path}")
            return False
        try:
            self.df_findings = pd.read_csv(self.csv_path, low_memory=False)
            logger.info(f"[LocalDataLoader] CSV chargé : {len(self.df_findings)} findings, {len(self.df_findings.columns)} colonnes")
            self._build_hierarchy()
            self._loaded_at = datetime.now(timezone.utc)
            self._ready = True
            logger.info(
                f"[LocalDataLoader] Hiérarchie construite : "
                f"{len(self.products)} produits, "
                f"{len(self.engagements)} engagements"
            )
            return True
        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur de chargement : {e}")
            return False

    def _build_hierarchy(self) -> None:
        self.products.clear()
        self.engagements.clear()
        self.tests.clear()
        self.findings_by_id.clear()

        if self.df_findings is None:
            return

        try:
            required_cols = ['id', 'product_id', 'engagement_id', 'product_name', 'engagement_name', 'severity_num']
            missing_cols = [col for col in required_cols if col not in self.df_findings.columns]
            if missing_cols:
                logger.error(f"[LocalDataLoader] Colonnes manquantes : {missing_cols}")
                raise KeyError(f"Missing columns: {missing_cols}")

            for _, row in self.df_findings[['product_id', 'product_name']].drop_duplicates().iterrows():
                prod_id = int(row['product_id'])
                self.products[prod_id] = {
                    'id': prod_id,
                    'name': row['product_name'],
                    'engagements': set(),
                }

            for _, row in self.df_findings[['product_id', 'engagement_id', 'engagement_name']].drop_duplicates().iterrows():
                eng_id  = int(row['engagement_id'])
                prod_id = int(row['product_id'])
                self.engagements[eng_id] = {
                    'id': eng_id,
                    'name': row['engagement_name'],
                    'product_id': prod_id,
                    'tests': set(),
                }
                if prod_id in self.products:
                    self.products[prod_id]['engagements'].add(eng_id)

            SEVERITY_MAP_NUM = {0: 'info', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}

            for _, row in self.df_findings.iterrows():
                finding_id = int(row['id'])
                eng_id     = int(row['engagement_id'])
                prod_id    = int(row['product_id'])

                finding_dict = row.to_dict()
                finding_dict['id']            = finding_id
                finding_dict['product_id']    = prod_id
                finding_dict['engagement_id'] = eng_id

                sev_num = row['severity_num']
                if pd.isna(sev_num):
                    severity = 'info'
                else:
                    try:
                        severity = SEVERITY_MAP_NUM.get(int(sev_num), 'info')
                    except (ValueError, TypeError):
                        severity = 'info'
                finding_dict['severity'] = severity

                tags_list = []
                if row.get('tag_urgent') == 1:        tags_list.append('urgent')
                if row.get('tag_in_production') == 1: tags_list.append('production')
                if row.get('tag_sensitive') == 1:     tags_list.append('sensitive')
                if row.get('tag_external') == 1:      tags_list.append('external')
                finding_dict['tags'] = tags_list

                self.findings_by_id[finding_id] = finding_dict

                if eng_id in self.engagements:
                    if 'findings' not in self.engagements[eng_id]:
                        self.engagements[eng_id]['findings'] = []
                    self.engagements[eng_id]['findings'].append(finding_id)

        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur _build_hierarchy : {e}")
            self._ready = False
            raise

    def get_findings_for_product(self, product_id: int) -> List[Dict]:
        if not self._ready or product_id not in self.products:
            return []
        results = []
        for eng_id in self.products[product_id].get('engagements', set()):
            if eng_id in self.engagements:
                for fid in self.engagements[eng_id].get('findings', []):
                    if fid in self.findings_by_id:
                        results.append(self.findings_by_id[fid])
        return results

    def get_findings_for_engagement(self, engagement_id: int) -> List[Dict]:
        if not self._ready or engagement_id not in self.engagements:
            return []
        return [
            self.findings_by_id[fid]
            for fid in self.engagements[engagement_id].get('findings', [])
            if fid in self.findings_by_id
        ]

    def get_all_findings(self) -> List[Dict]:
        return list(self.findings_by_id.values()) if self._ready else []

    def get_products(self) -> List[Dict]:
        return [
            {'id': p['id'], 'name': p['name'], 'engagement_count': len(p.get('engagements', []))}
            for p in self.products.values()
        ]

    def get_engagements_for_product(self, product_id: int) -> List[Dict]:
        if product_id not in self.products:
            return []
        results = []
        for eng_id in self.products[product_id].get('engagements', set()):
            if eng_id in self.engagements:
                eng = self.engagements[eng_id]
                results.append({'id': eng['id'], 'name': eng['name'], 'product_id': product_id})
        return results

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at


# ══════════════════════════════════════════════════════════════════════════════
# FindingInput — to_features() CORRIGÉ : 22 features exactes de train.py
# ══════════════════════════════════════════════════════════════════════════════

class FindingInput(BaseModel):
    severity:        str           = Field(..., description="critical, high, medium, low, info")
    cvss_score:      float         = Field(0.0, ge=0.0, le=10.0)
    title:           str           = ""
    description:     str           = ""
    file_path:       str           = ""
    tags:            List[str]     = []
    finding_id:      Optional[int] = None
    engagement_id:   Optional[int] = None
    product_id:      Optional[int] = None
    days_open:       int           = 0
    duplicate_count: int           = 0
    epss_score:      float         = 0.0
    epss_percentile: float         = 0.0
    has_cve:         int           = 0
    has_cwe:         int           = 0

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = ["critical", "high", "medium", "low", "info", "informational"]
        v_lower = v.lower().strip()
        if v_lower not in valid:
            raise ValueError(f"Severity doit être l'un de : {valid}")
        return v_lower

    @field_validator("cvss_score")
    @classmethod
    def round_cvss(cls, v: float) -> float:
        return round(v, 1)

    def to_features(self) -> Dict[str, Any]:
        """
        Génère exactement les 22 features de FEATURE_COLS (= train.py).
        NE contient PAS severity_num, exploit_risk, days_open_high,
        epss_score_norm, score_composite_*, cvss_severity_gap, etc.
        Ces colonnes sont dans EXCLUDE_COLS de train.py → jamais vues par le modèle.
        """
        tags_lower = [t.lower() for t in self.tags]

        tag_urgent        = 1 if any(t in tags_lower for t in ["urgent", "blocker", "p0", "p1"]) else 0
        tag_in_production = 1 if any(t in tags_lower for t in ["production", "prod", "prd", "live"]) else 0
        tag_sensitive     = 1 if any(t in tags_lower for t in ["sensitive", "pii", "gdpr", "confidential"]) else 0
        tag_external      = 1 if any(t in tags_lower for t in ["external", "internet-facing", "public", "exposed"]) else 0

        tags_count      = len(tags_lower)
        tags_count_norm = round(min(tags_count / 20.0, 1.0), 4)

        cvss_score_norm = round(min(self.cvss_score / 10.0, 1.0), 4)
        age_days        = self.days_open
        age_days_norm   = round(min(age_days / 365.0, 1.0), 4)

        epss_x_cvss   = round(self.cvss_score * self.epss_score, 4)
        has_high_epss = 1 if self.epss_score > 0.5 else 0

        cvss_x_has_cve = round(self.cvss_score * self.has_cve, 4)
        age_x_cvss     = round(age_days * self.cvss_score, 3)

        # context_score : score métier basé sur les tags (max 5 — preprocess.py étape 7)
        context_score = min(tag_in_production * 2 + tag_external * 2 + tag_sensitive * 1, 5)

        # exposure_norm = context_score / 5  (preprocess.py étape 10)
        exposure_norm = round(context_score / 5.0, 4)

        # delay_norm : urgence opérationnelle (preprocess.py étape 10)
        # 1 - (days_clipped / (days_clipped + 30)) → proche de 1 = trouvé récemment
        days_clipped = min(age_days, 365)
        delay_norm   = round(1.0 - (days_clipped / (days_clipped + 30)), 4)

        return {
            "cvss_score":        self.cvss_score,
            "cvss_score_norm":   cvss_score_norm,
            "has_cve":           self.has_cve,
            "has_cwe":           self.has_cwe,
            "epss_score":        self.epss_score,
            "epss_percentile":   self.epss_percentile,
            "has_high_epss":     has_high_epss,
            "epss_x_cvss":       epss_x_cvss,
            "age_days":          age_days,
            "age_days_norm":     age_days_norm,
            "delay_norm":        delay_norm,
            "tag_urgent":        tag_urgent,
            "tag_in_production": tag_in_production,
            "tag_sensitive":     tag_sensitive,
            "tag_external":      tag_external,
            "tags_count":        tags_count,
            "tags_count_norm":   tags_count_norm,
            "context_score":     context_score,
            "exposure_norm":     exposure_norm,
            "product_fp_rate":   0.0,
            "cvss_x_has_cve":    cvss_x_has_cve,
            "age_x_cvss":        age_x_cvss,
        }


class BatchInput(BaseModel):
    findings:      List[FindingInput] = Field(..., min_length=1, max_length=500)
    engagement_id: Optional[int]      = None


# ══════════════════════════════════════════════════════════════════════════════
# Response models
# ══════════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    request_id:    str
    finding_id:    Optional[int]
    engagement_id: Optional[int]
    product_id:    Optional[int]
    risk_class:    int
    risk_level:    str
    risk_color:    str
    risk_score:    float
    confidence:    float
    context_score: int
    cvss_score:    float          = 0.0
    cve:           Optional[str]  = None
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    predicted_at:  str


class BatchPredictionResponse(BaseModel):
    request_id:   str
    total:        int
    success:      int
    errors_count: int
    results:      List[PredictionResponse]
    errors:       List[Dict[str, Any]]
    summary:      Dict[str, int]
    processed_at: str


class HealthResponse(BaseModel):
    status:         str
    api_version:    str
    model_version:  str
    model_ready:    bool
    n_classes:      int
    n_features:     int
    uptime_seconds: float
    loaded_at:      Optional[str]


class ProductResponse(BaseModel):
    id:             int
    name:           str
    description:    Optional[str] = None
    created:        Optional[str] = None
    findings_count: Optional[int] = None


class EngagementResponse(BaseModel):
    id:             int
    name:           str
    product:        int
    product_name:   Optional[str] = None
    status:         str           = "active"
    created:        Optional[str] = None
    findings_count: Optional[int] = None


class FindingSummaryResponse(BaseModel):
    id:              int
    title:           str
    severity:        str
    cvss_score:      float
    cve:             Optional[str]   = None
    tags:            List[str]       = []
    test_id:         Optional[int]   = None
    engagement_id:   Optional[int]   = None
    engagement_name: Optional[str]   = None
    product_id:      Optional[int]   = None
    product_name:    Optional[str]   = None
    created:         Optional[str]   = None
    age_days:        Optional[int]   = None
    file_path:       Optional[str]   = None
    line:            Optional[int]   = None
    has_cve:         Optional[int]   = None
    description:     Optional[str]   = ""
    # ── Champs IA (remplis par le vrai modèle au démarrage) ──
    risk_class:      Optional[int]   = None
    risk_level:      Optional[str]   = None
    risk_color:      Optional[str]   = None
    ai_confidence:   Optional[float] = None
    ai_probabilities: Optional[Dict[str, float]] = None
    context_score:   Optional[int]   = None
    exposure_norm:   Optional[float] = None
    ai_risk_score_cont: Optional[float] = None   
    model_base_score:   Optional[float] = None   
    business_nudge:     Optional[float] = None   
    shap_features:      Optional[List[Dict[str, Any]]] = None  


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_age_days(created_str: Optional[str]) -> Optional[int]:
    if not created_str:
        return None
    try:
        d = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - d).days
    except Exception:
        return None


def _parse_tags(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def safe_str(val: Any, default: str = "") -> str:
    if val is None or val == "":
        return default
    try:
        if isinstance(val, float) and np.isnan(val):
            return default
        return str(val).strip()
    except (ValueError, TypeError):
        return default


def safe_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        if isinstance(val, float) and np.isnan(val):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        f_val = float(val)
        return default if np.isnan(f_val) else f_val
    except (ValueError, TypeError):
        return default


def _finding_to_response(f: Dict, scores_cache: Dict) -> FindingSummaryResponse:
    """
    Construit la réponse finding enrichie avec les VRAIS scores IA
    provenant du cache rempli par _score_all_findings_at_startup().
    """
    fid        = safe_int(f.get("id")) or 0
    score_data = scores_cache.get(str(fid), {})

    SEVERITY_MAP_NUM = {0: 'info', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
    VALID_SEVERITIES = ["critical", "high", "medium", "low", "info"]

    severity_raw = f.get("severity")
    cve_value    = safe_str(f.get("cve"))

    if severity_raw is None or (not isinstance(severity_raw, str) and pd.isna(severity_raw)):
        sev_num = f.get("severity_num")
        severity_raw = SEVERITY_MAP_NUM.get(int(sev_num), 'info') if sev_num and not pd.isna(sev_num) else 'info'

    if severity_raw is None or (isinstance(severity_raw, float) and np.isnan(severity_raw)):
        severity = "info"
    else:
        severity = str(severity_raw).strip().lower()
        if severity not in VALID_SEVERITIES:
            severity = "info"

    # Scores IA depuis le cache (rempli par le modèle au démarrage)
    ai_risk_class      = safe_int(score_data.get("ai_risk_score"))
    ai_risk_level      = safe_str(score_data.get("ai_risk_level"))
    ai_risk_color      = safe_str(score_data.get("ai_risk_color"))
    ai_confidence      = safe_float(score_data.get("ai_confidence"), 0.0) or None
    ai_probabilities   = score_data.get("ai_probabilities") or None
    ai_context_score   = safe_int(score_data.get("context_score"))
    ai_exposure_norm   = safe_float(score_data.get("exposure_norm"), 0.0) or None



    cont_score     = safe_float(score_data.get("ai_risk_score"))   # le score continu
    model_base     = safe_float(score_data.get("model_base_score"))
    nudge          = safe_float(score_data.get("business_nudge"))
    shap_feats     = score_data.get("shap_top_features") or None

    return FindingSummaryResponse(
        id              = fid,
        title           = safe_str(f.get("title"), "Unknown"),
        severity        = severity,
        cvss_score      = safe_float(f.get("cvss_score"), 0.0),
        tags            = _parse_tags(f.get("tags", [])),
        test_id         = safe_int(f.get("test_id")),
        engagement_id   = safe_int(f.get("engagement_id")),
        engagement_name = safe_str(f.get("engagement_name")),
        product_id      = safe_int(f.get("product_id")),
        product_name    = safe_str(f.get("product_name")),
        created         = safe_str(f.get("created")),
        age_days        = _compute_age_days(safe_str(f.get("created"))),
        file_path       = safe_str(f.get("file_path")),
        line            = safe_int(f.get("line")),
        has_cve         = 1 if f.get("cve") else 0,
        cve             = cve_value,
        description     = safe_str(f.get("description"), ""),
        risk_class      = ai_risk_class,
        risk_level      = ai_risk_level if ai_risk_level else None,
        risk_color      = ai_risk_color if ai_risk_color else None,
        ai_confidence   = ai_confidence,
        ai_probabilities= ai_probabilities,
        context_score   = ai_context_score,
        exposure_norm   = ai_exposure_norm,
        ai_risk_score_cont = cont_score,
        model_base_score   = model_base,
        business_nudge     = nudge,
        shap_features      = shap_feats,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ModelManager
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    def __init__(self, model_path: Path, meta_path: Path):
        self.model_path         = model_path
        self.meta_path          = meta_path
        self._model             = None
        self._metadata:         Dict[str, Any]     = {}
        self._loaded_at:        Optional[datetime] = None
        self._feature_columns:  List[str]          = FEATURE_COLS.copy()
        self._prediction_cache: Dict[str, Tuple]   = {}
        self._cache_max_size    = 100
        self._cache_hits        = 0
        self._cache_misses      = 0

    def is_ready(self) -> bool:
        return self._model is not None

    def get_model(self):
        return self._model

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    def predict_batch_cached(self, X: pd.DataFrame) -> Tuple[Any, Any]:
        import hashlib
        h = hashlib.md5(X.to_csv(index=False).encode()).hexdigest()
        if h in self._prediction_cache:
            self._cache_hits += 1
            result = self._prediction_cache.pop(h)
            self._prediction_cache[h] = result
            return result
        self._cache_misses += 1
        if len(self._prediction_cache) >= self._cache_max_size:
            del self._prediction_cache[next(iter(self._prediction_cache))]
        classes = self._model.predict(X).astype(int)
        probas  = self._model.predict_proba(X)
        self._prediction_cache[h] = (classes, probas)
        return classes, probas

    def load_model(self) -> bool:
        if not self.model_path.exists():
            logger.warning(f"Modèle introuvable : {self.model_path}")
            return False
        try:
            self._model     = joblib.load(self.model_path)
            self._loaded_at = datetime.now(timezone.utc)
            if self.meta_path.exists():
                with open(self.meta_path) as f:
                    self._metadata = json.load(f)
            # Priorité : feature_names_in_ > metadata > FEATURE_COLS
            if hasattr(self._model, "feature_names_in_"):
                self._feature_columns = list(self._model.feature_names_in_)
                logger.info(f"Features chargées depuis le modèle : {len(self._feature_columns)} colonnes")
            elif self._metadata.get("feature_columns"):
                self._feature_columns = self._metadata["feature_columns"]
                logger.info(f"Features chargées depuis metadata : {len(self._feature_columns)} colonnes")
            else:
                logger.warning("Features chargées depuis FEATURE_COLS (fallback)")
            logger.info(
                f"Modèle chargé — version={self._metadata.get('version','?')} "
                f"F1={self._metadata.get('metrics', {}).get('test_f1_weighted', 'N/A')}"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            self._model = None
            return False

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns

    @property
    def n_classes(self) -> int:
        return len(self._model.classes_) if self._model and hasattr(self._model, "classes_") else 4

    @property
    def classes(self) -> List[int]:
        return [int(c) for c in self._model.classes_] if self._model and hasattr(self._model, "classes_") else [0, 1, 2, 3]

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def model_version(self) -> str:
        return self._metadata.get("version", self._metadata.get("timestamp", "unknown"))


# ══════════════════════════════════════════════════════════════════════════════
# _score_all_findings_at_startup — scoring IA réel au démarrage
# Lit df_findings (CSV déjà en mémoire), aligne les 22 colonnes,
# appelle model.predict() + predict_proba() en batch sur tous les findings,
# écrit data/ai_scores_cache.json et met à jour _scores_cache_memory.
# ══════════════════════════════════════════════════════════════════════════════

def _score_all_findings_at_startup(
    loader: LocalDataLoader,
    manager: ModelManager,
) -> None:
    if loader.df_findings is None or loader.df_findings.empty:
        logger.warning("_score_all_findings_at_startup : DataFrame vide")
        return

    expected_cols = manager.feature_columns
    df = loader.df_findings.copy()

    # Colonnes manquantes → remplir à 0 (jamais dû arriver si CSV correct)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.warning(f"Colonnes manquantes dans CSV (remplies à 0) : {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[expected_cols].fillna(0).copy()
    X.columns = pd.Index(expected_cols)  # noms explicites → supprime warning sklearn

    n = len(X)
    logger.info(f"Scoring IA de {n} findings en batch…")

    try:
        predictions = manager.get_model().predict(X).astype(int)
        probas      = manager.get_model().predict_proba(X)
    except Exception as e:
        logger.error(f"Erreur scoring batch : {e}")
        raise

    model_classes   = manager.classes
    n_classes       = len(model_classes)

    # Distribution pour diagnostic
    from collections import Counter
    dist = Counter(int(p) for p in predictions)
    dist_labels = {CLASS_LABELS.get(k, str(k)): v for k, v in dist.items()}
    logger.info(f"Distribution IA : {dist_labels}")

    cache: Dict[str, Any] = {}
    now = datetime.now(timezone.utc).isoformat()
    all_findings = loader.get_all_findings()

    for i, finding in enumerate(all_findings):
        fid        = str(finding.get("id", i))
        pred       = int(predictions[i])
        proba_row  = probas[i].tolist()
        confidence = float(max(proba_row))
        label      = CLASS_LABELS.get(pred, f"class_{pred}")
        color      = CLASS_COLORS.get(label, "#888888")

        proba_dict = {
            CLASS_LABELS.get(c, str(c)): round(float(p), 4)
            for c, p in zip(model_classes, proba_row)
        }

        ctx_score = float(X["context_score"].iloc[i])   if "context_score"  in X.columns else 0.0
        exp_norm  = float(X["exposure_norm"].iloc[i])   if "exposure_norm"   in X.columns else 0.0
        dly_norm  = float(X["delay_norm"].iloc[i])      if "delay_norm"      in X.columns else 0.0

        cache[fid] = {
            "ai_risk_score":    pred,
            "ai_risk_level":    label,
            "ai_risk_color":    color,
            "ai_confidence":    round(confidence, 4),
            "ai_probabilities": proba_dict,
            "context_score":    int(round(ctx_score)),
            "exposure_norm":    round(exp_norm, 4),
            "delay_norm":       round(dly_norm, 4),
            "updated_at":       now,
        }

    # Sauvegarde atomique
    cache_file = Path("data/ai_scores_cache.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(cache_file)

    logger.info(f"✅ Cache IA sauvegardé : {len(cache)} findings → {cache_file}")

    # Mise à jour immédiate du cache mémoire (évite relecture fichier)
    global _scores_cache_memory, _scores_cache_loaded_at
    _scores_cache_memory    = cache
    _scores_cache_loaded_at = time.time()


# ══════════════════════════════════════════════════════════════════════════════
# Globals
# ══════════════════════════════════════════════════════════════════════════════

model_manager:     ModelManager               = ModelManager(MODEL_PATH, META_PATH)
shap_explainer                                = None
local_data_loader: Optional[LocalDataLoader] = None
_rate_limit_store: Dict[str, List[float]]     = defaultdict(list)


# ══════════════════════════════════════════════════════════════════════════════
# SHAP helper — extraction robuste du modèle de base
# Supporte : Pipeline, CalibratedClassifierCV, StackingClassifier
# ══════════════════════════════════════════════════════════════════════════════

def _extract_base_model_for_shap(model):
    """
    Extrait le modèle de base compatible TreeExplainer depuis n'importe
    quelle enveloppe sklearn (Pipeline, CalibratedClassifierCV, Stacking).
    """
    # Pipeline sklearn
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return _extract_base_model_for_shap(steps[-1])

    # CalibratedClassifierCV (sklearn >= 1.2 : .estimator ; ancien : .base_estimator)
    if hasattr(model, "calibrated_classifiers_"):
        cc   = model.calibrated_classifiers_[0]
        base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if base is not None:
            return _extract_base_model_for_shap(base)

    # StackingClassifier → premier estimateur de base
    if hasattr(model, "estimators_"):
        first = model.estimators_[0]
        if isinstance(first, tuple):
            first = first[1]
        return _extract_base_model_for_shap(first)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_data_loader, shap_explainer

    logger.info(f"Démarrage InvisiThreat AI Risk Engine API v{API_VERSION}")

    # 1. Chargement du modèle
    if model_manager.load_model():
        logger.info("✅ Modèle ML chargé")

        # 2. SHAP — utilise _extract_base_model_for_shap pour CalibratedClassifierCV
        try:
            import shap

            base_for_shap = _extract_base_model_for_shap(model_manager.get_model())
            logger.info(f"SHAP — modèle de base extrait : {type(base_for_shap).__name__}")

            class SimpleShapExplainer:
                def __init__(self, mdl):
                    self._exp   = shap.TreeExplainer(mdl)
                    self._ready = True

                def is_ready(self) -> bool:
                    return self._ready

                def explain(self, X: pd.DataFrame, pred_class: int) -> Dict[str, Any]:
                    try:
                        sv   = self._exp.shap_values(X)
                        sv_c = sv[pred_class][0] if isinstance(sv, list) else sv[0]
                        top  = sorted(range(len(sv_c)), key=lambda i: abs(sv_c[i]), reverse=True)[:10]
                        ev   = self._exp.expected_value
                        base = float(ev[pred_class] if isinstance(ev, (list, np.ndarray)) else ev)
                        return {
                            "top_features": [
                                {
                                    "feature":       X.columns[i],
                                    "shap_value":    round(float(sv_c[i]), 4),
                                    "feature_value": round(float(X.iloc[0, i]), 4),
                                    "direction":     "+" if sv_c[i] > 0 else "-",
                                }
                                for i in top
                            ],
                            "base_value": round(base, 4),
                        }
                    except Exception as e:
                        logger.warning(f"SHAP explain échoué : {e}")
                        return {"top_features": [], "base_value": 0}

            shap_explainer = SimpleShapExplainer(base_for_shap)
            logger.info("✅ SHAP explainer chargé")
        except ImportError:
            logger.warning("⚠️  SHAP non installé — explications désactivées")
        except Exception as e:
            logger.warning(f"⚠️  SHAP explainer échoué : {e}")
    else:
        logger.warning("⚠️  Modèle non chargé — prédictions retourneront 503")

    # 3. Chargement des données CSV
    try:
        local_data_loader = LocalDataLoader(CSV_FINDINGS_PATH)
        if local_data_loader.load():
            logger.info(
                f"✅ LocalDataLoader initialisé : "
                f"{len(local_data_loader.products)} produits, "
                f"{len(local_data_loader.engagements)} engagements, "
                f"{len(local_data_loader.findings_by_id)} findings"
            )

            # 4. SCORING IA AU DÉMARRAGE ────────────────────────────────────
            # Score TOUS les findings avec le vrai modèle en un seul batch.
            # Remplit data/ai_scores_cache.json ET _scores_cache_memory.
            # Tous les endpoints /defectdojo/findings utiliseront ces scores réels.
            logger.info("Utilisation du cache IA existant (data/ai_scores_cache.json)")

            # ─────────────────────────────────────────────────────────────────
        else:
            logger.warning("⚠️  LocalDataLoader : CSV non accessible, mode dégradé")
    except Exception as e:
        logger.error(f"❌ LocalDataLoader échoué : {e}")

    yield
    logger.info("API arrêtée")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "InvisiThreat AI Risk Engine",
    description = "API de scoring IA pour les vulnérabilités de sécurité",
    version     = API_VERSION,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


@app.get("/debug/severity-check", tags=["Debug"])
async def debug_severity_check() -> Dict[str, Any]:
    loader       = _require_local_loader()
    scores_cache = get_scores_cache()
    findings     = loader.get_all_findings()[:10]
    results = []
    for f in findings:
        response = _finding_to_response(f, scores_cache)
        results.append({
            "id":                    f.get("id"),
            "severity_in_response":  response.severity,
            "ai_risk_level":         response.risk_level,
            "ai_confidence":         response.ai_confidence,
            "ai_probabilities":      response.ai_probabilities,
        })
    dist = {}
    for v in scores_cache.values():
        lv = v.get("ai_risk_level", "N/A")
        dist[lv] = dist.get(lv, 0) + 1
    return {
        "total_findings":  len(loader.get_all_findings()),
        "scores_in_cache": len(scores_cache),
        "distribution_ai": dist,
        "sample_findings": results,
    }


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start     = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now       = time.time()
    _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(status_code=429, content={"detail": "Rate limit dépassé", "request_id": request_id})
    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"]      = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"
    logger.info(f"[{request_id}] {request.method} {request.url.path} → {response.status_code} ({duration:.1f}ms)")
    return response


def _require_local_loader() -> LocalDataLoader:
    if local_data_loader is None or not local_data_loader.is_ready:
        raise HTTPException(status_code=503, detail="LocalDataLoader non prêt.")
    return local_data_loader


# ── Info ──────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    return {
        "service":           "InvisiThreat AI Risk Engine",
        "version":           API_VERSION,
        "status":            "running",
        "model_ready":       model_manager.is_ready(),
        "data_loader_ready": local_data_loader is not None and local_data_loader.is_ready,
        "docs":              "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check() -> HealthResponse:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()
    return HealthResponse(
        status         = "healthy",
        api_version    = API_VERSION,
        model_version  = model_manager.model_version,
        model_ready    = True,
        n_classes      = model_manager.n_classes,
        n_features     = len(model_manager.feature_columns),
        uptime_seconds = round(uptime, 1),
        loaded_at      = model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Dict[str, Any]:
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()
    ld = local_data_loader
    scores_dist: Dict[str, int] = {}
    for v in _scores_cache_memory.values():
        lv = v.get("ai_risk_level", "N/A")
        scores_dist[lv] = scores_dist.get(lv, 0) + 1
    return {
        "invisithreat_api_version":             API_VERSION,
        "invisithreat_model_ready":             1 if model_manager.is_ready() else 0,
        "invisithreat_model_f1_score":          model_manager.get_metadata().get("metrics", {}).get("test_f1_weighted", 0),
        "invisithreat_uptime_seconds":          uptime,
        "invisithreat_n_classes":               model_manager.n_classes,
        "invisithreat_n_features":              len(model_manager.feature_columns),
        "invisithreat_cache_entries":           len(_cache_store),
        "invisithreat_prediction_cache_hits":   model_manager._cache_hits,
        "invisithreat_prediction_cache_misses": model_manager._cache_misses,
        "invisithreat_scores_cache_entries":    len(_scores_cache_memory),
        "invisithreat_ai_distribution":         scores_dist,
        "invisithreat_data_loader_products":    len(ld.products)     if ld and ld.is_ready else 0,
        "invisithreat_data_loader_engagements": len(ld.engagements)  if ld and ld.is_ready else 0,
        "invisithreat_data_loader_findings":    len(ld.findings_by_id) if ld and ld.is_ready else 0,
    }


@app.get("/model/info", tags=["Model"])
async def model_info() -> Dict[str, Any]:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "model_version":   model_manager.model_version,
        "n_classes":       model_manager.n_classes,
        "classes":         model_manager.classes,
        "class_labels":    CLASS_LABELS,
        "n_features":      len(model_manager.feature_columns),
        "feature_columns": model_manager.feature_columns,
        "loaded_at":       model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
        "metrics":         model_manager.get_metadata().get("metrics", {}),
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model() -> Dict[str, Any]:
    if model_manager.load_model():
        # Re-score tout après rechargement du modèle
        if local_data_loader and local_data_loader.is_ready:
            try:
                _score_all_findings_at_startup(local_data_loader, model_manager)
            except Exception as e:
                logger.warning(f"Re-scoring après reload échoué : {e}")
        return {
            "status":        "reloaded",
            "model_version": model_manager.model_version,
            "loaded_at":     model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
        }
    raise HTTPException(status_code=500, detail="Rechargement du modèle échoué")


# ── DefectDojo endpoints ──────────────────────────────────────────────────────

@app.get("/defectdojo/products", response_model=List[ProductResponse], tags=["DefectDojo"])
async def get_products() -> List[ProductResponse]:
    loader = _require_local_loader()
    return [ProductResponse(id=p['id'], name=p['name']) for p in loader.get_products()]


@app.get("/defectdojo/engagements", response_model=List[EngagementResponse], tags=["DefectDojo"])
async def get_engagements(product_id: Optional[int] = None) -> List[EngagementResponse]:
    loader  = _require_local_loader()
    results = []
    if product_id is not None:
        if product_id not in loader.products:
            return results
        for e in loader.get_engagements_for_product(product_id):
            results.append(EngagementResponse(
                id           = safe_int(e['id']) or 0,
                name         = safe_str(e['name'], f"engagement-{e['id']}"),
                product      = safe_int(e['product_id']) or 0,
                product_name = safe_str(loader.products.get(e['product_id'], {}).get('name', 'Unknown')),
            ))
    else:
        for eng in loader.engagements.values():
            prod_id = safe_int(eng['product_id'])
            results.append(EngagementResponse(
                id           = safe_int(eng['id']) or 0,
                name         = safe_str(eng['name'], f"engagement-{eng['id']}"),
                product      = prod_id or 0,
                product_name = safe_str(loader.products.get(prod_id, {}).get('name', 'Unknown')),
            ))
    return results


@app.get("/defectdojo/findings", response_model=List[FindingSummaryResponse], tags=["DefectDojo"])
async def get_findings(
    engagement_id: Optional[int] = None,
    product_id:    Optional[int] = None,
    limit:         int           = 2000,
) -> List[FindingSummaryResponse]:
    loader       = _require_local_loader()
    scores_cache = get_scores_cache()
    try:
        if engagement_id is not None:
            if engagement_id not in loader.engagements:
                raise HTTPException(404, detail=f"Engagement {engagement_id} introuvable")
            raw = loader.get_findings_for_engagement(engagement_id)
        elif product_id is not None:
            if product_id not in loader.products:
                raise HTTPException(404, detail=f"Produit {product_id} introuvable")
            raw = loader.get_findings_for_product(product_id)
        else:
            raw = loader.get_all_findings()
        results = [_finding_to_response(f, scores_cache) for f in raw]
        logger.info(f"[get_findings] engagement={engagement_id}, product={product_id} → {len(results)} findings")
        return results[:limit]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[get_findings] Erreur : {e}")
        raise HTTPException(500, detail=str(e))


@app.get("/defectdojo/products/{product_id}/findings", response_model=List[FindingSummaryResponse], tags=["DefectDojo"])
async def get_product_findings(product_id: int, limit: int = 2000) -> List[FindingSummaryResponse]:
    loader = _require_local_loader()
    if product_id not in loader.products:
        raise HTTPException(404, detail=f"Produit {product_id} introuvable")
    raw          = loader.get_findings_for_product(product_id)
    scores_cache = get_scores_cache()
    return [_finding_to_response(f, scores_cache) for f in raw][:limit]


@app.get("/defectdojo/engagements/{engagement_id}/findings", response_model=List[FindingSummaryResponse], tags=["DefectDojo"])
async def get_engagement_findings(engagement_id: int, limit: int = 2000) -> List[FindingSummaryResponse]:
    loader = _require_local_loader()
    if engagement_id not in loader.engagements:
        raise HTTPException(404, detail=f"Engagement {engagement_id} introuvable")
    raw          = loader.get_findings_for_engagement(engagement_id)
    scores_cache = get_scores_cache()
    return [_finding_to_response(f, scores_cache) for f in raw][:limit]


@app.get("/defectdojo/findings/{finding_id}", response_model=FindingSummaryResponse, tags=["DefectDojo"])
async def get_finding_by_id(finding_id: int) -> FindingSummaryResponse:
    loader = _require_local_loader()
    if finding_id not in loader.findings_by_id:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")
    scores_cache = get_scores_cache()
    return _finding_to_response(loader.findings_by_id[finding_id], scores_cache)


# ── Admin ─────────────────────────────────────────────────────────────────────

@app.post("/data/refresh", tags=["Admin"])
async def refresh_data() -> Dict[str, Any]:
    loader = _require_local_loader()
    try:
        if loader.load():
            # Re-score après refresh des données
            if model_manager.is_ready():
                _score_all_findings_at_startup(loader, model_manager)
            return {
                "status":      "refreshed",
                "products":    len(loader.products),
                "engagements": len(loader.engagements),
                "findings":    len(loader.findings_by_id),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
        raise HTTPException(500, detail="Impossible de recharger le CSV")
    except Exception as e:
        logger.error(f"Data refresh échoué : {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/cache/invalidate", tags=["Admin"])
async def invalidate_findings_cache(prefix: str = "") -> Dict[str, Any]:
    global _scores_cache_loaded_at
    count = invalidate_cache(prefix)
    if prefix in ("", "scores"):
        _scores_cache_loaded_at = 0.0
    logger.info(f"Cache invalidé : {count} entrées (prefix='{prefix}')")
    return {"invalidated": count, "prefix": prefix}


# ── Prediction endpoints ──────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(finding: FindingInput, request: Request) -> PredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    rid = getattr(request.state, "request_id", "unknown")
    try:
        features      = finding.to_features()
        X             = pd.DataFrame([features])
        expected_cols = model_manager.feature_columns
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_cols]
        X.columns = pd.Index(expected_cols)

        risk_class = int(model_manager.get_model().predict(X)[0])
        probas     = model_manager.get_model().predict_proba(X)[0]
        classes    = model_manager.classes
        confidence = round(float(max(probas)), 4)
        risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / max(classes), 2)
        proba_dict = {CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, probas)}

        logger.info(f"[{rid}] Predict: {finding.title[:50]!r} → {CLASS_LABELS.get(risk_class)} (conf={confidence:.2f})")

        return PredictionResponse(
            request_id    = rid,
            finding_id    = finding.finding_id,
            engagement_id = finding.engagement_id,
            product_id    = finding.product_id,
            risk_class    = risk_class,
            risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
            risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, ""), "#888"),
            risk_score    = risk_score,
            confidence    = confidence,
            context_score = int(features["context_score"]),
            cvss_score    = finding.cvss_score,
            probabilities = proba_dict,
            features_used = {c: round(float(X[c].iloc[0]), 4) for c in expected_cols[:10]},
            predicted_at  = datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{rid}] Erreur prédiction : {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput, request: Request) -> BatchPredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    rid   = getattr(request.state, "request_id", "unknown")
    start = time.perf_counter()
    try:
        records       = [f.to_features() for f in batch.findings]
        X             = pd.DataFrame(records)
        expected_cols = model_manager.feature_columns
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_cols]
        X.columns = pd.Index(expected_cols)

        classes_arr, probas_arr = model_manager.predict_batch_cached(X)
        classes = model_manager.classes
        results = []
        for idx, finding in enumerate(batch.findings):
            risk_class = int(classes_arr[idx])
            probas     = probas_arr[idx]
            confidence = round(float(max(probas)), 4)
            risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / max(classes), 2)
            proba_dict = {CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, probas)}
            results.append(PredictionResponse(
                request_id    = rid,
                finding_id    = finding.finding_id,
                engagement_id = finding.engagement_id,
                product_id    = finding.product_id,
                risk_class    = risk_class,
                risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
                risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, ""), "#888"),
                risk_score    = risk_score,
                confidence    = confidence,
                context_score = int(records[idx]["context_score"]),
                probabilities = proba_dict,
                features_used = {},
                predicted_at  = datetime.now(timezone.utc).isoformat(),
            ))
        summary = {lv: sum(1 for r in results if r.risk_level.lower() == lv)
                   for lv in ["critical", "high", "medium", "low"]}
        elapsed = time.perf_counter() - start
        logger.info(f"[{rid}] Batch: {len(results)} findings en {elapsed:.2f}s")
        return BatchPredictionResponse(
            request_id   = rid,
            total        = len(batch.findings),
            success      = len(results),
            errors_count = 0,
            results      = results,
            errors       = [],
            summary      = summary,
            processed_at = datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{rid}] Erreur batch : {e}")
        raise HTTPException(500, detail=str(e))


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLMRequest(BaseModel):
    finding_id:  Optional[int]       = None
    title:       str
    severity:    str
    cvss_score:  Optional[float]     = 0.0
    description: Optional[str]       = ""
    cve:         Optional[str]       = ""
    cwe:         Optional[str]       = ""
    file_path:   Optional[str]       = ""
    tags:        Optional[List[str]] = []
    risk_level:  Optional[str]       = ""


class LLMExplanationResponse(BaseModel):
    finding_id:              Optional[int] = None
    summary:                 str
    impact:                  str
    root_cause:              Optional[str] = None
    exploitation_difficulty: Optional[str] = None
    priority_note:           str
    from_cache:              bool          = False


class LLMRecommendationResponse(BaseModel):
    finding_id:      Optional[int]       = None
    title:           str
    recommendations: List[str]
    references:      Optional[List[str]] = []
    verification:    Optional[str]       = None
    prevention:      Optional[str]       = None
    from_cache:      bool                = False


def _explanation_fallback(fid, title, severity, cvss) -> LLMExplanationResponse:
    return LLMExplanationResponse(
        finding_id              = fid,
        summary                 = f"Vulnérabilité {severity.upper()} : {title}",
        impact                  = f"Sévérité {severity} (CVSS: {cvss or 'N/A'}) — risque de sécurité.",
        root_cause              = "Analyse automatique indisponible.",
        exploitation_difficulty = "Moyenne",
        priority_note           = (
            "Immédiat" if severity.lower() in ("critical", "high")
            else "Semaine" if severity.lower() == "medium"
            else "Mois"
        ),
    )


def _recommendation_fallback(fid) -> LLMRecommendationResponse:
    return LLMRecommendationResponse(
        finding_id      = fid,
        title           = "Recommandations générales",
        recommendations = [
            "Analyser le code source autour du fichier concerné",
            "Appliquer les correctifs recommandés par OWASP",
            "Tester la correction avec un scanner de sécurité",
            "Re-scanner après application du correctif",
        ],
        references   = ["https://owasp.org/www-project-top-ten/"],
        verification = "Re-lancer le scan et vérifier l'absence du finding",
        prevention   = "Former les développeurs aux bonnes pratiques de sécurité",
    )


@app.post("/explain/llm", response_model=LLMExplanationResponse, tags=["LLM"])
async def explain_with_llm(request: LLMRequest):
    result = _llm_explain(request.model_dump(), use_cache=True, mode="explanation") if _LLM_AVAILABLE else None
    if not result or not result.get("summary"):
        return _explanation_fallback(request.finding_id, request.title, request.severity, request.cvss_score or 0)
    return LLMExplanationResponse(
        finding_id              = request.finding_id,
        summary                 = result["summary"],
        impact                  = result.get("impact", ""),
        root_cause              = result.get("root_cause"),
        exploitation_difficulty = result.get("exploitation_difficulty"),
        priority_note           = result.get("priority_note", ""),
        from_cache              = result.get("from_cache", False),
    )


@app.post("/recommend/llm", response_model=LLMRecommendationResponse, tags=["LLM"])
async def recommend_with_llm(request: LLMRequest):
    result = _llm_explain(request.model_dump(), use_cache=True, mode="recommendation") if _LLM_AVAILABLE else None
    if not result or not result.get("recommendations"):
        return _recommendation_fallback(request.finding_id)
    return LLMRecommendationResponse(
        finding_id      = request.finding_id,
        title           = result.get("title", "Recommandations IA"),
        recommendations = result.get("recommendations", []),
        references      = result.get("references", []),
        verification    = result.get("verification"),
        prevention      = result.get("prevention"),
        from_cache      = result.get("from_cache", False),
    )


@app.get("/defectdojo/findings/{finding_id}/explain", response_model=LLMExplanationResponse, tags=["LLM"])
async def explain_finding_from_dojo(finding_id: int):
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id=finding_id, title=finding.title, severity=finding.severity,
        cvss_score=finding.cvss_score, description=finding.description or "",
        file_path=finding.file_path or "", tags=finding.tags or [],
        risk_level=finding.risk_level or "",
    )
    return await explain_with_llm(req)


@app.get("/defectdojo/findings/{finding_id}/recommend", response_model=LLMRecommendationResponse, tags=["LLM"])
async def recommend_finding_from_dojo(finding_id: int):
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id=finding_id, title=finding.title, severity=finding.severity,
        cvss_score=finding.cvss_score, description=finding.description or "",
        file_path=finding.file_path or "", tags=finding.tags or [],
        risk_level=finding.risk_level or "",
    )
    return await recommend_with_llm(req)


@app.get("/llm/health", tags=["LLM"])
async def llm_health():
    try:
        r = requests.get(f"{os.getenv('OLLAMA_URL', 'http://192.168.11.170:11434')}/api/tags", timeout=5)
        return {"status": "ok", "models": [m["name"] for m in r.json().get("models", [])],
                "current": os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ── Jira ──────────────────────────────────────────────────────────────────────

@app.post(
    "/defectdojo/findings/{finding_id}/create-jira-issue",
    response_model=JiraIssueResponse,
    tags=["Jira"],
    summary="Crée une issue Jira pour un finding avec score IA + LLM",
)
async def create_jira_issue_for_finding(finding_id: int):
    existing = jira_service._find_existing_issue(finding_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Un ticket Jira existe déjà : {existing.key}")

    if local_data_loader is None or not local_data_loader.is_ready:
        raise HTTPException(503, detail="LocalDataLoader non prêt")

    finding = local_data_loader.findings_by_id.get(finding_id)
    if not finding:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")

    # Récupérer le score depuis le cache (rempli au démarrage par le vrai modèle)
    scores_cache = get_scores_cache()
    cached_score = scores_cache.get(str(finding_id), {})

    try:
        if cached_score:
            # Utiliser le score IA déjà calculé au démarrage
            risk_class = cached_score.get("ai_risk_score", 0)
            risk_level = cached_score.get("ai_risk_level", "Low")
            confidence = cached_score.get("ai_confidence", 0.0)
            proba_dict = cached_score.get("ai_probabilities", {})
            classes    = model_manager.classes
            probas_vals = [proba_dict.get(CLASS_LABELS.get(c, str(c)), 0.0) for c in classes]
            risk_score = round(sum(c * p for c, p in zip(classes, probas_vals)) * 100 / max(classes), 2)
        else:
            # Fallback : recalculer via FindingInput
            inp = FindingInput(
                severity        = finding.get("severity", "info"),
                cvss_score      = safe_float(finding.get("cvss_score"), 0.0),
                title           = finding.get("title", ""),
                description     = finding.get("description", ""),
                tags            = finding.get("tags", []),
                days_open       = _compute_age_days(finding.get("created")) or 0,
                epss_score      = safe_float(finding.get("epss_score"), 0.0),
                epss_percentile = safe_float(finding.get("epss_percentile"), 0.0),
                has_cve         = 1 if finding.get("cve") else 0,
                has_cwe         = 1 if finding.get("cwe") else 0,
            )
            feat     = inp.to_features()
            expected = model_manager.feature_columns
            X        = pd.DataFrame([{c: feat.get(c, 0.0) for c in expected}])
            X.columns = pd.Index(expected)
            classes_arr, probas_arr = model_manager.predict_batch_cached(X)
            risk_class  = int(classes_arr[0])
            probas_vals = probas_arr[0]
            classes     = model_manager.classes
            confidence  = round(float(max(probas_vals)), 4)
            risk_score  = round(sum(c * p for c, p in zip(classes, probas_vals)) * 100 / max(classes), 2)
            proba_dict  = {CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, probas_vals)}
            risk_level  = CLASS_LABELS.get(risk_class, "Unknown")

        ai_result = {
            "risk_level":    risk_level,
            "risk_score":    risk_score,
            "confidence":    confidence,
            "probabilities": proba_dict,
        }
    except Exception as e:
        logger.warning(f"Prédiction IA échouée pour finding {finding_id} : {e}")
        ai_result = {"risk_level": finding.get("severity", "medium").capitalize(),
                     "risk_score": 50, "confidence": 0.0, "probabilities": {}}

    explanation = recommendation = None
    if _LLM_AVAILABLE:
        try:
            llm_req = LLMRequest(
                finding_id=finding_id, title=finding.get("title", ""),
                severity=ai_result.get("risk_level", "medium"),
                cvss_score=finding.get("cvss_score", 0.0),
                description=finding.get("description", ""),
                cve=finding.get("cve", ""), tags=finding.get("tags", []),
                risk_level=ai_result.get("risk_level"),
            )
            explanation    = await explain_with_llm(llm_req)
            recommendation = await recommend_with_llm(llm_req)
        except Exception as e:
            logger.warning(f"LLM indisponible pour finding {finding_id} : {e}")

    try:
        result = jira_service.create_security_issue(
            finding            = finding,
            ai_prediction      = ai_result,
            llm_explanation    = explanation.dict() if explanation else None,
            llm_recommendation = recommendation.dict() if recommendation else None,
        )
    except Exception as e:
        logger.error(f"Erreur création issue Jira : {e}")
        raise HTTPException(502, detail=f"Erreur Jira : {str(e)}")

    message = ("Issue déjà existante" if result.get("already_exists")
               else f"Issue créée avec succès : {result['jira_key']}")
    return JiraIssueResponse(
        key            = result["jira_key"],
        id             = result.get("jira_id", ""),
        self           = result.get("jira_self", ""),
        url            = result.get("jira_url"),
        already_exists = result.get("already_exists", False),
        message        = message,
    )


@app.get("/defectdojo/findings/{finding_id}/jira-issue", tags=["Jira"])
async def get_jira_issue_for_finding(finding_id: int):
    if not local_data_loader or not local_data_loader.is_ready:
        raise HTTPException(503, "LocalDataLoader non prêt")
    if finding_id not in local_data_loader.findings_by_id:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")
    try:
        issue = jira_service._find_existing_issue(finding_id)
        if issue:
            return {"exists": True, "jira_key": issue.key,
                    "jira_url": f"{jira_service.server}/browse/{issue.key}",
                    "created": issue.fields.created if hasattr(issue, 'fields') else None}
        return {"exists": False, "jira_key": None, "jira_url": None}
    except Exception as e:
        logger.error(f"Erreur vérification Jira pour finding {finding_id}: {e}")
        return {"exists": False, "jira_key": None, "jira_url": None}


@app.get("/jira/health", response_model=JiraHealthResponse, tags=["Jira"])
async def jira_health():
    health = jira_service.health_check()
    return JiraHealthResponse(**health)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port        = int(os.getenv("API_PORT",  "8081"))
    host        = os.getenv("API_HOST",      "0.0.0.0")
    reload_mode = os.getenv("API_RELOAD",    "false").lower() == "true"
    logger.info(f"Démarrage API sur {host}:{port}")
    logger.info(f"Documentation : http://localhost:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=reload_mode, log_level="info")