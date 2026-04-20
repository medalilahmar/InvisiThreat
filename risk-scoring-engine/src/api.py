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


#  CONSTANTES
API_VERSION = "3.1.0"
APP_START   = datetime.now(timezone.utc)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline_latest.pkl"))
META_PATH  = Path(os.getenv("META_PATH",  "models/pipeline_latest_meta.json"))

DEFECTDOJO_URL     = os.getenv("DEFECTDOJO_URL", "http://localhost:8080")
DEFECTDOJO_API_KEY = os.getenv("DEFECTDOJO_API_KEY")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   "60"))

# TTL du cache (secondes)
CACHE_TTL_PRODUCTS    = 300
CACHE_TTL_TESTS       = 300
CACHE_TTL_FINDINGS    = 60
SCORES_CACHE_TTL      = 60

# Chemin CSV local (source de vérité)
CSV_FINDINGS_PATH = Path(os.getenv("CSV_FINDINGS_PATH", "data/processed/findings_clean.csv"))

RISK_LEVELS: Dict[int, str]  = {0: "info", 1: "low", 2: "medium", 3: "high", 4: "critical"}
CLASS_LABELS: Dict[int, str] = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
CLASS_COLORS: Dict[str, str] = {
    "Info":     "#95a5a6",
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

FEATURE_COLS: List[str] = [
    "cvss_score", "cvss_score_norm", "age_days", "age_days_norm",
    "has_cve", "has_cwe", "tags_count", "tags_count_norm",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "product_fp_rate", "cvss_x_has_cve", "age_x_cvss",
    "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm",
    "exploit_risk", "context_score", "days_open_high",
    "severity_num", "cvss_severity_gap", "severity_x_active",
    "cvss_x_severity", "severity_x_urgent", "score_composite_raw", "score_composite_adj",
]


#  CACHE GÉNÉRIQUE (TTL)
_cache_store: Dict[str, Tuple[Any, float]] = {}
_scores_cache_memory: Dict[str, Any]       = {}
_scores_cache_loaded_at: float             = 0.0


def get_cached_or_fetch(cache_key: str, fetch_func, ttl: int = CACHE_TTL_PRODUCTS) -> Any:
    now   = time.time()
    entry = _cache_store.get(cache_key)
    if entry is not None:
        data, ts = entry
        if now - ts < ttl:
            logger.debug(f"Cache HIT : {cache_key}")
            return data
    logger.debug(f"Cache MISS : {cache_key}")
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


#  LOCAL DATA LOADER (CSV First)
#  Charge findings_clean.csv et construit
#  la hiérarchie complète en mémoire
class LocalDataLoader:
    """
    Charge et gère les données locales depuis CSV.
    Source de vérité pour les findings avec hiérarchie
    stricte : product → engagement → test → finding
    """

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
        """Charge les données depuis CSV et construit la hiérarchie."""
        if not self.csv_path.exists():
            logger.warning(f"CSV introuvable : {self.csv_path}")
            return False

        try:
            self.df_findings = pd.read_csv(self.csv_path)
            logger.info(f"[LocalDataLoader] CSV chargé : {len(self.df_findings)} findings")

            # Construire les maps
            self._build_hierarchy()
            self._loaded_at = datetime.now(timezone.utc)
            self._ready = True
            logger.info(
                f"[LocalDataLoader] Hiérarchie construite : "
                f"{len(self.products)} produits, "
                f"{len(self.engagements)} engagements, "
                f"{len(self.tests)} tests"
            )
            return True

        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur de chargement : {e}")
            return False

    def _build_hierarchy(self) -> None:
        """Construit les maps de hiérarchie depuis le CSV."""
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
                logger.error(f"[LocalDataLoader] Missing columns: {missing_cols}")
                raise KeyError(f"Missing columns: {missing_cols}")

            # Produits
            for _, row in self.df_findings[
                ['product_id', 'product_name']
            ].drop_duplicates().iterrows():
                prod_id = int(row['product_id'])
                self.products[prod_id] = {
                    'id': prod_id,
                    'name': row['product_name'],
                    'engagements': set(),
                }

            # Engagements
            for _, row in self.df_findings[
                ['product_id', 'engagement_id', 'engagement_name']
            ].drop_duplicates().iterrows():
                eng_id = int(row['engagement_id'])
                prod_id = int(row['product_id'])
                self.engagements[eng_id] = {
                    'id': eng_id,
                    'name': row['engagement_name'],
                    'product_id': prod_id,
                    'tests': set(),
                }
                if prod_id in self.products:
                    self.products[prod_id]['engagements'].add(eng_id)

            # Findings
            # ✅ MAP severity_num → severity_label
            SEVERITY_MAP = {
                0: 'info',
                1: 'low',
                2: 'medium',
                3: 'high',
                4: 'critical'
            }
        
            for _, row in self.df_findings.iterrows():
                finding_id = int(row['id'])
                eng_id = int(row['engagement_id'])
                prod_id = int(row['product_id'])

                finding_dict = row.to_dict()
                finding_dict['id'] = finding_id
                finding_dict['product_id'] = prod_id
                finding_dict['engagement_id'] = eng_id

                # ✅ CONVERSION : severity_num → severity_label
                severity_num = row['severity_num']
            
                if pd.isna(severity_num):
                    severity = 'info'
                else:
                    try:
                        severity = SEVERITY_MAP.get(int(severity_num), 'info')
                    except (ValueError, TypeError):
                        severity = 'info'

                finding_dict['severity'] = severity

                tags_list = []
                if row.get('tag_urgent') == 1 or row.get('tag_urgent') is True:
                    tags_list.append('urgent')
                if row.get('tag_in_production') == 1 or row.get('tag_in_production') is True:
                    tags_list.append('production')
                if row.get('tag_sensitive') == 1 or row.get('tag_sensitive') is True:
                    tags_list.append('sensitive')
                if row.get('tag_external') == 1 or row.get('tag_external') is True:
                    tags_list.append('external')
                finding_dict['tags'] = tags_list

                self.findings_by_id[finding_id] = finding_dict

                # Lier engagement si nécessaire
                if eng_id in self.engagements:
                    if 'findings' not in self.engagements[eng_id]:
                        self.engagements[eng_id]['findings'] = []
                    self.engagements[eng_id]['findings'].append(finding_id)

            self._loaded_at = datetime.now(timezone.utc)
            self._ready = True
            logger.info(
                f"[LocalDataLoader] Hiérarchie construite : "
                f"{len(self.products)} produits, "
                f"{len(self.engagements)} engagements, "
                f"{len(self.tests)} tests"
            )

        except Exception as e:
            logger.error(f"[LocalDataLoader] Erreur de chargement : {e}")
            self._ready = False
            raise

    def get_findings_for_product(self, product_id: int) -> List[Dict]:
        """
        Retourne TOUS les findings d'un produit
        en descendant strictement : produit → engagements → findings
        """
        if not self._ready:
            return []

        results = []
        if product_id not in self.products:
            return results

        prod = self.products[product_id]
        # Descendre sur tous les engagements du produit
        for eng_id in prod.get('engagements', set()):
            if eng_id in self.engagements:
                eng_findings = self.engagements[eng_id].get('findings', [])
                for finding_id in eng_findings:
                    if finding_id in self.findings_by_id:
                        results.append(self.findings_by_id[finding_id])

        logger.info(f"[LocalDataLoader] Product {product_id} : {len(results)} findings")
        return results

    def get_findings_for_engagement(self, engagement_id: int) -> List[Dict]:
        """Retourne les findings d'un engagement spécifique."""
        if not self._ready:
            return []

        results = []
        if engagement_id not in self.engagements:
            return results

        eng = self.engagements[engagement_id]
        for finding_id in eng.get('findings', []):
            if finding_id in self.findings_by_id:
                results.append(self.findings_by_id[finding_id])

        logger.info(f"[LocalDataLoader] Engagement {engagement_id} : {len(results)} findings")
        return results

    def get_all_findings(self) -> List[Dict]:
        """Retourne TOUS les findings du CSV."""
        if not self._ready:
            return []
        return list(self.findings_by_id.values())

    def get_products(self) -> List[Dict]:
        """Retourne tous les produits."""
        return [
            {
                'id': p['id'],
                'name': p['name'],
                'engagement_count': len(p.get('engagements', [])),
            }
            for p in self.products.values()
        ]

    def get_engagements_for_product(self, product_id: int) -> List[Dict]:
        """Retourne les engagements d'un produit."""
        if product_id not in self.products:
            return []

        results = []
        for eng_id in self.products[product_id].get('engagements', set()):
            if eng_id in self.engagements:
                eng = self.engagements[eng_id]
                results.append({
                    'id': eng['id'],
                    'name': eng['name'],
                    'product_id': product_id,
                })
        return results

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at


#  MODÈLES PYDANTIC
class FindingInput(BaseModel):
    severity:        str            = Field(..., description="critical, high, medium, low, info")
    cvss_score:      float          = Field(0.0, ge=0.0, le=10.0)
    title:           str            = ""
    description:     str            = ""
    file_path:       str            = ""
    tags:            List[str]      = []
    finding_id:      Optional[int]  = None
    engagement_id:   Optional[int]  = None
    product_id:      Optional[int]  = None
    days_open:       int            = 0
    duplicate_count: int            = 0
    epss_score:      float          = 0.0
    epss_percentile: float          = 0.0
    has_cve:         int            = 0
    has_cwe:         int            = 0

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
        tags_lower    = [t.lower() for t in self.tags]
        severity_map  = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0, "informational": 0}
        severity_num  = severity_map.get(self.severity.lower(), 0)

        tag_weights = {
            "production": 2, "prod": 2, "prd": 2, "live": 2, "main": 1,
            "external": 2, "internet-facing": 2, "public": 2, "exposed": 2,
            "sensitive": 1, "pii": 1, "gdpr": 1, "confidential": 1,
            "urgent": 3, "blocker": 3, "critical": 3,
        }
        context_score     = min(sum(tag_weights.get(t, 0) for t in tags_lower), 10)
        tag_urgent        = 1 if any(t in tags_lower for t in ["urgent", "blocker", "critical"]) else 0
        tag_in_production = 1 if any(t in tags_lower for t in ["production", "prod", "prd", "live"]) else 0
        tag_sensitive     = 1 if any(t in tags_lower for t in ["sensitive", "pii", "gdpr", "confidential"]) else 0
        tag_external      = 1 if any(t in tags_lower for t in ["external", "internet-facing", "public", "exposed"]) else 0

        cvss_score_norm = round(min(self.cvss_score / 10.0, 1.0), 4)
        age_days_norm   = round(min(self.days_open / 365.0, 1.0), 4)
        tags_count_norm = round(min(len(tags_lower) / 4.0, 1.0), 4)
        epss_score_norm = round(min(self.epss_score, 1.0), 4)
        has_high_epss   = 1 if self.epss_score > 0.5 else 0
        days_open_high  = 1 if self.days_open > 30 else 0

        text        = f"{self.title} {self.description}".lower()
        has_exploit = any(kw in text for kw in ["exploit", "metasploit", "poc", "public exploit"])
        exploit_risk = round(
            min((self.cvss_score * 0.7 + severity_num * 0.3) * (1.5 if has_exploit else 1), 10.0), 4
        )

        cvss_x_has_cve    = round(self.cvss_score * self.has_cve, 4)
        age_x_cvss        = round(self.days_open * self.cvss_score, 3)
        epss_x_cvss       = round(self.cvss_score * self.epss_score, 4)
        cvss_severity_gap = round(abs(self.cvss_score - (severity_num * 2.5)), 2)
        severity_x_active = severity_num * 1
        cvss_x_severity   = round(self.cvss_score * severity_num, 3)
        severity_x_urgent = severity_num * tag_urgent

        score_composite_raw = round(
            (self.cvss_score * 0.35 + exploit_risk * 0.25 + context_score * 0.15
             + self.epss_score * 10 * 0.15 + (1 if self.has_cve else 0) * 10 * 0.10), 2
        )
        age_penalty         = max(0, 1 - (self.days_open / 180))
        score_composite_adj = round(score_composite_raw * age_penalty, 2)

        return {
            "cvss_score": self.cvss_score, "cvss_score_norm": cvss_score_norm,
            "age_days": self.days_open,    "age_days_norm": age_days_norm,
            "has_cve": self.has_cve,       "has_cwe": self.has_cwe,
            "tags_count": tag_urgent + tag_in_production + tag_sensitive + tag_external,
            "tags_count_norm": tags_count_norm,
            "tag_urgent": tag_urgent,      "tag_in_production": tag_in_production,
            "tag_sensitive": tag_sensitive,"tag_external": tag_external,
            "product_fp_rate": 0.0,
            "cvss_x_has_cve": cvss_x_has_cve, "age_x_cvss": age_x_cvss,
            "epss_score": self.epss_score, "epss_percentile": self.epss_percentile,
            "has_high_epss": has_high_epss, "epss_x_cvss": epss_x_cvss,
            "epss_score_norm": epss_score_norm,
            "exploit_risk": exploit_risk,  "context_score": context_score,
            "days_open_high": days_open_high,
            "severity_num": severity_num,  "cvss_severity_gap": cvss_severity_gap,
            "severity_x_active": severity_x_active, "cvss_x_severity": cvss_x_severity,
            "severity_x_urgent": severity_x_urgent,
            "score_composite_raw": score_composite_raw,
            "score_composite_adj": score_composite_adj,
        }


class BatchInput(BaseModel):
    findings:      List[FindingInput] = Field(..., min_length=1, max_length=500)
    engagement_id: Optional[int]      = None


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
    cvss_score: float = 0.0  
    cve: Optional[str] = None 
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
    cve: Optional[str] = None
    tags:            List[str]      = []
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
    risk_class:      Optional[int]   = None
    risk_level:      Optional[str]   = None
    ai_confidence:   Optional[float] = None
    context_score:   Optional[int]   = None


#  HELPERS
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


def _finding_to_response(f: Dict, scores_cache: Dict) -> FindingSummaryResponse:
   
    fid        = safe_int(f.get("id")) or 0
    score_data = scores_cache.get(str(fid), {})

    SEVERITY_MAP = {0: 'info', 1: 'low', 2: 'medium', 3: 'high', 4: 'critical'}
    VALID_SEVERITIES = ["critical", "high", "medium", "low", "info"]
    
    severity_raw = f.get("severity")
    cve_value = safe_str(f.get("cve"))
    
    if severity_raw is None or (pd.isna(severity_raw) if not isinstance(severity_raw, str) else severity_raw == ''):
        severity_num = f.get("severity_num")
        if severity_num is not None and not pd.isna(severity_num):
            severity_raw = SEVERITY_MAP.get(int(severity_num), 'info')
        else:
            severity_raw = 'info'
    
    if severity_raw is None or (isinstance(severity_raw, float) and np.isnan(severity_raw)):
        severity = "info"
    else:
        severity = str(severity_raw).strip().lower()
        if severity not in VALID_SEVERITIES:
            severity = "info"

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
        risk_class      = safe_int(score_data.get("ai_risk_score")),
        risk_level      = safe_str(score_data.get("ai_risk_level")),
        ai_confidence   = safe_float(score_data.get("ai_confidence"), 0.0),
        context_score   = safe_int(score_data.get("context_score")),
    )

def safe_str(val: Any, default: str = "") -> str:
    """Convertit en string, gère NaN et None."""
    if val is None or val == "":
        return default
    try:
        if isinstance(val, float):
            if np.isnan(val):
                return default
            return str(val)
        return str(val).strip()
    except (ValueError, TypeError):
        return default


def safe_int(val: Any) -> Optional[int]:
    """Convertit en int, retourne None si NaN ou invalide."""
    if val is None or val == "":
        return None
    try:
        if isinstance(val, float):
            if np.isnan(val):
                return None
            return int(val)
        return int(val)
    except (ValueError, TypeError):
        return None


def safe_float(val: Any, default: float = 0.0) -> float:
    """Convertit en float, retourne default si NaN."""
    if val is None or val == "":
        return default
    try:
        f_val = float(val)
        if np.isnan(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default


class ModelManager:
    def __init__(self, model_path: Path, meta_path: Path):
        self.model_path          = model_path
        self.meta_path           = meta_path
        self._model              = None
        self._metadata:          Dict[str, Any]          = {}
        self._loaded_at:         Optional[datetime]      = None
        self._feature_columns:   List[str]               = FEATURE_COLS.copy()
        self._prediction_cache:  Dict[str, Tuple]        = {}
        self._cache_max_size     = 100
        self._cache_hits         = 0
        self._cache_misses       = 0

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
            if hasattr(self._model, "feature_names_in_"):
                self._feature_columns = list(self._model.feature_names_in_)
            elif self._metadata.get("feature_columns"):
                self._feature_columns = self._metadata["feature_columns"]
            logger.info(f"Modèle chargé — F1={self._metadata.get('metrics', {}).get('test_f1_weighted', 'N/A')}")
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
        return len(self._model.classes_) if self._model and hasattr(self._model, "classes_") else 5

    @property
    def classes(self) -> List[int]:
        return [int(c) for c in self._model.classes_] if self._model and hasattr(self._model, "classes_") else [0,1,2,3,4]

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def model_version(self) -> str:
        return self._metadata.get("timestamp", "unknown")



model_manager:     ModelManager               = ModelManager(MODEL_PATH, META_PATH)
shap_explainer                                = None
local_data_loader: Optional[LocalDataLoader] = None
_rate_limit_store: Dict[str, List[float]]     = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global local_data_loader, shap_explainer

    logger.info(f"Démarrage InvisiThreat AI Risk Engine API v{API_VERSION}")

    # ── Chargement modèle ────────────────────
    if model_manager.load_model():
        logger.info("✅ Modèle ML chargé")
        try:
            import shap
            estimator = model_manager.get_model()
            if hasattr(estimator, "named_steps"):
                estimator = estimator.named_steps[list(estimator.named_steps)[-1]]

            class SimpleShapExplainer:
                def __init__(self, mdl):
                    self._exp   = shap.TreeExplainer(mdl)
                    self._ready = True

                def is_ready(self) -> bool:
                    return self._ready

                def explain(self, X: pd.DataFrame, pred_class: int) -> Dict[str, Any]:
                    try:
                        sv  = self._exp.shap_values(X)
                        sv  = sv[pred_class][0] if isinstance(sv, list) else sv[0]
                        top = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:10]
                        ev  = self._exp.expected_value
                        base = float(ev[pred_class] if isinstance(ev, (list, np.ndarray)) else ev)
                        return {
                            "top_features": [
                                {"feature": X.columns[i], "shap_value": round(float(sv[i]), 4),
                                 "feature_value": round(float(X.iloc[0, i]), 4)}
                                for i in top
                            ],
                            "base_value": round(base, 4),
                        }
                    except Exception as e:
                        logger.warning(f"SHAP explain échoué : {e}")
                        return {"top_features": [], "base_value": 0}

            shap_explainer = SimpleShapExplainer(estimator)
            logger.info("✅ SHAP explainer chargé")
        except ImportError:
            logger.warning("⚠️ SHAP non installé — explications désactivées")
        except Exception as e:
            logger.warning(f"⚠️ SHAP explainer échoué : {e}")
    else:
        logger.warning("❌ Modèle non chargé — prédictions retourneront 503")

    # ── LocalDataLoader (CSV First) ──────────
    try:
        local_data_loader = LocalDataLoader(CSV_FINDINGS_PATH)
        if local_data_loader.load():
            logger.info(
                f"✅ LocalDataLoader initialisé depuis CSV : "
                f"{len(local_data_loader.products)} produits, "
                f"{len(local_data_loader.engagements)} engagements, "
                f"{len(local_data_loader.findings_by_id)} findings"
            )
        else:
            logger.warning("⚠️ LocalDataLoader : CSV non accessible, mode dégradé")
    except Exception as e:
        logger.error(f"❌ LocalDataLoader échoué : {e}")

    yield
    logger.info("API arrêtée")






#  APPLICATION FASTAPI
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
    """
    ✅ Affiche les 10 premiers findings avec severité brute vs traitée
    """
    loader = _require_local_loader()
    scores_cache = get_scores_cache()
    
    findings = loader.get_all_findings()[:10]
    
    results = []
    for f in findings:
        response = _finding_to_response(f, scores_cache)
        results.append({
            "id": f.get("id"),
            "severity_raw_from_dict": f.get("severity"),
            "severity_type": type(f.get("severity")).__name__,
            "severity_in_response": response.severity,
            "match": f.get("severity") == response.severity if f.get("severity") else False,
        })
    
    return {
        "total_findings": len(loader.get_all_findings()),
        "sample_findings": results,
    }


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start     = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now       = time.time()

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit dépassé", "request_id": request_id},
        )
    _rate_limit_store[client_ip].append(now)

    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"]      = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.1f}ms) IP={client_ip}"
    )
    return response


# ─────────────────────────────────────────────
#  GUARDS
# ─────────────────────────────────────────────
def _require_local_loader() -> LocalDataLoader:
    if local_data_loader is None or not local_data_loader.is_ready:
        raise HTTPException(
            status_code=503,
            detail="LocalDataLoader non prêt. CSV données non chargées.",
        )
    return local_data_loader


# ─────────────────────────────────────────────
#  ENDPOINTS — INFO
# ─────────────────────────────────────────────
@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    return {
        "service":                 "InvisiThreat AI Risk Engine",
        "version":                 API_VERSION,
        "status":                  "running",
        "model_ready":             model_manager.is_ready(),
        "data_loader_ready":       local_data_loader is not None and local_data_loader.is_ready,
        "docs":                    "/docs",
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
        "invisithreat_data_loader_products":    len(ld.products) if ld and ld.is_ready else 0,
        "invisithreat_data_loader_engagements": len(ld.engagements) if ld and ld.is_ready else 0,
        "invisithreat_data_loader_findings":    len(ld.findings_by_id) if ld and ld.is_ready else 0,
    }


# ─────────────────────────────────────────────
#  ENDPOINTS — MODÈLE
# ─────────────────────────────────────────────
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
        return {
            "status":        "reloaded",
            "model_version": model_manager.model_version,
            "loaded_at":     model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
        }
    raise HTTPException(status_code=500, detail="Rechargement du modèle échoué")


# ─────────────────────────────────────────────
#  ENDPOINTS — DEFECTDOJO (LOCAL CSV FIRST)
#  Hiérarchie stricte : Product → Engagement → Test → Finding
# ─────────────────────────────────────────────

@app.get("/defectdojo/products", response_model=List[ProductResponse], tags=["DefectDojo"])
async def get_products() -> List[ProductResponse]:
    """Liste tous les produits depuis le CSV local."""
    loader = _require_local_loader()
    products = loader.get_products()
    return [
        ProductResponse(
            id   = p['id'],
            name = p['name'],
        )
        for p in products
    ]


@app.get("/defectdojo/engagements", response_model=List[EngagementResponse], tags=["DefectDojo"])
async def get_engagements(product_id: Optional[int] = None) -> List[EngagementResponse]:
    """
    Liste les engagements.
    Si product_id est fourni, filtre sur ce produit uniquement.
    """
    loader = _require_local_loader()

    results = []
    if product_id is not None:
        # Filtre : descendre produit → engagements
        if product_id not in loader.products:
            logger.warning(f"[get_engagements] Produit {product_id} introuvable")
            return results

        engs = loader.get_engagements_for_product(product_id)
        results = [
            EngagementResponse(
                id           = safe_int(e['id']) or 0,
                name         = safe_str(e['name'], f"engagement-{e['id']}"),
                product      = safe_int(e['product_id']) or 0,
                product_name = safe_str(loader.products.get(e['product_id'], {}).get('name', 'Unknown')),
            )
            for e in engs
        ]
    else:
        # Retour tous les engagements
        for eng in loader.engagements.values():
            prod_id = safe_int(eng['product_id'])
            results.append(
                EngagementResponse(
                    id           = safe_int(eng['id']) or 0,
                    name         = safe_str(eng['name'], f"engagement-{eng['id']}"),
                    product      = prod_id or 0,
                    product_name = safe_str(loader.products.get(prod_id, {}).get('name', 'Unknown')),
                )
            )

    logger.info(f"[get_engagements] product_id={product_id} → {len(results)} engagements")
    return results


@app.get(
    "/defectdojo/findings",
    response_model=List[FindingSummaryResponse],
    tags=["DefectDojo"],
)
async def get_findings(
    engagement_id:    Optional[int] = None,
    product_id:       Optional[int] = None,
    limit:            int           = 2000,
) -> List[FindingSummaryResponse]:
    """
    Récupère les findings en respectant strictement la hiérarchie CSV.
    - Si engagement_id : retourne findings de cet engagement uniquement
    - Si product_id : descend produit → engagements → findings
    - Sinon : retourne tous les findings
    """
    loader = _require_local_loader()
    scores_cache = get_scores_cache()

    raw_findings = []

    try:
        if engagement_id is not None:
            # Cas 1 : filtre par engagement
            if engagement_id not in loader.engagements:
                raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} introuvable")
            raw_findings = loader.get_findings_for_engagement(engagement_id)

        elif product_id is not None:
            # Cas 2 : descend produit → engagements → findings
            if product_id not in loader.products:
                raise HTTPException(status_code=404, detail=f"Produit {product_id} introuvable")
            raw_findings = loader.get_findings_for_product(product_id)

        else:
            # Cas 3 : tous les findings
            raw_findings = loader.get_all_findings()

        # Convertir avec gestion complète des NaN
        results = [_finding_to_response(f, scores_cache) for f in raw_findings]
        logger.info(
            f"[get_findings] engagement={engagement_id}, product={product_id} → {len(results)} findings"
        )
        return results[:limit]

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[get_findings] Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/defectdojo/products/{product_id}/findings",
    response_model=List[FindingSummaryResponse],
    tags=["DefectDojo"],
)
async def get_product_findings(
    product_id: int,
    limit:      int = 2000,
) -> List[FindingSummaryResponse]:
    """
    Findings d'un produit.
    Descend strictement : produit → engagements → findings.
    Le compte retourné est identique au CSV (source de vérité).
    """
    loader = _require_local_loader()

    # Vérifier que le produit existe
    if product_id not in loader.products:
        raise HTTPException(status_code=404, detail=f"Produit {product_id} introuvable")

    try:
        raw_findings = loader.get_findings_for_product(product_id)
        scores_cache = get_scores_cache()
        results = [_finding_to_response(f, scores_cache) for f in raw_findings]

        logger.info(f"[get_product_findings] product={product_id} → {len(results)} findings")
        return results[:limit]

    except Exception as e:
        logger.error(f"[get_product_findings] Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/defectdojo/engagements/{engagement_id}/findings",
    response_model=List[FindingSummaryResponse],
    tags=["DefectDojo"],
)
async def get_engagement_findings(
    engagement_id: int,
    limit:         int = 2000,
) -> List[FindingSummaryResponse]:
    """
    Findings d'un engagement précis.
    Remonte uniquement ses findings (pas d'autres engagements mélangés).
    """
    loader = _require_local_loader()

    # Vérifier que l'engagement existe
    if engagement_id not in loader.engagements:
        raise HTTPException(status_code=404, detail=f"Engagement {engagement_id} introuvable")

    try:
        raw_findings = loader.get_findings_for_engagement(engagement_id)
        scores_cache = get_scores_cache()
        results = [_finding_to_response(f, scores_cache) for f in raw_findings]

        logger.info(f"[get_engagement_findings] engagement={engagement_id} → {len(results)} findings")
        return results[:limit]

    except Exception as e:
        logger.error(f"[get_engagement_findings] Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/defectdojo/findings/{finding_id}",
    response_model=FindingSummaryResponse,
    tags=["DefectDojo"],
)
async def get_finding_by_id(finding_id: int) -> FindingSummaryResponse:
    """Retourne un finding individuel depuis le CSV."""
    loader = _require_local_loader()

    if finding_id not in loader.findings_by_id:
        raise HTTPException(status_code=404, detail=f"Finding {finding_id} introuvable")

    try:
        f = loader.findings_by_id[finding_id]
        scores_cache = get_scores_cache()
        return _finding_to_response(f, scores_cache)

    except Exception as e:
        logger.error(f"[get_finding_by_id] Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))
# ─────────────────────────────────────────────
#  ENDPOINTS — ADMIN
# ─────────────────────────────────────────────
@app.post("/data/refresh", tags=["Admin"])
async def refresh_data() -> Dict[str, Any]:
    """
    Force le rechargement des données depuis le CSV.
    À appeler après modification du CSV ou nouveau export DefectDojo.
    """
    loader = _require_local_loader()

    try:
        if loader.load():
            return {
                "status":       "refreshed",
                "products":     len(loader.products),
                "engagements":  len(loader.engagements),
                "findings":     len(loader.findings_by_id),
                "timestamp":    datetime.now(timezone.utc).isoformat(),
            }
        else:
            raise HTTPException(status_code=500, detail="Impossible de recharger le CSV")
    except Exception as e:
        logger.error(f"Data refresh échoué : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cache/invalidate", tags=["Admin"])
async def invalidate_findings_cache(prefix: str = "") -> Dict[str, Any]:
    global _scores_cache_loaded_at
    count = invalidate_cache(prefix)
    if prefix in ("", "scores"):
        _scores_cache_loaded_at = 0.0
    logger.info(f"Cache invalidé : {count} entrées (prefix='{prefix}')")
    return {"invalidated": count, "prefix": prefix}


#  ENDPOINTS — PRÉDICTION IA

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(finding: FindingInput, request: Request) -> PredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    rid = getattr(request.state, "request_id", "unknown")
    try:
        features      = finding.to_features()
        X             = pd.DataFrame([features])
        expected_cols = model_manager.feature_columns
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0.0
        X          = X[expected_cols]
        risk_class = int(model_manager.get_model().predict(X)[0])
        probas     = model_manager.get_model().predict_proba(X)[0]
        classes    = model_manager.classes
        risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / 4, 2)
        confidence = round(float(max(probas)), 4)
        proba_dict = {CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, probas)}
        logger.info(f"[{rid}] Predict: {finding.title[:50]!r} → {CLASS_LABELS.get(risk_class)} (conf={confidence:.2f})")
        return PredictionResponse(
            request_id    = rid,
            finding_id    = finding.finding_id,
            engagement_id = finding.engagement_id,
            product_id    = finding.product_id,
            risk_class    = risk_class,
            risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
            risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, "Info"), "#888"),
            risk_score    = risk_score,
            confidence    = confidence,
            context_score = features["context_score"],
            probabilities = proba_dict,
            features_used = {c: round(float(X[c].iloc[0]), 4) for c in expected_cols[:10]},
            predicted_at  = datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{rid}] Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput, request: Request) -> BatchPredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
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
        classes_arr, probas_arr = model_manager.predict_batch_cached(X)
        classes = model_manager.classes
        results = []
        for idx, finding in enumerate(batch.findings):
            risk_class = int(classes_arr[idx])
            probas     = probas_arr[idx]
            confidence = round(float(max(probas)), 4)
            risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / 4, 2)
            proba_dict = {CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, probas)}
            results.append(PredictionResponse(
                request_id    = rid,
                finding_id    = finding.finding_id,
                engagement_id = finding.engagement_id,
                product_id    = finding.product_id,
                risk_class    = risk_class,
                risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
                risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, "Info"), "#888"),
                risk_score    = risk_score,
                confidence    = confidence,
                context_score = records[idx]["context_score"],
                probabilities = proba_dict,
                features_used = {},
                predicted_at  = datetime.now(timezone.utc).isoformat(),
            ))
        summary = {lv: sum(1 for r in results if r.risk_level.lower() == lv)
                   for lv in ["critical", "high", "medium", "low", "info"]}
        elapsed = time.perf_counter() - start
        logger.info(f"[{rid}] Batch: {len(results)} findings in {elapsed:.2f}s")
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
        raise HTTPException(status_code=500, detail=str(e))


#  ENDPOINTS — LLM

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
        references  = ["https://owasp.org/www-project-top-ten/"],
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
        finding_id  = finding_id, title=finding.title,
        severity    = finding.severity, cvss_score=finding.cvss_score,
        description = finding.description or "", file_path=finding.file_path or "",
        tags        = finding.tags or [], risk_level=finding.risk_level or "",
    )
    return await explain_with_llm(req)


@app.get("/defectdojo/findings/{finding_id}/recommend", response_model=LLMRecommendationResponse, tags=["LLM"])
async def recommend_finding_from_dojo(finding_id: int):
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id  = finding_id, title=finding.title,
        severity    = finding.severity, cvss_score=finding.cvss_score,
        description = finding.description or "", file_path=finding.file_path or "",
        tags        = finding.tags or [], risk_level=finding.risk_level or "",
    )
    return await recommend_with_llm(req)


@app.get("/llm/health", tags=["LLM"])
async def llm_health():
    try:
        r = requests.get(
            f"{os.getenv('OLLAMA_URL', 'http://192.168.11.170:11434')}/api/tags", timeout=5
        )
        return {
            "status":  "ok",
            "models":  [m["name"] for m in r.json().get("models", [])],
            "current": os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    import uvicorn
    port        = int(os.getenv("API_PORT",  "8081"))
    host        = os.getenv("API_HOST",      "0.0.0.0")
    reload_mode = os.getenv("API_RELOAD",    "false").lower() == "true"
    logger.info(f"Démarrage API sur {host}:{port}")
    logger.info(f"Documentation : http://localhost:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=reload_mode, log_level="info")