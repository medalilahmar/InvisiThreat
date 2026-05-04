from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _configure_logging(level: str = "INFO") -> logging.Logger:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("invisithreat.predict_live")


logger = _configure_logging(os.environ.get("LOG_LEVEL", "INFO"))


# Matching train.py CLASS_LABELS exactly: {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
RISK_LEVELS: Dict[int, str] = {
    0: "low",
    1: "medium",
    2: "high",
    3: "critical",
}

SEVERITY_MAP: Dict[str, int] = {
    "critical": 3,
    "high": 2,
    "medium": 1,
    "low": 0,
    "info": 0,
    "informational": 0,
}

# Class score anchors — midpoint of each class on a [0, 10] scale.
# Maps class index (0=Low, 1=Medium, 2=High, 3=Critical) to a representative score.
# These are the "true" positions trained into the model — NOT business formula weights.
CLASS_SCORE_ANCHORS = np.array([1.5, 4.0, 7.0, 9.5])

# Maximum nudge (±) that business signals can apply on top of the model's expected score.
# Keeps the model's judgment primary while adding intra-class granularity.
# Set to 1.2 so high-confidence findings (where model_score ≈ class anchor) still
# show meaningful spread within the class band.
BUSINESS_NUDGE_MAX = 1.2

# Features EXACTEMENT comme dans train.py → FEATURE_COLS
# Ne jamais modifier cette liste sans réentraîner le modèle
EXPECTED_FEATURES = [
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

# Columns that were explicitly excluded during training (train.py EXCLUDE_COLS).
# Any of these present in a CSV must NOT be fed to the model — data leakage guard.
TRAIN_EXCLUDED_COLS = {
    "days_to_fix",
    "risk_class",
    "risk_score",           # ← target-derived, leaks ground truth
    "severity_num",
    "is_mitigated",
    "out_of_scope",
    "is_false_positive",
    "label_source",
    "score_composite_raw",  # ← derived from risk_score, leaks ground truth
    "cvss_x_severity",
    "severity_x_active",
    "severity_x_urgent",
    "cvss_severity_gap",
}


@dataclass
class PredictionResult:
    # Champs sans valeur par défaut
    finding_id: int
    title: str
    severity: str
    ai_risk_score: float
    ai_risk_level: str
    ai_confidence: float
    context_score: float
    exposure_norm: float
    shap_top_features: List[Dict[str, Any]]
    ai_prediction_timestamp: str
    model_version: str
    tags: List[str]

    probabilities: Dict[str, float] = field(default_factory=dict)
    model_base_score: float = 0.0
    business_nudge: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    blocked: bool
    threshold: float
    critical_count: int
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class ModelMetadata:
    version: str
    feature_columns: List[str]
    metrics: Dict[str, Any]
    trained_at: str
    model_type: str


# ─────────────────────────────────────────────────────────────────────────────
# CSV LOADER
# Mode CSV : lit les features DIRECTEMENT depuis findings_clean.csv
# sans recalculer quoi que ce soit via FeatureExtractor.
#
# IMPORTANT — data-leakage guard:
#   findings_clean.csv may contain columns like risk_score / score_composite_raw
#   that were computed AFTER labelling in preprocess.py and were explicitly
#   excluded from training (EXCLUDE_COLS in train.py).  We strip them here
#   so the predictor never sees them — identical behaviour to training.
# ─────────────────────────────────────────────────────────────────────────────

class CsvFindingsLoader:
    """
    Charge findings_clean.csv qui contient déjà toutes les features
    calculées par preprocess.py. Ne recalcule rien.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV findings file not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        if len(self.df) == 0:
            raise ValueError(f"CSV file is empty: {self.csv_path}")

        # ── Data-leakage guard ──────────────────────────────────────────────
        leaked = TRAIN_EXCLUDED_COLS.intersection(self.df.columns)
        if leaked:
            logger.warning(
                "DATA LEAKAGE GUARD — dropping %d column(s) that were excluded "
                "from training (EXCLUDE_COLS): %s",
                len(leaked), sorted(leaked),
            )
            self.df = self.df.drop(columns=list(leaked))

        logger.info("Loaded %d findings from CSV: %s", len(self.df), self.csv_path)
        logger.info("CSV columns after leakage guard: %d", len(self.df.columns))

    def get_findings(self) -> List[Dict]:
        return self.df.to_dict("records")

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()


# ─────────────────────────────────────────────────────────────────────────────
# DEFECTDOJO CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DefectDojoClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        page_size: int = 100,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        start = time.monotonic()
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.monotonic() - start
            logger.debug("GET %s -> %d (%.2fs)", url, response.status_code, elapsed)
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error on GET %s: %s", url, exc)
            raise
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection failed on GET %s: %s", url, exc)
            raise
        except requests.exceptions.Timeout:
            logger.error("Timeout on GET %s (limit=%ds)", url, self.timeout)
            raise

    def _patch(self, url: str, data: Dict) -> Optional[Dict]:
        start = time.monotonic()
        try:
            response = self._session.patch(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.monotonic() - start
            logger.debug("PATCH %s -> %d (%.2fs)", url, response.status_code, elapsed)
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error on PATCH %s: %s", url, exc)
            return None

    def iter_findings(
        self, engagement_id: int, active_only: bool = True
    ) -> Generator[Dict, None, None]:
        url = f"{self.base_url}/api/v2/findings/"
        params: Dict[str, Any] = {"engagement": engagement_id, "limit": self.page_size}
        if active_only:
            params["active"] = True

        page_num = 1
        while url:
            try:
                data = self._get(url, params=params)
            except Exception as exc:
                logger.error("Pagination stopped at page %d: %s", page_num, exc)
                break
            results = data.get("results", [])
            logger.debug("Page %d -> %d findings", page_num, len(results))
            yield from results
            url = data.get("next")
            params = None
            page_num += 1

    def get_findings(self, engagement_id: int, active_only: bool = True) -> List[Dict]:
        findings = list(self.iter_findings(engagement_id, active_only))
        logger.info(
            "Fetched %d findings from DefectDojo (engagement=%d)", len(findings), engagement_id
        )
        return findings

    def update_finding(self, finding_id: int, payload: Dict) -> bool:
        url = f"{self.base_url}/api/v2/findings/{finding_id}/"
        result = self._patch(url, payload)
        return result is not None


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTOR — utilisé UNIQUEMENT en mode DefectDojo (API live)
# En mode CSV, on lit les features directement depuis le CSV preprocessé.
#
# IMPORTANT: must produce EXACTLY the same features as preprocess.py,
# in the same order as EXPECTED_FEATURES / train.py FEATURE_COLS.
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Utilisé uniquement en mode DefectDojo (findings bruts de l'API).
    En mode CSV, findings_clean.csv contient déjà les features calculées.
    """

    @staticmethod
    def parse_tags(raw_tags: Any) -> List[str]:
        if not raw_tags:
            return []
        if isinstance(raw_tags, list):
            return [str(t).strip().lower() for t in raw_tags if t]
        if isinstance(raw_tags, str):
            try:
                parsed = json.loads(raw_tags)
                if isinstance(parsed, list):
                    return [str(t).strip().lower() for t in parsed if t]
            except (json.JSONDecodeError, ValueError):
                pass
            return [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        return []

    @staticmethod
    def severity_numeric(severity: str) -> int:
        return SEVERITY_MAP.get(severity.lower().strip(), 0)

    @classmethod
    def extract(cls, finding: Dict) -> Dict[str, Any]:
        tags = cls.parse_tags(finding.get("tags"))
        severity = str(finding.get("severity", "")).lower().strip()
        cvss = float(finding.get("cvss_score") or 0)
        days_open = int(finding.get("days_open") or finding.get("age_days") or 0)
        epss = float(finding.get("epss_score") or 0)
        epss_percentile = float(finding.get("epss_percentile") or 0)
        has_cve = int(bool(finding.get("has_cve") or finding.get("cve")))
        has_cwe = int(bool(finding.get("has_cwe") or finding.get("cwe")))

        tag_urgent       = 1 if any(t in tags for t in ("urgent", "blocker", "p0", "p1")) else 0
        tag_in_production = 1 if any(t in tags for t in ("production", "prod", "prd", "live")) else 0
        tag_sensitive    = 1 if any(t in tags for t in ("sensitive", "pii", "gdpr", "confidential")) else 0
        tag_external     = 1 if any(t in tags for t in ("external", "internet-facing", "public", "exposed")) else 0

        tags_count      = len(tags)
        tags_count_norm = min(tags_count / 20, 1.0)
        cvss_score_norm = cvss / 10.0
        age_days        = days_open
        age_days_norm   = min(age_days / 365, 1.0)
        age_x_cvss      = age_days * cvss
        cvss_x_has_cve  = cvss * has_cve
        epss_x_cvss     = epss * cvss
        has_high_epss   = 1 if epss > 0.5 else 0

        # context_score: max 5 pts — must match preprocess.py exactly
        context_score = min(tag_in_production * 2 + tag_external * 2 + tag_sensitive * 1, 5)
        exposure_norm = context_score / 5.0

        # delay_norm: operational urgency — must match preprocess.py exactly
        days_clipped = min(age_days, 365)
        delay_norm = round(1.0 - (days_clipped / (days_clipped + 30)), 4)

        product_fp_rate = 0.0

        return {
            "cvss_score":        cvss,
            "cvss_score_norm":   cvss_score_norm,
            "has_cve":           has_cve,
            "has_cwe":           has_cwe,
            "epss_score":        epss,
            "epss_percentile":   epss_percentile,
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
            "exposure_norm":     round(exposure_norm, 4),
            "product_fp_rate":   product_fp_rate,
            "cvss_x_has_cve":    cvss_x_has_cve,
            "age_x_cvss":        age_x_cvss,
        }

    @classmethod
    def extract_batch(cls, findings: List[Dict]) -> pd.DataFrame:
        records = [cls.extract(f) for f in findings]
        df = pd.DataFrame(records).fillna(0)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# RISK PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

class RiskPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self._shap_available = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info("Loading model from: %s", self.model_path)
        self.model = joblib.load(self.model_path)

        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)

            self.metadata = ModelMetadata(
                version=raw.get("version", self.model_path.stem),
                feature_columns=raw.get("feature_columns", EXPECTED_FEATURES),
                metrics=raw.get("metrics", {}),
                trained_at=raw.get("trained_at", "unknown"),
                model_type=raw.get("model_type", "unknown"),
            )

            # reverse_mapping: encoded class index → real class index
            # Only needed if train.py used a MERGE_MAP that remapped labels.
            # Standard train.py with CLASS_LABELS={0:Low,1:Med,2:High,3:Crit}
            # uses identity mapping — reverse_mapping will be empty or absent.
            self.reverse_mapping = raw.get("reverse_mapping", {})
            if self.reverse_mapping:
                self.reverse_mapping = {int(k): int(v) for k, v in self.reverse_mapping.items()}
                logger.info("Label reverse mapping loaded: %s", self.reverse_mapping)
            else:
                logger.info("No reverse mapping — standard 4-class: 0=Low 1=Med 2=High 3=Crit")

            self._build_class_labels()

            f1 = self.metadata.metrics.get("test_f1_weighted", 0)
            logger.info(
                "Model loaded — version=%s type=%s F1=%.4f",
                self.metadata.version,
                self.metadata.model_type,
                f1,
            )
        else:
            logger.warning("Metadata file not found: %s", meta_path)
            self.metadata = ModelMetadata(
                version=self.model_path.stem,
                feature_columns=EXPECTED_FEATURES,
                metrics={},
                trained_at="unknown",
                model_type="unknown",
            )
            self.reverse_mapping = {}
            self._build_class_labels()

        try:
            import shap  # noqa: F401
            self._shap_available = True
            logger.debug("SHAP library available")
        except ImportError:
            logger.info("SHAP not installed — feature explanations disabled")

    def _build_class_labels(self) -> None:
        """
        Builds encoded_class_index → human-readable label mapping.

        With a standard train.py (no MERGE_MAP / no reverse_mapping):
          encoded 0 → Low, 1 → Medium, 2 → High, 3 → Critical

        With a reverse_mapping (e.g. MERGE_MAP remapped labels before encoding):
          encoded index → real class index → RISK_LEVELS label
        """
        if self.reverse_mapping:
            self._class_labels = {}
            for enc, real in self.reverse_mapping.items():
                if real == 4:
                    self._class_labels[enc] = "critical"
                else:
                    self._class_labels[enc] = RISK_LEVELS.get(real, f"class_{real}")
        else:
            # Standard case matching train.py CLASS_LABELS
            self._class_labels = {i: RISK_LEVELS.get(i, f"class_{i}") for i in range(4)}

        logger.info("Class labels mapping: %s", self._class_labels)

    # ── SHAP model extraction ─────────────────────────────────────────────────

    # Tree-based class name fragments that shap.TreeExplainer supports natively.
    _TREE_TYPES = (
        "RandomForest", "ExtraTrees", "GradientBoosting",
        "XGB", "LGB", "LGBM", "CatBoost", "DecisionTree",
        "HistGradientBoosting",
    )

    @classmethod
    def _is_tree_based(cls, model: Any) -> bool:
        return any(t in type(model).__name__ for t in cls._TREE_TYPES)

    def _unwrap_to_tree_model(self, model: Any) -> Any:
        
        if getattr(self, "_unwrap_depth", 0) > 10:
            logger.warning("SHAP unwrap depth limit reached")
            return None
        self._unwrap_depth = getattr(self, "_unwrap_depth", 0) + 1

        model_name = type(model).__name__.lower()

        # 1. StackingClassifier → chercher le meilleur estimator
        if "stackingclassifier" in model_name:
            best_estimator = None
            best_priority = -1
            priority = {"lightgbm": 3, "lgbm": 3, "xgboost": 2, "xgb": 2, 
                       "randomforest": 1, "random forest": 1}

            for name, estimator in model.estimators_:
                if isinstance(estimator, tuple):
                    estimator = estimator[1]
                est_name = type(estimator).__name__.lower()

                # Cherche le meilleur selon la priorité
                for key, prio in priority.items():
                    if key in est_name and prio > best_priority:
                        best_priority = prio
                        best_estimator = estimator
                        break

            if best_estimator is not None:
                self._unwrap_depth = 0
                logger.debug("SHAP — Selected best base estimator: %s", type(best_estimator).__name__)
                return best_estimator

            # Fallback : premier estimator
            first = model.estimators_[0]
            if isinstance(first, tuple):
                first = first[1]
            return self._unwrap_to_tree_model(first)

        # 2. Pipeline
        if hasattr(model, "named_steps"):
            last = list(model.named_steps.values())[-1]
            return self._unwrap_to_tree_model(last)

        # 3. CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_"):
            cc = model.calibrated_classifiers_[0]
            inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if inner is not None:
                return self._unwrap_to_tree_model(inner)

        self._unwrap_depth = 0
        return model

    def _align_features(self, df: pd.DataFrame, source: str = "api") -> pd.DataFrame:
        expected = self.metadata.feature_columns if self.metadata else EXPECTED_FEATURES

        # ── Data-leakage guard (defensive, CsvFindingsLoader already strips these) ──
        leaked = TRAIN_EXCLUDED_COLS.intersection(df.columns)
        if leaked:
            logger.warning(
                "_align_features: dropping leaked columns: %s", sorted(leaked)
            )
            df = df.drop(columns=list(leaked))

        missing = [c for c in expected if c not in df.columns]
        extra   = [c for c in df.columns if c not in expected]

        if missing:
            logger.warning("Missing features (filled with 0): %s", missing)
            for col in missing:
                df[col] = 0.0

        if extra and source == "api":
            logger.debug("Extra columns ignored: %s", extra[:5])

        # Exact column order + float dtype — identical to training
        result = df[expected].copy().astype(float)
        return result

    # ── SHAP computation ──────────────────────────────────────────────────────

    def _compute_shap(
        self, X: pd.DataFrame, top_n: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Compute SHAP values using the unwrapped base tree model.

        For StackingClassifier: always uses a BASE estimator (e.g. LGBMClassifier),
        never the meta-learner — the meta-learner operates on OOF predictions,
        not on original features, so its SHAP values would be meaningless.

        SHAP array formats handled:
          list of (n_samples, n_features)   → one array per class (common in RF/LGBM)
          (n_samples, n_features, n_classes) → 3-D array (some XGBoost versions)
          (n_samples, n_features)            → single 2-D array (binary / regression)
        """
        if not self._shap_available:
            return [[] for _ in range(len(X))]

        try:
            import shap

            tree_model = self._unwrap_to_tree_model(self.model)

            if tree_model is None:
                logger.warning(
                    "SHAP — no tree-based estimator found in model pipeline. "
                    "Model type: %s. SHAP disabled.",
                    type(self.model).__name__,
                )
                return [[] for _ in range(len(X))]

            logger.debug(
                "SHAP — using base estimator: %s", type(tree_model).__name__
            )

            explainer = shap.TreeExplainer(tree_model)

            # check_additivity=False: calibration shifts probabilities so the
            # raw tree's SHAP sum won't exactly match the pipeline's output.
            shap_values = explainer.shap_values(X, check_additivity=False)

            feature_names = X.columns.tolist()
            n_features    = len(feature_names)

            # Predicted classes from the FULL pipeline (not the raw base model)
            predicted_classes = self.model.predict(X).astype(int)

            results = []
            for i in range(len(X)):
                # ── Select the SHAP slice for the predicted class ───────────
                if isinstance(shap_values, list):
                    # list[n_classes] of (n_samples, n_features)
                    pred_cls = int(predicted_classes[i])
                    cls_idx  = pred_cls if pred_cls < len(shap_values) else 0
                    sv = shap_values[cls_idx][i]
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                    # (n_samples, n_features, n_classes)
                    pred_cls = int(predicted_classes[i])
                    cls_idx  = pred_cls if pred_cls < shap_values.shape[2] else 0
                    sv = shap_values[i, :, cls_idx]
                else:
                    # (n_samples, n_features)
                    sv = shap_values[i]

                if len(sv) != n_features:
                    logger.warning(
                        "SHAP row %d: sv length %d != n_features %d — skipping",
                        i, len(sv), n_features,
                    )
                    results.append([])
                    continue

                top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
                top_features = [
                    {
                        "feature":    feature_names[j],
                        "value":      round(float(X.iloc[i, j]), 4),
                        "shap_value": round(float(sv[j]), 4),
                        "direction":  "+" if sv[j] > 0 else "-",
                    }
                    for j in top_idx
                ]
                results.append(top_features)

            return results

        except Exception as exc:
            logger.warning(
                "SHAP computation failed: %s — model=%s",
                str(exc), type(self.model).__name__,
                exc_info=True,
            )
            return [[] for _ in range(len(X))]

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _compute_ai_score(
        self,
        encoded_class: int,
        proba: np.ndarray,
        row: pd.Series,
        n_classes: int,
    ) -> Tuple[float, float, float]:          # <-- retourne (final, base, nudge)
        anchors = CLASS_SCORE_ANCHORS[:n_classes]
        model_score = float(np.dot(proba, anchors))

        cvss_norm = float(row.get("cvss_score_norm", 0.0))
        epss      = float(row.get("epss_score", 0.0))
        context   = float(row.get("context_score", 0.0))
        context_norm = context / 5.0

        nudge_raw = (
            0.40 * (cvss_norm - 0.5) +
            0.40 * (min(epss * 2.0, 1.0) - 0.5) +
            0.20 * (context_norm - 0.5)
        )
        nudge = nudge_raw * BUSINESS_NUDGE_MAX * 2.0
        nudge = float(np.clip(nudge, -BUSINESS_NUDGE_MAX, BUSINESS_NUDGE_MAX))

        ai_score = model_score + nudge
        ai_score = round(float(np.clip(ai_score, 0.0, 10.0)), 2)
        return ai_score, round(model_score, 4), round(nudge, 4)

    # ── Batch entry points ────────────────────────────────────────────────────

    def predict_batch_from_csv(
        self,
        findings: List[Dict],
        compute_shap: bool = True,
    ) -> List[PredictionResult]:
        """
        CSV mode: features already computed by preprocess.py.
        Does NOT call FeatureExtractor.
        """
        if not findings:
            return []

        df_raw = pd.DataFrame(findings)
        logger.info(
            "CSV mode — using preprocessed features (%d rows, %d cols)",
            len(df_raw), len(df_raw.columns),
        )
        X = self._align_features(df_raw, source="csv")
        return self._run_prediction(findings, X, compute_shap)

    def predict_batch(
        self,
        findings: List[Dict],
        compute_shap: bool = True,
    ) -> List[PredictionResult]:
        """
        DefectDojo mode: recomputes features from raw findings via FeatureExtractor.
        """
        if not findings:
            return []

        X_raw = FeatureExtractor.extract_batch(findings)
        X = self._align_features(X_raw, source="api")
        return self._run_prediction(findings, X, compute_shap)

    def _run_prediction(
        self,
        findings: List[Dict],
        X: pd.DataFrame,
        compute_shap: bool,
    ) -> List[PredictionResult]:
        """
        Core inference loop.

        Scoring is model-first:
          ai_risk_score = E[score | proba] + small_business_nudge
        See _compute_ai_score() for full documentation.
        """
        encoded_classes = self.model.predict(X).astype(int)
        probas          = self.model.predict_proba(X)
        shap_results    = self._compute_shap(X) if compute_shap else [[] for _ in findings]

        model_version = self.metadata.version if self.metadata else "unknown"
        now           = datetime.now(timezone.utc).isoformat()
        n_classes     = probas.shape[1]

        # Distribution log
        unique, counts_arr = np.unique(encoded_classes, return_counts=True)
        dist = {
            self._class_labels.get(int(u), str(u)): int(c)
            for u, c in zip(unique, counts_arr)
        }
        logger.info("Prediction distribution: %s", dist)

        # Probability column labels (index → human label)
        prob_labels = [self._class_labels.get(i, f"class_{i}") for i in range(n_classes)]

        results: List[PredictionResult] = []

        for idx, finding in enumerate(findings):
            encoded    = int(encoded_classes[idx])
            proba      = probas[idx]
            row        = X.iloc[idx]
            confidence = float(np.max(proba))

            ai_score, model_base, nudge_val = self._compute_ai_score(encoded, proba, row, n_classes)
            risk_level = self._class_labels.get(encoded, "unknown")

            proba_dict = {
                prob_labels[j]: round(float(proba[j]), 4)
                for j in range(n_classes)
            }

            tags_raw = finding.get("tags", [])
            tags = (
                FeatureExtractor.parse_tags(tags_raw)
                if isinstance(tags_raw, (str, list))
                else []
            )

            context  = float(row.get("context_score", 0.0))
            exposure = float(row.get("exposure_norm",  0.0))

            results.append(
                PredictionResult(
                    finding_id=int(finding.get("id", idx)),
                    title=str(finding.get("title", f"Finding {idx}")),
                    severity=str(finding.get("severity", "")).lower(),
                    ai_risk_score=ai_score,
                    model_base_score=model_base,
                    business_nudge=nudge_val,
                    ai_risk_level=risk_level,
                    ai_confidence=round(confidence, 4),
                    context_score=round(context, 2),
                    exposure_norm=round(exposure, 2),
                    shap_top_features=shap_results[idx],
                    ai_prediction_timestamp=now,
                    model_version=model_version,
                    tags=tags,
                    probabilities=proba_dict,
                )
            )

        return results


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PUBLISHER
# ─────────────────────────────────────────────────────────────────────────────

class ResultPublisher:
    def __init__(self, client: Optional[DefectDojoClient] = None) -> None:
        self._client = client

    def publish_to_defectdojo(
        self, results: List[PredictionResult], dry_run: bool = False
    ) -> Tuple[int, int]:
        if not self._client:
            logger.warning("DefectDojo client not available — skipping publication")
            return 0, 0

        success = 0
        failures = 0
        for result in results:
            if result.finding_id == 0:
                continue
            new_ai_tag = f"ai-risk-{result.ai_risk_level}"
            clean_tags = [t for t in result.tags if not t.startswith("ai-risk-")] + [new_ai_tag]
            payload = {"tags": clean_tags, "notes": self._build_note(result)}

            if dry_run:
                logger.info(
                    "[DRY-RUN] Would update finding %d -> %s (%.0f%%)",
                    result.finding_id,
                    result.ai_risk_level,
                    result.ai_confidence * 100,
                )
                success += 1
                continue

            ok = self._client.update_finding(result.finding_id, payload)
            if ok:
                success += 1
            else:
                failures += 1

        logger.info("DefectDojo: %d updated, %d failed", success, failures)
        if not dry_run:
            self._save_scores_to_cache(results)
        return success, failures

    @staticmethod
    def publish_to_file(results: List[PredictionResult], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "summary": ResultPublisher._build_summary(results),
            "findings": [r.to_dict() for r in results],
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Predictions saved: %s (%d findings)", output_path, len(results))

    @staticmethod
    def _build_note(result: PredictionResult) -> str:
        lines = [
            f"[InvisiThreat AI] Risk: {result.ai_risk_level.upper()} — Score: {result.ai_risk_score}/10",
            f"Confidence: {result.ai_confidence:.0%} | Context: {result.context_score} pts",
            f"Model: {result.model_version} | {result.ai_prediction_timestamp}",
        ]
        if result.shap_top_features:
            lines.append("Top features:")
            for feat in result.shap_top_features:
                lines.append(
                    f"  {feat['direction']} {feat['feature']}={feat['value']} "
                    f"(SHAP={feat['shap_value']:+.3f})"
                )
        if result.probabilities:
            prob_str = " | ".join(f"{k}:{v:.0%}" for k, v in result.probabilities.items())
            lines.append(f"Probabilities: {prob_str}")
        return "\n".join(lines)

    @staticmethod
    def _build_summary(results: List[PredictionResult]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for r in results:
            counts[r.ai_risk_level] = counts.get(r.ai_risk_level, 0) + 1
        confidences = [r.ai_confidence for r in results]
        return {
            "by_risk_level": counts,
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
            "min_confidence": round(float(np.min(confidences)), 4) if confidences else 0,
        }

    @staticmethod
    def _save_scores_to_cache(results: List[PredictionResult]) -> None:
        cache_file = Path("data/ai_scores_cache.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache: Dict = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        for r in results:
            cache[str(r.finding_id)] = {
                "ai_risk_score":  r.ai_risk_score,
                "ai_risk_level":  r.ai_risk_level,
                "ai_confidence":  r.ai_confidence,
                "context_score":  r.context_score,
                "probabilities":  r.probabilities,
                "shap_top_features":     r.shap_top_features,
                "model_base_score":  r.model_base_score,     
                "business_nudge":    r.business_nudge,       
                "updated_at":     datetime.now(timezone.utc).isoformat(),
            }

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info("Cache saved: %d findings -> %s", len(results), cache_file)


# ─────────────────────────────────────────────────────────────────────────────
# SECURITY GATE
# ─────────────────────────────────────────────────────────────────────────────

class SecurityGate:
    def __init__(self, threshold: float = 7.0) -> None:
        self.threshold = threshold

    def evaluate(self, results: List[PredictionResult], ci_mode: bool = False):
        critical = [r for r in results if r.ai_risk_score >= self.threshold]
        decision = GateDecision(
            blocked=len(critical) > 0,
            threshold=self.threshold,
            critical_count=len(critical),
            critical_findings=[
                {
                    "id":            r.finding_id,
                    "title":         r.title,
                    "ai_risk_score": r.ai_risk_score,
                    "ai_risk_level": r.ai_risk_level,
                    "ai_confidence": r.ai_confidence,
                }
                for r in critical
            ],
            message=(
                f"Security Gate: {len(critical)} finding(s) at or above threshold {self.threshold}"
                if critical
                else "Security Gate: All findings within acceptable risk levels"
            ),
        )

        if decision.blocked:
            logger.warning("=" * 70)
            logger.warning(decision.message)
            for f in decision.critical_findings[:10]:
                logger.warning(
                    "  [%s] %s (score=%.2f, conf=%.0f%%)",
                    f["ai_risk_level"].upper(),
                    f["title"][:70],
                    f["ai_risk_score"],
                    f["ai_confidence"] * 100,
                )
            logger.warning("=" * 70)
            if ci_mode:
                logger.error("CI/CD deployment blocked by Security Gate")
                sys.exit(1)
        else:
            logger.info(decision.message)

        return decision


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PREDICTOR (DefectDojo mode)
# ─────────────────────────────────────────────────────────────────────────────

class LivePredictor:
    def __init__(
        self,
        predictor: RiskPredictor,
        publisher: ResultPublisher,
        gate: SecurityGate,
        dd_client: Optional[DefectDojoClient] = None,
    ) -> None:
        self._predictor = predictor
        self._publisher = publisher
        self._gate      = gate
        self._dd_client = dd_client

    @classmethod
    def from_env(
        cls,
        model_path: Path,
        dd_url: Optional[str] = None,
        dd_api_key: Optional[str] = None,
        gate_threshold: float = 7.0,
    ) -> "LivePredictor":
        resolved_url = dd_url or os.environ.get("DEFECTDOJO_URL", "http://localhost:8080")
        resolved_key = dd_api_key or os.environ.get("DEFECTDOJO_API_KEY", "")
        dd_client = DefectDojoClient(base_url=resolved_url, api_key=resolved_key) if resolved_key else None
        if not dd_client:
            logger.warning("DEFECTDOJO_API_KEY not set — DefectDojo updates disabled")
        return cls(
            RiskPredictor(model_path),
            ResultPublisher(dd_client),
            SecurityGate(gate_threshold),
            dd_client,
        )

    def run(
        self,
        engagement_id: int,
        active_only: bool = True,
        update_dojo: bool = True,
        dry_run: bool = False,
        output_path: Optional[Path] = None,
        ci_mode: bool = False,
        compute_shap: bool = True,
        batch_size: int = 200,
    ) -> List[PredictionResult]:
        if not self._dd_client:
            logger.error("Cannot fetch findings without DefectDojo client")
            return []

        findings = self._dd_client.get_findings(engagement_id, active_only)
        if not findings:
            logger.warning("No findings for engagement %d", engagement_id)
            return []

        all_results: List[PredictionResult] = []
        total_batches = (len(findings) + batch_size - 1) // batch_size
        for bi in range(total_batches):
            batch = findings[bi * batch_size: (bi + 1) * batch_size]
            logger.info("Batch %d/%d (%d findings)", bi + 1, total_batches, len(batch))
            all_results.extend(
                self._predictor.predict_batch(batch, compute_shap=compute_shap)
            )

        self._log_summary(all_results)

        if output_path:
            self._publisher.publish_to_file(all_results, output_path)
        if update_dojo:
            self._publisher.publish_to_defectdojo(all_results, dry_run=dry_run)

        self._gate.evaluate(all_results, ci_mode=ci_mode)
        return all_results

    @staticmethod
    def _log_summary(results: List[PredictionResult]) -> None:
        counts: Dict[str, int] = {}
        for r in results:
            counts[r.ai_risk_level] = counts.get(r.ai_risk_level, 0) + 1
        confidences = [r.ai_confidence for r in results]
        logger.info("Summary (%d findings):", len(results))
        for level in ["critical", "high", "medium", "low", "unknown"]:
            count = counts.get(level, 0)
            if count:
                bar = "█" * min(count, 40)
                logger.info("  %-10s %4d  %s", level.upper(), count, bar)
        if confidences:
            logger.info(
                "  Avg confidence: %.1f%% | Min: %.1f%%",
                np.mean(confidences) * 100,
                np.min(confidences) * 100,
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    for base in (Path(__file__).parent, Path(__file__).parent.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    return p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InvisiThreat AI Risk Scoring Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_argument_group("Source")
    src.add_argument("--use-csv", action="store_true", default=False,
                     help="Load findings from findings_clean.csv (preprocessed features)")
    src.add_argument("--csv-path", default="data/processed/findings_clean.csv")
    src.add_argument("--engagement-id", type=int, help="DefectDojo engagement ID")

    dojo = parser.add_argument_group("DefectDojo")
    dojo.add_argument("--dd-url")
    dojo.add_argument("--dd-api-key")
    dojo.add_argument("--update-dojo", action="store_true", default=False)

    model = parser.add_argument_group("Model")
    model.add_argument("--model-path", default="models/pipeline_latest.pkl")

    pred = parser.add_argument_group("Prediction")
    pred.add_argument("--batch-size", type=int, default=200)
    pred.add_argument("--no-shap", dest="compute_shap", action="store_false", default=True)
    pred.add_argument("--gate-threshold", type=float, default=7.0)

    out = parser.add_argument_group("Output")
    out.add_argument("--output-file")
    out.add_argument("--dry-run", action="store_true", default=False)

    dbg = parser.add_argument_group("Debug")
    dbg.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    dbg.add_argument("--ci-mode", action="store_true", default=False)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("InvisiThreat AI Risk Scoring Engine starting")
    logger.info("Mode: %s", "CSV (preprocessed)" if args.use_csv else "DefectDojo API")

    model_path = _resolve_model_path(args.model_path)

    # ── CSV MODE ──────────────────────────────────────────────────────────────
    if args.use_csv:
        logger.info("=" * 70)
        logger.info("CSV MODE — reading preprocessed features from findings_clean.csv")
        logger.info("=" * 70)

        try:
            loader = CsvFindingsLoader(args.csv_path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Cannot load CSV: %s", exc)
            sys.exit(1)

        try:
            predictor = RiskPredictor(model_path)
        except FileNotFoundError as exc:
            logger.error("Cannot load model: %s", exc)
            sys.exit(2)

        publisher = ResultPublisher(client=None)
        gate      = SecurityGate(threshold=args.gate_threshold)

        findings = loader.get_findings()
        logger.info(
            "Scoring %d findings with model %s…",
            len(findings),
            predictor.metadata.version,
        )

        all_results = predictor.predict_batch_from_csv(
            findings, compute_shap=args.compute_shap
        )

        output_path = (
            Path(args.output_file)
            if args.output_file
            else Path("logs/predictions_csv.json")
        )
        publisher.publish_to_file(all_results, output_path)
        publisher._save_scores_to_cache(all_results)
        gate.evaluate(all_results, ci_mode=args.ci_mode)

        logger.info("Done. Results: %s", output_path)

    # ── DEFECTDOJO MODE ───────────────────────────────────────────────────────
    else:
        logger.info("=" * 70)
        logger.info("PRODUCTION MODE — DefectDojo API")
        logger.info("=" * 70)

        if not args.engagement_id:
            logger.error("--engagement-id required for DefectDojo mode")
            sys.exit(1)

        try:
            live = LivePredictor.from_env(
                model_path=model_path,
                dd_url=args.dd_url,
                dd_api_key=args.dd_api_key,
                gate_threshold=args.gate_threshold,
            )
        except FileNotFoundError as exc:
            logger.error("Cannot load model: %s", exc)
            sys.exit(2)

        output_path = (
            Path(args.output_file)
            if args.output_file
            else Path(f"logs/predictions_eng_{args.engagement_id}.json")
        )
        ci_mode = args.ci_mode or os.environ.get("CI", "").lower() in ("true", "1", "yes")

        live.run(
            engagement_id=args.engagement_id,
            active_only=True,
            update_dojo=args.update_dojo,
            dry_run=args.dry_run,
            output_path=output_path,
            ci_mode=ci_mode,
            compute_shap=args.compute_shap,
            batch_size=args.batch_size,
        )

        logger.info("Production mode completed")


if __name__ == "__main__":
    main()