"""
config.py — Constantes globales, variables d'environnement, FEATURE_COLS.
NE PAS MODIFIER FEATURE_COLS sans réentraîner le modèle.
"""
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── Version & timing ──────────────────────────────────────────────────────────
API_VERSION = "3.2.0"

# ── Chemins modèle ────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline_latest.pkl"))
META_PATH  = Path(os.getenv("META_PATH",  "models/pipeline_latest_meta.json"))

# ── DefectDojo ────────────────────────────────────────────────────────────────
DEFECTDOJO_URL     = os.getenv("DEFECTDOJO_URL", "http://localhost:8080")
DEFECTDOJO_API_KEY = os.getenv("DEFECTDOJO_API_KEY")

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   "60"))

# ── Cache TTL (secondes) ──────────────────────────────────────────────────────
CACHE_TTL_PRODUCTS = 300
CACHE_TTL_TESTS    = 300
CACHE_TTL_FINDINGS = 60
SCORES_CACHE_TTL   = 60

# ── Données CSV ───────────────────────────────────────────────────────────────
CSV_FINDINGS_PATH = Path(os.getenv("CSV_FINDINGS_PATH", "data/processed/findings_clean.csv"))

# ── Labels & couleurs ─────────────────────────────────────────────────────────
RISK_LEVELS:  Dict[int, str] = {0: "low", 1: "medium", 2: "high", 3: "critical"}
CLASS_LABELS: Dict[int, str] = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
CLASS_COLORS: Dict[str, str] = {
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

# ── FEATURE_COLS — IDENTIQUE à train.py ───────────────────────────────────────
# Colonnes interdites supprimées : severity_num, exploit_risk, days_open_high,
# epss_score_norm, score_composite_raw, score_composite_adj,
# cvss_severity_gap, severity_x_active, cvss_x_severity, severity_x_urgent
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