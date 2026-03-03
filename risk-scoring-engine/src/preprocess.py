"""
preprocess.py — AI Risk Engine
================================
Prétraitement complet des vulnérabilités DefectDojo.

Corrections appliquées (v2.1) :
  #1  Cible ML → régression sur composite_risk normalisé (pas fixed_in_30d binaire)
  #2  composite_risk normalisé 0-10 (suppression division par 5)
  #3  Normalisation MinMax de toutes les features (0-1)
  #4  Extraction sémantique des tags (urgent, prod, critical, etc.)
  #5  Clamping 99e percentile sur toutes les features numériques
  #6  Features d'interaction (cvss × severity, cvss × has_cve, etc.)
  #7  Validation croisée K-Fold (non implémentée ici mais données prêtes)
  #8  Cible continue (régression) ou 5 classes ordinales
  #9  [v2.1] Cast product_id / engagement_id stable (int → str uniquement
      pour la jointure, puis recast en Int64 nullable après merge)
  #10 [v2.1] Jointure robuste engagement_name (tolère float/int/str ids)
  #11 [v2.1] product_name garanti présent grâce à fetch_data.py v2 ;
      fallback merge conservé pour compatibilité ascendante
"""

import ast
import json
import logging
import os
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/preprocess.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SEVERITY_MAP = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Info": 0}

URGENT_TAGS      = {"urgent", "critical", "p0", "p1", "blocker", "asap", "emergency"}
PRODUCTION_TAGS  = {"prod", "production", "live", "prd"}
SENSITIVE_TAGS   = {"sensitive", "pii", "gdpr", "confidential", "secret"}
EXTERNAL_TAGS    = {"external", "internet-facing", "public", "exposed"}

KEEP_COLS = [
    "id", "title", "product_id", "engagement_id",
    "product_name", "engagement_name",
    "file_path", "line", "description",
    "severity_num", "cvss_score", "age_days",
    "has_cve", "has_cwe", "tags_count",
    "is_false_positive", "is_active",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "severity_x_active", "product_fp_rate", "cvss_severity_gap",
    "cvss_x_severity", "cvss_x_has_cve", "severity_x_urgent", "age_x_cvss",
    "cvss_score_norm", "severity_norm", "age_days_norm",
    "tags_count_norm", "cvss_severity_gap_norm",
    "composite_risk",
    "risk_score",
    "risk_class",
    "days_to_fix",
]


# ══════════════════════════════════════════════
# 1. CHARGEMENT
# ══════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "findings":    RAW_DIR / "findings_raw.csv",
        "products":    RAW_DIR / "products.csv",
        "engagements": RAW_DIR / "engagements.csv",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")

    findings    = pd.read_csv(files["findings"])
    products    = pd.read_csv(files["products"])
    engagements = pd.read_csv(files["engagements"])

    logger.info(f"Findings bruts  : {len(findings)} lignes")
    logger.info(f"Produits        : {len(products)} lignes")
    logger.info(f"Engagements     : {len(engagements)} lignes")
    return findings, products, engagements


# ══════════════════════════════════════════════
# 2. UTILITAIRES
# ══════════════════════════════════════════════

def safe_col(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


def _normalize_tz(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def clamp_percentile(series: pd.Series, p: float = 99) -> pd.Series:
    upper = series.quantile(p / 100)
    if upper > 0:
        return series.clip(upper=upper)
    return series


def _safe_int_col(df: pd.DataFrame, col: str) -> pd.Series:
    """
    FIX #9 — Convertit une colonne id en Int64 nullable.
    Tolère NaN, float (ex: 2.0), string ('2'), int natif.
    Évite les problèmes de jointure dues aux types mixtes.
    """
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")


def _normalize_id_series(series: pd.Series) -> pd.Series:
    """
    FIX #9/#10 — Normalise une série d'ids en string propre ('1', '2', …)
    pour les jointures. Gère float ('2.0' → '2'), NaN → ''.
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "")
    )


# ══════════════════════════════════════════════
# 3. FEATURES DE DATE
# ══════════════════════════════════════════════

def build_date_features(data: pd.DataFrame) -> pd.DataFrame:
    now = pd.Timestamp(datetime.utcnow())

    date_col = next((c for c in ["date", "created"] if c in data.columns), None)
    if date_col:
        discovery = _normalize_tz(pd.to_datetime(data[date_col], errors="coerce"))
        raw_age   = (now - discovery).dt.days.fillna(0).clip(lower=0)
        data["age_days"] = clamp_percentile(raw_age, 99)
        logger.info(f"age_days : médiane={data['age_days'].median():.0f}j  "
                    f"max_clampé={raw_age.quantile(0.99):.0f}j")
    else:
        data["age_days"] = 0.0
        discovery = None
        logger.warning("Colonne de date absente — age_days forcé à 0")

    fix_col = next((c for c in ["mitigated_date", "last_reviewed"] if c in data.columns), None)
    if fix_col and discovery is not None:
        fix_date            = _normalize_tz(pd.to_datetime(data[fix_col], errors="coerce"))
        data["days_to_fix"] = (fix_date - discovery).dt.days
    else:
        data["days_to_fix"] = np.nan
        logger.warning("days_to_fix absent — ignoré (cible ML = risk_score)")

    return data


# ══════════════════════════════════════════════
# 4. FEATURES DE SÉVÉRITÉ & CVSS
# ══════════════════════════════════════════════

def build_severity_features(data: pd.DataFrame) -> pd.DataFrame:
    if "severity" in data.columns:
        data["severity_num"] = data["severity"].map(SEVERITY_MAP).fillna(0).astype(int)
    else:
        data["severity_num"] = 0

    cvss_col = next((c for c in ["cvssv3_score", "cvss_score"] if c in data.columns), None)
    if cvss_col:
        data["cvss_score"] = pd.to_numeric(data[cvss_col], errors="coerce").fillna(0).clip(0, 10)
    else:
        data["cvss_score"] = 0.0

    data["cvss_score"] = clamp_percentile(data["cvss_score"], 99)

    cvss_norm               = data["cvss_score"] / 10 * 4
    data["cvss_severity_gap"] = (cvss_norm - data["severity_num"]).abs().round(3)
    data["cvss_severity_gap"] = clamp_percentile(data["cvss_severity_gap"], 99)

    return data


# ══════════════════════════════════════════════
# 5. FEATURES BINAIRES & TAGS
# ══════════════════════════════════════════════

def _parse_tags(tag_field) -> list[str]:
    if pd.isna(tag_field):
        return []
    if isinstance(tag_field, list):
        return [str(t).lower().strip() for t in tag_field]
    if isinstance(tag_field, str):
        try:
            parsed = ast.literal_eval(tag_field)
            if isinstance(parsed, list):
                return [str(t).lower().strip() for t in parsed]
        except (ValueError, SyntaxError):
            pass
        return [t.lower().strip() for t in tag_field.split(",") if t.strip()]
    return []


def _has_tag_match(tags: list[str], keyword_set: set[str]) -> int:
    return int(any(t in keyword_set for t in tags))


def build_binary_features(data: pd.DataFrame) -> pd.DataFrame:
    data["has_cve"]            = data["cve"].notna().astype(int) if "cve" in data.columns else 0
    data["has_cwe"]            = data["cwe"].notna().astype(int) if "cwe" in data.columns else 0
    data["is_false_positive"]  = safe_col(data, "false_p", False).fillna(False).astype(int)
    data["is_active"]          = safe_col(data, "active",  True).fillna(True).astype(int)

    parsed_tags = (
        data["tags"].apply(_parse_tags)
        if "tags" in data.columns
        else pd.Series([[] for _ in range(len(data))], index=data.index)
    )

    data["tags_count"]        = clamp_percentile(parsed_tags.apply(len), 99)
    data["tag_urgent"]        = parsed_tags.apply(lambda t: _has_tag_match(t, URGENT_TAGS))
    data["tag_in_production"] = parsed_tags.apply(lambda t: _has_tag_match(t, PRODUCTION_TAGS))
    data["tag_sensitive"]     = parsed_tags.apply(lambda t: _has_tag_match(t, SENSITIVE_TAGS))
    data["tag_external"]      = parsed_tags.apply(lambda t: _has_tag_match(t, EXTERNAL_TAGS))

    return data


# ══════════════════════════════════════════════
# 6. FEATURES CONTEXTUELLES & INTERACTION
# ══════════════════════════════════════════════

def build_contextual_features(data: pd.DataFrame) -> pd.DataFrame:
    data["severity_x_active"] = data["severity_num"] * data["is_active"]
    # FIX #9 : product_id peut être Int64 après recast — groupby tolère Int64
    data["product_fp_rate"] = (
        data.groupby("product_id")["is_false_positive"]
        .transform("mean")
        .round(4)
    )
    return data


def build_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    data["cvss_x_severity"]   = data["cvss_score"] * data["severity_num"]
    data["cvss_x_has_cve"]    = data["cvss_score"] * data["has_cve"]
    data["severity_x_urgent"] = data["severity_num"] * data["tag_urgent"]
    data["age_x_cvss"]        = data["age_days"] * data["cvss_score"]

    for col in ["cvss_x_severity", "cvss_x_has_cve", "age_x_cvss"]:
        data[col] = clamp_percentile(data[col], 99)

    return data


# ══════════════════════════════════════════════
# 7. NORMALISATION
# ══════════════════════════════════════════════

def build_normalized_features(data: pd.DataFrame) -> pd.DataFrame:
    norm_map = {
        "cvss_score":        "cvss_score_norm",
        "severity_num":      "severity_norm",
        "age_days":          "age_days_norm",
        "tags_count":        "tags_count_norm",
        "cvss_severity_gap": "cvss_severity_gap_norm",
    }
    for src, dst in norm_map.items():
        if src in data.columns:
            col_min, col_max = data[src].min(), data[src].max()
            data[dst] = (
                ((data[src] - col_min) / (col_max - col_min)).round(4)
                if col_max > col_min else 0.0
            )
        else:
            data[dst] = 0.0
    return data


# ══════════════════════════════════════════════
# 8. SCORE COMPOSITE & CIBLE ML
# ══════════════════════════════════════════════

def build_composite_risk(data: pd.DataFrame) -> pd.DataFrame:
    raw_score = (
        data["cvss_score"]        * 0.35 +
        data["severity_num"]      * 1.50 +
        (data["age_days"] / 30)   * 0.20 +
        data["has_cve"]           * 2.00 +
        data["tag_urgent"]        * 1.50 +
        data["tag_in_production"] * 1.20 +
        data["tag_external"]      * 0.80 +
        data["tag_sensitive"]     * 0.70 +
        data["tags_count"]        * 0.05
    )

    score_min, score_max = raw_score.min(), raw_score.max()
    data["composite_risk"] = (
        ((raw_score - score_min) / (score_max - score_min) * 10).round(3)
        if score_max > score_min else 5.0
    )

    logger.info(
        f"composite_risk : min={data['composite_risk'].min():.2f}  "
        f"max={data['composite_risk'].max():.2f}  "
        f"médiane={data['composite_risk'].median():.2f}"
    )
    return data


def build_ml_target(data: pd.DataFrame) -> pd.DataFrame:
    data["risk_score"] = data["composite_risk"].round(3)

    bins   = [-np.inf, 2.0, 4.0, 6.0, 8.0, np.inf]
    labels = [0, 1, 2, 3, 4]
    data["risk_class"] = pd.cut(
        data["risk_score"], bins=bins, labels=labels, right=True
    ).astype(int)

    dist = data["risk_class"].value_counts().sort_index()
    logger.info("Distribution risk_class :\n" + dist.to_string())
    logger.info(
        f"risk_score : min={data['risk_score'].min():.2f}  "
        f"max={data['risk_score'].max():.2f}  "
        f"écart-type={data['risk_score'].std():.2f}"
    )
    return data


# ══════════════════════════════════════════════
# 9. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════

def preprocess_findings(
    findings:    pd.DataFrame,
    products:    pd.DataFrame,
    engagements: pd.DataFrame,
) -> pd.DataFrame:
    data = findings.copy()

    # ── FIX #9 : ids stables Int64 dès le début ──────────────────────────
    data["product_id"]    = _safe_int_col(data, "product_id")
    data["engagement_id"] = _safe_int_col(data, "engagement_id")

    # ── product_name ──────────────────────────────────────────────────────
    if "product_name" in data.columns and data["product_name"].notna().any():
        # fetch_data.py v2 garantit déjà ce champ — chemin nominal
        logger.info("product_name présent dans les données brutes (chemin nominal)")
        # Combler les éventuels NaN résiduels par jointure
        mask_missing = data["product_name"].isna()
        if mask_missing.any():
            logger.warning(f"{mask_missing.sum()} product_name manquants — complétion par jointure")
            _fill_product_name(data, products, mask_missing)
    else:
        logger.warning("product_name absent — jointure complète avec products.csv")
        _fill_product_name(data, products, pd.Series(True, index=data.index))

    # ── engagement_name ───────────────────────────────────────────────────
    # FIX #10 : fetch_data.py n'exporte pas engagement_name → jointure robuste
    if "engagement_name" not in data.columns or data["engagement_name"].isna().all():
        logger.info("Résolution engagement_name via engagements.csv")
        _fill_engagement_name(data, engagements)
    else:
        mask_missing = data["engagement_name"].isna()
        if mask_missing.any():
            _fill_engagement_name_partial(data, engagements, mask_missing)

    # ── Pipeline features ─────────────────────────────────────────────────
    data = build_date_features(data)
    data = build_severity_features(data)
    data = build_binary_features(data)
    data = build_contextual_features(data)
    data = build_interaction_features(data)
    data = build_normalized_features(data)
    data = build_composite_risk(data)
    data = build_ml_target(data)

    final_cols = [c for c in KEEP_COLS if c in data.columns]
    result = data[final_cols].copy()

    # FIX #9 : recast final pour export propre
    for id_col in ["product_id", "engagement_id"]:
        if id_col in result.columns:
            result[id_col] = result[id_col].astype("Int64")

    logger.info(f"Findings nettoyés : {len(result)} lignes × {len(result.columns)} colonnes")
    return result


def _fill_product_name(
    data:     pd.DataFrame,
    products: pd.DataFrame,
    mask:     pd.Series,
) -> None:
    """
    FIX #9/#11 — Jointure product_name robuste.
    Normalise les ids en string propre avant le merge pour éviter
    les échecs silencieux liés aux types (int vs float vs str).
    """
    if "id" not in products.columns or "name" not in products.columns:
        logger.warning("products.csv sans colonnes id/name — product_name non résolu")
        data.loc[mask, "product_name"] = data.loc[mask, "product_name"].fillna("Unknown")
        return

    prod_lookup = (
        products[["id", "name"]]
        .assign(pid_str=lambda d: _normalize_id_series(d["id"]))
        .drop_duplicates("pid_str")
        .set_index("pid_str")["name"]
    )

    data.loc[mask, "product_name"] = (
        _normalize_id_series(data.loc[mask, "product_id"])
        .map(prod_lookup)
        .fillna("Unknown")
        .values
    )

    resolved = data.loc[mask, "product_name"].ne("Unknown").sum()
    logger.info(f"product_name résolu pour {resolved}/{mask.sum()} findings")


def _fill_engagement_name(data: pd.DataFrame, engagements: pd.DataFrame) -> None:
    """
    FIX #10 — Jointure engagement_name robuste (même logique de normalisation).
    """
    data["engagement_name"] = ""

    if "id" not in engagements.columns or "name" not in engagements.columns:
        logger.warning("engagements.csv sans colonnes id/name — engagement_name vide")
        return

    eng_lookup = (
        engagements[["id", "name"]]
        .assign(eid_str=lambda d: _normalize_id_series(d["id"]))
        .drop_duplicates("eid_str")
        .set_index("eid_str")["name"]
    )

    data["engagement_name"] = (
        _normalize_id_series(data["engagement_id"])
        .map(eng_lookup)
        .fillna("")
        .values
    )

    resolved = data["engagement_name"].ne("").sum()
    logger.info(f"engagement_name résolu pour {resolved}/{len(data)} findings")


def _fill_engagement_name_partial(
    data:        pd.DataFrame,
    engagements: pd.DataFrame,
    mask:        pd.Series,
) -> None:
    _fill_engagement_name(
        data.loc[mask].copy(),  # travaille sur une copie locale
        engagements,
    )
    # re-injecte dans data original
    tmp = data.loc[mask, "engagement_id"].copy().to_frame()
    tmp["engagement_name"] = ""
    _fill_engagement_name(tmp, engagements)
    data.loc[mask, "engagement_name"] = tmp["engagement_name"].values


# ══════════════════════════════════════════════
# 10. VALIDATION
# ══════════════════════════════════════════════

def validate_output(df: pd.DataFrame) -> bool:
    ok = True

    required = ["severity_num", "cvss_score", "age_days", "risk_score", "risk_class"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Colonnes obligatoires manquantes : {missing}")
        ok = False

    if len(df) == 0:
        logger.error("DataFrame vide après prétraitement.")
        ok = False

    if "risk_score" in df.columns:
        if df["risk_score"].max() > 10.01 or df["risk_score"].min() < -0.01:
            logger.error(
                f"risk_score hors plage [0-10] : "
                f"min={df['risk_score'].min():.3f}  max={df['risk_score'].max():.3f}"
            )
            ok = False
        if df["risk_score"].std() < 0.01:
            logger.warning("risk_score quasi-constant — modèle ne pourra pas apprendre.")

    # Vérification distribution par produit
    if "product_name" in df.columns:
        dist = df["product_name"].value_counts()
        unknown_pct = (df["product_name"] == "Unknown").mean() * 100
        logger.info(f"Distribution findings par produit :\n{dist.to_string()}")
        if unknown_pct > 10:
            logger.warning(
                f"{unknown_pct:.1f}% de findings avec product_name='Unknown' "
                f"— vérifier fetch_data.py (enrich_findings)"
            )

    null_pct  = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 30]
    if not high_null.empty:
        logger.warning(f"Colonnes avec >30% de NaN :\n{high_null.to_string()}")

    return ok


# ══════════════════════════════════════════════
# 11. SAUVEGARDE & RAPPORT
# ══════════════════════════════════════════════

def save_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", dir=path.parent) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    logger.info(f"💾 Sauvegardé : {path} ({len(df)} lignes)")


def save_data_report(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)
    report = {
        "timestamp":       datetime.now().isoformat(),
        "n_rows":          len(df),
        "n_cols":          len(df.columns),
        "columns":         list(df.columns),
        "null_pct":        df.isnull().mean().round(4).to_dict(),
        "risk_score_stats": {
            "min":    round(float(df["risk_score"].min()), 3),
            "max":    round(float(df["risk_score"].max()), 3),
            "mean":   round(float(df["risk_score"].mean()), 3),
            "std":    round(float(df["risk_score"].std()), 3),
            "median": round(float(df["risk_score"].median()), 3),
        } if "risk_score" in df.columns else {},
        "risk_class_dist": (
            df["risk_class"].value_counts().sort_index().to_dict()
            if "risk_class" in df.columns else {}
        ),
        "severity_dist": (
            df["severity_num"].value_counts().sort_index().to_dict()
            if "severity_num" in df.columns else {}
        ),
        "product_dist": (
            df["product_name"].value_counts().to_dict()
            if "product_name" in df.columns else {}
        ),
        "tag_stats": {
            k: int(df[k].sum()) if k in df.columns else 0
            for k in ["tag_urgent", "tag_in_production", "tag_sensitive", "tag_external"]
        },
    }
    report_path = PROCESSED_DIR / "data_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📋 Rapport qualité : {report_path}")


# ══════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("🧹 Prétraitement — AI Risk Engine v2.1")
    logger.info("=" * 60)

    findings, products, engagements = load_data()
    df_clean = preprocess_findings(findings, products, engagements)

    if not validate_output(df_clean):
        logger.error("Validation échouée — vérifiez les données brutes.")
        raise SystemExit(1)

    output_path = PROCESSED_DIR / "findings_clean.csv"
    save_atomic(df_clean, output_path)
    save_data_report(df_clean)

    logger.info("\nAperçu des 3 premières lignes :")
    logger.info(f"\n{df_clean.head(3).to_string()}")

    logger.info("\nStatistiques numériques :")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    logger.info(f"\n{df_clean[numeric_cols].describe().round(3).to_string()}")

    logger.info("\nPrétraitement v2.1 terminé avec succès.")


if __name__ == "__main__":
    main()