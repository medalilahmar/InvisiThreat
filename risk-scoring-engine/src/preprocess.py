import json
import logging
import re
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT_DIR      = Path(__file__).resolve().parent.parent
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

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



SEVERITY_MAP = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1, "Info": 0}

MERGE_MAP          = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
MERGED_CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}

URGENT_TAGS     = {"urgent", "critical", "p0", "p1", "blocker", "asap", "emergency"}
PRODUCTION_TAGS = {"prod", "production", "live", "prd"}
SENSITIVE_TAGS  = {"sensitive", "pii", "gdpr", "confidential", "secret"}
EXTERNAL_TAGS   = {"external", "internet-facing", "public", "exposed"}

EPSS_THRESHOLD   = 0.5
CVE_PATTERN      = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
EXPOSURE_MAX     = 5.0
DELAY_CLIP_MAX   = 365
MIN_ROWS_PER_CLASS = 30



FEATURE_COLS = [
    "cvss_score",
    "cvss_score_norm",
    "has_cve",
    "has_cwe",
    "epss_score",
    "epss_percentile",
    "has_high_epss",
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
    "epss_x_cvss",
]

EXCLUDE_FROM_ML = [
    "days_to_fix",
    "risk_class",
    "risk_score",
    "severity_num",
    "is_false_positive",
    "is_mitigated",
    "out_of_scope",
    "label_source",
    "score_composite_raw",
    "cvss_x_severity",
    "severity_x_active",
    "severity_x_urgent",
    "cvss_severity_gap",
]

KEEP_COLS = FEATURE_COLS + [
    "id", "title", "product_id", "engagement_id",
    "product_name", "engagement_name", "file_path", "line", "description",
    "is_false_positive", "is_active", "is_mitigated", "out_of_scope",
    "severity", "severity_num",
    "cvss_severity_gap", "severity_x_active", "cvss_x_severity", "severity_x_urgent",
    "score_composite_raw",
    "risk_score", "risk_class", "label_source", "days_to_fix", "created",
    "cve", "sample_weight",
]



def load_data() -> tuple:
    files = {
        "findings":    RAW_DIR / "findings_raw.csv",
        "products":    RAW_DIR / "products.csv",
        "engagements": RAW_DIR / "engagements.csv",
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier manquant : {path}\n  -> Lancer fetch_data.py d'abord"
            )
    findings    = pd.read_csv(files["findings"])
    products    = pd.read_csv(files["products"])
    engagements = pd.read_csv(files["engagements"])
    logger.info(
        f"Donnees chargees : {len(findings)} findings | "
        f"{len(products)} produits | {len(engagements)} engagements"
    )
    return findings, products, engagements


# ─── UTILITAIRES ──────────────────────────────────────────────────────────────

def safe_col(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


def _normalize_tz(series: pd.Series) -> pd.Series:
    if series.dt.tz is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series


def clamp_percentile(series: pd.Series, p: float = 99) -> pd.Series:
    upper = series.quantile(p / 100)
    return series.clip(upper=upper) if upper > 0 else series


def _safe_int_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")


def _normalize_id_series(series: pd.Series) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64").astype(str).replace("<NA>", "")
    )


def robust_minmax(series: pd.Series, p_low: float = 1, p_high: float = 99) -> pd.Series:
    lo = series.quantile(p_low  / 100)
    hi = series.quantile(p_high / 100)
    if hi <= lo:
        return pd.Series(0.0, index=series.index)
    return ((series.clip(lower=lo, upper=hi) - lo) / (hi - lo)).round(4)


# ─── ETAPE 3 : FEATURES TEMPORELLES ──────────────────────────────────────────

def build_date_features(data: pd.DataFrame) -> pd.DataFrame:
    now      = pd.Timestamp(datetime.utcnow())
    date_col = next((c for c in ["date", "created"] if c in data.columns), None)

    if date_col:
        discovery        = _normalize_tz(pd.to_datetime(data[date_col], errors="coerce"))
        data["age_days"] = clamp_percentile(
            (now - discovery).dt.days.fillna(0).clip(lower=0), 99
        )
    else:
        data["age_days"] = 0.0
        discovery        = None
        logger.warning("Aucune colonne date trouvee — age_days = 0")

    fix_col = next(
        (c for c in ["mitigated_date", "last_reviewed"] if c in data.columns), None
    )
    if fix_col and discovery is not None:
        fix_date            = _normalize_tz(pd.to_datetime(data[fix_col], errors="coerce"))
        data["days_to_fix"] = (fix_date - discovery).dt.days
    else:
        data["days_to_fix"] = np.nan

    dtf_count = int(data["days_to_fix"].notna().sum())
    logger.info(
        f"days_to_fix disponible : {dtf_count}/{len(data)} "
        f"({dtf_count / len(data) * 100:.1f}%)"
    )
    if dtf_count < 50:
        logger.warning(
            f"Seulement {dtf_count} findings ont days_to_fix — "
            "delay_norm utilisera la mediane comme fallback"
        )
    return data


# ─── ETAPE 4 : CVSS / SEVERITE ────────────────────────────────────────────────

def build_severity_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    severity_num : mapping ordinal DefectDojo — utilise comme source du label uniquement
    cvss_score   : score technique [0-10] — feature ML independante
    cvss_severity_gap : signal de discordance scanner/analyste — audit uniquement
    """
    data["severity_num"] = (
        data["severity"].map(SEVERITY_MAP).fillna(0).astype(int)
        if "severity" in data.columns else 0
    )
    cvss_col = next(
        (c for c in ["cvssv3_score", "cvssv4_score", "cvss_score"] if c in data.columns),
        None
    )
    if cvss_col:
        # CVSS est borne [0,10] par standard — clip suffit, clamp_percentile inutile
        data["cvss_score"] = (
            pd.to_numeric(data[cvss_col], errors="coerce").fillna(0).clip(0, 10)
        )
    else:
        data["cvss_score"] = 0.0
        logger.warning("Aucun score CVSS trouve — cvss_score = 0")

    # cvss_severity_gap : derive de severity_num -> exclue de FEATURE_COLS, audit uniquement
    cvss_norm_4               = data["cvss_score"] / 10 * 4
    data["cvss_severity_gap"] = clamp_percentile(
        (cvss_norm_4 - data["severity_num"]).abs().round(3), 99
    )
    return data


# ─── EXTRACTION CVE VECTORISEE ────────────────────────────────────────────────

def _extract_cve_vectorized(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(CVE-\d{4}-\d{4,})", flags=re.IGNORECASE, expand=False)
        .str.upper()
    )


# ─── ETAPE 5 : FEATURES BINAIRES & TAGS ──────────────────────────────────────

def _parse_tags(tag_field) -> list:
    if pd.isna(tag_field):
        return []
    if isinstance(tag_field, list):
        return [str(t).lower().strip() for t in tag_field]
    if isinstance(tag_field, str):
        s = tag_field.strip()
        for parser in (json.loads, __import__("ast").literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [str(t).lower().strip() for t in parsed]
                if isinstance(parsed, str):
                    return [parsed.lower().strip()] if parsed.strip() else []
            except Exception:
                continue
        return [t.lower().strip() for t in s.split(",") if t.strip()]
    return []


def _has_tag_match(tags: list, kw: set) -> int:
    return int(any(t in kw for t in tags))


def build_binary_features(data: pd.DataFrame) -> pd.DataFrame:
    cve_series = pd.Series(pd.NA, index=data.index, dtype="object")
    for field in ("cve", "vulnerability_ids", "title"):
        if field in data.columns:
            cve_series = cve_series.combine_first(
                _extract_cve_vectorized(data[field].fillna(""))
            )
    data["cve"]     = cve_series
    data["has_cve"] = data["cve"].notna().astype(int)
    data["has_cwe"] = data["cwe"].notna().astype(int) if "cwe" in data.columns else 0

    data["is_false_positive"] = safe_col(data, "false_p", False).fillna(False).astype(int)
    data["is_active"]         = safe_col(data, "active",  True ).fillna(True ).astype(int)
    data["is_mitigated"]      = safe_col(data, "is_mitigated", False).fillna(False).astype(bool)
    data["out_of_scope"]      = safe_col(data, "out_of_scope", False).fillna(False).astype(bool)

    parsed_tags = (
        data["tags"].apply(_parse_tags) if "tags" in data.columns
        else pd.Series([[] for _ in range(len(data))], index=data.index)
    )
    data["tags_count"]        = clamp_percentile(parsed_tags.apply(len), 99)
    data["tag_urgent"]        = parsed_tags.apply(lambda t: _has_tag_match(t, URGENT_TAGS))
    data["tag_in_production"] = parsed_tags.apply(lambda t: _has_tag_match(t, PRODUCTION_TAGS))
    data["tag_sensitive"]     = parsed_tags.apply(lambda t: _has_tag_match(t, SENSITIVE_TAGS))
    data["tag_external"]      = parsed_tags.apply(lambda t: _has_tag_match(t, EXTERNAL_TAGS))

    cve_pct = data["has_cve"].mean() * 100
    logger.info(f"CVE detectees : {int(data['has_cve'].sum())}/{len(data)} ({cve_pct:.1f}%)")
    return data


# ─── ETAPE 6 : FEATURES EPSS ──────────────────────────────────────────────────

def build_epss_features(data: pd.DataFrame) -> pd.DataFrame:
    data["epss_score"] = (
        pd.to_numeric(data.get("epss_score", 0), errors="coerce")
        .fillna(0.0).clip(0, 1)
    )
    data["epss_percentile"] = (
        pd.to_numeric(data.get("epss_percentile", 0), errors="coerce")
        .fillna(0.0).clip(0, 1)
    )
    data["has_high_epss"] = (data["epss_score"] > EPSS_THRESHOLD).astype(int)
    data["epss_x_cvss"]   = (data["epss_score"] * data["cvss_score"]).round(4)

    epss_positive = int((data["epss_score"] > 0).sum())
    logger.info(f"EPSS > 0 : {epss_positive}/{len(data)} findings")
    if epss_positive == 0:
        logger.warning("Tous les scores EPSS sont a 0 — verifier fetch_data.py et le cache EPSS")
    return data


# ─── ETAPE 7 : FEATURES CONTEXTUELLES ────────────────────────────────────────

def build_contextual_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    severity_x_active : derive de severity_num -> EXCLUDE_FROM_ML, audit uniquement
    product_fp_rate   : lissage Laplace (sum+1)/(count+2) — feature ML stable
    """
    data["severity_x_active"] = data["severity_num"] * data["is_active"]

    def laplace_fp_rate(x: pd.Series) -> float:
        return (x.sum() + 1) / (len(x) + 2)

    data["product_fp_rate"] = (
        data.groupby("product_id")["is_false_positive"]
        .transform(laplace_fp_rate)
        .round(4)
    )
    fp_min = data["product_fp_rate"].min()
    fp_max = data["product_fp_rate"].max()
    logger.info(f"product_fp_rate (Laplace) : min={fp_min:.4f} | max={fp_max:.4f}")
    return data


# ─── ETAPE 8 : FEATURES D'INTERACTION ────────────────────────────────────────

def build_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    cvss_x_severity / severity_x_urgent : derives de severity_num
    -> calcules ici pour audit, exclus de FEATURE_COLS via EXCLUDE_FROM_ML
    cvss_x_has_cve / age_x_cvss         : features ML propres (dans FEATURE_COLS)
    """
    data["cvss_x_severity"]   = data["cvss_score"] * data["severity_num"]
    data["cvss_x_has_cve"]    = data["cvss_score"] * data["has_cve"]
    data["severity_x_urgent"] = data["severity_num"] * data["tag_urgent"]
    data["age_x_cvss"]        = data["age_days"] * data["cvss_score"]

    for col in ["cvss_x_severity", "cvss_x_has_cve", "age_x_cvss"]:
        data[col] = clamp_percentile(data[col], 99)
    return data


# ─── ETAPE 9 : NORMALISATION ──────────────────────────────────────────────────

def build_normalized_features(data: pd.DataFrame) -> pd.DataFrame:
    data["cvss_score_norm"] = (data["cvss_score"] / 10).round(4)
    data["age_days_norm"]   = robust_minmax(data["age_days"])
    data["tags_count_norm"] = robust_minmax(data["tags_count"])
    return data


# ─── ETAPE 10a : FEATURES D'EXPOSITION ───────────────────────────────────────

def build_exposure_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    context_score : ponderation metier SIEM
      external      +2 (risque exploitation externe)
      production    +2 (impact business direct)
      sensitive     +1 (impact reglementaire)
    exposure_norm : context_score / EXPOSURE_MAX -> [0,1]
    """
    data["context_score"] = (
        data["tag_external"]      * 2 +
        data["tag_in_production"] * 2 +
        data["tag_sensitive"]     * 1
    ).astype(int)

    data["exposure_norm"] = (
        data["context_score"].clip(0, EXPOSURE_MAX) / EXPOSURE_MAX
    ).round(4)
    return data


# ─── ETAPE 10b : DELAY NORM ───────────────────────────────────────────────────

def build_delay_norm(data: pd.DataFrame) -> pd.DataFrame:
    """
    Signal SOC d'urgence operationnelle :
      delay_norm = 1 -> correction rapide -> forte priorite operationnelle
      delay_norm = 0 -> correction lente ou non traitee

    Formule : 1 - (days_to_fix / (days_to_fix + 30))
    NaN -> mediane (evite biais si on utilisait 0)
    """
    dtf_raw     = pd.to_numeric(
        data.get("days_to_fix", pd.Series(np.nan, index=data.index)),
        errors="coerce"
    )
    dtf_clipped = dtf_raw.clip(0, DELAY_CLIP_MAX)
    median_val  = dtf_clipped.median()

    if pd.isna(median_val):
        median_val = 30.0
        logger.warning("Mediane days_to_fix indisponible — fallback = 30 jours")

    dtf_filled         = dtf_clipped.fillna(median_val)
    norm               = dtf_filled / (dtf_filled + 30)
    data["delay_norm"] = (1 - norm).clip(0, 1).round(4)

    logger.info(
        f"delay_norm : mean={data['delay_norm'].mean():.3f} | "
        f"min={data['delay_norm'].min():.3f} | "
        f"max={data['delay_norm'].max():.3f}"
    )
    return data


# ─── ETAPE 11 : LABEL + SAMPLE WEIGHTS ───────────────────────────────────────

def build_label_from_severity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Label : severity_num -> MERGE_MAP -> 4 classes (0=Low 1=Med 2=High 3=Crit)
    Exclus : faux positifs et out_of_scope -> risk_class = NaN
    score_composite_adj supprime (identique a raw, residuel v7)
    """
    is_fp      = data.get("is_false_positive",
                           pd.Series(0, index=data.index)).fillna(0).astype(bool)
    is_oos     = data.get("out_of_scope",
                           pd.Series(False, index=data.index)).fillna(False).astype(bool)
    is_invalid = is_fp | is_oos

    risk_class_raw             = data["severity_num"].map(MERGE_MAP)
    risk_class_raw[is_invalid] = np.nan

    data["risk_class"]          = risk_class_raw
    data["label_source"]        = "severity_defectdojo_v9"
    data["score_composite_raw"] = (data["severity_num"] / 4.0).clip(0, 1)
    data["risk_score"]          = (data["score_composite_raw"] * 10).round(3)

    valid_mask     = data["risk_class"].notna()
    active_count   = int(valid_mask.sum())
    excluded_count = int(is_invalid.sum())

    logger.info(
        f"Label : {active_count} findings valides | {excluded_count} exclus (FP/OOS)"
    )

    if active_count > 0:
        dist = data.loc[valid_mask, "risk_class"].value_counts().sort_index()
        logger.info("Distribution risk_class :")
        for cls, count in dist.items():
            name = MERGED_CLASS_NAMES.get(int(cls), "?")
            pct  = count / active_count * 100
            logger.info(f"  {name:<10} (classe {cls}) : {count:5d} ({pct:5.1f}%)")

        dominant_pct = float(dist.max() / active_count)
        if dominant_pct > 0.70:
            dominant_cls = MERGED_CLASS_NAMES.get(int(dist.idxmax()), "?")
            logger.warning(
                f"Classe dominante : {dominant_cls} = {dominant_pct * 100:.1f}% "
                "-> sample_weight compensera ce desequilibre"
            )

        _log_feature_correlations(data, valid_mask)

    return data


def _log_feature_correlations(data: pd.DataFrame, valid_mask: pd.Series) -> None:
    """
    Monitoring correlations features/label.
    severity_num/risk_class : correlation elevee attendue (source du label).
    CVSS/risk_class > 0.95  : warning seulement (correlation structurelle).
    CVSS/EPSS > 0.70        : alerte redondance features.
    """
    rc = data.loc[valid_mask, "risk_class"].astype(float)

    checks = {
        "severity_num": (0.80, 0.99, "attendu eleve — source du label"),
        "cvss_score":   (None, 0.95, "warning si > 0.95 (dominance structurelle)"),
        "epss_score":   (None, 0.60, "signal secondaire"),
        "delay_norm":   (None, 0.50, "signal SOC comportemental"),
    }

    logger.info("Correlations features / risk_class :")
    for col, (low_warn, high_warn, note) in checks.items():
        if col not in data.columns:
            continue
        r    = float(data.loc[valid_mask, col].corr(rc))
        flag = ""
        if high_warn and abs(r) > high_warn:
            flag = " [WARNING : correlation elevee]"
        elif low_warn and abs(r) < low_warn:
            flag = " [WARNING : correlation plus faible qu attendue]"
        logger.info(f"  {col:<20} r = {r:+.4f}  ({note}){flag}")

    if "cvss_score" in data.columns and "epss_score" in data.columns:
        r_ce = float(
            data.loc[valid_mask, "cvss_score"].corr(data.loc[valid_mask, "epss_score"])
        )
        flag = (
            " [WARNING : redondance elevee — verifier poids EPSS]"
            if abs(r_ce) > 0.70 else " [OK]"
        )
        logger.info(f"  cvss_score / epss_score   r = {r_ce:+.4f}{flag}")


def build_sample_weights(data: pd.DataFrame) -> pd.DataFrame:
    valid_mask = data["risk_class"].notna()
    if not valid_mask.any():
        data["sample_weight"] = 1.0
        return data

    class_counts = data.loc[valid_mask, "risk_class"].value_counts()
    total        = valid_mask.sum()
    n_classes    = len(class_counts)
    weight_map   = {
        cls: min(total / (n_classes * count), 10.0)
        for cls, count in class_counts.items()
    }

    data["sample_weight"] = data["risk_class"].map(weight_map).fillna(1.0).round(4)

    logger.info("Poids de classes (plafonnes a 10.0) :")
    for cls, w in sorted(weight_map.items()):
        logger.info(f"  {MERGED_CLASS_NAMES.get(int(cls), '?'):<10} -> {w:.3f}")
    return data


# ─── RESOLUTION PRODUIT / ENGAGEMENT ──────────────────────────────────────────

def _fill_product_name(data: pd.DataFrame, products: pd.DataFrame,
                        mask: pd.Series) -> None:
    if "id" not in products.columns or "name" not in products.columns:
        data.loc[mask, "product_name"] = data.loc[mask, "product_name"].fillna("Unknown")
        return
    prod_lookup = (
        products[["id", "name"]]
        .assign(pid_str=lambda d: _normalize_id_series(d["id"]))
        .drop_duplicates("pid_str").set_index("pid_str")["name"]
    )
    data.loc[mask, "product_name"] = (
        _normalize_id_series(data.loc[mask, "product_id"])
        .map(prod_lookup).fillna("Unknown").values
    )


def _fill_engagement_name(data: pd.DataFrame, engagements: pd.DataFrame) -> None:
    data["engagement_name"] = ""
    if "id" not in engagements.columns or "name" not in engagements.columns:
        return
    eng_lookup = (
        engagements[["id", "name"]]
        .assign(eid_str=lambda d: _normalize_id_series(d["id"]))
        .drop_duplicates("eid_str").set_index("eid_str")["name"]
    )
    data["engagement_name"] = (
        _normalize_id_series(data["engagement_id"])
        .map(eng_lookup).fillna("").values
    )


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def preprocess_findings(
    findings: pd.DataFrame,
    products: pd.DataFrame,
    engagements: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline preprocessing v9.0 — 11 etapes.

    1-2  : IDs et resolution noms produit/engagement
    3    : Dates (age_days, days_to_fix brut)
    4    : CVSS / Severite (cvss_severity_gap = audit only)
    5    : Binaires, CVE, Tags
    6    : EPSS
    7    : Contexte (product_fp_rate Laplace, severity_x_active = audit only)
    8    : Interactions (cvss_x_severity / severity_x_urgent = audit only)
    9    : Normalisation (cvss_norm, age_norm, tags_norm)
    10   : Exposition (exposure_norm) + delay_norm (signal SOC)
    11   : Label severity -> 4 classes + sample_weight
    """
    logger.info("Etape 1/11 : normalisation des IDs")
    data                  = findings.copy()
    data["product_id"]    = _safe_int_col(data, "product_id")
    data["engagement_id"] = _safe_int_col(data, "engagement_id")

    logger.info("Etape 2/11 : resolution produits et engagements")
    if "product_name" in data.columns and data["product_name"].notna().any():
        mask = data["product_name"].isna()
        if mask.any():
            _fill_product_name(data, products, mask)
    else:
        _fill_product_name(data, products, pd.Series(True, index=data.index))
    if "engagement_name" not in data.columns or data["engagement_name"].isna().all():
        _fill_engagement_name(data, engagements)

    logger.info("Etape 3/11 : features temporelles (age_days, days_to_fix)")
    data = build_date_features(data)

    logger.info("Etape 4/11 : CVSS et severite")
    data = build_severity_features(data)

    logger.info("Etape 5/11 : features binaires, CVE, tags")
    data = build_binary_features(data)

    logger.info("Etape 6/11 : features EPSS")
    data = build_epss_features(data)

    logger.info("Etape 7/11 : features contextuelles (product_fp_rate Laplace)")
    data = build_contextual_features(data)

    logger.info("Etape 8/11 : features d'interaction")
    data = build_interaction_features(data)

    logger.info("Etape 9/11 : normalisation (cvss_norm, age_norm, tags_norm)")
    data = build_normalized_features(data)

    logger.info("Etape 10/11 : exposition (exposure_norm) et delay_norm")
    data = build_exposure_features(data)
    data = build_delay_norm(data)

    logger.info("Etape 11/11 : label severity -> 4 classes + sample_weight")
    data = build_label_from_severity(data)
    data = build_sample_weights(data)

    final_cols = [c for c in KEEP_COLS if c in data.columns]
    result     = data[final_cols].copy()
    for col in ["product_id", "engagement_id"]:
        if col in result.columns:
            result[col] = result[col].astype("Int64")

    logger.info(
        f"Preprocessing termine : {len(result)} lignes x {len(result.columns)} colonnes"
    )
    return result


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def validate_output(df: pd.DataFrame) -> bool:
    ok = True

    required = [
        "cvss_score", "epss_score", "age_days", "delay_norm",
        "exposure_norm", "risk_class", "sample_weight",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        for col in missing:
            logger.error(f"Colonne obligatoire manquante : {col}")
        ok = False

    if len(df) == 0:
        logger.error("DataFrame vide apres preprocessing")
        return False

    valid = df["risk_class"].notna() if "risk_class" in df.columns else pd.Series(dtype=bool)
    if valid.sum() == 0:
        logger.error("Aucun finding avec risk_class valide")
        ok = False

    # Anti-leakage : aucune colonne de EXCLUDE_FROM_ML ne doit etre dans FEATURE_COLS
    leakage = [c for c in EXCLUDE_FROM_ML if c in FEATURE_COLS]
    if leakage:
        logger.error(f"DATA LEAKAGE detecte dans FEATURE_COLS : {leakage}")
        ok = False
    else:
        logger.info("Anti-leakage : OK — aucune colonne suspecte dans FEATURE_COLS")

    # Correlation CVSS/label — warning uniquement (correlation structurelle attendue)
    if "cvss_score" in df.columns and valid.sum() > 0:
        r = float(df.loc[valid, "cvss_score"].corr(df.loc[valid, "risk_class"].astype(float)))
        if abs(r) > 0.95:
            logger.warning(
                f"cvss_score / risk_class r = {r:.4f} > 0.95 "
                "— correlation structurelle elevee (label derive de severity)"
            )
        else:
            logger.info(f"Correlation cvss / risk_class : r = {r:.4f} [OK]")

    # Distribution des classes
    if valid.sum() > 0:
        dist     = df.loc[valid, "risk_class"].value_counts(normalize=True)
        dominant = float(dist.max())
        n_unique = df.loc[valid, "risk_class"].nunique()
        logger.info(
            f"Classes distinctes : {n_unique} | "
            f"Classe dominante : {dominant * 100:.1f}%"
        )
        if n_unique < 4:
            logger.warning(
                f"Seulement {n_unique} classes presentes — "
                "qcut dans train.py utilisera le fallback cut"
            )

        # Alerte si une classe a trop peu d'exemples pour l'entrainement
        counts = df.loc[valid, "risk_class"].value_counts()
        sparse = counts[counts < MIN_ROWS_PER_CLASS]
        if not sparse.empty:
            for cls, cnt in sparse.items():
                name = MERGED_CLASS_NAMES.get(int(cls), "?")
                logger.warning(
                    f"Classe {name} : seulement {cnt} exemples "
                    f"(seuil recommande : {MIN_ROWS_PER_CLASS})"
                )

    # Validation delay_norm
    if "delay_norm" in df.columns:
        dn = df["delay_norm"]
        if dn.isna().any():
            logger.error("delay_norm contient des NaN — verifier build_delay_norm()")
            ok = False
        elif not ((dn >= 0) & (dn <= 1)).all():
            logger.error("delay_norm hors [0,1] — verifier le clipping")
            ok = False
        else:
            logger.info(f"delay_norm : OK [{dn.min():.3f}, {dn.max():.3f}]")

    return ok


# ─── SAUVEGARDE ───────────────────────────────────────────────────────────────

def save_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".csv", dir=path.parent, encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    logger.info(f"Fichier sauvegarde : {path} ({len(df)} lignes)")


def save_data_report(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)
    valid    = df["risk_class"].notna() if "risk_class" in df.columns else pd.Series(True, index=df.index)
    df_valid = df.loc[valid]

    def safe_corr(col_a: str, col_b: str) -> Optional[float]:
        if col_a in df_valid.columns and col_b in df_valid.columns and len(df_valid) > 0:
            return round(float(df_valid[col_a].corr(df_valid[col_b].astype(float))), 4)
        return None

    report = {
        "version":   "9.0",
        "timestamp": datetime.now().isoformat(),
        "label_strategy": (
            "severity_defectdojo_v9 : severity_num -> MERGE_MAP -> 4 classes. "
            "Aucune correction heuristique. L'intelligence = modele ML (LightGBM)."
        ),
        "class_names":     MERGED_CLASS_NAMES,
        "n_rows_total":    len(df),
        "n_rows_training": int(valid.sum()),
        "n_rows_excluded": int((~valid).sum()),
        "n_cols":          len(df.columns),
        "n_features_ml":   len(FEATURE_COLS),
        "leakage_status": (
            "CLEAN"
            if not [c for c in EXCLUDE_FROM_ML if c in FEATURE_COLS]
            else "LEAKAGE DETECTED"
        ),
        "correlations": {
            "severity_num_vs_risk_class": safe_corr("severity_num", "risk_class"),
            "cvss_score_vs_risk_class":   safe_corr("cvss_score",   "risk_class"),
            "epss_score_vs_risk_class":   safe_corr("epss_score",   "risk_class"),
            "delay_norm_vs_risk_class":   safe_corr("delay_norm",   "risk_class"),
            "cvss_vs_epss_redundancy":    safe_corr("cvss_score",   "epss_score"),
        },
        "risk_class_dist": (
            df_valid["risk_class"].value_counts().sort_index().to_dict()
            if "risk_class" in df_valid.columns else {}
        ),
        "signal_coverage": {
            "days_to_fix_count":   int(df["days_to_fix"].notna().sum())
                                   if "days_to_fix" in df.columns else 0,
            "days_to_fix_pct":     round(df["days_to_fix"].notna().mean() * 100, 1)
                                   if "days_to_fix" in df.columns else 0.0,
            "epss_positive_count": int((df["epss_score"] > 0).sum())
                                   if "epss_score" in df.columns else 0,
            "has_cve_count":       int(df["has_cve"].sum())
                                   if "has_cve" in df.columns else 0,
        },
        "delay_norm_stats": (
            df["delay_norm"].describe().round(4).to_dict()
            if "delay_norm" in df.columns else {}
        ),
        "product_fp_rate_laplace": {
            "min": round(float(df["product_fp_rate"].min()), 4)
                   if "product_fp_rate" in df.columns else None,
            "max": round(float(df["product_fp_rate"].max()), 4)
                   if "product_fp_rate" in df.columns else None,
        },
        "sample_weight_generated": "sample_weight" in df.columns,
        "feature_cols_count":      len(FEATURE_COLS),
        "feature_cols":            FEATURE_COLS,
    }

    path = PROCESSED_DIR / "data_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Rapport JSON sauvegarde : {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("InvisiThreat — Preprocessing v9.0")
    logger.info("  [L1] Label    : severity DefectDojo -> MERGE_MAP -> 4 classes")
    logger.info("  [L2] Features : normalises [0,1] — cvss, epss, delay, exposure")
    logger.info("  [L3] Anti-leakage : severity_num / derives exclus de FEATURE_COLS")
    logger.info("  [L4] product_fp_rate : lissage Laplace (sum+1)/(count+2)")
    logger.info("  [L5] Monitoring : correlation CVSS/EPSS + distribution classes")
    logger.info("=" * 70)
    logger.info(f"Repertoire racine : {ROOT_DIR}")

    try:
        findings, products, engagements = load_data()
        df_clean = preprocess_findings(findings, products, engagements)

        if not validate_output(df_clean):
            logger.error("Validation echouee — corriger les erreurs ci-dessus avant de continuer")
            raise SystemExit(1)

        save_atomic(df_clean, PROCESSED_DIR / "findings_clean.csv")
        save_data_report(df_clean)

        logger.info("=" * 70)
        logger.info("Preprocessing v9.0 termine avec succes")
        logger.info("  -> Prochaine etape : python src/train.py")
        logger.info("=" * 70)

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()