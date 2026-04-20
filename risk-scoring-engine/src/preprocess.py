import ast
import json
import logging
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

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

URGENT_TAGS     = {"urgent", "critical", "p0", "p1", "blocker", "asap", "emergency"}
PRODUCTION_TAGS = {"prod", "production", "live", "prd"}
SENSITIVE_TAGS  = {"sensitive", "pii", "gdpr", "confidential", "secret"}
EXTERNAL_TAGS   = {"external", "internet-facing", "public", "exposed"}

EPSS_THRESHOLD = 0.5


FEATURE_COLS = [
    "cvss_score", "cvss_score_norm",
    "age_days", "age_days_norm",
    "has_cve", "has_cwe",
    "tags_count", "tags_count_norm",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "product_fp_rate",
    "cvss_x_has_cve", "age_x_cvss",
    "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm",
    "exploit_risk",
    "context_score",
    "days_open_high",
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
    "score_composite_adj",
]

# ✅ CORRECTION #1 : Ajouter "cve" à KEEP_COLS pour le conserver dans findings_clean.csv
KEEP_COLS = FEATURE_COLS + [
    "id", "title", "product_id", "engagement_id",
    "product_name", "engagement_name", "file_path", "line", "description",
    "is_false_positive", "is_active", "is_mitigated", "out_of_scope",
    "severity", "severity_num",
    "cvss_severity_gap", "severity_x_active", "cvss_x_severity", "severity_x_urgent",
    "score_composite_raw", "score_composite_adj",
    "risk_score", "risk_class", "label_source", "days_to_fix", "created",
    "cve",  # ← NOUVEAU
]


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

    logger.info(f"Charge : {len(findings)} findings | {len(products)} produits | {len(engagements)} engagements")
    return findings, products, engagements


def safe_col(df, col, default=0):
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)

def _normalize_tz(series):
    if series.dt.tz is not None:
        return series.dt.tz_convert("UTC").dt.tz_localize(None)
    return series

def clamp_percentile(series, p=99):
    upper = series.quantile(p / 100)
    return series.clip(upper=upper) if upper > 0 else series

def _safe_int_col(df, col):
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")
    return pd.to_numeric(df[col], errors="coerce").astype("Int64")

def _normalize_id_series(series):
    return (
        pd.to_numeric(series, errors="coerce")
        .astype("Int64")
        .astype(str)
        .replace("<NA>", "")
    )


def build_date_features(data: pd.DataFrame) -> pd.DataFrame:
    now      = pd.Timestamp(datetime.utcnow())
    date_col = next((c for c in ["date", "created"] if c in data.columns), None)

    if date_col:
        discovery        = _normalize_tz(pd.to_datetime(data[date_col], errors="coerce"))
        raw_age          = (now - discovery).dt.days.fillna(0).clip(lower=0)
        data["age_days"] = clamp_percentile(raw_age, 99)
    else:
        data["age_days"] = 0.0
        discovery        = None
        logger.warning("Aucune colonne date trouvee — age_days = 0")

    fix_col = next((c for c in ["mitigated_date", "last_reviewed"] if c in data.columns), None)
    if fix_col and discovery is not None:
        fix_date            = _normalize_tz(pd.to_datetime(data[fix_col], errors="coerce"))
        data["days_to_fix"] = (fix_date - discovery).dt.days
    else:
        data["days_to_fix"] = np.nan

    return data


# ✅ CORRECTION #2 : Corriger la priorité CVSS pour inclure cvssv4_score
def build_severity_features(data: pd.DataFrame) -> pd.DataFrame:
    data["severity_num"] = (
        data["severity"].map(SEVERITY_MAP).fillna(0).astype(int)
        if "severity" in data.columns else 0
    )
    # Priorité : cvssv3_score > cvssv4_score > cvss_score
    cvss_col = next((c for c in ["cvssv3_score", "cvssv4_score", "cvss_score"] if c in data.columns), None)
    data["cvss_score"] = (
        clamp_percentile(
            pd.to_numeric(data[cvss_col], errors="coerce").fillna(0).clip(0, 10), 99
        )
        if cvss_col else 0.0
    )
    cvss_norm                 = data["cvss_score"] / 10 * 4
    data["cvss_severity_gap"] = clamp_percentile(
        (cvss_norm - data["severity_num"]).abs().round(3), 99
    )
    return data


# FIX #1: Robust tag parsing.
def _parse_tags(tag_field) -> list:
    if pd.isna(tag_field):
        return []
    if isinstance(tag_field, list):
        return [str(t).lower().strip() for t in tag_field]
    if isinstance(tag_field, str):
        s = tag_field.strip()
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(t).lower().strip() for t in parsed]
            if isinstance(parsed, str):
                return [parsed.lower().strip()] if parsed.strip() else []
        except (ValueError, SyntaxError):
            pass
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
            return [p.lower() for p in parts if p]
        return [t.lower().strip() for t in s.split(",") if t.strip()]
    return []


def _has_tag_match(tags: list, keyword_set: set) -> int:
    return int(any(t in keyword_set for t in tags))


# ✅ CORRECTION #3 : Ajouter fonction d'extraction des CVE depuis vulnerability_ids
def _extract_cve_from_vuln_ids(vuln_ids_str) -> str:
    """
    Extrait le premier CVE d'une colonne vulnerability_ids au format JSON/dict string.
    Exemple: "[{'vulnerability_id': 'CVE-2020-28500'}]" → "CVE-2020-28500"
    """
    if pd.isna(vuln_ids_str):
        return None
    try:
        parsed = ast.literal_eval(str(vuln_ids_str))
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    for key in ['vulnerability_id', 'id', 'cve']:
                        if key in item:
                            cve_val = str(item[key]).strip()
                            if cve_val.startswith("CVE-"):
                                return cve_val
    except (ValueError, SyntaxError):
        pass
    return None


def build_binary_features(data: pd.DataFrame) -> pd.DataFrame:
    # ✅ CORRECTION #4 : Extraire les CVE depuis vulnerability_ids
    if "vulnerability_ids" in data.columns:
        data["cve"] = data["vulnerability_ids"].apply(_extract_cve_from_vuln_ids)
    else:
        data["cve"] = None
    
    # has_cve = 1 si CVE existe
    data["has_cve"]           = data["cve"].notna().astype(int)
    data["has_cwe"]           = data["cwe"].notna().astype(int) if "cwe" in data.columns else 0
    data["is_false_positive"] = safe_col(data, "false_p", False).fillna(False).astype(int)
    data["is_active"]         = safe_col(data, "active", True).fillna(True).astype(int)
    data["is_mitigated"]      = safe_col(data, "is_mitigated", False).fillna(False).astype(bool)
    data["out_of_scope"]      = safe_col(data, "out_of_scope", False).fillna(False).astype(bool)

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

    # Diagnostic log
    logger.info(
        f"Tag parsing complete — "
        f"tag_in_production: {int(data['tag_in_production'].sum())} hits | "
        f"tag_urgent: {int(data['tag_urgent'].sum())} hits | "
        f"tag_sensitive: {int(data['tag_sensitive'].sum())} hits | "
        f"tag_external: {int(data['tag_external'].sum())} hits"
    )
    
    # ✅ CORRECTION #5 : Log sur l'extraction des CVE
    cve_count = data["has_cve"].sum()
    logger.info(f"CVE extraction : {int(cve_count)} findings avec CVE détecté")
    
    return data


def build_epss_features(data: pd.DataFrame) -> pd.DataFrame:
    data["epss_score"]      = pd.to_numeric(
        data.get("epss_score", 0), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    data["epss_percentile"] = pd.to_numeric(
        data.get("epss_percentile", 0), errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0)
    data["has_high_epss"]   = (data["epss_score"] > EPSS_THRESHOLD).astype(int)
    data["epss_x_cvss"]     = (data["epss_score"] * data["cvss_score"]).round(4)

    col_min, col_max        = data["epss_score"].min(), data["epss_score"].max()
    data["epss_score_norm"] = (
        ((data["epss_score"] - col_min) / (col_max - col_min)).round(4)
        if col_max > col_min else 0.0
    )
    return data


def build_contextual_features(data: pd.DataFrame) -> pd.DataFrame:
    data["severity_x_active"] = data["severity_num"] * data["is_active"]
    data["product_fp_rate"]   = (
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


def build_normalized_features(data: pd.DataFrame) -> pd.DataFrame:
    norm_map = {
        "cvss_score":        "cvss_score_norm",
        "severity_num":      "severity_norm",
        "age_days":          "age_days_norm",
        "tags_count":        "tags_count_norm",
        "cvss_severity_gap": "cvss_severity_gap_norm",
    }
    for src, dst in norm_map.items():
        col_min = data[src].min() if src in data.columns else 0
        col_max = data[src].max() if src in data.columns else 0
        data[dst] = (
            ((data[src] - col_min) / (col_max - col_min)).round(4)
            if src in data.columns and col_max > col_min else 0.0
        )
    return data


def build_new_features(data: pd.DataFrame) -> pd.DataFrame:
    data["exploit_risk"] = (data["epss_score"] * data["cvss_score"]).round(4)
    data["exploit_risk"] = clamp_percentile(data["exploit_risk"], 99)

    data["context_score"] = (
        data["tag_in_production"] * 2 +
        data["tag_external"]      * 2 +
        data["tag_sensitive"]     * 1
    ).astype(int)

    data["days_open_high"] = (
        data["age_days"] * (data["severity_num"] >= 3).astype(int)
    ).round(0)
    data["days_open_high"] = clamp_percentile(data["days_open_high"], 99)

    logger.info(
        f"Nouvelles features — "
        f"exploit_risk max={data['exploit_risk'].max():.2f} | "
        f"context_score max={data['context_score'].max()} | "
        f"days_open_high max={data['days_open_high'].max():.0f}"
    )
    return data


def build_advanced_ml_target(data: pd.DataFrame) -> pd.DataFrame:
    is_fp = data.get(
        "is_false_positive", pd.Series(0, index=data.index)
    ).fillna(0).astype(bool)

    is_out_of_scope = data.get(
        "out_of_scope", pd.Series(False, index=data.index)
    ).fillna(False).astype(bool)

    is_invalid = is_fp | is_out_of_scope

    severity_num = data["severity_num"].fillna(0) if "severity_num" in data.columns \
                   else pd.Series(0, index=data.index)

    dtf = pd.to_numeric(
        data.get("days_to_fix", pd.Series(np.nan, index=data.index)),
        errors="coerce"
    )

    score = severity_num / 4.0

    fix_factor   = pd.Series(1.0, index=data.index)
    label_source = pd.Series("severity_only", index=data.index)

    has_fix = dtf.notna()

    mask_fast = has_fix & (dtf < 7)
    fix_factor[mask_fast]   = 1.40
    label_source[mask_fast] = "fast_fix_boost_1.40"

    mask_moderate = has_fix & (dtf >= 7) & (dtf < 30)
    fix_factor[mask_moderate]   = 1.15
    label_source[mask_moderate] = "moderate_fix_boost_1.15"

    mask_slow = has_fix & (dtf > 90) & (dtf <= 180)
    fix_factor[mask_slow]   = 0.70
    label_source[mask_slow] = "slow_fix_penalty_0.70"

    mask_very_slow = has_fix & (dtf > 180)
    fix_factor[mask_very_slow]   = 0.55
    label_source[mask_very_slow] = "very_slow_penalty_0.55"

    score_adj = (score * fix_factor).clip(0.0, 1.0)

    score_adj[is_invalid]    = np.nan
    label_source[is_invalid] = "excluded"

    risk_score = (score_adj * 10).round(3)

    bins = [-0.001, 1.5, 3.8, 6.2, 8.7, 10.001]
    cats = [0, 1, 2, 3, 4]
    risk_class = pd.cut(risk_score, bins=bins, labels=cats).astype("float")
    risk_class[is_invalid] = np.nan

    data["score_composite_raw"] = score
    data["score_composite_adj"] = score_adj
    data["risk_score"]          = risk_score
    data["risk_class"]          = risk_class
    data["label_source"]        = label_source

    valid_mask = data["risk_class"].notna()
    data.loc[valid_mask, "risk_class"] = data.loc[valid_mask, "risk_class"].astype(int)

    active_count   = valid_mask.sum()
    excluded_count = is_invalid.sum()
    has_fix_count  = has_fix.sum()

    logger.info(f"Findings pour training : {active_count} | Exclus (FP/OOS) : {excluded_count}")
    logger.info(f"Avec days_to_fix       : {has_fix_count} ({has_fix_count/len(data)*100:.1f}%)")

    if valid_mask.sum() > 0:
        dist = data.loc[valid_mask, "risk_class"].value_counts().sort_index()
        class_names = {0:"Info", 1:"Low", 2:"Medium", 3:"High", 4:"Critical"}
        logger.info("Distribution risk_class :")
        for cls, count in dist.items():
            pct = count / valid_mask.sum() * 100
            logger.info(f"  {class_names.get(int(cls),'?'):<10} ({cls}): {count:5d} ({pct:5.1f}%)")

        src_dist = data["label_source"].value_counts()
        logger.info("Sources des labels :")
        for src, count in src_dist.items():
            logger.info(f"  {src:<35} : {count:5d}")

        if has_fix_count < 100:
            logger.warning(
                f"Seulement {has_fix_count} findings ont days_to_fix. "
                f"Le label = severity brute pour la majorite. "
                f"Plus de vulnerabilites corrigees = meilleur modele."
            )
    else:
        logger.error("Aucun sample valide pour le training !")

    return data


def _fill_product_name(data, products, mask):
    if "id" not in products.columns or "name" not in products.columns:
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


def _fill_engagement_name(data, engagements):
    data["engagement_name"] = ""
    if "id" not in engagements.columns or "name" not in engagements.columns:
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


def preprocess_findings(findings, products, engagements) -> pd.DataFrame:
    data                  = findings.copy()
    data["product_id"]    = _safe_int_col(data, "product_id")
    data["engagement_id"] = _safe_int_col(data, "engagement_id")

    if "product_name" in data.columns and data["product_name"].notna().any():
        mask_missing = data["product_name"].isna()
        if mask_missing.any():
            _fill_product_name(data, products, mask_missing)
    else:
        _fill_product_name(data, products, pd.Series(True, index=data.index))

    if "engagement_name" not in data.columns or data["engagement_name"].isna().all():
        _fill_engagement_name(data, engagements)

    data = build_date_features(data)
    data = build_severity_features(data)
    data = build_binary_features(data)
    data = build_epss_features(data)
    data = build_contextual_features(data)
    data = build_interaction_features(data)
    data = build_normalized_features(data)
    data = build_new_features(data)
    data = build_advanced_ml_target(data)

    final_cols = [c for c in KEEP_COLS if c in data.columns]
    result     = data[final_cols].copy()

    for id_col in ["product_id", "engagement_id"]:
        if id_col in result.columns:
            result[id_col] = result[id_col].astype("Int64")

    logger.info(f"Preprocessing complet : {len(result)} lignes x {len(result.columns)} colonnes")
    return result


def validate_output(df: pd.DataFrame) -> bool:
    ok = True

    required = [
        "severity_num", "cvss_score", "age_days", "risk_score", "risk_class",
        "epss_score", "exploit_risk", "context_score", "days_open_high",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Colonnes manquantes : {missing}")
        ok = False

    if len(df) == 0:
        logger.error("DataFrame vide apres preprocessing")
        ok = False

    valid = df["risk_class"].notna() if "risk_class" in df.columns else pd.Series(dtype=bool)
    if valid.sum() == 0:
        logger.error("Aucun risk_class valide — verifier les donnees brutes")
        ok = False

    if "risk_score" in df.columns and valid.sum() > 0:
        std = df.loc[valid, "risk_score"].std()
        if std < 0.01:
            logger.warning(f"risk_score quasi-constant (std={std:.4f}) — le modele ne pourra pas apprendre")

    leakage = [c for c in EXCLUDE_FROM_ML if c in FEATURE_COLS]
    if leakage:
        logger.error(f"DATA LEAKAGE DETECTE dans FEATURE_COLS : {leakage}")
        ok = False
    else:
        logger.info("Verification leakage : OK — aucune colonne interdite dans FEATURE_COLS")

    return ok


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", dir=path.parent) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    logger.info(f"Sauvegarde : {path} ({len(df)} lignes)")


def save_data_report(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)

    valid    = df["risk_class"].notna() if "risk_class" in df.columns else pd.Series(True, index=df.index)
    df_valid = df.loc[valid]

    report = {
        "version":   "5.0",
        "timestamp": datetime.now().isoformat(),
        "label_strategy": "severity_only + days_to_fix_correction (zero leakage)",
        "n_rows_total":    len(df),
        "n_rows_training": int(valid.sum()),
        "n_rows_excluded": int((~valid).sum()),
        "n_cols":          len(df.columns),
        "leakage_status":  "CLEAN — severity_num et days_to_fix absents de FEATURE_COLS",
        "label_source_dist": (
            df["label_source"].value_counts().to_dict()
            if "label_source" in df.columns else {}
        ),
        "days_to_fix_coverage": {
            "count": int(df["days_to_fix"].notna().sum()),
            "pct":   round(df["days_to_fix"].notna().mean() * 100, 1),
        } if "days_to_fix" in df.columns else {},
        "risk_class_dist": (
            df_valid["risk_class"].value_counts().sort_index().to_dict()
            if "risk_class" in df_valid.columns else {}
        ),
        "feature_stats": {
            "exploit_risk_mean":   round(float(df["exploit_risk"].mean()), 4) if "exploit_risk" in df.columns else 0,
            "context_score_dist":  df["context_score"].value_counts().to_dict() if "context_score" in df.columns else {},
            "days_open_high_mean": round(float(df["days_open_high"].mean()), 1) if "days_open_high" in df.columns else 0,
            "epss_with_score":     int((df["epss_score"] > 0).sum()) if "epss_score" in df.columns else 0,
        },
    }

    path = PROCESSED_DIR / "data_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Rapport data : {path}")


def main() -> None:
    
    logger.info("=" * 70)
    logger.info("InvisiThreat AI Risk Engine — Preprocessing v5.0")
    logger.info("=" * 70)

    try:
        findings, products, engagements = load_data()
        df_clean = preprocess_findings(findings, products, engagements)

        if not validate_output(df_clean):
            logger.error("Validation echouee — verifier les donnees brutes")
            raise SystemExit(1)

        save_atomic(df_clean, PROCESSED_DIR / "findings_clean.csv")
        save_data_report(df_clean)

        logger.info("=" * 70)
        logger.info("Preprocessing termine avec succes — v5.0")
        logger.info("=" * 70)

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()