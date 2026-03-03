import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scoring.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_engine.scoring")

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
PROJECT_ROOT   = SCRIPT_DIR.parent

MODELS_DIR     = PROJECT_ROOT / "models"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR    = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "pipeline_latest.pkl"
META_PATH  = MODELS_DIR / "pipeline_latest_meta.json"
CLEAN_CSV  = DATA_PROCESSED / "findings_clean.csv"
RAW_CSV    = DATA_RAW / "findings_raw.csv"

FEATURE_COLS = [
    "severity_num", "cvss_score", "age_days",
    "has_cve", "has_cwe", "tags_count",
    "is_false_positive", "is_active",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "severity_x_active", "product_fp_rate", "cvss_severity_gap",
    "cvss_x_severity", "cvss_x_has_cve", "severity_x_urgent", "age_x_cvss",
    "cvss_score_norm", "severity_norm", "age_days_norm",
    "tags_count_norm", "cvss_severity_gap_norm",
]

CLASS_LABELS = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
CLASS_COLORS = {
    "Info":     "#64748b",
    "Low":      "#22c55e",
    "Medium":   "#f59e0b",
    "High":     "#f97316",
    "Critical": "#ef4444",
}
CLASS_BG = {
    "Info":     "#f1f5f9",
    "Low":      "#f0fdf4",
    "Medium":   "#fffbeb",
    "High":     "#fff7ed",
    "Critical": "#fef2f2",
}
SEVERITY_MAP_INV = {4: "Critical", 3: "High", 2: "Medium", 1: "Low", 0: "Info"}


# ══════════════════════════════════════════════
# 1. CHARGEMENT MODELE
# ══════════════════════════════════════════════

def load_model() -> tuple:
    """Charge le pipeline sklearn et ses metadonnees."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}\n"
            "Executez d'abord : python main.py train"
        )

    pipeline = joblib.load(MODEL_PATH)
    meta     = {}
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)

    rf        = pipeline.named_steps.get("model")
    n_classes = len(rf.classes_) if rf is not None else 5
    logger.info(
        f"Modele charge — version={meta.get('timestamp', '?')}  "
        f"classes={n_classes}  features={meta.get('n_features', len(FEATURE_COLS))}"
    )
    return pipeline, meta


# ══════════════════════════════════════════════
# 2. CHARGEMENT DONNEES
# ══════════════════════════════════════════════

def load_data(source: str = "processed", product_id: Optional[int] = None) -> pd.DataFrame:
    """
    Charge les donnees depuis la source choisie.
    source='processed' -> findings_clean.csv (recommande)
    source='raw'       -> findings_raw.csv (necessite features a la volee)
    """
    path = CLEAN_CSV if source == "processed" else RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Executez d'abord : python main.py {'preprocess' if source == 'processed' else 'fetch'}"
        )

    df = pd.read_csv(path)
    logger.info(f"Donnees chargees : {len(df)} findings depuis {path.name}")

    if "product_name" in df.columns:
        logger.info("product_name present dans les donnees.")
        sample = df["product_name"].dropna().iloc[0] if not df["product_name"].isna().all() else "TOUS NaN"
        logger.info(f"Exemple product_name : {sample}")
    else:
        logger.warning("product_name absent — l'ID produit sera utilise a la place.")

    if "engagement_name" in df.columns:
        logger.info("engagement_name present dans les donnees.")

    # FIX D : dedoublonnage AVANT tout traitement pour ne pas biaiser les stats
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["id"])
        removed = before - len(df)
        if removed:
            logger.info(f"Dedoublonnage par ID : {before} -> {len(df)} findings ({removed} doublons supprimes)")

    if product_id is not None:
        df = df[df["product_id"] == product_id].copy()
        logger.info(f"Filtre product_id={product_id} -> {len(df)} findings")
        if len(df) == 0:
            raise ValueError(f"Aucun finding pour product_id={product_id}")

    return df


# ══════════════════════════════════════════════
# 3. PREDICTION
# ══════════════════════════════════════════════

def _fill_missing_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Calcule les features manquantes a la volee si non presentes dans le CSV.

    FIX A (scoring) — product_fp_rate :
      En inference, on n'a pas de train/test. On utilise les taux sauvegardes
      dans les metadonnees du modele (meta['product_fp_rates']) s'ils existent,
      sinon on calcule sur toutes les donnees disponibles (comportement acceptable
      en production car il n'y a pas de fuite : on ne cherche pas a generaliser
      a un test set, on score tous les findings).
    """
    # --- Features derivees de base ---
    if "severity_x_active" not in df.columns and all(c in df.columns for c in ["severity_num", "is_active"]):
        df["severity_x_active"] = df["severity_num"] * df["is_active"]

    if "cvss_severity_gap" not in df.columns and "cvss_score" in df.columns and "severity_num" in df.columns:
        cvss_norm = df["cvss_score"] / 10 * 4
        df["cvss_severity_gap"] = (cvss_norm - df["severity_num"]).abs().round(3)

    # FIX A : product_fp_rate depuis metadonnees du modele en priorite
    if "product_fp_rate" not in df.columns and all(c in df.columns for c in ["product_id", "is_false_positive"]):
        saved_rates = meta.get("product_fp_rates", {})
        if saved_rates:
            # Utilise les taux appris sur le train (coherence avec le modele)
            logger.info("product_fp_rate : utilisation des taux sauvegardes dans les metadonnees du modele")
            df["product_fp_rate"] = (
                df["product_id"].astype(str)
                .map({str(k): v for k, v in saved_rates.items()})
                .fillna(float(np.mean(list(saved_rates.values()))) if saved_rates else 0.0)
            )
        else:
            # Fallback : calcul sur toutes les donnees disponibles
            # Acceptable en inference (pas de generalisation a un test set)
            logger.info("product_fp_rate : calcul sur toutes les donnees disponibles (pas de taux sauvegardes)")
            df["product_fp_rate"] = (
                df.groupby("product_id")["is_false_positive"]
                .transform("mean")
                .round(4)
            )

    # --- Features d'interaction ---
    for col in ["cvss_x_severity", "cvss_x_has_cve", "severity_x_urgent", "age_x_cvss"]:
        if col not in df.columns:
            if col == "cvss_x_severity" and "cvss_score" in df.columns and "severity_num" in df.columns:
                df[col] = df["cvss_score"] * df["severity_num"]
            elif col == "cvss_x_has_cve" and "cvss_score" in df.columns and "has_cve" in df.columns:
                df[col] = df["cvss_score"] * df["has_cve"]
            elif col == "severity_x_urgent" and "severity_num" in df.columns:
                df[col] = df["severity_num"] * df.get("tag_urgent", 0)
            elif col == "age_x_cvss" and "age_days" in df.columns and "cvss_score" in df.columns:
                df[col] = df["age_days"] * df["cvss_score"]

    # --- Normalisation ---
    norm_map = {
        "cvss_score_norm":        ("cvss_score",        10.0),
        "severity_norm":          ("severity_num",       4.0),
        "age_days_norm":          ("age_days",         365.0),
        "tags_count_norm":        ("tags_count",        20.0),
        "cvss_severity_gap_norm": ("cvss_severity_gap",  4.0),
    }
    for dst, (src, divisor) in norm_map.items():
        if dst not in df.columns and src in df.columns:
            df[dst] = (df[src] / divisor).clip(0, 1).round(4)

    # --- Tags semantiques par defaut ---
    for tag_col in ["tag_urgent", "tag_in_production", "tag_sensitive", "tag_external"]:
        if tag_col not in df.columns:
            df[tag_col] = 0

    return df


def run_scoring(df: pd.DataFrame, pipeline, meta: dict) -> pd.DataFrame:
    """
    Applique le pipeline de prediction sur tous les findings.

    FIX B — Calcul risk_score robuste :
      L'ancienne formule supposait model_classes = [0,1,2,3,4] et divisait
      par 4 (max theorique = classe 4). Si une classe est absente du modele
      entrainee (ex: pas de 'Info' dans le train), le denominateur est faux.
      Correction : on utilise le max theorique = max(model_classes),
      ce qui garantit un score dans [0, 10] quelle que soit la configuration.
    """
    df = _fill_missing_features(df.copy(), meta)

    # Features disponibles — on impute a 0 les manquantes
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        logger.warning(f"Features manquantes (imputees a 0) : {sorted(missing)}")
        for col in missing:
            df[col] = 0.0

    X = df[FEATURE_COLS].fillna(0).copy()

    logger.info(f"Scoring de {len(X)} findings...")
    start = datetime.now()

    raw_classes   = pipeline.predict(X)
    raw_probas    = pipeline.predict_proba(X)
    model_classes = pipeline.named_steps["model"].classes_

    # FIX B : denominateur = max(model_classes) pour etre robuste
    max_class = max(int(c) for c in model_classes)
    if max_class == 0:
        max_class = 1  # protection division par zero

    risk_scores = np.array([
        sum(int(c) * p for c, p in zip(model_classes, probas)) / max_class * 10
        for probas in raw_probas
    ]).round(2)

    df["predicted_class"] = raw_classes.astype(int)
    df["predicted_level"] = [CLASS_LABELS.get(int(c), "Unknown") for c in raw_classes]
    df["risk_score"]      = risk_scores
    df["confidence"]      = raw_probas.max(axis=1).round(4)

    # Probabilites par classe
    for i, cls in enumerate(model_classes):
        label = CLASS_LABELS.get(int(cls), str(cls))
        df[f"proba_{label.lower()}"] = raw_probas[:, i].round(4)

    df["scored_at"] = datetime.now(timezone.utc).isoformat()

    duration = (datetime.now() - start).total_seconds()
    logger.info(
        f"Scoring termine en {duration:.2f}s — {len(df)} findings  |  "
        f"Critical: {(df['predicted_level']=='Critical').sum()}  "
        f"High: {(df['predicted_level']=='High').sum()}  "
        f"Medium: {(df['predicted_level']=='Medium').sum()}"
    )
    return df


# ══════════════════════════════════════════════
# 4. AGREGATION DES STATISTIQUES
# ══════════════════════════════════════════════

def compute_stats(df: pd.DataFrame) -> dict:
    """Calcule toutes les statistiques d'agregation pour les rapports."""
    total = len(df)
    now   = datetime.now(timezone.utc).isoformat()

    level_dist = df["predicted_level"].value_counts().to_dict()
    level_pct  = {k: round(v / total * 100, 1) for k, v in level_dist.items()}

    global_score = round(float(df["risk_score"].mean()), 2)
    median_score = round(float(df["risk_score"].median()), 2)
    max_score    = round(float(df["risk_score"].max()), 2)

    # Stats par produit
    by_product = []
    if "product_id" in df.columns:
        logger.info(f"Stats par produit : {df['product_id'].nunique()} produits distincts")
        for pid, grp in df.groupby("product_id"):
            product_name   = grp["product_name"].iloc[0] if "product_name" in grp.columns else str(pid)
            critical_count = (grp["predicted_level"] == "Critical").sum()
            high_count     = (grp["predicted_level"] == "High").sum()
            by_product.append({
                "product_id":      int(pid),
                "product_name":    product_name,
                "total":           len(grp),
                "mean_risk_score": round(float(grp["risk_score"].mean()), 2),
                "max_risk_score":  round(float(grp["risk_score"].max()), 2),
                "critical_count":  int(critical_count),
                "high_count":      int(high_count),
                "priority_score":  round(float(critical_count * 10 + high_count * 5 + grp["risk_score"].sum()), 1),
            })
        by_product.sort(key=lambda x: x["priority_score"], reverse=True)

    # FIX C : tri par risk_score AVANT de construire le top
    top_critical = df.sort_values("risk_score", ascending=False).head(50)

    top_list = []
    for _, row in top_critical.iterrows():
        top_list.append({
            "id":              int(row["id"]) if pd.notna(row.get("id")) else None,
            "title":           str(row.get("title", "N/A"))[:80],
            "predicted_level": row["predicted_level"],
            "risk_score":      float(row["risk_score"]),
            "confidence":      float(row["confidence"]),
            "cvss_score":      float(row.get("cvss_score", 0)),
            "severity_num":    int(row.get("severity_num", 0)),
            "age_days":        int(row.get("age_days", 0)),
            "has_cve":         int(row.get("has_cve", 0)),
            "product_id":      int(row["product_id"]) if pd.notna(row.get("product_id")) else None,
            "product_name":    str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else "",
            "engagement_name": str(row.get("engagement_name", "")) if pd.notna(row.get("engagement_name")) else "",
            "file_path":       str(row.get("file_path", "")) if pd.notna(row.get("file_path")) else "",
            "line":            int(row["line"]) if pd.notna(row.get("line")) else None,
            "description":     str(row.get("description", ""))[:200] if pd.notna(row.get("description")) else "",
        })

    # Matrice severite declaree vs predite
    confusion = {}
    if "severity_num" in df.columns:
        for sev_num in sorted(df["severity_num"].unique()):
            sev_label = SEVERITY_MAP_INV.get(int(sev_num), str(sev_num))
            sub = df[df["severity_num"] == sev_num]
            confusion[sev_label] = sub["predicted_level"].value_counts().to_dict()

    return {
        "generated_at":          now,
        "total_findings":        total,
        "global_score":          global_score,
        "median_score":          median_score,
        "max_score":             max_score,
        "level_dist":            level_dist,
        "level_pct":             level_pct,
        "by_product":            by_product,
        "top_critical":          top_list,
        "severity_vs_predicted": confusion,
    }


# ══════════════════════════════════════════════
# 5. EXPORT JSON
# ══════════════════════════════════════════════

def save_json_report(stats: dict, df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [c for c in [
        "id", "title",
        "product_id", "product_name",
        "engagement_id", "engagement_name",
        "file_path", "line", "description",
        "predicted_class", "predicted_level",
        "risk_score", "confidence",
        "cvss_score", "severity_num",
        "age_days", "has_cve", "has_cwe",
        "is_active", "is_false_positive",
        "scored_at",
    ] if c in df.columns]

    report = {
        "meta": {
            "report_version": "2.1",
            "generated_at":   stats["generated_at"],
            "engine":         "AI Risk Engine",
        },
        "summary": {
            "total_findings":    stats["total_findings"],
            "global_risk_score": stats["global_score"],
            "median_risk_score": stats["median_score"],
            "max_risk_score":    stats["max_score"],
            "distribution":      stats["level_dist"],
            "distribution_pct":  stats["level_pct"],
        },
        "by_product":            stats["by_product"],
        "top_critical_findings": stats["top_critical"],
        "severity_vs_predicted": stats["severity_vs_predicted"],
        "all_findings": json.loads(
            df[export_cols].to_json(orient="records", default_handler=str)
        ),
    }

    path = output_dir / "scoring_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Rapport JSON -> {path}")
    return path


# ══════════════════════════════════════════════
# 6. EXPORT CSV
# ══════════════════════════════════════════════

def save_csv_scored(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "findings_scored.csv"
    df.to_csv(path, index=False)
    logger.info(f"CSV enrichi -> {path}  ({len(df)} lignes)")
    return path


# ══════════════════════════════════════════════
# 7. RAPPORT HTML
# ══════════════════════════════════════════════

def save_html_report(stats: dict, meta: dict, df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dist       = stats["level_dist"]
    pct        = stats["level_pct"]
    total      = stats["total_findings"]
    top        = stats["top_critical"]
    by_product = stats["by_product"]
    gen_at     = stats["generated_at"][:19].replace("T", " ")
    model_ver  = meta.get("timestamp", "N/A")

    def badge(level: str) -> str:
        color = CLASS_COLORS.get(level, "#888")
        bg    = CLASS_BG.get(level, "#f8f8f8")
        return (
            f'<span style="background:{bg};color:{color};border:1.5px solid {color};'
            f'padding:2px 10px;border-radius:20px;font-size:0.78rem;font-weight:700;">'
            f'{level}</span>'
        )

    def score_bar(score: float) -> str:
        pct_val = min(score / 10 * 100, 100)
        if score >= 8:   bar_color = CLASS_COLORS["Critical"]
        elif score >= 6: bar_color = CLASS_COLORS["High"]
        elif score >= 4: bar_color = CLASS_COLORS["Medium"]
        elif score >= 2: bar_color = CLASS_COLORS["Low"]
        else:            bar_color = CLASS_COLORS["Info"]
        return (
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden">'
            f'<div style="width:{pct_val:.0f}%;height:100%;background:{bar_color};border-radius:3px"></div>'
            f'</div><span style="font-weight:700;color:{bar_color};min-width:36px;text-align:right">'
            f'{score:.1f}</span></div>'
        )

    dist_cards = ""
    for lvl in ["Critical", "High", "Medium", "Low", "Info"]:
        count = dist.get(lvl, 0)
        pct_v = pct.get(lvl, 0.0)
        color = CLASS_COLORS[lvl]
        bg    = CLASS_BG[lvl]
        dist_cards += f"""
        <div style="background:{bg};border:1.5px solid {color}22;border-radius:14px;
                    padding:20px 24px;position:relative;overflow:hidden">
          <div style="position:absolute;top:0;left:0;height:4px;width:{min(pct_v,100)}%;
                      background:{color};border-radius:2px 0 0 0"></div>
          <div style="font-size:2rem;font-weight:800;color:{color};line-height:1">{count}</div>
          <div style="font-size:0.82rem;font-weight:600;color:{color};opacity:0.8;
                      text-transform:uppercase;letter-spacing:0.08em;margin-top:4px">{lvl}</div>
          <div style="font-size:0.75rem;color:#94a3b8;margin-top:6px">{pct_v}% du total</div>
        </div>"""

    top_rows = ""
    for i, f in enumerate(top[:15], 1):
        lvl          = f["predicted_level"]
        prod_display = f.get("product_name") or str(f.get("product_id", "—"))
        cve          = "CVE" if f.get("has_cve") else "—"
        top_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9"
            onmouseover="this.style.background='#f8fafc'"
            onmouseout="this.style.background='transparent'">
          <td style="padding:12px 16px;font-weight:700;color:#64748b;font-size:0.8rem">{i}</td>
          <td style="padding:12px 8px">{badge(lvl)}</td>
          <td style="padding:12px 16px;font-size:0.85rem;max-width:320px;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
              title="{f['title']}">{f['title']}</td>
          <td style="padding:12px 16px;min-width:150px">{score_bar(f['risk_score'])}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;color:#64748b">{f['cvss_score']}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;color:#64748b">{f['age_days']}j</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;
                     color:{'#22c55e' if cve=='CVE' else '#cbd5e1'}">{cve}</td>
          <td style="padding:12px 16px;font-size:0.82rem;color:#64748b">{prod_display}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.78rem;
                     color:{'#22c55e' if f['confidence']>=0.8 else '#f59e0b' if f['confidence']>=0.6 else '#ef4444'}">{f['confidence']:.0%}</td>
        </tr>"""

    prod_rows = ""
    for p in by_product[:10]:
        score     = p["mean_risk_score"]
        sc_color  = CLASS_COLORS["Critical"] if score >= 7 else CLASS_COLORS["High"] if score >= 5 else CLASS_COLORS["Medium"]
        prod_name = p.get("product_name", p["product_id"])
        prod_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9">
          <td style="padding:11px 16px;font-weight:600;color:#1e293b">{prod_name}</td>
          <td style="padding:11px 16px;text-align:center">{p['total']}</td>
          <td style="padding:11px 16px;text-align:center;color:{CLASS_COLORS['Critical']};font-weight:700">{p['critical_count']}</td>
          <td style="padding:11px 16px;text-align:center;color:{CLASS_COLORS['High']};font-weight:700">{p['high_count']}</td>
          <td style="padding:11px 16px;min-width:130px">{score_bar(score)}</td>
          <td style="padding:11px 16px;text-align:center;font-size:0.82rem;color:{sc_color};font-weight:700">{p['max_risk_score']}</td>
        </tr>"""

    # FIX C : liste complete triee par risk_score DESC
    df_display = df.sort_values("risk_score", ascending=False).head(500)
    full_rows  = ""
    for _, row in df_display.iterrows():
        lvl         = row.get("predicted_level", "Unknown")
        title       = str(row.get("title", ""))[:60]
        risk_score  = row.get("risk_score", 0)
        file_path   = str(row.get("file_path", ""))[:40] if pd.notna(row.get("file_path")) else "—"
        line_str    = str(int(row["line"])) if pd.notna(row.get("line")) else "—"
        desc_raw    = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
        desc        = (desc_raw[:60] + "…") if len(desc_raw) > 60 else desc_raw
        prod_name   = str(row.get("product_name", row.get("product_id", "")))[:20]
        eng_name    = str(row.get("engagement_name", ""))[:20] if pd.notna(row.get("engagement_name")) else "—"
        full_rows  += f"""
        <tr style="border-bottom:1px solid #f1f5f9">
          <td style="padding:8px 12px;font-size:0.8rem">{row.get('id','')}</td>
          <td style="padding:8px 12px">{badge(lvl)}</td>
          <td style="padding:8px 12px;font-size:0.8rem;max-width:200px;overflow:hidden;
                     text-overflow:ellipsis" title="{row.get('title','')}">{title}</td>
          <td style="padding:8px 12px;min-width:100px">{score_bar(risk_score)}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{file_path}</td>
          <td style="padding:8px 12px;text-align:center;font-size:0.8rem;color:#64748b">{line_str}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b;max-width:150px;
                     overflow:hidden;text-overflow:ellipsis">{desc}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{prod_name}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{eng_name}</td>
        </tr>"""

    if len(df) > 500:
        full_rows += f'<tr><td colspan="9" style="padding:16px;text-align:center;color:#94a3b8;font-style:italic">… et {len(df)-500} autres findings (voir CSV/JSON)</td></tr>'

    chart_labels = json.dumps(["Info", "Low", "Medium", "High", "Critical"])
    chart_values = json.dumps([dist.get(l, 0) for l in ["Info", "Low", "Medium", "High", "Critical"]])
    chart_colors = json.dumps([CLASS_COLORS[l] for l in ["Info", "Low", "Medium", "High", "Critical"]])

    critical_rate     = round((dist.get("Critical", 0) + dist.get("High", 0)) / max(total, 1) * 100, 1)
    risk_level_global = (
        "CRITIQUE" if stats["global_score"] >= 7 else
        "ELEVE"    if stats["global_score"] >= 5 else
        "MODERE"   if stats["global_score"] >= 3 else "FAIBLE"
    )
    risk_color_global = (
        CLASS_COLORS["Critical"] if stats["global_score"] >= 7 else
        CLASS_COLORS["High"]     if stats["global_score"] >= 5 else
        CLASS_COLORS["Medium"]   if stats["global_score"] >= 3 else CLASS_COLORS["Low"]
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Risk Engine — Scoring Report</title>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    :root{{--bg:#f0f4f8;--surface:#ffffff;--border:#e2e8f0;--text:#1e293b;--muted:#64748b;--accent:#6366f1}}
    body{{font-family:'Sora',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}}
    .topbar{{background:#0f172a;color:white;padding:0 32px;height:56px;display:flex;
             align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100}}
    .topbar-brand{{display:flex;align-items:center;gap:12px;font-weight:700;font-size:0.95rem}}
    .dot{{width:8px;height:8px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite}}
    @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
    .topbar-meta{{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#64748b}}
    .container{{max-width:1300px;margin:0 auto;padding:32px 24px 64px}}
    .page-header{{margin-bottom:32px}}
    .page-header h1{{font-size:1.75rem;font-weight:800;color:#0f172a;letter-spacing:-0.02em}}
    .page-header p{{color:var(--muted);font-size:0.88rem;margin-top:6px}}
    .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
    .kpi-card{{background:var(--surface);border:1px solid var(--border);border-radius:14px;
               padding:20px 24px;position:relative;overflow:hidden}}
    .kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent)}}
    .kpi-label{{font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;
                color:var(--muted);margin-bottom:8px}}
    .kpi-value{{font-size:2.2rem;font-weight:800;line-height:1;color:#0f172a}}
    .kpi-sub{{font-size:0.78rem;color:var(--muted);margin-top:6px}}
    .section{{background:var(--surface);border:1px solid var(--border);border-radius:16px;
              margin-bottom:24px;overflow:hidden}}
    .section-header{{padding:18px 24px 16px;border-bottom:1px solid var(--border);
                     display:flex;align-items:center;justify-content:space-between}}
    .section-title{{font-size:0.9rem;font-weight:700;color:#0f172a;display:flex;align-items:center;gap:10px}}
    .section-body{{padding:20px 24px}}
    .dist-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
    table{{width:100%;border-collapse:collapse}}
    th{{padding:10px 16px;font-size:0.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.06em;color:var(--muted);text-align:left;background:#f8fafc;
        border-bottom:1px solid var(--border)}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
    .tag{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.72rem;
          font-weight:700;letter-spacing:0.04em;text-transform:uppercase}}
    .alert-banner{{border-radius:12px;padding:14px 20px;display:flex;align-items:center;
                   gap:14px;margin-bottom:24px;font-size:0.88rem}}
    @media(max-width:900px){{.dist-grid{{grid-template-columns:repeat(3,1fr)}}.two-col{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<div class="topbar">
  <div class="topbar-brand"><div class="dot"></div>AI Risk Engine</div>
  <div class="topbar-meta">Genere le {gen_at} UTC &nbsp;|&nbsp; Modele : {model_ver}</div>
</div>
<div class="container">
  <div class="page-header">
    <h1>Rapport de Scoring des Vulnerabilites</h1>
    <p>{total} findings analyses &nbsp;&middot;&nbsp; Score global : <strong>{stats['global_score']}/10</strong>
       &nbsp;&middot;&nbsp; Niveau : <strong style="color:{risk_color_global}">{risk_level_global}</strong></p>
  </div>

  {"" if critical_rate < 20 else f'''
  <div class="alert-banner" style="background:#fef2f2;border:1.5px solid #fca5a5">
    <span style="font-size:1.4rem">&#128680;</span>
    <div><strong style="color:#dc2626">{critical_rate}% des findings sont High ou Critical.</strong>
    <span style="color:#ef4444;margin-left:8px">Action immediate recommandee sur les {dist.get("Critical",0)} findings critiques.</span></div>
  </div>'''}

  <div class="kpi-grid">
    <div class="kpi-card" style="--accent:{risk_color_global}">
      <div class="kpi-label">Score Global</div>
      <div class="kpi-value" style="color:{risk_color_global}">{stats['global_score']}</div>
      <div class="kpi-sub">sur 10 &middot; niveau {risk_level_global}</div>
    </div>
    <div class="kpi-card"><div class="kpi-label">Total Findings</div>
      <div class="kpi-value">{total}</div><div class="kpi-sub">analyses par l'IA</div></div>
    <div class="kpi-card" style="--accent:{CLASS_COLORS['Critical']}">
      <div class="kpi-label">Critiques</div>
      <div class="kpi-value" style="color:{CLASS_COLORS['Critical']}">{dist.get('Critical',0)}</div>
      <div class="kpi-sub">{pct.get('Critical',0)}% du total</div></div>
    <div class="kpi-card" style="--accent:{CLASS_COLORS['High']}">
      <div class="kpi-label">High</div>
      <div class="kpi-value" style="color:{CLASS_COLORS['High']}">{dist.get('High',0)}</div>
      <div class="kpi-sub">{pct.get('High',0)}% du total</div></div>
    <div class="kpi-card"><div class="kpi-label">Score Median</div>
      <div class="kpi-value" style="font-size:1.8rem">{stats['median_score']}</div>
      <div class="kpi-sub">max : {stats['max_score']}</div></div>
  </div>

  <div class="two-col">
    <div class="section">
      <div class="section-header"><div class="section-title">Distribution par niveau</div></div>
      <div class="section-body"><div class="dist-grid">{dist_cards}</div></div>
    </div>
    <div class="section">
      <div class="section-header"><div class="section-title">Repartition visuelle</div></div>
      <div class="section-body" style="display:flex;align-items:center;justify-content:center;height:200px">
        <canvas id="distChart" style="max-height:190px"></canvas>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">
      <div class="section-title">Top Findings (par score IA decroissant)</div>
      <span class="tag" style="background:#f1f5f9;color:#334155">{len(top)} findings</span>
    </div>
    <div style="overflow-x:auto"><table>
      <thead><tr>
        <th>#</th><th>Niveau</th><th>Titre</th><th>Score IA</th>
        <th>CVSS</th><th>Age</th><th>CVE</th><th>Produit</th><th>Confiance</th>
      </tr></thead>
      <tbody>{top_rows}</tbody>
    </table></div>
  </div>

  {"" if not by_product else f'''
  <div class="section">
    <div class="section-header"><div class="section-title">Risque par Produit (top 10)</div></div>
    <div style="overflow-x:auto"><table>
      <thead><tr>
        <th>Produit</th><th>Findings</th><th>Critical</th><th>High</th>
        <th>Score moyen</th><th>Score max</th>
      </tr></thead>
      <tbody>{prod_rows}</tbody>
    </table></div>
  </div>'''}

  <div class="section">
    <div class="section-header">
      <div class="section-title">Liste complete des findings (top 500 par score)</div>
      <span class="tag" style="background:#f1f5f9;color:#334155">{len(df)} total</span>
    </div>
    <div style="overflow-x:auto;max-height:500px;overflow-y:auto"><table>
      <thead><tr>
        <th>ID</th><th>Niveau</th><th>Titre</th><th>Score IA</th>
        <th>Fichier</th><th>Ligne</th><th>Description</th><th>Produit</th><th>Engagement</th>
      </tr></thead>
      <tbody>{full_rows}</tbody>
    </table></div>
  </div>

  <div style="text-align:center;margin-top:40px;color:#94a3b8;font-size:0.78rem">
    <span style="font-family:'IBM Plex Mono',monospace">AI Risk Engine v2.1</span>
    &nbsp;&middot;&nbsp; {gen_at} UTC &nbsp;&middot;&nbsp; Modele : {model_ver}
  </div>
</div>

<script>
  new Chart(document.getElementById('distChart').getContext('2d'), {{
    type: 'doughnut',
    data: {{
      labels: {chart_labels},
      datasets: [{{
        data: {chart_values},
        backgroundColor: {chart_colors},
        borderWidth: 2,
        borderColor: '#ffffff',
        hoverOffset: 6,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: true,
      plugins: {{
        legend: {{ position: 'right', labels: {{ font: {{ family: 'Sora', size: 11 }}, padding: 12 }} }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}} : ${{ctx.parsed}} (${{(ctx.parsed/{total}*100).toFixed(1)}}%)` }} }}
      }},
      cutout: '62%',
    }}
  }});
</script>
</body>
</html>"""

    path = output_dir / "scoring_report.html"
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Rapport HTML -> {path}")
    return path


# ══════════════════════════════════════════════
# 8. ORCHESTRATION
# ══════════════════════════════════════════════

def run(
    source:     str           = "processed",
    output_dir: Path          = REPORTS_DIR,
    top:        int           = 20,
    product_id: Optional[int] = None,
) -> dict:
    logger.info("=" * 56)
    logger.info("AI Risk Engine — Scoring Pipeline v2.1")
    logger.info("=" * 56)

    pipeline, meta = load_model()

    # FIX D : dedoublonnage dans load_data(), AVANT scoring et stats
    df = load_data(source=source, product_id=product_id)

    # meta transmis a run_scoring pour FIX A (product_fp_rates)
    df_scored = run_scoring(df, pipeline, meta)

    stats      = compute_stats(df_scored)
    output_dir = Path(output_dir)

    json_path = save_json_report(stats, df_scored, output_dir)
    csv_path  = save_csv_scored(df_scored, output_dir)
    html_path = save_html_report(stats, meta, df_scored, output_dir)

    logger.info("=" * 56)
    logger.info("Scoring termine")
    logger.info(f"   Score global : {stats['global_score']}/10")
    logger.info(f"   Critical     : {stats['level_dist'].get('Critical', 0)}")
    logger.info(f"   High         : {stats['level_dist'].get('High', 0)}")
    logger.info(f"   HTML         : {html_path}")
    logger.info(f"   JSON         : {json_path}")
    logger.info(f"   CSV          : {csv_path}")
    logger.info("=" * 56)

    return {
        "stats":     stats,
        "html_path": str(html_path),
        "json_path": str(json_path),
        "csv_path":  str(csv_path),
    }


# ══════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scoring.py",
        description="AI Risk Engine — Score automatique des vulnerabilites",
    )
    parser.add_argument("--source", choices=["processed", "raw"], default="processed")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--product-id", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(
        source     = args.source,
        output_dir = args.output_dir,
        top        = args.top,
        product_id = args.product_id,
    )