import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import sys

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configuration des logs
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

# ====================== CONFIG ======================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


CLASS_COLORS = {
    "Critical": "#ef4444", 
    "High": "#f97316", 
    "Medium": "#f59e0b", 
    "Low": "#3b82f6", 
    "Info": "#64748b",
    "Unknown": "#888888"
}
CLASS_BG = {
    "Critical": "#fef2f2", 
    "High": "#fff7ed", 
    "Medium": "#fffbeb", 
    "Low": "#eff6ff", 
    "Info": "#f8fafc",
    "Unknown": "#f8f8f8"
}

MODEL_PATH = MODELS_DIR / "pipeline_latest.pkl"
META_PATH = MODELS_DIR / "pipeline_latest_meta.json"
CLEAN_CSV = DATA_PROCESSED / "findings_clean.csv"

CLASS_LABELS = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
SEVERITY_MAP_INV = {4: "Critical", 3: "High", 2: "Medium", 1: "Low", 0: "Info"}

def load_model():
    """Charge le pipeline scikit-learn et ses métadonnées."""
    if not MODEL_PATH.exists():
        alt_path = PROJECT_ROOT / "src" / "models" / "pipeline_latest.pkl"
        if alt_path.exists():
            model_p, meta_p = alt_path, PROJECT_ROOT / "src" / "models" / "pipeline_latest_meta.json"
        else:
            raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
    else:
        model_p, meta_p = MODEL_PATH, META_PATH

    pipeline = joblib.load(model_p)
    meta = {}
    if meta_p.exists():
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)

    # Détection des features
    model_features = []
    # On essaie d'extraire les noms de colonnes si le modèle a été entraîné avec un DataFrame
    base_estimator = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(base_estimator, "feature_names_in_"):
        model_features = list(base_estimator.feature_names_in_)
    
    if not model_features:
        logger.warning("⚠️ Features non détectées dans l'objet, utilisation du set v4.4")
        model_features = [
            "cvss_score", "cvss_score_norm", "age_days", "age_days_norm",
            "has_cve", "has_cwe", "tags_count", "tags_count_norm",
            "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
            "product_fp_rate", "cvss_x_has_cve", "age_x_cvss",
            "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm"
           
        ]

    logger.info(f"✅ Modèle chargé ({len(model_features)} features)")
    return pipeline, meta, model_features


def load_data(product_id: Optional[int] = None) -> pd.DataFrame:
    """Charge les données nettoyées depuis le CSV."""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"❌ Clean data not found at {CLEAN_CSV}")
    
    df = pd.read_csv(CLEAN_CSV)
    if product_id:
        df = df[df["product_id"] == product_id]
        logger.info(f"📂 Filtré pour le produit ID: {product_id} ({len(df)} findings)")
    else:
        logger.info(f"📂 Chargement de {len(df)} findings globaux")
    return df


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Synchronisé avec la logique de Training v4.4"""
    df = df.copy()
    
    # 1. Normalisation des noms de colonnes pour matcher .get()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # 2. Mapping de Sévérité (Crucial pour days_open_high)
    sev_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    if "severity" in df.columns:
        df["severity_num"] = df["severity"].astype(str).str.lower().map(sev_map).fillna(0).astype(int)
    else:
        df["severity_num"] = 0

    # 3. Extraction des valeurs de base avec fallback
    cvss = df.get("cvss_score", df.get("cvss", 0.0)).fillna(0.0)
    age = df.get("age_days", df.get("age", 0)).fillna(0).astype(int)
    epss = df.get("epss_score", df.get("epss", 0.0)).fillna(0.0)
    
    # 4. Reconstruction des Features v4.4
    df["cvss_score"] = cvss
    df["cvss_score_norm"] = (cvss / 10).clip(0, 1)
    df["age_days"] = age
    df["age_days_norm"] = (age / 365).clip(0, 1)
    df["epss_score"] = epss
    df["epss_percentile"] = df.get("epss_percentile", 0.0).fillna(0.0)
    df["epss_score_norm"] = epss.clip(0, 1)
    
    df["has_cve"] = df.get("has_cve", 0).fillna(0).astype(int)
    df["has_cwe"] = df.get("has_cwe", 0).fillna(0).astype(int)
    
    # Tags
    tag_cols = ["tag_urgent", "tag_in_production", "tag_sensitive", "tag_external"]
    for col in tag_cols:
        df[col] = df.get(col, 0).fillna(0).astype(int)
    
    df["tags_count"] = df[tag_cols].sum(axis=1)
    df["tags_count_norm"] = (df["tags_count"] / 4).clip(0, 1)
    
    # Interactions et scores complexes
    df["cvss_x_has_cve"] = df["cvss_score"] * df["has_cve"]
    df["age_x_cvss"] = df["age_days"] * df["cvss_score"]
    df["epss_x_cvss"] = (df["epss_score"] * df["cvss_score"]).round(4)
    df["has_high_epss"] = (df["epss_score"] > 0.5).astype(int)
    df["exploit_risk"] = df["epss_x_cvss"]
    df["context_score"] = (df["tag_in_production"] * 2 + df["tag_external"] * 2 + df["tag_sensitive"] * 1)
    df["days_open_high"] = df["age_days"] * (df["severity_num"] >= 3).astype(int)
    
    if "product_fp_rate" not in df.columns:
        df["product_fp_rate"] = 0.05 # Neutre

    return df

def run_scoring(df: pd.DataFrame, pipeline, model_features: list) -> pd.DataFrame:
    df_feat = _build_features(df)
    
    # Préparation de X (Strict respect de l'ordre des colonnes du modèle)
    X = pd.DataFrame(index=df_feat.index)
    for col in model_features:
        X[col] = df_feat.get(col, 0.0)
    
    logger.info(f"🚀 Scoring {len(X)} findings avec {X.shape[1]} features...")
    
    # Prédiction
    raw_classes = pipeline.predict(X)
    raw_probas = pipeline.predict_proba(X)
    
    # Calcul du Risk Score (0-10) basé sur l'espérance mathématique des probas
    # Score = sum( proba_classe_i * i ) / 4 * 10
    risk_scores = np.array([
        sum(i * p for i, p in enumerate(probas)) / 4 * 10
        for probas in raw_probas
    ]).round(2)

    df_res = df.copy()
    df_res["predicted_class"] = raw_classes.astype(int)
    df_res["predicted_level"] = [CLASS_LABELS.get(int(c), "Unknown") for c in raw_classes]
    df_res["risk_score"] = risk_scores
    df_res["confidence"] = raw_probas.max(axis=1).round(4)
    
    # Réinjection des features pour le rapport si besoin
    for col in ["cvss_score", "epss_score", "age_days"]:
        if col in df_feat.columns:
            df_res[col] = df_feat[col]

    return df_res

def compute_stats(df: pd.DataFrame) -> dict:
    total = len(df)
    now   = datetime.now(timezone.utc).isoformat()

    level_dist = df["predicted_level"].value_counts().to_dict()
    level_pct  = {k: round(v / total * 100, 1) for k, v in level_dist.items()}

    global_score = round(float(df["risk_score"].mean()), 2)
    median_score = round(float(df["risk_score"].median()), 2)
    max_score    = round(float(df["risk_score"].max()), 2)

    by_product = []
    if "product_id" in df.columns:
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

    top_critical = df.sort_values("risk_score", ascending=False).head(50)
    top_list     = []
    for _, row in top_critical.iterrows():
        top_list.append({
            "id":              int(row["id"]) if pd.notna(row.get("id")) else None,
            "title":           str(row.get("title", "N/A"))[:80],
            "predicted_level": row["predicted_level"],
            "risk_score":      float(row["risk_score"]),
            "confidence":      float(row["confidence"]),
            "cvss_score":      float(row.get("cvss_score", 0)),
            "epss_score":      float(row.get("epss_score", 0)),
            "age_days":        int(row.get("age_days", 0)),
            "has_cve":         int(row.get("has_cve", 0)),
            "product_id":      int(row["product_id"]) if pd.notna(row.get("product_id")) else None,
            "product_name":    str(row.get("product_name", "")) if pd.notna(row.get("product_name")) else "",
            "engagement_name": str(row.get("engagement_name", "")) if pd.notna(row.get("engagement_name")) else "",
            "file_path":       str(row.get("file_path", "")) if pd.notna(row.get("file_path")) else "",
            "line":            int(row["line"]) if pd.notna(row.get("line")) else None,
            "description":     str(row.get("description", ""))[:200] if pd.notna(row.get("description")) else "",
        })

    confusion = {}
    if "severity_num" in df.columns:
        for sev_num in sorted(df["severity_num"].unique()):
            sev_label          = SEVERITY_MAP_INV.get(int(sev_num), str(sev_num))
            sub                = df[df["severity_num"] == sev_num]
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


def save_json_report(stats: dict, df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    export_cols = [c for c in [
        "id", "title",
        "product_id", "product_name",
        "engagement_id", "engagement_name",
        "file_path", "line", "description",
        "predicted_class", "predicted_level",
        "risk_score", "confidence",
        "cvss_score", "epss_score", "epss_percentile",
        "age_days", "has_cve", "has_cwe",
        "is_active", "is_false_positive",
        "scored_at",
    ] if c in df.columns]

    report = {
        "meta": {
            "report_version": "2.2",
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
        "all_findings":          json.loads(
            df[export_cols].to_json(orient="records", default_handler=str)
        ),
    }

    path = output_dir / "scoring_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON report → {path}")
    return path


def save_csv_scored(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "findings_scored.csv"
    df.to_csv(path, index=False)
    logger.info(f"CSV scored → {path} ({len(df)} rows)")
    return path


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
    for i, f in enumerate(top[:20], 1):
        lvl          = f["predicted_level"]
        prod_display = f.get("product_name") or str(f.get("product_id", "—"))
        cve_tag      = "CVE" if f.get("has_cve") else "—"
        epss_val     = f"{f.get('epss_score', 0.0):.3f}"
        top_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9"
            onmouseover="this.style.background='#f8fafc'"
            onmouseout="this.style.background='transparent'">
          <td style="padding:12px 16px;font-weight:700;color:#64748b;font-size:0.8rem">{i}</td>
          <td style="padding:12px 8px">{badge(lvl)}</td>
          <td style="padding:12px 16px;font-size:0.85rem;max-width:280px;
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
              title="{f['title']}">{f['title']}</td>
          <td style="padding:12px 16px;min-width:150px">{score_bar(f['risk_score'])}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;color:#64748b">{f['cvss_score']}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;color:#64748b">{epss_val}</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;color:#64748b">{f['age_days']}j</td>
          <td style="padding:12px 16px;text-align:center;font-size:0.82rem;
                     color:{'#22c55e' if cve_tag=='CVE' else '#cbd5e1'}">{cve_tag}</td>
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

    df_display = df.sort_values("risk_score", ascending=False).head(500)
    full_rows  = ""
    for _, row in df_display.iterrows():
        lvl        = row.get("predicted_level", "Unknown")
        title      = str(row.get("title", ""))[:60]
        risk_score = row.get("risk_score", 0)
        file_path  = str(row.get("file_path", ""))[:40] if pd.notna(row.get("file_path")) else "—"
        line_str   = str(int(row["line"])) if pd.notna(row.get("line")) else "—"
        desc_raw   = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
        desc       = (desc_raw[:60] + "…") if len(desc_raw) > 60 else desc_raw
        prod_name  = str(row.get("product_name", row.get("product_id", "")))[:20]
        eng_name   = str(row.get("engagement_name", ""))[:20] if pd.notna(row.get("engagement_name")) else "—"
        epss_disp  = f"{float(row.get('epss_score', 0)):.3f}"
        full_rows += f"""
        <tr style="border-bottom:1px solid #f1f5f9">
          <td style="padding:8px 12px;font-size:0.8rem">{row.get('id','')}</td>
          <td style="padding:8px 12px">{badge(lvl)}</td>
          <td style="padding:8px 12px;font-size:0.8rem;max-width:200px;overflow:hidden;
                     text-overflow:ellipsis" title="{row.get('title','')}">{title}</td>
          <td style="padding:8px 12px;min-width:100px">{score_bar(risk_score)}</td>
          <td style="padding:8px 12px;text-align:center;font-size:0.8rem;color:#64748b">{epss_disp}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{file_path}</td>
          <td style="padding:8px 12px;text-align:center;font-size:0.8rem;color:#64748b">{line_str}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b;max-width:150px;
                     overflow:hidden;text-overflow:ellipsis">{desc}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{prod_name}</td>
          <td style="padding:8px 12px;font-size:0.8rem;color:#64748b">{eng_name}</td>
        </tr>"""

    if len(df) > 500:
        full_rows += (
            f'<tr><td colspan="10" style="padding:16px;text-align:center;'
            f'color:#94a3b8;font-style:italic">'
            f'… et {len(df)-500} autres findings (voir CSV/JSON)</td></tr>'
        )

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
    .section-title{{font-size:0.9rem;font-weight:700;color:#0f172a}}
    .section-body{{padding:20px 24px}}
    .dist-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}
    table{{width:100%;border-collapse:collapse}}
    th{{padding:10px 16px;font-size:0.72rem;font-weight:700;text-transform:uppercase;
        letter-spacing:0.06em;color:var(--muted);text-align:left;background:#f8fafc;
        border-bottom:1px solid var(--border)}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
    .alert-banner{{border-radius:12px;padding:14px 20px;display:flex;align-items:center;
                   gap:14px;margin-bottom:24px;font-size:0.88rem}}
    @media(max-width:900px){{.dist-grid{{grid-template-columns:repeat(3,1fr)}}.two-col{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<div class="topbar">
  <div class="topbar-brand"><div class="dot"></div>AI Risk Engine</div>
  <div class="topbar-meta">Generated {gen_at} UTC &nbsp;|&nbsp; Model: {model_ver}</div>
</div>
<div class="container">
  <div class="page-header">
    <h1>Vulnerability Scoring Report</h1>
    <p>{total} findings analyzed &nbsp;&middot;&nbsp; Global score: <strong>{stats['global_score']}/10</strong>
       &nbsp;&middot;&nbsp; Level: <strong style="color:{risk_color_global}">{risk_level_global}</strong></p>
  </div>

  {"" if critical_rate < 20 else f'''
  <div class="alert-banner" style="background:#fef2f2;border:1.5px solid #fca5a5">
    <span style="font-size:1.4rem">&#128680;</span>
    <div><strong style="color:#dc2626">{critical_rate}% of findings are High or Critical.</strong>
    <span style="color:#ef4444;margin-left:8px">Immediate action recommended on {dist.get("Critical",0)} critical findings.</span></div>
  </div>'''}

  <div class="kpi-grid">
    <div class="kpi-card" style="--accent:{risk_color_global}">
      <div class="kpi-label">Global Score</div>
      <div class="kpi-value" style="color:{risk_color_global}">{stats['global_score']}</div>
      <div class="kpi-sub">out of 10 &middot; {risk_level_global}</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total Findings</div>
      <div class="kpi-value">{total}</div>
      <div class="kpi-sub">analyzed by AI</div>
    </div>
    <div class="kpi-card" style="--accent:{CLASS_COLORS['Critical']}">
      <div class="kpi-label">Critical</div>
      <div class="kpi-value" style="color:{CLASS_COLORS['Critical']}">{dist.get('Critical',0)}</div>
      <div class="kpi-sub">{pct.get('Critical',0)}% of total</div>
    </div>
    <div class="kpi-card" style="--accent:{CLASS_COLORS['High']}">
      <div class="kpi-label">High</div>
      <div class="kpi-value" style="color:{CLASS_COLORS['High']}">{dist.get('High',0)}</div>
      <div class="kpi-sub">{pct.get('High',0)}% of total</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Median Score</div>
      <div class="kpi-value" style="font-size:1.8rem">{stats['median_score']}</div>
      <div class="kpi-sub">max: {stats['max_score']}</div>
    </div>
  </div>

  <div class="two-col">
    <div class="section">
      <div class="section-header"><div class="section-title">Distribution by level</div></div>
      <div class="section-body"><div class="dist-grid">{dist_cards}</div></div>
    </div>
    <div class="section">
      <div class="section-header"><div class="section-title">Visual breakdown</div></div>
      <div class="section-body" style="display:flex;align-items:center;justify-content:center;height:200px">
        <canvas id="distChart" style="max-height:190px"></canvas>
      </div>
    </div>
  </div>

  <div class="section">
    <div class="section-header">
      <div class="section-title">Top Findings (by AI score descending)</div>
      <span style="font-size:0.78rem;color:#64748b">{len(top)} findings</span>
    </div>
    <div style="overflow-x:auto"><table>
      <thead><tr>
        <th>#</th><th>Level</th><th>Title</th><th>AI Score</th>
        <th>CVSS</th><th>EPSS</th><th>Age</th><th>CVE</th><th>Product</th><th>Confidence</th>
      </tr></thead>
      <tbody>{top_rows}</tbody>
    </table></div>
  </div>

  {"" if not by_product else f'''
  <div class="section">
    <div class="section-header"><div class="section-title">Risk by Product (top 10)</div></div>
    <div style="overflow-x:auto"><table>
      <thead><tr>
        <th>Product</th><th>Findings</th><th>Critical</th><th>High</th>
        <th>Mean Score</th><th>Max Score</th>
      </tr></thead>
      <tbody>{prod_rows}</tbody>
    </table></div>
  </div>'''}

  <div class="section">
    <div class="section-header">
      <div class="section-title">All findings (top 500 by score)</div>
      <span style="font-size:0.78rem;color:#64748b">{len(df)} total</span>
    </div>
    <div style="overflow-x:auto;max-height:500px;overflow-y:auto"><table>
      <thead><tr>
        <th>ID</th><th>Level</th><th>Title</th><th>AI Score</th>
        <th>EPSS</th><th>File</th><th>Line</th><th>Description</th><th>Product</th><th>Engagement</th>
      </tr></thead>
      <tbody>{full_rows}</tbody>
    </table></div>
  </div>

  <div style="text-align:center;margin-top:40px;color:#94a3b8;font-size:0.78rem">
    <span style="font-family:'IBM Plex Mono',monospace">AI Risk Engine v2.2</span>
    &nbsp;&middot;&nbsp; {gen_at} UTC &nbsp;&middot;&nbsp; Model: {model_ver}
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
    logger.info(f"HTML report → {path}")
    return path

def run(
    source: str = "processed",
    output_dir: Path = REPORTS_DIR,
    product_id: Optional[int] = None,
) -> dict:
    logger.info("🚀 AI Risk Engine — Scoring Pipeline v2.2")

    pipeline, meta, model_features = load_model()
    df = load_data(product_id=product_id)
    df_scored = run_scoring(df, pipeline, model_features)
    stats = compute_stats(df_scored)

    output_dir = Path(output_dir)
    json_path = save_json_report(stats, df_scored, output_dir)
    csv_path = save_csv_scored(df_scored, output_dir)
    html_path = save_html_report(stats, meta, df_scored, output_dir)

    logger.info(f"🎯 Done — Global Score: {stats['global_score']}/10 | Critical: {stats['level_dist'].get('Critical', 0)}")
    logger.info(f"HTML → {html_path}")

    return {
        "stats": stats,
        "html_path": str(html_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Risk Engine — Automatic vulnerability scoring")
    parser.add_argument("--source", choices=["processed", "raw"], default="processed")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--product-id", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(source=args.source, output_dir=args.output_dir, product_id=args.product_id)