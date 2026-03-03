"""
scripts/generate_ia_report.py — AI Risk Engine v2.0
=====================================================
Génère un rapport HTML autonome, professionnel et interactif
à partir du scoring IA des vulnérabilités.

Esthétique : Dark SOC (Security Operations Center)
  — Police monospace pour les données, display pour les titres
  — Palette sombre avec accents néon par niveau de risque
  — Animations d'entrée fluides, micro-interactions hover
  — Tableau filtrable/triable en JavaScript pur
  — Graphiques Chart.js intégrés (doughnut + bar + timeline)
  — 100% autonome (aucune dépendance serveur, un seul fichier .html)

Usage :
  python scripts/generate_ia_report.py
  python scripts/generate_ia_report.py --input reports/scoring_report.json
  python scripts/generate_ia_report.py --output reports/custom_report.html
  python scripts/generate_ia_report.py --title "Audit Q1 2026"
  python scripts/generate_ia_report.py --open   # Ouvre dans le navigateur
"""

import argparse
import json
import logging
import os
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/report.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_engine.report")

# ──────────────────────────────────────────────
# Chemins par défaut
# ──────────────────────────────────────────────
SRC_DIR     = Path(__file__).parent.parent
REPORTS_DIR = SRC_DIR / "reports"
DEFAULT_JSON = REPORTS_DIR / "scoring_report.json"
DEFAULT_HTML = REPORTS_DIR / "ia_report.html"

# ──────────────────────────────────────────────
# Palette de couleurs SOC
# ──────────────────────────────────────────────
COLORS = {
    "Critical": {"hex": "#ff3b5c", "glow": "rgba(255,59,92,0.35)",  "bg": "rgba(255,59,92,0.08)"},
    "High":     {"hex": "#ff8c42", "glow": "rgba(255,140,66,0.35)", "bg": "rgba(255,140,66,0.08)"},
    "Medium":   {"hex": "#ffd166", "glow": "rgba(255,209,102,0.3)", "bg": "rgba(255,209,102,0.07)"},
    "Low":      {"hex": "#06d6a0", "glow": "rgba(6,214,160,0.3)",   "bg": "rgba(6,214,160,0.07)"},
    "Info":     {"hex": "#4cc9f0", "glow": "rgba(76,201,240,0.3)",  "bg": "rgba(76,201,240,0.07)"},
}
LEVEL_ORDER = ["Critical", "High", "Medium", "Low", "Info"]


# ══════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ══════════════════════════════════════════════

def load_report_data(json_path: Path) -> dict:
    """Charge le rapport JSON produit par scoring.py."""
    if not json_path.exists():
        raise FileNotFoundError(
            f"Rapport JSON introuvable : {json_path}\n"
            "Exécutez d'abord : python scoring.py"
        )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Données chargées : {json_path.name}")
    return data


def _safe_get(data: dict, *keys, default=None):
    """Accès sécurisé à une clé imbriquée."""
    val = data
    for k in keys:
        if not isinstance(val, dict):
            return default
        val = val.get(k, default)
    return val if val is not None else default


# ══════════════════════════════════════════════
# 2. COMPOSANTS HTML
# ══════════════════════════════════════════════

def _level_badge(level: str, size: str = "md") -> str:
    """Badge coloré avec glow effect pour un niveau de risque."""
    c = COLORS.get(level, COLORS["Info"])
    fs = "0.65rem" if size == "sm" else "0.72rem" if size == "md" else "0.82rem"
    return (
        f'<span class="badge badge-{level.lower()}" '
        f'style="color:{c["hex"]};background:{c["bg"]};'
        f'border:1px solid {c["hex"]}44;font-size:{fs}">'
        f'{level}</span>'
    )


def _score_pill(score: float) -> str:
    """Pill avec couleur et barre proportionnelle."""
    if score >= 8:   lvl = "Critical"
    elif score >= 6: lvl = "High"
    elif score >= 4: lvl = "Medium"
    elif score >= 2: lvl = "Low"
    else:            lvl = "Info"
    c   = COLORS[lvl]["hex"]
    pct = min(score / 10 * 100, 100)
    return (
        f'<div class="score-pill">'
        f'<div class="score-bar-track">'
        f'<div class="score-bar-fill" style="width:{pct:.0f}%;background:{c};'
        f'box-shadow:0 0 6px {COLORS[lvl]["glow"]}"></div>'
        f'</div>'
        f'<span style="color:{c};font-weight:700;font-size:0.82rem;min-width:32px">'
        f'{score:.1f}</span>'
        f'</div>'
    )


def _confidence_icon(conf: float) -> str:
    """Icône de confiance colorée."""
    if conf >= 0.85: return f'<span style="color:#06d6a0" title="Confiance élevée">◉ {conf:.0%}</span>'
    if conf >= 0.65: return f'<span style="color:#ffd166" title="Confiance moyenne">◎ {conf:.0%}</span>'
    return f'<span style="color:#ff8c42" title="Confiance faible">○ {conf:.0%}</span>'


def _build_kpi_cards(summary: dict) -> str:
    """Génère les 6 cartes KPI principales."""
    dist     = summary.get("distribution", {})
    total    = summary.get("total_findings", 0)
    g_score  = summary.get("global_risk_score", 0)
    m_score  = summary.get("median_risk_score", 0)
    critical = dist.get("Critical", 0)
    high     = dist.get("High", 0)
    urgent   = critical + high
    urgent_pct = round(urgent / max(total, 1) * 100, 1)

    if g_score >= 7:   risk_label, risk_lvl = "CRITIQUE",  "Critical"
    elif g_score >= 5: risk_label, risk_lvl = "ÉLEVÉ",     "High"
    elif g_score >= 3: risk_label, risk_lvl = "MODÉRÉ",    "Medium"
    else:              risk_label, risk_lvl = "FAIBLE",     "Low"
    rc = COLORS[risk_lvl]["hex"]
    rg = COLORS[risk_lvl]["glow"]

    kpis = [
        {
            "label": "SCORE GLOBAL",
            "value": f"{g_score}",
            "unit":  "/ 10",
            "sub":   f"Niveau {risk_label}",
            "color": rc,
            "glow":  rg,
            "icon":  "⬡",
        },
        {
            "label": "TOTAL FINDINGS",
            "value": f"{total:,}",
            "unit":  "",
            "sub":   "analysés par l'IA",
            "color": "#4cc9f0",
            "glow":  COLORS["Info"]["glow"],
            "icon":  "◈",
        },
        {
            "label": "CRITIQUES",
            "value": str(critical),
            "unit":  "",
            "sub":   f"{dist.get('Critical', 0) / max(total,1)*100:.1f}% du total",
            "color": COLORS["Critical"]["hex"],
            "glow":  COLORS["Critical"]["glow"],
            "icon":  "⚠",
        },
        {
            "label": "HIGH",
            "value": str(high),
            "unit":  "",
            "sub":   f"{high / max(total,1)*100:.1f}% du total",
            "color": COLORS["High"]["hex"],
            "glow":  COLORS["High"]["glow"],
            "icon":  "▲",
        },
        {
            "label": "URGENT (C+H)",
            "value": str(urgent),
            "unit":  "",
            "sub":   f"{urgent_pct}% nécessitent action",
            "color": "#c77dff",
            "glow":  "rgba(199,125,255,0.3)",
            "icon":  "⚡",
        },
        {
            "label": "SCORE MÉDIAN",
            "value": f"{m_score}",
            "unit":  "/ 10",
            "sub":   f"max : {summary.get('max_risk_score', 0)}",
            "color": "#9d8df1",
            "glow":  "rgba(157,141,241,0.3)",
            "icon":  "◐",
        },
    ]

    cards = ""
    for i, k in enumerate(kpis):
        cards += f"""
        <div class="kpi-card fade-in" style="animation-delay:{i*0.07}s;--card-color:{k['color']};--card-glow:{k['glow']}">
          <div class="kpi-icon">{k['icon']}</div>
          <div class="kpi-label">{k['label']}</div>
          <div class="kpi-value" style="color:{k['color']}">{k['value']}<span class="kpi-unit">{k['unit']}</span></div>
          <div class="kpi-sub">{k['sub']}</div>
        </div>"""
    return cards


def _build_dist_bars(summary: dict) -> str:
    """Barres de distribution horizontales animées."""
    dist  = summary.get("distribution", {})
    total = summary.get("total_findings", 1)
    rows  = ""
    for lvl in LEVEL_ORDER:
        count = dist.get(lvl, 0)
        pct   = count / total * 100
        c     = COLORS[lvl]
        rows += f"""
          <div class="dist-row fade-in">
            <div class="dist-label">{_level_badge(lvl)}</div>
            <div class="dist-bar-wrap">
              <div class="dist-bar-fill" style="width:0%;background:{c['hex']};
                   box-shadow:0 0 10px {c['glow']}" data-target="{pct:.1f}"></div>
            </div>
            <div class="dist-count">{count}</div>
            <div class="dist-pct" style="color:{c['hex']}">{pct:.1f}%</div>
          </div>"""
    return rows


def _build_findings_table(findings: list) -> str:
    """Tableau interactif complet de TOUS les findings scorés."""
    if not findings:
        return '<tr><td colspan="11" class="empty-state">Aucun finding à afficher</td></tr>'

    rows = ""
    for i, f in enumerate(findings):
        lvl    = f.get("predicted_level", "Info")
        title  = str(f.get("title", "N/A"))
        title_disp = (title[:65] + "…") if len(title) > 65 else title
        fid    = f.get("id") or "—"
        score  = float(f.get("risk_score", 0))
        conf   = float(f.get("confidence", 0))
        cvss   = float(f.get("cvss_score", 0))
        age    = int(f.get("age_days", 0))
        has_cve = f.get("has_cve", 0)
        sev_num = int(f.get("severity_num", 0))
        sev_map = {4: "Critical", 3: "High", 2: "Medium", 1: "Low", 0: "Info"}
        sev_orig = sev_map.get(sev_num, "?")

        # Product + Engagement
        prod_name = str(f.get("product_name") or f.get("product_id") or "—")
        eng_name  = str(f.get("engagement_name") or "—")

        # File path + line
        file_path = str(f.get("file_path") or "")
        line_num  = f.get("line")
        if file_path:
            # Affiche seulement le nom de fichier court + chemin en tooltip
            short_path = file_path.replace("\\", "/").split("/")[-1]
            full_path  = file_path.replace("\\", "/")
            line_str   = f":{line_num}" if line_num else ""
            location_html = (
                f'<span class="file-path" title="{full_path}{line_str}">'
                f'<span class="file-icon">📄</span>'
                f'<span class="file-name">{short_path}</span>'
                f'{"<span class=\'file-line\'>L" + str(line_num) + "</span>" if line_num else ""}'
                f'</span>'
            )
        else:
            location_html = '<span class="muted">—</span>'

        # Divergence sévérité déclarée vs prédite
        diverge_attr = ' class="row-diverge"' if sev_orig != lvl else ""
        diverge_title = f' title="⚠ Déclaré {sev_orig} → Prédit {lvl}"' if sev_orig != lvl else ""

        rows += f"""
        <tr data-level="{lvl}" data-score="{score}"{diverge_attr}{diverge_title}>
          <td class="td-rank">{i+1}</td>
          <td class="td-badge">{_level_badge(lvl)}</td>
          <td class="td-title" title="{title}">{title_disp}</td>
          <td class="td-score">{_score_pill(score)}</td>
          <td class="td-product">
            <div class="product-cell">
              <span class="product-name">{prod_name}</span>
              <span class="engagement-name">{eng_name}</span>
            </div>
          </td>
          <td class="td-location">{location_html}</td>
          <td class="td-cvss"><span class="mono" style="color:{'#ff3b5c' if cvss>=9 else '#ff8c42' if cvss>=7 else '#ffd166' if cvss>=4 else '#06d6a0'}">{cvss:.1f}</span></td>
          <td class="td-age"><span class="mono" style="color:{'#ff8c42' if age>90 else '#4cc9f0'}">{age}j</span></td>
          <td class="td-cve">{'<span class="cve-yes">CVE</span>' if has_cve else '<span class="cve-no">—</span>'}</td>
          <td class="td-conf">{_confidence_icon(conf)}</td>
          <td class="td-id"><span class="mono muted">#{fid}</span></td>
        </tr>"""
    return rows


def _build_product_table(by_product: list) -> str:
    """Tableau des risques par produit avec nom et engagements."""
    if not by_product:
        return '<tr><td colspan="7" class="empty-state">Aucun produit</td></tr>'
    rows = ""
    for p in by_product[:20]:
        pid       = p.get("product_id", "?")
        pname     = p.get("product_name") or str(pid)
        eng       = p.get("engagements", "—")
        total     = p.get("total", 0)
        crit      = p.get("critical_count", 0)
        high      = p.get("high_count", 0)
        medium    = p.get("medium_count", 0)
        mean_s    = float(p.get("mean_risk_score", 0))
        ps        = float(p.get("priority_score", 0))
        max_ps    = max((x.get("priority_score", 1) for x in by_product), default=1)
        bar_w     = ps / max_ps * 100
        rows += f"""
        <tr>
          <td>
            <div class="product-cell">
              <span class="product-name" style="color:#e2eaf5">{pname}</span>
              <span class="engagement-name">ID: {pid}</span>
            </div>
          </td>
          <td class="td-center" style="color:#94a3b8;font-family:var(--font-mono);font-size:0.78rem">{eng}</td>
          <td class="td-center mono">{total}</td>
          <td class="td-center"><span style="color:{COLORS['Critical']['hex']};font-weight:700">{crit}</span></td>
          <td class="td-center"><span style="color:{COLORS['High']['hex']};font-weight:700">{high}</span></td>
          <td class="td-center"><span style="color:{COLORS['Medium']['hex']};font-weight:700">{medium}</span></td>
          <td style="min-width:140px">{_score_pill(mean_s)}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="flex:1;height:4px;background:#1e2a3a;border-radius:2px">
                <div style="width:{bar_w:.0f}%;height:100%;background:#c77dff;border-radius:2px;
                     box-shadow:0 0 6px rgba(199,125,255,0.4)"></div>
              </div>
              <span class="mono" style="color:#c77dff;font-size:0.75rem">{ps:.0f}</span>
            </div>
          </td>
        </tr>"""
    return rows


def _build_chart_data(summary: dict) -> str:
    """Prépare les données JSON pour Chart.js."""
    dist = summary.get("distribution", {})
    counts  = [dist.get(l, 0) for l in LEVEL_ORDER]
    colors  = [COLORS[l]["hex"] for l in LEVEL_ORDER]
    glows   = [COLORS[l]["glow"] for l in LEVEL_ORDER]

    by_product = summary.get("by_product", []) if isinstance(summary.get("by_product"), list) else []
    # Prendre seulement si vient de la clé top-level (scoring.py met by_product dans summary via stats)
    prod_labels = [f"P-{p['product_id']}" for p in by_product[:8]]
    prod_scores = [round(p.get("mean_risk_score", 0), 2) for p in by_product[:8]]
    prod_colors = [
        COLORS["Critical"]["hex"] if s >= 7 else
        COLORS["High"]["hex"]     if s >= 5 else
        COLORS["Medium"]["hex"]   if s >= 3 else
        COLORS["Low"]["hex"]
        for s in prod_scores
    ]

    return f"""
    const DIST_LABELS  = {json.dumps(LEVEL_ORDER)};
    const DIST_COUNTS  = {json.dumps(counts)};
    const DIST_COLORS  = {json.dumps(colors)};
    const DIST_GLOWS   = {json.dumps(glows)};
    const PROD_LABELS  = {json.dumps(prod_labels)};
    const PROD_SCORES  = {json.dumps(prod_scores)};
    const PROD_COLORS  = {json.dumps(prod_colors)};
    """


# ══════════════════════════════════════════════
# 3. TEMPLATE HTML PRINCIPAL
# ══════════════════════════════════════════════

def generate_html(data: dict, title: str = "AI Risk Engine — Rapport de Scoring") -> str:
    """Assemble le rapport HTML complet."""

    summary    = _safe_get(data, "summary", default={})
    meta       = _safe_get(data, "meta", default={})
    findings   = _safe_get(data, "all_findings", default=[])
    # by_product est à la racine du JSON (pas dans summary)
    by_product = _safe_get(data, "by_product", default=[])

    # Injecter by_product dans summary uniquement pour les graphiques Chart.js
    summary["by_product"] = by_product

    total     = summary.get("total_findings", len(findings))
    gen_at    = meta.get("generated_at", datetime.now(timezone.utc).isoformat())[:19].replace("T", " ")
    model_ver = meta.get("model_version", _safe_get(meta, "engine", default="v2.0"))

    dist = summary.get("distribution", {})
    critical = dist.get("Critical", 0)
    high     = dist.get("High", 0)
    show_alert = (critical + high) / max(total, 1) > 0.15

    # Trier les findings par score décroissant
    findings_sorted = sorted(findings, key=lambda x: float(x.get("risk_score", 0)), reverse=True)

    kpi_cards     = _build_kpi_cards(summary)
    dist_bars     = _build_dist_bars(summary)
    findings_rows = _build_findings_table(findings_sorted)
    product_rows  = _build_product_table(by_product)
    chart_data    = _build_chart_data(summary)

    alert_html = ""
    if show_alert:
        alert_html = f"""
        <div class="alert-critical fade-in">
          <div class="alert-icon">⚠</div>
          <div class="alert-body">
            <div class="alert-title">ACTION IMMÉDIATE REQUISE</div>
            <div class="alert-text">
              <strong>{critical} findings Critiques</strong> et <strong>{high} findings High</strong>
              ont été identifiés — soit <strong>{(critical+high)/max(total,1)*100:.1f}%</strong>
              du portefeuille total. Priorisation immédiate recommandée.
            </div>
          </div>
          <div class="alert-score">{critical + high}<span>urgents</span></div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Clash+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    /* ─── Reset & Base ─── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg-base:    #080d14;
      --bg-surface: #0d1520;
      --bg-card:    #111c2b;
      --bg-raised:  #162033;
      --border:     #1e2d42;
      --border-lit: #2a3f5c;
      --text:       #e2eaf5;
      --text-muted: #5a7394;
      --text-dim:   #3a5070;
      --accent:     #3b82f6;
      --font-mono:  'JetBrains Mono', monospace;
      --font-ui:    'DM Sans', sans-serif;
      --font-display: 'Clash Display', 'DM Sans', sans-serif;
      --radius:     10px;
      --radius-lg:  16px;
      --shadow:     0 4px 24px rgba(0,0,0,0.5);
      --transition: 0.18s cubic-bezier(0.4,0,0.2,1);
    }}

    html {{ scroll-behavior: smooth; }}

    body {{
      font-family: var(--font-ui);
      background: var(--bg-base);
      color: var(--text);
      min-height: 100vh;
      font-size: 14px;
      line-height: 1.6;
    }}

    /* ─── Scanline texture overlay ─── */
    body::before {{
      content: '';
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
      );
      pointer-events: none;
      z-index: 1000;
    }}

    /* ─── Noise grain ─── */
    body::after {{
      content: '';
      position: fixed;
      inset: 0;
      background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
      pointer-events: none;
      z-index: 999;
      opacity: 0.4;
    }}

    /* ─── Animations ─── */
    @keyframes fadeInUp {{
      from {{ opacity: 0; transform: translateY(16px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes scanIn {{
      from {{ opacity: 0; clip-path: inset(0 100% 0 0); }}
      to   {{ opacity: 1; clip-path: inset(0 0% 0 0); }}
    }}
    @keyframes glow-pulse {{
      0%, 100% {{ opacity: 0.6; }}
      50%       {{ opacity: 1; }}
    }}
    @keyframes bar-grow {{
      from {{ width: 0 !important; }}
    }}
    .fade-in {{
      animation: fadeInUp 0.5s ease both;
    }}

    /* ─── Topbar ─── */
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 200;
      background: rgba(8,13,20,0.92);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border);
      height: 52px;
      display: flex;
      align-items: center;
      padding: 0 28px;
      justify-content: space-between;
      gap: 16px;
    }}
    .topbar-left {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .topbar-logo {{
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 0.82rem;
      color: var(--text);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .topbar-logo .live-dot {{
      width: 7px; height: 7px;
      background: #06d6a0;
      border-radius: 50%;
      animation: glow-pulse 2s ease infinite;
      box-shadow: 0 0 6px #06d6a0;
    }}
    .topbar-sep {{ width: 1px; height: 20px; background: var(--border); }}
    .topbar-meta {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--text-muted);
    }}
    .topbar-right {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .topbar-tag {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      padding: 3px 10px;
      border-radius: 20px;
      border: 1px solid var(--border-lit);
      color: var(--text-muted);
      letter-spacing: 0.04em;
    }}

    /* ─── Layout ─── */
    .container {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 32px 24px 80px;
    }}

    /* ─── Page header ─── */
    .page-header {{
      margin-bottom: 36px;
      padding-bottom: 28px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }}
    .page-header h1 {{
      font-family: var(--font-display);
      font-size: 1.9rem;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.03em;
      line-height: 1.1;
    }}
    .page-header h1 span {{
      display: block;
      font-size: 0.78rem;
      font-family: var(--font-mono);
      font-weight: 400;
      color: var(--text-muted);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .header-meta {{
      text-align: right;
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--text-muted);
      line-height: 2;
    }}

    /* ─── Alert banner ─── */
    .alert-critical {{
      background: rgba(255,59,92,0.06);
      border: 1px solid rgba(255,59,92,0.3);
      border-left: 3px solid #ff3b5c;
      border-radius: var(--radius-lg);
      padding: 18px 24px;
      display: flex;
      align-items: center;
      gap: 18px;
      margin-bottom: 28px;
      box-shadow: 0 0 30px rgba(255,59,92,0.08);
    }}
    .alert-icon {{
      font-size: 1.6rem;
      color: #ff3b5c;
      flex-shrink: 0;
      animation: glow-pulse 1.5s ease infinite;
    }}
    .alert-body {{ flex: 1; }}
    .alert-title {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      color: #ff3b5c;
      margin-bottom: 4px;
    }}
    .alert-text {{ font-size: 0.86rem; color: #c0cfe0; }}
    .alert-text strong {{ color: #ff3b5c; }}
    .alert-score {{
      font-family: var(--font-mono);
      font-size: 2rem;
      font-weight: 700;
      color: #ff3b5c;
      text-align: center;
      flex-shrink: 0;
      line-height: 1;
    }}
    .alert-score span {{
      display: block;
      font-size: 0.65rem;
      color: var(--text-muted);
      letter-spacing: 0.06em;
    }}

    /* ─── KPI grid ─── */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      padding: 20px 22px 18px;
      position: relative;
      overflow: hidden;
      cursor: default;
      transition: border-color var(--transition), box-shadow var(--transition), transform var(--transition);
    }}
    .kpi-card::before {{
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at 20% 20%, var(--card-glow, transparent) 0%, transparent 60%);
      opacity: 0;
      transition: opacity 0.3s;
    }}
    .kpi-card:hover {{
      border-color: var(--card-color, var(--border-lit));
      box-shadow: 0 0 20px var(--card-glow, transparent), var(--shadow);
      transform: translateY(-2px);
    }}
    .kpi-card:hover::before {{ opacity: 1; }}
    .kpi-icon {{
      font-size: 1.1rem;
      color: var(--card-color, var(--text-muted));
      margin-bottom: 10px;
      opacity: 0.8;
    }}
    .kpi-label {{
      font-family: var(--font-mono);
      font-size: 0.62rem;
      letter-spacing: 0.12em;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .kpi-value {{
      font-family: var(--font-display);
      font-size: 2.1rem;
      font-weight: 700;
      line-height: 1;
      letter-spacing: -0.02em;
    }}
    .kpi-unit {{
      font-size: 0.9rem;
      font-weight: 400;
      color: var(--text-muted);
      margin-left: 2px;
    }}
    .kpi-sub {{
      font-size: 0.73rem;
      color: var(--text-muted);
      margin-top: 8px;
    }}

    /* ─── Cards / Sections ─── */
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      overflow: hidden;
      margin-bottom: 20px;
    }}
    .card-header {{
      padding: 16px 22px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--bg-raised);
    }}
    .card-title {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .card-title-icon {{
      color: var(--accent);
      font-size: 0.9rem;
    }}
    .card-body {{ padding: 20px 22px; }}
    .card-count {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--text-muted);
      background: var(--bg-base);
      padding: 3px 10px;
      border-radius: 20px;
      border: 1px solid var(--border);
    }}

    /* ─── 2-col grid ─── */
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .three-col {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}

    /* ─── Distribution bars ─── */
    .dist-row {{
      display: grid;
      grid-template-columns: 90px 1fr 50px 60px;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .dist-label {{ font-size: 0.78rem; }}
    .dist-bar-wrap {{
      height: 8px;
      background: var(--bg-base);
      border-radius: 4px;
      overflow: hidden;
    }}
    .dist-bar-fill {{
      height: 100%;
      border-radius: 4px;
      transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
    }}
    .dist-count {{
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: var(--text);
      text-align: right;
    }}
    .dist-pct {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      text-align: right;
    }}

    /* ─── Badges ─── */
    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-family: var(--font-mono);
      font-weight: 600;
      letter-spacing: 0.05em;
      white-space: nowrap;
    }}

    /* ─── Score pill ─── */
    .score-pill {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .score-bar-track {{
      flex: 1;
      height: 5px;
      background: var(--bg-base);
      border-radius: 3px;
      overflow: hidden;
    }}
    .score-bar-fill {{
      height: 100%;
      border-radius: 3px;
      transition: width 0.8s ease;
    }}

    /* ─── Filter bar ─── */
    .filter-bar {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .filter-input {{
      background: var(--bg-base);
      border: 1px solid var(--border-lit);
      border-radius: 8px;
      padding: 7px 14px 7px 36px;
      color: var(--text);
      font-family: var(--font-mono);
      font-size: 0.75rem;
      outline: none;
      transition: border-color var(--transition);
      width: 220px;
      position: relative;
    }}
    .filter-input:focus {{ border-color: var(--accent); }}
    .search-wrap {{ position: relative; }}
    .search-wrap::before {{
      content: '⌕';
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-muted);
      font-size: 0.9rem;
      pointer-events: none;
    }}
    .filter-btn {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid var(--border-lit);
      background: transparent;
      color: var(--text-muted);
      cursor: pointer;
      letter-spacing: 0.05em;
      transition: all var(--transition);
    }}
    .filter-btn:hover,
    .filter-btn.active {{
      color: var(--text);
      border-color: var(--btn-color, var(--accent));
      background: var(--btn-bg, rgba(59,130,246,0.08));
      box-shadow: 0 0 12px var(--btn-glow, rgba(59,130,246,0.2));
    }}
    .filter-btn-critical {{
      --btn-color: {COLORS['Critical']['hex']};
      --btn-bg:    {COLORS['Critical']['bg']};
      --btn-glow:  {COLORS['Critical']['glow']};
    }}
    .filter-btn-high {{
      --btn-color: {COLORS['High']['hex']};
      --btn-bg:    {COLORS['High']['bg']};
      --btn-glow:  {COLORS['High']['glow']};
    }}
    .filter-btn-medium {{
      --btn-color: {COLORS['Medium']['hex']};
      --btn-bg:    {COLORS['Medium']['bg']};
      --btn-glow:  {COLORS['Medium']['glow']};
    }}
    .filter-btn-low {{
      --btn-color: {COLORS['Low']['hex']};
      --btn-bg:    {COLORS['Low']['bg']};
      --btn-glow:  {COLORS['Low']['glow']};
    }}
    .filter-btn-info {{
      --btn-color: {COLORS['Info']['hex']};
      --btn-bg:    {COLORS['Info']['bg']};
      --btn-glow:  {COLORS['Info']['glow']};
    }}

    /* ─── Table ─── */
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{
      padding: 10px 14px;
      font-family: var(--font-mono);
      font-size: 0.62rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--text-muted);
      background: var(--bg-base);
      border-bottom: 1px solid var(--border);
      text-align: left;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
      transition: color var(--transition);
    }}
    thead th:hover {{ color: var(--text); }}
    thead th .sort-arrow {{ margin-left: 4px; opacity: 0.4; font-size: 0.6rem; }}
    thead th.sorted {{ color: var(--accent); }}
    thead th.sorted .sort-arrow {{ opacity: 1; color: var(--accent); }}
    tbody tr {{
      border-bottom: 1px solid var(--border);
      transition: background var(--transition);
    }}
    tbody tr:hover {{ background: var(--bg-raised); }}
    tbody tr.row-diverge {{ border-left: 2px solid #ffd166; }}
    tbody tr[data-level="Critical"] td:first-child {{
      border-left: 2px solid {COLORS['Critical']['hex']};
    }}
    tbody tr[data-level="High"] td:first-child {{
      border-left: 2px solid {COLORS['High']['hex']};
    }}
    td {{ padding: 10px 14px; font-size: 0.82rem; vertical-align: middle; }}
    .td-rank {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--text-dim);
      text-align: center;
      width: 36px;
    }}
    .td-badge  {{ white-space: nowrap; width: 90px; }}
    .td-title  {{ max-width: 300px; color: #c0cfe0; font-size: 0.83rem; }}
    .td-score  {{ min-width: 150px; }}
    .td-cvss, .td-age, .td-cve, .td-conf, .td-id {{ text-align: center; white-space: nowrap; }}
    .td-center {{ text-align: center; }}
    .cve-yes {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: #ff3b5c;
      background: rgba(255,59,92,0.1);
      border: 1px solid rgba(255,59,92,0.3);
      padding: 2px 7px;
      border-radius: 4px;
    }}
    .cve-no {{ color: var(--text-dim); font-size: 0.8rem; }}
    .empty-state {{ text-align: center; color: var(--text-muted); padding: 40px; }}
    .hidden {{ display: none !important; }}

    /* ─── Product / Engagement cell ─── */
    .product-cell {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .product-name {{
      font-size: 0.82rem;
      font-weight: 600;
      color: #c0cfe0;
      white-space: nowrap;
    }}
    .engagement-name {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: var(--text-muted);
      white-space: nowrap;
    }}

    /* ─── File path cell ─── */
    .file-path {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      background: var(--bg-base);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 2px 8px;
      cursor: help;
      max-width: 180px;
      overflow: hidden;
    }}
    .file-icon {{ font-size: 0.75rem; flex-shrink: 0; }}
    .file-name {{
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: #4cc9f0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .file-line {{
      font-family: var(--font-mono);
      font-size: 0.65rem;
      color: #ffd166;
      flex-shrink: 0;
    }}
    .td-product {{ min-width: 140px; }}
    .td-location {{ min-width: 160px; }}
    .muted {{ color: var(--text-muted); }}
    .section-sep {{
      height: 1px;
      background: linear-gradient(to right, transparent, var(--border), transparent);
      margin: 32px 0;
    }}

    /* ─── Chart containers ─── */
    .chart-wrap {{
      position: relative;
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    /* ─── Footer ─── */
    .footer {{
      text-align: center;
      padding: 40px 0 20px;
      font-family: var(--font-mono);
      font-size: 0.68rem;
      color: var(--text-dim);
      border-top: 1px solid var(--border);
      margin-top: 48px;
      letter-spacing: 0.06em;
    }}
    .footer a {{ color: var(--text-muted); text-decoration: none; }}
    .footer a:hover {{ color: var(--text); }}

    /* ─── Responsive ─── */
    @media (max-width: 1024px) {{
      .two-col, .three-col {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 700px) {{
      .kpi-grid {{ grid-template-columns: 1fr 1fr; }}
      .page-header {{ flex-direction: column; align-items: flex-start; }}
      .header-meta {{ text-align: left; }}
      .topbar-right {{ display: none; }}
    }}
    @media print {{
      .topbar, .filter-bar {{ display: none; }}
      .card {{ break-inside: avoid; }}
      body {{ background: white; color: black; }}
    }}
  </style>
</head>
<body>

<!-- ─── TOPBAR ─── -->
<nav class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">
      <div class="live-dot"></div>
      AI·RISK·ENGINE
    </div>
    <div class="topbar-sep"></div>
    <div class="topbar-meta">SECURITY VULNERABILITY REPORT</div>
  </div>
  <div class="topbar-right">
    <div class="topbar-tag">v2.0</div>
    <div class="topbar-tag">Modèle : {model_ver[:16] if len(model_ver) > 16 else model_ver}</div>
    <div class="topbar-tag">{gen_at} UTC</div>
  </div>
</nav>

<!-- ─── MAIN CONTAINER ─── -->
<div class="container">

  <!-- PAGE HEADER -->
  <div class="page-header fade-in">
    <div>
      <h1>
        <span>// Security Intelligence Report</span>
        {title.replace('AI Risk Engine — ', '')}
      </h1>
    </div>
    <div class="header-meta">
      <div>GÉNÉRÉ LE &nbsp; {gen_at} UTC</div>
      <div>FINDINGS &nbsp; {total:,} analysés</div>
      <div>MODÈLE &nbsp;&nbsp; {model_ver}</div>
    </div>
  </div>

  <!-- ALERT BANNER -->
  {alert_html}

  <!-- KPI CARDS -->
  <div class="kpi-grid">
    {kpi_cards}
  </div>

  <!-- CHARTS + DISTRIBUTION -->
  <div class="three-col">

    <!-- Distribution bars -->
    <div class="card fade-in" style="animation-delay:0.2s">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-icon">▣</span>
          Distribution par niveau de risque
        </div>
        <span class="card-count">{total} findings</span>
      </div>
      <div class="card-body">
        {dist_bars}
      </div>
    </div>

    <!-- Doughnut chart -->
    <div class="card fade-in" style="animation-delay:0.3s">
      <div class="card-header">
        <div class="card-title">
          <span class="card-title-icon">◎</span>
          Répartition visuelle
        </div>
      </div>
      <div class="card-body">
        <div class="chart-wrap" style="height:220px">
          <canvas id="doughnutChart"></canvas>
        </div>
      </div>
    </div>

  </div>

  <!-- BAR CHART by product (if data) -->
  {"" if not by_product else '''
  <div class="card fade-in" style="animation-delay:0.35s;margin-bottom:20px">
    <div class="card-header">
      <div class="card-title">
        <span class="card-title-icon">◈</span>
        Score moyen par Produit (top 8)
      </div>
    </div>
    <div class="card-body" style="height:180px">
      <canvas id="productChart"></canvas>
    </div>
  </div>'''}

  <div class="section-sep"></div>

  <!-- FINDINGS TABLE -->
  <div class="card fade-in" style="animation-delay:0.4s">
    <div class="card-header">
      <div class="card-title">
        <span class="card-title-icon">⬡</span>
        Tous les Findings — Priorisés par Score IA
      </div>
      <div class="filter-bar">
        <div class="search-wrap">
          <input
            type="text"
            id="searchInput"
            class="filter-input"
            placeholder="Rechercher..."
            oninput="filterTable()"
          >
        </div>
        <button class="filter-btn filter-btn-critical" onclick="setLevelFilter('Critical')">Critical</button>
        <button class="filter-btn filter-btn-high"     onclick="setLevelFilter('High')">High</button>
        <button class="filter-btn filter-btn-medium"   onclick="setLevelFilter('Medium')">Medium</button>
        <button class="filter-btn filter-btn-low"      onclick="setLevelFilter('Low')">Low</button>
        <button class="filter-btn filter-btn-info"     onclick="setLevelFilter('Info')">Info</button>
        <button class="filter-btn" onclick="setLevelFilter('')" id="btn-all" style="color:#c0cfe0;border-color:#3b82f6;background:rgba(59,130,246,0.08)">Tous</button>
      </div>
    </div>
    <div class="table-wrap">
      <table id="findingsTable">
        <thead>
          <tr>
            <th onclick="sortTable(0)">#<span class="sort-arrow">↕</span></th>
            <th onclick="sortTable(1)">Niveau<span class="sort-arrow">↕</span></th>
            <th onclick="sortTable(2)">Titre</th>
            <th onclick="sortTable(3)" class="sorted">Score IA<span class="sort-arrow">↓</span></th>
            <th>Produit / Engagement</th>
            <th>Fichier / Ligne</th>
            <th onclick="sortTable(6)">CVSS<span class="sort-arrow">↕</span></th>
            <th onclick="sortTable(7)">Âge<span class="sort-arrow">↕</span></th>
            <th>CVE</th>
            <th onclick="sortTable(9)">Confiance<span class="sort-arrow">↕</span></th>
            <th>ID</th>
          </tr>
        </thead>
        <tbody id="findingsBody">
          {findings_rows}
        </tbody>
      </table>
    </div>
    <div style="padding:12px 22px;border-top:1px solid var(--border);
                font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);
                display:flex;justify-content:space-between;align-items:center">
      <span id="rowCount">{len(findings_sorted)} findings affichés sur {len(findings_sorted)}</span>
      <span>ℹ Cliquez sur un en-tête pour trier &nbsp;·&nbsp; Jaune = divergence sévérité déclarée/prédite &nbsp;·&nbsp; Hover 📄 pour chemin complet</span>
    </div>
  </div>

  <!-- PRODUCTS TABLE -->
  {"" if not by_product else f'''
  <div class="card fade-in" style="animation-delay:0.5s">
    <div class="card-header">
      <div class="card-title">
        <span class="card-title-icon">◈</span>
        Risque par Produit — Top {min(len(by_product),15)}
      </div>
      <span class="card-count">{len(by_product)} produits</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Produit</th>
            <th>Engagement(s)</th>
            <th class="td-center">Total</th>
            <th class="td-center">Critical</th>
            <th class="td-center">High</th>
            <th class="td-center">Medium</th>
            <th>Score Moyen</th>
            <th>Priorité</th>
          </tr>
        </thead>
        <tbody>{product_rows}</tbody>
      </table>
    </div>
  </div>'''}

  <!-- FOOTER -->
  <div class="footer">
    AI RISK ENGINE v2.0 &nbsp;·&nbsp;
    RAPPORT GÉNÉRÉ LE {gen_at} UTC &nbsp;·&nbsp;
    MODÈLE {model_ver} &nbsp;·&nbsp;
    {total:,} FINDINGS ANALYSÉS
  </div>

</div><!-- /container -->

<!-- ─── JAVASCRIPT ─── -->
<script>
/* ── Chart data ── */
{chart_data}

/* ── Doughnut chart ── */
(function() {{
  const ctx = document.getElementById('doughnutChart');
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'doughnut',
    data: {{
      labels: DIST_LABELS,
      datasets: [{{
        data: DIST_COUNTS,
        backgroundColor: DIST_COLORS.map(c => c + '22'),
        borderColor: DIST_COLORS,
        borderWidth: 2,
        hoverBackgroundColor: DIST_COLORS.map(c => c + '44'),
        hoverOffset: 8,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {{
        legend: {{
          position: 'right',
          labels: {{
            font: {{ family: 'JetBrains Mono', size: 11 }},
            color: '#5a7394',
            padding: 14,
            usePointStyle: true,
            pointStyleWidth: 8,
          }}
        }},
        tooltip: {{
          backgroundColor: '#0d1520',
          borderColor: '#1e2d42',
          borderWidth: 1,
          titleFont: {{ family: 'JetBrains Mono', size: 11 }},
          bodyFont:  {{ family: 'JetBrains Mono', size: 11 }},
          callbacks: {{
            label: ctx => {{
              const total = ctx.dataset.data.reduce((a,b)=>a+b,0);
              const pct = (ctx.parsed / total * 100).toFixed(1);
              return `  ${{ctx.label}} : ${{ctx.parsed}} (${{pct}}%)`;
            }}
          }}
        }}
      }},
      animation: {{ animateRotate: true, duration: 900, easing: 'easeInOutQuart' }}
    }}
  }});
}})();

/* ── Bar chart by product ── */
(function() {{
  const ctx = document.getElementById('productChart');
  if (!ctx || !PROD_LABELS.length) return;
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: PROD_LABELS,
      datasets: [{{
        label: 'Score moyen',
        data: PROD_SCORES,
        backgroundColor: PROD_COLORS.map(c => c + '33'),
        borderColor: PROD_COLORS,
        borderWidth: 2,
        borderRadius: 6,
        borderSkipped: false,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        y: {{
          min: 0, max: 10,
          grid: {{ color: '#1e2d42' }},
          ticks: {{ font: {{ family: 'JetBrains Mono', size: 10 }}, color: '#5a7394' }},
        }},
        x: {{
          grid: {{ display: false }},
          ticks: {{ font: {{ family: 'JetBrains Mono', size: 10 }}, color: '#5a7394' }},
        }}
      }},
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          backgroundColor: '#0d1520',
          borderColor: '#1e2d42',
          borderWidth: 1,
          titleFont: {{ family: 'JetBrains Mono', size: 11 }},
          bodyFont:  {{ family: 'JetBrains Mono', size: 11 }},
        }}
      }},
      animation: {{ duration: 800, easing: 'easeInOutCubic' }}
    }}
  }});
}})();

/* ── Animate distribution bars on load ── */
(function() {{
  function animateBars() {{
    document.querySelectorAll('.dist-bar-fill').forEach(bar => {{
      const target = bar.dataset.target || '0';
      bar.style.width = target + '%';
    }});
  }}
  if ('IntersectionObserver' in window) {{
    const obs = new IntersectionObserver(entries => {{
      entries.forEach(e => {{ if (e.isIntersecting) {{ animateBars(); obs.disconnect(); }} }});
    }}, {{ threshold: 0.3 }});
    const el = document.querySelector('.dist-bar-wrap');
    if (el) obs.observe(el);
  }} else {{
    setTimeout(animateBars, 400);
  }}
}})();

/* ── Table filtering ── */
let currentLevel = '';

function setLevelFilter(level) {{
  currentLevel = level;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  const btnMap = {{
    'Critical': 'filter-btn-critical', 'High': 'filter-btn-high',
    'Medium': 'filter-btn-medium',     'Low':  'filter-btn-low',
    'Info':   'filter-btn-info',       '':     null,
  }};
  if (level === '') {{
    document.getElementById('btn-all').classList.add('active');
  }} else {{
    document.querySelectorAll('.filter-btn-' + level.toLowerCase()).forEach(b => b.classList.add('active'));
  }}
  filterTable();
}}

function filterTable() {{
  const search = document.getElementById('searchInput').value.toLowerCase();
  const rows   = document.querySelectorAll('#findingsBody tr');
  let visible  = 0;
  rows.forEach(row => {{
    const rowLevel = row.dataset.level || '';
    const rowText  = row.textContent.toLowerCase();
    const matchLvl = currentLevel === '' || rowLevel === currentLevel;
    const matchTxt = search === '' || rowText.includes(search);
    if (matchLvl && matchTxt) {{
      row.classList.remove('hidden');
      visible++;
    }} else {{
      row.classList.add('hidden');
    }}
  }});
  const el = document.getElementById('rowCount');
  if (el) el.textContent = visible + ' findings affichés sur ' + rows.length;
}}

/* ── Table sorting ── */
let sortCol = 3;
let sortAsc = false;

function sortTable(col) {{
  const tbody = document.getElementById('findingsBody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  if (sortCol === col) {{ sortAsc = !sortAsc; }} else {{ sortCol = col; sortAsc = false; }}

  // Update header indicators
  document.querySelectorAll('thead th').forEach((th, i) => {{
    th.classList.remove('sorted');
    const arrow = th.querySelector('.sort-arrow');
    if (arrow) arrow.textContent = '↕';
  }});
  const headers = document.querySelectorAll('thead th');
  if (headers[col]) {{
    headers[col].classList.add('sorted');
    const arrow = headers[col].querySelector('.sort-arrow');
    if (arrow) arrow.textContent = sortAsc ? '↑' : '↓';
  }}

  const LEVEL_RANK = {{ Critical: 4, High: 3, Medium: 2, Low: 1, Info: 0 }};

  rows.sort((a, b) => {{
    let va = a.cells[col]?.textContent.trim() || '';
    let vb = b.cells[col]?.textContent.trim() || '';

    if (col === 0) {{ va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }}
    else if (col === 1) {{ va = LEVEL_RANK[a.dataset.level] || 0; vb = LEVEL_RANK[b.dataset.level] || 0; }}
    else if (col === 3) {{ va = parseFloat(a.dataset.score) || 0; vb = parseFloat(b.dataset.score) || 0; }}
    else if (col === 6) {{ va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }}
    else if (col === 7) {{ va = parseInt(va) || 0; vb = parseInt(vb) || 0; }}
    else if (col === 9) {{ va = parseFloat(va.replace('%','').replace('◉','').replace('◎','').replace('○','').trim()) || 0; vb = parseFloat(vb.replace('%','').replace('◉','').replace('◎','').replace('○','').trim()) || 0; }}

    if (typeof va === 'number') return sortAsc ? va - vb : vb - va;
    return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
  }});

  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════
# 4. POINT D'ENTRÉE
# ══════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/generate_ia_report.py",
        description="AI Risk Engine — Génère le rapport HTML interactif",
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_JSON,
        help=f"Rapport JSON source (défaut: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_HTML,
        help=f"Fichier HTML de sortie (défaut: {DEFAULT_HTML})",
    )
    parser.add_argument(
        "--title", type=str, default="AI Risk Engine — Rapport de Scoring",
        help="Titre personnalisé du rapport",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Ouvre le rapport dans le navigateur après génération",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logger.info("=" * 56)
    logger.info("📊 Génération rapport HTML — AI Risk Engine v2.0")
    logger.info("=" * 56)

    data = load_report_data(args.input)
    html = generate_html(data, title=args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = args.output.stat().st_size / 1024
    logger.info(f"✅ Rapport HTML généré : {args.output}  ({size_kb:.1f} KB)")
    logger.info(f"   Ouvrez dans votre navigateur pour visualiser le dashboard.")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
        logger.info("   Navigateur ouvert automatiquement.")


if __name__ == "__main__":
    main()