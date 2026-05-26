"""
analytics_router.py — Endpoints /analytics/stats et /analytics/products/{id}/stats
Calculs côté backend avec cache + RBAC automatique.
"""
import logging
from collections import defaultdict
from datetime import datetime, timedelta , timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func


from auth.security import get_accessible_product_ids, get_current_user
from database.connection import get_db
from database.models import User , FindingScoreHistory
from server.cache import get_scores_cache
from server.dependencies import require_local_loader
from server.utils import safe_float, safe_int, safe_str
from collections import defaultdict




logger = logging.getLogger("invisithreat.analytics")

router = APIRouter(prefix="/analytics", tags=["📊 Analytics"])

# ── Cache mémoire simple ──────────────────────────────────────────────────────
_analytics_cache: Dict[str, Any] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _cache_get(key: str) -> Optional[Any]:
    entry = _analytics_cache.get(key)
    if not entry:
        return None
    if (datetime.utcnow() - entry["ts"]).total_seconds() > CACHE_TTL_SECONDS:
        del _analytics_cache[key]
        return None
    return entry["data"]


def _cache_set(key: str, data: Any) -> None:
    _analytics_cache[key] = {"data": data, "ts": datetime.utcnow()}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_risk_score(critical: int, high: int, medium: int, low: int, total: int) -> float:
    """Score de risque de 0 à 100."""
    if total == 0:
        return 0.0
    weighted = (critical * 10 + high * 5 + medium * 2 + low * 1)
    max_possible = total * 10
    return round((weighted / max_possible) * 100, 1)


def _compute_mttr(findings: List[Dict]) -> Dict[str, float]:
    """Mean Time To Resolve par sévérité (en jours)."""
    age_by_sev = defaultdict(list)
    for f in findings:
        sev = safe_str(f.get("severity"), "info").lower()
        age = safe_float(f.get("age_days"), 0.0)
        if age and age > 0:
            age_by_sev[sev].append(age)
    return {
        sev: round(sum(ages) / len(ages), 1)
        for sev, ages in age_by_sev.items()
        if ages
    }


def _build_product_stats(product_id: int, product_name: str, findings: List[Dict], scores_cache: Dict) -> Dict:
    """Construit les stats complètes d'un produit."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    cvss_scores = []
    age_list = []
    vuln_types = defaultdict(int)
    timeline_by_month = defaultdict(lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})

    for f in findings:
        sev = safe_str(f.get("severity"), "info").lower()
        if sev in counts:
            counts[sev] += 1
        else:
            counts["info"] += 1

        cvss = safe_float(f.get("cvss_score"), 0.0)
        if cvss and cvss > 0:
            cvss_scores.append(cvss)

        age = safe_float(f.get("age_days"), 0.0)
        if age:
            age_list.append(age)

        # Type de vulnérabilité depuis le titre
        title = safe_str(f.get("title"), "").lower()
        for vtype in ["sql injection", "xss", "csrf", "ssrf", "idor", "rce", "lfi", "xxe", "open redirect", "insecure"]:
            if vtype in title:
                vuln_types[vtype.upper()] += 1
                break
        else:
            vuln_types["Autre"] += 1

        # Timeline par mois
        created = safe_str(f.get("created"), "")
        if created:
            try:
                dt = datetime.fromisoformat(created[:10])
                month_key = dt.strftime("%Y-%m")
                sev_key = sev if sev in ["critical", "high", "medium", "low"] else "low"
                timeline_by_month[month_key][sev_key] += 1
            except Exception:
                pass

    total = sum(counts.values())
    risk_score = _compute_risk_score(counts["critical"], counts["high"], counts["medium"], counts["low"], total)

    # Top 8 vuln types
    top_vuln = sorted(vuln_types.items(), key=lambda x: x[1], reverse=True)[:8]

    # Timeline triée (6 derniers mois)
    sorted_months = sorted(timeline_by_month.items())[-6:]
    timeline = [{"month": k, **v} for k, v in sorted_months]

    # Radar profil de risque (6 axes)
    radar = [
        {"axis": "Injection",      "value": min(counts["critical"] * 10, 100)},
        {"axis": "Auth",           "value": min(counts["high"] * 5, 100)},
        {"axis": "Exposition",     "value": min(total * 2, 100)},
        {"axis": "Config",         "value": min(counts["medium"] * 3, 100)},
        {"axis": "Crypto",         "value": min(len([f for f in findings if "crypt" in safe_str(f.get("title"), "").lower()]) * 15, 100)},
        {"axis": "Dépendances",    "value": min(len([f for f in findings if "depend" in safe_str(f.get("title"), "").lower()]) * 10, 100)},
    ]

    return {
        "id":             product_id,
        "name":           product_name,
        "total_findings": total,
        "critical":       counts["critical"],
        "high":           counts["high"],
        "medium":         counts["medium"],
        "low":            counts["low"],
        "info":           counts["info"],
        "avg_cvss":       round(sum(cvss_scores) / len(cvss_scores), 2) if cvss_scores else 0.0,
        "avg_age_days":   round(sum(age_list) / len(age_list), 1) if age_list else 0.0,
        "risk_score":     risk_score,
        "mttr":           _compute_mttr(findings),
        "top_vuln_types": [{"name": k, "value": v} for k, v in top_vuln],
        "timeline":       timeline,
        "radar":          radar,
        "funnel": {
            "detected":    total,
            "high_critical": counts["critical"] + counts["high"],
            "medium":      counts["medium"],
            "low_info":    counts["low"] + counts["info"],
        },
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_analytics_stats(
    accessible_ids: List[int] = Depends(get_accessible_product_ids),
    current_user: User = Depends(get_current_user),
) -> Dict:
    """
    Stats globales analytics — filtrées par RBAC.
    Admin/Analyst → tous les produits.
    Manager/Developer → seulement leurs produits assignés.
    """
    cache_key = f"analytics_stats_{current_user.id}"
    cached = _cache_get(cache_key)
    if cached:
        logger.info(f"[analytics/stats] Cache hit pour user {current_user.id}")
        return cached

    loader = require_local_loader()
    scores_cache = get_scores_cache()

    all_products = loader.get_products()

    # Filtrage RBAC
    if accessible_ids:
        products = [p for p in all_products if p["id"] in accessible_ids]
    else:
        products = all_products

    # Calcul global
    total_findings = 0
    total_critical = 0
    total_high = 0
    total_medium = 0
    total_low = 0
    total_info = 0
    all_cvss = []
    all_ages = []
    by_product = []
    heatmap = []
    global_vuln_types = defaultdict(int)
    global_timeline = defaultdict(lambda: {"critical": 0, "high": 0, "medium": 0, "low": 0})

    for product in products:
        pid = product["id"]
        pname = product["name"]
        findings = loader.get_findings_for_product(pid)

        stats = _build_product_stats(pid, pname, findings, scores_cache)

        total_findings += stats["total_findings"]
        total_critical += stats["critical"]
        total_high     += stats["high"]
        total_medium   += stats["medium"]
        total_low      += stats["low"]
        total_info     += stats["info"]

        if stats["avg_cvss"] > 0:
            all_cvss.append(stats["avg_cvss"])
        if stats["avg_age_days"] > 0:
            all_ages.append(stats["avg_age_days"])

        # Pour le bar chart top produits
        by_product.append({
            "id":            pid,
            "name":          pname,
            "totalFindings": stats["total_findings"],
            "critical":      stats["critical"],
            "high":          stats["high"],
            "medium":        stats["medium"],
            "low":           stats["low"],
            "risk_score":    stats["risk_score"],
            "avg_cvss":      stats["avg_cvss"],
        })

        # Heatmap
        heatmap.append({
            "product": pname[:20],
            "critical": stats["critical"],
            "high":     stats["high"],
            "medium":   stats["medium"],
            "low":      stats["low"],
            "info":     stats["info"],
        })

        # Vuln types globaux
        for vt in stats["top_vuln_types"]:
            global_vuln_types[vt["name"]] += vt["value"]

        # Timeline globale
        for t in stats["timeline"]:
            month = t["month"]
            for sev in ["critical", "high", "medium", "low"]:
                global_timeline[month][sev] += t.get(sev, 0)

    # Tri
    by_product.sort(key=lambda x: x["risk_score"], reverse=True)
    top_products = by_product[:10]
    top_vuln_global = sorted(global_vuln_types.items(), key=lambda x: x[1], reverse=True)[:10]
    timeline_sorted = [{"month": k, **v} for k, v in sorted(global_timeline.items())][-12:]

    global_risk_score = _compute_risk_score(total_critical, total_high, total_medium, total_low, total_findings)

    # Funnel global
    funnel = [
        {"name": "Détectés",       "value": total_findings},
        {"name": "Critiques+High", "value": total_critical + total_high},
        {"name": "Medium",         "value": total_medium},
        {"name": "Low+Info",       "value": total_low + total_info},
    ]

    # Distribution sévérité
    severity_distribution = [
        {"name": "Critical", "value": total_critical, "color": "#ff4757"},
        {"name": "High",     "value": total_high,     "color": "#ff6b35"},
        {"name": "Medium",   "value": total_medium,   "color": "#ffd32a"},
        {"name": "Low",      "value": total_low,      "color": "#2ed573"},
        {"name": "Info",     "value": total_info,     "color": "#95a5a6"},
    ]

    # MTTR global approximatif
    mttr_global = {
        "critical": round(sum(all_ages[:len(all_ages)//4 or 1]) / max(len(all_ages)//4, 1), 1),
        "high":     round(sum(all_ages) / max(len(all_ages), 1) * 0.8, 1),
        "medium":   round(sum(all_ages) / max(len(all_ages), 1) * 1.2, 1),
        "low":      round(sum(all_ages) / max(len(all_ages), 1) * 2.0, 1),
    }

    result = {
        "summary": {
            "total_findings":  total_findings,
            "total_products":  len(products),
            "total_critical":  total_critical,
            "total_high":      total_high,
            "total_medium":    total_medium,
            "total_low":       total_low,
            "total_info":      total_info,
            "avg_cvss":        round(sum(all_cvss) / len(all_cvss), 2) if all_cvss else 0.0,
            "avg_age_days":    round(sum(all_ages) / len(all_ages), 1) if all_ages else 0.0,
            "global_risk_score": global_risk_score,
            "urgent_count":    total_critical + total_high,
            "urgent_ratio":    round((total_critical + total_high) / max(total_findings, 1) * 100, 1),
        },
        "severity_distribution": severity_distribution,
        "by_product":            by_product,
        "top_products":          top_products,
        "heatmap":               heatmap,
        "top_vuln_types":        [{"name": k, "value": v} for k, v in top_vuln_global],
        "timeline":              timeline_sorted,
        "funnel":                funnel,
        "mttr":                  mttr_global,
        "role":                  current_user.role,
        "filtered":              bool(accessible_ids),
    }

    _cache_set(cache_key, result)
    logger.info(f"[analytics/stats] Calculé pour user {current_user.id} ({current_user.role}) — {total_findings} findings, {len(products)} produits")
    return result


@router.get("/products/{product_id}/stats")
def get_product_analytics(
    product_id: int,
    accessible_ids: List[int] = Depends(get_accessible_product_ids),
    current_user: User = Depends(get_current_user),
) -> Dict:
    """Stats détaillées d'un produit — vérifie l'accès RBAC."""
    # Vérification accès
    if accessible_ids and product_id not in accessible_ids:
        raise HTTPException(403, detail="Accès non autorisé à ce produit")

    cache_key = f"analytics_product_{product_id}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    loader = require_local_loader()
    scores_cache = get_scores_cache()

    if product_id not in loader.products:
        raise HTTPException(404, detail=f"Produit {product_id} introuvable")

    product_name = loader.products[product_id]["name"]
    findings = loader.get_findings_for_product(product_id)
    result = _build_product_stats(product_id, product_name, findings, scores_cache)

    _cache_set(cache_key, result)
    logger.info(f"[analytics/products/{product_id}] {len(findings)} findings calculés")
    return result



@router.delete("/cache")
def clear_analytics_cache(current_user: User = Depends(get_current_user)) -> Dict:
    """Vide le cache analytics — admin seulement."""
    if current_user.role != "admin":
        raise HTTPException(403, detail="Réservé aux administrateurs")
    count = len(_analytics_cache)
    _analytics_cache.clear()
    return {"cleared": count, "message": "Cache analytics vidé"}





@router.get("/heatmap")
def get_code_heatmap(
    engagement_id: Optional[int] = Query(default=None),
    product_name:  Optional[str] = Query(default=None),
    db:            Session       = Depends(get_db),
    _:             object        = Depends(get_current_user)
):
    """
    Retourne l'arborescence des fichiers colorée par nombre de findings.
    Utilisé pour la heatmap du code dans le dashboard.
    """

    # Récupère les findings depuis le cache local (LocalDataLoader)
    # ou depuis DefectDojo directement
    try:
        from server.dependencies import get_local_data_loader
        loader   = get_local_data_loader()
        findings = loader.get_all_findings()
    except Exception:
        return {"tree": [], "stats": {}}

    # Filtre par engagement ou produit si demandé
    if engagement_id:
        findings = [
            f for f in findings
            if f.get("engagement_id") == engagement_id
        ]
    if product_name:
        findings = [
            f for f in findings
            if f.get("product_name") == product_name
        ]

    # Groupe par file_path
    file_stats: dict = defaultdict(lambda: {
        "findings":  [],
        "critical":  0,
        "high":      0,
        "medium":    0,
        "low":       0,
        "total":     0,
        "max_score": 0.0,
    })

    for f in findings:
        # file_path peut être NaN (float) depuis pandas/CSV
        raw_path = f.get("file_path")
        if raw_path is None or (isinstance(raw_path, float)):
            path = "unknown/"
        else:
            path = str(raw_path).strip()
            if not path or path in ("", "None", "null", "nan"):
                path = "unknown/"

        # severity peut aussi être NaN
        raw_sev = f.get("severity")
        sev = str(raw_sev).lower() if raw_sev and not isinstance(raw_sev, float) else ""

        # score peut être NaN ou absent
        try:
            score = float(f.get("ai_risk_score_cont") or 0)
            if score != score:  # détecte NaN
                score = 0.0
        except (ValueError, TypeError):
            score = 0.0

        file_stats[path]["total"]    += 1
        file_stats[path]["max_score"] = max(
            file_stats[path]["max_score"], score
        )
        file_stats[path]["findings"].append(f.get("id"))

        if sev == "critical": file_stats[path]["critical"] += 1
        elif sev == "high":   file_stats[path]["high"]     += 1
        elif sev == "medium": file_stats[path]["medium"]   += 1
        elif sev == "low":    file_stats[path]["low"]      += 1

    # Construit l'arbre
    def build_tree(file_stats: dict) -> list:
        tree: dict = {}

        for path, stats in file_stats.items():
            parts = path.replace("\\", "/").split("/")
            node  = tree

            for i, part in enumerate(parts):
                if part not in node:
                    node[part] = {
                        "_children": {},
                        "_stats": {
                            "findings": [],
                            "critical": 0, "high": 0,
                            "medium":   0, "low":  0,
                            "total":    0, "max_score": 0.0,
                            "is_file":  False,
                            "path":     "/".join(parts[:i+1])
                        }
                    }

                # Si c'est le dernier élément → c'est un fichier
                if i == len(parts) - 1:
                    node[part]["_stats"].update({
                        **stats,
                        "is_file": True,
                        "path":    path
                    })
                else:
                    # Accumule les stats dans les dossiers parents
                    s = node[part]["_stats"]
                    s["total"]    += stats["total"]
                    s["critical"] += stats["critical"]
                    s["high"]     += stats["high"]
                    s["medium"]   += stats["medium"]
                    s["low"]      += stats["low"]
                    s["max_score"] = max(s["max_score"], stats["max_score"])

                node = node[part]["_children"]

        # Convertit en liste récursive
        def to_list(node: dict, depth: int = 0) -> list:
            result = []
            for name, data in sorted(node.items()):
                children = to_list(data["_children"], depth + 1)
                result.append({
                    "name":     name,
                    "depth":    depth,
                    "stats":    data["_stats"],
                    "children": children
                })
            return result

        return to_list(tree)

    tree = build_tree(file_stats)

    # Stats globales
    total_files    = len(file_stats)
    total_findings = sum(s["total"]    for s in file_stats.values())
    total_critical = sum(s["critical"] for s in file_stats.values())

    return {
        "tree": tree,
        "stats": {
            "total_files":    total_files,
            "total_findings": total_findings,
            "total_critical": total_critical,
            "top_risky_files": sorted(
                [
                    {
                        "path":     path,
                        "total":    s["total"],
                        "critical": s["critical"],
                        "score":    round(s["max_score"], 2)
                    }
                    for path, s in file_stats.items()
                ],
                key=lambda x: (x["critical"], x["total"]),
                reverse=True
            )[:10]
        }
    }



@router.get("/history/{finding_id}")
def get_finding_history(
    finding_id: int,
    db:         Session = Depends(get_db),
    _:          object  = Depends(get_current_user),
):
    records = (
        db.query(FindingScoreHistory)
        .filter_by(finding_id=finding_id)
        .order_by(FindingScoreHistory.scored_at)
        .all()
    )
    return {
        "finding_id": finding_id,
        "count":      len(records),
        "history": [
            {
                "score":      r.ai_risk_score,
                "level":      r.risk_level,
                "confidence": r.confidence,
                "date":       r.scored_at.isoformat(),
            }
            for r in records
        ],
    }


@router.get("/trends")
def get_global_trends(
    days:         int            = Query(default=30, ge=7, le=365),
    product_name: Optional[str]  = Query(default=None),
    db:           Session        = Depends(get_db),
    _:            object         = Depends(get_current_user),
):
    since   = datetime.now(timezone.utc) - timedelta(days=days)
    query   = db.query(FindingScoreHistory).filter(FindingScoreHistory.scored_at >= since)

    if product_name:
        query = query.filter_by(product_name=product_name)

    records = query.order_by(FindingScoreHistory.scored_at).all()

    weeks: dict = {}
    for r in records:
        week_key = r.scored_at.strftime("%Y-W%W")
        if week_key not in weeks:
            weeks[week_key] = {
                "week": week_key, "scores": [],
                "critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0,
            }
        weeks[week_key]["scores"].append(r.ai_risk_score)
        weeks[week_key]["total"] += 1
        level = (r.risk_level or "").lower()
        if level in weeks[week_key]:
            weeks[week_key][level] += 1

    trends = []
    for week_data in sorted(weeks.values(), key=lambda x: x["week"]):
        scores = week_data.pop("scores")
        week_data["avg_score"] = round(sum(scores) / len(scores), 2) if scores else 0
        trends.append(week_data)

    return {"period_days": days, "product": product_name or "all", "trends": trends}


@router.get("/summary")
def get_summary(
    db: Session = Depends(get_db),
    _:  object  = Depends(get_current_user),
):
    result = db.query(
        func.avg(FindingScoreHistory.ai_risk_score).label("avg_score"),
        func.count(FindingScoreHistory.id).label("total"),
    ).first()

    since_week   = datetime.now(timezone.utc) - timedelta(days=7)
    since_2weeks = datetime.now(timezone.utc) - timedelta(days=14)

    recent = db.query(FindingScoreHistory).filter(FindingScoreHistory.scored_at >= since_week).all()
    distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in recent:
        level = (r.risk_level or "").lower()
        if level in distribution:
            distribution[level] += 1

    prev_week_avg = db.query(func.avg(FindingScoreHistory.ai_risk_score)).filter(
        FindingScoreHistory.scored_at >= since_2weeks,
        FindingScoreHistory.scored_at <  since_week,
    ).scalar()

    curr_week_avg = db.query(func.avg(FindingScoreHistory.ai_risk_score)).filter(
        FindingScoreHistory.scored_at >= since_week,
    ).scalar()

    trend = 0.0
    if prev_week_avg and curr_week_avg:
        trend = round(float(curr_week_avg) - float(prev_week_avg), 2)

    return {
        "global_avg_score":   round(float(result.avg_score or 0), 2),
        "total_scored":       result.total or 0,
        "last_7days":         distribution,
        "trend_vs_last_week": trend,
        "trend_direction":    "up" if trend > 0 else "down" if trend < 0 else "stable",
    }