"""
routers/findings.py — Endpoints /defectdojo/products, /engagements, /findings.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from server.cache import get_scores_cache
from server.config import CLASS_LABELS, CLASS_COLORS
from server.dependencies import require_local_loader
from server.schemas import (
    EngagementResponse,
    FindingSummaryResponse,
    ProductResponse,
)

from typing import List, Optional
from fastapi import Depends
from auth.security import get_accessible_product_ids

from server.utils import (
    _compute_age_days,
    _parse_tags,
    safe_float,
    safe_int,
    safe_str,
)

logger = logging.getLogger("invisithreat.findings")

router = APIRouter(tags=["DefectDojo"])

VALID_SEVERITIES = ["critical", "high", "medium", "low", "info"]
SEVERITY_MAP_NUM = {0: "info", 1: "low", 2: "medium", 3: "high", 4: "critical"}


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _finding_to_response(f: Dict, scores_cache: Dict) -> FindingSummaryResponse:
    """Construit la réponse finding enrichie avec les scores IA du cache."""
    fid        = safe_int(f.get("id")) or 0
    score_data = scores_cache.get(str(fid), {})

    severity_raw = f.get("severity")
    cve_value    = safe_str(f.get("cve"))

    if severity_raw is None or (not isinstance(severity_raw, str) and pd.isna(severity_raw)):
        sev_num = f.get("severity_num")
        severity_raw = (
            SEVERITY_MAP_NUM.get(int(sev_num), "info")
            if sev_num and not pd.isna(sev_num)
            else "info"
        )

    if severity_raw is None or (isinstance(severity_raw, float) and np.isnan(severity_raw)):
        severity = "info"
    else:
        severity = str(severity_raw).strip().lower()
        if severity not in VALID_SEVERITIES:
            severity = "info"

    # Scores IA depuis le cache
    ai_risk_class    = safe_int(score_data.get("ai_risk_score"))
    ai_risk_level    = safe_str(score_data.get("ai_risk_level"))
    ai_risk_color    = safe_str(score_data.get("ai_risk_color"))
    ai_confidence    = safe_float(score_data.get("ai_confidence"), 0.0) or None
    ai_probabilities = score_data.get("ai_probabilities") or None
    ai_context_score = safe_int(score_data.get("context_score"))
    ai_exposure_norm = safe_float(score_data.get("exposure_norm"), 0.0) or None

    cont_score  = safe_float(score_data.get("ai_risk_score"))
    model_base  = safe_float(score_data.get("model_base_score"))
    nudge       = safe_float(score_data.get("business_nudge"))
    shap_feats  = score_data.get("shap_top_features") or None

    return FindingSummaryResponse(
        id               = fid,
        title            = safe_str(f.get("title"), "Unknown"),
        severity         = severity,
        cvss_score       = safe_float(f.get("cvss_score"), 0.0),
        tags             = _parse_tags(f.get("tags", [])),
        test_id          = safe_int(f.get("test_id")),
        engagement_id    = safe_int(f.get("engagement_id")),
        engagement_name  = safe_str(f.get("engagement_name")),
        product_id       = safe_int(f.get("product_id")),
        product_name     = safe_str(f.get("product_name")),
        created          = safe_str(f.get("created")),
        age_days         = _compute_age_days(safe_str(f.get("created"))),
        file_path        = safe_str(f.get("file_path")),
        line             = safe_int(f.get("line")),
        has_cve          = 1 if f.get("cve") else 0,
        cve              = cve_value,
        description      = safe_str(f.get("description"), ""),
        risk_class       = ai_risk_class,
        risk_level       = ai_risk_level if ai_risk_level else None,
        risk_color       = ai_risk_color if ai_risk_color else None,
        ai_confidence    = ai_confidence,
        ai_probabilities = ai_probabilities,
        context_score    = ai_context_score,
        exposure_norm    = ai_exposure_norm,
        ai_risk_score_cont = cont_score,
        model_base_score   = model_base,
        business_nudge     = nudge,
        shap_features      = shap_feats,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/defectdojo/products", response_model=List[ProductResponse])
async def get_products(
    accessible_ids: List[int] = Depends(get_accessible_product_ids),
) -> List[ProductResponse]:
    loader = require_local_loader()
    all_products = loader.get_products()

    products = []
    for p in all_products:
        pid      = p['id']
        findings = loader.get_findings_for_product(pid)
        products.append(ProductResponse(
            id             = pid,
            name           = p.get('name', ''),
            description    = p.get('description', None),
            created        = p.get('created', None),
            findings_count = len(findings),
        ))

    if accessible_ids:
        products = [p for p in products if p.id in accessible_ids]
    return products


@router.get("/defectdojo/engagements", response_model=List[EngagementResponse])
async def get_engagements(product_id: Optional[int] = None) -> List[EngagementResponse]:
    loader  = require_local_loader()
    results = []
    if product_id is not None:
        if product_id not in loader.products:
            return results
        for e in loader.get_engagements_for_product(product_id):
            results.append(EngagementResponse(
                id           = safe_int(e["id"]) or 0,
                name         = safe_str(e["name"], f"engagement-{e['id']}"),
                product      = safe_int(e["product_id"]) or 0,
                product_name = safe_str(loader.products.get(e["product_id"], {}).get("name", "Unknown")),
            ))
    else:
        for eng in loader.engagements.values():
            prod_id = safe_int(eng["product_id"])
            results.append(EngagementResponse(
                id           = safe_int(eng["id"]) or 0,
                name         = safe_str(eng["name"], f"engagement-{eng['id']}"),
                product      = prod_id or 0,
                product_name = safe_str(loader.products.get(prod_id, {}).get("name", "Unknown")),
            ))
    return results


@router.get("/defectdojo/findings", response_model=List[FindingSummaryResponse])
async def get_findings(
    engagement_id: Optional[int] = None,
    product_id:    Optional[int] = None,
    limit:         int           = 2000,
) -> List[FindingSummaryResponse]:
    loader       = require_local_loader()
    scores_cache = get_scores_cache()
    try:
        if engagement_id is not None:
            if engagement_id not in loader.engagements:
                raise HTTPException(404, detail=f"Engagement {engagement_id} introuvable")
            raw = loader.get_findings_for_engagement(engagement_id)
        elif product_id is not None:
            if product_id not in loader.products:
                raise HTTPException(404, detail=f"Produit {product_id} introuvable")
            raw = loader.get_findings_for_product(product_id)
        else:
            raw = loader.get_all_findings()
        results = [_finding_to_response(f, scores_cache) for f in raw]
        logger.info(f"[get_findings] engagement={engagement_id}, product={product_id} → {len(results)} findings")
        return results[:limit]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[get_findings] Erreur : {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/defectdojo/products/{product_id}/findings", response_model=List[FindingSummaryResponse])
async def get_product_findings(product_id: int, limit: int = 2000) -> List[FindingSummaryResponse]:
    loader = require_local_loader()
    if product_id not in loader.products:
        raise HTTPException(404, detail=f"Produit {product_id} introuvable")
    raw          = loader.get_findings_for_product(product_id)
    scores_cache = get_scores_cache()
    return [_finding_to_response(f, scores_cache) for f in raw][:limit]


@router.get("/defectdojo/engagements/{engagement_id}/findings", response_model=List[FindingSummaryResponse])
async def get_engagement_findings(engagement_id: int, limit: int = 2000) -> List[FindingSummaryResponse]:
    loader = require_local_loader()
    if engagement_id not in loader.engagements:
        raise HTTPException(404, detail=f"Engagement {engagement_id} introuvable")
    raw          = loader.get_findings_for_engagement(engagement_id)
    scores_cache = get_scores_cache()
    return [_finding_to_response(f, scores_cache) for f in raw][:limit]


@router.get("/defectdojo/findings/{finding_id}", response_model=FindingSummaryResponse)
async def get_finding_by_id(finding_id: int) -> FindingSummaryResponse:
    loader = require_local_loader()
    if finding_id not in loader.findings_by_id:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")
    scores_cache = get_scores_cache()
    return _finding_to_response(loader.findings_by_id[finding_id], scores_cache)


@router.get("/defectdojo/engagements/{engagement_id}/tests")
async def get_engagement_tests(engagement_id: int) -> List[dict]:
    loader = require_local_loader()
    if engagement_id not in loader.engagements:
        raise HTTPException(404, detail=f"Engagement {engagement_id} introuvable")

    raw_findings = loader.get_findings_for_engagement(engagement_id)

    tests_map: dict = {}
    for f in raw_findings:
        tid = f.get("test_id")
        if tid is None:
            continue
        tid = int(tid)
        if tid not in tests_map:
            tests_map[tid] = {
                "id":             tid,
                "title":          f.get("test_type_name") or f"Test #{tid}",
                "test_type_name": f.get("test_type_name") or "",
                "findings_count": 0,
            }
        tests_map[tid]["findings_count"] += 1

    return sorted(tests_map.values(), key=lambda t: t["id"])