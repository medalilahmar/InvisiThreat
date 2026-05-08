"""
routers/jira.py — Endpoints Jira : create-jira-issue, get-jira-issue, health.

FIX : local_data_loader et model_manager ne sont PLUS importés comme valeurs
      figées au chargement du module. On utilise :
        - Depends(require_local_loader) → injecte le loader vivant à chaque requête
        - get_model_manager()           → lit l'instance courante du module dependencies
"""
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from jira_service import jira_service
from server.cache import get_scores_cache
from server.config import CLASS_LABELS, CLASS_COLORS
from server.dependencies import get_model_manager, require_local_loader
from server.data_loader import LocalDataLoader
from server.routers.llm import explain_with_llm, recommend_with_llm, _LLM_AVAILABLE
from server.schemas import (
    JiraHealthResponse,
    JiraIssueResponse,
    LLMRequest,
    FindingInput,
)
from server.utils import _compute_age_days, safe_float, safe_int, safe_str

logger = logging.getLogger("invisithreat.jira")

router = APIRouter(tags=["Jira"])


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/defectdojo/findings/{finding_id}/create-jira-issue",
    response_model=JiraIssueResponse,
    summary="Crée une issue Jira pour un finding avec score IA + LLM",
)
async def create_jira_issue_for_finding(
    finding_id: int,
    loader: LocalDataLoader = Depends(require_local_loader),   # ← injecté, jamais None
) -> JiraIssueResponse:

    # Vérification doublon Jira
    existing = jira_service._find_existing_issue(finding_id)
    if existing:
        raise HTTPException(409, detail=f"Un ticket Jira existe déjà : {existing.key}")

    # loader est garanti non-None et is_ready grâce à require_local_loader
    finding = loader.findings_by_id.get(finding_id)
    if not finding:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")

    # model_manager lu via getter — toujours l'instance post-lifespan
    mm = get_model_manager()

    # Score IA depuis le cache
    scores_cache = get_scores_cache()
    cached_score = scores_cache.get(str(finding_id), {})

    try:
        if cached_score:
            risk_class  = cached_score.get("ai_risk_score", 0)
            risk_level  = cached_score.get("ai_risk_level", "Low")
            confidence  = cached_score.get("ai_confidence", 0.0)
            proba_dict  = cached_score.get("ai_probabilities", {})
            classes     = mm.classes
            probas_vals = [proba_dict.get(CLASS_LABELS.get(c, str(c)), 0.0) for c in classes]
            risk_score  = round(sum(c * p for c, p in zip(classes, probas_vals)) * 100 / max(classes), 2)
        else:
            # Fallback : recalcul via FindingInput
            inp = FindingInput(
                severity        = finding.get("severity", "info"),
                cvss_score      = safe_float(finding.get("cvss_score"), 0.0),
                title           = finding.get("title", ""),
                description     = finding.get("description", ""),
                tags            = finding.get("tags", []),
                days_open       = _compute_age_days(finding.get("created")) or 0,
                epss_score      = safe_float(finding.get("epss_score"), 0.0),
                epss_percentile = safe_float(finding.get("epss_percentile"), 0.0),
                has_cve         = 1 if finding.get("cve") else 0,
                has_cwe         = 1 if finding.get("cwe") else 0,
            )
            feat     = inp.to_features()
            expected = mm.feature_columns
            X        = pd.DataFrame([{c: feat.get(c, 0.0) for c in expected}])
            X.columns = pd.Index(expected)
            classes_arr, probas_arr = mm.predict_batch_cached(X)
            risk_class  = int(classes_arr[0])
            probas_vals = probas_arr[0]
            classes     = mm.classes
            confidence  = round(float(max(probas_vals)), 4)
            risk_score  = round(sum(c * p for c, p in zip(classes, probas_vals)) * 100 / max(classes), 2)
            proba_dict  = {
                CLASS_LABELS.get(c, str(c)): round(float(p), 4)
                for c, p in zip(classes, probas_vals)
            }
            risk_level = CLASS_LABELS.get(risk_class, "Unknown")

        ai_result = {
            "risk_level":    risk_level,
            "risk_score":    risk_score,
            "confidence":    confidence,
            "probabilities": proba_dict,
        }
    except Exception as e:
        logger.warning(f"Prédiction IA échouée pour finding {finding_id} : {e}")
        ai_result = {
            "risk_level":    finding.get("severity", "medium").capitalize(),
            "risk_score":    50,
            "confidence":    0.0,
            "probabilities": {},
        }

    # LLM (optionnel)
    explanation = recommendation = None
    if _LLM_AVAILABLE:
        try:
            llm_req = LLMRequest(
                finding_id  = finding_id,
                title       = finding.get("title", ""),
                severity    = ai_result.get("risk_level", "medium"),
                cvss_score  = finding.get("cvss_score", 0.0),
                description = finding.get("description", ""),
                cve         = finding.get("cve", ""),
                tags        = finding.get("tags", []),
                risk_level  = ai_result.get("risk_level"),
            )
            explanation    = await explain_with_llm(llm_req)
            recommendation = await recommend_with_llm(llm_req)
        except Exception as e:
            logger.warning(f"LLM indisponible pour finding {finding_id} : {e}")

    # Création Jira
    try:
        result = jira_service.create_security_issue(
            finding            = finding,
            ai_prediction      = ai_result,
            llm_explanation    = explanation.dict() if explanation else None,
            llm_recommendation = recommendation.dict() if recommendation else None,
        )
    except Exception as e:
        logger.error(f"Erreur création issue Jira : {e}")
        raise HTTPException(502, detail=f"Erreur Jira : {str(e)}")

    message = (
        "Issue déjà existante"
        if result.get("already_exists")
        else f"Issue créée avec succès : {result['jira_key']}"
    )
    return JiraIssueResponse(
        key            = result["jira_key"],
        id             = result.get("jira_id", ""),
        self           = result.get("jira_self", ""),
        url            = result.get("jira_url"),
        already_exists = result.get("already_exists", False),
        message        = message,
    )


@router.get("/defectdojo/findings/{finding_id}/jira-issue")
async def get_jira_issue_for_finding(
    finding_id: int,
    loader: LocalDataLoader = Depends(require_local_loader),  # ← idem
):
    if finding_id not in loader.findings_by_id:
        raise HTTPException(404, detail=f"Finding {finding_id} introuvable")
    try:
        issue = jira_service._find_existing_issue(finding_id)
        if issue:
            return {
                "exists":   True,
                "jira_key": issue.key,
                "jira_url": f"{jira_service.server}/browse/{issue.key}",
                "created":  issue.fields.created if hasattr(issue, "fields") else None,
            }
        return {"exists": False, "jira_key": None, "jira_url": None}
    except Exception as e:
        logger.error(f"Erreur vérification Jira pour finding {finding_id}: {e}")
        return {"exists": False, "jira_key": None, "jira_url": None}


@router.get("/jira/health", response_model=JiraHealthResponse)
async def jira_health() -> JiraHealthResponse:
    health = jira_service.health_check()
    return JiraHealthResponse(**health)