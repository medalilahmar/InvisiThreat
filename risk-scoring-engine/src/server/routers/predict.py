"""
routers/predict.py — Endpoints /predict et /predict/batch.

FIX : model_manager n'est PLUS importé comme valeur figée.
      On utilise get_model_manager() à chaque requête.
"""
import logging
import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from server.config import CLASS_LABELS, CLASS_COLORS
from server.dependencies import get_model_manager          # ← getter dynamique
from server.schemas import (
    BatchInput,
    BatchPredictionResponse,
    FindingInput,
    PredictionResponse,
)

logger = logging.getLogger("invisithreat.predict")

router = APIRouter(tags=["Prediction"])


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_X(features: dict, expected_cols) -> pd.DataFrame:
    X = pd.DataFrame([features])
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[expected_cols]
    X.columns = pd.Index(expected_cols)
    return X


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictionResponse)
async def predict(finding: FindingInput, request: Request) -> PredictionResponse:
    mm = get_model_manager()                           # ← lu à chaque requête
    if not mm.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    rid = getattr(request.state, "request_id", "unknown")
    try:
        features      = finding.to_features()
        expected_cols = mm.feature_columns
        X             = _build_X(features, expected_cols)

        risk_class = int(mm.get_model().predict(X)[0])
        probas     = mm.get_model().predict_proba(X)[0]
        classes    = mm.classes
        confidence = round(float(max(probas)), 4)
        risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / max(classes), 2)
        proba_dict = {
            CLASS_LABELS.get(c, str(c)): round(float(p), 4)
            for c, p in zip(classes, probas)
        }

        logger.info(
            f"[{rid}] Predict: {finding.title[:50]!r} → "
            f"{CLASS_LABELS.get(risk_class)} (conf={confidence:.2f})"
        )

        return PredictionResponse(
            request_id    = rid,
            finding_id    = finding.finding_id,
            engagement_id = finding.engagement_id,
            product_id    = finding.product_id,
            risk_class    = risk_class,
            risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
            risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, ""), "#888"),
            risk_score    = risk_score,
            confidence    = confidence,
            context_score = int(features["context_score"]),
            cvss_score    = finding.cvss_score,
            probabilities = proba_dict,
            features_used = {c: round(float(X[c].iloc[0]), 4) for c in expected_cols[:10]},
            predicted_at  = datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{rid}] Erreur prédiction : {e}")
        raise HTTPException(500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchInput, request: Request) -> BatchPredictionResponse:
    mm = get_model_manager()                           # ← lu à chaque requête
    if not mm.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    rid   = getattr(request.state, "request_id", "unknown")
    start = time.perf_counter()
    try:
        records       = [f.to_features() for f in batch.findings]
        expected_cols = mm.feature_columns
        X             = pd.DataFrame(records)
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[expected_cols]
        X.columns = pd.Index(expected_cols)

        classes_arr, probas_arr = mm.predict_batch_cached(X)
        classes = mm.classes
        results = []
        for idx, finding in enumerate(batch.findings):
            risk_class = int(classes_arr[idx])
            probas     = probas_arr[idx]
            confidence = round(float(max(probas)), 4)
            risk_score = round(sum(c * p for c, p in zip(classes, probas)) * 100 / max(classes), 2)
            proba_dict = {
                CLASS_LABELS.get(c, str(c)): round(float(p), 4)
                for c, p in zip(classes, probas)
            }
            results.append(PredictionResponse(
                request_id    = rid,
                finding_id    = finding.finding_id,
                engagement_id = finding.engagement_id,
                product_id    = finding.product_id,
                risk_class    = risk_class,
                risk_level    = CLASS_LABELS.get(risk_class, "Unknown"),
                risk_color    = CLASS_COLORS.get(CLASS_LABELS.get(risk_class, ""), "#888"),
                risk_score    = risk_score,
                confidence    = confidence,
                context_score = int(records[idx]["context_score"]),
                probabilities = proba_dict,
                features_used = {},
                predicted_at  = datetime.now(timezone.utc).isoformat(),
            ))

        summary = {
            lv: sum(1 for r in results if r.risk_level.lower() == lv)
            for lv in ["critical", "high", "medium", "low"]
        }
        elapsed = time.perf_counter() - start
        logger.info(f"[{rid}] Batch: {len(results)} findings en {elapsed:.2f}s")

        return BatchPredictionResponse(
            request_id   = rid,
            total        = len(batch.findings),
            success      = len(results),
            errors_count = 0,
            results      = results,
            errors       = [],
            summary      = summary,
            processed_at = datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{rid}] Erreur batch : {e}")
        raise HTTPException(500, detail=str(e))