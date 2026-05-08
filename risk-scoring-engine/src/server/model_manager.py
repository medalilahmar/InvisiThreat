"""
model_manager.py — ModelManager, scoring batch au démarrage, helper SHAP.
"""
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from server.config import (
    CLASS_COLORS,
    CLASS_LABELS,
    FEATURE_COLS,
    MODEL_PATH,
    META_PATH,
)

logger = logging.getLogger("invisithreat.model_manager")


# ══════════════════════════════════════════════════════════════════════════════
# SHAP helper — extraction robuste depuis Pipeline / CalibratedClassifierCV
# ══════════════════════════════════════════════════════════════════════════════

def _extract_base_model_for_shap(model):
    """
    Extrait le modèle de base compatible TreeExplainer depuis n'importe
    quelle enveloppe sklearn (Pipeline, CalibratedClassifierCV, Stacking).
    """
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        return _extract_base_model_for_shap(steps[-1])

    if hasattr(model, "calibrated_classifiers_"):
        cc   = model.calibrated_classifiers_[0]
        base = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if base is not None:
            return _extract_base_model_for_shap(base)

    if hasattr(model, "estimators_"):
        first = model.estimators_[0]
        if isinstance(first, tuple):
            first = first[1]
        return _extract_base_model_for_shap(first)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# ModelManager
# ══════════════════════════════════════════════════════════════════════════════

class ModelManager:

    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        self.model_path         = model_path
        self.meta_path          = meta_path
        self._model             = None
        self._metadata:         Dict[str, Any]     = {}
        self._loaded_at:        Optional[datetime] = None
        self._feature_columns:  List[str]          = FEATURE_COLS.copy()
        self._prediction_cache: Dict[str, Tuple]   = {}
        self._cache_max_size    = 100
        self._cache_hits        = 0
        self._cache_misses      = 0

    # ── Status ────────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._model is not None

    def get_model(self):
        return self._model

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        if not self.model_path.exists():
            logger.warning(f"Modèle introuvable : {self.model_path}")
            return False
        try:
            self._model     = joblib.load(self.model_path)
            self._loaded_at = datetime.now(timezone.utc)

            if self.meta_path.exists():
                with open(self.meta_path) as f:
                    self._metadata = json.load(f)

            # Priorité : feature_names_in_ > metadata > FEATURE_COLS
            if hasattr(self._model, "feature_names_in_"):
                self._feature_columns = list(self._model.feature_names_in_)
                logger.info(f"Features chargées depuis le modèle : {len(self._feature_columns)} colonnes")
            elif self._metadata.get("feature_columns"):
                self._feature_columns = self._metadata["feature_columns"]
                logger.info(f"Features chargées depuis metadata : {len(self._feature_columns)} colonnes")
            else:
                logger.warning("Features chargées depuis FEATURE_COLS (fallback)")

            logger.info(
                f"Modèle chargé — version={self._metadata.get('version', '?')} "
                f"F1={self._metadata.get('metrics', {}).get('test_f1_weighted', 'N/A')}"
            )
            return True
        except Exception as e:
            logger.error(f"Erreur chargement modèle : {e}")
            self._model = None
            return False

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_batch_cached(self, X: pd.DataFrame) -> Tuple[Any, Any]:
        import hashlib
        h = hashlib.md5(X.to_csv(index=False).encode()).hexdigest()
        if h in self._prediction_cache:
            self._cache_hits += 1
            result = self._prediction_cache.pop(h)
            self._prediction_cache[h] = result
            return result
        self._cache_misses += 1
        if len(self._prediction_cache) >= self._cache_max_size:
            del self._prediction_cache[next(iter(self._prediction_cache))]
        classes = self._model.predict(X).astype(int)
        probas  = self._model.predict_proba(X)
        self._prediction_cache[h] = (classes, probas)
        return classes, probas

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns

    @property
    def n_classes(self) -> int:
        return len(self._model.classes_) if self._model and hasattr(self._model, "classes_") else 4

    @property
    def classes(self) -> List[int]:
        return (
            [int(c) for c in self._model.classes_]
            if self._model and hasattr(self._model, "classes_")
            else [0, 1, 2, 3]
        )

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def model_version(self) -> str:
        return self._metadata.get("version", self._metadata.get("timestamp", "unknown"))


# ══════════════════════════════════════════════════════════════════════════════
# Scoring batch au démarrage
# ══════════════════════════════════════════════════════════════════════════════

def score_all_findings_at_startup(
    loader,           # LocalDataLoader — évite import circulaire
    manager: ModelManager,
) -> Dict[str, Any]:
    """
    Score TOUS les findings avec le vrai modèle en un seul batch.
    Écrit data/ai_scores_cache.json et retourne le dict du cache.
    """
    if loader.df_findings is None or loader.df_findings.empty:
        logger.warning("score_all_findings_at_startup : DataFrame vide")
        return {}

    expected_cols = manager.feature_columns
    df = loader.df_findings.copy()

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        logger.warning(f"Colonnes manquantes dans CSV (remplies à 0) : {missing}")
        for col in missing:
            df[col] = 0.0

    X = df[expected_cols].fillna(0).copy()
    X.columns = pd.Index(expected_cols)

    n = len(X)
    logger.info(f"Scoring IA de {n} findings en batch…")

    try:
        predictions = manager.get_model().predict(X).astype(int)
        probas      = manager.get_model().predict_proba(X)
    except Exception as e:
        logger.error(f"Erreur scoring batch : {e}")
        raise

    model_classes = manager.classes
    dist          = Counter(int(p) for p in predictions)
    dist_labels   = {CLASS_LABELS.get(k, str(k)): v for k, v in dist.items()}
    logger.info(f"Distribution IA : {dist_labels}")

    cache: Dict[str, Any] = {}
    now          = datetime.now(timezone.utc).isoformat()
    all_findings = loader.get_all_findings()

    for i, finding in enumerate(all_findings):
        fid        = str(finding.get("id", i))
        pred       = int(predictions[i])
        proba_row  = probas[i].tolist()
        confidence = float(max(proba_row))
        label      = CLASS_LABELS.get(pred, f"class_{pred}")
        color      = CLASS_COLORS.get(label, "#888888")

        proba_dict = {
            CLASS_LABELS.get(c, str(c)): round(float(p), 4)
            for c, p in zip(model_classes, proba_row)
        }

        ctx_score = float(X["context_score"].iloc[i]) if "context_score" in X.columns else 0.0
        exp_norm  = float(X["exposure_norm"].iloc[i])  if "exposure_norm"  in X.columns else 0.0
        dly_norm  = float(X["delay_norm"].iloc[i])     if "delay_norm"     in X.columns else 0.0

        cache[fid] = {
            "ai_risk_score":    pred,
            "ai_risk_level":    label,
            "ai_risk_color":    color,
            "ai_confidence":    round(confidence, 4),
            "ai_probabilities": proba_dict,
            "context_score":    int(round(ctx_score)),
            "exposure_norm":    round(exp_norm, 4),
            "delay_norm":       round(dly_norm, 4),
            "updated_at":       now,
        }

    # Sauvegarde atomique
    cache_file = Path("data/ai_scores_cache.json")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(cache_file)

    logger.info(f"✅ Cache IA sauvegardé : {len(cache)} findings → {cache_file}")
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# SimpleShapExplainer
# ══════════════════════════════════════════════════════════════════════════════

class SimpleShapExplainer:
    """Wrapper SHAP léger autour de TreeExplainer."""

    def __init__(self, model):
        import shap
        base = _extract_base_model_for_shap(model)
        logger.info(f"SHAP — modèle de base extrait : {type(base).__name__}")
        self._exp   = shap.TreeExplainer(base)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def explain(self, X: pd.DataFrame, pred_class: int) -> Dict[str, Any]:
        try:
            sv   = self._exp.shap_values(X)
            sv_c = sv[pred_class][0] if isinstance(sv, list) else sv[0]
            top  = sorted(range(len(sv_c)), key=lambda i: abs(sv_c[i]), reverse=True)[:10]
            ev   = self._exp.expected_value
            base = float(ev[pred_class] if isinstance(ev, (list, np.ndarray)) else ev)
            return {
                "top_features": [
                    {
                        "feature":       X.columns[i],
                        "shap_value":    round(float(sv_c[i]), 4),
                        "feature_value": round(float(X.iloc[0, i]), 4),
                        "direction":     "+" if sv_c[i] > 0 else "-",
                    }
                    for i in top
                ],
                "base_value": round(base, 4),
            }
        except Exception as e:
            logger.warning(f"SHAP explain échoué : {e}")
            return {"top_features": [], "base_value": 0}