import logging
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("invisithreat.explainer")

CLASS_LABELS: Dict[int, str] = {
    0: "Info",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical"
}


class FindingExplainer:
    def __init__(self):
        self._explainer = None
        self._imputer = None
        self._scaler = None
        self._model = None
        self._ready = False
        self._feature_names = []

    def _extract_base_model(self, model):
        if hasattr(model, "calibrated_classifiers"):
            if len(model.calibrated_classifiers) > 0:
                base = model.calibrated_classifiers[0].base_estimator
                logger.info(f"Extracted base model: {type(base).__name__}")
                return base
        return model

    def load(self, pipeline) -> bool:
        try:
            if hasattr(pipeline, "named_steps"):
                self._model = pipeline.named_steps.get("model")
                self._imputer = pipeline.named_steps.get("imputer")
                self._scaler = pipeline.named_steps.get("scaler")
            else:
                self._model = pipeline

            if self._model is None:
                logger.warning("No model found in pipeline")
                return False

            base_model = self._extract_base_model(self._model)
            self._explainer = shap.TreeExplainer(base_model)

            if hasattr(base_model, "feature_names_in_"):
                self._feature_names = list(base_model.feature_names_in_)
            elif hasattr(self._model, "feature_names_in_"):
                self._feature_names = list(self._model.feature_names_in_)

            self._ready = True
            logger.info(f"SHAP explainer loaded with {len(self._feature_names)} features")
            return True

        except Exception as e:
            logger.warning(f"SHAP explainer failed to load: {e}")
            self._ready = False
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def explain(self, X_row: pd.DataFrame, pred_class: int) -> Dict[str, Any]:
        if not self._ready:
            raise RuntimeError("Explainer not loaded")

        model_features = self.feature_names
        if not model_features:
            raise RuntimeError("No feature names available")

        for col in model_features:
            if col not in X_row.columns:
                X_row[col] = 0.0

        X_row = X_row[model_features]

        X_transformed = X_row
        if self._imputer is not None:
            X_transformed = self._imputer.transform(X_transformed)
        if self._scaler is not None:
            X_transformed = self._scaler.transform(X_transformed)

        X_transformed = pd.DataFrame(X_transformed, columns=model_features)

        shap_values = self._explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            if len(shap_values) > pred_class:
                sv_for_class = shap_values[pred_class][0]
            else:
                sv_for_class = shap_values[0][0]
            if hasattr(self._explainer, "expected_value"):
                if isinstance(self._explainer.expected_value, list):
                    base_value = float(self._explainer.expected_value[pred_class])
                else:
                    base_value = float(self._explainer.expected_value)
            else:
                base_value = 0.0
        elif shap_values.ndim == 3:
            sv_for_class = shap_values[0, :, pred_class]
            if hasattr(self._explainer, "expected_value"):
                if isinstance(self._explainer.expected_value, list):
                    base_value = float(self._explainer.expected_value[pred_class])
                else:
                    base_value = float(self._explainer.expected_value)
            else:
                base_value = 0.0
        else:
            sv_for_class = shap_values[0]
            if hasattr(self._explainer, "expected_value"):
                base_value = float(self._explainer.expected_value)
            else:
                base_value = 0.0

        contributions = {
            feat: round(float(sv_for_class[idx]), 6)
            for idx, feat in enumerate(model_features)
        }

        sorted_contribs = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        top_increasing = [
            {
                "feature": k,
                "shap_value": v,
                "feature_value": round(float(X_row[k].iloc[0]), 4)
            }
            for k, v in sorted_contribs if v > 0
        ][:5]

        top_decreasing = [
            {
                "feature": k,
                "shap_value": v,
                "feature_value": round(float(X_row[k].iloc[0]), 4)
            }
            for k, v in sorted_contribs if v < 0
        ][:5]

        prediction_score = base_value + float(np.sum(sv_for_class))

        return {
            "predicted_class": pred_class,
            "predicted_label": CLASS_LABELS.get(pred_class, str(pred_class)),
            "base_value": round(base_value, 6),
            "prediction_score": round(prediction_score, 6),
            "top_increasing": top_increasing,
            "top_decreasing": top_decreasing,
            "all_contributions": contributions,
        }


finding_explainer = FindingExplainer()