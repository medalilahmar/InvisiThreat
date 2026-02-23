import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = os.getenv('MODEL_PATH', 'models/rf_model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print(f" Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    print(f" Erreur chargement modèle : {e}")
    model = None

app = FastAPI(title="Risk Scoring Engine", version="1.0.0")

class FindingInput(BaseModel):
    severity_num: int
    cvss_score: float
    age_days: int
    has_cve: int
    has_cwe: int
    tags_count: int
    is_false_positive: int
    is_active: int
    product_id: Optional[int] = None
    engagement_id: Optional[int] = None

    class Config:
        schema_extra = {
            "example": {
                "severity_num": 3,
                "cvss_score": 7.5,
                "age_days": 30,
                "has_cve": 1,
                "has_cwe": 1,
                "tags_count": 2,
                "is_false_positive": 0,
                "is_active": 1,
                "product_id": 42,
                "engagement_id": 123
            }
        }

@app.get("/")
def root():
    return {"message": "Risk Scoring Engine API is running"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: FindingInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        feature_order = [
            'severity_num', 'cvss_score', 'age_days', 'has_cve', 'has_cwe',
            'tags_count', 'is_false_positive', 'is_active'
        ]
        input_dict = data.dict()
        X = pd.DataFrame([[input_dict[f] for f in feature_order]], columns=feature_order)

        score = model.predict(X)[0]
        score = round(float(score), 2)

        if score >= 8:
            level = "Critical"
        elif score >= 6:
            level = "High"
        elif score >= 4:
            level = "Medium"
        elif score >= 2:
            level = "Low"
        else:
            level = "Info"

        return {
            "ai_score": score,
            "risk_level": level,
            "product_id": data.product_id,
            "engagement_id": data.engagement_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

