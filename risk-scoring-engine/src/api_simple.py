from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("invisithreat.api")

app = FastAPI(title="InvisiThreat AI Risk Engine")

MODEL_PATH = Path("models/pipeline_latest.pkl")
model = joblib.load(MODEL_PATH)

# Liste exacte des features attendues par le modèle
FEATURE_COLS = [
    "cvss_score", "cvss_score_norm", "age_days", "age_days_norm",
    "has_cve", "has_cwe", "tags_count", "tags_count_norm",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "product_fp_rate", "cvss_x_has_cve", "age_x_cvss",
    "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm",
    "exploit_risk", "context_score", "days_open_high",
]

class FindingInput(BaseModel):
    severity: str
    cvss_score: float = 0
    title: str = ""
    description: str = ""
    tags: List[str] = []
    days_open: int = 0
    epss_score: float = 0
    epss_percentile: float = 0
    has_cve: int = 0
    has_cwe: int = 0
    finding_id: Optional[int] = None
    engagement_id: Optional[int] = None
    product_id: Optional[int] = None


def calculate_features(finding: FindingInput) -> dict:
    """Calcule les features exactement comme dans train.py."""
    
    severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
    severity_num = severity_map.get(finding.severity.lower(), 2)
    
    tags_lower = [t.lower() for t in finding.tags]
    
    tag_urgent = 1 if any(t in tags_lower for t in ["urgent", "critical", "blocker"]) else 0
    tag_in_production = 1 if any(t in tags_lower for t in ["production", "prod", "live"]) else 0
    tag_sensitive = 1 if any(t in tags_lower for t in ["sensitive", "pii", "gdpr"]) else 0
    tag_external = 1 if any(t in tags_lower for t in ["external", "internet-facing", "public"]) else 0
    tags_count = len(finding.tags)
    
    cvss_score_norm = finding.cvss_score / 10
    age_days_norm = min(finding.days_open / 365, 1.0)
    tags_count_norm = min(tags_count / 20, 1.0)
    
    cvss_x_has_cve = finding.cvss_score * finding.has_cve
    age_x_cvss = finding.days_open * finding.cvss_score
    
    has_high_epss = 1 if finding.epss_score > 0.5 else 0
    epss_x_cvss = finding.epss_score * finding.cvss_score
    epss_score_norm = finding.epss_score
    
    context_score = (tag_in_production * 2 + tag_external * 2 + 
                     tag_sensitive * 1 + tag_urgent * 3)
    context_score = min(context_score, 10)
    
    exploit_risk = (finding.cvss_score * 0.7 + severity_num * 0.3)
    days_open_high = 1 if finding.days_open > 30 else 0
    
    return {
        "cvss_score": finding.cvss_score,
        "cvss_score_norm": round(cvss_score_norm, 4),
        "age_days": finding.days_open,
        "age_days_norm": round(age_days_norm, 4),
        "has_cve": finding.has_cve,
        "has_cwe": finding.has_cwe,
        "tags_count": tags_count,
        "tags_count_norm": round(tags_count_norm, 4),
        "tag_urgent": tag_urgent,
        "tag_in_production": tag_in_production,
        "tag_sensitive": tag_sensitive,
        "tag_external": tag_external,
        "product_fp_rate": 0.0,
        "cvss_x_has_cve": round(cvss_x_has_cve, 4),
        "age_x_cvss": round(age_x_cvss, 3),
        "epss_score": finding.epss_score,
        "epss_percentile": finding.epss_percentile,
        "has_high_epss": has_high_epss,
        "epss_x_cvss": round(epss_x_cvss, 4),
        "epss_score_norm": round(epss_score_norm, 4),
        "exploit_risk": round(exploit_risk, 4),
        "context_score": context_score,
        "days_open_high": days_open_high,
    }


@app.post("/predict")
async def predict(finding: FindingInput):
    try:
        features = calculate_features(finding)
        X = pd.DataFrame([features])[FEATURE_COLS]
        
        risk_class = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        
        risk_levels = ["Info", "Low", "Medium", "High", "Critical"]
        colors = {"Info": "#95a5a6", "Low": "#2ecc71", "Medium": "#f39c12", 
                  "High": "#e67e22", "Critical": "#e74c3c"}
        
        return {
            "finding_id": finding.finding_id,
            "engagement_id": finding.engagement_id,
            "risk_class": risk_class,
            "risk_level": risk_levels[risk_class],
            "risk_color": colors.get(risk_levels[risk_class], "#888888"),
            "confidence": float(max(proba)),
            "context_score": features["context_score"],
            "predicted_at": pd.Timestamp.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}


@app.get("/model/features")
async def model_features():
    return {"feature_columns": FEATURE_COLS, "n_features": len(FEATURE_COLS)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_simple:app", host="0.0.0.0", port=8081, reload=True)