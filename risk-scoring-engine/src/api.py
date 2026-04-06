import json
import logging
import os
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("invisithreat.api")

API_VERSION = "2.1.0"
APP_START = datetime.now(timezone.utc)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pipeline_latest.pkl"))
META_PATH = Path(os.getenv("META_PATH", "models/pipeline_latest_meta.json"))

DEFECTDOJO_URL = os.getenv("DEFECTDOJO_URL", "http://192.168.11.170:8080")
DEFECTDOJO_API_KEY = os.getenv("DEFECTDOJO_API_KEY", "a8506b7874b044ed31f8d6b847ca4d6b15bdb868")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

EPSS_THRESHOLD = 0.5

RISK_LEVELS: Dict[int, str] = {
    0: "info",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical",
}

CLASS_LABELS: Dict[int, str] = {
    0: "Info",
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Critical",
}

CLASS_COLORS: Dict[str, str] = {
    "Info": "#95a5a6",
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}

FEATURE_COLS: List[str] = [
    "cvss_score", "severity_numeric", "days_open", "duplicate_count",
    "context_score", "exploit_risk", "age_factor", "tags_count",
    "tag_production", "tag_external", "tag_sensitive", "tag_urgent",
    "tag_sca", "tag_api", "epss_score", "epss_percentile",
    "age_x_cvss", "cvss_x_severity", "cvss_x_epss",
    "severity_x_urgent", "days_open_high", "has_cve", "has_cwe",
]


class FindingInput(BaseModel):
    severity: str = Field(..., description="critical, high, medium, low, info")
    cvss_score: float = Field(0.0, ge=0.0, le=10.0)
    title: str = ""
    description: str = ""
    file_path: str = ""
    tags: List[str] = []
    finding_id: Optional[int] = None
    engagement_id: Optional[int] = None
    product_id: Optional[int] = None
    days_open: int = 0
    duplicate_count: int = 0
    epss_score: float = 0.0
    epss_percentile: float = 0.0
    has_cve: int = 0
    has_cwe: int = 0

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = ["critical", "high", "medium", "low", "info", "informational"]
        v_lower = v.lower().strip()
        if v_lower not in valid:
            raise ValueError(f"Severity must be one of {valid}")
        return v_lower

    @field_validator("cvss_score")
    @classmethod
    def round_cvss(cls, v: float) -> float:
        return round(v, 1)

    def to_features(self) -> Dict[str, Any]:
        tags_lower = [t.lower() for t in self.tags]
        
        severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        severity_numeric = severity_map.get(self.severity.lower(), 0)
        
        tag_weights = {
            "production": 2, "prod": 2, "prd": 2, "live": 2, "main": 1,
            "external": 2, "internet-facing": 2, "public": 2, "exposed": 2,
            "sensitive": 1, "pii": 1, "gdpr": 1, "confidential": 1,
            "urgent": 3, "blocker": 3, "critical": 3,
        }
        context_score = sum(tag_weights.get(t, 0) for t in tags_lower)
        context_score = min(context_score, 10)
        
        tag_production = 1 if any(t in tags_lower for t in ["production", "prod", "prd", "live"]) else 0
        tag_external = 1 if any(t in tags_lower for t in ["external", "internet-facing", "public", "exposed"]) else 0
        tag_sensitive = 1 if any(t in tags_lower for t in ["sensitive", "pii", "gdpr", "confidential"]) else 0
        tag_urgent = 1 if any(t in tags_lower for t in ["urgent", "blocker", "critical"]) else 0
        tag_sca = 1 if "sca" in tags_lower else 0
        tag_api = 1 if any(t in tags_lower for t in ["api", "endpoint", "rest"]) else 0
        
        age_factor = max(0.1, 1.0 - (self.days_open / 180.0))
        
        text = f"{self.title} {self.description}".lower()
        has_exploit = any(kw in text for kw in ["exploit", "metasploit", "poc", "public exploit"])
        exploit_risk = (self.cvss_score * 0.7 + severity_numeric * 0.3)
        if has_exploit:
            exploit_risk *= 1.5
        exploit_risk = min(exploit_risk, 10.0)
        
        days_open_high = 1 if self.days_open > 30 else 0
        age_x_cvss = round(self.days_open * self.cvss_score, 3)
        cvss_x_severity = round(self.cvss_score * severity_numeric, 3)
        cvss_x_epss = round(self.cvss_score * self.epss_score, 4)
        severity_x_urgent = severity_numeric * tag_urgent
        tags_count = len(tags_lower)
        
        return {
            "cvss_score": self.cvss_score,
            "severity_numeric": severity_numeric,
            "days_open": self.days_open,
            "duplicate_count": self.duplicate_count,
            "context_score": context_score,
            "exploit_risk": round(exploit_risk, 4),
            "age_factor": round(age_factor, 4),
            "tags_count": tags_count,
            "tag_production": tag_production,
            "tag_external": tag_external,
            "tag_sensitive": tag_sensitive,
            "tag_urgent": tag_urgent,
            "tag_sca": tag_sca,
            "tag_api": tag_api,
            "epss_score": self.epss_score,
            "epss_percentile": self.epss_percentile,
            "age_x_cvss": age_x_cvss,
            "cvss_x_severity": cvss_x_severity,
            "cvss_x_epss": cvss_x_epss,
            "severity_x_urgent": severity_x_urgent,
            "days_open_high": days_open_high,
            "has_cve": self.has_cve,
            "has_cwe": self.has_cwe,
        }


class BatchInput(BaseModel):
    findings: List[FindingInput] = Field(..., min_length=1, max_length=500)
    engagement_id: Optional[int] = None


class PredictionResponse(BaseModel):
    request_id: str
    finding_id: Optional[int]
    engagement_id: Optional[int]
    product_id: Optional[int]
    risk_class: int
    risk_level: str
    risk_color: str
    risk_score: float
    confidence: float
    context_score: int
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    predicted_at: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    total: int
    success: int
    errors_count: int
    results: List[PredictionResponse]
    errors: List[Dict[str, Any]]
    summary: Dict[str, int]
    processed_at: str


class HealthResponse(BaseModel):
    status: str
    api_version: str
    model_version: str
    model_ready: bool
    n_classes: int
    n_features: int
    uptime_seconds: float
    loaded_at: Optional[str]


class ProductResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created: Optional[str] = None
    findings_count: Optional[int] = None


class EngagementResponse(BaseModel):
    id: int
    name: str
    product: int
    product_name: Optional[str] = None
    status: str = "active"
    created: Optional[str] = None
    findings_count: Optional[int] = None


class FindingSummaryResponse(BaseModel):
    id: int
    title: str
    severity: str
    cvss_score: float
    tags: List[str] = []
    engagement_id: Optional[int] = None
    product_id: Optional[int] = None
    created: Optional[str] = None
    risk_class: Optional[int] = None
    risk_level: Optional[str] = None



class DefectDojoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_products(self) -> List[Dict]:
        products = []
        url = f"{self.base_url}/api/v2/products/"
        while url:
            data = self._get(url)
            products.extend(data.get("results", []))
            url = data.get("next")
        logger.info(f"Fetched {len(products)} products from DefectDojo")
        return products

    def get_engagements(self, product_id: Optional[int] = None) -> List[Dict]:
        engagements = []
        params = {}
        if product_id:
            params["product"] = product_id
        url = f"{self.base_url}/api/v2/engagements/"
        while url:
            data = self._get(url, params=params)
            engagements.extend(data.get("results", []))
            url = data.get("next")
            params = None
        logger.info(f"Fetched {len(engagements)} engagements from DefectDojo")
        return engagements

    def get_findings(self, engagement_id: Optional[int] = None, product_id: Optional[int] = None, limit: int = 100) -> List[Dict]:
        findings = []
        params = {"limit": min(limit, 100)}
        if engagement_id:
            params["engagement"] = engagement_id
        if product_id:
            params["product"] = product_id
        url = f"{self.base_url}/api/v2/findings/"
        while url and len(findings) < limit:
            data = self._get(url, params=params)
            findings.extend(data.get("results", []))
            url = data.get("next")
            params = None
        logger.info(f"Fetched {len(findings)} findings from DefectDojo")
        return findings[:limit]


def get_defectdojo_client() -> Optional[DefectDojoClient]:
    if DEFECTDOJO_API_KEY:
        return DefectDojoClient(DEFECTDOJO_URL, DEFECTDOJO_API_KEY)
    return None


class ModelManager:
    def __init__(self, model_path: Path, meta_path: Path):
        self.model_path = model_path
        self.meta_path = meta_path
        self._model = None
        self._metadata = {}
        self._loaded_at = None
        self._feature_columns = FEATURE_COLS.copy()

    def is_ready(self) -> bool:
        return self._model is not None

    def get_model(self):
        return self._model

    def get_metadata(self) -> dict:
        return self._metadata

    def load_model(self) -> bool:
        if not self.model_path.exists():
            logger.warning(f"Model not found: {self.model_path}")
            return False
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
            self._loaded_at = datetime.now(timezone.utc)
            
            if self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    self._metadata = json.load(f)
                logger.info(f"Metadata loaded: version={self._metadata.get('timestamp', 'unknown')}")
            
            if hasattr(self._model, "feature_names_in_"):
                self._feature_columns = list(self._model.feature_names_in_)
            elif self._metadata.get("feature_columns"):
                self._feature_columns = self._metadata["feature_columns"]
            
            f1 = self._metadata.get('metrics', {}).get('test_f1_weighted', 'N/A')
            logger.info(f"Model loaded successfully - F1={f1}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            return False

    @property
    def feature_columns(self) -> List[str]:
        return self._feature_columns

    @property
    def n_classes(self) -> int:
        if self._model and hasattr(self._model, "classes_"):
            return len(self._model.classes_)
        return 5

    @property
    def classes(self) -> List[int]:
        if self._model and hasattr(self._model, "classes_"):
            return [int(c) for c in self._model.classes_]
        return [0, 1, 2, 3, 4]

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def model_version(self) -> str:
        return self._metadata.get("timestamp", "unknown")


class ShapExplainer:
    def __init__(self):
        self._explainer = None
        self._is_ready = False
        self._feature_names = FEATURE_COLS.copy()

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._explainer is not None

    def load(self, model) -> None:
        try:
            import shap
            estimator = model
            if hasattr(model, "named_steps"):
                step_names = list(model.named_steps.keys())
                estimator = model.named_steps[step_names[-1]]
            self._explainer = shap.TreeExplainer(estimator)
            self._is_ready = True
            logger.info("SHAP explainer loaded successfully")
        except ImportError:
            logger.warning("SHAP not installed - explanations disabled")
        except Exception as e:
            logger.warning(f"SHAP explainer failed to load: {e}")

    def explain(self, X: pd.DataFrame, pred_class: int) -> Dict[str, Any]:
        if not self._is_ready:
            return {"error": "SHAP explainer not available"}
        try:
            import shap
            shap_values = self._explainer.shap_values(X)
            if isinstance(shap_values, list):
                sv = shap_values[pred_class][0]
            else:
                sv = shap_values[0]
            feature_names = X.columns.tolist()
            top_indices = np.argsort(np.abs(sv))[::-1][:5]
            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_names[idx],
                    "value": round(float(X.iloc[0, idx]), 4),
                    "shap_value": round(float(sv[idx]), 4),
                    "impact": "+" if sv[idx] > 0 else "-",
                })
            return {
                "top_features": top_features,
                "base_value": round(float(self._explainer.expected_value), 4),
            }
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            return {"error": str(e)}


model_manager = ModelManager(MODEL_PATH, META_PATH)
shap_explainer = ShapExplainer()
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting InvisiThreat AI Risk Engine API v{API_VERSION}")
    if model_manager.load_model():
        if shap_explainer:
            shap_explainer.load(model_manager.get_model())
    else:
        logger.warning("Model not loaded - predictions will return 503")
    yield
    logger.info("API shutdown")


app = FastAPI(
    title="InvisiThreat AI Risk Engine",
    description="API de scoring IA pour les vulnerabilites de securite",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded", "request_id": request_id},
        )
    _rate_limit_store[client_ip].append(now)
    
    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"
    
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"-> {response.status_code} ({duration:.1f}ms) IP={client_ip}"
    )
    return response


@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    return {
        "service": "InvisiThreat AI Risk Engine",
        "version": API_VERSION,
        "status": "running",
        "model_ready": model_manager.is_ready(),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check() -> HealthResponse:
    if not model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()
    
    return HealthResponse(
        status="healthy",
        api_version=API_VERSION,
        model_version=model_manager.model_version,
        model_ready=True,
        n_classes=model_manager.n_classes,
        n_features=len(model_manager.feature_columns),
        uptime_seconds=round(uptime, 1),
        loaded_at=model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Dict[str, Any]:
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()
    return {
        "invisithreat_api_version": API_VERSION,
        "invisithreat_model_ready": 1 if model_manager.is_ready() else 0,
        "invisithreat_model_f1_score": model_manager.get_metadata().get("metrics", {}).get("test_f1_weighted", 0),
        "invisithreat_uptime_seconds": uptime,
        "invisithreat_n_classes": model_manager.n_classes,
        "invisithreat_n_features": len(model_manager.feature_columns),
    }


@app.get("/model/info", tags=["Model"])
async def model_info() -> Dict[str, Any]:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_manager.model_version,
        "n_classes": model_manager.n_classes,
        "classes": model_manager.classes,
        "class_labels": CLASS_LABELS,
        "n_features": len(model_manager.feature_columns),
        "feature_columns": model_manager.feature_columns,
        "loaded_at": model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
        "metrics": model_manager.get_metadata().get("metrics", {}),
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model() -> Dict[str, Any]:
    try:
        if model_manager.load_model():
            if shap_explainer:
                shap_explainer.load(model_manager.get_model())
            return {
                "status": "reloaded",
                "model_version": model_manager.model_version,
                "loaded_at": model_manager.loaded_at.isoformat() if model_manager.loaded_at else None,
            }
        else:
            raise HTTPException(status_code=500, detail="Model reload failed")
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/defectdojo/products", response_model=List[ProductResponse], tags=["DefectDojo"])
async def get_products() -> List[ProductResponse]:
    client = get_defectdojo_client()
    if not client:
        raise HTTPException(status_code=503, detail="DefectDojo client not configured")
    
    try:
        products = client.get_products()
        return [
            ProductResponse(
                id=p.get("id"),
                name=p.get("name"),
                description=p.get("description"),
                created=p.get("created"),
                findings_count=None,
            )
            for p in products
        ]
    except Exception as e:
        logger.error(f"Failed to fetch products: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/defectdojo/engagements", response_model=List[EngagementResponse], tags=["DefectDojo"])
async def get_engagements(product_id: Optional[int] = None) -> List[EngagementResponse]:
    client = get_defectdojo_client()
    if not client:
        raise HTTPException(status_code=503, detail="DefectDojo client not configured")
    
    try:
        engagements = client.get_engagements(product_id=product_id)
        
        products = {}
        try:
            prod_list = client.get_products()
            products = {p["id"]: p["name"] for p in prod_list}
        except Exception:
            pass
        
        return [
            EngagementResponse(
                id=e.get("id"),
                name=e.get("name"),
                product=e.get("product"),
                product_name=products.get(e.get("product")),
                status=e.get("status", "active"),
                created=e.get("created"),
                findings_count=None,
            )
            for e in engagements
        ]
    except Exception as e:
        logger.error(f"Failed to fetch engagements: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/defectdojo/findings", response_model=List[FindingSummaryResponse], tags=["DefectDojo"])
async def get_findings(
    engagement_id: Optional[int] = None,
    product_id: Optional[int] = None,
    limit: int = 100
) -> List[FindingSummaryResponse]:
    client = get_defectdojo_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="DefectDojo not configured. Set DEFECTDOJO_URL and DEFECTDOJO_API_KEY environment variables."
        )
    
    try:
        findings = client.get_findings(engagement_id=engagement_id, product_id=product_id, limit=min(limit, 500))
        
        results = []
        for f in findings:
            tags = f.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            
            eng_id = f.get("engagement")
            if eng_id is None:
                eng_id = f.get("engagement_id")
            
            prod_id = f.get("product")
            if prod_id is None:
                prod_id = f.get("product_id")
            
            results.append(
                FindingSummaryResponse(
                    id=f.get("id", 0),
                    title=f.get("title", "Unknown"),
                    severity=f.get("severity", "info"),
                    cvss_score=float(f.get("cvss_score") or 0),
                    tags=tags,
                    engagement_id=int(eng_id) if eng_id is not None else None,
                    product_id=int(prod_id) if prod_id is not None else None,
                    created=f.get("created"),
                    risk_class=None,
                    risk_level=None,
                )
            )
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch findings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/defectdojo/engagements/{engagement_id}/findings", response_model=List[FindingSummaryResponse], tags=["DefectDojo"])
async def get_engagement_findings(engagement_id: int, limit: int = 500) -> List[FindingSummaryResponse]:
    client = get_defectdojo_client()
    if not client:
        raise HTTPException(
            status_code=503,
            detail="DefectDojo not configured. Set DEFECTDOJO_URL and DEFECTDOJO_API_KEY environment variables."
        )
    
    try:
        findings = client.get_findings(engagement_id=engagement_id, limit=min(limit, 500))
        
        results = []
        for f in findings:
            tags = f.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            
            results.append(
                FindingSummaryResponse(
                    id=f.get("id", 0),
                    title=f.get("title", "Unknown"),
                    severity=f.get("severity", "info"),
                    cvss_score=float(f.get("cvss_score") or 0),
                    tags=tags,
                    engagement_id=engagement_id,
                    product_id=f.get("product"),
                    created=f.get("created"),
                    risk_class=None,
                    risk_level=None,
                )
            )
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch findings for engagement {engagement_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(finding: FindingInput, request: Request) -> PredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded - retry in a few seconds"
        )
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        features = finding.to_features()
        feature_df = pd.DataFrame([features])
        
        expected_cols = model_manager.feature_columns
        for col in expected_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        X = feature_df[expected_cols]
        
        risk_class = int(model_manager.get_model().predict(X)[0])
        probabilities = model_manager.get_model().predict_proba(X)[0]
        
        classes = model_manager.classes
        risk_score = round(sum(c * p for c, p in zip(classes, probabilities)) * 100 / 4, 2)
        confidence = round(float(max(probabilities)), 4)
        
        proba_dict = {
            CLASS_LABELS.get(c, str(c)): round(float(p), 4)
            for c, p in zip(classes, probabilities)
        }
        
        features_used = {col: round(float(X[col].iloc[0]), 4) for col in expected_cols[:10]}
        
        logger.info(
            f"[{request_id}] Prediction: {finding.title[:50]} -> "
            f"{RISK_LEVELS.get(risk_class, 'unknown')} (conf={confidence:.2f})"
        )
        
        return PredictionResponse(
            request_id=request_id,
            finding_id=finding.finding_id,
            engagement_id=finding.engagement_id,
            product_id=finding.product_id,
            risk_class=risk_class,
            risk_level=CLASS_LABELS.get(risk_class, "Unknown"),
            risk_color=CLASS_COLORS.get(CLASS_LABELS.get(risk_class, "Info"), "#888888"),
            risk_score=risk_score,
            confidence=confidence,
            context_score=features["context_score"],
            probabilities=proba_dict,
            features_used=features_used,
            predicted_at=datetime.now(timezone.utc).isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(batch: BatchInput, request: Request) -> BatchPredictionResponse:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    results = []
    errors = []
    
    for idx, finding in enumerate(batch.findings):
        try:
            features = finding.to_features()
            feature_df = pd.DataFrame([features])
            
            expected_cols = model_manager.feature_columns
            for col in expected_cols:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            X = feature_df[expected_cols]
            
            risk_class = int(model_manager.get_model().predict(X)[0])
            probabilities = model_manager.get_model().predict_proba(X)[0]
            
            classes = model_manager.classes
            risk_score = round(sum(c * p for c, p in zip(classes, probabilities)) * 100 / 4, 2)
            confidence = round(float(max(probabilities)), 4)
            
            proba_dict = {
                CLASS_LABELS.get(c, str(c)): round(float(p), 4)
                for c, p in zip(classes, probabilities)
            }
            
            results.append(PredictionResponse(
                request_id=request_id,
                finding_id=finding.finding_id,
                engagement_id=finding.engagement_id,
                product_id=finding.product_id,
                risk_class=risk_class,
                risk_level=CLASS_LABELS.get(risk_class, "Unknown"),
                risk_color=CLASS_COLORS.get(CLASS_LABELS.get(risk_class, "Info"), "#888888"),
                risk_score=risk_score,
                confidence=confidence,
                context_score=features["context_score"],
                probabilities=proba_dict,
                features_used={},
                predicted_at=datetime.now(timezone.utc).isoformat(),
            ))
        except Exception as e:
            errors.append({"index": idx, "finding_id": finding.finding_id, "error": str(e)})
    
    summary = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for r in results:
        level = r.risk_level.lower()
        if level in summary:
            summary[level] += 1
    
    logger.info(f"[{request_id}] Batch: {len(results)} success, {len(errors)} errors")
    
    return BatchPredictionResponse(
        request_id=request_id,
        total=len(batch.findings),
        success=len(results),
        errors_count=len(errors),
        results=results,
        errors=errors,
        summary=summary,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/explain", tags=["Prediction"])
async def explain(finding: FindingInput, request: Request) -> Dict[str, Any]:
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not shap_explainer.is_ready:
        raise HTTPException(status_code=503, detail="SHAP explainer not available")
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    try:
        features = finding.to_features()
        feature_df = pd.DataFrame([features])
        
        expected_cols = model_manager.feature_columns
        for col in expected_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        X = feature_df[expected_cols]
        
        risk_class = int(model_manager.get_model().predict(X)[0])
        probabilities = model_manager.get_model().predict_proba(X)[0]
        
        explanation = shap_explainer.explain(X, risk_class)
        
        return {
            "request_id": request_id,
            "finding_id": finding.finding_id,
            "risk_class": risk_class,
            "risk_level": CLASS_LABELS.get(risk_class, "Unknown"),
            "confidence": round(float(max(probabilities)), 4),
            "explanation": explanation,
            "predicted_at": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.exception(f"[{request_id}] Explain error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8081"))
    host = os.getenv("API_HOST", "0.0.0.0")
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting API on {host}:{port}")
    logger.info(f"Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )