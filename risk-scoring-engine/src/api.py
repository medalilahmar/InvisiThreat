"""
api.py — AI Risk Engine v2.0
================================
API REST FastAPI pour la prédiction de risque
"""
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ===== IMPORT DU MODEL_MANAGER POUR PARTAGE AVEC LES TESTS =====
import sys
sys.path.insert(0, str(Path(__file__).parent / "core"))
try:
    from core.model_loader import model_manager
    # Forcer le chargement du modèle au démarrage
    model_manager.load_model()
    print(f"🚀 Modèle chargé au démarrage via model_manager: {model_manager.is_ready()}")
except ImportError as e:
    print(f"⚠️ model_manager non trouvé: {e}")
    model_manager = None
# ===== FIN DE L'AJOUT =====

# Configuration des logs
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_engine.api")

# Constantes
MODEL_PATH   = Path(os.getenv("MODEL_PATH", "models/pipeline_latest.pkl"))
META_PATH    = Path(os.getenv("META_PATH",  "models/pipeline_latest_meta.json"))
API_VERSION  = "2.0.0"
APP_START    = datetime.now(timezone.utc)

FEATURE_COLS = [
    "severity_num",
    "cvss_score",
    "age_days",
    "has_cve",
    "has_cwe",
    "tags_count",
    "is_false_positive",
    "is_active",
    "tag_urgent",
    "tag_in_production",
    "tag_sensitive",
    "tag_external",
    "severity_x_active",
    "product_fp_rate",
    "cvss_severity_gap",
    "cvss_x_severity",
    "cvss_x_has_cve",
    "severity_x_urgent",
    "age_x_cvss",
    "cvss_score_norm",
    "severity_norm",
    "age_days_norm",
    "tags_count_norm",
    "cvss_severity_gap_norm",
]

CLASS_LABELS = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
CLASS_COLORS = {
    "Info":     "#95a5a6",
    "Low":      "#2ecc71",
    "Medium":   "#f39c12",
    "High":     "#e67e22",
    "Critical": "#e74c3c",
}

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW   = int(os.getenv("RATE_LIMIT_WINDOW",   "60"))

# ===== MODEL STATE MODIFIÉ POUR UTILISER MODEL_MANAGER =====
class ModelState:
    """Conteneur pour le pipeline et ses métadonnées."""
    
    @property
    def pipeline(self):
        """Retourne le pipeline depuis model_manager"""
        if model_manager and model_manager.is_ready():
            return model_manager.get_model()
        return None
    
    @property
    def meta(self):
        """Retourne les métadonnées depuis model_manager"""
        if model_manager:
            return model_manager.get_metadata()
        return {}
    
    @property
    def loaded_at(self):
        """Retourne la date de chargement"""
        if model_manager and hasattr(model_manager, '_loaded_at'):
            return model_manager._loaded_at
        return None
    
    @property
    def model_version(self):
        """Retourne la version du modèle"""
        meta = self.meta
        return meta.get("timestamp", "unknown") if meta else "unknown"
    
    @property
    def n_features(self):
        """Retourne le nombre de features"""
        return len(FEATURE_COLS)
    
    @property
    def n_classes(self):
        """Retourne le nombre de classes"""
        if self.pipeline and hasattr(self.pipeline, 'named_steps'):
            rf = self.pipeline.named_steps.get("model")
            if rf and hasattr(rf, 'classes_'):
                return len(rf.classes_)
        return 5  # Valeur par défaut
    
    @property
    def classes(self):
        """Retourne les classes"""
        if self.pipeline and hasattr(self.pipeline, 'named_steps'):
            rf = self.pipeline.named_steps.get("model")
            if rf and hasattr(rf, 'classes_'):
                return [int(c) for c in rf.classes_]
        return [0, 1, 2, 3, 4]  # Valeur par défaut
    
    @property
    def is_ready(self) -> bool:
        """Vérifie si le modèle est prêt"""
        return model_manager is not None and model_manager.is_ready()
    
    def load(self) -> None:
        """Charge le modèle (délègue à model_manager)"""
        if model_manager:
            model_manager.load_model()
            logger.info(f"Pipeline chargé ✓ — version={self.model_version}")
        else:
            logger.error("model_manager non disponible")

model_state = ModelState()

# ══════════════════════════════════════════════
# Lifespan (chargement/libération au démarrage)
# ══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pattern moderne FastAPI : remplace @app.on_event."""
    logger.info("=" * 50)
    logger.info(f"🚀 Démarrage AI Risk Engine API v{API_VERSION}")
    logger.info("=" * 50)
    try:
        model_state.load()
    except Exception as e:
        logger.error(f"❌ Chargement pipeline échoué : {e}")
        logger.warning("L'API démarre sans modèle — /predict renverra 503.")
    yield
    logger.info("🛑 Arrêt de l'API.")

# ══════════════════════════════════════════════
# Application FastAPI
# ══════════════════════════════════════════════

app = FastAPI(
    title="AI Risk Engine",
    description=(
        "API de priorisation des vulnérabilités de sécurité.\n\n"
        "Prédit le niveau de risque (Info → Critical) à partir des métadonnées "
        "d'un finding DefectDojo."
    ),
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

# ===== Rendre model_manager accessible depuis l'app =====
app.state.model_manager = model_manager

# ══════════════════════════════════════════════
# Middleware : logging + rate limiting
# ══════════════════════════════════════════════

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Ajoute request_id, log chaque requête, rate limiting léger."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    now       = time.time()
    window    = _rate_limit_store[client_ip]
    # Nettoie les entrées hors fenêtre
    _rate_limit_store[client_ip] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit dépassé. Réessayez dans 60s.", "request_id": request_id},
        )
    _rate_limit_store[client_ip].append(now)

    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"

    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code}  ({duration:.1f}ms)  IP={client_ip}"
    )
    return response

# ══════════════════════════════════════════════
# Schémas Pydantic
# ══════════════════════════════════════════════

class FindingInput(BaseModel):
    """Données d'entrée d'une vulnérabilité — validées strictement."""

    # Champs obligatoires
    severity_num:      int   = Field(..., ge=0, le=4,  description="Sévérité encodée : Info=0, Low=1, Medium=2, High=3, Critical=4")
    cvss_score:        float = Field(..., ge=0, le=10, description="Score CVSS v3 (0-10)")
    age_days:          int   = Field(..., ge=0,        description="Âge du finding en jours")
    has_cve:           int   = Field(..., ge=0, le=1,  description="1 si un CVE est associé")
    has_cwe:           int   = Field(..., ge=0, le=1,  description="1 si un CWE est associé")
    tags_count:        int   = Field(..., ge=0,        description="Nombre de tags")
    is_false_positive: int   = Field(..., ge=0, le=1,  description="1 si marqué faux positif")
    is_active:         int   = Field(..., ge=0, le=1,  description="1 si le finding est actif")

    # Tags sémantiques (optionnels, défaut=0)
    tag_urgent:        int   = Field(default=0, ge=0, le=1, description="Tag urgent/p0/blocker détecté")
    tag_in_production: int   = Field(default=0, ge=0, le=1, description="Tag prod/production détecté")
    tag_sensitive:     int   = Field(default=0, ge=0, le=1, description="Tag PII/GDPR/sensitive détecté")
    tag_external:      int   = Field(default=0, ge=0, le=1, description="Tag internet-facing/public détecté")

    # Features contextuelles (calculées automatiquement si absentes)
    severity_x_active:     Optional[float] = Field(default=None)
    product_fp_rate:       Optional[float] = Field(default=None, ge=0, le=1)
    cvss_severity_gap:     Optional[float] = Field(default=None, ge=0)

    # Features d'interaction (calculées automatiquement si absentes)
    cvss_x_severity:   Optional[float] = Field(default=None)
    cvss_x_has_cve:    Optional[float] = Field(default=None)
    severity_x_urgent: Optional[float] = Field(default=None)
    age_x_cvss:        Optional[float] = Field(default=None)

    # Features normalisées (calculées automatiquement si absentes)
    cvss_score_norm:        Optional[float] = Field(default=None, ge=0, le=1)
    severity_norm:          Optional[float] = Field(default=None, ge=0, le=1)
    age_days_norm:          Optional[float] = Field(default=None, ge=0, le=1)
    tags_count_norm:        Optional[float] = Field(default=None, ge=0, le=1)
    cvss_severity_gap_norm: Optional[float] = Field(default=None, ge=0, le=1)

    # Métadonnées (non utilisées par le modèle)
    product_id:    Optional[int] = Field(default=None)
    engagement_id: Optional[int] = Field(default=None)
    finding_id:    Optional[int] = Field(default=None, description="ID DefectDojo du finding")

    @field_validator("cvss_score")
    @classmethod
    def round_cvss(cls, v: float) -> float:
        return round(v, 1)

    @model_validator(mode="after")
    def compute_derived_features(self) -> "FindingInput":
        """Calcule automatiquement les features dérivées si non fournies."""
        if self.severity_x_active is None:
            self.severity_x_active = float(self.severity_num * self.is_active)
        if self.cvss_severity_gap is None:
            cvss_norm = self.cvss_score / 10 * 4
            self.cvss_severity_gap = round(abs(cvss_norm - self.severity_num), 3)
        if self.product_fp_rate is None:
            self.product_fp_rate = float(self.is_false_positive)  # proxy local
        if self.cvss_x_severity is None:
            self.cvss_x_severity = round(self.cvss_score * self.severity_num, 3)
        if self.cvss_x_has_cve is None:
            self.cvss_x_has_cve = round(self.cvss_score * self.has_cve, 3)
        if self.severity_x_urgent is None:
            self.severity_x_urgent = float(self.severity_num * self.tag_urgent)
        if self.age_x_cvss is None:
            self.age_x_cvss = round(self.age_days * self.cvss_score, 3)
        # Normalisation simple 0-1
        if self.cvss_score_norm is None:
            self.cvss_score_norm = round(self.cvss_score / 10, 4)
        if self.severity_norm is None:
            self.severity_norm = round(self.severity_num / 4, 4)
        if self.age_days_norm is None:
            self.age_days_norm = round(min(self.age_days / 365, 1.0), 4)
        if self.tags_count_norm is None:
            self.tags_count_norm = round(min(self.tags_count / 20, 1.0), 4)
        if self.cvss_severity_gap_norm is None:
            self.cvss_severity_gap_norm = round(min(self.cvss_severity_gap / 4, 1.0), 4)
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "severity_num": 3,
                "cvss_score": 7.5,
                "age_days": 45,
                "has_cve": 1,
                "has_cwe": 1,
                "tags_count": 3,
                "is_false_positive": 0,
                "is_active": 1,
                "tag_urgent": 1,
                "tag_in_production": 1,
                "tag_sensitive": 0,
                "tag_external": 1,
                "product_id": 42,
                "engagement_id": 123,
                "finding_id": 9876,
            }
        }
    }


class BatchInput(BaseModel):
    """Entrée pour la prédiction en lot."""
    findings: list[FindingInput] = Field(..., min_length=1, max_length=500)


class PredictionResult(BaseModel):
    """Résultat d'une prédiction individuelle."""
    request_id:   Optional[str]
    finding_id:   Optional[int]
    product_id:   Optional[int]
    engagement_id: Optional[int]
    risk_class:   int
    risk_level:   str
    risk_color:   str
    risk_score:   float
    probabilities: dict[str, float]
    confidence:   float
    features_used: dict[str, float]
    predicted_at:  str


# ══════════════════════════════════════════════
# Utilitaires de prédiction
# ══════════════════════════════════════════════

def _to_dataframe(finding: FindingInput) -> pd.DataFrame:
    """Convertit un FindingInput en DataFrame ordonné pour le pipeline."""
    row = finding.model_dump()
    values = [row.get(col, 0.0) or 0.0 for col in FEATURE_COLS]
    return pd.DataFrame([values], columns=FEATURE_COLS)


def _predict_one(finding: FindingInput, request_id: str = "") -> PredictionResult:
    """Effectue une prédiction sur un seul finding."""
    if model_state.pipeline is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    X = _to_dataframe(finding)

    raw_class  = int(model_state.pipeline.predict(X)[0])
    raw_probas = model_state.pipeline.predict_proba(X)[0]
    classes    = model_state.classes

    # Probabilités par label
    probas_dict = {
        CLASS_LABELS.get(c, str(c)): round(float(p), 4)
        for c, p in zip(classes, raw_probas)
    }

    # Score continu 0-10 : moyenne pondérée par classe
    risk_score = round(float(sum(c * p for c, p in zip(classes, raw_probas)) * 10 / 4), 2)

    risk_level = CLASS_LABELS.get(raw_class, "Unknown")
    confidence = round(float(max(raw_probas)), 4)

    # Features utilisées (pour transparence / debug)
    features_used = {col: round(float(X[col].iloc[0]), 4) for col in FEATURE_COLS}

    return PredictionResult(
        request_id    = request_id,
        finding_id    = finding.finding_id,
        product_id    = finding.product_id,
        engagement_id = finding.engagement_id,
        risk_class    = raw_class,
        risk_level    = risk_level,
        risk_color    = CLASS_COLORS.get(risk_level, "#888888"),
        risk_score    = risk_score,
        probabilities = probas_dict,
        confidence    = confidence,
        features_used = features_used,
        predicted_at  = datetime.now(timezone.utc).isoformat(),
    )


# ══════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    return {
        "service":   "AI Risk Engine",
        "version":   API_VERSION,
        "status":    "running",
        "model_ready": model_state.is_ready,
        "docs":      "/docs",
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check enrichi : uptime, version modèle, état."""
    uptime_seconds = (datetime.now(timezone.utc) - APP_START).total_seconds()
    if not model_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé. Vérifiez les logs de démarrage.",
        )
    return {
        "status":         "healthy",
        "api_version":    API_VERSION,
        "model_version":  model_state.model_version,
        "model_ready":    True,
        "n_classes":      model_state.n_classes,
        "n_features":     model_state.n_features,
        "loaded_at":      model_state.loaded_at.isoformat() if model_state.loaded_at else None,
        "uptime_seconds": round(uptime_seconds, 1),
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Retourne les métadonnées complètes du modèle chargé."""
    if not model_state.is_ready:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")
    return {
        "model_version":  model_state.model_version,
        "n_classes":      model_state.n_classes,
        "classes":        model_state.classes,
        "class_labels":   CLASS_LABELS,
        "n_features":     model_state.n_features,
        "feature_cols":   FEATURE_COLS,
        "loaded_at":      model_state.loaded_at.isoformat() if model_state.loaded_at else None,
        "meta":           model_state.meta,
    }


@app.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict(data: FindingInput, request: Request):
    """
    Prédit le niveau de risque d'un finding unique.

    Retourne :
    - `risk_class`    : classe 0-4 (Info → Critical)
    - `risk_level`    : label textuel
    - `risk_score`    : score continu 0-10
    - `probabilities` : probabilité par classe
    - `confidence`    : probabilité de la classe prédite
    - `features_used` : valeurs des features envoyées au modèle
    """
    if not model_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible. Réessayez dans quelques secondes.",
        )

    request_id = getattr(request.state, "request_id", "")
    try:
        result = _predict_one(data, request_id=request_id)
        logger.info(
            f"[{request_id}] Prédiction → {result.risk_level} "
            f"(score={result.risk_score}  conf={result.confidence:.2f})"
        )
        return result
    except Exception as e:
        logger.exception(f"[{request_id}] Erreur prédiction : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(batch: BatchInput, request: Request):
    """
    Prédit le niveau de risque pour un lot de findings (max 500).

    Retourne les prédictions triées par risk_score décroissant
    (les plus critiques en premier).
    """
    if not model_state.is_ready:
        raise HTTPException(status_code=503, detail="Modèle non disponible.")

    request_id = getattr(request.state, "request_id", "")
    results    = []
    errors     = []

    for i, finding in enumerate(batch.findings):
        try:
            result = _predict_one(finding, request_id=request_id)
            results.append(result.model_dump())
        except Exception as e:
            errors.append({"index": i, "finding_id": finding.finding_id, "error": str(e)})
            logger.warning(f"[{request_id}] Batch erreur index={i} : {e}")

    results.sort(key=lambda r: r["risk_score"], reverse=True)

    logger.info(
        f"[{request_id}] Batch : {len(results)} succès, {len(errors)} erreurs "
        f"sur {len(batch.findings)} findings"
    )
    return {
        "request_id":   request_id,
        "total":        len(batch.findings),
        "success":      len(results),
        "errors_count": len(errors),
        "results":      results,
        "errors":       errors,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Recharge le pipeline depuis le disque sans redémarrer l'API."""
    try:
        model_state.load()
        return {
            "status":        "reloaded",
            "model_version": model_state.model_version,
            "loaded_at":     model_state.loaded_at.isoformat(),
        }
    except Exception as e:
        logger.error(f"Rechargement modèle échoué : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rechargement échoué : {str(e)}",
        )



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true",
        log_level="info",
    )