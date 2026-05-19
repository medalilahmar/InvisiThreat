"""
api_simple.py — Point d'entrée InvisiThreat AI Risk Engine.

Lancer l'API :
    python api_simple.py
    uvicorn api_simple:app --host 0.0.0.0 --port 8081 --reload
"""
import logging
import os
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from server.cache import _cache_store, _scores_cache_memory, invalidate_cache, invalidate_scores_cache
from server.config import API_VERSION, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
from server.dependencies import (
    APP_START,
    _rate_limit_store,
    lifespan,
    get_local_data_loader,   # ← getter dynamique
    get_model_manager,       # ← getter dynamique
    require_local_loader,
)
from server.model_manager import score_all_findings_at_startup
from server.schemas import HealthResponse
from server.routers import findings, predict, llm, jira
from server.routers.analytics_router import router as analytics_router

from database.connection import engine, Base, SessionLocal
from database.models import Project, User
from auth.security import get_accessible_product_ids, get_current_user
from server.routers.auth_router import router as auth_router
from server.routers.admin_router import router as admin_router
from server.routers.notifications_router import router as notifications_router
from server.routers.projects import router as projects_router


# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("invisithreat.main")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "InvisiThreat AI Risk Engine",
    description = "API de scoring IA pour les vulnérabilités de sécurité",
    version     = API_VERSION,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(findings.router)
app.include_router(predict.router)
app.include_router(llm.router)
app.include_router(jira.router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(analytics_router)
app.include_router(notifications_router)
app.include_router(projects_router)

# ══════════════════════════════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def request_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start     = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"
    now       = time.time()

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return JSONResponse(
            status_code = 429,
            content     = {"detail": "Rate limit dépassé", "request_id": request_id},
        )
    _rate_limit_store[client_ip].append(now)

    response = await call_next(request)
    duration = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"]      = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration:.1f}"
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.1f}ms)"
    )
    return response


# ══════════════════════════════════════════════════════════════════════════════
# Info & monitoring
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root() -> Dict[str, Any]:
    mm = get_model_manager()
    ld = get_local_data_loader()
    return {
        "service":           "InvisiThreat AI Risk Engine",
        "version":           API_VERSION,
        "status":            "running",
        "model_ready":       mm.is_ready(),
        "data_loader_ready": ld is not None and ld.is_ready,
        "docs":              "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check() -> HealthResponse:
    from fastapi import HTTPException
    mm = get_model_manager()
    if not mm.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()
    return HealthResponse(
        status         = "healthy",
        api_version    = API_VERSION,
        model_version  = mm.model_version,
        model_ready    = True,
        n_classes      = mm.n_classes,
        n_features     = len(mm.feature_columns),
        uptime_seconds = round(uptime, 1),
        loaded_at      = mm.loaded_at.isoformat() if mm.loaded_at else None,
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Dict[str, Any]:
    mm     = get_model_manager()
    ld     = get_local_data_loader()
    uptime = (datetime.now(timezone.utc) - APP_START).total_seconds()

    scores_dist: Dict[str, int] = {}
    for v in _scores_cache_memory.values():
        lv = v.get("ai_risk_level", "N/A")
        scores_dist[lv] = scores_dist.get(lv, 0) + 1

    return {
        "invisithreat_api_version":             API_VERSION,
        "invisithreat_model_ready":             1 if mm.is_ready() else 0,
        "invisithreat_model_f1_score":          mm.get_metadata().get("metrics", {}).get("test_f1_weighted", 0),
        "invisithreat_uptime_seconds":          uptime,
        "invisithreat_n_classes":               mm.n_classes,
        "invisithreat_n_features":              len(mm.feature_columns),
        "invisithreat_cache_entries":           len(_cache_store),
        "invisithreat_prediction_cache_hits":   mm._cache_hits,
        "invisithreat_prediction_cache_misses": mm._cache_misses,
        "invisithreat_scores_cache_entries":    len(_scores_cache_memory),
        "invisithreat_ai_distribution":         scores_dist,
        "invisithreat_data_loader_products":    len(ld.products)       if ld and ld.is_ready else 0,
        "invisithreat_data_loader_engagements": len(ld.engagements)    if ld and ld.is_ready else 0,
        "invisithreat_data_loader_findings":    len(ld.findings_by_id) if ld and ld.is_ready else 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/model/info", tags=["Model"])
async def model_info() -> Dict[str, Any]:
    from fastapi import HTTPException
    mm = get_model_manager()
    if not mm.is_ready():
        raise HTTPException(503, detail="Modèle non chargé")
    return {
        "model_version":   mm.model_version,
        "n_classes":       mm.n_classes,
        "classes":         mm.classes,
        "class_labels":    {str(k): v for k, v in __import__("server.config", fromlist=["CLASS_LABELS"]).CLASS_LABELS.items()},
        "n_features":      len(mm.feature_columns),
        "feature_columns": mm.feature_columns,
        "loaded_at":       mm.loaded_at.isoformat() if mm.loaded_at else None,
        "metrics":         mm.get_metadata().get("metrics", {}),
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model() -> Dict[str, Any]:
    from fastapi import HTTPException
    mm = get_model_manager()
    ld = get_local_data_loader()
    if mm.load_model():
        if ld and ld.is_ready:
            try:
                from server.cache import set_scores_cache
                cache = score_all_findings_at_startup(ld, mm)
                set_scores_cache(cache)
            except Exception as e:
                logger.warning(f"Re-scoring après reload échoué : {e}")
        return {
            "status":        "reloaded",
            "model_version": mm.model_version,
            "loaded_at":     mm.loaded_at.isoformat() if mm.loaded_at else None,
        }
    raise HTTPException(500, detail="Rechargement du modèle échoué")


@app.post("/data/refresh", tags=["Admin"])
async def refresh_data() -> Dict[str, Any]:
    from fastapi import HTTPException
    mm     = get_model_manager()
    loader = require_local_loader()
    try:
        if loader.load():
            if mm.is_ready():
                from server.cache import set_scores_cache
                cache = score_all_findings_at_startup(loader, mm)
                set_scores_cache(cache)
            return {
                "status":      "refreshed",
                "products":    len(loader.products),
                "engagements": len(loader.engagements),
                "findings":    len(loader.findings_by_id),
                "timestamp":   datetime.now(timezone.utc).isoformat(),
            }
        raise HTTPException(500, detail="Impossible de recharger le CSV")
    except Exception as e:
        logger.error(f"Data refresh échoué : {e}")
        raise HTTPException(500, detail=str(e))


@app.post("/cache/invalidate", tags=["Admin"])
async def invalidate_findings_cache(prefix: str = "") -> Dict[str, Any]:
    count = invalidate_cache(prefix)
    if prefix in ("", "scores"):
        invalidate_scores_cache()
    logger.info(f"Cache invalidé : {count} entrées (prefix='{prefix}')")
    return {"invalidated": count, "prefix": prefix}


@app.get("/debug/severity-check", tags=["Debug"])
async def debug_severity_check() -> Dict[str, Any]:
    from server.cache import get_scores_cache
    from server.routers.findings import _finding_to_response
    loader       = require_local_loader()
    scores_cache = get_scores_cache()
    findings_sample = loader.get_all_findings()[:10]
    results = []
    for f in findings_sample:
        response = _finding_to_response(f, scores_cache)
        results.append({
            "id":                   f.get("id"),
            "severity_in_response": response.severity,
            "ai_risk_level":        response.risk_level,
            "ai_confidence":        response.ai_confidence,
            "ai_probabilities":     response.ai_probabilities,
        })
    dist = {}
    for v in scores_cache.values():
        lv = v.get("ai_risk_level", "N/A")
        dist[lv] = dist.get(lv, 0) + 1
    return {
        "total_findings":  len(loader.get_all_findings()),
        "scores_in_cache": len(scores_cache),
        "distribution_ai": dist,
        "sample_findings": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RBAC : /auth/me
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/auth/me", tags=["🔐 Authentification"])
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id":       current_user.id,
        "username": current_user.username,
        "email":    current_user.email,
        "role":     current_user.role,
        "status":   current_user.status,
        "projects": [{"id": p.id, "name": p.name} for p in current_user.projects],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entrée principale
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port        = int(os.getenv("API_PORT",  "8081"))
    host        = os.getenv("API_HOST",      "0.0.0.0")
    reload_mode = os.getenv("API_RELOAD",    "false").lower() == "true"

    logger.info(f"Démarrage API sur {host}:{port}")
    logger.info(f"Documentation : http://localhost:{port}/docs")

    uvicorn.run("api_simple:app", host=host, port=port, reload=reload_mode, log_level="info")