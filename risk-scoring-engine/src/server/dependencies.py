"""
dependencies.py — Globals partagés, lifespan FastAPI, dépendance require_local_loader.

FIX : les singletons mutables ne sont JAMAIS importés directement par les routers.
      Les routers appellent get_local_data_loader() / get_model_manager() à chaque
      requête, ce qui lit toujours la valeur courante du module — initialisée par lifespan.
"""
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException

from server.cache import get_scores_cache, set_scores_cache
from server.config import API_VERSION, CSV_FINDINGS_PATH, MODEL_PATH, META_PATH
from server.data_loader import LocalDataLoader
from server.model_manager import ModelManager, SimpleShapExplainer, score_all_findings_at_startup

logger = logging.getLogger("invisithreat.dependencies")

# ── Singletons (valeurs mutables — NE PAS importer directement depuis les routers) ──
_model_manager:     ModelManager                  = ModelManager(MODEL_PATH, META_PATH)
_shap_explainer:    Optional[SimpleShapExplainer] = None
_local_data_loader: Optional[LocalDataLoader]     = None

# ── Rate limiting ─────────────────────────────────────────────────────────────
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)

# ── Startup timestamp ─────────────────────────────────────────────────────────
APP_START = datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# Getters — à utiliser dans TOUS les routers à la place des imports directs
# ══════════════════════════════════════════════════════════════════════════════

def get_model_manager() -> ModelManager:
    """Retourne toujours l'instance courante, même après reload."""
    return _model_manager


def get_local_data_loader() -> Optional[LocalDataLoader]:
    """Retourne l'instance courante du LocalDataLoader (None si pas encore initialisé)."""
    return _local_data_loader






# ══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _local_data_loader, _shap_explainer

    logger.info(f"Démarrage InvisiThreat AI Risk Engine API v{API_VERSION}")

    # 1 — Chargement du modèle ML
    if _model_manager.load_model():
        logger.info("✅ Modèle ML chargé")

        # 2 — SHAP
        try:
            _shap_explainer = SimpleShapExplainer(_model_manager.get_model())
            logger.info("✅ SHAP explainer chargé")
        except ImportError:
            logger.warning("⚠️  SHAP non installé — explications désactivées")
        except Exception as e:
            logger.warning(f"⚠️  SHAP explainer échoué : {e}")
    else:
        logger.warning("⚠️  Modèle non chargé — prédictions retourneront 503")

    # 3 — Chargement des données CSV
    try:
        _local_data_loader = LocalDataLoader(CSV_FINDINGS_PATH)
        if _local_data_loader.load():
            logger.info(
                f"✅ LocalDataLoader initialisé : "
                f"{len(_local_data_loader.products)} produits, "
                f"{len(_local_data_loader.engagements)} engagements, "
                f"{len(_local_data_loader.findings_by_id)} findings"
            )
            logger.info("Utilisation du cache IA existant (data/ai_scores_cache.json)")
        else:
            logger.warning("⚠️  LocalDataLoader : CSV non accessible, mode dégradé")
    except Exception as e:
        logger.error(f"❌ LocalDataLoader échoué : {e}")

    # 4 — RBAC + synchronisation des projets
    from database.connection import engine, Base, SessionLocal
    from database.models import Project

    Base.metadata.create_all(bind=engine)
    logger.info("✅ Tables RBAC vérifiées/créées")

    if _local_data_loader and _local_data_loader.is_ready:
        db = SessionLocal()
        try:
            existing_ids = {p.id for p in db.query(Project).all()}
            count = 0
            for prod_id, prod_data in _local_data_loader.products.items():
                if prod_id not in existing_ids:
                    db.add(Project(id=prod_id, name=prod_data["name"]))
                    count += 1
            db.commit()
            if count:
                logger.info(f"✅ {count} nouveau(x) projet(s) synchronisé(s)")
        except Exception as e:
            logger.error(f"❌ Erreur sync projets : {e}")
            db.rollback()
        finally:
            db.close()

    yield
    logger.info("API arrêtée")


# ══════════════════════════════════════════════════════════════════════════════
# Dépendances FastAPI (Depends)
# ══════════════════════════════════════════════════════════════════════════════

def require_local_loader() -> LocalDataLoader:
    """Dépendance FastAPI — lève 503 si le loader n'est pas prêt."""
    loader = _local_data_loader
    if loader is None or not loader.is_ready:
        raise HTTPException(status_code=503, detail="LocalDataLoader non prêt.")
    return loader


def require_model_manager() -> ModelManager:
    """Dépendance FastAPI — lève 503 si le modèle n'est pas prêt."""
    if not _model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Modèle ML non chargé.")
    return _model_manager