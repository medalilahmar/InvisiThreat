import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ModelManager:
    """Gère le chargement et l'accès au modèle ML"""
    
    _instance = None
    _model = None
    _metadata = None
    _loaded_at = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.model_path = Path("models/pipeline_latest.pkl")
        self.meta_path = Path("models/pipeline_latest_meta.json")
    
    def load_model(self) -> bool:
        """Charge le modèle depuis le disque"""
        try:
            if not self.model_path.exists():
                logger.error(f"Modèle non trouvé: {self.model_path}")
                return False
            
            self._model = joblib.load(self.model_path)
            self._loaded_at = datetime.now()
            
            if self.meta_path.exists():
                with open(self.meta_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
            
            logger.info(f"✅ Modèle chargé: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def get_model(self):
        if self._model is None:
            self.load_model()
        return self._model
    
    def get_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            return {}
        return self._metadata
    
    def is_ready(self) -> bool:
        return self._model is not None

model_manager = ModelManager()