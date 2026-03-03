import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
APP_PATH = str(PROJECT_ROOT / "app")
def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert "model_ready" in data

def test_health_model_info(test_client):
    response = test_client.get("/health")
    data = response.json()
    
    assert "model_version" in data
    assert "n_classes" in data
    assert "n_features" in data
    assert "loaded_at" in data

def test_health_uptime(test_client):
    response = test_client.get("/health")
    data = response.json()
    
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0

def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "AI Risk Engine"
    assert "version" in data
    assert "status" in data