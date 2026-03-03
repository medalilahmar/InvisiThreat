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

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_PATH = str(PROJECT_ROOT / "src")
APP_PATH = str(PROJECT_ROOT / "app")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if APP_PATH not in sys.path:
    sys.path.insert(0, APP_PATH)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f" Chemins ajoutés pour les tests:")
print(f"   - {SRC_PATH}")
print(f"   - {APP_PATH}")
print(f"   - {PROJECT_ROOT}")

try:
    from core.model_loader import model_manager
    print(" Import réussi: from core.model_loader import model_manager")
except ImportError:
    try:
        from src.core.model_loader import model_manager
        print(" Import réussi: from src.core.model_loader import model_manager")
    except ImportError:
        try:
            from app.core.model_loader import model_manager
            print(" Import réussi: from app.core.model_loader import model_manager")
        except ImportError:
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "src" / "core"))
                sys.path.insert(0, str(PROJECT_ROOT / "app" / "core"))
                import model_loader
                model_manager = model_loader.model_manager
                print(" Import réussi: import model_loader")
            except ImportError as e:
                print(f" Échec de tous les imports: {e}")
                class MockModelManager:
                    def load_model(self): return False
                    def get_model(self): return None
                    def is_ready(self): return False
                    def get_metadata(self): return {}
                model_manager = MockModelManager()
                print(" Utilisation d'un mock ModelManager")

@pytest.fixture
def sample_finding():
    return {
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
        "finding_id": 9876,
        "product_id": 42,
        "engagement_id": 123
    }

@pytest.fixture
def sample_findings_batch():
    return [
        {
            "severity_num": 4,
            "cvss_score": 9.5,
            "age_days": 1,
            "has_cve": 1,
            "has_cwe": 1,
            "tags_count": 5,
            "is_false_positive": 0,
            "is_active": 1,
            "tag_urgent": 1,
            "tag_in_production": 1,
            "tag_sensitive": 1,
            "tag_external": 1,
            "finding_id": 1001
        },
        {
            "severity_num": 2,
            "cvss_score": 5.0,
            "age_days": 30,
            "has_cve": 0,
            "has_cwe": 1,
            "tags_count": 2,
            "is_false_positive": 0,
            "is_active": 1,
            "tag_urgent": 0,
            "tag_in_production": 1,
            "tag_sensitive": 0,
            "tag_external": 0,
            "finding_id": 1002
        },
        {
            "severity_num": 0,
            "cvss_score": 1.5,
            "age_days": 90,
            "has_cve": 0,
            "has_cwe": 0,
            "tags_count": 0,
            "is_false_positive": 1,
            "is_active": 0,
            "tag_urgent": 0,
            "tag_in_production": 0,
            "tag_sensitive": 0,
            "tag_external": 0,
            "finding_id": 1003
        }
    ]

@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame({
        'id': [1, 2, 3],
        'title': ['SQL Injection', 'XSS', 'Info Leak'],
        'severity': ['High', 'Medium', 'Low'],
        'cvss': [7.5, 5.0, 2.5],
        'created': ['2026-01-01', '2026-01-15', '2026-02-01'],
        'cve': ['CVE-2024-1234', None, None],
        'cwe': [89, 79, None],
        'tags': [['urgent', 'prod'], ['prod'], []],
        'false_p': [False, False, True],
        'active': [True, True, True]
    })

@pytest.fixture
def sample_processed_dataframe():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'severity_num': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'cvss_score': np.random.uniform(0, 10, n_samples),
        'age_days': np.random.randint(1, 100, n_samples),
        'has_cve': np.random.choice([0, 1], n_samples),
        'has_cwe': np.random.choice([0, 1], n_samples),
        'tags_count': np.random.randint(0, 5, n_samples),
        'is_false_positive': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'is_active': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'tag_urgent': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'tag_in_production': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'tag_sensitive': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'tag_external': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'product_fp_rate': np.random.uniform(0, 0.1, n_samples),
        'cvss_severity_gap': np.random.uniform(0, 4, n_samples),
        'risk_class': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.1, 0.4, 0.3, 0.15, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    df['severity_x_active'] = df['severity_num'] * df['is_active']
    df['cvss_x_severity'] = df['cvss_score'] * df['severity_num']
    df['cvss_x_has_cve'] = df['cvss_score'] * df['has_cve']
    df['severity_x_urgent'] = df['severity_num'] * df['tag_urgent']
    df['age_x_cvss'] = df['age_days'] * df['cvss_score']
    df['cvss_score_norm'] = df['cvss_score'] / 10
    df['severity_norm'] = df['severity_num'] / 4
    df['age_days_norm'] = df['age_days'] / 365
    df['tags_count_norm'] = df['tags_count'] / 10
    df['cvss_severity_gap_norm'] = df['cvss_severity_gap'] / 4
    
    return df

@pytest.fixture
def mock_model():
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(
            n_estimators=10,
            max_depth=5,
            random_state=42
        ))
    ])
    
    X = np.random.rand(100, 24)
    y = np.random.choice([0, 1, 2, 3, 4], 100)
    pipeline.fit(X, y)
    
    return pipeline

@pytest.fixture
def temp_model_path(mock_model):
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        joblib.dump(mock_model, f.name)
        yield f.name
    Path(f.name).unlink()  

@pytest.fixture
def test_client():
    """Client de test FastAPI"""
    try:
        from fastapi.testclient import TestClient
        
        try:
            from api import app
            print(" Import réussi: from api import app")
        except ImportError:
            try:
                from src.api import app
                print(" Import réussi: from src.api import app")
            except ImportError:
                try:
                    from app.main import app
                    print(" Import réussi: from app.main import app")
                except ImportError as e:
                    print(f" Échec import app: {e}")
                    return None
        
        try:
            app.state.model_manager = model_manager
        except:
            pass
            
        return TestClient(app)
    except Exception as e:
        print(f" Erreur création client: {e}")
        return None

@pytest.fixture(scope="session", autouse=True)
def load_model_for_tests():
    """Charge le modèle avant tous les tests et le partage avec l'API"""
    try:
        model_path = Path("models/pipeline_latest.pkl")
        if not model_path.exists():
            print(f" Modèle non trouvé à: {model_path.absolute()}")
        else:
            print(f" Modèle trouvé à: {model_path.absolute()}")
        
        result = model_manager.load_model()
        print(f" Modèle chargé: {result}")
        
        try:
            from api import app
            app.state.model_manager = model_manager
            print(" Modèle partagé avec l'API")
        except:
            try:
                from src.api import app
                app.state.model_manager = model_manager
                print(" Modèle partagé avec l'API (src)")
            except:
                try:
                    from app.main import app
                    app.state.model_manager = model_manager
                    print(" Modèle partagé avec l'API (app)")
                except Exception as e:
                    print(f" Impossible de partager le modèle: {e}")
            
    except Exception as e:
        print(f" Modèle non chargé: {e}")