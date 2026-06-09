import pytest
import os
from nova.app import app
from nova.database import init_db, DB_FILE

@pytest.fixture
def client():
    # S'assurer d'initialiser la base de données de test
    init_db()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
    # Nettoyage après les tests
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except OSError:
            pass

def test_index(client):
    """Test la page d'accueil."""
    rv = client.get('/')
    assert b"Projet Nova" in rv.data

def test_login_vulnerable(client):
    """Test l'injection SQL de base."""
    rv = client.get('/login?username=admin')
    assert rv.status_code == 200
    assert b"admin" in rv.data
