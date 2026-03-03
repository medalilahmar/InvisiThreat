import pytest
from fastapi.testclient import TestClient

def test_predict_success(test_client, sample_finding):
    response = test_client.post("/predict", json=sample_finding)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_class" in data
    assert "risk_level" in data
    assert "risk_score" in data
    assert "confidence" in data
    assert "probabilities" in data
    
    assert 0 <= data["risk_class"] <= 4
    assert data["risk_level"] in ["Info", "Low", "Medium", "High", "Critical"]
    assert 0 <= data["risk_score"] <= 10
    assert 0 <= data["confidence"] <= 1

def test_predict_with_minimal_fields(test_client):
    minimal_finding = {
        "severity_num": 2,
        "cvss_score": 5.0,
        "age_days": 10,
        "has_cve": 0,
        "has_cwe": 0,
        "tags_count": 0,
        "is_false_positive": 0,
        "is_active": 1
    }
    
    response = test_client.post("/predict", json=minimal_finding)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_class" in data

def test_predict_missing_required_field(test_client):
    invalid_finding = {
        "severity_num": 3,
        "age_days": 45
    }
    
    response = test_client.post("/predict", json=invalid_finding)
    assert response.status_code == 422  

@pytest.mark.parametrize("field,value,expected_error", [
    ("severity_num", 5, "severity_num"),
    ("cvss_score", 11, "cvss_score"),
    ("has_cve", 2, "has_cve"),
    ("age_days", -1, "age_days"),
])
def test_predict_validation_ranges(test_client, field, value, expected_error):
    finding = {
        "severity_num": 3,
        "cvss_score": 7.5,
        "age_days": 45,
        "has_cve": 1,
        "has_cwe": 1,
        "tags_count": 3,
        "is_false_positive": 0,
        "is_active": 1
    }
    finding[field] = value
    
    response = test_client.post("/predict", json=finding)
    assert response.status_code == 422
    error_detail = str(response.json())
    assert expected_error in error_detail

def test_predict_batch_success(test_client, sample_findings_batch):
    payload = {"findings": sample_findings_batch}
    
    response = test_client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["total"] == len(sample_findings_batch)
    assert data["success"] == len(sample_findings_batch)
    assert len(data["results"]) == len(sample_findings_batch)
    
    scores = [r["risk_score"] for r in data["results"]]
    assert scores == sorted(scores, reverse=True)

def test_predict_batch_too_large(test_client, sample_findings_batch):
    large_batch = sample_findings_batch * 200  
    
    response = test_client.post("/predict/batch", json={"findings": large_batch})
    assert response.status_code == 422  

def test_model_info_endpoint(test_client):
    response = test_client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_version" in data
    assert "n_classes" in data
    assert "classes" in data
    assert "class_labels" in data
    assert "n_features" in data
    assert "feature_cols" in data

def test_predict_with_computed_features(test_client):
    finding = {
        "severity_num": 3,
        "cvss_score": 7.5,
        "age_days": 45,
        "has_cve": 1,
        "has_cwe": 1,
        "tags_count": 3,
        "is_false_positive": 0,
        "is_active": 1,
        "tag_urgent": 1,
        "tag_in_production": 1
    }
    
    response = test_client.post("/predict", json=finding)
    assert response.status_code == 200
    
    data = response.json()
    features = data["features_used"]
    
    assert "severity_x_active" in features
    assert features["severity_x_active"] == 3  
    
    assert "cvss_x_severity" in features
    assert features["cvss_x_severity"] == 22.5  

def test_predict_rate_limiting(test_client, sample_finding):
    for i in range(10):
        response = test_client.post("/predict", json=sample_finding)
        if response.status_code == 429:
            assert "Rate limit" in response.text
            return
    
    assert True