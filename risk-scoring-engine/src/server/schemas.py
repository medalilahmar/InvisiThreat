"""
schemas.py — Tous les modèles Pydantic (request + response).
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from server.config import CLASS_LABELS, CLASS_COLORS, FEATURE_COLS


# ══════════════════════════════════════════════════════════════════════════════
# Jira
# ══════════════════════════════════════════════════════════════════════════════

class JiraCreateIssueRequest(BaseModel):
    finding_id:   int
    title:        str
    severity:     str
    cvss_score:   float
    description:  Optional[str]       = None
    cve:          Optional[str]       = None
    cwe:          Optional[str]       = None
    file_path:    Optional[str]       = None
    line:         Optional[int]       = None
    tags:         Optional[List[str]] = None
    risk_level:   Optional[str]       = None
    ai_score:     Optional[float]     = None
    ai_confidence: Optional[float]   = None
    engagement_id: Optional[int]     = None
    product_name:  Optional[str]     = None

    class Config:
        schema_extra = {
            "example": {
                "finding_id": 42,
                "title":      "SQL Injection in login form",
                "severity":   "CRITICAL",
                "cvss_score": 9.8,
            }
        }


class JiraIssueResponse(BaseModel):
    key:            str
    id:             str
    self:           str
    url:            Optional[str]  = None
    already_exists: bool           = False
    message:        Optional[str]  = None


class JiraHealthResponse(BaseModel):
    status:      str
    jira_server: Optional[str] = None
    project_key: Optional[str] = None
    connected:   bool
    message:     Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# Finding input
# ══════════════════════════════════════════════════════════════════════════════

class FindingInput(BaseModel):
    severity:        str           = Field(..., description="critical, high, medium, low, info")
    cvss_score:      float         = Field(0.0, ge=0.0, le=10.0)
    title:           str           = ""
    description:     str           = ""
    file_path:       str           = ""
    tags:            List[str]     = []
    finding_id:      Optional[int] = None
    engagement_id:   Optional[int] = None
    product_id:      Optional[int] = None
    days_open:       int           = 0
    duplicate_count: int           = 0
    epss_score:      float         = 0.0
    epss_percentile: float         = 0.0
    has_cve:         int           = 0
    has_cwe:         int           = 0

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        valid = ["critical", "high", "medium", "low", "info", "informational"]
        v_lower = v.lower().strip()
        if v_lower not in valid:
            raise ValueError(f"Severity doit être l'un de : {valid}")
        return v_lower

    @field_validator("cvss_score")
    @classmethod
    def round_cvss(cls, v: float) -> float:
        return round(v, 1)

    def to_features(self) -> Dict[str, Any]:
        """
        Génère exactement les 22 features de FEATURE_COLS (= train.py).
        Ne contient PAS severity_num, exploit_risk, days_open_high,
        epss_score_norm, score_composite_*, cvss_severity_gap, etc.
        """
        tags_lower = [t.lower() for t in self.tags]

        tag_urgent        = 1 if any(t in tags_lower for t in ["urgent", "blocker", "p0", "p1"]) else 0
        tag_in_production = 1 if any(t in tags_lower for t in ["production", "prod", "prd", "live"]) else 0
        tag_sensitive     = 1 if any(t in tags_lower for t in ["sensitive", "pii", "gdpr", "confidential"]) else 0
        tag_external      = 1 if any(t in tags_lower for t in ["external", "internet-facing", "public", "exposed"]) else 0

        tags_count      = len(tags_lower)
        tags_count_norm = round(min(tags_count / 20.0, 1.0), 4)

        cvss_score_norm = round(min(self.cvss_score / 10.0, 1.0), 4)
        age_days        = self.days_open
        age_days_norm   = round(min(age_days / 365.0, 1.0), 4)

        epss_x_cvss   = round(self.cvss_score * self.epss_score, 4)
        has_high_epss = 1 if self.epss_score > 0.5 else 0

        cvss_x_has_cve = round(self.cvss_score * self.has_cve, 4)
        age_x_cvss     = round(age_days * self.cvss_score, 3)

        # context_score : score métier basé sur les tags (max 5 — preprocess.py étape 7)
        context_score = min(tag_in_production * 2 + tag_external * 2 + tag_sensitive * 1, 5)

        # exposure_norm = context_score / 5  (preprocess.py étape 10)
        exposure_norm = round(context_score / 5.0, 4)

        # delay_norm : urgence opérationnelle (preprocess.py étape 10)
        days_clipped = min(age_days, 365)
        delay_norm   = round(1.0 - (days_clipped / (days_clipped + 30)), 4)

        return {
            "cvss_score":        self.cvss_score,
            "cvss_score_norm":   cvss_score_norm,
            "has_cve":           self.has_cve,
            "has_cwe":           self.has_cwe,
            "epss_score":        self.epss_score,
            "epss_percentile":   self.epss_percentile,
            "has_high_epss":     has_high_epss,
            "epss_x_cvss":       epss_x_cvss,
            "age_days":          age_days,
            "age_days_norm":     age_days_norm,
            "delay_norm":        delay_norm,
            "tag_urgent":        tag_urgent,
            "tag_in_production": tag_in_production,
            "tag_sensitive":     tag_sensitive,
            "tag_external":      tag_external,
            "tags_count":        tags_count,
            "tags_count_norm":   tags_count_norm,
            "context_score":     context_score,
            "exposure_norm":     exposure_norm,
            "product_fp_rate":   0.0,
            "cvss_x_has_cve":    cvss_x_has_cve,
            "age_x_cvss":        age_x_cvss,
        }


class BatchInput(BaseModel):
    findings:      List[FindingInput] = Field(..., min_length=1, max_length=500)
    engagement_id: Optional[int]      = None


# ══════════════════════════════════════════════════════════════════════════════
# Prediction responses
# ══════════════════════════════════════════════════════════════════════════════

class PredictionResponse(BaseModel):
    request_id:    str
    finding_id:    Optional[int]
    engagement_id: Optional[int]
    product_id:    Optional[int]
    risk_class:    int
    risk_level:    str
    risk_color:    str
    risk_score:    float
    confidence:    float
    context_score: int
    cvss_score:    float               = 0.0
    cve:           Optional[str]       = None
    probabilities: Dict[str, float]
    features_used: Dict[str, float]
    predicted_at:  str


class BatchPredictionResponse(BaseModel):
    request_id:   str
    total:        int
    success:      int
    errors_count: int
    results:      List[PredictionResponse]
    errors:       List[Dict[str, Any]]
    summary:      Dict[str, int]
    processed_at: str


# ══════════════════════════════════════════════════════════════════════════════
# Health / monitoring
# ══════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status:         str
    api_version:    str
    model_version:  str
    model_ready:    bool
    n_classes:      int
    n_features:     int
    uptime_seconds: float
    loaded_at:      Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# DefectDojo responses
# ══════════════════════════════════════════════════════════════════════════════

class ProductResponse(BaseModel):
    id:             int
    name:           str
    description:    Optional[str] = None
    created:        Optional[str] = None
    findings_count: Optional[int] = None


class EngagementResponse(BaseModel):
    id:             int
    name:           str
    product:        int
    product_name:   Optional[str] = None
    status:         str           = "active"
    created:        Optional[str] = None
    findings_count: Optional[int] = None


class FindingSummaryResponse(BaseModel):
    id:              int
    title:           str
    severity:        str
    cvss_score:      float
    cve:             Optional[str]       = None
    tags:            List[str]           = []
    test_id:         Optional[int]       = None
    engagement_id:   Optional[int]       = None
    engagement_name: Optional[str]       = None
    product_id:      Optional[int]       = None
    product_name:    Optional[str]       = None
    created:         Optional[str]       = None
    age_days:        Optional[int]       = None
    file_path:       Optional[str]       = None
    line:            Optional[int]       = None
    has_cve:         Optional[int]       = None
    description:     Optional[str]       = ""
    # ── Champs IA ──────────────────────────────────────────────────────────
    risk_class:          Optional[int]               = None
    risk_level:          Optional[str]               = None
    risk_color:          Optional[str]               = None
    ai_confidence:       Optional[float]             = None
    ai_probabilities:    Optional[Dict[str, float]]  = None
    context_score:       Optional[int]               = None
    exposure_norm:       Optional[float]             = None
    ai_risk_score_cont:  Optional[float]             = None
    model_base_score:    Optional[float]             = None
    business_nudge:      Optional[float]             = None
    shap_features:       Optional[List[Dict[str, Any]]] = None


# ══════════════════════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════════════════════

class LLMRequest(BaseModel):
    finding_id:  Optional[int]       = None
    title:       str
    severity:    str
    cvss_score:  Optional[float]     = 0.0
    description: Optional[str]       = ""
    cve:         Optional[str]       = ""
    cwe:         Optional[str]       = ""
    file_path:   Optional[str]       = ""
    tags:        Optional[List[str]] = []
    risk_level:  Optional[str]       = ""


class LLMExplanationResponse(BaseModel):
    finding_id:              Optional[int] = None
    summary:                 str
    impact:                  str
    root_cause:              Optional[str] = None
    exploitation_difficulty: Optional[str] = None
    priority_note:           str
    from_cache:              bool          = False


class LLMRecommendationResponse(BaseModel):
    finding_id:      Optional[int]       = None
    title:           str
    recommendations: List[str]
    references:      Optional[List[str]] = []
    verification:    Optional[str]       = None
    prevention:      Optional[str]       = None
    from_cache:      bool                = False