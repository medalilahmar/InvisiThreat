"""
routers/llm.py
--------------
FastAPI router exposing LLM-powered vulnerability analysis endpoints.

Endpoints:
  POST /explain/llm                                  — explain a finding (free-form request body)
  POST /recommend/llm                                — recommend remediation (free-form request body)
  GET  /defectdojo/findings/{id}/explain             — explain a finding by ID
  GET  /defectdojo/findings/{id}/recommend           — recommend remediation by ID
  GET  /defectdojo/findings/{id}/solution            — generate a code-level fix by ID
  POST /defectdojo/findings/{id}/autofix             — apply fix and open a GitHub Pull Request
  GET  /llm/health                                   — check Ollama connectivity and loaded models

The LLM backend is the local Ollama instance configured via the OLLAMA_URL
and OLLAMA_MODEL environment variables (see llm_service.py).
"""

import logging
import os
from typing import Optional
from datetime import datetime

import requests
from fastapi import APIRouter, HTTPException
from github import Github, GithubException
from pydantic import BaseModel

from server.dependencies import get_local_data_loader

from server.routers.findings import get_finding_by_id
from server.schemas import (
    LLMExplanationResponse,
    LLMRecommendationResponse,
    LLMRequest,
)
from database.connection import get_db
from sqlalchemy.orm import Session
from fastapi import Depends
from notifications.service import notify_pr_merged

logger = logging.getLogger("invisithreat.llm")

router = APIRouter(tags=["LLM"])

# GitHub integration settings — only required for the /autofix endpoint
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_BRANCH = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

# Ollama settings — read here so the health endpoint stays consistent
# with whatever llm_service.py is using.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.11.170:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

# Guard import: if llm_service is not installed the router still loads,
# all LLM endpoints simply return their static fallback responses.
try:
    from llm_service import explain_finding as _llm_explain
    _LLM_AVAILABLE = True
    logger.info("LLM service loaded successfully — backend: Ollama")
except ImportError:
    _LLM_AVAILABLE = False
    logger.warning(
        "llm_service module not found — all LLM endpoints will return fallback responses."
    )

    def _llm_explain(*args, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Response schemas specific to this router
# ---------------------------------------------------------------------------


class LLMSolutionResponse(BaseModel):
    finding_id: int
    vulnerable_snippet: Optional[str] = None
    fixed_snippet: Optional[str] = None
    explanation: str
    confidence: float
    file_path: Optional[str] = None
    line: Optional[int] = None
    has_file: bool = False
    from_cache: bool = False


class AutoFixResponse(BaseModel):
    pr_url: str
    pr_number: int
    branch_name: str
    status: str


# ---------------------------------------------------------------------------
# Static fallback helpers — returned when the LLM is unavailable or fails
# ---------------------------------------------------------------------------


def _explanation_fallback(
    fid: int, title: str, severity: str, cvss
) -> LLMExplanationResponse:
    """Return a minimal explanation response when the LLM cannot be reached."""
    return LLMExplanationResponse(
        finding_id=fid,
        summary=f"Vulnerability {severity.upper()}: {title}",
        impact=f"Severity {severity} (CVSS: {cvss or 'N/A'}) — security risk requiring review.",
        root_cause="Automatic analysis unavailable.",
        exploitation_difficulty="Medium",
        priority_note=(
            "Immediate"
            if severity.lower() in ("critical", "high")
            else "Week"
            if severity.lower() == "medium"
            else "Month"
        ),
    )


def _recommendation_fallback(fid: int) -> LLMRecommendationResponse:
    """Return a generic remediation plan when the LLM cannot be reached."""
    return LLMRecommendationResponse(
        finding_id=fid,
        title="General Remediation Recommendations",
        recommendations=[
            "Review the source code around the affected file and identify all occurrences of the vulnerable pattern.",
            "Apply the OWASP-recommended fix for this vulnerability class following your language and framework guidelines.",
            "Write tests covering both normal use cases and adversarial inputs to prevent regressions.",
            "Re-run the security scanner after applying the fix to confirm the finding is resolved.",
        ],
        references=["https://owasp.org/www-project-top-ten/"],
        verification="Re-run the SAST scanner and confirm zero findings for this vulnerability class.",
        prevention="Integrate a SAST rule in the CI/CD pipeline to catch this class of vulnerability in future pull requests.",
    )


def _solution_fallback(
    fid: int, title: str, severity: str
) -> LLMSolutionResponse:
    """Return an empty solution response when the LLM cannot generate a fix."""
    return LLMSolutionResponse(
        finding_id=fid,
        vulnerable_snippet=None,
        fixed_snippet=None,
        explanation=(
            f"Automatic code fix unavailable for '{title}' ({severity}). "
            "Verify that the finding is a SAST type with file_path and line populated."
        ),
        confidence=0.0,
        has_file=False,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_github_url(url: str) -> tuple[str, str]:
    """
    Extract (owner, repo_name) from a GitHub repository URL.
    Supports both HTTPS and SSH formats, with or without .git suffix.
    """
    url = url.rstrip("/").replace(".git", "")
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return parts[-2], parts[-1]


def _get_finding_dict(finding_id: int) -> dict:
    from server.dependencies import get_local_data_loader
    loader = get_local_data_loader()

    if loader is None or not loader.is_ready:
        raise HTTPException(503, "Data loader non prêt — attendez le démarrage ou appelez /data/refresh")

    finding = loader.findings_by_id.get(finding_id)
    if not finding:
        raise HTTPException(404, f"Finding #{finding_id} introuvable")

    return finding




def _get_engagement_repo_url(engagement_id: int, loader) -> str:
    """Retourne source_code_management_uri de l'engagement, ou chaîne vide."""
    if loader is None or engagement_id is None:
        return ""
    engagement = loader.engagements_by_id.get(engagement_id) if hasattr(loader, "engagements_by_id") else None
    if engagement:
        return (engagement.get("source_code_management_uri") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Existing endpoints — explanation and recommendation (unchanged logic)
# ---------------------------------------------------------------------------


@router.post("/explain/llm", response_model=LLMExplanationResponse)
async def explain_with_llm(request: LLMRequest) -> LLMExplanationResponse:
    """Explain a vulnerability from a free-form LLMRequest body."""
    result = (
        _llm_explain(request.model_dump(), use_cache=True, mode="explanation")
        if _LLM_AVAILABLE
        else None
    )
    if not result or not result.get("summary"):
        return _explanation_fallback(
            request.finding_id, request.title, request.severity, request.cvss_score or 0
        )
    return LLMExplanationResponse(
        finding_id=request.finding_id,
        summary=result["summary"],
        impact=result.get("impact", ""),
        root_cause=result.get("root_cause"),
        exploitation_difficulty=result.get("exploitation_difficulty"),
        priority_note=result.get("priority_note", ""),
        from_cache=result.get("from_cache", False),
    )


@router.post("/recommend/llm", response_model=LLMRecommendationResponse)
async def recommend_with_llm(request: LLMRequest) -> LLMRecommendationResponse:
    """Return a remediation plan for a vulnerability from a free-form LLMRequest body."""
    result = (
        _llm_explain(request.model_dump(), use_cache=True, mode="recommendation")
        if _LLM_AVAILABLE
        else None
    )
    if not result or not result.get("recommendations"):
        return _recommendation_fallback(request.finding_id)
    return LLMRecommendationResponse(
        finding_id=request.finding_id,
        title=result.get("title", "AI Recommendations"),
        recommendations=result.get("recommendations", []),
        references=result.get("references", []),
        verification=result.get("verification"),
        prevention=result.get("prevention"),
        from_cache=result.get("from_cache", False),
    )


@router.get(
    "/defectdojo/findings/{finding_id}/explain",
    response_model=LLMExplanationResponse,
)
async def explain_finding_from_dojo(finding_id: int) -> LLMExplanationResponse:
    """Explain a DefectDojo finding by ID."""
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id=finding_id,
        title=finding.title,
        severity=finding.severity,
        cvss_score=finding.cvss_score,
        description=finding.description or "",
        file_path=finding.file_path or "",
        tags=finding.tags or [],
        risk_level=finding.risk_level or "",
    )
    return await explain_with_llm(req)


@router.get(
    "/defectdojo/findings/{finding_id}/recommend",
    response_model=LLMRecommendationResponse,
)
async def recommend_finding_from_dojo(finding_id: int) -> LLMRecommendationResponse:
    """Return a remediation plan for a DefectDojo finding by ID."""
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id=finding_id,
        title=finding.title,
        severity=finding.severity,
        cvss_score=finding.cvss_score,
        description=finding.description or "",
        file_path=finding.file_path or "",
        tags=finding.tags or [],
        risk_level=finding.risk_level or "",
    )
    return await recommend_with_llm(req)


# ---------------------------------------------------------------------------
# Solution endpoint — AI code-level fix
# ---------------------------------------------------------------------------


@router.get(
    "/defectdojo/findings/{finding_id}/solution",
    response_model=LLMSolutionResponse,
)
async def solution_finding_from_dojo(finding_id: int) -> LLMSolutionResponse:
    """
    Generate an AI code fix for a SAST finding.
    Requires file_path, line, description, and mitigation to be populated
    in the finding (from findings_clean.csv or equivalent source).
    """
    finding = _get_finding_dict(finding_id)

    file_path = (
        finding.get("file_path") or finding.get("sast_source_file_path") or ""
    ).strip()
    line_raw = finding.get("line") or finding.get("sast_source_line")
    line_num = (
        int(line_raw)
        if line_raw and str(line_raw) not in ("", "nan")
        else None
    )
    is_static = str(finding.get("static_finding", "")).lower() in ("true", "1", "yes")
    has_file = bool(file_path and line_num and is_static)

    if not _LLM_AVAILABLE:
        return _solution_fallback(
            finding_id, finding.get("title", ""), finding.get("severity", "")
        )

    result = _llm_explain(finding, use_cache=True, mode="solution")

    if not result or not result.get("fixed_snippet"):
        return _solution_fallback(
            finding_id, finding.get("title", ""), finding.get("severity", "")
        )

    return LLMSolutionResponse(
        finding_id=finding_id,
        vulnerable_snippet=result.get("vulnerable_snippet"),
        fixed_snippet=result.get("fixed_snippet"),
        explanation=result.get("explanation", ""),
        confidence=float(result.get("confidence", 0.5)),
        file_path=file_path or None,
        line=line_num,
        has_file=has_file,
        from_cache=result.get("from_cache", False),
    )


# ---------------------------------------------------------------------------
# Auto-fix endpoint — applies the AI fix and opens a GitHub Pull Request
# ---------------------------------------------------------------------------
import math

def _safe_str(val, default: str = "") -> str:
    """Convertit une valeur pandas (potentiellement nan/None) en string propre."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return str(val).strip()

@router.post(
    "/defectdojo/findings/{finding_id}/autofix",
    response_model=AutoFixResponse,
)
async def autofix_finding(finding_id: int,db: Session = Depends(get_db),) -> AutoFixResponse:
    """
    Full automated fix workflow (sans suppression de branche).
    """
    if not GITHUB_TOKEN:
        raise HTTPException(503, "GITHUB_TOKEN not configured")

    finding = _get_finding_dict(finding_id)
    title     = _safe_str(finding.get("title"))
    file_path = _safe_str(finding.get("file_path")) or _safe_str(finding.get("sast_source_file_path"))
    line_raw  = finding.get("line") or finding.get("sast_source_line")
    line_num  = int(float(line_raw)) if line_raw and _safe_str(line_raw) not in ("", "nan") else None
    branch    = _safe_str(finding.get("branch")) or GITHUB_BRANCH
    repo_url  = _safe_str(finding.get("repo_url"))

    if not file_path or not line_num:
        raise HTTPException(400, "file_path and line required")

    repo_url = (finding.get("repo_url") or "").strip()
    if not repo_url:
        raise HTTPException(400, "repo_url missing")

    if not _LLM_AVAILABLE:
        raise HTTPException(503, "LLM service not available")

    result = _llm_explain(finding, use_cache=True, mode="solution")
    if not result or not result.get("fixed_snippet"):
        raise HTTPException(422, "Could not generate fix")

    fixed_snippet = result["fixed_snippet"]

    try:
        g = Github(GITHUB_TOKEN)
        owner, repo_name = _parse_github_url(repo_url)
        repo = g.get_repo(f"{owner}/{repo_name}")
    except (GithubException, ValueError) as exc:
        raise HTTPException(400, f"GitHub error: {exc}")

    # Nom de branche unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fix_branch = f"fix/invisithreat-{finding_id}_{timestamp}"

    try:
        file_obj = repo.get_contents(file_path, ref=branch)
        original_content = file_obj.decoded_content.decode("utf-8")
        file_sha = file_obj.sha
    except GithubException:
        raise HTTPException(404, f"File not found: {file_path} on branch {branch}")

    lines = original_content.split("\n")
    target_index = max(0, line_num - 1)
    fixed_lines = fixed_snippet.split("\n")
    new_content = "\n".join(lines[:target_index] + fixed_lines + lines[target_index + 1:])

    if new_content.strip() == original_content.strip():
        raise HTTPException(400, "No changes to commit")

    try:
        base_sha = repo.get_branch(branch).commit.sha
        repo.create_git_ref(ref=f"refs/heads/{fix_branch}", sha=base_sha)
    except GithubException as exc:
        raise HTTPException(500, f"Failed to create branch: {exc.data}")

    commit_message = (
        f"fix(security): auto-fix finding #{finding_id}\n\n"
        f"Title : {title}\nFile : {file_path}\nLine : {line_num}\n"
        f"Tool : InvisiThreat AI (Ollama / {OLLAMA_MODEL})"
    )
    try:
        repo.update_file(
            path=file_path,
            message=commit_message,
            content=new_content,
            sha=file_sha,
            branch=fix_branch,
        )
    except GithubException as exc:
        raise HTTPException(500, f"Failed to commit: {exc.data}")

    pr_body = (
        f"## Security Auto-Fix — InvisiThreat\n\n"
        f"### Finding #{finding_id} — {title}\n\n"
        f"| Field | Value |\n"
        f"|---|---|\n"
        f"| File | `{file_path}` |\n"
        f"| Line | `{line_num}` |\n"
        f"| Branch | `{fix_branch}` -> `{branch}` |\n"
        f"| AI Model | Ollama / {OLLAMA_MODEL} |\n\n"
        f"### Review Checklist\n"
        f"- [ ] Business logic is preserved\n"
        f"- [ ] Vulnerability is correctly fixed\n"
        f"- [ ] All tests pass\n"
        f"- [ ] No regression introduced\n\n"
        f"---\n*Auto-generated by InvisiThreat AI Auto-Fix*\n"
    )
    try:
        pr = repo.create_pull(
            title=f"[Security Fix] #{finding_id} — {title[:60]}",
            body=pr_body,
            head=fix_branch,
            base=branch,
        )
    except GithubException as exc:
        raise HTTPException(500, f"Failed to create PR: {exc.data}")

    logger.info("PR #%d created: %s", pr.number, pr.html_url)
    # ── Notification ──────────────────────────────────────────────────
    try:
        notify_pr_merged(
            db=db,
            pr_title=f"[Security Fix] #{finding_id} — {title[:60]}",
            pr_url=pr.html_url,
            finding_id=finding_id,
        )
    except Exception as e:
        logger.warning(f"[NOTIF] notify_pr_merged failed: {e}")
    # ──────────────────────────────────────────────────────────────────
    return AutoFixResponse(
        pr_url=pr.html_url,
        pr_number=pr.number,
        branch_name=fix_branch,
        status="open",
    )



class AutoFixCapabilityResponse(BaseModel):
    """Response indicating whether a finding can be auto-fixed."""
    finding_id: int
    can_autofix: bool
    reason: str  # Explication du pourquoi
    missing_fields: list[str]  # Champs manquants
    requirements: dict  # Détails des requirements


def _can_autofix_finding(finding: dict, loader) -> AutoFixCapabilityResponse:
    import math
    fid = finding.get("id", 0)
    missing = []

    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")

    is_static = str(finding.get("static_finding", "")).lower() in ("true", "1", "yes")
    if not is_static:
        missing.append("static_finding=true")

    file_path = finding.get("file_path") or finding.get("sast_source_file_path") or ""
    file_path = "" if isinstance(file_path, float) else str(file_path).strip()
    if not file_path:
        missing.append("file_path")

    line = finding.get("line") or finding.get("sast_source_line")
    line_invalid = (
        line is None
        or (isinstance(line, float) and math.isnan(line))
        or str(line).strip() in ("", "nan")
    )
    if line_invalid:
        missing.append("line")

    repo_url_raw = finding.get("repo_url")
    repo_url = (
        ""
        if repo_url_raw is None or (isinstance(repo_url_raw, float) and math.isnan(repo_url_raw))
        else str(repo_url_raw).strip()
    )
    if not repo_url:
        missing.append("repo_url")

    logger.info(
        f"[can_autofix] id={fid} "
        f"is_static={is_static} "
        f"file_path={repr(file_path)} "
        f"line={repr(line)} "
        f"repo_url={repr(repo_url)}"
    )

    can_fix = len(missing) == 0
    reason = (
        "✅ Finding peut être autofixée"
        if can_fix
        else f"❌ Champs manquants: {', '.join(missing)}"
    )

    return AutoFixCapabilityResponse(
        finding_id=fid,
        can_autofix=can_fix,
        reason=reason,
        missing_fields=missing,
        requirements={
            "is_static":        is_static,
            "has_file_path":    bool(file_path),
            "has_line":         not line_invalid,
            "has_repo_url":     bool(repo_url),
            "has_github_token": bool(GITHUB_TOKEN),
        }
    )


@router.get("/defectdojo/findings/{finding_id}/can-autofix", response_model=AutoFixCapabilityResponse)
async def check_autofix_capability(finding_id: int) -> AutoFixCapabilityResponse:
    from server.dependencies import get_local_data_loader
    loader = get_local_data_loader()
    if loader is None or not loader.is_ready:
        raise HTTPException(503, "Data loader not ready")
    finding = loader.findings_by_id.get(finding_id)
    if not finding:
        raise HTTPException(404, f"Finding #{finding_id} not found")
    return _can_autofix_finding(finding, loader)


# ---------------------------------------------------------------------------
# Health check — verifies Ollama connectivity and lists available models
# ---------------------------------------------------------------------------


@router.get("/llm/health")
async def llm_health():
    """
    Probe the Ollama server and return its status and list of loaded models.

    Returns:
      {"status": "ok",    "models": [...], "current": "<model>"}  — server reachable
      {"status": "error", "detail": "<reason>"}                   — server unreachable
    """
    try:
        response = requests.get(
            f"{OLLAMA_URL.rstrip('/')}/api/tags",
            timeout=5,
        )
        response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        return {
            "status": "ok",
            "models": models,
            "current": OLLAMA_MODEL,
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "detail": f"Cannot connect to Ollama at {OLLAMA_URL} — verify the VM is running.",
        }
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "detail": f"Connection to Ollama at {OLLAMA_URL} timed out.",
        }
    except Exception as exc:
        return {
            "status": "error",
            "detail": str(exc),
        }