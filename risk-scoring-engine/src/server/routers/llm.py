"""
routers/llm.py — Endpoints /explain/llm, /recommend/llm, /llm/health
et les variantes /defectdojo/findings/{id}/explain|recommend|solution|autofix.
"""
import logging
import os

import requests
from fastapi import APIRouter, HTTPException
from github import Github, GithubException
from pydantic import BaseModel
from typing import Optional

from server.routers.findings import get_finding_by_id
from server.schemas import (
    LLMExplanationResponse,
    LLMRecommendationResponse,
    LLMRequest,
)

logger = logging.getLogger("invisithreat.llm")

router = APIRouter(tags=["LLM"])

GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN", "")
GITHUB_BRANCH  = os.getenv("GITHUB_DEFAULT_BRANCH", "main")

try:
    from llm_service import explain_finding as _llm_explain
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    def _llm_explain(*args, **kwargs):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Schémas nouveaux
# ══════════════════════════════════════════════════════════════════════════════

class LLMSolutionResponse(BaseModel):
    finding_id:         int
    vulnerable_snippet: Optional[str] = None
    fixed_snippet:      Optional[str] = None
    explanation:        str
    confidence:         float
    file_path:          Optional[str] = None
    line:               Optional[int] = None
    has_file:           bool = False
    from_cache:         bool = False

class AutoFixResponse(BaseModel):
    pr_url:      str
    pr_number:   int
    branch_name: str
    status:      str


# ══════════════════════════════════════════════════════════════════════════════
# Fallbacks
# ══════════════════════════════════════════════════════════════════════════════

def _explanation_fallback(fid, title, severity, cvss) -> LLMExplanationResponse:
    return LLMExplanationResponse(
        finding_id              = fid,
        summary                 = f"Vulnérabilité {severity.upper()} : {title}",
        impact                  = f"Sévérité {severity} (CVSS: {cvss or 'N/A'}) — risque de sécurité.",
        root_cause              = "Analyse automatique indisponible.",
        exploitation_difficulty = "Moyenne",
        priority_note           = (
            "Immédiat" if severity.lower() in ("critical", "high")
            else "Semaine" if severity.lower() == "medium"
            else "Mois"
        ),
    )


def _recommendation_fallback(fid) -> LLMRecommendationResponse:
    return LLMRecommendationResponse(
        finding_id      = fid,
        title           = "Recommandations générales",
        recommendations = [
            "Analyser le code source autour du fichier concerné",
            "Appliquer les correctifs recommandés par OWASP",
            "Tester la correction avec un scanner de sécurité",
            "Re-scanner après application du correctif",
        ],
        references   = ["https://owasp.org/www-project-top-ten/"],
        verification = "Re-lancer le scan et vérifier l'absence du finding",
        prevention   = "Former les développeurs aux bonnes pratiques de sécurité",
    )


def _solution_fallback(fid, title, severity) -> LLMSolutionResponse:
    return LLMSolutionResponse(
        finding_id         = fid,
        vulnerable_snippet = None,
        fixed_snippet      = None,
        explanation        = (
            f"Correction automatique indisponible pour '{title}' ({severity}). "
            "Vérifiez que le finding est de type SAST avec file_path et line renseignés."
        ),
        confidence = 0.0,
        has_file   = False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers GitHub
# ══════════════════════════════════════════════════════════════════════════════

def _parse_github_url(url: str) -> tuple[str, str]:
    url = url.rstrip("/").replace(".git", "")
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError(f"URL GitHub invalide : {url}")
    return parts[-2], parts[-1]


def _get_finding_dict(finding_id: int) -> dict:
    """Récupère le finding brut depuis le data_loader."""
    try:
        from server.data_loader import get_local_loader
        loader = get_local_loader()
    except ImportError:
        try:
            from api import local_data_loader as loader
        except ImportError:
            raise HTTPException(503, "Data loader non disponible")
    if not loader or not loader.is_ready:
        raise HTTPException(503, "Données non chargées")
    f = loader.findings_by_id.get(finding_id)
    if not f:
        raise HTTPException(404, f"Finding #{finding_id} introuvable")
    return f


# ══════════════════════════════════════════════════════════════════════════════
# Endpoints existants — inchangés
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/explain/llm", response_model=LLMExplanationResponse)
async def explain_with_llm(request: LLMRequest) -> LLMExplanationResponse:
    result = _llm_explain(request.model_dump(), use_cache=True, mode="explanation") if _LLM_AVAILABLE else None
    if not result or not result.get("summary"):
        return _explanation_fallback(
            request.finding_id, request.title, request.severity, request.cvss_score or 0
        )
    return LLMExplanationResponse(
        finding_id              = request.finding_id,
        summary                 = result["summary"],
        impact                  = result.get("impact", ""),
        root_cause              = result.get("root_cause"),
        exploitation_difficulty = result.get("exploitation_difficulty"),
        priority_note           = result.get("priority_note", ""),
        from_cache              = result.get("from_cache", False),
    )


@router.post("/recommend/llm", response_model=LLMRecommendationResponse)
async def recommend_with_llm(request: LLMRequest) -> LLMRecommendationResponse:
    result = _llm_explain(request.model_dump(), use_cache=True, mode="recommendation") if _LLM_AVAILABLE else None
    if not result or not result.get("recommendations"):
        return _recommendation_fallback(request.finding_id)
    return LLMRecommendationResponse(
        finding_id      = request.finding_id,
        title           = result.get("title", "Recommandations IA"),
        recommendations = result.get("recommendations", []),
        references      = result.get("references", []),
        verification    = result.get("verification"),
        prevention      = result.get("prevention"),
        from_cache      = result.get("from_cache", False),
    )


@router.get("/defectdojo/findings/{finding_id}/explain", response_model=LLMExplanationResponse)
async def explain_finding_from_dojo(finding_id: int) -> LLMExplanationResponse:
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id  = finding_id,
        title       = finding.title,
        severity    = finding.severity,
        cvss_score  = finding.cvss_score,
        description = finding.description or "",
        file_path   = finding.file_path or "",
        tags        = finding.tags or [],
        risk_level  = finding.risk_level or "",
    )
    return await explain_with_llm(req)


@router.get("/defectdojo/findings/{finding_id}/recommend", response_model=LLMRecommendationResponse)
async def recommend_finding_from_dojo(finding_id: int) -> LLMRecommendationResponse:
    finding = await get_finding_by_id(finding_id)
    req = LLMRequest(
        finding_id  = finding_id,
        title       = finding.title,
        severity    = finding.severity,
        cvss_score  = finding.cvss_score,
        description = finding.description or "",
        file_path   = finding.file_path or "",
        tags        = finding.tags or [],
        risk_level  = finding.risk_level or "",
    )
    return await recommend_with_llm(req)


# ══════════════════════════════════════════════════════════════════════════════
# NOUVEAU — Solution IA
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/defectdojo/findings/{finding_id}/solution", response_model=LLMSolutionResponse)
async def solution_finding_from_dojo(finding_id: int) -> LLMSolutionResponse:
    """
    Génère un fix de code IA pour un finding SAST.
    Utilise file_path, line, description, mitigation depuis findings_clean.csv.
    """
    finding = _get_finding_dict(finding_id)

    file_path  = (finding.get("file_path") or finding.get("sast_source_file_path") or "").strip()
    line_raw   = finding.get("line") or finding.get("sast_source_line")
    line_num   = int(line_raw) if line_raw and str(line_raw) not in ("", "nan") else None
    is_static  = str(finding.get("static_finding", "")).lower() in ("true", "1", "yes")
    has_file   = bool(file_path and line_num and is_static)

    if not _LLM_AVAILABLE:
        return _solution_fallback(finding_id, finding.get("title", ""), finding.get("severity", ""))

    result = _llm_explain(finding, use_cache=True, mode="solution")

    if not result or not result.get("fixed_snippet"):
        return _solution_fallback(finding_id, finding.get("title", ""), finding.get("severity", ""))

    return LLMSolutionResponse(
        finding_id         = finding_id,
        vulnerable_snippet = result.get("vulnerable_snippet"),
        fixed_snippet      = result.get("fixed_snippet"),
        explanation        = result.get("explanation", ""),
        confidence         = float(result.get("confidence", 0.5)),
        file_path          = file_path or None,
        line               = line_num,
        has_file           = has_file,
        from_cache         = result.get("from_cache", False),
    )


# ══════════════════════════════════════════════════════════════════════════════
# NOUVEAU — Auto-Fix → Pull Request GitHub
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/defectdojo/findings/{finding_id}/autofix", response_model=AutoFixResponse)
async def autofix_finding(finding_id: int) -> AutoFixResponse:
    """
    1. Récupère la solution IA (fixed_snippet)
    2. Récupère repo_url + branch depuis le finding (propagés depuis l'engagement)
    3. Crée une branche fix/invisithreat-{id}
    4. Remplace les lignes vulnérables + commit
    5. Ouvre une Pull Request
    """
    if not GITHUB_TOKEN:
        raise HTTPException(503, "GITHUB_TOKEN non configuré dans le .env")

    # ── 1. Récupérer le finding complet ──────────────────────────────────────
    finding   = _get_finding_dict(finding_id)
    title     = finding.get("title", "")
    file_path = (finding.get("file_path") or finding.get("sast_source_file_path") or "").strip()
    line_raw  = finding.get("line") or finding.get("sast_source_line")
    line_num  = int(line_raw) if line_raw and str(line_raw) not in ("", "nan") else None
    repo_url  = (finding.get("repo_url") or "").strip()
    branch    = (finding.get("branch") or GITHUB_BRANCH).strip()

    if not file_path or not line_num:
        raise HTTPException(400, "file_path et line sont requis pour l'auto-fix (finding SAST uniquement)")
    if not repo_url:
        raise HTTPException(400, "repo_url absent — vérifiez source_code_management_uri dans l'engagement DefectDojo")

    # ── 2. Générer la solution IA ─────────────────────────────────────────────
    if not _LLM_AVAILABLE:
        raise HTTPException(503, "Service LLM non disponible")

    result = _llm_explain(finding, use_cache=True, mode="solution")
    if not result or not result.get("fixed_snippet"):
        raise HTTPException(422, "La solution IA n'a pas pu générer un fix pour ce finding")

    fixed_snippet = result["fixed_snippet"]

    # ── 3. GitHub : accès repo ────────────────────────────────────────────────
    try:
        g            = Github(GITHUB_TOKEN)
        owner, rname = _parse_github_url(repo_url)
        repo         = g.get_repo(f"{owner}/{rname}")
    except GithubException as e:
        raise HTTPException(400, f"Accès repo GitHub impossible : {e.data}")
    except ValueError as e:
        raise HTTPException(400, str(e))

    fix_branch = f"fix/invisithreat-{finding_id}"

    # ── 4. Récupérer le fichier original ─────────────────────────────────────
    try:
        file_obj = repo.get_contents(file_path, ref=branch)
        original = file_obj.decoded_content.decode("utf-8")
        file_sha = file_obj.sha
    except GithubException:
        raise HTTPException(404, f"Fichier introuvable sur GitHub : {file_path} (branche {branch})")

    # ── 5. Appliquer le fix sur la ligne ─────────────────────────────────────
    lines      = original.split("\n")
    idx        = max(0, line_num - 1)
    fixed_lines = fixed_snippet.split("\n")
    new_content = "\n".join(lines[:idx] + fixed_lines + lines[idx + 1:])

    if new_content.strip() == original.strip():
        raise HTTPException(400, "Le code corrigé est identique à l'original")

    # ── 6. Créer la branche fix ───────────────────────────────────────────────
    try:
        base_sha = repo.get_branch(branch).commit.sha
        try:
            repo.get_git_ref(f"heads/{fix_branch}").delete()
        except GithubException:
            pass
        repo.create_git_ref(ref=f"refs/heads/{fix_branch}", sha=base_sha)
    except GithubException as e:
        raise HTTPException(500, f"Erreur création branche : {e.data}")

    # ── 7. Commit ─────────────────────────────────────────────────────────────
    commit_msg = (
        f"fix(security): auto-fix finding #{finding_id}\n\n"
        f"Title : {title}\n"
        f"File  : {file_path}\n"
        f"Line  : {line_num}\n"
        f"Tool  : InvisiThreat AI (Gemini)"
    )
    try:
        repo.update_file(
            path    = file_path,
            message = commit_msg,
            content = new_content,
            sha     = file_sha,
            branch  = fix_branch,
        )
    except GithubException as e:
        raise HTTPException(500, f"Erreur commit : {e.data}")

    # ── 8. Pull Request ───────────────────────────────────────────────────────
    pr_body = f"""## 🔐 Security Auto-Fix — InvisiThreat

### Finding #{finding_id} — {title}

| Champ | Valeur |
|---|---|
| Fichier | `{file_path}` |
| Ligne | `{line_num}` |
| Branche | `{fix_branch}` → `{branch}` |
| Modèle IA | Gemini |

### Checklist review
- [ ] La logique métier est préservée
- [ ] La vulnérabilité est corrigée
- [ ] Les tests passent
- [ ] Pas de régression introduite

---
*Auto-généré par InvisiThreat AI Auto-Fix*
"""
    try:
        pr = repo.create_pull(
            title = f"[Security Fix] #{finding_id} — {title[:60]}",
            body  = pr_body,
            head  = fix_branch,
            base  = branch,
        )
    except GithubException as e:
        raise HTTPException(500, f"Erreur création PR : {e.data}")

    logger.info(f"✅ PR #{pr.number} créée : {pr.html_url}")

    return AutoFixResponse(
        pr_url      = pr.html_url,
        pr_number   = pr.number,
        branch_name = fix_branch,
        status      = "open",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Health check LLM — inchangé
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/llm/health")
async def llm_health():
    ollama_url = os.getenv("OLLAMA_URL", "http://192.168.11.170:11434")
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return {
            "status":  "ok",
            "models":  [m["name"] for m in r.json().get("models", [])],
            "current": os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b"),
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}