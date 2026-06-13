"""
llm_service.py
--------------
LLM service module for vulnerability analysis using a local Ollama instance.

This module replaces the previous Gemini-based implementation with a call
to a self-hosted Ollama server running on the local network. It provides
three analysis modes for security findings:
  - explanation   : summarizes what the vulnerability is and its danger
  - recommendation: provides a structured remediation plan
  - solution      : generates a concrete code-level fix

Results are cached on disk to avoid redundant LLM calls.

Configuration (environment variables):
  OLLAMA_URL          : Base URL of the Ollama server (default: http://192.168.11.170:11434)
  OLLAMA_MODEL        : Model name to use (default: deepseek-coder:6.7b)
  LLM_CACHE_PATH      : Path to the on-disk JSON cache (default: data/llm_cache.json)
  LLM_CONNECT_TIMEOUT : TCP connection timeout in seconds (default: 5)
  LLM_READ_TIMEOUT    : Response read timeout in seconds (default: 90)
  LLM_MAX_TOKENS      : Maximum tokens to generate per response (default: 900)
""" 

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all values can be overridden via environment variables
# ---------------------------------------------------------------------------

OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://192.168.11.170:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

LLM_CACHE_PATH: Path = Path(os.getenv("LLM_CACHE_PATH", "data/llm_cache.json"))

LLM_CONNECT_TIMEOUT: int = int(os.getenv("LLM_CONNECT_TIMEOUT", "5"))
LLM_READ_TIMEOUT: int = int(os.getenv("LLM_READ_TIMEOUT", "300"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "200"))

# Thread lock used to serialize all cache read/write operations
_cache_lock = threading.Lock()

# Log the active configuration at import time to make deployment issues obvious
logger.info(
    "LLM service initialized — backend: Ollama | url: %s | model: %s | "
    "connect_timeout: %ds | read_timeout: %ds | max_tokens: %d",
    OLLAMA_URL,
    OLLAMA_MODEL,
    LLM_CONNECT_TIMEOUT,
    LLM_READ_TIMEOUT,
    LLM_MAX_TOKENS,
)



import math

def _safe_str(val, default: str = "") -> str:
    """Convertit n'importe quelle valeur (NaN, None, float...) en string propre."""
    if val is None:
        return default
    try:
        if isinstance(val, float) and math.isnan(val):
            return default
    except Exception:
        pass
    return str(val).strip()

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    """Load the JSON cache from disk. Returns an empty dict on any error."""
    if LLM_CACHE_PATH.exists():
        try:
            with open(LLM_CACHE_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.warning("Failed to load LLM cache (%s) — starting empty.", exc)
    return {}


def _save_cache(cache: dict) -> None:
    """Persist the cache dict to disk, creating parent directories if needed."""
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


def _cache_key(finding: dict, mode: str) -> str:
    """
    Derive a deterministic MD5 key from the fields that uniquely identify
    a finding + analysis mode combination.
    """
    key_data = {
        "mode": mode,
        "title": finding.get("title", ""),
        "severity": finding.get("severity", ""),
        "cvss_score": finding.get("cvss_score", 0),
        "description": (finding.get("description", "") or "")[:200],
        "file_path": finding.get("file_path", ""),
        "line": finding.get("line", ""),
        "cve": finding.get("cve", ""),
        "cwe": finding.get("cwe", ""),
        "tags": sorted(finding.get("tags", [])),
        "risk_level": finding.get("risk_level", ""),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_explanation_prompt(finding: dict) -> str:
    """
    Build the prompt used for the 'explanation' mode.
    Asks the model to produce a structured JSON object describing
    the vulnerability, its impact, root cause, difficulty and priority.
    """
    title = finding.get("title", "Unknown")
    severity = finding.get("severity", "medium")
    cvss = finding.get("cvss_score", 0)
    desc = _safe_str(finding.get("description"))[:300]
    fp = finding.get("file_path", "")
    cve = finding.get("cve", "") or "N/A"
    cwe = finding.get("cwe", "") or "N/A"

    return f"""Vous êtes un ingénieur senior en sécurité applicative rédigeant un rapport de vulnérabilité détaillé pour une équipe de développement. Vous DEVEZ répondre entièrement en français. Toutes les valeurs des champs doivent être rédigées en français.
Votre objectif est d'expliquer clairement CE QU'EST la vulnérabilité, POURQUOI elle est dangereuse, et À QUEL POINT il est urgent de la corriger.
Rédigez en français clair et professionnel. Soyez spécifique et concret. Évitez les phrases vagues ou génériques.

Détails de la vulnérabilité :
- Titre : {title}
- Sévérité : {severity} | Score CVSS : {cvss}
- CVE : {cve} | CWE : {cwe}
- Fichier affecté : {fp}
- Description : {desc}

Répondez avec un seul objet JSON en utilisant EXACTEMENT ces clés :

"summary"
  2 à 3 phrases. Expliquer ce qu'est cette vulnérabilité, où elle existe dans le code, et ce qui la rend dangereuse.
  Être spécifique à la vulnérabilité ci-dessus — pas générique.

"impact"
  3 à 4 phrases. Décrire concrètement ce qu'un attaquant peut faire s'il exploite cette faille.
  Mentionner les données, systèmes ou utilisateurs affectés. Inclure les conséquences métier potentielles (fuite de données, indisponibilité, risque légal, etc.).

"root_cause"
  2 phrases. Expliquer la raison technique exacte pour laquelle cette vulnérabilité existe.
  Être spécifique sur le pattern de code ou la mauvaise pratique responsable.

"exploitation_difficulty"
  Un parmi : Facile / Moyen / Difficile
  Suivi d'un tiret et d'1 phrase expliquant pourquoi (outils nécessaires, niveau requis, prérequis).
  Exemple : "Facile — Aucun outil spécial requis ; un payload basique suffit à déclencher la vulnérabilité."

"priority_note"
  Commencer par : Immédiat / 48h / Semaine / Mois
  Suivi de 2 phrases : expliquer le niveau d'urgence et ce qui pourrait arriver si non corrigé.
  Exemple : "Immédiat — Cette vulnérabilité est activement exploitée et ne nécessite aucune authentification. La laisser non corrigée même 24h augmente significativement le risque de compromission."

Produire uniquement l'objet JSON brut. Pas de balises markdown, pas de texte avant ou après.

Exemple de format de sortie :
{{
  "summary": "Cette vulnérabilité XSS existe sur la page de profil utilisateur où les données non filtrées sont injectées directement dans le DOM HTML. Elle permet à un attaquant d'exécuter du JavaScript arbitraire dans le navigateur d'autres utilisateurs. La page affectée est accessible à tous les utilisateurs authentifiés, ce qui amplifie la surface d'attaque.",
  "impact": "Un attaquant peut voler les cookies de session et détourner des comptes authentifiés. Des scripts malveillants peuvent effectuer des actions au nom des victimes, comme changer leur mot de passe ou exfiltrer des données personnelles. Si un compte administrateur est compromis, l'attaquant obtient des privilèges élevés sur toute l'application. Cela peut conduire à une compromission massive de comptes utilisateurs.",
  "root_cause": "L'application insère les données utilisateur directement dans le template HTML sans encodage des caractères spéciaux. Le développeur a utilisé innerHTML au lieu de textContent, permettant au navigateur d'interpréter les balises HTML et les scripts injectés.",
  "exploitation_difficulty": "Facile — Un attaquant n'a besoin que d'enregistrer un script malveillant dans son champ de profil ; aucun outil spécial ni contournement d'authentification n'est requis.",
  "priority_note": "Immédiat — Les vulnérabilités XSS stockées peuvent affecter chaque utilisateur visitant la page infectée, y compris les administrateurs. Tout délai dans la correction augmente le risque de détournement de sessions en masse."
}}

Rédigez maintenant le JSON pour la vulnérabilité décrite ci-dessus :"""


def _build_recommendation_prompt(finding: dict) -> str:
    """
    Build the prompt used for the 'recommendation' mode.
    Asks the model for a structured remediation plan with concrete steps,
    references, verification method and long-term prevention guidance.
    """
    title = finding.get("title", "Unknown")
    severity = finding.get("severity", "medium")
    cvss = finding.get("cvss_score", 0)
    desc = _safe_str(finding.get("description"))[:300]

    fp = finding.get("file_path", "")
    cve = finding.get("cve", "") or "N/A"
    cwe = finding.get("cwe", "") or "N/A"

    return f"""Vous êtes un ingénieur senior en sécurité applicative rédigeant un plan de remédiation détaillé pour une équipe de développement. Vous DEVEZ répondre entièrement en français. Toutes les valeurs des champs doivent être rédigées en français.
Votre objectif est de fournir des étapes claires, concrètes et complètes pour corriger cette vulnérabilité.
Chaque étape doit être suffisamment spécifique pour qu'un développeur puisse l'implémenter directement.
Ne PAS écrire des conseils vagues comme "assainir les entrées" — expliquer exactement comment et pourquoi.

Détails de la vulnérabilité :
- Titre : {title}
- Sévérité : {severity} | Score CVSS : {cvss}
- CVE : {cve} | CWE : {cwe}
- Fichier affecté : {fp}
- Description : {desc}

Répondez avec un seul objet JSON en utilisant EXACTEMENT ces clés :

"title"
  Un titre clair et spécifique pour ce plan de remédiation.
  Pas seulement le nom de la vulnérabilité — décrire l'approche de correction.
  Exemple : "Correction du XSS stocké sur la page de profil par encodage de sortie"

"recommendations"
  Tableau de 4 à 6 étapes de remédiation sous forme de chaînes.
  Chaque étape doit :
  - Commencer par un verbe d'action (Remplacer, Utiliser, Ajouter, Configurer, Valider, Appliquer, etc.)
  - Être spécifique à ce type de vulnérabilité
  - Inclure une courte explication du POURQUOI cette étape est importante

"references"
  Tableau de 2 à 3 URLs faisant autorité, pertinentes pour ce type de vulnérabilité.
  Utiliser OWASP, NIST, CWE, CVE, ou les avis de sécurité des éditeurs.

"verification"
  2 phrases. Expliquer comment vérifier que le correctif a été correctement appliqué.
  Inclure des méthodes automatisées (scanner, vérification CI) et manuelles (revue de code, test de pénétration).

"prevention"
  2 phrases. Expliquer comment prévenir cette classe de vulnérabilité dans toute la base de code à l'avenir.
  Se concentrer sur les améliorations de processus, les outils (SAST/DAST en CI), ou les décisions d'architecture.

Produire uniquement l'objet JSON brut. Pas de balises markdown, pas de texte avant ou après.

Exemple de format de sortie :
{{
  "title": "Correction du XSS stocké sur la page de profil par encodage de sortie et CSP",
  "recommendations": [
    "Remplacer tous les usages de innerHTML par textContent ou utiliser DOMPurify pour assainir le HTML avant le rendu, empêchant le navigateur d'interpréter les balises script injectées.",
    "Appliquer un encodage de sortie côté serveur sur tous les champs utilisateur via une bibliothèque comme html.escape() en Python avant d'insérer les données dans les templates HTML.",
    "Mettre en place un en-tête Content Security Policy strict avec 'script-src self' pour bloquer l'exécution de scripts inline injectés même si l'encodage est manqué à un endroit.",
    "Ajouter une validation serveur pour rejeter les entrées contenant des balises HTML ou des gestionnaires d'événements JavaScript dans les champs qui ne doivent pas contenir de balisage.",
    "Auditer tous les templates pour détecter d'autres instances de données utilisateur non encodées et appliquer le même correctif de manière cohérente dans toute la base de code."
  ],
  "references": [
    "https://owasp.org/www-community/attacks/xss/",
    "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    "https://cwe.mitre.org/data/definitions/79.html"
  ],
  "verification": "Relancer le scanner SAST et l'outil DAST sur l'endpoint corrigé et confirmer qu'aucune vulnérabilité XSS ne subsiste. Tester manuellement en tentant d'enregistrer un payload <script>alert(1)</script> et vérifier qu'il est affiché en texte brut.",
  "prevention": "Ajouter une règle SAST dans le pipeline CI/CD qui bloque les pull requests contenant des usages de innerHTML ou des variables de template non encodées. Exiger une formation OWASP XSS pour tous les développeurs et intégrer l'encodage de sortie dans la checklist de revue de code."
}}

Rédigez maintenant le JSON pour la vulnérabilité décrite ci-dessus :"""


def _build_solution_prompt(finding: dict) -> str:
    """
    Build the prompt used for the 'solution' mode.
    Asks the model for a minimal, concrete code-level fix with a before/after
    snippet, a short explanation, and a confidence score.
    """
    title      = _safe_str(finding.get("title"), "Unknown")
    severity   = _safe_str(finding.get("severity"), "medium")
    cvss       = finding.get("cvss_score", 0)
    desc       = _safe_str(finding.get("description"))[:400]
    mitigation = _safe_str(finding.get("mitigation"))[:300]
    fp         = _safe_str(finding.get("file_path")) or _safe_str(finding.get("sast_source_file_path"))
    line       = finding.get("line") or finding.get("sast_source_line", "")
    cve        = _safe_str(finding.get("cve")) or "N/A"
    cwe        = _safe_str(finding.get("cwe")) or "N/A"
    component  = _safe_str(finding.get("component_name"))
    comp_ver   = _safe_str(finding.get("component_version"))
    sast_src   = _safe_str(finding.get("sast_source_object"))
    sast_sink  = _safe_str(finding.get("sast_sink_object"))
    is_static  = str(finding.get("static_finding", "")).lower() in ("true", "1", "yes")

    # Build optional context block based on available finding metadata
    context_block = ""
    if is_static and fp:
        context_block = f"- Fichier affecté : {fp}" + (f" (ligne {line})" if line else "")
    elif component:
        context_block = f"- Composant vulnérable : {component} {comp_ver}".strip()
    if sast_src:
        context_block += f"\n- Objet source : {sast_src}"
    if sast_sink:
        context_block += f"\n- Objet sink : {sast_sink}"

    return f"""Vous êtes un ingénieur senior en sécurité applicative fournissant un correctif de code exact pour une vulnérabilité. Vous DEVEZ répondre entièrement en français. Toutes les valeurs des champs, à l'exception des extraits de code, doivent être rédigées en français.
Votre objectif est de produire une correction de code minimale, précise et immédiatement applicable.
Soyez spécifique — N'écrivez PAS de conseils génériques. Écrivez du vrai code.

Détails de la vulnérabilité :
- Titre : {title}
- Sévérité : {severity} | Score CVSS : {cvss}
- CVE : {cve} | CWE : {cwe}
{context_block}
- Description : {desc}
- Indication de remédiation : {mitigation}

Répondez avec un seul objet JSON en utilisant EXACTEMENT ces clés :

"vulnerable_snippet"
  Le pattern de code vulnérable exact (2 à 5 lignes). Du vrai code, pas une description.

"fixed_snippet"
  La version corrigée du même code (2 à 5 lignes). Modifications minimales uniquement.
  Ne PAS réécrire le fichier entier — corriger uniquement la partie vulnérable.

"explanation"
  2 à 3 phrases. Expliquer exactement ce qui a été modifié et pourquoi cela corrige la vulnérabilité.
  Être spécifique sur la propriété de sécurité ajoutée ou restaurée.

"confidence"
  Un float entre 0.0 et 1.0 représentant la confiance dans ce correctif.
  Utiliser 0.90+ si file_path et line sont disponibles. Utiliser 0.65-0.75 pour des correctifs génériques.

Produire uniquement l'objet JSON brut. Pas de balises markdown, pas de texte avant ou après.

Exemple de format de sortie :
{{
  "vulnerable_snippet": "cursor.execute(\\"SELECT * FROM users WHERE id = \\" + user_id)",
  "fixed_snippet": "cursor.execute(\\"SELECT * FROM users WHERE id = %s\\", (user_id,))",
  "explanation": "Le code original concaténait directement les entrées utilisateur dans la requête SQL, permettant une injection SQL. Le correctif utilise une requête paramétrée qui sépare les données du code, empêchant l'exécution de tout SQL injecté.",
  "confidence": 0.92
}}

Rédigez maintenant le JSON pour la vulnérabilité décrite ci-dessus :"""


# ---------------------------------------------------------------------------
# Ollama API call
# ---------------------------------------------------------------------------


def _call_ollama(prompt: str) -> Optional[str]:
    """
    Send a generation request to the local Ollama server using the
    /api/generate endpoint (non-streaming, single-shot completion).

    Returns the raw text content on success, or None on any error.

    Checklist of things to verify if this call fails:
      1. Ollama is running on the VM:      `systemctl status ollama`
      2. The model is pulled:              `ollama list`  (should show OLLAMA_MODEL)
      3. The VM is reachable from here:    `curl http://192.168.11.170:11434/api/tags`
      4. Firewall allows port 11434 from this host
      5. OLLAMA_URL env var is set correctly if the default IP changed
    """
    endpoint = f"{OLLAMA_URL.rstrip('/')}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,  # Receive the full response in one HTTP reply
        "options": {
            "temperature": 0.2,
            "num_predict": LLM_MAX_TOKENS,
            "num_ctx": 1024,
            "top_k": 10,
            "top_p": 0.85,
        },
    }

    logger.debug(
        "Sending request to Ollama — endpoint: %s | model: %s | prompt_length: %d chars",
        endpoint,
        OLLAMA_MODEL,
        len(prompt),
    )

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        # Ollama /api/generate returns {"response": "<text>", "done": true, ...}
        text = data.get("response", "").strip()

        if not text:
            logger.error(
                "Ollama returned an empty 'response' field. "
                "Full payload keys: %s",
                list(data.keys()),
            )
            return None

        logger.debug("Ollama raw response preview: %.400s", text)
        return text

    except requests.exceptions.ConnectionError as exc:
        logger.error(
            "Could not connect to Ollama at %s. "
            "Verify the VM is running and the URL is correct. Error: %s",
            OLLAMA_URL,
            exc,
        )
    except requests.exceptions.Timeout:
        logger.error(
            "Ollama request timed out after %ds read timeout. "
            "Consider increasing LLM_READ_TIMEOUT or using a smaller model.",
            LLM_READ_TIMEOUT,
        )
    except requests.exceptions.HTTPError as exc:
        logger.error(
            "Ollama returned HTTP %d: %s",
            exc.response.status_code,
            exc.response.text[:300],
        )
    except Exception as exc:
        logger.error("Unexpected error calling Ollama: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Robust JSON parser
# ---------------------------------------------------------------------------

# Substrings that indicate the model echoed back the prompt template
# instead of filling it with real content.
PLACEHOLDER_PATTERNS = [
    "one sentence",
    "two sentence",
    "easy/medium/hard",
    "immediate/48h",
    "week/month",
    "explain the",
    "summary of the",
    "technical cause here",
    "how to verify",
    "prevention measure",
    "short title",
    # Équivalents français
    "une phrase",
    "deux phrases",
    "facile/moyen/difficile",
    "immédiat/48h",
    "semaine/mois",
    "expliquer le",
    "résumé de la",
    "cause technique ici",
    "comment vérifier",
    "mesure de prévention",
    "titre court",
]


def _is_placeholder(value: str) -> bool:
    """Return True if the string looks like an unfilled template placeholder."""
    if not isinstance(value, str):
        return False
    lower = value.lower()
    return any(pattern in lower for pattern in PLACEHOLDER_PATTERNS)


def _parse_json_response(raw: str, fallback: dict) -> dict:
    """
    Extract and parse a JSON object from the raw LLM output.

    Handles common model output issues:
      - Markdown code fences (```json ... ```)
      - Leading/trailing prose before/after the JSON object
      - List values for scalar fields (joined into a single string)
      - Unfilled template placeholders (triggers fallback)

    Returns the parsed dict or the provided fallback dict on failure.
    """
    import re

    if not raw:
        logger.warning("Empty raw response — returning fallback.")
        return fallback

    text = raw.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Isolate the outermost JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        logger.warning(
            "No JSON object found in LLM response — returning fallback. "
            "Raw preview: %.200s",
            raw,
        )
        return fallback

    text = text[start:end]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "JSON decode error: %s — raw preview: %.200s — returning fallback.",
            exc,
            text,
        )
        return fallback

    # Normalize unexpected dict or list values for scalar fields
    for key in list(result.keys()):
        val = result[key]
        if isinstance(val, dict):
            # Flatten dict values into a space-joined string
            result[key] = " ".join(str(v) for v in val.values() if v)
        elif isinstance(val, list) and key not in ("recommendations", "references"):
            # Flatten list values for non-array fields
            result[key] = " ".join(str(v) for v in val if v)

    # Detect and reject responses that still contain template placeholders
    has_placeholders = any(
        _is_placeholder(v)
        for k, v in result.items()
        if k not in ("recommendations", "references")
    )
    if has_placeholders:
        logger.warning(
            "LLM response contains unfilled template placeholders — returning fallback. "
            "Keys present: %s",
            list(result.keys()),
        )
        return fallback

    return result


# ---------------------------------------------------------------------------
# Corrupted cache cleanup
# ---------------------------------------------------------------------------


def clean_corrupted_cache() -> int:
    """
    Scan the on-disk cache and remove entries that contain either:
      - Nested dict values (malformed structure from an older model output)
      - Unfilled template placeholder strings

    Returns the number of entries removed.
    """

    def _is_corrupted(entry: dict) -> bool:
        for key, val in entry.items():
            if key in ("from_cache", "_fallback"):
                continue
            if isinstance(val, dict):
                return True
            if _is_placeholder(str(val)):
                return True
        return False

    with _cache_lock:
        cache = _load_cache()
        original_size = len(cache)
        cleaned = {k: v for k, v in cache.items() if not _is_corrupted(v)}
        removed = original_size - len(cleaned)
        if removed > 0:
            _save_cache(cleaned)
            logger.info(
                "Cache cleanup complete — removed %d corrupted entries out of %d total.",
                removed,
                original_size,
            )
        else:
            logger.info(
                "Cache is clean — %d entries inspected, none corrupted.",
                original_size,
            )
    return removed


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def explain_finding(
    finding: dict,
    use_cache: bool = True,
    mode: str = "explanation",
) -> dict:
    """
    Analyze a security finding using the local Ollama LLM.

    Parameters
    ----------
    finding   : dict   Normalized finding dict from the scanner.
    use_cache : bool   If True, read from and write to the on-disk JSON cache.
    mode      : str    One of 'explanation', 'recommendation', or 'solution'.

    Returns
    -------
    dict  The structured analysis result. Always contains a 'from_cache' key
          (bool) and a '_fallback' key (bool) indicating whether the LLM
          response was used or the static fallback was returned instead.
    """
    if mode not in ("explanation", "recommendation", "solution"):
        logger.warning(
            "Unknown mode '%s' — defaulting to 'explanation'.", mode
        )
        mode = "explanation"

    title = finding.get("title", "unknown")
    cache_key = _cache_key(finding, mode)

    # --- Cache lookup ---
    if use_cache:
        with _cache_lock:
            cache = _load_cache()
        if cache_key in cache:
            logger.info(
                "Cache hit [%s] — title: %s | key: %.8s",
                mode,
                title,
                cache_key,
            )
            cached = cache[cache_key]
            cached["from_cache"] = True
            return cached

    # --- Build prompt and define static fallback ---
    if mode == "explanation":
        prompt = _build_explanation_prompt(finding)
        fallback: dict = {
            "summary": (
                f"La vulnérabilité '{title}' a été détectée dans la base de code et nécessite une révision. "
                f"Elle est classifiée avec une sévérité {finding.get('severity', 'inconnue')} et un score CVSS "
                f"de {finding.get('cvss_score', 'N/A')}. "
                "Une revue manuelle de sécurité du fichier affecté est fortement recommandée."
            ),
            "impact": (
                "Cette vulnérabilité peut permettre à un attaquant de compromettre la confidentialité, "
                "l'intégrité ou la disponibilité du système affecté. "
                "Des données sensibles, des comptes utilisateurs ou des ressources système critiques pourraient être exposés ou manipulés. "
                "L'étendue de l'impact dépend du contexte de déploiement et du niveau d'accès de l'attaquant. "
                "Une évaluation manuelle approfondie est nécessaire pour déterminer l'impact complet sur l'application."
            ),
            "root_cause": (
                "La cause racine n'a pas pu être déterminée automatiquement pour ce constat. "
                "Veuillez consulter la documentation de la vulnérabilité et effectuer une revue manuelle du code "
                "du fichier affecté pour identifier le pattern de code responsable."
            ),
            "exploitation_difficulty": (
                "Moyen — La difficulté n'a pas pu être évaluée automatiquement. "
                "Supposez qu'un attaquant motivé pourrait exploiter cette faille avec un effort modéré et des outils standard."
            ),
            "priority_note": (
                "Immédiat — Ce constat de sévérité critique ou élevée doit être traité dès que possible "
                "pour minimiser la fenêtre d'exposition."
                if str(finding.get("severity", "")).lower() in ("critical", "high")
                else (
                    "Semaine — Ce constat de sévérité moyenne doit être planifié pour remédiation "
                    "dans le sprint de développement en cours."
                    if str(finding.get("severity", "")).lower() == "medium"
                    else "Mois — Ce constat de faible sévérité doit être suivi dans le backlog et "
                    "résolu lors d'un prochain cycle de release."
                )
            ),
        }

    elif mode == "recommendation":
        prompt = _build_recommendation_prompt(finding)
        fallback = {
            "title": f"Plan de remédiation pour : {title}",
            "recommendations": [
                "Examiner le fichier affecté et identifier tous les emplacements où ce pattern de vulnérabilité apparaît avant toute modification, afin d'assurer un correctif complet et cohérent.",
                "Appliquer le correctif recommandé par l'OWASP pour cette classe de vulnérabilité, en suivant les directives de codage sécurisé spécifiques à votre langage et framework.",
                "Écrire des tests unitaires et d'intégration couvrant les cas d'usage normaux et les entrées adversariales liées à ce type de vulnérabilité pour prévenir les régressions futures.",
                "Relancer le scanner de sécurité après application du correctif pour confirmer que le constat n'apparaît plus et qu'aucun nouveau problème n'a été introduit.",
                "Documenter le correctif dans le journal de sécurité interne afin que l'équipe puisse s'y référer pour des vulnérabilités similaires à l'avenir.",
            ],
            "references": [
                "https://owasp.org/www-project-top-ten/",
                "https://cwe.mitre.org/",
            ],
            "verification": (
                "Relancer les scanners SAST et DAST sur le code corrigé et confirmer l'absence de constats pour cette classe de vulnérabilité. "
                "Effectuer une revue de code par les pairs pour valider la conformité aux standards de codage sécurisé et réaliser un test de pénétration ciblé sur l'endpoint affecté."
            ),
            "prevention": (
                "Ajouter une règle SAST automatisée dans le pipeline CI/CD pour détecter cette classe de vulnérabilité dans les futures pull requests et bloquer les fusions jusqu'à résolution. "
                "Intégrer ce type de vulnérabilité dans la prochaine formation de sensibilisation à la sécurité des développeurs et imposer une checklist de sécurité obligatoire lors des revues de code."
            ),
        }

    else:  # mode == "solution"
        prompt = _build_solution_prompt(finding)
        safe_mitigation = _safe_str(finding.get("mitigation"))
        mitigation_text = safe_mitigation if safe_mitigation else "directives OWASP"
        fallback = {
            "vulnerable_snippet": (
                f"# Pattern vulnérable pour : {title}\n"
                "# Code exact non disponible — vérifiez les champs file_path et static_finding."
            ),
            "fixed_snippet": (
                "# Appliquer la remédiation décrite dans le constat.\n"
                f"# Référence : {mitigation_text[:200]}"
            ),
            "explanation": (
                f"Il s'agit d'une vulnérabilité de sévérité {finding.get('severity', 'inconnue')} ({title}). "
                "Le correctif de code automatique n'a pas pu être généré — le chemin du fichier ou les informations "
                "de constat statique sont peut-être manquants. Veuillez appliquer la remédiation manuellement."
            ),
            "confidence": 0.40,
        }

    # --- Call the local Ollama instance ---
    logger.info(
        "Calling Ollama [%s] — model: %s | title: %s",
        mode,
        OLLAMA_MODEL,
        title,
    )
    raw = _call_ollama(prompt)

    # --- Parse and cache the result ---
    if raw:
        result = _parse_json_response(raw, fallback)
        result["from_cache"] = False
        result["_fallback"] = result is fallback

        if use_cache and not result.get("_fallback"):
            with _cache_lock:
                cache = _load_cache()
                cache[cache_key] = result
                _save_cache(cache)
            logger.debug(
                "Result cached [%s] — key: %.8s | title: %s",
                mode,
                cache_key,
                title,
            )

        return result

    # --- LLM call failed entirely — return static fallback ---
    logger.warning(
        "Ollama call failed [%s] — returning static fallback for: %s",
        mode,
        title,
    )
    fallback["from_cache"] = False
    fallback["_fallback"] = True
    return fallback