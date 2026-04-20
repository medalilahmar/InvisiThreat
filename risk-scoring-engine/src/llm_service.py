import json
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional


import requests

logger = logging.getLogger(__name__)


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("⚠️ GEMINI_API_KEY non définie — le service LLM utilisera le fallback uniquement")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

LLM_CACHE_PATH   = Path("data/llm_cache.json")
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "900"))
LLM_READ_TIMEOUT = int(os.getenv("LLM_READ_TIMEOUT", "90"))

_cache_lock = threading.Lock()



def _load_cache() -> dict:
    if LLM_CACHE_PATH.exists():
        try:
            with open(LLM_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _cache_key(finding: dict, mode: str) -> str:
    key_data = {
        "mode": mode,
        "title": finding.get("title", ""),
        "severity": finding.get("severity", ""),
        "cvss_score": finding.get("cvss_score", 0),
        "description": (finding.get("description", "") or "")[:200],
        "file_path": finding.get("file_path", ""),
        "cve": finding.get("cve", ""),
        "cwe": finding.get("cwe", ""),
        "tags": sorted(finding.get("tags", [])),
        "risk_level": finding.get("risk_level", ""),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()



def _build_explanation_prompt(finding: dict) -> str:
    title    = finding.get("title", "Unknown")
    severity = finding.get("severity", "medium")
    cvss     = finding.get("cvss_score", 0)
    desc     = (finding.get("description", "") or "")[:400]
    fp       = finding.get("file_path", "")
    cve      = finding.get("cve", "") or "N/A"
    cwe      = finding.get("cwe", "") or "N/A"

    return f"""You are a senior application security engineer writing a detailed vulnerability report for a development team.
Your goal is to explain clearly WHAT the vulnerability is, WHY it is dangerous, and HOW urgent it is to fix.
Write in clear, professional English. Be specific and concrete. Avoid vague or generic sentences.

Vulnerability details:
- Title: {title}
- Severity: {severity} | CVSS Score: {cvss}
- CVE: {cve} | CWE: {cwe}
- Affected file: {fp}
- Description: {desc}

Respond with a single JSON object using EXACTLY these keys:

"summary"
  2 to 3 sentences. Explain what this vulnerability is, where it exists in the code, and what makes it dangerous.
  Be specific to the vulnerability above — not generic.

"impact"
  3 to 4 sentences. Describe concretely what an attacker can do if they exploit this.
  Mention affected data, systems, or users. Include potential business consequences (data breach, downtime, legal risk, etc.).

"root_cause"
  2 sentences. Explain the exact technical reason why this vulnerability exists.
  Be specific about the code pattern or bad practice responsible.

"exploitation_difficulty"
  One of: Easy / Medium / Hard
  Followed by a dash and 1 sentence explaining why (tools needed, skill level, prerequisites).
  Example: "Easy — No special tools required; a basic payload is enough to trigger the vulnerability."

"priority_note"
  Start with one of: Immediate / 48h / Week / Month
  Followed by 2 sentences: explain the urgency level and what could happen if not fixed in time.
  Example: "Immediate — This vulnerability is actively exploited in the wild and requires no authentication. Leaving it unpatched even for 24 hours significantly increases the risk of a breach."

Output only the raw JSON object. No markdown fences, no text before or after.

Example output format:
{{
  "summary": "This cross-site scripting (XSS) vulnerability exists in the user profile page where unsanitized user input is rendered directly into the HTML DOM. It allows an attacker to inject arbitrary JavaScript that executes in the context of other users' browsers. The vulnerability is especially dangerous because the affected page is accessible to all authenticated users.",
  "impact": "An attacker can steal session cookies and hijack authenticated user sessions, gaining full access to their accounts. Malicious scripts can also be used to perform actions on behalf of victims, such as changing passwords or exfiltrating personal data. If an admin account is compromised, the attacker gains elevated privileges over the entire application. This can lead to a full account takeover incident affecting all users of the platform.",
  "root_cause": "The application renders user-supplied profile data directly into the HTML template without encoding special characters. The developer used innerHTML instead of textContent, which allows HTML tags and script elements to be interpreted by the browser.",
  "exploitation_difficulty": "Easy — An attacker only needs to save a malicious script in their profile field; no special tools or authentication bypass is required.",
  "priority_note": "Immediate — Stored XSS vulnerabilities can affect every user who visits the infected page, including administrators. Delaying the fix increases the risk of a mass session hijacking incident."
}}

Now write the JSON for the vulnerability described above:"""


def _build_recommendation_prompt(finding: dict) -> str:
    title    = finding.get("title", "Unknown")
    severity = finding.get("severity", "medium")
    cvss     = finding.get("cvss_score", 0)
    desc     = (finding.get("description", "") or "")[:400]
    fp       = finding.get("file_path", "")
    cve      = finding.get("cve", "") or "N/A"
    cwe      = finding.get("cwe", "") or "N/A"

    return f"""You are a senior application security engineer writing a detailed remediation plan for a development team.
Your goal is to provide clear, actionable, and complete steps to fix this vulnerability.
Each step must be specific enough that a developer can implement it directly.
Do NOT write vague advice like "sanitize inputs" — explain exactly how and why.

Vulnerability details:
- Title: {title}
- Severity: {severity} | CVSS Score: {cvss}
- CVE: {cve} | CWE: {cwe}
- Affected file: {fp}
- Description: {desc}

Respond with a single JSON object using EXACTLY these keys:

"title"
  A clear, specific title for this remediation plan.
  Not just the vulnerability name — describe the fix approach.
  Example: "Fix Stored XSS in Profile Page by Applying Output Encoding"

"recommendations"
  Array of 4 to 6 remediation steps as strings.
  Each step must:
  - Start with an action verb (Replace, Use, Add, Configure, Validate, Enforce, etc.)
  - Be specific to this vulnerability type
  - Include a short explanation of WHY this step matters
  Example:
  "Replace all innerHTML assignments with textContent or a trusted sanitization library (e.g., DOMPurify) to prevent browser interpretation of injected HTML tags.",
  "Apply context-aware output encoding on all user-supplied data before rendering it in HTML, JavaScript, or URL contexts using a server-side encoding library.",
  "Implement a strict Content Security Policy (CSP) header that blocks inline scripts and limits script sources to trusted domains, reducing XSS impact even if injection occurs.",
  "Add server-side input validation to reject or strip HTML tags from fields that are not expected to contain markup.",
  "Conduct a full audit of all template rendering functions to identify other locations in the codebase where user data is rendered without encoding."

"references"
  Array of 2 to 3 authoritative URLs relevant to this specific vulnerability type.
  Use OWASP, NIST, CWE, CVE, or vendor security advisories.

"verification"
  2 sentences. Explain how to verify the fix was applied correctly.
  Include both automated (scanner, CI check) and manual (code review, pen test) methods.

"prevention"
  2 sentences. Explain how to prevent this entire class of vulnerability across the codebase in the future.
  Focus on process improvements, tooling (SAST/DAST in CI), or architecture decisions.

Output only the raw JSON object. No markdown fences, no text before or after.

Example output format:
{{
  "title": "Fix Stored XSS in Profile Page by Applying Output Encoding and CSP",
  "recommendations": [
    "Replace all innerHTML assignments with textContent or use DOMPurify to sanitize HTML before rendering, preventing the browser from interpreting injected script tags.",
    "Apply server-side output encoding on all user-supplied fields using a library such as OWASP Java Encoder or Python's html.escape() before inserting data into HTML templates.",
    "Implement a strict Content Security Policy (CSP) header with 'script-src self' to block execution of injected inline scripts even if encoding is missed in one location.",
    "Add server-side input validation to reject inputs containing HTML tags or JavaScript event handlers in fields that are not expected to contain markup.",
    "Audit all template files for other instances of unencoded user data rendering and apply the same encoding fix consistently across the codebase."
  ],
  "references": [
    "https://owasp.org/www-community/attacks/xss/",
    "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
    "https://cwe.mitre.org/data/definitions/79.html"
  ],
  "verification": "Re-run the SAST scanner and DAST tool against the fixed endpoint and confirm zero XSS findings remain. Manually test by attempting to save a <script>alert(1)</script> payload in the profile field and verify it is rendered as plain text.",
  "prevention": "Add a SAST rule in the CI/CD pipeline that fails pull requests containing innerHTML assignments or unencoded template variables. Require all developers to complete OWASP XSS training and enforce output encoding as a mandatory code review checklist item."
}}

Now write the JSON for the vulnerability described above:"""


# ---------------------------------------------------------------------------
# Appel Gemini API
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> Optional[str]:
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": LLM_MAX_TOKENS,
            "topK": 30,
            "topP": 0.90,
        },
    }

    try:
        resp = requests.post(
            GEMINI_URL,
            json=payload,
            timeout=LLM_READ_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            logger.error("❌ Gemini: aucun candidat dans la réponse")
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            logger.error("❌ Gemini: aucune partie dans le candidat")
            return None

        full = "".join(p.get("text", "") for p in parts).strip()
        logger.debug(f"[Gemini raw] {full[:400]}")
        return full

    except requests.exceptions.Timeout:
        logger.error("❌ Gemini Timeout — augmenter LLM_READ_TIMEOUT")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ Gemini ConnectionError: {e}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"❌ Gemini HTTP {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        logger.error(f"❌ Gemini erreur: {e}")
    return None


# ---------------------------------------------------------------------------
# Parsing JSON robuste
# ---------------------------------------------------------------------------

PLACEHOLDER_PATTERNS = [
    "one sentence", "two sentence", "easy/medium/hard",
    "immediate/48h", "week/month", "explain the",
    "summary of the", "technical cause here", "how to verify",
    "prevention measure", "short title",
]


def _is_placeholder(val: str) -> bool:
    if not isinstance(val, str):
        return False
    val_lower = val.lower()
    return any(p in val_lower for p in PLACEHOLDER_PATTERNS)


def _parse_json_response(raw: str, fallback: dict) -> dict:
    if not raw:
        return fallback

    text = raw.strip()

    if "```" in text:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e} — raw: {text[:200]}")
        return fallback

    for key in list(result.keys()):
        val = result[key]
        if isinstance(val, dict):
            result[key] = " ".join(str(v) for v in val.values() if v)
        elif isinstance(val, list) and key not in ("recommendations", "references"):
            result[key] = " ".join(str(v) for v in val if v)

    has_placeholders = any(
        _is_placeholder(v)
        for k, v in result.items()
        if k not in ("recommendations", "references")
    )
    if has_placeholders:
        logger.warning(f"LLM returned template placeholders — fallback. keys={list(result.keys())}")
        return fallback

    return result


# ---------------------------------------------------------------------------
# Nettoyage du cache corrompu
# ---------------------------------------------------------------------------

def clean_corrupted_cache() -> int:
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
            logger.info(f"🧹 Cache nettoyé : {removed} entrées corrompues supprimées sur {original_size}")
        else:
            logger.info(f"✅ Cache propre : {original_size} entrées, aucune corrompue")
    return removed


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

def explain_finding(
    finding: dict,
    use_cache: bool = True,
    mode: str = "explanation",
) -> dict:
    if mode not in ("explanation", "recommendation"):
        mode = "explanation"

    cache_key = _cache_key(finding, mode)

    if use_cache:
        with _cache_lock:
            cache = _load_cache()
        if cache_key in cache:
            logger.info(f"✅ Cache hit [{mode}] — {finding.get('title', '?')}")
            cached = cache[cache_key]
            cached["from_cache"] = True
            return cached

    if mode == "explanation":
        prompt = _build_explanation_prompt(finding)
        fallback: dict = {
            "summary": (
                f"The vulnerability '{finding.get('title', 'unknown')}' has been detected in the codebase and requires review. "
                f"It is classified as {finding.get('severity', 'unknown')} severity with a CVSS score of {finding.get('cvss_score', 'N/A')}. "
                "A manual security review of the affected file is strongly recommended."
            ),
            "impact": (
                "This vulnerability may allow an attacker to compromise the confidentiality, integrity, or availability of the affected system. "
                "Sensitive data, user accounts, or critical system resources could be exposed or manipulated. "
                "The exact blast radius depends on the deployment context and the attacker's access level. "
                "A thorough manual assessment is needed to determine the full impact on the application and its users."
            ),
            "root_cause": (
                "The root cause could not be automatically determined for this finding. "
                "Please refer to the vulnerability documentation and perform a manual code review of the affected file to identify the exact code pattern responsible."
            ),
            "exploitation_difficulty": (
                "Medium — Difficulty could not be automatically assessed. "
                "Assume a motivated attacker could exploit this with moderate effort and standard security tools."
            ),
            "priority_note": (
                "Immediate — This critical or high severity finding should be addressed as soon as possible to minimize exposure window."
                if str(finding.get("severity", "")).lower() in ("critical", "high")
                else "Week — This medium severity finding should be scheduled for remediation within the current development sprint."
                if str(finding.get("severity", "")).lower() == "medium"
                else "Month — This lower severity finding should be tracked in the backlog and resolved in an upcoming release cycle."
            ),
        }
    else:
        prompt = _build_recommendation_prompt(finding)
        fallback = {
            "title": f"Remediation Plan for: {finding.get('title', 'Unknown Vulnerability')}",
            "recommendations": [
                "Review the affected file and identify all locations where this vulnerability pattern appears before making any changes to ensure a complete fix.",
                "Apply the OWASP-recommended fix for this vulnerability class, following secure coding guidelines specific to your language and framework.",
                "Write unit and integration tests that cover both normal use cases and adversarial inputs related to this vulnerability type to prevent regressions.",
                "Re-run the security scanner after applying the fix to confirm the finding no longer appears and no new issues were introduced.",
                "Document the fix in your internal security changelog so the team can reference the remediation approach for similar vulnerabilities in the future.",
            ],
            "references": [
                "https://owasp.org/www-project-top-ten/",
                "https://cwe.mitre.org/",
            ],
            "verification": (
                "Re-run the SAST and DAST scanners against the patched code and confirm zero findings for this vulnerability class. "
                "Perform a peer code review to validate the fix aligns with secure coding standards and conduct a targeted penetration test on the affected endpoint."
            ),
            "prevention": (
                "Add an automated SAST rule in the CI/CD pipeline to detect this vulnerability class in future pull requests and block merges until resolved. "
                "Include this vulnerability type in the next developer security awareness training and enforce a mandatory security checklist during code reviews."
            ),
        }

    logger.info(f"🤖 Calling Gemini [{mode}] — {finding.get('title', '?')}")
    raw = _call_gemini(prompt)

    if raw:
        result = _parse_json_response(raw, fallback)
        result["from_cache"] = False
        result["_fallback"] = (result is fallback)

        if use_cache and not result.get("_fallback"):
            with _cache_lock:
                cache = _load_cache()
                cache[cache_key] = result
                _save_cache(cache)

        return result

    logger.warning(f"⚠️ Fallback [{mode}] — {finding.get('title', '?')}")
    fallback["from_cache"] = False
    fallback["_fallback"] = True
    return fallback