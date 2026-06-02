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
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "900"))

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
    desc = _safe_str(finding.get("description"))[:400]
    fp = finding.get("file_path", "")
    cve = finding.get("cve", "") or "N/A"
    cwe = finding.get("cwe", "") or "N/A"

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
    """
    Build the prompt used for the 'recommendation' mode.
    Asks the model for a structured remediation plan with concrete steps,
    references, verification method and long-term prevention guidance.
    """
    title = finding.get("title", "Unknown")
    severity = finding.get("severity", "medium")
    cvss = finding.get("cvss_score", 0)
    desc = _safe_str(finding.get("description"))[:400]

    fp = finding.get("file_path", "")
    cve = finding.get("cve", "") or "N/A"
    cwe = finding.get("cwe", "") or "N/A"

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
        context_block = f"- Affected file: {fp}" + (f" (line {line})" if line else "")
    elif component:
        context_block = f"- Vulnerable component: {component} {comp_ver}".strip()
    if sast_src:
        context_block += f"\n- Source object: {sast_src}"
    if sast_sink:
        context_block += f"\n- Sink object: {sast_sink}"

    return f"""You are a senior application security engineer providing an exact code fix for a vulnerability.
Your goal is to produce a minimal, precise, and immediately applicable code correction.
Be specific — do NOT write generic advice. Write actual code.

Vulnerability details:
- Title: {title}
- Severity: {severity} | CVSS Score: {cvss}
- CVE: {cve} | CWE: {cwe}
{context_block}
- Description: {desc}
- Mitigation hint: {mitigation}

Respond with a single JSON object using EXACTLY these keys:

"vulnerable_snippet"
  The exact vulnerable code pattern (2 to 5 lines). Real code, not a description.

"fixed_snippet"
  The corrected version of the same code (2 to 5 lines). Minimal changes only.
  Do NOT rewrite the entire file — only fix the vulnerable part.

"explanation"
  2 to 3 sentences. Explain exactly what was changed and why it fixes the vulnerability.
  Be specific about the security property that was added or restored.

"confidence"
  A float between 0.0 and 1.0 representing your confidence in this fix.
  Use 0.90+ if file_path and line are available. Use 0.65-0.75 for generic pattern fixes.

Output only the raw JSON object. No markdown fences, no text before or after.

Example output format:
{{
  "vulnerable_snippet": "cursor.execute(\\"SELECT * FROM users WHERE id = \\" + user_id)",
  "fixed_snippet": "cursor.execute(\\"SELECT * FROM users WHERE id = %s\\", (user_id,))",
  "explanation": "The original code concatenated user input directly into the SQL query, enabling SQL injection. The fix uses a parameterized query which separates data from code, preventing any injected SQL from being executed.",
  "confidence": 0.92
}}

Now write the JSON for the vulnerability described above:"""


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
            "top_k": 30,
            "top_p": 0.90,
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
                f"The vulnerability '{title}' has been detected in the codebase and requires review. "
                f"It is classified as {finding.get('severity', 'unknown')} severity with a CVSS score "
                f"of {finding.get('cvss_score', 'N/A')}. "
                "A manual security review of the affected file is strongly recommended."
            ),
            "impact": (
                "This vulnerability may allow an attacker to compromise the confidentiality, "
                "integrity, or availability of the affected system. "
                "Sensitive data, user accounts, or critical system resources could be exposed or manipulated. "
                "The exact blast radius depends on the deployment context and the attacker's access level. "
                "A thorough manual assessment is needed to determine the full impact on the application."
            ),
            "root_cause": (
                "The root cause could not be automatically determined for this finding. "
                "Please refer to the vulnerability documentation and perform a manual code review "
                "of the affected file to identify the exact code pattern responsible."
            ),
            "exploitation_difficulty": (
                "Medium — Difficulty could not be automatically assessed. "
                "Assume a motivated attacker could exploit this with moderate effort and standard tools."
            ),
            "priority_note": (
                "Immediate — This critical or high severity finding should be addressed as soon as "
                "possible to minimize the exposure window."
                if str(finding.get("severity", "")).lower() in ("critical", "high")
                else (
                    "Week — This medium severity finding should be scheduled for remediation within "
                    "the current development sprint."
                    if str(finding.get("severity", "")).lower() == "medium"
                    else "Month — This lower severity finding should be tracked in the backlog and "
                    "resolved in an upcoming release cycle."
                )
            ),
        }

    elif mode == "recommendation":
        prompt = _build_recommendation_prompt(finding)
        fallback = {
            "title": f"Remediation Plan for: {title}",
            "recommendations": [
                "Review the affected file and identify all locations where this vulnerability pattern "
                "appears before making changes to ensure a complete and consistent fix.",
                "Apply the OWASP-recommended fix for this vulnerability class, following secure coding "
                "guidelines specific to your language and framework.",
                "Write unit and integration tests that cover both normal use cases and adversarial inputs "
                "related to this vulnerability type to prevent future regressions.",
                "Re-run the security scanner after applying the fix to confirm the finding no longer "
                "appears and that no new issues were introduced by the change.",
                "Document the fix in your internal security changelog so the team can reference the "
                "remediation approach for similar vulnerabilities in the future.",
            ],
            "references": [
                "https://owasp.org/www-project-top-ten/",
                "https://cwe.mitre.org/",
            ],
            "verification": (
                "Re-run the SAST and DAST scanners against the patched code and confirm zero findings "
                "for this vulnerability class. "
                "Perform a peer code review to validate the fix aligns with secure coding standards "
                "and conduct a targeted penetration test on the affected endpoint."
            ),
            "prevention": (
                "Add an automated SAST rule in the CI/CD pipeline to detect this vulnerability class "
                "in future pull requests and block merges until resolved. "
                "Include this vulnerability type in the next developer security awareness training "
                "and enforce a mandatory security checklist during code reviews."
            ),
        }

    else:  # mode == "solution"
        prompt = _build_solution_prompt(finding)
        safe_mitigation = _safe_str(finding.get("mitigation"))
        mitigation_text = safe_mitigation if safe_mitigation else "OWASP guidelines"
        fallback = {
            "vulnerable_snippet": (
                f"# Vulnerable pattern for: {title}\n"
                "# Could not retrieve exact code — check file_path and static_finding fields."
            ),
            "fixed_snippet": (
                "# Apply the mitigation described in the finding.\n"
                f"# Refer to: {mitigation_text[:200]}"
            ),
            "explanation": (
                f"This is a {finding.get('severity', 'unknown')} severity vulnerability ({title}). "
                "Automatic code fix could not be generated — the file path or static finding "
                "information may be missing. Please apply the mitigation manually."
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