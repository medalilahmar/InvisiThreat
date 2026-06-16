import os
import logging
from typing import Optional
from jira import JIRA, JIRAError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

RISK_TO_JIRA_PRIORITY = {
    "Critical": "Highest",
    "High":     "High",
    "Medium":   "Medium",
    "Low":      "Low",
}


class JiraService:
    """
    Service pour la création automatique de tickets Jira
    à partir de findings DefectDojo avec scores IA InvisiThreat.
    """

    def __init__(self):
        self.server           = os.getenv("JIRA_SERVER", "")
        self.email            = os.getenv("JIRA_EMAIL", "")
        self.api_token        = os.getenv("JIRA_API_TOKEN", "")
        self.project_key      = os.getenv("JIRA_PROJECT_KEY", "SEC")
        self.issue_type       = os.getenv("JIRA_ISSUE_TYPE", "Bug")
        self.label_prefix     = os.getenv("JIRA_LABEL_PREFIX", "invisitreat")
        self.default_assignee = os.getenv("JIRA_ASSIGNEE_ID", None)

        self._client: Optional[JIRA] = None
        self._connected = False

    # ------------------------------------------------------------------
    # Client Jira (lazy init)
    # ------------------------------------------------------------------

    @property
    def client(self) -> JIRA:
        if self._client is None:
            if not all([self.server, self.email, self.api_token]):
                raise ValueError(
                    "Variables JIRA_SERVER, JIRA_EMAIL et JIRA_API_TOKEN "
                    "manquantes dans .env"
                )
            try:
                self._client = JIRA(
                    server=self.server,
                    basic_auth=(self.email, self.api_token),
                    options={"verify": True},
                )
                logger.info("Connexion Jira etablie : %s", self.server)
                self._connected = True
            except Exception as exc:
                logger.error("Erreur connexion Jira : %s", exc)
                self._connected = False
                raise
        return self._client

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def create_security_issue(
        self,
        finding: dict,
        ai_prediction: dict,
        llm_explanation: Optional[dict] = None,
        llm_recommendation: Optional[dict] = None,
    ) -> dict:
        """
        Cree un ticket Jira pour un finding DefectDojo.
        llm_explanation et llm_recommendation sont optionnels :
        ils peuvent etre None a la creation et ajoutes plus tard
        via update_issue_with_llm (background task).
        """
        existing = self._find_existing_issue(finding.get("id"))
        if existing:
            logger.info("Issue Jira existante trouvee : %s", existing.key)
            return {
                "jira_key":       existing.key,
                "jira_url":       f"{self.server}/browse/{existing.key}",
                "jira_id":        existing.id,
                "jira_self":      existing.self,
                "already_exists": True,
            }

        summary     = self._build_summary(finding, ai_prediction)
        description = self._build_description(
            finding, ai_prediction, llm_explanation, llm_recommendation
        )
        priority = RISK_TO_JIRA_PRIORITY.get(
            ai_prediction.get("risk_level", "Medium"), "Medium"
        )
        labels = self._build_labels(finding, ai_prediction)

        issue_fields: dict = {
            "project":   {"key": self.project_key},
            "summary":   summary,
            "description": description,
            "issuetype": {"name": self.issue_type},
            "priority":  {"name": priority},
            "labels":    labels,
        }
        if self.default_assignee:
            issue_fields["assignee"] = {"accountId": self.default_assignee}

        try:
            issue = self.client.create_issue(fields=issue_fields)
            logger.info("Issue Jira creee : %s", issue.key)
            self._add_ai_comment(issue.key, ai_prediction)
            return {
                "jira_key":       issue.key,
                "jira_url":       f"{self.server}/browse/{issue.key}",
                "jira_id":        issue.id,
                "jira_self":      issue.self,
                "already_exists": False,
            }
        except JIRAError as exc:
            logger.error(
                "Erreur creation issue Jira : %s - %s", exc.status_code, exc.text
            )
            raise Exception(f"Jira API Error: {exc.text}") from exc

    def update_issue_with_llm(
        self,
        jira_key: str,
        llm_explanation: Optional[dict],
        llm_recommendation: Optional[dict],
    ) -> None:
        """
        Enrichit la description d'un ticket existant avec les resultats LLM.
        Appelee en background task depuis routers/jira.py apres la reponse HTTP.
        Ne leve aucune exception — les erreurs sont uniquement loggees.
        """
        try:
            issue   = self.client.issue(jira_key)
            current = issue.fields.description or ""

            addition = self._build_llm_section(llm_explanation, llm_recommendation)
            if not addition:
                logger.info(
                    "[update_issue_with_llm] Aucun contenu LLM a ajouter pour %s",
                    jira_key,
                )
                return

            # Remplacer la ligne d'attente par le contenu reel
            updated = current.replace(
                "Enrichissement LLM en cours — cette section sera completee automatiquement.",
                "",
            )
            issue.update(fields={"description": updated + addition})
            logger.info("Ticket %s enrichi avec les resultats LLM.", jira_key)

        except JIRAError as exc:
            logger.warning(
                "Mise a jour LLM du ticket %s echouee (JIRAError) : %s",
                jira_key, exc.text,
            )
        except Exception as exc:
            logger.warning(
                "Mise a jour LLM du ticket %s echouee : %s", jira_key, exc
            )

    # ------------------------------------------------------------------
    # Constructeurs internes
    # ------------------------------------------------------------------

    def _build_summary(self, finding: dict, ai: dict) -> str:
        risk_level = ai.get("risk_level", "Unknown")
        title      = finding.get("title", "Vulnerability")
        product    = finding.get("product_name", "Unknown")
        cve        = finding.get("cve", "")
        cve_part   = f" [{cve}]" if cve else ""

        summary = f"[{risk_level}]{cve_part} {title} - {product}"
        return summary[:255]

    def _build_description(
        self,
        finding: dict,
        ai: dict,
        explanation: Optional[dict],
        recommendation: Optional[dict],
    ) -> str:
        risk_level = ai.get("risk_level", "N/A")
        risk_score = ai.get("risk_score", "N/A")
        confidence = round(ai.get("confidence", 0) * 100, 1)
        probas     = ai.get("probabilities", {})

        lines = [
            "h2. Finding Information",
            "",
            f"|| Field || Value ||",
            f"| DefectDojo ID | {finding.get('id', 'N/A')} |",
            f"| Title | {finding.get('title', 'N/A')} |",
            f"| Scanner Severity | {finding.get('severity', 'N/A')} |",
            f"| CVE | {finding.get('cve', 'N/A') or 'N/A'} |",
            f"| CWE | {finding.get('cwe', 'N/A') or 'N/A'} |",
            f"| File | {finding.get('file_path', 'N/A')} |",
            f"| Line | {finding.get('line', 'N/A')} |",
            f"| CVSS Score | {finding.get('cvss_score', 'N/A')}/10 |",
            f"| EPSS Score | {finding.get('epss_score', 'N/A')} |",
            f"| Product | {finding.get('product_name', 'N/A')} |",
            f"| Engagement | {finding.get('engagement_name', 'N/A')} |",
            f"| Active | {'Yes' if finding.get('active') else 'No'} |",
            f"| Verified | {'Yes' if finding.get('verified') else 'No'} |",
            f"| False Positive | {'Yes' if finding.get('false_p') else 'No'} |",
            "",
            "h2. AI Risk Assessment (InvisiThreat)",
            "",
            f"|| Metric || Value ||",
            f"| Risk Level | {risk_level} |",
            f"| AI Score | {risk_score}/100 |",
            f"| Confidence | {confidence}% |",
            f"| Low probability | {round(probas.get('Low', 0) * 100, 1)}% |",
            f"| Medium probability | {round(probas.get('Medium', 0) * 100, 1)}% |",
            f"| High probability | {round(probas.get('High', 0) * 100, 1)}% |",
            f"| Critical probability | {round(probas.get('Critical', 0) * 100, 1)}% |",
            "",
        ]

        # Section LLM — placeholder si pas encore disponible
        if not explanation and not recommendation:
            lines += [
                "h2. LLM Analysis",
                "",
                "Enrichissement LLM en cours — cette section sera completee automatiquement.",
                "",
            ]
        else:
            lines += self._build_llm_section(explanation, recommendation).splitlines()

        lines += [
            "----",
            "Issue created automatically by InvisiThreat AI Security Engine.",
            f"Source : DefectDojo Finding #{finding.get('id')}",
        ]

        return "\n".join(lines)

    def _build_llm_section(
        self,
        explanation: Optional[dict],
        recommendation: Optional[dict],
    ) -> str:
        """
        Construit le bloc texte LLM (explanation + recommendation).
        Retourne une chaine vide si les deux sont absents ou vides.
        """
        lines = []

        if explanation and isinstance(explanation, dict):
            lines += [
                "",
                "h2. LLM Explanation",
                "",
                f"*Summary* : {explanation.get('summary', 'N/A')}",
                "",
                f"*Impact* : {explanation.get('impact', 'N/A')}",
                "",
            ]
            if explanation.get("root_cause"):
                lines += [f"*Root Cause* : {explanation['root_cause']}", ""]
            if explanation.get("exploitation_difficulty"):
                lines += [
                    f"*Exploitation Difficulty* : {explanation['exploitation_difficulty']}",
                    "",
                ]
            if explanation.get("priority_note"):
                lines += [f"*Recommended Timeframe* : {explanation['priority_note']}", ""]

        if recommendation and isinstance(recommendation, dict):
            lines += ["", "h2. Remediation Recommendations", ""]

            recs = recommendation.get("recommendations", [])
            if recs:
                lines.append("*Remediation Steps* :")
                for i, step in enumerate(recs, 1):
                    lines.append(f"{i}. {step}")
                lines.append("")

            refs = recommendation.get("references", [])
            if refs:
                lines.append("*References* :")
                for ref in refs:
                    lines.append(f"- {ref}")
                lines.append("")

            if recommendation.get("verification"):
                lines += [f"*Verification* : {recommendation['verification']}", ""]

            if recommendation.get("prevention"):
                lines += [f"*Prevention* : {recommendation['prevention']}", ""]

        return "\n".join(lines)

    def _build_labels(self, finding: dict, ai: dict) -> list:
        prefix = self.label_prefix
        labels = [
            prefix,
            f"risk-{ai.get('risk_level', 'unknown').lower()}",
            f"severity-{finding.get('severity', 'unknown').lower()}",
            f"defectdojo-finding-{finding.get('id')}",
        ]
        if finding.get("cve"):
            labels.append("has-cve")
        if finding.get("cwe"):
            labels.append("has-cwe")
        if finding.get("active"):
            labels.append("active")
        if finding.get("verified"):
            labels.append("verified")

        return [label.replace(" ", "-").lower() for label in labels]

    def _find_existing_issue(self, finding_id) -> Optional[object]:
        if not finding_id:
            return None

        jql = (
            f'project = "{self.project_key}" '
            f'AND labels = "defectdojo-finding-{finding_id}" '
            f'AND status != Done'
        )
        try:
            results = self.client.search_issues(jql, maxResults=1)
            return results[0] if results else None
        except JIRAError as exc:
            logger.warning(
                "Erreur recherche issue existante pour finding %s : %s",
                finding_id, exc.text,
            )
            return None

    def _add_ai_comment(self, issue_key: str, ai: dict) -> None:
        body = (
            f"InvisiThreat AI Score — "
            f"Risk Level : {ai.get('risk_level')} | "
            f"Score : {ai.get('risk_score')}/100 | "
            f"Confidence : {round(ai.get('confidence', 0) * 100, 1)}%"
        )
        try:
            self.client.add_comment(issue_key, body)
            logger.debug("Commentaire IA ajoute a %s", issue_key)
        except JIRAError as exc:
            logger.warning(
                "Impossible d'ajouter un commentaire a %s : %s", issue_key, exc.text
            )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> dict:
        try:
            if not self._client:
                _ = self.client

            project     = self.client.project(self.project_key)
            issue_types = [it.name for it in self.client.issue_types()]

            return {
                "status":       "ok",
                "connected":    True,
                "jira_server":  self.server,
                "project_key":  self.project_key,
                "project_name": project.name,
                "issue_types":  issue_types,
                "message":      f"Connected to Jira - Project {project.name}",
            }
        except Exception as exc:
            logger.error("Health check Jira echoue : %s", exc)
            return {
                "status":      "error",
                "connected":   False,
                "jira_server": self.server,
                "project_key": self.project_key,
                "detail":      str(exc),
                "message":     f"Connection error : {exc}",
            }


jira_service = JiraService()