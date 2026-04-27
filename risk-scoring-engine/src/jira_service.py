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

RISK_TO_EMOJI = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
}


class JiraService:
    """
    Service pour la création automatique de tickets Jira
    à partir de findings DefectDojo avec scores IA
    """

    def __init__(self):
        self.server      = os.getenv("JIRA_SERVER", "")
        self.email       = os.getenv("JIRA_EMAIL", "")
        self.api_token   = os.getenv("JIRA_API_TOKEN", "")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "SEC")
        self.issue_type  = os.getenv("JIRA_ISSUE_TYPE", "Bug")
        self.label_prefix = os.getenv("JIRA_LABEL_PREFIX", "invisitreat")
        self.default_assignee = os.getenv("JIRA_ASSIGNEE_ID", None)

        self._client: Optional[JIRA] = None
        self._connected = False

    @property
    def client(self) -> JIRA:
        """Lazy initialization du client Jira"""
        if self._client is None:
            if not all([self.server, self.email, self.api_token]):
                raise ValueError(
                    "Variables JIRA_SERVER, JIRA_EMAIL et JIRA_API_TOKEN manquantes dans .env"
                )
            try:
                self._client = JIRA(
                    server=self.server,
                    basic_auth=(self.email, self.api_token),
                    options={"verify": True},
                )
                logger.info(f"✓ Connexion Jira établie : {self.server}")
                self._connected = True
            except Exception as e:
                logger.error(f"✗ Erreur connexion Jira : {e}")
                self._connected = False
                raise
        return self._client

    def create_security_issue(
        self,
        finding: dict,
        ai_prediction: dict,
        llm_explanation: Optional[dict] = None,
        llm_recommendation: Optional[dict] = None,
    ) -> dict:
        
        
        existing = self._find_existing_issue(finding.get("id"))
        if existing:
            logger.info(f"✓ Issue Jira existante trouvée : {existing.key}")
            return {
                "jira_key": existing.key,
                "jira_url": f"{self.server}/browse/{existing.key}",
                "jira_id": existing.id,
                "jira_self": existing.self,
                "already_exists": True,
            }

        summary     = self._build_summary(finding, ai_prediction)
        description = self._build_description(
            finding, ai_prediction, llm_explanation, llm_recommendation
        )
        priority    = RISK_TO_JIRA_PRIORITY.get(
            ai_prediction.get("risk_level", "Medium"), "Medium"
        )
        labels      = self._build_labels(finding, ai_prediction)

        issue_fields = {
            "project":     {"key": self.project_key},
            "summary":     summary,
            "description": description,
            "issuetype":   {"name": self.issue_type},
            "priority":    {"name": priority},
            "labels":      labels,
        }

        if self.default_assignee:
            issue_fields["assignee"] = {"accountId": self.default_assignee}

        try:
            issue = self.client.create_issue(fields=issue_fields)
            logger.info(f"✓ Issue Jira créée : {issue.key}")

            self._add_ai_comment(issue.key, ai_prediction)

            return {
                "jira_key": issue.key,
                "jira_url": f"{self.server}/browse/{issue.key}",
                "jira_id": issue.id,
                "jira_self": issue.self,
                "already_exists": False,
            }
        except JIRAError as e:
            logger.error(f"✗ Erreur création issue Jira : {e.status_code} – {e.text}")
            raise Exception(f"Jira API Error: {e.text}")

    def _build_summary(self, finding: dict, ai: dict) -> str:
        emoji     = RISK_TO_EMOJI.get(ai.get("risk_level", ""), "⚪")
        title     = finding.get("title", "Vulnerability")
        product   = finding.get("product_name", "Unknown")
        cve       = finding.get("cve", "")
        cve_part  = f" [{cve}]" if cve else ""
        
        summary = f"{emoji} [{ai.get('risk_level','?')}]{cve_part} {title} – {product}"
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
            "## 🔍 Informations du Finding",
            f"- **ID DefectDojo** : {finding.get('id', 'N/A')}",
            f"- **Titre** : {finding.get('title', 'N/A')}",
            f"- **Sévérité scanner** : {finding.get('severity', 'N/A')}",
            f"- **CVE** : {finding.get('cve', 'N/A') or 'N/A'}",
            f"- **CWE** : {finding.get('cwe', 'N/A') or 'N/A'}",
            f"- **Fichier** : `{finding.get('file_path', 'N/A')}`",
            f"- **Ligne** : {finding.get('line', 'N/A')}",
            f"- **CVSS Score** : {finding.get('cvss_score', 'N/A')}/10",
            f"- **EPSS Score** : {finding.get('epss_score', 'N/A')}",
            f"- **Produit** : {finding.get('product_name', 'N/A')}",
            f"- **Engagement** : {finding.get('engagement_name', 'N/A')}",
            f"- **Statut** : {'✅ Actif' if finding.get('active') else '❌ Inactif'}",
            f"- **Vérifié** : {'✅ Oui' if finding.get('verified') else '❌ Non'}",
            f"- **Faux positif** : {'⚠️ Oui' if finding.get('false_p') else '❌ Non'}",
            "",
            "## 🤖 Score de Risque IA (InvisiThreat)",
            f"- **Classe de risque** : **{risk_level}**",
            f"- **Score IA** : **{risk_score}/100**",
            f"- **Confiance** : **{confidence}%**",
            "- **Distribution probabilités** :",
            f"  - 🟢 Low: {round(probas.get('Low', 0)*100,1)}%",
            f"  - 🟡 Medium: {round(probas.get('Medium', 0)*100,1)}%",
            f"  - 🟠 High: {round(probas.get('High', 0)*100,1)}%",
            f"  - 🔴 Critical: {round(probas.get('Critical', 0)*100,1)}%",
            "",
        ]

        if explanation and isinstance(explanation, dict):
            lines += [
                "## 💡 Explication IA (LLM)",
                f"**Résumé** : {explanation.get('summary', 'N/A')}",
                "",
                f"**Impact** : {explanation.get('impact', 'N/A')}",
                "",
            ]
            if explanation.get('root_cause'):
                lines.append(f"**Cause racine** : {explanation.get('root_cause')}")
                lines.append("")
            if explanation.get('priority_note'):
                lines.append(f"**Délai recommandé** : {explanation.get('priority_note')}")
                lines.append("")

        if recommendation and isinstance(recommendation, dict):
            lines += [
                "## 🛠️ Recommandations (LLM)",
            ]
            if recommendation.get('recommendations'):
                lines.append("**Étapes de remédiation** :")
                for i, step in enumerate(recommendation.get('recommendations', []), 1):
                    lines.append(f"{i}. {step}")
                lines.append("")
            
            if recommendation.get('references'):
                lines.append("**Références** :")
                for ref in recommendation.get('references', []):
                    lines.append(f"- 🔗 {ref}")
                lines.append("")
            
            if recommendation.get('verification'):
                lines.append(f"**Vérification** : {recommendation.get('verification')}")
                lines.append("")

        lines += [
            "---",
            f"*Issue créée automatiquement par InvisiThreat AI Security Engine*",
            f"*Source : DefectDojo Finding #{finding.get('id')}*",
        ]

        return "\n".join(lines)

    def _build_labels(self, finding: dict, ai: dict) -> list:
        prefix = self.label_prefix
        labels = [
            f"{prefix}",
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
        
        return [l.replace(" ", "-").lower() for l in labels]

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
        except JIRAError as e:
            logger.warning(f"Erreur lors de la recherche d'issue existante : {e.text}")
            return None

    def _add_ai_comment(self, issue_key: str, ai: dict):
        body = (
            f"*Score IA InvisiThreat mis à jour :*\n"
            f"Classe : *{ai.get('risk_level')}* | "
            f"Score : *{ai.get('risk_score')}/100* | "
            f"Confiance : *{round(ai.get('confidence',0)*100,1)}%*"
        )
        try:
            self.client.add_comment(issue_key, body)
            logger.debug(f"✓ Commentaire IA ajouté à {issue_key}")
        except JIRAError as e:
            logger.warning(f"⚠️ Impossible d'ajouter un commentaire à {issue_key} : {e.text}")

    def health_check(self) -> dict:
        try:
            if not self._client:
                # Essayer d'établir la connexion
                _ = self.client
            
            project = self.client.project(self.project_key)
            issue_types = [it.name for it in self.client.issue_types()]
            
            return {
                "status": "ok",
                "connected": True,
                "jira_server": self.server,
                "project_key": self.project_key,
                "project_name": project.name,
                "issue_types": issue_types,
                "message": f"Connecté à Jira - Projet {project.name}"
            }
        except Exception as e:
            logger.error(f"✗ Erreur health check Jira: {e}")
            return {
                "status": "error",
                "connected": False,
                "jira_server": self.server,
                "project_key": self.project_key,
                "detail": str(e),
                "message": f"Erreur de connexion : {str(e)}"
            }


jira_service = JiraService()