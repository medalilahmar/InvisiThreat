import argparse
import logging
import os
import sys
from typing import Set

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DefectDojoClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def get_findings(self, engagement_id: int, active_only: bool = True) -> list:
        findings = []
        params = {"engagement": engagement_id, "limit": 100, "offset": 0}
        if active_only:
            params["active"] = "true"

        while True:
            resp = self.session.get(
                f"{self.base_url}/api/v2/findings/",
                params=params,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            findings.extend(data.get("results", []))
            if not data.get("next"):
                break
            params["offset"] += 100

        logger.info(f"Fetched {len(findings)} findings")
        return findings

    def patch_finding(self, finding_id: int, tags: list) -> bool:
        try:
            self.session.patch(
                f"{self.base_url}/api/v2/findings/{finding_id}/",
                json={"tags": tags}
            ).raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to update finding {finding_id}: {e}")
            return False


def compute_tags(finding: dict) -> Set[str]:
    tags: Set[str] = set()

    for t in finding.get("tags", []):
        tags.add(t.lower().strip())

    severity = (finding.get("severity") or "").title()
    title = (finding.get("title") or "").lower()
    desc = (finding.get("description") or "").lower()
    file_path = (finding.get("file_path") or "").lower()
    component = (finding.get("component_name") or "").lower()

    text = f"{title} {desc} {file_path} {component}"

    if severity in ["Critical", "High"]:
        tags.add("urgent")
    if severity == "Critical":
        tags.add("blocker")

    prod_keywords = ["prod", "production", "live", "prd", "main", "master"]
    if any(k in text for k in prod_keywords):
        tags.add("production")
        tags.add("prod")

    external_keywords = [
        "external", "internet", "public", "exposed",
        "http", "api endpoint", "frontend", "zap"
    ]
    if any(k in text for k in external_keywords):
        tags.add("external")
        tags.add("internet-facing")

    sensitive_keywords = [
        "password", "secret", "token", "pii", "gdpr",
        "credential", "api key", "auth", "login", "user",
        "personal data", "confidential"
    ]
    if any(k in text for k in sensitive_keywords):
        tags.add("sensitive")
        tags.add("pii")

    sca_keywords = ["package.json", "dependency", "vulnerable", "npm", "pip", "maven"]
    if any(k in text for k in sca_keywords):
        tags.add("sca")

    api_keywords = ["/api/", "rest", "graphql", "endpoint"]
    if any(k in text for k in api_keywords):
        tags.add("api")

    return tags


def main():
    parser = argparse.ArgumentParser(
        description="InvisiThreat - Intelligent Tagging for DefectDojo Findings"
    )
    parser.add_argument(
        "--dd-url",
        default=os.environ.get("DEFECTDOJO_URL"),
        help="DefectDojo URL (or DEFECTDOJO_URL env var)"
    )
    parser.add_argument(
        "--dd-api-key",
        default=os.environ.get("DEFECTDOJO_API_KEY"),
        help="DefectDojo API key (or DEFECTDOJO_API_KEY env var)"
    )
    parser.add_argument(
        "--engagement-id",
        required=True,
        type=int,
        help="Engagement ID to process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation mode - no changes made"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include inactive findings"
    )

    args = parser.parse_args()

    if not args.dd_url:
        logger.error("DefectDojo URL not provided")
        sys.exit(1)

    if not args.dd_api_key:
        logger.error("DefectDojo API key not provided")
        sys.exit(1)

    client = DefectDojoClient(args.dd_url, args.dd_api_key)

    logger.info("=" * 60)
    logger.info("InvisiThreat - Intelligent Tagging")
    logger.info("=" * 60)
    logger.info(f"URL: {args.dd_url}")
    logger.info(f"Engagement ID: {args.engagement_id}")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"Include inactive: {args.all}")
    logger.info("=" * 60)

    findings = client.get_findings(args.engagement_id, active_only=not args.all)

    if not findings:
        logger.warning(f"No findings found for engagement {args.engagement_id}")
        sys.exit(0)

    updated = 0
    for finding in findings:
        finding_id = finding["id"]
        current_tags = set(t.lower().strip() for t in finding.get("tags", []))
        new_tags = compute_tags(finding)

        if new_tags == current_tags:
            continue

        added = new_tags - current_tags
        tag_list = sorted(new_tags)

        if args.dry_run:
            logger.info(f"[DRY-RUN] Finding {finding_id} would add: {sorted(added)}")
            updated += 1
        else:
            if client.patch_finding(finding_id, tag_list):
                updated += 1
                logger.info(f"Updated finding {finding_id} added: {sorted(added)}")

    logger.info("=" * 60)
    logger.info(f"Total findings processed: {len(findings)}")
    logger.info(f"Total findings updated: {updated}")
    if args.dry_run:
        logger.info("Mode: DRY-RUN - no actual changes made")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()