from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _configure_logging(level: str = "INFO") -> logging.Logger:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("invisithreat.predict_live")


logger = _configure_logging(os.environ.get("LOG_LEVEL", "INFO"))


RISK_LEVELS: Dict[int, str] = {
    0: "info",
    1: "low",
    2: "medium",
    3: "high",
    4: "critical",
}

SEVERITY_MAP: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "info": 0,
    "informational": 0,
}

CONTEXT_TAG_WEIGHTS: Dict[str, int] = {
    "production": 2,
    "prod": 2,
    "prd": 2,
    "live": 2,
    "main": 1,
    "external": 2,
    "internet-facing": 2,
    "public": 2,
    "exposed": 2,
    "internet": 2,
    "sensitive": 1,
    "pii": 1,
    "gdpr": 1,
    "confidential": 1,
    "secret": 1,
    "token": 1,
    "urgent": 3,
    "blocker": 3,
    "p0": 3,
    "p1": 2,
    "sca": 0,
    "api": 0,
}

CONTEXT_SCORE_MAX = 10


@dataclass
class PredictionResult:
    finding_id: int
    title: str
    severity: str
    ai_risk_score: int
    ai_risk_level: str
    ai_confidence: float
    context_score: int
    exploit_risk: float
    shap_top_features: List[Dict[str, Any]]
    ai_prediction_timestamp: str
    model_version: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    blocked: bool
    threshold: int
    critical_count: int
    critical_findings: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


@dataclass
class ModelMetadata:
    version: str
    feature_columns: List[str]
    metrics: Dict[str, Any]
    trained_at: str
    model_type: str


class DefectDojoClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
        max_retries: int = 3,
        page_size: int = 100,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "PATCH"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        start = time.monotonic()
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.monotonic() - start
            logger.debug("GET %s -> %d (%.2fs)", url, response.status_code, elapsed)
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error on GET %s: %s", url, exc)
            raise
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection failed on GET %s: %s", url, exc)
            raise
        except requests.exceptions.Timeout:
            logger.error("Timeout on GET %s (limit=%ds)", url, self.timeout)
            raise

    def _patch(self, url: str, data: Dict) -> Optional[Dict]:
        start = time.monotonic()
        try:
            response = self._session.patch(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.monotonic() - start
            logger.debug("PATCH %s -> %d (%.2fs)", url, response.status_code, elapsed)
            return response.json()
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error on PATCH %s: %s", url, exc)
            return None

    def iter_findings(
        self,
        engagement_id: int,
        active_only: bool = True,
    ) -> Generator[Dict, None, None]:
        url = f"{self.base_url}/api/v2/findings/"
        params: Dict[str, Any] = {
            "engagement": engagement_id,
            "limit": self.page_size,
        }
        if active_only:
            params["active"] = True

        page_num = 1
        while url:
            try:
                data = self._get(url, params=params)
            except Exception as exc:
                logger.error("Pagination stopped at page %d: %s", page_num, exc)
                break

            results = data.get("results", [])
            logger.debug("Page %d -> %d findings", page_num, len(results))
            yield from results

            url = data.get("next")
            params = None
            page_num += 1

    def get_findings(self, engagement_id: int, active_only: bool = True) -> List[Dict]:
        findings = list(self.iter_findings(engagement_id, active_only))
        logger.info("Fetched %d findings for engagement %d", len(findings), engagement_id)
        return findings

    def update_finding(self, finding_id: int, payload: Dict) -> bool:
        url = f"{self.base_url}/api/v2/findings/{finding_id}/"
        result = self._patch(url, payload)
        return result is not None


class FeatureExtractor:
    @staticmethod
    def parse_tags(raw_tags: Any) -> List[str]:
        if not raw_tags:
            return []
        if isinstance(raw_tags, list):
            return [str(t).strip().lower() for t in raw_tags if t]
        if isinstance(raw_tags, str):
            return [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
        return []

    @staticmethod
    def context_score(tags: List[str]) -> int:
        score = sum(CONTEXT_TAG_WEIGHTS.get(tag, 0) for tag in tags)
        return min(score, CONTEXT_SCORE_MAX)

    @staticmethod
    def severity_numeric(severity: str) -> int:
        return SEVERITY_MAP.get(severity.lower().strip(), 0)

    @staticmethod
    def exploit_risk(finding: Dict) -> float:
        cvss = float(finding.get("cvss_score") or 0)
        sev_num = SEVERITY_MAP.get(
            str(finding.get("severity", "")).lower().strip(), 0
        )
        description = str(finding.get("description", "")).lower()

        has_exploit = any(
            kw in description
            for kw in ("exploit", "metasploit", "public exploit", "proof of concept", "poc")
        )
        has_cve = "cve-" in description

        base = cvss * 0.7 + sev_num * 0.3
        if has_exploit:
            base *= 1.5
        elif has_cve:
            base *= 1.2

        return round(min(base, 10.0), 4)

    @staticmethod
    def age_factor(days_open: int) -> float:
        return round(max(0.1, 1.0 - (days_open / 180.0)), 4)

    @classmethod
    def extract(cls, finding: Dict) -> Dict[str, Any]:
        tags = cls.parse_tags(finding.get("tags"))
        days_open = int(finding.get("days_open") or 0)
        severity = str(finding.get("severity", "")).lower().strip()

        features: Dict[str, Any] = {
            "cvss_score": float(finding.get("cvss_score") or 0),
            "severity_numeric": cls.severity_numeric(severity),
            "days_open": days_open,
            "duplicate_count": int(finding.get("duplicate_count") or 0),
            "context_score": cls.context_score(tags),
            "exploit_risk": cls.exploit_risk(finding),
            "age_factor": cls.age_factor(days_open),
            "tags_count": len(tags),
            "tag_production": int(any(t in tags for t in ("production", "prod", "prd", "live"))),
            "tag_external": int(any(t in tags for t in ("external", "internet-facing", "public", "exposed"))),
            "tag_sensitive": int(any(t in tags for t in ("sensitive", "pii", "gdpr", "confidential"))),
            "tag_urgent": int(any(t in tags for t in ("urgent", "blocker", "p0", "p1"))),
            "tag_sca": int("sca" in tags),
            "tag_api": int("api" in tags),
        }

        return features

    @classmethod
    def extract_batch(cls, findings: List[Dict]) -> pd.DataFrame:
        records = [cls.extract(f) for f in findings]
        df = pd.DataFrame(records).fillna(0)
        return df


class RiskPredictor:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self._shap_available = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                "Run: python main.py run --engagement-id <N>"
            )

        logger.info("Loading model from %s", self.model_path)
        self.model = joblib.load(self.model_path)

        meta_path = self.model_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.metadata = ModelMetadata(
                version=raw.get("version", self.model_path.stem),
                feature_columns=raw.get("feature_columns", []),
                metrics=raw.get("metrics", {}),
                trained_at=raw.get("trained_at", "unknown"),
                model_type=raw.get("model_type", "unknown"),
            )
            f1 = self.metadata.metrics.get("test_f1_weighted", 0)
            logger.info(
                "Model loaded - version=%s type=%s F1=%.4f",
                self.metadata.version,
                self.metadata.model_type,
                f1,
            )
        else:
            logger.warning("Metadata file missing: %s", meta_path)
            self.metadata = ModelMetadata(
                version=self.model_path.stem,
                feature_columns=[],
                metrics={},
                trained_at="unknown",
                model_type="unknown",
            )

        try:
            import shap
            self._shap_available = True
            logger.debug("SHAP available")
        except ImportError:
            logger.info("SHAP not installed - explanations disabled")

    def _extract_base_model(self, model):
        if hasattr(model, "calibrated_classifiers"):
            if len(model.calibrated_classifiers) > 0:
                return model.calibrated_classifiers[0].base_estimator
        return model

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.metadata or not self.metadata.feature_columns:
            return df

        expected = self.metadata.feature_columns
        for col in expected:
            if col not in df.columns:
                logger.debug("Missing feature, filled with 0: %s", col)
                df[col] = 0

        extra = [c for c in df.columns if c not in expected]
        if extra:
            logger.debug("Extra features ignored: %s", extra)

        return df[expected]

    def _compute_shap(
        self, X: pd.DataFrame, top_n: int = 3
    ) -> List[List[Dict[str, Any]]]:
        if not self._shap_available:
            return [[] for _ in range(len(X))]

        try:
            import shap

            base_model = self._extract_base_model(self.model)
            if hasattr(self.model, "named_steps"):
                step_names = list(self.model.named_steps.keys())
                estimator = self.model.named_steps[step_names[-1]]
                base_model = self._extract_base_model(estimator)

            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)

            results = []
            for i in range(len(X)):
                if isinstance(shap_values, list):
                    pred_class = int(self.model.predict(X.iloc[[i]])[0])
                    sv = shap_values[pred_class][i]
                else:
                    sv = shap_values[i]

                feature_names = X.columns.tolist()
                top_indices = np.argsort(np.abs(sv))[::-1][:top_n]
                top_features = [
                    {
                        "feature": feature_names[j],
                        "value": round(float(X.iloc[i, j]), 4),
                        "shap_value": round(float(sv[j]), 4),
                        "direction": "+" if sv[j] > 0 else "-",
                    }
                    for j in top_indices
                ]
                results.append(top_features)

            return results

        except Exception as exc:
            logger.warning("SHAP calculation failed (fallback empty): %s", exc)
            return [[] for _ in range(len(X))]

    def predict_batch(
        self, findings: List[Dict], compute_shap: bool = True
    ) -> List[PredictionResult]:
        if not findings:
            return []

        X_raw = FeatureExtractor.extract_batch(findings)
        X = self._align_features(X_raw)

        risk_classes = self.model.predict(X).astype(int)
        risk_probas = self.model.predict_proba(X)

        shap_results = self._compute_shap(X) if compute_shap else [[] for _ in findings]

        model_version = self.metadata.version if self.metadata else "unknown"
        now = datetime.now(timezone.utc).isoformat()

        results: List[PredictionResult] = []
        for idx, finding in enumerate(findings):
            tags = FeatureExtractor.parse_tags(finding.get("tags"))
            risk_class = int(risk_classes[idx])
            confidence = float(np.max(risk_probas[idx]))

            results.append(
                PredictionResult(
                    finding_id=int(finding.get("id", 0)),
                    title=str(finding.get("title", "")),
                    severity=str(finding.get("severity", "")).lower(),
                    ai_risk_score=risk_class,
                    ai_risk_level=RISK_LEVELS.get(risk_class, "unknown"),
                    ai_confidence=round(confidence, 4),
                    context_score=FeatureExtractor.context_score(tags),
                    exploit_risk=FeatureExtractor.exploit_risk(finding),
                    shap_top_features=shap_results[idx],
                    ai_prediction_timestamp=now,
                    model_version=model_version,
                    tags=tags,
                )
            )

        return results


class ResultPublisher:
    def __init__(self, client: Optional[DefectDojoClient] = None) -> None:
        self._client = client

    def publish_to_defectdojo(
        self, results: List[PredictionResult], dry_run: bool = False
    ) -> Tuple[int, int]:
        if not self._client:
            logger.warning("DefectDojo client missing - publication skipped")
            return 0, 0

        success = 0
        failures = 0

        for result in results:
            if result.finding_id == 0:
                logger.debug("Finding without ID ignored: %s", result.title)
                continue

            new_ai_tag = f"ai-risk-{result.ai_risk_level}"
            clean_tags = [
                t for t in result.tags if not t.startswith("ai-risk-")
            ] + [new_ai_tag]

            payload = {
                "tags": clean_tags,
                "notes": self._build_note(result),
            }

            if dry_run:
                logger.info(
                    "[DRY-RUN] Would update finding %d -> %s (conf=%.2f)",
                    result.finding_id,
                    result.ai_risk_level,
                    result.ai_confidence,
                )
                success += 1
                continue

            ok = self._client.update_finding(result.finding_id, payload)
            if ok:
                success += 1
            else:
                failures += 1
                logger.warning("Failed to update finding %d", result.finding_id)

        logger.info(
            "DefectDojo: %d updated, %d failed (total=%d)",
            success,
            failures,
            len(results),
        )
        return success, failures

    @staticmethod
    def publish_to_file(results: List[PredictionResult], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(results),
            "summary": ResultPublisher._build_summary(results),
            "findings": [r.to_dict() for r in results],
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)

        logger.info("Results saved to %s (%d findings)", output_path, len(results))

    @staticmethod
    def _build_note(result: PredictionResult) -> str:
        lines = [
            f"[InvisiThreat AI] Score: {result.ai_risk_score}/4 ({result.ai_risk_level.upper()})",
            f"Confidence: {result.ai_confidence:.0%} | Context score: {result.context_score}",
            f"Model: {result.model_version} | {result.ai_prediction_timestamp}",
        ]
        if result.shap_top_features:
            lines.append("Top features:")
            for feat in result.shap_top_features:
                lines.append(
                    f"  {feat['direction']} {feat['feature']} = {feat['value']} "
                    f"(SHAP: {feat['shap_value']:+.3f})"
                )
        return "\n".join(lines)

    @staticmethod
    def _build_summary(results: List[PredictionResult]) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for r in results:
            counts[r.ai_risk_level] = counts.get(r.ai_risk_level, 0) + 1

        confidences = [r.ai_confidence for r in results]
        return {
            "by_level": counts,
            "avg_confidence": round(float(np.mean(confidences)), 4) if confidences else 0,
            "min_confidence": round(float(np.min(confidences)), 4) if confidences else 0,
        }


class SecurityGate:
    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold

    def evaluate(
        self, results: List[PredictionResult], ci_mode: bool = False
    ) -> GateDecision:
        critical = [r for r in results if r.ai_risk_score >= self.threshold]

        decision = GateDecision(
            blocked=len(critical) > 0,
            threshold=self.threshold,
            critical_count=len(critical),
            critical_findings=[
                {
                    "id": r.finding_id,
                    "title": r.title,
                    "ai_risk_score": r.ai_risk_score,
                    "ai_risk_level": r.ai_risk_level,
                    "ai_confidence": r.ai_confidence,
                    "top_reason": r.shap_top_features[0]["feature"]
                    if r.shap_top_features
                    else "n/a",
                }
                for r in critical
            ],
            message=(
                f"Security Gate: {len(critical)} critical finding(s) detected"
                if critical
                else "Security Gate: No critical findings"
            ),
        )

        if decision.blocked:
            logger.warning("=" * 60)
            logger.warning(decision.message)
            logger.warning("Block threshold: risk_score >= %d", self.threshold)
            for f in decision.critical_findings[:10]:
                logger.warning(
                    "  [%s] %s (score=%d, conf=%.0f%%)",
                    f["ai_risk_level"].upper(),
                    f["title"][:70],
                    f["ai_risk_score"],
                    f["ai_confidence"] * 100,
                )
            if len(decision.critical_findings) > 10:
                logger.warning(
                    "  ... and %d more findings", len(decision.critical_findings) - 10
                )
            logger.warning("=" * 60)

            if ci_mode:
                logger.error("CI/CD: Deployment blocked by Security Gate")
                sys.exit(1)
        else:
            logger.info(decision.message)

        return decision


class LivePredictor:
    def __init__(
        self,
        predictor: RiskPredictor,
        publisher: ResultPublisher,
        gate: SecurityGate,
        dd_client: Optional[DefectDojoClient] = None,
    ) -> None:
        self._predictor = predictor
        self._publisher = publisher
        self._gate = gate
        self._dd_client = dd_client

    @classmethod
    def from_env(
        cls,
        model_path: Path,
        dd_url: Optional[str] = None,
        dd_api_key: Optional[str] = None,
        gate_threshold: int = 3,
        page_size: int = 100,
    ) -> "LivePredictor":
        resolved_url = dd_url or os.environ.get("DEFECTDOJO_URL", "http://localhost:8080")
        resolved_key = dd_api_key or os.environ.get("DEFECTDOJO_API_KEY", "")

        dd_client: Optional[DefectDojoClient] = None
        if resolved_key:
            dd_client = DefectDojoClient(
                base_url=resolved_url,
                api_key=resolved_key,
                page_size=page_size,
            )
        else:
            logger.warning("DEFECTDOJO_API_KEY not set - DefectDojo operations disabled")

        predictor = RiskPredictor(model_path)
        publisher = ResultPublisher(dd_client)
        gate = SecurityGate(threshold=gate_threshold)

        return cls(predictor, publisher, gate, dd_client)

    def run(
        self,
        engagement_id: int,
        active_only: bool = True,
        update_dojo: bool = True,
        dry_run: bool = False,
        output_path: Optional[Path] = None,
        ci_mode: bool = False,
        compute_shap: bool = True,
        batch_size: int = 200,
    ) -> List[PredictionResult]:
        start_time = time.monotonic()

        if not self._dd_client:
            logger.error("Cannot fetch findings without DefectDojo client")
            return []

        findings = self._dd_client.get_findings(engagement_id, active_only)
        if not findings:
            logger.warning("No findings for engagement %d", engagement_id)
            return []

        all_results: List[PredictionResult] = []
        total_batches = (len(findings) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch = findings[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            logger.info(
                "Batch %d/%d - %d findings",
                batch_idx + 1,
                total_batches,
                len(batch),
            )
            batch_results = self._predictor.predict_batch(batch, compute_shap=compute_shap)
            all_results.extend(batch_results)

        self._log_summary(all_results)
        elapsed = time.monotonic() - start_time
        rate = len(all_results) / elapsed if elapsed > 0 else 0
        logger.info(
            "Prediction completed: %d findings in %.2fs (%.0f findings/s)",
            len(all_results),
            elapsed,
            rate,
        )

        if output_path:
            self._publisher.publish_to_file(all_results, output_path)

        if update_dojo:
            self._publisher.publish_to_defectdojo(all_results, dry_run=dry_run)

        self._gate.evaluate(all_results, ci_mode=ci_mode)

        return all_results

    @staticmethod
    def _log_summary(results: List[PredictionResult]) -> None:
        counts: Dict[str, int] = {}
        for r in results:
            counts[r.ai_risk_level] = counts.get(r.ai_risk_level, 0) + 1

        confidences = [r.ai_confidence for r in results]
        avg_conf = round(float(np.mean(confidences)) * 100, 1) if confidences else 0
        min_conf = round(float(np.min(confidences)) * 100, 1) if confidences else 0

        logger.info("Prediction summary (%d findings):", len(results))
        order = ["critical", "high", "medium", "low", "info"]
        for level in order:
            count = counts.get(level, 0)
            if count:
                bar = "#" * min(count, 40)
                logger.info("  %-10s %4d  %s", level.upper(), count, bar)
        logger.info("  Avg confidence: %.1f%%  (min: %.1f%%)", avg_conf, min_conf)


def _resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return Path(__file__).parent / p


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="InvisiThreat - AI Risk Scoring Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engagement-id",
        type=int,
        required=True,
        help="DefectDojo engagement ID to score",
    )
    parser.add_argument(
        "--dd-url",
        default=None,
        help="DefectDojo URL (default: $DEFECTDOJO_URL)",
    )
    parser.add_argument(
        "--dd-api-key",
        default=None,
        help="DefectDojo API key (default: $DEFECTDOJO_API_KEY)",
    )
    parser.add_argument(
        "--model-path",
        default="models/pipeline_latest.pkl",
        help="Path to model .pkl file",
    )
    parser.add_argument(
        "--update-dojo",
        action="store_true",
        default=False,
        help="Write scores to DefectDojo",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate without writing to DefectDojo",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="JSON output file path",
    )
    parser.add_argument(
        "--gate-threshold",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="Minimum score to block pipeline (0=info to 4=critical)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of findings per batch",
    )
    parser.add_argument(
        "--all",
        dest="active_only",
        action="store_false",
        default=True,
        help="Include inactive findings",
    )
    parser.add_argument(
        "--no-shap",
        dest="compute_shap",
        action="store_false",
        default=True,
        help="Disable SHAP calculations",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    model_path = _resolve_model_path(args.model_path)

    try:
        live = LivePredictor.from_env(
            model_path=model_path,
            dd_url=args.dd_url,
            dd_api_key=args.dd_api_key,
            gate_threshold=args.gate_threshold,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(2)

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        script_dir = Path(__file__).parent
        output_path = script_dir / f"logs/predictions_eng_{args.engagement_id}.json"

    ci_mode = os.environ.get("CI", "").lower() in ("true", "1", "yes")

    live.run(
        engagement_id=args.engagement_id,
        active_only=args.active_only,
        update_dojo=args.update_dojo,
        dry_run=args.dry_run,
        output_path=output_path,
        ci_mode=ci_mode,
        compute_shap=args.compute_shap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()