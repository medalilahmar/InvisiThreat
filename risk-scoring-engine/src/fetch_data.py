import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

BASE_URL        = os.getenv("DEFECTDOJO_URL", "").rstrip("/")
API_TOKEN       = os.getenv("DEFECTDOJO_API_KEY", "")
PAGE_LIMIT      = int(os.getenv("DEFECTDOJO_PAGE_LIMIT", "100"))
TIMEOUT         = int(os.getenv("DEFECTDOJO_TIMEOUT", "30"))
EPSS_API_URL    = "https://api.first.org/data/v1/epss"
EPSS_BATCH_SIZE = 100
EPSS_TIMEOUT    = 15

if not BASE_URL or not API_TOKEN:
    print("Variables DEFECTDOJO_URL et DEFECTDOJO_TOKEN requises dans .env")
    sys.exit(1)

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/fetch_data.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

EXPECTED_COLS = {
    "products":    {"id", "name"},
    "engagements": {"id", "product"},
    "tests":       {"id", "engagement"},
    "findings":    {"id", "title", "severity"},
}



def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Authorization": f"Token {API_TOKEN}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    })
    return session



def fetch_all_pages(session: requests.Session, url: str, resource: str) -> list[dict]:
    results: list[dict] = []
    page = 1

    while url:
        try:
            resp = session.get(url, timeout=TIMEOUT)

            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                logger.warning(f"[{resource}] Rate limit — waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data  = resp.json()
            batch = data.get("results", [])
            results.extend(batch)
            url   = data.get("next")
            page += 1

            if url:
                time.sleep(0.3)

        except requests.exceptions.Timeout:
            logger.error(f"[{resource}] Timeout page {page}")
            break
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{resource}] HTTP {e.response.status_code}: {e}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"[{resource}] Network error: {e}")
            break
        except Exception as e:
            logger.error(f"[{resource}] Unexpected error: {e}")
            break

    logger.info(f"[{resource}] {len(results)} items fetched ({page - 1} page(s))")
    return results



def fetch_products(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/products/?limit={PAGE_LIMIT}", "products")


def fetch_engagements(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/engagements/?limit={PAGE_LIMIT}", "engagements")


def fetch_tests(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/tests/?limit={PAGE_LIMIT}", "tests")


def fetch_all_findings(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/findings/?limit={PAGE_LIMIT}", "findings_all")



def build_lookup_table(
    tests:       list[dict],
    engagements: list[dict],
    products:    list[dict],
) -> dict[int, dict]:
    eng_map: dict[int, dict] = {int(e["id"]): e for e in engagements if e.get("id") is not None}
    prod_map: dict[int, dict] = {int(p["id"]): p for p in products if p.get("id") is not None}

    lookup: dict[int, dict] = {}
    unresolved = 0

    for t in tests:
        test_id = t.get("id")
        eng_id  = t.get("engagement")

        if test_id is None:
            continue

        test_id = int(test_id)

        if eng_id is None:
            unresolved += 1
            lookup[test_id] = {
                "engagement_id":   None,
                "engagement_name": "",
                "product_id":      None,
                "product_name":    "Unknown",
            }
            continue

        eng_id    = int(eng_id)
        eng       = eng_map.get(eng_id, {})
        prod_id   = eng.get("product")
        prod_id   = int(prod_id) if prod_id is not None else None
        prod      = prod_map.get(prod_id, {}) if prod_id else {}
        prod_name = prod.get("name", f"Unknown-prod{prod_id}") if prod_id else "Unknown"
        eng_name  = eng.get("name", f"engagement-{eng_id}")

        lookup[test_id] = {
            "engagement_id":   eng_id,
            "engagement_name": eng_name,
            "product_id":      prod_id,
            "product_name":    prod_name,
        }

    if unresolved:
        logger.warning(f"{unresolved} test(s) without engagement_id in lookup table")

    logger.info(f"Lookup table built: {len(lookup)} test_id(s) mapped")
    return lookup



def enrich_findings(findings: list[dict], lookup: dict[int, dict]) -> list[dict]:
    enriched   = []
    unresolved = 0

    for f in findings:
        row     = dict(f)
        test_id = f.get("test")
        info    = lookup.get(int(test_id)) if test_id is not None else None

        if info:
            row["product_id"]      = info["product_id"]
            row["product_name"]    = info["product_name"]
            row["engagement_id"]   = info["engagement_id"]
            row["engagement_name"] = info["engagement_name"]
        else:
            row["product_id"]      = None
            row["product_name"]    = "Unknown"
            row["engagement_id"]   = None
            row["engagement_name"] = ""
            unresolved += 1

        enriched.append(row)

    if unresolved:
        logger.warning(f"{unresolved} finding(s) with unresolved test_id")

    return enriched



def _extract_cve(finding: dict) -> Optional[str]:
    cve_direct = finding.get("cve")
    if cve_direct and isinstance(cve_direct, str):
        val = cve_direct.strip().upper()
        if val.startswith("CVE-"):
            return val

    vuln_ids = finding.get("vulnerability_ids")
    if vuln_ids:
        if isinstance(vuln_ids, str):
            try:
                import ast
                vuln_ids = ast.literal_eval(vuln_ids)
            except Exception:
                pass
        if isinstance(vuln_ids, list):
            for entry in vuln_ids:
                vid = None
                if isinstance(entry, dict):
                    vid = entry.get("id") or entry.get("vulnerability_id")
                elif isinstance(entry, str):
                    vid = entry
                if vid and isinstance(vid, str):
                    vid = vid.strip().upper()
                    if vid.startswith("CVE-"):
                        return vid

    return None


def batch_fetch_epss(cves: list[str]) -> dict[str, dict]:
    cache: dict[str, dict] = {}

    for i in range(0, len(cves), EPSS_BATCH_SIZE):
        batch  = cves[i:i + EPSS_BATCH_SIZE]
        params = {"cve": ",".join(batch)}

        try:
            resp = requests.get(EPSS_API_URL, params=params, timeout=EPSS_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                cve_id = item.get("cve", "").upper()
                if cve_id:
                    cache[cve_id] = {
                        "epss_score":      float(item.get("epss", 0.0)),
                        "epss_percentile": float(item.get("percentile", 0.0)),
                    }

        except requests.exceptions.RequestException as e:
            logger.warning(f"[EPSS] Batch {i // EPSS_BATCH_SIZE + 1} failed: {e}")
        except Exception as e:
            logger.warning(f"[EPSS] Unexpected error on batch {i // EPSS_BATCH_SIZE + 1}: {e}")

        time.sleep(0.5)

    logger.info(f"[EPSS] {len(cache)}/{len(cves)} CVE(s) scored")
    return cache


def enrich_findings_with_epss(findings: list[dict]) -> list[dict]:
    cves = list({
        _extract_cve(f)
        for f in findings
        if _extract_cve(f) is not None
    })

    logger.info(f"[EPSS] Fetching scores for {len(cves)} unique CVE(s)")
    epss_cache = batch_fetch_epss(cves) if cves else {}

    enriched = []
    hits     = 0

    for f in findings:
        row = dict(f)
        cve = _extract_cve(f)

        if cve and cve in epss_cache:
            row["epss_score"]      = epss_cache[cve]["epss_score"]
            row["epss_percentile"] = epss_cache[cve]["epss_percentile"]
            hits += 1
        else:
            row["epss_score"]      = float(f.get("epss_score") or 0.0)
            row["epss_percentile"] = float(f.get("epss_percentile") or 0.0)

        enriched.append(row)

    logger.info(f"[EPSS] {hits}/{len(findings)} finding(s) enriched with EPSS score")
    return enriched



def validate_data(data: list[dict], resource: str) -> bool:
    if not data:
        logger.warning(f"[{resource}] No data received")
        return False
    missing = EXPECTED_COLS.get(resource, set()) - set(data[0].keys())
    if missing:
        logger.warning(f"[{resource}] Missing columns: {missing}")
        return False
    return True



def save_atomic(data: list[dict], filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / filename
    df     = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(
        "w", delete=False, suffix=".csv", dir=RAW_DIR, encoding="utf-8"
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    shutil.move(tmp_path, target)
    logger.info(f"Saved: {target} ({len(df)} rows, {len(df.columns)} cols)")
    return target



def save_collection_report(stats: dict):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now().isoformat(), "base_url": BASE_URL, **stats}
    path   = RAW_DIR / "collection_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Collection report saved: {path}")



def main():
    logger.info("Starting DefectDojo data collection — AI Risk Engine")
    logger.info(f"Target: {BASE_URL}")

    session = create_session()
    stats   = {"products": 0, "engagements": 0, "tests": 0, "findings": 0, "errors": []}

    products = fetch_products(session)
    if not validate_data(products, "products"):
        logger.error("Invalid products data — aborting")
        sys.exit(1)
    save_atomic(products, "products.csv")
    stats["products"] = len(products)

    engagements = fetch_engagements(session)
    validate_data(engagements, "engagements")
    save_atomic(engagements, "engagements.csv")
    stats["engagements"] = len(engagements)

    tests = fetch_tests(session)
    validate_data(tests, "tests")
    save_atomic(tests, "tests.csv")
    stats["tests"] = len(tests)

    lookup = build_lookup_table(tests, engagements, products)

    raw_findings = fetch_all_findings(session)
    if not raw_findings:
        logger.error("No findings retrieved — aborting")
        sys.exit(1)
    validate_data(raw_findings, "findings")

    findings = enrich_findings(raw_findings, lookup)
    findings = enrich_findings_with_epss(findings)

    dist: dict[str, int] = {}
    for f in findings:
        key      = f.get("product_name") or f"product_id={f.get('product_id', '?')}"
        dist[key] = dist.get(key, 0) + 1

    stats["findings"]                 = len(findings)
    stats["distribution_by_product"]  = dist

    save_atomic(findings, "findings_raw.csv")

    df_all = pd.DataFrame(findings)
    if "product_name" in df_all.columns:
        for prod_name, group in df_all.groupby("product_name"):
            safe = "".join(c if c.isalnum() else "_" for c in str(prod_name))
            save_atomic(group.to_dict("records"), f"findings_{safe}.csv")

    save_collection_report(stats)

    logger.info(
        f"Collection complete — "
        f"{stats['products']} products, "
        f"{stats['engagements']} engagements, "
        f"{stats['tests']} tests, "
        f"{stats['findings']} findings"
    )


if __name__ == "__main__":
    main()