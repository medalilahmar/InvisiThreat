import json
import logging
import os
import re
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

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

BASE_URL        = os.getenv("DEFECTDOJO_URL", "").rstrip("/")
API_TOKEN       = os.getenv("DEFECTDOJO_API_KEY", "")
PAGE_LIMIT      = int(os.getenv("DEFECTDOJO_PAGE_LIMIT", "100"))
TIMEOUT         = int(os.getenv("DEFECTDOJO_TIMEOUT", "30"))
EPSS_API_URL    = "https://api.first.org/data/v1/epss"
EPSS_BATCH_SIZE = 100
EPSS_TIMEOUT    = 15
EPSS_CACHE_FILE = ROOT_DIR / "data" / "epss_cache.json"

CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)

if not BASE_URL or not API_TOKEN:
    print("Variables DEFECTDOJO_URL et DEFECTDOJO_API_KEY requises dans .env")
    sys.exit(1)

LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "fetch_data.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = ROOT_DIR / "data" / "raw"

EXPECTED_COLS = {
    "products":    {"id", "name"},
    "engagements": {"id", "product"},
    "tests":       {"id", "engagement"},
    "findings":    {"id", "title", "severity"},
}



def load_epss_cache() -> dict:
    """Charge le cache EPSS depuis le disque (CVE → {epss_score, epss_percentile})."""
    if EPSS_CACHE_FILE.exists():
        try:
            with open(EPSS_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Cache EPSS chargé : {len(cache)} entrée(s)")
            return cache
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache EPSS illisible, réinitialisation : {e}")
    return {}


def save_epss_cache(cache: dict):
    """Sauvegarde atomique du cache EPSS."""
    EPSS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".json",
            dir=EPSS_CACHE_FILE.parent, encoding="utf-8"
        ) as tmp:
            json.dump(cache, tmp, indent=2)
            tmp_path = tmp.name
        shutil.move(tmp_path, EPSS_CACHE_FILE)
        logger.info(f"Cache EPSS sauvegardé : {len(cache)} entrée(s)")
    except OSError as e:
        logger.error(f"Impossible de sauvegarder le cache EPSS : {e}")
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)



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
                logger.warning(f"[{resource}] Rate limit — attente {wait}s")
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
            logger.error(f"[{resource}] Erreur réseau : {e}")
            break
        except Exception as e:
            logger.error(f"[{resource}] Erreur inattendue : {e}")
            break

    logger.info(f"[{resource}] {len(results)} éléments récupérés ({page - 1} page(s))")
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
    eng_map:  dict[int, dict] = {int(e["id"]): e for e in engagements if e.get("id") is not None}
    prod_map: dict[int, dict] = {int(p["id"]): p for p in products   if p.get("id") is not None}

    lookup:    dict[int, dict] = {}
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
        logger.warning(f"{unresolved} test(s) sans engagement_id dans la lookup table")

    logger.info(f"Lookup table construite : {len(lookup)} test_id(s) mappés")
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
        logger.warning(f"{unresolved} finding(s) avec test_id non résolu")

    return enriched



def _extract_cve(finding: dict) -> Optional[str]:
   

    def _normalize(val) -> Optional[str]:
        """Retourne un CVE normalisé si val en contient un, sinon None."""
        if not val:
            return None
        s = str(val).strip()
        m = CVE_PATTERN.search(s)
        return m.group(0).upper() if m else None

    cve = _normalize(finding.get("cve"))
    if cve:
        return cve

    vuln_ids = finding.get("vulnerability_ids")

    if vuln_ids:
        if isinstance(vuln_ids, str):
            stripped = vuln_ids.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    vuln_ids = json.loads(stripped)
                except json.JSONDecodeError:
                    cve = _normalize(stripped)
                    if cve:
                        return cve
                    vuln_ids = []
            else:
                cve = _normalize(stripped)
                if cve:
                    return cve
                vuln_ids = []

        if isinstance(vuln_ids, list):
            for entry in vuln_ids:
                if isinstance(entry, dict):
                    for key in ("vulnerability_id", "cve", "id", "name", "value"):
                        cve = _normalize(entry.get(key))
                        if cve:
                            return cve
                    for v in entry.values():
                        cve = _normalize(v)
                        if cve:
                            return cve
                else:
                    cve = _normalize(entry)
                    if cve:
                        return cve

        elif isinstance(vuln_ids, dict):
            for key in ("vulnerability_id", "cve", "id", "name", "value"):
                cve = _normalize(vuln_ids.get(key))
                if cve:
                    return cve

    for field in ("title", "description"):
        cve = _normalize(finding.get(field, ""))
        if cve:
            return cve

    return None


# ─── EPSS BATCH + CACHE ───────────────────────────────────────────────────────

def batch_fetch_epss(cves: list[str], cache: dict) -> dict:
    
    to_fetch = [c for c in cves if c not in cache]

    if not to_fetch:
        logger.info("EPSS : tous les CVE sont déjà en cache")
        return cache

    logger.info(f"EPSS : {len(to_fetch)} CVE à récupérer (cache : {len(cache)})")

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))

    fetched = 0
    for i in range(0, len(to_fetch), EPSS_BATCH_SIZE):
        batch  = to_fetch[i : i + EPSS_BATCH_SIZE]
        params = {"cve": ",".join(batch), "limit": EPSS_BATCH_SIZE}
        try:
            resp = session.get(EPSS_API_URL, params=params, timeout=EPSS_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                cve_id = item.get("cve", "").upper()
                if cve_id:
                    cache[cve_id] = {
                        "epss_score":       float(item.get("epss",       0.0)),
                        "epss_percentile":  float(item.get("percentile", 0.0)),
                    }
                    fetched += 1

            
            returned_cves = {item.get("cve", "").upper() for item in data.get("data", [])}
            for cve_id in batch:
                if cve_id not in returned_cves:
                    cache.setdefault(cve_id, {"epss_score": 0.0, "epss_percentile": 0.0})

        except requests.exceptions.Timeout:
            logger.warning(f"EPSS batch {i // EPSS_BATCH_SIZE + 1} : timeout")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"EPSS batch {i // EPSS_BATCH_SIZE + 1} : HTTP {e.response.status_code}")
        except Exception as e:
            logger.warning(f"EPSS batch {i // EPSS_BATCH_SIZE + 1} : {e}")

        time.sleep(0.5)

    logger.info(f"EPSS : {fetched} score(s) récupérés depuis l'API FIRST")
    return cache



def enrich_findings_with_epss(findings: list[dict]) -> list[dict]:
   
    cve_map: dict[int, str] = {}  
    for idx, f in enumerate(findings):
        cve = _extract_cve(f)
        if cve:
            cve_map[idx] = cve

    all_cves = list(set(cve_map.values()))
    logger.info(
        f"EPSS : {len(cve_map)} finding(s) avec CVE "
        f"({len(all_cves)} CVE unique(s) sur {len(findings)} findings)"
    )

    if not all_cves:
        logger.warning("Aucun CVE détecté — vérifier les champs vulnerability_ids / cve")
        for f in findings:
            f.setdefault("epss_score",      0.0)
            f.setdefault("epss_percentile", 0.0)
        return findings

    cache = load_epss_cache()
    cache = batch_fetch_epss(all_cves, cache)
    save_epss_cache(cache)

    enriched_count = 0
    for idx, f in enumerate(findings):
        cve = cve_map.get(idx)
        if cve and cve in cache:
            entry = cache[cve]
            f["epss_score"]      = entry["epss_score"]
            f["epss_percentile"] = entry["epss_percentile"]
            if entry["epss_score"] > 0:
                enriched_count += 1
        else:
            f.setdefault("epss_score",      0.0)
            f.setdefault("epss_percentile", 0.0)

    logger.info(
        f"EPSS : {enriched_count}/{len(cve_map)} finding(s) enrichis "
        f"avec un score > 0"
    )
    return findings


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def validate_data(data: list[dict], resource: str) -> bool:
    if not data:
        logger.warning(f"[{resource}] Aucune donnée reçue")
        return False
    missing = EXPECTED_COLS.get(resource, set()) - set(data[0].keys())
    if missing:
        logger.warning(f"[{resource}] Colonnes manquantes : {missing}")
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
    logger.info(f"Sauvegardé : {target} ({len(df)} lignes, {len(df.columns)} cols)")
    return target


def save_collection_report(stats: dict):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": datetime.now().isoformat(), "base_url": BASE_URL, **stats}
    path   = RAW_DIR / "collection_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Rapport de collecte sauvegardé : {path}")



def main():
    logger.info("Démarrage de la collecte DefectDojo — AI Risk Engine")
    logger.info(f"Cible : {BASE_URL}")

    session = create_session()
    stats   = {"products": 0, "engagements": 0, "tests": 0, "findings": 0, "errors": []}

    products = fetch_products(session)
    if not validate_data(products, "products"):
        logger.error("Données produits invalides — abandon")
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
        logger.error("Aucun finding récupéré — abandon")
        sys.exit(1)
    validate_data(raw_findings, "findings")

    findings = enrich_findings(raw_findings, lookup)
    findings = enrich_findings_with_epss(findings)

    dist: dict[str, int] = {}
    for f in findings:
        key      = f.get("product_name") or f"product_id={f.get('product_id', '?')}"
        dist[key] = dist.get(key, 0) + 1

    with_cve   = sum(1 for f in findings if _extract_cve(f))
    with_epss  = sum(1 for f in findings if (f.get("epss_score") or 0) > 0)
    stats["findings"]                 = len(findings)
    stats["findings_with_cve"]        = with_cve
    stats["findings_with_epss_score"] = with_epss
    stats["distribution_by_product"]  = dist

    save_atomic(findings, "findings_raw.csv")

    df_all = pd.DataFrame(findings)
    if "product_name" in df_all.columns:
        for prod_name, group in df_all.groupby("product_name"):
            safe = "".join(c if c.isalnum() else "_" for c in str(prod_name))
            save_atomic(group.to_dict("records"), f"findings_{safe}.csv")

    save_collection_report(stats)

    logger.info(
        f"Collecte terminée — "
        f"{stats['products']} produits, "
        f"{stats['engagements']} engagements, "
        f"{stats['tests']} tests, "
        f"{stats['findings']} findings "
        f"({with_cve} avec CVE, {with_epss} avec score EPSS > 0)"
    )


if __name__ == "__main__":
    main()