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

BASE_URL   = os.getenv('DEFECTDOJO_URL', '').rstrip('/')
API_TOKEN  = os.getenv('DEFECTDOJO_TOKEN', '')
PAGE_LIMIT = int(os.getenv('DEFECTDOJO_PAGE_LIMIT', '100'))
TIMEOUT    = int(os.getenv('DEFECTDOJO_TIMEOUT', '30'))

if not BASE_URL or not API_TOKEN:
    print("Variables DEFECTDOJO_URL et DEFECTDOJO_TOKEN requises dans .env")
    sys.exit(1)

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/fetch_data.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path('data/raw')

EXPECTED_COLS = {
    'products':    {'id', 'name'},
    'engagements': {'id', 'product'},
    'tests':       {'id', 'engagement'},
    'findings':    {'id', 'title', 'severity'},
}


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION
# ─────────────────────────────────────────────────────────────────────────────

def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    session.headers.update({
        'Authorization': f'Token {API_TOKEN}',
        'Content-Type':  'application/json',
        'Accept':        'application/json',
    })
    return session


# ─────────────────────────────────────────────────────────────────────────────
#  PAGINATION
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_pages(
    session:  requests.Session,
    url:      str,
    resource: str,
) -> list[dict]:
    results: list[dict] = []
    page = 1

    while url:
        try:
            resp = session.get(url, timeout=TIMEOUT)

            if resp.status_code == 429:
                wait = int(resp.headers.get('Retry-After', 10))
                logger.warning(f"[{resource}] Rate limit — attente {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            batch = data.get('results', [])
            results.extend(batch)

            url = data.get('next')
            page += 1
            if url:
                time.sleep(0.3)

        except requests.exceptions.Timeout:
            logger.error(f"[{resource}] Timeout page {page} — arrêt")
            break
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{resource}] HTTP {e.response.status_code} : {e}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"[{resource}] Erreur réseau : {e}")
            break
        except Exception as e:
            logger.error(f"[{resource}] Erreur inattendue : {e}")
            break

    logger.info(f"[{resource}] Total : {len(results)} éléments ({page-1} page(s))")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  FETCH HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_products(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/products/?limit={PAGE_LIMIT}", 'products')


def fetch_engagements(session: requests.Session) -> list[dict]:
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/engagements/?limit={PAGE_LIMIT}", 'engagements')


def fetch_tests(session: requests.Session) -> list[dict]:
    """
    Récupère tous les Tests DefectDojo.

    HIÉRARCHIE DefectDojo :
        Product  →  Engagement  →  Test  →  Finding

    Un finding expose uniquement 'test' (test_id).
    Le Test expose 'engagement' (engagement_id).
    L'Engagement expose 'product' (product_id).

    Cette table est donc le SEUL pont entre findings et produits
    dans cette instance DefectDojo.
    """
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/tests/?limit={PAGE_LIMIT}", 'tests')


def fetch_all_findings(session: requests.Session) -> list[dict]:
    """Récupère tous les findings en une seule passe globale sans filtre."""
    return fetch_all_pages(session, f"{BASE_URL}/api/v2/findings/?limit={PAGE_LIMIT}", 'findings_all')


# ─────────────────────────────────────────────────────────────────────────────
#  TABLE DE JOINTURE  test_id → engagement_id → product_id → product_name
# ─────────────────────────────────────────────────────────────────────────────

def build_lookup_table(
    tests:       list[dict],
    engagements: list[dict],
    products:    list[dict],
) -> dict[int, dict]:
    """
    Construit un dictionnaire :
        test_id  →  {
            'engagement_id': int,
            'engagement_name': str,
            'product_id': int,
            'product_name': str,
        }

    Résolution en deux passes :
        1. test_id      → engagement_id  (via tests)
        2. engagement_id → product_id   (via engagements)
        3. product_id   → product_name  (via products)
    """
    # Index engagement_id → engagement dict
    eng_map: dict[int, dict] = {}
    for e in engagements:
        eid = e.get('id')
        if eid is not None:
            eng_map[int(eid)] = e

    # Index product_id → product dict
    prod_map: dict[int, dict] = {}
    for p in products:
        pid = p.get('id')
        if pid is not None:
            prod_map[int(pid)] = p

    lookup: dict[int, dict] = {}
    unresolved_tests = 0

    for t in tests:
        test_id = t.get('id')
        eng_id  = t.get('engagement')

        if test_id is None:
            continue

        test_id = int(test_id)

        # Résolution engagement
        if eng_id is None:
            unresolved_tests += 1
            lookup[test_id] = {
                'engagement_id':   None,
                'engagement_name': '',
                'product_id':      None,
                'product_name':    'Unknown',
            }
            continue

        eng_id = int(eng_id)
        eng    = eng_map.get(eng_id, {})

        # Résolution product
        prod_id = eng.get('product')
        if prod_id is not None:
            prod_id = int(prod_id)

        prod      = prod_map.get(prod_id, {}) if prod_id else {}
        prod_name = prod.get('name', f'Unknown-prod{prod_id}') if prod_id else 'Unknown'
        eng_name  = eng.get('name', f'engagement-{eng_id}')

        lookup[test_id] = {
            'engagement_id':   eng_id,
            'engagement_name': eng_name,
            'product_id':      prod_id,
            'product_name':    prod_name,
        }

    if unresolved_tests:
        logger.warning(f"{unresolved_tests} test(s) sans engagement_id dans la table de lookup")

    logger.info(f"Table de lookup construite : {len(lookup)} test_id mappés")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
#  ENRICHISSEMENT DES FINDINGS
# ─────────────────────────────────────────────────────────────────────────────

def enrich_findings(
    findings: list[dict],
    lookup:   dict[int, dict],
) -> list[dict]:
    """
    Injecte product_id, product_name, engagement_id, engagement_name
    sur chaque finding via la table de lookup test_id → product.

    C'est la seule méthode fiable quand l'API n'expose que 'test' sur
    le finding.
    """
    enriched   = []
    unresolved = 0

    for f in findings:
        row     = dict(f)
        test_id = f.get('test')

        if test_id is not None:
            test_id = int(test_id)
            info    = lookup.get(test_id)
        else:
            info = None

        if info:
            row['product_id']      = info['product_id']
            row['product_name']    = info['product_name']
            row['engagement_id']   = info['engagement_id']
            row['engagement_name'] = info['engagement_name']
        else:
            row['product_id']      = None
            row['product_name']    = 'Unknown'
            row['engagement_id']   = None
            row['engagement_name'] = ''
            unresolved += 1

        enriched.append(row)

    if unresolved:
        logger.warning(
            f"{unresolved} finding(s) avec test_id absent de la table de lookup. "
            f"Vérifier que /api/v2/tests/ retourne bien tous les tests."
        )

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
#  DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def log_distribution(findings: list[dict]) -> dict:
    dist: dict[str, int] = {}
    for f in findings:
        key = f.get('product_name') or f"product_id={f.get('product_id', '?')}"
        dist[key] = dist.get(key, 0) + 1

    logger.info("═" * 52)
    logger.info("  Distribution réelle des findings par produit")
    logger.info("═" * 52)
    for name, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * min(count // 10, 40)
        logger.info(f"  {name:<35} {count:>5}  {bar}")
    logger.info("═" * 52)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_data(data: list[dict], resource: str) -> bool:
    if not data:
        logger.warning(f"[{resource}] Aucune donnée reçue.")
        return False
    actual_cols = set(data[0].keys())
    expected    = EXPECTED_COLS.get(resource, set())
    missing     = expected - actual_cols
    if missing:
        logger.warning(f"[{resource}] Colonnes manquantes : {missing}")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  SAUVEGARDE ATOMIQUE
# ─────────────────────────────────────────────────────────────────────────────

def save_atomic(data: list[dict], filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / filename
    df     = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(
        'w', delete=False, suffix='.csv', dir=RAW_DIR, encoding='utf-8'
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    shutil.move(tmp_path, target)
    logger.info(f"💾 Sauvegardé : {target} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return target


# ─────────────────────────────────────────────────────────────────────────────
#  RAPPORT JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_collection_report(stats: dict):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = {'timestamp': datetime.now().isoformat(), 'base_url': BASE_URL, **stats}
    path   = RAW_DIR / 'collection_report.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📋 Rapport de collecte : {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info(" Récupération des données DefectDojo — AI Risk Engine")
    logger.info(f"   URL : {BASE_URL}")
    logger.info("=" * 60)

    session = create_session()
    stats   = {'products': 0, 'engagements': 0, 'tests': 0, 'findings': 0, 'errors': []}

    # ── 1. Produits ───────────────────────────────────────────────────────
    products = fetch_products(session)
    if not validate_data(products, 'products'):
        logger.error("Données produits invalides — arrêt.")
        sys.exit(1)
    save_atomic(products, 'products.csv')
    stats['products'] = len(products)
    logger.info(f"Produits : {[p['name'] for p in products]}")

    # ── 2. Engagements ────────────────────────────────────────────────────
    engagements = fetch_engagements(session)
    validate_data(engagements, 'engagements')
    save_atomic(engagements, 'engagements.csv')
    stats['engagements'] = len(engagements)
    logger.info(f"Engagements : {len(engagements)}")
    for e in engagements:
        logger.info(f"   engagement id={e['id']}  product={e.get('product')}  name={e.get('name')}")

    # ── 3. Tests — pont entre findings et engagements ─────────────────────
    logger.info("Récupération des Tests (pont finding → engagement → product)…")
    tests = fetch_tests(session)
    validate_data(tests, 'tests')
    save_atomic(tests, 'tests.csv')
    stats['tests'] = len(tests)
    logger.info(f"Tests récupérés : {len(tests)}")
    for t in tests:
        logger.info(f"   test id={t['id']}  engagement={t.get('engagement')}  title={t.get('title','?')}")

    # ── 4. Table de lookup test_id → product ──────────────────────────────
    lookup = build_lookup_table(tests, engagements, products)

    # ── 5. Findings — passe unique globale ───────────────────────────────
    logger.info("Récupération globale des findings (passe unique sans filtre)…")
    raw_findings = fetch_all_findings(session)
    if not raw_findings:
        logger.error("Aucun finding récupéré — arrêt.")
        sys.exit(1)
    validate_data(raw_findings, 'findings')

    # ── 6. Enrichissement via lookup ──────────────────────────────────────
    findings = enrich_findings(raw_findings, lookup)

    # ── 7. Distribution réelle ────────────────────────────────────────────
    dist = log_distribution(findings)
    stats['findings']                = len(findings)
    stats['distribution_by_product'] = dist

    # ── 8. Sauvegarde globale ─────────────────────────────────────────────
    save_atomic(findings, 'findings_raw.csv')

    # ── 9. Un CSV par produit ─────────────────────────────────────────────
    df_all = pd.DataFrame(findings)
    if 'product_name' in df_all.columns:
        logger.info("Sauvegarde individuelle par produit :")
        for prod_name, group in df_all.groupby('product_name'):
            safe = "".join(c if c.isalnum() else "_" for c in str(prod_name))
            save_atomic(group.to_dict('records'), f'findings_{safe}.csv')

    # ── 10. Rapport ───────────────────────────────────────────────────────
    save_collection_report(stats)

    logger.info("\n" + "=" * 60)
    logger.info(" Collecte terminée")
    logger.info(f"   {stats['products']}  produits")
    logger.info(f"   {stats['engagements']}  engagements")
    logger.info(f"   {stats['tests']}  tests")
    logger.info(f"   {stats['findings']} findings")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()