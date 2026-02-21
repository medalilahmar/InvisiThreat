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

# ---------------------------------------------------------------------------
# Chargement de l'environnement
# ---------------------------------------------------------------------------
load_dotenv()

BASE_URL   = os.getenv('DEFECTDOJO_URL', '').rstrip('/')
API_TOKEN  = os.getenv('DEFECTDOJO_TOKEN', '')
PAGE_LIMIT = int(os.getenv('DEFECTDOJO_PAGE_LIMIT', '100'))
TIMEOUT    = int(os.getenv('DEFECTDOJO_TIMEOUT', '30'))

if not BASE_URL or not API_TOKEN:
    print("❌ Variables DEFECTDOJO_URL et DEFECTDOJO_TOKEN requises dans .env")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Répertoires
# ---------------------------------------------------------------------------
RAW_DIR = Path('data/raw')

# ---------------------------------------------------------------------------
# Colonnes minimales attendues par ressource (pour validation)
# ---------------------------------------------------------------------------
EXPECTED_COLS = {
    'products':    {'id', 'name'},
    'engagements': {'id', 'product'},
    'findings':    {'id', 'title', 'severity'},
}


# ---------------------------------------------------------------------------
# 1. Session HTTP robuste
# ---------------------------------------------------------------------------

def create_session() -> requests.Session:
    """
    Crée une session HTTP réutilisable avec :
      - retry exponentiel sur les erreurs 429, 500, 502, 503, 504
      - headers d'authentification injectés une seule fois
    """
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,          # attentes : 1.5s, 3s, 6s, 12s
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


# ---------------------------------------------------------------------------
# 2. Pagination générique
# ---------------------------------------------------------------------------

def fetch_all_pages(
    session:  requests.Session,
    url:      str,
    resource: str,
) -> list[dict]:
    """
    Parcourt toutes les pages d'un endpoint paginé DefectDojo.
    Gère le rate limiting via l'en-tête Retry-After.
    """
    results: list[dict] = []
    page = 1

    while url:
        logger.debug(f"[{resource}] page {page} → {url}")
        try:
            resp = session.get(url, timeout=TIMEOUT)

            # Rate limiting explicite
            if resp.status_code == 429:
                wait = int(resp.headers.get('Retry-After', 10))
                logger.warning(f"[{resource}] Rate limit — attente {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            batch = data.get('results', [])
            results.extend(batch)
            logger.debug(f"[{resource}] page {page} : {len(batch)} éléments reçus")

            url = data.get('next')
            page += 1

            # Politesse inter-requêtes
            if url:
                time.sleep(0.3)

        except requests.exceptions.Timeout:
            logger.error(f"[{resource}] Timeout après {TIMEOUT}s sur page {page} — arrêt")
            break
        except requests.exceptions.HTTPError as e:
            logger.error(f"[{resource}] Erreur HTTP {e.response.status_code} : {e}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"[{resource}] Erreur réseau : {e}")
            break
        except Exception as e:
            logger.error(f"[{resource}] Erreur inattendue : {e}")
            break

    logger.info(f"[{resource}] Total récupéré : {len(results)} éléments ({page-1} page(s))")
    return results


# ---------------------------------------------------------------------------
# 3. Fonctions métier de récupération
# ---------------------------------------------------------------------------

def fetch_products(session: requests.Session) -> list[dict]:
    url = f"{BASE_URL}/api/v2/products/?limit={PAGE_LIMIT}"
    return fetch_all_pages(session, url, 'products')


def fetch_engagements(
    session:    requests.Session,
    product_id: Optional[int] = None,
) -> list[dict]:
    url = f"{BASE_URL}/api/v2/engagements/?limit={PAGE_LIMIT}"
    if product_id is not None:
        url += f"&product={product_id}"
    return fetch_all_pages(session, url, f'engagements/product={product_id}')


def fetch_findings(
    session:       requests.Session,
    engagement_id: Optional[int] = None,
) -> list[dict]:
    url = f"{BASE_URL}/api/v2/findings/?limit={PAGE_LIMIT}"
    if engagement_id is not None:
        url += f"&engagement={engagement_id}"
    return fetch_all_pages(session, url, f'findings/engagement={engagement_id}')


# ---------------------------------------------------------------------------
# 4. Validation des données
# ---------------------------------------------------------------------------

def validate_data(data: list[dict], resource: str) -> bool:
    """
    Vérifie que les colonnes minimales attendues sont présentes.
    Log un warning si des colonnes critiques manquent.
    """
    if not data:
        logger.warning(f"[{resource}] Aucune donnée reçue.")
        return False

    actual_cols  = set(data[0].keys())
    expected     = EXPECTED_COLS.get(resource, set())
    missing      = expected - actual_cols

    if missing:
        logger.warning(f"[{resource}] Colonnes manquantes : {missing}")
        return False

    return True


# ---------------------------------------------------------------------------
# 5. Sauvegarde atomique
# ---------------------------------------------------------------------------

def save_atomic(data: list[dict], filename: str) -> Path:
    """
    Sauvegarde atomique : écrit dans un fichier temporaire puis déplace.
    Garantit qu'un CSV partiellement écrit n'écrase jamais le fichier précédent.
    """
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


# ---------------------------------------------------------------------------
# 6. Déduplication des findings
# ---------------------------------------------------------------------------

def deduplicate_findings(findings: list[dict]) -> list[dict]:
    """
    Supprime les doublons de findings (même id peut apparaître
    dans plusieurs engagements si l'API le retourne plusieurs fois).
    """
    seen    = set()
    unique  = []
    duplicates = 0

    for f in findings:
        fid = f.get('id')
        if fid not in seen:
            seen.add(fid)
            unique.append(f)
        else:
            duplicates += 1

    if duplicates:
        logger.warning(f"Doublons supprimés : {duplicates} findings")

    logger.info(f"Findings uniques : {len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# 7. Rapport de collecte
# ---------------------------------------------------------------------------

def save_collection_report(stats: dict):
    """Exporte un rapport JSON horodaté récapitulant la collecte."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        'timestamp': datetime.now().isoformat(),
        'base_url':  BASE_URL,
        **stats,
    }
    path = RAW_DIR / 'collection_report.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📋 Rapport de collecte : {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("📡 Récupération des données DefectDojo — AI Risk Engine")
    logger.info(f"   URL : {BASE_URL}")
    logger.info("=" * 60)

    session = create_session()
    stats   = {'products': 0, 'engagements': 0, 'findings': 0, 'errors': []}

    # ── 1. Produits ──────────────────────────────────────────────────────────
    products = fetch_products(session)
    if not validate_data(products, 'products'):
        logger.error("❌ Données produits invalides — arrêt.")
        sys.exit(1)

    save_atomic(products, 'products.csv')
    stats['products'] = len(products)

    # ── 2. Engagements & Findings par produit ────────────────────────────────
    all_engagements: list[dict] = []
    all_findings:    list[dict] = []

    for prod in products:
        prod_id   = prod['id']
        prod_name = prod.get('name', 'N/A')
        logger.info(f"\n🔍 Produit {prod_id} : {prod_name}")

        # Engagements
        engagements = fetch_engagements(session, product_id=prod_id)
        all_engagements.extend(engagements)

        prod_findings_count = 0

        for eng in engagements:
            eng_id   = eng['id']
            findings = fetch_findings(session, engagement_id=eng_id)

            # Injection des IDs de contexte
            for f in findings:
                f['product_id']    = prod_id
                f['engagement_id'] = eng_id
                f['product_name']  = prod_name

            all_findings.extend(findings)
            prod_findings_count += len(findings)
            logger.info(
                f"   ↳ Engagement {eng_id} : {len(findings)} findings"
            )

        logger.info(
            f"   Produit {prod_id} total : "
            f"{len(engagements)} engagement(s), {prod_findings_count} finding(s)"
        )

    # ── 3. Déduplication ─────────────────────────────────────────────────────
    all_findings = deduplicate_findings(all_findings)

    # ── 4. Validation & sauvegarde ───────────────────────────────────────────
    validate_data(all_engagements, 'engagements')
    validate_data(all_findings,    'findings')

    save_atomic(all_engagements, 'engagements.csv')
    save_atomic(all_findings,    'findings_raw.csv')

    stats['engagements'] = len(all_engagements)
    stats['findings']    = len(all_findings)

    # ── 5. Rapport ────────────────────────────────────────────────────────────
    save_collection_report(stats)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Collecte terminée avec succès")
    logger.info(f"   {stats['products']} produits")
    logger.info(f"   {stats['engagements']} engagements")
    logger.info(f"   {stats['findings']} findings uniques")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()