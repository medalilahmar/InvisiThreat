import ast
import json
import logging
import os
import shutil
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/preprocess.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
RAW_DIR       = Path('data/raw')
PROCESSED_DIR = Path('data/processed')

SEVERITY_MAP = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'Info': 0}

# Colonnes finales exportées
KEEP_COLS = [
    # Identifiants
    'id', 'title', 'product_id', 'engagement_id',
    # Features ML
    'severity_num', 'cvss_score', 'age_days',
    'has_cve', 'has_cwe', 'tags_count',
    'is_false_positive', 'is_active',
    # Features contextuelles enrichies
    'severity_x_active', 'product_fp_rate', 'cvss_severity_gap',
    # Score métier (règle déterministe, pas cible ML)
    'composite_risk',
    # Vraie cible ML
    'days_to_fix', 'fixed_in_30d',
]


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les trois CSV bruts avec validation d'existence."""
    files = {
        'findings':    RAW_DIR / 'findings_raw.csv',
        'products':    RAW_DIR / 'products.csv',
        'engagements': RAW_DIR / 'engagements.csv',
    }
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")

    findings    = pd.read_csv(files['findings'])
    products    = pd.read_csv(files['products'])
    engagements = pd.read_csv(files['engagements'])

    logger.info(f"Findings bruts  : {len(findings)} lignes")
    logger.info(f"Produits        : {len(products)} lignes")
    logger.info(f"Engagements     : {len(engagements)} lignes")
    return findings, products, engagements


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def safe_col(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    """Retourne la colonne si elle existe, sinon une série constante."""
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


def count_tags(tag_field) -> int:
    """Compte le nombre de tags dans un champ texte ou liste."""
    if pd.isna(tag_field):
        return 0
    if isinstance(tag_field, list):
        return len(tag_field)
    if isinstance(tag_field, str):
        try:
            parsed = ast.literal_eval(tag_field)
            if isinstance(parsed, list):
                return len(parsed)
        except (ValueError, SyntaxError):
            pass
        return len(tag_field.split(',')) if tag_field.strip() else 0
    return 0


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def _normalize_tz(series: pd.Series) -> pd.Series:
    """
    Normalise une série de dates : supprime le fuseau horaire (UTC → naive).
    DefectDojo retourne des dates tz-aware (ex: 2024-01-15T10:00:00Z),
    pandas.Timestamp.now() est tz-naive → il faut uniformiser avant toute soustraction.
    """
    if series.dt.tz is not None:
        return series.dt.tz_convert('UTC').dt.tz_localize(None)
    return series


def build_date_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calcule age_days (découverte → aujourd'hui) et days_to_fix (→ correction)."""
    now = pd.Timestamp(datetime.utcnow())   # toujours UTC tz-naive

    # Colonne de découverte
    date_col = next((c for c in ['date', 'created'] if c in data.columns), None)
    if date_col:
        discovery = _normalize_tz(pd.to_datetime(data[date_col], errors='coerce'))
        raw_age   = (now - discovery).dt.days.fillna(0).clip(lower=0)
        # Clamp au percentile 99 pour éviter les outliers aberrants
        p99 = raw_age.quantile(0.99) if raw_age.max() > 0 else 1
        data['age_days'] = raw_age.clip(upper=p99)
        logger.info(f"age_days : médiane={data['age_days'].median():.0f}j, max clampé={p99:.0f}j")
    else:
        data['age_days'] = 0
        discovery = None
        logger.warning("Colonne de date absente — age_days forcé à 0")

    # Délai de correction (vraie cible ML)
    fix_col = next((c for c in ['mitigated_date', 'last_reviewed'] if c in data.columns), None)
    if fix_col and discovery is not None:
        fix_date            = _normalize_tz(pd.to_datetime(data[fix_col], errors='coerce'))
        data['days_to_fix'] = (fix_date - discovery).dt.days
    else:
        data['days_to_fix'] = np.nan
        if fix_col is None:
            logger.warning(
                "Colonne de correction absente (mitigated_date / last_reviewed). "
                "days_to_fix = NaN → fixed_in_30d utilisera le fallback composite_risk."
            )

    return data


def build_severity_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encode la sévérité et calcule le gap avec le score CVSS."""
    if 'severity' in data.columns:
        data['severity_num'] = data['severity'].map(SEVERITY_MAP).fillna(0).astype(int)
    else:
        data['severity_num'] = 0

    # CVSS
    cvss_col = next((c for c in ['cvssv3_score', 'cvss_score'] if c in data.columns), None)
    if cvss_col:
        data['cvss_score'] = pd.to_numeric(data[cvss_col], errors='coerce').fillna(0).clip(0, 10)
    else:
        data['cvss_score'] = 0.0

    # Gap entre CVSS normalisé [0-4] et sévérité interne → détecte les incohérences
    cvss_norm             = data['cvss_score'] / 10 * 4
    data['cvss_severity_gap'] = (cvss_norm - data['severity_num']).abs().round(3)

    return data


def build_binary_features(data: pd.DataFrame) -> pd.DataFrame:
    """CVE, CWE, faux positifs, statut actif."""
    data['has_cve']          = data['cve'].notna().astype(int) if 'cve' in data.columns else 0
    data['has_cwe']          = data['cwe'].notna().astype(int) if 'cwe' in data.columns else 0
    data['is_false_positive'] = safe_col(data, 'false_p', False).fillna(False).astype(int)
    data['is_active']         = safe_col(data, 'active', True).fillna(True).astype(int)
    data['tags_count']        = (
        data['tags'].apply(count_tags) if 'tags' in data.columns
        else pd.Series(0, index=data.index)
    )
    return data


def build_contextual_features(data: pd.DataFrame) -> pd.DataFrame:
    """Features enrichies croisant plusieurs dimensions."""
    # Interaction sévérité × actif
    data['severity_x_active'] = data['severity_num'] * data['is_active']

    # Taux de faux positifs par produit (bruit moyen du produit)
    data['product_fp_rate'] = data.groupby('product_id')['is_false_positive'].transform('mean').round(4)

    return data


def build_composite_risk(data: pd.DataFrame) -> pd.DataFrame:
    """
    Score de risque métier déterministe (règle d'expert).
    Conservé pour affichage dashboard — ce N'EST PAS la cible ML.
    """
    data['composite_risk'] = (
        data['cvss_score']          * 0.4  +
        data['severity_num']        * 1.5  +
        (data['age_days'] / 30)     * 0.2  +
        data['has_cve']             * 2.0  +
        data['tags_count']          * 0.1
    ) / 5.0
    return data


def build_ml_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la vraie variable cible ML : fixed_in_30d.
      1 = finding corrigé en ≤ 30 jours
      0 = non corrigé ou toujours actif

    Fallback automatique si days_to_fix est absent :
      On seuille composite_risk à sa médiane (proxy temporaire).
    """
    if data['days_to_fix'].notna().sum() > 0:
        data['fixed_in_30d'] = data['days_to_fix'].between(0, 30).astype(int)
        coverage = data['fixed_in_30d'].notna().mean() * 100
        logger.info(f"Cible ML : fixed_in_30d (couverture {coverage:.1f}%)")
    else:
        logger.warning(
            "⚠️  Fallback cible ML : composite_risk seuillé à la médiane. "
            "Remplacez par des dates réelles dès que possible."
        )
        median = data['composite_risk'].median()
        data['fixed_in_30d'] = (data['composite_risk'] >= median).astype(int)

    ratio = data['fixed_in_30d'].mean() * 100
    logger.info(f"Distribution fixed_in_30d → positifs : {ratio:.1f}%  |  négatifs : {100-ratio:.1f}%")
    return data


# ---------------------------------------------------------------------------
# 4. Pipeline principal de prétraitement
# ---------------------------------------------------------------------------

def preprocess_findings(
    findings: pd.DataFrame,
    products: pd.DataFrame,
    engagements: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prétraitement complet : nettoyage, features, cible, jointures.
    Retourne un DataFrame prêt pour l'entraînement.
    """
    data = findings.copy()

    # IDs de contexte
    data['product_id']    = safe_col(data, 'product_id', 0)
    data['engagement_id'] = safe_col(data, 'engagement_id', 0)

    # Enrichissement depuis products (ex: criticité du produit)
    if 'id' in products.columns:
        prod_meta = products[['id']].copy()
        prod_meta = prod_meta.rename(columns={'id': 'product_id'})
        # Ajouter d'autres colonnes produit ici si disponibles (ex: business_criticality)
        data = data.merge(prod_meta, on='product_id', how='left')

    # Features
    data = build_date_features(data)
    data = build_severity_features(data)
    data = build_binary_features(data)
    data = build_contextual_features(data)
    data = build_composite_risk(data)
    data = build_ml_target(data)

    # Sélection des colonnes finales (uniquement celles présentes)
    final_cols = [c for c in KEEP_COLS if c in data.columns]
    result     = data[final_cols].copy()

    logger.info(f"Findings nettoyés : {len(result)} lignes, {len(result.columns)} colonnes")
    return result


# ---------------------------------------------------------------------------
# 5. Validation du DataFrame final
# ---------------------------------------------------------------------------

def validate_output(df: pd.DataFrame) -> bool:
    """Vérifie la cohérence minimale du DataFrame avant sauvegarde."""
    ok = True

    # Colonnes ML obligatoires
    required = ['severity_num', 'cvss_score', 'age_days', 'fixed_in_30d']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"Colonnes obligatoires manquantes : {missing}")
        ok = False

    # Pas de DataFrame vide
    if len(df) == 0:
        logger.error("DataFrame vide après prétraitement.")
        ok = False

    # Rapport valeurs manquantes
    null_pct = df.isnull().mean() * 100
    high_null = null_pct[null_pct > 30]
    if not high_null.empty:
        logger.warning(f"Colonnes avec >30% de NaN :\n{high_null.to_string()}")

    return ok


# ---------------------------------------------------------------------------
# 6. Sauvegarde atomique
# ---------------------------------------------------------------------------

def save_atomic(df: pd.DataFrame, path: Path):
    """
    Sauvegarde atomique via fichier temporaire :
    évite un CSV corrompu si le processus est interrompu.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.csv', dir=path.parent) as tmp:
        df.to_csv(tmp.name, index=False)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    logger.info(f"💾 Sauvegardé : {path} ({len(df)} lignes)")


def save_data_report(df: pd.DataFrame):
    """Exporte un rapport JSON de qualité des données."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    report = {
        'timestamp':   datetime.now().isoformat(),
        'n_rows':      len(df),
        'n_cols':      len(df.columns),
        'columns':     list(df.columns),
        'null_pct':    df.isnull().mean().round(4).to_dict(),
        'target_dist': df['fixed_in_30d'].value_counts(normalize=True).round(4).to_dict()
                       if 'fixed_in_30d' in df.columns else {},
        'severity_dist': df['severity_num'].value_counts().to_dict()
                         if 'severity_num' in df.columns else {},
    }
    report_path = PROCESSED_DIR / 'data_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"📋 Rapport qualité : {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("🧹 Prétraitement des données — AI Risk Engine")
    logger.info("=" * 60)

    # 1. Chargement
    findings, products, engagements = load_data()

    # 2. Prétraitement complet
    df_clean = preprocess_findings(findings, products, engagements)

    # 3. Validation
    if not validate_output(df_clean):
        logger.error("❌ Validation échouée — vérifiez les données brutes.")
        raise SystemExit(1)

    # 4. Sauvegarde atomique
    output_path = PROCESSED_DIR / 'findings_clean.csv'
    save_atomic(df_clean, output_path)

    # 5. Rapport qualité
    save_data_report(df_clean)

    # 6. Aperçu
    logger.info("\n🔍 Aperçu des 3 premières lignes :")
    logger.info(f"\n{df_clean.head(3).to_string()}")

    logger.info("\n📊 Statistiques numériques :")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    logger.info(f"\n{df_clean[numeric_cols].describe().round(3).to_string()}")

    logger.info("\n✅ Prétraitement terminé avec succès.")


if __name__ == '__main__':
    main()