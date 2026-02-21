import os
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        logging.FileHandler('logs/train.log', encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DATA_PATH   = Path('data/processed/findings_clean.csv')
MODELS_DIR  = Path('models')
REPORTS_DIR = Path('reports')

FEATURE_COLS = [
    'severity_num',
    'cvss_score',
    'age_days',
    'has_cve',
    'has_cwe',
    'tags_count',
    'is_false_positive',
    'is_active',
]

PARAM_DIST = {
    'model__n_estimators':     [50, 100, 200, 300],
    'model__max_depth':        [5, 10, 15, None],
    'model__min_samples_leaf': [1, 2, 5, 10],
    'model__max_features':     ['sqrt', 'log2'],
    'model__class_weight':     ['balanced', None],
}


# ---------------------------------------------------------------------------
# Helper : folds adaptatifs
# ---------------------------------------------------------------------------

def _n_splits(y: pd.Series) -> int:
    """min(5, taille classe minoritaire) — evite n_splits > n_samples."""
    min_class_count = int(y.value_counts().min())
    return max(2, min(5, min_class_count))


def _has_two_classes(series: pd.Series) -> bool:
    return series.nunique() >= 2


# ---------------------------------------------------------------------------
# 1. Chargement
# ---------------------------------------------------------------------------

def load_clean_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Donnees chargees : {len(df)} lignes, {len(df.columns)} colonnes")
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")
    return df


# ---------------------------------------------------------------------------
# 2. Diagnostic des colonnes de dates
# ---------------------------------------------------------------------------

def _diagnose_days_to_fix(df: pd.DataFrame):
    """Log un diagnostic complet de la colonne days_to_fix."""
    if 'days_to_fix' not in df.columns:
        logger.info("days_to_fix : colonne absente")
        return

    col   = df['days_to_fix']
    valid = col.dropna()
    logger.info(f"days_to_fix — total: {len(col)}, valides: {len(valid)}, NaN: {col.isna().sum()}")
    if len(valid) > 0:
        logger.info(f"  min={valid.min():.0f}  max={valid.max():.0f}  "
                    f"median={valid.median():.0f}  mean={valid.mean():.1f}")
        logger.info(f"  <= 30j : {(valid <= 30).sum()} ({(valid <= 30).mean()*100:.1f}%)")
        logger.info(f"   > 30j : {(valid  > 30).sum()} ({(valid  > 30).mean()*100:.1f}%)")
        logger.info(f"  negatifs (date fix < date decouverte) : {(valid < 0).sum()}")


# ---------------------------------------------------------------------------
# 3. Construction de la cible — cascade de fallbacks
# ---------------------------------------------------------------------------

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Cascade de strategies pour garantir une cible binaire valide.

    Strategie 1 — days_to_fix avec seuil adaptatif
      Si tous les findings sont corriges en <= 30j, on cherche un seuil
      qui cree deux classes equilibrees (median de days_to_fix).

    Strategie 2 — composite_risk > percentile 70
      Top 30% = haut risque.

    Strategie 3 — severity_num >= 3 (High ou Critical)
      Toujours disponible, semantiquement clair.
    """
    _diagnose_days_to_fix(df)

    # --- Strategie 1 : days_to_fix ---
    if 'days_to_fix' in df.columns:
        col   = df['days_to_fix']
        valid = col.dropna()

        if len(valid) > 10:
            # Tentative seuil 30j
            target_30 = col.between(0, 30).astype(int)
            if _has_two_classes(target_30):
                logger.info("Cible : fixed_in_30d (seuil 30j) — 2 classes OK")
                return _log_and_return(target_30, "fixed_in_30d seuil=30j")

            # Seuil 30j ne donne qu'une classe -> essai seuil adaptatif (mediane)
            median_fix = valid.median()
            logger.warning(
                f"Seuil 30j ne produit qu'une classe "
                f"(tous les findings ont days_to_fix <= 30j).\n"
                f"   Tentative seuil adaptatif : mediane = {median_fix:.0f}j"
            )
            target_med = (col > median_fix).astype(int)
            if _has_two_classes(target_med):
                logger.info(f"Cible : days_to_fix > {median_fix:.0f}j (mediane) — 2 classes OK")
                return _log_and_return(target_med, f"days_to_fix > mediane ({median_fix:.0f}j)")

    # --- Strategie 2 : composite_risk ---
    if 'composite_risk' in df.columns:
        threshold = df['composite_risk'].quantile(0.70)
        target_cr = (df['composite_risk'] > threshold).astype(int)
        if _has_two_classes(target_cr):
            logger.warning(
                "Fallback strategie 2 : composite_risk > percentile 70\n"
                "   (top 30%% = haut risque). Remplacez par des dates reelles."
            )
            return _log_and_return(target_cr, "composite_risk > p70")

    # --- Strategie 3 : severity_num ---
    if 'severity_num' in df.columns:
        target_sev = (df['severity_num'] >= 3).astype(int)  # High ou Critical
        if _has_two_classes(target_sev):
            logger.warning(
                "Fallback strategie 3 : severity_num >= 3 (High / Critical).\n"
                "   Remplacez par des dates reelles des que possible."
            )
            return _log_and_return(target_sev, "severity >= High")

    raise ValueError(
        "Impossible de construire une cible binaire avec vos donnees.\n"
        "Toutes les strategies ont echoue (une seule classe dans chaque tentative).\n"
        "Verifiez la diversite de vos findings dans DefectDojo."
    )


def _log_and_return(target: pd.Series, label: str) -> pd.Series:
    """Log la distribution et retourne la cible."""
    n_pos = int(target.sum())
    n_neg = len(target) - n_pos
    ratio = target.mean() * 100
    logger.info(
        f"[{label}] positifs: {n_pos} ({ratio:.1f}%) | negatifs: {n_neg} ({100-ratio:.1f}%)"
    )
    if ratio < 5 or ratio > 95:
        logger.warning(f"Cible desequilibree ({ratio:.1f}%%). class_weight='balanced' sera force.")
    return target


# ---------------------------------------------------------------------------
# 4. Features
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, target: pd.Series):
    valid_idx = target.notna()
    X = df.loc[valid_idx, FEATURE_COLS].copy()
    y = target.loc[valid_idx].copy()
    logger.info(f"Echantillons valides : {len(X)}")
    return X, y


# ---------------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(force_balanced: bool = False) -> Pipeline:
    cw = 'balanced' if force_balanced else None
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestClassifier(class_weight=cw, random_state=42, n_jobs=-1)),
    ])


# ---------------------------------------------------------------------------
# 6. Tuning
# ---------------------------------------------------------------------------

def tune_pipeline(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    k      = _n_splits(y_train)
    n_iter = min(20, 50)
    logger.info(f"Recherche hyperparametres : StratifiedKFold k={k}, n_iter={n_iter}")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST,
        n_iter=n_iter,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        error_score=0,
        verbose=0,
    )
    search.fit(X_train, y_train)
    logger.info(f"Meilleurs params : {search.best_params_}")
    logger.info(f"Meilleur F1 (CV) : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------

def evaluate(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    logger.info("\nEvaluation du modele")

    k = _n_splits(y_train)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', error_score=0)
    logger.info(f"F1 cross-val : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    y_pred = pipeline.predict(X_test)
    f1     = f1_score(y_test, y_pred, zero_division=0)
    logger.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    logger.info(f"F1-score test : {f1:.4f}")

    auc           = None
    model_classes = pipeline.named_steps['model'].classes_
    if len(model_classes) == 2 and y_test.nunique() == 2:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        logger.info(f"ROC-AUC test  : {auc:.4f}")
    else:
        logger.warning(f"ROC-AUC non calcule : classes modele={model_classes}")

    return {
        'cv_f1_mean':   round(float(cv_scores.mean()), 4),
        'cv_f1_std':    round(float(cv_scores.std()),  4),
        'test_f1':      round(float(f1), 4),
        'test_roc_auc': round(float(auc), 4) if auc is not None else 'N/A',
    }


# ---------------------------------------------------------------------------
# 8. Rapports
# ---------------------------------------------------------------------------

def save_reports(pipeline: Pipeline, X_test, y_test):
    REPORTS_DIR.mkdir(exist_ok=True)

    y_pred      = pipeline.predict(X_test)
    model_cls   = pipeline.named_steps['model'].classes_
    disp_labels = ['Faible risque', 'Haut risque'] if len(model_cls) == 2 else [str(c) for c in model_cls]

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Matrice de confusion - Test set')
    plt.tight_layout()
    path_cm = REPORTS_DIR / 'confusion_matrix.png'
    plt.savefig(path_cm, dpi=150)
    plt.close()
    logger.info(f"Matrice de confusion -> {path_cm}")

    rf_model    = pipeline.named_steps['model']
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(FEATURE_COLS)), importances[indices], color='steelblue')
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels([FEATURE_COLS[i] for i in indices], rotation=35, ha='right')
    ax.set_title('Importance des features (RandomForest)')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    path_fi = REPORTS_DIR / 'feature_importance.png'
    plt.savefig(path_fi, dpi=150)
    plt.close()
    logger.info(f"Importance features -> {path_fi}")

    logger.info("\nTop 5 features :")
    for rank, idx in enumerate(indices[:5], 1):
        logger.info(f"   {rank}. {FEATURE_COLS[idx]:<25} {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# 9. Sauvegarde
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: Pipeline, metrics: dict):
    MODELS_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path  = MODELS_DIR / f'pipeline_{timestamp}.pkl'
    latest_path = MODELS_DIR / 'pipeline_latest.pkl'
    meta_path   = MODELS_DIR / 'pipeline_latest_meta.json'

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, latest_path)
    logger.info(f"Pipeline sauvegarde -> {model_path}")
    logger.info(f"Lien latest         -> {latest_path}")

    meta = {
        'timestamp':    timestamp,
        'features':     FEATURE_COLS,
        'n_classes':    int(len(pipeline.named_steps['model'].classes_)),
        'classes':      pipeline.named_steps['model'].classes_.tolist(),
        'metrics':      metrics,
        'model_params': pipeline.named_steps['model'].get_params(),
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadonnees         -> {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("Entrainement du modele — AI Risk Engine")
    logger.info("=" * 60)

    df     = load_clean_data()
    target = build_target(df)
    X, y   = prepare_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train : {len(X_train)} | Test : {len(X_test)}")

    ratio          = y.mean() * 100
    force_balanced = ratio < 20 or ratio > 80
    pipeline       = build_pipeline(force_balanced=force_balanced)
    pipeline       = tune_pipeline(pipeline, X_train, y_train)
    metrics        = evaluate(pipeline, X_train, y_train, X_test, y_test)

    save_reports(pipeline, X_test, y_test)
    save_pipeline(pipeline, metrics)

    logger.info("\nEntrainement termine avec succes.")


if __name__ == '__main__':
    main()