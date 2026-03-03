"""
train.py — AI Risk Engine v2.1
================================
Entraînement du modèle de classification multi-classes (5 niveaux de risque).

Corrections v2.1 :
  PART 1 — FEATURE_COLS complet (base + sémantique + interaction + normalisé)
  PART 2 — Cible = risk_class (5 classes ordinales, produite par preprocess.py)
  PART 3 — Validation de toutes les colonnes
  PART 4 — Métriques multi-classes (F1 weighted/macro, ROC-AUC OvR robuste)
  PART 5 — Matrice de confusion 5x5 avec labels explicites
  PART 6 — Grille hyperparamètres adaptée multi-classes + class_weight
  PART 7 — _n_splits robuste pour 5 classes
  FIX A  — product_fp_rate recalculée sur train uniquement (suppression data leakage)
  FIX B  — CV dans evaluate() sur pipeline non-fitté (mesure honnête)
  FIX C  — ROC-AUC robuste aux classes absentes du test set
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
DATA_PATH   = Path("data/processed/findings_clean.csv")
MODELS_DIR  = Path("models")
REPORTS_DIR = Path("reports")

# PART 1 — Features complètes
# Note : product_fp_rate est incluse mais recalculée sur train uniquement (FIX A)
FEATURE_COLS = [
    # Base
    "severity_num",
    "cvss_score",
    "age_days",
    "has_cve",
    "has_cwe",
    "tags_count",
    "is_false_positive",
    "is_active",
    # Tags sémantiques
    "tag_urgent",
    "tag_in_production",
    "tag_sensitive",
    "tag_external",
    # Contextuelles
    "severity_x_active",
    "product_fp_rate",
    "cvss_severity_gap",
    # Interaction
    "cvss_x_severity",
    "cvss_x_has_cve",
    "severity_x_urgent",
    "age_x_cvss",
    # Normalisées
    "cvss_score_norm",
    "severity_norm",
    "age_days_norm",
    "tags_count_norm",
    "cvss_severity_gap_norm",
]

CLASS_LABELS = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
CLASS_NAMES  = [CLASS_LABELS[i] for i in range(5)]

# PART 6 — Grille hyperparamètres (class_weight géré ici, pas dans build_pipeline)
PARAM_DIST_RF = {
    "model__n_estimators":      [100, 200, 300, 500],
    "model__max_depth":         [8, 12, 16, 20, None],
    "model__min_samples_leaf":  [1, 2, 4, 6],
    "model__min_samples_split": [2, 4, 8],
    "model__max_features":      ["sqrt", "log2", 0.3],
    "model__class_weight":      ["balanced", "balanced_subsample", None],
    "model__criterion":         ["gini", "entropy"],
}


# ──────────────────────────────────────────────
# PART 7 — Folds adaptatifs
# ──────────────────────────────────────────────

def _n_splits(y: pd.Series) -> int:
    """
    Nombre de folds StratifiedKFold optimal.
    Contrainte : n_splits <= taille de la classe la plus petite.
    """
    min_class_count = int(y.value_counts().min())
    n = max(2, min(5, min_class_count))
    logger.info(f"StratifiedKFold : k={n} (classe minoritaire = {min_class_count} samples)")
    return n


# ══════════════════════════════════════════════
# 1. CHARGEMENT & VALIDATION
# ══════════════════════════════════════════════

def load_clean_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Données chargées : {len(df)} lignes x {len(df.columns)} colonnes")

    # PART 3 — Validation colonnes
    missing_targets  = {"risk_class", "risk_score"} - set(df.columns)
    missing_features = set(FEATURE_COLS) - set(df.columns)

    if missing_targets:
        logger.warning(f"Colonnes cible manquantes : {sorted(missing_targets)}")
    if missing_features:
        logger.warning(f"Features manquantes : {sorted(missing_features)}")

    return df


# ══════════════════════════════════════════════
# 2. CIBLE (PART 2)
# ══════════════════════════════════════════════

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    PART 2 — risk_class (5 classes ordinales) depuis preprocess.py v2.1.
    Fallbacks pour compatibilité ascendante.
    """
    if "risk_class" in df.columns:
        target = df["risk_class"].astype(int)
        logger.info(f"Cible : risk_class ({target.nunique()} classes distinctes)")
        _log_class_distribution(target)
        return target

    if "risk_score" in df.columns:
        logger.warning("risk_class absent — reconstruction depuis risk_score.")
        bins   = [-np.inf, 2.0, 4.0, 6.0, 8.0, np.inf]
        target = pd.cut(df["risk_score"], bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
        _log_class_distribution(target)
        return target

    if "composite_risk" in df.columns:
        logger.warning("Fallback composite_risk — mettez a jour preprocess.py.")
        norm   = df["composite_risk"] / df["composite_risk"].max() * 10
        target = pd.cut(norm, bins=[-np.inf, 2, 4, 6, 8, np.inf], labels=[0, 1, 2, 3, 4]).astype(int)
        _log_class_distribution(target)
        return target

    raise ValueError(
        "Impossible de construire la cible ML. "
        "Colonnes risk_class, risk_score et composite_risk toutes absentes."
    )


def _log_class_distribution(target: pd.Series) -> None:
    dist  = target.value_counts().sort_index()
    total = len(target)
    logger.info("Distribution des classes :")
    for cls, count in dist.items():
        label = CLASS_LABELS.get(int(cls), f"Class {cls}")
        pct   = count / total * 100
        bar   = "X" * int(pct / 5)
        logger.info(f"   {label:<10} ({cls}) : {count:4d} samples  {pct:5.1f}%  {bar}")
    min_pct = dist.min() / total * 100
    max_pct = dist.max() / total * 100
    if max_pct / (min_pct + 1e-6) > 5:
        logger.warning(
            f"Desequilibre : {min_pct:.1f}% vs {max_pct:.1f}% — "
            "class_weight='balanced' sera applique."
        )


# ══════════════════════════════════════════════
# 3. PREPARATION DES FEATURES (FIX A)
# ══════════════════════════════════════════════

def prepare_features(
    df:        pd.DataFrame,
    target:    pd.Series,
    train_idx: pd.Index = None,
) -> tuple:
    """
    FIX A — Recalcul de product_fp_rate UNIQUEMENT sur train.

    Pourquoi c'est important :
      product_fp_rate = taux de faux positifs par produit.
      Si calculée sur train+test, le modele voit de l'information du test
      pendant l'entrainement (data leakage). En la recalculant sur train
      puis en la propageant au test par lookup produit, la mesure est honnete.
    """
    valid_idx   = target.notna()
    available   = [c for c in FEATURE_COLS if c in df.columns]
    unavailable = set(FEATURE_COLS) - set(available)

    if unavailable:
        logger.warning(f"Features ignorees (absentes) : {sorted(unavailable)}")

    X = df.loc[valid_idx, available].copy()
    y = target.loc[valid_idx].astype(int).copy()

    # FIX A : recalcul product_fp_rate sur train uniquement
    if (
        "product_fp_rate" in available
        and "product_id" in df.columns
        and train_idx is not None
    ):
        train_mask   = X.index.isin(train_idx)
        X_train_part = X.loc[train_mask].copy()
        X_train_part["_product_id"] = df.loc[X_train_part.index, "product_id"].values

        fp_rate_train = (
            X_train_part.groupby("_product_id")["is_false_positive"]
            .mean()
        )

        product_ids = df.loc[X.index, "product_id"].values
        X["product_fp_rate"] = (
            pd.Series(product_ids, index=X.index)
            .map(fp_rate_train)
            .fillna(fp_rate_train.mean() if len(fp_rate_train) > 0 else 0.0)
        )
        logger.info("product_fp_rate recalculee sur train uniquement (FIX A — data leakage supprime)")

    logger.info(f"Samples valides : {len(X)} | Features : {len(available)}")
    return X, y, available


# ══════════════════════════════════════════════
# 4. PIPELINE
# ══════════════════════════════════════════════

def build_pipeline() -> Pipeline:
    """
    Pipeline : imputation -> standardisation -> RandomForest multi-classes.
    class_weight gere dans PARAM_DIST_RF pour eviter contradiction.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
        )),
    ])


# ══════════════════════════════════════════════
# 5. TUNING (PART 6)
# ══════════════════════════════════════════════

def tune_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """PART 6 — RandomizedSearchCV avec f1_weighted."""
    k      = _n_splits(y_train)
    n_iter = 30
    logger.info(f"Optimisation : k={k} folds, n_iter={n_iter}")

    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=PARAM_DIST_RF,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
        error_score=0.0,
        verbose=0,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info(f"Meilleurs params     : {search.best_params_}")
    logger.info(f"Meilleur F1-weighted : {search.best_score_:.4f}")
    return search.best_estimator_


# ══════════════════════════════════════════════
# 6. EVALUATION (PART 4 + FIX B + FIX C)
# ══════════════════════════════════════════════

def evaluate(
    best_pipeline:     Pipeline,
    unfitted_pipeline: Pipeline,
    X_train:           pd.DataFrame,
    y_train:           pd.Series,
    X_test:            pd.DataFrame,
    y_test:            pd.Series,
    feature_cols:      list,
) -> dict:
    """
    PART 4 — Metriques completes multi-classes.

    FIX B : cross_val_score utilise unfitted_pipeline (clone non entraine)
      pour une mesure honnete de generalisation. Le CV avec le pipeline fitté
      serait optimiste car les hyperparametres ont ete choisis sur ces donnees.

    FIX C : ROC-AUC robuste aux classes absentes du test set.
      Si une classe n'apparait pas dans y_test, on filtre les colonnes
      de probabilite correspondantes pour eviter un crash silencieux.
    """
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION DU MODELE")
    logger.info("=" * 50)

    # ── FIX B : CV sur pipeline NON fitte ────────────────────────────────
    k  = _n_splits(y_train)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    cv_f1w = cross_val_score(unfitted_pipeline, X_train, y_train, cv=cv,
                              scoring="f1_weighted", error_score=0.0)
    cv_f1m = cross_val_score(unfitted_pipeline, X_train, y_train, cv=cv,
                              scoring="f1_macro", error_score=0.0)
    logger.info(f"CV F1-weighted (non-fitte) : {cv_f1w.mean():.4f} +/- {cv_f1w.std():.4f}")
    logger.info(f"CV F1-macro    (non-fitte) : {cv_f1m.mean():.4f} +/- {cv_f1m.std():.4f}")

    # ── Metriques test ────────────────────────────────────────────────────
    y_pred       = best_pipeline.predict(X_test)
    y_proba      = best_pipeline.predict_proba(X_test)
    f1_weighted  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro     = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None,       zero_division=0)

    logger.info(f"\nF1-weighted (test) : {f1_weighted:.4f}")
    logger.info(f"F1-macro    (test) : {f1_macro:.4f}")

    classes_present = sorted(y_test.unique())
    labels_present  = [CLASS_LABELS.get(c, str(c)) for c in classes_present]
    logger.info("\nRapport de classification :")
    logger.info(classification_report(
        y_test, y_pred,
        labels=classes_present,
        target_names=labels_present,
        zero_division=0,
    ))

    # ── FIX C : ROC-AUC robuste ───────────────────────────────────────────
    auc_ovr       = None
    model_classes = list(best_pipeline.named_steps["model"].classes_)
    try:
        present_indices  = [model_classes.index(c) for c in classes_present if c in model_classes]
        y_proba_filtered = y_proba[:, present_indices]

        if len(classes_present) >= 2:
            if len(classes_present) == 2:
                auc_ovr = roc_auc_score(y_test, y_proba_filtered[:, 1])
            else:
                auc_ovr = roc_auc_score(
                    y_test, y_proba_filtered,
                    multi_class="ovr",
                    average="weighted",
                    labels=classes_present,
                )
            logger.info(f"ROC-AUC OvR (weighted) : {auc_ovr:.4f}")
    except Exception as e:
        logger.warning(f"ROC-AUC non calculable : {e}")

    # ── F1 par classe ─────────────────────────────────────────────────────
    logger.info("\nF1 par classe :")
    for i, cls in enumerate(model_classes):
        if i < len(f1_per_class):
            logger.info(f"   {CLASS_LABELS.get(int(cls), str(cls)):<10} : {f1_per_class[i]:.4f}")

    return {
        "cv_f1_weighted_mean": round(float(cv_f1w.mean()), 4),
        "cv_f1_weighted_std":  round(float(cv_f1w.std()),  4),
        "cv_f1_macro_mean":    round(float(cv_f1m.mean()), 4),
        "cv_f1_macro_std":     round(float(cv_f1m.std()),  4),
        "test_f1_weighted":    round(float(f1_weighted), 4),
        "test_f1_macro":       round(float(f1_macro), 4),
        "test_roc_auc_ovr":    round(float(auc_ovr), 4) if auc_ovr is not None else "N/A",
        "f1_per_class": {
            CLASS_LABELS.get(int(cls), str(cls)): round(float(f1_per_class[i]), 4)
            for i, cls in enumerate(model_classes) if i < len(f1_per_class)
        },
    }


# ══════════════════════════════════════════════
# 7. RAPPORTS VISUELS (PART 5)
# ══════════════════════════════════════════════

def save_reports(pipeline: Pipeline, X_test: pd.DataFrame,
                 y_test: pd.Series, feature_cols: list) -> None:
    """PART 5 — Graphiques adaptes aux 5 classes avec labels explicites."""
    REPORTS_DIR.mkdir(exist_ok=True)

    y_pred        = pipeline.predict(X_test)
    model_classes = pipeline.named_steps["model"].classes_
    present_labels = [CLASS_LABELS.get(int(c), str(c)) for c in model_classes]

    # Matrice de confusion 5x5
    cm = confusion_matrix(y_test, y_pred, labels=model_classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_labels, yticklabels=present_labels, ax=ax)
    ax.set_title("Matrice de confusion — 5 niveaux de risque", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predit", fontsize=11)
    ax.set_ylabel("Reel", fontsize=11)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    logger.info(f"Matrice de confusion -> {REPORTS_DIR / 'confusion_matrix.png'}")

    # Importance des features
    rf_model    = pipeline.named_steps["model"]
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    colors = [
        "#e74c3c" if importances[i] > 0.10 else
        "#e67e22" if importances[i] > 0.05 else "#3498db"
        for i in indices
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(feature_cols)), importances[indices], color=colors, edgecolor="white")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels([feature_cols[i] for i in indices], rotation=40, ha="right", fontsize=9)
    ax.set_title("Importance des features (RandomForest)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Importance relative")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#e74c3c", label="> 10%"),
        Patch(facecolor="#e67e22", label="5-10%"),
        Patch(facecolor="#3498db", label="< 5%"),
    ], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    logger.info(f"Importance features -> {REPORTS_DIR / 'feature_importance.png'}")

    # Distribution reel vs predit + F1 par classe
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    real_dist = pd.Series(y_test.values).map(CLASS_LABELS).value_counts().reindex(CLASS_NAMES, fill_value=0)
    pred_dist = pd.Series(y_pred).map(CLASS_LABELS).value_counts().reindex(CLASS_NAMES, fill_value=0)
    x, width  = np.arange(len(CLASS_NAMES)), 0.35
    axes[0].bar(x - width/2, real_dist.values, width, label="Reel",   color="#2ecc71", alpha=0.8)
    axes[0].bar(x + width/2, pred_dist.values, width, label="Predit", color="#3498db", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES)
    axes[0].set_title("Distribution Reel vs Predit")
    axes[0].set_ylabel("Nombre de samples")
    axes[0].legend()

    classes_in_test = sorted(y_test.unique())
    f1_per = f1_score(y_test, y_pred, average=None, zero_division=0, labels=classes_in_test)
    names_in_test = [CLASS_LABELS.get(c, str(c)) for c in classes_in_test]
    bar_colors = ["#e74c3c" if f < 0.5 else "#e67e22" if f < 0.75 else "#2ecc71" for f in f1_per]
    axes[1].bar(names_in_test, f1_per, color=bar_colors, edgecolor="white")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("F1-score par classe")
    axes[1].set_ylabel("F1-score")
    axes[1].axhline(0.75, color="gray", linestyle="--", alpha=0.6, label="Seuil 0.75")
    axes[1].legend(fontsize=9)

    plt.suptitle("Analyse des predictions — AI Risk Engine v2.1", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "prediction_analysis.png", dpi=150)
    plt.close()
    logger.info(f"Analyse predictions -> {REPORTS_DIR / 'prediction_analysis.png'}")

    logger.info("\nTop 10 features :")
    for rank, idx in enumerate(indices[:10], 1):
        bar = "X" * int(importances[idx] * 100)
        logger.info(f"   {rank:2}. {feature_cols[idx]:<30} {importances[idx]:.4f}  {bar}")


# ══════════════════════════════════════════════
# 8. SAUVEGARDE
# ══════════════════════════════════════════════

def save_pipeline(pipeline: Pipeline, metrics: dict, feature_cols: list) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path  = MODELS_DIR / f"pipeline_{timestamp}.pkl"
    latest_path = MODELS_DIR / "pipeline_latest.pkl"
    meta_path   = MODELS_DIR / "pipeline_latest_meta.json"

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, latest_path)
    logger.info(f"Pipeline -> {model_path}")
    logger.info(f"Latest   -> {latest_path}")

    rf   = pipeline.named_steps["model"]
    meta = {
        "version":      "2.1",
        "timestamp":    timestamp,
        "task":         "multiclass_classification",
        "n_classes":    int(len(rf.classes_)),
        "classes":      [int(c) for c in rf.classes_],
        "class_labels": {str(k): v for k, v in CLASS_LABELS.items()},
        "features":     feature_cols,
        "n_features":   len(feature_cols),
        "metrics":      metrics,
        "model_params": rf.get_params(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Metadonnees -> {meta_path}")


# ══════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("Entrainement — AI Risk Engine v2.1")
    logger.info("=" * 60)

    # Chargement
    df     = load_clean_data()
    target = build_target(df)

    # FIX A : split AVANT prepare_features pour recalculer product_fp_rate sur train
    valid_mask = target.notna()
    df_valid   = df.loc[valid_mask].copy()
    y_valid    = target.loc[valid_mask].astype(int)

    train_idx, test_idx = train_test_split(
        df_valid.index, test_size=0.2, random_state=42, stratify=y_valid
    )
    logger.info(f"Train : {len(train_idx)} samples | Test : {len(test_idx)} samples")

    X, y, feature_cols = prepare_features(df_valid, y_valid, train_idx=train_idx)
    X_train, X_test    = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test    = y.loc[train_idx], y.loc[test_idx]
    logger.info(f"Shape : X_train={X_train.shape}, X_test={X_test.shape}")

    # FIX B : pipeline non-fitte conserve pour CV honnete dans evaluate()
    unfitted_pipeline = build_pipeline()

    # Tuning (PART 6)
    best_pipeline = tune_pipeline(build_pipeline(), X_train, y_train)

    # Evaluation (PART 4 + FIX B + FIX C)
    metrics = evaluate(
        best_pipeline, unfitted_pipeline,
        X_train, y_train, X_test, y_test,
        feature_cols,
    )

    save_reports(best_pipeline, X_test, y_test, feature_cols)

    save_pipeline(best_pipeline, metrics, feature_cols)

    logger.info("\nEntrainement v2.1 termine avec succes.")
    logger.info(f"   F1-weighted : {metrics['test_f1_weighted']:.4f}")
    logger.info(f"   F1-macro    : {metrics['test_f1_macro']:.4f}")
    logger.info(f"   ROC-AUC OvR : {metrics['test_roc_auc_ovr']}")


if __name__ == "__main__":
    main()