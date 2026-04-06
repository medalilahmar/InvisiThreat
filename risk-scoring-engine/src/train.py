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
import shap

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings("ignore")


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

DATA_PATH = Path("data/processed/findings_clean.csv")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
SHAP_DIR = REPORTS_DIR / "shap"

FEATURE_COLS = [
    "cvss_score", "cvss_score_norm", "age_days", "age_days_norm",
    "has_cve", "has_cwe", "tags_count", "tags_count_norm",
    "tag_urgent", "tag_in_production", "tag_sensitive", "tag_external",
    "product_fp_rate", "cvss_x_has_cve", "age_x_cvss",
    "epss_score", "epss_percentile", "has_high_epss", "epss_x_cvss", "epss_score_norm",
    "exploit_risk", "context_score", "days_open_high",
]

EXCLUDE_COLS = [
    "days_to_fix", "risk_class", "risk_score",
    "is_mitigated", "out_of_scope", "is_false_positive",
    "label_source", "score_composite_raw", "score_composite_adj"
]

CLASS_LABELS = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}

PARAM_DIST_RF = {
    "model__n_estimators": [100, 150, 200, 300],
    "model__max_depth": [10, 15, 20, 25, None],
    "model__min_samples_leaf": [2, 4, 6, 8],
    "model__min_samples_split": [4, 8, 12],
    "model__max_features": ["sqrt", "log2", 0.4],
    "model__class_weight": ["balanced", "balanced_subsample"],
    "model__criterion": ["gini", "entropy"],
}

PARAM_DIST_XGB = {
    "model__n_estimators": [100, 150, 200, 300],
    "model__max_depth": [5, 7, 10, 12],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight": [1, 2, 3, 5],
    "model__gamma": [0, 0.1, 0.2],
}

PARAM_DIST_LGBM = {
    "model__n_estimators": [100, 150, 200, 300],
    "model__max_depth": [8, 12, 16, 20],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__num_leaves": [31, 50, 70],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}


def get_model_object(pipeline_or_calibrated):
    """ Extraire le modèle selon son type (Pipeline ou CalibratedClassifierCV)."""
    if isinstance(pipeline_or_calibrated, CalibratedClassifierCV):
        return pipeline_or_calibrated.estimator.named_steps["model"]
    else:
        return pipeline_or_calibrated.named_steps["model"]


def get_pipeline_transformers(pipeline_or_calibrated):
    """ Extraire les transformers (imputer, scaler) selon le type de pipeline."""
    if isinstance(pipeline_or_calibrated, CalibratedClassifierCV):
        imputer = pipeline_or_calibrated.estimator.named_steps["imputer"]
        scaler = pipeline_or_calibrated.estimator.named_steps["scaler"]
    else:
        imputer = pipeline_or_calibrated.named_steps["imputer"]
        scaler = pipeline_or_calibrated.named_steps["scaler"]
    return imputer, scaler


def load_clean_data() -> pd.DataFrame:
    """Charger et valider données."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f" Fichier manquant : {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    leakage_found = [c for c in EXCLUDE_COLS if c in FEATURE_COLS]
    if leakage_found:
        logger.error(f" DATA LEAKAGE DETECTED in FEATURE_COLS: {leakage_found}")
        raise ValueError("Data leakage violation!")
    
    logger.info(f"✓ Loaded: {len(df)} rows × {len(df.columns)} cols")
    logger.info(f"✓ Features available: {len([c for c in FEATURE_COLS if c in df.columns])}/{len(FEATURE_COLS)}")
    
    return df

def build_target(df: pd.DataFrame) -> pd.Series:
    """Extraire et valider la cible."""
    if "risk_class" not in df.columns:
        raise ValueError("risk_class absent — relance preprocess.py")
    
    target = pd.to_numeric(df["risk_class"], errors="coerce")
    valid = target.notna()
    
    logger.info(f"✓ Target samples: {valid.sum()} valid | {(~valid).sum()} excluded")
    
    if "label_source" in df.columns:
        dist = df.loc[valid, "label_source"].value_counts()
        logger.info(f"Label sources:\n{dist.to_string()}")
    
    log_class_distribution(target.dropna().astype(int), "Target distribution")
    return target

def log_class_distribution(target: pd.Series, msg="Distribution"):
    """Log distribution des classes."""
    if len(target) == 0:
        return
    dist = target.value_counts().sort_index()
    total = len(target)
    logger.info(f"\n{msg}:")
    for cls, count in dist.items():
        label = CLASS_LABELS.get(int(cls), f"Class {cls}")
        pct = count / total * 100
        logger.info(f"  {label:<12} ({cls}): {count:5d} samples  ({pct:5.1f}%)")

def prepare_features(df, target, train_idx=None):
    """Préparer features pour ML."""
    valid_idx = target.notna()
    available = [c for c in FEATURE_COLS if c in df.columns]
    unavailable = set(FEATURE_COLS) - set(available)
    
    if unavailable:
        logger.warning(f"  Missing features: {sorted(unavailable)}")
    
    X = df.loc[valid_idx, available].copy()
    y = target.loc[valid_idx].astype(int).copy()
    
    if "product_fp_rate" in available and "product_id" in df.columns and train_idx is not None:
        train_mask = X.index.isin(train_idx)
        train_ids = X.index[train_mask]
        
        if len(train_ids) > 0:
            fp_df = df.loc[train_ids, ["product_id", "is_false_positive"]].copy()
            fp_df["is_false_positive"] = pd.to_numeric(
                fp_df["is_false_positive"], errors="coerce"
            ).fillna(0)
            fp_rate_train = fp_df.groupby("product_id")["is_false_positive"].mean()
            product_ids = df.loc[X.index, "product_id"].values
            
            X["product_fp_rate"] = (
                pd.Series(product_ids, index=X.index)
                .map(fp_rate_train)
                .fillna(fp_rate_train.mean() if len(fp_rate_train) > 0 else 0.0)
            )
    
    logger.info(f"✓ Features: {X.shape} | Target: {y.shape}")
    return X, y, available


def build_rf_pipeline():
    """RandomForest pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1, verbose=0)),
    ])

def build_xgb_pipeline():
    """XGBoost pipeline (si disponible)."""
    if not XGBOOST_AVAILABLE:
        return None
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            random_state=42, n_jobs=-1,
            use_label_encoder=False, verbosity=0, eval_metric="mlogloss"
        )),
    ])

def build_lgbm_pipeline():
    """LightGBM pipeline (si disponible)."""
    if not LIGHTGBM_AVAILABLE:
        return None
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)),
    ])


def random_search_cv(pipeline, param_dist, X_train, y_train, name="model"):
    """Tuning avec RandomizedSearchCV + stratified k-fold."""
    min_class_count = int(y_train.value_counts().min())
    k = max(2, min(5, min_class_count // 10))
    
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=20, cv=cv,
        scoring="f1_weighted", n_jobs=-1, random_state=42, error_score=0.0,
        verbose=0, refit=True,
    )
    
    logger.info(f"[{name}] Hyperparameter search ({k}-fold stratified CV)...")
    search.fit(X_train, y_train)
    logger.info(f"[{name}] Best F1-weighted (CV): {search.best_score_:.4f}")
    
    return search.best_estimator_, search.best_score_

def train_all_models(X_train, y_train):
    """Entraîner tous les modèles disponibles et retourner le meilleur."""
    results = {}
    
    logger.info("\n" + "="*70)
    logger.info("Training RandomForest...")
    logger.info("="*70)
    rf_pipe, rf_score = random_search_cv(
        build_rf_pipeline(), PARAM_DIST_RF, X_train, y_train, "RandomForest"
    )
    results["RandomForest"] = (rf_pipe, rf_score)
    
    if XGBOOST_AVAILABLE:
        logger.info("\n" + "="*70)
        logger.info("Training XGBoost...")
        logger.info("="*70)
        xgb_pipe, xgb_score = random_search_cv(
            build_xgb_pipeline(), PARAM_DIST_XGB, X_train, y_train, "XGBoost"
        )
        results["XGBoost"] = (xgb_pipe, xgb_score)
    else:
        logger.warning("⚠️  XGBoost not available")
    
    if LIGHTGBM_AVAILABLE:
        logger.info("\n" + "="*70)
        logger.info("Training LightGBM...")
        logger.info("="*70)
        lgbm_pipe, lgbm_score = random_search_cv(
            build_lgbm_pipeline(), PARAM_DIST_LGBM, X_train, y_train, "LightGBM"
        )
        results["LightGBM"] = (lgbm_pipe, lgbm_score)
    else:
        logger.warning("LightGBM not available")
    
    best_name = max(results, key=lambda k: results[k][1])
    best_pipeline = results[best_name][0]
    
    logger.info("\n" + "="*70)
    logger.info("MODEL COMPARISON & RANKING")
    logger.info("="*70)
    for rank, (name, (_, score)) in enumerate(
        sorted(results.items(), key=lambda x: x[1][1], reverse=True), 1
    ):
        status = "  🏆 ← SELECTED" if name == best_name else ""
        logger.info(f"  #{rank} {name:<15} F1-weighted: {score:.4f}{status}")
    logger.info("="*70 + "\n")
    
    return best_pipeline, best_name, results


def calibrate_pipeline(pipeline, X_train, y_train, cv=None):
    """Calibrer probabilités avec Platt scaling."""
    logger.info("Calibrating model probabilities (Platt scaling)...")
    
    if cv is None:
        min_class_count = int(y_train.value_counts().min())
        cv = max(2, min(3, min_class_count))  # Max 3, min 2
        logger.info(f"  Auto-adjusted CV folds: {cv} (min class count: {min_class_count})")
    
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=cv)
    calibrated.fit(X_train, y_train)
    logger.info("✓ Calibration complete")
    return calibrated


def evaluate(pipeline, X_train, y_train, X_test, y_test, feature_cols):
    logger.info("\n" + "="*70)
    logger.info("MODEL EVALUATION ON TEST SET")
    logger.info("="*70 + "\n")
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_per_cls = f1_score(y_test, y_pred, average=None, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    logger.info(f"  Metrics:")
    logger.info(f"  F1-weighted: {f1_weighted:.4f}")
    logger.info(f"  F1-macro:    {f1_macro:.4f}")
    logger.info(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    classes_present = sorted(y_test.unique())
    labels_present = [CLASS_LABELS.get(c, str(c)) for c in classes_present]
    
    logger.info("\n" + classification_report(
        y_test, y_pred,
        labels=classes_present,
        target_names=labels_present,
        zero_division=0,
    ))
    
    auc_ovr = None
    try:
        if len(classes_present) >= 2:
            auc_ovr = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted",
                labels=classes_present,
            )
            logger.info(f"  ROC-AUC OvR (weighted): {auc_ovr:.4f}")
    except Exception as e:
        logger.warning(f"ROC-AUC computation failed: {e}")
    
    return {
        "test_f1_weighted": round(float(f1_weighted), 4),
        "test_f1_macro": round(float(f1_macro), 4),
        "test_balanced_accuracy": round(float(balanced_acc), 4),
        "test_roc_auc_ovr": round(float(auc_ovr), 4) if auc_ovr is not None else "N/A",
        "f1_per_class": {
            CLASS_LABELS.get(int(cls), str(cls)): round(float(f1_per_cls[i]), 4)
            for i, cls in enumerate(classes_present) if i < len(f1_per_cls)
        },
    }


def save_reports(pipeline, X_test, y_test, feature_cols, model_name="model"):
    """Générer rapports visuels (confusion, importance, SHAP)."""
    REPORTS_DIR.mkdir(exist_ok=True)
    
    y_pred = pipeline.predict(X_test)
    model_obj = get_model_object(pipeline)
    model_classes = model_obj.classes_
    class_labels = [CLASS_LABELS.get(int(c), str(c)) for c in model_classes]
    
    logger.info("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred, labels=model_classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax,
                cbar_kws={"label": "Count"})
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("✓ Confusion matrix saved")
    
    if hasattr(model_obj, "feature_importances_"):
        logger.info("Generating feature importance...")
        importances = model_obj.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = [
            "#e74c3c" if importances[i] > 0.10 else
            "#f39c12" if importances[i] > 0.05 else "#3498db"
            for i in indices
        ]
        ax.bar(range(len(feature_cols)), importances[indices], color=colors, edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels([feature_cols[i] for i in indices], rotation=45, ha="right", fontsize=9)
        ax.set_title(f"Feature Importance — {model_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Importance")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Feature importance saved")
        
        logger.info("\nTop 15 Features:")
        for rank, idx in enumerate(np.argsort(importances)[::-1][:15], 1):
            logger.info(f"  {rank:2}. {feature_cols[idx]:<40} {importances[idx]:.4f}")

def explain_with_shap(pipeline, X_test, feature_cols, sample_size=200):
    """Générer explicabilité SHAP."""
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    
    model = get_model_object(pipeline)
    imputer, scaler = get_pipeline_transformers(pipeline)
    
    sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
    X_transformed = pd.DataFrame(
        scaler.transform(imputer.transform(sample)),
        columns=feature_cols,
    )
    
    try:
        logger.info("Generating SHAP explanations...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        
        if isinstance(shap_values, list):
            class_idx = min(2, len(shap_values) - 1)
            sv_2d = shap_values[class_idx]
        else:
            sv_2d = shap_values
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(sv_2d, X_transformed, feature_names=feature_cols,
                         show=False, plot_type="bar", max_display=15)
        plt.title(f"SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        
        logger.info("✓ SHAP explanations saved")
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")


def save_model_atomic(pipeline, metrics, feature_cols, model_name, all_results):
    """Sauvegarde atomique du modèle."""
    MODELS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = MODELS_DIR / f"pipeline_{timestamp}.pkl"
    latest_path = MODELS_DIR / "pipeline_latest.pkl"
    
    try:
        joblib.dump(pipeline, model_path)
        joblib.dump(pipeline, latest_path)
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Latest: {latest_path}")
    except Exception as e:
        logger.error(f" Error saving model: {e}", exc_info=True)
        raise
    
    model_obj = get_model_object(pipeline)
    
    meta = {
        "version": "4.4",
        "timestamp": timestamp,
        "model_type": model_name,
        "task": "multiclass_classification (Info/Low/Medium/High/Critical)",
        "label_strategy": "severity_based_composite (no_leakage)",
        "n_classes": int(len(model_obj.classes_)),
        "classes": [int(c) for c in model_obj.classes_],
        "class_labels": CLASS_LABELS,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "metrics": metrics,
        "model_comparison": {
            name: round(float(score), 4) for name, (_, score) in all_results.items()
        },
        "leakage_check": {
            "excluded_features": EXCLUDE_COLS,
            "feature_cols_count": len(FEATURE_COLS),
            "status": "OK — Zero leakage",
        },
    }
    
    meta_path = MODELS_DIR / "pipeline_latest_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    
    logger.info(f"Metadata: {meta_path}\n")


def main():
    logger.info("\n" + "="*70)
    logger.info("InvisiThreat AI Risk Engine — Training v4.4 (COMPLETE)")
    logger.info("="*70 + "\n")
    
    df = load_clean_data()
    target = build_target(df)
    
    valid_mask = target.notna()
    df_valid = df.loc[valid_mask].copy()
    y_raw = target.loc[valid_mask].astype(int)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_valid = pd.Series(y_encoded, index=y_raw.index)
    
    mapping = {int(src): int(dst) for src, dst in zip(le.classes_, range(len(le.classes_)))}
    logger.info(f"✓ Labels re-mapped for XGBoost compatibility: {mapping}")
    
    train_idx, test_idx = train_test_split(
        df_valid.index, test_size=0.20, random_state=42, stratify=y_valid
    )
    logger.info(f"\n✓ Split: {len(train_idx)} train | {len(test_idx)} test\n")
    
    X, y, feature_cols = prepare_features(df_valid, y_valid, train_idx=train_idx)
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    
    log_class_distribution(y_train, "Training set (Encoded)")
    log_class_distribution(y_test, "Test set (Encoded)")
    
    best_pipeline, best_name, all_results = train_all_models(X_train, y_train)
    
    best_pipeline_calibrated = calibrate_pipeline(best_pipeline, X_train, y_train)
    
    metrics = evaluate(best_pipeline_calibrated, X_train, y_train, X_test, y_test, feature_cols)
    
    save_reports(best_pipeline_calibrated, X_test, y_test, feature_cols, model_name=best_name)
    explain_with_shap(best_pipeline_calibrated, X_test, feature_cols, sample_size=min(200, len(X_test)))
    
    save_model_atomic(best_pipeline_calibrated, metrics, feature_cols, best_name, all_results)
    
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)
    logger.info(f"Best Model: {best_name}")
    logger.info(f"F1-weighted: {metrics['test_f1_weighted']}")
    logger.info(f"F1-macro: {metrics['test_f1_macro']}")
    logger.info(f"Balanced Accuracy: {metrics['test_balanced_accuracy']}")
    logger.info(f"ROC-AUC: {metrics['test_roc_auc_ovr']}")
    logger.info(f"Data Leakage:  ZERO")
    logger.info(f"Classes processed: {list(le.classes_)} (Mapped to sequential integers)")
    logger.info("="*70 + "\n")
    logger.info("Training complete! Model ready for deployment.\n")

if __name__ == "__main__":
    main()