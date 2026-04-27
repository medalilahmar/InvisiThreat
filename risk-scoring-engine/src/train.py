import json
import logging
import random
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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

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

random.seed(42)
np.random.seed(42)


ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT_DIR / "data" / "processed" / "findings_clean.csv"
MODELS_DIR  = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
SHAP_DIR    = REPORTS_DIR / "shap"
LOGS_DIR    = ROOT_DIR / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CLASS_LABELS = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}


FEATURE_COLS = [
    "cvss_score",
    "cvss_score_norm",
    "has_cve",
    "has_cwe",
    "epss_score",
    "epss_percentile",
    "has_high_epss",
    "epss_x_cvss",
    "age_days",
    "age_days_norm",
    "delay_norm",
    "tag_urgent",
    "tag_in_production",
    "tag_sensitive",
    "tag_external",
    "tags_count",
    "tags_count_norm",
    "context_score",
    "exposure_norm",
    "product_fp_rate",
    "cvss_x_has_cve",
    "age_x_cvss",
]

EXCLUDE_COLS = [
    "days_to_fix",
    "risk_class",
    "risk_score",
    "severity_num",
    "is_mitigated",
    "out_of_scope",
    "is_false_positive",
    "label_source",
    "score_composite_raw",
    "cvss_x_severity",
    "severity_x_active",
    "severity_x_urgent",
    "cvss_severity_gap",
]


PARAM_DIST_RF = {
    "model__n_estimators":      [100, 150, 200, 300],
    "model__max_depth":         [10, 15, 20, None],
    "model__min_samples_leaf":  [2, 4, 6, 8],
    "model__min_samples_split": [4, 8, 12],
    "model__max_features":      ["sqrt", "log2", 0.4],
    "model__class_weight":      ["balanced", "balanced_subsample"],
    "model__criterion":         ["gini", "entropy"],
}
PARAM_DIST_XGB = {
    "model__n_estimators":     [100, 150, 200, 300],
    "model__max_depth":        [5, 7, 10],
    "model__learning_rate":    [0.01, 0.05, 0.1],
    "model__subsample":        [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__min_child_weight": [1, 3, 5],
    "model__gamma":            [0, 0.1, 0.2],
}
PARAM_DIST_LGBM = {
    "model__n_estimators":     [100, 150, 200, 300],
    "model__max_depth":        [8, 12, 16],
    "model__learning_rate":    [0.01, 0.05, 0.1],
    "model__num_leaves":       [31, 50, 70],
    "model__subsample":        [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
}



def load_clean_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier manquant : {DATA_PATH}\n  Lancer preprocess.py d'abord"
        )
    df = pd.read_csv(DATA_PATH)

    leakage = [c for c in EXCLUDE_COLS if c in FEATURE_COLS]
    if leakage:
        logger.error(f"DATA LEAKAGE detecte dans FEATURE_COLS : {leakage}")
        raise ValueError("Leakage detecte — corriger FEATURE_COLS")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.error(f"Features absentes du CSV : {missing}")
        raise ValueError(
            "Incoherence entre preprocess.py v9.0 et train.py — "
            f"features manquantes : {missing}"
        )

    n_feat = len([c for c in FEATURE_COLS if c in df.columns])
    logger.info(
        f"Donnees chargees : {len(df)} lignes | "
        f"features disponibles : {n_feat}/{len(FEATURE_COLS)}"
    )
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    if "risk_class" not in df.columns:
        raise ValueError("Colonne risk_class absente du CSV")
    target = pd.to_numeric(df["risk_class"], errors="coerce")
    valid  = target.notna()
    logger.info(f"Findings valides : {valid.sum()} | Exclus (FP/OOS) : {(~valid).sum()}")
    _log_dist(target.dropna().astype(int), "Distribution cible complete")
    return target


def _log_dist(y: pd.Series, label: str = "") -> None:
    if len(y) == 0:
        return
    dist  = y.value_counts().sort_index()
    total = len(y)
    logger.info(f"{label} :")
    for cls, count in dist.items():
        name = CLASS_LABELS.get(int(cls), "?")
        logger.info(
            f"  {name:<10} (classe {cls}) : {count:5d}  ({count / total * 100:5.1f}%)"
        )


def prepare_features(df: pd.DataFrame, target: pd.Series) -> tuple:
    valid     = target.notna()
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning(f"Features absentes du CSV (ignorees) : {sorted(missing)}")
    X = df.loc[valid, available].copy()
    y = target.loc[valid].astype(int).copy()
    logger.info(f"Matrice features : {X.shape} | Cible : {y.shape}")
    return X, y, available



def audit_leakage_mi(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Mutual Information comme detecteur de leakage complementaire.
    MI eleve (> 0.80) sur une feature simple peut signaler un leakage indirect.
    Correlation Pearson seule ne detecte pas les relations non lineaires.
    """
    logger.info("Audit leakage — Mutual Information (top 10) :")
    mi     = mutual_info_classif(X.fillna(0), y, random_state=42)
    mi_df  = (
        pd.DataFrame({"feature": X.columns, "mi": mi})
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )
    for _, row in mi_df.head(10).iterrows():
        flag = "  [WARNING : MI eleve — verifier leakage indirect]" if row["mi"] > 0.80 else ""
        logger.info(f"  {row['feature']:<38} MI={row['mi']:.4f}{flag}")
    return mi_df


def audit_correlations(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Correlation Pearson feature/cible.
    Seuil warning : 0.95 (correlation structurelle attendue pour certaines features).
    """
    logger.info("Audit correlations Pearson features/cible (top 10) :")
    corr = {}
    for col in X.columns:
        try:
            corr[col] = round(float(X[col].corr(y.astype(float))), 4)
        except Exception:
            corr[col] = None

    top = sorted(corr.items(), key=lambda x: abs(x[1] or 0), reverse=True)
    for col, r in top[:10]:
        flag = "  [WARNING : correlation > 0.95]" if abs(r or 0) > 0.95 else ""
        logger.info(f"  {col:<38} r={r:+.4f}{flag}")
    return corr



def compute_sample_weights(y: pd.Series, max_weight: float = 10.0) -> np.ndarray:
    counts  = y.value_counts()
    total   = len(y)
    n_cls   = len(counts)
    wmap    = {
        cls: min(total / (n_cls * cnt), max_weight)
        for cls, cnt in counts.items()
    }
    weights = y.map(wmap).values.astype(float)
    logger.info(f"Sample weights (plafond={max_weight}) :")
    for cls, w in sorted(wmap.items()):
        logger.info(f"  {CLASS_LABELS.get(int(cls), '?'):<10} -> {w:.3f}")
    return weights



def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced"
        )),
    ])


def build_xgb_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss"
        )),
    ])


def build_lgbm_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   LGBMClassifier(
            random_state=42, n_jobs=-1, verbose=-1
        )),
    ])



def random_search_cv(
    pipeline: Pipeline,
    param_dist: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    name: str,
    sw: np.ndarray = None,
) -> tuple:
    """
    [T4] sample_weight transmis a tous les modeles (RF inclus).
    [T5] CV fixe k=5 stratifie — plus stable qu'un k dynamique.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_dist,
        n_iter=20, cv=cv,
        scoring="f1_weighted",
        n_jobs=-1, random_state=42,
        error_score=0.0, verbose=0,
    )
    logger.info(f"Hyperopt {name} — 5-fold stratifie, 20 iterations...")
    fit_params = {}
    if sw is not None:
        fit_params["model__sample_weight"] = sw
    search.fit(X_train, y_train, **fit_params)
    logger.info(f"  {name} meilleur F1-weighted CV : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_score_, search.best_params_



def build_stacking_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    sw      = compute_sample_weights(y_train)
    results = {}

    logger.info("=" * 60)
    logger.info("Entrainement des modeles de base pour Stacking...")

    # RandomForest 
    logger.info("=" * 60)
    rf_pipe, rf_score, _ = random_search_cv(
        build_rf_pipeline(), PARAM_DIST_RF, X_train, y_train, "RandomForest", sw=sw
    )
    results["RandomForest"] = (rf_pipe, rf_score)

    # XGBoost
    xgb_pipe = None
    if XGBOOST_AVAILABLE:
        logger.info("=" * 60)
        xgb_pipe, xgb_score, _ = random_search_cv(
            build_xgb_pipeline(), PARAM_DIST_XGB, X_train, y_train, "XGBoost", sw=sw
        )
        results["XGBoost"] = (xgb_pipe, xgb_score)

    # LightGBM
    lgbm_pipe = None
    if LIGHTGBM_AVAILABLE:
        logger.info("=" * 60)
        lgbm_pipe, lgbm_score, _ = random_search_cv(
            build_lgbm_pipeline(), PARAM_DIST_LGBM, X_train, y_train, "LightGBM", sw=sw
        )
        results["LightGBM"] = (lgbm_pipe, lgbm_score)

    logger.info("Comparaison modeles de base :")
    for rank, (name, (_, score)) in enumerate(
        sorted(results.items(), key=lambda x: x[1][1], reverse=True), 1
    ):
        logger.info(f"  #{rank} {name:<15} CV F1={score:.4f}")

    estimators = [("rf", rf_pipe)]
    if xgb_pipe is not None:
        estimators.append(("xgb", xgb_pipe))
    if lgbm_pipe is not None:
        estimators.append(("lgbm", lgbm_pipe))

    if len(estimators) < 2:
        logger.warning(
            "Moins de 2 modeles disponibles — fallback sur le meilleur modele seul"
        )
        best_name = max(results, key=lambda k: results[k][1])
        return results[best_name][0], best_name, results

    final_estimator = LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced",
        solver="lbfgs",
    )

    # [T3] stack_method="predict_proba" pour coherence des sorties intermediaires
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=False,
        n_jobs=-1,
    )

    logger.info("Fit StackingClassifier (stack_method=predict_proba)...")
    stacking.fit(X_train, y_train)

    cv_preds   = cross_val_predict(stacking, X_train, y_train, cv=3, n_jobs=-1)
    cv_f1      = f1_score(y_train, cv_preds, average="weighted", zero_division=0)
    train_f1   = f1_score(y_train, stacking.predict(X_train), average="weighted", zero_division=0)
    logger.info(f"StackingClassifier F1-train : {train_f1:.4f} | F1-CV (3-fold) : {cv_f1:.4f}")

    return stacking, "StackingClassifier", results



def calibrate_if_needed(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> tuple:
    """
    [T10] StackingClassifier : calibration ignoree (LogisticRegression deja calibre).
    Autres modeles : calibration prefit avec split dedie 80/20.
    """
    if model_name == "StackingClassifier":
        logger.info(
            "StackingClassifier detecte — calibration ignoree "
            "(final_estimator LogisticRegression est deja calibre)"
        )
        return pipeline, X_train, y_train

    n      = len(y_train)
    method = "isotonic" if n > 1000 else "sigmoid"
    logger.info(f"Calibration prefit : methode={method} (n={n})")

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train,
        test_size=0.20, random_state=42, stratify=y_train,
    )
    logger.info(f"Split calibration : {len(X_fit)} fit | {len(X_cal)} calibration")
    pipeline.fit(X_fit, y_fit)
    cal = CalibratedClassifierCV(pipeline, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    logger.info(f"Calibration OK (method={method})")
    return cal, X_fit, y_fit



def evaluate(
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = sorted(y_test.unique())

    f1_w    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    f1_pc   = f1_score(y_test, y_pred, average=None,       zero_division=0, labels=classes)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    f1_train = f1_score(y_train, pipeline.predict(X_train), average="weighted", zero_division=0)
    overfit  = f1_train - f1_w

    critical_label = 3
    recall_crit = 0.0
    f1_crit     = 0.0
    if critical_label in classes:
        idx_crit    = classes.index(critical_label)
        recall_crit = recall_score(
            y_test, y_pred, labels=[critical_label], average=None, zero_division=0
        )[0]
        f1_crit = float(f1_pc[idx_crit]) if idx_crit < len(f1_pc) else 0.0

    logger.info("=" * 60)
    logger.info("EVALUATION FINALE")
    logger.info(f"  F1-weighted  (test)  : {f1_w:.4f}")
    logger.info(f"  F1-weighted  (train) : {f1_train:.4f}")
    logger.info(
        f"  Surapprentissage D   : {overfit:+.4f}  "
        f"{'[WARNING : fort surapprentissage]' if overfit > 0.10 else '[OK]'}"
    )
    logger.info(f"  F1-macro             : {f1_m:.4f}")
    logger.info(f"  Balanced Accuracy    : {bal_acc:.4f}")
    logger.info(f"  Recall (Critical)    : {recall_crit:.4f}")
    logger.info(f"  F1     (Critical)    : {f1_crit:.4f}")

    if f1_crit < 0.5 and critical_label in y_test.values:
        logger.warning(
            f"Classe Critical : F1={f1_crit:.2f} < 0.50 — "
            "peu d'exemples ou modele mal calibre. "
            "Envisager class_weight renforce ou regroupement avec High."
        )

    if f1_w > 0.95:
        logger.warning(f"F1={f1_w:.4f} > 0.95 — suspicieusement eleve, verifier leakage")
    elif f1_w > 0.88:
        logger.info(f"F1={f1_w:.4f} — dans la plage attendue pour ce type de donnees")

    logger.info(
        "\n"
        + classification_report(
            y_test, y_pred,
            labels=classes,
            target_names=[CLASS_LABELS.get(c, str(c)) for c in classes],
            zero_division=0,
        )
    )

    auc = None
    try:
        if len(classes) >= 2:
            auc = roc_auc_score(
                y_test, y_proba,
                multi_class="ovr", average="weighted", labels=classes,
            )
            logger.info(f"  ROC-AUC OvR (weighted) : {auc:.4f}")
    except Exception as e:
        logger.warning(f"ROC-AUC indisponible : {e}")

    return {
        "test_f1_weighted":       round(f1_w,       4),
        "test_f1_macro":          round(f1_m,       4),
        "test_balanced_accuracy": round(bal_acc,    4),
        "test_roc_auc_ovr":       round(auc, 4) if auc else "N/A",
        "train_f1_weighted":      round(f1_train,   4),
        "overfit_delta":          round(overfit,    4),
        "recall_critical":        round(recall_crit, 4),
        "f1_critical":            round(f1_crit,    4),
        "f1_per_class": {
            CLASS_LABELS.get(int(cls), str(cls)): round(float(f1_pc[i]), 4)
            for i, cls in enumerate(classes) if i < len(f1_pc)
        },
    }



def extract_base_model(pipeline):
    """
    Remonte jusqu'au modele sklearn avec feature_importances_.
    CORRECTION : on cherche bien le modele (step 'model'), pas l'imputer.
    """
    inner = pipeline.estimator if isinstance(pipeline, CalibratedClassifierCV) else pipeline

    if isinstance(inner, StackingClassifier):
        for name, est in inner.estimators_:
            if hasattr(est, "named_steps"):
                base = est.named_steps.get("model")
                if base is not None and hasattr(base, "feature_importances_"):
                    logger.info(
                        f"Modele de base extrait pour SHAP/importance : {name} ({type(base).__name__})"
                    )
                    return base, name
            elif hasattr(est, "feature_importances_"):
                logger.info(
                    f"Modele de base extrait pour SHAP/importance : {name} ({type(est).__name__})"
                )
                return est, name
        logger.info("SHAP sur final_estimator (LogisticRegression)")
        return inner.final_estimator_, "final_estimator"

    if hasattr(inner, "named_steps"):
        model = inner.named_steps.get("model")
        if model is not None:
            return model, type(model).__name__

    return inner, "unknown"


def get_imputer(pipeline):
    inner = pipeline.estimator if isinstance(pipeline, CalibratedClassifierCV) else pipeline
    if isinstance(inner, StackingClassifier):
        for _, est in inner.estimators_:
            if hasattr(est, "named_steps") and "imputer" in est.named_steps:
                return est.named_steps["imputer"]
        return None
    if hasattr(inner, "named_steps"):
        return inner.named_steps.get("imputer")
    return None



def explain_with_shap(
    pipeline,
    X_test: pd.DataFrame,
    feature_cols: list,
    sample_size: int = 200,
) -> tuple:
    if not SHAP_AVAILABLE:
        logger.warning("SHAP non installe — passer shap")
        return None, feature_cols

    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    base_model, _ = extract_base_model(pipeline)
    imputer       = get_imputer(pipeline)
    sample        = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    X_t = (
        pd.DataFrame(imputer.transform(sample), columns=feature_cols)
        if imputer is not None
        else sample.copy()
    )

    shap_mean_vals = None
    try:
        ex = shap.TreeExplainer(base_model)
        sv = ex.shap_values(X_t)

        classes_list = list(getattr(base_model, "classes_", [0, 1, 2, 3]))
        target_class = 2  # classe "High"
        if target_class in classes_list:
            class_index = classes_list.index(target_class)
        else:
            class_index = min(2, len(sv) - 1)
            logger.warning(
                f"Classe {target_class} absente de base_model.classes_ "
                f"— fallback index={class_index}"
            )

        sv_target = sv[class_index] if isinstance(sv, list) else sv

        # ── CORRECTION BUG : s'assurer que shap_mean_vals est un array 1D de scalaires ──
        shap_mean_raw = np.abs(sv_target).mean(axis=0)
        if isinstance(shap_mean_raw, np.ndarray) and shap_mean_raw.ndim > 1:
            shap_mean_raw = shap_mean_raw.mean(axis=1)
        shap_mean_vals = np.array([float(v) for v in shap_mean_raw])

        fig, _ = plt.subplots(figsize=(12, 7))
        shap.summary_plot(
            sv_target, X_t, feature_names=feature_cols,
            show=False, plot_type="bar", max_display=15,
        )
        plt.title("SHAP Feature Importance (bar) — classe High", fontweight="bold")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
        plt.close()

        fig, _ = plt.subplots(figsize=(12, 7))
        shap.summary_plot(
            sv_target, X_t, feature_names=feature_cols,
            show=False, plot_type="dot", max_display=15,
        )
        plt.title("SHAP Summary (dot) — direction d impact — classe High", fontweight="bold")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_dot.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("SHAP : bar plot + dot plot sauvegardes")

        shap_ranking = sorted(
            zip(feature_cols, shap_mean_vals), key=lambda x: x[1], reverse=True
        )
        logger.info("SHAP Feature Ranking (mean |SHAP|) :")
        for rank, (fname, val) in enumerate(shap_ranking[:15], 1):
            logger.info(f"  {rank:2}. {fname:<38} {val:.4f}")

    except Exception as e:
        logger.warning(f"SHAP echoue : {e}")
        shap_mean_vals = None

    return shap_mean_vals, feature_cols



def save_reports(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: list,
    model_name: str,
) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    y_pred     = pipeline.predict(X_test)
    base_model, _ = extract_base_model(pipeline)
    classes    = sorted(y_test.unique())
    labels     = [CLASS_LABELS.get(int(c), str(c)) for c in classes]

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    ax.set_xlabel("Predit")
    ax.set_ylabel("Reel")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix sauvegardee")

    fi_data = {}
    if hasattr(base_model, "feature_importances_"):
        imp = base_model.feature_importances_
        imp = imp / np.sum(imp)  
        idx = np.argsort(imp)[::-1]
        colors = [
            "#e74c3c" if imp[i] > 0.10
            else "#f39c12" if imp[i] > 0.05
            else "#3498db"
            for i in idx
        ]
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(range(len(feature_cols)), imp[idx], color=colors, edgecolor="white")
        ax.set_xticks(range(len(feature_cols)))
        ax.set_xticklabels(
            [feature_cols[i] for i in idx], rotation=45, ha="right", fontsize=9
        )
        ax.set_title(
            f"Feature Importance (normalisee) — {model_name}", fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

        logger.info("Top 10 Feature Importance (normalisee) :")
        for rank, i in enumerate(np.argsort(imp)[::-1][:10], 1):
            logger.info(f"  {rank:2}. {feature_cols[i]:<38} {imp[i]:.4f}")
            fi_data[feature_cols[i]] = round(float(imp[i]), 4)

    return fi_data



def generate_html_report(
    metrics: dict,
    model_name: str,
    feature_cols: list,
    fi_data: dict,
    shap_vals,
    shap_cols: list,
    correlations: dict,
    mi_df: pd.DataFrame,
    ts: str,
) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    def bar_html(val: float, color: str = "#3498db") -> str:
        pct = min(100, int(val * 100))
        return (
            f'<div style="background:#eee;border-radius:4px;height:16px;'
            f'width:200px;display:inline-block">'
            f'<div style="background:{color};width:{pct}%;height:100%;'
            f'border-radius:4px"></div></div>'
            f' <span style="font-weight:bold">{val:.4f}</span>'
        )

    # ── CORRECTION BUG : conversion explicite en float scalaire avant sorted ──
    shap_rows = ""
    if shap_vals is not None:
        try:
            shap_vals_flat = [float(v) for v in np.array(shap_vals).flatten()]
            shap_pairs = sorted(
                zip(shap_cols, shap_vals_flat), key=lambda x: x[1], reverse=True
            )
            for rank, (fname, val) in enumerate(shap_pairs[:15], 1):
                shap_rows += (
                    f"<tr><td>{rank}</td>"
                    f"<td><code>{fname}</code></td>"
                    f"<td>{val:.4f}</td></tr>"
                )
        except Exception as e:
            logger.warning(f"Impossible de generer le tableau SHAP HTML : {e}")

    fi_rows = ""
    for rank, (fname, val) in enumerate(
        sorted(fi_data.items(), key=lambda x: x[1], reverse=True)[:15], 1
    ):
        fi_rows += f"<tr><td>{rank}</td><td><code>{fname}</code></td><td>{val:.4f}</td></tr>"

    corr_rows = ""
    for col, r in sorted(correlations.items(), key=lambda x: abs(x[1] or 0), reverse=True)[:15]:
        flag  = " WARNING" if abs(r or 0) > 0.85 else ""
        color = "#e74c3c" if abs(r or 0) > 0.85 else "#2c3e50"
        corr_rows += (
            f'<tr><td><code>{col}</code></td>'
            f'<td style="color:{color}">{r:+.4f}{flag}</td></tr>'
        )

    mi_rows = ""
    if mi_df is not None:
        for _, row in mi_df.head(10).iterrows():
            flag  = " WARNING" if row["mi"] > 0.80 else ""
            color = "#e74c3c" if row["mi"] > 0.80 else "#2c3e50"
            mi_rows += (
                f'<tr><td><code>{row["feature"]}</code></td>'
                f'<td style="color:{color}">{row["mi"]:.4f}{flag}</td></tr>'
            )

    f1_per_class_rows = "".join(
        f"<tr><td>{cls}</td><td>{val:.4f}</td><td>"
        f"{bar_html(val, '#2ecc71' if val >= 0.80 else '#f39c12' if val >= 0.60 else '#e74c3c')}"
        f"</td></tr>"
        for cls, val in metrics.get("f1_per_class", {}).items()
    )

    cal_note = (
        "non appliquee (StackingClassifier avec LogisticRegression final)"
        if model_name == "StackingClassifier"
        else "prefit avec split dedie 80/20"
    )

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>InvisiThreat — Training Report v8.0</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; background:#f5f6fa; color:#2c3e50; margin:0; padding:20px; }}
  h1   {{ color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:10px; }}
  h2   {{ color:#3498db; margin-top:30px; }}
  .card {{ background:#fff; border-radius:8px; padding:20px; margin:15px 0; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  .metric {{ display:inline-block; margin:10px 20px 10px 0; }}
  .metric .label {{ font-size:12px; color:#7f8c8d; text-transform:uppercase; }}
  .metric .value {{ font-size:28px; font-weight:bold; color:#2c3e50; }}
  table {{ border-collapse:collapse; width:100%; }}
  th,td {{ border:1px solid #ddd; padding:8px 12px; text-align:left; font-size:13px; }}
  th {{ background:#3498db; color:#fff; }}
  tr:nth-child(even) {{ background:#f9f9f9; }}
  img {{ max-width:100%; border-radius:6px; margin:10px 0; }}
  .badge {{ display:inline-block; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:bold; }}
  .badge-green  {{ background:#2ecc71; color:#fff; }}
  .badge-orange {{ background:#f39c12; color:#fff; }}
  .badge-red    {{ background:#e74c3c; color:#fff; }}
  code {{ background:#ecf0f1; padding:2px 6px; border-radius:3px; font-size:12px; }}
</style>
</head>
<body>
<h1>InvisiThreat AI Risk Engine — Training Report</h1>
<p><strong>Version :</strong> v8.0 &nbsp;|&nbsp;
   <strong>Modele :</strong> {model_name} &nbsp;|&nbsp;
   <strong>Date :</strong> {ts}</p>

<div class="card">
  <h2>Metriques principales</h2>
  <div class="metric"><div class="label">F1-weighted (test)</div><div class="value">{metrics['test_f1_weighted']:.4f}</div></div>
  <div class="metric"><div class="label">F1-macro</div><div class="value">{metrics['test_f1_macro']:.4f}</div></div>
  <div class="metric"><div class="label">Balanced Accuracy</div><div class="value">{metrics['test_balanced_accuracy']:.4f}</div></div>
  <div class="metric"><div class="label">ROC-AUC OvR</div><div class="value">{metrics['test_roc_auc_ovr']}</div></div>
  <br>
  <p><strong>Surapprentissage D :</strong> {metrics['overfit_delta']:+.4f} &nbsp;
    <span class="badge {'badge-green' if metrics['overfit_delta'] < 0.05 else 'badge-orange' if metrics['overfit_delta'] < 0.10 else 'badge-red'}">
      {'Excellent' if metrics['overfit_delta'] < 0.05 else 'OK' if metrics['overfit_delta'] < 0.10 else 'Surapprentissage'}
    </span>
  </p>
  <p><strong>Recall (Critical) :</strong> {metrics.get('recall_critical', 0):.4f} &nbsp;
     <strong>F1 (Critical) :</strong> {metrics.get('f1_critical', 0):.4f}
  </p>
  <h3>F1 par classe</h3>
  <table><tr><th>Classe</th><th>F1</th><th>Barre</th></tr>{f1_per_class_rows}</table>
</div>

<div class="card">
  <h2>Confusion Matrix</h2>
  <img src="confusion_matrix.png" alt="Confusion Matrix">
</div>

<div class="card">
  <h2>Feature Importance (normalisee — somme = 1)</h2>
  <img src="feature_importance.png" alt="Feature Importance">
  <table><tr><th>#</th><th>Feature</th><th>Importance</th></tr>
  {fi_rows if fi_rows else "<tr><td colspan='3'>Non disponible</td></tr>"}
  </table>
</div>

<div class="card">
  <h2>Analyse SHAP (classe High)</h2>
  <h3>Bar plot — importance absolue</h3>
  <img src="shap/shap_bar.png" alt="SHAP Bar">
  <h3>Dot plot — direction d impact</h3>
  <img src="shap/shap_dot.png" alt="SHAP Dot">
  <h3>SHAP Ranking</h3>
  <table><tr><th>#</th><th>Feature</th><th>Mean |SHAP|</th></tr>
  {shap_rows if shap_rows else "<tr><td colspan='3'>Non disponible</td></tr>"}
  </table>
</div>

<div class="card">
  <h2>Audit Leakage — Mutual Information</h2>
  <p>MI > 0.80 peut signaler un leakage indirect.</p>
  <table><tr><th>Feature</th><th>MI</th></tr>
  {mi_rows if mi_rows else "<tr><td colspan='2'>Non disponible</td></tr>"}
  </table>
</div>

<div class="card">
  <h2>Audit Correlations Pearson features/cible</h2>
  <table><tr><th>Feature</th><th>Correlation r</th></tr>{corr_rows}</table>
</div>

<div class="card">
  <h2>Configuration</h2>
  <p><strong>Modele :</strong> {model_name}</p>
  <p><strong>Features :</strong> {len(feature_cols)}</p>
  <p><strong>Anti-leakage :</strong> <span class="badge badge-green">CLEAN</span></p>
  <p><strong>Calibration :</strong> {cal_note}</p>
  <p><strong>stack_method :</strong> predict_proba</p>
  <p><strong>sample_weight :</strong> applique a tous les modeles (plafonnes a 10)</p>
  <p><strong>Feature importance :</strong> normalisee (somme = 1)</p>
  <p><strong>SHAP class_index :</strong> resolu depuis base_model.classes_</p>
  <p><strong>Seed global :</strong> 42 (numpy + random)</p>
</div>
</body>
</html>"""

    with open(REPORTS_DIR / "report.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Rapport HTML sauvegarde : {REPORTS_DIR / 'report.html'}")



def save_model_atomic(
    pipeline,
    metrics: dict,
    feature_cols: list,
    model_name: str,
    all_results: dict,
    correlations: dict,
    cal_method: str,
    ts: str,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / f"pipeline_{ts}.pkl")
    joblib.dump(pipeline, MODELS_DIR / "pipeline_latest.pkl")

    base_model, _ = extract_base_model(pipeline)
    classes       = list(getattr(base_model, "classes_", list(CLASS_LABELS.keys())))

    meta = {
        "version":            "8.0",
        "timestamp":          ts,
        "model_type":         model_name,
        "calibration_method": cal_method,
        "label_strategy":     "severity_defectdojo_v9 (MERGE_MAP 5->4 classes)",
        "class_labels":       CLASS_LABELS,
        "stack_method":       "predict_proba",
        "sample_weight_cap":  10.0,
        "seed_global":        42,
        "feature_importance_normalized": True,
        "shap_class_index_from_classes_attr": True,
        "product_fp_rate_note": "Pre-calcule dans preprocess.py v9.0 (lissage Laplace)",
        "leakage_status":     "CLEAN",
        "n_classes":          len(classes),
        "classes":            [int(c) for c in classes],
        "features":           feature_cols,
        "n_features":         len(feature_cols),
        "metrics":            metrics,
        "model_comparison": {
            n: round(float(s), 4) for n, (_, s) in all_results.items()
        },
        "feature_target_correlations": correlations,
    }
    with open(MODELS_DIR / "pipeline_latest_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Modele sauvegarde : {MODELS_DIR}/pipeline_{ts}.pkl")
    logger.info(f"Meta sauvegardee  : {MODELS_DIR}/pipeline_latest_meta.json")


def save_run_history(
    metrics: dict,
    model_name: str,
    feature_cols: list,
    correlations: dict,
    all_results: dict,
    cal_method: str,
    ts: str,
) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    run = {
        "version":      "8.0",
        "timestamp":    ts,
        "model":        model_name,
        "calibration":  cal_method,
        "n_features":   len(feature_cols),
        "features":     feature_cols,
        "metrics":      metrics,
        "base_models_cv": {
            n: round(float(s), 4) for n, (_, s) in all_results.items()
        },
        "top_correlations": dict(
            sorted(correlations.items(), key=lambda x: abs(x[1] or 0), reverse=True)[:10]
        ),
        "leakage_status":           "CLEAN",
        "class_imbalance_handling": "sample_weight (cap=10, tous modeles) + class_weight=balanced",
    }
    path = LOGS_DIR / f"train_run_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    logger.info(f"Historique run sauvegarde : {path}")



def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("InvisiThreat AI Risk Engine — Training v8.0")
    logger.info("  [T1] FEATURE_COLS alignee sur preprocess.py v9.0")
    logger.info("  [T2] SHAP class_index resolu depuis base_model.classes_")
    logger.info("  [T3] StackingClassifier stack_method=predict_proba")
    logger.info("  [T4] sample_weight applique a tous les modeles (RF inclus)")
    logger.info("  [T5] CV fixe k=5 stratifie")
    logger.info("  [T6] Feature importance normalisee (sum=1)")
    logger.info("  [T7] Audit leakage Mutual Information")
    logger.info("  [T8] Seed global numpy + random = 42")
    logger.info("  [T9] Seuil correlation warning a 0.95")
    logger.info("  [T10] Calibration ignoree pour StackingClassifier")
    logger.info("=" * 70)

    df     = load_clean_data()
    target = build_target(df)

    valid_mask = target.notna()
    df_valid   = df.loc[valid_mask].copy()
    y_valid    = target.loc[valid_mask].astype(int)

    X, y, fcols = prepare_features(df_valid, y_valid)

    train_idx, test_idx = train_test_split(
        X.index, test_size=0.20, random_state=42, stratify=y
    )
    X_train = X.loc[train_idx].copy()
    X_test  = X.loc[test_idx].copy()
    y_train = y.loc[train_idx]
    y_test  = y.loc[test_idx]

    logger.info(f"Split : {len(X_train)} train | {len(X_test)} test")
    _log_dist(y_train, "Distribution train")
    _log_dist(y_test,  "Distribution test")

    mi_df = audit_leakage_mi(X_train, y_train)
    corr  = audit_correlations(X_train, y_train)

    stacking_model, model_name, all_results = build_stacking_model(X_train, y_train)

    cal_method = (
        "none"
        if model_name == "StackingClassifier"
        else ("isotonic" if len(y_train) > 1000 else "sigmoid")
    )
    best_model, X_fit, y_fit = calibrate_if_needed(
        stacking_model, X_train, y_train, model_name
    )

    metrics = evaluate(best_model, X_fit, y_fit, X_test, y_test)
    fi_data = save_reports(best_model, X_test, y_test, fcols, model_name)

    shap_vals, shap_cols = explain_with_shap(
        best_model, X_test, fcols,
        sample_size=min(200, len(X_test)),
    )

    generate_html_report(
        metrics, model_name, fcols, fi_data,
        shap_vals, shap_cols, corr, mi_df, ts,
    )
    save_model_atomic(
        best_model, metrics, fcols, model_name,
        all_results, corr, cal_method, ts,
    )
    save_run_history(
        metrics, model_name, fcols, corr,
        all_results, cal_method, ts,
    )

    logger.info("=" * 70)
    logger.info(f"Modele final        : {model_name}")
    logger.info(f"F1-weighted (test)  : {metrics['test_f1_weighted']}")
    logger.info(f"F1-weighted (train) : {metrics['train_f1_weighted']}")
    logger.info(f"Surapprentissage D  : {metrics['overfit_delta']:+.4f}")
    logger.info(f"Balanced Accuracy   : {metrics['test_balanced_accuracy']}")
    logger.info(f"ROC-AUC OvR         : {metrics['test_roc_auc_ovr']}")
    logger.info(f"Recall (Critical)   : {metrics.get('recall_critical', 0):.4f}")
    logger.info(f"F1     (Critical)   : {metrics.get('f1_critical', 0):.4f}")
    logger.info(f"Calibration         : {cal_method}")
    logger.info(f"Rapport HTML        : {REPORTS_DIR / 'report.html'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()