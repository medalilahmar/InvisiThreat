import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/main.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("risk_engine.main")


SRC_DIR       = Path(__file__).parent
DATA_RAW      = SRC_DIR / "data" / "raw"
DATA_PROCESSED = SRC_DIR / "data" / "processed"
MODELS_DIR    = SRC_DIR / "models"
REPORTS_DIR   = SRC_DIR / "reports"
LOGS_DIR      = SRC_DIR / "logs"

PIPELINE_STEPS = {
    "fetch":      SRC_DIR / "fetch_data.py",
    "preprocess": SRC_DIR / "preprocess.py",
    "train":      SRC_DIR / "train.py",
    "serve":      SRC_DIR / "api.py",
}

EXPECTED_FILES = {
    "fetch": [
        DATA_RAW / "findings_raw.csv",
        DATA_RAW / "products.csv",
        DATA_RAW / "engagements.csv",
    ],
    "preprocess": [
        DATA_PROCESSED / "findings_clean.csv",
        DATA_PROCESSED / "data_report.json",
    ],
    "train": [
        MODELS_DIR / "pipeline_latest.pkl",
        MODELS_DIR / "pipeline_latest_meta.json",
    ],
}


def _banner(title: str, char: str = "═") -> None:
    line = char * 56
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def _check_python() -> None:
    """Vérifie que Python >= 3.10."""
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ requis. Version actuelle : {sys.version}")
        sys.exit(1)


def _check_dependencies() -> bool:
    """Vérifie que les dépendances critiques sont installées."""
    required = [
        "pandas", "numpy", "sklearn", "joblib",
        "fastapi", "uvicorn", "pydantic",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != "sklearn" else "sklearn")
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Dépendances manquantes : {missing}")
        logger.error("Installez-les avec : pip install -r requirements.txt")
        return False
    return True


def _run_script(script_path: Path, step_name: str) -> bool:
    """
    Exécute un script Python en subprocess.
    Capture stdout/stderr et les redirige vers le logger.
    Retourne True si succès, False sinon.
    """
    if not script_path.exists():
        logger.error(f"Script introuvable : {script_path}")
        return False

    logger.info(f"▶  Exécution : {script_path.name}")
    start = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(SRC_DIR),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"   │ {line}")

        for line in result.stderr.splitlines():
            if line.strip():
                level = logging.WARNING if result.returncode == 0 else logging.ERROR
                logger.log(level, f"   │ {line}")

        duration = time.perf_counter() - start

        if result.returncode == 0:
            logger.info(f"✅ {step_name} terminé en {duration:.1f}s")
            return True
        else:
            logger.error(f"❌ {step_name} échoué (code={result.returncode}) en {duration:.1f}s")
            return False

    except Exception as e:
        logger.exception(f"❌ Erreur inattendue lors de {step_name} : {e}")
        return False


def _validate_outputs(step: str) -> bool:
    """Vérifie que les fichiers de sortie attendus existent après une étape."""
    expected = EXPECTED_FILES.get(step, [])
    missing  = [f for f in expected if not f.exists()]
    if missing:
        logger.warning(f"Fichiers attendus après '{step}' introuvables :")
        for f in missing:
            logger.warning(f"   ✗ {f}")
        return False
    logger.info(f"Sorties validées pour '{step}' ✓")
    return True


def _init_directories() -> None:
    """Crée l'arborescence du projet si elle n'existe pas."""
    dirs = [
        DATA_RAW, DATA_PROCESSED,
        MODELS_DIR, REPORTS_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Arborescence projet initialisée ✓")


def cmd_fetch() -> bool:
    """Étape 1 : Récupère les données depuis DefectDojo."""
    _banner("ÉTAPE 1 / 3 — FETCH DATA")
    success = _run_script(PIPELINE_STEPS["fetch"], "fetch_data")
    if success:
        _validate_outputs("fetch")
    return success


def cmd_preprocess() -> bool:
    """Étape 2 : Prétraitement et feature engineering."""
    _banner("ÉTAPE 2 / 3 — PREPROCESS")

    raw_csv = DATA_RAW / "findings_raw.csv"
    if not raw_csv.exists():
        logger.error(f"Données brutes introuvables : {raw_csv}")
        logger.error("Exécutez d'abord : python main.py fetch")
        return False

    success = _run_script(PIPELINE_STEPS["preprocess"], "preprocess")
    if success:
        _validate_outputs("preprocess")
        _print_data_report()
    return success


def cmd_train() -> bool:
    """Étape 3 : Entraînement du modèle ML."""
    _banner("ÉTAPE 3 / 3 — TRAIN MODEL")

    clean_csv = DATA_PROCESSED / "findings_clean.csv"
    if not clean_csv.exists():
        logger.error(f"Données prétraitées introuvables : {clean_csv}")
        logger.error("Exécutez d'abord : python main.py preprocess")
        return False

    success = _run_script(PIPELINE_STEPS["train"], "train")
    if success:
        _validate_outputs("train")
        _print_model_meta()
    return success


def cmd_serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Lance l'API FastAPI avec uvicorn."""
    _banner("API — AI RISK ENGINE")

    model_pkl = MODELS_DIR / "pipeline_latest.pkl"
    if not model_pkl.exists():
        logger.warning(
            f"Modèle introuvable : {model_pkl}\n"
            "L'API démarrera mais /predict renverra 503.\n"
            "Exécutez d'abord : python main.py run"
        )

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn non installé. Installez-le : pip install uvicorn")
        sys.exit(1)

    logger.info(f"Démarrage API sur http://{host}:{port}")
    logger.info(f"Documentation : http://localhost:{port}/docs")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        app_dir=str(SRC_DIR),
    )


def cmd_run(skip_fetch: bool = False) -> bool:
    """Pipeline complet : fetch → preprocess → train."""
    _banner("PIPELINE COMPLET — AI RISK ENGINE v2.0", char="█")
    logger.info(f"Démarrage : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    total_start = time.perf_counter()

    results: dict[str, bool] = {}

    if not skip_fetch:
        results["fetch"] = cmd_fetch()
        if not results["fetch"]:
            logger.error("Pipeline interrompu : fetch échoué.")
            _print_summary(results, time.perf_counter() - total_start)
            return False
    else:
        logger.info("⏭  Fetch ignoré (--skip-fetch)")

    results["preprocess"] = cmd_preprocess()
    if not results["preprocess"]:
        logger.error("Pipeline interrompu : prétraitement échoué.")
        _print_summary(results, time.perf_counter() - total_start)
        return False

    results["train"] = cmd_train()

    total_duration = time.perf_counter() - total_start
    _print_summary(results, total_duration)
    return all(results.values())


def cmd_status() -> None:
    """Affiche un tableau de bord de l'état du projet."""
    _banner("STATUS — AI RISK ENGINE")

    print("\n📁 Données")
    _status_file(DATA_RAW / "findings_raw.csv",         "Findings bruts")
    _status_file(DATA_RAW / "products.csv",             "Produits")
    _status_file(DATA_RAW / "engagements.csv",          "Engagements")
    _status_file(DATA_PROCESSED / "findings_clean.csv", "Findings prétraités")

    report_path = DATA_PROCESSED / "data_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        print(f"\n📊 Données prétraitées (rapport : {report.get('timestamp', '?')[:19]})")
        print(f"   Lignes        : {report.get('n_rows', '?')}")
        print(f"   Colonnes      : {report.get('n_cols', '?')}")
        risk_stats = report.get("risk_score_stats", {})
        if risk_stats:
            print(f"   risk_score    : min={risk_stats.get('min')}  max={risk_stats.get('max')}  "
                  f"mean={risk_stats.get('mean')}  std={risk_stats.get('std')}")
        dist = report.get("risk_class_dist", {})
        if dist:
            labels = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
            print("   Distribution  :", "  ".join(
                f"{labels.get(int(k), k)}={v}" for k, v in sorted(dist.items(), key=lambda x: int(x[0]))
            ))

    print("\n🤖 Modèle")
    _status_file(MODELS_DIR / "pipeline_latest.pkl",       "Pipeline sklearn")
    _status_file(MODELS_DIR / "pipeline_latest_meta.json", "Métadonnées")
    _print_model_meta(verbose=True)

    print("\n📈 Rapports")
    _status_file(REPORTS_DIR / "confusion_matrix.png",   "Matrice de confusion")
    _status_file(REPORTS_DIR / "feature_importance.png", "Importance features")
    _status_file(REPORTS_DIR / "prediction_analysis.png", "Analyse prédictions")

    print("\n📋 Logs")
    for log_file in sorted(LOGS_DIR.glob("*.log")):
        size_kb = log_file.stat().st_size / 1024
        mtime   = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"   {'✓':2} {log_file.name:<25} {size_kb:6.1f} KB   {mtime}")

    print()

def _status_file(path: Path, label: str) -> None:
    """Affiche le statut d'un fichier (présent/absent + taille)."""
    if path.exists():
        size_kb = path.stat().st_size / 1024
        mtime   = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"   ✅ {label:<35} {size_kb:7.1f} KB   {mtime}")
    else:
        print(f"   ❌ {label:<35} (absent)")


def _print_data_report() -> None:
    """Affiche un résumé du rapport de données après preprocess."""
    report_path = DATA_PROCESSED / "data_report.json"
    if not report_path.exists():
        return
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    logger.info("── Rapport données ──────────────────────────")
    logger.info(f"   Lignes    : {report.get('n_rows')}")
    logger.info(f"   Colonnes  : {report.get('n_cols')}")
    stats = report.get("risk_score_stats", {})
    if stats:
        logger.info(
            f"   risk_score : min={stats.get('min')}  max={stats.get('max')}  "
            f"mean={stats.get('mean')}  std={stats.get('std')}"
        )
    dist = report.get("risk_class_dist", {})
    if dist:
        labels = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
        logger.info("   Classes   : " + "  ".join(
            f"{labels.get(int(k), k)}={v}" for k, v in sorted(dist.items(), key=lambda x: int(x[0]))
        ))


def _print_model_meta(verbose: bool = False) -> None:
    """Affiche les métadonnées du modèle entraîné."""
    meta_path = MODELS_DIR / "pipeline_latest_meta.json"
    if not meta_path.exists():
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if verbose:
        print(f"   Version       : {meta.get('timestamp', '?')}")
        print(f"   Task          : {meta.get('task', '?')}")
        print(f"   Classes       : {meta.get('n_classes')} ({meta.get('classes')})")
        print(f"   Features      : {meta.get('n_features')}")
        metrics = meta.get("metrics", {})
        if metrics:
            print(f"   F1-weighted   : {metrics.get('test_f1_weighted', '?')}")
            print(f"   F1-macro      : {metrics.get('test_f1_macro', '?')}")
            print(f"   ROC-AUC OvR   : {metrics.get('test_roc_auc_ovr', '?')}")
    else:
        metrics = meta.get("metrics", {})
        logger.info("── Métadonnées modèle ───────────────────────")
        logger.info(f"   Version     : {meta.get('timestamp', '?')}")
        logger.info(f"   Classes     : {meta.get('n_classes')}")
        logger.info(f"   Features    : {meta.get('n_features')}")
        if metrics:
            logger.info(f"   F1-weighted : {metrics.get('test_f1_weighted', '?')}")
            logger.info(f"   ROC-AUC OvR : {metrics.get('test_roc_auc_ovr', '?')}")


def _print_summary(results: dict[str, bool], duration: float) -> None:
    """Affiche le récapitulatif final du pipeline."""
    _banner("RÉSUMÉ DU PIPELINE")
    icons = {True: "✅", False: "❌"}
    for step, ok in results.items():
        logger.info(f"   {icons[ok]}  {step:<15} {'OK' if ok else 'ÉCHOUÉ'}")
    logger.info(f"\n   Durée totale : {duration:.1f}s")
    if all(results.values()):
        logger.info("\n Pipeline terminé avec succès.")
        logger.info("   Lancez l'API : python main.py serve")
    else:
        logger.error("\n Pipeline terminé avec des erreurs. Consultez logs/main.log")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="AI Risk Engine — Orchestrateur du pipeline DevSecOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py run                  # Pipeline complet (fetch + preprocess + train)
  python main.py run --skip-fetch     # Pipeline sans fetch (données déjà présentes)
  python main.py fetch                # Récupère les données DefectDojo uniquement
  python main.py preprocess           # Prétraitement uniquement
  python main.py train                # Entraînement uniquement
  python main.py serve                # Lance l'API sur le port 8000
  python main.py serve --port 9000    # Lance l'API sur un port custom
  python main.py serve --reload       # API en mode développement (hot-reload)
  python main.py status               # Affiche l'état complet du projet
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    p_run = subparsers.add_parser("run", help="Pipeline complet (fetch + preprocess + train)")
    p_run.add_argument("--skip-fetch", action="store_true",
                       help="Ignore l'étape fetch (données brutes déjà présentes)")

    subparsers.add_parser("fetch", help="Récupère les données depuis DefectDojo")

    subparsers.add_parser("preprocess", help="Prétraitement et feature engineering")

    subparsers.add_parser("train", help="Entraîne et sauvegarde le modèle ML")

    p_serve = subparsers.add_parser("serve", help="Lance l'API FastAPI")
    p_serve.add_argument("--host",   default="0.0.0.0",  help="Hôte (défaut: 0.0.0.0)")
    p_serve.add_argument("--port",   default=8000, type=int, help="Port (défaut: 8000)")
    p_serve.add_argument("--reload", action="store_true",   help="Hot-reload (développement)")

    subparsers.add_parser("status", help="Affiche l'état du projet")

    return parser


def main() -> None:
    _check_python()

    parser = build_parser()
    args   = parser.parse_args()

    _init_directories()

    if args.command != "status" and not _check_dependencies():
        sys.exit(1)

    if args.command == "run":
        success = cmd_run(skip_fetch=args.skip_fetch)
        sys.exit(0 if success else 1)

    elif args.command == "fetch":
        success = cmd_fetch()
        sys.exit(0 if success else 1)

    elif args.command == "preprocess":
        success = cmd_preprocess()
        sys.exit(0 if success else 1)

    elif args.command == "train":
        success = cmd_train()
        sys.exit(0 if success else 1)

    elif args.command == "serve":
        cmd_serve(host=args.host, port=args.port, reload=args.reload)

    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()