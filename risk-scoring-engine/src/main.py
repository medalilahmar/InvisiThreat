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

SRC_DIR        = Path(__file__).parent
DATA_RAW       = SRC_DIR / "data" / "raw"
DATA_PROCESSED = SRC_DIR / "data" / "processed"
MODELS_DIR     = SRC_DIR / "models"
REPORTS_DIR    = SRC_DIR / "reports"
LOGS_DIR       = SRC_DIR / "logs"

SCRIPTS = {
    "fetch":      SRC_DIR / "fetch_data.py",
    "preprocess": SRC_DIR / "preprocess.py",
    "train":      SRC_DIR / "train.py",
    "tag":        SRC_DIR / "tag_findings.py",
    "predict":    SRC_DIR / "predict_live.py",
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

DEFECTDOJO_URL = os.environ.get("DEFECTDOJO_URL", "http://192.168.11.170:8080")
DEFECTDOJO_API_KEY = os.environ.get("DEFECTDOJO_API_KEY", "")



def _banner(title: str, char: str = "═") -> None:
    line = char * 60
    logger.info(line)
    logger.info(f"  {title}")
    logger.info(line)


def _check_python() -> None:
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ requis. Version actuelle : {sys.version}")
        sys.exit(1)


def _check_dependencies() -> bool:
    required = ["pandas", "numpy", "sklearn", "joblib", "requests"]
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


def _run_script(script_path: Path, step_name: str, extra_args: list = None) -> bool:
    
    if not script_path.exists():
        logger.error(f"Script introuvable : {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    logger.info(f" Exécution : {script_path.name} {' '.join(extra_args or [])}")
    start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
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
            logger.info(f" {step_name} terminé en {duration:.1f}s")
            return True
        else:
            logger.error(f" {step_name} échoué (code={result.returncode})")
            return False

    except Exception as e:
        logger.exception(f"Erreur lors de {step_name} : {e}")
        return False


def _validate_outputs(step: str) -> bool:
    expected = EXPECTED_FILES.get(step, [])
    missing = [f for f in expected if not f.exists()]
    if missing:
        logger.warning(f"Fichiers attendus après '{step}' introuvables :")
        for f in missing:
            logger.warning(f"   ✗ {f}")
        return False
    logger.info(f" Sorties validées pour '{step}'")
    return True


def _init_directories() -> None:
    dirs = [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, LOGS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug("Arborescence initialisée")


def _print_data_report() -> None:
    report_path = DATA_PROCESSED / "data_report.json"
    if not report_path.exists():
        return

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    logger.info(" Rapport des données")
    logger.info(f"   Lignes      : {report.get('n_rows')}")
    logger.info(f"   Colonnes    : {report.get('n_cols')}")

    stats = report.get("risk_score_stats", {})
    if stats:
        logger.info(f"   risk_score  : min={stats.get('min')}  max={stats.get('max')}  "
                    f"mean={stats.get('mean')}  std={stats.get('std')}")

    dist = report.get("risk_class_dist", {})
    if dist:
        labels = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
        dist_str_list = []
        for k, v in sorted(dist.items(), key=lambda x: float(x[0])):
            try:
                key_int = int(float(k))  
            except (ValueError, TypeError):
                key_int = k
            label = labels.get(key_int, str(k))
            dist_str_list.append(f"{label}={v}")
        
        logger.info(f"   Classes     : {', '.join(dist_str_list)}")



def _print_model_meta() -> None:
    meta_path = MODELS_DIR / "pipeline_latest_meta.json"
    if not meta_path.exists():
        return

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    logger.info("🤖 Métadonnées du modèle")
    logger.info(f"   Version     : {meta.get('timestamp', '?')}")
    logger.info(f"   Classes     : {meta.get('n_classes')}")
    logger.info(f"   Features    : {meta.get('n_features')}")

    metrics = meta.get("metrics", {})
    if metrics:
        f1 = metrics.get('test_f1_weighted', '?')
        roc_auc = metrics.get('test_roc_auc_ovr', '?')
        logger.info(f"   F1-weighted : {f1 if f1 is not None else 'N/A'}")
        logger.info(f"   ROC-AUC OvR : {roc_auc if roc_auc is not None else 'N/A'}")


def _print_summary(results: dict[str, bool], duration: float) -> None:
    """Affiche le résumé du pipeline"""
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


def _status_file(path: Path, label: str) -> None:
    """Affiche le statut d'un fichier"""
    if path.exists():
        size_kb = path.stat().st_size / 1024
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"    {label:<35} {size_kb:7.1f} KB   {mtime}")
    else:
        print(f"    {label:<35} (absent)")



def cmd_fetch() -> bool:
    """Étape 1 : Récupère les données depuis DefectDojo"""
    _banner("ÉTAPE 1 — FETCH DATA")
    success = _run_script(SCRIPTS["fetch"], "fetch_data")
    if success:
        _validate_outputs("fetch")
    return success


def cmd_preprocess() -> bool:
    """Étape 2 : Prétraitement et feature engineering"""
    _banner("ÉTAPE 2 — PREPROCESS")

    raw_csv = DATA_RAW / "findings_raw.csv"
    if not raw_csv.exists():
        logger.error(f"Données brutes introuvables : {raw_csv}")
        logger.error("Exécutez d'abord : python main.py fetch")
        return False

    success = _run_script(SCRIPTS["preprocess"], "preprocess")
    if success:
        _validate_outputs("preprocess")
        _print_data_report()
    return success


def cmd_train() -> bool:
    """Étape 3 : Entraînement du modèle ML"""
    _banner("ÉTAPE 3 — TRAIN MODEL")

    clean_csv = DATA_PROCESSED / "findings_clean.csv"
    if not clean_csv.exists():
        logger.error(f"Données prétraitées introuvables : {clean_csv}")
        logger.error("Exécutez d'abord : python main.py preprocess")
        return False

    success = _run_script(SCRIPTS["train"], "train")
    if success:
        _validate_outputs("train")
        _print_model_meta()
    return success


def cmd_tag(engagement_id: int, dry_run: bool = False, all_findings: bool = False) -> bool:
    """Commande : Tagging intelligent des findings"""
    _banner("TAGGING INTELLIGENT")

    args = [
        "--dd-url", DEFECTDOJO_URL,
        "--engagement-id", str(engagement_id)
    ]

    if DEFECTDOJO_API_KEY:
        args.extend(["--dd-api-key", DEFECTDOJO_API_KEY])

    if dry_run:
        args.append("--dry-run")
    if all_findings:
        args.append("--all")

    success = _run_script(SCRIPTS["tag"], "Tagging intelligent", args)
    return success


def cmd_predict(engagement_id: int, update_dojo: bool = True) -> bool:
    """Commande : Scoring IA en live"""
    _banner("AI RISK SCORING")

    model_path = MODELS_DIR / "pipeline_latest.pkl"
    if not model_path.exists():
        logger.error(f"Modèle introuvable : {model_path}")
        logger.error("Exécutez d'abord : python main.py run")
        return False

    args = [
        "--engagement-id", str(engagement_id),
        "--dd-url", DEFECTDOJO_URL
    ]

    if DEFECTDOJO_API_KEY:
        args.extend(["--dd-api-key", DEFECTDOJO_API_KEY])

    if update_dojo:
        args.append("--update-dojo")

    success = _run_script(SCRIPTS["predict"], "Scoring IA", args)
    return success


def cmd_full(engagement_id: int, dry_run: bool = False) -> bool:
    """Pipeline complet : Tagging + Predict"""
    _banner("PIPELINE COMPLET — TAG + PREDICT", char="█")

    results = {}

    logger.info(f" Étape 1/2 — Tagging (engagement {engagement_id})")
    results["tag"] = cmd_tag(engagement_id, dry_run=dry_run)

    if not results["tag"]:
        logger.error(" Tagging échoué → arrêt du pipeline")
        return False

    logger.info(f"\n Étape 2/2 — Scoring IA")
    results["predict"] = cmd_predict(engagement_id, update_dojo=not dry_run)

    _banner("RÉSULTAT")
    for step, ok in results.items():
        logger.info(f"   {'✅' if ok else '❌'}  {step}")
    logger.info(f"\n   Engagement : {engagement_id}")
    logger.info(f"   Mode dry-run : {'oui' if dry_run else 'non'}")

    return all(results.values())


def cmd_run(skip_fetch: bool = False, engagement_id: int = None) -> bool:
    """Pipeline complet : tag → fetch → preprocess → train"""
    _banner("PIPELINE COMPLET — ML TRAINING", char="█")
    logger.info(f"Démarrage : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    total_start = time.perf_counter()
    results: dict[str, bool] = {}

    if engagement_id:
        logger.info(f" Étape 0 — Tagging intelligent (engagement {engagement_id})")
        results["tag"] = cmd_tag(engagement_id, dry_run=False)
        if not results["tag"]:
            logger.warning(" Tagging partiel, continuation du pipeline...")
    else:
        logger.info("ℹ️ Aucun engagement spécifié, tagging ignoré")
        results["tag"] = True  

    if not skip_fetch:
        results["fetch"] = cmd_fetch()
        if not results["fetch"]:
            logger.error("Pipeline interrompu : fetch échoué.")
            _print_summary(results, time.perf_counter() - total_start)
            return False
    else:
        logger.info("⏭ Fetch ignoré (--skip-fetch)")

    results["preprocess"] = cmd_preprocess()
    if not results["preprocess"]:
        logger.error("Pipeline interrompu : prétraitement échoué.")
        _print_summary(results, time.perf_counter() - total_start)
        return False

    results["train"] = cmd_train()

    total_duration = time.perf_counter() - total_start
    _print_summary(results, total_duration)

    return all(results.values())


def cmd_serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Lance l'API FastAPI"""
    _banner("API — AI RISK ENGINE")

    model_path = MODELS_DIR / "pipeline_latest.pkl"
    if not model_path.exists():
        logger.warning(f" Modèle introuvable : {model_path}")
        logger.warning("L'API démarrera mais les prédictions renverront 503")

    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn non installé. Installez : pip install uvicorn")
        sys.exit(1)

    logger.info(f" Démarrage API sur http://{host}:{port}")
    logger.info(f" Documentation : http://localhost:{port}/docs")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        app_dir=str(SRC_DIR),
    )


def cmd_status() -> None:
    _banner("STATUS — AI RISK ENGINE")

    print("\n Données")
    _status_file(DATA_RAW / "findings_raw.csv", "Findings bruts")
    _status_file(DATA_RAW / "products.csv", "Produits")
    _status_file(DATA_RAW / "engagements.csv", "Engagements")
    _status_file(DATA_PROCESSED / "findings_clean.csv", "Findings prétraités")

    report_path = DATA_PROCESSED / "data_report.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        print(f"\n Données prétraitées ({report.get('timestamp', '?')[:19]})")
        print(f"   Lignes      : {report.get('n_rows', '?')}")
        print(f"   Colonnes    : {report.get('n_cols', '?')}")
        dist = report.get("risk_class_dist", {})
        if dist:
            labels = {0: "Info", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
            dist_str_list = []
            for k, v in sorted(dist.items(), key=lambda x: float(x[0])):
                try:
                    key_int = int(float(k))
                except (ValueError, TypeError):
                    key_int = k
                label = labels.get(key_int, str(k))
                dist_str_list.append(f"{label}={v}")
            print(f"   Distribution: {', '.join(dist_str_list)}")

    print("\n Modèle")
    _status_file(MODELS_DIR / "pipeline_latest.pkl", "Pipeline sklearn")
    _status_file(MODELS_DIR / "pipeline_latest_meta.json", "Métadonnées")

    meta_path = MODELS_DIR / "pipeline_latest_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        metrics = meta.get("metrics", {})
        if metrics:
            f1 = metrics.get('test_f1_weighted', '?')
            roc_auc = metrics.get('test_roc_auc_ovr', '?')
            print(f"   F1-weighted : {f1 if f1 is not None else 'N/A'}")
            print(f"   ROC-AUC OvR : {roc_auc if roc_auc is not None else 'N/A'}")

    print("\n Rapports")
    _status_file(REPORTS_DIR / "confusion_matrix.png", "Matrice de confusion")
    _status_file(REPORTS_DIR / "feature_importance.png", "Importance features")

    print("\n Logs")
    for log_file in sorted(LOGS_DIR.glob("*.log")):
        size_kb = log_file.stat().st_size / 1024
        mtime = datetime.fromtimestamp(log_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"   ✅ {log_file.name:<25} {size_kb:6.1f} KB   {mtime}")

    print()



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="InvisiThreat AI Risk Engine — Orchestrateur complet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
╔══════════════════════════════════════════════════════════════════════╗
║ COMMANDES PRINCIPALES                                                ║
╠══════════════════════════════════════════════════════════════════════╣
║ python main.py run --engagement-id 5     # Pipeline ML avec tagging  ║
║ python main.py run --skip-fetch          # ML sans re-fetch          ║
║ python main.py tag --engagement-id 5     # Tagging intelligent       ║
║ python main.py predict --engagement-id 5 # Scoring IA en live        ║
║ python main.py full --engagement-id 5    # Tagging + Scoring complet ║
║ python main.py serve                     # API FastAPI               ║
║ python main.py status                    # État complet du projet    ║
╚══════════════════════════════════════════════════════════════════════╝

Exemples détaillés :
  # Pipeline ML avec tagging (RECOMMANDÉ)
  python main.py run --engagement-id 123
  
  # Pipeline ML sans re-fetch (données déjà présentes)
  python main.py run --skip-fetch --engagement-id 123
  
  # Tagging intelligent (post-import)
  python main.py tag --engagement-id 123 --dry-run
  python main.py tag --engagement-id 123
  
  # Scoring IA en live
  python main.py predict --engagement-id 123
  
  # Pipeline complet CI/CD (tag + score)
  python main.py full --engagement-id 123
  
  # API pour dashboard React
  python main.py serve --port 8000
  
  # Vérifier l'état
  python main.py status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    p_run = subparsers.add_parser("run", help="Pipeline complet ML (tag + fetch + preprocess + train)")
    p_run.add_argument("--skip-fetch", action="store_true", help="Ignore l'étape fetch")
    p_run.add_argument("--engagement-id", type=int, help="ID de l'engagement pour le tagging préalable (RECOMMANDÉ)")

    subparsers.add_parser("fetch", help="Récupère les données depuis DefectDojo")
    subparsers.add_parser("preprocess", help="Prétraitement et feature engineering")
    subparsers.add_parser("train", help="Entraîne et sauvegarde le modèle ML")

    p_tag = subparsers.add_parser("tag", help="Tagging intelligent des findings")
    p_tag.add_argument("--engagement-id", required=True, type=int, help="ID de l'engagement")
    p_tag.add_argument("--dry-run", action="store_true", help="Mode simulation (sans modification)")
    p_tag.add_argument("--all", action="store_true", help="Tagger aussi les findings inactifs")

    p_predict = subparsers.add_parser("predict", help="Scoring IA en live")
    p_predict.add_argument("--engagement-id", required=True, type=int, help="ID de l'engagement")

    p_full = subparsers.add_parser("full", help="Pipeline complet (tagging + scoring)")
    p_full.add_argument("--engagement-id", required=True, type=int, help="ID de l'engagement")
    p_full.add_argument("--dry-run", action="store_true", help="Mode simulation")

    p_serve = subparsers.add_parser("serve", help="Lance l'API FastAPI")
    p_serve.add_argument("--host", default="0.0.0.0", help="Hôte (défaut: 0.0.0.0)")
    p_serve.add_argument("--port", default=8000, type=int, help="Port (défaut: 8000)")
    p_serve.add_argument("--reload", action="store_true", help="Hot-reload (développement)")

    subparsers.add_parser("status", help="Affiche l'état complet du projet")

    return parser



def main() -> None:
    _check_python()
    parser = build_parser()
    args = parser.parse_args()

    _init_directories()

    if args.command not in ["status", "serve"] and not _check_dependencies():
        sys.exit(1)

    if args.command == "run":
        success = cmd_run(
            skip_fetch=getattr(args, 'skip_fetch', False),
            engagement_id=getattr(args, 'engagement_id', None) 
        )
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

    elif args.command == "tag":
        success = cmd_tag(args.engagement_id, getattr(args, 'dry_run', False), getattr(args, 'all', False))
        sys.exit(0 if success else 1)

    elif args.command == "predict":
        success = cmd_predict(args.engagement_id)
        sys.exit(0 if success else 1)

    elif args.command == "full":
        success = cmd_full(args.engagement_id, getattr(args, 'dry_run', False))
        sys.exit(0 if success else 1)

    elif args.command == "serve":
        cmd_serve(host=args.host, port=args.port, reload=args.reload)

    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()