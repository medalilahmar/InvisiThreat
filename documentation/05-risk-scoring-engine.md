# Moteur de scoring IA — Risk Scoring Engine

Ce module est le cœur intelligent d'InvisiThreat. Il transforme les vulnérabilités remontées par DefectDojo en un score de risque contextualisé (0–10) grâce à un pipeline de machine learning, puis expose ces scores pour le dashboard, Jira et le pipeline CI/CD.

---

## Table des matières

- [1. Vue d'ensemble](#1-vue-densemble)
- [2. Architecture logique](#2-architecture-logique)
- [3. Structure des dossiers](#3-structure-des-dossiers)
- [4. Pipeline de données](#4-pipeline-de-données)
- [5. Cycle de vie du modèle](#5-cycle-de-vie-du-modèle)
- [6. Inférence en production](#6-inférence-en-production-predict_livepy)
- [7. Anti-leakage et robustesse](#7-anti-leakage-et-robustesse)
- [8. Intégration dans l'écosystème](#8-intégration-dans-lécosystème)

---

## 1. Vue d'ensemble

Le Risk Scoring Engine assure plusieurs fonctions :

- Récupération et enrichissement des vulnérabilités (findings) depuis DefectDojo.
- Nettoyage, feature engineering et calcul d'un score IA avec un modèle supervisé.
- Inférence en temps réel via un service dédié ou en batch.
- Enrichissement automatique des tags dans DefectDojo (tagging intelligent).
- Intégrations avancées : Jira, Gemini (LLM), cache EPSS.

L'ensemble est conteneurisé et exécutable aussi bien en local que dans le pipeline CI/CD.

---

## 2. Architecture logique

```
[DefectDojo API]
       ↓
[fetch_data.py]      → récupère les findings actifs
       ↓
[preprocess.py]      → nettoyage + extraction des features
       ↓
[model_manager.py]   → chargement du modèle (pipeline_latest.pkl)
       ↓
[predict_live.py]    → scoring batch ou temps réel
       ↓
[api.py / api_simple.py]  → endpoints REST (FastAPI)
       ↓
[tag_findings.py]    → réécriture des tags dans DefectDojo
       ↓
[jira_service.py]    → création de tickets Jira
       ↓
[llm_service.py]     → analyse sémantique via Gemini
```

**Composants externes :** DefectDojo (source et cible), Jira Cloud, Gemini API, PostgreSQL (utilisateurs), Redis (cache).

---

## 3. Structure des dossiers

```
risk-scoring-engine/
├── src/
│   ├── fetch_data.py       # Collecte des données DefectDojo
│   ├── preprocess.py       # Nettoyage et feature engineering
│   ├── train.py            # Entraînement du modèle ML
│   ├── predict_live.py     # Inférence temps réel / batch
│   ├── tag_findings.py     # Tagging intelligent des vulnérabilités
│   ├── api_simple.py       # Serveur API FastAPI
│   ├── model_manager.py    # Gestion du modèle (chargement, cache)
│   └── server/             # Architecture modulaire de l'API
├── data/
│   ├── raw/                # Données brutes (CSV de DefectDojo)
│   ├── processed/          # Données nettoyées (findings_clean.csv)
│   └── epss_cache.json     # Cache des scores EPSS
├── models/
│   ├── pipeline_latest.pkl
│   └── pipeline_latest_meta.json
└── reports/                # Rapports d'entraînement et SHAP
```

---

## 4. Pipeline de données

### 4.1 Récupération — `fetch_data.py`

- Se connecte à l'API DefectDojo et pagine l'ensemble des produits, engagements, tests et findings.
- Enrichit chaque finding avec le nom du produit et de l'engagement.
- Extrait les identifiants CVE de manière robuste (JSON, liste, texte).
- Interroge l'API FIRST.org pour obtenir les scores EPSS, avec cache local.
- Sauvegarde les données brutes dans `data/raw/` et un rapport de collecte.

### 4.2 Nettoyage et features — `preprocess.py`

Ce script transforme les données brutes en un dataset prêt pour l'entraînement.

**Nettoyage :** gestion des valeurs manquantes, normalisation des types, exclusion des faux positifs et hors-scope.

**Ingénierie de features (> 20 caractéristiques) :**

| Catégorie | Features |
|-----------|----------|
| Temporelles | `age_days`, `days_to_fix`, `delay_norm` |
| CVSS | `cvss_score`, `cvss_score_norm` |
| EPSS | `epss_score`, `epss_percentile`, `has_high_epss`, `epss_x_cvss` |
| Tags | `tag_urgent`, `tag_in_production`, `tag_sensitive`, `tag_external`, `tags_count` |
| Contexte | `context_score`, `exposure_norm`, `product_fp_rate` |
| Interactions | `cvss_x_has_cve`, `age_x_cvss`, `epss_x_cvss` |

**Anti-leakage :** toute colonne dérivée du label (`severity_num`, interactions avec la sévérité) est exclue des features d'entraînement.

**Label :** la sévérité DefectDojo est fusionnée en 4 classes (`Low`, `Medium`, `High`, `Critical`) stockée dans `risk_class`.

**Équilibrage :** des `sample_weight` sont calculés pour compenser le déséquilibre des classes.

Sortie : `findings_clean.csv` + rapport JSON de qualité.

### 4.3 Entraînement — `train.py`

- Recherche d'hyperparamètres par `RandomizedSearchCV` pour RandomForest, XGBoost et LightGBM.
- Construction d'un **StackingClassifier** (méta-classifieur : régression logistique calibrée).
- Calibration des probabilités (isotonic/sigmoid) pour les modèles non stackés.
- Évaluation : F1, balanced accuracy, recall Critical, ROC-AUC, matrice de confusion.
- Explicabilité : calcul des valeurs SHAP (classe High) et génération de graphiques.
- Persistance : sauvegarde de `pipeline_latest.pkl` et de ses métadonnées, historique des runs.

### 4.4 Tagging intelligent — `tag_findings.py`

- Exécuté après l'import des scans dans DefectDojo.
- Analyse lexicographique des findings (titre, description, chemin, composant).
- Ajoute automatiquement des tags : `urgent`, `blocker`, `production`, `external`, `sensitive`, `pii`, `api`, `sca`.
- Ces tags enrichissent les features utilisées par le modèle (ex. `tag_external` augmente le score de risque).

---

## 5. Cycle de vie du modèle

| Étape | Description |
|-------|-------------|
| **Entraînement** | `train.py` lit `findings_clean.csv`, entraîne et sauvegarde le modèle. |
| **Validation** | Métriques et rapports (matrice de confusion, SHAP) générés dans `reports/`. |
| **Promotion** | Le meilleur modèle est copié en `pipeline_latest.pkl`. |
| **Inférence** | `model_manager.py` charge ce modèle en mémoire (singleton). |
| **Monitoring** | `monitor_model.py` surveille la dérive des prédictions. |

---

## 6. Inférence en production (`predict_live.py`)

Deux modes disponibles :

**Mode CSV** — lit `findings_clean.csv`, applique le modèle et sauvegarde les scores.

**Mode API DefectDojo** — récupère les findings d'un engagement, les prédit par lots, met à jour les tags `ai-risk-*` et ajoute une note détaillée (score, confiance, SHAP).

Le score final `ai_risk_score` combine le score du modèle et un **nudge métier** prenant en compte CVSS, EPSS et exposition.

Un **Security Gate** bloque le pipeline CI si un score dépasse un seuil (par défaut `7.0`).

---

## 7. Anti-leakage et robustesse

- Les colonnes interdites (ex. `severity_num`, `cvss_x_severity`) sont supprimées avant l'inférence, garantissant que le modèle ne triche pas.
- Les features sont toujours réalignées dans l'ordre exact de l'entraînement.
- Les valeurs manquantes sont imputées de manière cohérente avec le preprocessing.

---

## 8. Intégration dans l'écosystème

| Composant | Rôle |
|-----------|------|
| `fetch_data.py` | Alimente le dataset d'entraînement et le cache EPSS. |
| `preprocess.py` | Génère les données propres pour `train.py`. |
| `train.py` | Produit le modèle et les métadonnées utilisés par l'API et `predict_live.py`. |
| `tag_findings.py` | Enrichit les tags, qui deviennent des features du modèle. |
| API backend | Expose les endpoints `/predict` et `/findings` pour le dashboard et les intégrations. |
| Pipeline CI/CD | Appelle `predict_live.py` en mode `--ci-mode` pour jouer le Security Gate. |