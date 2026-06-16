# 🔄 Pipeline CI/CD DevSecOps — InvisiThreat

> Pipeline GitHub Actions multi-projets assurant l'analyse de sécurité continue, la construction Docker, le déploiement Kubernetes (k3s) et la mise à jour du modèle IA.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Déclencheurs (Triggers)](#2-déclencheurs-triggers)
3. [Variables d'environnement globales](#3-variables-denvironnement-globales)
4. [Structure et flux d'exécution](#4-structure-et-flux-dexécution)
5. [Job 1 — Détection des projets modifiés (`detect`)](#5-job-1--détection-des-projets-modifiés-detect)
6. [Job 2 — Analyse de sécurité (`security`)](#6-job-2--analyse-de-sécurité-security)
7. [Job 3 — Construction et push Docker (`build-and-push`)](#7-job-3--construction-et-push-docker-build-and-push)
8. [Job 4 — Déploiement Kubernetes (`deploy`)](#8-job-4--déploiement-kubernetes-deploy)
9. [Actions composites réutilisables](#9-actions-composites-réutilisables)
   - [load-config](#91-load-config--chargement-de-la-configuration-projet)
   - [dd-setup](#92-dd-setup--initialisation-defectdojo)
   - [dd-import](#93-dd-import--import-générique-defectdojo)
   - [dd-import-zap](#94-dd-import-zap--import-zap-defectdojo)
   - [zap-scan](#95-zap-scan--analyse-dast)
   - [deploy](#96-deploy--déploiement-kubernetes-k3s)
10. [Résumé final du pipeline](#10-résumé-final-du-pipeline)

---

## 1. Vue d'ensemble

Le pipeline DevSecOps d'InvisiThreat est un workflow **GitHub Actions** conçu pour gérer plusieurs projets applicatifs en parallèle. Il détecte automatiquement les projets dont le code a été modifié, exécute les analyses de sécurité appropriées, construit et déploie les services sur un cluster Kubernetes léger (k3s), et maintient à jour le modèle d'intelligence artificielle de scoring de risque.

```
detect → security → build-and-push → deploy
```

Chaque étape est conditionnelle : un échec à la porte de sécurité bloque le déploiement.

---

## 2. Déclencheurs (Triggers)

| Événement | Branches | Filtrage | Particularité |
|-----------|----------|----------|---------------|
| **Push** | `main`, `develop` | Modifications dans `projects/**`, `risk-scoring-engine/**`, `security-dashboard/**` | Exécution automatique |
| **Pull Request** | `main`, `develop` | Modifications dans `projects/**` | Exécution à l'ouverture/mise à jour de la PR |
| **Workflow dispatch** | — | Projet choisi manuellement parmi une liste | Déclenchement manuel avec paramètres personnalisés |

### Paramètres du déclenchement manuel

Lors d'un `workflow_dispatch`, l'utilisateur peut configurer :

- **Projet cible** : nom d'un projet spécifique, ou vide pour l'auto-détection
- **Déployer** : déployer les conteneurs après les scans (défaut : `true`)
- **Forcer le réentraînement ML** : forcer la mise à jour du modèle de scoring IA (défaut : `false`)

---

## 3. Variables d'environnement globales

Accessibles dans tous les jobs du workflow :

```yaml
env:
  DD_URL:       http://localhost:8080   # URL de l'instance DefectDojo
  PROJECTS_DIR: projects                # Répertoire racine des projets
  APP_NAME:     InvisiThreat            # Nom de l'application
  NAMESPACE:    invisithreat            # Namespace Kubernetes cible
  REGISTRY:     localhost:5000          # Registre Docker local
```

---

## 4. Structure et flux d'exécution

```
┌─────────┐     ┌──────────┐     ┌────────────────┐     ┌────────┐
│ detect  │────▶│ security │────▶│ build-and-push │────▶│ deploy │
└─────────┘     └──────────┘     └────────────────┘     └────────┘
     │               │
     │    (matrice de projets, en parallèle)
     │               │
     │         ┌─────┴──────┐
     │         │  projet A  │
     │         │  projet B  │
     │         │  projet N  │
     │         └────────────┘
     │
     └── Sorties : matrix JSON, has_projects, build_infra
```

**Règles de conditionnalité :**

- `security` s'exécute uniquement si `has_projects == true`
- `build-and-push` s'exécute si `security` réussit, le déploiement est activé, et on n'est pas en PR
- `deploy` s'exécute si `build-and-push` réussit, hors PR, déploiement activé

---

## 5. Job 1 — Détection des projets modifiés (`detect`)

**Rôle** : Construire la matrice JSON des projets impactés par les modifications du commit.

### Outputs

| Nom | Description |
|-----|-------------|
| `matrix` | Liste JSON des projets modifiés (ex : `["juice-shop","nova"]`) |
| `has_projects` | `true` si au moins un projet a changé |
| `build_infra` | `false` dans cette version (réservé pour usage futur) |

### Logique de détection

```
workflow_dispatch avec projet explicite
        │
        ▼ Oui ──▶ Utiliser uniquement ce projet
        │
        ▼ Non
        │
  Analyser git diff
        │
  Pour chaque fichier modifié sous projects/<nom>/
        │
  Vérifier l'existence de project.config.yml
        │
  Ajouter le projet (dédoublonnage)
        │
  Sérialiser en JSON → outputs
```

---

## 6. Job 2 — Analyse de sécurité (`security`)

- **Condition** : `has_projects == true`
- **Stratégie** : Matrice avec `fail-fast: false` — les projets s'analysent en parallèle, un échec n'annule pas les autres
- **Timeout** : 120 minutes par projet

### Étapes d'exécution par projet

#### 6.1 Checkout et chargement de la configuration

Récupération du code source, puis lecture du fichier `projects/<nom>/project.config.yml` via l'action composite `load-config` qui expose toutes les métadonnées du projet (type, commandes, seuils de sécurité, etc.).

#### 6.2 Préparation de DefectDojo

Un playbook Ansible s'assure que l'instance DefectDojo locale est opérationnelle avant tout import.

#### 6.3 Installation des dépendances (avec cache)

Installation conditionnelle selon le type de projet, avec un cache manuel basé sur le hash des fichiers de configuration :

| Type | Fichiers de cache | Comportement |
|------|------------------|--------------|
| `nodejs` | `package.json`, `package-lock.json` | `npm install` avec cache sha256 |
| `python` | `requirements.txt`, `pyproject.toml`, `setup.cfg/py` | Virtualenv + ajout au site-packages |
| `java` | `pom.xml` ou `build.gradle` | Cache `~/.m2` Maven |
| `static` | — | Aucune installation nécessaire |

#### 6.4 Configuration de l'engagement DefectDojo

L'action `dd-setup` crée ou met à jour un **produit** et un **engagement** dans DefectDojo, lié à la branche et au SHA actuel. Elle retourne l'`engagement_id`, le nom, le `product_id`, le tag d'environnement (`prod`, `staging`, `dev`) et les dates de la période trimestrielle.

#### 6.5 Snapshot des findings avant scan

Appel API DefectDojo pour récupérer le nombre de findings actifs (total, Critical, High) **avant** l'import des nouveaux résultats. Ces valeurs servent à calculer les nouveaux findings introduits par le commit.

#### 6.6 SAST — Analyse statique avec Semgrep

- Récupération des rulesets depuis la config projet (champ `semgrep_configs`)
- Fallback automatique selon le type de projet si aucune règle n'est définie :
  - `nodejs` → `p/security-audit`, `p/owasp-top-ten`, `p/nodejs`
  - `python` → `p/python`, `p/security-audit`
  - etc.
- Exécution via l'image Docker officielle `returntocorp/semgrep`
- Sortie : `semgrep-results.json`

#### 6.7 SCA — Analyse de dépendances avec Snyk

```
snyk test  ──▶  Analyse les dépendances selon le fichier détecté
               (package-lock.json / requirements.txt / pom.xml)
               Seuil de sévérité : défini par snyk_threshold

snyk monitor ──▶  Publie un snapshot sur le tableau de bord Snyk
                  pour le suivi dans le temps
```

Sortie : `snyk-results.json`

#### 6.8 DAST — Analyse dynamique avec OWASP ZAP

1. **Démarrage de l'application** : `start_command` lancé en arrière-plan, boucle d'attente sur `target_url + health_path`
2. **Scan OpenAPI** : si `zap_openapi_endpoint` est défini dans la config
3. **Full Scan** : AJAX Spider + Active Scan, avec authentification JWT si configurée
4. **Arrêt de l'application** : toujours exécuté, même en cas d'erreur

Sorties : `zap-openapi.xml`, `zap-fullscan.xml`

#### 6.9 Upload des artefacts

Tous les rapports de sécurité (Semgrep JSON, Snyk JSON, ZAP XML) sont sauvegardés comme **artefacts GitHub** pour consultation ultérieure.

#### 6.10 Import dans DefectDojo

Chaque rapport est importé dans l'engagement DefectDojo avec des tags automatiques (`sast`, `sca`, `openapi`, `fullscan`, branche, nom du projet, environnement).

#### 6.11 Tagging intelligent

Le script `risk-scoring-engine/src/tag_findings.py` applique des tags supplémentaires aux findings (ex : `false-positive-probable`, `IA-critique`) en utilisant le moteur de scoring IA.

#### 6.12 Security Gate (porte de sécurité)

```
Findings après import
        │
        ▼
Calcul des nouveaux findings
  new_critical = après_critical - avant_critical
  new_high     = après_high     - avant_high
        │
        ▼
Vérification des seuils (project.config.yml)
  fail_on_new_critical: true/false
  fail_on_new_high:     true/false
        │
  Seuil dépassé ──▶  exit 1 → pipeline stoppé
        │
  OK    ──▶  gate_passed = true → suite du pipeline
```

### Outputs du job `security`

| Nom | Description |
|-----|-------------|
| `engagement_id` | ID de l'engagement DefectDojo |
| `engagement_name` | Nom de l'engagement |
| `product_id` | ID du produit DefectDojo |
| `env_tag` | Tag d'environnement (`prod`, `staging`, `dev`) |
| `new_critical` | Nombre de nouveaux findings Critical |
| `new_high` | Nombre de nouveaux findings High |
| `gate_passed` | `true` si les seuils ne sont pas dépassés |

---

## 7. Job 3 — Construction et push Docker (`build-and-push`)

**Conditions d'exécution** :
- `security` n'a pas échoué
- `deploy != 'false'`
- Pas de Pull Request
- Des projets ont été détectés

### Étapes

1. Vérification des Dockerfiles dans `risk-scoring-engine/` et `security-dashboard/`
2. Démarrage du registre local (conteneur `registry:2` sur le port 5000, si absent)
3. **Build & Push du Risk Scoring Engine** :
   - Cache Docker classique (`DOCKER_BUILDKIT=0`)
   - Tags : `latest` et SHA du commit
4. **Build & Push du Dashboard** :
   - Build en deux étapes pour optimiser le cache : le stage `builder` est poussé séparément, puis le stage final réutilise ce cache
   - Tags identiques

### Outputs

| Nom | Description |
|-----|-------------|
| `image_tag` | SHA du commit utilisé comme tag d'image |

---

## 8. Job 4 — Déploiement Kubernetes (`deploy`)

**Conditions** : `build-and-push` réussi, hors PR, déploiement activé.

### Étapes

1. Affichage des paramètres (namespace, SHA, `force_retrain`)
2. Appel de l'action composite `deploy` qui :
   - Met à jour les déploiements Kubernetes avec les nouvelles images
   - Applique tous les secrets (PostgreSQL, JWT, API tokens, Gemini, Jira, SMTP…)
   - Déclenche la mise à jour du modèle ML si `run_ml_update == true`
3. Vérifications post-déploiement :
   - Liste des pods et services dans le namespace
   - Derniers événements Kubernetes
   - Dernières lignes de logs du Risk Scoring Engine et du Dashboard
4. Test de connectivité API via `curl` sur `/health` et `/docs`

---

## 9. Actions composites réutilisables

Les actions composites locales (dans `.github/actions/`) factorisent la logique répétitive du pipeline.

### 9.1 `load-config` — Chargement de la configuration projet

**Fichier** : `.github/actions/load-config/action.yml`

Lit le fichier `projects/<nom>/project.config.yml` et expose toutes ses valeurs comme outputs. Permet à chaque projet d'avoir sa propre configuration de sécurité sans duplication de code.

#### Inputs

| Nom | Description | Requis | Défaut |
|-----|-------------|--------|--------|
| `project` | Nom du projet (ex. `juice-shop`) | Oui | — |
| `projects_dir` | Dossier racine des projets | Non | `projects` |

#### Outputs (19 au total)

**Identité** : `project_name`, `project_description`

**Type et chemins** : `project_type` (`nodejs`, `python`, `java`, `static`), `app_dir`, `install_dir`

**Exécution** : `start_command`, `health_path`, `startup_wait`, `target_url`, `install_command`

**SAST** : `semgrep_configs` (liste CSV des rulesets)

**SCA** : `snyk_threshold` (`low`, `medium`, `high`)

**DAST** : `zap_timeout`, `zap_jwt_login_endpoint`, `zap_jwt_login_body`, `zap_jwt_token_path`, `zap_openapi_endpoint`

**Porte de sécurité** : `fail_on_new_critical`, `fail_on_new_high`

#### Fonctionnement interne

1. Vérification de l'existence du fichier `project.config.yml`
2. Installation de `yq` v4.44.1 si absent
3. Parsing YAML avec une fonction `yq_get` qui gère les valeurs par défaut et les fallbacks automatiques :

| Paramètre | Fallback si absent |
|-----------|-------------------|
| `start_command` (nodejs) | `npm start` |
| `start_command` (python) | `python app.py` |
| `install_command` (nodejs) | `npm install --no-audit --no-fund --legacy-peer-deps` |
| `install_command` (python) | `pip install -r requirements.txt` |
| `health_path` | `/` |
| `startup_wait` | `15` secondes |
| `snyk_threshold` | `low` |
| `zap_timeout` | `30` minutes |
| `semgrep_configs` | `p/security-audit` |

4. Validation : `project_type` obligatoire et dans la liste autorisée, `app_port` entier
5. Export vers `$GITHUB_OUTPUT`

#### Exemple de `project.config.yml`

```yaml
project:
  name: "Juice Shop"
  description: "Application e-commerce vulnérable pour tests"
  type: nodejs

app:
  port: 3000
  start_command: "npm start"
  health_check_path: "/health"
  startup_wait: 20
  install_dir: "frontend"           # optionnel
  install_command: "npm ci --production"  # optionnel

security:
  fail_on_new_critical: true
  fail_on_new_high: false
  snyk_severity_threshold: medium
  zap_full_scan_timeout: 60
  semgrep_rulesets:
    - p/security-audit
    - p/owasp-top-ten
    - p/nodejs
  zap_jwt_login_endpoint: "/api/login"
  zap_jwt_login_body: '{"email":"admin@example.com","password":"admin123"}'
  zap_jwt_token_path: ".token"
  zap_openapi_endpoint: "/api-docs"
```

---

### 9.2 `dd-setup` — Initialisation DefectDojo

**Fichier** : `.github/actions/dd-setup/action.yml`

Crée ou met à jour le produit et l'engagement dans DefectDojo avant chaque série de scans. Exécutée une fois par projet et par branche.

#### Inputs

| Nom | Description | Requis |
|-----|-------------|--------|
| `dd_url` | URL DefectDojo | Oui |
| `dd_api_key` | Token API DefectDojo | Oui |
| `app_name` | Nom du produit | Oui |
| `branch` | Branche Git | Oui |
| `run_id` | ID du run GitHub Actions | Oui |
| `sha` | SHA du commit | Oui |
| `repo_url` | URL du dépôt | Oui |

#### Outputs

| Nom | Description |
|-----|-------------|
| `product_id` | ID du produit DefectDojo |
| `engagement_id` | ID de l'engagement |
| `engagement_name` | Nom de l'engagement (ex : `main - 2026-Q2`) |
| `start_date` / `end_date` | Dates de la période trimestrielle |
| `period` | Trimestre (`Q1`, `Q2`, `Q3`, `Q4`) |
| `env_tag` | Tag d'environnement (`prod`, `staging`, `dev`) |

#### Logique clé

**Tag d'environnement automatique :**

```
main / master  →  prod
staging / preprod  →  staging
toute autre branche  →  dev
```

**Engagements trimestriels :**

Les scans sont regroupés par trimestre calendaire pour conserver l'historique sans accumulation infinie :

```
Janvier–Mars   →  Q1
Avril–Juin     →  Q2
Juillet–Sept.  →  Q3
Octobre–Déc.   →  Q4

Nom de l'engagement : "<branche> - <année>-<trimestre>"
Exemple : "main - 2026-Q2"
```

Si l'engagement existe déjà, il est mis à jour (`PATCH`). Sinon, il est créé (`POST`) avec le lien vers le dépôt source.

---

### 9.3 `dd-import` — Import générique DefectDojo

**Fichier** : `.github/actions/dd-import/action.yml`

Importe un rapport de scan (SAST, SCA, DAST) dans DefectDojo via l'endpoint `reimport-scan`.

#### Inputs principaux

| Nom | Description | Requis |
|-----|-------------|--------|
| `dd_url` / `dd_api_key` | Connexion DefectDojo | Oui |
| `engagement_id` | ID de l'engagement cible | Oui |
| `scan_type` | Type de parser (ex : `Semgrep JSON Report`) | Oui |
| `file` | Chemin du rapport à importer | Oui |
| `test_title` | Titre unique du Test DefectDojo | Oui |
| `tags` | Tags séparés par virgules | Non |

#### Outputs

| Nom | Description |
|-----|-------------|
| `test_id` | ID du Test créé/mis à jour (`skipped` si fichier vide) |
| `new_findings` | Nombre de nouveaux findings importés |

#### Comportement clé

- Si le fichier est vide ou absent → mode **skip** sans erreur
- Utilise `reimport-scan` (pas `import-scan`) pour **dédupliquer** les findings entre exécutions
- Les findings résolus (absents du nouveau rapport) sont **automatiquement fermés**
- Les tags sont passés comme champs `-F tags=...` séparés (un par valeur)

#### Paramètres fixes envoyés à l'API

| Paramètre | Valeur |
|-----------|--------|
| `auto_create_context` | `true` |
| `close_old_findings` | `true` |
| `minimum_severity` | `Info` |
| `active` | `true` |
| `verified` | `false` |
| `environment` | `Development` |

---

### 9.4 `dd-import-zap` — Import ZAP DefectDojo

**Fichier** : `.github/actions/dd-import-zap/action.yml`

Variante spécialisée de `dd-import` pour les rapports XML OWASP ZAP.

#### Différences par rapport à `dd-import`

| Aspect | `dd-import` | `dd-import-zap` |
|--------|-------------|-----------------|
| Format | JSON (Semgrep, Snyk) | XML (ZAP) |
| `scan_type` | Variable | `ZAP Scan` (fixe) |
| Tags par défaut | `""` | `dast,zap,external,internet-facing` |
| Détection placeholder | Non | Oui (`<OWASPZAPReport/>`) |

Le `scan_type` est codé en dur car cette action est exclusivement dédiée aux rapports ZAP XML.

Les tags par défaut `external` et `internet-facing` indiquent l'origine d'un scan dynamique externe, ce qui influence le poids du scoring IA.

---

### 9.5 `zap-scan` — Analyse DAST

**Fichier** : `.github/actions/zap-scan/action.yml`

Exécute les analyses dynamiques OWASP ZAP avec adaptation automatique au projet.

#### Inputs

| Nom | Description | Requis | Défaut |
|-----|-------------|--------|--------|
| `target_url` | URL de l'application cible | Oui | — |
| `app_dir` | Répertoire de travail | Oui | — |
| `jwt_login_endpoint` | Endpoint POST de login JWT | Non | `""` |
| `jwt_login_body` | Corps JSON du login | Non | `""` |
| `jwt_token_path` | Chemin `jq` pour extraire le token | Non | `.token` |
| `openapi_endpoint` | Endpoint OpenAPI (vide = désactivé) | Non | `""` |
| `full_scan_timeout` | Timeout du scan complet (minutes) | Non | `30` |

#### Outputs

| Nom | Description |
|-----|-------------|
| `fullscan_report_xml` | Chemin absolu du rapport XML Full Scan |
| `openapi_report_xml` | Chemin absolu du rapport XML OpenAPI |
| `total_alerts` | Nombre total d'alertes uniques |
| `high_alerts` | Nombre d'alertes High/Critical |
| `medium_alerts` | Nombre d'alertes Medium |
| `detected_tags` | Tags contextuels détectés automatiquement |

#### Séquence d'exécution

```
Étape 1 : Healthcheck de la cible (5 tentatives, 3s d'intervalle)
    │
Étape 2 : Obtention du JWT (si jwt_login_endpoint configuré)
    │
Étape 3 : Préparation du Swagger
          ├── Recherche locale (swagger.yml, openapi.yaml…)
          └── Téléchargement depuis l'app si absent
              Patch des URLs relatives → absolues
    │
Étape 4 : Création de la fonction zap_run (factorisation Docker)
    │
Étape 5 : ZAP OpenAPI Scan (si openapi_endpoint défini)
          zap-api-scan.py -t <url> -f openapi -T 15
    │
Étape 6 : ZAP Full Scan
          zap-full-scan.py -t <url> -T <timeout>
          (AJAX Spider + Active Scan)
    │
Étape 7 : Résolution des chemins absolus des rapports
    │
Étape 8 : Consolidation et tagging contextuel
```

#### Détection automatique de tags contextuels

Les alertes ZAP sont analysées pour détecter des motifs et attribuer des tags enrichissant le scoring :

| Motif dans les alertes | Tag ajouté |
|------------------------|-----------|
| `password`, `credential`, `token`, `pii`, `private key`… | `sensitive`, `pii` |
| `sql injection`, `xss`, `command injection`, `rce`… | `blocker` |
| `cors`, `access-control`, `csp`, `clickjack`… | `exposed` |
| `information disclosure`, `directory browsing`, `debug`… | `internet-facing` |

Ces tags, préfixés de `dast,zap`, sont injectés dans DefectDojo et exploités par `preprocess.py` pour pondérer le `context_score` des findings dans le modèle IA.

---

### 9.6 `deploy` — Déploiement Kubernetes (k3s)

**Fichier** : `.github/actions/deploy/action.yml`

Orchestre le déploiement complet de la plateforme sur le cluster k3s en trois phases.

#### Inputs principaux

| Nom | Description | Requis |
|-----|-------------|--------|
| `namespace` | Namespace Kubernetes | Non (`invisithreat`) |
| `sha` | SHA du commit (tag image Docker) | Oui |
| `defectdojo_api_key` | Clé API DefectDojo | Oui |
| `postgres_password` | Mot de passe PostgreSQL | Oui |
| `jwt_secret_key` | Clé JWT | Oui |
| `gemini_api_key` | Clé API Gemini | Oui |
| `jira_api_token` / `jira_email` / `jira_server` | Connexion Jira | Oui |
| `smtp_password` | Mot de passe SMTP | Oui |
| `run_ml_update` | Exécuter la mise à jour ML ? | Non (`"true"`) |

#### Outputs

| Nom | Description |
|-----|-------------|
| `node_ip` | IP interne du nœud k3s |
| `dashboard_url` | URL du dashboard React |
| `api_url` | URL de l'API FastAPI |

#### Phase 1 — Installation k3s (idempotente via Ansible)

| Playbook | Rôle |
|----------|------|
| `playbook-k3s-install.yml` | Installation du binaire k3s, démarrage du service |
| *(config kubectl)* | Lien symbolique `/usr/local/bin/kubectl`, copie de `k3s.yaml` |
| `playbook-metallb-install.yml` | Load-balancer pour services `LoadBalancer` |
| `playbook-ingress-install.yml` | NGINX Ingress Controller |
| `playbook-monitoring-install.yml` | Prometheus + Grafana dans le namespace `monitoring` |

#### Phase 2 — Déploiement Kubernetes (ordre strict)

```
1. Namespace          → infrastructure/kubernetes/namespace.yaml
2. Secrets            → kubectl create secret (--from-literal)
3. ConfigMap          → infrastructure/kubernetes/configmap.yaml
4. Service DefectDojo → ExternalName pointant vers l'IP du nœud:8080
5. PersistentVolumes  → infrastructure/kubernetes/storage/
6. PostgreSQL         → déploiement + rollout status (300s)
7. Redis              → déploiement + rollout status (300s)
8. Risk Scoring Engine + Dashboard → déploiement + kubectl set image + rollout (240s)
```

> **Service ExternalName DefectDojo** : DefectDojo tourne sur l'hôte local (`localhost:8080`), inaccessible depuis les pods. L'action crée un `Service` de type `ExternalName` et un `Endpoints` qui mappe l'IP du nœud sur le port 8080, permettant aux pods d'appeler `defectdojo-external` en DNS interne.

#### Phase 3 — Mise à jour du modèle ML (`run_ml_update == true`)

```
1. pip install -r requirements.txt
        │
2. fetch_data.py     → data/raw/findings_raw.csv
        │
3. preprocess.py     → data/processed/findings_clean.csv
        │
4. Copie atomique dans le pod :
   fichier.pkl  →  fichier.pkl.tmp  →  mv  →  fichier.pkl
        │
5. kubectl rollout restart deployment/risk-scoring-engine
   + boucle d'attente sur /ready
        │
6. create_admin.py  (idempotent, crée l'admin si absent)
        │
7. predict_live.py  → logs/predictions_latest.json
        │
8. Copie de data/ai_scores_cache.json dans le pod
        │
9. Vérification /api/ready et /api/model/info
```

> **Copie atomique** : les fichiers sont copiés sous le suffixe `.tmp` avant d'être renommés en une seule commande `mv`. Cela garantit **zéro downtime** lors de la mise à jour du modèle.

---

## 10. Résumé final du pipeline

À la fin du job `deploy`, un bloc récapitulatif complet est affiché dans les logs :

```
DEVSECOPS PIPELINE COMPLET — InvisiThreat

  ── SÉCURITÉ ──────────────────────────────────────────
  Engagement : ... (ID=...)
  Produit    : (ID=...)
  Période    : ... → ...
  Env        : prod / staging / dev
  Critical   : X nouveau(x)
  High       : Y nouveau(x)
  Gate       : true / false
  Lien DD    : http://localhost:8080/engagements/.../finding/

  ── BUILD ─────────────────────────────────────────────
  Image tag  : <sha>
  Registre   : localhost:5000

  ── DÉPLOIEMENT ───────────────────────────────────────
  Namespace  : invisithreat
  SHA        : ...
  Branche    : main
  ML Retrain : true / false

  ── URLs ──────────────────────────────────────────────
  Dashboard  : http://invisithreat.local
  API FastAPI: http://invisithreat.local/docs
  DefectDojo : http://localhost:8080

  ── ÉTAT KUBERNETES ───────────────────────────────────
  <liste des déploiements et pods>
```

Ce résumé permet aux développeurs et aux équipes sécurité de connaître immédiatement l'état de la dernière intégration.

---

> **Fichier suivant** : [05-risk-scoring-engine.md](./05-risk-scoring-engine.md) — Moteur de scoring IA