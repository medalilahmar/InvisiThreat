# Architecture — InvisiThreat

> Vue d'ensemble du système, flux de données, composants et choix techniques de la plateforme DevSecOps InvisiThreat.

---

## Table des matières

1. [Présentation générale](#1-présentation-générale)
2. [Schéma d'architecture global](#2-schéma-darchitecture-global)
3. [Composants du système](#3-composants-du-système)
4. [Flux de données détaillé](#4-flux-de-données-détaillé)
5. [Couche d'analyse de sécurité](#5-couche-danalyse-de-sécurité)
6. [Couche de centralisation — DefectDojo](#6-couche-de-centralisation--defectdojo)
7. [Couche d'intelligence artificielle — Risk Scoring Engine](#7-couche-dintelligence-artificielle--risk-scoring-engine)
8. [Couche applicative — API Backend et Dashboard](#8-couche-applicative--api-backend-et-dashboard)
9. [Couche infrastructure](#9-couche-infrastructure)
10. [Couche monitoring](#10-couche-monitoring)
11. [Interactions entre composants](#11-interactions-entre-composants)
12. [Choix techniques et justifications](#12-choix-techniques-et-justifications)
13. [Sécurité de l'architecture](#13-sécurité-de-larchitecture)

---

## 1. Présentation générale

InvisiThreat est une plateforme DevSecOps qui automatise l'intégralité du cycle de détection, d'analyse et de priorisation des vulnérabilités applicatives. Elle s'articule autour d'un pipeline CI/CD GitHub Actions qui orchestre plusieurs outils de sécurité, centralise leurs résultats dans DefectDojo, les enrichit avec un moteur de scoring basé sur le machine learning, et les expose via une API REST et un tableau de bord interactif.

Le projet cible l'application volontairement vulnérable **OWASP Juice Shop** comme sujet d'analyse, ce qui garantit un flux constant de vulnérabilités réelles pour alimenter et valider le moteur IA.

### Objectifs architecturaux

| Objectif | Traduction technique |
|----------|---------------------|
| Automatisation complète | Pipeline CI/CD sans intervention manuelle |
| Reproductibilité | Infrastructure as Code via Ansible |
| Priorisation intelligente | Modèle ML supervisé avec stacking |
| Traçabilité totale | Lien SHA commit → image → déploiement → finding |
| Observabilité | Stack Prometheus, Grafana, Loki |
| Sécurité by design | JWT, secrets Kubernetes, Security Gate |

---

## 2. Schéma d'architecture global

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dépôt GitHub                             │
│          (Code source OWASP Juice Shop + InvisiThreat)          │
└────────────────────────┬────────────────────────────────────────┘
                         │ push / PR / dispatch
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GitHub Actions (CI/CD)                        │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌────────┐ │
│  │  detect  │──▶│ security │──▶│build-and-push │──▶│ deploy │ │
│  └──────────┘   └────┬─────┘   └───────────────┘   └────────┘ │
│                      │                                          │
│            ┌─────────┼─────────┐                               │
│            ▼         ▼         ▼                               │
│         Semgrep    Snyk      ZAP                               │
│         (SAST)    (SCA)    (DAST)                              │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DefectDojo                                │
│         (Centralisation, déduplication, normalisation)          │
│              Déployé via Docker Compose + Ansible               │
└────────────────────────┬────────────────────────────────────────┘
                         │ API REST /api/v2/
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Risk Scoring Engine                            │
│   fetch_data → preprocess → train → predict → tag_findings     │
│        Modèle ML : RandomForest + XGBoost + LightGBM           │
│                  Explications via Gemini API                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌─────────────────────┐   ┌─────────────────────────────────────┐
│    API Backend      │   │         Dashboard React              │
│    (FastAPI)        │   │  Findings, scores, LLM, Jira, PR    │
│  JWT, rate limit,   │   │  TypeScript, Tailwind, React Query  │
│  cache, audit       │   │                                     │
└─────────────────────┘   └─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│               Infrastructure Kubernetes (K3s)                   │
│                                                                 │
│  Namespace invisithreat          Namespace monitoring           │
│  ├── Risk Scoring Engine         ├── Prometheus                 │
│  ├── Dashboard                   ├── Grafana                    │
│  ├── PostgreSQL                  ├── Loki                       │
│  ├── Redis                       ├── Promtail                   │
│  └── PVC (models, data, db)      └── Node Exporter              │
│                                                                 │
│  Registre Docker local (port 5000)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Composants du système

InvisiThreat est composé de neuf composants principaux organisés en couches fonctionnelles.

| Composant | Rôle | Technologie | Hébergement |
|-----------|------|-------------|-------------|
| Pipeline CI/CD | Orchestration de toute la chaîne | GitHub Actions | GitHub |
| DefectDojo | Centralisation des vulnérabilités | Python / Django | Docker Compose (hôte) |
| Risk Scoring Engine | Scoring IA et tagging intelligent | Python / ML | Pod K3s |
| API Backend | Exposition REST des données | FastAPI | Pod K3s |
| Dashboard | Interface utilisateur | React / TypeScript | Pod K3s |
| PostgreSQL | Stockage des données applicatives | PostgreSQL 16 | Pod K3s |
| Redis | Cache distribué des scores | Redis 7 | Pod K3s |
| Registre Docker | Images Docker locales | Registry v2 | Docker (hôte) |
| Stack Monitoring | Observabilité et logs | Prometheus / Grafana / Loki | Namespace monitoring |

---

## 4. Flux de données détaillé

Le flux de données d'InvisiThreat suit une chaîne linéaire et séquentielle, depuis le commit du développeur jusqu'à l'affichage du score de risque dans le tableau de bord.

### Étape 1 — Déclenchement du pipeline

Un développeur pousse du code sur la branche `main` ou `develop`. GitHub Actions détecte le push et identifie les projets modifiés en analysant les fichiers changés. Seuls les projets dont le répertoire contient un fichier `project.config.yml` sont inclus dans la matrice d'analyse.

### Étape 2 — Analyse de sécurité multi-dimensionnelle

Pour chaque projet détecté, trois scanners s'exécutent en parallèle (un par type d'analyse) :

**Semgrep** analyse le code source statiquement et produit un fichier JSON listant les patterns de vulnérabilités détectés selon les rulesets configurés dans `project.config.yml`.

**Snyk** analyse les dépendances tierces du projet et identifie les CVE connues. Il produit un fichier JSON et publie un snapshot sur son tableau de bord pour le suivi continu.

**OWASP ZAP** démarre l'application cible, effectue un crawl AJAX et un scan actif, puis produit des rapports XML pour le scan OpenAPI et le scan complet.

### Étape 3 — Centralisation dans DefectDojo

Chaque rapport de scan est importé dans DefectDojo via l'endpoint `reimport-scan` de son API REST. Ce mécanisme de réimport garantit que les findings déjà connus ne sont pas dupliqués, et que les findings résolus (absents du nouveau rapport) sont automatiquement fermés.

Chaque import est rattaché à un engagement trimestriel spécifique à la branche et au projet, ce qui permet de suivre l'évolution de la sécurité dans le temps.

### Étape 4 — Tagging intelligent

Le script `tag_findings.py` du Risk Scoring Engine interroge DefectDojo pour récupérer les findings récemment importés et leur applique des tags contextuels supplémentaires. Ces tags enrichissent les métadonnées des findings avec des informations métier (exposition externe, données sensibles, probabilité de faux positif) qui seront utilisées comme features par le modèle ML.

### Étape 5 — Security Gate

La porte de sécurité compare le nombre de findings critiques et de haute sévérité avant et après l'import. Si les nouveaux findings dépassent les seuils définis dans `project.config.yml`, le pipeline s'arrête avec une erreur et le déploiement est bloqué. Cela garantit qu'aucune régression de sécurité ne passe en production.

### Étape 6 — Construction et déploiement

Si la porte de sécurité est franchie, le pipeline construit les images Docker du Risk Scoring Engine et du Dashboard, les pousse vers le registre local taguées avec le SHA du commit, puis déploie les nouvelles images sur le cluster K3s.

### Étape 7 — Mise à jour du modèle ML

Le pipeline déclenche la mise à jour du modèle de scoring : collecte des données depuis DefectDojo, nettoyage et enrichissement, copie atomique dans le pod actif, redémarrage et vérification de disponibilité. Les scores sont alors recalculés sur l'ensemble des findings.

### Étape 8 — Exposition et visualisation

L'API FastAPI expose les findings enrichis de leurs scores via des endpoints REST sécurisés. Le Dashboard React interroge l'API et affiche une vue interactive des vulnérabilités avec leurs scores, explications LLM et suggestions de remédiation.

---

## 5. Couche d'analyse de sécurité

### Approche multi-dimensionnelle

InvisiThreat couvre les trois dimensions de l'analyse de sécurité applicative :

| Dimension | Outil | Ce qui est analysé |
|-----------|-------|-------------------|
| SAST (Static Application Security Testing) | Semgrep | Code source — patterns de vulnérabilités, mauvaises pratiques |
| SCA (Software Composition Analysis) | Snyk | Dépendances tierces — CVE connues, licences |
| DAST (Dynamic Application Security Testing) | OWASP ZAP | Application en cours d'exécution — comportement réel |

Cette combinaison est essentielle car chaque approche détecte des catégories de vulnérabilités que les autres ne voient pas. Le SAST détecte les erreurs de logique dans le code mais ne voit pas les problèmes de configuration à l'exécution. Le DAST voit le comportement réel de l'application mais ne peut pas inspecter le code source. Le SCA identifie les vulnérabilités connues dans les bibliothèques tierces.

### Configuration par projet

Chaque projet dispose de son propre fichier `project.config.yml` qui définit les paramètres spécifiques à chaque scanner : rulesets Semgrep, seuil de sévérité Snyk, timeout ZAP, endpoint OpenAPI, configuration de l'authentification JWT pour ZAP, et seuils de la porte de sécurité. Cette approche permet d'adapter le niveau d'analyse à chaque projet sans modifier le pipeline.

### Intégration avec DefectDojo

Les résultats des trois scanners sont importés dans DefectDojo avec des tags automatiques qui identifient le type d'outil, la branche, l'environnement et le projet. Ces tags sont hérités par les findings et constituent la base des features utilisées par le modèle ML pour calculer le score de risque.

---

## 6. Couche de centralisation — DefectDojo

### Rôle central dans l'architecture

DefectDojo joue un rôle de pivot dans l'architecture d'InvisiThreat. Il reçoit les résultats de tous les scanners, les normalise dans un format unifié, les déduplique, et les expose via une API REST standardisée. Sans DefectDojo, chaque composant du système devrait gérer ses propres formats de données et ses propres mécanismes de déduplication.

### Modèle de données

DefectDojo organise les données selon une hiérarchie à trois niveaux :

**Produit** : correspond à un projet applicatif (par exemple, Juice Shop). Chaque projet dans `projects/` crée ou réutilise un produit DefectDojo.

**Engagement** : correspond à une période d'analyse, nommée selon la convention `<branche> - <année>-<trimestre>` (par exemple, `main - 2026-Q2`). Les engagements sont trimestriels pour conserver l'historique sans accumulation infinie.

**Finding** : correspond à une vulnérabilité individuelle détectée par un scanner. Chaque finding porte ses métadonnées (sévérité, outil, branche, commit, tags) et son état (actif, résolu, faux positif).

### Mécanisme de réimport

L'action `dd-import` utilise l'endpoint `reimport-scan` plutôt que `import-scan`. Cette différence est fondamentale : le réimport met à jour les findings existants pour le même test (même titre, même engagement) au lieu d'en créer de nouveaux. Les findings qui n'apparaissent plus dans le rapport sont automatiquement fermés, offrant une vue toujours à jour de l'état de sécurité réel.

### Gestion des environnements

DefectDojo reçoit un tag d'environnement automatique pour chaque import, calculé selon la branche en cours : `main` et `master` reçoivent le tag `prod`, `staging` et `preprod` reçoivent `staging`, et toute autre branche reçoit `dev`. Ce tag est crucial pour le Risk Scoring Engine, qui pondère différemment un finding détecté en production par rapport à un finding sur une branche de développement.

---

## 7. Couche d'intelligence artificielle — Risk Scoring Engine

### Objectif du scoring

Le scoring IA résout un problème concret : lorsqu'un projet génère des centaines de findings, les équipes de sécurité ne peuvent pas tous les traiter avec la même priorité. Le Risk Scoring Engine calcule un score de risque contextuel de 0 à 10 pour chaque finding, permettant de concentrer les efforts sur les vulnérabilités qui représentent un danger réel dans le contexte spécifique de l'application.

### Pipeline de données

Le Risk Scoring Engine suit un pipeline en plusieurs étapes séquentielles :

**Collecte (`fetch_data.py`)** : interroge l'API DefectDojo pour extraire l'ensemble des findings actifs avec leurs métadonnées. Les données sont sauvegardées dans `data/raw/findings_raw.csv`.

**Prétraitement (`preprocess.py`)** : nettoie les données, gère les valeurs manquantes, encode les variables catégorielles, enrichit les findings avec le score EPSS (Exploit Prediction Scoring System) depuis l'API FIRST, et calcule un score contextuel basé sur les tags. La sortie est sauvegardée dans `data/processed/findings_clean.csv`.

**Entraînement (`train.py`)** : entraîne un modèle supervisé par stacking combinant RandomForest, XGBoost et LightGBM. Le méta-modèle apprend à combiner les prédictions des trois modèles de base pour minimiser l'erreur. Le pipeline entraîné est sérialisé dans `models/pipeline_latest.pkl`.

**Inférence (`predict_live.py`)** : charge le modèle entraîné et calcule les scores pour tous les findings courants. Les scores sont écrits dans `data/ai_scores_cache.json` et mis à jour dans DefectDojo.

**Tagging (`tag_findings.py`)** : applique des tags métier supplémentaires aux findings selon leur score et leur contexte, enrichissant les métadonnées pour l'affichage dans le Dashboard.

### Features utilisées par le modèle

Le modèle ML utilise plusieurs catégories de features pour calculer le score de risque :

| Catégorie | Features | Description |
|-----------|----------|-------------|
| Sévérité de base | Score CVSS, sévérité DefectDojo | Gravité intrinsèque de la vulnérabilité |
| Exploitabilité | Score EPSS | Probabilité d'exploitation dans les 30 jours |
| Contexte d'exposition | Tags `prod`, `external`, `internet-facing` | Impact potentiel selon l'environnement |
| Type de finding | Tags `sast`, `sca`, `dast` | Origine et nature du finding |
| Criticité métier | Tags `sensitive`, `pii`, `blocker` | Données sensibles ou blocage critique |
| Historique | Ancienneté, récidive | Vulnérabilités persistantes |

### Explications LLM

Pour chaque finding, le Risk Scoring Engine peut générer une explication textuelle en langage naturel via l'API Gemini. Cette explication contextualise le score calculé, décrit le risque dans les termes du projet, et suggère des pistes de remédiation adaptées. Les explications sont mises en cache dans Redis pour éviter des appels API répétés.

---

## 8. Couche applicative — API Backend et Dashboard

### API Backend (FastAPI)

L'API Backend expose les données de sécurité enrichies et la logique métier de la plateforme. Elle s'appuie sur FastAPI pour ses performances élevées, sa validation automatique des données et sa génération automatique de documentation Swagger.

**Sécurité de l'API :**

| Mécanisme | Implémentation |
|-----------|---------------|
| Authentification | JWT tokens avec expiration configurable |
| Autorisation | RBAC avec vérification de rôle par endpoint |
| Rate limiting | Limite de requêtes par utilisateur et par IP |
| Validation | Validation automatique des payloads via Pydantic |
| Audit | Journalisation de tous les accès et modifications |

**Endpoints principaux :**

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/findings` | GET | Liste des findings avec scores, filtres et pagination |
| `/api/predict` | POST | Prédiction de score en temps réel pour un finding |
| `/api/explain/{id}` | GET | Explication LLM d'un finding spécifique |
| `/api/jira/create-ticket` | POST | Création d'un ticket Jira depuis un finding |
| `/api/stats` | GET | Statistiques globales et métriques du modèle |
| `/api/model/info` | GET | Informations sur le modèle actif (version, accuracy) |
| `/health` | GET | Healthcheck de l'API |
| `/api/v1/ready` | GET | Disponibilité incluant le chargement du modèle ML |

### Dashboard React

Le Dashboard est une application monopage (SPA) construite avec React et TypeScript. Il communique exclusivement avec l'API Backend et ne contacte jamais DefectDojo directement.

**Pages principales :**

| Page | Description |
|------|-------------|
| Dashboard | Vue d'ensemble avec statistiques, tendances et findings critiques récents |
| Findings | Liste complète avec filtres multi-critères, tri et pagination |
| Risk Analysis | Détail d'un finding : score IA, explication LLM, historique |
| Remediation | Suggestions de correctifs automatiques et création de PR GitHub |
| Jira Integration | Création et synchronisation des tickets de remédiation |
| Model Performance | Métriques du modèle IA : accuracy, précision, rappel, distribution des scores |

**Gestion de l'état :**

Zustand gère l'état global de l'application (authentification, préférences, filtres actifs). React Query gère les appels API avec mise en cache automatique, invalidation des données et retry sur erreur. Cette combinaison minimise les re-renders inutiles et garantit une expérience fluide même avec de nombreux findings.

---

## 9. Couche infrastructure

### Décomposition en deux niveaux

L'infrastructure d'InvisiThreat se décompose en deux niveaux distincts mais complémentaires :

**Niveau hôte** : DefectDojo et le registre Docker local sont déployés directement sur le serveur via Docker Compose et gérés par Ansible. Ce choix est justifié par la nature de DefectDojo, qui est une application lourde avec de nombreuses dépendances et qui n'a pas besoin d'être redémarrée fréquemment.

**Niveau Kubernetes** : les services applicatifs d'InvisiThreat (Risk Scoring Engine, Dashboard, PostgreSQL, Redis) sont déployés dans un cluster K3s. Ce choix permet de bénéficier des mécanismes de Kubernetes (healthchecks, redémarrage automatique, mise à jour sans downtime, gestion des secrets) tout en maintenant une empreinte légère grâce à K3s.

### Réseau et communication

La communication entre les pods Kubernetes et DefectDojo (hors cluster) est assurée par un Service Kubernetes de type ExternalName combiné à un objet Endpoints manuel. Ce mécanisme exploite CoreDNS (inclus dans K3s) pour résoudre le nom `defectdojo-external` en l'IP du nœud hôte, permettant aux pods d'accéder à DefectDojo comme s'il était dans le cluster.

La communication entre les services applicatifs utilise les noms DNS internes Kubernetes (`postgres-svc`, `redis-svc`, `risk-scoring-engine-svc`), ce qui rend les configurations indépendantes des adresses IP.

### Gestion des secrets

Les secrets ne transitent jamais en clair dans le pipeline ou dans les manifests. Ils sont stockés dans GitHub Secrets, injectés par le pipeline lors du déploiement via la commande `kubectl create secret`, et consommés par les pods via le mécanisme `envFrom` de Kubernetes. Cette chaîne garantit que les valeurs sensibles ne sont jamais visibles dans les logs du pipeline ni dans le dépôt Git.

---

## 10. Couche monitoring

### Philosophie d'observabilité

L'observabilité d'InvisiThreat repose sur trois piliers : les métriques (Prometheus), les logs (Loki) et la visualisation (Grafana). Ces trois éléments permettent de répondre à deux types de questions : les métriques répondent à "combien ?" et "à quelle fréquence ?", tandis que les logs répondent à "pourquoi ?" et "que s'est-il passé exactement ?".

### Sources de métriques

| Source | Outil | Métriques exposées |
|--------|-------|-------------------|
| Pods applicatifs | Auto-découverte Prometheus | Requêtes API, latence, erreurs |
| Système hôte | Node Exporter (DaemonSet) | CPU, mémoire, disque, réseau |
| Cluster Kubernetes | Kube State Metrics | État des déploiements, pods, PVC |
| Base de données | PostgreSQL Exporter | Connexions, taille des tables, requêtes lentes |

### Alerting

Prometheus peut être configuré pour déclencher des alertes lorsque des seuils critiques sont atteints : pod en état `CrashLoopBackOff`, mémoire saturée à plus de 90 %, latence API supérieure à 2 secondes, ou score de risque critique détecté sur un nouveau finding. Ces alertes peuvent être routées vers un canal de notification (email, Slack) via Alertmanager.

---

## 11. Interactions entre composants

### Tableau des dépendances

| Composant | Dépend de | Protocole |
|-----------|-----------|-----------|
| Pipeline CI/CD | Tous les composants | GitHub Actions / SSH / kubectl |
| Risk Scoring Engine | DefectDojo, PostgreSQL, Redis, Gemini API | HTTP REST / SQL / Redis protocol |
| API Backend | PostgreSQL, Redis, Risk Scoring Engine | SQL / Redis protocol / HTTP |
| Dashboard | API Backend | HTTP REST |
| DefectDojo | Docker Compose (hôte) | HTTP |
| Prometheus | Tous les pods annotés | HTTP (scrape) |
| Grafana | Prometheus, Loki | HTTP |
| Loki | Promtail | HTTP |

### Ordre de démarrage critique

Le démarrage des services suit un ordre précis pour éviter les erreurs de dépendance :

```
PostgreSQL (prêt)
      │
      ▼
Redis (prêt)
      │
      ▼
Risk Scoring Engine (modèle ML chargé → /api/v1/ready)
      │
      ▼
Dashboard (Nginx prêt → /ready)
      │
      ▼
Pipeline : vérification des endpoints /health et /docs
```

DefectDojo démarre indépendamment de cette chaîne (playbook `ensure-defectdojo.yml`) car il est hors du cluster Kubernetes.

---

## 12. Choix techniques et justifications

### Pourquoi GitHub Actions pour le CI/CD

GitHub Actions offre une intégration native avec le dépôt Git, un système d'actions composites réutilisables, et la possibilité d'utiliser des self-hosted runners directement sur le serveur cible. Ce dernier point est crucial pour InvisiThreat car il permet au pipeline d'accéder directement au cluster K3s et à DefectDojo sans passer par un tunnel ou une exposition réseau.

### Pourquoi DefectDojo comme agrégateur

DefectDojo est la référence open source pour la gestion des vulnérabilités en contexte DevSecOps. Il supporte nativement les formats de sortie de Semgrep, Snyk et ZAP, dispose d'une API REST complète, et offre des mécanismes de déduplication et de suivi dans le temps qui auraient été très coûteux à développer de zéro.

### Pourquoi le stacking ML (RandomForest + XGBoost + LightGBM)

Chaque modèle de base capture des patterns différents dans les données : RandomForest est robuste aux valeurs aberrantes, XGBoost excelle sur les données tabulaires structurées, et LightGBM est très efficace sur les features catégorielles après encodage. Le méta-modèle de stacking apprend la meilleure façon de combiner leurs prédictions, produisant un résultat plus précis et plus stable qu'aucun modèle seul.

### Pourquoi K3s plutôt que Docker Compose pour les services applicatifs

Docker Compose n'offre pas de mécanismes de healthcheck avancés, de redémarrage automatique fiable, ni de mise à jour sans downtime. K3s apporte ces fonctionnalités nativement via les probes Kubernetes, le `rollout restart`, et la stratégie `RollingUpdate`. Il permet également une gestion centralisée des secrets via les objets Kubernetes Secret, évitant d'exposer des valeurs sensibles dans des fichiers `.env`.

### Pourquoi FastAPI pour le backend

FastAPI génère automatiquement une documentation Swagger interactive depuis les annotations Python, ce qui est précieux pour le développement et les tests. Sa validation automatique des payloads via Pydantic réduit le code de validation manuel. Ses performances asynchrones (basées sur Starlette et Uvicorn) sont adaptées à un service qui effectue simultanément des appels à PostgreSQL, Redis, DefectDojo et l'API Gemini.

### Pourquoi React et TypeScript pour le Dashboard

React est le choix naturel pour une application avec de nombreuses interactions utilisateur (filtres, tri, pagination, formulaires). TypeScript apporte la sécurité du typage statique, particulièrement utile pour manipuler des structures de données complexes comme les findings DefectDojo et les scores ML. Tailwind CSS permet une personnalisation fine de l'interface sans écrire de CSS personnalisé.

---

## 13. Sécurité de l'architecture

### Principes appliqués

| Principe | Implémentation dans InvisiThreat |
|----------|----------------------------------|
| Least privilege | Chaque service n'accède qu'aux ressources dont il a besoin |
| Defense in depth | Security Gate + JWT + RBAC + NetworkPolicies |
| Secrets management | GitHub Secrets → Kubernetes Secrets → variables d'environnement |
| Audit trail | Logs structurés JSON, centralisés dans Loki |
| Immutable infrastructure | Images taguées par SHA, pas de modification en production |
| Zero trust | Chaque appel API est authentifié, même entre services internes |

### Security Gate comme mécanisme de protection

La porte de sécurité est un élément architectural central, pas un simple contrôle. Elle garantit que le pipeline de déploiement est également un pipeline de sécurité : aucune image ne peut être construite et poussée en production si de nouveaux findings critiques ont été introduits. Cette intégration rend la sécurité obligatoire plutôt qu'optionnelle dans le cycle de développement.

### Isolation réseau

Les services applicatifs sont isolés dans le namespace `invisithreat`. La communication avec DefectDojo (hors cluster) se fait uniquement via le Service ExternalName défini dans les manifests, limitant la surface d'exposition. Les bases de données (PostgreSQL et Redis) sont exposées uniquement en ClusterIP, inaccessibles depuis l'extérieur du cluster.

---

> **Document suivant** : [02-setup.md](./02-setup.md) — Guide d'installation et de configuration