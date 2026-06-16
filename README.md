# InvisiThreat — Plateforme DevSecOps Intelligente

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Security](https://img.shields.io/badge/security-DevSecOps-orange)
![AI](https://img.shields.io/badge/AI-Risk%20Scoring-brightgreen)
![Kubernetes](https://img.shields.io/badge/k8s-K3s-326ce5)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF)

**InvisiThreat** est une plateforme DevSecOps de fin d'études (PFE) qui automatise la détection, l'analyse et la priorisation des vulnérabilités applicatives à l'aide de l'intelligence artificielle. Elle cible l'application volontairement vulnérable **OWASP Juice Shop** et couvre l'intégralité de la chaîne DevSecOps : analyse statique (SAST), analyse de dépendances (SCA), analyse dynamique (DAST), agrégation dans DefectDojo, scoring IA, API REST, tableau de bord interactif et infrastructure entièrement automatisée.

---

## Table des matières

1. [Architecture globale](#1-architecture-globale)
2. [Technologies utilisées](#2-technologies-utilisées)
3. [Structure du projet](#3-structure-du-projet)
4. [Fonctionnalités principales](#4-fonctionnalités-principales)
5. [Déploiement de l'infrastructure](#5-déploiement-de-linfrastructure)
6. [Développement local — Backend et Frontend](#6-développement-local--backend-et-frontend)
7. [Pipeline CI/CD — GitHub Actions](#7-pipeline-cicd--github-actions)
8. [Documentation technique](#8-documentation-technique)
9. [Métriques et KPIs](#9-métriques-et-kpis)
10. [Auteur](#10-auteur)

---

## 1. Architecture globale

InvisiThreat suit une architecture en couches, orchestrée par un pipeline CI/CD GitHub Actions qui alimente chaque composant de manière automatique et séquentielle.

```
Code source (push / PR)
          |
          v
  GitHub Actions (CI/CD)
          |
    +-----+-----+-----+
    |     |     |     |
    v     v     v     v
 Semgrep Snyk  ZAP   Infra
 (SAST) (SCA) (DAST) Check
    |     |     |     |
    +-----+-----+-----+
          |
          v
      DefectDojo
  (Centralisation et déduplication des findings)
          |
          v
  Risk Scoring Engine
  (Preprocessing, entraînement ML, scoring 0-10)
          |
          v
     API Backend (FastAPI)
  (Endpoints REST, JWT, cache, LLM, Jira)
          |
          v
   Dashboard React
  (Visualisation, filtres, correctifs, tickets)
          |
          v
     Infrastructure K3s
  (Kubernetes, Ansible, Prometheus, Grafana, Loki)
```

Toute l'infrastructure (cluster Kubernetes, monitoring, services applicatifs, secrets) est provisionnée **automatiquement** par le pipeline CI/CD via des playbooks Ansible et des manifests Kubernetes. Aucune intervention manuelle n'est nécessaire après la configuration initiale du serveur et des secrets GitHub.

---

## 2. Technologies utilisées

### Backend et Intelligence Artificielle

| Technologie | Rôle | Version |
|-------------|------|---------|
| Python | Langage principal | 3.8+ |
| FastAPI | API REST | 0.95+ |
| Scikit-learn | Pipeline ML | 1.0+ |
| XGBoost | Modèle de scoring (stacking) | 1.5+ |
| LightGBM | Modèle de scoring (stacking) | 3.3+ |
| SHAP | Explicabilité du modèle | 0.41+ |
| Pandas | Traitement des données | 1.3+ |
| Gemini API | Génération d'explications LLM | — |

### Frontend

| Technologie | Rôle | Version |
|-------------|------|---------|
| React | Framework UI | 18 |
| TypeScript | Typage statique | 4.9+ |
| Vite | Bundler | 4.0+ |
| Tailwind CSS | Styles | 3.3+ |
| Zustand | Gestion d'état | — |
| React Query | Fetching et cache | 4.0+ |

### Infrastructure et DevOps

| Technologie | Rôle | Version |
|-------------|------|---------|
| Docker | Conteneurisation | 20.10+ |
| K3s | Cluster Kubernetes léger | latest |
| Ansible | Provisionnement IaC | 2.9+ |
| GitHub Actions | Pipeline CI/CD | — |
| Prometheus | Métriques | 2.40+ |
| Grafana | Visualisation monitoring | 9.0+ |
| Loki | Centralisation des logs | 2.7+ |
| Nginx | Ingress Controller | 1.23+ |
| PostgreSQL | Base de données | — |
| Redis | Cache distribué | — |

### Intégrations externes

| Outil | Rôle |
|-------|------|
| DefectDojo | Centralisation et gestion des vulnérabilités |
| Snyk | Analyse des dépendances (SCA) |
| Semgrep | Analyse statique du code (SAST) |
| OWASP ZAP | Analyse dynamique (DAST) |
| Jira | Gestion des tickets de remédiation |
| GitHub API | Création automatique de Pull Requests |

---

## 3. Structure du projet

```
InvisiThreat/
├── .github/
│   ├── workflows/
│   │   └── devsecops.yml          # Pipeline CI/CD principal
│   └── actions/                   # Actions composites réutilisables
│       ├── load-config/           # Chargement de la config projet
│       ├── dd-setup/              # Initialisation DefectDojo
│       ├── dd-import/             # Import générique DefectDojo
│       ├── dd-import-zap/         # Import ZAP DefectDojo
│       ├── zap-scan/              # Orchestration des scans ZAP
│       └── deploy/                # Déploiement Kubernetes + ML
│
├── documentation/                 # Documentation technique complète
│   ├── index.md
│   ├── 01-architecture.md
│   ├── 02-setup.md
│   ├── 03-securite.md
│   ├── 04-cicd.md
│   ├── 05-risk-scoring-engine.md
│   ├── 06-api-backend.md
│   ├── 07-infrastructure.md
│   └── 08-security-dashboard.md
│
├── infrastructure/
│   ├── ansible/                   # Playbooks Ansible (K3s, monitoring, DefectDojo)
│   └── kubernetes/                # Manifests Kubernetes
│       ├── namespace.yaml
│       ├── configmap.yaml
│       ├── storage/
│       ├── postgres/
│       ├── redis/
│       ├── risk-scoring-engine/
│       ├── dashboard/
│       └── external-services.yaml
│
├── risk-scoring-engine/           # Moteur IA
│   ├── src/
│   │   ├── fetch_data.py          # Collecte des findings DefectDojo
│   │   ├── preprocess.py          # Nettoyage et feature engineering
│   │   ├── train.py               # Entraînement du modèle
│   │   ├── predict_live.py        # Inférence en temps réel
│   │   ├── tag_findings.py        # Tagging intelligent
│   │   ├── api_simple.py          # Serveur FastAPI
│   │   └── create_admin.py        # Initialisation admin
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── models/
│   └── reports/
│
├── security-dashboard/            # Application React
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── store/
│   └── package.json
│
├── projects/                      # Projets analysés (ex: juice-shop)
│   └── <nom-projet>/
│       └── project.config.yml     # Configuration par projet
│
├── docker-compose.yml             # Stack DefectDojo locale
├── docker-compose.override.yml
├── .gitignore
└── README.md
```

---

## 4. Fonctionnalités principales

### Analyse de sécurité automatisée

Le pipeline exécute trois types d'analyses à chaque modification du code source :

- **SAST (Semgrep)** : analyse statique du code source, détection de patterns de vulnérabilités, rulesets configurables par projet
- **SCA (Snyk)** : analyse des dépendances tierces, détection de CVE connues, monitoring continu
- **DAST (OWASP ZAP)** : analyse dynamique sur l'application en cours d'exécution, scan OpenAPI et Full Scan, authentification JWT supportée

### Moteur de scoring IA

Chaque vulnérabilité reçoit un score de risque contextuel de 0 à 10, calculé par un modèle de machine learning supervisé utilisant le stacking (RandomForest + XGBoost + LightGBM). Le score intègre le CVSS, le score EPSS (probabilité d'exploitation), le contexte d'exposition (production, externe, données sensibles) et les tags détectés automatiquement. Les explications textuelles sont générées via l'API Gemini.

### API REST sécurisée

L'API FastAPI expose les données de sécurité et la logique métier avec authentification JWT, rate limiting, cache des scores et audit trail complet. Elle propose des endpoints de prédiction en temps réel, d'explication LLM et d'intégration Jira.

### Dashboard interactif

Le dashboard React offre une visualisation complète des vulnérabilités avec leurs scores IA, des filtres multi-critères, des explications LLM détaillées, des suggestions de correctifs et la création de tickets Jira directement depuis l'interface.

### Infrastructure 100 % automatisée

Le cluster Kubernetes (K3s), la stack de monitoring (Prometheus, Grafana, Loki), les services applicatifs (PostgreSQL, Redis, API, Dashboard) et les secrets sont provisionnés et mis à jour automatiquement par le pipeline CI/CD. Aucune intervention manuelle n'est requise après la configuration initiale.

---

## 5. Déploiement de l'infrastructure

### Comment fonctionne l'automatisation complète

L'ensemble de l'infrastructure est géré par le pipeline GitHub Actions. Dès qu'un push est effectué sur `main` ou `develop`, le pipeline :

1. Détecte les projets modifiés
2. Exécute les scans SAST, SCA et DAST
3. Importe les résultats dans DefectDojo
4. Applique le tagging intelligent et calcule les scores IA
5. Construit et pousse les images Docker vers le registre local
6. Provisionne ou met à jour le cluster K3s via Ansible (idempotent)
7. Déploie tous les services Kubernetes dans le bon ordre
8. Injecte les secrets depuis GitHub Secrets
9. Met à jour le modèle ML et redémarre les services
10. Vérifie la santé de tous les endpoints

Tout ce processus ne nécessite aucune action manuelle. Il suffit de configurer le serveur cible et les secrets GitHub une seule fois.

---

### Prérequis sur le serveur cible

Le serveur (VPS, machine physique ou VM) doit disposer de :

- Ubuntu 20.04 ou équivalent
- Minimum 8 Go de RAM, 4 CPUs, 50 Go de stockage
- Docker installé
- Python 3.8+ installé
- Un utilisateur avec accès `sudo` sans mot de passe (pour Ansible)
- SSH accessible depuis le runner GitHub Actions

---

### Configuration du runner GitHub Actions

Le runner GitHub Actions doit avoir accès au serveur cible pour exécuter les playbooks Ansible et les commandes `kubectl`. Il est recommandé d'utiliser un **self-hosted runner** installé directement sur le serveur cible.

**Installation du runner sur le serveur :**

1. Dans le dépôt GitHub, aller dans `Settings > Actions > Runners > New self-hosted runner`
2. Suivre les instructions d'installation pour Linux
3. Démarrer le runner en tant que service :

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

Le runner s'enregistre automatiquement et est prêt à recevoir les jobs du pipeline.

---

### Secrets GitHub à configurer

Dans `Settings > Secrets and variables > Actions`, ajouter les secrets suivants :

| Secret | Description |
|--------|-------------|
| `DEFECTDOJO_API_KEY` | Clé API de l'instance DefectDojo |
| `SNYK_TOKEN` | Token d'authentification Snyk |
| `GEMINI_API_KEY` | Clé API Google Gemini (LLM) |
| `JIRA_API_TOKEN` | Token API Jira Cloud |
| `JIRA_EMAIL` | Email du compte Jira |
| `JIRA_SERVER` | URL du serveur Jira (ex : `https://yourorg.atlassian.net`) |
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL |
| `JWT_SECRET_KEY` | Clé secrète pour les tokens JWT |
| `API_ADMIN_TOKEN` | Token admin pour l'API FastAPI |
| `SMTP_PASSWORD` | Mot de passe SMTP pour les notifications email |
| `GITHUB_TOKEN` | Token GitHub (disponible automatiquement dans Actions) |

Une fois ces secrets en place et le runner configuré, le pipeline est entièrement opérationnel. Chaque push sur le dépôt déclenche automatiquement l'ensemble de la chaîne DevSecOps.

---

### Ordre de déploiement automatique des services

Le pipeline déploie les services dans l'ordre suivant pour garantir la cohérence des dépendances :

```
Namespace Kubernetes
        |
        v
Secrets et ConfigMaps
        |
        v
Service ExternalName (DefectDojo)
        |
        v
PersistentVolumeClaims (stockage)
        |
        v
PostgreSQL  →  attente rollout (300s)
        |
        v
Redis  →  attente rollout (300s)
        |
        v
Risk Scoring Engine + Dashboard
→  mise à jour image (SHA du commit)
→  attente rollout (240s)
        |
        v
Mise à jour du modèle ML
→  fetch → preprocess → copie atomique → redémarrage
        |
        v
Vérification des endpoints /health et /docs
```

---

## 6. Développement local — Backend et Frontend

Cette section concerne uniquement le développement et le test locaux du backend et du frontend. En production, tout est déployé automatiquement par le pipeline.

### Backend — Risk Scoring Engine et API FastAPI

**Prérequis** : Python 3.8+

```bash
cd risk-scoring-engine
pip install -r requirements.txt
```

Lancer l'API en mode développement :

```bash
python src/api_simple.py
```

L'API est accessible sur `http://localhost:8000`. La documentation interactive Swagger est disponible sur `http://localhost:8000/docs`.

Pour exécuter manuellement le pipeline de données :

```bash
# Collecte des findings depuis DefectDojo
python src/fetch_data.py

# Nettoyage et feature engineering
python src/preprocess.py

# Entraînement du modèle
python src/train.py

# Inférence sur les données courantes
python src/predict_live.py
```

---

### Frontend — Dashboard React

**Prérequis** : Node.js 18+

```bash
cd security-dashboard
npm install
npm run dev
```

Le dashboard est accessible sur `http://localhost:5173`.

Pour construire la version de production :

```bash
npm run build
```

---

## 7. Pipeline CI/CD — GitHub Actions

Le pipeline est défini dans `.github/workflows/devsecops.yml`. Il se compose de quatre jobs principaux exécutés séquentiellement :

| Job | Rôle | Condition |
|-----|------|-----------|
| `detect` | Détecte les projets modifiés et construit la matrice d'exécution | Toujours |
| `security` | Exécute SAST, SCA, DAST et vérifie la porte de sécurité | Projets détectés |
| `build-and-push` | Construit et pousse les images Docker | Sécurité réussie, hors PR |
| `deploy` | Déploie sur K3s et met à jour le modèle ML | Build réussi, hors PR |

### Déclencheurs

| Événement | Branches | Filtrage |
|-----------|----------|----------|
| Push | `main`, `develop` | `projects/**`, `risk-scoring-engine/**`, `security-dashboard/**` |
| Pull Request | `main`, `develop` | `projects/**` |
| Déclenchement manuel | — | Projet et options configurables |

### Porte de sécurité (Security Gate)

Le pipeline intègre une porte de sécurité qui bloque automatiquement le déploiement si de nouveaux findings critiques ou de haute sévérité sont détectés au-delà des seuils définis dans `project.config.yml`. Cette porte garantit qu'aucune régression de sécurité ne passe en production.

Pour la documentation complète du pipeline, consulter [documentation/04-cicd.md](./documentation/04-cicd.md).

---

## 8. Documentation technique

La documentation complète est disponible dans le dossier `documentation/` :

| Fichier | Contenu |
|---------|---------|
| [index.md](./documentation/index.md) | Index général, guide de démarrage, flux de travail complet |
| [01-architecture.md](./documentation/01-architecture.md) | Architecture globale, flux de données, choix techniques |
| [02-setup.md](./documentation/02-setup.md) | Guide d'installation et de configuration |
| [03-securite.md](./documentation/03-securite.md) | Outils de sécurité : Semgrep, Snyk, ZAP |
| [04-cicd.md](./documentation/04-cicd.md) | Pipeline CI/CD, jobs, actions composites |
| [05-risk-scoring-engine.md](./documentation/05-risk-scoring-engine.md) | Moteur IA : données, preprocessing, modèle, inférence |
| [06-api-backend.md](./documentation/06-api-backend.md) | API FastAPI : endpoints, sécurité, middlewares |
| [07-infrastructure.md](./documentation/07-infrastructure.md) | Ansible, K3s, Kubernetes, monitoring |
| [08-security-dashboard.md](./documentation/08-security-dashboard.md) | Dashboard React : pages, composants, intégrations |

---

## 9. Métriques et KPIs

| Métrique | Description | Cible |
|----------|-------------|-------|
| MTTR | Temps moyen de résolution des vulnérabilités | < 72 heures |
| Findings critiques ouverts | Vulnérabilités critiques non résolues | 0 |
| Couverture de scan | Pourcentage du code analysé | > 90 % |
| Précision du modèle IA | Accuracy du scoring de risque | > 85 % |
| Durée du pipeline | Temps total d'exécution CI/CD | < 30 minutes |
| Taux de faux positifs | Findings incorrectement signalés | < 10 % |

---

## 10. Auteur

**Projet de fin d'études (PFE) — 2025-2026**

- Auteur : [Votre nom]
- Institution : [Votre école / université]
- Encadrant : [Nom de l'encadrant]
- Licence : MIT

---

*InvisiThreat — Sécurité intelligente, pipeline automatisé, infrastructure reproductible.*