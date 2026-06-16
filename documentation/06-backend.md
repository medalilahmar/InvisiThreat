# API Backend — Serveur FastAPI

Le backend expose les fonctionnalités du Risk Scoring Engine via une API REST documentée automatiquement (Swagger). Il est le point d'entrée unique pour le dashboard React, les intégrations Jira/LLM, et les outils d'administration.

---

## Table des matières

- [1. Présentation générale](#1-présentation-générale)
- [2. Structure du serveur](#2-structure-du-serveur)
- [3. Middlewares et sécurité](#3-middlewares-et-sécurité)
- [4. Endpoints principaux](#4-endpoints-principaux)
- [5. Cache et performances](#5-cache-et-performances)
- [6. Gestion des dépendances](#6-gestion-des-dépendances)
- [7. Exemple d'appel](#7-exemple-dappel)
- [8. Intégration avec le reste d'InvisiThreat](#8-intégration-avec-le-reste-dinvisithreat)
- [9. Sécurité et bonnes pratiques](#9-sécurité-et-bonnes-pratiques)

---

## 1. Présentation générale

L'API est construite avec **FastAPI** (asynchrone) et s'appuie sur :

- **ModelManager** : chargement et cache du modèle de scoring.
- **LocalDataLoader** : chargement des données DefectDojo (fichier CSV) en mémoire.
- **Cache mémoire** : scores IA pré-calculés pour des réponses rapides.
- **Authentification JWT** (RBAC basique).
- **Middlewares** : CORS, compression GZip, rate limiting, logging.

L'application est lancée via `api_simple.py` (ou `uvicorn api_simple:app`) et écoute par défaut sur le port `8081`.

---

## 2. Structure du serveur

```
src/
├── api_simple.py          # Point d'entrée, configuration, lifespan, middlewares
└── server/
    ├── config.py          # Paramétrage (labels, couleurs, chemins)
    ├── dependencies.py    # Injection de dépendances (ModelManager, DataLoader)
    ├── model_manager.py   # Singleton du modèle + cache de prédictions
    ├── data_loader.py     # Chargement et indexation du CSV
    ├── cache.py           # Gestion du cache scores / prédictions
    ├── schemas.py         # Modèles Pydantic (requêtes / réponses)
    ├── utils.py           # Fonctions utilitaires (parsing tags, normalisation)
    └── routers/
        ├── predict.py              # /predict et /predict/batch
        ├── findings.py             # /defectdojo/findings, /products, /engagements
        ├── auth_router.py          # Authentification (login, token)
        ├── admin_router.py         # Administration (reload, cache, refresh)
        ├── analytics_router.py
        ├── llm.py
        ├── jira.py
        └── notifications_router.py
```

---

## 3. Middlewares et sécurité

| Middleware | Description |
|------------|-------------|
| **Rate limiting** | 100 requêtes / IP / 60 secondes. Retourne `429` en cas de dépassement. |
| **CORS** | Origines définies via `CORS_ORIGINS` (toutes autorisées en développement). |
| **GZip** | Compression activée pour les réponses volumineuses. |
| **Request ID** | Identifiant unique tracé dans les logs et les en-têtes de chaque requête. |
| **JWT** | La plupart des endpoints sont protégés. Les routes admin exigent le rôle `admin`. |

---

## 4. Endpoints principaux

### 4.1 Santé et informations

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Statut général, version, disponibilité du modèle. |
| `GET` | `/health` | Vérification détaillée (modèle chargé, uptime, métadonnées). |
| `GET` | `/metrics` | Métriques Prometheus : cache hits, distribution des scores, etc. |
| `GET` | `/model/info` | Version du modèle, classes, features, F1 score. |
| `POST` | `/model/reload` | Recharge le modèle et rescorte les findings. |

### 4.2 Router `predict` — Scoring en temps réel

**`POST /predict`** — soumet une vulnérabilité unique et retourne son score IA.

Corps de la requête :

```json
{
  "finding_id": 123,
  "severity": "high",
  "cvss_score": 7.5,
  "tags": ["urgent", "production"],
  "has_cve": 1,
  "age_days": 45
}
```

Champs retournés :

| Champ | Description |
|-------|-------------|
| `risk_level` | `low`, `medium`, `high` ou `critical` |
| `risk_score` | Score continu (0–100) |
| `confidence` | Probabilité maximale |
| `probabilities` | Détail par classe |
| `context_score`, `cvss_score` | Données contextuelles |

**`POST /predict/batch`** — scoring par lot. Retourne un résumé statistique et la liste complète des résultats.

### 4.3 Router `findings` — Consultation des vulnérabilités enrichies

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/defectdojo/products` | Liste tous les produits (filtrée selon les droits RBAC). |
| `GET` | `/defectdojo/engagements?product_id=...` | Liste les engagements d'un produit. |
| `GET` | `/defectdojo/findings` | Findings avec scores IA (params : `engagement_id`, `product_id`, `limit`). |
| `GET` | `/defectdojo/findings/{finding_id}` | Détail d'un finding unique. |
| `GET` | `/defectdojo/products/{product_id}/findings` | Raccourci par produit. |
| `GET` | `/defectdojo/engagements/{engagement_id}/findings` | Raccourci par engagement. |

Tous les retours incluent :

```
ai_risk_score_cont   # score continu 0–10
risk_level           # low / medium / high / critical
risk_color
ai_confidence
context_score
exposure_norm
model_base_score
business_nudge
cve
age_days
tags
```

---

## 5. Cache et performances

- Un cache mémoire `_scores_cache_memory` stocke les scores de tous les findings, calculés au démarrage via `score_all_findings_at_startup()`.
- Le cache est rafraîchi lors d'un `POST /model/reload` ou `POST /data/refresh`.
- Le `ModelManager` embarque son propre cache de prédictions (`predict_batch_cached`) pour éviter de recalculer des features identiques.

---

## 6. Gestion des dépendances

| Dépendance FastAPI | Rôle |
|--------------------|------|
| `get_model_manager()` | Fournit le modèle chargé (rechargement dynamique sans redémarrage). |
| `require_local_loader()` | Donne accès aux données DefectDojo indexées. |
| `get_current_user()` | Authentification et RBAC. |

---

## 7. Exemple d'appel

```bash
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{
    "severity": "high",
    "cvss_score": 7.5,
    "tags": ["urgent", "production"],
    "has_cve": 1,
    "age_days": 45
  }'
```

Réponse typique :

```json
{
  "request_id": "abc123",
  "risk_level": "high",
  "risk_score": 72.5,
  "confidence": 0.87,
  "probabilities": {
    "low": 0.05,
    "medium": 0.08,
    "high": 0.87,
    "critical": 0.00
  }
}
```

---

## 8. Intégration avec le reste d'InvisiThreat

| Composant | Lien |
|-----------|------|
| Dashboard React | Consomme les endpoints `/findings` et `/predict`. |
| Pipeline CI/CD | Appelle `predict_live.py` en mode `--ci-mode`, même modèle que l'API. |
| DefectDojo | L'API lit via `LocalDataLoader` ; `tag_findings.py` et `predict_live.py` écrivent les tags et notes. |
| Jira / Gemini | Routes `/jira` et `/llm` pour la création de tickets et les recommandations. |

---

## 9. Sécurité et bonnes pratiques

- Les clés secrètes (API DefectDojo, Jira, Gemini) ne sont jamais loguées.
- L'API est conçue pour être placée derrière un reverse proxy (nginx) en production avec HTTPS.
- Le rate limiting et l'authentification JWT protègent les endpoints sensibles.
- Les logs incluent un `request_id` pour faciliter le traçage.
- Les dépendances sont chargées dynamiquement pour permettre le rechargement à chaud du modèle.