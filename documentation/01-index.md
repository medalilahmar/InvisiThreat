# Documentation — InvisiThreat

InvisiThreat est une plateforme DevSecOps complète qui intègre l'analyse de sécurité continue (SAST, SCA, DAST) avec un moteur de scoring de risque basé sur l'intelligence artificielle. Ce document constitue le point d'entrée de toute la documentation technique du projet.

---

## Table des matières

1. [Présentation du projet](#1-présentation-du-projet)
2. [Composants principaux](#2-composants-principaux)
3. [Index des fichiers de documentation](#3-index-des-fichiers-de-documentation)
4. [Concepts clés](#4-concepts-clés)
5. [Guide de démarrage rapide](#5-guide-de-démarrage-rapide)
6. [Flux de travail complet](#6-flux-de-travail-complet)
7. [Métriques et KPIs](#7-métriques-et-kpis)
8. [Sécurité et bonnes pratiques](#8-sécurité-et-bonnes-pratiques)
9. [Contributions et support](#9-contributions-et-support)
10. [Ressources additionnelles](#10-ressources-additionnelles)

---

## 1. Présentation du projet

InvisiThreat automatise la détection, l'analyse et la priorisation des vulnérabilités de sécurité dans les applications modernes. Le projet repose sur un pipeline CI/CD GitHub Actions qui orchestre plusieurs scanners de sécurité, centralise les résultats dans DefectDojo, les enrichit avec un score de risque calculé par un modèle de machine learning, et les expose via une API et un tableau de bord interactif.

Le projet a été conçu comme un projet de mémoire de fin d'études, démontrant les meilleures pratiques en DevSecOps, sécurité applicative et automatisation par l'intelligence artificielle.

---

## 2. Composants principaux

| Composant | Rôle | Technologie |
|-----------|------|-------------|
| Pipeline CI/CD | Orchestration des scanners de sécurité | GitHub Actions |
| DefectDojo | Centralisation et gestion des vulnérabilités | Python / Django |
| Risk Scoring Engine | Attribution de scores de risque IA | Python / XGBoost / LightGBM |
| API Backend | Exposition des données et logique métier | FastAPI / Python |
| Dashboard | Interface utilisateur d'analyse | React / TypeScript |
| Infrastructure | Provisionnement et orchestration | Ansible / Kubernetes (K3s) |
| Monitoring | Supervision et logs centralisés | Prometheus / Grafana / Loki |
| Base de données | Stockage des données applicatives | PostgreSQL |
| Cache | Gestion du cache distribué | Redis |

---

## 3. Index des fichiers de documentation

| Fichier | Description |
|---------|-------------|
| [01-architecture.md](./01-architecture.md) | Vue d'ensemble du système, flux de données, diagrammes et choix techniques |
| [02-setup.md](./02-setup.md) | Guide d'installation complète et de configuration de l'environnement |
| [03-securite.md](./03-securite.md) | Détail des outils de sécurité (Semgrep, Snyk, ZAP) et de leur intégration |
| [04-cicd.md](./04-cicd.md) | Pipeline CI/CD GitHub Actions : jobs, étapes et automatisation |
| [05-risk-scoring-engine.md](./05-risk-scoring-engine.md) | Moteur IA : collecte de données, preprocessing, entraînement et inférence |
| [06-api-backend.md](./06-api-backend.md) | API REST FastAPI : endpoints, middlewares, sécurité et routage |
| [07-infrastructure.md](./07-infrastructure.md) | Infrastructure : Ansible, K3s, manifests Kubernetes et monitoring |
| [08-security-dashboard.md](./08-security-dashboard.md) | Dashboard React : pages, composants et intégrations externes |

---

## 4. Concepts clés

### DefectDojo — Plateforme de gestion des vulnérabilités

DefectDojo est le point de centralisation de toutes les données de sécurité du projet. Il reçoit les résultats des trois types de scanners (SAST, SCA, DAST), les déduplique, les normalise et les expose via une API REST. Il permet également de générer des rapports de conformité et de suivre l'évolution des vulnérabilités dans le temps.

Documentation détaillée : [01-architecture.md](./01-architecture.md) et [03-securite.md](./03-securite.md)

---

### Risk Scoring Engine — Moteur de scoring IA

Le Risk Scoring Engine enrichit chaque vulnérabilité avec un score de risque contextuel compris entre 0 et 10. Il analyse les scores CVSS et EPSS, identifie les tags contextuels (production, exposition externe, données sensibles) et utilise un modèle supervisé par stacking (RandomForest + XGBoost + LightGBM). Les explications textuelles sont générées via l'API Gemini.

Architecture interne :

- Data Pipeline : Fetch → Preprocess → Feature Engineering
- Training Pipeline : Collect → Train → Validate → Deploy
- Inference Pipeline : Score → Explain → Cache

Documentation détaillée : [05-risk-scoring-engine.md](./05-risk-scoring-engine.md)

---

### API Backend — Serveur FastAPI

L'API Backend expose les données de sécurité et la logique métier de la plateforme. Elle est sécurisée par authentification JWT avec rate limiting par utilisateur et validation des inputs.

Endpoints principaux :

| Endpoint | Description |
|----------|-------------|
| `GET /api/findings` | Récupère les vulnérabilités avec leurs scores |
| `POST /api/predict` | Prédiction de risque en temps réel |
| `GET /api/explain/{finding_id}` | Explication LLM d'une vulnérabilité |
| `POST /api/jira/create-ticket` | Création automatique d'un ticket Jira |
| `GET /api/stats` | Statistiques et métriques du modèle |

Documentation détaillée : [06-api-backend.md](./06-api-backend.md)

---

### Dashboard React — Interface utilisateur

Le Dashboard offre une visualisation interactive des vulnérabilités avec filtrage multi-critères, export CSV/PDF, notifications en temps réel et mode sombre/clair.

Pages principales :

| Page | Description |
|------|-------------|
| Dashboard | Vue d'ensemble des findings et statistiques globales |
| Findings | Liste filtrable, triable et paginée des vulnérabilités |
| Risk Analysis | Détails d'un finding avec score IA et explication |
| Remediation | Suggestions de correctifs automatiques |
| Jira Integration | Création et synchronisation des tickets Jira |
| Model Performance | Métriques et performance du modèle IA |

Documentation détaillée : [08-security-dashboard.md](./08-security-dashboard.md)

---

### Infrastructure — Provisionnement automatisé

L'infrastructure repose sur Kubernetes (K3s) pour l'orchestration et Ansible pour le provisionnement en tant que code (IaC). La supervision est assurée par Prometheus pour les métriques, Grafana pour la visualisation et Loki pour la centralisation des logs.

Documentation détaillée : [07-infrastructure.md](./07-infrastructure.md)

---

## 5. Guide de démarrage rapide

### Prérequis

| Catégorie | Exigence |
|-----------|----------|
| Système | Ubuntu 20.04 ou équivalent |
| Outils | Docker, Ansible, Python 3.8+, Node.js 18+, kubectl |
| Ressources | Minimum 8 Go RAM, 4 CPUs, 50 Go de stockage |

### Étapes d'installation

**Étape 1 — Cloner le dépôt**

    git clone https://github.com/votre-org/InvisiThreat.git
    cd InvisiThreat

**Étape 2 — Configurer les variables d'environnement**

    cp .env.example .env
    nano .env

**Étape 3 — Déployer l'infrastructure**

    ansible-playbook -i infrastructure/ansible/inventory.yml \
                      infrastructure/ansible/playbook-k3s-install.yml
    kubectl get pods -A -w

La finalisation prend environ 15 à 20 minutes.

**Étape 4 — Configurer les secrets GitHub**

Dans les paramètres du dépôt GitHub, ajouter les secrets suivants :

- `SNYK_TOKEN`
- `DEFECTDOJO_API_KEY`
- `GEMINI_API_KEY`
- `JIRA_API_TOKEN`

**Étape 5 — Lancer le pipeline CI/CD**

    git push origin main

Suivre les logs du workflow sur GitHub Actions.

**Étape 6 — Accéder au dashboard**

    kubectl port-forward -n invisithreat svc/dashboard 30080:80

Ouvrir ensuite http://localhost:30080 dans le navigateur.

### Vérification du déploiement

    # Vérifier tous les pods
    kubectl get pods -n invisithreat

    # Consulter les logs de l'API
    kubectl logs -n invisithreat deployment/api-backend -f

    # Accéder à DefectDojo
    kubectl port-forward -n invisithreat svc/defectdojo 8080:80
    # http://localhost:8080

    # Accéder à Grafana
    kubectl port-forward -n invisithreat svc/grafana 3000:80
    # http://localhost:3000 (admin / admin)

---

## 6. Flux de travail complet

Le flux de travail d'InvisiThreat suit une chaîne linéaire depuis le code source jusqu'au tableau de bord :

```
Code source (push)
        |
        v
GitHub Actions (Pipeline CI/CD)
        |
   +----+----+----+
   |    |    |    |
   v    v    v    v
Semgrep Snyk ZAP Infra
(SAST) (SCA) (DAST) Scan
   |    |    |    |
   +----+----+----+
        |
        v
   DefectDojo
(Centralisation des findings)
        |
        v
Risk Scoring Engine
  (Scores IA + Tags)
        |
        v
   API Backend
  (Exposition REST)
        |
        v
 Dashboard React
 (Visualisation)
```

---

## 7. Métriques et KPIs

| Métrique | Description | Cible |
|----------|-------------|-------|
| MTTR | Temps moyen de résolution des vulnérabilités | < 72 heures |
| Findings critiques non résolus | Nombre de vulnérabilités critiques ouvertes | 0 |
| Coverage | Couverture du code scanné | > 90 % |
| Model Accuracy | Précision du modèle de scoring IA | > 85 % |
| Pipeline Duration | Durée totale d'exécution du pipeline | < 30 minutes |
| False Positives Rate | Taux de faux positifs détectés | < 10 % |

---

## 8. Sécurité et bonnes pratiques

Les principes de sécurité suivants sont appliqués dans l'ensemble du projet :

| Principe | Implémentation |
|----------|---------------|
| Authentification | JWT tokens avec rotation automatique |
| Autorisation | RBAC (Role-Based Access Control) |
| Chiffrement | TLS 1.3 pour toutes les communications |
| Secrets | Gestion via GitHub Secrets |
| Audit | Logs complets de tous les accès et modifications |
| Isolation | Namespaces Kubernetes séparés par environnement |
| Conformité | OWASP Top 10, CWE/SANS Top 25, CVSS v3.1, EPSS |

---

## 9. Contributions et support

### Comment contribuer

1. Forker le dépôt
2. Créer une branche : `git checkout -b feature/ma-fonctionnalite`
3. Committer les modifications : `git commit -am 'Ajoute ma fonctionnalité'`
4. Pousser la branche : `git push origin feature/ma-fonctionnalite`
5. Créer une Pull Request avec une description détaillée

### Signaler un problème

Ouvrir une GitHub Issue en précisant :

- Description claire du problème
- Étapes pour reproduire
- Logs et messages d'erreur
- Résultat attendu vs résultat obtenu

### Contacts

- Email : support@invisithreat.dev
- GitHub Discussions : Ouvrir une discussion dans le dépôt

---

## 10. Ressources additionnelles

### Documentation externe

- [OWASP Juice Shop](https://owasp.org/www-project-juice-shop/)
- [DefectDojo Documentation](https://defectdojo.github.io/django-DefectDojo/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Ansible Documentation](https://docs.ansible.com/)

### Standards et certifications appliqués

| Standard | Description |
|----------|-------------|
| OWASP Top 10 | Référentiel des 10 risques les plus critiques |
| CWE / SANS Top 25 | Liste des faiblesses logicielles les plus dangereuses |
| CVSS v3.1 | Système de notation des vulnérabilités |
| EPSS | Score de probabilité d'exploitation |

---

## Checklist d'initialisation

- [ ] Cloner le dépôt
- [ ] Configurer les variables d'environnement
- [ ] Installer les prérequis (Docker, Ansible, kubectl)
- [ ] Exécuter le playbook Ansible
- [ ] Configurer les secrets GitHub
- [ ] Déclencher le premier pipeline
- [ ] Vérifier le dashboard
- [ ] Configurer Jira (optionnel)
- [ ] Lire la documentation technique complète

---

*Dernière mise à jour : juin 2026 — Version 1.0.0*