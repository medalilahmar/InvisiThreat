# 🔐 SecureVault API
## Enterprise Password & Secrets Management Platform

> **⚠️ AVERTISSEMENT** : Ce projet est **intentionnellement vulnérable**. Il est conçu exclusivement pour des démonstrations de pipelines DevSecOps (SAST, DAST, SCA). **Ne jamais déployer en production.**

---

## 📋 Table des matières

1. [Présentation du projet](#présentation)
2. [Architecture & Structure](#architecture)
3. [Prérequis](#prérequis)
4. [Installation](#installation)
5. [Lancement de l'application](#lancement)
6. [Vulnérabilités incluses](#vulnérabilités)
7. [Intégration dans le pipeline](#pipeline)
8. [Résultats attendus par outil](#résultats-attendus)

---

## 📌 Présentation

**SecureVault API** est une API REST simulant une plateforme de gestion de mots de passe et secrets d'entreprise. Elle est construite avec **Node.js / Express** et contient volontairement des vulnérabilités de sécurité couvrant les catégories **OWASP Top 10**, permettant de valider un pipeline DevSecOps complet.

| Attribut       | Détail                          |
|----------------|---------------------------------|
| Langage        | JavaScript (Node.js 16+)        |
| Framework      | Express 4.17.1                  |
| Base de données| MySQL (simulé en mémoire)        |
| Port par défaut| 3000                            |
| Taille projet  | ~15 fichiers / ~800 lignes      |
| Scan estimé    | < 4 minutes (SAST + SCA + DAST) |

---

## 🏗️ Architecture & Structure

```
SecureVault-API/
├── src/
│   ├── app.js                    # Point d'entrée principal
│   ├── config/
│   │   └── config.js             # Configuration (secrets hardcodés)
│   ├── controllers/
│   │   ├── authController.js     # Login, register, mot de passe
│   │   ├── vaultController.js    # Gestion des secrets
│   │   ├── adminController.js    # Panel administration
│   │   └── userController.js     # Profils utilisateurs
│   ├── middlewares/
│   │   └── authMiddleware.js     # Vérification JWT
│   ├── models/                   # Modèles de données
│   ├── routes/
│   │   ├── authRoutes.js
│   │   ├── vaultRoutes.js
│   │   ├── adminRoutes.js
│   │   └── userRoutes.js
│   └── utils/
│       └── database.js           # Couche d'accès DB
├── public/
│   └── index.html                # Interface de connexion
├── views/                        # Templates EJS
├── tests/                        # Tests unitaires
├── docs/                         # Documentation
├── .env                          # Variables d'environnement (secrets exposés)
├── .gitignore
└── package.json                  # Dépendances vulnérables
```

---

## ✅ Prérequis

```bash
node --version    # >= 16.x
npm --version     # >= 7.x
```

---

## 🚀 Installation

```bash
# 1. Cloner ou extraire le projet
cd SecureVault-API

# 2. Installer les dépendances
npm install

# Durée estimée : 20-40 secondes
```

---

## ▶️ Lancement de l'application

```bash
# Démarrage standard
npm start

# L'application démarre en ~2-3 secondes
# Accès : http://localhost:3000
```

**Endpoints disponibles pour le DAST :**

| Méthode | Endpoint                    | Description                    |
|---------|-----------------------------|--------------------------------|
| POST    | /api/auth/login             | Authentification               |
| POST    | /api/auth/register          | Inscription                    |
| POST    | /api/auth/forgot-password   | Réinitialisation mot de passe  |
| GET     | /api/vault/secret/:id       | Récupérer un secret (IDOR)     |
| GET     | /api/vault/export           | Export vault (Cmd Injection)   |
| POST    | /api/vault/import           | Import vault (Deserialization) |
| GET     | /api/vault/backup/download  | Téléchargement (Path Traversal)|
| POST    | /api/vault/fetch-external   | Fetch URL externe (SSRF)       |
| GET     | /api/admin/debug            | Infos debug (sans auth!)       |
| GET     | /api/admin/diagnostic       | Diagnostic (Cmd Injection)     |
| GET     | /api/users/search           | Recherche (XSS réfléchi)       |

---

## 🐛 Vulnérabilités incluses

### 🔴 CRITIQUE

| ID  | Type                        | Fichier                        | Description                              |
|-----|-----------------------------|--------------------------------|------------------------------------------|
| V01 | SQL Injection               | authController.js:14           | Query construite par concat. de strings  |
| V02 | Hardcoded Secrets           | config/config.js               | Clés AWS, JWT, DB en clair               |
| V03 | Command Injection           | vaultController.js:57          | `exec()` avec input utilisateur          |
| V04 | Insecure Deserialization    | vaultController.js:68          | `node-serialize` sur données utilisateur |
| V05 | Privilege Escalation        | authController.js:46           | `role` contrôlé par l'utilisateur        |
| V06 | Broken Access Control       | adminRoutes.js:8               | `/debug` sans aucune authentification    |
| V07 | Secrets in .env committed   | .env                           | Clés AWS/Stripe dans le repo git         |

### 🟠 ÉLEVÉ

| ID  | Type                        | Fichier                        | Description                              |
|-----|-----------------------------|--------------------------------|------------------------------------------|
| V08 | XSS Stocké/Réfléchi         | vaultController.js / userCtrl  | Contenu HTML non sanitisé                |
| V09 | Path Traversal              | vaultController.js:78          | Chemin fichier non validé                |
| V10 | SSRF                        | vaultController.js:91          | URL externe non validée                  |
| V11 | Weak Cryptography           | vaultController.js / authCtrl  | DES + MD5 pour chiffrement/hash          |
| V12 | Weak JWT Config             | authController.js / middleware | Secret faible, algo non forcé            |
| V13 | IDOR                        | vaultController.js:14          | Pas de vérification de propriété         |
| V14 | XXE                         | adminController.js:24          | Parser XML sans restriction d'entités    |
| V15 | Sensitive Data in Response  | authController.js:35           | Hash mot de passe retourné dans response |
| V16 | Session Misconfiguration    | app.js:30                      | Cookie sans Secure/HttpOnly              |

### 🟡 MOYEN

| ID  | Type                        | Fichier                        | Description                              |
|-----|-----------------------------|--------------------------------|------------------------------------------|
| V17 | CORS Wildcard               | app.js:18                      | Toutes origines autorisées               |
| V18 | Stack Trace Exposed         | app.js:52                      | Détails d'erreur dans la réponse         |
| V19 | Open Redirect               | userController.js:37           | Paramètre `redirect` non validé          |
| V20 | No CSRF Protection          | adminController.js:40          | Pas de token CSRF sur actions critiques  |
| V21 | Sensitive Data in JWT       | authController.js:28           | Email et ID interne dans le payload      |
| V22 | JWT Storage in localStorage | public/index.html              | Token XSS-accessible                     |

### 🔵 FAIBLE / INFO

| ID  | Type                        | Fichier                        | Description                              |
|-----|-----------------------------|--------------------------------|------------------------------------------|
| V23 | Version Disclosure          | app.js:45                      | Header X-Powered-By avec version         |
| V24 | Username Enumeration        | userController.js:44           | Messages d'erreur différents             |
| V25 | Sensitive Data in Logs      | authController.js:17           | Requêtes SQL loggées (avec mots de passe)|
| V26 | No Rate Limiting            | authRoutes.js                  | Brute force possible sur /login          |

---

## 🔄 Intégration dans le pipeline

### SAST — Analyse statique du code source
```bash
# Semgrep (recommandé)
semgrep --config=p/nodejs --config=p/security-audit .

# ESLint Security
npx eslint --plugin security src/

# SonarQube
sonar-scanner -Dsonar.projectKey=securevault
```

### SCA — Analyse des dépendances
```bash
# npm audit natif (rapide, ~10s)
npm audit --json

# Snyk
snyk test

# OWASP Dependency-Check
dependency-check --project SecureVault --scan .
```

**Dépendances volontairement vulnérables :**

| Package            | Version | CVE(s) notables                    | Sévérité |
|--------------------|---------|-------------------------------------|----------|
| lodash             | 4.17.4  | CVE-2019-10744 (Prototype Pollution)| Critical |
| marked             | 0.3.6   | CVE-2022-21681 (ReDoS)              | High     |
| express            | 4.17.1  | Plusieurs CVEs connues              | Medium   |
| node-serialize     | 0.0.4   | CVE-2017-5941 (RCE)                 | Critical |
| serialize-javascript| 1.7.0  | CVE-2020-7660 (XSS)                 | High     |
| axios              | 0.19.0  | CVE-2020-28168 (SSRF)               | Medium   |
| xml2js             | 0.4.19  | CVE-2023-0842 (XXE)                 | Medium   |
| express-session    | 1.15.6  | Fixation de session                 | Medium   |
| mongoose           | 5.9.7   | NoSQL Injection                     | High     |

### DAST — Analyse dynamique (application en cours d'exécution)
```bash
# 1. Démarrer l'application
npm start &

# 2. OWASP ZAP (scan de base rapide)
zap-baseline.py -t http://localhost:3000

# 3. Nuclei
nuclei -u http://localhost:3000 -t vulnerabilities/

# 4. Nikto
nikto -h http://localhost:3000
```

---

## 📊 Résultats attendus par outil

| Outil         | Type | Vulnérabilités attendues        | Temps estimé |
|---------------|------|---------------------------------|--------------|
| Semgrep       | SAST | 15-20 findings (Critical/High)  | ~30s         |
| npm audit     | SCA  | 8-12 CVEs                       | ~10s         |
| Snyk          | SCA  | 10-15 issues avec remediation   | ~20s         |
| OWASP ZAP     | DAST | 8-12 alertes (XSS, SQLi, etc.)  | ~90s         |
| Nikto         | DAST | 5-10 findings                   | ~60s         |

**Temps total pipeline estimé : 3 à 4 minutes ✅**
