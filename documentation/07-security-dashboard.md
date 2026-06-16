# Security Dashboard — Interface Utilisateur React

Le **Security Dashboard** est l'interface web de la plateforme **InvisiThreat**.
Développée en **React 18 + TypeScript** avec **Vite**, elle permet de visualiser, filtrer et interagir avec les vulnérabilités enrichies par l'IA.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du frontend](#2-architecture-du-frontend)
3. [Gestion de l'état et des données](#3-gestion-de-létat-et-des-données)
4. [Services API](#4-services-api)
5. [Routage et protection des pages](#5-routage-et-protection-des-pages)
6. [Principales pages](#6-principales-pages)
7. [Composants clés](#7-composants-clés)
8. [Gestion des erreurs et feedback](#8-gestion-des-erreurs-et-feedback)
9. [Authentification et sécurité](#9-authentification-et-sécurité)
10. [Déploiement](#10-déploiement)
11. [Intégration avec le backend](#11-intégration-avec-le-backend)
12. [Bonnes pratiques](#12-bonnes-pratiques)
13. [Résumé des fonctionnalités](#13-résumé-des-fonctionnalités)

---

## 1. Vue d'ensemble

Le dashboard offre les fonctionnalités suivantes :

- **Consultation des vulnérabilités** : liste complète avec scores IA, filtres par produit et engagement, recherche et tri multicritères
- **Détail d'une vulnérabilité** : métadonnées DefectDojo, score IA détaillé, explications et recommandations générées par LLM
- **Correction automatique** : suggestion de correctif de code (SAST) et création d'une Pull Request GitHub
- **Intégration Jira** : création de tickets directement depuis l'interface
- **Authentification** : gestion des utilisateurs avec rôles (admin, developer, analyst) et token JWT
- **Tableaux de bord analytiques** : métriques du modèle IA, répartition des risques, tendances temporelles
- **Notifications** : alertes en temps réel via polling

L'application suit une architecture modulaire avec une séparation claire entre la couche de services API, la gestion d'état et les composants d'interface.

---

## 2. Architecture du frontend

### Structure des fichiers

L'application est organisée selon une architecture en couches :

```
security-dashboard/
├── public/
├── src/
│   ├── api/              # Clients HTTP et services par domaine
│   ├── auth/             # Contexte, guards, hooks et stockage JWT
│   ├── components/       # Composants UI réutilisables et layout
│   ├── features/         # Pages métier (findings, analytics, admin…)
│   ├── routes/           # Configuration du routage et ProtectedRoute
│   ├── store/            # État global via Zustand
│   ├── types/            # Types TypeScript partagés
│   └── utils/            # Formateurs, validateurs, constantes
├── Dockerfile
├── nginx-spa.conf
├── vite.config.ts
└── tailwind.config.js
```

### Architecture en couches

Les données circulent de bas en haut selon le flux suivant :

```
Backend FastAPI (:8081)
        │
        ▼
API Services (Axios)
        │
        ▼
Store (Zustand — État global)
        │
        ▼
Features (Pages métier)
        │
        ▼
UI Components (Composants React)
```

---

## 3. Gestion de l'état et des données

### 3.1 Store global — Zustand

**Zustand** gère l'état global de manière légère et performante. Il couvre principalement l'authentification, les notifications et les préférences d'interface. Les avantages retenus sont l'absence de prop drilling, des mises à jour granulaires par souscription, et une persistance simple via `localStorage`.

### 3.2 Cache et requêtes — React Query

**React Query** (`@tanstack/react-query`) prend en charge toutes les données asynchrones issues du backend. Il assure la mise en cache automatique (5 minutes de fraîcheur, 10 minutes avant garbage collection), la déduplication des requêtes parallèles, le rechargement en arrière-plan et la gestion centralisée des états `loading`, `error` et `success`.

### 3.3 Token JWT

Le token JWT est stocké dans `localStorage` et injecté automatiquement dans chaque requête HTTP via un intercepteur Axios. En cas d'expiration (réponse 401), le token est supprimé et l'utilisateur est redirigé vers la page de connexion.

---

## 4. Services API

Tous les appels vers le backend FastAPI (port 8081) sont centralisés dans le dossier `src/api/services/`, organisés par domaine fonctionnel.

| Service         | Fichier            | Endpoints clés                                              |
|-----------------|--------------------|-------------------------------------------------------------|
| Findings        | `findings.ts`      | `GET /defectdojo/findings`, `GET /defectdojo/findings/{id}` |
| Products        | `products.ts`      | `GET /defectdojo/products`                                  |
| Engagements     | `engagements.ts`   | `GET /defectdojo/engagements`                               |
| Predictions     | `predictions.ts`   | `POST /predict`, `POST /predict/batch`                      |
| Explanations    | `explanations.ts`  | `POST /defectdojo/findings/{id}/explain`                    |
| Recommandations | `explanations.ts`  | `POST /defectdojo/findings/{id}/recommend`                  |
| Solutions       | `explanations.ts`  | `POST /defectdojo/findings/{id}/solution`                   |
| Jira            | `jira.ts`          | `POST /defectdojo/findings/{id}/create-jira-issue`          |
| Authentication  | `auth.ts`          | `POST /auth/login`, `POST /auth/register`, `GET /auth/me`   |
| Notifications   | `notifications.ts` | `GET /notifications`, `POST /notifications/mark-read`       |

Le client Axios est configuré avec un timeout de 30 secondes et une base URL lue depuis la variable d'environnement `VITE_API_BASE_URL`.

---

## 5. Routage et protection des pages

### 5.1 Table de routage

| Chemin          | Page              | Accès  | Rôle requis |
|-----------------|-------------------|--------|-------------|
| `/login`        | LoginPage         | Public | —           |
| `/register`     | RegisterPage      | Public | —           |
| `/`             | Redirect findings | Privé  | User        |
| `/findings`     | FindingsPage      | Privé  | User        |
| `/findings/:id` | FindingDetailPage | Privé  | User        |
| `/products`     | ProductsPage      | Privé  | User        |
| `/engagements`  | EngagementsPage   | Privé  | User        |
| `/analytics`    | AnalyticsPage     | Privé  | Admin       |
| `/admin/users`  | UsersPage         | Privé  | Admin       |
| `/profile`      | ProfilePage       | Privé  | User        |
| `/model`        | ModelMetricsPage  | Privé  | User        |

### 5.2 Protection des routes

Toutes les routes privées sont enveloppées dans un composant `ProtectedRoute` qui vérifie l'état d'authentification depuis le store Zustand. Si l'utilisateur n'est pas connecté, il est redirigé automatiquement vers `/login`. Les routes réservées aux administrateurs font l'objet d'une vérification supplémentaire du rôle.

---

## 6. Principales pages

### 6.1 FindingsPage — Liste des vulnérabilités

Page centrale de l'application. Elle affiche un tableau dynamique des findings issus d'un engagement sélectionné.

**Fonctionnalités disponibles**

- Sélection de l'engagement via un dropdown
- Tableau paginé localement (20 éléments par page)
- Filtres par sévérité : `Critical`, `High`, `Medium`, `Low`
- Recherche textuelle sur le titre, le chemin, le tag ou l'identifiant
- Tri multicritères : score IA, CVSS, âge, sévérité
- Export en CSV et PDF
- Statistiques résumées : total des findings, nombre de critiques, score moyen

**Flux de récupération des données**

1. Chargement de la liste des engagements disponibles
2. Sélection d'un engagement par l'utilisateur
3. Récupération des findings de cet engagement
4. Appel batch des scores IA
5. Affichage du tableau enrichi

### 6.2 FindingDetailPage — Détail d'une vulnérabilité

Page d'analyse complète d'un finding. Elle est organisée en sections successives.

```
┌─────────────────────────────────────────────┐
│  En-tête : titre, sévérité, score IA         │
├─────────────────────────────────────────────┤
│  Métadonnées : produit, fichier, ligne       │
├─────────────────────────────────────────────┤
│  Score IA + barre de confiance               │
├─────────────────────────────────────────────┤
│  SHAP : top features influençant le score    │
├─────────────────────────────────────────────┤
│  Explication LLM : résumé, impact, cause     │
├─────────────────────────────────────────────┤
│  Recommandations : étapes, CVE/CWE           │
├─────────────────────────────────────────────┤
│  Solution & Autofix : code diff + PR         │
├─────────────────────────────────────────────┤
│  Jira : ticket existant ou création          │
└─────────────────────────────────────────────┘
```

**Description des sections**

- **En-tête** : titre du finding, sévérité, niveau de risque IA, badge de confiance et statut
- **Métadonnées** : produit, engagement, fichier source, numéro de ligne, dates de découverte et de dernière modification
- **Score IA** : valeur continue (0–10), classe de risque (`LOW` / `MEDIUM` / `HIGH` / `CRITICAL`), distribution des probabilités et niveau de confiance
- **SHAP** : top 5 des features ayant influencé le score, avec leur magnitude respective (affiché si disponible)
- **Explication LLM** : résumé synthétique, impact potentiel, cause racine identifiée, difficulté d'exploitation estimée
- **Recommandations LLM** : étapes de remédiation numérotées, références CVE et CWE associées, critères de vérification de la correction
- **Solution & Autofix** : comparaison code vulnérable / code corrigé, bouton de création de Pull Request GitHub
- **Jira** : affichage des détails du ticket existant ou bouton de création d'un nouveau ticket

### 6.3 AnalyticsPage — Tableaux de bord IA

Accessible aux administrateurs uniquement. Regroupe l'ensemble des métriques de performance du modèle IA.

**Graphiques disponibles**

- Répartition des vulnérabilités par niveau de risque IA (camembert)
- Évolution des scores dans le temps (courbe temporelle)
- Distribution CVSS versus score IA (nuage de points)
- Matrice de confusion, F1-score et balanced accuracy du modèle
- Nombre de prédictions générées par jour

### 6.4 UsersPage — Gestion des utilisateurs

Réservée aux administrateurs. Permet de superviser l'ensemble des comptes de la plateforme.

**Actions disponibles**

- Lister tous les utilisateurs avec leur statut (actif, en attente, bloqué)
- Valider ou rejeter les inscriptions en attente
- Modifier le rôle d'un utilisateur (`developer`, `analyst`, `admin`)
- Bloquer ou débloquer un compte

---

## 7. Composants clés

### 7.1 SolutionAndAutofix

Composant intégré dans `FindingDetailPage`. Il gère le cycle complet de correction automatique d'une vulnérabilité SAST.

**Flux de fonctionnement**

1. Vérification de l'éligibilité au correctif automatique
2. Affichage de la comparaison code vulnérable / code corrigé
3. Déclenchement de la création de PR via le bouton dédié
4. Le backend crée une branche, effectue le commit et ouvre la PR sur GitHub
5. Le frontend affiche le lien vers la PR créée

**États du composant**

| État           | Description                              |
|----------------|------------------------------------------|
| `no-source`    | Aucun fichier source trouvé              |
| `invalid-line` | Numéro de ligne invalide ou introuvable  |
| `no-github`    | Dépôt GitHub non configuré              |
| `eligible`     | Autofix possible                         |
| `loading`      | Génération du correctif en cours         |
| `success`      | Pull Request créée avec succès           |
| `error`        | Erreur lors de la création               |

### 7.2 Layout

Fournit la structure visuelle commune à toutes les pages privées de l'application.

- **Navbar** : logo et titre, navigation principale, cloche de notifications avec badge, menu utilisateur (profil, déconnexion)
- **Sidebar** : navigation rapide vers les pages principales, repliable sur mobile
- **Main Content** : zone de rendu des pages, responsive via Tailwind CSS

### 7.3 Composants UI réutilisables

Ensemble de composants stylisés disponibles pour toute l'interface.

| Composant      | Utilisation                                     |
|----------------|-------------------------------------------------|
| `Button`       | Actions générales (primary, secondary, danger)  |
| `Card`         | Conteneurs de sections de contenu               |
| `Modal`        | Dialogues de confirmation et formulaires        |
| `Badge`        | Étiquettes colorées par sévérité ou statut      |
| `Spinner`      | Indicateur de chargement                        |
| `SeverityChip` | Affichage de la sévérité avec couleur et icône  |
| `ProgressBar`  | Visualisation de la confiance (0–100 %)         |
| `Table`        | Tableaux paginés avec tri                       |
| `Select`       | Dropdown personnalisé                           |
| `Input`        | Champ texte avec validation intégrée            |

---

## 8. Gestion des erreurs et feedback

### 8.1 États de chargement

Pendant tout appel API (LLM, Jira, prédictions batch), l'interface affiche un indicateur de chargement. En cas d'erreur, un message explicite est présenté à l'utilisateur. Une fois les données disponibles, le contenu s'affiche à la place des indicateurs.

### 8.2 Messages d'erreur contextuels

| Situation                     | Message affiché à l'utilisateur                               |
|-------------------------------|---------------------------------------------------------------|
| Timeout LLM                   | Invitation à réessayer ultérieurement                        |
| Service Jira indisponible     | Indication d'indisponibilité temporaire                      |
| Ticket Jira déjà existant     | Notification du conflit avec le ticket existant              |
| Permissions insuffisantes     | Message d'accès refusé avec rôle requis                      |

### 8.3 Indicateur de cache

Lorsque la réponse LLM provient du cache Redis, un indicateur visuel le signale à l'utilisateur pour lui indiquer que la réponse n'est pas générée en temps réel.

### 8.4 Fallback gracieux

Si le LLM échoue complètement, le backend renvoie une réponse statique de secours. Le frontend affiche une alerte informative indiquant que l'explication provient du modèle par défaut.

---

## 9. Authentification et sécurité

### 9.1 Flux d'authentification

1. L'utilisateur soumet ses identifiants via la page de connexion ou d'inscription
2. Le frontend envoie une requête `POST /auth/login` avec l'email et le mot de passe
3. Le backend valide les identifiants et génère un token JWT signé
4. Le frontend stocke le token dans `localStorage`
5. Le token est injecté automatiquement dans l'en-tête `Authorization` de toutes les requêtes suivantes

### 9.2 Expiration du token

Lorsque le backend retourne une réponse 401, le frontend supprime le token, réinitialise l'état d'authentification dans le store et redirige l'utilisateur vers la page de connexion.

### 9.3 Protection des routes

Chaque route privée est gardée par le composant `ProtectedRoute`. Si `isAuthenticated` est `false`, l'utilisateur est redirigé vers `/login`. Les pages réservées aux administrateurs font l'objet d'une vérification supplémentaire du rôle.

### 9.4 HTTPS obligatoire en production

Les tokens JWT transitent dans les en-têtes HTTP. En production, toute communication doit être chiffrée via HTTPS. Toute communication sur HTTP non sécurisé doit être rejetée au niveau du serveur.

---

## 10. Déploiement

### 10.1 Développement local

L'application se lance avec `npm run dev` et est accessible sur le port 5173. Un proxy Vite redirige automatiquement les appels API vers le backend sur le port 8081, évitant les problèmes CORS en développement.

### 10.2 Build production

La commande `npm run build` compile le TypeScript et bundle l'application. Le dossier `dist/` généré contient les assets statiques prêts à être servis.

### 10.3 Docker — Build multi-étapes

L'image Docker repose sur un build multi-étapes :

- **Stage 1 (builder)** : image Node.js Alpine, installation des dépendances et compilation de l'application
- **Stage 2 (runtime)** : image Nginx Alpine servant les fichiers statiques du dossier `dist/`

### 10.4 Configuration nginx

Nginx est configuré pour le mode Single Page Application : toutes les routes inconnues retournent `index.html` afin que React Router puisse les gérer côté client. Les assets statiques (JS, CSS, polices, images) sont mis en cache pendant un an avec l'en-tête `Cache-Control: public, immutable`.

### 10.5 Pipeline CI/CD

Le pipeline GitHub Actions construit l'image Docker, la pousse vers le registre local, puis déploie les manifests Kubernetes via `kubectl apply`. Le déploiement est déclenché automatiquement à chaque merge sur la branche principale.

---

## 11. Intégration avec le backend

Le dashboard communique exclusivement avec l'API FastAPI exposée sur le port 8081.

### Endpoints consommés

| Domaine    | Endpoints                                                                  |
|------------|----------------------------------------------------------------------------|
| Données    | `/defectdojo/findings`, `/defectdojo/products`, `/defectdojo/engagements`  |
| Scoring    | `/predict/batch`                                                           |
| LLM        | `/defectdojo/findings/{id}/explain`, `/recommend`, `/solution`, `/autofix` |
| Jira       | `/defectdojo/findings/{id}/create-jira-issue`                              |
| Auth       | `/auth/login`, `/auth/register`, `/auth/me`                                |
| Notifs     | `/notifications`, `/notifications/mark-read`                               |

### Flux de communication

```
React (port 5173)
        │
        │  proxy Vite en développement
        │  nginx en production
        │
        ▼
FastAPI Backend (port 8081)
        │
        ├── DefectDojo API  (port 8080)
        ├── PostgreSQL      (port 5432)
        ├── Redis           (port 6379)
        ├── LLM Service     (Ollama / OpenAI / Anthropic)
        └── Jira API
```

---

## 12. Bonnes pratiques

### 12.1 TypeScript strict

L'ensemble du code est écrit en TypeScript avec le mode strict activé : `noImplicitAny`, `strictNullChecks` et `strictFunctionTypes` sont tous activés. Cela garantit la cohérence des types à travers toute l'application et réduit les erreurs à l'exécution.

### 12.2 Séparation des responsabilités

| Couche        | Responsabilité                         |
|---------------|----------------------------------------|
| Composants    | Rendu visuel uniquement                |
| Hooks         | Logique métier réutilisable            |
| Services      | Appels API et transformation des données |
| Store         | État global partagé entre composants  |

### 12.3 Optimisation réseau

React Query gère automatiquement le cache, la déduplication des requêtes parallèles et la revalidation en arrière-plan. Les routes peu fréquentées (Analytics, Admin) sont chargées en lazy loading via `React.lazy` pour réduire le bundle initial.

### 12.4 Responsive design

Tailwind CSS assure l'adaptation à tous les formats d'écran. Les grilles passent automatiquement d'une colonne sur mobile à plusieurs colonnes sur tablette et desktop, sans CSS personnalisé.

### 12.5 Accessibilité

- Attributs `aria-*` présents sur tous les éléments interactifs
- Contraste des couleurs conforme au niveau WCAG AA
- Navigation complète au clavier sur l'ensemble des composants

---

## 13. Résumé des fonctionnalités

| Fonctionnalité              | Description                                           | Page                  |
|-----------------------------|-------------------------------------------------------|-----------------------|
| Liste des vulnérabilités    | Vue tabulaire avec filtres, tri et export             | `FindingsPage`        |
| Détail d'une vulnérabilité  | Métadonnées complètes, score IA, analyse LLM          | `FindingDetailPage`   |
| Correction automatique      | Suggestion de correctif et création de PR GitHub      | `SolutionAndAutofix`  |
| Création de ticket Jira     | Création directe depuis le dashboard                  | `FindingDetailPage`   |
| Tableaux de bord IA         | Métriques du modèle et tendances                      | `AnalyticsPage`       |
| Gestion des utilisateurs    | Validation des comptes et attribution de rôles        | `UsersPage`           |
| Authentification            | Login et Register avec token JWT                      | `LoginPage`, `RegisterPage` |
| Notifications               | Alertes en temps réel via polling                     | `NotificationBell`    |