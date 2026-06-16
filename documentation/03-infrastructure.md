# Infrastructure — InvisiThreat

> Déploiement et orchestration de l'ensemble des composants InvisiThreat sur un cluster Kubernetes léger (K3s) et DefectDojo, entièrement provisionnés et gérés via Ansible.

---

## Table des matières

1. [Vue d'ensemble de l'infrastructure](#1-vue-densemble-de-linfrastructure)
2. [K3s — Cluster Kubernetes léger](#2-k3s--cluster-kubernetes-léger)
3. [Provisionnement Ansible — K3s et services](#3-provisionnement-ansible--k3s-et-services)
4. [Provisionnement Ansible — DefectDojo](#4-provisionnement-ansible--defectdojo)
5. [Architecture des manifests Kubernetes](#5-architecture-des-manifests-kubernetes)
6. [Configuration centralisée](#6-configuration-centralisée)
7. [Accès à DefectDojo depuis le cluster](#7-accès-à-defectdojo-depuis-le-cluster)
8. [Déploiements applicatifs](#8-déploiements-applicatifs)
9. [Stockage persistant](#9-stockage-persistant)
10. [Registre Docker local](#10-registre-docker-local)
11. [Monitoring — Prometheus, Grafana, Loki](#11-monitoring--prometheus-grafana-loki)
12. [Intégration avec le pipeline CI/CD](#12-intégration-avec-le-pipeline-cicd)
13. [Dépannage](#13-dépannage)

---

## 1. Vue d'ensemble de l'infrastructure

L'infrastructure d'InvisiThreat repose sur deux niveaux complémentaires : DefectDojo, déployé via Docker Compose sur l'hôte, et un cluster Kubernetes mono-nœud (K3s) qui héberge l'ensemble des services applicatifs. Les deux niveaux sont entièrement provisionnés par des playbooks Ansible, ce qui garantit une infrastructure reproductible, idempotente et intégrable dans le pipeline CI/CD.

### Schéma général

```
Serveur hôte (Ubuntu)
│
├── DefectDojo (Docker Compose, hors cluster)
│       └── Géré par Ansible (rôle defectdojo + playbook ensure-defectdojo)
│       └── Accessible via Service ExternalName depuis les pods K3s
│
└── K3s (Cluster Kubernetes)
        │
        ├── Namespace : invisithreat
        │       ├── Risk Scoring Engine (API FastAPI + modèle ML)
        │       ├── Dashboard (React)
        │       ├── PostgreSQL
        │       ├── Redis
        │       └── PersistentVolumeClaims (modèles, données, base)
        │
        ├── Namespace : monitoring
        │       ├── Prometheus
        │       ├── Grafana
        │       ├── Loki
        │       ├── Promtail (DaemonSet)
        │       ├── Node Exporter (DaemonSet)
        │       ├── PostgreSQL Exporter
        │       └── Kube State Metrics
        │
        └── Registre Docker local (port 5000)
```

### Principes de conception

**Idempotence** : tous les playbooks Ansible et manifests Kubernetes peuvent être exécutés plusieurs fois sans effet de bord. Si un composant est déjà installé dans la bonne configuration, il n'est pas réinstallé.

**Reproductibilité** : l'ensemble de l'infrastructure peut être recréé de zéro sur un nouveau serveur en déclenchant simplement le pipeline CI/CD, sans aucune intervention manuelle au-delà de la configuration initiale du runner et des secrets GitHub.

**Séparation des responsabilités** : les paramètres non sensibles sont centralisés dans une ConfigMap Kubernetes, les secrets sont injectés depuis GitHub Secrets, et les données persistantes sont stockées dans des volumes dédiés.

---

## 2. K3s — Cluster Kubernetes léger

### Pourquoi K3s

K3s est une distribution Kubernetes certifiée, optimisée pour les environnements à ressources limitées. Contrairement à un cluster Kubernetes standard, K3s fonctionne dans environ 500 Mo de RAM, ce qui le rend adapté à un serveur de développement ou un environnement de démonstration.

| Caractéristique | K3s | Kubernetes standard |
|----------------|-----|---------------------|
| Empreinte mémoire | ~500 Mo | ~2 Go |
| Installation | Script one-liner | Procédure multi-étapes |
| CoreDNS | Inclus | À installer séparément |
| Load balancer interne | Inclus (servicelb) | À installer séparément |
| Provisionneur de volumes | Inclus (local-path) | À installer séparément |
| Compatibilité manifests | Complète | Complète |

### Configuration spécifique à InvisiThreat

K3s est installé sans Traefik (l'Ingress Controller par défaut), car le projet utilise NGINX Ingress Controller à la place. Le kubeconfig est configuré avec des permissions accessibles à l'utilisateur courant. Le registre Docker local (`localhost:5000`) est déclaré comme registre de confiance dans K3s, permettant aux pods de télécharger les images construites par le pipeline sans passer par un registre externe.

---

## 3. Provisionnement Ansible — K3s et services

### Playbooks disponibles

L'infrastructure K3s est provisionnée par quatre playbooks Ansible indépendants, chacun responsable d'un périmètre précis. Ils sont tous exécutés par le pipeline CI/CD dans le bon ordre, mais peuvent également être lancés manuellement.

| Playbook | Rôle |
|----------|------|
| `playbook-k3s-install.yml` | Installation et configuration du cluster K3s |
| `playbook-metallb-install.yml` | Déploiement du load balancer MetalLB |
| `playbook-ingress-install.yml` | Déploiement du NGINX Ingress Controller |
| `playbook-monitoring-install.yml` | Déploiement de la stack Prometheus, Grafana, Loki |

### Playbook K3s — Étapes d'installation

**Vérification de présence** : avant toute installation, le playbook vérifie si K3s est déjà présent. Si c'est le cas, l'installation est ignorée pour respecter l'idempotence.

**Installation** : K3s est installé via son script officiel avec la désactivation de Traefik et l'écriture du kubeconfig avec des permissions accessibles.

**Démarrage du service** : le service K3s est activé au démarrage et lancé immédiatement. Le playbook attend que le nœud passe à l'état `Ready` avant de continuer.

**Configuration de kubectl** : le fichier kubeconfig généré par K3s est copié dans `~/.kube/config` pour permettre l'utilisation de `kubectl` sans paramètre supplémentaire.

**Création des répertoires persistants** : les répertoires de stockage sont créés sur l'hôte sous `/data/invisithreat/` :

| Répertoire | Contenu |
|-----------|---------|
| `/data/invisithreat/postgres` | Données PostgreSQL |
| `/data/invisithreat/models` | Modèles de machine learning |
| `/data/invisithreat/data/raw` | Données brutes depuis DefectDojo |
| `/data/invisithreat/data/processed` | Données nettoyées et enrichies |

**Configuration du registre local** : un fichier de configuration est écrit dans `/etc/rancher/k3s/registries.yaml` pour indiquer à K3s de faire confiance au registre local sur le port 5000, sans vérification TLS. K3s est ensuite redémarré pour prendre en compte cette configuration.

**Démarrage du registre Docker** : un conteneur `registry:2` est démarré sur le port 5000 via Docker, configuré pour redémarrer automatiquement.

**Validation** : le playbook vérifie que le nœud est en état `Ready` et affiche un résumé de l'installation.

### Playbook Monitoring — Composants déployés

Le playbook de monitoring déploie la stack d'observabilité complète dans le namespace `monitoring`. Chaque composant a un rôle précis dans la chaîne d'observation.

**Prometheus** collecte les métriques toutes les 15 secondes avec une rétention de 24 heures. Il découvre automatiquement les cibles à scrapper grâce aux annotations Kubernetes.

**Grafana** expose une interface de visualisation avec des dashboards pré-configurés pour les métriques applicatives, système et Kubernetes.

**Loki** agrège les logs de tous les pods du cluster, alimenté par Promtail déployé en DaemonSet sur chaque nœud.

**Node Exporter** expose les métriques système du serveur hôte (CPU, mémoire, disque, réseau) via un DaemonSet.

**PostgreSQL Exporter** expose les métriques internes de PostgreSQL vers Prometheus (connexions actives, taille des tables, temps de requête).

**Kube State Metrics** expose l'état des objets Kubernetes (pods, déploiements, PVC) sous forme de métriques Prometheus.

---

## 4. Provisionnement Ansible — DefectDojo

### Vue d'ensemble

Le déploiement et la maintenance de DefectDojo sont pilotés par deux éléments Ansible distincts selon le contexte d'utilisation :

| Composant | Fichier | Contexte d'utilisation |
|-----------|---------|------------------------|
| Playbook de vérification rapide | `ensure-defectdojo.yml` | Appelé à chaque exécution du pipeline CI/CD |
| Rôle complet | `roles/defectdojo` | Utilisé lors du setup initial d'une nouvelle machine |

DefectDojo s'appuie sur Docker Compose pour exécuter ses conteneurs, et sur un service systemd pour garantir le redémarrage automatique au boot du serveur.

---

### 4.1 Playbook `ensure-defectdojo.yml` — Vérification et démarrage rapide

Ce playbook est le point d'entrée du pipeline CI/CD. Il ne réalise pas une installation complète, mais garantit que l'API DefectDojo est accessible avant chaque exécution des scans.

#### Logique de fonctionnement

Le playbook suit un flux conditionnel en trois cas :

**Cas 1 — DefectDojo répond déjà** : une requête HTTP est envoyée sur l'endpoint `/api/v2/`. Si le service renvoie l'un des codes HTTP attendus (200, 301, 401, 403, etc.), le playbook se termine immédiatement sans aucune action. C'est le cas le plus courant lors des exécutions successives du pipeline.

**Cas 2 — Le dossier existe mais les conteneurs sont arrêtés** : DefectDojo a déjà été installé lors d'une exécution précédente, mais les conteneurs ont été stoppés (redémarrage du serveur, arrêt manuel). Le playbook relance les conteneurs via Docker Compose sans recloner le dépôt.

**Cas 3 — Le dossier n'existe pas** : première exécution sur la machine. Le playbook clone le dépôt officiel DefectDojo depuis GitHub, puis démarre les conteneurs. Il attend ensuite que l'API soit disponible avant de rendre la main au pipeline.

#### Paramètres de configuration

| Variable | Valeur par défaut | Description |
|----------|-------------------|-------------|
| `dojo_path` | `/home/user/django-DefectDojo` | Chemin local du dépôt DefectDojo |
| `api_url` | `http://localhost:8080/api/v2/` | Endpoint utilisé pour le healthcheck |
| `max_retries` | `30` | Nombre maximum de tentatives de connexion |
| `retry_delay` | `5` | Intervalle en secondes entre chaque tentative |
| `success_codes` | `200, 301, 302, 401, 403` | Codes HTTP indiquant que le service est vivant |

#### Idempotence garantie

L'idempotence est respectée à chaque niveau du playbook. Exécuter le playbook dix fois d'affilée dans le même état produit exactement le même résultat qu'une seule exécution, sans créer de doublons ni provoquer de redémarrages inutiles.

---

### 4.2 Rôle Ansible `defectdojo` — Installation complète

Le rôle `defectdojo` est utilisé pour provisionner entièrement un nouvel environnement. Il est appelé lors du setup initial d'une machine ou pour une reconfiguration complète, mais pas à chaque exécution du pipeline.

#### Structure du rôle

```
infrastructure/ansible/roles/defectdojo/
├── handlers/
│   └── main.yml          # Rechargement de systemd après modification du service
├── tasks/
│   └── main.yml          # Séquence complète d'installation
└── templates/
    └── defectdojo.service.j2   # Template du service systemd
```

#### Étapes d'installation

**Mise à jour du gestionnaire de paquets** : la liste des paquets disponibles est rafraîchie pour éviter les erreurs lors des installations suivantes.

**Installation des dépendances système** : les outils nécessaires sont installés — `ca-certificates` pour valider les certificats SSL, `curl` pour les tests HTTP, `gnupg` pour la vérification des signatures GPG, et `git` pour le clonage des dépôts.

**Installation de Docker** : la clé GPG officielle Docker est ajoutée au trousseau système, le dépôt stable Docker est configuré, puis Docker Engine, Docker CLI, containerd et le plugin Docker Compose v2 sont installés.

**Démarrage du service Docker** : Docker est activé et démarré via systemd, configuré pour démarrer automatiquement au reboot.

**Ajout de l'utilisateur au groupe Docker** : l'utilisateur Ansible est ajouté au groupe `docker` pour pouvoir exécuter les commandes Docker sans élévation de privilèges. À noter que cette appartenance offre des droits équivalents à `sudo` et doit être restreinte en production.

**Clonage du dépôt DefectDojo** : le dépôt officiel est cloné dans le répertoire défini par `dojo_path`. Si le dossier existe déjà, le clonage est ignoré.

**Création du service systemd** : un fichier de service systemd est généré depuis le template Jinja2 et déposé dans `/etc/systemd/system/defectdojo.service`. Systemd est rechargé pour prendre en compte ce nouveau service.

**Démarrage du service DefectDojo** : le service est activé et démarré. Il démarre les conteneurs Docker Compose en arrière-plan et s'arrête proprement avec un délai de grâce de 30 secondes.

**Vérification de disponibilité** : le playbook attend jusqu'à 300 secondes (60 tentatives de 5 secondes) que l'API réponde correctement avant de valider l'installation.

#### Service systemd — Comportement

Le service systemd généré par le template contrôle le cycle de vie de DefectDojo de manière fiable :

| Directive | Valeur | Signification |
|-----------|--------|--------------|
| `Type` | `oneshot` | Exécution unique au démarrage du service |
| `RemainAfterExit` | `yes` | Le service reste actif même après la fin du processus de démarrage |
| `ExecStart` | `docker compose up --detach` | Démarre les conteneurs en arrière-plan |
| `ExecStop` | `docker compose down --timeout 30` | Arrête proprement avec 30s de délai |
| `Restart` | `on-failure` | Redémarrage automatique en cas d'échec |
| `RestartSec` | `10` | Délai de 10 secondes avant relance |
| `StandardOutput` | `journal` | Logs dirigés vers journald |

Les logs du service sont consultables via `journalctl -u defectdojo`, ce qui permet de diagnostiquer rapidement tout problème de démarrage.

---

### 4.3 Inventaire et playbook principal

#### Inventaire (`inventory.yml`)

L'inventaire définit les machines cibles sur lesquelles les playbooks s'exécutent. Dans la configuration par défaut d'InvisiThreat, la cible est la machine locale (runner CI/CD = serveur hôte), ce qui évite d'avoir à configurer une connexion SSH.

| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `ansible_host` | `localhost` | Adresse de la machine cible |
| `ansible_connection` | `local` | Connexion locale, sans SSH |
| `ansible_user` | Utilisateur courant | Utilisateur qui exécute les commandes |

Pour cibler une machine distante, il suffit de remplacer `ansible_connection: local` par `ansible_connection: ssh` et de renseigner l'adresse IP, l'utilisateur SSH et le chemin vers la clé privée.

#### Playbook principal (`playbook.yml`)

Le playbook principal orchestre l'application du rôle `defectdojo` sur les hôtes définis dans l'inventaire. Il s'exécute avec élévation de privilèges (`become: true`) car l'installation de Docker et la création du service systemd nécessitent des droits root.

---

### 4.4 Intégration dans le pipeline CI/CD

Le pipeline GitHub Actions appelle `ensure-defectdojo.yml` au début du job `security`, avant toute interaction avec DefectDojo. Ce séquencement garantit que l'API est disponible avant que les imports de résultats de scan ne commencent.

**Flux d'exécution dans le pipeline :**

```
Push / PR
    │
    ▼
Pipeline CI/CD lancé
    │
    ▼
ensure-defectdojo.yml
  ├── Vérifie l'API DefectDojo
  ├── Clone ou redémarre si nécessaire
  └── Attend la disponibilité
    │
    ▼
Scans de sécurité (SAST, SCA, DAST)
    │
    ▼
Import des résultats dans DefectDojo
(API garantie disponible)
```

**Avantages de cette approche :**

- Si le serveur a redémarré entre deux exécutions du pipeline, DefectDojo repart automatiquement sans intervention.
- L'import des résultats ne commence que lorsque l'API est réellement prête, évitant les erreurs de connexion.
- Exécuter le pipeline plusieurs fois ne crée aucun effet de bord sur DefectDojo.

---

### 4.5 Bonnes pratiques et évolutions

**Paramétrage du chemin** : le chemin `dojo_path` est actuellement configuré statiquement. Pour un environnement multi-utilisateur, il est recommandé de le rendre paramétrable via un fichier de variables externe passé au moment de l'exécution du playbook.

**Séparation des environnements** : pour passer à l'échelle, l'inventaire peut être dupliqué en trois fichiers distincts (`dev.yml`, `staging.yml`, `prod.yml`) avec des variables propres à chaque environnement, permettant de cibler le bon serveur selon le contexte.

**Healthcheck avancé** : le healthcheck actuel teste uniquement la disponibilité HTTP. Il peut être enrichi pour vérifier que la base de données DefectDojo est également accessible et que la version de l'API correspond à celle attendue.

**Rôle `common`** : pour factoriser les tâches communes à tous les serveurs (timezone, mises à jour système, outils de base, configuration du pare-feu), un rôle `common` peut être ajouté et placé en tête de la liste des rôles dans le playbook principal.

---

## 5. Architecture des manifests Kubernetes

### Organisation des fichiers

Les manifests Kubernetes sont organisés par service dans le répertoire `infrastructure/kubernetes/`. Chaque service dispose de son propre sous-répertoire.

| Fichier / Répertoire | Contenu |
|---------------------|---------|
| `namespace.yaml` | Définition du namespace `invisithreat` |
| `configmap.yaml` | Paramètres de configuration centralisés |
| `external-services.yaml` | Service ExternalName et Endpoints pour DefectDojo |
| `storage/` | Définitions des PersistentVolumeClaims |
| `postgres/` | Déploiement et service PostgreSQL |
| `redis/` | Déploiement et service Redis |
| `risk-scoring-engine/` | Déploiement, service et configmap du moteur IA |
| `dashboard/` | Déploiement, service et configmap du dashboard React |

### Namespace

Le namespace `invisithreat` isole l'ensemble des ressources applicatives du reste du cluster. Il est étiqueté avec le nom de l'application et l'outil qui le gère (GitHub Actions), ce qui facilite les recherches et l'audit.

---

## 6. Configuration centralisée

### ConfigMap `invisithreat-config`

La ConfigMap centralise tous les paramètres non sensibles. Elle est injectée dans tous les pods comme source de variables d'environnement, permettant de modifier une configuration sans reconstruire les images Docker.

**Paramètres de connectivité :**

| Clé | Valeur | Description |
|-----|--------|-------------|
| `DEFECTDOJO_URL` | `http://defectdojo-external:8080` | URL DefectDojo accessible depuis les pods |
| `POSTGRES_HOST` | `postgres-svc` | Nom DNS interne du service PostgreSQL |
| `POSTGRES_PORT` | `5432` | Port PostgreSQL standard |
| `POSTGRES_DB` | `invisithreat` | Nom de la base de données |
| `REDIS_HOST` | `redis-svc` | Nom DNS interne du service Redis |
| `REDIS_PORT` | `6379` | Port Redis standard |

**Paramètres applicatifs :**

| Clé | Valeur | Description |
|-----|--------|-------------|
| `MODEL_PATH` | `/app/models/pipeline_latest.pkl` | Chemin du modèle ML dans le pod |
| `DATA_PATH` | `/app/data` | Répertoire racine des données |
| `LOG_LEVEL` | `INFO` | Niveau de verbosité des logs |
| `LOG_FORMAT` | `JSON` | Format des logs pour Loki |

**Seuils de risque :**

| Clé | Valeur | Signification |
|-----|--------|--------------|
| `RISK_THRESHOLDS_CRITICAL` | `8.0` | Score ≥ 8.0 → Critique |
| `RISK_THRESHOLDS_HIGH` | `6.0` | Score ≥ 6.0 → Élevé |
| `RISK_THRESHOLDS_MEDIUM` | `4.0` | Score ≥ 4.0 → Moyen |
| `RISK_THRESHOLDS_LOW` | `2.0` | Score ≥ 2.0 → Faible |

### Secret `invisithreat-secrets`

Les valeurs sensibles ne sont jamais stockées dans les manifests ou dans le dépôt Git. Elles sont injectées par le pipeline CI/CD depuis GitHub Secrets sous forme d'un objet Secret Kubernetes.

| Secret | Description |
|--------|-------------|
| `POSTGRES_PASSWORD` | Mot de passe PostgreSQL |
| `JWT_SECRET_KEY` | Clé de signature des tokens JWT |
| `API_ADMIN_TOKEN` | Token d'administration de l'API |
| `DEFECTDOJO_API_KEY` | Clé API DefectDojo |
| `GEMINI_API_KEY` | Clé API Google Gemini |
| `JIRA_API_TOKEN` | Token API Jira |
| `SMTP_PASSWORD` | Mot de passe SMTP |

Chaque pod reçoit automatiquement les variables de la ConfigMap et du Secret via le mécanisme `envFrom` de Kubernetes, sans modification du code applicatif.

---

## 7. Accès à DefectDojo depuis le cluster

### Problématique

DefectDojo est déployé sur l'hôte via Docker Compose, en dehors du cluster Kubernetes. Les pods ne peuvent pas accéder directement à `localhost:8080` car, dans un contexte Kubernetes, `localhost` désigne le pod lui-même et non l'hôte.

### Solution — Service ExternalName et Endpoints manuels

Le manifest `external-services.yaml` crée deux ressources Kubernetes complémentaires :

Un **Service sans sélecteur** nommé `defectdojo-external` dans le namespace `invisithreat`, qui expose le port 8080 et agit comme un point d'entrée DNS interne au cluster.

Un **objet Endpoints** portant le même nom que le Service, qui pointe vers l'IP réelle du nœud K3s sur le port 8080. Cette IP est injectée dynamiquement par le pipeline lors du déploiement.

Grâce à CoreDNS (inclus dans K3s), les pods peuvent résoudre `defectdojo-external` et accéder à DefectDojo via `http://defectdojo-external:8080/api/v2/`, exactement comme si le service était dans le cluster.

---

## 8. Déploiements applicatifs

### 8.1 Risk Scoring Engine

Le Risk Scoring Engine héberge l'API FastAPI et le modèle de machine learning. C'est le service central du système.

| Paramètre | Valeur |
|-----------|--------|
| Image | `localhost:5000/invisithreat/risk-scoring-engine:<sha>` |
| Port interne | `8081` |
| Exposition externe | NodePort `30081` |
| CPU (demandé / limité) | `200m` / `1000m` |
| Mémoire (demandée / limitée) | `256 Mi` / `1 Gi` |
| Sonde de vivacité | `GET /health` (délai initial : 30s) |
| Sonde de disponibilité | `GET /api/v1/ready` (délai initial : 10s) |

**Volumes montés :**

| Volume | Chemin dans le pod | Contenu |
|--------|--------------------|---------|
| `pvc-models` | `/app/models` | Fichiers `.pkl` du modèle ML |
| `pvc-data` | `/app/data` | Données brutes et traitées |

La sonde de disponibilité interroge `/api/v1/ready`, qui vérifie que le modèle ML est chargé en mémoire avant d'accepter du trafic. Cela évite que l'API reçoive des requêtes pendant le chargement du modèle après un redémarrage.

---

### 8.2 Dashboard React

Le Dashboard est une application React compilée en production et servie par Nginx embarqué dans l'image Docker.

| Paramètre | Valeur |
|-----------|--------|
| Image | `localhost:5000/invisithreat/dashboard:<sha>` |
| Port interne | `80` (Nginx) |
| Exposition externe | NodePort `30080` |
| CPU (demandé / limité) | `100m` / `500m` |
| Mémoire (demandée / limitée) | `128 Mi` / `512 Mi` |
| Sonde de vivacité | `GET /health` (délai initial : 30s) |
| Sonde de disponibilité | `GET /ready` (délai initial : 10s) |

Le Dashboard communique avec le Risk Scoring Engine via le nom de service DNS interne `risk-scoring-engine-svc:8081`.

---

### 8.3 PostgreSQL

PostgreSQL stocke les utilisateurs, sessions, scores calculés et métadonnées applicatives.

| Paramètre | Valeur |
|-----------|--------|
| Image | `postgres:16-alpine` |
| Port | `5432` |
| Type de service | ClusterIP (interne uniquement) |
| CPU (demandé / limité) | `250m` / `1000m` |
| Mémoire (demandée / limitée) | `512 Mi` / `2 Gi` |
| Stratégie de mise à jour | `Recreate` |
| Volume | `pvc-postgres` monté sur `/var/lib/postgresql/data` |

La stratégie `Recreate` est imposée par le mode d'accès `ReadWriteOnce` des volumes, qui interdit le montage simultané par deux pods. Elle implique une courte interruption lors des mises à jour, mais garantit l'intégrité des données.

---

### 8.4 Redis

Redis sert de cache distribué pour les scores IA pré-calculés, accélérant les réponses de l'API.

| Paramètre | Valeur |
|-----------|--------|
| Image | `redis:7-alpine` |
| Port | `6379` |
| Type de service | ClusterIP (interne uniquement) |
| CPU (demandé / limité) | `50m` / `200m` |
| Mémoire (demandée / limitée) | `64 Mi` / `200 Mi` |
| Persistance | Désactivée (cache pur) |
| Politique d'éviction | `allkeys-lru` |
| Mémoire maximale | `200 Mo` |

La persistance est volontairement désactivée. En cas de redémarrage, le cache est reconstruit lors des premières requêtes.

---

## 9. Stockage persistant

### StorageClass `local-path`

K3s inclut par défaut le provisionneur `local-path`, qui crée des volumes directement sur le système de fichiers du nœud dans `/var/lib/rancher/k3s/storage/`. C'est adapté à l'architecture mono-nœud d'InvisiThreat.

### PersistentVolumeClaims

Tous les PVC utilisent le mode d'accès `ReadWriteOnce`, autorisant le montage par un seul pod à la fois.

| PVC | Taille | Chemin dans le pod | Usage |
|-----|--------|--------------------|-------|
| `pvc-postgres` | 2 Gi | `/var/lib/postgresql/data` | Données PostgreSQL |
| `pvc-models` | 2 Gi | `/app/models` | Fichiers `.pkl` et métadonnées ML |
| `pvc-data` | 3 Gi | `/app/data` | Findings bruts et données traitées |

---

## 10. Registre Docker local

### Rôle du registre

Le pipeline CI/CD pousse les images Docker vers un registre local (`localhost:5000`) plutôt qu'un registre public. Cela supprime toute dépendance à Internet pour le déploiement, réduit le temps de pull des images grâce au réseau local, et élimine les limites de taux imposées par les registres publics.

### Gestion des tags

Chaque image est poussée avec deux tags complémentaires :

| Tag | Usage |
|-----|-------|
| `latest` | Toujours associé à la dernière version construite |
| `<sha-commit>` | Permet de retrouver exactement le code source correspondant |

Le déploiement Kubernetes utilise le tag SHA pour garantir que la bonne version est déployée, même en cas de builds successifs rapides.

---

## 11. Monitoring — Prometheus, Grafana, Loki

### Architecture de la stack

La stack de monitoring est déployée dans le namespace `monitoring`, isolé des services applicatifs. Elle observe à la fois le cluster Kubernetes et les services applicatifs.

```
Sources                              Collecte             Visualisation
───────                              ────────             ─────────────
Pods annotés (applicatifs)      ──▶
Node Exporter (système hôte)    ──▶  Prometheus  ──▶  Grafana
PostgreSQL Exporter (DB)        ──▶  (15s)
Kube State Metrics (K8s)        ──▶

Logs des pods                   ──▶  Promtail  ──▶  Loki  ──▶  Grafana
```

### Prometheus

Prometheus collecte les métriques toutes les 15 secondes et les conserve 24 heures. Il découvre automatiquement les cibles grâce aux annotations Kubernetes : tout pod annoté `prometheus.io/scrape: "true"` est automatiquement inclus dans la collecte.

### Grafana

Trois sources de données sont préconfigurées : Prometheus pour les métriques et Loki pour les logs. Des dashboards sont fournis pour la santé du cluster, les métriques applicatives (requêtes API, latence, erreurs), les métriques PostgreSQL et les logs applicatifs en temps réel.

### Loki et Promtail

Loki indexe uniquement les métadonnées des logs (labels), pas leur contenu, ce qui le rend très efficace en stockage. Promtail, déployé en DaemonSet, collecte les logs de tous les pods depuis les fichiers système et les envoie à Loki avec les labels appropriés (namespace, pod, conteneur).

### Node Exporter et Kube State Metrics

Node Exporter expose les métriques du système d'exploitation hôte (CPU, mémoire, disque, réseau) via un DaemonSet. Kube State Metrics surveille l'état des objets Kubernetes (pods en erreur, déploiements dégradés, PVC non liés) et les expose comme métriques Prometheus.

---

## 12. Intégration avec le pipeline CI/CD

### Déclenchement automatique

À chaque push sur `main` ou `develop`, le pipeline prend en charge l'intégralité du cycle de vie de l'infrastructure. Les playbooks Ansible étant idempotents, leur exécution est rapide si l'infrastructure est déjà en place.

Le pipeline exécute dans l'ordre :

1. Playbook K3s (idempotent — ignoré si déjà installé)
2. Playbook MetalLB et NGINX Ingress
3. Playbook Monitoring
4. Playbook DefectDojo (`ensure-defectdojo.yml`)
5. Application des manifests Kubernetes
6. Injection des secrets depuis GitHub Secrets
7. Mise à jour des images avec le SHA du commit
8. Attente de la disponibilité de chaque service
9. Vérification des endpoints de santé

### Ordre de déploiement garanti

| Étape | Ressource | Attente avant de continuer |
|-------|-----------|---------------------------|
| 1 | Namespace | Immédiat |
| 2 | Secrets et ConfigMap | Immédiat |
| 3 | Service ExternalName DefectDojo | Immédiat |
| 4 | PersistentVolumeClaims | Immédiat |
| 5 | PostgreSQL | Rollout complet (300s max) |
| 6 | Redis | Rollout complet (300s max) |
| 7 | Risk Scoring Engine | Rollout complet (240s max) |
| 8 | Dashboard | Rollout complet (240s max) |
| 9 | Mise à jour du modèle ML | Endpoint `/ready` disponible |

---

## 13. Dépannage

### Diagnostic général

En cas de problème, la première étape est toujours de vérifier l'état des pods et les événements du cluster. Les événements Kubernetes contiennent des messages précis indiquant la cause du problème (image introuvable, ressources insuffisantes, volume non lié, etc.).

### Problèmes courants et solutions

**DefectDojo ne répond pas**

Vérifier l'état du service systemd (`systemctl status defectdojo`) et consulter les logs via `journalctl -u defectdojo`. Si les conteneurs Docker sont arrêtés, relancer le service. Si c'est la première installation, vérifier que Docker est bien installé et que l'utilisateur appartient au groupe `docker`.

**K3s ne démarre pas**

La cause la plus fréquente est un conflit de port ou des fichiers résiduels d'une installation précédente. Vérifier les logs du service K3s via `journalctl -u k3s` pour identifier le message d'erreur précis.

**Un pod reste en état `Pending`**

Cela indique généralement un problème de ressources (CPU ou mémoire insuffisants) ou un PVC qui ne peut pas être lié. La commande `kubectl describe pod <nom>` affiche le détail de la cause dans la section `Events`.

**Un pod est en état `CrashLoopBackOff`**

Le conteneur démarre mais s'arrête immédiatement. Consulter les logs du pod et les logs de l'exécution précédente pour identifier l'erreur applicative. Les causes fréquentes sont une variable d'environnement manquante ou un secret mal configuré.

**Les images ne peuvent pas être téléchargées**

Vérifier que le registre Docker local est démarré sur le port 5000, que la configuration K3s est correcte et que K3s a été redémarré après cette configuration. Vérifier également que les images ont bien été construites et poussées par le pipeline.

**DefectDojo n'est pas accessible depuis les pods**

Vérifier que le Service `defectdojo-external` et l'objet `Endpoints` existent dans le namespace `invisithreat`, et que l'IP dans les Endpoints correspond à l'IP actuelle du nœud. Si l'IP a changé, relancer le déploiement.

**Le stockage est saturé**

Vérifier l'occupation des répertoires `/data/invisithreat/` et `/var/lib/rancher/k3s/storage/`. Les données brutes de `fetch_data.py` et les anciennes versions des modèles peuvent occuper un espace important. Un nettoyage manuel des fichiers obsolètes peut être nécessaire.

---

> **Document suivant** : [08-security-dashboard.md](./08-security-dashboard.md) — Dashboard React : pages, composants et intégrations