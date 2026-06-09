# Nova : Projet de Test Vulnérable (SAST / DAST / SCA)

Ce projet est délibérément structuré de manière professionnelle mais contient des vulnérabilités de sécurité critiques. Il a été conçu pour servir de benchmark ou d'application de test pour les scanners de vulnérabilités.

> [!WARNING]
> Ce code est vulnérable. Ne l'exécutez pas sur un réseau public ou dans un environnement de production.

---

## 🛠️ Outils de scan à tester sur ce projet

Ce projet a été conçu pour déclencher des alertes sur les trois types de scanners suivants :

### 1. SCA (Software Composition Analysis)
* **Objectif** : Identifier les dépendances tierces vulnérables.
* **Alertes attendues** : Les packages listés dans [requirements.txt](file:///c:/Users/ASUS/OneDrive/Bureau/Nova/requirements.txt) comme `Flask==2.0.1`, `Jinja2==3.0.1`, `requests==2.25.1`, et `PyYAML==5.3.1` possèdent des CVE (Common Vulnerabilities and Exposures) critiques connues.
* **Outils conseillés** : `pip-audit`, `Snyk`, `GitHub Dependency Graph / Dependabot`, `Trivy`.

### 2. SAST (Static Application Security Testing)
* **Objectif** : Analyse du code source statique à la recherche de faiblesses d'implémentation.
* **Alertes attendues** :
  * **Secrets codés en dur** : Clé d'API AWS et clé secrète Flask définies en clair dans `src/nova/app.py`.
  * **Injection SQL** : Requête SQL concaténée dynamiquement dans `src/nova/database.py`.
  * **Injection de commande** : Utilisation directe d'entrées utilisateur dans `os.popen()` dans `src/nova/app.py`.
  * **Désérialisation non sécurisée** : Utilisation de `yaml.load()` avec `Loader=yaml.Loader` et de `pickle.loads()`.
  * **Path Traversal** : Utilisation directe d'entrées utilisateur pour construire des chemins de fichiers et les renvoyer via `send_file()`.
* **Outils conseillés** : `Bandit` (spécifique Python), `Semgrep`, `SonarQube`.

### 3. DAST (Dynamic Application Security Testing)
* **Objectif** : Analyse de l'application en cours d'exécution via des requêtes HTTP malveillantes.
* **Alertes attendues** :
  * Cross-Site Scripting (XSS) réfléchi sur l'URL `/xss?name=...`.
  * SQL Injection sur l'URL `/login?username=...`.
  * Exécution de commandes à distance (RCE) via `/ping?ip=...`.
  * Lecture arbitraire de fichiers système via `/read?file=...`.
* **Outils conseillés** : `OWASP ZAP`, `Nikto`, `Nuclei`.

---

## 🚀 Installation & Exécution locale

### Prérequis
- Python 3.9 ou supérieur

### Étapes
1. **Créer un environnement virtuel** :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Linux/Mac
   .venv\Scripts\activate     # Sur Windows (PowerShell/CMD)
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application web** :
   ```bash
   python src/nova/app.py
   ```
   L'application sera accessible sur `http://127.0.0.1:5000/`.
