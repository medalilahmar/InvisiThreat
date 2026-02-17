# 🛡️ InvisiThreat - Plateforme de Sécurité DevSecOps

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Security](https://img.shields.io/badge/security-devsecops-orange)

## 📋 Vue d'ensemble

**InvisiThreat** intègre OWASP Juice Shop avec un pipeline DevSecOps complet :
- 🔍 **SAST** avec Semgrep
- 📦 **SCA** avec Snyk  
- 🌐 **DAST** avec OWASP ZAP
- 📊 **Aggrégation** dans DefectDojo

## 🚀 Démarrage rapide

```bash
git clone https://github.com/VOTRE-UTILISATEUR/InvisiThreat.git
cd InvisiThreat
git submodule update --init --recursive
cd juice-shop
npm install
npm run build
npm start