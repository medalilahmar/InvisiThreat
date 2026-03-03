#!/usr/bin/env python
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    def __init__(self):
        print("Initialisation du ModelMonitor...")
        
        self.model = joblib.load('models/pipeline_latest.pkl')
        self.model_path = 'models/pipeline_latest.pkl'
        
        self.meta_path = 'models/pipeline_latest_meta.json'
        if Path(self.meta_path).exists():
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        self.train_data = pd.read_csv('data/processed/findings_clean.csv')
        self.train_stats = self.train_data.describe()
        
        self.metrics_history = []
        
        Path('reports').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        print(f"Modele charge: {self.metadata.get('timestamp', 'unknown')}")
        print(f"Donnees entrainement: {len(self.train_data)} lignes")
    
    def _calculate_drift(self, train_stats, new_stats, threshold=0.1):
        drift_scores = []
        numeric_features = ['severity_num', 'cvss_score', 'age_days', 'has_cve', 
                           'has_cwe', 'tags_count', 'risk_score']
        
        for feature in numeric_features:
            if feature in train_stats.columns and feature in new_stats.columns:
                train_mean = train_stats[feature]['mean']
                new_mean = new_stats[feature]['mean']
                
                if abs(train_mean) > 0.001:
                    drift = abs(new_mean - train_mean) / abs(train_mean)
                else:
                    drift = abs(new_mean - train_mean)
                
                drift = min(drift, 1.0)
                drift_scores.append(drift)
                
                if drift > threshold:
                    print(f"Derive sur {feature}: {drift:.2%}")
        
        global_drift = np.mean(drift_scores) if drift_scores else 0
        return global_drift
    
    def alert(self, message, level="warning"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_message = message.encode('ascii', 'ignore').decode('ascii')
        alert_msg = f"[{timestamp}] [{level.upper()}] {clean_message}"
        
        print(f"[{timestamp}] [{level.upper()}] {message}")
        
        try:
            with open('logs/alerts.log', 'a', encoding='utf-8') as f:
                f.write(alert_msg + '\n')
        except:
            with open('logs/alerts.log', 'a', encoding='ascii', errors='ignore') as f:
                f.write(alert_msg + '\n')
    
    def evaluate_drift(self, new_data_path=None, new_data=None):
        if new_data_path:
            new_data = pd.read_csv(new_data_path)
            print(f"Nouvelles donnees: {new_data_path}")
        elif new_data is not None:
            print(f"Nouvelles donnees: {len(new_data)} lignes")
        else:
            raise ValueError("Specifiez new_data_path ou new_data")
        
        new_stats = new_data.describe()
        drift_score = self._calculate_drift(self.train_stats, new_stats)
        
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'drift_score': drift_score,
            'n_samples': len(new_data)
        })
        
        if drift_score > 0.1:
            self.alert(f"Data drift detecte ! Score: {drift_score:.2%}")
            self.alert("Reentrainement recommande")
        else:
            print(f"Pas de derive significative: {drift_score:.2%}")
        
        return drift_score
    
    def plot_feature_distributions(self, new_data, save=True):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        features = ['severity_num', 'cvss_score', 'age_days', 
                   'has_cve', 'tags_count', 'risk_score']
        
        for i, feature in enumerate(features):
            if feature in self.train_data.columns and feature in new_data.columns:
                ax = axes[i]
                ax.hist(self.train_data[feature].dropna(), bins=20, 
                       alpha=0.5, label='Train', color='blue', density=True)
                ax.hist(new_data[feature].dropna(), bins=20, 
                       alpha=0.5, label='Nouvelles', color='red', density=True)
                ax.set_title(feature)
                ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig('reports/feature_drift_comparison.png', dpi=150)
            print("Graphique sauvegarde: reports/feature_drift_comparison.png")
        
        return fig
    
    def generate_report(self, new_data=None):
        print("Generation du rapport de performance...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = self.metadata.get('metrics', {})
        if metrics:
            bars = ax1.bar(['F1-weighted', 'F1-macro', 'ROC-AUC'], 
                          [metrics.get('test_f1_weighted', 0),
                           metrics.get('test_f1_macro', 0),
                           metrics.get('test_roc_auc_ovr', 0)],
                          color=['#2ecc71', '#3498db', '#9b59b6'])
            ax1.set_ylim(0, 1.1)
            ax1.set_title('Metriques du Modele')
            ax1.set_ylabel('Score')
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
        
        ax2 = fig.add_subplot(gs[0, 1:])
        if hasattr(self.model.named_steps['model'], 'feature_importances_'):
            importances = self.model.named_steps['model'].feature_importances_
            feature_names = self.train_data.columns[:len(importances)]
            indices = np.argsort(importances)[-10:]
            ax2.barh(range(10), importances[indices], color='#3498db')
            ax2.set_yticks(range(10))
            ax2.set_yticklabels([feature_names[i] for i in indices])
            ax2.set_title('Top 10 Features Importantes')
            ax2.set_xlabel('Importance')
        
        ax3 = fig.add_subplot(gs[1, 0])
        if self.metrics_history:
            timestamps = [m['timestamp'] for m in self.metrics_history]
            drift_scores = [m['drift_score'] for m in self.metrics_history]
            ax3.plot(timestamps, drift_scores, marker='o', color='#e74c3c')
            ax3.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Seuil alerte')
            ax3.set_title('Evolution du Data Drift')
            ax3.set_ylabel('Drift Score')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
        
        ax4 = fig.add_subplot(gs[1, 1])
        if new_data is not None and 'risk_class' in new_data.columns:
            risk_counts = new_data['risk_class'].value_counts().sort_index()
            labels = ['Info', 'Low', 'Medium', 'High', 'Critical']
            colors = ['#95a5a6', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
            bars = ax4.bar([labels[i] for i in risk_counts.index], 
                          risk_counts.values, color=colors)
            ax4.set_title('Distribution des Risques (Nouvelles donnees)')
            ax4.set_ylabel('Nombre')
        
        ax5 = fig.add_subplot(gs[1, 2])
        if new_data is not None and 'age_days' in new_data.columns:
            ax5.hist(new_data['age_days'], bins=20, color='#3498db', edgecolor='white')
            ax5.set_title('Age des Vulnerabilites (jours)')
            ax5.set_xlabel('Jours')
            ax5.set_ylabel('Nombre')
        
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        model_version = self.metadata.get('timestamp', 'N/A')
        if isinstance(model_version, str):
            model_version = model_version.encode('ascii', 'ignore').decode('ascii')
        
        f1_score = metrics.get('test_f1_weighted', 'N/A')
        if isinstance(f1_score, float):
            f1_score = f"{f1_score:.4f}"
        
        last_drift = f"{self.metrics_history[-1]['drift_score']:.2%}" if self.metrics_history else 'N/A'
        
        info_text = f"""
RAPPORT DE MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MODELE:
- Version: {model_version}
- Classes: {self.metadata.get('n_classes', 5)} (Info -> Critical)
- Features: {self.metadata.get('n_features', 24)}
- F1-weighted: {f1_score}

DONNEES:
- Entrainement: {len(self.train_data)} echantillons
- Dernier drift: {last_drift}

PERFORMANCES:
- Requetes: {len(self.metrics_history)} analyses
- Statut: {'Sain' if (self.metrics_history and self.metrics_history[-1]['drift_score'] < 0.1) else 'A surveiller'}
"""
        
        ax6.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa'))
        
        fig.suptitle('AI Risk Engine - Dashboard de Monitoring', fontsize=16, fontweight='bold')
        
        report_path = 'reports/model_performance_dashboard.png'
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard sauvegarde: {report_path}")
        
        html_report = self._generate_html_report()
        html_path = 'reports/monitoring_report.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"Rapport HTML sauvegarde: {html_path}")
        
        plt.close()
        return report_path
    
    def _generate_html_report(self):
        metrics = self.metadata.get('metrics', {})
        last_drift = f"{self.metrics_history[-1]['drift_score']:.2%}" if self.metrics_history else 'N/A'
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI Risk Engine - Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #f8f9fa; border-radius: 5px; min-width: 200px; }}
        .good {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AI Risk Engine - Monitoring Report</h1>
        <p>Genere le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Informations du Modele</h2>
        <div class="metric">
            <h3>Version</h3>
            <p>{self.metadata.get('timestamp', 'N/A')}</p>
        </div>
        <div class="metric">
            <h3>Classes</h3>
            <p>{self.metadata.get('n_classes', 5)}</p>
        </div>
        <div class="metric">
            <h3>F1-Score</h3>
            <p>{metrics.get('test_f1_weighted', 'N/A')}</p>
        </div>
        
        <h2>Etat de Sante</h2>
        <div class="metric">
            <h3>Statut</h3>
            <p class="{'good' if (self.metrics_history and self.metrics_history[-1]['drift_score'] < 0.1) else 'warning'}">
                {'Operationnel' if (self.metrics_history and self.metrics_history[-1]['drift_score'] < 0.1) else 'A surveiller'}
            </p>
        </div>
        <div class="metric">
            <h3>Dernier Drift</h3>
            <p>{last_drift}</p>
        </div>
        
        <h2>Images</h2>
        <img src="model_performance_dashboard.png" alt="Performance Dashboard">
        <br><br>
        <img src="feature_drift_comparison.png" alt="Feature Drift Comparison">
    </div>
</body>
</html>"""
        return html
    
    def run_periodic_check(self, data_path, interval_hours=24):
        import time
        while True:
            print(f"\nVerification periodique - {datetime.now()}")
            drift = self.evaluate_drift(new_data_path=data_path)
            self.generate_report()
            print(f"Prochaine verification dans {interval_hours}h")
            time.sleep(interval_hours * 3600)

if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING DU MODELE AI RISK ENGINE")
    print("=" * 60)
    
    monitor = ModelMonitor()
    
    print("\nGeneration de donnees de test realistes...")
    n_samples = 1000
    
    new_data = pd.DataFrame({
        'severity_num': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.13, 0.46, 0.27, 0.11, 0.03]),
        'cvss_score': np.random.uniform(0, 10, n_samples),
        'age_days': np.random.randint(1, 100, n_samples),
        'has_cve': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        'has_cwe': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'tags_count': np.random.poisson(1, n_samples),
        'risk_score': np.random.uniform(0, 10, n_samples),
        'risk_class': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.13, 0.46, 0.27, 0.11, 0.03])
    })
    
    print("\nEvaluation du data drift...")
    drift_score = monitor.evaluate_drift(new_data=new_data)
    print(f"Score de derive: {drift_score:.2%}")
    
    print("\nGeneration des graphiques de comparaison...")
    monitor.plot_feature_distributions(new_data)
    
    print("\nGeneration du dashboard...")
    report_path = monitor.generate_report(new_data=new_data)
    
    print(f"\nDashboard genere: {report_path}")
    print("Rapport HTML: reports/monitoring_report.html")
    print("\nOuvrez le rapport dans votre navigateur pour voir le dashboard !")