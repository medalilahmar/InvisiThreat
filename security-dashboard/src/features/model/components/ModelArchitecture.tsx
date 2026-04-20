
import type { ModelMetrics as IModelMetrics, HealthStatus } from '../../../types/model';

interface ModelArchitectureProps {
  metrics: IModelMetrics | null;
  health: HealthStatus | null;
}

export function ModelArchitecture({ metrics, health }: ModelArchitectureProps) {
  if (!metrics || !health) return null;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('fr-FR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}j ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  return (
    <section className="model-architecture">
      <div className="section-label">
        <span className="fp-label-dot" />
        ARCHITECTURE
      </div>
      <h2 className="section-title">
        Pipeline d'<span>Entraînement</span>
      </h2>

      <div className="arch-grid">
        {/* Model Info */}
        <div className="arch-card">
          <div className="arch-card-header">
            <span className="arch-icon">📦</span>
            <h3>Modèle</h3>
          </div>
          <div className="arch-info-item">
            <label>Version:</label>
            <code>{metrics.model_version}</code>
          </div>
          <div className="arch-info-item">
            <label>Type:</label>
            <span className="arch-badge">Random Forest + Calibration</span>
          </div>
          <div className="arch-info-item">
            <label>Chargé:</label>
            <span>{formatDate(metrics.loaded_at)}</span>
          </div>
          <div className="arch-info-item">
            <label>Uptime:</label>
            <span>{formatUptime(health.uptime_seconds)}</span>
          </div>
        </div>

        {/* Configuration */}
        <div className="arch-card">
          <div className="arch-card-header">
            <span className="arch-icon">⚙️</span>
            <h3>Configuration</h3>
          </div>
          <div className="arch-info-item">
            <label>Classes:</label>
            <span>{metrics.n_classes} (Info → Critical)</span>
          </div>
          <div className="arch-info-item">
            <label>Features:</label>
            <span>{metrics.n_features} caractéristiques</span>
          </div>
          <div className="arch-info-item">
            <label>Stratégie:</label>
            <span className="arch-badge">Multiclass Classification</span>
          </div>
          <div className="arch-info-item">
            <label>Calibration:</label>
            <span className="arch-badge">Platt Scaling</span>
          </div>
        </div>

        {/* Validation */}
        <div className="arch-card">
          <div className="arch-card-header">
            <span className="arch-icon">✅</span>
            <h3>Validation</h3>
          </div>
          <div className="arch-info-item">
            <label>CV Strategy:</label>
            <span className="arch-badge">5-Fold Stratified</span>
          </div>
          <div className="arch-info-item">
            <label>Mean F1 (CV):</label>
            <span style={{ color: '#2ed573', fontWeight: 'bold' }}>
              {(metrics.metrics.cv_f1_weighted_mean * 100).toFixed(2)}%
            </span>
          </div>
          <div className="arch-info-item">
            <label>Std Dev:</label>
            <span>±{(metrics.metrics.cv_f1_weighted_std * 100).toFixed(2)}%</span>
          </div>
          <div className="arch-info-item">
            <label>Data Leakage:</label>
            <span className="arch-badge success">✓ ZERO</span>
          </div>
        </div>
      </div>

      {/* Pipeline Steps */}
      <div className="pipeline-steps">
        <h3>🔄 Pipeline d'Inférence</h3>
        <div className="steps-container">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>SimpleImputer</h4>
              <p>Stratégie: median</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>StandardScaler</h4>
              <p>Normalisation (μ=0, σ=1)</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>RandomForestClassifier</h4>
              <p>Ensemble de décision</p>
            </div>
          </div>
          <div className="step-arrow">→</div>
          <div className="step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>CalibratedClassifier</h4>
              <p>Platt Scaling (probas)</p>
            </div>
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div className="arch-explanation">
        <h3>💡 Comment ��a marche ?</h3>
        <div className="explanation-text">
          <p>
            Le modèle est un <strong>classifieur multiclasse</strong> utilisant <strong>Random Forest</strong> 
            avec calibration Platt. Pour chaque vulnérabilité:
          </p>
          <ol>
            <li><strong>Imputation:</strong> Les valeurs manquantes sont remplies par la médiane</li>
            <li><strong>Normalisation:</strong> Les features sont standardisées (μ=0, σ=1)</li>
            <li><strong>Classification:</strong> {metrics.n_features} features décrivent le risque</li>
            <li><strong>Probabilités:</strong> Sortie calibrée pour confiabilité maximale</li>
          </ol>
          <p>
            Résultat: Un score de risque de <strong>0 à 4</strong> avec probabilité d'appartenance à chaque classe.
          </p>
        </div>
      </div>
    </section>
  );
}