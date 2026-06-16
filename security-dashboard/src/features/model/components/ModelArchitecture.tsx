import type { ModelMetrics as IModelMetrics, HealthStatus } from '../../../types/model';

interface ModelArchitectureProps {
  metrics: IModelMetrics | null;
  health:  HealthStatus  | null;
}

export function ModelArchitecture({ metrics, health }: ModelArchitectureProps) {
  if (!metrics || !health) return null;

  const formatDate = (dateString: string) =>
    new Date(dateString).toLocaleString('fr-FR', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });

  const formatUptime = (seconds: number) => {
    const days  = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins  = Math.floor((seconds % 3600) / 60);
    if (days  > 0) return `${days}j ${hours}h`;
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

        {/* Modèle */}
        <div className="arch-card">
          <div className="arch-card-header">
            <div className="arch-icon-wrap">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              </svg>
            </div>
            <h3>Modèle</h3>
          </div>
          <div className="arch-info-item">
            <label>Version</label>
            <code>{metrics.model_version}</code>
          </div>
          <div className="arch-info-item">
            <label>Type</label>
            <span className="arch-badge">Random Forest + Calibration</span>
          </div>
          <div className="arch-info-item">
            <label>Chargé</label>
            <span>{formatDate(metrics.loaded_at)}</span>
          </div>
          <div className="arch-info-item">
            <label>Uptime</label>
            <span>{formatUptime(health.uptime_seconds)}</span>
          </div>
        </div>

        {/* Configuration */}
        <div className="arch-card">
          <div className="arch-card-header">
            <div className="arch-icon-wrap">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14" />
              </svg>
            </div>
            <h3>Configuration</h3>
          </div>
          <div className="arch-info-item">
            <label>Classes</label>
            <span>{metrics.n_classes} (Info → Critical)</span>
          </div>
          <div className="arch-info-item">
            <label>Features</label>
            <span>{metrics.n_features} caractéristiques</span>
          </div>
          <div className="arch-info-item">
            <label>Stratégie</label>
            <span className="arch-badge">Multiclass Classification</span>
          </div>
          <div className="arch-info-item">
            <label>Calibration</label>
            <span className="arch-badge">Platt Scaling</span>
          </div>
        </div>

        {/* Validation */}
        <div className="arch-card">
          <div className="arch-card-header">
            <div className="arch-icon-wrap arch-icon-wrap--success">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12" />
              </svg>
            </div>
            <h3>Validation</h3>
          </div>
          <div className="arch-info-item">
            <label>CV Strategy</label>
            <span className="arch-badge">5-Fold Stratified</span>
          </div>
          <div className="arch-info-item">
            <label>Mean F1 (CV)</label>
            <span className="arch-metric arch-metric--success">
              {(metrics.metrics.cv_f1_weighted_mean * 100).toFixed(2)}%
            </span>
          </div>
          <div className="arch-info-item">
            <label>Std Dev</label>
            <span>±{(metrics.metrics.cv_f1_weighted_std * 100).toFixed(2)}%</span>
          </div>
          <div className="arch-info-item">
            <label>Data Leakage</label>
            <span className="arch-badge arch-badge--success">ZERO</span>
          </div>
        </div>

      </div>

      

      {/* Explanation */}
      <div className="arch-explanation">
        <h3>Fonctionnement</h3>
        <div className="explanation-text">
          <p>
            Le modèle est un <strong>classifieur multiclasse</strong> utilisant{' '}
            <strong>Random Forest</strong> avec calibration Platt.
            Pour chaque vulnérabilité :
          </p>
          <ol>
            <li><strong>Imputation :</strong> Les valeurs manquantes sont remplies par la médiane</li>
            <li><strong>Normalisation :</strong> Les features sont standardisées (μ=0, σ=1)</li>
            <li><strong>Classification :</strong> {metrics.n_features} features décrivent le risque</li>
            <li><strong>Probabilités :</strong> Sortie calibrée pour fiabilité maximale</li>
          </ol>
          <p>
            Résultat : un score de risque de <strong>0 à 4</strong> avec probabilité
            d'appartenance à chaque classe.
          </p>
        </div>
      </div>
    </section>
  );
}