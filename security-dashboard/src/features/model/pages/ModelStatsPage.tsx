import { useModelInfo } from '../hooks/useModelInfo';
import { ModelMetrics }      from '../components/ModelMetrics';
import { ModelArchitecture } from '../components/ModelArchitecture';
import { ModelPerformance }  from '../components/ModelPerformance';
import { FeatureExplorer }   from '../components/FeatureExplorer';
import { ModelTerminal }     from '../components/ModelTerminal';
import { ModelActions }      from '../components/ModelActions';
import '../styles/ModelStats.css';

export function ModelStatsPage() {
  const { modelInfo, health, loading, error, reloadModel, reloading, refetch } =
    useModelInfo();

  if (loading) {
    return (
      <div className="model-loading home-root">
        <div className="bg-grid" />
        <div className="bg-radials" />
        <div className="scan-line" />
        <div className="loader-content">
          <div className="loader-spinner" />
          <p className="loader-label">Chargement du modèle IA...</p>
        </div>
      </div>
    );
  }

  if (error || !modelInfo || !health) {
    return (
      <div className="model-error home-root">
        <div className="bg-grid" />
        <div className="bg-radials" />
        <div className="error-content">
          <div className="error-icon-wrap">
            <svg
              width="32" height="32" viewBox="0 0 24 24" fill="none"
              stroke="var(--severity-critical)" strokeWidth="1.8"
              strokeLinecap="round" strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="error-title">Erreur de chargement</h2>
          <p className="error-message">{error || 'Impossible de charger le modèle'}</p>
          <button className="btn btn-primary" onClick={refetch}>
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="model-stats-page home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <header className="model-header">
        <div className="header-content">
          <div className="header-label">
            <span className="fp-label-dot" />
            MACHINE LEARNING
          </div>
          <h1 className="header-title">
            InvisiThreat AI Risk <span>Engine</span>
          </h1>
          <p className="header-subtitle">
            Classifieur multiclasse pour l'évaluation automatique du risque de vulnérabilités
          </p>
          <div className="header-badges">
            <span className="badge badge-low">Modèle Actif</span>
            <span className="badge badge-info">v{modelInfo.model_version}</span>
            <span className="badge badge-info">{modelInfo.n_features} Features</span>
          </div>
        </div>
      </header>

      <main className="model-content">
        <ModelMetrics      metrics={modelInfo} />
        <ModelPerformance  metrics={modelInfo} />
        <ModelArchitecture metrics={modelInfo} health={health} />
        <FeatureExplorer   metrics={modelInfo} />
        <ModelTerminal />
        <ModelActions
          onReload={reloadModel}
          isReloading={reloading}
        />
      </main>
    </div>
  );
}

export default ModelStatsPage;