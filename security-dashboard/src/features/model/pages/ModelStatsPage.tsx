import { useModelInfo } from '../hooks/useModelInfo';
import { ModelMetrics } from '../components/ModelMetrics';
import { ModelArchitecture } from '../components/ModelArchitecture';
import { ModelPerformance } from '../components/ModelPerformance';
import { FeatureExplorer } from '../components/FeatureExplorer';
import { ModelTerminal } from '../components/ModelTerminal';
import { ModelActions } from '../components/ModelActions';
import '../styles/ModelStats.css';

export function ModelStatsPage() {
  const { modelInfo, health, loading, error, reloadModel, reloading, refetch } =
    useModelInfo();

  if (loading) {
    return (
      <div className="model-loading">
        <div className="loader-content">
          <div className="loader-spinner" />
          <p>Chargement du modèle IA...</p>
        </div>
      </div>
    );
  }

  if (error || !modelInfo || !health) {
    return (
      <div className="model-error">
        <div className="error-content">
          <h2>❌ Erreur</h2>
          <p>{error || 'Impossible de charger le modèle'}</p>
          <button className="btn btn-primary" onClick={refetch}>
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="model-stats-page">
      {/* Header */}
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
            <span className="badge badge-success">✓ Modèle Actif</span>
            <span className="badge badge-info">v{modelInfo.model_version}</span>
            <span className="badge badge-info">{modelInfo.n_features} Features</span>
          </div>
        </div>
      </header>

      <main className="model-content">
        <ModelMetrics metrics={modelInfo} />
        <ModelPerformance metrics={modelInfo} />
        <ModelArchitecture metrics={modelInfo} health={health} />
        <FeatureExplorer metrics={modelInfo} />
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