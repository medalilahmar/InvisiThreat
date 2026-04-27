import { JiraResultPanelProps } from '../../../types/jira';
import { JiraActionButton } from './JiraActionButton';
import './JiraResultPanel.css';


export function JiraResultPanel({
  loading,
  error,
  success,
  jiraUrl,
  jiraKey,
  onRetry,
}: JiraResultPanelProps) {
  return (
    <div className="jira-result-panel">
      <div className="jira-result-header">
        <div className="jira-result-icon">🔗</div>
        <div>
          <div className="jira-result-title">Export vers Jira</div>
          <div className="jira-result-subtitle">INTÉGRATION AUTOMATIQUE</div>
        </div>
      </div>

      <div className="jira-result-content">
        {success && jiraKey && jiraUrl ? (
          <div className="jira-success-box">
            <div className="jira-success-icon">✓</div>
            <div className="jira-success-text">
              <p>
                <strong>Ticket Jira créé avec succès !</strong>
              </p>
              <a
                href={jiraUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="jira-success-link"
              >
                Ouvrir {jiraKey} dans Jira →
              </a>
            </div>
          </div>
        ) : error ? (
          <div className="jira-error-box">
            <div className="jira-error-icon">⚠️</div>
            <div className="jira-error-text">
              <p>
                <strong>Erreur lors de la création</strong>
              </p>
              <p className="jira-error-message">{error}</p>
              <button className="jira-retry-btn" onClick={onRetry} disabled={loading}>
                ↺ Réessayer
              </button>
            </div>
          </div>
        ) : loading ? (
          <div className="jira-loading-box">
            <div className="jira-loading-spinner" />
            <p>Création du ticket Jira en cours...</p>
            <p className="jira-loading-hint">Les informations du finding sont envoyées au backend</p>
          </div>
        ) : (
          <div className="jira-empty-box">
            <div className="jira-empty-icon">🎫</div>
            <p>
              Cliquez sur le bouton ci-dessous pour créer automatiquement un ticket Jira avec
              toutes les informations du finding.
            </p>
          </div>
        )}

        <JiraActionButton
          loading={loading}
          success={success}
          error={error}
          jiraUrl={jiraUrl}
          jiraKey={jiraKey}
          onClick={onRetry}
        />
      </div>
    </div>
  );
}