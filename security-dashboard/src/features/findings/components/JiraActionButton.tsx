import { JiraActionButtonProps } from '../../../types/jira';
import './JiraActionButton.css';


export function JiraActionButton({
  loading,
  success,
  error,
  jiraUrl,
  jiraKey,
  onClick,
}: JiraActionButtonProps) {
  if (success && jiraUrl && jiraKey) {
    return (
      <a
        href={jiraUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="jira-action-btn success"
      >
        <span className="jira-icon">✓</span>
        <span className="jira-text">Ticket Jira: {jiraKey}</span>
        <span className="jira-external">↗</span>
      </a>
    );
  }

  return (
    <button
      className={`jira-action-btn ${loading ? 'loading' : error ? 'error' : 'default'}`}
      onClick={onClick}
      disabled={loading}
      title={error || 'Créer un ticket Jira'}
    >
      {loading ? (
        <>
          <span className="jira-spinner" />
          <span className="jira-text">Création...</span>
        </>
      ) : error ? (
        <>
          <span className="jira-icon">⚠️</span>
          <span className="jira-text">{error}</span>
        </>
      ) : (
        <>
          <span className="jira-icon">🎫</span>
          <span className="jira-text">Créer ticket Jira</span>
        </>
      )}
    </button>
  );
}