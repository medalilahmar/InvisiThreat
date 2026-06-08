import React, { useEffect, useState } from 'react';
import { useSolution } from '../hooks/useSolution';
import { useAutofix } from '../hooks/useAutofix';

interface Props {
  findingId: number;
}

export const SolutionAndAutofix: React.FC<Props> = ({ findingId }) => {
  const { solution, loading: solutionLoading, error: solutionError, fetchSolution } = useSolution();
  const {
    capability,
    capabilityLoading,
    autofixResult,
    autofixLoading,
    autofixError,
    checkCapability,
    executeAutofix,
  } = useAutofix();

  const [activeTab, setActiveTab] = useState<'vulnerable' | 'fixed'>('vulnerable');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (findingId) checkCapability(findingId);
  }, [findingId, checkCapability]);

  const handleShowSolution = () => fetchSolution(findingId);

  const handleAutofix = async () => {
    if (window.confirm('L\'autofix va créer une branche, committer le correctif et ouvrir une Pull Request sur GitHub. Voulez-vous continuer ?')) {
      await executeAutofix(findingId);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  };

  const confidencePct = solution ? Math.round(solution.confidence * 100) : 0;
  const confidenceColor =
    confidencePct >= 80 ? 'var(--severity-low)'    :
    confidencePct >= 50 ? 'var(--severity-medium)' :
                          'var(--severity-critical)';

  return (
    <div className="fdp-llm-panel" style={{ '--llm-accent': 'var(--severity-low)' } as React.CSSProperties}>

      {/* ══ Header — identique à LLMPanel ══════════════════════════════════ */}
      <div className="fdp-llm-header">
        <div className="fdp-llm-title-group">
          <div className="fdp-llm-icon-box" style={{ background: 'var(--glass-success)' }}>
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none"
              stroke="var(--severity-low)" strokeWidth="1.4">
              <path d="M3 8l3 3 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div className="fdp-llm-h">Correction automatique</div>
            <span className="fdp-llm-model-pill">deepseek-coder · Ollama</span>
          </div>
        </div>

        {/* ── Boutons d'action dans le header ── */}
        <div className="saf-header-actions">
          {/* Bouton Solution */}
          <button
            onClick={handleShowSolution}
            disabled={solutionLoading}
            className={`fdp-llm-btn${solution ? ' regen' : ''}`}
            style={{ '--btn-accent': 'var(--accent)' } as React.CSSProperties}
          >
            {solutionLoading ? (
              <><span className="fdp-spinner" /> Génération...</>
            ) : (
              <>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="1.8"
                  strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8"  x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {solution ? '↺ Regénérer' : '✦ Voir la solution'}
              </>
            )}
          </button>

          {/* Bouton Autofix */}
          {capabilityLoading ? (
            <span className="saf-checking">
              <span className="fdp-spinner"
                style={{ borderTopColor: 'var(--muted)', borderColor: 'var(--border)' } as React.CSSProperties} />
              Vérification...
            </span>
          ) : capability?.can_autofix ? (
            <button
              onClick={handleAutofix}
              disabled={autofixLoading}
              className="fdp-llm-btn"
              style={{
                '--btn-accent': 'var(--severity-low)',
                background: 'linear-gradient(135deg, var(--severity-low), #1a9e4f)',
              } as React.CSSProperties}
            >
              {autofixLoading ? (
                <><span className="fdp-spinner" /> Création PR...</>
              ) : (
                <>
                  <svg width="13" height="13" viewBox="0 0 15 15" fill="none"
                    stroke="currentColor" strokeWidth="1.5">
                    <circle cx="4.5" cy="3.5" r="1.5" />
                    <circle cx="10.5" cy="11.5" r="1.5" />
                    <circle cx="10.5" cy="3.5" r="1.5" />
                    <path d="M4.5 5v4a2 2 0 002 2h2.5" strokeLinecap="round" />
                    <path d="M10.5 5v2" strokeLinecap="round" />
                  </svg>
                  Autofix → PR GitHub
                </>
              )}
            </button>
          ) : capability && !capability.can_autofix ? (
            <div className="saf-unavail-inline">
              <svg width="10" height="10" viewBox="0 0 12 12" fill="none"
                stroke="var(--severity-critical)" strokeWidth="1.5">
                <circle cx="6" cy="6" r="5" />
                <path d="M6 3.5v3M6 8.5h.01" strokeLinecap="round" />
              </svg>
              Autofix indisponible
              {capability.missing_fields?.length > 0 && (
                <span className="saf-unavail-fields">
                  — champs manquants : {capability.missing_fields.join(', ')}
                </span>
              )}
            </div>
          ) : null}
        </div>
      </div>

      {/* ══ État vide — identique à LLMPanel ═══════════════════════════════ */}
      {!solutionLoading && !solutionError && !solution && (
        <div className="fdp-llm-empty">
          <div className="fdp-llm-empty-icon" style={{ color: 'var(--severity-low)' }}>
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none"
              stroke="currentColor" strokeWidth="1.4">
              <path d="M3 8l3 3 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <p>
            Cliquez sur <strong>«&nbsp;Voir la solution&nbsp;»</strong> pour générer un correctif IA,
            ou <strong>«&nbsp;Autofix&nbsp;»</strong> pour créer une PR GitHub automatiquement.
          </p>
        </div>
      )}

      {/* ══ Loading — identique à LLMPanel ═════════════════════════════════ */}
      {solutionLoading && (
        <div className="fdp-llm-loading">
          <div className="fdp-progress-track">
            <div className="fdp-progress-fill"
              style={{ '--pf-color': 'var(--severity-low)' } as React.CSSProperties} />
          </div>
          <p>Analyse du code et génération du correctif... (1–2 min)</p>
        </div>
      )}

      {/* ══ Erreur solution ═════════════════════════════════════════════════ */}
      {solutionError && !solutionLoading && (
        <div className="fdp-error-box">
          ⚠️ {solutionError}
          <button className="fdp-error-retry" onClick={handleShowSolution}>Réessayer</button>
        </div>
      )}

      {/* ══ Résultat solution ═══════════════════════════════════════════════ */}
      {solution && !solutionLoading && (
        <div className="fdp-llm-body">

          {/* Cache pill */}
          {solution.from_cache && (
            <div className="fdp-cache-pill">⚡ Depuis le cache</div>
          )}

          <div className="fdp-llm-sections">

            {/* ── Section fichier ── */}
            {solution.has_file && solution.file_path && (
              <div className="fdp-llm-section">
                <div className="fdp-llm-sec-icon" style={{ background: 'var(--bg4)' }}>
                  <svg width="13" height="13" viewBox="0 0 15 15" fill="none"
                    stroke="currentColor" strokeWidth="1.4">
                    <path d="M3 2h6l3 3v8H3V2z" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M9 2v3h3"           strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <div style={{ flex: 1 }}>
                  <div className="fdp-llm-sec-label" style={{ color: 'var(--muted)' }}>FICHIER CONCERNÉ</div>
                  <div className="fdp-file-path-box" style={{ marginTop: 6, marginBottom: 0 }}>
                    <code className="fdp-filepath-code">{solution.file_path}</code>
                    {solution.line && <span className="fdp-line-badge">L.{solution.line}</span>}
                  </div>
                </div>
              </div>
            )}

            {/* ── Section code (onglets vulnérable / corrigé) ── */}
            <div className="fdp-llm-section" style={{ flexDirection: 'column', gap: 0 }}>

              {/* Tab bar */}
              <div className="saf-tab-bar">
                <button
                  className={`saf-tab${activeTab === 'vulnerable' ? ' saf-tab--active saf-tab--danger' : ''}`}
                  onClick={() => setActiveTab('vulnerable')}
                >
                  <svg width="9" height="9" viewBox="0 0 12 12" fill="none"
                    stroke="currentColor" strokeWidth="1.5">
                    <circle cx="6" cy="6" r="5" />
                    <path d="M6 3.5v3M6 8.5h.01" strokeLinecap="round" />
                  </svg>
                  Code vulnérable
                </button>

                <button
                  className={`saf-tab${activeTab === 'fixed' ? ' saf-tab--active saf-tab--success' : ''}`}
                  onClick={() => setActiveTab('fixed')}
                >
                  <svg width="9" height="9" viewBox="0 0 12 12" fill="none"
                    stroke="currentColor" strokeWidth="1.5">
                    <path d="M2 6l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  Code corrigé
                </button>

                <button
                  className="saf-copy-btn"
                  onClick={() => handleCopy(
                    activeTab === 'vulnerable'
                      ? (solution.vulnerable_snippet ?? '')
                      : (solution.fixed_snippet     ?? '')
                  )}
                >
                  {copied ? (
                    <>
                      <svg width="9" height="9" viewBox="0 0 12 12" fill="none"
                        stroke="var(--severity-low)" strokeWidth="1.5">
                        <path d="M2 6l3 3 5-5" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Copié
                    </>
                  ) : (
                    <>
                      <svg width="9" height="9" viewBox="0 0 12 12" fill="none"
                        stroke="currentColor" strokeWidth="1.5">
                        <rect x="4" y="4" width="7" height="7" rx="1" />
                        <path d="M2 8V2h6" strokeLinecap="round" strokeLinejoin="round" />
                      </svg>
                      Copier
                    </>
                  )}
                </button>
              </div>

              {/* Code panel */}
              <pre className={`saf-pre${activeTab === 'vulnerable' ? ' saf-pre--danger' : ' saf-pre--success'}`}>
                <code>{
                  activeTab === 'vulnerable'
                    ? (solution.vulnerable_snippet || 'Non disponible')
                    : (solution.fixed_snippet     || 'Non disponible')
                }</code>
              </pre>
            </div>

            {/* ── Section explication ── */}
            <div className="fdp-llm-section">
              <div className="fdp-llm-sec-icon" style={{ background: 'var(--glass-success)' }}>💡</div>
              <div style={{ flex: 1 }}>
                <div className="fdp-llm-sec-label" style={{ color: 'var(--severity-low)' }}>EXPLICATION DU CORRECTIF</div>
                <p className="fdp-llm-sec-text">{solution.explanation}</p>

                {/* Confiance IA inline */}
                <div className="saf-confidence-row">
                  <span className="fdp-llm-sec-label" style={{ color: 'var(--dimmed)' }}>CONFIANCE IA</span>
                  <div className="saf-confidence-bar-wrap">
                    <div className="fdp-progress-track" style={{ flex: 1 }}>
                      <div style={{
                        height: '100%',
                        width: `${confidencePct}%`,
                        background: confidenceColor,
                        borderRadius: 2,
                        transition: 'width .6s ease',
                      }} />
                    </div>
                    <span className="saf-confidence-pct" style={{ color: confidenceColor }}>
                      {confidencePct}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

          </div>{/* /fdp-llm-sections */}
        </div>
      )}

      {/* ══ PR success ══════════════════════════════════════════════════════ */}
      {autofixResult && (
        <div className="fdp-error-box saf-pr-success">
          <svg width="14" height="14" viewBox="0 0 15 15" fill="none"
            stroke="var(--severity-low)" strokeWidth="1.5">
            <circle cx="4.5" cy="3.5" r="1.5" />
            <circle cx="10.5" cy="11.5" r="1.5" />
            <circle cx="10.5" cy="3.5" r="1.5" />
            <path d="M4.5 5v4a2 2 0 002 2h2.5" strokeLinecap="round" />
            <path d="M10.5 5v2" strokeLinecap="round" />
          </svg>
          Pull Request créée avec succès !
          <a
            href={autofixResult.pr_url}
            target="_blank"
            rel="noopener noreferrer"
            className="saf-pr-link"
          >
            → Voir la PR #{autofixResult.pr_number}
          </a>
        </div>
      )}

      {/* ══ Autofix error ════════════════════════════════════════════════════ */}
      {autofixError && (
        <div className="fdp-error-box">
          ⚠️ {autofixError}
          <button className="fdp-error-retry" onClick={handleAutofix}>Réessayer</button>
        </div>
      )}

    </div>
  );
};