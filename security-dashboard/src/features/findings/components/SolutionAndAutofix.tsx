import React, { useEffect } from 'react';
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

  useEffect(() => {
    if (findingId) checkCapability(findingId);
  }, [findingId, checkCapability]);

  const handleShowSolution = () => fetchSolution(findingId);
  const handleAutofix = async () => {
    if (window.confirm('⚠️ L\'autofix va créer une branche, committer le correctif et ouvrir une Pull Request sur GitHub. Voulez-vous continuer ?')) {
      await executeAutofix(findingId);
    }
  };

  return (
    <div className="fdp-card" style={{ marginTop: '1.5rem' }}>
      <div className="fdp-card-header">
        <div className="fdp-card-icon" style={{ background: 'rgba(0,232,122,.09)' }}>
          <svg width="14" height="14" viewBox="0 0 15 15" fill="none" stroke="#00e87a" strokeWidth="1.4">
            <path d="M12 5.5L9 8.5L7 6.5L3 10.5" strokeLinecap="round" strokeLinejoin="round"/>
            <path d="M12 5.5H8.5M12 5.5V9" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </div>
        <div>
          <div className="fdp-card-title">Correction automatique (IA)</div>
          <div className="fdp-card-subtitle">GÉNÉRATION DE CORRECTIF & AUTOFIX GITHUB</div>
        </div>
      </div>

      <div className="fdp-llm-body" style={{ padding: '0 0 0.5rem 0' }}>
        <div className="flex gap-3 items-center flex-wrap" style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '1rem' }}>
          <button
            onClick={handleShowSolution}
            disabled={solutionLoading}
            className="fdp-llm-btn"
            style={{ '--btn-accent': '#00d4ff' } as React.CSSProperties}
          >
            {solutionLoading ? 'Génération...' : '💡 Voir la solution suggérée'}
          </button>

          {capabilityLoading ? (
            <span className="text-gray-500" style={{ fontSize: '0.8rem' }}>Vérification...</span>
          ) : capability?.can_autofix ? (
            <button
              onClick={handleAutofix}
              disabled={autofixLoading}
              className="fdp-llm-btn"
              style={{ '--btn-accent': '#00e87a' } as React.CSSProperties}
            >
              {autofixLoading ? 'Création de la PR...' : '🚀 Lancer l\'autofix (PR GitHub)'}
            </button>
          ) : capability && !capability.can_autofix ? (
            <div style={{ color: '#f5a623', fontSize: '0.8rem' }}>
              ❌ Autofix indisponible : {capability.reason}
              {capability.missing_fields.length > 0 && (
                <span style={{ display: 'block', fontSize: '0.7rem' }}>Champs manquants : {capability.missing_fields.join(', ')}</span>
              )}
            </div>
          ) : null}
        </div>

        {solutionError && <div style={{ color: '#ff3b5c', fontSize: '0.8rem', marginBottom: '0.5rem' }}>{solutionError}</div>}

        {solution && (
          <div style={{ marginTop: '1rem', padding: '1rem', background: 'rgba(255,255,255,.03)', borderRadius: '12px', border: '1px solid rgba(255,255,255,.1)' }}>
            {solution.has_file && solution.file_path && (
              <p style={{ fontSize: '0.75rem', color: '#aaa', marginBottom: '0.5rem' }}>
                📄 Fichier : {solution.file_path} {solution.line && `(ligne ${solution.line})`}
              </p>
            )}
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.25rem', color: '#ff6b35' }}>Code vulnérable :</div>
              <pre style={{ background: 'rgba(255,59,92,.1)', padding: '0.5rem', borderRadius: '8px', fontSize: '0.7rem', overflowX: 'auto' }}>
                {solution.vulnerable_snippet || 'Non disponible'}
              </pre>
            </div>
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.25rem', color: '#00e87a' }}>Code corrigé :</div>
              <pre style={{ background: 'rgba(0,232,122,.05)', padding: '0.5rem', borderRadius: '8px', fontSize: '0.7rem', overflowX: 'auto' }}>
                {solution.fixed_snippet || 'Non disponible'}
              </pre>
            </div>
            <p style={{ fontSize: '0.8rem', lineHeight: 1.4 }}>{solution.explanation}</p>
            <p style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '0.5rem' }}>
              Confiance IA : {(solution.confidence * 100).toFixed(0)}%
              {solution.from_cache && ' (depuis cache)'}
            </p>
          </div>
        )}

        {autofixResult && (
          <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(0,232,122,.1)', borderRadius: '12px', border: '1px solid rgba(0,232,122,.3)' }}>
            <p style={{ color: '#00e87a', fontSize: '0.8rem' }}>✅ Pull Request créée avec succès !</p>
            <a href={autofixResult.pr_url} target="_blank" rel="noopener noreferrer" style={{ color: '#00d4ff', fontSize: '0.8rem', textDecoration: 'underline' }}>
              Voir la PR #{autofixResult.pr_number}
            </a>
          </div>
        )}

        {autofixError && (
          <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(255,59,92,.1)', borderRadius: '12px', border: '1px solid rgba(255,59,92,.3)' }}>
            <p style={{ color: '#ff3b5c', fontSize: '0.8rem' }}>Erreur : {autofixError}</p>
          </div>
        )}
      </div>
    </div>
  );
};