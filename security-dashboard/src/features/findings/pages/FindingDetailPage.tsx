import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';
import { predictionsApi, PredictionRequest } from '../../../api/services/predictions';
import { explanationsApi, LLMExplanation, LLMRecommendation } from '../../../api/services/explanations';
import { jiraApi } from '../../../api/services/jira';
import type { JiraIntegrationState } from '../../../types/jira';

import './FindingDetailPage.css';

export default function FindingDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const findingId = parseInt(id || '0', 10);

  const [llmExplanation, setLlmExplanation] = useState<LLMExplanation | null>(null);
  const [llmRecommendation, setLlmRecommendation] = useState<LLMRecommendation | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [recommendationError, setRecommendationError] = useState<string | null>(null);

  const [jiraState, setJiraState] = useState<JiraIntegrationState>({
    loading: false,
    error: null,
    success: false,
    jiraKey: null,
    jiraUrl: null,
  });

  useEffect(() => {
    if (findingId) {
      const checkExistingTicket = async () => {
        try {
          console.log('[Jira] Checking for existing issue...');
          const result = await jiraApi.checkIssue(findingId);
          
          if (result.exists && result.jira_key) {
            console.log('[Jira] Found existing issue:', result.jira_key);
            setJiraState({
              loading: false,
              error: null,
              success: true,
              jiraKey: result.jira_key,
              jiraUrl: result.jira_url,
            });
          }
        } catch (err) {
          console.error('[Jira] Erreur vérification initiale:', err);
        }
      };

      checkExistingTicket();
    }
  }, [findingId]);

  const { data: finding, isLoading: loadingFinding, error } = useQuery({
    queryKey: ['finding', findingId],
    queryFn: () => findingsApi.getOne(findingId).then(res => res.data),
    enabled: !!findingId,
  });

  const findingsForBatch: PredictionRequest[] = finding ? [{
    severity: finding.severity,
    cvss_score: finding.cvss_score || 0,
    title: finding.title,
    tags: finding.tags,
    days_open: finding.age_days || 0,
    finding_id: finding.id,
    engagement_id: finding.engagement_id ?? undefined,
  }] : [];

  const { data: scoreResult } = useQuery({
    queryKey: ['score', findingId],
    queryFn: () => predictionsApi.predictBatch(findingsForBatch).then(res => res.data),
    enabled: !!finding,
  });

  const aiScore = scoreResult?.results[0];

  const buildLLMPayload = () => ({
    finding_id: finding!.id,
    title: finding!.title ?? '',
    severity: finding!.severity ?? 'medium',
    cvss_score: finding!.cvss_score ?? 0,
    description: finding!.description ?? '',
    cve: (finding as any)?.cve ?? '',
    cwe: (finding as any)?.cwe ?? '',
    file_path: finding!.file_path ?? '',
    tags: finding!.tags ?? [],
    risk_level: aiScore?.risk_level ?? '',
  });

  const handleExplanation = async () => {
    if (!finding) return;
    setExplanationLoading(true);
    setExplanationError(null);

    try {
      const res = await explanationsApi.explain(buildLLMPayload());

      const data = res.data;
      if (!data?.summary) {
        console.error('[LLM explain] Réponse inattendue du backend:', data);
        setExplanationError("Réponse LLM invalide — champ 'summary' manquant");
        return;
      }

      setLlmExplanation(data);
    } catch (e: any) {
      const status = e?.response?.status;
      const code = e?.code;

      if (status === 404) {
        setExplanationError('Finding introuvable dans DefectDojo');
      } else if (code === 'ECONNABORTED') {
        setExplanationError('Délai dépassé — le modèle est trop lent, réessayez dans quelques secondes');
      } else if (status === 503) {
        setExplanationError('Service Ollama indisponible — vérifiez que le modèle est chargé');
      } else {
        setExplanationError(`Erreur lors de l'explication (${status ?? code ?? 'inconnue'})`);
      }

      console.error('[LLM explain] Erreur:', {
        status,
        code,
        data: e?.response?.data,
        message: e?.message,
      });
    } finally {
      setExplanationLoading(false);
    }
  };

  const handleRecommendation = async () => {
    if (!finding) return;
    setRecommendationLoading(true);
    setRecommendationError(null);

    try {
      const res = await explanationsApi.recommend(buildLLMPayload());

      const data = res.data;
      if (!data?.recommendations?.length) {
        console.error('[LLM recommend] Réponse inattendue du backend:', data);
        setRecommendationError("Réponse LLM invalide — champ 'recommendations' manquant");
        return;
      }

      setLlmRecommendation(data);
    } catch (e: any) {
      const status = e?.response?.status;
      const code = e?.code;

      if (status === 404) {
        setRecommendationError('Finding introuvable dans DefectDojo');
      } else if (code === 'ECONNABORTED') {
        setRecommendationError('Délai dépassé — le modèle est trop lent, réessayez dans quelques secondes');
      } else if (status === 503) {
        setRecommendationError('Service Ollama indisponible — vérifiez que le modèle est chargé');
      } else {
        setRecommendationError(`Erreur lors des recommandations (${status ?? code ?? 'inconnue'})`);
      }

      console.error('[LLM recommend] Erreur:', {
        status,
        code,
        data: e?.response?.data,
        message: e?.message,
      });
    } finally {
      setRecommendationLoading(false);
    }
  };

  const handleCreateJiraIssue = async () => {
    if (!finding) return;

    setJiraState({ loading: true, error: null, success: false, jiraKey: null, jiraUrl: null });

    const payload = {
      finding_id: finding.id,
      title: finding.title,
      severity: finding.severity.toUpperCase(),
      cvss_score: finding.cvss_score ?? 0,
      description: finding.description ?? '',
      cve: (finding as any).cve ?? '',
      cwe: (finding as any).cwe ?? '',
      file_path: finding.file_path ?? '',
      line: (finding as any).line,
      tags: finding.tags ?? [],
      risk_level: aiScore?.risk_level ?? 'UNKNOWN',
      ai_score: aiScore?.risk_class,
      ai_confidence: aiScore?.confidence,
      engagement_id: finding.engagement_id,
      product_name: finding.product_name,
    };

    try {
      const data = await jiraApi.createIssue(payload);
      setJiraState({
        loading: false,
        error: null,
        success: true,
        jiraKey: data.key,
        jiraUrl: jiraApi.getIssueUrl(data.key),
      });
    } catch (error: any) {
      let errorMessage = 'Erreur lors de la création du ticket Jira';
      if (error.response?.status === 409) {
        const existing = await jiraApi.checkIssue(findingId);
        if (existing.exists && existing.jira_key) {
          setJiraState({
            loading: false,
            error: null,
            success: true,
            jiraKey: existing.jira_key,
            jiraUrl: existing.jira_url,
          });
          return;
        }
        errorMessage = ' Un ticket Jira existe déjà pour ce finding';
      } else if (error.response?.status === 502) {
        errorMessage = ' Backend InvisiThreat indisponible';
      } else if (error.response?.status === 503) {
        errorMessage = ' Service Jira indisponible';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = ' Délai réseau dépassé';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      setJiraState({ loading: false, error: errorMessage, success: false, jiraKey: null, jiraUrl: null });
    }
  };

  const handleResetJira = () => {
    setJiraState({ loading: false, error: null, success: false, jiraKey: null, jiraUrl: null });
  };

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return '#ff4757';
      case 'high':     return '#ff6b35';
      case 'medium':   return '#ffd32a';
      case 'low':      return '#2ed573';
      default:         return '#95a5a6';
    }
  };

  const getSeverityClass = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'sev-crit';
      case 'high':     return 'sev-high';
      case 'medium':   return 'sev-med';
      case 'low':      return 'sev-low';
      default:         return 'sev-info';
    }
  };

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleDateString('fr-FR', {
      year: 'numeric', month: 'long', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  };

  const getRiskMessage = () => {
    const rc = aiScore?.risk_class;
    if (rc === undefined) return 'Analyse IA en cours...';
    if (rc >= 3) return 'Traitement dans les 24h · Priorité absolue';
    if (rc >= 2) return 'Correction recommandée sous 7 jours';
    return 'À surveiller lors des prochaines maintenances';
  };

  if (loadingFinding) return (
    <div className="home-root" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
      <div className="bg-grid" /><div className="bg-radials" /><div className="scan-line" />
      <div style={{ textAlign: 'center', position: 'relative', zIndex: 10 }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.14em', color: 'rgba(255,255,255,0.3)', marginBottom: 12 }}>CHARGEMENT DU FINDING</div>
        <div style={{ width: 200, height: 2, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ height: '100%', background: 'linear-gradient(90deg,#00d4ff,#008fbb)', animation: 'progressBar 2s ease infinite', borderRadius: 2 }} />
        </div>
      </div>
    </div>
  );

  if (error || !finding) return (
    <div className="home-root" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#ff4757', position: 'relative', zIndex: 10 }}>Finding non trouvé</div>
    </div>
  );

  const sevColor = getSeverityColor(finding.severity);

  return (
    <div className="finding-detail-page home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="fdp-page">

        <button className="fdp-back-btn" onClick={() => navigate(-1)}>
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M9 2L4 7l5 5" />
          </svg>
          RETOUR AUX FINDINGS
        </button>

        <header className="fdp-header fu1">
          <div className="fdp-finding-id">FINDING · #{finding.id} · DEVSECOPS PLATFORM</div>
          <h1 className="fdp-title">{finding.title}</h1>
          <div className="fdp-badge-row">
            <span className={`fdp-sev-badge ${getSeverityClass(finding.severity)}`}>
              {finding.severity.toUpperCase()}
            </span>
            <span className="fdp-ai-badge">
              <span className="fdp-ai-dot" />
              Score IA: {aiScore?.risk_class ?? '?'}/4 · {(aiScore?.risk_level || 'N/A').toUpperCase()}
            </span>
            {finding.has_cve && (
              <span className="fdp-sev-badge" style={{ background: 'rgba(255,71,87,.08)', color: 'rgba(255,71,87,.7)', borderColor: 'rgba(255,71,87,.2)' }}>
                CVE PRÉSENT
              </span>
            )}
          </div>
          <div className="fdp-meta-row">
            <span className="fdp-meta-item">
              Engagement <span className="fdp-meta-dot" />
              <span className="fdp-meta-val">{finding.engagement_name || finding.engagement_id || 'N/A'}</span>
            </span>
            <span className="fdp-meta-item">
              Produit <span className="fdp-meta-dot" />
              <span className="fdp-meta-val">{finding.product_name || finding.product_id || 'N/A'}</span>
            </span>
            <span className="fdp-meta-item">
              Créé le <span className="fdp-meta-dot" />
              <span className="fdp-meta-val">{formatDate(finding.created)}</span>
            </span>
          </div>
        </header>

        <div className="fdp-stats-grid fu2">
          <div className="fdp-stat-card" style={{ '--acc-color': '#ff4757' } as React.CSSProperties}>
            <div className="fdp-stat-label">CVSS Score</div>
            <div className="fdp-stat-val" style={{ color: '#ff4757' }}>
              {finding.cvss_score ?? 'N/A'}
              {finding.cvss_score && <span className="fdp-stat-unit">/10</span>}
            </div>
          </div>
          <div className="fdp-stat-card" style={{ '--acc-color': '#ffd32a' } as React.CSSProperties}>
            <div className="fdp-stat-label">Âge</div>
            <div className="fdp-stat-val" style={{ color: '#ffd32a' }}>
              {finding.age_days ?? 'N/A'}
              {finding.age_days && <span className="fdp-stat-unit">j</span>}
            </div>
          </div>
          <div className="fdp-stat-card" style={{ '--acc-color': '#a29bfe' } as React.CSSProperties}>
            <div className="fdp-stat-label">Confiance IA</div>
            <div className="fdp-stat-val" style={{ color: '#a29bfe' }}>
              {aiScore?.confidence ? `${(aiScore.confidence * 100).toFixed(0)}` : 'N/A'}
              {aiScore?.confidence && <span className="fdp-stat-unit">%</span>}
            </div>
          </div>
          <div className="fdp-stat-card" style={{ '--acc-color': '#00d4ff' } as React.CSSProperties}>
            <div className="fdp-stat-label">Score Contexte</div>
            <div className="fdp-stat-val" style={{ color: '#00d4ff' }}>
              {aiScore?.context_score ?? 'N/A'}
              {aiScore?.context_score && <span className="fdp-stat-unit">/10</span>}
            </div>
          </div>
          <div className="fdp-stat-card" style={{ '--acc-color': '#ff4757' } as React.CSSProperties}>
            <div className="fdp-stat-label">CVE</div>
            <div className="fdp-stat-val" style={{ fontSize: 14, marginTop: 4, color: finding.has_cve ? '#ff4757' : 'rgba(255,255,255,0.3)' }}>
              {(finding as any).cve ?? (finding.has_cve ? 'Présent' : 'Non')}
            </div>
          </div>
          <div className="fdp-stat-card" style={{ '--acc-color': '#2ed573' } as React.CSSProperties}>
            <div className="fdp-stat-label">Tags</div>
            <div className="fdp-tags-wrap">
              {finding.tags?.length
                ? finding.tags.map((tag: string) => <span key={tag} className="fdp-tag-pill">{tag}</span>)
                : <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 12 }}>—</span>}
            </div>
          </div>
        </div>

        <div className="fdp-two-col fu3">

          <div className="fdp-card">
            <div className="fdp-card-header">
              <div className="fdp-card-icon" style={{ background: 'rgba(0,212,255,.1)' }}>📄</div>
              <div>
                <div className="fdp-card-title">Fichier concerné</div>
                <div className="fdp-card-subtitle">LOCALISATION DE LA VULNÉRABILITÉ</div>
              </div>
            </div>
            <div className="fdp-file-path-box">
              <code>{finding.file_path ?? 'Non spécifié'}</code>
              {finding.line && <span className="fdp-line-badge">L.{finding.line}</span>}
            </div>
            <div className="fdp-sep" />
            <div className="fdp-card-header" style={{ borderBottom: 'none', paddingBottom: 0, marginBottom: 12 }}>
              <div className="fdp-card-icon" style={{ background: 'rgba(255,255,255,.05)' }}>📝</div>
              <div className="fdp-card-title">Description</div>
            </div>
            <p className="fdp-desc-text">{finding.description ?? 'Aucune description disponible.'}</p>
          </div>

          <div className="fdp-card">
            <div className="fdp-card-header">
              <div className="fdp-card-icon" style={{ background: 'rgba(162,155,254,.1)' }}>🤖</div>
              <div>
                <div className="fdp-card-title">Analyse de risque IA</div>
                <div className="fdp-card-subtitle">DEEPSEEK · MODÈLE PRÉDICTIF v2</div>
              </div>
            </div>
            <div className="fdp-risk-bar" style={{ background: `${sevColor}10`, borderLeftColor: sevColor }}>
              <div>
                <div className="fdp-risk-label">NIVEAU DE RISQUE</div>
                <div className="fdp-risk-val" style={{ color: sevColor }}>
                  {(aiScore?.risk_level || 'Non déterminé').toUpperCase()}
                </div>
              </div>
              <div className="fdp-risk-msg">{getRiskMessage()}</div>
            </div>
            <p className="fdp-risk-text">
              {aiScore?.risk_class !== undefined && aiScore.risk_class >= 3
                ? '⚠️ Cette vulnérabilité présente un risque ÉLEVÉ et doit être traitée en priorité absolue.'
                : aiScore?.risk_class !== undefined && aiScore.risk_class >= 2
                ? '📌 Cette vulnérabilité présente un risque MODÉRÉ. Une correction est recommandée rapidement.'
                : 'ℹ️ Cette vulnérabilité présente un risque FAIBLE. À surveiller lors des prochaines maintenances.'}
              {aiScore?.context_score && aiScore.context_score > 5 &&
                ` Le score de contexte élevé (${aiScore.context_score}/10) indique un environnement critique.`}
            </p>
            <div className="fdp-risk-classes-label">CLASSES DE RISQUE · DISTRIBUTION IA</div>
            <div className="fdp-risk-score-row">
              {['low', 'medium', 'high', 'critical'].map((level, i) => {
                const colors = ['#2ed573', '#ffd32a', '#ff6b35', '#ff4757'];
                const val = scoreResult?.results[0]?.probabilities?.[level] ?? [0.02, 0.04, 0.06, 0.88][i];
                return (
                  <div key={level} className="fdp-risk-score-box"
                    style={level === (aiScore?.risk_level?.toLowerCase()) ? { background: `${colors[i]}08`, borderColor: `${colors[i]}30` } : undefined}>
                    <div className="fdp-risk-score-num" style={{ color: colors[i] }}>
                      {typeof val === 'number' ? val.toFixed(2) : '—'}
                    </div>
                    <div className="fdp-risk-score-name">{level.toUpperCase().slice(0, 3)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* ── JIRA PANEL ── */}
        <div className="jira-result-panel fu3">
          <div className="jira-result-header">
            <div className="jira-result-icon">🔗</div>
            <div>
              <div className="jira-result-title">Export vers Jira</div>
              <div className="jira-result-subtitle">INTÉGRATION AUTOMATIQUE</div>
            </div>
          </div>

          <div className="jira-result-content">
            {jiraState.success && jiraState.jiraKey && jiraState.jiraUrl ? (
              <div className="jira-success-box">
                <div className="jira-success-icon">✓</div>
                <div className="jira-success-text">
                  <p>
                    <strong>Ticket Jira créé avec succès !</strong>
                  </p>
                  <a
                    href={jiraState.jiraUrl || ''}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="jira-success-link"
                  >
                    Ouvrir {jiraState.jiraKey} dans Jira →
                  </a>
                </div>
              </div>
            ) : jiraState.error ? (
              <div className="jira-error-box">
                <div className="jira-error-icon">⚠️</div>
                <div className="jira-error-text">
                  <p>
                    <strong>Erreur lors de la création</strong>
                  </p>
                  <p className="jira-error-message">{jiraState.error}</p>
                  <button 
                    className="jira-retry-btn" 
                    onClick={handleCreateJiraIssue} 
                    disabled={jiraState.loading}
                  >
                    ↺ Réessayer
                  </button>
                </div>
              </div>
            ) : jiraState.loading ? (
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

            {/* Jira Action Button */}
            {jiraState.success && jiraState.jiraUrl && jiraState.jiraKey ? (
              <>
                <a
                  href={jiraState.jiraUrl || ''}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="jira-action-btn success"
                >
                  <span className="jira-icon">✓</span>
                  <span className="jira-text">Ticket Jira: {jiraState.jiraKey}</span>
                  <span className="jira-external">↗</span>
                </a>
                <button
                  onClick={handleResetJira}
                  className="jira-action-btn secondary"
                  style={{ marginTop: 8 }}
                >
                  + Nouveau ticket
                </button>
              </>
            ) : (
              <button
                className={`jira-action-btn ${jiraState.loading ? 'loading' : jiraState.error ? 'error' : 'default'}`}
                onClick={handleCreateJiraIssue}
                disabled={jiraState.loading}
                title={jiraState.error || 'Créer un ticket Jira'}
              >
                {jiraState.loading ? (
                  <>
                    <span className="jira-spinner" />
                    <span className="jira-text">Création...</span>
                  </>
                ) : jiraState.error ? (
                  <>
                    <span className="jira-icon">⚠️</span>
                    <span className="jira-text">{jiraState.error}</span>
                  </>
                ) : (
                  <>
                    <span className="jira-icon">🎫</span>
                    <span className="jira-text">Créer ticket Jira</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* ── LLM Panel: Explication ── */}
        <LLMPanel
          icon="📖"
          iconBg="rgba(0,212,255,.1)"
          title="Explication IA"
          loading={explanationLoading}
          error={explanationError}
          hasResult={!!llmExplanation}
          onAction={handleExplanation}
          actionLabel={llmExplanation ? '↺ Regénérer l\'explication' : '✦ Expliquer avec IA'}
          emptyIcon="🧠"
          emptyText={<>Cliquez sur <strong>"Expliquer avec IA"</strong> pour obtenir une analyse détaillée de la vulnérabilité.</>}
          loadingText="Analyse en cours... (peut prendre 1-2 minutes)"
        >
          {llmExplanation && (
            <>
              {(llmExplanation._fallback || llmExplanation.from_cache) && (
                <div className="fdp-cache-pill">
                  ⚡ {llmExplanation._fallback ? 'Réponse hors-ligne' : 'Cache'}
                </div>
              )}
              <LLMSection icon="📖" iconBg="rgba(0,212,255,.08)" label="EXPLICATION" labelColor="var(--acc)">
                <p className="fdp-llm-sec-text">{llmExplanation.summary}</p>
              </LLMSection>
              <LLMSection icon="💥" iconBg="rgba(255,71,87,.08)" label="IMPACT" labelColor="#ff4757">
                <p className="fdp-llm-sec-text">{llmExplanation.impact}</p>
              </LLMSection>
              <LLMSection icon="⏱" iconBg="rgba(255,211,42,.08)" label="DÉLAI RECOMMANDÉ" labelColor="#ffd32a">
                <p className="fdp-llm-sec-text" style={{ color: '#ffd32a', opacity: 0.85, fontWeight: 500 }}>
                  {llmExplanation.priority_note}
                </p>
              </LLMSection>
            </>
          )}
        </LLMPanel>

        {/* ── LLM Panel: Recommandations ── */}
        <LLMPanel
          icon="🔧"
          iconBg="rgba(46,213,115,.1)"
          title="Recommandations IA"
          loading={recommendationLoading}
          error={recommendationError}
          hasResult={!!llmRecommendation}
          onAction={handleRecommendation}
          actionLabel={llmRecommendation ? '↺ Regénérer les recommandations' : '✦ Recommandations IA'}
          emptyIcon="🛠️"
          emptyText={<>Cliquez sur <strong>"Recommandations IA"</strong> pour obtenir des étapes de correction détaillées.</>}
          loadingText="Génération des recommandations... (peut prendre 1-2 minutes)"
        >
          {llmRecommendation && (
            <>
              {(llmRecommendation._fallback || llmRecommendation.from_cache) && (
                <div className="fdp-cache-pill">
                  ⚡ {llmRecommendation._fallback ? 'Réponse hors-ligne' : 'Cache'}
                </div>
              )}
              <LLMSection icon="🔧" iconBg="rgba(46,213,115,.08)" label="ÉTAPES DE REMÉDIATION" labelColor="#2ed573">
                <ol className="fdp-llm-steps">
                  {(llmRecommendation.recommendations || []).map((step: string, i: number) => (
                    <li key={i}>
                      <span className="fdp-step-num">{i + 1}</span>
                      {step}
                    </li>
                  ))}
                </ol>
              </LLMSection>
              {llmRecommendation.references && llmRecommendation.references.length > 0 && (
                <div className="fdp-llm-refs">
                  <div className="fdp-llm-sec-label" style={{ color: 'rgba(255,255,255,.25)', marginBottom: 10 }}>RÉFÉRENCES</div>
                  {llmRecommendation.references.map((ref: string, i: number) => (
                    <a key={i} href={ref} target="_blank" rel="noopener noreferrer" className="fdp-ref-link">
                      🔗 {ref}
                    </a>
                  ))}
                </div>
              )}
            </>
          )}
        </LLMPanel>

        <div className="fdp-card fu5">
          <div className="fdp-card-header">
            <div className="fdp-card-icon" style={{ background: 'rgba(255,255,255,.05)' }}>🔧</div>
            <div>
              <div className="fdp-card-title">Informations techniques</div>
              <div className="fdp-card-subtitle">MÉTADONNÉES SYSTÈME</div>
            </div>
          </div>
          <div className="fdp-tech-grid">
            <div className="fdp-tech-item"><strong>ID UNIQUE</strong><code>#{finding.id}</code></div>
            <div className="fdp-tech-item"><strong>ENGAGEMENT ID</strong><code>{finding.engagement_id ?? 'N/A'}</code></div>
            <div className="fdp-tech-item"><strong>PRODUIT ID</strong><code>{finding.product_id ?? 'N/A'}</code></div>
            <div className="fdp-tech-item"><strong>DATE CRÉATION</strong><code>{formatDate(finding.created)}</code></div>
          </div>
        </div>

      </div>
    </div>
  );
}


interface LLMPanelProps {
  icon: string;
  iconBg: string;
  title: string;
  loading: boolean;
  error: string | null;
  hasResult: boolean;
  onAction: () => void;
  actionLabel: string;
  emptyIcon: string;
  emptyText: React.ReactNode;
  loadingText: string;
  children?: React.ReactNode;
}

function LLMPanel({ icon, iconBg, title, loading, error, hasResult, onAction, actionLabel, emptyIcon, emptyText, loadingText, children }: LLMPanelProps) {
  const showEmpty = !loading && !error && !hasResult;
  return (
    <div className="fdp-llm-panel fu4">
      <div className="fdp-llm-header">
        <div className="fdp-llm-title-group">
          <div className="fdp-llm-icon-box" style={{ background: iconBg }}>{icon}</div>
          <div className="fdp-llm-h">{title}</div>
          <span className="fdp-llm-model-pill">deepseek-coder · Ollama</span>
        </div>
        <button
          className={`fdp-llm-btn${hasResult ? ' regen' : ''}`}
          onClick={onAction}
          disabled={loading}
        >
          {loading ? (
            <><span className="fdp-spinner" /> Génération...</>
          ) : (
            actionLabel
          )}
        </button>
      </div>

      {showEmpty && (
        <div className="fdp-llm-empty">
          <div className="fdp-llm-empty-ico">{emptyIcon}</div>
          <p>{emptyText}</p>
        </div>
      )}

      {loading && (
        <div className="fdp-llm-loading">
          <div className="fdp-progress-track"><div className="fdp-progress-fill" /></div>
          <p>{loadingText}</p>
        </div>
      )}

      {error && !loading && (
        <div className="fdp-error-box">
          ⚠️ {error}
          <button
            style={{ marginLeft: 12, fontSize: 11, opacity: 0.6, cursor: 'pointer', background: 'none', border: '1px solid currentColor', borderRadius: 4, padding: '2px 8px', color: 'inherit' }}
            onClick={onAction}
          >
            Réessayer
          </button>
        </div>
      )}

      {hasResult && !loading && (
        <div className="fdp-llm-body">{children}</div>
      )}
    </div>
  );
}

interface LLMSectionProps {
  icon: string;
  iconBg: string;
  label: string;
  labelColor: string;
  children: React.ReactNode;
}

function LLMSection({ icon, iconBg, label, labelColor, children }: LLMSectionProps) {
  return (
    <div className="fdp-llm-section">
      <div className="fdp-llm-sec-icon" style={{ background: iconBg }}>{icon}</div>
      <div style={{ flex: 1 }}>
        <div className="fdp-llm-sec-label" style={{ color: labelColor }}>{label}</div>
        {children}
      </div>
    </div>
  );
}