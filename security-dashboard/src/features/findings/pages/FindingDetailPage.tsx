import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';
import { predictionsApi, PredictionRequest } from '../../../api/services/predictions';
import { explanationsApi, LLMExplanation, LLMRecommendation } from '../../../api/services/explanations';
import { jiraApi } from '../../../api/services/jira';
import type { JiraIntegrationState } from '../../../types/jira';

import './FindingDetailPage.css';

const getSeverityColor = (severity: string) => {
  switch (severity?.toLowerCase()) {
    case 'critical': return '#ff3b5c';
    case 'high':     return '#ff6b35';
    case 'medium':   return '#f5a623';
    case 'low':      return '#00e87a';
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
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
};

export default function FindingDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const findingId = parseInt(id || '0', 10);

  const [llmExplanation, setLlmExplanation]       = useState<LLMExplanation | null>(null);
  const [llmRecommendation, setLlmRecommendation] = useState<LLMRecommendation | null>(null);
  const [explanationLoading, setExplanationLoading]       = useState(false);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [explanationError, setExplanationError]       = useState<string | null>(null);
  const [recommendationError, setRecommendationError] = useState<string | null>(null);
  const [descExpanded, setDescExpanded] = useState(false);

  const [jiraState, setJiraState] = useState<JiraIntegrationState>({
    loading: false, error: null, success: false, jiraKey: null, jiraUrl: null,
  });

  useEffect(() => {
    if (!findingId) return;
    const checkExistingTicket = async () => {
      try {
        const result = await jiraApi.checkIssue(findingId);
        if (result.exists && result.jira_key) {
          setJiraState({ loading: false, error: null, success: true, jiraKey: result.jira_key, jiraUrl: result.jira_url });
        }
      } catch (err) {
        console.error('[Jira] Erreur vérification initiale:', err);
      }
    };
    checkExistingTicket();
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
        setExplanationError("Réponse LLM invalide — champ 'summary' manquant");
        return;
      }
      setLlmExplanation(data);
    } catch (e: any) {
      const status = e?.response?.status;
      const code   = e?.code;
      if      (status === 404)          setExplanationError('Finding introuvable dans DefectDojo');
      else if (code === 'ECONNABORTED') setExplanationError('Délai dépassé — réessayez dans quelques secondes');
      else if (status === 503)          setExplanationError('Service Ollama indisponible');
      else                              setExplanationError(`Erreur lors de l'explication (${status ?? code ?? 'inconnue'})`);
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
        setRecommendationError("Réponse LLM invalide — champ 'recommendations' manquant");
        return;
      }
      setLlmRecommendation(data);
    } catch (e: any) {
      const status = e?.response?.status;
      const code   = e?.code;
      if      (status === 404)          setRecommendationError('Finding introuvable dans DefectDojo');
      else if (code === 'ECONNABORTED') setRecommendationError('Délai dépassé — réessayez dans quelques secondes');
      else if (status === 503)          setRecommendationError('Service Ollama indisponible');
      else                              setRecommendationError(`Erreur lors des recommandations (${status ?? code ?? 'inconnue'})`);
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
      setJiraState({ loading: false, error: null, success: true, jiraKey: data.key, jiraUrl: jiraApi.getIssueUrl(data.key) });
    } catch (error: any) {
      let errorMessage = 'Erreur lors de la création du ticket Jira';
      if (error.response?.status === 409) {
        const existing = await jiraApi.checkIssue(findingId);
        if (existing.exists && existing.jira_key) {
          setJiraState({ loading: false, error: null, success: true, jiraKey: existing.jira_key, jiraUrl: existing.jira_url });
          return;
        }
        errorMessage = 'Un ticket Jira existe déjà pour ce finding';
      } else if (error.response?.status === 502) {
        errorMessage = 'Backend InvisiThreat indisponible';
      } else if (error.response?.status === 503) {
        errorMessage = 'Service Jira indisponible';
      } else if (error.code === 'ECONNABORTED') {
        errorMessage = 'Délai réseau dépassé';
      } else if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      setJiraState({ loading: false, error: errorMessage, success: false, jiraKey: null, jiraUrl: null });
    }
  };

  const handleResetJira = () => {
    setJiraState({ loading: false, error: null, success: false, jiraKey: null, jiraUrl: null });
  };

  const getRiskMessage = () => {
    const rc = aiScore?.risk_class;
    if (rc === undefined) return 'Analyse IA en cours...';
    if (rc >= 3) return 'Priorité absolue · 24h';
    if (rc >= 2) return 'Correction sous 7 jours';
    return 'Surveillance planifiée';
  };

  if (loadingFinding) return (
    <div className="home-root fdp-loading-screen">
      <div className="bg-grid" /><div className="bg-radials" /><div className="scan-line" />
      <div className="fdp-loading-inner">
        <div className="fdp-loading-label">CHARGEMENT DU FINDING</div>
        <div className="fdp-loading-track"><div className="fdp-loading-fill" /></div>
      </div>
    </div>
  );

  if (error || !finding) return (
    <div className="home-root fdp-loading-screen">
      <div className="bg-grid" /><div className="bg-radials" />
      <div className="fdp-error-screen">
        <span>⚠</span> Finding non trouvé
      </div>
    </div>
  );

  const sevColor = getSeverityColor(finding.severity);
  const description = finding.description ?? '';
  const descTruncated = description.length > 320;

  return (
    <div className="finding-detail-page home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="fdp-page fdp-page-v2">

        <div className="fdp-topbar fu">
          <button className="fdp-back-btn" onClick={() => navigate(-1)}>
            <svg width="13" height="13" viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
              <path d="M9 2L4 7l5 5" />
            </svg>
            Retour aux findings
          </button>
          <div className="fdp-breadcrumb">
            <span>Findings</span>
            <span className="fdp-bc-sep">/</span>
            <span className="fdp-bc-active">#{finding.id}</span>
          </div>
        </div>

        <header className="fdp-hero fu1">
          <div className="fdp-hero-eyebrow">
            <span className="fdp-finding-ref">FINDING · #{finding.id}</span>
            <span className="fdp-hero-divider" />
            <span className="fdp-platform-tag">DEVSECOPS PLATFORM</span>
          </div>

          <h1 className="fdp-title">{finding.title}</h1>

          <div className="fdp-hero-footer">
            <div className="fdp-badge-row">
              <span className={`fdp-sev-badge ${getSeverityClass(finding.severity)}`}>
                {finding.severity.toUpperCase()}
              </span>
              <span className="fdp-ai-badge">
                <span className="fdp-ai-dot" />
                Score IA {aiScore?.risk_class ?? '?'}/4 · {(aiScore?.risk_level || 'N/A').toUpperCase()}
              </span>
              {finding.has_cve && (
                <span className="fdp-sev-badge sev-cve">CVE</span>
              )}
            </div>

            <div className="fdp-meta-row">
              <MetaItem label="Engagement" value={String(finding.engagement_name || finding.engagement_id || 'N/A')} />
              <MetaItem label="Produit"    value={String(finding.product_name    || finding.product_id    || 'N/A')} />
              <MetaItem label="Créé le"   value={formatDate(finding.created)} />
            </div>
          </div>
        </header>

        <div className="fdp-stats-grid fu2">
          <StatCard label="CVSS Score"     value={finding.cvss_score ?? 'N/A'} unit="/10"  color="#ff3b5c" />
          <StatCard label="Âge"            value={finding.age_days ?? 'N/A'}   unit="jours" color="#f5a623" />
          <StatCard label="Confiance IA"   value={aiScore?.confidence ? `${(aiScore.confidence * 100).toFixed(0)}` : 'N/A'} unit="%" color="#b48eff" />
          <StatCard label="Score Contexte" value={aiScore?.context_score ?? 'N/A'} unit="/10" color="#00d4ff" />
          <StatCard label="CVE"            value={(finding as any).cve ?? (finding.has_cve ? 'Présent' : '—')} color={finding.has_cve ? '#ff3b5c' : undefined} textSize />
          <div className="fdp-stat-card" style={{ '--acc-color': '#00e87a' } as React.CSSProperties}>
            <div className="fdp-stat-label">TAGS</div>
            <div className="fdp-tags-wrap">
              {finding.tags?.length
                ? finding.tags.map((tag: string) => <span key={tag} className="fdp-tag-pill">{tag}</span>)
                : <span className="fdp-empty-val">—</span>}
            </div>
          </div>
        </div>

        <div className="fdp-main-grid fu3">

          <div className="fdp-col-primary">

            <div className="fdp-card fdp-card-file">
              <div className="fdp-card-header">
                <div className="fdp-card-icon" style={{ background: 'rgba(0,212,255,.09)' }}>
                  <svg width="14" height="14" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.4">
                    <path d="M3 2h6l3 3v8H3V2z" strokeLinecap="round" strokeLinejoin="round"/>
                    <path d="M9 2v3h3" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div>
                  <div className="fdp-card-title">Fichier concerné</div>
                  <div className="fdp-card-subtitle">LOCALISATION DE LA VULNÉRABILITÉ</div>
                </div>
              </div>
              <div className="fdp-file-path-box">
                <code className="fdp-filepath-code">{finding.file_path ?? 'Non spécifié'}</code>
                {(finding as any).line && <span className="fdp-line-badge">L.{(finding as any).line}</span>}
              </div>
            </div>

            <div className="fdp-card fdp-card-desc">
              <div className="fdp-card-header">
                <div className="fdp-card-icon" style={{ background: 'rgba(255,255,255,.055)' }}>
                  <svg width="14" height="14" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.4">
                    <path d="M2 4h11M2 7h9M2 10h7" strokeLinecap="round"/>
                  </svg>
                </div>
                <div className="fdp-card-title">Description</div>
              </div>

              <div className={`fdp-desc-wrapper${descExpanded ? ' expanded' : ''}`}>
                <p className="fdp-desc-text">{description || 'Aucune description disponible.'}</p>
                {!descExpanded && descTruncated && <div className="fdp-desc-fade" />}
              </div>

              {descTruncated && (
                <button className="fdp-show-more-btn" onClick={() => setDescExpanded(v => !v)}>
                  {descExpanded
                    ? <><ChevronUp /> Voir moins</>
                    : <><ChevronDown /> Voir plus</>}
                </button>
              )}
            </div>
          </div>

          <div className="fdp-col-risk">
            <div className="fdp-card fdp-card-risk">
              <div className="fdp-card-header">
                <div className="fdp-card-icon" style={{ background: 'rgba(180,142,255,.1)' }}>
                  <svg width="14" height="14" viewBox="0 0 15 15" fill="none" stroke="#b48eff" strokeWidth="1.4">
                    <circle cx="7.5" cy="7.5" r="5.5" />
                    <path d="M7.5 4.5v3.5M7.5 10h.01" strokeLinecap="round"/>
                  </svg>
                </div>
                <div>
                  <div className="fdp-card-title">Analyse de risque IA</div>
                  <div className="fdp-card-subtitle">DEEPSEEK · MODÈLE PRÉDICTIF v2</div>
                </div>
              </div>

              <div className="fdp-risk-bar" style={{ background: `${sevColor}0c`, borderLeftColor: sevColor }}>
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
                  ? '⚠️ Risque ÉLEVÉ — traitement en priorité absolue requis.'
                  : aiScore?.risk_class !== undefined && aiScore.risk_class >= 2
                  ? '📌 Risque MODÉRÉ — correction recommandée rapidement.'
                  : 'ℹ️ Risque FAIBLE — à surveiller lors des prochaines maintenances.'}
                {aiScore?.context_score && aiScore.context_score > 5 &&
                  ` Score de contexte élevé (${aiScore.context_score}/10) — environnement critique.`}
              </p>

              <div className="fdp-risk-classes-label">DISTRIBUTION DES CLASSES</div>
              <div className="fdp-risk-score-row">
                {(['low','medium','high','critical'] as const).map((level, i) => {
                  const colors   = ['#00e87a','#f5a623','#ff6b35','#ff3b5c'];
                  const fallback = [0.02,0.04,0.06,0.88][i];
                  const val      = scoreResult?.results[0]?.probabilities?.[level] ?? fallback;
                  const active   = level === aiScore?.risk_level?.toLowerCase();
                  return (
                    <div key={level} className={`fdp-risk-score-box${active ? ' active' : ''}`}
                      style={{ '--rc': colors[i] } as React.CSSProperties}>
                      <div className="fdp-risk-score-num" style={{ color: colors[i] }}>
                        {typeof val === 'number' ? val.toFixed(2) : '—'}
                      </div>
                      <div className="fdp-risk-score-name">{level.toUpperCase().slice(0,3)}</div>
                      {active && <div className="fdp-risk-score-active-bar" style={{ background: colors[i] }} />}
                    </div>
                  );
                })}
              </div>
            </div>

            <JiraPanel
              jiraState={jiraState}
              onCreateIssue={handleCreateJiraIssue}
              onReset={handleResetJira}
            />
          </div>
        </div>

        <div className="fdp-ai-grid fu4">

          <LLMPanel
            icon={
              <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="#00d4ff" strokeWidth="1.4">
                <circle cx="8" cy="8" r="6" /><path d="M8 5v3.5l2.5 1.5" strokeLinecap="round"/>
              </svg>
            }
            iconBg="rgba(0,212,255,.09)"
            accentColor="#00d4ff"
            title="Explication IA"
            loading={explanationLoading}
            error={explanationError}
            hasResult={!!llmExplanation}
            onAction={handleExplanation}
            actionLabel={llmExplanation ? '↺ Regénérer' : '✦ Expliquer avec IA'}
            emptyText={<>Cliquez sur <strong>«&nbsp;Expliquer avec IA&nbsp;»</strong> pour obtenir une analyse détaillée.</>}
            loadingText="Analyse en cours... (1–2 min)"
          >
            {llmExplanation && (
              <>
                {(llmExplanation._fallback || llmExplanation.from_cache) && (
                  <div className="fdp-cache-pill">
                    ⚡ {llmExplanation._fallback ? 'Réponse hors-ligne' : 'Depuis le cache'}
                  </div>
                )}
                <div className="fdp-llm-sections">
                  <LLMSection icon="📖" iconBg="rgba(0,212,255,.07)"  label="EXPLICATION"      labelColor="var(--accent)">
                    <p className="fdp-llm-sec-text">{llmExplanation.summary}</p>
                  </LLMSection>
                  <LLMSection icon="💥" iconBg="rgba(255,59,92,.07)"  label="IMPACT"           labelColor="#ff3b5c">
                    <p className="fdp-llm-sec-text">{llmExplanation.impact}</p>
                  </LLMSection>
                  <LLMSection icon="⏱" iconBg="rgba(245,166,35,.07)" label="DÉLAI RECOMMANDÉ" labelColor="#f5a623">
                    <p className="fdp-llm-sec-text" style={{ color: '#f5a623', opacity: .82, fontWeight: 500 }}>
                      {llmExplanation.priority_note}
                    </p>
                  </LLMSection>
                </div>
              </>
            )}
          </LLMPanel>

          <LLMPanel
            icon={
              <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="#00e87a" strokeWidth="1.4">
                <path d="M3 8.5l3 3 7-7" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            }
            iconBg="rgba(0,232,122,.09)"
            accentColor="#00e87a"
            title="Recommandations IA"
            loading={recommendationLoading}
            error={recommendationError}
            hasResult={!!llmRecommendation}
            onAction={handleRecommendation}
            actionLabel={llmRecommendation ? '↺ Regénérer' : '✦ Recommandations IA'}
            emptyText={<>Cliquez sur <strong>«&nbsp;Recommandations IA&nbsp;»</strong> pour obtenir les étapes de correction.</>}
            loadingText="Génération des recommandations... (1–2 min)"
          >
            {llmRecommendation && (
              <>
                {(llmRecommendation._fallback || llmRecommendation.from_cache) && (
                  <div className="fdp-cache-pill">
                    ⚡ {llmRecommendation._fallback ? 'Réponse hors-ligne' : 'Depuis le cache'}
                  </div>
                )}
                <div className="fdp-llm-sections">
                  <LLMSection icon="🔧" iconBg="rgba(0,232,122,.07)" label="ÉTAPES DE REMÉDIATION" labelColor="#00e87a">
                    <ol className="fdp-llm-steps">
                      {(llmRecommendation.recommendations || []).map((step: string, i: number) => (
                        <li key={i}>
                          <span className="fdp-step-num">{i + 1}</span>
                          {step}
                        </li>
                      ))}
                    </ol>
                  </LLMSection>
                  {(llmRecommendation.references ?? []).length > 0 && (
                    <div className="fdp-llm-refs">
                      <div className="fdp-llm-sec-label" style={{ color: 'rgba(255,255,255,.22)', marginBottom: 9 }}>RÉFÉRENCES</div>
                      {(llmRecommendation.references ?? []).map((ref: string, i: number) => (
                        <a key={i} href={ref} target="_blank" rel="noopener noreferrer" className="fdp-ref-link">
                          🔗 {ref}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </LLMPanel>

        </div>

        <div className="fdp-card fu5">
          <div className="fdp-card-header">
            <div className="fdp-card-icon" style={{ background: 'rgba(255,255,255,.045)' }}>
              <svg width="14" height="14" viewBox="0 0 15 15" fill="none" stroke="currentColor" strokeWidth="1.4">
                <circle cx="7.5" cy="7.5" r="5.5" />
                <path d="M7.5 5v3.5l2 1.5" strokeLinecap="round"/>
              </svg>
            </div>
            <div>
              <div className="fdp-card-title">Informations techniques</div>
              <div className="fdp-card-subtitle">MÉTADONNÉES SYSTÈME</div>
            </div>
          </div>
          <div className="fdp-tech-grid">
            <TechItem label="ID UNIQUE"     value={`#${finding.id}`} />
            <TechItem label="ENGAGEMENT ID" value={String(finding.engagement_id ?? 'N/A')} />
            <TechItem label="PRODUIT ID"    value={String(finding.product_id    ?? 'N/A')} />
            <TechItem label="DATE CRÉATION" value={formatDate(finding.created)} />
          </div>
        </div>

      </div>
    </div>
  );
}

function ChevronDown() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M2 4l4 4 4-4" />
    </svg>
  );
}

function ChevronUp() {
  return (
    <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M2 8l4-4 4 4" />
    </svg>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  textSize?: boolean;
}

function StatCard({ label, value, unit, color, textSize }: StatCardProps) {
  const style = color ? { '--acc-color': color } as React.CSSProperties : undefined;
  return (
    <div className="fdp-stat-card" style={style}>
      <div className="fdp-stat-label">{label}</div>
      <div className="fdp-stat-val" style={color ? { color } : undefined}>
        <span style={textSize ? { fontSize: 13.5, marginTop: 3, display: 'block' } : undefined}>
          {value}
        </span>
        {unit && value !== 'N/A' && value !== '—' && (
          <span className="fdp-stat-unit">{unit}</span>
        )}
      </div>
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <span className="fdp-meta-item">
      {label} <span className="fdp-meta-dot" /> <span className="fdp-meta-val">{value}</span>
    </span>
  );
}

function TechItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="fdp-tech-item">
      <strong>{label}</strong>
      <code>{value}</code>
    </div>
  );
}

interface JiraPanelProps {
  jiraState: JiraIntegrationState;
  onCreateIssue: () => void;
  onReset: () => void;
}

function JiraPanel({ jiraState, onCreateIssue, onReset }: JiraPanelProps) {
  return (
    <div className="fdp-jira-panel">
      <div className="fdp-jira-header">
        <div className="fdp-jira-title-row">
          <div className="fdp-jira-logo">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" style={{ color: '#4d9fff' }}>
              <path d="M11.53 2.195a.694.694 0 00-.98 0L5.83 6.913a.694.694 0 000 .981l4.72 4.718a.694.694 0 00.98 0l4.72-4.718a.694.694 0 000-.981L11.53 2.195zm0 9.198a.694.694 0 00-.98 0L5.83 16.11a.694.694 0 000 .982l4.72 4.718a.694.694 0 00.98 0l4.72-4.718a.694.694 0 000-.982l-4.72-4.717z" />
            </svg>
          </div>
          <div>
            <div className="fdp-jira-title">Intégration Jira</div>
            <div className="fdp-jira-subtitle">EXPORT AUTOMATIQUE</div>
          </div>
        </div>
        {jiraState.success && <span className="fdp-jira-status-pill success">✓ Lié</span>}
        {jiraState.error   && <span className="fdp-jira-status-pill error">✕ Erreur</span>}
      </div>

      <div className="fdp-jira-body">
        {jiraState.success && jiraState.jiraKey && jiraState.jiraUrl ? (
          <div className="fdp-jira-success">
            <div className="fdp-jira-ticket-display">
              <div className="fdp-jira-ticket-key">{jiraState.jiraKey}</div>
              <div className="fdp-jira-ticket-label">Ticket créé avec succès</div>
            </div>
            <div className="fdp-jira-actions">
              <a href={jiraState.jiraUrl} target="_blank" rel="noopener noreferrer" className="fdp-jira-btn primary">
                <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M2 10L10 2M10 2H5M10 2v5" strokeLinecap="round" />
                </svg>
                Ouvrir dans Jira
              </a>
              <button className="fdp-jira-btn secondary" onClick={onReset}>+ Nouveau</button>
            </div>
          </div>
        ) : jiraState.error ? (
          <div className="fdp-jira-error-state">
            <div className="fdp-jira-error-msg">{jiraState.error}</div>
            <button className="fdp-jira-btn primary" onClick={onCreateIssue} disabled={jiraState.loading}>
              ↺ Réessayer
            </button>
          </div>
        ) : jiraState.loading ? (
          <div className="fdp-jira-loading-state">
            <div className="fdp-jira-spinner" />
            <span>Création en cours...</span>
          </div>
        ) : (
          <div className="fdp-jira-idle">
            <p className="fdp-jira-hint">Exportez ce finding directement vers votre board Jira en un clic.</p>
            <button className="fdp-jira-btn primary full" onClick={onCreateIssue}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                <path d="M11.53 2.195a.694.694 0 00-.98 0L5.83 6.913a.694.694 0 000 .981l4.72 4.718a.694.694 0 00.98 0l4.72-4.718a.694.694 0 000-.981L11.53 2.195zm0 9.198a.694.694 0 00-.98 0L5.83 16.11a.694.694 0 000 .982l4.72 4.718a.694.694 0 00.98 0l4.72-4.718a.694.694 0 000-.982l-4.72-4.717z" />
              </svg>
              Créer un ticket Jira
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

interface LLMPanelProps {
  icon: React.ReactNode;
  iconBg: string;
  accentColor: string;
  title: string;
  loading: boolean;
  error: string | null;
  hasResult: boolean;
  onAction: () => void;
  actionLabel: string;
  emptyText: React.ReactNode;
  loadingText: string;
  children?: React.ReactNode;
}

function LLMPanel({ icon, iconBg, accentColor, title, loading, error, hasResult, onAction, actionLabel, emptyText, loadingText, children }: LLMPanelProps) {
  const showEmpty = !loading && !error && !hasResult;
  return (
    <div className="fdp-llm-panel" style={{ '--llm-accent': accentColor } as React.CSSProperties}>
      <div className="fdp-llm-header">
        <div className="fdp-llm-title-group">
          <div className="fdp-llm-icon-box" style={{ background: iconBg }}>{icon}</div>
          <div>
            <div className="fdp-llm-h">{title}</div>
            <span className="fdp-llm-model-pill">deepseek-coder · Ollama</span>
          </div>
        </div>
        <button
          className={`fdp-llm-btn${hasResult ? ' regen' : ''}`}
          style={{ '--btn-accent': accentColor } as React.CSSProperties}
          onClick={onAction}
          disabled={loading}
        >
          {loading ? <><span className="fdp-spinner" /> Génération...</> : actionLabel}
        </button>
      </div>

      {showEmpty && (
        <div className="fdp-llm-empty">
          <div className="fdp-llm-empty-icon" style={{ color: accentColor }}>
            {icon}
          </div>
          <p>{emptyText}</p>
        </div>
      )}

      {loading && (
        <div className="fdp-llm-loading">
          <div className="fdp-progress-track">
            <div className="fdp-progress-fill" style={{ '--pf-color': accentColor } as React.CSSProperties} />
          </div>
          <p>{loadingText}</p>
        </div>
      )}

      {error && !loading && (
        <div className="fdp-error-box">
          ⚠️ {error}
          <button className="fdp-error-retry" onClick={onAction}>Réessayer</button>
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