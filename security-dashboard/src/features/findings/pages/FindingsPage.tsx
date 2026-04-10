import { useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';
import { predictionsApi, PredictionRequest } from '../../../api/services/predictions';
import './FindingsPage.css';

export default function FindingsPage() {
  const [searchParams] = useSearchParams();
  const engagementId = searchParams.get('engagementId');
  const parsedId = engagementId ? parseInt(engagementId, 10) : null;

  // 1. Récupérer les findings bruts
  const { data: rawFindings = [], isLoading: loadingFindings } = useQuery({
    queryKey: ['findings', parsedId],
    queryFn: () => findingsApi.getByEngagement(parsedId!).then(res => res.data),
    enabled: !!parsedId,
  });

  // 2. Préparer la requête batch
  const findingsForBatch: PredictionRequest[] = rawFindings.map(f => ({
    severity: f.severity,
    cvss_score: f.cvss_score || 0,
    title: f.title,
    tags: f.tags,
    days_open: 0,
    finding_id: f.id,
    engagement_id: f.engagement_id ?? parsedId,
  }));

  // 3. Obtenir les scores IA
  const { data: batchResults, isLoading: loadingScores } = useQuery({
    queryKey: ['scores', rawFindings],
    queryFn: () => predictionsApi.predictBatch(findingsForBatch).then(res => res.data),
    enabled: rawFindings.length > 0,
  });

  if (!parsedId) return <div className="error">Aucun engagement sélectionné</div>;
  if (loadingFindings) return <div className="loading">Chargement des findings...</div>;
  if (rawFindings.length === 0) return <div className="error">Aucun finding trouvé</div>;
  if (loadingScores) return <div className="loading">Calcul des scores IA...</div>;

  // Fusion
  const findings = rawFindings.map((f, idx) => ({
    ...f,
    ai_risk_class: batchResults?.results[idx]?.risk_class,
    ai_risk_level: batchResults?.results[idx]?.risk_level,
    ai_confidence: batchResults?.results[idx]?.confidence,
  }));

  return (
    <div className="findings-page">
      <div className="section-header">
        <div className="section-label">FINDINGS</div>
        <h2 className="section-title">Vulnérabilités – Engagement #{parsedId}</h2>
      </div>
      <div className="findings-table-container">
        <table className="findings-table">
          <thead><tr><th>ID</th><th>Titre</th><th>Sévérité</th><th>CVSS</th><th>Tags</th><th>Score IA</th><th>Confiance</th></tr></thead>
          <tbody>
            {findings.map(f => (
              <tr key={f.id} onClick={() => window.location.href = `/findings/${f.id}`}>
                <td>{f.id}</td>
                <td>{f.title}</td>
                <td><span className={`severity-badge ${f.severity}`}>{f.severity}</span></td>
                <td>{f.cvss_score}</td>
                <td><div className="tags-list">{f.tags.slice(0,3).map(t => <span key={t} className="tag-pill">{t}</span>)}</div></td>
                <td>{f.ai_risk_class !== undefined ? `${f.ai_risk_class}/4` : 'N/A'}</td>
                <td>{f.ai_confidence ? `${(f.ai_confidence * 100).toFixed(0)}%` : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}