import { useSearchParams, Link } from 'react-router-dom';
import { useEngagements } from '../hooks/useEngagements';
import './EngagementsPage.css';

export default function EngagementsPage() {
  const [searchParams] = useSearchParams();
  const productId = searchParams.get('productId');
  const { data: engagements, isLoading, error } = useEngagements(Number(productId));

  if (!productId) return <div className="error">Aucun produit sélectionné.</div>;
  if (isLoading) return <div className="loading">Chargement des engagements...</div>;
  if (error) return <div className="error">Erreur : {error.message}</div>;

  return (
    <div className="engagements-page">
      <div className="section-header">
        <div className="section-label">ENGAGEMENTS</div>
        <h2 className="section-title">Scans pour le <span>produit #{productId}</span></h2>
        <p className="section-subtitle">Choisissez un engagement pour voir ses vulnérabilités.</p>
      </div>

      <div className="engagements-grid">
        {engagements?.map((eng) => (
          <Link
            key={eng.id}
            to={`/findings?engagementId=${eng.id}`}
            className="engagement-card"
          >
            <div className="engagement-card-icon">🔍</div>
            <div className="engagement-card-name">{eng.name}</div>
            <div className="engagement-card-meta">
              <span>Statut : {eng.status}</span>
              <span>Créé le {new Date(eng.created!).toLocaleDateString()}</span>
              <span>{eng.findings_count ?? 0} findings</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}