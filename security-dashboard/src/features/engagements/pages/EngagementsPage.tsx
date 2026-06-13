import { Link } from 'react-router-dom';
import { useHashedProductId } from '../../../utils/useHashedParams';
import { encodeId } from '../../../utils/hashId';
import { useEngagements } from '../hooks/useEngagements';
import './EngagementsPage.css';

function IconSearch() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.6"
      strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

export default function EngagementsPage() {
  const { productId } = useHashedProductId();
  const { data: engagements, isLoading, error } = useEngagements(productId ?? 0);

  if (!productId) return (
    <div className="ep-state ep-state--error">Aucun produit sélectionné.</div>
  );
  if (isLoading) return (
    <div className="ep-state ep-state--loading">
      <div className="ep-spinner" />
      Chargement des engagements...
    </div>
  );
  if (error) return (
    <div className="ep-state ep-state--error">Erreur : {error.message}</div>
  );

  return (
    <div className="engagements-page home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="ep-inner">
        <div className="section-header">
          <div className="section-label">
            <span className="fp-label-dot" />
            ENGAGEMENTS
          </div>
          <h2 className="section-title">
            Scans pour le <span>produit #{productId}</span>
          </h2>
          <p className="section-subtitle">
            Choisissez un engagement pour voir ses vulnérabilités.
          </p>
        </div>

        <div className="engagements-grid">
          {engagements?.map((eng) => (
            <Link
              key={eng.id}
              to={`/findings?engagementId=${encodeId(eng.id)}`}
              className="engagement-card"
            >
              <div className="engagement-card-icon">
                <IconSearch />
              </div>
              <div className="engagement-card-name">{eng.name}</div>
              <div className="engagement-card-meta">
                <span>Statut : {eng.status}</span>
                <span>Créé le {new Date(eng.created!).toLocaleDateString('fr-FR')}</span>
                
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}