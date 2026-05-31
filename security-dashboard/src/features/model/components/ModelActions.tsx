// frontend/src/features/model/components/ModelActions.tsx
import { useState } from 'react';

interface ModelActionsProps {
  onReload: () => Promise<boolean>;
  isReloading: boolean;
}

export function ModelActions({ onReload, isReloading }: ModelActionsProps) {
  const [showConfirm, setShowConfirm] = useState(false);
  const [successMsg, setSuccessMsg]   = useState('');
  const [isError, setIsError]         = useState(false);

  const handleReload = async () => {
    setShowConfirm(false);
    const success = await onReload();
    setIsError(!success);
    if (success) {
      setSuccessMsg('Modèle rechargé avec succès');
      setTimeout(() => setSuccessMsg(''), 3000);
    } else {
      setSuccessMsg('Erreur lors du rechargement');
      setTimeout(() => setSuccessMsg(''), 4000);
    }
  };

  return (
    <section className="model-actions">
      <div className="section-label">
        <span className="fp-label-dot" />
        MANAGEMENT
      </div>
      <h2 className="section-title">
        Gestion du <span>Modèle</span>
      </h2>

      {successMsg && (
        <div className={`action-message${isError ? ' action-message--error' : ' action-message--success'}`}>
          {successMsg}
        </div>
      )}

      <div className="actions-grid">
        <div className="action-card">
          <div className="action-icon-wrap action-icon-wrap--reload">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M1 4v6h6" />
              <path d="M23 20v-6h-6" />
              <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10M23 14l-4.64 4.36A9 9 0 0 1 3.51 15" />
            </svg>
          </div>
          <h3>Recharger le Modèle</h3>
          <p>Force le rechargement depuis les fichiers stockés</p>
          {showConfirm ? (
            <div className="action-confirm">
              <p>Êtes-vous sûr ?</p>
              <div className="action-buttons">
                <button
                  className="btn btn-danger"
                  onClick={handleReload}
                  disabled={isReloading}
                >
                  {isReloading ? 'Rechargement...' : 'Confirmer'}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => setShowConfirm(false)}
                  disabled={isReloading}
                >
                  Annuler
                </button>
              </div>
            </div>
          ) : (
            <button
              className="btn btn-primary"
              onClick={() => setShowConfirm(true)}
              disabled={isReloading}
            >
              Recharger
            </button>
          )}
        </div>

        <div className="action-card">
          <div className="action-icon-wrap action-icon-wrap--export">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </div>
          <h3>Exporter Métriques</h3>
          <p>Télécharge les métriques au format JSON</p>
          <button className="btn btn-ghost">Télécharger</button>
        </div>

        <div className="action-card">
          <div className="action-icon-wrap action-icon-wrap--reports">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M3 9h18M9 21V9" />
            </svg>
          </div>
          <h3>Voir les Rapports</h3>
          <p>Accédez aux graphiques et explications SHAP</p>
          <button className="btn btn-ghost">Ouvrir Rapports</button>
        </div>

        <div className="action-card">
          <div className="action-icon-wrap action-icon-wrap--test">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18" />
            </svg>
          </div>
          <h3>Tester le Modèle</h3>
          <p>Testez le modèle avec des données d'exemple</p>
          <button className="btn btn-ghost">Test Interactif</button>
        </div>
      </div>
    </section>
  );
}