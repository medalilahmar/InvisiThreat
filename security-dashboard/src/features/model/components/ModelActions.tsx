// frontend/src/features/model/components/ModelActions.tsx
import { useState } from 'react';

interface ModelActionsProps {
  onReload: () => Promise<boolean>;
  isReloading: boolean;
  // ← modelVersion supprimé car non utilisé
}

export function ModelActions({ onReload, isReloading }: ModelActionsProps) {
  const [showConfirm, setShowConfirm] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  const handleReload = async () => {
    setShowConfirm(false);
    const success = await onReload();
    if (success) {
      setSuccessMsg('✅ Modèle rechargé avec succès');
      setTimeout(() => setSuccessMsg(''), 3000);
    } else {
      setSuccessMsg('❌ Erreur lors du rechargement');
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

      {successMsg && <div className="action-message">{successMsg}</div>}

      <div className="actions-grid">
        <div className="action-card">
          <div className="action-icon">🔄</div>
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
                  className="btn btn-secondary"
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
          <div className="action-icon">📥</div>
          <h3>Exporter Métriques</h3>
          <p>Télécharge les métriques au format JSON</p>
          <button className="btn btn-secondary">Télécharger</button>
        </div>

        <div className="action-card">
          <div className="action-icon">📊</div>
          <h3>Voir les Rapports</h3>
          <p>Accédez aux graphiques et explications SHAP</p>
          <button className="btn btn-secondary">Ouvrir Rapports</button>
        </div>

        <div className="action-card">
          <div className="action-icon">🧪</div>
          <h3>Tester le Modèle</h3>
          <p>Testez le modèle avec des données d'exemple</p>
          <button className="btn btn-secondary">Test Interactive</button>
        </div>
      </div>
    </section>
  );
}