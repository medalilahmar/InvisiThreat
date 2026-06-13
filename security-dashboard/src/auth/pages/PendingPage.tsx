import { Link } from 'react-router-dom';
import './PendingPage.css';

export default function PendingPage() {
  return (
    <div className="pp-root">
      <div className="pp-bg-glow" />

      <div className="pp-card">
        <div className="pp-card-top-line" />

        {/* Icône */}
        <div className="pp-icon-wrap">
          <span className="pp-icon">⏳</span>
        </div>

        {/* Titre */}
        <h2 className="pp-title">Validation du compte en attente</h2>
        <p className="pp-desc">
          Votre compte a été créé avec succès. Un administrateur
          doit valider votre accès et vous assigner à vos projets
          avant de pouvoir vous connecter.
        </p>

        {/* Info box */}
        <div className="pp-info">
          <span className="pp-info-icon">💡</span>
          <span>Vous serez notifié une fois que votre compte sera approuvé par l'équipe de sécurité.</span>
        </div>

        {/* Status */}
        <div className="pp-status">
          <span className="pp-status-dot" />
          <span className="pp-status-text">En attente de révision de l'administrateur</span>
        </div>

        {/* Lien retour */}
        <Link to="/login" className="pp-back">
          ← Retour à la page de connexion
        </Link>
      </div>
    </div>
  );
}