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
        <h2 className="pp-title">Account pending approval</h2>
        <p className="pp-desc">
          Your account has been created successfully. An administrator
          must validate your access and assign you to your projects
          before you can sign in.
        </p>

        {/* Info box */}
        <div className="pp-info">
          <span className="pp-info-icon">💡</span>
          <span>You will be notified once your account is approved by the security team.</span>
        </div>

        {/* Status */}
        <div className="pp-status">
          <span className="pp-status-dot" />
          <span className="pp-status-text">Awaiting administrator review</span>
        </div>

        {/* Lien retour */}
        <Link to="/login" className="pp-back">
          ← Back to sign in
        </Link>
      </div>
    </div>
  );
}