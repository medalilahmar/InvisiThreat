import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../auth/hooks/useAuth';
import './BlockedPage.css';

export default function BlockedPage() {
  const navigate = useNavigate();
  const { user, isActive, isPending, logout } = useAuth();

  useEffect(() => {
    if (isActive)   navigate('/dashboard', { replace: true });
    else if (isPending) navigate('/pending', { replace: true });
  }, [isActive, isPending, navigate]);

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login', { replace: true });
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  return (
    <div className="bp-root">
      <div className="bp-bg-glow" />

      {/* ── Panel gauche — Branding ── */}
      <div className="bp-left">
        <div className="bp-left-content">

          <div className="bp-brand-mark">🛡️</div>
          <h1 className="bp-brand-name">Invisi<span>Threat</span></h1>
          <p className="bp-brand-desc">
            Votre accès à la plateforme InvisiThreat a été restreint.
            Veuillez consulter les informations ci-dessous et contacter votre administrateur.
          </p>

          {/* Compte info */}
          <div className="bp-account-box">
            <div className="bp-account-box-title">Détails du compte</div>

            <div className="bp-info-row">
              <span className="bp-info-key">Nom d'utilisateur</span>
              <span className="bp-info-val">{user?.username ?? '—'}</span>
            </div>
            <div className="bp-info-row">
              <span className="bp-info-key">Email</span>
              <span className="bp-info-val">{user?.email ?? '—'}</span>
            </div>
            <div className="bp-info-row">
              <span className="bp-info-key">Role</span>
              <span className="bp-info-val">{user?.role ?? 'Developer'}</span>
            </div>
            {user?.job_title && (
              <div className="bp-info-row">
                <span className="bp-info-key">Poste</span>
                <span className="bp-info-val">{user.job_title}</span>
              </div>
            )}
            {user?.department && (
              <div className="bp-info-row">
                <span className="bp-info-key">Service / Département</span>
                <span className="bp-info-val">{user.department}</span>
              </div>
            )}
            <div className="bp-info-row">
              <span className="bp-info-key">Statut</span>
              <span className="bp-status-badge">BLOQUÉ</span>
            </div>
          </div>

          {/* Status indicator */}
          <div className="bp-system-status">
            <span className="bp-status-dot" />
            Accès au compte restreint
          </div>

        </div>
      </div>

      {/* ── Panel droit — Informations ── */}
      <div className="bp-right">
        <div className="bp-content-wrap">

          {/* Icône principale */}
          <div className="bp-icon-wrap">
            <span className="bp-icon">🚫</span>
          </div>

          <div className="bp-form-header">
            <h2 className="bp-title">Compte bloqué</h2>
            <p className="bp-subtitle">
              Votre compte a été suspendu et vous ne pouvez pas accéder à la plateforme pour le moment.
            </p>
          </div>

          {/* Raisons */}
          <div className="bp-section">
            <div className="bp-section-label">Pourquoi mon compte a-t-il été bloqué?</div>
            <div className="bp-reasons">
              {[
                { icon: '⚠️', text: 'Violation de la politique de sécurité' },
                { icon: '🔍', text: 'Activité suspecte détectée' },
                { icon: '👤', text: 'Décision administrative' },
                { icon: '📋', text: 'Violation des conditions d\'utilisation' },
              ].map((r, i) => (
                <div className="bp-reason-item" key={i}>
                  <span className="bp-reason-icon">{r.icon}</span>
                  <span className="bp-reason-text">{r.text}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Contact */}
          <div className="bp-section">
            <div className="bp-section-label">Contact support</div>
            <p className="bp-support-desc">
              Pour faire appel de cette décision ou en savoir plus, contactez l'équipe de support :
            </p>
            <div className="bp-contacts">
              <a href="mailto:support@invisithreat.com" className="bp-contact">
                <span className="bp-contact-icon">📧</span>
                <span>support@invisithreat.com</span>
              </a>
              <div className="bp-contact">
                <span className="bp-contact-icon">👤</span>
                <span>Contactez votre administrateur directement</span>
              </div>
            </div>
          </div>

          {/* Action */}
          <button onClick={handleLogout} className="bp-btn-logout">
            Se déconnecter
          </button>

          <p className="bp-footer-text">
            InvisiThreat © 2024 · Security Intelligence Platform
          </p>

        </div>
      </div>
    </div>
  );
}