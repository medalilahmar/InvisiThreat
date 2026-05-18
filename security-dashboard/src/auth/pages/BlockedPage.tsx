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
            Your access to the InvisiThreat platform has been restricted.
            Please review the information below and contact your administrator.
          </p>

          {/* Compte info */}
          <div className="bp-account-box">
            <div className="bp-account-box-title">Account details</div>

            <div className="bp-info-row">
              <span className="bp-info-key">Username</span>
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
                <span className="bp-info-key">Job title</span>
                <span className="bp-info-val">{user.job_title}</span>
              </div>
            )}
            {user?.department && (
              <div className="bp-info-row">
                <span className="bp-info-key">Department</span>
                <span className="bp-info-val">{user.department}</span>
              </div>
            )}
            <div className="bp-info-row">
              <span className="bp-info-key">Status</span>
              <span className="bp-status-badge">BLOCKED</span>
            </div>
          </div>

          {/* Status indicator */}
          <div className="bp-system-status">
            <span className="bp-status-dot" />
            Account access restricted
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
            <h2 className="bp-title">Account blocked</h2>
            <p className="bp-subtitle">
              Your account has been suspended and you cannot access the platform at this time.
            </p>
          </div>

          {/* Raisons */}
          <div className="bp-section">
            <div className="bp-section-label">Why was my account blocked?</div>
            <div className="bp-reasons">
              {[
                { icon: '⚠️', text: 'Security policy violation' },
                { icon: '🔍', text: 'Suspicious activity detected' },
                { icon: '👤', text: 'Administrative decision' },
                { icon: '📋', text: 'Terms of service breach' },
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
              To appeal this decision or learn more, reach out to the support team:
            </p>
            <div className="bp-contacts">
              <a href="mailto:support@invisithreat.com" className="bp-contact">
                <span className="bp-contact-icon">📧</span>
                <span>support@invisithreat.com</span>
              </a>
              <div className="bp-contact">
                <span className="bp-contact-icon">👤</span>
                <span>Contact your administrator directly</span>
              </div>
            </div>
          </div>

          {/* Action */}
          <button onClick={handleLogout} className="bp-btn-logout">
            Sign out
          </button>

          <p className="bp-footer-text">
            InvisiThreat © 2024 · Security Intelligence Platform
          </p>

        </div>
      </div>
    </div>
  );
}