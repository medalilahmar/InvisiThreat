import { useState } from 'react';
import { useLocation,useNavigate, Link } from 'react-router-dom';
import { login as doLogin } from '../services/authService';
import { useAuth } from '../hooks/useAuth';
import './LoginPage.css';

export default function LoginPage() {
  const location = useLocation();
  const from = (location.state as any)?.from?.pathname || '/dashboard';
  const { login } = useAuth();
  const navigate = useNavigate();

  const [form,         setForm]         = useState({ username: '', password: '' });
  const [error,        setError]        = useState('');
  const [loading,      setLoading]      = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const result = await doLogin(form);

      // ── Login réussi ───────────────────────────────────────────────────
      login(result.user, result.access_token);

      if (result.user.status === 'pending') {
        navigate('/auth/pending', { replace: true });
        return;
      }
      if (result.user.status === 'blocked') {
        navigate('/auth/blocked', { replace: true });
        return;
      }

      navigate(result.user.role === 'admin' ? '/admin' : from, { replace: true });

    } catch (err: any) {
      // ── Lire le LoginError structuré depuis authService ───────────────
      const code:    string = err?.code    ?? 'UNKNOWN';
      const message: string = err?.message ?? 'An error occurred';

      if (code === 'ACCOUNT_LOCKED') {
        const minutes = err?.minutesLeft ?? 30;
        setError(`🔒 Account locked. Try again in ${minutes} minutes.`);

      } else if (code === 'ACCOUNT_PENDING') {
        navigate('/auth/pending', { replace: true });
        return;

      } else if (code === 'ACCOUNT_BLOCKED') {
        navigate('/auth/blocked', { replace: true });
        return;

      } else if (code === 'INVALID_CREDENTIALS') {
        setError('Invalid username or password');

      } else {
        setError(message || 'An error occurred. Please try again.');
      }

      setLoading(false);
    }
  };

  return (
    <div className="lp-root">

      {/* ── Panel gauche — Branding ── */}
      <div className="lp-left">
        <div className="lp-left-noise" />
        <div className="lp-left-glow" />

        <div className="lp-left-content">
          <div className="lp-brand-mark">
            <span className="lp-brand-icon">🛡️</span>
          </div>

          <h1 className="lp-brand-name">
            Invisi<span>Threat</span>
          </h1>
          <p className="lp-brand-desc">
            Security Intelligence Platform for modern SOC teams. Monitor,
            analyze and respond to threats in real time.
          </p>

          <div className="lp-stats">
            <div className="lp-stat">
              <span className="lp-stat-value">99.9%</span>
              <span className="lp-stat-label">Uptime</span>
            </div>
            <div className="lp-stat-divider" />
            <div className="lp-stat">
              <span className="lp-stat-value">24/7</span>
              <span className="lp-stat-label">Monitoring</span>
            </div>
            <div className="lp-stat-divider" />
            <div className="lp-stat">
              <span className="lp-stat-value">TLS 1.3</span>
              <span className="lp-stat-label">Encrypted</span>
            </div>
          </div>

          <div className="lp-system-status">
            <span className="lp-status-dot" />
            All systems operational
          </div>
        </div>

        <div className="lp-left-version">v2.4.1 · SOC Command Center</div>
      </div>

      {/* ── Panel droit — Formulaire ── */}
      <div className="lp-right">
        <div className="lp-form-wrap">

          <div className="lp-form-header">
            <Link to="/" className="lp-back-home">
              ← Home
            </Link>
            <h2 className="lp-form-title">Welcome back</h2>
            <p className="lp-form-subtitle">Sign in to your account to continue</p>
          </div>

          <form className="lp-form" onSubmit={handleSubmit}>

            <div className="lp-field">
              <label className="lp-label" htmlFor="username">Username</label>
              <input
                id="username"
                className="lp-input"
                type="text"
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
                placeholder="Enter your username"
                required
                autoFocus
                autoComplete="username"
              />
            </div>

            <div className="lp-field">
              <div className="lp-label-row">
                <label className="lp-label" htmlFor="password">Password</label>
              </div>
              <div className="lp-input-wrap">
                <input
                  id="password"
                  className="lp-input"
                  type={showPassword ? 'text' : 'password'}
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  placeholder="••••••••••••"
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="lp-eye-btn"
                  onClick={() => setShowPassword(!showPassword)}
                  tabIndex={-1}
                >
                  {showPassword ? '🙈' : '👁'}
                </button>
              </div>
            </div>

            {error && (
              <div className="lp-error">
                <span className="lp-error-icon">⚠</span>
                <span>{error}</span>
              </div>
            )}

            <button type="submit" className="lp-btn" disabled={loading}>
              {loading ? (
                <><span className="lp-spinner" /> Authenticating…</>
              ) : (
                'Sign in'
              )}
            </button>

          </form>

          <p className="lp-footer-text">
            Don't have an account?{' '}
            <Link to="/register" className="lp-footer-link">Create access →</Link>
          </p>

        </div>
      </div>

    </div>
  );
}