import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../auth/hooks/useAuth';
import './LoginPage.css';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [shakeForm, setShakeForm] = useState(false);
  const usernameRef = useRef<HTMLInputElement>(null);

  const { login, loading, error, clearError, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: Location })?.from?.pathname || '/dashboard';

  useEffect(() => {
    if (isAuthenticated) navigate(from, { replace: true });
  }, [isAuthenticated, navigate, from]);

  useEffect(() => {
    usernameRef.current?.focus();
  }, []);

  useEffect(() => {
    if (error) {
      setShakeForm(true);
      const t = setTimeout(() => setShakeForm(false), 600);
      return () => clearTimeout(t);
    }
  }, [error]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    if (!username.trim() || !password.trim()) return;
    await login({ username, password });
  };

  return (
    <div className="login-root">
      {/* Animated grid background */}
      <div className="login-bg">
        <div className="login-bg__grid" />
        <div className="login-bg__glow login-bg__glow--1" />
        <div className="login-bg__glow login-bg__glow--2" />
        <div className="login-bg__scanline" />
      </div>

      <div className="login-layout">
        {/* Left branding panel */}
        <div className="login-brand">
          <div className="login-brand__inner">
            <div className="login-brand__logo-wrap">
              <svg className="login-brand__icon" viewBox="0 0 48 48" fill="none">
                <polygon points="24,4 44,14 44,34 24,44 4,34 4,14" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.3"/>
                <polygon points="24,10 38,17 38,31 24,38 10,31 10,17" stroke="currentColor" strokeWidth="1" fill="none" opacity="0.5"/>
                <circle cx="24" cy="24" r="6" fill="currentColor" opacity="0.9"/>
                <line x1="24" y1="4" x2="24" y2="10" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="44" y1="14" x2="38" y2="17" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="44" y1="34" x2="38" y2="31" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="24" y1="44" x2="24" y2="38" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="4" y1="34" x2="10" y2="31" stroke="currentColor" strokeWidth="1.5"/>
                <line x1="4" y1="14" x2="10" y2="17" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
              <span className="login-brand__wordmark">InvisiThreat</span>
            </div>

            <div className="login-brand__tagline">
              <p>Security Intelligence</p>
              <p>Platform</p>
            </div>

            <div className="login-brand__stats">
              <div className="login-brand__stat">
                <span className="login-brand__stat-value">99.9<span>%</span></span>
                <span className="login-brand__stat-label">Uptime</span>
              </div>
              <div className="login-brand__stat-divider" />
              <div className="login-brand__stat">
                <span className="login-brand__stat-value">256<span>-bit</span></span>
                <span className="login-brand__stat-label">Encryption</span>
              </div>
              <div className="login-brand__stat-divider" />
              <div className="login-brand__stat">
                <span className="login-brand__stat-value">24/7</span>
                <span className="login-brand__stat-label">Monitoring</span>
              </div>
            </div>

            <div className="login-brand__badges">
              <span className="login-brand__badge">● THREAT DETECTION</span>
              <span className="login-brand__badge">● ENGAGEMENT TRACKING</span>
              <span className="login-brand__badge">● FINDINGS ANALYSIS</span>
            </div>
          </div>
        </div>

        {/* Right form panel */}
        <div className="login-panel">
          <div className={`login-form-wrap ${shakeForm ? 'login-form-wrap--shake' : ''}`}>
            <div className="login-form-header">
              <div className="login-form-header__eyebrow">
                <span className="login-form-header__dot" />
                SECURE ACCESS
              </div>
              <h1 className="login-form-header__title">Welcome back</h1>
              <p className="login-form-header__sub">Sign in to your dashboard</p>
            </div>

            <form className="login-form" onSubmit={handleSubmit} noValidate>
              <div className="login-field">
                <label className="login-field__label" htmlFor="username">Username</label>
                <div className="login-field__control">
                  <svg className="login-field__icon" viewBox="0 0 20 20" fill="none">
                    <circle cx="10" cy="7" r="3.5" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M3 17c0-3.314 3.134-6 7-6s7 2.686 7 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  <input
                    ref={usernameRef}
                    id="username"
                    type="text"
                    className="login-field__input"
                    placeholder="Enter username"
                    value={username}
                    onChange={(e) => { setUsername(e.target.value); clearError(); }}
                    autoComplete="username"
                    autoCapitalize="none"
                    spellCheck={false}
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="login-field">
                <label className="login-field__label" htmlFor="password">Password</label>
                <div className="login-field__control">
                  <svg className="login-field__icon" viewBox="0 0 20 20" fill="none">
                    <rect x="4" y="9" width="12" height="9" rx="2" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M7 9V6a3 3 0 016 0v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                    <circle cx="10" cy="13.5" r="1.5" fill="currentColor"/>
                  </svg>
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    className="login-field__input"
                    placeholder="Enter password"
                    value={password}
                    onChange={(e) => { setPassword(e.target.value); clearError(); }}
                    autoComplete="current-password"
                    disabled={loading}
                  />
                  <button
                    type="button"
                    className="login-field__toggle"
                    onClick={() => setShowPassword((v) => !v)}
                    tabIndex={-1}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? (
                      <svg viewBox="0 0 20 20" fill="none">
                        <path d="M3 3l14 14M8.46 8.54A3 3 0 0013.46 13.46M6.22 5.22A8.95 8.95 0 002 10c1.273 3.3 4.46 5.5 8 5.5a8.8 8.8 0 003.78-.85M9 4.55A8.75 8.75 0 0118 10a8.9 8.9 0 01-1.27 2.73" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                      </svg>
                    ) : (
                      <svg viewBox="0 0 20 20" fill="none">
                        <path d="M2 10c1.273-3.3 4.46-5.5 8-5.5S16.727 6.7 18 10c-1.273 3.3-4.46 5.5-8 5.5S3.273 13.3 2 10z" stroke="currentColor" strokeWidth="1.5"/>
                        <circle cx="10" cy="10" r="2.5" stroke="currentColor" strokeWidth="1.5"/>
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              {error && (
                <div className="login-error" role="alert">
                  <svg viewBox="0 0 20 20" fill="none">
                    <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="1.5"/>
                    <path d="M10 6v4M10 14h.01" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                  {error}
                </div>
              )}

              <button
                type="submit"
                className={`login-submit ${loading ? 'login-submit--loading' : ''}`}
                disabled={loading || !username.trim() || !password.trim()}
              >
                {loading ? (
                  <>
                    <span className="login-submit__spinner" />
                    Authenticating…
                  </>
                ) : (
                  <>
                    Sign in
                    <svg viewBox="0 0 20 20" fill="none">
                      <path d="M4 10h12M11 6l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </>
                )}
              </button>
            </form>

            <p className="login-footer-note">
              Protected by InvisiThreat Security · All access is logged
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}