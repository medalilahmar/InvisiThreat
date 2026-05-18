import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { register } from '../services/authService';
import './RegisterPage.css';

export default function RegisterPage() {
  const navigate = useNavigate();
  const [step, setStep] = useState<1 | 2>(1);

  const [form, setForm] = useState({
    username:   '',
    email:      '',
    password:   '',
    confirm:    '',
    job_title:  '',
    department: '',
    phone:      '',
    avatar_url: '',
  });

  const [error,       setError]       = useState('');
  const [loading,     setLoading]     = useState(false);
  const [showPass,    setShowPass]    = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);

  const set = (field: string) =>
    (e: React.ChangeEvent<HTMLInputElement>) =>
      setForm(f => ({ ...f, [field]: e.target.value }));

  /* ── Étape 1 → 2 ── */
  const handleNext = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (!form.username.trim()) { setError('Username is required'); return; }
    if (!form.email.trim())    { setError('Email is required'); return; }
    if (form.password.length < 8) { setError('Password too short — minimum 8 characters'); return; }
    if (form.password !== form.confirm) { setError('Passwords do not match'); return; }
    setStep(2);
  };

  /* ── Submit final ── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await register({
        username:   form.username,
        email:      form.email,
        password:   form.password,
        job_title:  form.job_title  || undefined,
        department: form.department || undefined,
        phone:      form.phone      || undefined,
        avatar_url: form.avatar_url || undefined,
      });
      navigate('/pending');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Error creating account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rp-root">

      {/* ── Panel gauche — Branding ── */}
      <div className="rp-left">
        <div className="rp-bg-glow" />
        <div className="rp-left-content">

          <div className="rp-brand-mark">🛡️</div>
          <h1 className="rp-brand-name">Invisi<span>Threat</span></h1>
          <p className="rp-brand-desc">
            Join the InvisiThreat platform. Your account will be reviewed
            by an administrator before you can access the SOC dashboard.
          </p>

          <div className="rp-steps">
            <div className={`rp-step ${step >= 1 ? 'rp-step--active' : ''}`}>
              <div className="rp-step-num">
                {step > 1 ? <span className="rp-step-check">✓</span> : '01'}
              </div>
              <div className="rp-step-info">
                <span className="rp-step-title">Account credentials</span>
                <span className="rp-step-desc">Username, email & password</span>
              </div>
            </div>

            <div className={`rp-step-line ${step > 1 ? 'rp-step-line--done' : ''}`} />

            <div className={`rp-step ${step >= 2 ? 'rp-step--active' : 'rp-step--dim'}`}>
              <div className="rp-step-num">02</div>
              <div className="rp-step-info">
                <span className="rp-step-title">Professional profile</span>
                <span className="rp-step-desc">Optional details</span>
              </div>
            </div>

            <div className="rp-step-line" />

            <div className="rp-step rp-step--dim">
              <div className="rp-step-num">03</div>
              <div className="rp-step-info">
                <span className="rp-step-title">Admin approval</span>
                <span className="rp-step-desc">Access granted after review</span>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* ── Panel droit — Formulaire ── */}
      <div className="rp-right">
        <div className="rp-form-wrap">

          {/* Progress bar */}
          <div className="rp-progress">
            <div className="rp-progress-track">
              <div className="rp-progress-fill" style={{ width: step === 1 ? '50%' : '100%' }} />
            </div>
            <span className="rp-progress-label">Step {step} of 2</span>
          </div>

          {/* ══ ÉTAPE 1 ══ */}
          {step === 1 && (
            <div className="rp-panel rp-panel--enter">

              <div className="rp-form-header">
                <h2 className="rp-form-title">Account credentials</h2>
                <p className="rp-form-subtitle">Create your login details to get started</p>
              </div>

              <form className="rp-form" onSubmit={handleNext}>

                <div className="rp-field">
                  <label className="rp-label" htmlFor="rp-username">Username *</label>
                  <input id="rp-username" className="rp-input" type="text"
                    value={form.username} onChange={set('username')}
                    placeholder="john_doe" required autoFocus autoComplete="username" />
                </div>

                <div className="rp-field">
                  <label className="rp-label" htmlFor="rp-email">Email address *</label>
                  <input id="rp-email" className="rp-input" type="email"
                    value={form.email} onChange={set('email')}
                    placeholder="john@company.com" required autoComplete="email" />
                </div>

                <div className="rp-field">
                  <label className="rp-label" htmlFor="rp-password">Password *</label>
                  <div className="rp-input-wrap">
                    <input id="rp-password" className="rp-input"
                      type={showPass ? 'text' : 'password'}
                      value={form.password} onChange={set('password')}
                      placeholder="Min 8 characters" required autoComplete="new-password" />
                    <button type="button" className="rp-eye-btn"
                      onClick={() => setShowPass(p => !p)} tabIndex={-1}>
                      {showPass ? '🙈' : '👁'}
                    </button>
                  </div>
                </div>

                <div className="rp-field">
                  <label className="rp-label" htmlFor="rp-confirm">Confirm password *</label>
                  <div className="rp-input-wrap">
                    <input id="rp-confirm" className="rp-input"
                      type={showConfirm ? 'text' : 'password'}
                      value={form.confirm} onChange={set('confirm')}
                      placeholder="••••••••••••" required autoComplete="new-password" />
                    <button type="button" className="rp-eye-btn"
                      onClick={() => setShowConfirm(p => !p)} tabIndex={-1}>
                      {showConfirm ? '🙈' : '👁'}
                    </button>
                  </div>
                </div>

                {error && (
                  <div className="rp-error">
                    <span className="rp-error-icon">⚠</span>
                    <span>{error}</span>
                  </div>
                )}

                <button type="submit" className="rp-btn">Continue →</button>

              </form>

              <p className="rp-footer-text">
                Already have an account?{' '}
                <Link to="/login" className="rp-footer-link">Sign in →</Link>
              </p>
            </div>
          )}

          {/* ══ ÉTAPE 2 ══ */}
          {step === 2 && (
            <div className="rp-panel rp-panel--enter">

              <div className="rp-form-header">
                <h2 className="rp-form-title">Professional profile</h2>
                <p className="rp-form-subtitle">
                  Optional — helps your admin assign you to the right projects
                </p>
              </div>

              <form className="rp-form" onSubmit={handleSubmit}>

                <div className="rp-field-row">
                  <div className="rp-field">
                    <label className="rp-label" htmlFor="rp-job">Job title</label>
                    <input id="rp-job" className="rp-input" type="text"
                      value={form.job_title} onChange={set('job_title')}
                      placeholder="Security Engineer" autoFocus />
                  </div>
                  <div className="rp-field">
                    <label className="rp-label" htmlFor="rp-dept">Department</label>
                    <input id="rp-dept" className="rp-input" type="text"
                      value={form.department} onChange={set('department')}
                      placeholder="DevSecOps" />
                  </div>
                </div>

                <div className="rp-field-row">
                  <div className="rp-field">
                    <label className="rp-label" htmlFor="rp-phone">Phone</label>
                    <input id="rp-phone" className="rp-input" type="tel"
                      value={form.phone} onChange={set('phone')}
                      placeholder="+33 6 12 34 56 78" />
                  </div>
                  <div className="rp-field">
                    <label className="rp-label" htmlFor="rp-avatar">Avatar URL</label>
                    <input id="rp-avatar" className="rp-input" type="url"
                      value={form.avatar_url} onChange={set('avatar_url')}
                      placeholder="https://..." />
                  </div>
                </div>

                {/* Récap */}
                <div className="rp-recap">
                  <div className="rp-recap-title">Account summary</div>
                  <div className="rp-recap-row">
                    <span className="rp-recap-key">Username</span>
                    <span className="rp-recap-val">{form.username}</span>
                  </div>
                  <div className="rp-recap-row">
                    <span className="rp-recap-key">Email</span>
                    <span className="rp-recap-val">{form.email}</span>
                  </div>
                </div>

                {error && (
                  <div className="rp-error">
                    <span className="rp-error-icon">⚠</span>
                    <span>{error}</span>
                  </div>
                )}

                <div className="rp-btn-row">
                  <button type="button" className="rp-btn-back"
                    onClick={() => { setStep(1); setError(''); }}>
                    ← Back
                  </button>
                  <button type="submit" className="rp-btn rp-btn--flex" disabled={loading}>
                    {loading
                      ? <><span className="rp-spinner" /> Creating…</>
                      : 'Create account →'}
                  </button>
                </div>

              </form>

              <p className="rp-footer-text">
                Already have an account?{' '}
                <Link to="/login" className="rp-footer-link">Sign in →</Link>
              </p>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}