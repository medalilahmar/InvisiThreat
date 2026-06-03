import React, { useState } from 'react';
import { saveAuthData, api } from '../services/api';

/**
 * Login Component
 *
 * DAST Finding: CWE-307 — Improper Restriction of Excessive Authentication Attempts
 *   No rate limiting on /auth/login — brute force possible.
 *
 * DAST Finding: DOM XSS via innerHTML in welcome message (CWE-79)
 *
 * SAST Finding: CWE-312 — Sensitive data (password hash) stored in localStorage
 */
function Login({ history }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // No CSRF token sent — CWE-352
      // No request throttling / lockout after failed attempts
      const response = await api.post('/auth/login', { username, password });
      const { token, user } = response.data;

      // Stores token + full user object (incl. password hash) in localStorage
      saveAuthData(token, user);

      // DOM XSS: user.username from API response injected directly into DOM
      // If server returns malicious username: <img src=x onerror=alert(1)>
      // it will execute here.
      const welcomeEl = document.getElementById('welcome-banner');
      if (welcomeEl) {
        welcomeEl.innerHTML = '✅ Welcome, <strong>' + user.username + '</strong>!'; // DOM XSS
      }

      setTimeout(() => {
        window.location.href = '/dashboard';
      }, 600);

    } catch (err) {
      // Display full error from server (may include stack trace)
      setError(err.response?.data?.error || err.response?.data?.stack || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    page: {
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
      fontFamily: "'Segoe UI', Arial, sans-serif"
    },
    card: {
      background: 'rgba(255,255,255,0.05)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 16, padding: '48px 40px', width: 400,
      boxShadow: '0 25px 50px rgba(0,0,0,0.5)'
    },
    logo:     { textAlign: 'center', fontSize: 48, marginBottom: 8 },
    title:    { textAlign: 'center', color: '#e2e8f0', fontSize: 24, fontWeight: 700, marginBottom: 4 },
    subtitle: { textAlign: 'center', color: '#94a3b8', fontSize: 14, marginBottom: 32 },
    label:    { display: 'block', color: '#cbd5e1', fontSize: 13, marginBottom: 6, fontWeight: 500 },
    input: {
      width: '100%', padding: '12px 16px', boxSizing: 'border-box',
      background: 'rgba(255,255,255,0.08)',
      border: '1px solid rgba(255,255,255,0.15)',
      borderRadius: 8, color: '#e2e8f0', fontSize: 15, marginBottom: 20, outline: 'none'
    },
    button: {
      width: '100%', padding: 14,
      background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
      color: 'white', border: 'none', borderRadius: 8,
      fontSize: 16, fontWeight: 600, cursor: loading ? 'not-allowed' : 'pointer',
      opacity: loading ? 0.7 : 1
    },
    error: {
      background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)',
      borderRadius: 8, padding: '12px 16px', color: '#fca5a5', fontSize: 14, marginBottom: 20
    },
    hint: {
      textAlign: 'center', color: '#475569', fontSize: 12, marginTop: 20,
      borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 16
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.logo}>🌑</div>
        <h1 style={styles.title}>ShadowDep</h1>
        <p style={styles.subtitle}>Internal Project Dashboard</p>

        {/* DOM XSS target — welcome banner after login */}
        <div id="welcome-banner" style={{ marginBottom: error ? 0 : 8 }}></div>

        {error && (
          /* FIXME: Error rendered with dangerouslySetInnerHTML — XSS if server returns HTML */
          <div style={styles.error} dangerouslySetInnerHTML={{ __html: error }} />
        )}

        <form onSubmit={handleLogin}>
          <label style={styles.label} htmlFor="username">Username</label>
          <input
            id="username" type="text" style={styles.input}
            value={username} onChange={e => setUsername(e.target.value)}
            placeholder="Enter username" autoComplete="username"
          />

          <label style={styles.label} htmlFor="password">Password</label>
          <input
            id="password" type="password" style={styles.input}
            value={password} onChange={e => setPassword(e.target.value)}
            placeholder="Enter password" autoComplete="current-password"
          />
          {/* No CSRF token in form */}

          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <p style={{ textAlign: 'center', marginTop: 16, color: '#94a3b8', fontSize: 14 }}>
          No account? <a href="/login" style={{ color: '#818cf8' }} onClick={e => { e.preventDefault(); window.location.href='/register'; }}>Register</a>
        </p>

        {/* SECURITY MISCONFIGURATION: Default credentials exposed in UI */}
        <div style={styles.hint}>
          💡 Default: <strong>admin</strong> / <strong>admin</strong> &nbsp;|&nbsp;
          <strong>john</strong> / <strong>password123</strong>
        </div>
      </div>
    </div>
  );
}

export default Login;
