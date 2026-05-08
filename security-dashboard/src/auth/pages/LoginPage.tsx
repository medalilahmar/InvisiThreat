import { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import { login as doLogin } from '../services/authService';

export default function LoginPage() {
  const location = useLocation();
  const from = (location.state as any)?.from?.pathname || '/dashboard';

  const [form,    setForm]    = useState({ username: '', password: '' });
  const [error,   setError]   = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const { user, access_token } = await doLogin(form);
      console.log('✅ Token reçu :', access_token.slice(0, 20) + '...');
      console.log('✅ User reçu :', user);
      console.log('✅ Role :', user.role);

      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));

      // Vérification immédiate
      console.log('✅ localStorage token :', localStorage.getItem('token')?.slice(0, 20) + '...');

      window.location.href = user.role === 'admin' ? '/admin' : from;

    } catch (err: any) {
      console.error('❌ Erreur login :', err);
      setError(err.response?.data?.detail || 'Identifiants incorrects');
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      background: '#0a0a0f', padding: '1rem'
    }}>
      <div style={{
        width: '100%', maxWidth: 420,
        background: '#13131a',
        border: '0.5px solid rgba(255,255,255,0.08)',
        borderRadius: 16, padding: '2rem'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ fontSize: 36, marginBottom: 8 }}>🛡️</div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 500, color: '#fff' }}>
            InvisiThreat
          </h1>
          <p style={{ margin: '6px 0 0', color: 'rgba(255,255,255,0.45)', fontSize: 14 }}>
            Security Intelligence Platform
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div>
            <label style={{ display: 'block', fontSize: 13, color: 'rgba(255,255,255,0.55)', marginBottom: 6 }}>
              Nom d'utilisateur
            </label>
            <input
              type="text"
              value={form.username}
              onChange={e => setForm({ ...form, username: e.target.value })}
              placeholder="votre_username"
              required
              autoFocus
              style={{
                width: '100%', boxSizing: 'border-box',
                padding: '10px 14px', borderRadius: 8,
                background: '#1e1e2e',
                border: '0.5px solid rgba(255,255,255,0.12)',
                color: '#fff', fontSize: 14, outline: 'none'
              }}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: 13, color: 'rgba(255,255,255,0.55)', marginBottom: 6 }}>
              Mot de passe
            </label>
            <input
              type="password"
              value={form.password}
              onChange={e => setForm({ ...form, password: e.target.value })}
              placeholder="••••••••"
              required
              style={{
                width: '100%', boxSizing: 'border-box',
                padding: '10px 14px', borderRadius: 8,
                background: '#1e1e2e',
                border: '0.5px solid rgba(255,255,255,0.12)',
                color: '#fff', fontSize: 14, outline: 'none'
              }}
            />
          </div>

          {error && (
            <div style={{
              background: 'rgba(220,38,38,0.12)',
              border: '0.5px solid rgba(220,38,38,0.4)',
              color: '#f87171', borderRadius: 8,
              padding: '10px 14px', fontSize: 13
            }}>
              ⚠️ {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%', padding: '11px',
              background: loading ? 'rgba(99,102,241,0.5)' : '#6366f1',
              color: '#fff', border: 'none', borderRadius: 8,
              fontSize: 14, fontWeight: 500, cursor: loading ? 'not-allowed' : 'pointer',
              marginTop: 4
            }}
          >
            {loading ? 'Connexion...' : 'Se connecter'}
          </button>
        </form>

        <p style={{ textAlign: 'center', color: 'rgba(255,255,255,0.35)', fontSize: 13, marginTop: '1.5rem' }}>
          Pas encore de compte ?{' '}
          <Link to="/register" style={{ color: '#818cf8', textDecoration: 'none' }}>
            Créer un compte
          </Link>
        </p>
      </div>
    </div>
  );
}