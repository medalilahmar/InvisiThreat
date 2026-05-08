import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { register } from '../services/authService';

export default function RegisterPage() {
  const navigate  = useNavigate();
  const [form,    setForm]    = useState({ username: '', email: '', password: '', confirm: '' });
  const [error,   setError]   = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    if (form.password !== form.confirm) {
      setError('Les mots de passe ne correspondent pas');
      return;
    }
    if (form.password.length < 8) {
      setError('Mot de passe trop court — minimum 8 caractères');
      return;
    }
    setLoading(true);
    try {
      await register({ username: form.username, email: form.email, password: form.password });
      navigate('/pending');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Erreur lors de la création du compte');
    } finally {
      setLoading(false);
    }
  };

  const fields = [
    { key: 'username', label: "Nom d'utilisateur", type: 'text',     placeholder: 'john_doe' },
    { key: 'email',    label: 'Adresse email',      type: 'email',    placeholder: 'john@company.com' },
    { key: 'password', label: 'Mot de passe',        type: 'password', placeholder: 'Min 8 caractères' },
    { key: 'confirm',  label: 'Confirmer',            type: 'password', placeholder: '••••••••' },
  ] as const;

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
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 500, color: '#fff' }}>Créer un compte</h1>
          <p style={{ margin: '6px 0 0', color: 'rgba(255,255,255,0.45)', fontSize: 14 }}>
            InvisiThreat Platform
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {fields.map(f => (
            <div key={f.key}>
              <label style={{ display: 'block', fontSize: 13, color: 'rgba(255,255,255,0.55)', marginBottom: 6 }}>
                {f.label}
              </label>
              <input
                type={f.type}
                value={form[f.key]}
                onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                placeholder={f.placeholder}
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
          ))}

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
              background: loading ? 'rgba(16,185,129,0.4)' : '#10b981',
              color: '#fff', border: 'none', borderRadius: 8,
              fontSize: 14, fontWeight: 500,
              cursor: loading ? 'not-allowed' : 'pointer', marginTop: 4
            }}
          >
            {loading ? 'Création...' : 'Créer mon compte'}
          </button>
        </form>

        <p style={{ textAlign: 'center', color: 'rgba(255,255,255,0.35)', fontSize: 13, marginTop: '1.5rem' }}>
          Déjà un compte ?{' '}
          <Link to="/login" style={{ color: '#818cf8', textDecoration: 'none' }}>
            Se connecter
          </Link>
        </p>
      </div>
    </div>
  );
}