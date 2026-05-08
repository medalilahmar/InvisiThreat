import { Link } from 'react-router-dom';

export default function PendingPage() {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      alignItems: 'center', justifyContent: 'center',
      background: '#0a0a0f', padding: '1rem'
    }}>
      <div style={{
        textAlign: 'center', maxWidth: 440,
        background: '#13131a',
        border: '0.5px solid rgba(255,255,255,0.08)',
        borderRadius: 16, padding: '2.5rem 2rem'
      }}>
        <div style={{ fontSize: 56, marginBottom: '1rem' }}>⏳</div>
        <h2 style={{ margin: '0 0 12px', fontSize: 20, fontWeight: 500, color: '#fff' }}>
          Compte en attente de validation
        </h2>
        <p style={{ margin: '0 0 1.5rem', color: 'rgba(255,255,255,0.5)', fontSize: 14, lineHeight: 1.7 }}>
          Votre compte a bien été créé. Un administrateur doit valider
          votre accès et vous affecter à vos projets avant que vous
          puissiez vous connecter.
        </p>
        <div style={{
          background: 'rgba(99,102,241,0.12)',
          border: '0.5px solid rgba(99,102,241,0.3)',
          borderRadius: 8, padding: '12px 16px',
          color: '#a5b4fc', fontSize: 13,
          marginBottom: '1.5rem', lineHeight: 1.6
        }}>
          💡 Vous serez notifié une fois votre compte approuvé par l'équipe sécurité.
        </div>
        <Link
          to="/login"
          style={{ color: '#818cf8', fontSize: 13, textDecoration: 'none' }}
        >
          ← Retour à la connexion
        </Link>
      </div>
    </div>
  );
}