import { useState, useRef } from 'react';
import { useAuth } from '../../../auth/hooks/useAuth';
import {
  updateProfile,
  changePassword,
  saveUser,
} from '../../../auth/services/authService';
import './ProfilePage.css';

// ── Configs ───────────────────────────────────────────────────────────────────

const ROLE_COLORS: Record<string, { bg: string; color: string; label: string; glow: string }> = {
  admin:     { bg: 'rgba(239,68,68,0.18)',   color: '#fca5a5', label: '👑 Admin',      glow: 'rgba(239,68,68,0.25)' },
  manager:   { bg: 'rgba(234,179,8,0.18)',   color: '#fde047', label: '🎯 Manager',    glow: 'rgba(234,179,8,0.25)' },
  analyst:   { bg: 'rgba(99,102,241,0.18)',  color: '#a5b4fc', label: '🔍 Analyst',    glow: 'rgba(99,102,241,0.25)' },
  developer: { bg: 'rgba(34,197,94,0.18)',   color: '#86efac', label: '⚙️ Developer',  glow: 'rgba(34,197,94,0.25)' },
};

const STATUS_CONFIG: Record<string, { color: string; label: string; dot: string; bg: string }> = {
  active:  { color: '#86efac', label: 'Actif',      dot: '#22c55e', bg: 'rgba(34,197,94,0.15)' },
  pending: { color: '#fde047', label: 'En attente', dot: '#eab308', bg: 'rgba(234,179,8,0.15)' },
  blocked: { color: '#fca5a5', label: 'Bloqué',     dot: '#ef4444', bg: 'rgba(239,68,68,0.15)' },
};

// ── Composant principal ───────────────────────────────────────────────────────

export default function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const fileRef = useRef<HTMLInputElement>(null);

  const [avatar, setAvatar] = useState<string | null>(
    localStorage.getItem(`avatar_${user?.id}`) || null
  );
  const [form, setForm] = useState({
    username: user?.username || '',
    email:    user?.email    || '',
  });
  const [passwords, setPasswords] = useState({
    current: '',
    next:    '',
    confirm: '',
  });
  const [showPass, setShowPass] = useState({ current: false, next: false, confirm: false });
  const [tab,    setTab]    = useState<'info' | 'security'>('info');
  const [saving, setSaving] = useState(false);
  const [msg,    setMsg]    = useState<{ text: string; type: 'ok' | 'err' } | null>(null);

  if (!user) return null;

  const role   = ROLE_COLORS[user.role]     || ROLE_COLORS.developer;
  const status = STATUS_CONFIG[user.status] || STATUS_CONFIG.active;
  const initials = user.username.slice(0, 2).toUpperCase();

  const notify = (text: string, type: 'ok' | 'err' = 'ok') => {
    setMsg({ text, type });
    setTimeout(() => setMsg(null), 3500);
  };

  // ── Avatar ────────────────────────────────────────────────────────────────

  const handleAvatar = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) { notify('Image trop lourde (max 2 Mo)', 'err'); return; }
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setAvatar(result);
      localStorage.setItem(`avatar_${user.id}`, result);
      notify('Photo de profil mise à jour ✓');
    };
    reader.readAsDataURL(file);
  };

  const removeAvatar = () => {
    setAvatar(null);
    localStorage.removeItem(`avatar_${user.id}`);
    notify('Photo supprimée');
  };

  // ── Sauvegarder le profil ─────────────────────────────────────────────────

  const saveProfile = async () => {
    if (!form.username.trim()) { notify('Le nom d\'utilisateur est requis', 'err'); return; }
    if (!form.email.trim())    { notify('L\'email est requis', 'err'); return; }
    setSaving(true);
    try {
      const updatedUser = await updateProfile({ username: form.username.trim(), email: form.email.trim() });
      saveUser(updatedUser);
      await refreshUser();
      notify('Profil mis à jour avec succès ✓');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Erreur lors de la mise à jour', 'err');
    } finally {
      setSaving(false);
    }
  };

  // ── Changer le mot de passe ───────────────────────────────────────────────

  const savePassword = async () => {
    if (!passwords.current)                   { notify('Mot de passe actuel requis', 'err'); return; }
    if (passwords.next.length < 8)            { notify('Minimum 8 caractères requis', 'err'); return; }
    if (passwords.next !== passwords.confirm) { notify('Les mots de passe ne correspondent pas', 'err'); return; }
    setSaving(true);
    try {
      await changePassword(passwords.current, passwords.next);
      setPasswords({ current: '', next: '', confirm: '' });
      notify('Mot de passe changé avec succès ✓');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Erreur lors du changement', 'err');
    } finally {
      setSaving(false);
    }
  };

  const strength = passwordStrength(passwords.next);

  // ── Rendu ─────────────────────────────────────────────────────────────────

  return (
    <div className="profile-root">

      {/* Background ambient glow */}
      <div className="profile-bg-glow" />

      <div className="profile-container">

        {/* ── Page Header ──────────────────────────────────────────────── */}
        <header className="profile-header">
          <div className="profile-header-left">
            <div className="profile-breadcrumb">
              <span className="breadcrumb-icon">⚙️</span>
              <span className="breadcrumb-sep">/</span>
              <span className="breadcrumb-current">Mon profil</span>
            </div>
            <h1 className="profile-title">Paramètres du profil</h1>
            <p className="profile-subtitle">Gérez vos informations personnelles, votre sécurité et vos préférences</p>
          </div>
          <div className="profile-header-right">
            <div className="profile-header-meta">
              <div className="header-stat">
                <span className="header-stat-dot" style={{ background: status.dot }} />
                <span className="header-stat-label">{status.label}</span>
              </div>
              <div className="header-divider" />
              <div className="header-role-badge" style={{ background: role.bg, color: role.color }}>
                {role.label}
              </div>
            </div>
          </div>
        </header>

        {/* ── Notification Toast ───────────────────────────────────────── */}
        {msg && (
          <div className={`profile-toast ${msg.type === 'ok' ? 'toast-ok' : 'toast-err'}`}>
            <span className="toast-icon">{msg.type === 'ok' ? '✓' : '⚠'}</span>
            <span className="toast-text">{msg.text}</span>
          </div>
        )}

        {/* ── Main Grid ────────────────────────────────────────────────── */}
        <div className="profile-grid">

          {/* ── LEFT COLUMN ─────────────────────────────────────────────── */}
          <aside className="profile-sidebar">

            {/* Avatar Card */}
            <div className="profile-card avatar-card">
              <div className="avatar-glow-ring" style={{ '--glow': role.glow } as React.CSSProperties} />
              <div className="avatar-wrapper">
                {avatar ? (
                  <img src={avatar} alt="avatar" className="avatar-img" />
                ) : (
                  <div className="avatar-initials" style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}>
                    {initials}
                  </div>
                )}
                <div className="avatar-status-dot" style={{ background: status.dot }} />
                <button className="avatar-edit-btn" onClick={() => fileRef.current?.click()} title="Changer la photo">
                  <span>📷</span>
                </button>
              </div>

              <div className="avatar-info">
                <h2 className="avatar-name">{user.username}</h2>
                <p className="avatar-email">{user.email}</p>
              </div>

              <div className="avatar-badges">
                <span className="badge-role" style={{ background: role.bg, color: role.color, boxShadow: `0 2px 12px ${role.glow}` }}>
                  {role.label}
                </span>
                <span className="badge-status" style={{ background: status.bg, color: status.color }}>
                  <span className="badge-dot" style={{ background: status.dot }} />
                  {status.label}
                </span>
              </div>

              <div className="avatar-actions">
                <button className="btn-avatar-change" onClick={() => fileRef.current?.click()}>
                  <span>📷</span> Changer la photo
                </button>
                {avatar && (
                  <button className="btn-avatar-remove" onClick={removeAvatar}>
                    <span>🗑</span> Supprimer
                  </button>
                )}
              </div>
              <input ref={fileRef} type="file" accept="image/*" onChange={handleAvatar} style={{ display: 'none' }} />
            </div>

            {/* Info Card */}
            <div className="profile-card info-card">
              <div className="card-section-label">
                <span className="section-label-dot" />
                Informations du compte
              </div>
              <div className="info-rows">
                <InfoRow label="Identifiant" value={`#${user.id}`} icon="🔑" />
                <InfoRow label="Rôle" value={role.label} icon="🎭" color={role.color} />
                <InfoRow label="Statut" value={status.label} icon="📡" color={status.color} />
                <InfoRow
                  label="Membre depuis"
                  value={user.created_at
                    ? new Date(user.created_at).toLocaleDateString('fr-FR', { year: 'numeric', month: 'long' })
                    : '—'}
                  icon="📅"
                />
              </div>
            </div>

            {/* Projects Card */}
            <div className="profile-card projects-card">
              <div className="card-section-label">
                <span className="section-label-dot" />
                Projets assignés
                <span className="projects-count">{user.projects?.length || 0}</span>
              </div>
              {user.projects && user.projects.length > 0 ? (
                <div className="projects-list">
                  {user.projects.map((p: any) => (
                    <div key={p.id} className="project-item">
                      <div className="project-dot" />
                      <span className="project-name">{p.name}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="projects-empty">
                  <span className="projects-empty-icon">📂</span>
                  <span>Aucun projet assigné</span>
                </div>
              )}
            </div>

          </aside>

          {/* ── RIGHT COLUMN ─────────────────────────────────────────────── */}
          <main className="profile-main">
            <div className="profile-card main-card">

              {/* Tabs */}
              <div className="profile-tabs">
                {(['info', 'security'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`profile-tab ${tab === t ? 'tab-active' : ''}`}
                  >
                    <span className="tab-icon">{t === 'info' ? '👤' : '🔒'}</span>
                    {t === 'info' ? 'Informations personnelles' : 'Sécurité & Mot de passe'}
                    {tab === t && <span className="tab-indicator" />}
                  </button>
                ))}
              </div>

              <div className="tab-content">

                {/* ── Tab Info ──────────────────────────────────────────── */}
                {tab === 'info' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Informations personnelles</h3>
                        <p className="panel-desc">Mettez à jour votre nom d'utilisateur et votre adresse email</p>
                      </div>
                      <div className="panel-header-icon">👤</div>
                    </div>

                    <div className="fields-grid">
                      <Field label="Nom d'utilisateur" hint="Visible par tous les membres de l'équipe" icon="@">
                        <input
                          value={form.username}
                          onChange={e => setForm({ ...form, username: e.target.value })}
                          className="profile-input"
                          placeholder="votre_username"
                        />
                      </Field>
                      <Field label="Adresse email" hint="Utilisée pour les notifications et la connexion" icon="✉">
                        <input
                          type="email"
                          value={form.email}
                          onChange={e => setForm({ ...form, email: e.target.value })}
                          className="profile-input"
                          placeholder="votre@email.com"
                        />
                      </Field>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">💡 Les modifications seront visibles immédiatement</div>
                      <button onClick={saveProfile} disabled={saving} className={`btn-primary ${saving ? 'btn-loading' : ''}`}>
                        {saving ? (
                          <><span className="btn-spinner" /> Enregistrement...</>
                        ) : (
                          <><span>✓</span> Enregistrer les modifications</>
                        )}
                      </button>
                    </div>
                  </div>
                )}

                {/* ── Tab Security ────────────────────────────────────────── */}
                {tab === 'security' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Changer le mot de passe</h3>
                        <p className="panel-desc">Utilisez un mot de passe fort d'au moins 8 caractères</p>
                      </div>
                      <div className="panel-header-icon">🔒</div>
                    </div>

                    <div className="fields-stack">
                      <Field label="Mot de passe actuel" hint="Votre mot de passe de connexion actuel" icon="🔑">
                        <div className="input-password-wrap">
                          <input
                            type={showPass.current ? 'text' : 'password'}
                            value={passwords.current}
                            onChange={e => setPasswords({ ...passwords, current: e.target.value })}
                            className="profile-input"
                            placeholder="••••••••"
                          />
                          <button className="toggle-pass" onClick={() => setShowPass(s => ({ ...s, current: !s.current }))}>
                            {showPass.current ? '🙈' : '👁'}
                          </button>
                        </div>
                      </Field>

                      <Field label="Nouveau mot de passe" hint="Minimum 8 caractères recommandé" icon="🔐">
                        <div className="input-password-wrap">
                          <input
                            type={showPass.next ? 'text' : 'password'}
                            value={passwords.next}
                            onChange={e => setPasswords({ ...passwords, next: e.target.value })}
                            className="profile-input"
                            placeholder="••••••••"
                          />
                          <button className="toggle-pass" onClick={() => setShowPass(s => ({ ...s, next: !s.next }))}>
                            {showPass.next ? '🙈' : '👁'}
                          </button>
                        </div>
                        {passwords.next && (
                          <div className="strength-meter">
                            <div className="strength-bars">
                              {[1, 2, 3, 4].map(i => (
                                <div
                                  key={i}
                                  className="strength-bar"
                                  style={{ background: strength >= i ? strengthColor(strength) : 'rgba(255,255,255,0.08)' }}
                                />
                              ))}
                            </div>
                            <span className="strength-label" style={{ color: strengthColor(strength) }}>
                              {['', 'Faible', 'Moyen', 'Fort', 'Très fort'][strength]}
                            </span>
                          </div>
                        )}
                      </Field>

                      <Field label="Confirmer le nouveau mot de passe" hint="Répétez exactement votre nouveau mot de passe" icon="✅">
                        <div className="input-password-wrap">
                          <input
                            type={showPass.confirm ? 'text' : 'password'}
                            value={passwords.confirm}
                            onChange={e => setPasswords({ ...passwords, confirm: e.target.value })}
                            className={`profile-input ${passwords.confirm && passwords.next !== passwords.confirm ? 'input-error' : ''}`}
                            placeholder="••••••••"
                          />
                          <button className="toggle-pass" onClick={() => setShowPass(s => ({ ...s, confirm: !s.confirm }))}>
                            {showPass.confirm ? '🙈' : '👁'}
                          </button>
                        </div>
                        {passwords.confirm && passwords.next !== passwords.confirm && (
                          <div className="field-error">⚠ Les mots de passe ne correspondent pas</div>
                        )}
                        {passwords.confirm && passwords.next === passwords.confirm && passwords.confirm.length > 0 && (
                          <div className="field-success">✓ Les mots de passe correspondent</div>
                        )}
                      </Field>
                    </div>

                    {/* Security Tips */}
                    <div className="security-tips">
                      <div className="tips-header">
                        <span className="tips-icon">💡</span>
                        <span className="tips-title">Conseils de sécurité</span>
                      </div>
                      <div className="tips-grid">
                        {[
                          { icon: '📏', text: 'Minimum 8 caractères' },
                          { icon: '🔡', text: 'Majuscules & minuscules' },
                          { icon: '🔢', text: 'Chiffres inclus' },
                          { icon: '✳️', text: 'Caractères spéciaux (!@#$)' },
                        ].map(tip => (
                          <div key={tip.text} className="tip-item">
                            <span>{tip.icon}</span>
                            <span>{tip.text}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">🔒 Votre session restera active après le changement</div>
                      <button onClick={savePassword} disabled={saving} className={`btn-primary ${saving ? 'btn-loading' : ''}`}>
                        {saving ? (
                          <><span className="btn-spinner" /> Enregistrement...</>
                        ) : (
                          <><span>🔒</span> Changer le mot de passe</>
                        )}
                      </button>
                    </div>
                  </div>
                )}

              </div>
            </div>
          </main>
        </div>

      </div>
    </div>
  );
}

// ── Utilitaires ───────────────────────────────────────────────────────────────

function passwordStrength(pwd: string): number {
  let score = 0;
  if (pwd.length >= 8)           score++;
  if (/[A-Z]/.test(pwd))         score++;
  if (/[0-9]/.test(pwd))         score++;
  if (/[^A-Za-z0-9]/.test(pwd))  score++;
  return score;
}

function strengthColor(score: number): string {
  return ['', '#ef4444', '#eab308', '#3b82f6', '#22c55e'][score];
}

// ── Sous-composants ───────────────────────────────────────────────────────────

function Field({ label, hint, icon, children }: {
  label: string;
  hint?: string;
  icon?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="form-field">
      <div className="field-label-row">
        {icon && <span className="field-icon">{icon}</span>}
        <label className="field-label">{label}</label>
      </div>
      {children}
      {hint && <p className="field-hint">{hint}</p>}
    </div>
  );
}

function InfoRow({ label, value, icon, color }: { label: string; value: string; icon?: string; color?: string }) {
  return (
    <div className="info-row">
      <div className="info-row-left">
        {icon && <span className="info-row-icon">{icon}</span>}
        <span className="info-row-label">{label}</span>
      </div>
      <span className="info-row-value" style={color ? { color } : undefined}>{value}</span>
    </div>
  );
}