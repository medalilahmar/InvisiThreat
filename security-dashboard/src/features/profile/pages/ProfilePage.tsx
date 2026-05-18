import { useState, useRef, useCallback } from 'react';
import { useAuth } from '../../../auth/hooks/useAuth';
import {
  updateProfile,
  changePassword,
  saveUser,
} from '../../../auth/services/authService';
import './ProfilePage.css';

// ── Configs ───────────────────────────────────────────────────────────────────

const ROLE_COLORS: Record<string, { bg: string; color: string; label: string; glow: string }> = {
  admin:     { bg: 'rgba(239,68,68,0.18)',   color: '#fca5a5', label: ' Admin',     glow: 'rgba(239,68,68,0.25)'   },
  manager:   { bg: 'rgba(234,179,8,0.18)',   color: '#fde047', label: ' Manager',   glow: 'rgba(234,179,8,0.25)'   },
  analyst:   { bg: 'rgba(99,102,241,0.18)',  color: '#a5b4fc', label: ' Analyst',   glow: 'rgba(99,102,241,0.25)'  },
  developer: { bg: 'rgba(34,197,94,0.18)',   color: '#86efac', label: ' Developer', glow: 'rgba(34,197,94,0.25)'   },
};

const STATUS_CONFIG: Record<string, { color: string; label: string; dot: string; bg: string }> = {
  active:  { color: '#86efac', label: 'Active',   dot: '#22c55e', bg: 'rgba(34,197,94,0.15)'  },
  pending: { color: '#fde047', label: 'Pending',  dot: '#eab308', bg: 'rgba(234,179,8,0.15)'  },
  blocked: { color: '#fca5a5', label: 'Blocked',  dot: '#ef4444', bg: 'rgba(239,68,68,0.15)'  },
};

type TabId = 'profile' | 'integrations' | 'notifications' | 'security';

const TABS: { id: TabId; icon: string; label: string }[] = [
  { id: 'profile',       icon: '', label: 'Profile'       },
  { id: 'integrations',  icon: '', label: 'Integrations'  },
  { id: 'notifications', icon: '', label: 'Notifications' },
  { id: 'security',      icon: '', label: 'Security'      },
];

// ── Helper formatDate ─────────────────────────────────────────────────────────
const fmt = (iso?: string | null): string => {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
};

// ── Toggle Switch ─────────────────────────────────────────────────────────────
function Toggle({ checked, onChange, disabled }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`pf-toggle ${checked ? 'pf-toggle--on' : ''}`}
    >
      <span className="pf-toggle-thumb" />
    </button>
  );
}

// ── Password Strength ─────────────────────────────────────────────────────────
function passwordStrength(pwd: string): number {
  let s = 0;
  if (pwd.length >= 8)           s++;
  if (/[A-Z]/.test(pwd))         s++;
  if (/[0-9]/.test(pwd))         s++;
  if (/[^A-Za-z0-9]/.test(pwd))  s++;
  return s;
}

const strengthColor = (s: number) =>
  ['', '#ef4444', '#eab308', '#3b82f6', '#22c55e'][s];

const strengthLabel = (s: number) =>
  ['', 'Weak', 'Fair', 'Strong', 'Very strong'][s];

// ── Field wrapper ─────────────────────────────────────────────────────────────
function Field({ label, hint, icon, children, optional }: {
  label: string; hint?: string; icon?: string;
  children: React.ReactNode; optional?: boolean;
}) {
  return (
    <div className="pf-field">
      <div className="pf-field-label-row">
        {icon && <span className="pf-field-icon">{icon}</span>}
        <label className="pf-field-label">{label}</label>
        {optional && <span className="pf-field-optional">optional</span>}
      </div>
      {children}
      {hint && <p className="pf-field-hint">{hint}</p>}
    </div>
  );
}

// ── InfoRow ───────────────────────────────────────────────────────────────────
function InfoRow({ label, value, icon, color, mono }: {
  label: string; value: string; icon?: string; color?: string; mono?: boolean;
}) {
  return (
    <div className="info-row">
      <div className="info-row-left">
        {icon && <span className="info-row-icon">{icon}</span>}
        <span className="info-row-label">{label}</span>
      </div>
      <span
        className="info-row-value"
        style={{
          color: color || undefined,
          fontFamily: mono ? "'JetBrains Mono', monospace" : undefined,
        }}
      >
        {value}
      </span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function ProfilePage() {
  const { user, refreshUser } = useAuth();
  const fileRef = useRef<HTMLInputElement>(null);

  // ── Avatar local ──────────────────────────────────────────────────────────
  const [avatar, setAvatar] = useState<string | null>(
    localStorage.getItem(`avatar_${user?.id}`) || user?.avatar_url || null
  );

  // ── Tab actif ─────────────────────────────────────────────────────────────
  const [tab, setTab] = useState<TabId>('profile');

  // ── Toast ─────────────────────────────────────────────────────────────────
  const [msg, setMsg] = useState<{ text: string; type: 'ok' | 'err' } | null>(null);
  const notify = useCallback((text: string, type: 'ok' | 'err' = 'ok') => {
    setMsg({ text, type });
    setTimeout(() => setMsg(null), 3500);
  }, []);

  // ── Saving state ─────────────────────────────────────────────────────────
  const [saving, setSaving] = useState(false);

  // ── Tab Profile — form ───────────────────────────────────────────────────
  const [profileForm, setProfileForm] = useState({
    username:   user?.username   || '',
    email:      user?.email      || '',
    job_title:  user?.job_title  || '',
    department: user?.department || '',
    phone:      user?.phone      || '',
  });

  // ── Tab Integrations — form ──────────────────────────────────────────────
  const [githubForm, setGithubForm] = useState({
    username: user?.github_username || '',
    token:    '',
  });
  const [jiraForm, setJiraForm] = useState({
    email: user?.jira_email || '',
    token: '',
  });
  const [showGithubToken, setShowGithubToken] = useState(false);
  const [showJiraToken,   setShowJiraToken]   = useState(false);

  // ── Tab Notifications — state ────────────────────────────────────────────
  const [notifSaving, setNotifSaving] = useState(false);
  const [notifPrefs, setNotifPrefs] = useState({
    notify_on_new_finding: user?.notify_on_new_finding ?? true,
    notify_on_pr_merged:   user?.notify_on_pr_merged   ?? true,
  });

  // ── Tab Security — form ──────────────────────────────────────────────────
  const [passwords, setPasswords] = useState({ current: '', next: '', confirm: '' });
  const [showPass, setShowPass] = useState({ current: false, next: false, confirm: false });

  if (!user) return null;

  const role   = ROLE_COLORS[user.role]     || ROLE_COLORS.developer;
  const status = STATUS_CONFIG[user.status] || STATUS_CONFIG.active;
  const initials = user.username.slice(0, 2).toUpperCase();

  // ── Avatar handlers ───────────────────────────────────────────────────────
  const handleAvatar = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) { notify('Image too large (max 2 MB)', 'err'); return; }
    const reader = new FileReader();
    reader.onload = async () => {
      const result = reader.result as string;
      setAvatar(result);
      localStorage.setItem(`avatar_${user.id}`, result);
      try {
        const updated = await updateProfile({ avatar_url: result });
        saveUser(updated);
        await refreshUser();
        notify('Profile photo updated ✓');
      } catch {
        notify('Photo saved locally only — sync failed', 'err');
      }
    };
    reader.readAsDataURL(file);
  };

  const removeAvatar = async () => {
    setAvatar(null);
    localStorage.removeItem(`avatar_${user.id}`);
    try {
      const updated = await updateProfile({ avatar_url: null });
      saveUser(updated);
      await refreshUser();
    } catch {}
    notify('Photo removed');
  };

  // ── Save Profile ──────────────────────────────────────────────────────────
  const saveProfile = async () => {
    if (!profileForm.username.trim()) { notify('Username is required', 'err'); return; }
    if (!profileForm.email.trim())    { notify('Email is required', 'err'); return; }
    setSaving(true);
    try {
      const updated = await updateProfile({
        username:   profileForm.username.trim(),
        email:      profileForm.email.trim(),
        job_title:  profileForm.job_title  || null,
        department: profileForm.department || null,
        phone:      profileForm.phone      || null,
      });
      saveUser(updated);
      await refreshUser();
      notify('Profile updated successfully ✓');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Error updating profile', 'err');
    } finally {
      setSaving(false);
    }
  };

  // ── Save Integrations ─────────────────────────────────────────────────────
  const saveIntegrations = async () => {
    setSaving(true);
    try {
      const updated = await updateProfile({
        github_username: githubForm.username || null,
        github_token:    githubForm.token    || undefined,
        jira_email:      jiraForm.email      || null,
        jira_token:      jiraForm.token      || undefined,
      });
      saveUser(updated);
      await refreshUser();
      setGithubForm(f => ({ ...f, token: '' }));
      setJiraForm(f => ({ ...f, token: '' }));
      notify('Integrations saved ✓');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Error saving integrations', 'err');
    } finally {
      setSaving(false);
    }
  };

  // ── Save Notification preference (auto-save) ──────────────────────────────
  const saveNotifPref = async (key: keyof typeof notifPrefs, value: boolean) => {
    const next = { ...notifPrefs, [key]: value };
    setNotifPrefs(next);
    setNotifSaving(true);
    try {
      const updated = await updateProfile(next);
      saveUser(updated);
      await refreshUser();
      notify('Preferences saved ✓');
    } catch {
      setNotifPrefs(notifPrefs); // rollback
      notify('Error saving preferences', 'err');
    } finally {
      setNotifSaving(false);
    }
  };

  // ── Save Password ─────────────────────────────────────────────────────────
  const savePassword = async () => {
    if (!passwords.current)                   { notify('Current password required', 'err'); return; }
    if (passwords.next.length < 8)            { notify('Minimum 8 characters required', 'err'); return; }
    if (passwords.next !== passwords.confirm) { notify('Passwords do not match', 'err'); return; }
    setSaving(true);
    try {
      await changePassword(passwords.current, passwords.next);
      setPasswords({ current: '', next: '', confirm: '' });
      notify('Password changed successfully ✓');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Error changing password', 'err');
    } finally {
      setSaving(false);
    }
  };

  const strength = passwordStrength(passwords.next);

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="profile-root">
      <div className="profile-bg-glow" />

      <div className="profile-container">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="profile-header">
          <div className="profile-header-left">
            <div className="profile-breadcrumb">
              <span className="breadcrumb-icon"></span>
              <span className="breadcrumb-sep">/</span>
              <span className="breadcrumb-current">Settings</span>
            </div>
            <h1 className="profile-title">Account Settings</h1>
            <p className="profile-subtitle">
              Manage your personal information, integrations, and security preferences
            </p>
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

        {/* ── Toast ───────────────────────────────────────────────────────── */}
        {msg && (
          <div className={`profile-toast ${msg.type === 'ok' ? 'toast-ok' : 'toast-err'}`}>
            <span className="toast-icon">{msg.type === 'ok' ? '✓' : '⚠'}</span>
            <span className="toast-text">{msg.text}</span>
          </div>
        )}

        {/* ── Grid ────────────────────────────────────────────────────────── */}
        <div className="profile-grid">

          {/* ── SIDEBAR ──────────────────────────────────────────────────── */}
          <aside className="profile-sidebar">

            {/* Avatar Card */}
            <div className="profile-card avatar-card">
              <div className="avatar-glow-ring" style={{ '--glow': role.glow } as React.CSSProperties} />

              <div className="avatar-wrapper">
                {avatar ? (
                  <img src={avatar} alt="avatar" className="avatar-img" />
                ) : (
                  <div className="avatar-initials"
                    style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}>
                    {initials}
                  </div>
                )}
                <div className="avatar-status-dot" style={{ background: status.dot }} />
                <button className="avatar-edit-btn" onClick={() => fileRef.current?.click()}>
                  <span>📷</span>
                </button>
              </div>

              <div className="avatar-info">
                <h2 className="avatar-name">{user.username}</h2>
                <p className="avatar-email">{user.email}</p>
                {user.job_title && (
                  <p className="avatar-jobtitle">{user.job_title}
                    {user.department && <span className="avatar-dept"> · {user.department}</span>}
                  </p>
                )}
              </div>

              <div className="avatar-badges">
                <span className="badge-role"
                  style={{ background: role.bg, color: role.color, boxShadow: `0 2px 12px ${role.glow}` }}>
                  {role.label}
                </span>
                <span className="badge-status" style={{ background: status.bg, color: status.color }}>
                  <span className="badge-dot" style={{ background: status.dot }} />
                  {status.label}
                </span>
              </div>

              <div className="avatar-actions">
                <button className="btn-avatar-change" onClick={() => fileRef.current?.click()}>
                  <span>📷</span> Upload photo
                </button>
                {avatar && (
                  <button className="btn-avatar-remove" onClick={removeAvatar}>
                    <span>🗑</span> Remove
                  </button>
                )}
              </div>

              {/* ── URL field ── */}
              <div className="avatar-url-section">
                <div className="avatar-url-label">-</div>
                <div className="avatar-url-row">
                  <input
                    className="profile-input avatar-url-input"
                    type="text"
                    placeholder="https://example.com/photo.jpg"
                    defaultValue={user.avatar_url || ''}
                    onBlur={async (e) => {
                      const url = e.target.value.trim();
                      if (!url || url === (user.avatar_url || '')) return;
                      try {
                        const updated = await updateProfile({ avatar_url: url });
                        saveUser(updated);
                        await refreshUser();
                        setAvatar(url);
                        localStorage.setItem(`avatar_${user.id}`, url);
                        notify('Avatar URL saved ✓');
                      } catch {
                        notify('Invalid URL or save failed', 'err');
                      }
                    }}
                  />
                </div>
                <div className="avatar-url-hint">
                  -
                </div>
              </div>
              <input ref={fileRef} type="file" accept="image/*"
                onChange={handleAvatar} style={{ display: 'none' }} />
            </div>

            {/* Account Info Card */}
            <div className="profile-card info-card">
              <div className="card-section-label">
                <span className="section-label-dot" />
                Account info
              </div>
              <div className="info-rows">
                <InfoRow label="ID"          value={`#${user.id}`}  icon="" mono />
                <InfoRow label="Role"        value={role.label}     icon="" color={role.color} />
                <InfoRow label="Status"      value={status.label}   icon="" color={status.color} />
                <InfoRow label="Member since"
                  value={user.created_at
                    ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long' })
                    : '—'}
                  icon=""
                />
                <InfoRow label="Last login"  value={fmt(user.last_login)}  icon="" />
                <InfoRow label="Projects"    value={`${user.projects?.length || 0} assigned`} icon="" />
              </div>
            </div>

            {/* Projects Card */}
            <div className="profile-card projects-card">
              <div className="card-section-label">
                <span className="section-label-dot" />
                Assigned projects
                <span className="projects-count">{user.projects?.length || 0}</span>
              </div>
              {user.projects && user.projects.length > 0 ? (
                <div className="projects-list">
                  {user.projects.map((p: any) => (
                    <div key={p.id} className="project-item">
                      <div className="project-dot" />
                      <span className="project-name">{p.name}</span>
                      <span className="project-id">#{p.id}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="projects-empty">
                  <span className="projects-empty-icon">📂</span>
                  <span>No projects assigned</span>
                </div>
              )}
            </div>

          </aside>

          {/* ── MAIN ─────────────────────────────────────────────────────── */}
          <main className="profile-main">
            <div className="profile-card main-card">

              {/* Tabs */}
              <div className="profile-tabs">
                {TABS.map(t => (
                  <button
                    key={t.id}
                    onClick={() => setTab(t.id)}
                    className={`profile-tab ${tab === t.id ? 'tab-active' : ''}`}
                  >
                    <span className="tab-icon">{t.icon}</span>
                    <span className="tab-label">{t.label}</span>
                    {tab === t.id && <span className="tab-indicator" />}
                  </button>
                ))}
              </div>

              <div className="tab-content">

                {/* ══════════════════════════════════════════════════════════
                    TAB — PROFILE
                ══════════════════════════════════════════════════════════ */}
                {tab === 'profile' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Personal information</h3>
                        <p className="panel-desc">
                          Update your username, email and professional details
                        </p>
                      </div>
                      <div className="panel-header-icon">👤</div>
                    </div>

                    {/* Section — Identity */}
                    <div className="pf-section">
                      <div className="pf-section-title">Identity</div>
                      <div className="fields-grid">
                        <Field label="Username" icon="@"
                          hint="Visible to all team members">
                          <input
                            className="profile-input"
                            value={profileForm.username}
                            onChange={e => setProfileForm(f => ({ ...f, username: e.target.value }))}
                            placeholder="your_username"
                          />
                        </Field>
                        <Field label="Email address" icon="✉"
                          hint="Used for login and notifications">
                          <input
                            type="email"
                            className="profile-input"
                            value={profileForm.email}
                            onChange={e => setProfileForm(f => ({ ...f, email: e.target.value }))}
                            placeholder="you@company.com"
                          />
                        </Field>
                      </div>
                    </div>

                    {/* Section — Professional */}
                    <div className="pf-section">
                      <div className="pf-section-title">Professional details</div>
                      <div className="fields-grid">
                        <Field label="Job title" icon="💼" optional>
                          <input
                            className="profile-input"
                            value={profileForm.job_title}
                            onChange={e => setProfileForm(f => ({ ...f, job_title: e.target.value }))}
                            placeholder="Security Engineer"
                          />
                        </Field>
                        <Field label="Department" icon="🏢" optional>
                          <input
                            className="profile-input"
                            value={profileForm.department}
                            onChange={e => setProfileForm(f => ({ ...f, department: e.target.value }))}
                            placeholder="DevSecOps"
                          />
                        </Field>
                        <Field label="Phone" icon="📞" optional>
                          <input
                            type="tel"
                            className="profile-input"
                            value={profileForm.phone}
                            onChange={e => setProfileForm(f => ({ ...f, phone: e.target.value }))}
                            placeholder="+1 555 000 0000"
                          />
                        </Field>
                      </div>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        💡 Changes are applied immediately after saving
                      </div>
                      <button onClick={saveProfile} disabled={saving}
                        className={`btn-primary ${saving ? 'btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Saving…</>
                          : <><span>✓</span> Save changes</>
                        }
                      </button>
                    </div>
                  </div>
                )}

                {/* ══════════════════════════════════════════════════════════
                    TAB — INTEGRATIONS
                ══════════════════════════════════════════════════════════ */}
                {tab === 'integrations' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Third-party integrations</h3>
                        <p className="panel-desc">
                          Connect your GitHub and Jira accounts. Tokens are encrypted and never displayed.
                        </p>
                      </div>
                      <div className="panel-header-icon">🔗</div>
                    </div>

                    {/* GitHub */}
                    <div className="pf-integration-card">
                      <div className="pf-integration-header">
                        <div className="pf-integration-brand">
                          <span className="pf-integration-logo">⚡</span>
                          <div>
                            <div className="pf-integration-name">GitHub</div>
                            <div className="pf-integration-desc">Source code & pull requests</div>
                          </div>
                        </div>
                        <div className={`pf-integration-badge ${user.github_username ? 'badge-connected' : 'badge-disconnected'}`}>
                          {user.github_username ? '● Connected' : '○ Not connected'}
                        </div>
                      </div>
                      <div className="fields-grid">
                        <Field label="GitHub username" icon="@" optional>
                          <input
                            className="profile-input"
                            value={githubForm.username}
                            onChange={e => setGithubForm(f => ({ ...f, username: e.target.value }))}
                            placeholder="octocat"
                          />
                        </Field>
                        <Field label="Personal access token" icon="🔑" optional
                          hint="Leave blank to keep existing token">
                          <div className="input-password-wrap">
                            <input
                              type={showGithubToken ? 'text' : 'password'}
                              className="profile-input"
                              value={githubForm.token}
                              onChange={e => setGithubForm(f => ({ ...f, token: e.target.value }))}
                              placeholder={user.github_username ? '••••••••••••••••' : 'ghp_xxxxxxxxxxxx'}
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowGithubToken(v => !v)}>
                              {showGithubToken ? '🙈' : '👁'}
                            </button>
                          </div>
                        </Field>
                      </div>
                      {user.github_username && (
                        <div className="pf-integration-current">
                          <span className="pf-integration-current-label">Connected as</span>
                          <span className="pf-integration-current-value">@{user.github_username}</span>
                        </div>
                      )}
                    </div>

                    {/* Jira */}
                    <div className="pf-integration-card">
                      <div className="pf-integration-header">
                        <div className="pf-integration-brand">
                          <span className="pf-integration-logo">📋</span>
                          <div>
                            <div className="pf-integration-name">Jira</div>
                            <div className="pf-integration-desc">Issue tracking & project management</div>
                          </div>
                        </div>
                        <div className={`pf-integration-badge ${user.jira_email ? 'badge-connected' : 'badge-disconnected'}`}>
                          {user.jira_email ? '● Connected' : '○ Not connected'}
                        </div>
                      </div>
                      <div className="fields-grid">
                        <Field label="Jira email" icon="✉" optional>
                          <input
                            type="email"
                            className="profile-input"
                            value={jiraForm.email}
                            onChange={e => setJiraForm(f => ({ ...f, email: e.target.value }))}
                            placeholder="you@company.atlassian.net"
                          />
                        </Field>
                        <Field label="API token" icon="🔑" optional
                          hint="Leave blank to keep existing token">
                          <div className="input-password-wrap">
                            <input
                              type={showJiraToken ? 'text' : 'password'}
                              className="profile-input"
                              value={jiraForm.token}
                              onChange={e => setJiraForm(f => ({ ...f, token: e.target.value }))}
                              placeholder={user.jira_email ? '••••••••••••••••' : 'ATATxxxxxxxx'}
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowJiraToken(v => !v)}>
                              {showJiraToken ? '🙈' : '👁'}
                            </button>
                          </div>
                        </Field>
                      </div>
                      {user.jira_email && (
                        <div className="pf-integration-current">
                          <span className="pf-integration-current-label">Connected as</span>
                          <span className="pf-integration-current-value">{user.jira_email}</span>
                        </div>
                      )}
                    </div>

                    {/* Security note */}
                    <div className="pf-security-note">
                      <span>🔐</span>
                      <span>Tokens are encrypted server-side and never displayed after saving.</span>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        🔗 Disconnect by clearing both fields and saving
                      </div>
                      <button onClick={saveIntegrations} disabled={saving}
                        className={`btn-primary ${saving ? 'btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Saving…</>
                          : <><span>✓</span> Save integrations</>
                        }
                      </button>
                    </div>
                  </div>
                )}

                {/* ══════════════════════════════════════════════════════════
                    TAB — NOTIFICATIONS
                ══════════════════════════════════════════════════════════ */}
                {tab === 'notifications' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Notification preferences</h3>
                        <p className="panel-desc">
                          Choose what events you want to be notified about.
                          Changes are saved automatically.
                        </p>
                      </div>
                      <div className="panel-header-icon">🔔</div>
                    </div>

                    <div className="pf-notif-list">

                      {/* Preference 1 */}
                      <div className="pf-notif-row">
                        <div className="pf-notif-info">
                          <div className="pf-notif-icon-wrap">🔍</div>
                          <div>
                            <div className="pf-notif-title">New security finding</div>
                            <div className="pf-notif-desc">
                              Get notified when a new vulnerability or finding is discovered in your projects
                            </div>
                          </div>
                        </div>
                        <Toggle
                          checked={notifPrefs.notify_on_new_finding}
                          onChange={v => saveNotifPref('notify_on_new_finding', v)}
                          disabled={notifSaving}
                        />
                      </div>

                      {/* Preference 2 */}
                      <div className="pf-notif-row">
                        <div className="pf-notif-info">
                          <div className="pf-notif-icon-wrap">🔀</div>
                          <div>
                            <div className="pf-notif-title">Pull request merged</div>
                            <div className="pf-notif-desc">
                              Get notified when a pull request is merged in one of your assigned projects
                            </div>
                          </div>
                        </div>
                        <Toggle
                          checked={notifPrefs.notify_on_pr_merged}
                          onChange={v => saveNotifPref('notify_on_pr_merged', v)}
                          disabled={notifSaving}
                        />
                      </div>

                    </div>

                    {/* Info box */}
                    <div className="pf-notif-info-box">
                      <span>💡</span>
                      <span>
                        Notifications are sent to <strong>{user.email}</strong>.
                        To change your notification email, update it in the Profile tab.
                      </span>
                    </div>
                  </div>
                )}

                {/* ══════════════════════════════════════════════════════════
                    TAB — SECURITY
                ══════════════════════════════════════════════════════════ */}
                {tab === 'security' && (
                  <div className="tab-panel">
                    <div className="panel-header">
                      <div className="panel-header-text">
                        <h3 className="panel-title">Security settings</h3>
                        <p className="panel-desc">
                          Manage your password and review your account security status
                        </p>
                      </div>
                      <div className="panel-header-icon">🔒</div>
                    </div>

                    {/* Audit timestamps */}
                    <div className="pf-section">
                      <div className="pf-section-title">Account audit</div>
                      <div className="pf-audit-grid">
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon">📅</span>
                          <div>
                            <div className="pf-audit-label">Account created</div>
                            <div className="pf-audit-value">{fmt(user.created_at)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon">🕐</span>
                          <div>
                            <div className="pf-audit-label">Last login</div>
                            <div className="pf-audit-value">{fmt(user.last_login)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon">🔑</span>
                          <div>
                            <div className="pf-audit-label">Password changed</div>
                            <div className="pf-audit-value">{fmt(user.password_changed_at)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon">
                            {(user.failed_login_attempts ?? 0) > 0 ? '⚠️' : '✅'}
                          </span>
                          <div>
                            <div className="pf-audit-label">Failed login attempts</div>
                            <div className="pf-audit-value"
                              style={{ color: (user.failed_login_attempts ?? 0) > 0 ? '#fbbf24' : '#86efac' }}>
                              {user.failed_login_attempts ?? 0}
                              {(user.failed_login_attempts ?? 0) === 0 ? ' — All good' : ' — Contact admin if unexpected'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Change password */}
                    <div className="pf-section">
                      <div className="pf-section-title">Change password</div>
                      <div className="fields-stack">

                        <Field label="Current password" icon="🔑"
                          hint="Your current login password">
                          <div className="input-password-wrap">
                            <input
                              type={showPass.current ? 'text' : 'password'}
                              className="profile-input"
                              value={passwords.current}
                              onChange={e => setPasswords(p => ({ ...p, current: e.target.value }))}
                              placeholder="••••••••"
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowPass(s => ({ ...s, current: !s.current }))}>
                              {showPass.current ? '🙈' : '👁'}
                            </button>
                          </div>
                        </Field>

                        <Field label="New password" icon="🔐"
                          hint="Minimum 8 characters">
                          <div className="input-password-wrap">
                            <input
                              type={showPass.next ? 'text' : 'password'}
                              className="profile-input"
                              value={passwords.next}
                              onChange={e => setPasswords(p => ({ ...p, next: e.target.value }))}
                              placeholder="••••••••"
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowPass(s => ({ ...s, next: !s.next }))}>
                              {showPass.next ? '🙈' : '👁'}
                            </button>
                          </div>
                          {passwords.next && (
                            <div className="strength-meter">
                              <div className="strength-bars">
                                {[1, 2, 3, 4].map(i => (
                                  <div key={i} className="strength-bar"
                                    style={{
                                      background: strength >= i
                                        ? strengthColor(strength)
                                        : 'rgba(255,255,255,0.08)',
                                    }} />
                                ))}
                              </div>
                              <span className="strength-label"
                                style={{ color: strengthColor(strength) }}>
                                {strengthLabel(strength)}
                              </span>
                            </div>
                          )}
                        </Field>

                        <Field label="Confirm new password" icon="✅">
                          <div className="input-password-wrap">
                            <input
                              type={showPass.confirm ? 'text' : 'password'}
                              className={`profile-input ${
                                passwords.confirm && passwords.next !== passwords.confirm
                                  ? 'input-error' : ''
                              }`}
                              value={passwords.confirm}
                              onChange={e => setPasswords(p => ({ ...p, confirm: e.target.value }))}
                              placeholder="••••••••"
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowPass(s => ({ ...s, confirm: !s.confirm }))}>
                              {showPass.confirm ? '🙈' : '👁'}
                            </button>
                          </div>
                          {passwords.confirm && passwords.next !== passwords.confirm && (
                            <div className="field-error">⚠ Passwords do not match</div>
                          )}
                          {passwords.confirm && passwords.next === passwords.confirm && passwords.confirm.length > 0 && (
                            <div className="field-success">✓ Passwords match</div>
                          )}
                        </Field>

                      </div>
                    </div>

                    {/* Tips */}
                    <div className="security-tips">
                      <div className="tips-header">
                        <span className="tips-icon">💡</span>
                        <span className="tips-title">Password tips</span>
                      </div>
                      <div className="tips-grid">
                        {[
                          { icon: '📏', text: 'At least 8 characters' },
                          { icon: '🔡', text: 'Mix uppercase & lowercase' },
                          { icon: '🔢', text: 'Include numbers' },
                          { icon: '✳️', text: 'Add special characters (!@#$)' },
                        ].map(tip => (
                          <div key={tip.text} className="tip-item">
                            <span>{tip.icon}</span>
                            <span>{tip.text}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        🔒 Your session will remain active after changing your password
                      </div>
                      <button onClick={savePassword} disabled={saving}
                        className={`btn-primary ${saving ? 'btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Saving…</>
                          : <><span>🔒</span> Change password</>
                        }
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