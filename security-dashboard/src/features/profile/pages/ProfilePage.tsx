import { useState, useRef, useCallback } from 'react';
import { useAuth } from '../../../auth/hooks/useAuth';
import {
  updateProfile,
  changePassword,
  saveUser,
} from '../../../auth/services/authService';
import './ProfilePage.css';

// ── Icons ────────────────────────────────────────────────────────────────────

function IconHome({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 10.5L12 3l9 7.5" />
      <path d="M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5" />
    </svg>
  );
}

function IconUser({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function IconLink({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}

function IconBell({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  );
}

function IconLock({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function IconCamera({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  );
}

function IconTrash({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m5 0V4a2 2 0 0 1 2-2h0a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function IconMail({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <path d="m22 6-10 7L2 6" />
    </svg>
  );
}

function IconBriefcase({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  );
}

function IconBuilding({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="2" width="16" height="20" rx="1" />
      <path d="M9 22v-5h6v5" />
    </svg>
  );
}

function IconPhone({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
    </svg>
  );
}

function IconKey({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="7.5" cy="15.5" r="5.5" />
      <path d="M21 2l-9.6 9.6M15.5 7.5l3 3L22 7l-3-3" />
    </svg>
  );
}

function IconEye({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function IconEyeOff({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.94 17.94A10.94 10.94 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  );
}

function IconCode({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="16 18 22 12 16 6" />
      <polyline points="8 6 2 12 8 18" />
    </svg>
  );
}

function IconClipboard({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
      <rect x="8" y="2" width="8" height="4" rx="1" />
    </svg>
  );
}

function IconSearch({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function IconGitMerge({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="18" cy="18" r="3" />
      <circle cx="6" cy="6" r="3" />
      <path d="M6 21V9a9 9 0 0 0 9 9" />
    </svg>
  );
}

function IconInfo({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8"  x2="12.01" y2="8" />
    </svg>
  );
}

function IconCalendar({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8"  y1="2" x2="8"  y2="6" />
      <line x1="3"  y1="10" x2="21" y2="10" />
    </svg>
  );
}

function IconClock({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function IconCheck({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconWarning({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9"  x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IconFolder({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
    </svg>
  );
}

// ── Configs ──────────────────────────────────────────────────────────────────

const ROLE_COLORS: Record<string, { bg: string; color: string; label: string; glow: string }> = {
  admin:     { bg: 'var(--severity-critical-bg)', color: 'var(--severity-critical)', label: 'Admin',     glow: 'var(--severity-critical-border)' },
  manager:   { bg: 'var(--severity-medium-bg)',   color: 'var(--severity-medium)',   label: 'Manager',   glow: 'var(--severity-medium-border)'   },
  analyst:   { bg: 'var(--severity-info-bg)',     color: 'var(--severity-info)',     label: 'Analyst',   glow: 'var(--severity-info-border)'     },
  developer: { bg: 'var(--severity-low-bg)',      color: 'var(--severity-low)',      label: 'Developer', glow: 'var(--severity-low-border)'      },
};

const STATUS_CONFIG: Record<string, { color: string; label: string; dot: string; bg: string }> = {
  active:  { color: 'var(--severity-low)',      label: 'Active',  dot: 'var(--severity-low)',      bg: 'var(--severity-low-bg)'      },
  pending: { color: 'var(--severity-medium)',   label: 'Pending', dot: 'var(--severity-medium)',   bg: 'var(--severity-medium-bg)'   },
  blocked: { color: 'var(--severity-critical)', label: 'Blocked', dot: 'var(--severity-critical)', bg: 'var(--severity-critical-bg)' },
};

type TabId = 'profile' | 'integrations' | 'notifications' | 'security';

const TABS: { id: TabId; icon: React.ReactNode; label: string }[] = [
  { id: 'profile',       icon: <IconUser />, label: 'Profil'       },
  { id: 'integrations',  icon: <IconLink />, label: 'Intégrations'  },
  { id: 'notifications', icon: <IconBell />, label: 'Notifications' },
  { id: 'security',      icon: <IconLock />, label: 'Sécurité'      },
];

// ── Helper formatDate ────────────────────────────────────────────────────────
const fmt = (iso?: string | null): string => {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
};

// ── Toggle Switch ────────────────────────────────────────────────────────────
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
      className={`pf-toggle${checked ? ' pf-toggle--on' : ''}`}
    >
      <span className="pf-toggle-thumb" />
    </button>
  );
}

// ── Password Strength ────────────────────────────────────────────────────────
function passwordStrength(pwd: string): number {
  let s = 0;
  if (pwd.length >= 8)          s++;
  if (/[A-Z]/.test(pwd))        s++;
  if (/[0-9]/.test(pwd))        s++;
  if (/[^A-Za-z0-9]/.test(pwd)) s++;
  return s;
}

const STRENGTH_COLORS = ['', 'var(--severity-critical)', 'var(--severity-medium)', 'var(--accent)', 'var(--severity-low)'];
const STRENGTH_LABELS = ['', 'Faible', 'Moyen', 'Fort', 'Très fort'];

const strengthColor = (s: number) => STRENGTH_COLORS[s];
const strengthLabel = (s: number) => STRENGTH_LABELS[s];

// ── Field wrapper ────────────────────────────────────────────────────────────
function Field({ label, hint, icon, children, optional }: {
  label: string; hint?: string; icon?: React.ReactNode;
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

// ── InfoRow ──────────────────────────────────────────────────────────────────
function InfoRow({ label, value, color, mono }: {
  label: string; value: string; color?: string; mono?: boolean;
}) {
  return (
    <div className="info-row">
      <div className="info-row-left">
        <span className="info-row-label">{label}</span>
      </div>
      <span
        className="info-row-value"
        style={{
          color: color || undefined,
          fontFamily: mono ? 'var(--font-mono)' : undefined,
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

  const role     = ROLE_COLORS[user.role]     || ROLE_COLORS.developer;
  const status   = STATUS_CONFIG[user.status] || STATUS_CONFIG.active;
  const initials = user.username.slice(0, 2).toUpperCase();

  // ── Avatar handlers ───────────────────────────────────────────────────────
  const handleAvatar = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 2 * 1024 * 1024) { notify('Image trop volumineuse (2 MB max)', 'err'); return; }
    const reader = new FileReader();
    reader.onload = async () => {
      const result = reader.result as string;
      setAvatar(result);
      localStorage.setItem(`avatar_${user.id}`, result);
      try {
        const updated = await updateProfile({ avatar_url: result });
        saveUser(updated);
        await refreshUser();
        notify('Photo de profil mise à jour');
      } catch {
        notify('Photo enregistrée localement seulement — échec de la synchronisation', 'err');
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
    notify('Photo supprimée');
  };

  // ── Save Profile ──────────────────────────────────────────────────────────
  const saveProfile = async () => {
    if (!profileForm.username.trim()) { notify('Nom d’utilisateur requis', 'err'); return; }
    if (!profileForm.email.trim())    { notify('Email requis', 'err'); return; }
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
      notify('Profil mis à jour avec succès');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Erreur lors de la mise à jour du profil', 'err');
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
      notify('Intégrations enregistrées');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Erreur lors de l’enregistrement des intégrations', 'err');
    } finally {
      setSaving(false);
    }
  };

  // ── Save Notification preference (auto-save) ─────────────────────────────
  const saveNotifPref = async (key: keyof typeof notifPrefs, value: boolean) => {
    const next = { ...notifPrefs, [key]: value };
    setNotifPrefs(next);
    setNotifSaving(true);
    try {
      const updated = await updateProfile(next);
      saveUser(updated);
      await refreshUser();
      notify('Preferences enregistrées');
    } catch {
      setNotifPrefs(notifPrefs); // rollback
      notify('Erreur lors de l’enregistrement des préférences', 'err');
    } finally {
      setNotifSaving(false);
    }
  };

  // ── Save Password ──────────────────────────────────────────────────────────
  const savePassword = async () => {
    if (!passwords.current)                   { notify('Mot de passe actuel requis', 'err'); return; }
    if (passwords.next.length < 8)            { notify('Minimum 8 caractères requis', 'err'); return; }
    if (passwords.next !== passwords.confirm) { notify('Les mots de passe ne correspondent pas', 'err'); return; }
    setSaving(true);
    try {
      await changePassword(passwords.current, passwords.next);
      setPasswords({ current: '', next: '', confirm: '' });
      notify('Mot de passe modifié avec succès');
    } catch (err: any) {
      notify(err?.response?.data?.detail || 'Erreur lors du changement de mot de passe', 'err');
    } finally {
      setSaving(false);
    }
  };

  const strength = passwordStrength(passwords.next);
  const failedAttempts = user.failed_login_attempts ?? 0;

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="profile-root home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="profile-container">

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header className="profile-header">
          <div className="profile-header-left">
            <div className="profile-breadcrumb">
              <span className="breadcrumb-icon"><IconHome /></span>
              <span className="breadcrumb-sep">/</span>
              <span className="breadcrumb-current">Paramètres</span>
            </div>
            <h1 className="profile-title">Paramètres du compte</h1>
            <p className="profile-subtitle">
              Gérez vos informations personnelles, vos intégrations et vos préférences de sécurité
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
            <span className="toast-icon">{msg.type === 'ok' ? <IconCheck /> : <IconWarning />}</span>
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
                  <div className="avatar-initials">
                    {initials}
                  </div>
                )}
                <div className="avatar-status-dot" style={{ background: status.dot }} />
                <button className="avatar-edit-btn" onClick={() => fileRef.current?.click()}>
                  <IconCamera />
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
                  <IconCamera /> Téléverser une photo
                </button>
                {avatar && (
                  <button className="btn-avatar-remove" onClick={removeAvatar}>
                    <IconTrash /> Supprimer
                  </button>
                )}
              </div>

              {/* ── URL field ── */}
              <div className="avatar-url-section">
                <div className="avatar-url-label">Ou collez une URL d’image</div>
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
                        notify('URL de l’avatar enregistrée');
                      } catch {
                        notify('URL invalide ou échec de la sauvegarde', 'err');
                      }
                    }}
                  />
                </div>
                <div className="avatar-url-hint">
                  Les images carrées fonctionnent mieux (min 200×200px)
                </div>
              </div>
              <input ref={fileRef} type="file" accept="image/*"
                onChange={handleAvatar} className="ap-hidden-input" />
            </div>

            {/* Account Info Card */}
            <div className="profile-card info-card">
              <div className="card-section-label">
                <span className="section-label-dot" />
                Informations du compte
              </div>
              <div className="info-rows">
                <InfoRow label="ID"           value={`#${user.id}`}  mono />
                <InfoRow label="Rôle"         value={role.label}     color={role.color} />
                <InfoRow label="Statut"       value={status.label}   color={status.color} />
                <InfoRow label="Membre depuis"
                  value={user.created_at
                    ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long' })
                    : '—'}
                />
                <InfoRow label="Dernière connexion"  value={fmt(user.last_login)} />
                <InfoRow label="Projets"    value={`${user.projects?.length || 0} assignés`} />
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
                      <span className="project-id">#{p.id}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="projects-empty">
                  <span className="projects-empty-icon"><IconFolder /></span>
                  <span>Aucun projet assigné</span>
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
                    className={`profile-tab${tab === t.id ? ' tab-active' : ''}`}
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
                        <h3 className="panel-title">Informations personnelles</h3>
                        <p className="panel-desc">
                          Modifiez votre nom d'utilisateur, votre email et vos informations professionnelles
                        </p>
                      </div>
                      <div className="panel-header-icon"><IconUser size={20} /></div>
                    </div>

                    {/* Section — Identity */}
                    <div className="pf-section">
                      <div className="pf-section-title">Identité</div>
                      <div className="fields-grid">
                        <Field label="Nom d'utilisateur" icon="@"
                          hint="	Visible par tous les membres de l'équipe">
                          <input
                            className="profile-input"
                            value={profileForm.username}
                            onChange={e => setProfileForm(f => ({ ...f, username: e.target.value }))}
                            placeholder="votre_nom_utilisateur"
                          />
                        </Field>
                        <Field label="Adresse e-mail" icon={<IconMail />}
                          hint="Utilisé pour la connexion et les notifications">
                          <input
                            type="email"
                            className="profile-input"
                            value={profileForm.email}
                            onChange={e => setProfileForm(f => ({ ...f, email: e.target.value }))}
                            placeholder="vous@entreprise.com"
                          />
                        </Field>
                      </div>
                    </div>

                    {/* Section — Professional */}
                    <div className="pf-section">
                      <div className="pf-section-title">Informations professionnelles</div>
                      <div className="fields-grid">
                        <Field label="Titre du poste" icon={<IconBriefcase />} optional>
                          <input
                            className="profile-input"
                            value={profileForm.job_title}
                            onChange={e => setProfileForm(f => ({ ...f, job_title: e.target.value }))}
                            placeholder="Ingénieur de sécurité"
                          />
                        </Field>
                        <Field label="Département" icon={<IconBuilding />} optional>
                          <input
                            className="profile-input"
                            value={profileForm.department}
                            onChange={e => setProfileForm(f => ({ ...f, department: e.target.value }))}
                            placeholder="DevSecOps"
                          />
                        </Field>
                        <Field label="Téléphone" icon={<IconPhone />} optional>
                          <input
                            type="tel"
                            className="profile-input"
                            value={profileForm.phone}
                            onChange={e => setProfileForm(f => ({ ...f, phone: e.target.value }))}
                            placeholder="+216 0000 0000"
                          />
                        </Field>
                      </div>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        <IconInfo /> Les modifications sont appliquées immédiatement après la sauvegarde
                      </div>
                      <button onClick={saveProfile} disabled={saving}
                        className={`btn-primary${saving ? ' btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Enregistrement…</>
                          : <><IconCheck /> Enregistrer les modifications</>
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
                        <h3 className="panel-title">Intégrations tierces</h3>
                        <p className="panel-desc">
                          Connectez vos comptes GitHub et Jira. Les jetons sont chiffrés et jamais affichés.
                        </p>
                      </div>
                      <div className="panel-header-icon"><IconLink size={20} /></div>
                    </div>

                    {/* GitHub */}
                    <div className="pf-integration-card">
                      <div className="pf-integration-header">
                        <div className="pf-integration-brand">
                          <span className="pf-integration-logo"><IconCode /></span>
                          <div>
                            <div className="pf-integration-name">GitHub</div>
                            <div className="pf-integration-desc">Code source et demandes de tirage</div>
                          </div>
                        </div>
                        <div className={`pf-integration-badge ${user.github_username ? 'badge-connected' : 'badge-disconnected'}`}>
                          {user.github_username ? 'Connecté' : 'Non connecté'}
                        </div>
                      </div>
                      <div className="fields-grid">
                        <Field label="Nom d’utilisateur GitHub" icon="@" optional>
                          <input
                            className="profile-input"
                            value={githubForm.username}
                            onChange={e => setGithubForm(f => ({ ...f, username: e.target.value }))}
                            placeholder="octocat"
                          />
                        </Field>
                        <Field label="Jeton d'accès personnel" icon={<IconKey />} optional
                          hint="Laissez vide pour conserver le jeton existant">
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
                              {showGithubToken ? <IconEyeOff /> : <IconEye />}
                            </button>
                          </div>
                        </Field>
                      </div>
                      {user.github_username && (
                        <div className="pf-integration-current">
                          <span className="pf-integration-current-label">Connecté en tant que</span>
                          <span className="pf-integration-current-value">@{user.github_username}</span>
                        </div>
                      )}
                    </div>

                    {/* Jira */}
                    <div className="pf-integration-card">
                      <div className="pf-integration-header">
                        <div className="pf-integration-brand">
                          <span className="pf-integration-logo"><IconClipboard /></span>
                          <div>
                            <div className="pf-integration-name">Jira</div>
                            <div className="pf-integration-desc">Suivi des tickets et gestion de projet</div>
                          </div>
                        </div>
                        <div className={`pf-integration-badge ${user.jira_email ? 'badge-connected' : 'badge-disconnected'}`}>
                          {user.jira_email ? 'Connecté' : 'Non connecté'}
                        </div>
                      </div>
                      <div className="fields-grid">
                        <Field label="Email Jira" icon={<IconMail />} optional>
                          <input
                            type="email"
                            className="profile-input"
                            value={jiraForm.email}
                            onChange={e => setJiraForm(f => ({ ...f, email: e.target.value }))}
                            placeholder="vous@entreprise.atlassian.net"
                          />
                        </Field>
                        <Field label="Jeton API" icon={<IconKey />} optional
                          hint="Laissez vide pour conserver le jeton existant">
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
                              {showJiraToken ? <IconEyeOff /> : <IconEye />}
                            </button>
                          </div>
                        </Field>
                      </div>
                      {user.jira_email && (
                        <div className="pf-integration-current">
                          <span className="pf-integration-current-label">Connecté en tant que</span>
                          <span className="pf-integration-current-value">{user.jira_email}</span>
                        </div>
                      )}
                    </div>

                    {/* Security note */}
                    <div className="pf-security-note">
                      <IconLock />
                      <span>Les jetons sont chiffrés côté serveur et jamais affichés après l'enregistrement.</span>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        <IconLink /> Déconnectez-vous en vidant les deux champs et en enregistrant
                      </div>
                      <button onClick={saveIntegrations} disabled={saving}
                        className={`btn-primary${saving ? ' btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Enregistrement…</>
                          : <><IconCheck /> Enregistrer les intégrations</>
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
                        <h3 className="panel-title">Préférences de notification</h3>
                        <p className="panel-desc">
                          Choisissez les événements pour lesquels vous voulez être notifié.
                          Les modifications sont enregistrées automatiquement.
                        </p>
                      </div>
                      <div className="panel-header-icon"><IconBell size={20} /></div>
                    </div>

                    <div className="pf-notif-list">

                      {/* Preference 1 */}
                      <div className="pf-notif-row">
                        <div className="pf-notif-info">
                          <div className="pf-notif-icon-wrap"><IconSearch /></div>
                          <div>
                            <div className="pf-notif-title">Nouvelle faille de sécurité</div>
                            <div className="pf-notif-desc">
                              Soyez notifié lorsqu'une nouvelle vulnérabilité est découverte dans vos projets
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
                          <div className="pf-notif-icon-wrap"><IconGitMerge /></div>
                          <div>
                            <div className="pf-notif-title">Pull request Demande</div>
                            <div className="pf-notif-desc">
                              Soyez notifié lorsqu'une pull request est fusionnée dans l'un de vos projets assignés
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
                      <IconInfo />
                      <span>
                        Les notifications sont envoyées à <strong>{user.email}</strong>.
                        Pour modifier votre adresse email de notification, mettez-la à jour dans l'onglet Profil.
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
                        <h3 className="panel-title">Paramètres de sécurité</h3>
                        <p className="panel-desc">
                          Gérez votre mot de passe et consultez l'état de sécurité de votre compte
                        </p>
                      </div>
                      <div className="panel-header-icon"><IconLock size={20} /></div>
                    </div>

                    {/* Audit timestamps */}
                    <div className="pf-section">
                      <div className="pf-section-title">Audit du compte</div>
                      <div className="pf-audit-grid">
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon"><IconCalendar /></span>
                          <div>
                            <div className="pf-audit-label">Compte créé</div>
                            <div className="pf-audit-value">{fmt(user.created_at)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon"><IconClock /></span>
                          <div>
                            <div className="pf-audit-label">Dernière connexion</div>
                            <div className="pf-audit-value">{fmt(user.last_login)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon"><IconKey /></span>
                          <div>
                            <div className="pf-audit-label">Mot de passe changé</div>
                            <div className="pf-audit-value">{fmt(user.password_changed_at)}</div>
                          </div>
                        </div>
                        <div className="pf-audit-item">
                          <span className="pf-audit-icon" style={{ color: failedAttempts > 0 ? 'var(--severity-medium)' : 'var(--severity-low)' }}>
                            {failedAttempts > 0 ? <IconWarning /> : <IconCheck />}
                          </span>
                          <div>
                            <div className="pf-audit-label">Tentatives de connexion échouées</div>
                            <div className="pf-audit-value"
                              style={{ color: failedAttempts > 0 ? 'var(--severity-medium)' : 'var(--severity-low)' }}>
                              {failedAttempts}
                              {failedAttempts === 0 ? ' — Tout va bien' : ' — Contactez ladministrateur si imprévu'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Change password */}
                    <div className="pf-section">
                      <div className="pf-section-title">Changer le mot de passe</div>
                      <div className="fields-stack">

                        <Field label="Mot de passe actuel" icon={<IconKey />}
                          hint="Votre mot de passe actuel">
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
                              {showPass.current ? <IconEyeOff /> : <IconEye />}
                            </button>
                          </div>
                        </Field>

                        <Field label="Nouveau mot de passe" icon={<IconLock />}
                          hint="Minimum 8 caractères">
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
                              {showPass.next ? <IconEyeOff /> : <IconEye />}
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
                                        : 'var(--border)',
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

                        <Field label="Confirmer le nouveau mot de passe" icon={<IconCheck />}>
                          <div className="input-password-wrap">
                            <input
                              type={showPass.confirm ? 'text' : 'password'}
                              className={`profile-input${
                                passwords.confirm && passwords.next !== passwords.confirm
                                  ? ' input-error' : ''
                              }`}
                              value={passwords.confirm}
                              onChange={e => setPasswords(p => ({ ...p, confirm: e.target.value }))}
                              placeholder="••••••••"
                            />
                            <button className="toggle-pass"
                              onClick={() => setShowPass(s => ({ ...s, confirm: !s.confirm }))}>
                              {showPass.confirm ? <IconEyeOff /> : <IconEye />}
                            </button>
                          </div>
                          {passwords.confirm && passwords.next !== passwords.confirm && (
                            <div className="field-error"><IconWarning /> Les mots de passe ne correspondent pas</div>
                          )}
                          {passwords.confirm && passwords.next === passwords.confirm && passwords.confirm.length > 0 && (
                            <div className="field-success"><IconCheck /> Les mots de passe correspondent</div>
                          )}
                        </Field>

                      </div>
                    </div>

                    {/* Tips */}
                    <div className="security-tips">
                      <div className="tips-header">
                        <span className="tips-icon"><IconInfo /></span>
                        <span className="tips-title">Conseils pour le mot de passe</span>
                      </div>
                      <div className="tips-grid">
                        {[
                          'Au moins 8 caractères',
                          'Mélanger majuscules et minuscules',
                          'Inclure des chiffres',
                          'Ajouter des caractères spéciaux (!@#$)',
                        ].map(text => (
                          <div key={text} className="tip-item">
                            <IconCheck />
                            <span>{text}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="panel-actions">
                      <div className="action-hint">
                        <IconLock /> Votre session restera active après avoir changé votre mot de passe
                      </div>
                      <button onClick={savePassword} disabled={saving}
                        className={`btn-primary${saving ? ' btn-loading' : ''}`}>
                        {saving
                          ? <><span className="btn-spinner" /> Enregistrement…</>
                          : <><IconLock /> Changer le mot de passe</>
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