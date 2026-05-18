import { useEffect, useState, useCallback } from 'react';
import { apiClient } from '../../../api/client';
import './AdminPanel.css';

// ── Types ─────────────────────────────────────────────────────────────────────

interface Project { id: number; name: string; }

interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  status: string;
  projects: Project[];
  created_at?: string;
  last_login?: string | null;
  avatar_url?: string | null;
  locked_until?: string | null;
  failed_login_attempts?: number;
  job_title?: string | null;
  department?: string | null;
}

interface Stats {
  users: {
    total: number;
    pending: number;
    active: number;
    blocked: number;
    by_role: Record<string, number>;
  };
  projects: { total: number };
}

// ── Helper : extraire message d'erreur API ────────────────────────────────────

function extractError(err: any): string {
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) {
    return detail.map((e: any) => {
      const field = e.loc?.slice(1).join('.') || '';
      return field ? `${field}: ${e.msg}` : e.msg;
    }).join(' · ');
  }
  if (typeof detail === 'string') return detail;
  return err?.message || 'Une erreur est survenue';
}

// ── Helper : est-ce que le compte est verrouillé ? ────────────────────────────

function isLocked(user: User): boolean {
  return !!(user.locked_until && new Date(user.locked_until) > new Date());
}

function minutesLeft(user: User): number {
  if (!user.locked_until) return 0;
  return Math.ceil((new Date(user.locked_until).getTime() - Date.now()) / 60000);
}

// ── Helper : formater date ────────────────────────────────────────────────────

function fmt(iso?: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('fr-FR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

// ── Export CSV via apiClient (token dans header, pas dans l'URL) ──────────────

async function downloadCSV() {
  const res = await apiClient.get('/admin/users/export', { responseType: 'blob' });
  const url = URL.createObjectURL(new Blob([res.data]));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'users_export.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ═════════════════════════════════════════════════════════════════════════════
// COMPOSANT PRINCIPAL
// ═════════════════════════════════════════════════════════════════════════════

function UserAvatar({ user }: { user: User }) {
  const [imgError, setImgError] = useState(false);
  const initials = user.username.slice(0, 2).toUpperCase();
  const colors: Record<string, string> = {
    admin:     'linear-gradient(135deg, #ef4444, #dc2626)',
    manager:   'linear-gradient(135deg, #eab308, #ca8a04)',
    analyst:   'linear-gradient(135deg, #6366f1, #4f46e5)',
    developer: 'linear-gradient(135deg, #22c55e, #16a34a)',
  };
  const base = {
    width: 36, height: 36,
    borderRadius: '50%',
    border: '2px solid var(--border)',
    flexShrink: 0 as const,
  };

  if (user.avatar_url && !imgError) {
    return (
      <img
        src={user.avatar_url}
        alt={user.username}
        style={{ ...base, objectFit: 'cover' as const }}
        onError={() => setImgError(true)}
      />
    );
  }

  return (
    <div style={{
      ...base,
      background: colors[user.role] || colors.developer,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontSize: 13, fontWeight: 700, color: '#fff',
    }}>
      {initials}
    </div>
  );
}

export default function AdminPanel() {
  const [users,        setUsers]        = useState<User[]>([]);
  const [allProjects,  setAllProjects]  = useState<Project[]>([]);
  const [stats,        setStats]        = useState<Stats | null>(null);
  const [loading,      setLoading]      = useState(true);
  const [error,        setError]        = useState('');
  const [search,       setSearch]       = useState('');
  const [filterRole,   setFilterRole]   = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [actionMsg,    setActionMsg]    = useState<{ text: string; type: 'ok' | 'err' }>({ text: '', type: 'ok' });
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [assigning,    setAssigning]    = useState(false);
  const [resetting,    setResetting]    = useState(false);
  const [newPassword,  setNewPassword]  = useState('');
  const [creating,     setCreating]     = useState(false);
  const [newUser,      setNewUser]      = useState({
    username: '', email: '', password: '', role: 'developer'
  });

  // ── Fetch ───────────────────────────────────────────────────────────────────
  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [usersRes, projectsRes, statsRes] = await Promise.all([
        apiClient.get('/admin/users'),
        apiClient.get('/admin/projects'),
        apiClient.get('/admin/stats'),
      ]);
      setUsers(usersRes.data);
      setAllProjects(projectsRes.data);
      setStats(statsRes.data);
    } catch {
      setError('Impossible de charger les données.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  // ── Toast ───────────────────────────────────────────────────────────────────
  const notify = useCallback((text: string, type: 'ok' | 'err' = 'ok') => {
    setActionMsg({ text, type });
    setTimeout(() => setActionMsg({ text: '', type: 'ok' }), 3500);
  }, []);

  // ── Filtres ─────────────────────────────────────────────────────────────────
  const filteredUsers = users.filter(u => {
    const q = search.toLowerCase();
    const matchSearch = !search ||
      u.username.toLowerCase().includes(q) ||
      u.email.toLowerCase().includes(q);
    const matchRole   = !filterRole   || u.role   === filterRole;
    const matchStatus = !filterStatus || u.status === filterStatus;
    return matchSearch && matchRole && matchStatus;
  });

  // ── Actions ─────────────────────────────────────────────────────────────────
  const approveUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/approve`);
      notify('Utilisateur approuvé ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const blockUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/block`);
      notify('Utilisateur bloqué ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const unblockUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/unblock`);
      notify('Utilisateur débloqué ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const unlockUser = async (id: number) => {
    try {
      await apiClient.put(`/admin/users/${id}/unlock`);
      notify('Compte déverrouillé ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const deleteUser = async (id: number, username: string) => {
    if (!confirm(`Supprimer définitivement "${username}" ?`)) return;
    try {
      await apiClient.delete(`/admin/users/${id}`);
      notify('Utilisateur supprimé ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const changeRole = async (id: number, newRole: string) => {
    try {
      await apiClient.put(`/admin/users/${id}`, { role: newRole });
      notify('Rôle mis à jour ✓'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const assignProjects = async (userId: number, projectIds: number[]) => {
    try {
      await apiClient.post(`/admin/users/${userId}/projects`, projectIds);
      notify('Projets assignés ✓');
      setAssigning(false); setSelectedUser(null); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const resetPassword = async () => {
    if (!selectedUser) return;
    if (newPassword.length < 8) { notify('Minimum 8 caractères', 'err'); return; }
    try {
      await apiClient.post(`/admin/users/${selectedUser.id}/reset-password`, {
        new_password: newPassword,
      });
      notify('Mot de passe réinitialisé ✓');
      setResetting(false); setNewPassword(''); setSelectedUser(null);
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const createUser = async () => {
    if (!newUser.username || !newUser.email || !newUser.password) {
      notify('Tous les champs sont requis', 'err'); return;
    }
    try {
      await apiClient.post('/auth/register', {
        username: newUser.username,
        email:    newUser.email,
        password: newUser.password,
      });
      notify('Utilisateur créé ✓');
      setCreating(false);
      setNewUser({ username: '', email: '', password: '', role: 'developer' });
      fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const exportCSV = async () => {
    try {
      await downloadCSV();
      notify('Export CSV téléchargé ✓');
    } catch (err) { notify(extractError(err), 'err'); }
  };

  // ── Loading / Error ─────────────────────────────────────────────────────────
  if (loading) return <div className="admin-loading">Chargement...</div>;
  if (error)   return <div style={{ padding: '2rem', color: 'var(--accent2)' }}>{error}</div>;

  // ═══════════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════════

  return (
    <div className="admin-container">

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="admin-header">
        <h1>🛡️ Administration</h1>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => setCreating(true)} className="btn-secondary">
            + Nouvel utilisateur
          </button>
          <button onClick={exportCSV} className="btn-outline">⬇ Export CSV</button>
          <button onClick={fetchAll}  className="btn-outline">↻ Actualiser</button>
        </div>
      </div>

      {/* ── Toast ───────────────────────────────────────────────────────────── */}
      {actionMsg.text && (
        <div className={`admin-notification ${actionMsg.type === 'err' ? 'admin-notification--err' : ''}`}>
          {actionMsg.type === 'ok' ? '✓' : '⚠'} {actionMsg.text}
        </div>
      )}

      {/* ── Stats ───────────────────────────────────────────────────────────── */}
      {stats && (
        <div className="stats-grid">
          {[
            { label: 'Total',      value: stats.users.total,    color: 'var(--accent)'  },
            { label: 'Actifs',     value: stats.users.active,   color: 'var(--green)'   },
            { label: 'En attente', value: stats.users.pending,  color: 'var(--accent3)' },
            { label: 'Bloqués',    value: stats.users.blocked,  color: 'var(--accent2)' },
            { label: 'Projets',    value: stats.projects.total, color: 'var(--purple)'  },
          ].map(({ label, value, color }) => (
            <div key={label} className="stat-card" style={{ borderLeftColor: color }}>
              <div className="stat-label">{label}</div>
              <div className="stat-value" style={{ color }}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Rôles ───────────────────────────────────────────────────────────── */}
      {stats && (
        <div className="roles-grid">
          {Object.entries(stats.users.by_role).map(([role, count]) => (
            <div key={role} className="role-card">
              <div className="role-name">{role}</div>
              <div className="role-count">{count as number}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Filtres ─────────────────────────────────────────────────────────── */}
      <div className="filters-bar">
        <input
          placeholder="🔍 Rechercher..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="admin-input"
        />
        <select value={filterRole} onChange={e => setFilterRole(e.target.value)}
          className="admin-select">
          <option value="">Tous les rôles</option>
          <option value="admin">Admin</option>
          <option value="manager">Manager</option>
          <option value="analyst">Analyst</option>
          <option value="developer">Developer</option>
        </select>
        <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
          className="admin-select">
          <option value="">Tous les statuts</option>
          <option value="active">Actif</option>
          <option value="pending">En attente</option>
          <option value="blocked">Bloqué</option>
        </select>
        {(search || filterRole || filterStatus) && (
          <button
            onClick={() => { setSearch(''); setFilterRole(''); setFilterStatus(''); }}
            className="btn-outline"
          >
            ✕ Réinitialiser
          </button>
        )}
      </div>

      {/* Compteur */}
      <div style={{ color: 'var(--dimmed)', fontSize: 13, marginBottom: '1rem' }}>
        {filteredUsers.length} utilisateur{filteredUsers.length !== 1 ? 's' : ''} affiché{filteredUsers.length !== 1 ? 's' : ''}
      </div>

      {/* ── Table ───────────────────────────────────────────────────────────── */}
      <div className="table-container">
        <table className="admin-table">
          <thead>
            <tr>
              {['ID', 'Utilisateur', 'Email', 'Rôle', 'Statut', 'Sécurité', 'Projets', 'Actions'].map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredUsers.map(user => (
              <tr key={user.id}>

                {/* ID */}
                <td style={{ color: 'var(--dimmed)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>
                  #{user.id}
                </td>

                {/* Utilisateur */}
                <td>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <UserAvatar user={user} />
                    <div>
                      <div className="username">{user.username}</div>
                      {user.job_title && (
                        <div className="date">{user.job_title}
                          {user.department && ` · ${user.department}`}
                        </div>
                      )}
                      {user.created_at && (
                        <div className="date">
                          Créé {new Date(user.created_at).toLocaleDateString('fr-FR')}
                        </div>
                      )}
                    </div>
                  </div>
                </td>

                {/* Email */}
                <td className="email">{user.email}</td>

                {/* Rôle */}
                <td>
                  <select
                    value={user.role}
                    onChange={e => changeRole(user.id, e.target.value)}
                    className="admin-select"
                    style={{ padding: '4px 8px', fontSize: 13 }}
                  >
                    <option value="admin">Admin</option>
                    <option value="manager">Manager</option>
                    <option value="analyst">Analyst</option>
                    <option value="developer">Developer</option>
                  </select>
                </td>

                {/* Statut */}
                <td>
                  <span className={`status-badge ${user.status}`}>
                    {user.status === 'active'  ? '● Actif'      :
                     user.status === 'pending' ? '◐ En attente' : '○ Bloqué'}
                  </span>
                </td>

                {/* ── Sécurité (colonne nouvelle) ── */}
                <td>
                  {isLocked(user) ? (
                    <div className="security-cell security-cell--locked">
                      <span className="security-icon">🔒</span>
                      <div>
                        <div className="security-label">Verrouillé</div>
                        <div className="security-sub">{minutesLeft(user)} min restantes</div>
                      </div>
                    </div>
                  ) : (user.failed_login_attempts ?? 0) > 0 ? (
                    <div className="security-cell security-cell--warn">
                      <span className="security-icon">⚠️</span>
                      <div>
                        <div className="security-label">{user.failed_login_attempts} échec(s)</div>
                        <div className="security-sub">
                          {fmt(user.last_login)}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="security-cell security-cell--ok">
                      <div className="security-label">Secured</div>
                      <div className="security-sub">Last login: {fmt(user.last_login)}</div>
                    </div>
                  )}
                </td>

                {/* Projets */}
                <td>
                  <div style={{ fontSize: 12, color: 'var(--muted)', maxWidth: 160 }}>
                    {user.projects.length > 0
                      ? user.projects.map(p => p.name).join(', ')
                      : <span style={{ color: 'var(--dimmed)' }}>Aucun</span>
                    }
                  </div>
                  <button
                    onClick={() => { setSelectedUser(user); setAssigning(true); }}
                    className="link-btn"
                  >
                    ✎ Modifier
                  </button>
                </td>

                {/* Actions */}
                <td style={{ whiteSpace: 'nowrap' }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>

                    {/* Approve si pending */}
                    {user.status === 'pending' && (
                      <button onClick={() => approveUser(user.id)}
                        className="btn-small success" title="Approuver">
                        ✓ Approuver
                      </button>
                    )}

                    {/* Block si active */}
                    {user.status === 'active' && (
                      <button onClick={() => blockUser(user.id)}
                        className="btn-small danger" title="Bloquer">
                        Bloquer
                      </button>
                    )}

                    {/* Unblock si blocked */}
                    {user.status === 'blocked' && (
                      <button onClick={() => unblockUser(user.id)}
                        className="btn-small warning" title="Débloquer">
                        Débloquer
                      </button>
                    )}

                    {/* Unlock si verrouillé temporairement */}
                    {isLocked(user) && (
                      <button onClick={() => unlockUser(user.id)}
                        className="btn-small warning" title="Déverrouiller">
                        🔓 Déverrouiller
                      </button>
                    )}

                    {/* Reset password */}
                    <button
                      onClick={() => { setSelectedUser(user); setResetting(true); }}
                      className="btn-small info" title="Réinitialiser le mot de passe"
                    >
                      🔑
                    </button>

                    {/* Delete */}
                    <button
                      onClick={() => deleteUser(user.id, user.username)}
                      className="btn-small delete" title="Supprimer"
                    >
                      🗑
                    </button>

                  </div>
                </td>

              </tr>
            ))}
          </tbody>
        </table>

        {filteredUsers.length === 0 && (
          <div className="no-results">Aucun utilisateur trouvé</div>
        )}
      </div>

      {/* ── Modal assignation projets ────────────────────────────────────────── */}
      {assigning && selectedUser && (
        <Modal
          title={`Projets de @${selectedUser.username}`}
          onClose={() => { setAssigning(false); setSelectedUser(null); }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
            {allProjects.map(proj => (
              <label key={proj.id}
                style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', padding: '6px 0' }}>
                <input
                  type="checkbox"
                  checked={selectedUser.projects.some(p => p.id === proj.id)}
                  onChange={() => {
                    setSelectedUser(prev => {
                      if (!prev) return prev;
                      const exists = prev.projects.some(p => p.id === proj.id);
                      return {
                        ...prev,
                        projects: exists
                          ? prev.projects.filter(p => p.id !== proj.id)
                          : [...prev.projects, proj],
                      };
                    });
                  }}
                />
                <span style={{ color: 'var(--text)' }}>{proj.name}</span>
                <span style={{ fontSize: 11, color: 'var(--dimmed)' }}>#{proj.id}</span>
              </label>
            ))}
            {allProjects.length === 0 && (
              <p style={{ color: 'var(--dimmed)' }}>Aucun projet disponible</p>
            )}
          </div>
          <div style={{ marginTop: 12, fontSize: 12, color: 'var(--dimmed)' }}>
            {selectedUser.projects.length} projet(s) sélectionné(s)
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button onClick={() => { setAssigning(false); setSelectedUser(null); }}
              className="btn-outline">Annuler</button>
            <button
              onClick={() => assignProjects(selectedUser.id, selectedUser.projects.map(p => p.id))}
              className="btn-secondary">Enregistrer</button>
          </div>
        </Modal>
      )}

      {/* ── Modal reset mot de passe ─────────────────────────────────────────── */}
      {resetting && selectedUser && (
        <Modal
          title={`Réinitialiser — @${selectedUser.username}`}
          onClose={() => { setResetting(false); setSelectedUser(null); setNewPassword(''); }}
        >
          <input
            type="password"
            placeholder="Nouveau mot de passe (min 8 caractères)"
            value={newPassword}
            onChange={e => setNewPassword(e.target.value)}
            className="admin-input"
            style={{ width: '100%', boxSizing: 'border-box' }}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button
              onClick={() => { setResetting(false); setSelectedUser(null); setNewPassword(''); }}
              className="btn-outline">Annuler</button>
            <button onClick={resetPassword} className="btn-secondary">Réinitialiser</button>
          </div>
        </Modal>
      )}

      {/* ── Modal création utilisateur ───────────────────────────────────────── */}
      {creating && (
        <Modal title="Créer un utilisateur" onClose={() => setCreating(false)}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input
              placeholder="Nom d'utilisateur"
              value={newUser.username}
              onChange={e => setNewUser({ ...newUser, username: e.target.value })}
              className="admin-input"
              style={{ width: '100%', boxSizing: 'border-box' }}
            />
            <input
              placeholder="Email"
              type="email"
              value={newUser.email}
              onChange={e => setNewUser({ ...newUser, email: e.target.value })}
              className="admin-input"
              style={{ width: '100%', boxSizing: 'border-box' }}
            />
            <input
              placeholder="Mot de passe (min 8 caractères)"
              type="password"
              value={newUser.password}
              onChange={e => setNewUser({ ...newUser, password: e.target.value })}
              className="admin-input"
              style={{ width: '100%', boxSizing: 'border-box' }}
            />
            <select
              value={newUser.role}
              onChange={e => setNewUser({ ...newUser, role: e.target.value })}
              className="admin-select"
              style={{ width: '100%' }}
            >
              <option value="developer">Developer</option>
              <option value="analyst">Analyst</option>
              <option value="manager">Manager</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button onClick={() => setCreating(false)} className="btn-outline">Annuler</button>
            <button onClick={createUser} className="btn-secondary">Créer</button>
          </div>
        </Modal>
      )}

    </div>
  );
}

// ── Modal ─────────────────────────────────────────────────────────────────────

function Modal({ title, children, onClose }: {
  title: string; children: React.ReactNode; onClose: () => void;
}) {
  return (
    <div style={{
      position: 'fixed', inset: 0,
      background: 'rgba(0,0,0,0.75)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 50,
      backdropFilter: 'blur(4px)',
    }}>
      <div style={{
        background: 'var(--bg3)',
        border: '1px solid var(--border)',
        borderRadius: 16,
        padding: '1.5rem',
        width: 440,
        maxHeight: '80vh',
        overflowY: 'auto',
        boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
      }}>
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', marginBottom: '1.2rem',
        }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>
            {title}
          </h2>
          <button onClick={onClose} style={{
            background: 'none', border: 'none',
            color: 'var(--muted)', cursor: 'pointer', fontSize: 20,
            width: 32, height: 32, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            borderRadius: 8, transition: 'background 0.2s',
          }}>✕</button>
        </div>
        {children}
      </div>
    </div>
  );
}