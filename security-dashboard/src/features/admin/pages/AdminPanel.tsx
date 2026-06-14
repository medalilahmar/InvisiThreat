import { useEffect, useState, useCallback } from 'react';
import { apiClient } from '../../../api/client';
import './AdminPanel.css';

// ── Types ────────────────────────────────────────────────────────────────

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

// ── Helpers ──────────────────────────────────────────────────────────────

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

function isLocked(user: User): boolean {
  return !!(user.locked_until && new Date(user.locked_until) > new Date());
}

function minutesLeft(user: User): number {
  if (!user.locked_until) return 0;
  return Math.ceil((new Date(user.locked_until).getTime() - Date.now()) / 60000);
}

function fmt(iso?: string | null): string {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString('fr-FR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

async function downloadCSV() {
  const res = await apiClient.get('/admin/users/export', { responseType: 'blob' });
  const url = URL.createObjectURL(new Blob([res.data]));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'users_export.csv';
  a.click();
  URL.revokeObjectURL(url);
}

// ── Icons ────────────────────────────────────────────────────────────────

function IconShield() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}

function IconDownload() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  );
}

function IconRefresh() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 4v6h6M23 20v-6h-6" />
      <path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10M23 14l-4.64 4.36A9 9 0 0 1 3.51 15" />
    </svg>
  );
}

function IconPlus() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5"  y1="12" x2="19" y2="12" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconWarning() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9"  x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

function IconLock() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

function IconUnlock() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 9.9-1" />
    </svg>
  );
}

function IconKey() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="7.5" cy="15.5" r="5.5" />
      <path d="M21 2l-9.6 9.6M15.5 7.5l3 3L22 7l-3-3" />
    </svg>
  );
}

function IconTrash() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6m5 0V4a2 2 0 0 1 2-2h0a2 2 0 0 1 2 2v2" />
    </svg>
  );
}

function IconPencil() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z" />
    </svg>
  );
}

function IconSearch() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function IconX() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6"  y2="18" />
      <line x1="6"  y1="6" x2="18" y2="18" />
    </svg>
  );
}

// ── UserAvatar ───────────────────────────────────────────────────────────

const ROLE_AVATAR_CLASS: Record<string, string> = {
  admin:     'ap-avatar--admin',
  manager:   'ap-avatar--manager',
  analyst:   'ap-avatar--analyst',
  developer: 'ap-avatar--developer',
};

function UserAvatar({ user }: { user: User }) {
  const [imgError, setImgError] = useState(false);
  const initials = user.username.slice(0, 2).toUpperCase();

  if (user.avatar_url && !imgError) {
    return (
      <img
        src={user.avatar_url}
        alt={user.username}
        className="ap-avatar ap-avatar--img"
        onError={() => setImgError(true)}
      />
    );
  }

  return (
    <div className={`ap-avatar ${ROLE_AVATAR_CLASS[user.role] || ROLE_AVATAR_CLASS.developer}`}>
      {initials}
    </div>
  );
}

// ── Status badge ─────────────────────────────────────────────────────────

const STATUS_CLASSES: Record<string, string> = {
  active:  'status-badge--active',
  pending: 'status-badge--pending',
  blocked: 'status-badge--blocked',
};

const STATUS_LABELS: Record<string, string> = {
  active:  'Actif',
  pending: 'En attente',
  blocked: 'Bloqué',
};

// ═══════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

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
    username: '', email: '', password: '', role: 'developer',
  });

  // ── Fetch ──────────────────────────────────────────────────────────────
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

  // ── Toast ──────────────────────────────────────────────────────────────
  const notify = useCallback((text: string, type: 'ok' | 'err' = 'ok') => {
    setActionMsg({ text, type });
    setTimeout(() => setActionMsg({ text: '', type: 'ok' }), 3500);
  }, []);

  // ── Filtres ────────────────────────────────────────────────────────────
  const filteredUsers = users.filter(u => {
    const q = search.toLowerCase();
    const matchSearch = !search ||
      u.username.toLowerCase().includes(q) ||
      u.email.toLowerCase().includes(q);
    const matchRole   = !filterRole   || u.role   === filterRole;
    const matchStatus = !filterStatus || u.status === filterStatus;
    return matchSearch && matchRole && matchStatus;
  });

  // ── Actions ────────────────────────────────────────────────────────────
  const approveUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/approve`);
      notify('Utilisateur approuvé'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const blockUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/block`);
      notify('Utilisateur bloqué'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const unblockUser = async (id: number) => {
    try {
      await apiClient.post(`/admin/users/${id}/unblock`);
      notify('Utilisateur débloqué'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const unlockUser = async (id: number) => {
    try {
      await apiClient.put(`/admin/users/${id}/unlock`);
      notify('Compte déverrouillé'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const deleteUser = async (id: number, username: string) => {
    if (!confirm(`Supprimer définitivement "${username}" ?`)) return;
    try {
      await apiClient.delete(`/admin/users/${id}`);
      notify('Utilisateur supprimé'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const changeRole = async (id: number, newRole: string) => {
    try {
      await apiClient.put(`/admin/users/${id}`, { role: newRole });
      notify('Rôle mis à jour'); fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const assignProjects = async (userId: number, projectIds: number[]) => {
    try {
      await apiClient.post(`/admin/users/${userId}/projects`, projectIds);
      notify('Projets assignés');
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
      notify('Mot de passe réinitialisé');
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
      notify('Utilisateur créé');
      setCreating(false);
      setNewUser({ username: '', email: '', password: '', role: 'developer' });
      fetchAll();
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const exportCSV = async () => {
    try {
      await downloadCSV();
      notify('Export CSV téléchargé');
    } catch (err) { notify(extractError(err), 'err'); }
  };

  const resetFilters = () => { setSearch(''); setFilterRole(''); setFilterStatus(''); };

  // ── Loading / Error ────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="ap-state ap-state--loading">
        <div className="ap-spinner" />
        Chargement...
      </div>
    );
  }

  if (error) {
    return <div className="ap-state ap-state--error">{error}</div>;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════════

  return (
    <div className="admin-container home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="ap-inner">

        {/* ── Header ── */}
        <div className="admin-header">
          <h1 className="ap-title">
            <span className="ap-title-icon"><IconShield /></span>
            Administration
          </h1>
          <div className="ap-header-actions">
            <button onClick={() => setCreating(true)} className="btn btn-primary btn-sm">
              <IconPlus /> Nouvel utilisateur
            </button>
            <button onClick={exportCSV} className="btn btn-ghost btn-sm">
              <IconDownload /> Export CSV
            </button>
            <button onClick={fetchAll} className="btn btn-ghost btn-sm">
              <IconRefresh /> Actualiser
            </button>
          </div>
        </div>

        {/* ── Toast ── */}
        {actionMsg.text && (
          <div className={`admin-notification${actionMsg.type === 'err' ? ' admin-notification--err' : ''}`}>
            {actionMsg.type === 'ok' ? <IconCheck /> : <IconWarning />}
            {actionMsg.text}
          </div>
        )}

        {/* ── Stats ── */}
        {stats && (
          <div className="stats-grid">
            {[
              { label: 'Total',      value: stats.users.total,    cls: 'ap-stat--accent'   },
              { label: 'Actifs',     value: stats.users.active,   cls: 'ap-stat--success'  },
              { label: 'En attente', value: stats.users.pending,  cls: 'ap-stat--warning'  },
              { label: 'Bloqués',    value: stats.users.blocked,  cls: 'ap-stat--danger'   },
              { label: 'Projets',    value: stats.projects.total, cls: 'ap-stat--purple'   },
            ].map(({ label, value, cls }) => (
              <div key={label} className={`stat-card ${cls}`}>
                <div className="stat-label">{label}</div>
                <div className="stat-value">{value}</div>
              </div>
            ))}
          </div>
        )}

        {/* ── Rôles ── */}
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

        {/* ── Filtres ── */}
        <div className="filters-bar">
          <div className="ap-search-wrap">
            <span className="ap-search-icon"><IconSearch /></span>
            <input
              placeholder="Rechercher..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="admin-input ap-search-input"
            />
          </div>

          <select value={filterRole} onChange={e => setFilterRole(e.target.value)} className="admin-select">
            <option value="">Tous les rôles</option>
            <option value="admin">Admin</option>
            <option value="manager">Manager</option>
            <option value="analyst">Analyst</option>
            <option value="developer">Developer</option>
          </select>

          <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="admin-select">
            <option value="">Tous les statuts</option>
            <option value="active">Actif</option>
            <option value="pending">En attente</option>
            <option value="blocked">Bloqué</option>
          </select>

          {(search || filterRole || filterStatus) && (
            <button onClick={resetFilters} className="btn btn-ghost btn-sm">
              <IconX /> Réinitialiser
            </button>
          )}
        </div>

        {/* ── Compteur ── */}
        <div className="ap-results-count">
          {filteredUsers.length} utilisateur{filteredUsers.length !== 1 ? 's' : ''} affiché{filteredUsers.length !== 1 ? 's' : ''}
        </div>

        {/* ── Table ── */}
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
                  <td className="ap-id-cell">#{user.id}</td>

                  {/* Utilisateur */}
                  <td>
                    <div className="ap-user-cell">
                      <UserAvatar user={user} />
                      <div>
                        <div className="username">{user.username}</div>
                        {user.job_title && (
                          <div className="date">
                            {user.job_title}
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
                      className="admin-select admin-select--compact"
                    >
                      <option value="admin">Admin</option>
                      <option value="manager">Manager</option>
                      <option value="analyst">Analyst</option>
                      <option value="developer">Developer</option>
                    </select>
                  </td>

                  {/* Statut */}
                  <td>
                    <span className={`status-badge ${STATUS_CLASSES[user.status] ?? ''}`}>
                      {STATUS_LABELS[user.status] ?? user.status}
                    </span>
                  </td>

                  {/* Sécurité */}
                  <td>
                    {isLocked(user) ? (
                      <div className="security-cell security-cell--locked">
                        <span className="security-icon"><IconLock /></span>
                        <div>
                          <div className="security-label">Verrouillé</div>
                          <div className="security-sub">{minutesLeft(user)} min restantes</div>
                        </div>
                      </div>
                    ) : (user.failed_login_attempts ?? 0) > 0 ? (
                      <div className="security-cell security-cell--warn">
                        <span className="security-icon"><IconWarning /></span>
                        <div>
                          <div className="security-label">{user.failed_login_attempts} échec(s)</div>
                          <div className="security-sub">{fmt(user.last_login)}</div>
                        </div>
                      </div>
                    ) : (
                      <div className="security-cell security-cell--ok">
                        <div className="security-label">Sécurisé</div>
                        <div className="security-sub">Dernière connexion : {fmt(user.last_login)}</div>
                      </div>
                    )}
                  </td>

                  {/* Projets */}
                  <td>
                    <div className="ap-projects-cell">
                      {user.projects.length > 0
                        ? user.projects.map(p => p.name).join(', ')
                        : <span className="ap-projects-empty">Aucun</span>}
                    </div>
                    <button
                      onClick={() => { setSelectedUser(user); setAssigning(true); }}
                      className="link-btn"
                    >
                      <IconPencil /> Modifier
                    </button>
                  </td>

                  {/* Actions */}
                  <td className="ap-actions-cell">
                    <div className="ap-actions-row">

                      {user.status === 'pending' && (
                        <button onClick={() => approveUser(user.id)} className="btn-small success" title="Approuver">
                          <IconCheck /> Approuver
                        </button>
                      )}

                      {user.status === 'active' && (
                        <button onClick={() => blockUser(user.id)} className="btn-small danger" title="Bloquer">
                          Bloquer
                        </button>
                      )}

                      {user.status === 'blocked' && (
                        <button onClick={() => unblockUser(user.id)} className="btn-small warning" title="Débloquer">
                          Débloquer
                        </button>
                      )}

                      {isLocked(user) && (
                        <button onClick={() => unlockUser(user.id)} className="btn-small warning" title="Déverrouiller">
                          <IconUnlock /> Déverrouiller
                        </button>
                      )}

                      <button
                        onClick={() => { setSelectedUser(user); setResetting(true); }}
                        className="btn-small info"
                        title="Réinitialiser le mot de passe"
                      >
                        <IconKey />
                      </button>

                      <button
                        onClick={() => deleteUser(user.id, user.username)}
                        className="btn-small delete"
                        title="Supprimer"
                      >
                        <IconTrash />
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

        {/* ── Modal assignation projets ── */}
        {assigning && selectedUser && (
          <Modal
            title={`Projets de @${selectedUser.username}`}
            onClose={() => { setAssigning(false); setSelectedUser(null); }}
          >
            <div className="ap-projects-list">
              {allProjects.map(proj => (
                <label key={proj.id} className="ap-project-item">
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
                  <span className="ap-project-name">{proj.name}</span>
                  <span className="ap-project-id">#{proj.id}</span>
                </label>
              ))}
              {allProjects.length === 0 && (
                <p className="ap-empty-hint">Aucun projet disponible</p>
              )}
            </div>
            <div className="ap-selection-count">
              {selectedUser.projects.length} projet(s) sélectionné(s)
            </div>
            <div className="ap-modal-footer">
              <button onClick={() => { setAssigning(false); setSelectedUser(null); }} className="btn-outline">
                Annuler
              </button>
              <button
                onClick={() => assignProjects(selectedUser.id, selectedUser.projects.map(p => p.id))}
                className="btn-secondary"
              >
                Enregistrer
              </button>
            </div>
          </Modal>
        )}

        {/* ── Modal reset mot de passe ── */}
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
              className="admin-input ap-input-full"
            />
            <div className="ap-modal-footer">
              <button
                onClick={() => { setResetting(false); setSelectedUser(null); setNewPassword(''); }}
                className="btn-outline"
              >
                Annuler
              </button>
              <button onClick={resetPassword} className="btn-secondary">Réinitialiser</button>
            </div>
          </Modal>
        )}

        {/* ── Modal création utilisateur ── */}
        {creating && (
          <Modal title="Créer un utilisateur" onClose={() => setCreating(false)}>
            <div className="ap-form-stack">
              <input
                placeholder="Nom d'utilisateur"
                value={newUser.username}
                onChange={e => setNewUser({ ...newUser, username: e.target.value })}
                className="admin-input ap-input-full"
              />
              <input
                placeholder="Email"
                type="email"
                value={newUser.email}
                onChange={e => setNewUser({ ...newUser, email: e.target.value })}
                className="admin-input ap-input-full"
              />
              <input
                placeholder="Mot de passe (min 8 caractères)"
                type="password"
                value={newUser.password}
                onChange={e => setNewUser({ ...newUser, password: e.target.value })}
                className="admin-input ap-input-full"
              />
              <select
                value={newUser.role}
                onChange={e => setNewUser({ ...newUser, role: e.target.value })}
                className="admin-select ap-input-full"
              >
                <option value="developer">Developer</option>
                <option value="analyst">Analyst</option>
                <option value="manager">Manager</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <div className="ap-modal-footer">
              <button onClick={() => setCreating(false)} className="btn-outline">Annuler</button>
              <button onClick={createUser} className="btn-secondary">Créer</button>
            </div>
          </Modal>
        )}

      </div>
    </div>
  );
}

// ── Modal ────────────────────────────────────────────────────────────────

function Modal({ title, children, onClose }: {
  title: string; children: React.ReactNode; onClose: () => void;
}) {
  return (
    <div className="ap-modal-overlay">
      <div className="ap-modal">
        <div className="ap-modal-header">
          <h2 className="ap-modal-title">{title}</h2>
          <button onClick={onClose} className="ap-modal-close" aria-label="Fermer">
            <IconX />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}