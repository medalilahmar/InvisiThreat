import { useEffect, useState } from 'react';
import { apiClient } from '../../../api/client';

interface Project {
  id: number;
  name: string;
}

interface User {
  id: number;
  username: string;
  email: string;
  role: string;
  status: string;
  projects: Project[];
  created_at?: string;
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

export default function AdminPanel() {
  const [users, setUsers] = useState<User[]>([]);
  const [allProjects, setAllProjects] = useState<Project[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Recherche & filtres
  const [search, setSearch] = useState('');
  const [filterRole, setFilterRole] = useState('');
  const [filterStatus, setFilterStatus] = useState('');

  // Modales
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [assigning, setAssigning] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [newPassword, setNewPassword] = useState('');
  const [creating, setCreating] = useState(false);
  const [newUser, setNewUser] = useState({ username: '', email: '', password: '', role: 'developer' });
  const [actionMsg, setActionMsg] = useState('');

  // ── Chargement ────────────────────────────────────────────────────────────

  const fetchAll = async () => {
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
    } catch (err: any) {
      setError('Impossible de charger les données.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchAll(); }, []);

  const notify = (msg: string) => {
    setActionMsg(msg);
    setTimeout(() => setActionMsg(''), 3000);
  };

  // ── Filtrage local ────────────────────────────────────────────────────────

  const filteredUsers = users.filter(u => {
    const matchSearch = !search ||
      u.username.toLowerCase().includes(search.toLowerCase()) ||
      u.email.toLowerCase().includes(search.toLowerCase());
    const matchRole = !filterRole || u.role === filterRole;
    const matchStatus = !filterStatus || u.status === filterStatus;
    return matchSearch && matchRole && matchStatus;
  });

  // ── Actions utilisateurs ──────────────────────────────────────────────────

  const approveUser = async (id: number) => {
    await apiClient.post(`/admin/users/${id}/approve`);
    notify('Utilisateur approuvé ✓');
    fetchAll();
  };

  const blockUser = async (id: number) => {
    await apiClient.post(`/admin/users/${id}/block`);
    notify('Utilisateur bloqué ✓');
    fetchAll();
  };

  const unblockUser = async (id: number) => {
    await apiClient.post(`/admin/users/${id}/unblock`);
    notify('Utilisateur débloqué ✓');
    fetchAll();
  };

  const deleteUser = async (id: number, username: string) => {
    if (!confirm(`Supprimer définitivement "${username}" ?`)) return;
    await apiClient.delete(`/admin/users/${id}`);
    notify('Utilisateur supprimé ✓');
    fetchAll();
  };

  const changeRole = async (id: number, newRole: string) => {
    await apiClient.put(`/admin/users/${id}`, { role: newRole });
    notify('Rôle mis à jour ✓');
    fetchAll();
  };

  const assignProjects = async (userId: number, projectIds: number[]) => {
    try {
      await apiClient.post(`/admin/users/${userId}/projects`, projectIds);
      notify('Projets assignés ✓');
      setAssigning(false);
      setSelectedUser(null);
      fetchAll();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur lors de l\'assignation');
    }
  };

  const resetPassword = async () => {
    if (!selectedUser) return;
    if (newPassword.length < 8) { alert('Mot de passe trop court (min 8 caractères)'); return; }
    try {
      await apiClient.post(`/admin/users/${selectedUser.id}/reset-password`, { new_password: newPassword });
      notify('Mot de passe réinitialisé ✓');
      setResetting(false);
      setNewPassword('');
      setSelectedUser(null);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur');
    }
  };

  const createUser = async () => {
    if (!newUser.username || !newUser.email || !newUser.password) {
      alert('Tous les champs sont requis');
      return;
    }
    try {
      await apiClient.post('/auth/register', {
        username: newUser.username,
        email: newUser.email,
        password: newUser.password,
      });
      // Approuver automatiquement et définir le rôle
      const created = users.find(u => u.username === newUser.username);
      if (created) {
        await apiClient.post(`/admin/users/${created.id}/approve`);
        await apiClient.put(`/admin/users/${created.id}`, { role: newUser.role });
      }
      notify('Utilisateur créé ✓');
      setCreating(false);
      setNewUser({ username: '', email: '', password: '', role: 'developer' });
      fetchAll();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Erreur lors de la création');
    }
  };

  const exportCSV = () => {
    window.open('http://localhost:8081/admin/users/export?token=' + localStorage.getItem('token'));
  };

  // ── Rendu ─────────────────────────────────────────────────────────────────

  if (loading) return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0a0a0f', color: '#fff' }}>
      Chargement...
    </div>
  );

  if (error) return (
    <div style={{ padding: '2rem', color: '#f87171' }}>{error}</div>
  );

  return (
    <div style={{ padding: '2rem', background: '#0a0a0f', minHeight: '100vh', color: '#fff' }}>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700 }}>🛡️ Administration</h1>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => setCreating(true)} style={btnStyle('#6366f1')}>+ Nouvel utilisateur</button>
          <button onClick={exportCSV} style={btnStyle('#0ea5e9')}>⬇ Export CSV</button>
          <button onClick={fetchAll} style={btnStyle('#374151')}>↻ Actualiser</button>
        </div>
      </div>

      {/* Notification */}
      {actionMsg && (
        <div style={{ background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.4)', color: '#86efac', borderRadius: 8, padding: '10px 16px', marginBottom: '1rem' }}>
          ✓ {actionMsg}
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 12, marginBottom: '2rem' }}>
          <StatCard label="Total" value={stats.users.total} color="#3b82f6" />
          <StatCard label="Actifs" value={stats.users.active} color="#22c55e" />
          <StatCard label="En attente" value={stats.users.pending} color="#eab308" />
          <StatCard label="Bloqués" value={stats.users.blocked} color="#ef4444" />
          <StatCard label="Projets" value={stats.projects.total} color="#8b5cf6" />
        </div>
      )}

      {/* Rôles */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: '2rem' }}>
          {Object.entries(stats.users.by_role).map(([role, count]) => (
            <div key={role} style={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: 10, padding: '12px 16px' }}>
              <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase' }}>{role}</div>
              <div style={{ fontSize: 24, fontWeight: 700, marginTop: 4 }}>{count}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filtres */}
      <div style={{ display: 'flex', gap: 12, marginBottom: '1.5rem' }}>
        <input
          placeholder="🔍 Rechercher..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          style={inputStyle}
        />
        <select value={filterRole} onChange={e => setFilterRole(e.target.value)} style={selectStyle}>
          <option value="">Tous les rôles</option>
          <option value="admin">Admin</option>
          <option value="manager">Manager</option>
          <option value="analyst">Analyst</option>
          <option value="developer">Developer</option>
        </select>
        <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} style={selectStyle}>
          <option value="">Tous les statuts</option>
          <option value="active">Actif</option>
          <option value="pending">En attente</option>
          <option value="blocked">Bloqué</option>
        </select>
        {(search || filterRole || filterStatus) && (
          <button onClick={() => { setSearch(''); setFilterRole(''); setFilterStatus(''); }} style={btnStyle('#374151')}>
            ✕ Réinitialiser
          </button>
        )}
      </div>

      {/* Compteur */}
      <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 13, marginBottom: '1rem' }}>
        {filteredUsers.length} utilisateur{filteredUsers.length !== 1 ? 's' : ''} affiché{filteredUsers.length !== 1 ? 's' : ''}
      </div>

      {/* Table */}
      <div style={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.08)', borderRadius: 12, overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#1e1e2e' }}>
              {['ID', 'Utilisateur', 'Email', 'Rôle', 'Statut', 'Projets', 'Actions'].map(h => (
                <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontSize: 11, color: 'rgba(255,255,255,0.45)', textTransform: 'uppercase', letterSpacing: 1 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredUsers.map((user, i) => (
              <tr key={user.id} style={{ borderTop: '0.5px solid rgba(255,255,255,0.06)', background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)' }}>
                <td style={tdStyle}>{user.id}</td>
                <td style={tdStyle}>
                  <div style={{ fontWeight: 600 }}>{user.username}</div>
                  {user.created_at && <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>{new Date(user.created_at).toLocaleDateString('fr-FR')}</div>}
                </td>
                <td style={{ ...tdStyle, color: 'rgba(255,255,255,0.55)' }}>{user.email}</td>
                <td style={tdStyle}>
                  <select
                    value={user.role}
                    onChange={e => changeRole(user.id, e.target.value)}
                    style={{ background: '#1e1e2e', border: '0.5px solid rgba(255,255,255,0.12)', color: '#fff', borderRadius: 6, padding: '4px 8px', fontSize: 13 }}
                  >
                    <option value="admin">Admin</option>
                    <option value="manager">Manager</option>
                    <option value="analyst">Analyst</option>
                    <option value="developer">Developer</option>
                  </select>
                </td>
                <td style={tdStyle}>
                  <span style={{
                    padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 600,
                    background: user.status === 'active' ? 'rgba(34,197,94,0.15)' : user.status === 'pending' ? 'rgba(234,179,8,0.15)' : 'rgba(239,68,68,0.15)',
                    color: user.status === 'active' ? '#86efac' : user.status === 'pending' ? '#fde047' : '#fca5a5'
                  }}>
                    {user.status === 'active' ? '● Actif' : user.status === 'pending' ? '◐ En attente' : '○ Bloqué'}
                  </span>
                </td>
                <td style={tdStyle}>
                  <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.55)', maxWidth: 180 }}>
                    {user.projects.length > 0 ? user.projects.map(p => p.name).join(', ') : <span style={{ color: 'rgba(255,255,255,0.25)' }}>Aucun</span>}
                  </div>
                  <button onClick={() => { setSelectedUser(user); setAssigning(true); }} style={{ fontSize: 11, color: '#818cf8', background: 'none', border: 'none', cursor: 'pointer', padding: 0, marginTop: 4 }}>
                    ✎ Modifier
                  </button>
                </td>
                <td style={{ ...tdStyle, whiteSpace: 'nowrap' }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {user.status === 'pending' && (
                      <button onClick={() => approveUser(user.id)} style={btnSmall('#22c55e')}>Approuver</button>
                    )}
                    {user.status === 'active' && (
                      <button onClick={() => blockUser(user.id)} style={btnSmall('#ef4444')}>Bloquer</button>
                    )}
                    {user.status === 'blocked' && (
                      <button onClick={() => unblockUser(user.id)} style={btnSmall('#eab308')}>Débloquer</button>
                    )}
                    <button onClick={() => { setSelectedUser(user); setResetting(true); }} style={btnSmall('#6366f1')}>🔑</button>
                    <button onClick={() => deleteUser(user.id, user.username)} style={btnSmall('#374151')}>🗑</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filteredUsers.length === 0 && (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'rgba(255,255,255,0.25)' }}>
            Aucun utilisateur trouvé
          </div>
        )}
      </div>

      {/* Modal assignation projets */}
      {assigning && selectedUser && (
        <Modal title={`Projets de ${selectedUser.username}`} onClose={() => { setAssigning(false); setSelectedUser(null); }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 300, overflowY: 'auto' }}>
            {allProjects.map(proj => (
              <label key={proj.id} style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', padding: '6px 0' }}>
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
                          : [...prev.projects, proj]
                      };
                    });
                  }}
                />
                <span style={{ color: 'rgba(255,255,255,0.8)' }}>{proj.name}</span>
                <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>#{proj.id}</span>
              </label>
            ))}
            {allProjects.length === 0 && <p style={{ color: 'rgba(255,255,255,0.4)' }}>Aucun projet disponible</p>}
          </div>
          <div style={{ marginTop: 16, fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>
            {selectedUser.projects.length} projet(s) sélectionné(s)
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button onClick={() => { setAssigning(false); setSelectedUser(null); }} style={btnStyle('#374151')}>Annuler</button>
            <button onClick={() => assignProjects(selectedUser.id, selectedUser.projects.map(p => p.id))} style={btnStyle('#6366f1')}>Enregistrer</button>
          </div>
        </Modal>
      )}

      {/* Modal reset mot de passe */}
      {resetting && selectedUser && (
        <Modal title={`Réinitialiser le mot de passe de ${selectedUser.username}`} onClose={() => { setResetting(false); setSelectedUser(null); setNewPassword(''); }}>
          <input
            type="password"
            placeholder="Nouveau mot de passe (min 8 caractères)"
            value={newPassword}
            onChange={e => setNewPassword(e.target.value)}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button onClick={() => { setResetting(false); setSelectedUser(null); setNewPassword(''); }} style={btnStyle('#374151')}>Annuler</button>
            <button onClick={resetPassword} style={btnStyle('#6366f1')}>Réinitialiser</button>
          </div>
        </Modal>
      )}

      {/* Modal création utilisateur */}
      {creating && (
        <Modal title="Créer un utilisateur" onClose={() => setCreating(false)}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <input placeholder="Nom d'utilisateur" value={newUser.username} onChange={e => setNewUser({ ...newUser, username: e.target.value })} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }} />
            <input placeholder="Email" type="email" value={newUser.email} onChange={e => setNewUser({ ...newUser, email: e.target.value })} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }} />
            <input placeholder="Mot de passe" type="password" value={newUser.password} onChange={e => setNewUser({ ...newUser, password: e.target.value })} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' }} />
            <select value={newUser.role} onChange={e => setNewUser({ ...newUser, role: e.target.value })} style={{ ...selectStyle, width: '100%' }}>
              <option value="developer">Developer</option>
              <option value="analyst">Analyst</option>
              <option value="manager">Manager</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 20 }}>
            <button onClick={() => setCreating(false)} style={btnStyle('#374151')}>Annuler</button>
            <button onClick={createUser} style={btnStyle('#6366f1')}>Créer</button>
          </div>
        </Modal>
      )}
    </div>
  );
}

// ── Composants utilitaires ────────────────────────────────────────────────────

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div style={{ background: color + '22', border: `0.5px solid ${color}44`, borderRadius: 10, padding: '14px 18px' }}>
      <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase', letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 700, marginTop: 4, color: '#fff' }}>{value}</div>
    </div>
  );
}

function Modal({ title, children, onClose }: { title: string; children: React.ReactNode; onClose: () => void }) {
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }}>
      <div style={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.1)', borderRadius: 16, padding: '1.5rem', width: 420, maxHeight: '80vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>{title}</h2>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.5)', cursor: 'pointer', fontSize: 20 }}>✕</button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const tdStyle: React.CSSProperties = { padding: '12px 16px', fontSize: 14, verticalAlign: 'middle' };
const inputStyle: React.CSSProperties = { padding: '9px 14px', borderRadius: 8, background: '#1e1e2e', border: '0.5px solid rgba(255,255,255,0.12)', color: '#fff', fontSize: 14, outline: 'none' };
const selectStyle: React.CSSProperties = { padding: '9px 14px', borderRadius: 8, background: '#1e1e2e', border: '0.5px solid rgba(255,255,255,0.12)', color: '#fff', fontSize: 14, outline: 'none' };
const btnStyle = (bg: string): React.CSSProperties => ({ padding: '8px 16px', borderRadius: 8, background: bg, color: '#fff', border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 500 });
const btnSmall = (bg: string): React.CSSProperties => ({ padding: '4px 10px', borderRadius: 6, background: bg, color: '#fff', border: 'none', cursor: 'pointer', fontSize: 12 });