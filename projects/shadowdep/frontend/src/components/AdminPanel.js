import React, { useState, useEffect } from 'react';
import api from '../services/api';

/**
 * AdminPanel Component
 *
 * DAST Findings:
 *   - Broken Access Control: accessible to any authenticated user (CWE-285)
 *   - Command injection via /admin/exec (CWE-78)
 *   - Privilege escalation via /admin/users/role (CWE-269)
 *   - Sensitive data exposure: displays password hashes (CWE-312)
 *   - Client-side admin check via localStorage — easily bypassed (CWE-807)
 */
function AdminPanel() {
  const [users,       setUsers]       = useState([]);
  const [cmd,         setCmd]         = useState('whoami');
  const [cmdOutput,   setCmdOutput]   = useState('');
  const [cmdLoading,  setCmdLoading]  = useState(false);
  const [selectedUid, setSelectedUid] = useState('');
  const [newRole,     setNewRole]     = useState('admin');
  const [serverInfo,  setServerInfo]  = useState(null);
  const [activeTab,   setActiveTab]   = useState('users');

  useEffect(() => {
    // FIXME: Admin check is client-side only — trivially bypassed
    // Just set localStorage.setItem('role','admin') in browser console
    const role = localStorage.getItem('role');
    if (role !== 'admin') {
      console.warn('[SECURITY] Non-admin user accessing admin panel! Role from localStorage:', role);
      // Note: This warning is shown but page still loads — security theater
      // A real fix would redirect: window.location.href = '/dashboard';
    }

    fetchPanelData();
  }, []);

  const fetchPanelData = async () => {
    try {
      // /admin/panel returns users (with passwords) + env vars
      // requireAdmin middleware only checks token validity, not role
      const res = await api.get('/admin/panel');
      setUsers(res.data.users || []);
      setServerInfo(res.data.serverInfo || {});
    } catch (err) {
      // Fallback: try the public /api/users endpoint (no auth at all)
      try {
        const pub = await api.get('/api/users');
        setUsers(pub.data);
      } catch (e) {
        console.error('Cannot fetch users:', e);
      }
    }
  };

  const executeCommand = async () => {
    setCmdLoading(true);
    setCmdOutput('');
    try {
      // FIXME: Command injection on the server side
      // Any authenticated user can execute system commands
      // Test payloads:
      //   whoami
      //   dir C:\
      //   type C:\Windows\win.ini
      //   net user
      const res = await api.get(`/admin/exec?cmd=${encodeURIComponent(cmd)}`);
      setCmdOutput(res.data.output || '(no output)');
    } catch (err) {
      setCmdOutput(
        '❌ Error:\n' +
        (err.response?.data?.error || err.message) + '\n\n' +
        (err.response?.data?.stderr || '') + '\n\n' +
        // Stack trace from server also shown to user
        (err.response?.data?.stack || '')
      );
    } finally {
      setCmdLoading(false);
    }
  };

  const changeUserRole = async () => {
    if (!selectedUid) return alert('Select a user first');
    try {
      // Privilege escalation: any authenticated user can make anyone admin
      const res = await api.post('/admin/users/role', {
        userId: parseInt(selectedUid),
        role:   newRole
      });
      alert(`Role updated for user #${selectedUid} → ${newRole}`);
      fetchPanelData();
    } catch (err) {
      alert('Error: ' + (err.response?.data?.error || err.message));
    }
  };

  const s = {
    page:       { minHeight: '100vh', background: '#0f172a', fontFamily: "'Segoe UI',Arial,sans-serif" },
    navbar:     { background: 'rgba(30,41,59,0.95)', borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '0 32px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    brand:      { fontSize: 20, fontWeight: 700, color: '#818cf8' },
    container:  { maxWidth: 1200, margin: '0 auto', padding: '32px 24px' },
    card:       { background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, padding: 28, marginBottom: 24 },
    h2:         { color: '#e2e8f0', fontSize: 22, fontWeight: 700, marginBottom: 20 },
    termInput:  { flex: 1, padding: '12px 16px', background: '#111', border: '1px solid #333', borderRadius: 8, color: '#00ff00', fontFamily: 'monospace', fontSize: 14, outline: 'none' },
    termBtn:    { padding: '12px 24px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 700 },
    termOutput: { background: '#0a0a0a', border: '1px solid #222', borderRadius: 8, padding: 16, marginTop: 16, color: '#00ff00', fontFamily: 'monospace', fontSize: 13, whiteSpace: 'pre-wrap', minHeight: 80, maxHeight: 300, overflowY: 'auto' },
    table:      { width: '100%', borderCollapse: 'collapse' },
    th:         { padding: '10px 12px', textAlign: 'left', color: '#64748b', fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, borderBottom: '1px solid rgba(255,255,255,0.06)' },
    td:         { padding: '10px 12px', color: '#e2e8f0', fontSize: 13, borderBottom: '1px solid rgba(255,255,255,0.04)' },
    tdMono:     { padding: '10px 12px', color: '#ef4444', fontSize: 11, fontFamily: 'monospace', borderBottom: '1px solid rgba(255,255,255,0.04)', maxWidth: 300, wordBreak: 'break-all' },
    btnPrimary: { padding: '8px 16px', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 600 },
    btnWarn:    { padding: '8px 16px', background: '#f59e0b', color: '#000', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 13, fontWeight: 700 },
    tab:        (active) => ({ padding: '10px 20px', border: 'none', borderRadius: '8px 8px 0 0', cursor: 'pointer', fontSize: 14, fontWeight: 600, background: active ? 'rgba(99,102,241,0.2)' : 'transparent', color: active ? '#818cf8' : '#64748b' }),
    warn:       { background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '12px 16px', color: '#fca5a5', fontSize: 13, marginBottom: 20 },
    envBox:     { background: '#0a0a0a', border: '1px solid #222', borderRadius: 8, padding: 16, color: '#22c55e', fontFamily: 'monospace', fontSize: 11, whiteSpace: 'pre-wrap', maxHeight: 300, overflowY: 'auto' },
  };

  return (
    <div style={s.page}>
      <nav style={s.navbar}>
        <span style={s.brand}>🌑 ShadowDep</span>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          {/* Role from localStorage — tamper-prone */}
          <span style={{ color: '#fbbf24', fontSize: 13 }}>
            Role (localStorage): <strong>{localStorage.getItem('role') || 'user'}</strong>
          </span>
          <a href="/dashboard" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>← Dashboard</a>
        </div>
      </nav>

      <div style={s.container}>

        <div style={s.warn}>
          ⚠ <strong>Broken Access Control:</strong> This admin panel is accessible to ANY authenticated user
          because <code>requireAdmin</code> only checks token validity, not admin role.
          Even if role check were here, <code>localStorage.getItem('role')</code> is easily spoofed.
        </div>

        {/* ─── Tabs ──────────────────────────────────────────────────── */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 24, borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
          {['users', 'exec', 'env', 'roles'].map(tab => (
            <button key={tab} style={s.tab(activeTab === tab)} onClick={() => setActiveTab(tab)}>
              {{ users: '👥 Users', exec: '💻 Command Exec', env: '🔑 Environment', roles: '⚡ Privilege Escalation' }[tab]}
            </button>
          ))}
        </div>

        {/* ── USERS TAB ──────────────────────────────────────────────── */}
        {activeTab === 'users' && (
          <div style={s.card}>
            <h2 style={s.h2}>👥 All Users (including password column)</h2>
            <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
              Data from <code>GET /api/users</code> — public endpoint, no authentication required.
              The <strong style={{ color: '#ef4444' }}>Password</strong> column is returned by the API.
            </p>
            <table style={s.table}>
              <thead>
                <tr>
                  <th style={s.th}>ID</th>
                  <th style={s.th}>Username</th>
                  <th style={s.th}>Email</th>
                  <th style={s.th} title="CWE-312: Cleartext Storage">Password ⚠</th>
                  <th style={s.th}>Role</th>
                  <th style={s.th}>Created</th>
                  <th style={s.th}>Select</th>
                </tr>
              </thead>
              <tbody>
                {users.map(user => (
                  <tr key={user.id} style={{ cursor: 'pointer', background: selectedUid === String(user.id) ? 'rgba(99,102,241,0.1)' : 'transparent' }}>
                    <td style={s.td}>{user.id}</td>
                    <td style={s.td}><strong>{user.username}</strong></td>
                    <td style={s.td}>{user.email}</td>
                    <td style={s.tdMono}>{user.password}</td>{/* FIXME: Displaying passwords! */}
                    <td style={{ ...s.td, color: user.role === 'admin' ? '#f59e0b' : '#10b981' }}>
                      <strong>{user.role}</strong>
                    </td>
                    <td style={{ ...s.td, color: '#475569', fontSize: 12 }}>
                      {user.created_at ? new Date(user.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td style={s.td}>
                      <button style={s.btnPrimary} onClick={() => setSelectedUid(String(user.id))}>
                        Select
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── COMMAND EXEC TAB ───────────────────────────────────────── */}
        {activeTab === 'exec' && (
          <div style={s.card}>
            <h2 style={s.h2}>💻 Remote Command Execution</h2>
            <div style={s.warn}>
              ⚠ <strong>CWE-78 — OS Command Injection:</strong> Command executed via
              <code> child_process.exec('echo "Executing: ' + cmd + '" && ' + cmd)</code> with no sanitization.
            </div>
            <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
              Try: <code style={{ color: '#00ff00' }}>whoami</code> &nbsp;|&nbsp;
              <code style={{ color: '#00ff00' }}>dir</code> &nbsp;|&nbsp;
              <code style={{ color: '#00ff00' }}>type C:\Windows\win.ini</code> &nbsp;|&nbsp;
              <code style={{ color: '#00ff00' }}>net user</code>
            </p>
            <div style={{ display: 'flex', gap: 12, marginBottom: 8 }}>
              <input
                style={s.termInput}
                value={cmd}
                onChange={e => setCmd(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && executeCommand()}
                placeholder="Enter system command..."
              />
              <button style={s.termBtn} onClick={executeCommand} disabled={cmdLoading}>
                {cmdLoading ? '⏳ Running...' : '▶ Execute'}
              </button>
            </div>
            {cmdOutput && (
              <div style={s.termOutput}>{cmdOutput}</div>
            )}
          </div>
        )}

        {/* ── ENVIRONMENT TAB ────────────────────────────────────────── */}
        {activeTab === 'env' && (
          <div style={s.card}>
            <h2 style={s.h2}>🔑 Server Environment Variables</h2>
            <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
              From <code>GET /admin/panel</code> → <code>serverInfo.env</code>.
              Includes JWT_SECRET, DB_PASSWORD, AWS keys, etc.
            </p>
            {serverInfo ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                  {[
                    ['JWT_SECRET', serverInfo.env?.JWT_SECRET],
                    ['DB_PASSWORD', serverInfo.env?.DB_PASSWORD],
                    ['ADMIN_API_KEY', serverInfo.env?.ADMIN_API_KEY],
                    ['AWS_ACCESS_KEY_ID', serverInfo.env?.AWS_ACCESS_KEY_ID],
                    ['AWS_SECRET_ACCESS_KEY', serverInfo.env?.AWS_SECRET_ACCESS_KEY],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 12 }}>
                      <div style={{ color: '#64748b', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>{k}</div>
                      <div style={{ color: '#ef4444', fontFamily: 'monospace', fontSize: 13, wordBreak: 'break-all' }}>{v || '(not set)'}</div>
                    </div>
                  ))}
                </div>
                <p style={{ color: '#64748b', fontSize: 12, marginBottom: 8 }}>Full process.env dump:</p>
                <pre style={s.envBox}>{JSON.stringify(serverInfo.env, null, 2)}</pre>
              </>
            ) : (
              <p style={{ color: '#475569' }}>Loading environment data... (requires /admin/panel access)</p>
            )}
          </div>
        )}

        {/* ── PRIVILEGE ESCALATION TAB ───────────────────────────────── */}
        {activeTab === 'roles' && (
          <div style={s.card}>
            <h2 style={s.h2}>⚡ Privilege Escalation</h2>
            <div style={s.warn}>
              ⚠ <strong>CWE-269 — Improper Privilege Management:</strong>
              Any authenticated user can call <code>POST /admin/users/role</code> to change any user's role.
              The endpoint has no admin role check.
            </div>
            <p style={{ color: '#94a3b8', fontSize: 14, marginBottom: 20 }}>
              Select a user from the Users tab, choose a new role, and escalate:
            </p>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <input
                style={{ ...s.termInput, flex: '0 0 100px', color: '#e2e8f0' }}
                placeholder="User ID"
                value={selectedUid}
                onChange={e => setSelectedUid(e.target.value)}
              />
              <select
                value={newRole}
                onChange={e => setNewRole(e.target.value)}
                style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 14, outline: 'none' }}
              >
                <option value="admin">admin</option>
                <option value="superadmin">superadmin</option>
                <option value="user">user</option>
              </select>
              <button style={s.btnWarn} onClick={changeUserRole}>
                🚀 Escalate to {newRole}
              </button>
            </div>
            <p style={{ color: '#64748b', fontSize: 13 }}>
              Or use curl: <br />
              <code style={{ color: '#22c55e', fontSize: 12 }}>
                curl -X POST http://localhost:5000/admin/users/role \<br />
                &nbsp;&nbsp;-H "Authorization: Bearer YOUR_TOKEN" \<br />
                &nbsp;&nbsp;-H "Content-Type: application/json" \<br />
                &nbsp;&nbsp;-d '{JSON.stringify({ userId: 2, role: 'admin' })}'
              </code>
            </p>
          </div>
        )}

      </div>
    </div>
  );
}

export default AdminPanel;
