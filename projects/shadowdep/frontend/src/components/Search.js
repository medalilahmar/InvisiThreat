import React, { useState, useEffect } from 'react';
import api from '../services/api';

/**
 * Search Component
 *
 * DAST Findings:
 *   - DOM XSS: search term from URL injected via innerHTML (CWE-79)
 *   - The backend /projects/search endpoint has SQL Injection (CWE-89)
 *   - No rate limiting on search — enumeration possible
 */
function Search() {
  const [query,   setQuery]   = useState('');
  const [results, setResults] = useState([]);
  const [searched,setSearched]= useState(false);

  useEffect(() => {
    // ════════════════════════════════════════════════════════════
    // DOM XSS: search query read from URL ?q= param and injected
    // into the page unescaped via innerHTML.
    // Test: /search?q=<img src=x onerror=alert(document.cookie)>
    // ════════════════════════════════════════════════════════════
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q') || '';

    if (q) {
      setQuery(q);

      // DOM XSS — innerHTML with unsanitized URL parameter
      const displayEl = document.getElementById('search-result-header');
      if (displayEl) {
        displayEl.innerHTML = 'Results for: <strong style="color:#818cf8">' + q + '</strong>'; // DOM XSS
      }

      // Also inject q into a data attribute that gets reflected
      document.title = 'Search: ' + q + ' — ShadowDep';

      performSearch(q);
    }
  }, []);

  const performSearch = async (q) => {
    try {
      // FIXME: This backend endpoint has SQL injection vulnerability
      // Test payload: ' UNION SELECT id,username,password,role,NULL,NULL FROM users--
      const res = await api.get(`/api/projects?search=${encodeURIComponent(q)}`);
      setResults(res.data);
      setSearched(true);
    } catch (err) {
      console.error(err);
      // Also redirect to EJS search page which has reflected XSS
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    // Redirect to backend EJS page — which has reflected XSS in the output
    // The EJS page uses <%- query %> (unescaped) to display results
    window.location.href = `/projects/search?q=${query}`;
  };

  const s = {
    page:      { minHeight: '100vh', background: '#0f172a', fontFamily: "'Segoe UI',Arial,sans-serif" },
    navbar:    { background: 'rgba(30,41,59,0.95)', borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '0 32px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between' },
    brand:     { fontSize: 20, fontWeight: 700, color: '#818cf8' },
    container: { maxWidth: 860, margin: '40px auto', padding: '0 24px' },
    card:      { background: 'rgba(30,41,59,0.8)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 16, padding: 36, marginBottom: 24 },
    h2:        { color: '#e2e8f0', fontSize: 24, fontWeight: 700, marginBottom: 24 },
    input:     { flex: 1, padding: '12px 18px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, color: '#e2e8f0', fontSize: 15, outline: 'none' },
    btnPrimary:{ padding: '12px 24px', background: 'linear-gradient(135deg,#6366f1,#8b5cf6)', color: 'white', border: 'none', borderRadius: 10, fontSize: 15, cursor: 'pointer', fontWeight: 600 },
    resultCard:{ background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 12, padding: 20, marginBottom: 12 },
    warn:      { background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', borderRadius: 8, padding: '12px 16px', color: '#fca5a5', fontSize: 13, marginBottom: 20 },
  };

  return (
    <div style={s.page}>
      <nav style={s.navbar}>
        <span style={s.brand}>🌑 ShadowDep</span>
        <a href="/dashboard" style={{ color: '#94a3b8', textDecoration: 'none', fontSize: 14 }}>← Dashboard</a>
      </nav>

      <div style={s.container}>
        <div style={s.card}>
          <h2 style={s.h2}>🔍 Search Projects</h2>

          <div style={s.warn}>
            ⚠ <strong>Vulnerability Targets:</strong><br />
            <strong>SQL Injection</strong> (backend): <code>' OR '1'='1</code> &nbsp;|&nbsp;
            <code>' UNION SELECT id,username,password,role,NULL,NULL FROM users--</code><br />
            <strong>Reflected XSS</strong> (backend EJS): <code>&lt;script&gt;alert(document.cookie)&lt;/script&gt;</code><br />
            <strong>DOM XSS</strong> (this page): append <code>?q=&lt;img src=x onerror=alert(1)&gt;</code> to URL
          </div>

          {/* DOM XSS injection point from URL param */}
          <div id="search-result-header" style={{ color: '#94a3b8', fontSize: 16, marginBottom: 20 }}>
            Enter a search term below
          </div>

          <form onSubmit={handleSearch} style={{ display: 'flex', gap: 12, marginBottom: 32 }}>
            <input
              type="text"
              style={s.input}
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search projects... (try SQL injection or XSS payloads)"
            />
            <button type="submit" style={s.btnPrimary}>Search</button>
          </form>

          {searched && (
            <>
              <p style={{ color: '#64748b', fontSize: 13, marginBottom: 16 }}>
                {results.length} results from API (React) — or click Search to use EJS backend (reflected XSS)
              </p>
              {results.length === 0 ? (
                <p style={{ color: '#475569', textAlign: 'center', padding: 32 }}>No matching projects found.</p>
              ) : (
                results.map(project => (
                  <div key={project.id} style={s.resultCard}>
                    <a href={`/project/${project.id}`}
                      style={{ color: '#e2e8f0', fontSize: 16, fontWeight: 600, textDecoration: 'none' }}>
                      #{project.id} — {project.name}
                    </a>
                    {/*
                      DOM XSS: description rendered with dangerouslySetInnerHTML
                      Stored XSS executes here when attacker has a malicious project
                    */}
                    <div
                      style={{ color: '#94a3b8', fontSize: 13, marginTop: 8, lineHeight: 1.6 }}
                      dangerouslySetInnerHTML={{ __html: project.description }}
                    />
                    <div style={{ fontSize: 12, color: '#475569', marginTop: 8 }}>
                      Owner ID: {project.user_id} | Status: {project.status}
                    </div>
                  </div>
                ))
              )}
            </>
          )}
        </div>

        {/* Quick IDOR tester */}
        <div style={s.card}>
          <p style={{ color: '#64748b', fontSize: 13, fontWeight: 600, marginBottom: 12, textTransform: 'uppercase' }}>
            🔗 IDOR — Direct Object Reference Tester
          </p>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
            Project IDs are sequential. Access any project without ownership check:
          </p>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {[1, 2, 3, 4, 5, 6].map(id => (
              <a key={id} href={`/project/${id}`}
                style={{ padding: '8px 16px', background: 'rgba(99,102,241,0.15)', color: '#818cf8', borderRadius: 8, textDecoration: 'none', fontSize: 14 }}>
                Project #{id}
              </a>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Search;
