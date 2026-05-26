import { useState, useMemo, useCallback , useEffect } from 'react';
import {  useNavigate } from 'react-router-dom';
import { useHashedEngagementId } from '../../../utils/useHashedParams';
import { encodeId } from '../../../utils/hashId';
import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';
import { useAuth } from '../../../auth/hooks/useAuth';
import { predictionsApi, PredictionRequest } from '../../../api/services/predictions';
import type { Finding } from '../../../types/finding';
import FindingRowActions from '../components/FindingRowActions';
import { apiClient } from '../../../api/client';
import './FindingsPage.css';

type SortKey = 'risk' | 'cvss' | 'age' | 'severity' | 'id';
type FilterLevel = 'all' | 'critical' | 'high' | 'medium' | 'low';

const PAGE_SIZE = 20;

const SEV_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
const RISK_COLOR: Record<string, string> = {
  critical: 'var(--severity-critical)', high: 'var(--severity-high)', medium: 'var(--severity-medium)', low: 'var(--severity-low)',
};
const SEV_CLASS: Record<string, string> = {
  critical: 'fp-sev-crit', high: 'fp-sev-high', medium: 'fp-sev-med', low: 'fp-sev-low',
};

// ── Export helpers ──────────────────────────────────────────────────────────

function exportToExcel(findings: any[], engagementId: number | null) {
  // Build CSV content (opens perfectly in Excel)
  const headers = ['ID', 'Titre', 'Sévérité', 'CVSS', 'Âge (jours)', 'Tags', 'Score IA', 'Confiance', 'Fichier', 'Ligne', 'CVE'];
  const rows = findings.map(f => [
    f.id,
    `"${(f.title ?? '').replace(/"/g, '""')}"`,
    f.severity,
    f.cvss_score ?? '',
    f.age_days ?? '',
    `"${(f.tags ?? []).join(', ')}"`,
    f.ai_risk_class !== undefined ? `${f.ai_risk_class}/4` : '',
    f.ai_confidence !== undefined ? `${(f.ai_confidence * 100).toFixed(0)}%` : '',
    `"${(f.file_path ?? '').replace(/"/g, '""')}"`,
    f.line ?? '',
    f.has_cve ? 'Oui' : 'Non',
  ]);

  const csvContent = [headers.join(';'), ...rows.map(r => r.join(';'))].join('\n');
  const BOM = '\uFEFF'; // UTF-8 BOM for Excel
  const blob = new Blob([BOM + csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `findings-engagement-${engagementId ?? 'export'}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

function exportToPDF(findings: any[], engagementId: number | null) {
  const sevColor: Record<string, string> = {
    critical: '#ff4757', high: '#ff6b35', medium: '#ffd32a', low: '#2ed573', info: '#a0aec0',
  };

  const rows = findings.map(f => `
    <tr>
      <td>#${f.id}</td>
      <td class="title-cell">${escHtml(f.title)}</td>
      <td><span class="sev-badge sev-${f.severity}" style="color:${sevColor[f.severity] ?? '#888'}">${f.severity.toUpperCase()}</span></td>
      <td style="color:${RISK_COLOR[f.ai_risk_level ?? 'low'] ?? '#888'}">${f.cvss_score ?? '—'}</td>
      <td>${f.age_days ? `${f.age_days}j` : '—'}</td>
      <td>${(f.tags ?? []).slice(0, 3).join(', ') || '—'}</td>
      <td style="color:${RISK_COLOR[f.ai_risk_level ?? 'low'] ?? '#888'};font-weight:700">${f.ai_risk_class !== undefined ? `${f.ai_risk_class}/4` : '—'}</td>
      <td>${f.ai_confidence !== undefined ? `${(f.ai_confidence * 100).toFixed(0)}%` : '—'}</td>
      <td>${f.has_cve ? '<span style="color:#ff4757;font-weight:600">CVE</span>' : '—'}</td>
    </tr>
  `).join('');

  const html = `<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8"/>
<title>Findings – Engagement #${engagementId}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Courier New', monospace; background: #0d0f14; color: #e8edf2; padding: 32px; font-size: 11px; }
  .report-header { margin-bottom: 28px; border-bottom: 1px solid rgba(0,212,255,0.3); padding-bottom: 20px; }
  .report-title { font-size: 22px; font-weight: 800; letter-spacing: -0.02em; color: #fff; margin-bottom: 6px; }
  .report-title span { color: #00d4ff; }
  .report-meta { color: rgba(255,255,255,0.4); font-size: 10px; letter-spacing: 0.12em; }
  .report-stats { display: flex; gap: 24px; margin-bottom: 24px; }
  .stat { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 18px; }
  .stat-val { font-size: 24px; font-weight: 700; }
  .stat-lbl { font-size: 9px; letter-spacing: 0.14em; color: rgba(255,255,255,0.3); text-transform: uppercase; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; }
  thead tr { background: rgba(0,212,255,0.05); border-bottom: 1px solid rgba(0,212,255,0.2); }
  th { font-size: 9px; letter-spacing: 0.14em; text-transform: uppercase; color: rgba(255,255,255,0.4); padding: 10px 12px; text-align: left; white-space: nowrap; }
  td { padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); vertical-align: middle; color: rgba(255,255,255,0.8); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(255,255,255,0.02); }
  .title-cell { max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 600; color: #fff; }
  .sev-badge { font-size: 9px; font-weight: 700; padding: 3px 8px; border-radius: 4px; background: rgba(255,255,255,0.06); }
  .report-footer { margin-top: 24px; text-align: right; font-size: 9px; color: rgba(255,255,255,0.2); letter-spacing: 0.1em; }
  @media print {
    body { background: #fff; color: #111; }
    table, th, td { color: #111 !important; }
    .stat { background: #f5f5f5; border-color: #ddd; }
  }
</style>
</head>
<body>
  <div class="report-header">
    <div class="report-title">THREAT FINDINGS · <span>Engagement #${engagementId}</span></div>
    <div class="report-meta">GENERATED ${new Date().toLocaleDateString('fr-FR', { day: '2-digit', month: 'long', year: 'numeric', hour: '2-digit', minute: '2-digit' }).toUpperCase()} · INVISITHREAT DEVSECOPS PLATFORM</div>
  </div>
  <div class="report-stats">
    <div class="stat"><div class="stat-val" style="color:#00d4ff">${findings.length}</div><div class="stat-lbl">Total</div></div>
    <div class="stat"><div class="stat-val" style="color:#ff4757">${findings.filter(f => f.severity === 'critical').length}</div><div class="stat-lbl">Critical</div></div>
    <div class="stat"><div class="stat-val" style="color:#ff6b35">${findings.filter(f => f.severity === 'high').length}</div><div class="stat-lbl">High</div></div>
    <div class="stat"><div class="stat-val" style="color:#ffd32a">${findings.filter(f => f.has_cve).length}</div><div class="stat-lbl">CVE</div></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>#ID</th><th>Titre</th><th>Sévérité</th><th>CVSS</th><th>Âge</th><th>Tags</th><th>Score IA</th><th>Confiance</th><th>CVE</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
  <div class="report-footer">INVISITHREAT · RAPPORT CONFIDENTIEL · ${findings.length} FINDING${findings.length !== 1 ? 'S' : ''}</div>
  <script>window.onload = () => { window.print(); };<\/script>
</body>
</html>`;

  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const win = window.open(url, '_blank');
  if (!win) alert('Veuillez autoriser les pop-ups pour générer le PDF.');
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function escHtml(str: string) {
  return (str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Main component ──────────────────────────────────────────────────────────

export default function FindingsPage() {
  const { engagementId: parsedId } = useHashedEngagementId();

  const navigate = useNavigate();

  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<FilterLevel>('all');
  const [sortKey, setSortKey] = useState<SortKey>('risk');
  const [page, setPage] = useState(1);
  const [exporting, setExporting] = useState<'pdf' | 'excel' | null>(null);
  const { user } = useAuth();
  const isManager = user?.role === 'manager' || user?.role === 'admin'; 
  const [users, setUsers] = useState<{id: number; username: string; role: string}[]>([]);
  const [metadataMap, setMetadataMap] = useState<Record<number, any>>({});

  useEffect(() => {
    apiClient.get("/findings/users/assignable")
      .then(res => setUsers(res.data ?? []))
      .catch(() => setUsers([]));
  }, []);


  const { data: rawFindings = [], isLoading: loadingFindings } = useQuery({
    queryKey: ['findings', parsedId],
    queryFn: () => findingsApi.getByEngagement(parsedId!).then(res => res.data),
    enabled: !!parsedId,
  });

  const findingsForBatch: PredictionRequest[] = rawFindings.map((f: Finding) => ({
    severity: f.severity,
    cvss_score: f.cvss_score || 0,
    title: f.title,
    tags: f.tags,
    days_open: f.age_days || 0,
    epss_score: 0,
    has_cve: f.has_cve || 0,
    has_cwe: 0,
    finding_id: f.id,
    engagement_id: f.engagement_id ?? parsedId ?? undefined,
  }));

  const { data: batchResults, isLoading: loadingScores } = useQuery({
    queryKey: ['scores', rawFindings],
    queryFn: () => predictionsApi.predictBatch(findingsForBatch).then(res => res.data),
    enabled: rawFindings.length > 0,
  });

  const findings = useMemo(() => rawFindings.map((f: Finding, idx: number) => {
    const score = batchResults?.results[idx];
    return {
      ...f,
      ai_risk_class: score?.risk_class,
      ai_risk_level: score?.risk_level,
      ai_confidence: score?.confidence,
      context_score: score?.context_score,
    };
  }), [rawFindings, batchResults]);

  const stats = useMemo(() => ({
    total: findings.length,
    critical: findings.filter(f => f.severity === 'critical').length,
    high: findings.filter(f => f.severity === 'high').length,
    withCVE: findings.filter(f => f.has_cve).length,
    avgScore: findings.length
      ? (findings.reduce((a, f) => a + (f.ai_risk_class || 0), 0) / findings.length).toFixed(1)
      : '—',
  }), [findings]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return findings.filter(f => {
      const matchFilter = filter === 'all' || f.severity === filter;
      const matchSearch = !q ||
        f.title.toLowerCase().includes(q) ||
        (f.file_path || '').toLowerCase().includes(q) ||
        (f.tags || []).some((t: string) => t.includes(q)) ||
        String(f.id).includes(q);
      return matchFilter && matchSearch;
    });
  }, [findings, filter, search]);

  const sorted = useMemo(() => [...filtered].sort((a, b) => {
    const aPinned = metadataMap[a.id]?.is_pinned ? 1 : 0;
    const bPinned = metadataMap[b.id]?.is_pinned ? 1 : 0;
    if (bPinned !== aPinned) return bPinned - aPinned;

    if (sortKey === 'risk')     return (b.ai_risk_class || 0) - (a.ai_risk_class || 0) || (b.cvss_score || 0) - (a.cvss_score || 0);
    if (sortKey === 'cvss')     return (b.cvss_score || 0) - (a.cvss_score || 0);
    if (sortKey === 'age')      return (b.age_days || 0) - (a.age_days || 0);
    if (sortKey === 'severity') return (SEV_ORDER[a.severity] ?? 9) - (SEV_ORDER[b.severity] ?? 9);
    if (sortKey === 'id')       return a.id - b.id;
    return 0;
  }), [filtered, sortKey, metadataMap]);

  // ── Pagination ──
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const paginated = sorted.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  // APRÈS — stable
  const paginatedIds = useMemo(
    () => paginated.map(f => f.id),
    // Sérialise les IDs en string pour comparer par valeur, pas par référence
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [paginated.map(f => f.id).join(',')]
  );

  const refreshMetadata = useCallback(() => {
    if (paginatedIds.length === 0) return;
    apiClient.post("/findings/metadata/batch", paginatedIds)
      .then(res => setMetadataMap(prev => ({ ...prev, ...res.data })))
      .catch(() => {});
  }, [paginatedIds]);

  useEffect(() => {
    refreshMetadata();
  }, [paginatedIds]);

  // Reset to page 1 when filter/search/sort changes
  const handleFilter = useCallback((f: FilterLevel) => { setFilter(f); setPage(1); }, []);
  const handleSearch = useCallback((v: string) => { setSearch(v); setPage(1); }, []);
  const handleSort = useCallback((k: SortKey) => { setSortKey(k); setPage(1); }, []);

  // ── Export handlers ──
  const handleExportExcel = useCallback(async () => {
    setExporting('excel');
    try { exportToExcel(sorted, parsedId); }
    finally { setTimeout(() => setExporting(null), 800); }
  }, [sorted, parsedId]);

  const handleExportPDF = useCallback(async () => {
    setExporting('pdf');
    try { exportToPDF(sorted, parsedId); }
    finally { setTimeout(() => setExporting(null), 800); }
  }, [sorted, parsedId]);

  if (!parsedId) return <LoadingState message="Aucun engagement sélectionné" />;
  if (loadingFindings) return <LoadingState message="Chargement des findings..." progress />;
  if (rawFindings.length === 0) return <LoadingState message="Aucun finding trouvé" />;
  if (loadingScores) return <LoadingState message="Calcul des scores IA..." progress />;

  return (
    <div className="findings-page home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />

      <div className="fp-page">

        {/* ── Page Header ── */}
        <header className="fp-header fu">
          <div className="fp-page-label">
            <span className="fp-label-dot" />
            THREAT FINDINGS · DEVSECOPS PLATFORM
          </div>
          <h1 className="fp-page-title">
            Analyse IA{' '}
            <span className="fp-title-shimmer">Engagement #{parsedId}</span>
          </h1>
          <p className="fp-page-subtitle">
            Priorisation intelligente des vulnérabilités avec scoring IA et contexte sécurité.
          </p>
        </header>

        {/* ── Stats Row ── */}
        <div className="fp-stats-row fu1">
          <StatCard value={stats.total}    label="Total Findings" color="var(--accent)" pct={100} />
          <StatCard value={stats.critical} label="Critical"       color="var(--severity-critical)" pct={stats.total ? (stats.critical / stats.total) * 100 : 0} />
          <StatCard value={stats.high}     label="High"           color="var(--severity-high)" pct={stats.total ? (stats.high / stats.total) * 100 : 0} />
          <StatCard value={stats.withCVE}  label="CVE Présents"   color="var(--severity-medium)" pct={stats.total ? (stats.withCVE / stats.total) * 100 : 0} />
          <StatCard value={stats.avgScore} label="Score IA Moy."  color="var(--severity-info)" pct={parseFloat(stats.avgScore as string) / 4 * 100} />
        </div>

        {/* ── Toolbar ── */}
        <div className="fp-toolbar fu2">
          <div className="fp-search-wrap">
            <svg className="fp-search-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="6.5" cy="6.5" r="4" /><path d="M10.5 10.5L14 14" strokeLinecap="round" />
            </svg>
            <input
              className="fp-search-input"
              type="text"
              placeholder="Rechercher un finding, titre, fichier, tag..."
              value={search}
              onChange={e => handleSearch(e.target.value)}
            />
          </div>

          <div className="fp-filter-pills">
            {(['all', 'critical', 'high', 'medium', 'low'] as FilterLevel[]).map(level => (
              <button
                key={level}
                className={`fp-pill fp-pill-${level}${filter === level ? ' active' : ''}`}
                onClick={() => handleFilter(level)}
              >
                {level === 'all' ? 'Tous' : level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>

          <select
            className="fp-sort-select"
            value={sortKey}
            onChange={e => handleSort(e.target.value as SortKey)}
          >
            <option value="risk">Trier : Score IA</option>
            <option value="cvss">Trier : CVSS</option>
            <option value="age">Trier : Âge</option>
            <option value="severity">Trier : Sévérité</option>
            <option value="id">Trier : ID</option>
          </select>

          {/* ── Export buttons ── */}
          <div className="fp-export-group">
            <button
              className={`fp-export-btn fp-export-excel${exporting === 'excel' ? ' fp-export-loading' : ''}`}
              onClick={handleExportExcel}
              disabled={exporting !== null || sorted.length === 0}
              title="Exporter en Excel / CSV"
            >
              {exporting === 'excel' ? (
                <span className="fp-export-spinner" />
              ) : (
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13">
                  <rect x="2" y="2" width="12" height="12" rx="2" />
                  <path d="M5 6l3 4 3-4" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M8 10V5" strokeLinecap="round" />
                </svg>
              )}
              Excel
            </button>
            <button
              className={`fp-export-btn fp-export-pdf${exporting === 'pdf' ? ' fp-export-loading' : ''}`}
              onClick={handleExportPDF}
              disabled={exporting !== null || sorted.length === 0}
              title="Exporter en PDF"
            >
              {exporting === 'pdf' ? (
                <span className="fp-export-spinner" />
              ) : (
                <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" width="13" height="13">
                  <path d="M4 2h6l4 4v9a1 1 0 01-1 1H4a1 1 0 01-1-1V3a1 1 0 011-1z" />
                  <path d="M10 2v4h4" strokeLinecap="round" />
                  <path d="M6 9h4M6 12h2" strokeLinecap="round" />
                </svg>
              )}
              PDF
            </button>
          </div>
        </div>

        {/* ── Table ── */}
        <div className="fp-table-wrap fu3">
          <div className="fp-table-info-bar">
            <div className="fp-table-count">
              Affichage{' '}
              <span>{sorted.length === 0 ? 0 : (safePage - 1) * PAGE_SIZE + 1}–{Math.min(safePage * PAGE_SIZE, sorted.length)}</span>
              {' '}sur <span>{sorted.length}</span> finding{sorted.length !== 1 ? 's' : ''}
              {filter !== 'all' && <span className="fp-filter-active"> · filtre: {filter}</span>}
            </div>
            <div className="fp-table-eng">Engagement #{parsedId} · Scoring IA actif</div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="fp-tbl">
              <thead>
                <tr>
                  <SortTh label="#ID"       sortKey="id"       active={sortKey} onSort={handleSort} />
                  <th className="fp-th">Titre</th>
                  <SortTh label="Sévérité"  sortKey="severity" active={sortKey} onSort={handleSort} />
                  <SortTh label="CVSS"      sortKey="cvss"     active={sortKey} onSort={handleSort} />
                  <SortTh label="Âge"       sortKey="age"      active={sortKey} onSort={handleSort} />
                  <th className="fp-th">Tags</th>
                  <SortTh label="Score IA"  sortKey="risk"     active={sortKey} onSort={handleSort} />
                  <th className="fp-th">Confiance</th>
                  <th className="fp-th">Fichier</th>
                  <th className="fp-th">CVE</th>
                  <th className="fp-th" />
                </tr>
              </thead>
              <tbody>
                {paginated.length === 0 ? (
                  <tr><td colSpan={11} className="fp-no-results">AUCUN FINDING TROUVÉ</td></tr>
                ) : paginated.map(f => (
                  <tr
                    key={f.id}
                    className="fp-row"
                    data-severity={f.severity}
                    onClick={() => navigate(`/findings/${encodeId(f.id)}`)}

                  >
                    <td><span className="fp-mono-id">#{f.id}</span></td>
                    <td><span className="fp-cell-title" title={f.title}>{f.title}</span></td>
                    <td>
                      <span className={`fp-sev-badge ${SEV_CLASS[f.severity] ?? 'fp-sev-info'}`}>
                        {f.severity.toUpperCase()}
                      </span>
                    </td>
                    <td>
                      <span className="fp-cvss-val" style={{ color: RISK_COLOR[f.ai_risk_level ?? 'low'] ?? '#95a5a6' }}>
                        {f.cvss_score ?? '—'}
                      </span>
                    </td>
                    <td><span className="fp-age-val">{f.age_days ? `${f.age_days}j` : '—'}</span></td>
                    <td>
                      <div className="fp-tags-wrap">
                        {(f.tags ?? []).slice(0, 2).map((tag: string) => (
                          <span key={tag} className="fp-tpill">{tag}</span>
                        ))}
                        {(f.tags?.length ?? 0) > 2 && (
                          <span className="fp-tpill fp-tpill-more">+{f.tags.length - 2}</span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className="fp-ai-score" style={{ color: RISK_COLOR[f.ai_risk_level ?? 'low'] ?? '#95a5a6' }}>
                        {f.ai_risk_class !== undefined ? `${f.ai_risk_class}/4` : 'N/A'}
                      </span>
                    </td>
                    <td>
                      <span className="fp-ai-conf">
                        {f.ai_confidence !== undefined ? `${(f.ai_confidence * 100).toFixed(0)}%` : '—'}
                      </span>
                    </td>
                    <td>
                      {f.file_path ? (
                        <span className="fp-file-cell" title={f.file_path}>
                          {f.file_path.split(/[/\\]/).pop()}
                          {f.line && <span className="fp-file-line">:{f.line}</span>}
                        </span>
                      ) : <span className="fp-ai-conf">—</span>}
                    </td>
                    <td>
                      {f.has_cve
                        ? <span className="fp-cve-yes">CVE</span>
                        : <span className="fp-cve-no">—</span>}
                    </td>
                    <td onClick={e => e.stopPropagation()}>
                      <FindingRowActions
                        findingId={f.id}
                        findingTitle={f.title}
                        severity={f.severity}
                        aiRiskScore={f.ai_risk_class}
                        users={users}
                        metadata={metadataMap[f.id]}
                        onRefresh={refreshMetadata}
                        canManage={isManager}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* ── Pagination ── */}
          {totalPages > 1 && (
            <div className="fp-pagination">
              <button
                className="fp-page-btn"
                onClick={() => setPage(1)}
                disabled={safePage === 1}
                title="Première page"
              >
                «
              </button>
              <button
                className="fp-page-btn"
                onClick={() => setPage(p => Math.max(1, p - 1))}
                disabled={safePage === 1}
              >
                ‹
              </button>

              {getPaginationRange(safePage, totalPages).map((item, i) =>
                item === '...' ? (
                  <span key={`ellipsis-${i}`} className="fp-page-ellipsis">…</span>
                ) : (
                  <button
                    key={item}
                    className={`fp-page-btn fp-page-num${safePage === item ? ' active' : ''}`}
                    onClick={() => setPage(item as number)}
                  >
                    {item}
                  </button>
                )
              )}

              <button
                className="fp-page-btn"
                onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                disabled={safePage === totalPages}
              >
                ›
              </button>
              <button
                className="fp-page-btn"
                onClick={() => setPage(totalPages)}
                disabled={safePage === totalPages}
                title="Dernière page"
              >
                »
              </button>

              <span className="fp-page-info">
                Page <strong>{safePage}</strong> / {totalPages}
              </span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

// ── Pagination range helper ─────────────────────────────────────────────────

function getPaginationRange(current: number, total: number): (number | '...')[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const pages: (number | '...')[] = [1];
  if (current > 3) pages.push('...');
  for (let i = Math.max(2, current - 1); i <= Math.min(total - 1, current + 1); i++) pages.push(i);
  if (current < total - 2) pages.push('...');
  pages.push(total);
  return pages;
}

// ── Sub-components ──────────────────────────────────────────────────────────

function StatCard({ value, label, color, pct }: { value: string | number; label: string; color: string; pct: number }) {
  return (
    <div className="fp-stat-card" style={{ '--fp-c': color } as React.CSSProperties}>
      <div className="fp-stat-val" style={{ color }}>{value}</div>
      <div className="fp-stat-lbl">{label}</div>
      <div className="fp-stat-bar">
        <div className="fp-stat-bar-fill" style={{ width: `${Math.min(100, pct || 0)}%`, background: color }} />
      </div>
    </div>
  );
}

function SortTh({ label, sortKey, active, onSort }: {
  label: string; sortKey: SortKey; active: SortKey; onSort: (k: SortKey) => void;
}) {
  const isActive = sortKey === active;
  return (
    <th
      className={`fp-th fp-th-sort${isActive ? ' fp-th-active' : ''}`}
      onClick={() => onSort(sortKey)}
    >
      {label} <span className="fp-sort-icon">{isActive ? '↓' : '↕'}</span>
    </th>
  );
}

function LoadingState({ message, progress }: { message: string; progress?: boolean }) {
  return (
    <div className="home-root" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
      <div className="bg-grid" /><div className="bg-radials" /><div className="scan-line" />
      <div style={{ textAlign: 'center', position: 'relative', zIndex: 10 }}>
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.14em', color: 'var(--dimmed)', marginBottom: 16 }}>
          {message.toUpperCase()}
        </div>
        {progress && (
          <div style={{ width: 220, height: 2, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'linear-gradient(90deg, var(--accent), var(--accent-muted))', animation: 'progressBar 2s ease infinite', borderRadius: 2 }} />
          </div>
        )}
      </div>
    </div>
  );
}