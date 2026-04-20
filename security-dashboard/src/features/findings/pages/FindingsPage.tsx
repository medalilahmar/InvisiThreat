import { useState, useMemo } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';
import { predictionsApi, PredictionRequest } from '../../../api/services/predictions';
import type { Finding } from '../../../types/finding';
import './FindingsPage.css';

type SortKey = 'risk' | 'cvss' | 'age' | 'severity' | 'id';
type FilterLevel = 'all' | 'critical' | 'high' | 'medium' | 'low';

const SEV_ORDER: Record<string, number> = { critical: 0, high: 1, medium: 2, low: 3 };
const RISK_COLOR: Record<string, string> = {
  critical: '#ff4757', high: '#ff6b35', medium: '#ffd32a', low: '#2ed573',
};
const SEV_CLASS: Record<string, string> = {
  critical: 'fp-sev-crit', high: 'fp-sev-high', medium: 'fp-sev-med', low: 'fp-sev-low',
};

export default function FindingsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<FilterLevel>('all');
  const [sortKey, setSortKey] = useState<SortKey>('risk');

  const engagementId = searchParams.get('engagementId');
  const parsedId = engagementId ? parseInt(engagementId, 10) : null;

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
    if (sortKey === 'risk')     return (b.ai_risk_class || 0) - (a.ai_risk_class || 0) || (b.cvss_score || 0) - (a.cvss_score || 0);
    if (sortKey === 'cvss')     return (b.cvss_score || 0) - (a.cvss_score || 0);
    if (sortKey === 'age')      return (b.age_days || 0) - (a.age_days || 0);
    if (sortKey === 'severity') return (SEV_ORDER[a.severity] ?? 9) - (SEV_ORDER[b.severity] ?? 9);
    if (sortKey === 'id')       return a.id - b.id;
    return 0;
  }), [filtered, sortKey]);

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
          <StatCard value={stats.total}    label="Total Findings" color="#00d4ff" pct={100} />
          <StatCard value={stats.critical} label="Critical"       color="#ff4757" pct={stats.total ? (stats.critical / stats.total) * 100 : 0} />
          <StatCard value={stats.high}     label="High"           color="#ff6b35" pct={stats.total ? (stats.high / stats.total) * 100 : 0} />
          <StatCard value={stats.withCVE}  label="CVE Présents"   color="#ffd32a" pct={stats.total ? (stats.withCVE / stats.total) * 100 : 0} />
          <StatCard value={stats.avgScore} label="Score IA Moy."  color="#a29bfe" pct={parseFloat(stats.avgScore as string) / 4 * 100} />
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
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <div className="fp-filter-pills">
            {(['all', 'critical', 'high', 'medium', 'low'] as FilterLevel[]).map(level => (
              <button
                key={level}
                className={`fp-pill fp-pill-${level}${filter === level ? ' active' : ''}`}
                onClick={() => setFilter(level)}
              >
                {level === 'all' ? 'Tous' : level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>

          <select
            className="fp-sort-select"
            value={sortKey}
            onChange={e => setSortKey(e.target.value as SortKey)}
          >
            <option value="risk">Trier : Score IA</option>
            <option value="cvss">Trier : CVSS</option>
            <option value="age">Trier : Âge</option>
            <option value="severity">Trier : Sévérité</option>
            <option value="id">Trier : ID</option>
          </select>
        </div>

        {/* ── Table ── */}
        <div className="fp-table-wrap fu3">
          <div className="fp-table-info-bar">
            <div className="fp-table-count">
              Affichage <span>{sorted.length}</span> finding{sorted.length !== 1 ? 's' : ''}
              {filter !== 'all' && <span className="fp-filter-active"> · filtre: {filter}</span>}
            </div>
            <div className="fp-table-eng">Engagement #{parsedId} · Scoring IA actif</div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="fp-tbl">
              <thead>
                <tr>
                  <SortTh label="#ID"       sortKey="id"       active={sortKey} onSort={setSortKey} />
                  <th className="fp-th">Titre</th>
                  <SortTh label="Sévérité"  sortKey="severity" active={sortKey} onSort={setSortKey} />
                  <SortTh label="CVSS"      sortKey="cvss"     active={sortKey} onSort={setSortKey} />
                  <SortTh label="Âge"       sortKey="age"      active={sortKey} onSort={setSortKey} />
                  <th className="fp-th">Tags</th>
                  <SortTh label="Score IA"  sortKey="risk"     active={sortKey} onSort={setSortKey} />
                  <th className="fp-th">Confiance</th>
                  <th className="fp-th">Fichier</th>
                  <th className="fp-th">CVE</th>
                  <th className="fp-th" />
                </tr>
              </thead>
              <tbody>
                {sorted.length === 0 ? (
                  <tr><td colSpan={11} className="fp-no-results">AUCUN FINDING TROUVÉ</td></tr>
                ) : sorted.map(f => (
                  <tr
                    key={f.id}
                    className="fp-row"
                    data-severity={f.severity}
                    onClick={() => navigate(`/findings/${f.id}`)}
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
                    <td><span className="fp-row-arrow">→</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}

/* ── Sub-components ── */

function StatCard({ value, label, color, pct }: { value: string | number; label: string; color: string; pct: number }) {
  return (
    <div className="fp-stat-card" style={{ '--fp-c': color } as React.CSSProperties}>
      <div className="fp-stat-val" style={{ color }}>{value}</div>
      <div className="fp-stat-lbl">{label}</div>
      <div className="fp-stat-bar">
        <div className="fp-stat-bar-fill" style={{ width: `${Math.min(100, pct)}%`, background: color }} />
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
        <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.14em', color: 'rgba(255,255,255,0.3)', marginBottom: 16 }}>
          {message.toUpperCase()}
        </div>
        {progress && (
          <div style={{ width: 220, height: 2, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
            <div style={{ height: '100%', background: 'linear-gradient(90deg,#00d4ff,#008fbb)', animation: 'progressBar 2s ease infinite', borderRadius: 2 }} />
          </div>
        )}
      </div>
    </div>
  );
}