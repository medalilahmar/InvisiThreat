import { useDashboardData } from '../hooks/useDashboardData';
import { Link } from 'react-router-dom';
import { useState, useMemo, useEffect } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Legend, LineChart, Line, CartesianGrid,
} from 'recharts';
import './DashboardPage.css';

type RiskLevel = 'Critical' | 'High' | 'Medium' | 'Low' | 'Info';

interface RiskEntry { name: string; value: number; color: string; }

const RISK_COLORS: Record<RiskLevel, string> = {
  Critical: 'var(--severity-critical)',
  High:     'var(--severity-high)',
  Medium:   'var(--severity-medium)',
  Low:      'var(--severity-low)',
  Info:     'var(--severity-info)',
};

const RISK_ORDER: RiskLevel[] = ['Critical', 'High', 'Medium', 'Low', 'Info'];

/* ─── Icons ─────────────────────────────────────────────────────────────── */
function IconActivity({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}
function IconAlertTriangle({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
function IconShield({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    </svg>
  );
}
function IconZap({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}
function IconBox({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
    </svg>
  );
}
function IconSearch({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}
function IconX({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}
function IconChevronUp({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="18 15 12 9 6 15" />
    </svg>
  );
}
function IconChevronDown({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}
function IconChevronsUpDown({ size = 11 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="7 15 12 20 17 15" />
      <polyline points="7 9 12 4 17 9" />
    </svg>
  );
}
function IconBarChart2({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  );
}
function IconList({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </svg>
  );
}
function IconTrendingUp({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
      <polyline points="17 6 23 6 23 12" />
    </svg>
  );
}
function IconArrowRight({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}
function IconInfo({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

/* ─── KPI card icons map ─────────────────────────────────────────────────── */
const KPI_ICONS = [
  <IconActivity size={18} />,
  <IconAlertTriangle size={18} />,
  <IconShield size={18} />,
  <IconZap size={18} />,
  <IconBox size={18} />,
];

const KPI_COLORS = [
  'var(--accent)',
  'var(--severity-critical)',
  'var(--severity-high)',
  'var(--severity-critical)',
  'var(--severity-info)',
];

/* ─── Animated Counter ───────────────────────────────────────────────────── */
function AnimatedCounter({ target, duration = 1200 }: { target: number; duration?: number }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = target / (duration / 16);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setCount(target); clearInterval(timer); }
      else setCount(Math.floor(start));
    }, 16);
    return () => clearInterval(timer);
  }, [target, duration]);
  return <>{count.toLocaleString()}</>;
}

/* ─── Score Bar ──────────────────────────────────────────────────────────── */
function ScoreBar({ score, max = 10 }: { score: number; max?: number }) {
  const pct = Math.min((score / max) * 100, 100);
  const color = score >= 9 ? 'var(--severity-critical)'
    : score >= 7 ? 'var(--severity-high)'
    : score >= 4 ? 'var(--severity-medium)'
    : 'var(--severity-low)';
  return (
    <div className="score-pill">
      <span className="score-val" style={{ color }}>{score.toFixed(1)}</span>
      <div className="score-track">
        <div className="score-fill"
          style={{ '--bar-w': `${pct}%`, background: color } as React.CSSProperties} />
      </div>
    </div>
  );
}

/* ─── Risk Distribution Bars ─────────────────────────────────────────────── */
function RiskDistributionBars({
  riskData, total, onFilter, activeFilter,
}: {
  riskData: RiskEntry[];
  total: number;
  onFilter: (name: string | null) => void;
  activeFilter: string | null;
}) {
  return (
    <div className="risk-dist-bars">
      {riskData.map((entry) => {
        const pct = total > 0 ? ((entry.value / total) * 100).toFixed(1) : '0';
        const isActive = activeFilter === entry.name;
        return (
          <div key={entry.name}
            className={`dist-row${isActive ? ' dist-row--active' : ''}`}
            onClick={() => onFilter(isActive ? null : entry.name)}>
            <div className="dist-row-header">
              <span className="dist-row-name" style={{ color: entry.color }}>{entry.name}</span>
              <span className="dist-row-count">
                <strong>{entry.value}</strong>
                <span className="dist-row-pct">{pct}%</span>
              </span>
            </div>
            <div className="dist-bar-track">
              <div className="dist-bar-fill"
                style={{ '--bar-width': `${pct}%`, background: entry.color } as React.CSSProperties} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ─── Priority Bar ───────────────────────────────────────────────────────── */
function PriorityBar({ score, max }: { score: number; max: number }) {
  const pct = max > 0 ? (score / max) * 100 : 0;
  const color = pct > 75 ? 'var(--severity-critical)'
    : pct > 45 ? 'var(--severity-high)'
    : pct > 20 ? 'var(--severity-medium)'
    : 'var(--severity-low)';
  return (
    <div className="priority-bar-wrap">
      <div className="priority-bar-track">
        <div className="priority-bar-fill"
          style={{ '--bar-width': `${pct}%`, background: color } as React.CSSProperties} />
      </div>
      <span className="priority-bar-val" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  );
}

/* ─── Skeleton ───────────────────────────────────────────────────────────── */
function SkeletonCard() {
  return <div className="skeleton-card"><div className="skeleton-shimmer" /></div>;
}
function SkeletonRow() {
  return (
    <tr className="skeleton-row">
      {[...Array(8)].map((_, i) => (
        <td key={i}><div className="skeleton-cell skeleton-shimmer" /></td>
      ))}
    </tr>
  );
}

/* ─── Custom Tooltip ─────────────────────────────────────────────────────── */
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      {label && <p className="chart-tooltip-label">{label}</p>}
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color || 'var(--accent)' }}>
          {p.name}: <strong>{p.value}</strong>
        </p>
      ))}
    </div>
  );
};

/* ─── Tab definitions ────────────────────────────────────────────────────── */
const TABS = [
  { id: 'overview',  label: 'Vue globale',  icon: <IconBarChart2 size={14} /> },
  { id: 'products',  label: 'Produits',     icon: <IconList size={14} /> },
] as const;

type TabId = typeof TABS[number]['id'];

/* ═══════════════════════════════════════════════════════════════════════════
   MAIN
═══════════════════════════════════════════════════════════════════════════ */
export default function DashboardPage() {
  const { productSummaries, stats, isLoading, error } = useDashboardData();

  const [search,      setSearch]      = useState('');
  const [riskFilter,  setRiskFilter]  = useState<string | null>(null);
  const [sortKey,     setSortKey]     = useState<string>('totalFindings');
  const [sortDir,     setSortDir]     = useState<'asc' | 'desc'>('desc');
  const [activeTab,   setActiveTab]   = useState<TabId>('overview');

  const urgentRatio = stats
    ? ((stats.totalCritical + stats.totalHigh) / Math.max(stats.totalFindings, 1)) * 100
    : 0;

  const riskData = useMemo<RiskEntry[]>(() => {
    if (!stats?.riskDistribution) return [];
    return (stats.riskDistribution as RiskEntry[]).map((r) => ({
      ...r,
      color: r.color || RISK_COLORS[r.name as RiskLevel] || 'var(--muted)',
    }));
  }, [stats]);

  const filteredProducts = useMemo(() => {
    if (!productSummaries) return [];
    let list = [...productSummaries];
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((p) => p.name.toLowerCase().includes(q));
    }
    if (riskFilter) {
      const key = riskFilter.toLowerCase() as keyof typeof list[0];
      list = list.filter((p) => (p[key] as number) > 0);
    }
    list.sort((a, b) => {
      const av = a[sortKey as keyof typeof a] as number;
      const bv = b[sortKey as keyof typeof b] as number;
      return sortDir === 'desc' ? bv - av : av - bv;
    });
    return list;
  }, [productSummaries, search, riskFilter, sortKey, sortDir]);

  const withPriority = useMemo(() => filteredProducts.map((p) => ({
    ...p,
    priorityScore:
      (p.critical * 4 + p.high * 3 + p.medium * 2 + p.low) /
      Math.max(p.totalFindings, 1) * 10,
  })), [filteredProducts]);

  const maxPriority = useMemo(
    () => Math.max(...withPriority.map((p) => p.priorityScore), 1),
    [withPriority],
  );

  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    else { setSortKey(key); setSortDir('desc'); }
  };

  const SortIcon = ({ k }: { k: string }) =>
    sortKey === k ? (
      <span className="sort-icon sort-icon--active">
        {sortDir === 'desc' ? <IconChevronDown /> : <IconChevronUp />}
      </span>
    ) : (
      <span className="sort-icon"><IconChevronsUpDown /></span>
    );

  const KPI_CARDS = stats ? [
    { value: stats.totalFindings,                       label: 'Findings totaux',  colorIdx: 0 },
    { value: stats.totalCritical,                       label: 'Critiques',        colorIdx: 1 },
    { value: stats.totalHigh,                           label: 'High',             colorIdx: 2 },
    { value: stats.totalCritical + stats.totalHigh,     label: 'Urgents (C+H)',    colorIdx: 3 },
    { value: stats.totalProducts,                       label: 'Produits',         colorIdx: 4 },
  ] : [];

  /* ── Loading ─────────────────────────────────────────────────────────── */
  if (isLoading) return (
    <div className="dashboard-page">
      <div className="section-header">
        <div className="section-label">MONITORING</div>
        <h2 className="section-title">Vue d'ensemble <span>de la sécurité</span></h2>
        <p className="section-subtitle">Chargement des données en cours…</p>
      </div>
      <div className="stats-grid">
        {[...Array(5)].map((_, i) => <SkeletonCard key={i} />)}
      </div>
      <div className="charts-row">
        <div className="chart-card"><div className="skeleton-card skeleton-card--tall skeleton-shimmer" /></div>
        <div className="chart-card"><div className="skeleton-card skeleton-card--tall skeleton-shimmer" /></div>
      </div>
      <div className="products-summary-table">
        <table className="dashboard-table">
          <thead><tr>{[...Array(8)].map((_, i) => <th key={i}><div className="skeleton-cell skeleton-shimmer" /></th>)}</tr></thead>
          <tbody>{[...Array(5)].map((_, i) => <SkeletonRow key={i} />)}</tbody>
        </table>
      </div>
    </div>
  );

  /* ── Error ───────────────────────────────────────────────────────────── */
  if (error) return (
    <div className="dashboard-page">
      <div className="error-state">
        <span className="error-icon"><IconAlertTriangle size={36} /></span>
        <p className="error-msg">Erreur : {error.message}</p>
      </div>
    </div>
  );

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <div className="dashboard-page">

      {/* Section header */}
      <div className="section-header fu">
        <div className="section-label">MONITORING</div>
        <h2 className="section-title">Vue d'ensemble <span>de la sécurité</span></h2>
        <p className="section-subtitle">
          Synthèse de toutes les vulnérabilités, par produit et par niveau de risque.
        </p>
      </div>

      {/* Alert banner */}
      {urgentRatio > 15 && (
        <div className="alert-critical fu1">
          <div className="alert-critical-dot" />
          <div className="alert-critical-content">
            <strong>Action immédiate requise</strong>
            <span>
              {stats.totalCritical + stats.totalHigh} findings critiques/high représentent{' '}
              {urgentRatio.toFixed(0)}% du total — dépassement du seuil d'alerte (15%).
            </span>
          </div>
          <button className="alert-critical-btn"
            onClick={() => { setRiskFilter('Critical'); setActiveTab('products'); }}>
            Filtrer <IconArrowRight />
          </button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="stats-grid fu2">
        {KPI_CARDS.map((card, i) => (
          <div key={i} className="stat-card"
            style={{ '--card-accent': KPI_COLORS[card.colorIdx] } as React.CSSProperties}>
            <div className="stat-card-icon" style={{ color: KPI_COLORS[card.colorIdx] }}>
              {KPI_ICONS[card.colorIdx]}
            </div>
            <div className="stat-card-value" style={{ color: KPI_COLORS[card.colorIdx] }}>
              <AnimatedCounter target={card.value} />
            </div>
            <div className="stat-card-label">{card.label}</div>
            <div className="stat-card-glow" />
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="dashboard-tabs fu3">
        {TABS.map((tab) => (
          <button key={tab.id}
            className={`dash-tab${activeTab === tab.id ? ' dash-tab--active' : ''}`}
            onClick={() => setActiveTab(tab.id)}>
            <span className="dash-tab-icon">{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* ══ TAB: OVERVIEW ══════════════════════════════════════════════════ */}
      {activeTab === 'overview' && (
        <div className="charts-row fu4">

          {/* Donut + distribution */}
          <div className="chart-card chart-card--split">
            <h3 className="chart-title">
              <span className="chart-title-dot" style={{ background: 'var(--accent)' }} />
              Distribution des risques
            </h3>
            <div className="chart-split-inner">
              <ResponsiveContainer width="50%" height={260}>
                <PieChart>
                  <Pie data={riskData} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={58} outerRadius={98}
                    paddingAngle={3} labelLine={false}>
                    {riskData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color}
                        stroke="rgba(0,0,0,0.25)" strokeWidth={2} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="chart-split-right">
                <RiskDistributionBars
                  riskData={riskData}
                  total={stats.totalFindings}
                  onFilter={(name) => { setRiskFilter(name); setActiveTab('products'); }}
                  activeFilter={riskFilter}
                />
              </div>
            </div>
          </div>

          {/* Top products bar chart */}
          <div className="chart-card">
            <h3 className="chart-title">
              <span className="chart-title-dot" style={{ background: 'var(--purple)' }} />
              Top 8 produits — findings
            </h3>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={productSummaries.slice(0, 8)} layout="vertical"
                margin={{ left: 10, right: 20, top: 8, bottom: 8 }}>
                <XAxis type="number" tick={{ fill: 'var(--dimmed)', fontSize: 11 }}
                  axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={110}
                  tick={{ fill: 'var(--muted)', fontSize: 11 }}
                  axisLine={false} tickLine={false}
                  tickFormatter={(v: string) => v.length > 14 ? v.slice(0, 14) + '…' : v} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="critical" stackId="a" fill="var(--severity-critical)" name="Critique" barSize={14} />
                <Bar dataKey="high"     stackId="a" fill="var(--severity-high)"     name="High"     barSize={14} />
                <Bar dataKey="medium"   stackId="a" fill="var(--severity-medium)"   name="Medium"   barSize={14} />
                <Bar dataKey="low"      stackId="a" fill="var(--severity-low)"      name="Low"      barSize={14} radius={[0, 4, 4, 0]} />
                <Legend iconType="square" iconSize={8}
                  wrapperStyle={{ fontSize: 11, color: 'var(--dimmed)' }} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ══ TAB: PRODUCTS ══════════════════════════════════════════════════ */}
      {activeTab === 'products' && (
        <div className="products-summary-table fu4">

          {/* Filters */}
          <div className="table-filters">
            <div className="search-wrap">
              <span className="search-icon"><IconSearch size={15} /></span>
              <input className="search-input" type="text"
                placeholder="Rechercher un produit…"
                value={search} onChange={(e) => setSearch(e.target.value)} />
              {search && (
                <button className="search-clear" onClick={() => setSearch('')}>
                  <IconX size={12} />
                </button>
              )}
            </div>

            <div className="risk-filter-btns">
              <button
                className={`risk-btn${riskFilter === null ? ' risk-btn--active' : ''}`}
                onClick={() => setRiskFilter(null)}>
                Tous
              </button>
              {RISK_ORDER.map((level) => (
                <button key={level}
                  className={`risk-btn${riskFilter === level ? ' risk-btn--active' : ''}`}
                  style={{ '--btn-color': RISK_COLORS[level] } as React.CSSProperties}
                  onClick={() => setRiskFilter(riskFilter === level ? null : level)}>
                  {level}
                </button>
              ))}
            </div>

            <span className="table-count">
              {filteredProducts.length} produit{filteredProducts.length !== 1 ? 's' : ''}
            </span>
          </div>

          <div className="table-container">
            <table className="dashboard-table">
              <thead>
                <tr>
                  <th>Produit</th>
                  {[
                    { key: 'totalFindings', label: 'Total'      },
                    { key: 'critical',      label: 'Critique'   },
                    { key: 'high',          label: 'High'       },
                    { key: 'medium',        label: 'Medium'     },
                    { key: 'low',           label: 'Low'        },
                    { key: 'avgRiskScore',  label: 'Score CVSS' },
                    { key: 'priorityScore', label: 'Priorité'   },
                  ].map(({ key, label }) => (
                    <th key={key} className="th-sortable" onClick={() => handleSort(key)}>
                      <span className="th-inner">
                        {label}
                        <SortIcon k={key} />
                      </span>
                    </th>
                  ))}
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {withPriority.length === 0 && (
                  <tr>
                    <td colSpan={9} className="table-empty">
                      Aucun produit ne correspond aux filtres.
                    </td>
                  </tr>
                )}
                {withPriority.map((product) => (
                  <tr key={product.id}
                    className={`table-row${product.critical > 0 ? ' row--critical' : product.high > 5 ? ' row--high' : ''}`}>
                    <td className="td-product">
                      <div className="product-name-cell">
                        <div className="product-severity-dot" style={{
                          background: product.critical > 0 ? 'var(--severity-critical)'
                            : product.high > 0 ? 'var(--severity-high)'
                            : 'var(--severity-low)',
                        }} />
                        <span>{product.name}</span>
                      </div>
                    </td>
                    <td><strong>{product.totalFindings}</strong></td>
                    <td>
                      {product.critical > 0
                        ? <span className="badge badge--critical">{product.critical}</span>
                        : <span className="badge-zero">—</span>}
                    </td>
                    <td>
                      {product.high > 0
                        ? <span className="badge badge--high">{product.high}</span>
                        : <span className="badge-zero">—</span>}
                    </td>
                    <td style={{ color: 'var(--severity-medium)' }}>{product.medium || '—'}</td>
                    <td style={{ color: 'var(--severity-low)'    }}>{product.low    || '—'}</td>
                    <td><ScoreBar score={product.avgRiskScore} /></td>
                    <td><PriorityBar score={product.priorityScore} max={maxPriority} /></td>
                    <td>
                      <Link to={`/engagements?productId=${product.id}`} className="table-link">
                        Voir <IconArrowRight size={11} />
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      

    </div>
  );
}