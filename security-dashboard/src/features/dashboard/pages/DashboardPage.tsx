import { useDashboardData } from '../hooks/useDashboardData';
import { Link } from 'react-router-dom';
import { useState, useMemo, useEffect, useRef } from 'react';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Legend, LineChart, Line, CartesianGrid
} from 'recharts';
import './DashboardPage.css';

type RiskLevel = 'Critical' | 'High' | 'Medium' | 'Low' | 'Info';

interface RiskEntry { name: string; value: number; color: string; }

const RISK_COLORS: Record<RiskLevel, string> = {
  Critical: 'var(--accent2)',
  High:     'var(--orange)',
  Medium:   'var(--accent3)',
  Low:      'var(--green)',
  Info:     'var(--purple)',
};

const RISK_ORDER: RiskLevel[] = ['Critical', 'High', 'Medium', 'Low', 'Info'];

/* ─── Animated Counter ──────────────────────────────────────────────────── */
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

/* ─── Confidence Icon ───────────────────────────────────────────────────── */
function ConfidenceIcon({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  let cls = 'conf-low', symbol = '○';
  if (pct >= 85) { cls = 'conf-high'; symbol = '◉'; }
  else if (pct >= 65) { cls = 'conf-mid'; symbol = '◎'; }
  return (
    <span className={`conf-icon ${cls}`} title={`Confiance : ${pct}%`}>
      {symbol} <span className="conf-pct">{pct}%</span>
    </span>
  );
}

/* ─── Score Bar ─────────────────────────────────────────────────────────── */
function ScoreBar({ score, max = 10 }: { score: number; max?: number }) {
  const pct = Math.min((score / max) * 100, 100);
  const color = score >= 9 ? 'var(--accent2)'
    : score >= 7 ? 'var(--orange)'
    : score >= 4 ? 'var(--accent3)'
    : 'var(--green)';
  return (
    <div className="score-pill">
      <span className="score-val" style={{ color }}>{score.toFixed(1)}</span>
      <div className="score-track">
        <div
          className="score-fill"
          style={{ '--bar-w': `${pct}%`, background: color } as React.CSSProperties}
        />
      </div>
    </div>
  );
}

/* ─── Risk Distribution Bars ────────────────────────────────────────────── */
function RiskDistributionBars({
  riskData,
  total,
  onFilter,
  activeFilter,
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
          <div
            key={entry.name}
            className={`dist-row ${isActive ? 'dist-row--active' : ''}`}
            onClick={() => onFilter(isActive ? null : entry.name)}
          >
            <div className="dist-row-header">
              <span className="dist-row-name" style={{ color: entry.color }}>{entry.name}</span>
              <span className="dist-row-count">
                <strong>{entry.value}</strong>
                <span className="dist-row-pct">{pct}%</span>
              </span>
            </div>
            <div className="dist-bar-track">
              <div
                className="dist-bar-fill"
                style={{
                  '--bar-width': `${pct}%`,
                  background: entry.color,
                } as React.CSSProperties}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ─── Priority Bar ──────────────────────────────────────────────────────── */
function PriorityBar({ score, max }: { score: number; max: number }) {
  const pct = max > 0 ? (score / max) * 100 : 0;
  const color = pct > 75 ? 'var(--accent2)' : pct > 45 ? 'var(--orange)' : pct > 20 ? 'var(--accent3)' : 'var(--green)';
  return (
    <div className="priority-bar-wrap">
      <div className="priority-bar-track">
        <div
          className="priority-bar-fill"
          style={{ '--bar-width': `${pct}%`, background: color } as React.CSSProperties}
        />
      </div>
      <span className="priority-bar-val" style={{ color }}>{score.toFixed(0)}</span>
    </div>
  );
}

/* ─── Skeleton ──────────────────────────────────────────────────────────── */
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

/* ─── Custom Tooltip ────────────────────────────────────────────────────── */
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

/* ─── Main Dashboard Page ───────────────────────────────────────────────── */
export default function DashboardPage() {
  const { productSummaries, stats, isLoading, error } = useDashboardData();

  /* Filters & Sort */
  const [search, setSearch]           = useState('');
  const [riskFilter, setRiskFilter]   = useState<string | null>(null);
  const [sortKey, setSortKey]         = useState<string>('totalFindings');
  const [sortDir, setSortDir]         = useState<'asc' | 'desc'>('desc');
  const [activeTab, setActiveTab]     = useState<'overview' | 'products' | 'timeline'>('overview');

  /* Alert threshold: if (critical+high)/total > 15% */
  const urgentRatio = stats
    ? ((stats.totalCritical + stats.totalHigh) / Math.max(stats.totalFindings, 1)) * 100
    : 0;

  /* Risk data with fallback colors */
  const riskData = useMemo<RiskEntry[]>(() => {
    if (!stats?.riskDistribution) return [];
    return (stats.riskDistribution as RiskEntry[]).map((r) => ({
      ...r,
      color: r.color || RISK_COLORS[r.name as RiskLevel] || 'var(--muted)',
    }));
  }, [stats]);

  /* Filtered & sorted products */
  const filteredProducts = useMemo(() => {
    if (!productSummaries) return [];
    let list = [...productSummaries];

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter((p) =>
        p.name.toLowerCase().includes(q)
      );
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

  /* Priority score */
  const withPriority = useMemo(() => filteredProducts.map((p) => ({
    ...p,
    priorityScore: (p.critical * 4 + p.high * 3 + p.medium * 2 + p.low) /
      Math.max(p.totalFindings, 1) * 10,
  })), [filteredProducts]);

  const maxPriority = useMemo(
    () => Math.max(...withPriority.map((p) => p.priorityScore), 1),
    [withPriority]
  );

  /* Sort handler */
  const handleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    else { setSortKey(key); setSortDir('desc'); }
  };

  const SortIcon = ({ k }: { k: string }) =>
    sortKey === k ? (
      <span className="sort-icon sort-icon--active">{sortDir === 'desc' ? '↓' : '↑'}</span>
    ) : (
      <span className="sort-icon">↕</span>
    );

  /* ─── Loading skeleton ──────────────────────────────────────────────── */
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

  if (error) return (
    <div className="dashboard-page">
      <div className="error-state">
        <div className="error-icon">⚠</div>
        <p className="error-msg">Erreur : {error.message}</p>
      </div>
    </div>
  );

  return (
    <div className="dashboard-page">

      {/* ── Section header ─────────────────────────────────────────────── */}
      <div className="section-header fu">
        <div className="section-label">MONITORING</div>
        <h2 className="section-title">Vue d'ensemble <span>de la sécurité</span></h2>
        <p className="section-subtitle">
          Synthèse de toutes les vulnérabilités, par produit et par niveau de risque.
        </p>
      </div>

      {/* ── Alert Banner ───────────────────────────────────────────────── */}
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
          <button
            className="alert-critical-btn"
            onClick={() => { setRiskFilter('Critical'); setActiveTab('products'); }}
          >
            Filtrer →
          </button>
        </div>
      )}

      {/* ── KPI Cards ──────────────────────────────────────────────────── */}
      <div className="stats-grid fu2">
        {[
          { value: stats.totalFindings,                          label: 'Findings totaux',   color: 'var(--accent)',  icon: '🔍' },
          { value: stats.totalCritical,                          label: 'Critiques',          color: 'var(--accent2)', icon: '🚨' },
          { value: stats.totalHigh,                              label: 'High',               color: 'var(--orange)',  icon: '⚠' },
          { value: stats.totalCritical + stats.totalHigh,        label: 'Urgents (C+H)',      color: '#ff3366',        icon: '🔥' },
          { value: stats.totalProducts,                          label: 'Produits',           color: 'var(--purple)',  icon: '📦' },
        ].map((card, i) => (
          <div
            key={i}
            className="stat-card"
            style={{ '--card-accent': card.color } as React.CSSProperties}
          >
            <div className="stat-card-icon">{card.icon}</div>
            <div className="stat-card-value" style={{ color: card.color }}>
              <AnimatedCounter target={card.value} />
            </div>
            <div className="stat-card-label">{card.label}</div>
            <div className="stat-card-glow" />
          </div>
        ))}
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────── */}
      <div className="dashboard-tabs fu3">
        {(['overview', 'products', 'timeline'] as const).map((tab) => (
          <button
            key={tab}
            className={`dash-tab ${activeTab === tab ? 'dash-tab--active' : ''}`}
            onClick={() => setActiveTab(tab)}
          >
            {{ overview: '📊 Vue globale', products: '📋 Produits', timeline: '📈 Tendances' }[tab]}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB: OVERVIEW                                                    */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {activeTab === 'overview' && (
        <>
          {/* ── Charts row ─────────────────────────────────────────── */}
          <div className="charts-row fu4">

            {/* Donut + distribution bars */}
            <div className="chart-card chart-card--split">
              <h3 className="chart-title">
                <span className="chart-title-dot" style={{ background: 'var(--accent)' }} />
                Distribution des risques
              </h3>
              <div className="chart-split-inner">
                <ResponsiveContainer width="50%" height={260}>
                  <PieChart>
                    <Pie
                      data={riskData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={3}
                      labelLine={false}
                    >
                      {riskData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.color}
                          stroke="rgba(0,0,0,0.3)"
                          strokeWidth={2}
                        />
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
                <BarChart
                  data={productSummaries.slice(0, 8)}
                  layout="vertical"
                  margin={{ left: 10, right: 20, top: 8, bottom: 8 }}
                >
                  <XAxis type="number" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={110}
                    tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    tickFormatter={(v: string) => v.length > 14 ? v.slice(0, 14) + '…' : v}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="critical" stackId="a" fill="var(--accent2)" name="Critique" barSize={14} radius={[0, 0, 0, 0]} />
                  <Bar dataKey="high"     stackId="a" fill="var(--orange)"  name="High"     barSize={14} />
                  <Bar dataKey="medium"   stackId="a" fill="var(--accent3)" name="Medium"   barSize={14} />
                  <Bar dataKey="low"      stackId="a" fill="var(--green)"   name="Low"      barSize={14} radius={[0, 4, 4, 0]} />
                  <Legend
                    iconType="square"
                    iconSize={8}
                    wrapperStyle={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB: PRODUCTS                                                    */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {activeTab === 'products' && (
        <div className="products-summary-table fu4">
          {/* Filters bar */}
          <div className="table-filters">
            <div className="search-wrap">
              <span className="search-icon">⌕</span>
              <input
                className="search-input"
                type="text"
                placeholder="Rechercher un produit…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <button className="search-clear" onClick={() => setSearch('')}>✕</button>
              )}
            </div>
            <div className="risk-filter-btns">
              <button
                className={`risk-btn ${riskFilter === null ? 'risk-btn--active' : ''}`}
                onClick={() => setRiskFilter(null)}
              >
                Tous
              </button>
              {RISK_ORDER.map((level) => (
                <button
                  key={level}
                  className={`risk-btn ${riskFilter === level ? 'risk-btn--active' : ''}`}
                  style={{ '--btn-color': RISK_COLORS[level] } as React.CSSProperties}
                  onClick={() => setRiskFilter(riskFilter === level ? null : level)}
                >
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
                    { key: 'totalFindings', label: 'Total' },
                    { key: 'critical',      label: 'Critique' },
                    { key: 'high',          label: 'High' },
                    { key: 'medium',        label: 'Medium' },
                    { key: 'low',           label: 'Low' },
                    { key: 'avgRiskScore',  label: 'Score CVSS' },
                    { key: 'priorityScore', label: 'Priorité' },
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      className="th-sortable"
                      onClick={() => handleSort(key)}
                    >
                      {label} <SortIcon k={key} />
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
                {withPriority.map((product) => {
                  const urgentPct = ((product.critical + product.high) / Math.max(product.totalFindings, 1)) * 100;
                  return (
                    <tr
                      key={product.id}
                      className={`table-row ${product.critical > 0 ? 'row--critical' : product.high > 5 ? 'row--high' : ''}`}
                    >
                      <td className="td-product">
                        <div className="product-name-cell">
                          <div
                            className="product-severity-dot"
                            style={{
                              background: product.critical > 0 ? 'var(--accent2)'
                                : product.high > 0 ? 'var(--orange)'
                                : 'var(--green)',
                            }}
                          />
                          <span>{product.name}</span>
                        </div>
                      </td>
                      <td><strong>{product.totalFindings}</strong></td>
                      <td>
                        {product.critical > 0
                          ? <span className="badge badge--critical">{product.critical}</span>
                          : <span className="badge-zero">—</span>
                        }
                      </td>
                      <td>
                        {product.high > 0
                          ? <span className="badge badge--high">{product.high}</span>
                          : <span className="badge-zero">—</span>
                        }
                      </td>
                      <td style={{ color: 'var(--accent3)' }}>{product.medium || '—'}</td>
                      <td style={{ color: 'var(--green)' }}>{product.low || '—'}</td>
                      <td>
                        <ScoreBar score={product.avgRiskScore} />
                      </td>
                      <td>
                        <PriorityBar score={product.priorityScore} max={maxPriority} />
                      </td>
                      <td>
                        <Link
                          to={`/engagements?productId=${product.id}`}
                          className="table-link"
                        >
                          Voir →
                        </Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════════ */}
      {/* TAB: TIMELINE                                                    */}
      {/* ═══════════════════════════════════════════════════════════════ */}
      {activeTab === 'timeline' && (
        <div className="chart-card fu4" style={{ marginTop: '1.5rem' }}>
          <h3 className="chart-title">
            <span className="chart-title-dot" style={{ background: 'var(--accent)' }} />
            Évolution temporelle des findings
          </h3>
          <p className="chart-subtitle">
            Tendance des nouveaux findings critiques et high par période.
          </p>
          {/* Placeholder: replace with real time series data from your API */}
          <ResponsiveContainer width="100%" height={320}>
            <LineChart
              data={[
                { name: 'Jan', critical: 8,  high: 14 },
                { name: 'Fév', critical: 12, high: 18 },
                { name: 'Mar', critical: 6,  high: 22 },
                { name: 'Avr', critical: 15, high: 9  },
                { name: 'Mai', critical: 10, high: 25 },
                { name: 'Jun', critical: 7,  high: 13 },
              ]}
              margin={{ left: 0, right: 16, top: 12, bottom: 8 }}
            >
              <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }}
              />
              <Line
                type="monotone"
                dataKey="critical"
                stroke="var(--accent2)"
                strokeWidth={2.5}
                dot={{ fill: 'var(--accent2)', r: 4, strokeWidth: 0 }}
                activeDot={{ r: 6 }}
                name="Critique"
              />
              <Line
                type="monotone"
                dataKey="high"
                stroke="var(--orange)"
                strokeWidth={2.5}
                dot={{ fill: 'var(--orange)', r: 4, strokeWidth: 0 }}
                activeDot={{ r: 6 }}
                name="High"
              />
            </LineChart>
          </ResponsiveContainer>
          <p className="chart-note">
            ⚠ Données de démo — connectez votre endpoint de séries temporelles pour les données réelles.
          </p>
        </div>
      )}

    </div>
  );
}