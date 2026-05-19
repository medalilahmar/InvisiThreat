import { useState } from 'react';
import { useAnalyticsData, useProductAnalytics } from '../hooks/useAnalyticsData';
import { useAuth } from '../../../auth/hooks/useAuth';
import { RiskGauge } from '../components/RiskGauge';
import { HeatmapChart } from '../components/HeatmapChart';
import { RadarRiskChart } from '../components/RadarChart';
import { MttrChart } from '../components/MttrChart';
import { FunnelChart } from '../components/FunnelChart';
import { TimelineChart } from '../components/TimelineChart';
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

type Tab = 'overview' | 'trends' | 'products' | 'performance';



const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: 'var(--bg3)', border: '1px solid ${card.color}33', borderRadius: 8, padding: '10px 14px', fontSize: 12 }}>
      {label && <p style={{ color: 'rgba(255,255,255,0.5)', marginBottom: 6 }}>{label}</p>}
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color || '#fff', margin: '2px 0' }}>
          {p.name}: <strong>{p.value}</strong>
        </p>
      ))}
    </div>
  );
};

export default function AnalyticsPage() {
  const { user } = useAuth();
  const { data, isLoading, error } = useAnalyticsData();
  const [tab, setTab] = useState<Tab>('overview');
  const [selectedProduct, setSelectedProduct] = useState<number | null>(null);
  const { data: productData, isLoading: productLoading } = useProductAnalytics(selectedProduct);


  if (isLoading) return (
    <div style={pageStyle}>
      <div style={{ textAlign: 'center', padding: '4rem', color: 'rgba(255,255,255,0.4)' }}>
        <div style={{ fontSize: 32, marginBottom: 16 }}>📊</div>
        <div>Calcul des statistiques en cours...</div>
      </div>
    </div>
  );

  if (error || !data) return (
    <div style={pageStyle}>
      <div style={{ textAlign: 'center', padding: '4rem', color: '#fca5a5' }}>
        ⚠ Erreur lors du chargement des analytics
      </div>
    </div>
  );

  const { summary, severity_distribution, top_products, heatmap, top_vuln_types, timeline, funnel, mttr } = data;

  return (
    <div style={pageStyle}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{ fontSize: 11, color: '#6366f1', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6 }}>
              ANALYTICS
            </div>
            <h1 style={{ margin: 0, fontSize: 28, fontWeight: 700, letterSpacing: '-0.5px' }}>
              Analyse de sécurité
            </h1>
            <p style={{ margin: '6px 0 0', color: 'rgba(255,255,255,0.4)', fontSize: 14 }}>
              {data.filtered
                ? `Données filtrées — ${summary.total_products} produit(s) assigné(s)`
                : `Vue globale — ${summary.total_products} produits analysés`}
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: 'rgba(99,102,241,0.1)', border: '0.5px solid rgba(99,102,241,0.3)', borderRadius: 10 }}>
            <span style={{ fontSize: 12, color: '#a5b4fc' }}>
              {user?.role === 'admin' ? '👑 Admin' : user?.role === 'analyst' ? '🔍 Analyst' : user?.role === 'manager' ? '🎯 Manager' : '⚙️ Developer'}
            </span>
          </div>
        </div>
      </div>

      {/* ── Alerte urgente ─────────────────────────────────────────────────── */}
      {summary.urgent_ratio > 15 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', background: 'rgba(255,71,87,0.1)', border: '0.5px solid rgba(255,71,87,0.3)', borderRadius: 10, marginBottom: '1.5rem' }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ff4757', boxShadow: '0 0 8px #ff4757' }} />
          <div>
            <strong style={{ color: '#ff4757' }}>Action immédiate requise</strong>
            <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: 13, marginLeft: 8 }}>
              {summary.urgent_count} findings critiques/high ({summary.urgent_ratio}% du total)
            </span>
          </div>
        </div>
      )}

      {/* ── KPI Cards ──────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 10, marginBottom: '2rem' }}>
        {[
          { label: 'Total',        value: summary.total_findings,    color: '#6366f1', icon: '🔍' },
          { label: 'Critical',     value: summary.total_critical,    color: '#ff4757', icon: '🚨' },
          { label: 'High',         value: summary.total_high,        color: '#ff6b35', icon: '⚠️' },
          { label: 'Medium',       value: summary.total_medium,      color: '#ffd32a', icon: '🟡' },
          { label: 'Low',          value: summary.total_low,         color: '#2ed573', icon: '🟢' },
          { label: 'Score Risque', value: summary.global_risk_score, color: summary.global_risk_score > 70 ? '#ff4757' : summary.global_risk_score > 40 ? '#ffd32a' : '#2ed573', icon: '🎯', suffix: '/100' },
          { label: 'CVSS Moyen',   value: summary.avg_cvss,          color: '#8b5cf6', icon: '📊', decimal: true },
        ].map((card, i) => (
          <div key={i} style={{
            background: 'var(--bg3)',
            border: `0.5px solid ${card.color}33`,
            borderRadius: 12, padding: '14px 12px',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 20, marginBottom: 6 }}>{card.icon}</div>
            <div style={{ fontSize: 22, fontWeight: 800, color: card.color }}>
              {card.decimal ? card.value.toFixed(1) : Math.round(card.value as number).toLocaleString()}
              {card.suffix || ''}
            </div>
            <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              {card.label}
            </div>
          </div>
        ))}
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 4, background: 'rgba(255,255,255,0.04)', borderRadius: 12, padding: 4, marginBottom: '2rem' }}>
        {([
          { key: 'overview',     label: '📊 Vue globale' },
          { key: 'trends',       label: '📈 Tendances' },
          { key: 'products',     label: '🏭 Par produit' },
          { key: 'performance',  label: '⚡ Performance' },
        ] as const).map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            flex: 1, padding: '10px 16px', borderRadius: 10,
            background: tab === t.key ? '#6366f1' : 'transparent',
            color: tab === t.key ? '#fff' : 'rgba(255,255,255,0.4)',
            border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 500,
            transition: 'all 0.2s',
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/* TAB: OVERVIEW                                                      */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {tab === 'overview' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Row 1 — Gauge + Donut + Funnel */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle title="Score de risque global" dot="#ff4757" />
              <RiskGauge score={Math.round(summary.global_risk_score)} />
              <div style={{ textAlign: 'center', fontSize: 12, color: 'rgba(255,255,255,0.4)', marginTop: 8 }}>
                Basé sur {summary.total_findings.toLocaleString()} findings
              </div>
            </div>

            <div style={card}>
              <CardTitle title="Distribution des sévérités" dot="#6366f1" />
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <ResponsiveContainer width="50%" height={220}>
                  <PieChart>
                    <Pie data={severity_distribution} dataKey="value" cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3}>
                      {severity_distribution.map((entry, i) => (
                        <Cell key={i} fill={entry.color} stroke="rgba(0,0,0,0.3)" strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {severity_distribution.map((s, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: s.color }} />
                        <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.6)' }}>{s.name}</span>
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 700, color: s.color }}>{s.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={card}>
              <CardTitle title="Entonnoir de risque" dot="#ffd32a" />
              <FunnelChart data={funnel} />
            </div>
          </div>

          {/* Row 2 — Top produits + Heatmap */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle title="Top 10 produits — Score risque" dot="#ff6b35" />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <XAxis type="number" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" width={100} tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }} axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 12) + '…' : v} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="risk_score" name="Score risque" fill="#6366f1" radius={[0, 6, 6, 0]} maxBarSize={18}
                    label={{ position: 'right', fill: 'rgba(255,255,255,0.4)', fontSize: 10, formatter: (v: unknown) => `${Number(v)}` }} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={card}>
              <CardTitle title="Heatmap — Produits × Sévérité" dot="#8b5cf6" />
              <HeatmapChart data={heatmap} />
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/* TAB: TRENDS                                                        */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {tab === 'trends' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          <div style={card}>
            <CardTitle title="Évolution temporelle des findings" dot="#6366f1" />
            <p style={{ margin: '0 0 1rem', fontSize: 13, color: 'rgba(255,255,255,0.4)' }}>
              Tendance des findings par mois et par sévérité
            </p>
            {timeline.length > 0
              ? <TimelineChart data={timeline} />
              : <EmptyState msg="Pas assez de données temporelles disponibles" />
            }
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle title="Top 10 types de vulnérabilités" dot="#ff4757" />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_vuln_types} layout="vertical" margin={{ left: 10, right: 30 }}>
                  <XAxis type="number" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={90} tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Findings" fill="#ff4757" radius={[0, 6, 6, 0]} maxBarSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={card}>
              <CardTitle title="Stacked Bar — Findings par sévérité/produit" dot="#ffd32a" />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 6)} margin={{ left: 0, right: 16 }}>
                  <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 10 }} axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 8 ? v.slice(0, 8) + '…' : v} />
                  <YAxis tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }} />
                  <Bar dataKey="critical" stackId="a" fill="#ff4757" name="Critical" maxBarSize={40} />
                  <Bar dataKey="high"     stackId="a" fill="#ff6b35" name="High"     maxBarSize={40} />
                  <Bar dataKey="medium"   stackId="a" fill="#ffd32a" name="Medium"   maxBarSize={40} />
                  <Bar dataKey="low"      stackId="a" fill="#2ed573" name="Low"      maxBarSize={40} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/* TAB: PRODUCTS                                                      */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {tab === 'products' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* Sélecteur de produit */}
          <div style={card}>
            <CardTitle title="Sélectionner un produit" dot="#6366f1" />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              {data.by_product.map(p => (
                <button key={p.id} onClick={() => setSelectedProduct(selectedProduct === p.id ? null : p.id)} style={{
                  padding: '8px 14px', borderRadius: 8,
                  background: selectedProduct === p.id ? '#6366f1' : 'rgba(255,255,255,0.05)',
                  color: selectedProduct === p.id ? '#fff' : 'rgba(255,255,255,0.6)',
                  border: `0.5px solid ${selectedProduct === p.id ? '#6366f1' : 'rgba(255,255,255,0.1)'}`,
                  cursor: 'pointer', fontSize: 13, transition: 'all 0.2s',
                }}>
                  {p.name.length > 20 ? p.name.slice(0, 20) + '…' : p.name}
                  {p.critical > 0 && <span style={{ marginLeft: 6, color: '#ff4757', fontWeight: 700 }}>●</span>}
                </button>
              ))}
            </div>
          </div>

          {/* Dashboard produit sélectionné */}
          {selectedProduct && productLoading && (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'rgba(255,255,255,0.4)' }}>
              Chargement des stats du produit...
            </div>
          )}

          {selectedProduct && productData && !productLoading && (
            <>
              {/* KPIs produit */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10 }}>
                {[
                  { label: 'Total',    value: productData.total_findings, color: '#6366f1' },
                  { label: 'Critical', value: productData.critical,       color: '#ff4757' },
                  { label: 'High',     value: productData.high,           color: '#ff6b35' },
                  { label: 'Medium',   value: productData.medium,         color: '#ffd32a' },
                  { label: 'Low',      value: productData.low,            color: '#2ed573' },
                  { label: 'Score',    value: productData.risk_score,     color: productData.risk_score > 70 ? '#ff4757' : '#ffd32a', suffix: '/100' },
                ].map((k, i) => (
                  <div key={i} style={{ background: 'var(--bg3)', border: `0.5px solid ${k.color}33`, borderRadius: 10, padding: '12px', textAlign: 'center' }}>
                    <div style={{ fontSize: 22, fontWeight: 800, color: k.color }}>{k.value}{k.suffix || ''}</div>
                    <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 4 }}>{k.label}</div>
                  </div>
                ))}
              </div>

              {/* Radar + Timeline produit */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>
                <div style={card}>
                  <CardTitle title="Profil de risque" dot="#8b5cf6" />
                  <RadarRiskChart data={productData.radar} />
                </div>
                <div style={card}>
                  <CardTitle title="Évolution du produit" dot="#6366f1" />
                  {productData.timeline.length > 0
                    ? <TimelineChart data={productData.timeline} />
                    : <EmptyState msg="Pas de données temporelles pour ce produit" />
                  }
                </div>
              </div>

              {/* Top vuln types + Progress bars */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div style={card}>
                  <CardTitle title="Types de vulnérabilités" dot="#ff4757" />
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={productData.top_vuln_types} layout="vertical" margin={{ left: 10, right: 30 }}>
                      <XAxis type="number" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="name" width={80} tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="value" name="Findings" fill="#ff4757" radius={[0, 6, 6, 0]} maxBarSize={18} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={card}>
                  <CardTitle title="Taux de résolution par sévérité" dot="#2ed573" />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 8 }}>
                    {[
                      { sev: 'Critical', count: productData.critical, color: '#ff4757' },
                      { sev: 'High',     count: productData.high,     color: '#ff6b35' },
                      { sev: 'Medium',   count: productData.medium,   color: '#ffd32a' },
                      { sev: 'Low',      count: productData.low,      color: '#2ed573' },
                    ].map(({ sev, count, color }) => {
                      const pct = productData.total_findings > 0 ? Math.round((count / productData.total_findings) * 100) : 0;
                      return (
                        <div key={sev}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 12 }}>
                            <span style={{ color: 'rgba(255,255,255,0.6)' }}>{sev}</span>
                            <span style={{ color, fontWeight: 700 }}>{count} ({pct}%)</span>
                          </div>
                          <div style={{ background: 'rgba(255,255,255,0.06)', borderRadius: 4, height: 8 }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 4, transition: 'width 0.8s ease' }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </>
          )}

          {!selectedProduct && (
            <div style={{ textAlign: 'center', padding: '3rem', color: 'rgba(255,255,255,0.25)', fontSize: 14 }}>
              Sélectionnez un produit ci-dessus pour voir ses statistiques détaillées
            </div>
          )}
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/* TAB: PERFORMANCE                                                   */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {tab === 'performance' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle title="MTTR — Temps moyen de résolution" dot="#6366f1" />
              <p style={{ margin: '0 0 1rem', fontSize: 13, color: 'rgba(255,255,255,0.4)' }}>
                Basé sur l'âge moyen des findings par sévérité
              </p>
              <MttrChart mttr={mttr} />
            </div>

            <div style={card}>
              <CardTitle title="CVSS Moyen par produit" dot="#8b5cf6" />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 40 }}>
                  <XAxis type="number" domain={[0, 10]} tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={100} tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 11 }} axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 12) + '…' : v} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avg_cvss" name="CVSS Moyen" fill="#8b5cf6" radius={[0, 6, 6, 0]} maxBarSize={18}
                    label={{ position: 'right', fill: 'rgba(255,255,255,0.4)', fontSize: 10, formatter: (v: unknown) => `${Number(v).toFixed(1)}` }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Stats globales de performance */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
            {[
              { title: 'Âge moyen des findings',  value: `${summary.avg_age_days} jours`, subtitle: 'Depuis la date de création', color: '#ffd32a', icon: '⏱️' },
              { title: 'Ratio urgent',             value: `${summary.urgent_ratio}%`,      subtitle: 'Critical + High / Total',    color: summary.urgent_ratio > 15 ? '#ff4757' : '#2ed573', icon: '🎯' },
              { title: 'Score CVSS moyen',         value: summary.avg_cvss.toFixed(1),     subtitle: 'Tous produits confondus',    color: summary.avg_cvss > 7 ? '#ff4757' : summary.avg_cvss > 4 ? '#ffd32a' : '#2ed573', icon: '📊' },
            ].map((s, i) => (
              <div key={i} style={{ ...card, textAlign: 'center' }}>
                <div style={{ fontSize: 28, marginBottom: 8 }}>{s.icon}</div>
                <div style={{ fontSize: 32, fontWeight: 800, color: s.color }}>{s.value}</div>
                <div style={{ fontSize: 14, fontWeight: 600, marginTop: 8 }}>{s.title}</div>
                <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)', marginTop: 4 }}>{s.subtitle}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Composants utilitaires ────────────────────────────────────────────────────

function CardTitle({ title, dot }: { title: string; dot: string }) {
  return (
    <h3 style={{ margin: '0 0 1rem', fontSize: 14, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: dot, flexShrink: 0 }} />
      {title}
    </h3>
  );
}

function EmptyState({ msg }: { msg: string }) {
  return (
    <div style={{ textAlign: 'center', padding: '2rem', color: 'rgba(255,255,255,0.25)', fontSize: 13 }}>
      {msg}
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const pageStyle: React.CSSProperties = {
  minHeight: '100vh',
  background: 'var(--bg)',
  padding: '2rem',
  color: 'var(--text)',
};

const card: React.CSSProperties = {
  background: 'var(--bg3)',
  border: '1px solid var(--border)',
  borderRadius: 16,
  padding: '1.5rem',
};