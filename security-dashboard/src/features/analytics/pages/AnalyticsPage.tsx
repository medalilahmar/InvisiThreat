import { useState, useEffect } from 'react';
import { useAnalyticsData, useProductAnalytics } from '../hooks/useAnalyticsData';
import { useAuth } from '../../../auth/hooks/useAuth';
import { RiskGauge } from '../components/RiskGauge';
import { HeatmapChart } from '../components/HeatmapChart';
import { RadarRiskChart } from '../components/RadarChart';
import { MttrChart } from '../components/MttrChart';
import { FunnelChart } from '../components/FunnelChart';
import { TimelineChart } from '../components/TimelineChart';
import CodeHeatmap from "../components/CodeHeatmap";

import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

type Tab = 'overview' | 'trends' | 'products' | 'performance' | 'heatmap';

// ── Theme-aware getVar ────────────────────────────────────────────────────────
function useGetVar() {
  const [, forceUpdate] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => forceUpdate(n => n + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  return (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';
}

// ── Info Tooltip ──────────────────────────────────────────────────────────────
function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        style={{
          width: 16, height: 16,
          borderRadius: '50%',
          background: 'var(--bg4)',
          border: '1px solid var(--border2)',
          color: 'var(--dimmed)',
          fontSize: 9,
          fontWeight: 700,
          cursor: 'default',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          fontFamily: 'var(--font-mono)',
        }}
      >i</button>
      {open && (
        <div style={{
          position: 'absolute',
          bottom: 22,
          left: 0,
          background: 'var(--bg2)',
          border: '1px solid var(--border2)',
          borderRadius: 'var(--radius-md)',
          padding: '8px 12px',
          fontSize: 11,
          color: 'var(--muted)',
          boxShadow: 'var(--shadow-lg)',
          zIndex: 100,
          pointerEvents: 'none',
          fontFamily: 'var(--font-body)',
          lineHeight: 1.6,
          width: 240,
          whiteSpace: 'normal',
        }}>
          {text}
        </div>
      )}
    </div>
  );
}

// ── Custom Tooltip ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border2)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
      fontFamily: 'var(--font-mono)',
      boxShadow: 'var(--shadow-md)',
    }}>
      {label && <p style={{ color: 'var(--dimmed)', marginBottom: 6, fontSize: 11 }}>{label}</p>}
      {payload.map((p: any, i: number) => (
        <p key={i} style={{ color: p.color || 'var(--text)', margin: '2px 0' }}>
          {p.name}: <strong style={{ color: 'var(--text)' }}>{p.value}</strong>
        </p>
      ))}
    </div>
  );
};

// ── Card Title ────────────────────────────────────────────────────────────────
function CardTitle({ title, dot, info }: { title: string; dot: string; info?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '0 0 1rem' }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: dot, flexShrink: 0 }} />
      <h3 style={{ margin: 0, fontSize: 13, fontWeight: 700, color: 'var(--text)', fontFamily: 'var(--font-display)', letterSpacing: '-0.01em' }}>
        {title}
      </h3>
      {info && <InfoTooltip text={info} />}
    </div>
  );
}

// ── Empty State ───────────────────────────────────────────────────────────────
function EmptyState({ msg }: { msg: string }) {
  return (
    <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--dimmed)', fontSize: 13, fontFamily: 'var(--font-body)' }}>
      {msg}
    </div>
  );
}

// ── Product Button ────────────────────────────────────────────────────────────
function ProductBtn({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} style={{
      padding: '8px 14px',
      borderRadius: 8,
      background: active ? 'var(--purple)' : 'var(--bg4)',
      color: active ? 'var(--text-on-accent)' : 'var(--muted)',
      border: `1px solid ${active ? 'var(--purple)' : 'var(--border)'}`,
      cursor: 'pointer',
      fontSize: 12,
      fontFamily: 'var(--font-body)',
      fontWeight: 500,
      transition: 'all var(--transition-fast)',
    }}>
      {children}
    </button>
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

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function AnalyticsPage() {
  const { user } = useAuth();
  const { data, isLoading, error } = useAnalyticsData();
  const [tab, setTab] = useState<Tab>('overview');
  const [selectedProduct, setSelectedProduct] = useState<number | null>(null);
  const { data: productData, isLoading: productLoading } = useProductAnalytics(selectedProduct);
  const getVar = useGetVar();

  const SEV_COLORS = {
    critical: getVar('--severity-critical'),
    high:     getVar('--severity-high'),
    medium:   getVar('--severity-medium'),
    low:      getVar('--severity-low'),
    purple:   getVar('--purple'),
    dimmed:   getVar('--dimmed'),
    muted:    getVar('--muted'),
    text:     getVar('--text'),
  };

  if (isLoading) return (
    <div style={pageStyle}>
      <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--dimmed)', fontFamily: 'var(--font-body)' }}>
        <div style={{ fontSize: 13, marginBottom: 12, fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>
          CHARGEMENT
        </div>
        <div>Calcul des statistiques en cours...</div>
      </div>
    </div>
  );

  if (error || !data) return (
    <div style={pageStyle}>
      <div style={{ textAlign: 'center', padding: '4rem', color: 'var(--severity-critical)', fontFamily: 'var(--font-body)' }}>
        Erreur lors du chargement des analytics
      </div>
    </div>
  );

  const { summary, severity_distribution, top_products, heatmap, top_vuln_types, timeline, funnel, mttr } = data;

  const ROLE_LABELS: Record<string, string> = {
    admin:    'Admin',
    analyst:  'Analyst',
    manager:  'Manager',
    default:  'Developer',
  };

  return (
    <div style={pageStyle}>

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{
              fontSize: 10, color: 'var(--purple)',
              letterSpacing: '0.2em', textTransform: 'uppercase',
              marginBottom: 6, fontFamily: 'var(--font-mono)',
            }}>
              ◆ ANALYTICS
            </div>
            <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, letterSpacing: '-0.03em', color: 'var(--text-strong)', fontFamily: 'var(--font-display)' }}>
              Analyse de sécurité
            </h1>
            <p style={{ margin: '6px 0 0', color: 'var(--muted)', fontSize: 13, fontFamily: 'var(--font-body)' }}>
              {data.filtered
                ? `Données filtrées — ${summary.total_products} produit(s) assigné(s)`
                : `Vue globale — ${summary.total_products} produits analysés`}
            </p>
          </div>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8,
            padding: '8px 16px',
            background: 'var(--glass-purple)',
            border: '1px solid var(--severity-info-border)',
            borderRadius: 10,
          }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--purple)' }} />
            <span style={{ fontSize: 12, color: 'var(--purple)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {ROLE_LABELS[user?.role ?? 'default'] ?? 'Developer'}
            </span>
          </div>
        </div>
      </div>

      {/* ── Alerte urgente ─────────────────────────────────────────────────── */}
      {summary.urgent_ratio > 15 && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '12px 16px',
          background: 'var(--glass-danger)',
          border: '1px solid var(--border-danger)',
          borderLeft: '3px solid var(--severity-critical)',
          borderRadius: 10, marginBottom: '1.5rem',
        }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--severity-critical)', flexShrink: 0 }} />
          <div>
            <strong style={{ color: 'var(--severity-critical)', fontSize: 13 }}>Action immédiate requise</strong>
            <span style={{ color: 'var(--muted)', fontSize: 13, marginLeft: 8 }}>
              {summary.urgent_count} findings critiques/high ({summary.urgent_ratio}% du total)
            </span>
          </div>
        </div>
      )}

      {/* ── KPI Cards ──────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 10, marginBottom: '2rem' }}>
        {[
          { label: 'Total',        value: summary.total_findings,    color: 'var(--purple)',            icon: '◈' },
          { label: 'Critical',     value: summary.total_critical,    color: 'var(--severity-critical)', icon: '▲' },
          { label: 'High',         value: summary.total_high,        color: 'var(--severity-high)',     icon: '▲' },
          { label: 'Medium',       value: summary.total_medium,      color: 'var(--severity-medium)',   icon: '◆' },
          { label: 'Low',          value: summary.total_low,         color: 'var(--severity-low)',      icon: '◆' },
          {
            label: 'Score Risque',
            value: summary.global_risk_score,
            color: summary.global_risk_score > 70
              ? 'var(--severity-critical)'
              : summary.global_risk_score > 40
              ? 'var(--severity-medium)'
              : 'var(--severity-low)',
            icon: '◎', suffix: '/100',
          },
          { label: 'CVSS Moyen', value: summary.avg_cvss, color: 'var(--purple)', icon: '◉', decimal: true },
        ].map((c, i) => (
          <div key={i} style={{
            background: 'var(--bg3)',
            border: '1px solid var(--border)',
            borderTop: `2px solid ${c.color}`,
            borderRadius: 12, padding: '16px 12px', textAlign: 'center',
            transition: 'all var(--transition-fast)',
          }}
            onMouseEnter={e => (e.currentTarget.style.boxShadow = 'var(--shadow-md)')}
            onMouseLeave={e => (e.currentTarget.style.boxShadow = 'none')}
          >
            <div style={{ fontSize: 12, color: c.color, marginBottom: 10, fontFamily: 'var(--font-mono)', opacity: 0.8 }}>
              {c.icon}
            </div>
            <div style={{ fontSize: 22, fontWeight: 800, color: c.color, fontFamily: 'var(--font-display)', letterSpacing: '-0.02em', lineHeight: 1, marginBottom: 8 }}>
              {c.decimal ? (c.value as number).toFixed(1) : Math.round(c.value as number).toLocaleString()}
              {(c as any).suffix || ''}
            </div>
            <div style={{ fontSize: 9, color: 'var(--dimmed)', textTransform: 'uppercase', letterSpacing: '0.10em', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
              {c.label}
            </div>
          </div>
        ))}
      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 4, background: 'var(--bg2)', border: '1px solid var(--border)', borderRadius: 12, padding: 4, marginBottom: '2rem' }}>
        {([
          { key: 'overview',    label: 'Vue globale'   },
          { key: 'trends',      label: 'Tendances'     },
          { key: 'products',    label: 'Par produit'   },
          { key: 'performance', label: 'Performance'   },
          { key: 'heatmap',     label: 'Heatmap Code'  },
        ] as const).map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} style={{
            flex: 1, padding: '10px 16px', borderRadius: 10,
            background: tab === t.key ? 'var(--purple)' : 'transparent',
            color: tab === t.key ? 'var(--text-on-accent)' : 'var(--dimmed)',
            border: 'none', cursor: 'pointer',
            fontSize: 12, fontWeight: 600,
            fontFamily: 'var(--font-display)',
            transition: 'all var(--transition-fast)',
            letterSpacing: '-0.01em',
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

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr 1fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle
                title="Score de risque global"
                dot="var(--severity-critical)"
                info="Score composite calculé à partir des findings pondérés par sévérité, exposition et âge."
              />
              <RiskGauge score={Math.round(summary.global_risk_score)} />
              <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--dimmed)', marginTop: 8, fontFamily: 'var(--font-mono)' }}>
                Basé sur {summary.total_findings.toLocaleString()} findings
              </div>
            </div>

            <div style={card}>
              <CardTitle
                title="Distribution des sévérités"
                dot="var(--purple)"
                info="Répartition des findings par niveau de sévérité CVSS."
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <ResponsiveContainer width="50%" height={220}>
                  <PieChart>
                    <Pie data={severity_distribution} dataKey="value" cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3}>
                      {severity_distribution.map((entry, i) => (
                        <Cell key={i} fill={entry.color} stroke="var(--bg2)" strokeWidth={2} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {severity_distribution.map((s, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: s.color }} />
                        <span style={{ fontSize: 12, color: 'var(--muted)', fontFamily: 'var(--font-body)' }}>{s.name}</span>
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 700, color: s.color, fontFamily: 'var(--font-mono)' }}>{s.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={card}>
              <CardTitle
                title="Entonnoir de risque"
                dot="var(--severity-medium)"
                info="Distribution des findings par niveau de risque, du plus critique au plus faible."
              />
              <FunnelChart data={funnel} />
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle
                title="Top produits — Score risque"
                dot="var(--severity-high)"
                info="Classement des produits selon leur score de risque composite (0–100)."
              />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <XAxis type="number"
                    tick={{ fill: SEV_COLORS.dimmed, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false} domain={[0, 100]} unit="%" />
                  <YAxis type="category" dataKey="name" width={100}
                    tick={{ fill: SEV_COLORS.muted, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 12) + '…' : v} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="risk_score" name="Score risque" fill={SEV_COLORS.purple} radius={[0, 6, 6, 0]} maxBarSize={18}
                    label={{ position: 'right', fill: SEV_COLORS.dimmed, fontSize: 10, formatter: (v: unknown) => `${Number(v)}` }} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={card}>
              <CardTitle
                title="Heatmap — Produits × Sévérité"
                dot="var(--purple)"
                info="Carte de chaleur croisant les produits et les niveaux de sévérité."
              />
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
            <CardTitle
              title="Évolution temporelle des findings"
              dot="var(--purple)"
              info="Nombre de findings par sévérité, regroupés par mois."
            />
            <p style={{ margin: '0 0 1rem', fontSize: 12, color: 'var(--dimmed)', fontFamily: 'var(--font-body)' }}>
              Tendance des findings par mois et par sévérité
            </p>
            {timeline.length > 0
              ? <TimelineChart data={timeline} />
              : <EmptyState msg="Pas assez de données temporelles disponibles" />
            }
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>

            <div style={card}>
              <CardTitle
                title="Top 10 types de vulnérabilités"
                dot="var(--severity-critical)"
                info="Les types de vulnérabilités les plus fréquents dans vos findings."
              />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_vuln_types} layout="vertical" margin={{ left: 10, right: 30 }}>
                  <XAxis type="number"
                    tick={{ fill: SEV_COLORS.dimmed, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={90}
                    tick={{ fill: SEV_COLORS.muted, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" name="Findings" fill={SEV_COLORS.critical} radius={[0, 6, 6, 0]} maxBarSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={card}>
              <CardTitle
                title="Findings par sévérité / produit"
                dot="var(--severity-medium)"
                info="Stacked bar montrant la répartition des findings par sévérité pour chaque produit."
              />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 6)} margin={{ left: 0, right: 16 }}>
                  <XAxis dataKey="name"
                    tick={{ fill: SEV_COLORS.dimmed, fontSize: 10, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 8 ? v.slice(0, 8) + '…' : v} />
                  <YAxis tick={{ fill: SEV_COLORS.dimmed, fontSize: 11, fontFamily: getVar('--font-mono') }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend iconType="square" iconSize={8} wrapperStyle={{ fontSize: 11, color: SEV_COLORS.dimmed, fontFamily: getVar('--font-mono') }} />
                  <Bar dataKey="critical" stackId="a" fill={SEV_COLORS.critical} name="Critical" maxBarSize={40} />
                  <Bar dataKey="high"     stackId="a" fill={SEV_COLORS.high}     name="High"     maxBarSize={40} />
                  <Bar dataKey="medium"   stackId="a" fill={SEV_COLORS.medium}   name="Medium"   maxBarSize={40} />
                  <Bar dataKey="low"      stackId="a" fill={SEV_COLORS.low}      name="Low"      maxBarSize={40} radius={[4, 4, 0, 0]} />
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

          <div style={card}>
            <CardTitle
              title="Sélectionner un produit"
              dot="var(--purple)"
              info="Cliquez sur un produit pour afficher ses statistiques détaillées."
            />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              {data.by_product.map(p => (
                <ProductBtn
                  key={p.id}
                  active={selectedProduct === p.id}
                  onClick={() => setSelectedProduct(selectedProduct === p.id ? null : p.id)}
                >
                  {p.name.length > 20 ? p.name.slice(0, 20) + '…' : p.name}
                  {p.critical > 0 && (
                    <span style={{ marginLeft: 6, color: 'var(--severity-critical)', fontWeight: 700 }}>●</span>
                  )}
                </ProductBtn>
              ))}
            </div>
          </div>

          {selectedProduct && productLoading && (
            <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--dimmed)', fontFamily: 'var(--font-body)' }}>
              Chargement des stats du produit...
            </div>
          )}

          {selectedProduct && productData && !productLoading && (
            <>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: 10 }}>
                {[
                  { label: 'Total',    value: productData.total_findings, color: 'var(--purple)'            },
                  { label: 'Critical', value: productData.critical,       color: 'var(--severity-critical)' },
                  { label: 'High',     value: productData.high,           color: 'var(--severity-high)'     },
                  { label: 'Medium',   value: productData.medium,         color: 'var(--severity-medium)'   },
                  { label: 'Low',      value: productData.low,            color: 'var(--severity-low)'      },
                  {
                    label: 'Score', suffix: '/100',
                    value: productData.risk_score,
                    color: productData.risk_score > 70 ? 'var(--severity-critical)' : 'var(--severity-medium)',
                  },
                ].map((k, i) => (
                  <div key={i} style={{
                    background: 'var(--bg3)',
                    border: '1px solid var(--border)',
                    borderTop: `2px solid ${k.color}`,
                    borderRadius: 10, padding: '14px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 20, fontWeight: 800, color: k.color, fontFamily: 'var(--font-display)' }}>
                      {k.value}{(k as any).suffix || ''}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--dimmed)', marginTop: 6, fontFamily: 'var(--font-mono)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
                      {k.label}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.5fr', gap: '1.5rem' }}>
                <div style={card}>
                  <CardTitle title="Profil de risque" dot="var(--purple)" info="Radar des dimensions de risque pour ce produit." />
                  <RadarRiskChart data={productData.radar} />
                </div>
                <div style={card}>
                  <CardTitle title="Évolution du produit" dot="var(--purple)" info="Évolution des findings dans le temps pour ce produit." />
                  {productData.timeline.length > 0
                    ? <TimelineChart data={productData.timeline} />
                    : <EmptyState msg="Pas de données temporelles pour ce produit" />
                  }
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                <div style={card}>
                  <CardTitle title="Types de vulnérabilités" dot="var(--severity-critical)" info="Les types de vulnérabilités les plus fréquents pour ce produit." />
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={productData.top_vuln_types} layout="vertical" margin={{ left: 10, right: 30 }}>
                      <XAxis type="number" tick={{ fill: SEV_COLORS.dimmed, fontSize: 11, fontFamily: getVar('--font-mono') }} axisLine={false} tickLine={false} />
                      <YAxis type="category" dataKey="name" width={80} tick={{ fill: SEV_COLORS.muted, fontSize: 11, fontFamily: getVar('--font-mono') }} axisLine={false} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="value" name="Findings" fill={SEV_COLORS.critical} radius={[0, 6, 6, 0]} maxBarSize={18} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div style={card}>
                  <CardTitle title="Répartition par sévérité" dot="var(--severity-low)" info="Proportion de chaque sévérité par rapport au total du produit." />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 8 }}>
                    {[
                      { sev: 'Critical', count: productData.critical, color: SEV_COLORS.critical },
                      { sev: 'High',     count: productData.high,     color: SEV_COLORS.high     },
                      { sev: 'Medium',   count: productData.medium,   color: SEV_COLORS.medium   },
                      { sev: 'Low',      count: productData.low,      color: SEV_COLORS.low      },
                    ].map(({ sev, count, color }) => {
                      const pct = productData.total_findings > 0 ? Math.round((count / productData.total_findings) * 100) : 0;
                      return (
                        <div key={sev}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 12 }}>
                            <span style={{ color: 'var(--muted)', fontFamily: 'var(--font-body)' }}>{sev}</span>
                            <span style={{ color, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{count} ({pct}%)</span>
                          </div>
                          <div style={{ background: 'var(--bg4)', borderRadius: 4, height: 6 }}>
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
            <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--dimmed)', fontSize: 13, fontFamily: 'var(--font-body)' }}>
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
              <CardTitle
                title="MTTR — Temps moyen de résolution"
                dot="var(--purple)"
                info="Mean Time To Remediate : âge moyen des findings non résolus par sévérité."
              />
              <p style={{ margin: '0 0 1rem', fontSize: 12, color: 'var(--dimmed)', fontFamily: 'var(--font-body)' }}>
                Basé sur l'âge moyen des findings par sévérité
              </p>
              <MttrChart mttr={mttr} />
            </div>

            <div style={card}>
              <CardTitle
                title="CVSS Moyen par produit"
                dot="var(--purple)"
                info="Score CVSS moyen de tous les findings pour chaque produit."
              />
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={top_products.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 40 }}>
                  <XAxis type="number" domain={[0, 10]}
                    tick={{ fill: SEV_COLORS.dimmed, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="name" width={100}
                    tick={{ fill: SEV_COLORS.muted, fontSize: 11, fontFamily: getVar('--font-mono') }}
                    axisLine={false} tickLine={false}
                    tickFormatter={(v: string) => v.length > 12 ? v.slice(0, 12) + '…' : v} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avg_cvss" name="CVSS Moyen" fill={SEV_COLORS.purple} radius={[0, 6, 6, 0]} maxBarSize={18}
                    label={{ position: 'right', fill: SEV_COLORS.dimmed, fontSize: 10, formatter: (v: unknown) => `${Number(v).toFixed(1)}` }} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
            {[
              {
                title: 'Âge moyen des findings',
                value: `${summary.avg_age_days} jours`,
                subtitle: 'Depuis la date de création',
                color: 'var(--severity-medium)',
                icon: '◷',
              },
              {
                title: 'Ratio urgent',
                value: `${summary.urgent_ratio}%`,
                subtitle: 'Critical + High / Total',
                color: summary.urgent_ratio > 15 ? 'var(--severity-critical)' : 'var(--severity-low)',
                icon: '◎',
              },
              {
                title: 'Score CVSS moyen',
                value: summary.avg_cvss.toFixed(1),
                subtitle: 'Tous produits confondus',
                color: summary.avg_cvss > 7 ? 'var(--severity-critical)' : summary.avg_cvss > 4 ? 'var(--severity-medium)' : 'var(--severity-low)',
                icon: '◈',
              },
            ].map((s, i) => (
              <div key={i} style={{ ...card, textAlign: 'center' }}>
                <div style={{ fontSize: 18, color: s.color, marginBottom: 10, fontFamily: 'var(--font-mono)' }}>{s.icon}</div>
                <div style={{ fontSize: 30, fontWeight: 800, color: s.color, fontFamily: 'var(--font-display)', letterSpacing: '-0.03em' }}>{s.value}</div>
                <div style={{ fontSize: 13, fontWeight: 700, marginTop: 10, color: 'var(--text)', fontFamily: 'var(--font-display)' }}>{s.title}</div>
                <div style={{ fontSize: 11, color: 'var(--dimmed)', marginTop: 4, fontFamily: 'var(--font-body)' }}>{s.subtitle}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════════════════ */}
      {/* TAB: HEATMAP                                                       */}
      {/* ══════════════════════════════════════════════════════════════════ */}
      {tab === 'heatmap' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          <div style={card}>
            <CardTitle
              title="Filtrer par produit"
              dot="var(--purple)"
              info="Sélectionnez un produit pour filtrer la heatmap, ou laissez sur Tous."
            />
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              <ProductBtn active={selectedProduct === null} onClick={() => setSelectedProduct(null)}>
                Tous les produits
              </ProductBtn>
              {data.by_product.map(p => (
                <ProductBtn
                  key={p.id}
                  active={selectedProduct === p.id}
                  onClick={() => setSelectedProduct(selectedProduct === p.id ? null : p.id)}
                >
                  {p.name.length > 20 ? p.name.slice(0, 20) + '…' : p.name}
                  {p.critical > 0 && (
                    <span style={{ marginLeft: 6, color: 'var(--severity-critical)', fontWeight: 700 }}>●</span>
                  )}
                </ProductBtn>
              ))}
            </div>
          </div>

          <div style={card}>
            <CardTitle
              title="Heatmap du code source"
              dot="var(--severity-critical)"
              info="Visualisez les fichiers les plus vulnérables de votre codebase par nombre de findings."
            />
            <p style={{ margin: '0 0 1.5rem', fontSize: 12, color: 'var(--dimmed)', fontFamily: 'var(--font-body)' }}>
              Visualisez les fichiers les plus vulnérables de votre codebase
              {selectedProduct && (
                <span style={{ color: 'var(--purple)', marginLeft: 6, fontWeight: 600 }}>
                  — {data.by_product.find(p => p.id === selectedProduct)?.name}
                </span>
              )}
            </p>
            <CodeHeatmap
              productName={selectedProduct
                ? data.by_product.find(p => p.id === selectedProduct)?.name
                : undefined
              }
            />
          </div>
        </div>
      )}
    </div>
  );
}