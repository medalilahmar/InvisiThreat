import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useState, useEffect } from 'react';

const LINES = [
  { key: 'critical', name: 'Critical', color: '--severity-critical' },
  { key: 'high',     name: 'High',     color: '--severity-high'     },
  { key: 'medium',   name: 'Medium',   color: '--severity-medium'   },
  { key: 'low',      name: 'Low',      color: '--severity-low'      },
];

function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        style={{
          width: 18, height: 18,
          borderRadius: '50%',
          background: 'var(--bg4)',
          border: '1px solid var(--border2)',
          color: 'var(--dimmed)',
          fontSize: 10,
          fontWeight: 700,
          cursor: 'default',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
        }}
      >i</button>
      {open && (
        <div style={{
          position: 'absolute',
          bottom: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'var(--bg2)',
          border: '1px solid var(--border2)',
          borderRadius: 'var(--radius-md)',
          padding: '8px 12px',
          fontSize: 11,
          color: 'var(--muted)',
          whiteSpace: 'nowrap',
          boxShadow: 'var(--shadow-md)',
          zIndex: 50,
          pointerEvents: 'none',
          fontFamily: 'var(--font-body)',
          lineHeight: 1.5,
        }}>
          {text}
        </div>
      )}
    </div>
  );
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border2)',
      borderRadius: 'var(--radius-md)',
      padding: '10px 14px',
      fontSize: 12,
      fontFamily: 'var(--font-mono)',
      boxShadow: 'var(--shadow-md)',
      minWidth: 140,
    }}>
      <div style={{ color: 'var(--dimmed)', marginBottom: 8, fontSize: 11, letterSpacing: '0.05em' }}>
        {label}
      </div>
      {payload.map((p: any) => (
        <div key={p.dataKey} style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 4 }}>
          <span style={{ color: p.stroke, fontWeight: 600 }}>{p.name}</span>
          <span style={{ color: 'var(--text)', fontWeight: 700 }}>{p.value}</span>
        </div>
      ))}
    </div>
  );
}

export function TimelineChart({ data }: {
  data: { month: string; critical: number; high: number; medium: number; low: number }[]
}) {
  const formatted = data.map(d => ({ ...d, month: d.month.slice(0, 7) }));

  const [theme, setTheme] = useState(
    document.documentElement.getAttribute('data-theme') ?? 'dark'
  );

  useEffect(() => {
    const obs = new MutationObserver(() =>
      setTheme(document.documentElement.getAttribute('data-theme') ?? 'dark')
    );
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);

  const getVar = (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';

  const gridColor = getVar('--border');
  const tickColor = getVar('--dimmed');

  return (
    <div style={{
      background: 'var(--bg3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: '20px 16px 12px',
    }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, paddingLeft: 4 }}>
        <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 13, color: 'var(--text)' }}>
          Évolution des findings
        </span>
        <InfoTooltip text="Nombre de findings par sévérité, regroupés par mois." />
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={formatted} margin={{ left: -8, right: 12, top: 4, bottom: 4 }}>

          <CartesianGrid stroke={gridColor} strokeOpacity={0.3} vertical={false} />

          <XAxis
            dataKey="month"
            tick={{ fill: tickColor, fontSize: 11, fontFamily: getVar('--font-mono') }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: tickColor, fontSize: 11, fontFamily: getVar('--font-mono') }}
            axisLine={false}
            tickLine={false}
            allowDecimals={false}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            iconType="circle"
            iconSize={7}
            wrapperStyle={{
              fontSize: 11,
              color: tickColor,
              fontFamily: getVar('--font-mono'),
              paddingTop: 12,
            }}
          />

          {LINES.map(({ key, name, color }) => {
            const resolvedColor = getVar(color);
            return (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                name={name}
                stroke={resolvedColor}
                strokeWidth={2}
                dot={{ r: 3, fill: resolvedColor }}
                activeDot={{ r: 5 }}
              />
            );
          })}

        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}