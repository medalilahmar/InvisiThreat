import { useState, useEffect } from 'react';
import { HeatmapEntry } from '../hooks/useAnalyticsData';

const SEV_COLS = ['critical', 'high', 'medium', 'low', 'info'] as const;

const SEV_VARS: Record<string, string> = {
  critical: '--severity-critical',
  high:     '--severity-high',
  medium:   '--severity-medium',
  low:      '--severity-low',
  info:     '--severity-info',
};

const SEV_LABELS: Record<string, string> = {
  critical: 'Critique',
  high:     'Élevé',
  medium:   'Moyen',
  low:      'Faible',
  info:     'Info',
};

const INFO_TEXT = 'Carte de chaleur croisant les produits et les niveaux de sévérité. Plus la cellule est intense, plus le nombre de findings est élevé pour ce niveau de risque.';

function useThemeVar() {
  const [, forceUpdate] = useState(0);
  useEffect(() => {
    const obs = new MutationObserver(() => forceUpdate(n => n + 1));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => obs.disconnect();
  }, []);
  return (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';
}

function cellBg(value: number, max: number, sev: string, getVar: (v: string) => string): string {
  if (value === 0) return 'var(--bg4)';
  const intensity = Math.max(0.12, value / max);
  const base = getVar(SEV_VARS[sev] ?? '--muted');
  return `color-mix(in srgb, ${base} ${Math.round(intensity * 55)}%, var(--bg2))`;
}

function InfoTooltip({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        aria-label="Informations"
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
          bottom: 26,
          left: 0,
          background: 'var(--bg2)',
          border: '1px solid var(--border2)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 14px',
          fontSize: 11,
          color: 'var(--muted)',
          boxShadow: 'var(--shadow-lg)',
          zIndex: 100,
          pointerEvents: 'none',
          fontFamily: 'var(--font-body)',
          lineHeight: 1.6,
          width: 280,
          whiteSpace: 'normal',
        }}>
          {text}
        </div>
      )}
    </div>
  );
}

export function HeatmapChart({ data }: { data: HeatmapEntry[] }) {
  const getVar = useThemeVar();
  const maxVal = Math.max(...data.flatMap(d => SEV_COLS.map(s => d[s] || 0)), 1);

  return (
    <div style={{
      background: 'var(--bg3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      padding: '20px 16px',
    }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 18 }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: 13,
          color: 'var(--text)',
        }}>
        </span>
        <InfoTooltip text={INFO_TEXT} />
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'separate',
          borderSpacing: '0 4px',
          fontFamily: 'var(--font-mono)',
          fontSize: 12,
        }}>
          <thead>
            <tr>
              <th style={{
                textAlign: 'left',
                padding: '6px 12px 12px',
                color: 'var(--dimmed)',
                fontWeight: 600,
                fontSize: 10,
                letterSpacing: '0.10em',
                textTransform: 'uppercase',
                borderBottom: '1px solid var(--border2)',
                whiteSpace: 'nowrap',
              }}>
                Produit
              </th>
              {SEV_COLS.map(s => (
                <th key={s} style={{
                  padding: '6px 12px 12px',
                  textAlign: 'center',
                  fontWeight: 700,
                  fontSize: 10,
                  letterSpacing: '0.10em',
                  textTransform: 'uppercase',
                  color: `var(${SEV_VARS[s]})`,
                  borderBottom: '1px solid var(--border2)',
                  whiteSpace: 'nowrap',
                }}>
                  {SEV_LABELS[s]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                <td style={{
                  padding: '10px 12px',
                  color: 'var(--text)',
                  fontWeight: 600,
                  fontSize: 12,
                  whiteSpace: 'nowrap',
                  maxWidth: 160,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  borderBottom: '1px solid var(--border)',
                }}>
                  {row.product}
                </td>
                {SEV_COLS.map(sev => {
                  const val = row[sev] || 0;
                  return (
                    <td key={sev} style={{
                      padding: '10px 12px',
                      textAlign: 'center',
                      background: cellBg(val, maxVal, sev, getVar),
                      color: val > 0 ? `var(${SEV_VARS[sev]})` : 'var(--border2)',
                      fontWeight: val > 0 ? 700 : 400,
                      fontSize: val > 0 ? 13 : 11,
                      borderRadius: 8,
                      borderBottom: '1px solid var(--border)',
                      transition: 'background var(--transition-fast)',
                      minWidth: 60,
                    }}>
                      {val > 0 ? val : '—'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div style={{
        display: 'flex',
        gap: 16,
        marginTop: 16,
        paddingTop: 12,
        borderTop: '1px solid var(--border)',
        flexWrap: 'wrap',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: 10, color: 'var(--dimmed)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Légende
        </span>
        {SEV_COLS.map(s => (
          <div key={s} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 8, height: 8,
              borderRadius: '50%',
              background: `var(${SEV_VARS[s]})`,
              flexShrink: 0,
            }} />
            <span style={{ fontSize: 10, color: 'var(--muted)', letterSpacing: '0.05em' }}>
              {SEV_LABELS[s]}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}