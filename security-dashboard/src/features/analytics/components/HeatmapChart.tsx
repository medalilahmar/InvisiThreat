import { HeatmapEntry } from '../hooks/useAnalyticsData';

const SEV_COLS = ['critical', 'high', 'medium', 'low', 'info'] as const;
const SEV_COLORS: Record<string, string> = {
  critical: '#ff4757', high: '#ff6b35', medium: '#ffd32a', low: '#2ed573', info: '#95a5a6',
};

function cellColor(value: number, max: number, sev: string): string {
  if (value === 0) return 'rgba(255,255,255,0.03)';
  const intensity = Math.max(0.15, value / max);
  const base = SEV_COLORS[sev] || '#fff';
  return base + Math.round(intensity * 255).toString(16).padStart(2, '0');
}

export function HeatmapChart({ data }: { data: HeatmapEntry[] }) {
  const maxVal = Math.max(...data.flatMap(d => SEV_COLS.map(s => d[s] || 0)), 1);

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 3 }}>
        <thead>
          <tr>
            <th style={thStyle}>Produit</th>
            {SEV_COLS.map(s => (
              <th key={s} style={{ ...thStyle, color: SEV_COLORS[s] }}>
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <td style={{ ...tdStyle, color: 'rgba(255,255,255,0.7)', fontSize: 12, maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {row.product}
              </td>
              {SEV_COLS.map(sev => (
                <td key={sev} style={{
                  ...tdStyle,
                  background: cellColor(row[sev] || 0, maxVal, sev),
                  color: (row[sev] || 0) > 0 ? '#fff' : 'rgba(255,255,255,0.2)',
                  fontWeight: 600,
                  textAlign: 'center',
                  borderRadius: 6,
                }}>
                  {row[sev] || 0}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const thStyle: React.CSSProperties = { padding: '6px 10px', fontSize: 11, color: 'rgba(255,255,255,0.4)', textAlign: 'center', fontWeight: 500, textTransform: 'uppercase', letterSpacing: 1 };
const tdStyle: React.CSSProperties = { padding: '8px 12px', fontSize: 13 };