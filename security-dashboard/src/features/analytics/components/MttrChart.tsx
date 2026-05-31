import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLOR_VARS: Record<string, string> = {
  critical: '--severity-critical',
  high:     '--severity-high',
  medium:   '--severity-medium',
  low:      '--severity-low',
};

export function MttrChart({ mttr }: { mttr: Record<string, number> }) {
  const getVar = (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';

  const data = Object.entries(mttr).map(([sev, days]) => ({
    name:  sev.charAt(0).toUpperCase() + sev.slice(1),
    days,
    color: getVar(COLOR_VARS[sev] || '--muted'),
  }));

  const tickColor = getVar('--dimmed');

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
        <XAxis
          dataKey="name"
          tick={{ fill: tickColor, fontSize: 12, fontFamily: getVar('--font-mono') }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: tickColor, fontSize: 11, fontFamily: getVar('--font-mono') }}
          axisLine={false}
          tickLine={false}
          unit="j"
        />
        <Tooltip
          contentStyle={{
            background:   getVar('--bg2'),
            border:       `1px solid ${getVar('--border2')}`,
            borderRadius: 8,
            fontSize:     12,
            fontFamily:   getVar('--font-mono'),
            color:        getVar('--text'),
          }}
          formatter={(value) => [`${Number(value).toFixed(1)} jours`, 'MTTR']}
        />
        <Bar dataKey="days" radius={[6, 6, 0, 0]} maxBarSize={60}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}