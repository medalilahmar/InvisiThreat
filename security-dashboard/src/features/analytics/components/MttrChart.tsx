import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const COLORS: Record<string, string> = {
  critical: '#ff4757', high: '#ff6b35', medium: '#ffd32a', low: '#2ed573',
};

export function MttrChart({ mttr }: { mttr: Record<string, number> }) {
  const data = Object.entries(mttr).map(([sev, days]) => ({
    name: sev.charAt(0).toUpperCase() + sev.slice(1),
    days,
    color: COLORS[sev] || '#95a5a6',
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
        <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 12 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} axisLine={false} tickLine={false} unit="j" />
        <Tooltip
          contentStyle={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
          formatter={(value) => [`${Number(value).toFixed(1)} jours`, 'MTTR']} // ← fix
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