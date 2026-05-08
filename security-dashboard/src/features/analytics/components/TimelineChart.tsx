import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export function TimelineChart({ data }: { data: { month: string; critical: number; high: number; medium: number; low: number }[] }) {
  const formatted = data.map(d => ({
    ...d,
    month: d.month.slice(0, 7),
  }));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={formatted} margin={{ left: 0, right: 16, top: 8, bottom: 8 }}>
        <CartesianGrid stroke="rgba(255,255,255,0.04)" vertical={false} />
        <XAxis dataKey="month" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <YAxis tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11, color: 'rgba(255,255,255,0.4)' }} />
        <Line type="monotone" dataKey="critical" stroke="#ff4757" strokeWidth={2.5} dot={{ r: 3 }} name="Critical" />
        <Line type="monotone" dataKey="high"     stroke="#ff6b35" strokeWidth={2.5} dot={{ r: 3 }} name="High" />
        <Line type="monotone" dataKey="medium"   stroke="#ffd32a" strokeWidth={2}   dot={{ r: 3 }} name="Medium" />
        <Line type="monotone" dataKey="low"      stroke="#2ed573" strokeWidth={2}   dot={{ r: 3 }} name="Low" />
      </LineChart>
    </ResponsiveContainer>
  );
}