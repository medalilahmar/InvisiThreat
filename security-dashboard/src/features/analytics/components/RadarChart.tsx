import { Radar, RadarChart as ReRadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';

export function RadarRiskChart({ data }: { data: { axis: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ReRadarChart data={data}>
        <PolarGrid stroke="rgba(255,255,255,0.08)" />
        <PolarAngleAxis dataKey="axis" tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 11 }} />
        <Radar
          name="Risque"
          dataKey="value"
          stroke="#6366f1"
          fill="#6366f1"
          fillOpacity={0.3}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{ background: '#13131a', border: '0.5px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
          formatter={(value) => [`${Number(value)}/100`, 'Score']} // ← fix
        />
      </ReRadarChart>
    </ResponsiveContainer>
  );
}