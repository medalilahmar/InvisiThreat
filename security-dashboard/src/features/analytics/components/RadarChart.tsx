import { Radar, RadarChart as ReRadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts';

export function RadarRiskChart({ data }: { data: { axis: string; value: number }[] }) {
  const getVar = (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';

  const accentColor = getVar('--purple');
  const gridColor   = getVar('--border');
  const tickColor   = getVar('--dimmed');

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ReRadarChart data={data}>
        <PolarGrid stroke={gridColor} strokeOpacity={0.4} />
        <PolarAngleAxis
          dataKey="axis"
          tick={{ fill: tickColor, fontSize: 11, fontFamily: 'var(--font-mono)' }}
        />
        <Radar
          name="Risque"
          dataKey="value"
          stroke={accentColor}
          fill={accentColor}
          fillOpacity={0.18}
          strokeWidth={2}
        />
        <Tooltip
          contentStyle={{
            background: getVar('--bg2'),
            border: `1px solid ${getVar('--border2')}`,
            borderRadius: 8,
            fontSize: 12,
            fontFamily: getVar('--font-mono'),
            color: getVar('--text'),
          }}
          formatter={(value) => [`${Number(value)}/100`, 'Score']}
        />
      </ReRadarChart>
    </ResponsiveContainer>
  );
}