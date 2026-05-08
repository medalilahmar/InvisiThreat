import { RadialBarChart, RadialBar, ResponsiveContainer } from 'recharts';

export function RiskGauge({ score }: { score: number }) {
  const color = score >= 70 ? '#ff4757' : score >= 40 ? '#ffd32a' : '#2ed573';
  const label = score >= 70 ? 'CRITIQUE' : score >= 40 ? 'MODÉRÉ' : 'FAIBLE';

  const data = [
    { name: 'bg',    value: 100,   fill: 'rgba(255,255,255,0.06)' },
    { name: 'score', value: score, fill: color },
  ];

  return (
    <div style={{ position: 'relative', width: '100%', height: 200 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          cx="50%" cy="70%"
          innerRadius="60%" outerRadius="100%"
          startAngle={180} endAngle={0}
          data={data}
        >
          <RadialBar dataKey="value" cornerRadius={8} background={false} />
        </RadialBarChart>
      </ResponsiveContainer>
      <div style={{
        position: 'absolute', bottom: 20, left: '50%', transform: 'translateX(-50%)',
        textAlign: 'center',
      }}>
        <div style={{ fontSize: 36, fontWeight: 800, color, lineHeight: 1 }}>{score}</div>
        <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.4)', marginTop: 4, letterSpacing: 2 }}>{label}</div>
      </div>
    </div>
  );
}