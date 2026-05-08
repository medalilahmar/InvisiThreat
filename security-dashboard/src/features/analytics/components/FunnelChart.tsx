export function FunnelChart({ data }: { data: { name: string; value: number }[] }) {
  const max = data[0]?.value || 1;
  const colors = ['#6366f1', '#ff4757', '#ffd32a', '#2ed573'];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: '1rem 0' }}>
      {data.map((item, i) => {
        const pct = (item.value / max) * 100;
        return (
          <div key={i}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4, fontSize: 12 }}>
              <span style={{ color: 'rgba(255,255,255,0.6)' }}>{item.name}</span>
              <span style={{ color: colors[i], fontWeight: 700 }}>{item.value.toLocaleString()}</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.06)', borderRadius: 6, height: 28, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%',
                background: colors[i],
                borderRadius: 6,
                transition: 'width 0.8s ease',
                opacity: 0.85,
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}