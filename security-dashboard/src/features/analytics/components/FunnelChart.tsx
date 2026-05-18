import { useState } from 'react';
import '../pages/AnalyticsPage.css';

const FUNNEL_COLORS = [
  'var(--purple)',
  'var(--accent2)',
  'var(--accent3)',
  'var(--green)',
];

const INFO_TEXT = 'Shows how findings are distributed across risk levels. Each bar is relative to the highest value — a shorter bar means fewer findings at that severity.';

export function FunnelChart({ data }: { data: { name: string; value: number }[] }) {
  const max = data[0]?.value || 1;
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="ac-chart-wrap">

      {/* ── Info icon ── */}
      <div className="ac-info-btn-wrap">
        <button
          className="ac-info-btn"
          onMouseEnter={() => setShowInfo(true)}
          onMouseLeave={() => setShowInfo(false)}
          aria-label="Chart information"
        >
          ℹ
        </button>
        {showInfo && (
          <div className="ac-info-tooltip">
            {INFO_TEXT}
          </div>
        )}
      </div>

      {/* ── Bars ── */}
      <div className="ac-funnel">
        {data.map((item, i) => {
          const pct = (item.value / max) * 100;
          return (
            <div key={i} className="ac-funnel-row">
              <div className="ac-funnel-header">
                <span className="ac-funnel-name">{item.name}</span>
                <span className="ac-funnel-value" style={{ color: FUNNEL_COLORS[i] }}>
                  {item.value.toLocaleString()}
                </span>
              </div>
              <div className="ac-funnel-track">
                <div
                  className="ac-funnel-fill"
                  style={{ width: `${pct}%`, background: FUNNEL_COLORS[i] }}
                />
              </div>
            </div>
          );
        })}
      </div>

    </div>
  );
}