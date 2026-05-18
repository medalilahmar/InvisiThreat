import { useState } from 'react';
import { HeatmapEntry } from '../hooks/useAnalyticsData';

const SEV_COLS = ['critical', 'high', 'medium', 'low', 'info'] as const;

/* Valeurs RGB extraites des couleurs globales — pour rgba() dynamique */
const SEV_RGB: Record<string, string> = {
  critical: '255, 86, 104',
  high:     '255, 133, 85',
  medium:   '255, 224, 102',
  low:      '62, 255, 160',
  info:     '148, 163, 184',
};

const SEV_SOLID: Record<string, string> = {
  critical: '#ff5668',
  high:     '#ff8555',
  medium:   '#ffe066',
  low:      '#3effa0',
  info:     '#94a3b8',
};

const SEV_LABELS: Record<string, string> = {
  critical: 'Critique',
  high:     'Élevé',
  medium:   'Moyen',
  low:      'Faible',
  info:     'Info',
};

const INFO_TEXT = 'Carte de chaleur croisant les produits et les niveaux de sévérité. Plus la cellule est intense, plus le nombre de findings est élevé pour ce niveau de risque.';

/* rgba() — fonctionne en dark ET light mode */
function cellBg(value: number, max: number, sev: string): string {
  if (value === 0) return 'rgba(128, 128, 128, 0.06)';
  const intensity = Math.max(0.18, value / max);
  return `rgba(${SEV_RGB[sev] ?? '255,255,255'}, ${intensity.toFixed(2)})`;
}

export function HeatmapChart({ data }: { data: HeatmapEntry[] }) {
  const maxVal = Math.max(...data.flatMap(d => SEV_COLS.map(s => d[s] || 0)), 1);
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="ac-chart-wrap">

      {/* ── Info icon ── */}
      <div className="ac-info-btn-wrap">
        <button
          className="ac-info-btn"
          onMouseEnter={() => setShowInfo(true)}
          onMouseLeave={() => setShowInfo(false)}
          aria-label="Informations sur le graphique"
        >
          ℹ
        </button>
        {showInfo && (
          <div className="ac-info-tooltip">{INFO_TEXT}</div>
        )}
      </div>

      {/* ── Table ── */}
      <div className="ac-heatmap-wrap">
        <table className="ac-heatmap-table">
          <thead>
            <tr>
              <th className="ac-heatmap-th ac-heatmap-th--left">Produit</th>
              {SEV_COLS.map(s => (
                <th key={s} className="ac-heatmap-th" style={{ color: SEV_SOLID[s] }}>
                  {SEV_LABELS[s]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, i) => (
              <tr key={i}>
                <td className="ac-heatmap-td ac-heatmap-product">
                  {row.product}
                </td>
                {SEV_COLS.map(sev => {
                  const val = row[sev] || 0;
                  return (
                    <td
                      key={sev}
                      className="ac-heatmap-td ac-heatmap-cell"
                      style={{
                        background: cellBg(val, maxVal, sev),
                        color: val > 0 ? 'var(--text)' : 'var(--dimmed)',
                      }}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

    </div>
  );
}