import type { ModelMetrics as IModelMetrics } from '../../../types/model';

interface ModelMetricsProps {
  metrics: IModelMetrics | null;
}

const CLASS_COLORS: string[] = [
  'var(--severity-none)',
  'var(--severity-low)',
  'var(--severity-medium)',
  'var(--severity-high)',
  'var(--severity-critical)',
];

const performanceIconPath = {
  f1_weighted: 'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  f1_macro:    'M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z',
  roc_auc:     'M22 12h-4l-3 9L9 3l-3 9H2',
  cv_f1:       'M20 6L9 17l-5-5',
};

export function ModelMetrics({ metrics }: ModelMetricsProps) {
  if (!metrics) return null;

  const performanceMetrics = [
    {
      label:   'F1-Score (Weighted)',
      value:   metrics.metrics.test_f1_weighted,
      max:     1,
      color:   'var(--accent)',
      iconKey: 'f1_weighted' as const,
    },
    {
      label:   'F1-Score (Macro)',
      value:   metrics.metrics.test_f1_macro,
      max:     1,
      color:   'var(--severity-info)',
      iconKey: 'f1_macro' as const,
    },
    {
      label:   'ROC-AUC (OvR)',
      value:   metrics.metrics.test_roc_auc_ovr,
      max:     1,
      color:   'var(--severity-high)',
      iconKey: 'roc_auc' as const,
    },
    {
      label:   'CV F1 (Mean)',
      value:   metrics.metrics.cv_f1_weighted_mean,
      max:     1,
      color:   'var(--severity-low)',
      iconKey: 'cv_f1' as const,
    },
  ];

  return (
    <section className="model-metrics">
      <div className="section-label">
        <span className="fp-label-dot" />
        PERFORMANCE METRICS
      </div>
      <h2 className="section-title">
        Métriques d'<span>Évaluation</span>
      </h2>

      <div className="metrics-grid">
        {performanceMetrics.map((metric) => (
          <div key={metric.label} className="metric-card">
            <div className="metric-header">
              <div className="metric-icon-wrap" style={{ color: metric.color }}>
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d={performanceIconPath[metric.iconKey]} />
                </svg>
              </div>
              <h3 className="metric-label">{metric.label}</h3>
            </div>

            <div className="metric-value" style={{ color: metric.color }}>
              {(metric.value * 100).toFixed(1)}%
            </div>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{
                  width:      `${(metric.value / metric.max) * 100}%`,
                  background: metric.color,
                }}
              />
            </div>

            <div className="metric-detail">
              {metric.label === 'CV F1 (Mean)' && (
                <span className="metric-info">
                  ±{(metrics.metrics.cv_f1_weighted_std * 100).toFixed(2)}%
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="f1-per-class">
        <h3>F1-Score par Classe de Risque</h3>
        <div className="class-scores">
          {Object.entries(metrics.metrics.f1_per_class).map(([className, score], idx) => {
            const color = CLASS_COLORS[idx] ?? 'var(--severity-none)';
            return (
              <div key={className} className="class-score">
                <div className="class-header">
                  <span
                    className="class-icon"
                    style={{ background: color, color: 'var(--text-on-accent)' }}
                  >
                    {className[0].toUpperCase()}
                  </span>
                  <span className="class-name">{className}</span>
                </div>
                <div className="score-bar">
                  <div
                    className="score-fill"
                    style={{ width: `${score * 100}%`, background: color }}
                  />
                </div>
                <span className="score-value" style={{ color }}>
                  {(score * 100).toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}