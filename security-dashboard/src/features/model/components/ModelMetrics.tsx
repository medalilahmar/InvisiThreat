
import type { ModelMetrics as IModelMetrics } from '../../../types/model';

interface ModelMetricsProps {
  metrics: IModelMetrics | null;
}

export function ModelMetrics({ metrics }: ModelMetricsProps) {
  if (!metrics) return null;

  const performanceMetrics = [
    {
      label: 'F1-Score (Weighted)',
      value: metrics.metrics.test_f1_weighted,
      max: 1,
      color: '#00d4ff',
      icon: '🎯',
    },
    {
      label: 'F1-Score (Macro)',
      value: metrics.metrics.test_f1_macro,
      max: 1,
      color: '#a29bfe',
      icon: '⚖️',
    },
    {
      label: 'ROC-AUC (OvR)',
      value: metrics.metrics.test_roc_auc_ovr,
      max: 1,
      color: '#ff6b35',
      icon: '📈',
    },
    {
      label: 'CV F1 (Mean)',
      value: metrics.metrics.cv_f1_weighted_mean,
      max: 1,
      color: '#2ed573',
      icon: '✓',
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
        {performanceMetrics.map((metric, idx) => (
          <div key={idx} className="metric-card">
            <div className="metric-header">
              <span className="metric-icon">{metric.icon}</span>
              <h3 className="metric-label">{metric.label}</h3>
            </div>
            <div className="metric-value" style={{ color: metric.color }}>
              {(metric.value * 100).toFixed(1)}%
            </div>
            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{
                  width: `${(metric.value / metric.max) * 100}%`,
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
        <h3>📊 F1-Score par Classe de Risque</h3>
        <div className="class-scores">
          {Object.entries(metrics.metrics.f1_per_class).map(([className, score], idx) => {
            const colors = ['#95a5a6', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c'];
            return (
              <div key={className} className="class-score">
                <div className="class-header">
                  <span className="class-icon" style={{ background: colors[idx] }}>
                    {className[0]}
                  </span>
                  <span className="class-name">{className}</span>
                </div>
                <div className="score-bar">
                  <div
                    className="score-fill"
                    style={{
                      width: `${score * 100}%`,
                      background: colors[idx],
                    }}
                  />
                </div>
                <span className="score-value">{(score * 100).toFixed(1)}%</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}