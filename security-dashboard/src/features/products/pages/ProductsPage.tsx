import { useProducts } from '../hooks/useProducts';
import { Link } from 'react-router-dom';
import './ProductsPage.css';



export default function ProductsPage() {
  const { data: products, isLoading, error } = useProducts();
  

  // 🔥 Dynamic Risk Assessment System
  const getRiskLevel = (findings: number) => {
    if (findings > 50) return { level: 'CRITICAL', color: '#ff4757', icon: '🔴', severity: 'critical' };
    if (findings > 30) return { level: 'HIGH', color: '#ff6b35', icon: '🟠', severity: 'high' };
    if (findings > 15) return { level: 'MEDIUM', color: '#ffa502', icon: '🟡', severity: 'medium' };
    if (findings > 5) return { level: 'LOW', color: '#2ed573', icon: '🟢', severity: 'low' };
    return { level: 'SECURE', color: '#00d4ff', icon: '✅', severity: 'secure' };
  };

  const getSeverityWidth = (findings: number) => Math.min((findings / 100) * 100, 100);

  return (
    <div className="home-root">
      {/* Background Layers */}
      <div className="bg-grid"></div>
      <div className="bg-radials"></div>
      <div className="scan-line"></div>

      <section className="section products-section">
        <div className="section-inner">
          {/* Section Header */}
          <div className="section-header fu">
            <div className="section-label">
              <span style={{ marginRight: '8px' }}>◆</span>
              PORTEFEUILLE CYBER
            </div>

            <h2 className="section-title">
              Analyse <span>des assets</span>
            </h2>

            {!isLoading && !error && (
              <p className="section-subtitle">
                Visualisez l'état de sécurité de vos produits, analysez les vulnérabilités et maitrisez vos risques
              </p>
            )}

            {isLoading && (
              <p className="section-subtitle">
                <span style={{ opacity: 0.6 }}>⟳ Chargement des assets...</span>
              </p>
            )}

            {error && (
              <p className="section-subtitle" style={{ color: 'var(--accent2)' }}>
                ⚠️ {error.message}
              </p>
            )}

            {/* Quick Stats */}
            {!isLoading && !error && products && products.length > 0 && (
              <div className="quick-stats fu1">
                <div className="quick-stat-item">
                  <span className="quick-stat-value">{products.length}</span>
                  <span className="quick-stat-label">Assets</span>
                </div>
                <div className="quick-stat-divider"></div>
                <div className="quick-stat-item">
                  <span className="quick-stat-value">
                    {products.reduce((sum, p) => sum + (p.findings_count ?? 0), 0)}
                  </span>
                  <span className="quick-stat-label">Findings</span>
                </div>
                <div className="quick-stat-divider"></div>
                <div className="quick-stat-item">
                  <span className="quick-stat-value">
                    {Math.round(
                      (products.filter(p => (p.findings_count ?? 0) <= 10).length / products.length) * 100
                    )}%
                  </span>
                  <span className="quick-stat-label">Sécurisés</span>
                </div>
              </div>
            )}
          </div>

          {/* Products Grid */}
          <div className="features-grid">
            {/* Loading State */}
            {isLoading &&
              [...Array(6)].map((_, i) => (
                <div key={i} className={`feature-card skeleton fu${i % 7}`}>
                  <div
                    className="feature-card-icon skeleton-pulse"
                    style={{
                      background: 'linear-gradient(135deg, rgba(0,212,255,0.15), rgba(0,212,255,0.05))',
                      border: '1px solid rgba(0,212,255,0.25)',
                    }}
                  >
                    📦
                  </div>

                  <div className="skeleton-line" style={{ height: '20px', marginBottom: '12px' }}></div>
                  <div className="skeleton-line" style={{ height: '14px', marginBottom: '8px', width: '90%' }}></div>
                  <div className="skeleton-line" style={{ height: '14px', marginBottom: '8px', width: '95%' }}></div>

                  <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    <div className="skeleton-line" style={{ height: '12px', width: '70%' }}></div>
                    <div className="skeleton-line" style={{ height: '12px', width: '65%' }}></div>
                  </div>

                  <div className="severity-bar-skeleton" style={{ marginTop: '14px' }}></div>
                </div>
              ))}

            {/* Error State */}
            {error && !isLoading && (
              <div className="error-card fu">
                <div className="error-icon-large">⚠️</div>
                <h3 className="error-card-title">Erreur de chargement</h3>
                <p className="error-card-desc">{error.message}</p>
                <button
                  className="btn-error-retry"
                  onClick={() => window.location.reload()}
                >
                  ↻ Réessayer
                </button>
              </div>
            )}

            {/* Empty State */}
            {!isLoading && !error && (!products || products.length === 0) && (
              <div className="empty-card fu">
                <div className="empty-icon">📭</div>
                <h3 className="empty-title">Aucun asset</h3>
                <p className="empty-desc">
                  Commencez par créer votre premier produit pour démarrer l'analyse de sécurité
                </p>
              </div>
            )}

            {/* Products */}
            {!isLoading &&
              !error &&
              products &&
              products.map((product, index) => {
                const risk = getRiskLevel(product.findings_count ?? 0);
                const severity = getSeverityWidth(product.findings_count ?? 0);

                return (
                  <Link
                    key={product.id}
                    to={`/engagements?productId=${product.id}`}
                    className={`feature-card product-card fu${index % 7}`}
                    style={{
                      textDecoration: 'none',
                      color: 'inherit',
                      '--card-color': risk.color,
                      '--risk-severity': `${severity}%`
                    } as React.CSSProperties}
                  >
                    {/* Card Background Effects */}
                    <div className="card-effects">
                      <div className="card-glow"></div>
                      <div className="card-shine"></div>
                    </div>

                    {/* Risk Badge */}
                    <div className="product-risk-badge" style={{ color: risk.color, borderColor: `${risk.color}40` }}>
                      <span>{risk.icon}</span>
                      <span className="risk-label">{risk.level}</span>
                    </div>

                    {/* Icon */}
                    <div
                      className="feature-card-icon"
                      style={{
                        background: `linear-gradient(135deg, ${risk.color}15, ${risk.color}05)`,
                        border: `1.5px solid ${risk.color}40`,
                        boxShadow: `0 8px 24px ${risk.color}15`
                      }}
                    >
                      📦
                    </div>

                    {/* Name */}
                    <div className="feature-card-title">
                      {product.name}
                    </div>

                    {/* Description */}
                    <div className="feature-card-desc">
                      {product.description || 'Produit de sécurité professionnel'}
                    </div>

                    {/* Severity Bar */}
                    <div className="severity-indicator">
                      <div className="severity-header">
                        <span className="severity-label-small">Sévérité</span>
                        <span className="severity-percent" style={{ color: risk.color }}>
                          {severity.toFixed(0)}%
                        </span>
                      </div>
                      <div className="severity-bar-track">
                        <div
                          className="severity-bar-fill"
                          style={{
                            width: `${severity}%`,
                            background: `linear-gradient(90deg, ${risk.color}, ${risk.color}dd)`,
                            boxShadow: `0 0 16px ${risk.color}60`
                          }}
                        ></div>
                      </div>
                    </div>

                    {/* Meta Information */}
                    <div className="product-meta">
                      <div className="meta-item">
                        <span className="meta-emoji">📅</span>
                        <span className="meta-text">
                          {product.created
                            ? new Date(product.created).toLocaleDateString('fr-FR', {
                                year: '2-digit',
                                month: 'short',
                                day: 'numeric'
                              })
                            : 'N/A'}
                        </span>
                      </div>
                      <div className="meta-item">
                        <span className="meta-emoji">🛡️</span>
                        <span className="meta-text" style={{ color: risk.color, fontWeight: 600 }}>
                          {product.findings_count ?? 0} findings
                        </span>
                      </div>
                    </div>

                    {/* Hover Arrow */}
                    <div className="card-arrow">→</div>
                  </Link>
                );
              })}
          </div>
        </div>
      </section>
    </div>
  );
}