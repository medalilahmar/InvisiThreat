// frontend/src/features/model/components/FeatureExplorer.tsx
import { useState, useMemo } from 'react'; // ← Supprimé React
import { FEATURES_METADATA } from '../../../types/model';
import type { ModelMetrics as IModelMetrics } from '../../../types/model';

interface FeatureExplorerProps {
  metrics: IModelMetrics | null;
}

const impactColors = {
  Critique: '#e74c3c',
  Élevé: '#f39c12',
  Moyen: '#3498db',
};

const categoryIcons = {
  'Sévérité': '📊',
  'Contexte': '🌍',
  'CVE/CWE': '🔐',
  'Tags': '🏷️',
  'Exploit': '⚡',
  'Interaction': '🔗',
  'Historique': '📜',
};

export function FeatureExplorer({ metrics }: FeatureExplorerProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('Tous');
  const [searchTerm, setSearchTerm] = useState('');

  if (!metrics) return null;

  const categories = [
    'Tous',
    ...Array.from(
      new Set(Object.values(FEATURES_METADATA).map((f) => f.category))
    ).sort(),
  ];

  const filteredFeatures = useMemo(() => {
    return Object.entries(FEATURES_METADATA)
      .filter(([key]) => metrics.feature_columns.includes(key))
      .filter(
        ([, feature]) =>
          selectedCategory === 'Tous' || feature.category === selectedCategory
      )
      .filter(
        ([key, feature]) =>
          feature.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
          feature.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
          key.toLowerCase().includes(searchTerm.toLowerCase())
      )
      .sort(([, a], [, b]) => {
        const impactOrder = { Critique: 0, Élevé: 1, Moyen: 2 };
        return (impactOrder[a.impact] ?? 3) - (impactOrder[b.impact] ?? 3);
      });
  }, [selectedCategory, searchTerm, metrics.feature_columns]);

  return (
    <section className="feature-explorer">
      <div className="section-label">
        <span className="fp-label-dot" />
        FEATURES
      </div>
      <h2 className="section-title">
        Les {metrics.n_features} <span>Caractéristiques</span>
      </h2>

      <div className="explorer-controls">
        <input
          type="text"
          placeholder="🔍 Chercher une feature..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="search-input"
        />

        <div className="category-filter">
          {categories.map((cat) => (
            <button
              key={cat}
              className={`category-btn ${selectedCategory === cat ? 'active' : ''}`}
              onClick={() => setSelectedCategory(cat)}
            >
              {cat === 'Tous' ? '📋' : categoryIcons[cat as keyof typeof categoryIcons]}
              {cat}
            </button>
          ))}
        </div>
      </div>

      <div className="features-grid">
        {filteredFeatures.map(([key, feature]) => (
          <div key={key} className="feature-card">
            <div className="feature-top">
              <div className="feature-icons">
                <span className="feature-category-icon">
                  {categoryIcons[feature.category as keyof typeof categoryIcons]}
                </span>
              </div>
              <span
                className="impact-badge"
                style={{ background: impactColors[feature.impact] }}
              >
                {feature.impact}
              </span>
            </div>
            <h4 className="feature-name">{feature.name}</h4>
            <p className="feature-description">{feature.description}</p>
            <div className="feature-meta">
              <span className="meta-tag category">{feature.category}</span>
              <span className="meta-tag dimension">{feature.dimension}</span>
              <code className="feature-code">{key}</code>
            </div>
          </div>
        ))}
      </div>

      <div className="feature-stats">
        <div className="stat-box">
          <div className="stat-value">{metrics.n_features}</div>
          <div className="stat-label">Features Total</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">
            {Object.values(FEATURES_METADATA).filter((f) => f.impact === 'Critique').length}
          </div>
          <div className="stat-label">Critiques</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">{categories.length - 1}</div>
          <div className="stat-label">Catégories</div>
        </div>
        <div className="stat-box">
          <div className="stat-value">4</div>
          <div className="stat-label">Interactions</div>
        </div>
      </div>
    </section>
  );
}