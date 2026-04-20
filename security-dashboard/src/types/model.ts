export interface ModelMetrics {
  model_version: string;
  n_classes: number;
  n_features: number;
  classes: number[];
  class_labels: Record<string, string>;
  feature_columns: string[];
  loaded_at: string;
  metrics: {
    cv_f1_weighted_mean: number;
    cv_f1_weighted_std: number;
    cv_f1_macro_mean: number;
    cv_f1_macro_std: number;
    test_f1_weighted: number;
    test_f1_macro: number;
    test_roc_auc_ovr: number;
    f1_per_class: Record<string, number>;
  };
}

export interface HealthStatus {
  status: string;
  model_ready: boolean;
  uptime_seconds: number;
  api_version: string;
}

export interface FeatureInfo {
  name: string;
  category: 'Sévérité' | 'Contexte' | 'CVE/CWE' | 'Tags' | 'Exploit' | 'Interaction' | 'Historique';
  description: string;
  impact: 'Critique' | 'Élevé' | 'Moyen';
  dimension: string;
}

export const FEATURES_METADATA: Record<string, FeatureInfo> = {
  cvss_score: {
    name: 'CVSS Score',
    category: 'Sévérité',
    description: 'Score de vulnérabilité brut (0-10)',
    impact: 'Critique',
    dimension: 'Score'
  },
  cvss_score_norm: {
    name: 'CVSS Normalisé',
    category: 'Sévérité',
    description: 'CVSS normalisé entre 0 et 1',
    impact: 'Critique',
    dimension: 'Score'
  },
  age_days: {
    name: 'Âge (jours)',
    category: 'Contexte',
    description: 'Jours depuis découverte de la vulnérabilité',
    impact: 'Élevé',
    dimension: 'Temps'
  },
  age_days_norm: {
    name: 'Âge Normalisé',
    category: 'Contexte',
    description: 'Âge normalisé (0-1)',
    impact: 'Moyen',
    dimension: 'Temps'
  },
  has_cve: {
    name: 'CVE Détecté',
    category: 'CVE/CWE',
    description: 'Vulnérabilité enregistrée en CVE (augmente fiabilité)',
    impact: 'Critique',
    dimension: 'Booléen'
  },
  has_cwe: {
    name: 'CWE Détecté',
    category: 'CVE/CWE',
    description: 'Faiblesse enregistrée en CWE',
    impact: 'Élevé',
    dimension: 'Booléen'
  },
  tags_count: {
    name: 'Nombre de Tags',
    category: 'Contexte',
    description: 'Nombre total de tags contextuels',
    impact: 'Moyen',
    dimension: 'Compteur'
  },
  tags_count_norm: {
    name: 'Tags Normalisés',
    category: 'Contexte',
    description: 'Tags normalisés (0-1)',
    impact: 'Moyen',
    dimension: 'Score'
  },
  tag_urgent: {
    name: 'Tag: Urgent',
    category: 'Tags',
    description: 'Marqué comme urgent ou blocker',
    impact: 'Élevé',
    dimension: 'Booléen'
  },
  tag_in_production: {
    name: 'Tag: Production',
    category: 'Tags',
    description: 'En environnement de production',
    impact: 'Critique',
    dimension: 'Booléen'
  },
  tag_sensitive: {
    name: 'Tag: Sensible',
    category: 'Tags',
    description: 'Système ou données sensibles (PII, GDPR)',
    impact: 'Critique',
    dimension: 'Booléen'
  },
  tag_external: {
    name: 'Tag: Exposé',
    category: 'Tags',
    description: 'Exposé à Internet (accessible publiquement)',
    impact: 'Critique',
    dimension: 'Booléen'
  },
  product_fp_rate: {
    name: 'Taux FP Produit',
    category: 'Historique',
    description: 'Taux de faux positifs historique du produit',
    impact: 'Moyen',
    dimension: 'Ratio'
  },
  cvss_x_has_cve: {
    name: 'CVSS × CVE',
    category: 'Interaction',
    description: 'Interaction entre sévérité CVSS et présence CVE',
    impact: 'Critique',
    dimension: 'Interaction'
  },
  age_x_cvss: {
    name: 'Âge × CVSS',
    category: 'Interaction',
    description: 'Interaction entre âge et sévérité (risque chronique)',
    impact: 'Élevé',
    dimension: 'Interaction'
  },
  epss_score: {
    name: 'EPSS Score',
    category: 'Exploit',
    description: 'Exploit Prediction Scoring System (0-1)',
    impact: 'Critique',
    dimension: 'Score'
  },
  epss_percentile: {
    name: 'EPSS Percentile',
    category: 'Exploit',
    description: 'Percentile EPSS (0-100) vs toutes les vulnérabilités',
    impact: 'Élevé',
    dimension: 'Percentile'
  },
  has_high_epss: {
    name: 'EPSS Élevé',
    category: 'Exploit',
    description: 'EPSS > 0.5 (très probablement exploité)',
    impact: 'Critique',
    dimension: 'Booléen'
  },
  epss_x_cvss: {
    name: 'EPSS × CVSS',
    category: 'Interaction',
    description: 'Interaction entre exploit et sévérité',
    impact: 'Critique',
    dimension: 'Interaction'
  },
  epss_score_norm: {
    name: 'EPSS Normalisé',
    category: 'Exploit',
    description: 'EPSS normalisé (0-1)',
    impact: 'Élevé',
    dimension: 'Score'
  },
  exploit_risk: {
    name: 'Risque Exploit',
    category: 'Exploit',
    description: 'Score composite de risque d\'exploitation',
    impact: 'Critique',
    dimension: 'Score'
  },
  context_score: {
    name: 'Score Contexte',
    category: 'Contexte',
    description: 'Score composite du contexte de sécurité',
    impact: 'Élevé',
    dimension: 'Score'
  },
  days_open_high: {
    name: 'Jours Ouvert (High)',
    category: 'Contexte',
    description: 'Nombre de jours à risque élevé ou plus',
    impact: 'Élevé',
    dimension: 'Temps'
  },
};