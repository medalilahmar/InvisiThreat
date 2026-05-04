export interface Finding {
  id: number;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  cvss_score: number;
  tags: string[];
  engagement_id: number | null;
  engagement_name?: string | null;
  product_id?: number | null;
  product_name?: string | null;
  created?: string;
  age_days?: number | null;
  file_path?: string | null;
  line?: number | null;
  has_cve?: number | null;
  description?: string | null;
  cve?: string | null;
  cwe?: string | null;

  risk_class?: number | null;
  risk_level?: string | null;
  ai_confidence?: number | null;
  context_score?: number | null;

  ai_risk_class?: number | null;
  ai_risk_level?: string | null;


  ai_risk_score_cont?: number | null;
  model_base_score?:   number | null;
  business_nudge?:     number | null;
  shap_features?:      Array<{
    feature:    string;
    value:      number;
    shap_value: number;
    direction:  string;
  }> | null;
}