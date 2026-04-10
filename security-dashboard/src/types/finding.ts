export interface Finding {
  id: number;
  title: string;
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  cvss_score: number;
  tags: string[];
  engagement_id: number;
  product_id?: number;
  created?: string;
  risk_class?: number;
  risk_level?: string;
}