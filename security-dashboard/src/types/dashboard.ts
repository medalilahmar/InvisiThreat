export interface ProductSummary {
  id: number;
  name: string;
  totalFindings: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  info: number;
  avgRiskScore: number;
}

export interface DashboardStats {
  totalFindings: number;
  totalProducts: number;
  totalCritical: number;
  totalHigh: number;
  avgConfidence?: number;
  riskDistribution: { name: string; value: number }[];
  trendData?: { date: string; critical: number; high: number; medium: number }[];
}