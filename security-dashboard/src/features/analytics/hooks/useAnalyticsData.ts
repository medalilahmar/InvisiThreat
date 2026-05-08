import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../../api/client';

export interface AnalyticsSummary {
  total_findings: number;
  total_products: number;
  total_critical: number;
  total_high: number;
  total_medium: number;
  total_low: number;
  total_info: number;
  avg_cvss: number;
  avg_age_days: number;
  global_risk_score: number;
  urgent_count: number;
  urgent_ratio: number;
}

export interface ProductStat {
  id: number;
  name: string;
  totalFindings: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  risk_score: number;
  avg_cvss: number;
}

export interface HeatmapEntry {
  product: string;
  critical: number;
  high: number;
  medium: number;
  low: number;
  info: number;
}

export interface TimelineEntry {
  month: string;
  critical: number;
  high: number;
  medium: number;
  low: number;
}

export interface AnalyticsData {
  summary: AnalyticsSummary;
  severity_distribution: { name: string; value: number; color: string }[];
  by_product: ProductStat[];
  top_products: ProductStat[];
  heatmap: HeatmapEntry[];
  top_vuln_types: { name: string; value: number }[];
  timeline: TimelineEntry[];
  funnel: { name: string; value: number }[];
  mttr: Record<string, number>;
  role: string;
  filtered: boolean;
}

export interface ProductAnalytics {
  id: number;
  name: string;
  total_findings: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  info: number;
  avg_cvss: number;
  avg_age_days: number;
  risk_score: number;
  mttr: Record<string, number>;
  top_vuln_types: { name: string; value: number }[];
  timeline: { month: string; critical: number; high: number; medium: number; low: number }[];
  radar: { axis: string; value: number }[];
  funnel: Record<string, number>;
}

export const useAnalyticsData = () => {
  return useQuery<AnalyticsData>({
    queryKey: ['analytics-stats'],
    queryFn: async () => {
      const res = await apiClient.get('/analytics/stats');
      return res.data;
    },
    staleTime: 5 * 60 * 1000, // 5 minutes — correspond au TTL backend
    retry: 2,
  });
};

export const useProductAnalytics = (productId: number | null) => {
  return useQuery<ProductAnalytics>({
    queryKey: ['analytics-product', productId],
    queryFn: async () => {
      const res = await apiClient.get(`/analytics/products/${productId}/stats`);
      return res.data;
    },
    enabled: productId !== null,
    staleTime: 5 * 60 * 1000,
    retry: 2,
  });
};