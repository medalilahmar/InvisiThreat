import { useQuery } from '@tanstack/react-query';
import { productsApi } from '../../../api/services/products';
import { findingsApi } from '../../../api/services/findings';
import type { Finding } from '../../../types/finding';
import type { ProductSummary, DashboardStats } from '../../../types/dashboard';

export const useDashboardData = () => {
  const productsQuery = useQuery({
    queryKey: ['products'],
    queryFn: () => productsApi.getAll().then(res => res.data),
  });

  const findingsQuery = useQuery({
    queryKey: ['allFindings'],
    queryFn: () => findingsApi.getAll().then(res => res.data as Finding[]),
  });

  const isLoading = productsQuery.isLoading || findingsQuery.isLoading;
  const error = productsQuery.error || findingsQuery.error;

  const productSummaries: ProductSummary[] = [];
  let totalFindings = 0;
  let totalCritical = 0;
  let totalHigh = 0;
  const riskCounts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };

  if (productsQuery.data && findingsQuery.data) {
    const products = productsQuery.data;
    const findings = findingsQuery.data;

    const findingsByProduct = new Map<number, Finding[]>();
    findings.forEach((f: Finding) => {
      const pid = f.product_id;
      if (pid) {
        if (!findingsByProduct.has(pid)) findingsByProduct.set(pid, []);
        findingsByProduct.get(pid)!.push(f);
      }
    });

    products.forEach(product => {
      const productFindings = findingsByProduct.get(product.id) || [];
      const counts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
      productFindings.forEach(f => {
        const sev = f.severity?.toLowerCase();
        if (sev === 'critical') counts.critical++;
        else if (sev === 'high') counts.high++;
        else if (sev === 'medium') counts.medium++;
        else if (sev === 'low') counts.low++;
        else counts.info++;
      });
      const total = productFindings.length;
      productSummaries.push({
        id: product.id,
        name: product.name,
        totalFindings: total,
        ...counts,
        avgRiskScore: productFindings.reduce((acc, f) => acc + (f.cvss_score || 0), 0) / (total || 1),
      });
      totalFindings += total;
      totalCritical += counts.critical;
      totalHigh += counts.high;
      riskCounts.critical += counts.critical;
      riskCounts.high += counts.high;
      riskCounts.medium += counts.medium;
      riskCounts.low += counts.low;
      riskCounts.info += counts.info;
    });

    productSummaries.sort((a, b) => b.totalFindings - a.totalFindings);
  }

  const riskDistribution = [
    { name: 'Critique', value: riskCounts.critical, color: '#ff4757' },
    { name: 'High', value: riskCounts.high, color: '#ff6b35' },
    { name: 'Medium', value: riskCounts.medium, color: '#ffd32a' },
    { name: 'Low', value: riskCounts.low, color: '#2ed573' },
    { name: 'Info', value: riskCounts.info, color: '#95a5a6' },
  ].filter(d => d.value > 0);

  const stats: DashboardStats = {
    totalFindings,
    totalProducts: productsQuery.data?.length || 0,
    totalCritical,
    totalHigh,
    riskDistribution,
  };

  return { productSummaries, stats, isLoading, error };
};