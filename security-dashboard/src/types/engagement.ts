export interface Engagement {
  id: number;
  name: string;
  product: number;
  product_name?: string;
  status: string;
  created?: string;
  findings_count?: number;
}