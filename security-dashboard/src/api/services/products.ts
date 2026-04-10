import { apiClient } from '../client';

export interface Product {
  id: number;
  name: string;
  description?: string;
  created?: string;
  findings_count?: number;
}

export const productsApi = {
  getAll: () => apiClient.get<Product[]>('/defectdojo/products'),
};