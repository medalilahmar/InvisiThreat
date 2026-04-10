import { apiClient } from '../client';
import type { Engagement } from '../../types/engagement';

export const engagementsApi = {
  getByProduct: (productId: number) =>
    apiClient.get<Engagement[]>(`/defectdojo/engagements?product_id=${productId}`),
};