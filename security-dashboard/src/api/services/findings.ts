import { apiClient } from '../client';
import type { Finding } from '../../types/finding';

export const findingsApi = {
  getByEngagement: (engagementId: number, limit = 500) =>
    apiClient.get<Finding[]>(`/defectdojo/engagements/${engagementId}/findings?limit=${limit}`),
  getOne: (id: number) =>
    apiClient.get<Finding>(`/defectdojo/findings/${id}`),


  getAll: (limit = 2000) => apiClient.get(`/defectdojo/findings?limit=${limit}`),

};