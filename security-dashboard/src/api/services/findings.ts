import { apiClient } from '../client';
import type { Finding } from '../../types/finding';

export const findingsApi = {
  getByEngagement: (engagementId: number, limit = 500) =>
    apiClient.get<Finding[]>(`/defectdojo/engagements/${engagementId}/findings?limit=${limit}`),
  getOne: (id: number) =>
    apiClient.get<Finding>(`/defectdojo/findings/${id}`),


  getAll: (limit = 10000) => apiClient.get(`/defectdojo/findings?limit=${limit}`),


  getTestsByEngagement: (engagementId: number) =>
    apiClient.get<{ id: number; title: string; test_type_name: string; findings_count: number }[]>(
      `/defectdojo/engagements/${engagementId}/tests`
    ),
  

};