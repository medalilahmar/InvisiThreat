import { apiClient } from '../client';

export interface PredictionRequest {
  severity: string;
  cvss_score: number;
  title: string;
  tags: string[];
  days_open?: number;
  finding_id?: number;
  engagement_id?: number;
}

export const predictionsApi = {
  predictBatch: (findings: PredictionRequest[]) =>
    apiClient.post<{ results: any[] }>('/predict/batch', { findings }),
};