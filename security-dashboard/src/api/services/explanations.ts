import { apiClient, llmClient } from '../client';


export interface LLMExplanation {
  finding_id?: number;
  summary: string;
  impact: string;
  root_cause?: string;
  exploitation_difficulty?: string;
  priority_note: string;
  from_cache?: boolean;
  _fallback?: boolean;
}

export interface LLMRecommendation {
  finding_id?: number;
  title: string;
  recommendations: string[];
  references?: string[];
  verification?: string;
  prevention?: string;
  from_cache?: boolean;
  _fallback?: boolean;
}

export interface LLMExplanationRequest {
  finding_id?: number;
  title: string;
  severity: string;
  cvss_score?: number;
  description?: string;
  cve?: string;
  cwe?: string;
  file_path?: string;
  tags?: string[];
  risk_level?: string;
}

async function withRetry<T>(
  fn: () => Promise<T>,
  retries = 2,
  delayMs = 3000
): Promise<T> {
  try {
    return await fn();
  } catch (err: any) {
    const status = err?.response?.status;
    const isRetryable =
      !status ||
      status === 503 ||
      status === 502 ||
      err.code === 'ECONNABORTED';

    if (retries > 0 && isRetryable) {
      console.warn(`[LLM] Retry dans ${delayMs}ms... (${retries} restants)`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
      return withRetry(fn, retries - 1, delayMs * 1.5);
    }
    throw err;
  }
}

export const explanationsApi = {
  explain: (data: LLMExplanationRequest) =>
    withRetry(() => llmClient.post<LLMExplanation>('/explain/llm', data)),

  recommend: (data: LLMExplanationRequest) =>
    withRetry(() => llmClient.post<LLMRecommendation>('/recommend/llm', data)),

  health: () =>
    apiClient.get<{ status: string; models: string[]; current: string }>('/llm/health'),
};