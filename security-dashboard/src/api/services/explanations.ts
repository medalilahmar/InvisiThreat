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


export interface LLMSolution {
  finding_id: number;
  vulnerable_snippet: string | null;
  fixed_snippet: string | null;
  explanation: string;
  confidence: number;
  file_path?: string | null;
  line?: number | null;
  has_file: boolean;
  from_cache?: boolean;
}

export interface AutoFixResponse {
  pr_url: string;
  pr_number: number;
  branch_name: string;
  status: string;
}

export interface AutoFixCapability {
  finding_id: number;
  can_autofix: boolean;
  reason: string;
  missing_fields: string[];
  requirements: {
    is_static: boolean;
    has_file_path: boolean;
    has_line: boolean;
    has_repo_url: boolean;
    has_github_token: boolean;
  };
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


export const getSolution = (findingId: number) =>
  withRetry(() => llmClient.get<LLMSolution>(`/defectdojo/findings/${findingId}/solution`));


export const checkAutofixCapability = (findingId: number) =>
  withRetry(() => apiClient.get<AutoFixCapability>(`/defectdojo/findings/${findingId}/can-autofix`));


export const applyAutofix = (findingId: number) =>
  withRetry(() => apiClient.post<AutoFixResponse>(`/defectdojo/findings/${findingId}/autofix`));