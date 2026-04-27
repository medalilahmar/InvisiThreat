export interface JiraCreateIssueRequest {
  finding_id: number;
  title: string;
  severity: string;
  cvss_score: number;
  description?: string;
  cve?: string;
  cwe?: string;
  file_path?: string;
  line?: number;
  tags?: string[];
  risk_level?: string;
  ai_score?: number;
  ai_confidence?: number;
  engagement_id?: number | null;
  product_name?: string | null;
}

export interface JiraIssueResponse {
  key: string;
  id: string;
  self: string;
  url?: string;
}

export interface JiraHealthResponse {
  status: string;
  jira_server?: string;
  project_key?: string;
  connected: boolean;
}

export interface JiraErrorResponse {
  detail?: string;
  message?: string;
  error?: string;
  status?: number;
}

export interface JiraIntegrationState {
  loading: boolean;
  error: string | null;
  success: boolean;
  jiraKey: string | null;
  jiraUrl: string | null;
}

export interface JiraActionButtonProps {
  loading: boolean;
  success: boolean;
  error: string | null;
  jiraUrl: string | null;
  jiraKey: string | null;
  onClick: () => void;
}

export interface JiraResultPanelProps {
  loading: boolean;
  error: string | null;
  success: boolean;
  jiraUrl: string | null;
  jiraKey: string | null;
  onRetry: () => void;
}

export type JiraErrorCode = 
  | 'NOT_FOUND'
  | 'CONFLICT'
  | 'SERVER_ERROR'
  | 'SERVICE_UNAVAILABLE'
  | 'GATEWAY_TIMEOUT'
  | 'NETWORK_ERROR'
  | 'UNKNOWN';

export interface JiraError {
  code: JiraErrorCode;
  message: string;
  status?: number;
  details?: string;
}