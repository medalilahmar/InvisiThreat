export type UserRole   = 'admin' | 'manager' | 'analyst' | 'developer';
export type UserStatus = 'pending' | 'active' | 'blocked';

export interface AuthUser {
  id: number;
  username: string;
  email: string;
  role: UserRole;
  status: UserStatus;
  projects: { id: number; name: string }[];
  created_at?: string;
  last_login?: string | null;
  locked_until?: string | null;
  failed_login_attempts?: number;
  job_title?: string | null;
  department?: string | null;
  phone?: string | null;
  avatar_url?: string | null;
  is_active?: boolean;
  updated_at?: string;
  password_changed_at?: string;
  github_username?: string | null;
  jira_email?: string | null;
  notify_on_new_finding?: boolean;
  notify_on_pr_merged?: boolean;
}

export interface AuthState {
  user:            AuthUser | null;
  token:           string | null;
  isAuthenticated: boolean;
  isLoading:       boolean;
  
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  job_title?: string;
  department?: string;
  phone?: string;
  avatar_url?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type:   string;
  user:         AuthUser;
}

export type LoginErrorCode =
  | 'INVALID_CREDENTIALS'
  | 'ACCOUNT_LOCKED'
  | 'ACCOUNT_PENDING'
  | 'ACCOUNT_BLOCKED'
  | 'UNKNOWN';

export interface LoginError {
  code:         LoginErrorCode;
  message:      string;
  minutesLeft?: number;
}