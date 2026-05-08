export type UserRole   = 'admin' | 'manager' | 'analyst' | 'developer';
export type UserStatus = 'pending' | 'active' | 'blocked';

export interface AuthUser {
  id: number;
  username: string;
  email: string;
  role: UserRole;
  status: UserStatus;
  projects: { id: number; name: string }[];
  created_at?: string; // ← AJOUT
}

export interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: AuthUser;
}