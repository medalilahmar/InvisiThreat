export interface AuthState {
  isAuthenticated: boolean;
  user: AuthUser | null;
  loading: boolean;
  error: string | null;
}

export interface AuthUser {
  username: string;
  role: 'admin' | 'viewer';
}

export interface LoginCredentials {
  username: string;
  password: string;
}