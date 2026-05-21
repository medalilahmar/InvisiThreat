import axios from 'axios';
import {
  LoginRequest, RegisterRequest,
  AuthUser, TokenResponse, LoginError
} from '../types/auth.types';

const BASE_URL = import.meta.env.VITE_API_URL || '';
// ─── LocalStorage (inchangé) ──────────────────────────────────────────────────

export const saveToken    = (t: string)   => localStorage.setItem('token', t);
export const getToken     = ()            => localStorage.getItem('token');
export const removeToken  = ()            => localStorage.removeItem('token');
export const saveUser     = (u: AuthUser) => localStorage.setItem('user', JSON.stringify(u));
export const getSavedUser = (): AuthUser | null => {
  try { return JSON.parse(localStorage.getItem('user') || 'null'); }
  catch { return null; }
};
export const clearStorage = () => {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
};

// ─── Auth headers (inchangé) ──────────────────────────────────────────────────

const authHeaders = () => ({
  Authorization: `Bearer ${getToken()}`,
});

// ─── Helper : extraire les minutes depuis le message backend ──────────────────

const extractMinutes = (detail: string): number | undefined => {
  const match = detail.match(/(\d+)\s*minute/i);
  return match ? parseInt(match[1], 10) : undefined;
};

// ─── Login avec gestion structurée des erreurs 423/403 ───────────────────────

export const login = async (data: LoginRequest): Promise<TokenResponse> => {
  try {
    const params = new URLSearchParams();
    params.append('username', data.username);
    params.append('password', data.password);

    const res = await axios.post(`${BASE_URL}/auth/login`, params, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    const result: TokenResponse = res.data;
    saveToken(result.access_token);  // ← localStorage, inchangé
    saveUser(result.user);           // ← localStorage, inchangé
    return result;

  } catch (err: unknown) {
    if (!axios.isAxiosError(err)) {
      throw { code: 'UNKNOWN', message: 'Erreur réseau inattendue' } as LoginError;
    }

    const status = err.response?.status;
    const detail = err.response?.data?.detail ?? '';

    if (status === 423) {
      throw {
        code: 'ACCOUNT_LOCKED',
        message: detail || 'Compte verrouillé. Réessayez plus tard.',
        minutesLeft: extractMinutes(detail),
      } as LoginError;
    }

    if (status === 403) {
      const isPending = detail.toLowerCase().includes('attente');
      throw {
        code: isPending ? 'ACCOUNT_PENDING' : 'ACCOUNT_BLOCKED',
        message: detail,
      } as LoginError;
    }

    if (status === 400) {
      throw {
        code: 'INVALID_CREDENTIALS',
        message: detail || 'Nom d\'utilisateur ou mot de passe incorrect',
      } as LoginError;
    }

    throw {
      code: 'UNKNOWN',
      message: detail || 'Une erreur est survenue. Veuillez réessayer.',
    } as LoginError;
  }
};

// ─── Autres endpoints (inchangés) ─────────────────────────────────────────────

export const register = async (data: RegisterRequest) => {
  const res = await axios.post(`${BASE_URL}/auth/register`, data);
  return res.data;
};

export const getMe = async (): Promise<AuthUser> => {
  const res = await axios.get(`${BASE_URL}/auth/me`, {
    headers: authHeaders(),
  });
  return res.data;
};

export const logout = async () => {
  try {
    await axios.post(`${BASE_URL}/auth/logout`, {}, {
      headers: authHeaders(),
    });
  } catch { /* silencieux */ }
  clearStorage();
};

export const changePassword = async (
  current_password: string,
  new_password: string
) => {
  const res = await axios.put(
    `${BASE_URL}/auth/change-password`,
    { current_password, new_password },
    { headers: authHeaders() }
  );
  return res.data;
};

export const updateProfile = async (data: {
  username?: string;
  email?: string;
  
  job_title?: string | null;
  department?: string | null;
  phone?: string | null;
  avatar_url?: string | null;
  
  github_username?: string | null;
  github_token?: string | null;
  jira_email?: string | null;
  jira_token?: string | null;
  
  notify_on_new_finding?: boolean;
  notify_on_pr_merged?: boolean;
}): Promise<AuthUser> => {
  const res = await axios.put(
    `${BASE_URL}/auth/me`,
    data,
    { headers: authHeaders() }
  );
  saveUser(res.data);
  return res.data;
};