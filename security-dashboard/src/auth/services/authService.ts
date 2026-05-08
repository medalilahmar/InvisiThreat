import axios from 'axios';
import { LoginRequest, RegisterRequest, AuthUser, TokenResponse } from '../types/auth.types';

const BASE_URL = 'http://localhost:8081';

// ─── LocalStorage ─────────────────────────────────────────────────────────────

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

// ─── Auth headers ─────────────────────────────────────────────────────────────

const authHeaders = () => ({
  Authorization: `Bearer ${getToken()}`,
});

// ─── Endpoints ────────────────────────────────────────────────────────────────

export const login = async (data: LoginRequest): Promise<TokenResponse> => {
  const params = new URLSearchParams();
  params.append('username', data.username);
  params.append('password', data.password);

  const res = await axios.post(`${BASE_URL}/auth/login`, params, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });

  const result: TokenResponse = res.data;
  saveToken(result.access_token);
  saveUser(result.user);
  return result;
};

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

export const changePassword = async (current_password: string, new_password: string) => {
  const res = await axios.put(
    `${BASE_URL}/auth/change-password`,
    { current_password, new_password },
    { headers: authHeaders() }
  );
  return res.data;
};

// ← AJOUT : mise à jour du profil (email et/ou username)
export const updateProfile = async (data: {
  username?: string;
  email?: string;
}): Promise<AuthUser> => {
  const res = await axios.put(
    `${BASE_URL}/auth/me`,
    data,
    { headers: authHeaders() }
  );
  // On met à jour le user sauvegardé immédiatement
  saveUser(res.data);
  return res.data;
};