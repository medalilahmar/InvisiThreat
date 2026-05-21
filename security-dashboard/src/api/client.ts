import axios from 'axios';
import { getToken } from '../auth/services/authService';

const BASE_URL = import.meta.env.VITE_API_URL || '';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

export const llmClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 80000,
});

// ── Intercepteur requête : injecte le token ──────────────────────────────
apiClient.interceptors.request.use(
  (config) => {
    const token = getToken();
    console.log('🔑 Token injecté :', token?.slice(0, 20) || 'AUCUN');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Intercepteur réponse : log uniquement, PAS de suppression du token ───
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('❌ API Error:', error.response?.status, error.response?.data || error.message);
    // ← ON NE SUPPRIME PLUS LE TOKEN ICI
    return Promise.reject(error);
  }
);

// ── Intercepteurs LLM ────────────────────────────────────────────────────
llmClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

llmClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[LLM] Erreur:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);