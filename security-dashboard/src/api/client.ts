import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8081';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000, 
});


export const llmClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 80000, 
});

apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

llmClient.interceptors.request.use(
  (config) => {
    console.debug(`[LLM] → ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

llmClient.interceptors.response.use(
  (response) => {
    console.debug(`[LLM] ← ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    if (error.code === 'ECONNABORTED') {
      console.error('[LLM] Timeout dépassé — le modèle est trop lent. Augmenter le timeout ou réduire num_predict.');
    } else {
      console.error('[LLM] Erreur:', error.response?.data || error.message);
    }
    return Promise.reject(error);
  }
);