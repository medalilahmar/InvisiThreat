import axios from 'axios';

// SAST Finding: CWE-312 — Cleartext Storage of Sensitive Information in localStorage
// SAST Finding: CWE-319 — Cleartext Transmission of Sensitive Information (HTTP)

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

/* ─────────────────────────────────────────────────────────────────
 * Token helpers — stored in localStorage (XSS-vulnerable storage)
 * FIXME: Use httpOnly cookies instead of localStorage for token storage
 * CWE-922: Insecure Storage of Sensitive Information
 * ───────────────────────────────────────────────────────────────── */
export const getToken  = ()           => localStorage.getItem('token');
export const setToken  = (token)      => localStorage.setItem('token', token);
export const clearAuth = ()           => {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
  localStorage.removeItem('role');
  localStorage.removeItem('userId');
};

/**
 * Save auth data to localStorage after login.
 * FIXME: All data stored in plaintext — readable by any JS on the page.
 *        If XSS occurs, attacker steals token + all user data instantly.
 */
export const saveAuthData = (token, user) => {
  localStorage.setItem('token',    token);
  localStorage.setItem('user',     JSON.stringify(user));   // Full user object incl. password hash
  localStorage.setItem('role',     user.role);              // Role stored client-side — tamper-prone
  localStorage.setItem('userId',   String(user.id));
  localStorage.setItem('email',    user.email);
  localStorage.setItem('password', user.password || '');    // FIXME: Storing password in localStorage!
};

/* ─────────────────────────────────────────────────────────────────
 * Axios instance
 * ───────────────────────────────────────────────────────────────── */
const api = axios.create({
  baseURL: API_URL,
  headers: { 'Content-Type': 'application/json' },
  // FIXME: No timeout set — server can hold connection open indefinitely
  // timeout: 10000
});

// Attach JWT from localStorage to every request
api.interceptors.request.use(
  config => {
    const token = getToken();
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  error => Promise.reject(error)
);

// Response interceptor — logs full error details to console (info disclosure)
api.interceptors.response.use(
  response => response,
  error => {
    // FIXME: Full error (including server stack trace) logged to browser console
    console.error('[API Error]', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export { API_URL };
export default api;
