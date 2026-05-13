import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { AuthSliceState, AuthUser } from '../auth/types/auth.types';

const USER_KEY = 'invithreat_user';

const loadPersistedUser = (): AuthUser | null => {
  try {
    const raw = sessionStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
};

const initialState: AuthSliceState = {
  isAuthenticated: !!loadPersistedUser(),
  user:            loadPersistedUser(),
  token:           null,
  isLoading:       false,
  loading:         false,
  error:           null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart(state) {
      state.loading   = true;
      state.isLoading = true;
      state.error     = null;
    },
    loginSuccess(state, action: PayloadAction<AuthUser>) {
      state.isAuthenticated = true;
      state.user      = action.payload;
      state.loading   = false;
      state.isLoading = false;
      state.error     = null;
      sessionStorage.setItem(USER_KEY, JSON.stringify(action.payload));
    },
    loginFailure(state, action: PayloadAction<string>) {
      state.loading   = false;
      state.isLoading = false;
      state.error     = action.payload;
    },
    logout(state) {
      state.isAuthenticated = false;
      state.user      = null;
      state.token     = null;
      state.loading   = false;
      state.isLoading = false;
      state.error     = null;
      sessionStorage.removeItem(USER_KEY);
    },
    clearError(state) {
      state.error = null;
    },
  },
});

export const { loginStart, loginSuccess, loginFailure, logout, clearError } =
  authSlice.actions;

export default authSlice.reducer;