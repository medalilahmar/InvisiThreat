import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../../store/store';
import { loginStart, loginSuccess, loginFailure, logout, clearError } from '../../store/authSlice';
import { authService } from '../services/authService';
import type { LoginCredentials } from '../types/auth';

export const useAuth = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { isAuthenticated, user, loading, error } = useSelector(
    (state: RootState) => state.auth
  );

  const login = async (credentials: LoginCredentials) => {
    dispatch(loginStart());
    try {
      const user = await authService.login(credentials);
      dispatch(loginSuccess(user));
      return true;
    } catch (err) {
      dispatch(loginFailure(err instanceof Error ? err.message : 'Login failed'));
      return false;
    }
  };

  const handleLogout = () => dispatch(logout());
  const handleClearError = () => dispatch(clearError());

  return { isAuthenticated, user, loading, error, login, logout: handleLogout, clearError: handleClearError };
};