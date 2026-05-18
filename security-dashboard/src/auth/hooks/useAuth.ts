import {
  useState, useEffect, createContext,
  useContext, useCallback
} from 'react';
import { AuthUser, AuthState } from '../types/auth.types';
import {
  getToken, getSavedUser, getMe,
  logout as doLogout, saveUser
} from '../services/authService';

interface AuthContextType extends AuthState {
  login:       (user: AuthUser, token: string) => void;
  logout:      () => Promise<void>;
  refreshUser: () => Promise<void>;
  isAdmin:     boolean;
  isManager:   boolean;
  isAnalyst:   boolean;
  isDeveloper: boolean;
  isActive:   boolean;
  isPending:  boolean;
  isBlocked:  boolean;
  isLocked:   boolean;
}

export const AuthContext = createContext<AuthContextType | null>(null);

export function useAuthProvider(): AuthContextType {
  const [state, setState] = useState<AuthState>({
    user:            getSavedUser(),
    token:           getToken(),
    isAuthenticated: !!getToken() && !!getSavedUser(),
    isLoading:       !!getToken(),
     
  });

  useEffect(() => {
    const token = getToken();
    const savedUser = getSavedUser();

    console.log('🔄 useAuthProvider montage — token:', !!token, '— user:', !!savedUser);

    if (!token) {
      setState(s => ({ ...s, isLoading: false }));
      return;
    }

    // Si on a déjà le user sauvegardé, on affiche immédiatement
    // et on valide en arrière-plan
    if (savedUser) {
      setState({
        user: savedUser,
        token,
        isAuthenticated: true,
        isLoading: false, // ← on n'attend plus getMe() pour débloquer l'UI
      });
    }

    // Validation en arrière-plan
    getMe()
      .then(user => {
        console.log('✅ getMe() OK :', user);
        saveUser(user);
        setState({ user, token, isAuthenticated: true, isLoading: false });
      })
      .catch((err) => {
        console.error('❌ getMe() échoué :', err?.response?.status, err?.response?.data);
        if (err?.response?.status === 401) {
          doLogout();
          setState({ user: null, token: null, isAuthenticated: false, isLoading: false });
        }
        // Sinon (erreur réseau) on garde l'état déjà défini avec savedUser
      });
  }, []);

  const login = useCallback((user: AuthUser, token: string) => {
    setState({ user, token, isAuthenticated: true, isLoading: false });
  }, []);

  const logout = useCallback(async () => {
    await doLogout();
    setState({ user: null, token: null, isAuthenticated: false, isLoading: false });
  }, []);

  const refreshUser = useCallback(async () => {
    const user = await getMe();
    saveUser(user);
    setState(s => ({ ...s, user }));
  }, []);

  return {
    ...state,
    login,
    logout,
    refreshUser,
    isAdmin:     state.user?.role === 'admin',
    isManager:   state.user?.role === 'manager',
    isAnalyst:   state.user?.role === 'analyst',
    isDeveloper: state.user?.role === 'developer',
    isActive:    state.user?.status === 'active',
    isPending:  state.user?.status === 'pending',
    isBlocked:  state.user?.status === 'blocked',
    isLocked:   !!(state.user?.locked_until &&
                  new Date(state.user.locked_until) > new Date()),
  };
}

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside AuthProvider');
  return ctx;
}