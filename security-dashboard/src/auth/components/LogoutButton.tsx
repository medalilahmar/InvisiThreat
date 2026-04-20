// src/auth/components/LogoutButton.tsx
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../auth/hooks/useAuth';

interface LogoutButtonProps {
  className?: string;
}

export const LogoutButton = ({ className }: LogoutButtonProps) => {
  const { isAuthenticated, user, logout } = useAuth();
  const navigate = useNavigate();

  if (!isAuthenticated) return null;

  const handleLogout = () => {
    // 1. Rediriger d'abord vers la page d'accueil (non protégée)
    navigate('/', { replace: true });
    // 2. Après un court délai, effacer l'état d'authentification
    setTimeout(() => {
      logout();
    }, 100);
  };

  return (
    <button
      className={className || 'logout-button'}   
      onClick={handleLogout}
      title="Déconnexion"
    >
      <span className="logout-username">{user?.username}</span>
      <svg className="logout-icon" width="16" height="16" viewBox="0 0 20 20" fill="none">
        <path
          d="M13 15l4-5-4-5M17 10H8M10 4H5a1 1 0 00-1 1v10a1 1 0 001 1h5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
};