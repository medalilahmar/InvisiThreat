import { Link } from 'react-router-dom';
import Logo from '../../assets/invilogo.png';
import { LogoutButton } from '../../auth/components/LogoutButton';
import NotificationBell from './NotificationBell';
import { useAuth } from '../../auth/hooks/useAuth';
import { ThemeToggle } from '../ui/ThemeToggle';
import './Navbar.css';

interface NavbarProps {
  isAdmin?: boolean;
}

export function Navbar({ isAdmin = false }: NavbarProps) {
  const { user } = useAuth();
  
  // Les analyst ne peuvent pas voir Products, Engagements, Findings
  const isAnalyst = user?.role === 'analyst';

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <img src={Logo} alt="InvisiThreat Logo" className="navbar-logo-icon" />
        <span className="navbar-logo-text">Invisi<span>Threat</span></span>
      </Link>

      <ul className="navbar-links">
        <li><Link to="/dashboard">Dashboard</Link></li>
        
        {/* Masquer Products et Engagements pour les analyst */}
        {!isAnalyst && (
          <>
            <li><Link to="/products">Produits</Link></li>
          </>
        )}
        
        <li><Link to="/analytics">Analytics</Link></li>
        <li><Link to="/model-stats">Modèle</Link></li>
        <li><Link to="/profile">Profil</Link></li>
        
        {/* Admin link - réservé aux admin */}
        {isAdmin && (
          <li>
            <Link to="/admin" className="admin-link">Admin</Link>
          </li>
        )}
      </ul>

      <div className="navbar-actions">
        <NotificationBell />
        <ThemeToggle />
        <LogoutButton />
      </div>
      
    </nav>
  );
}