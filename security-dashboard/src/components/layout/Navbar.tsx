import { Link } from 'react-router-dom';
import Logo from '../../assets/invilogo.png';
import { LogoutButton } from '../../auth/components/LogoutButton';
import './Navbar.css';

export function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <img src={Logo} alt="InvisiThreat Logo" className="navbar-logo-icon" />
        <span className="navbar-logo-text">Invisi<span>Threat</span></span>
      </Link>

      <ul className="navbar-links">
        <li><Link to="/dashboard">Dashboard</Link></li>
        <li><Link to="/products">Produits</Link></li>
        <li><Link to="/engagements">Engagements</Link></li>  
        <li><Link to="/model-stats">Modèle</Link></li>
      </ul>

      <div className="navbar-actions">
        <LogoutButton />
      </div>
    </nav>
  );
}