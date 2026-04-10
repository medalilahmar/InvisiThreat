// src/components/layout/Navbar.tsx
import { Link } from 'react-router-dom';
import Logo from '../../assets/invilogo.png'; // adapte le chemin si nécessaire

export function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="navbar-logo">
        <img src={Logo} alt="InvisiThreat Logo" className="navbar-logo-icon" />
        <span className="navbar-logo-text">Invisi<span>Threat</span></span>
      </Link>
      <ul className="navbar-links">
        <li><Link to="/products">Produits</Link></li>
        <li><Link to="/engagements">Engagements</Link></li>
        <li><Link to="/findings">Findings</Link></li>
        <li><Link to="/model-stats">Modèle</Link></li>
      </ul>
      <button className="navbar-cta" onClick={() => window.location.href = '/products'}>
        Dashboard →
      </button>
    </nav>
  );
}