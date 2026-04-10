import { useState } from 'react';
import { NavLink } from 'react-router-dom';

export function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const toggleSidebar = () => setIsCollapsed(!isCollapsed);

  return (
    <aside className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        {!isCollapsed && <span className="sidebar-logo">InvisiThreat</span>}
        <button className="sidebar-toggle" onClick={toggleSidebar}>
          {isCollapsed ? '→' : '←'}
        </button>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <span className="sidebar-icon">🏠</span>
          {!isCollapsed && <span>Accueil</span>}
        </NavLink>

        <NavLink to="/products" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <span className="sidebar-icon">📦</span>
          {!isCollapsed && <span>Produits</span>}
        </NavLink>

        <NavLink to="/engagements" className="sidebar-link">
        <span className="sidebar-icon">🔍</span>
        <span>Engagements</span>
        </NavLink>

        <NavLink to="/findings" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <span className="sidebar-icon">⚠️</span>
          {!isCollapsed && <span>Findings</span>}
        </NavLink>

        <NavLink to="/model-stats" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <span className="sidebar-icon">📊</span>
          {!isCollapsed && <span>Modèle</span>}
        </NavLink>

        <NavLink to="/dashboard" className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}>
          <span className="sidebar-icon">📈</span>
          {!isCollapsed && <span>Dashboard</span>}
        </NavLink>
      </nav>
    </aside>
  );
}