import { Outlet } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Footer } from './Footer';
import { ParticleCanvas } from '../ui/ParticleCanvas';
import { useAuth } from '../../auth/hooks/useAuth';

export function PageLayout() {
  const { isAdmin } = useAuth();

  return (
    <>
      {/* Background principal */}
      <ParticleCanvas />

      {/* Fond de base (plus clair) */}
      <div className="global-background" />

      {/* Contenu de l'application */}
      <div style={{ position: 'relative', zIndex: 10 }}>
        <Navbar isAdmin={isAdmin} />
        
        <main className="main-content" style={{ paddingTop: '64px' }}>
          <Outlet />
        </main>
        
        <Footer />
      </div>
    </>
  );
}