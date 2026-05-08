import { Outlet } from 'react-router-dom';
import { Navbar } from './Navbar';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { ParticleCanvas } from '../ui/ParticleCanvas';
import { useAuth } from '../../auth/hooks/useAuth';



export function PageLayout() {
  const { isAdmin } = useAuth();
  return (
    <div className="home-root">
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />
      <ParticleCanvas />
      <Navbar isAdmin={isAdmin}  />

      <Sidebar />
      <main className="main-content" style={{ paddingTop: '64px' }}>
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}