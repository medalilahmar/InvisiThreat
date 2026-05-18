import { createBrowserRouter } from 'react-router-dom';
import { PageLayout } from '../components/layout/PageLayout';
import  ProtectedRoute  from '../auth/components/ProtectedRoute';
import Home from '../pages/Home';
import LoginPage from '../auth/pages/LoginPage';
import RegisterPage from '../auth/pages/RegisterPage';
import PendingPage from '../auth/pages/PendingPage';
import ProductsPage from '../features/products/pages/ProductsPage';
import EngagementsPage from '../features/engagements/pages/EngagementsPage';
import FindingsPage from '../features/findings/pages/FindingsPage';
import DashboardPage from '../features/dashboard/pages/DashboardPage';
import FindingDetailPage from '../features/findings/pages/FindingDetailPage';
import { ModelStatsPage } from '../features/model/pages/ModelStatsPage';
import AdminPanel from '../features/admin/pages/AdminPanel';
import ProfilePage from '../features/profile/pages/ProfilePage';
import AnalyticsPage from '../features/analytics/pages/AnalyticsPage';
import BlockedPage from '../auth/pages/BlockedPage';


export const router = createBrowserRouter([
  // ────── Routes publiques ──────
  { path: '/', element: <Home /> },
  { path: '/login', element: <LoginPage /> },
  { path: '/register', element: <RegisterPage /> },
  { path: '/auth/pending', element: <PendingPage /> },
  { path: '/auth/blocked', element: <BlockedPage /> },


  {
    element: <PageLayout />,
    children: [
      {
        path: '/dashboard',
        element: <ProtectedRoute><DashboardPage /></ProtectedRoute>,
      },
      {
        path: '/products',
        element: <ProtectedRoute blockedRoles={['analyst']}><ProductsPage /></ProtectedRoute>,
      },
      {
        path: '/model-stats',
        element: <ProtectedRoute><ModelStatsPage /></ProtectedRoute>,
      },
      {
        path: '/engagements',
        element: <ProtectedRoute  blockedRoles={['analyst']}><EngagementsPage /></ProtectedRoute>,
      },
      {
        path: '/findings',
        element: <ProtectedRoute blockedRoles={['analyst']} ><FindingsPage /></ProtectedRoute>,
      },
      {
        path: '/findings/:id',
        element: <ProtectedRoute blockedRoles={['analyst']}><FindingDetailPage /></ProtectedRoute>,
      },
      {
        path: '/profile',
        element: <ProtectedRoute><ProfilePage /></ProtectedRoute>,
      },
      // ── Route admin ──
      {
        path: '/admin/*',
        element: (
          <ProtectedRoute allowedRoles={['admin']}>
            <AdminPanel />
          </ProtectedRoute>
        ),
      },

      {
        path: '/analytics',
        element: (
          <ProtectedRoute>
            <AnalyticsPage />
          </ProtectedRoute>
        ),
      },
    ],
  },
]);