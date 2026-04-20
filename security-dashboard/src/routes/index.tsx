import { createBrowserRouter } from 'react-router-dom';
import { PageLayout } from '../components/layout/PageLayout';
import { ProtectedRoute } from '../auth/components/ProtectedRoute';
import Home from '../pages/Home';
import LoginPage from '../auth/pages/LoginPage';
import ProductsPage from '../features/products/pages/ProductsPage';
import EngagementsPage from '../features/engagements/pages/EngagementsPage';
import FindingsPage from '../features/findings/pages/FindingsPage';
import DashboardPage from '../features/dashboard/pages/DashboardPage';
import FindingDetailPage from '../features/findings/pages/FindingDetailPage';
import { ModelStatsPage } from '../features/model/pages/ModelStatsPage';

export const router = createBrowserRouter([


  { path: '/', element: <Home /> },
  { path: '/login', element: <LoginPage /> },

  {
    element: <PageLayout />,
    children: [
     
      {
        path: '/products',
        element: <ProtectedRoute><ProductsPage /></ProtectedRoute>,
      },
      {
        path: '/model-stats',
        element: <ProtectedRoute><ModelStatsPage /></ProtectedRoute>,
      },  
      {
        path: '/engagements',
        element: <ProtectedRoute><EngagementsPage /></ProtectedRoute>,
      },
      {
        path: '/findings',
        element: <ProtectedRoute><FindingsPage /></ProtectedRoute>,
      },
      {
        path: '/dashboard',
        element: <ProtectedRoute><DashboardPage /></ProtectedRoute>,
      },
      {
        path: '/findings/:id',
        element: <ProtectedRoute><FindingDetailPage /></ProtectedRoute>,
      },
    ],
  },
]);