import { createBrowserRouter } from 'react-router-dom';
import { PageLayout } from '../components/layout/PageLayout';
import Home from '../pages/Home';
import ProductsPage from '../features/products/pages/ProductsPage';
import EngagementsPage from '../features/engagements/pages/EngagementsPage';
import FindingsPage from '../features/findings/pages/FindingsPage';
import DashboardPage from '../features/dashboard/pages/DashboardPage';


export const router = createBrowserRouter([
  {
    element: <PageLayout />,  
    children: [
      {
        path: '/',
        element: <Home />,
      },
      {
        path: '/products',
        element: <ProductsPage />,
      },
      { path: '/engagements', element: <EngagementsPage /> },
      { path: '/findings', element: <FindingsPage /> },

      { path: 'dashboard', element: <DashboardPage /> },

     
    ],
  },
]);