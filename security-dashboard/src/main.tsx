import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Provider } from 'react-redux';
import { store } from './store/store';

import './assets/styles/globals.css';

import './index.css';
import App from './App.tsx';

const queryClient = new QueryClient();

const saved    = localStorage.getItem('invisithreat-theme');
const prefLight = window.matchMedia('(prefers-color-scheme: light)').matches;
const initial  = saved ?? (prefLight ? 'light' : 'dark');
document.documentElement.setAttribute('data-theme', initial);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </Provider>
  </StrictMode>
);