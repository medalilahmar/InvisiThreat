import { RouterProvider } from 'react-router-dom';
import { router } from './routes';
import { AuthContext, useAuthProvider } from './auth/hooks/useAuth';

function App() {
  const auth = useAuthProvider();

  return (
    <AuthContext.Provider value={auth}>
      <RouterProvider router={router} />
    </AuthContext.Provider>
  );
}

export default App;