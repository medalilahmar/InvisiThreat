import React from 'react';
import { BrowserRouter as Router, Route, Switch, Redirect } from 'react-router-dom';
import Login       from './components/Login';
import Dashboard   from './components/Dashboard';
import ProjectDetail from './components/ProjectDetail';
import Search      from './components/Search';
import AdminPanel  from './components/AdminPanel';

/**
 * DAST Finding: Client-side-only route guard
 * The isAuthenticated check reads from localStorage which can be tampered.
 * Even if the frontend blocks the route, the API has no proper auth.
 */
const isAuthenticated = () => !!localStorage.getItem('token'); // Easily bypassed

// FIXME: This "protected" wrapper is security theater
// Backend endpoints must enforce auth server-side
const ProtectedRoute = ({ component: Component, ...rest }) => (
  <Route
    {...rest}
    render={props =>
      isAuthenticated()
        ? <Component {...props} />
        : <Redirect to="/login" />
    }
  />
);

// FIXME: Admin route does NOT check if user has admin role
// It only checks token presence — matches the broken requireAdmin middleware
const AdminRoute = ({ component: Component, ...rest }) => (
  <Route
    {...rest}
    render={props =>
      isAuthenticated()            // BUG: Should also check localStorage.getItem('role') === 'admin'
        ? <Component {...props} /> // But even that would be tamper-prone
        : <Redirect to="/login" />
    }
  />
);

function App() {
  return (
    <Router>
      <Switch>
        <Route       exact path="/"          render={() => <Redirect to="/login" />} />
        <Route             path="/login"     component={Login} />
        <ProtectedRoute    path="/dashboard" component={Dashboard} />
        <ProtectedRoute    path="/project/:id" component={ProjectDetail} />
        <ProtectedRoute    path="/search"    component={Search} />
        {/* FIXME: Admin panel accessible to any authenticated user — no role check */}
        <AdminRoute        path="/admin"     component={AdminPanel} />
        <Route             render={() => <div style={{color:'white',padding:'40px'}}>404 — Page not found</div>} />
      </Switch>
    </Router>
  );
}

export default App;
