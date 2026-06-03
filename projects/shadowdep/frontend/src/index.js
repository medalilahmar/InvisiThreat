import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

// FIXME: No error boundary wrapping
// FIXME: React 16.8.0 has known XSS vulnerability in SVG namespace handling
// No performance monitoring, no CSP nonce

ReactDOM.render(
  // Note: React.StrictMode removed — hides additional warnings
  <App />,
  document.getElementById('root')
);
