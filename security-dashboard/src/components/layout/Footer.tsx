// src/components/layout/Footer.tsx
export function Footer() {
  return (
    <footer className="footer">
      <div className="footer-brand">
        <span style={{ fontSize: 20 }}>🛡️</span>
        <span className="footer-brand-name">Invisi<span>Threat</span></span>
        <span className="footer-version">v1.0.0</span>
      </div>
      <div className="footer-center">
        AI-POWERED DEVSECOPS · LIGHTGBM · F1=0.8937 · BUILT FOR SECURITY ENGINEERS
      </div>
      <div className="footer-right">
        Engagement #5 · 1311 findings · {new Date().getFullYear()}
      </div>
    </footer>
  );
}