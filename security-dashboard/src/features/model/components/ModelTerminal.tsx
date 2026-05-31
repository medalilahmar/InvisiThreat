export function ModelTerminal() {
  return (
    <section className="model-terminal-section">
      <div className="section-label">
        <span className="fp-label-dot" />
        INFERENCE
      </div>
      <h2 className="section-title">
        Utiliser le modèle via <span>l'API</span>
      </h2>

      <div className="terminals-grid">
        <div className="terminal">
          <div className="terminal-header">
            <div className="terminal-dots">
              <span className="terminal-dot terminal-dot--close"  />
              <span className="terminal-dot terminal-dot--min"    />
              <span className="terminal-dot terminal-dot--expand" />
            </div>
            <span className="terminal-route">POST /predict</span>
          </div>
          <div className="terminal-body">
            <div className="terminal-line terminal-line--comment"># Prédiction simple</div>
            <div className="terminal-line">$ curl -X POST http://localhost:8081/predict \</div>
            <div className="terminal-line terminal-line--indent">-H "Content-Type: application/json" \</div>
            <div className="terminal-line terminal-line--indent">-d '{`{`}</div>
            <div className="terminal-line terminal-line--indent2">"severity": "high",</div>
            <div className="terminal-line terminal-line--indent2">"cvss_score": 7.5,</div>
            <div className="terminal-line terminal-line--indent2">"tags": ["production"],</div>
            <div className="terminal-line terminal-line--indent2">"has_cve": 1</div>
            <div className="terminal-line terminal-line--indent">{`}`}'</div>
            <div className="terminal-line terminal-line--response">
              {`> {"risk_class": 3, "risk_level": "High", "confidence": 0.92}`}
            </div>
          </div>
        </div>

        <div className="terminal">
          <div className="terminal-header">
            <div className="terminal-dots">
              <span className="terminal-dot terminal-dot--close"  />
              <span className="terminal-dot terminal-dot--min"    />
              <span className="terminal-dot terminal-dot--expand" />
            </div>
            <span className="terminal-route">GET /model/info</span>
          </div>
          <div className="terminal-body">
            <div className="terminal-line terminal-line--comment"># Informations du modèle</div>
            <div className="terminal-line">$ curl http://localhost:8081/model/info</div>
            <div className="terminal-line terminal-line--response">
              {`> {"model_version": "20260420...", "n_classes": 5, "n_features": 20}`}
            </div>
          </div>
        </div>
      </div>

      <div className="api-doc-link">
        <a href="http://localhost:8081/docs" target="_blank" rel="noopener noreferrer">
          <svg
            width="14" height="14" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round"
          >
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          Voir la documentation Swagger complète
          <svg
            width="12" height="12" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2"
            strokeLinecap="round" strokeLinejoin="round"
          >
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </a>
      </div>
    </section>
  );
}