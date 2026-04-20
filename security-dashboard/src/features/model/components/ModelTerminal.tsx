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
            <div className="terminal-dot red" />
            <div className="terminal-dot yellow" />
            <div className="terminal-dot green" />
            <span>POST /predict</span>
          </div>
          <div className="terminal-body">
            <div className="terminal-line comment"># Prédiction simple</div>
            <div className="terminal-line">$ curl -X POST http://localhost:8081/predict \</div>
            <div className="terminal-line indent">-H "Content-Type: application/json" \</div>
            <div className="terminal-line indent">-d '{`{`}</div>
            <div className="terminal-line indent2">"severity": "high",</div>
            <div className="terminal-line indent2">"cvss_score": 7.5,</div>
            <div className="terminal-line indent2">"tags": ["production"],</div>
            <div className="terminal-line indent2">"has_cve": 1</div>
            <div className="terminal-line indent">{`}`}'</div>
            <div className="terminal-line response">
              {`> {"risk_class": 3, "risk_level": "High", "confidence": 0.92}`}
            </div>
          </div>
        </div>

        <div className="terminal">
          <div className="terminal-header">
            <div className="terminal-dot red" />
            <div className="terminal-dot yellow" />
            <div className="terminal-dot green" />
            <span>GET /model/info</span>
          </div>
          <div className="terminal-body">
            <div className="terminal-line comment"># Informations du modèle</div>
            <div className="terminal-line">$ curl http://localhost:8081/model/info</div>
            <div className="terminal-line response">
              {`> {"model_version": "20260420...", "n_classes": 5, "n_features": 20}`}
            </div>
          </div>
        </div>
      </div>

      <div className="api-doc-link">
        <a href="http://localhost:8081/docs" target="_blank" rel="noopener noreferrer">
          📖 Voir la documentation Swagger complète →
        </a>
      </div>
    </section>
  );
}