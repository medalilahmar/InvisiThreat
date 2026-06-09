import { useEffect, useRef, useState, useCallback } from 'react'
import './Home.css'
import Logo from '../assets/invilogo.png';
import { Link } from 'react-router-dom';
import { ThemeToggle } from '../components/ui/ThemeToggle';


// ─── Types ────────────────────────────────────────────────────────────────────
type NodeType = 'critical' | 'high' | 'normal'
interface Particle { x: number; y: number; vx: number; vy: number; r: number; pulse: number; type: NodeType }


// ─── Particle Canvas ──────────────────────────────────────────────────────────
function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    let animId: number

    const C = {
      critical: '255,71,87',
      high:     '255,107,53',
      accent:   '0,212,255',
    }

    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize)

    const nodes: Particle[] = Array.from({ length: 58 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.22,
      vy: (Math.random() - 0.5) * 0.22,
      r: Math.random() * 1.8 + 0.8,
      pulse: Math.random() * Math.PI * 2,
      type: Math.random() < 0.06 ? 'critical' : Math.random() < 0.14 ? 'high' : 'normal',
    }))

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy; n.pulse += 0.018
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1
      })
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x
          const dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 145) {
            const alpha = (1 - dist / 145) * 0.18
            const isCrit = nodes[i].type === 'critical' || nodes[j].type === 'critical'
            const isHigh = nodes[i].type === 'high'     || nodes[j].type === 'high'
            ctx.strokeStyle = isCrit
              ? `rgba(${C.critical},${alpha})`
              : isHigh
              ? `rgba(${C.high},${alpha * 0.75})`
              : `rgba(${C.accent},${alpha * 0.55})`
            ctx.lineWidth = isCrit ? 0.85 : 0.4
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.stroke()
          }
        }
      }
      nodes.forEach(n => {
        const pr = n.r + Math.sin(n.pulse) * 0.7
        const color = n.type === 'critical' ? `rgb(${C.critical})` : n.type === 'high' ? `rgb(${C.high})` : `rgb(${C.accent})`
        const alpha = n.type === 'critical' ? 0.92 : n.type === 'high' ? 0.72 : 0.48
        if (n.type === 'critical') {
          const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pr * 7)
          grd.addColorStop(0, `rgba(${C.critical},0.28)`)
          grd.addColorStop(1, `rgba(${C.critical},0)`)
          ctx.beginPath(); ctx.arc(n.x, n.y, pr * 7, 0, Math.PI * 2)
          ctx.fillStyle = grd; ctx.fill()
        }
        ctx.beginPath(); ctx.arc(n.x, n.y, pr, 0, Math.PI * 2)
        ctx.fillStyle = color.replace(')', `,${alpha})`).replace('rgb', 'rgba')
        ctx.fill()
      })
      animId = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(animId); window.removeEventListener('resize', resize) }
  }, [])

  return <canvas ref={canvasRef} className="particle-canvas" />
}

// ─── Animated Counter ─────────────────────────────────────────────────────────
function Counter({ to, suffix = '', duration = 2000 }: { to: number; suffix?: string; duration?: number }) {
  const [val, setVal] = useState(0)
  const started = useRef(false)
  const elRef   = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    const obs = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && !started.current) {
        started.current = true
        const t0 = performance.now()
        const tick = (now: number) => {
          const p = Math.min((now - t0) / duration, 1)
          setVal(Math.round((1 - Math.pow(1 - p, 3)) * to))
          if (p < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
      }
    }, { threshold: 0.5 })
    if (elRef.current) obs.observe(elRef.current)
    return () => obs.disconnect()
  }, [to, duration])
  return <span ref={elRef}>{val.toLocaleString()}{suffix}</span>
}

// ─── Typewriter ───────────────────────────────────────────────────────────────
function TypeWriter({ texts, speed = 58 }: { texts: string[]; speed?: number }) {
  const [display, setDisplay]   = useState('')
  const [idx, setIdx]           = useState(0)
  const [charIdx, setCharIdx]   = useState(0)
  const [deleting, setDeleting] = useState(false)
  useEffect(() => {
    const cur = texts[idx]
    if (!deleting && charIdx < cur.length) {
      const t = setTimeout(() => setCharIdx(c => c + 1), speed); return () => clearTimeout(t)
    }
    if (!deleting && charIdx === cur.length) {
      const t = setTimeout(() => setDeleting(true), 2400); return () => clearTimeout(t)
    }
    if (deleting && charIdx > 0) {
      const t = setTimeout(() => setCharIdx(c => c - 1), speed / 2.2); return () => clearTimeout(t)
    }
    if (deleting && charIdx === 0) { setDeleting(false); setIdx(i => (i + 1) % texts.length) }
  }, [charIdx, deleting, idx, texts, speed])
  useEffect(() => { setDisplay(texts[idx].slice(0, charIdx)) }, [charIdx, idx, texts])
  return <span>{display}<span className="hero-typewriter-cursor">█</span></span>
}

// ─── Feature Card ─────────────────────────────────────────────────────────────
function FeatureCard({ icon, title, desc, color, delay }: { icon: string; title: string; desc: string; color: string; delay: number }) {
  return (
    <div className="feature-card fu" style={{ '--card-color': color, animationDelay: `${delay}ms` } as React.CSSProperties}>
      <div className="feature-card-icon" style={{ background: `${color}14`, border: `1px solid ${color}24` }}>{icon}</div>
      <div className="feature-card-title">{title}</div>
      <div className="feature-card-desc">{desc}</div>
    </div>
  )
}

// ─── Model Bar ────────────────────────────────────────────────────────────────
function ModelBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="model-bar-item">
      <div className="model-bar-header">
        <span className="model-bar-name">{label}</span>
        <span className="model-bar-value" style={{ color }}>{value}%</span>
      </div>
      <div className="model-bar-track">
        <div className="model-bar-fill" style={{ width: `${value}%`, background: `linear-gradient(90deg, ${color}70, ${color})` }} />
      </div>
    </div>
  )
}

// ─── HOME PAGE ────────────────────────────────────────────────────────────────
export default function Home() {
  const scrollTo = useCallback((id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  const archSteps = [
    { icon: '👨‍💻', label: 'Push Développeur',  sub: 'git commit',        color: 'var(--accent)' },
    { icon: '🔍', label: 'Scanners',          sub: 'SAST · DAST · SCA', color: 'var(--accent3)' },
    { icon: '🗄️', label: 'DefectDojo',        sub: 'Findings DB',       color: 'var(--orange)' },
    { icon: '🏷️', label: 'Moteur de Tags',        sub: 'Étiquetage contextuel',   color: 'var(--green)' },
    { icon: '🤖', label: 'Scoreur IA',         sub: 'LightGBM',          color: 'var(--accent)' },
    { icon: '🚦', label: 'Porte de Sécurité',     sub: 'Bloquer / Autoriser',     color: 'var(--accent2)' },
  ]

  const features = [
    { icon: '🔬', color: 'var(--accent)', title: 'Notation de Risque par IA',     desc: 'Ensemble XGBoost · LightGBM · RandomForest. Chaque finding reçoit un score de risque de 0 à 4 pondéré par le contexte métier, et non seulement par la sévérité CVSS.' },
    { icon: '🏷️', color: 'var(--green)', title: 'Étiquetage Intelligent', desc: 'Détection automatique des findings en production, exposés publiquement, sensibles et urgents. Les tags alimentent directement le pipeline de scoring ML.' },
    { icon: '⚡', color: 'var(--accent3)', title: 'Porte de Sécurité',       desc: 'L’intégration CI/CD bloque automatiquement les déploiements lorsque un finding atteint un score ≥ 3 (High ou Critical). Protection zéro intervention' },
    { icon: '🧠', color: 'var(--orange)', title: 'Explicabilité SHAP', desc: 'Chaque décision de l’IA est entièrement expliquée. Voyez quelles caractéristiques ont influencé le score — pas de boîte noire, transparence totale pour votre équipe de sécurité.' },
    { icon: '📡', color: 'var(--accent2)', title: 'Unification des Scanners', desc: 'Résultats Semgrep (SAST), OWASP ZAP (DAST) et Snyk (SCA) unifiés via DefectDojo. Une seule source de vérité priorisée' },
    { icon: '📊', color: 'var(--purple)', title: 'Tableau de Bord en Temps Réel',      desc: 'Métriques en temps réel, graphiques de tendances, répartition des scanners et flux d’événements pipeline. Visibilité complète sur votre posture de sécurité' },
  ]

  const termLines = [
    { text: '$ semgrep --config auto ./src --output findings.json',       color: 'var(--muted)' },
    { text: '✓  Semgrep SAST: 487 findings — 0 new Critical',             color: 'var(--green)' },
    { text: '$ snyk test --severity-threshold=medium',                    color: 'var(--muted)' },
    { text: '⚠  Snyk SCA: 2 High vulnerabilities in node_modules',       color: 'var(--accent3)' },
    { text: '$ python main.py tag --engagement-id 5',                     color: 'var(--muted)' },
    { text: '🏷️  Tagged: production=847  external=312  sensitive=201',     color: 'var(--accent)' },
    { text: '$ python main.py predict --engagement-id 5',                 color: 'var(--muted)' },
    { text: '🤖 AI Scored: 1311 findings · max_risk=4 (Critical)',        color: 'var(--accent)' },
    { text: '⛔ SECURITY GATE: score 4 ≥ threshold 3 — DEPLOY BLOCKED',  color: 'var(--accent2)' },
  ]

  const riskClasses = [
    { score: '0', color: 'var(--accent)', label: 'Info' },
    { score: '1', color: 'var(--green)', label: 'Low'  },
    { score: '2', color: 'var(--accent3)', label: 'Med'  },
    { score: '3', color: 'var(--orange)', label: 'High' },
    { score: '4', color: 'var(--accent2)', label: 'Crit' },
  ]

  const techPills = [
    { label: 'LightGBM', color: 'var(--accent)' }, { label: 'FastAPI',    color: 'var(--green)' },
    { label: 'SHAP',     color: 'var(--accent3)' }, { label: 'DefectDojo', color: 'var(--orange)' },
    { label: 'React',    color: 'var(--accent)' }, { label: 'Prometheus', color: 'var(--accent2)' },
    { label: 'Docker',   color: 'var(--green)' }, { label: 'Semgrep',    color: 'var(--purple)' },
    { label: 'Snyk',     color: 'var(--accent3)' },
  ]

  return (
    <div className="home-root">

      {/* Background */}
      <div className="bg-grid" />
      <div className="bg-radials" />
      <div className="scan-line" />
      <ParticleCanvas />

      {/* ── NAVBAR ── */}
      <nav className="navbar">
        <a className="navbar-logo" href="#">
            <img 
                src={Logo} 
                alt="InvisiThreat Logo" 
                className="navbar-logo-icon"
            />
            <span className="navbar-logo-text">Invisi<span>Threat</span></span>
        </a>
        <ul className="navbar-links">
          {([['Features','features'],['Architecture','architecture'],['Model','model'],['Pipeline','pipeline']] as [string,string][]).map(([lbl, id]) => (
            <li key={id}><a href="#" onClick={e => { e.preventDefault(); scrollTo(id) }}>{lbl}</a></li>
          ))}
        </ul>
        <div className="navbar-actions">
          <ThemeToggle />
          <Link to="/login" className="navbar-cta">Ouvrir le Tableau de Bord →</Link>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="hero">
        <div className="hero-badge">
          <div className="hero-badge-dot" />
          <span className="hero-badge-text">PLATEFORME D’INTELLIGENCE DEVSECOPS PILOTÉE PAR IA</span>
        </div>
        <h1 className="hero-headline">
          <span className="hero-headline-plain">Rendez Chaque</span>
          <span className="hero-headline-shimmer">Menace Visible.</span>
        </h1>
        <div className="hero-typewriter">
          <TypeWriter texts={[
            'Notation des vulnérabilités avec l’IA.',
            'SAST · DAST · SCA — unifiés.',
            'Analyse de risque contextuelle',
            'LightGBM · F1=0.8937 · 23 features.',
            'Bloquez les menaces critiques avant leur mise en production.',
          ]} />
        </div>
        <p className="hero-description">
          InvisiThreat attribue des scores de risque générés par IA à chaque finding de sécurité provenant de vos scanners, en tenant compte de l’exposition en production, du contexte de données sensibles et de l’impact métier — pour que votre équipe se concentre sur ce qui compte vraiment
        </p>
        <div className="hero-cta-group" id="launch">
          <button className="btn-primary" onClick={() => alert('Run: python main.py serve --port 8000\nthen: npm run dev')}>
            <span></span> Lancer le Tableau de Bord
          </button>
          <button className="btn-ghost" onClick={() => scrollTo('architecture')}>
            <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>▶</span> Comment ça marche
          </button>
        </div>
        <div className="hero-tech-pills">
          {techPills.map(({ label, color }) => (
            <span key={label} className="tech-pill" style={{ background: `${color}12`, color, border: `1px solid ${color}26` }}>{label}</span>
          ))}
        </div>
        <div className="hero-scroll-hint">
          <span className="hero-scroll-hint-label">SCROLL</span>
          <div className="hero-scroll-hint-line" />
        </div>
      </section>

      {/* ── STATS TICKER ── */}
      <div className="stats-ticker">
        <div className="stats-ticker-inner">
          {[
            { to: 1311, suffix: '',  label: 'Findings analysés', color: 'var(--accent)'  },
            { to: 90,   suffix: '%', label: 'Précision du modèle',    color: 'var(--green)'   },
            { to: 1125, suffix: '',  label: 'Findings étiquetés',   color: 'var(--orange)'  },
            { to: 23,   suffix: '',  label: 'Caractéristiques ML',       color: 'var(--accent3)' },
          ].map(({ to, suffix, label, color }) => (
            <div key={label} className="stat-item">
              <div className="stat-value" style={{ color }}><Counter to={to} suffix={suffix} /></div>
              <div className="stat-label">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── FEATURES ── */}
      <section className="section" id="features">
        <div className="section-inner">
          <div className="section-header">
            <div className="section-label">Fonctionnalités</div>
            <h2 className="section-title">Tout ce que votre équipe de sécurité<br /><span>a réellement besoin.</span></h2>
            <p className="section-subtitle">Des résultats bruts des scanners jusqu’à des scores de risque priorisés et contextuels — automatiquement, en quelques secondes.</p>
          </div>
          <div className="features-grid">
            {features.map((f, i) => <FeatureCard key={f.title} {...f} delay={i * 90} />)}
          </div>
        </div>
      </section>

      {/* ── ARCHITECTURE ── */}
      <section className="arch-section" id="architecture">
        <div className="section-inner">
          <div className="section-header">
            <div className="section-label">Architecture</div>
            <h2 className="section-title">Comment <span>InvisiThreat</span> fonctionne</h2>
            <p className="section-subtitle">Un pipeline entièrement automatisé du commit de code jusqu’aux findings notés par risque — zéro étape manuelle</p>
          </div>
          <div className="arch-flow">
            {archSteps.map((step, i) => (
              <div key={step.label} style={{ display: 'flex', alignItems: 'center' }}>
                <div className="arch-step" style={{ background: `${step.color}08`, border: `1px solid ${step.color}1e` }}>
                  <div className="arch-step-icon" style={{ animationDelay: `${i * 0.28}s` }}>{step.icon}</div>
                  <div className="arch-step-label">{step.label}</div>
                  <div className="arch-step-sub" style={{ color: step.color }}>{step.sub}</div>
                </div>
                {i < archSteps.length - 1 && (
                  <div className="arch-arrow">
                    <div className="arch-arrow-line" style={{ background: `linear-gradient(90deg, ${step.color}50, ${archSteps[i+1].color}50)` }} />
                    <div className="arch-arrow-tip">▶</div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── MODEL STATS ── */}
      <section className="model-section" id="model">
        <div className="model-section-inner">
          <div className="model-text-side">
            <div className="section-label">Performance du Modèle</div>
            <h2 className="model-title">IA de niveau production.<br /><span>Pas juste une preuve de concept.</span></h2>
            <p className="model-description">
              Entraîné sur 1 311 findings de sécurité réels avec 23 caractéristiques élaborées.
              The LightGBM model was selected automatically via cross-validation,
              outperforming XGBoost and RandomForest on all metrics.
            </p>
            <div className="model-bars">
              <ModelBar label="F1-Weighted Score" value={89.4} color="var(--accent)" />
              <ModelBar label="ROC-AUC Score"     value={96.0} color="var(--green)" />
              <ModelBar label="Overall Accuracy"  value={90.0} color="var(--accent3)" />
            </div>
          </div>

          <div className="model-card">
            <div className="model-card-header">
              <div>
                <div className="model-card-name-label">ACTIVE MODEL</div>
                <div className="model-card-name">LightGBM</div>
              </div>
              <div className="model-card-badge"><span>● SELECTED</span></div>
            </div>
            <div className="model-card-metrics">
              {[{k:'F1-weighted',v:'0.8937',c:'var(--accent)'},{k:'ROC-AUC',v:'0.9603',c:'var(--green)'},{k:'Accuracy',v:'90.0%',c:'var(--accent3)'},{k:'Features',v:'23',c:'var(--orange)'}].map(({ k, v, c }) => (
                <div key={k} className="model-metric-box">
                  <div className="model-metric-value" style={{ color: c }}>{v}</div>
                  <div className="model-metric-label">{k}</div>
                </div>
              ))}
            </div>
            <div className="model-risk-classes-label">RISK CLASSES</div>
            <div className="model-risk-classes">
              {riskClasses.map(({ score, color, label }) => (
                <div key={score} className="risk-class-box" style={{ background: `${color}10`, border: `1px solid ${color}26` }}>
                  <div className="risk-class-score" style={{ color }}>{score}</div>
                  <div className="risk-class-name"  style={{ color }}>{label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── PIPELINE TERMINAL ── */}
      <section className="pipeline-section" id="pipeline">
        <div className="pipeline-section-inner">
          <div className="section-label">Intégration CI/CD</div>
          <h2 className="section-title" style={{ textAlign: 'center' }}>
            Automatisation de la sécurité <span style={{ color: 'var(--accent3)' }}>sans contact.</span>
          </h2>
          <p className="section-subtitle" style={{ margin: '0 auto' }}>
            Push de code — les scanners s’exécutent, l’IA note chaque finding, le déploiement est bloqué si le risque ≥ 3. 
      Aucune intervention humaine requise.
          </p>
          <div className="terminal">
            <div className="terminal-header">
              {['var(--accent2)','var(--accent3)','var(--green)'].map((bg, i) => <div key={i} className="terminal-dot" style={{ background: bg, opacity: 0.82 }} />)}
              <span className="terminal-title">github-actions / security-pipeline.yml — branche main</span>
            </div>
            <div className="terminal-body">
              {termLines.map((line, i) => (
                <span key={i} className="terminal-line" style={{ color: line.color, animationDelay: `${300 + i * 200}ms` }}>
                  {line.text}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── FINAL CTA ── */}
      <section className="cta-section">
        <div className="cta-glow-orb" />
        <div className="section-label">Commencer</div>
        <h2 className="cta-title">Prêt à voir vos<br /><span>menaces clairement?</span></h2>
        <p className="cta-desc">
          Démarrez le backend, ouvrez le dashboard, et obtenez des scores de risque alimentés par l'IA sur chaque vulnérabilité — en moins de 60 secondes.
        </p>
        <div className="cta-steps">
          {[
            { step: '01', cmd: 'python main.py serve', color: 'var(--accent)' },
            { step: '02', cmd: 'npm run dev',          color: 'var(--green)' },
            { step: '03', cmd: 'localhost:5173',       color: 'var(--accent3)' },
          ].map(({ step, cmd, color }) => (
            <div key={step} className="cta-step">
              <span className="cta-step-num" style={{ color }}>{step}</span>
              <div className="cta-step-divider" />
              <code className="cta-step-cmd" style={{ color }}>{cmd}</code>
            </div>
          ))}
        </div>
        <div className="cta-buttons">
          <button className="btn-primary" style={{ fontSize: 16, padding: '18px 44px' }}
                  onClick={() => scrollTo('launch')}>
            <span>🛡️</span> Open Dashboard
          </button>
          <a href="https://github.com" style={{ textDecoration: 'none' }}>
            <button className="btn-ghost" style={{ fontSize: 16, padding: '18px 44px' }}>
              <span>⭐</span> GitHub
            </button>
          </a>
        </div>
      </section>

     
    </div>
  )
}