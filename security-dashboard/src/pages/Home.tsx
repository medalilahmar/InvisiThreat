import { useEffect, useRef, useState, useCallback } from 'react'
import './Home.css'
import Logo from '../assets/invilogo.png';
import { Link } from 'react-router-dom';

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
            ctx.strokeStyle = isCrit ? `rgba(255,71,87,${alpha})` : isHigh ? `rgba(255,107,53,${alpha * 0.75})` : `rgba(0,212,255,${alpha * 0.55})`
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
        const color = n.type === 'critical' ? '#ff4757' : n.type === 'high' ? '#ff6b35' : '#00d4ff'
        const alpha = n.type === 'critical' ? 0.92 : n.type === 'high' ? 0.72 : 0.48
        if (n.type === 'critical') {
          const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pr * 7)
          grd.addColorStop(0, 'rgba(255,71,87,0.28)')
          grd.addColorStop(1, 'rgba(255,71,87,0)')
          ctx.beginPath(); ctx.arc(n.x, n.y, pr * 7, 0, Math.PI * 2)
          ctx.fillStyle = grd; ctx.fill()
        }
        ctx.beginPath(); ctx.arc(n.x, n.y, pr, 0, Math.PI * 2)
        ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0')
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
    { icon: '👨‍💻', label: 'Developer Push',  sub: 'git commit',        color: '#00d4ff' },
    { icon: '🔍', label: 'Scanners',          sub: 'SAST · DAST · SCA', color: '#ffd32a' },
    { icon: '🗄️', label: 'DefectDojo',        sub: 'Findings DB',       color: '#ff6b35' },
    { icon: '🏷️', label: 'Tag Engine',        sub: 'Context tagging',   color: '#2ed573' },
    { icon: '🤖', label: 'AI Scorer',         sub: 'LightGBM',          color: '#00d4ff' },
    { icon: '🚦', label: 'Security Gate',     sub: 'Block / Allow',     color: '#ff4757' },
  ]

  const features = [
    { icon: '🔬', color: '#00d4ff', title: 'AI Risk Scoring',     desc: 'XGBoost · LightGBM · RandomForest ensemble. Each finding gets a 0–4 risk score weighted by business context, not just CVSS severity.' },
    { icon: '🏷️', color: '#2ed573', title: 'Intelligent Tagging', desc: 'Automatic detection of production, external-facing, sensitive, and urgent findings. Tags feed directly into the ML scoring pipeline.' },
    { icon: '⚡', color: '#ffd32a', title: 'Security Gate',       desc: 'CI/CD integration automatically blocks deployments when any finding scores ≥ 3 (High or Critical). Zero-touch protection.' },
    { icon: '🧠', color: '#ff6b35', title: 'SHAP Explainability', desc: 'Every AI decision is fully explained. See which features drove the score — no black boxes, full transparency for your security team.' },
    { icon: '📡', color: '#ff4757', title: 'Scanner Unification', desc: 'Semgrep (SAST), OWASP ZAP (DAST), and Snyk (SCA) results unified via DefectDojo. One prioritized source of truth.' },
    { icon: '📊', color: '#a29bfe', title: 'Live Dashboard',      desc: 'Real-time metrics, trend charts, scanner breakdown, and pipeline event feed. Full visibility over your security posture.' },
  ]

  const termLines = [
    { text: '$ semgrep --config auto ./src --output findings.json',       color: 'rgba(255,255,255,0.45)' },
    { text: '✓  Semgrep SAST: 487 findings — 0 new Critical',             color: '#2ed573' },
    { text: '$ snyk test --severity-threshold=medium',                    color: 'rgba(255,255,255,0.45)' },
    { text: '⚠  Snyk SCA: 2 High vulnerabilities in node_modules',       color: '#ffd32a' },
    { text: '$ python main.py tag --engagement-id 5',                     color: 'rgba(255,255,255,0.45)' },
    { text: '🏷️  Tagged: production=847  external=312  sensitive=201',     color: '#00d4ff' },
    { text: '$ python main.py predict --engagement-id 5',                 color: 'rgba(255,255,255,0.45)' },
    { text: '🤖 AI Scored: 1311 findings · max_risk=4 (Critical)',        color: '#00d4ff' },
    { text: '⛔ SECURITY GATE: score 4 ≥ threshold 3 — DEPLOY BLOCKED',  color: '#ff4757' },
  ]

  const riskClasses = [
    { score: '0', color: '#00d4ff', label: 'Info' },
    { score: '1', color: '#2ed573', label: 'Low'  },
    { score: '2', color: '#ffd32a', label: 'Med'  },
    { score: '3', color: '#ff6b35', label: 'High' },
    { score: '4', color: '#ff4757', label: 'Crit' },
  ]

  const techPills = [
    { label: 'LightGBM', color: '#00d4ff' }, { label: 'FastAPI',    color: '#2ed573' },
    { label: 'SHAP',     color: '#ffd32a' }, { label: 'DefectDojo', color: '#ff6b35' },
    { label: 'React',    color: '#00d4ff' }, { label: 'Prometheus', color: '#ff4757' },
    { label: 'Docker',   color: '#2ed573' }, { label: 'Semgrep',    color: '#a29bfe' },
    { label: 'Snyk',     color: '#ffd32a' },
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
        <Link to="/login" className="navbar-cta">Open Dashboard →</Link>
      </nav>

      {/* ── HERO ── */}
      <section className="hero">
        <div className="hero-badge">
          <div className="hero-badge-dot" />
          <span className="hero-badge-text">AI-POWERED DEVSECOPS INTELLIGENCE PLATFORM</span>
        </div>
        <h1 className="hero-headline">
          <span className="hero-headline-plain">Make Every</span>
          <span className="hero-headline-shimmer">Threat Visible.</span>
        </h1>
        <div className="hero-typewriter">
          <TypeWriter texts={[
            'Scoring vulnerabilities with AI.',
            'SAST · DAST · SCA — unified.',
            'Context-aware risk analysis.',
            'LightGBM · F1=0.8937 · 23 features.',
            'Block critical threats before they ship.',
          ]} />
        </div>
        <p className="hero-description">
          InvisiThreat assigns AI-generated risk scores to every security finding from your scanners,
          factoring in production exposure, sensitive data context, and business impact —
          so your team acts on what truly matters.
        </p>
        <div className="hero-cta-group" id="launch">
          <button className="btn-primary" onClick={() => alert('Run: python main.py serve --port 8000\nthen: npm run dev')}>
            <span>🚀</span> Launch Dashboard
          </button>
          <button className="btn-ghost" onClick={() => scrollTo('architecture')}>
            <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>▶</span> How it works
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
            { to: 1311, suffix: '',  label: 'Findings Analyzed', color: 'var(--accent)'  },
            { to: 90,   suffix: '%', label: 'Model Accuracy',    color: 'var(--green)'   },
            { to: 1125, suffix: '',  label: 'Findings Tagged',   color: 'var(--orange)'  },
            { to: 23,   suffix: '',  label: 'ML Features',       color: 'var(--accent3)' },
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
            <div className="section-label">Capabilities</div>
            <h2 className="section-title">Everything your security team<br /><span>actually needs.</span></h2>
            <p className="section-subtitle">From raw scanner output to prioritized, context-aware risk scores — automatically, in seconds.</p>
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
            <h2 className="section-title">How <span>InvisiThreat</span> works</h2>
            <p className="section-subtitle">A fully automated pipeline from code commit to risk-scored findings — zero manual steps.</p>
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
            <div className="section-label">Model Performance</div>
            <h2 className="model-title">Production-grade AI.<br /><span>Not just a proof of concept.</span></h2>
            <p className="model-description">
              Trained on 1 311 real-world security findings with 23 engineered features.
              The LightGBM model was selected automatically via cross-validation,
              outperforming XGBoost and RandomForest on all metrics.
            </p>
            <div className="model-bars">
              <ModelBar label="F1-Weighted Score" value={89.4} color="#00d4ff" />
              <ModelBar label="ROC-AUC Score"     value={96.0} color="#2ed573" />
              <ModelBar label="Overall Accuracy"  value={90.0} color="#ffd32a" />
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
              {[{k:'F1-weighted',v:'0.8937',c:'#00d4ff'},{k:'ROC-AUC',v:'0.9603',c:'#2ed573'},{k:'Accuracy',v:'90.0%',c:'#ffd32a'},{k:'Features',v:'23',c:'#ff6b35'}].map(({ k, v, c }) => (
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
          <div className="section-label">CI/CD Integration</div>
          <h2 className="section-title" style={{ textAlign: 'center' }}>
            Zero-touch <span style={{ color: 'var(--accent3)' }}>security automation.</span>
          </h2>
          <p className="section-subtitle" style={{ margin: '0 auto' }}>
            Push code — scanners run, AI scores every finding, deploy is blocked if risk ≥ 3. No human intervention required.
          </p>
          <div className="terminal">
            <div className="terminal-header">
              {['#ff5f56','#ffbd2e','#27c93f'].map((bg, i) => <div key={i} className="terminal-dot" style={{ background: bg, opacity: 0.82 }} />)}
              <span className="terminal-title">github-actions / security-pipeline.yml — main branch</span>
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
        <div className="section-label">Get Started</div>
        <h2 className="cta-title">Ready to see your<br /><span>threats clearly?</span></h2>
        <p className="cta-desc">
          Start the backend, open the dashboard, and get AI-powered risk scores on every vulnerability — in under 60 seconds.
        </p>
        <div className="cta-steps">
          {[
            { step: '01', cmd: 'python main.py serve', color: '#00d4ff' },
            { step: '02', cmd: 'npm run dev',          color: '#2ed573' },
            { step: '03', cmd: 'localhost:5173',       color: '#ffd32a' },
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