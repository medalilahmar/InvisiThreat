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
    const C = { critical: '255,71,87', high: '255,107,53', accent: '0,212,255' }
    const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight }
    resize()
    window.addEventListener('resize', resize)
    const nodes: Particle[] = Array.from({ length: 58 }, () => ({
      x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.22, vy: (Math.random() - 0.5) * 0.22,
      r: Math.random() * 1.8 + 0.8, pulse: Math.random() * Math.PI * 2,
      type: Math.random() < 0.06 ? 'critical' : Math.random() < 0.14 ? 'high' : 'normal',
    }))
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      nodes.forEach(n => {
        n.x += n.vx; n.y += n.vy; n.pulse += 0.018
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1
      })
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < 145) {
            const alpha = (1 - dist / 145) * 0.18
            const isCrit = nodes[i].type === 'critical' || nodes[j].type === 'critical'
            const isHigh = nodes[i].type === 'high' || nodes[j].type === 'high'
            ctx.strokeStyle = isCrit ? `rgba(${C.critical},${alpha})` : isHigh ? `rgba(${C.high},${alpha * 0.75})` : `rgba(${C.accent},${alpha * 0.55})`
            ctx.lineWidth = isCrit ? 0.85 : 0.4
            ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke()
          }
        }
      }
      nodes.forEach(n => {
        const pr = n.r + Math.sin(n.pulse) * 0.7
        const color = n.type === 'critical' ? `rgb(${C.critical})` : n.type === 'high' ? `rgb(${C.high})` : `rgb(${C.accent})`
        const alpha = n.type === 'critical' ? 0.92 : n.type === 'high' ? 0.72 : 0.48
        if (n.type === 'critical') {
          const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pr * 7)
          grd.addColorStop(0, `rgba(${C.critical},0.28)`); grd.addColorStop(1, `rgba(${C.critical},0)`)
          ctx.beginPath(); ctx.arc(n.x, n.y, pr * 7, 0, Math.PI * 2); ctx.fillStyle = grd; ctx.fill()
        }
        ctx.beginPath(); ctx.arc(n.x, n.y, pr, 0, Math.PI * 2)
        ctx.fillStyle = color.replace(')', `,${alpha})`).replace('rgb', 'rgba'); ctx.fill()
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
  const elRef = useRef<HTMLSpanElement>(null)
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
  const [display, setDisplay] = useState('')
  const [idx, setIdx] = useState(0)
  const [charIdx, setCharIdx] = useState(0)
  const [deleting, setDeleting] = useState(false)
  useEffect(() => {
    const cur = texts[idx]
    if (!deleting && charIdx < cur.length) { const t = setTimeout(() => setCharIdx(c => c + 1), speed); return () => clearTimeout(t) }
    if (!deleting && charIdx === cur.length) { const t = setTimeout(() => setDeleting(true), 2400); return () => clearTimeout(t) }
    if (deleting && charIdx > 0) { const t = setTimeout(() => setCharIdx(c => c - 1), speed / 2.2); return () => clearTimeout(t) }
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

  // ── Données ───────────────────────────────────────────────────────────────

  const features = [
    { icon: '🔬', color: 'var(--accent)',  title: 'Notation de Risque par IA',   desc: 'LightGBM · XGBoost · RandomForest. Chaque finding reçoit un score de 0 à 4 basé sur le contexte métier et l\'exposition réelle — pas seulement le CVSS.' },
    { icon: '🏷️', color: 'var(--green)',  title: 'Étiquetage Intelligent',       desc: 'Tags auto-générés : production, external, sensitive, pii, blocker — détectés depuis la branche Git, les résultats ZAP et DefectDojo. Alimentent le scoring ML.' },
    { icon: '⚡', color: 'var(--accent3)', title: 'Porte de Sécurité CI/CD',      desc: 'Le déploiement est bloqué automatiquement si un finding atteint un score ≥ 3. Semgrep, Snyk et ZAP s\'exécutent en parallèle sur chaque projet modifié.' },
    { icon: '🧠', color: 'var(--orange)', title: 'Explicabilité SHAP',           desc: 'Chaque décision IA est entièrement expliquée. Visualisez les 23 features qui ont influencé le score — zéro boîte noire pour votre équipe de sécurité.' },
    { icon: '🔧', color: 'var(--accent2)', title: 'Auto-Fix par IA',             desc: 'L\'IA génère un correctif pour chaque finding SAST et ouvre automatiquement une Pull Request GitHub — du finding détecté au fix proposé en un seul clic.' },
    { icon: '📊', color: 'var(--purple)', title: 'Tableau de Bord en Temps Réel', desc: 'Métriques live, tendances, répartition par scanner et flux d\'événements pipeline. Déployé sur k3s avec PostgreSQL, Redis et Prometheus.' },
  ]

  // Flux unique : sécurité + CI/CD fusionnés
  const pipelineSteps = [
    { icon: '👨‍💻', label: 'Push',           sub: 'git commit',            color: 'var(--accent)'  },
    { icon: '🔍',  label: 'Scanners',        sub: 'SAST · DAST · SCA',    color: 'var(--accent3)' },
    { icon: '🗄️',  label: 'DefectDojo',      sub: 'Findings DB',           color: 'var(--orange)'  },
    { icon: '🏷️',  label: 'Tagging IA',      sub: 'Étiquetage contextuel', color: 'var(--green)'   },
    { icon: '🤖',  label: 'Scoring IA',      sub: 'LightGBM',              color: 'var(--accent)'  },
    { icon: '🚦',  label: 'Security Gate',   sub: 'Bloquer / Autoriser',   color: 'var(--accent2)' },
    { icon: '🐳',  label: 'Build & Push',    sub: 'Images Docker',         color: 'var(--orange)'  },
    { icon: '☸️',  label: 'Deploy k3s',      sub: 'Kubernetes',            color: 'var(--green)'   },
  ]

  const riskClasses = [
    { score: '0', color: 'var(--accent)',  label: 'Info' },
    { score: '1', color: 'var(--green)',   label: 'Low'  },
    { score: '2', color: 'var(--accent3)', label: 'Med'  },
    { score: '3', color: 'var(--orange)',  label: 'High' },
    { score: '4', color: 'var(--accent2)', label: 'Crit' },
  ]

  const techPills = [
    { label: 'LightGBM',  color: 'var(--accent)'  }, { label: 'FastAPI',    color: 'var(--green)'   },
    { label: 'SHAP',      color: 'var(--accent3)' }, { label: 'DefectDojo', color: 'var(--orange)'  },
    { label: 'React',     color: 'var(--accent)'  }, { label: 'Prometheus', color: 'var(--accent2)' },
    { label: 'Docker',    color: 'var(--green)'   }, { label: 'Semgrep',    color: 'var(--purple)'  },
    { label: 'Ansible',   color: 'var(--accent3)' }, { label: 'Kubernetes', color: 'var(--accent2)' },
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
          <img src={Logo} alt="InvisiThreat Logo" className="navbar-logo-icon" />
          <span className="navbar-logo-text">Invisi<span>Threat</span></span>
        </a>
        <ul className="navbar-links">
          {([
            ['Fonctionnalités', 'features'],
            ['Pipeline',        'pipeline'],
            ['Modèle IA',       'model'],
            ['Infrastructure',  'infra'],
          ] as [string, string][]).map(([lbl, id]) => (
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
          <span className="hero-badge-text">PLATEFORME DEVSECOPS PILOTÉE PAR IA</span>
        </div>
        <h1 className="hero-headline">
          <span className="hero-headline-plain">Rendez Chaque</span>
          <span className="hero-headline-shimmer">Menace Visible.</span>
        </h1>
        <div className="hero-typewriter">
          <TypeWriter texts={[
            'Notation des vulnérabilités par IA.',
            'SAST · DAST · SCA — unifiés.',
            'Auto-Fix : PR GitHub en un clic.',
            'LightGBM · F1=0.8937 · 23 features.',
            'Déployé sur Kubernetes, zéro downtime.',
          ]} />
        </div>
        <p className="hero-description">
          InvisiThreat analyse chaque finding de sécurité avec l'IA — score de risque contextuel,
          étiquetage intelligent, blocage automatique en CI/CD et correction automatique via Pull Request.
          Votre équipe se concentre sur ce qui compte vraiment.
        </p>
        <div className="hero-cta-group" id="launch">
          <Link to="/login" className="btn-primary">
            <span>🛡️</span> Ouvrir le Tableau de Bord
          </Link>
          <button className="btn-ghost" onClick={() => scrollTo('pipeline')}>
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

      {/* ── STATS ── */}
      <div className="stats-ticker">
        <div className="stats-ticker-inner">
          {[
            { to: 1311, suffix: '',  label: 'Findings analysés',   color: 'var(--accent)'  },
            { to: 90,   suffix: '%', label: 'Précision du modèle', color: 'var(--green)'   },
            { to: 1125, suffix: '',  label: 'Findings étiquetés',  color: 'var(--orange)'  },
            { to: 23,   suffix: '',  label: 'Features ML',         color: 'var(--accent3)' },
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
            <p className="section-subtitle">Du finding brut au correctif proposé — automatiquement, sans intervention humaine.</p>
          </div>
          <div className="features-grid">
            {features.map((f, i) => <FeatureCard key={f.title} {...f} delay={i * 90} />)}
          </div>
        </div>
      </section>

      {/* ── PIPELINE COMPLET ── */}
      <section className="arch-section" id="pipeline">
        <div className="section-inner">
          <div className="section-header">
            <div className="section-label">Pipeline</div>
            <h2 className="section-title">Du commit au déploiement<br /><span>sécurisé — en un seul flux.</span></h2>
            <p className="section-subtitle">
              GitHub Actions détecte les projets modifiés, lance les scanners en parallèle,
              score chaque finding par IA, puis bloque ou déploie sur Kubernetes.
            </p>
          </div>

          <div className="arch-flow" style={{ flexWrap: 'nowrap', overflowX: 'auto' }}>
            {pipelineSteps.map((step, i, arr) => (
              <div key={step.label} style={{ display: 'flex', alignItems: 'center' }}>
                <div className="arch-step" style={{ background: `${step.color}08`, border: `1px solid ${step.color}1e` }}>
                  <div className="arch-step-icon" style={{ animationDelay: `${i * 0.28}s` }}>{step.icon}</div>
                  <div className="arch-step-label">{step.label}</div>
                  <div className="arch-step-sub" style={{ color: step.color }}>{step.sub}</div>
                </div>
                {i < arr.length - 1 && (
                  <div className="arch-arrow">
                    <div className="arch-arrow-line" style={{ background: `linear-gradient(90deg, ${step.color}50, ${arr[i + 1].color}50)` }} />
                    <div className="arch-arrow-tip">▶</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Déclencheurs */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap', marginTop: '2rem' }}>
            {[
              { icon: '⚡', label: 'Push sur main',       sub: 'Scan + déploiement automatique',    color: 'var(--accent)'  },
              { icon: '⚙️', label: 'Ansible',        sub: 'Automatisation complète de la plateforme', color: 'var(--accent3)' },
              { icon: '🎛️', label: 'Déclenchement manuel', sub: 'Projet ciblé + force retrain ML',   color: 'var(--green)'   },
            ].map(({ icon, label, sub, color }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 10, background: `${color}06`, border: `1px solid ${color}18`, borderRadius: 10, padding: '0.75rem 1.25rem' }}>
                <span style={{ fontSize: 20 }}>{icon}</span>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color }}>{label}</div>
                  <div style={{ fontSize: 11, color: 'var(--muted)' }}>{sub}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── MODÈLE IA ── */}
      <section className="model-section" id="model">
        <div className="model-section-inner">
          <div className="model-text-side">
            <div className="section-label">Performance du Modèle</div>
            <h2 className="model-title">IA de niveau production.<br /><span>Pas juste une preuve de concept.</span></h2>
            <p className="model-description">
              Entraîné sur 1 311 findings réels avec 23 features élaborées.
              LightGBM sélectionné automatiquement par validation croisée —
              surpasse XGBoost et RandomForest sur toutes les métriques.
            </p>
            <div className="model-bars">
              <ModelBar label="F1-Weighted Score" value={89.4} color="var(--accent)"  />
              <ModelBar label="ROC-AUC Score"     value={96.0} color="var(--green)"   />
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
              {[
                { k: 'F1-weighted', v: '0.8937', c: 'var(--accent)'  },
                { k: 'ROC-AUC',     v: '0.9603', c: 'var(--green)'   },
                { k: 'Accuracy',    v: '90.0%',  c: 'var(--accent3)' },
                { k: 'Features',    v: '23',      c: 'var(--orange)'  },
              ].map(({ k, v, c }) => (
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

      {/* ── FINAL CTA ── */}
      <section className="cta-section" id="infra">
        <div className="cta-glow-orb" />
        <div className="section-label">Prêt à l'emploi</div>
        <h2 className="cta-title">
          Une plateforme complète.<br />
          <span>Découvrez-la maintenant.</span>
        </h2>
        <p className="cta-desc">
          De la détection à la correction automatique — InvisiThreat couvre l'intégralité
          du cycle de sécurité, déployé sur Kubernetes et prêt en production.
        </p>

        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center', flexWrap: 'wrap', marginBottom: '2.5rem' }}>
          {[
            { icon: '🤖', label: 'IA qui score chaque finding',     color: 'var(--accent)'  },
            { icon: '🔧', label: 'Auto-Fix via Pull Request',       color: 'var(--green)'   },
            { icon: '🚦', label: 'Security Gate automatique',       color: 'var(--accent2)' },
            { icon: '☸️', label: 'Kubernetes · Zéro downtime',      color: 'var(--accent3)' },
            { icon: '📊', label: 'Dashboard temps réel',            color: 'var(--orange)'  },
          ].map(({ icon, label, color }) => (
            <div key={label} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: `${color}08`, border: `1px solid ${color}20`,
              borderRadius: 8, padding: '0.5rem 1rem',
            }}>
              <span style={{ fontSize: 15 }}>{icon}</span>
              <span style={{ fontSize: 12, color, fontWeight: 500 }}>{label}</span>
            </div>
          ))}
        </div>

        <div className="cta-buttons">
          <Link to="/login" className="btn-primary" style={{ fontSize: 16, padding: '18px 44px', textDecoration: 'none' }}>
            <span>🛡️</span> Explorer la Plateforme
          </Link>
          <a href="https://github.com" style={{ textDecoration: 'none' }}>
            <button className="btn-ghost" style={{ fontSize: 16, padding: '18px 44px' }}>
              <span>⭐</span> Voir le Code Source
            </button>
          </a>
        </div>
      </section>

    </div>
  )
}