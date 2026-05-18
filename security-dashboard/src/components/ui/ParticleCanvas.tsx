import { useEffect, useRef } from 'react';

type NodeType = 'critical' | 'high' | 'normal';

interface Particle {
  x: number; y: number; vx: number; vy: number;
  r: number; pulse: number; type: NodeType;
}

export function ParticleCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: true })!;
    let animId: number | null = null;

    const isLight = () =>
      document.documentElement.getAttribute('data-theme') === 'light';

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    /* Moins de particules — plus épuré */
    const nodes: Particle[] = Array.from({ length: 38 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.18,   /* plus lent = plus calme */
      vy: (Math.random() - 0.5) * 0.18,
      r: Math.random() * 1.8 + 0.8,        /* plus petites */
      pulse: Math.random() * Math.PI * 2,
      type: Math.random() < 0.05 ? 'critical'
          : Math.random() < 0.12 ? 'high'
          : 'normal',
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const light = isLight();

      /* Opacités selon le thème */
      const connMax  = light ? 0.18 : 0.45;
      const dotAlpha = light ? 0.35 : 0.65;
      const glowA    = light ? 0.12 : 0.40;

      /* Mise à jour positions */
      nodes.forEach(n => {
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.018;
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });

      /* Connexions — seulement les proches */
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx   = nodes[i].x - nodes[j].x;
          const dy   = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const maxD = 140;

          if (dist < maxD) {
            const a      = (1 - dist / maxD) * connMax;
            const isCrit = nodes[i].type === 'critical' || nodes[j].type === 'critical';
            const isHigh = nodes[i].type === 'high'     || nodes[j].type === 'high';

            ctx.strokeStyle = isCrit
              ? `rgba(255,71,87,${a})`
              : isHigh
              ? `rgba(255,107,53,${a})`
              : light
              ? `rgba(0,150,200,${a})`      /* cyan plus sombre en light */
              : `rgba(0,212,255,${a})`;

            ctx.lineWidth = isCrit ? 1.2 : isHigh ? 0.8 : 0.6;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      /* Particules */
      nodes.forEach(n => {
        const pr    = n.r + Math.sin(n.pulse) * 0.6;
        const color = n.type === 'critical' ? '#ff4757'
                    : n.type === 'high'     ? '#ff6b35'
                    : light                 ? '#0097b8'   /* cyan sombre en light */
                    :                         '#00d4ff';

        /* Glow subtil — uniquement critical et high */
        if (n.type !== 'normal') {
          const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pr * 8);
          grd.addColorStop(0, n.type === 'critical'
            ? `rgba(255,71,87,${glowA})`
            : `rgba(255,107,53,${glowA * 0.8})`);
          grd.addColorStop(1, 'transparent');
          ctx.fillStyle = grd;
          ctx.beginPath();
          ctx.arc(n.x, n.y, pr * 8, 0, Math.PI * 2);
          ctx.fill();
        }

        /* Point */
        const a = n.type === 'normal' ? dotAlpha : dotAlpha * 1.2;
        ctx.beginPath();
        ctx.arc(n.x, n.y, pr, 0, Math.PI * 2);
        ctx.fillStyle = color + Math.round(Math.min(a, 1) * 255)
          .toString(16).padStart(2, '0');
        ctx.fill();
      });

      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animId) cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="particle-canvas" />;
}