// src/components/ui/ParticleCanvas.tsx
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
    const ctx = canvas.getContext('2d')!;
    let animId: number | null = null; // ✅ initialisé à null

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const nodes: Particle[] = Array.from({ length: 58 }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      vx: (Math.random() - 0.5) * 0.22,
      vy: (Math.random() - 0.5) * 0.22,
      r: Math.random() * 1.8 + 0.8,
      pulse: Math.random() * Math.PI * 2,
      type: Math.random() < 0.06 ? 'critical' : Math.random() < 0.14 ? 'high' : 'normal',
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Mise à jour des positions
      nodes.forEach(n => {
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += 0.018;
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });

      // Connexions entre particules
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 145) {
            const alpha = (1 - dist / 145) * 0.18;
            const isCrit = nodes[i].type === 'critical' || nodes[j].type === 'critical';
            const isHigh = nodes[i].type === 'high' || nodes[j].type === 'high';
            ctx.strokeStyle = isCrit
              ? `rgba(255,71,87,${alpha})`
              : isHigh
              ? `rgba(255,107,53,${alpha * 0.75})`
              : `rgba(0,212,255,${alpha * 0.55})`;
            ctx.lineWidth = isCrit ? 0.85 : 0.4;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      // Dessin des particules
      nodes.forEach(n => {
        const pr = n.r + Math.sin(n.pulse) * 0.7;
        const color = n.type === 'critical' ? '#ff4757' : n.type === 'high' ? '#ff6b35' : '#00d4ff';
        const alpha = n.type === 'critical' ? 0.92 : n.type === 'high' ? 0.72 : 0.48;

        if (n.type === 'critical') {
          const grd = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, pr * 7);
          grd.addColorStop(0, 'rgba(255,71,87,0.28)');
          grd.addColorStop(1, 'rgba(255,71,87,0)');
          ctx.beginPath();
          ctx.arc(n.x, n.y, pr * 7, 0, Math.PI * 2);
          ctx.fillStyle = grd;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, pr, 0, Math.PI * 2);
        ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
        ctx.fill();
      });

      animId = requestAnimationFrame(draw); // ✅ assignation correcte
    };

    draw();

    return () => {
      if (animId !== null) cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="particle-canvas" />;
}