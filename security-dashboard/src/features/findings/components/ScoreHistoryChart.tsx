import { useEffect, useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
  LabelList,
} from "recharts";
import { apiClient } from "../../../api/client";

interface HistoryPoint {
  score:      number;
  level:      string;
  confidence: number;
  date:       string;
}

interface Props {
  findingId: number;
}

/* ── CSS var helpers — recharts ne lit pas les CSS vars, on les résout via getComputedStyle ── */
function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ── Couleur severity via les variables du thème ── */
function levelCssVar(level: string): { color: string; bg: string; border: string } {
  switch (level?.toLowerCase()) {
    case "critical": return {
      color:  cssVar("--severity-critical"),
      bg:     cssVar("--severity-critical-bg"),
      border: cssVar("--severity-critical-border"),
    };
    case "high": return {
      color:  cssVar("--severity-high"),
      bg:     cssVar("--severity-high-bg"),
      border: cssVar("--severity-high-border"),
    };
    case "medium": return {
      color:  cssVar("--severity-medium"),
      bg:     cssVar("--severity-medium-bg"),
      border: cssVar("--severity-medium-border"),
    };
    case "low": return {
      color:  cssVar("--severity-low"),
      bg:     cssVar("--severity-low-bg"),
      border: cssVar("--severity-low-border"),
    };
    default: return {
      color:  cssVar("--accent"),
      bg:     cssVar("--accent-muted"),
      border: cssVar("--accent-border"),
    };
  }
}

/* ── Tooltip custom ── */
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  const sev = d?.level ? levelCssVar(d.level) : null;

  return (
    <div style={{
      background:   "var(--bg2)",
      border:       "1px solid var(--border-accent)",
      borderRadius: "var(--radius-md)",
      padding:      "10px 14px",
      fontSize:     "12px",
      color:        "var(--text)",
      minWidth:     "140px",
      fontFamily:   "var(--font-body)",
      boxShadow:    "var(--shadow-md)",
    }}>
      <div style={{ color: "var(--dimmed)", marginBottom: "8px", fontSize: "11px" }}>{label}</div>

      <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", marginBottom: "4px" }}>
        <span style={{ color: "var(--muted)" }}>Score IA</span>
        <span style={{ fontWeight: 600, color: "var(--accent)" }}>{d?.score} / 10</span>
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", gap: "16px", marginBottom: "10px" }}>
        <span style={{ color: "var(--muted)" }}>Confiance</span>
        <span style={{ color: "var(--accent)", opacity: 0.6 }}>{d?.confidence}%</span>
      </div>

      {sev && d?.level && (
        <div style={{
          display:       "inline-block",
          padding:       "2px 9px",
          borderRadius:  "var(--radius-pill)",
          fontSize:      "10px",
          fontWeight:    600,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          background:    sev.bg,
          color:         sev.color,
          border:        `1px solid ${sev.border}`,
        }}>
          {d.level}
        </div>
      )}
    </div>
  );
};

/* ── Label score sur les dots ── */
const ScoreLabel = ({ x, y, value }: any) => {
  if (value === undefined || value === null) return null;
  return (
    <text
      x={x}
      y={y - 10}
      textAnchor="middle"
      fill={cssVar("--accent")}
      fontSize={10}
      fontWeight={600}
      fontFamily={cssVar("--font-mono")}
    >
      {value}
    </text>
  );
};

/* ── Stat pill ── */
function StatPill({
  label, value, varColor,
}: { label: string; value: string; varColor: string }) {
  return (
    <div style={{
      background:   "var(--bg3)",
      border:       "1px solid var(--border)",
      borderRadius: "var(--radius-md)",
      padding:      "8px 14px",
      textAlign:    "center",
      transition:   "border-color var(--transition-fast)",
    }}>
      <div style={{
        color:         "var(--dimmed)",
        fontSize:      "10px",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        fontFamily:    "var(--font-body)",
        fontWeight:    600,
        marginBottom:  "3px",
      }}>
        {label}
      </div>
      <div style={{
        color:      varColor,
        fontSize:   "15px",
        fontWeight: 700,
        fontFamily: "var(--font-display)",
      }}>
        {value}
      </div>
    </div>
  );
}

export default function ScoreHistoryChart({ findingId }: Props) {
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        const res = await apiClient.get(`/analytics/history/${findingId}`);
        setHistory(res.data.history);
      } catch (err) {
        console.error("Erreur historique", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [findingId]);

  if (loading) {
    return (
      <div style={{
        color:      "var(--dimmed)",
        textAlign:  "center",
        padding:    "30px",
        fontSize:   "13px",
        fontFamily: "var(--font-body)",
      }}>
        Chargement de l'historique…
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div style={{
        textAlign:    "center",
        color:        "var(--dimmed)",
        padding:      "30px",
        background:   "var(--bg3)",
        borderRadius: "var(--radius-lg)",
        border:       "1px solid var(--border)",
        fontSize:     "13px",
        fontFamily:   "var(--font-body)",
      }}>
        Aucun historique disponible pour ce finding
      </div>
    );
  }

  const chartData = history.map(h => ({
    date:       new Date(h.date).toLocaleDateString("fr-FR", { day: "2-digit", month: "short" }),
    score:      parseFloat(h.score.toFixed(2)),
    confidence: parseFloat((h.confidence * 100).toFixed(0)),
    level:      h.level,
  }));

  const scores    = history.map(h => h.score);
  const minScore  = Math.min(...scores);
  const maxScore  = Math.max(...scores);
  const avgScore  = scores.reduce((a, b) => a + b, 0) / scores.length;
  const lastScore = history[history.length - 1]?.score || 0;
  const firstScore = history[0]?.score || 0;
  const trend     = lastScore - firstScore;

  const lastLevel = history[history.length - 1]?.level;
  const lastSev   = lastLevel ? levelCssVar(lastLevel) : null;

  /* Couleurs hardcodées pour recharts (ne lit pas les CSS vars) — calées sur le dark theme */
  const ACCENT         = "#00d4ff";
  const ACCENT_MUTED   = "rgba(0,212,255,0.12)";
  const GRID           = "rgba(148,163,184,0.08)";
  const AXIS_LINE      = "rgba(148,163,184,0.12)";
  const TICK_COLOR     = "#5a657a";
  const CRITICAL_STROKE = "rgba(255,71,87,0.40)";
  const MEDIUM_STROKE   = "rgba(255,211,42,0.40)";

  return (
    <div style={{
      background:   "var(--bg3)",
      border:       "1px solid var(--border)",
      borderRadius: "var(--radius-lg)",
      padding:      "var(--space-5)",
    }}>

      {/* ── Header ── */}
      <div style={{
        display:        "flex",
        justifyContent: "space-between",
        alignItems:     "flex-start",
        marginBottom:   "var(--space-4)",
      }}>
        <div>
          {/* Label section */}
          <div className="label" style={{ marginBottom: "6px" }}>
            Historique du score IA
          </div>

          {/* Score courant + badge niveau */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{
              color:      "var(--text-strong)",
              fontSize:   "22px",
              fontWeight: 700,
              fontFamily: "var(--font-display)",
              letterSpacing: "-0.02em",
            }}>
              {lastScore.toFixed(2)}
              <span style={{
                fontSize:   "13px",
                color:      "var(--dimmed)",
                fontWeight: 400,
                fontFamily: "var(--font-body)",
              }}> / 10</span>
            </span>

            {lastLevel && lastSev && (
              <span className={`badge badge-${lastLevel.toLowerCase()}`}>
                {lastLevel}
              </span>
            )}
          </div>

          <div style={{ color: "var(--dimmed)", fontSize: "11px", marginTop: "3px", fontFamily: "var(--font-body)" }}>
            {history.length} mesure{history.length > 1 ? "s" : ""}
          </div>
        </div>

        {/* Tendance */}
        <div style={{
          display:      "flex",
          alignItems:   "center",
          gap:          "5px",
          padding:      "6px 12px",
          background:   trend > 0
            ? "var(--severity-critical-bg)"
            : trend < 0
            ? "var(--severity-low-bg)"
            : "var(--bg4)",
          borderRadius: "var(--radius-pill)",
          border: trend > 0
            ? "1px solid var(--severity-critical-border)"
            : trend < 0
            ? "1px solid var(--severity-low-border)"
            : "1px solid var(--border2)",
        }}>
          <span style={{
            color:      trend > 0 ? "var(--severity-critical)" : trend < 0 ? "var(--severity-low)" : "var(--dimmed)",
            fontSize:   "12px",
            fontWeight: 700,
          }}>
            {trend > 0 ? "▲" : trend < 0 ? "▼" : "→"}
          </span>
          <span style={{
            color:      trend > 0 ? "var(--severity-critical)" : trend < 0 ? "var(--severity-low)" : "var(--dimmed)",
            fontSize:   "12px",
            fontWeight: 600,
            fontFamily: "var(--font-mono)",
          }}>
            {trend > 0 ? "+" : ""}{trend.toFixed(2)}
          </span>
        </div>
      </div>

      {/* ── Mini stats ── */}
      <div style={{
        display:             "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap:                 "var(--space-2)",
        marginBottom:        "var(--space-4)",
      }}>
        <StatPill label="Min"     value={minScore.toFixed(2)} varColor="var(--severity-low)"      />
        <StatPill label="Moyenne" value={avgScore.toFixed(2)} varColor="var(--accent)"             />
        <StatPill label="Max"     value={maxScore.toFixed(2)} varColor="var(--severity-critical)"  />
      </div>

      {/* ── Graphique ── */}
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={chartData} margin={{ top: 22, right: 12, left: -10, bottom: 0 }}>
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={ACCENT} stopOpacity={0.18} />
              <stop offset="95%" stopColor={ACCENT} stopOpacity={0}    />
            </linearGradient>
            <linearGradient id="confGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={ACCENT} stopOpacity={0.06} />
              <stop offset="95%" stopColor={ACCENT} stopOpacity={0}    />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke={GRID} />

          <XAxis
            dataKey="date"
            tick={{ fill: TICK_COLOR, fontSize: 11, fontFamily: "DM Sans, sans-serif" }}
            axisLine={{ stroke: AXIS_LINE }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 10]}
            tick={{ fill: TICK_COLOR, fontSize: 11, fontFamily: "DM Sans, sans-serif" }}
            axisLine={{ stroke: AXIS_LINE }}
            tickLine={false}
            tickCount={6}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Seuil Critical = 7 */}
          <ReferenceLine
            y={7}
            stroke={CRITICAL_STROKE}
            strokeDasharray="4 4"
            label={{
              value:    "Critical ×7",
              fill:     "#ff4757aa",
              fontSize: 10,
              position: "insideTopRight",
              fontFamily: "DM Sans, sans-serif",
            }}
          />

          {/* Seuil Medium = 4 */}
          <ReferenceLine
            y={4}
            stroke={MEDIUM_STROKE}
            strokeDasharray="4 4"
            label={{
              value:    "Medium ×4",
              fill:     "#ffd32aaa",
              fontSize: 10,
              position: "insideTopRight",
              fontFamily: "DM Sans, sans-serif",
            }}
          />

          {/* Courbe confiance (arrière-plan, pointillés) */}
          <Area
            type="monotone"
            dataKey="confidence"
            stroke={ACCENT_MUTED}
            strokeWidth={1}
            strokeDasharray="4 4"
            fill="url(#confGradient)"
            dot={false}
            activeDot={false}
          />

          {/* Courbe score (premier plan) */}
          <Area
            type="monotone"
            dataKey="score"
            stroke={ACCENT}
            strokeWidth={2}
            fill="url(#scoreGradient)"
            dot={{ fill: ACCENT, r: 4, strokeWidth: 0 }}
            activeDot={{ r: 6, fill: "#fff", stroke: ACCENT, strokeWidth: 2 }}
          >
            <LabelList dataKey="score" content={<ScoreLabel />} />
          </Area>
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}