import { useState, useRef, useEffect } from "react";
import { useFindingMetadata } from "../hooks/useFindingMetadata";
import { UserAvatar } from "../../../components/layout/UserAvatar";

interface AppUser {
  id:          number;
  username:    string;
  role:        string;
  avatar_url?: string | null;
}

interface Props {
  findingId:    number;
  findingTitle: string;
  severity?:    string;
  aiRiskScore?: number;
  users:        AppUser[];
  metadata:     any;
  onRefresh:    () => void;
  canManage?:   boolean;
}

const STATUS_COLORS: Record<string, string> = {
  open:        "var(--dimmed)",
  in_progress: "var(--severity-medium)",
  resolved:    "var(--severity-low)",
  wont_fix:    "var(--severity-critical)",
};

const STATUS_BG: Record<string, string> = {
  open:        "var(--bg4)",
  in_progress: "var(--severity-medium-bg)",
  resolved:    "var(--severity-low-bg)",
  wont_fix:    "var(--severity-critical-bg)",
};

const STATUS_LABELS: Record<string, string> = {
  open:        "Open",
  in_progress: "En cours",
  resolved:    "Résolu",
  wont_fix:    "Won't fix",
};

const STATUS_ICONS: Record<string, string> = {
  open:        "",
  in_progress: "",
  resolved:    "",
  wont_fix:    "",
};

export default function FindingRowActions({
  findingId, findingTitle, severity, aiRiskScore,
  users, metadata, onRefresh, canManage = false,
}: Props) {
  const { pin, unpin, assign, unassign, updateStatus } = useFindingMetadata(findingId);

  const [showAssign, setShowAssign] = useState(false);
  const [showStatus, setShowStatus] = useState(false);
  const [busy,       setBusy]       = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setShowAssign(false);
        setShowStatus(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handlePin = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (busy) return;
    setBusy(true);
    try {
      if (metadata?.is_pinned) await unpin();
      else await pin({ finding_title: findingTitle, severity, ai_risk_score: aiRiskScore });
      onRefresh();
    } finally { setBusy(false); }
  };

  const handleAssign = async (userId: number) => {
    setBusy(true);
    setShowAssign(false);
    try {
      await assign({ assigned_to_id: userId, finding_title: findingTitle, severity, ai_risk_score: aiRiskScore });
      onRefresh();
    } finally { setBusy(false); }
  };

  const handleUnassign = async () => {
    setBusy(true);
    setShowAssign(false);
    try { await unassign(); onRefresh(); }
    finally { setBusy(false); }
  };

  const handleStatus = async (status: string) => {
    setBusy(true);
    setShowStatus(false);
    try { await updateStatus(status); onRefresh(); }
    finally { setBusy(false); }
  };

  const currentStatus = metadata?.status || "open";

  /* user object pour avatar depuis les métadonnées (nom seul, pas d'avatar_url) */
  const assignedUser = metadata?.assigned_to
    ? users.find(u => u.username === metadata.assigned_to) ?? { username: metadata.assigned_to, role: "developer" }
    : null;

  return (
    <div
      ref={ref}
      style={{ display: "flex", alignItems: "center", gap: "6px" }}
      onClick={e => e.stopPropagation()}
    >

      {/* ── Épingler (manager/admin uniquement) ── */}
      {canManage ? (
        <button
          onClick={handlePin}
          disabled={busy}
          title={metadata?.is_pinned ? `Épinglé par ${metadata.pinned_by}` : "Épingler"}
          style={{
            background:  "none",
            border:      "none",
            cursor:      busy ? "not-allowed" : "pointer",
            padding:     "4px",
            fontSize:    "15px",
            opacity:     metadata?.is_pinned ? 1 : 0.22,
            transition:  "opacity 0.2s",
            flexShrink:  0,
          }}
        >
          📍
        </button>
      ) : (
        /* Lecture seule : afficher l'épingle si épinglé */
        metadata?.is_pinned && (
          <span
            title={`Épinglé par ${metadata.pinned_by}`}
            style={{ fontSize: "13px", opacity: 0.45, flexShrink: 0 }}
          >
            📍
          </span>
        )
      )}

      {/* ── Assigner ── */}
      {canManage ? (
        <div style={{ position: "relative" }}>
          <button
            onClick={e => { e.stopPropagation(); setShowAssign(!showAssign); setShowStatus(false); }}
            disabled={busy}
            title={assignedUser ? `Assigné à ${assignedUser.username}` : "Assigner"}
            style={{
              background:  "none",
              border:      "none",
              cursor:      busy ? "not-allowed" : "pointer",
              padding:     "0",
              display:     "flex",
              alignItems:  "center",
              flexShrink:  0,
            }}
          >
            {assignedUser ? (
              <UserAvatar user={assignedUser} size={35} />
            ) : (
              <div style={{
                width:          26,
                height:         26,
                borderRadius:   "50%",
                border:         "1px dashed var(--border2)",
                display:        "flex",
                alignItems:     "center",
                justifyContent: "center",
                color:          "var(--dimmed)",
                fontSize:       13,
                flexShrink:     0,
              }}>
                +
              </div>
            )}
          </button>

          {showAssign && (
            <div style={{
              position:     "absolute",
              top:          "calc(100% + 6px)",
              right:        0,
              background:   "var(--bg2)",
              border:       "1px solid var(--border2)",
              borderRadius: "var(--radius-lg)",
              minWidth:     "200px",
              zIndex:       999,
              boxShadow:    "var(--shadow-lg)",
              overflow:     "hidden",
            }}>
              {/* Retirer l'assignation */}
              {assignedUser && (
                <div
                  onClick={handleUnassign}
                  style={{
                    padding:      "9px 12px",
                    color:        "var(--severity-critical)",
                    fontSize:     "12px",
                    cursor:       "pointer",
                    borderBottom: "1px solid var(--border)",
                    fontFamily:   "var(--font-body)",
                  }}
                >
                  ✕ Retirer l'assignation
                </div>
              )}

              {/* Liste des users */}
              {users.map(u => (
                <div
                  key={u.id}
                  onClick={() => handleAssign(u.id)}
                  style={{
                    padding:        "8px 12px",
                    fontSize:       "12px",
                    cursor:         "pointer",
                    color:          "var(--text)",
                    display:        "flex",
                    alignItems:     "center",
                    justifyContent: "space-between",
                    gap:            "8px",
                    transition:     "background var(--transition-fast)",
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--bg3)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}
                >
                  <span style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <UserAvatar user={u} size={22} />
                    <span style={{ fontFamily: "var(--font-body)" }}>{u.username}</span>
                  </span>
                  <span style={{
                    fontSize:      "10px",
                    color:         "var(--dimmed)",
                    fontFamily:    "var(--font-mono)",
                    letterSpacing: "0.05em",
                  }}>
                    {u.role}
                  </span>
                </div>
              ))}

              {users.length === 0 && (
                <div style={{ padding: "10px 12px", color: "var(--dimmed)", fontSize: "12px" }}>
                  Aucun utilisateur
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        /* Lecture seule : avatar si assigné */
        assignedUser && (
          <UserAvatar user={assignedUser} size={24} />
        )
      )}

      {/* ── Statut ── */}
      {canManage ? (
        <div style={{ position: "relative" }}>
          <button
            onClick={e => { e.stopPropagation(); setShowStatus(!showStatus); setShowAssign(false); }}
            disabled={busy}
            title={`Statut : ${STATUS_LABELS[currentStatus]}`}
            style={{
              background:   STATUS_BG[currentStatus],
              border:       `1px solid ${STATUS_COLORS[currentStatus]}44`,
              borderRadius: "var(--radius-sm)",
              cursor:       busy ? "not-allowed" : "pointer",
              padding:      "3px 8px",
              fontSize:     "10px",
              fontWeight:   600,
              color:        STATUS_COLORS[currentStatus],
              fontFamily:   "var(--font-body)",
              letterSpacing:"0.04em",
              textTransform:"uppercase",
              whiteSpace:   "nowrap",
              flexShrink:   0,
            }}
          >
            {STATUS_ICONS[currentStatus]} {STATUS_LABELS[currentStatus]}
          </button>

          {showStatus && (
            <div style={{
              position:     "absolute",
              top:          "calc(100% + 6px)",
              right:        0,
              background:   "var(--bg2)",
              border:       "1px solid var(--border2)",
              borderRadius: "var(--radius-lg)",
              minWidth:     "160px",
              zIndex:       999,
              boxShadow:    "var(--shadow-lg)",
              overflow:     "hidden",
            }}>
              {Object.entries(STATUS_LABELS).map(([key, label]) => (
                <div
                  key={key}
                  onClick={() => handleStatus(key)}
                  style={{
                    padding:     "9px 12px",
                    fontSize:    "12px",
                    cursor:      "pointer",
                    color:       STATUS_COLORS[key],
                    background:  currentStatus === key ? STATUS_BG[key] : "transparent",
                    fontFamily:  "var(--font-body)",
                    fontWeight:  currentStatus === key ? 600 : 400,
                    display:     "flex",
                    alignItems:  "center",
                    gap:         "8px",
                    transition:  "background var(--transition-fast)",
                  }}
                  onMouseEnter={e => { if (currentStatus !== key) e.currentTarget.style.background = "var(--bg3)"; }}
                  onMouseLeave={e => { e.currentTarget.style.background = currentStatus === key ? STATUS_BG[key] : "transparent"; }}
                >
                  {STATUS_ICONS[key]} {label}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        /* Lecture seule : badge statut non cliquable */
        <span style={{
          background:    STATUS_BG[currentStatus],
          border:        `1px solid ${STATUS_COLORS[currentStatus]}44`,
          borderRadius:  "var(--radius-sm)",
          padding:       "3px 8px",
          fontSize:      "10px",
          fontWeight:    600,
          color:         STATUS_COLORS[currentStatus],
          fontFamily:    "var(--font-body)",
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          whiteSpace:    "nowrap",
          flexShrink:    0,
          opacity:       0.7,
        }}>
          {STATUS_ICONS[currentStatus]} {STATUS_LABELS[currentStatus]}
        </span>
      )}
    </div>
  );
}