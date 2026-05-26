import { useState, useEffect } from "react";
import { apiClient } from "../../../api/client";

interface AppUser {
  id:       number;
  username: string;
  email?:   string;
  role:     string;
}

interface Metadata {
  finding_id:     number;
  is_pinned:      boolean;
  pinned_by:      string | null;
  pinned_at:      string | null;
  assigned_to:    string | null;
  assigned_to_id: number | null;
  assigned_by:    string | null;
  assigned_at:    string | null;
  status:         string;
}

interface Props {
  findingId:    number;
  findingTitle: string;
  productName?: string;
  severity?:    string;
  aiRiskScore?: number;
  users:        AppUser[];
}

const STATUS_COLORS: Record<string, string> = {
  open:        "#666",
  in_progress: "#ffa502",
  resolved:    "#2ed573",
  wont_fix:    "#ff4757",
};

const STATUS_LABELS: Record<string, string> = {
  open:        "⬜ Ouvert",
  in_progress: "🔄 En cours",
  resolved:    "✅ Résolu",
  wont_fix:    "🚫 Won't fix",
};

export default function FindingActions({
  findingId, findingTitle, productName, severity, aiRiskScore, users
}: Props) {

  const [metadata,      setMetadata]      = useState<Metadata | null>(null);
  const [loading,       setLoading]       = useState(true);
  const [showAssignMenu, setShowAssignMenu] = useState(false);
  const [showStatusMenu, setShowStatusMenu] = useState(false);
  const [actionLoading,  setActionLoading]  = useState(false);
  const [message,        setMessage]        = useState<string | null>(null);

  // Fetch unique au montage — 1 seul finding, pas de problème de rate limit
  useEffect(() => {
    fetchMetadata();
  }, [findingId]);

  const fetchMetadata = async () => {
    try {
      setLoading(true);
      const res = await apiClient.get(`/findings/${findingId}/metadata`);
      setMetadata(res.data);
    } catch (err) {
      console.error("Erreur metadata", err);
    } finally {
      setLoading(false);
    }
  };

  const showMsg = (msg: string) => {
    setMessage(msg);
    setTimeout(() => setMessage(null), 3000);
  };

  const handlePin = async () => {
    try {
      setActionLoading(true);
      if (metadata?.is_pinned) {
        await apiClient.delete(`/findings/${findingId}/pin`);
        showMsg("Finding désépinglé");
      } else {
        await apiClient.post(`/findings/${findingId}/pin`, {
          finding_title:   findingTitle,
          product_name:    productName,
          severity,
          ai_risk_score:   aiRiskScore,
        });
        showMsg("Finding épinglé 📌");
      }
      await fetchMetadata();
    } catch (err: any) {
      showMsg(`❌ ${err.response?.data?.detail || err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleAssign = async (userId: number, username: string) => {
    try {
      setActionLoading(true);
      setShowAssignMenu(false);
      await apiClient.post(`/findings/${findingId}/assign`, {
        assigned_to_id: userId,
        finding_title:  findingTitle,
        product_name:   productName,
        severity,
        ai_risk_score:  aiRiskScore,
      });
      showMsg(`✅ Assigné à ${username}`);
      await fetchMetadata();
    } catch (err: any) {
      showMsg(`❌ ${err.response?.data?.detail || err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleUnassign = async () => {
    try {
      setActionLoading(true);
      setShowAssignMenu(false);
      await apiClient.delete(`/findings/${findingId}/assign`);
      showMsg("Assignation retirée");
      await fetchMetadata();
    } catch (err: any) {
      showMsg(`❌ ${err.response?.data?.detail || err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleStatus = async (status: string) => {
    try {
      setActionLoading(true);
      setShowStatusMenu(false);
      await apiClient.patch(`/findings/${findingId}/status`, { status });
      showMsg(`Statut → ${STATUS_LABELS[status]}`);
      await fetchMetadata();
    } catch (err: any) {
      showMsg(`❌ ${err.response?.data?.detail || err.message}`);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ color: "#666", fontSize: "13px" }}>
        Chargement...
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>

      {/* Message feedback */}
      {message && (
        <div style={{
          padding:      "8px 14px",
          background:   "rgba(0,212,255,0.1)",
          border:       "1px solid rgba(0,212,255,0.3)",
          borderRadius: "8px",
          color:        "#00d4ff",
          fontSize:     "13px",
        }}>
          {message}
        </div>
      )}

      <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>

        {/* ── Épingler ── */}
        <button
          onClick={handlePin}
          disabled={actionLoading}
          style={{
            display:      "flex",
            alignItems:   "center",
            gap:          "6px",
            padding:      "8px 16px",
            background:   metadata?.is_pinned
              ? "rgba(255,165,0,0.15)"
              : "rgba(255,255,255,0.05)",
            border:       `1px solid ${metadata?.is_pinned
              ? "rgba(255,165,0,0.5)"
              : "rgba(255,255,255,0.1)"}`,
            borderRadius: "8px",
            color:        metadata?.is_pinned ? "#ffa502" : "#aaa",
            fontSize:     "13px",
            cursor:       actionLoading ? "not-allowed" : "pointer",
            transition:   "all 0.2s",
          }}
        >
          📌 {metadata?.is_pinned ? "Désépingler" : "Épingler"}
        </button>

        {/* ── Assigner ── */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => { setShowAssignMenu(!showAssignMenu); setShowStatusMenu(false); }}
            disabled={actionLoading}
            style={{
              display:      "flex",
              alignItems:   "center",
              gap:          "6px",
              padding:      "8px 16px",
              background:   metadata?.assigned_to
                ? "rgba(46,213,115,0.15)"
                : "rgba(255,255,255,0.05)",
              border:       `1px solid ${metadata?.assigned_to
                ? "rgba(46,213,115,0.5)"
                : "rgba(255,255,255,0.1)"}`,
              borderRadius: "8px",
              color:        metadata?.assigned_to ? "#2ed573" : "#aaa",
              fontSize:     "13px",
              cursor:       actionLoading ? "not-allowed" : "pointer",
            }}
          >
            👤 {metadata?.assigned_to
              ? `Assigné : ${metadata.assigned_to}`
              : "Assigner"
            }
          </button>

          {showAssignMenu && (
            <div style={{
              position:     "absolute",
              top:          "calc(100% + 6px)",
              left:         0,
              background:   "#0d1117",
              border:       "1px solid rgba(255,255,255,0.15)",
              borderRadius: "10px",
              minWidth:     "220px",
              zIndex:       100,
              overflow:     "hidden",
              boxShadow:    "0 12px 40px rgba(0,0,0,0.5)",
            }}>
              {metadata?.assigned_to && (
                <div
                  onClick={handleUnassign}
                  style={{
                    padding:      "10px 14px",
                    color:        "#ff4757",
                    fontSize:     "13px",
                    cursor:       "pointer",
                    borderBottom: "1px solid rgba(255,255,255,0.07)",
                  }}
                >
                  ✕ Retirer l'assignation
                </div>
              )}
              {users.map(user => (
                <div
                  key={user.id}
                  onClick={() => handleAssign(user.id, user.username)}
                  style={{
                    padding:        "10px 14px",
                    color:          "#ddd",
                    fontSize:       "13px",
                    cursor:         "pointer",
                    display:        "flex",
                    justifyContent: "space-between",
                    alignItems:     "center",
                  }}
                >
                  <span>👤 {user.username}</span>
                  <span style={{
                    fontSize:     "10px",
                    color:        "#666",
                    background:   "rgba(255,255,255,0.05)",
                    padding:      "2px 6px",
                    borderRadius: "4px",
                  }}>
                    {user.role}
                  </span>
                </div>
              ))}
              {users.length === 0 && (
                <div style={{ padding: "12px", color: "#555", fontSize: "13px" }}>
                  Aucun développeur disponible
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Statut ── */}
        <div style={{ position: "relative" }}>
          <button
            onClick={() => { setShowStatusMenu(!showStatusMenu); setShowAssignMenu(false); }}
            disabled={actionLoading}
            style={{
              display:      "flex",
              alignItems:   "center",
              gap:          "6px",
              padding:      "8px 16px",
              background:   "rgba(255,255,255,0.05)",
              border:       "1px solid rgba(255,255,255,0.1)",
              borderRadius: "8px",
              color:        STATUS_COLORS[metadata?.status || "open"],
              fontSize:     "13px",
              cursor:       actionLoading ? "not-allowed" : "pointer",
            }}
          >
            {STATUS_LABELS[metadata?.status || "open"]}
          </button>

          {showStatusMenu && (
            <div style={{
              position:     "absolute",
              top:          "calc(100% + 6px)",
              left:         0,
              background:   "#0d1117",
              border:       "1px solid rgba(255,255,255,0.15)",
              borderRadius: "10px",
              minWidth:     "180px",
              zIndex:       100,
              overflow:     "hidden",
              boxShadow:    "0 12px 40px rgba(0,0,0,0.5)",
            }}>
              {Object.entries(STATUS_LABELS).map(([key, label]) => (
                <div
                  key={key}
                  onClick={() => handleStatus(key)}
                  style={{
                    padding:    "10px 14px",
                    color:      STATUS_COLORS[key],
                    fontSize:   "13px",
                    cursor:     "pointer",
                    background: metadata?.status === key
                      ? "rgba(255,255,255,0.05)"
                      : "transparent",
                  }}
                >
                  {label}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Infos assignation ── */}
      {metadata?.assigned_to && (
        <div style={{ fontSize: "12px", color: "#666", paddingLeft: "4px" }}>
          Assigné par{" "}
          <span style={{ color: "#aaa" }}>{metadata.assigned_by}</span>
          {metadata.assigned_at && (
            <span> · {new Date(metadata.assigned_at).toLocaleDateString("fr-FR")}</span>
          )}
        </div>
      )}

      {/* ── Infos épinglage ── */}
      {metadata?.is_pinned && (
        <div style={{ fontSize: "12px", color: "#666", paddingLeft: "4px" }}>
          Épinglé par{" "}
          <span style={{ color: "#ffa502" }}>{metadata.pinned_by}</span>
          {metadata.pinned_at && (
            <span> · {new Date(metadata.pinned_at).toLocaleDateString("fr-FR")}</span>
          )}
        </div>
      )}
    </div>
  );
}