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

const STATUS_CLASSES: Record<string, string> = {
  open:        'fa-status--open',
  in_progress: 'fa-status--progress',
  resolved:    'fa-status--resolved',
  wont_fix:    'fa-status--wont-fix',
};

const STATUS_LABELS: Record<string, string> = {
  open:        'Ouvert',
  in_progress: 'En cours',
  resolved:    'Résolu',
  wont_fix:    "Won't fix",
};

/* ── SVG icons ────────────────────────────────────────────────────────────── */

function IconPin() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  );
}

function IconUser() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>
  );
}

function IconChevron() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  );
}

function IconStatus() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  );
}

function IconRemove() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6"  y1="6" x2="18" y2="18"/>
    </svg>
  );
}

function IconCheck() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  );
}

function IconError() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8"  x2="12" y2="12"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
  );
}

/* ── Component ────────────────────────────────────────────────────────────── */

export default function FindingActions({
  findingId, findingTitle, productName, severity, aiRiskScore, users,
}: Props) {
  const [metadata,       setMetadata]       = useState<Metadata | null>(null);
  const [loading,        setLoading]        = useState(true);
  const [showAssignMenu, setShowAssignMenu] = useState(false);
  const [showStatusMenu, setShowStatusMenu] = useState(false);
  const [actionLoading,  setActionLoading]  = useState(false);
  const [message,        setMessage]        = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  useEffect(() => { fetchMetadata(); }, [findingId]);

  const fetchMetadata = async () => {
    try {
      setLoading(true);
      const res = await apiClient.get(`/findings/${findingId}/metadata`);
      setMetadata(res.data);
    } catch (err) {
      console.error('Erreur metadata', err);
    } finally {
      setLoading(false);
    }
  };

  const showMsg = (text: string, type: 'success' | 'error' = 'success') => {
    setMessage({ text, type });
    setTimeout(() => setMessage(null), 3000);
  };

  const handlePin = async () => {
    try {
      setActionLoading(true);
      if (metadata?.is_pinned) {
        await apiClient.delete(`/findings/${findingId}/pin`);
        showMsg('Finding désépinglé');
      } else {
        await apiClient.post(`/findings/${findingId}/pin`, {
          finding_title: findingTitle,
          product_name:  productName,
          severity,
          ai_risk_score: aiRiskScore,
        });
        showMsg('Finding épinglé');
      }
      await fetchMetadata();
    } catch (err: any) {
      showMsg(err.response?.data?.detail || err.message, 'error');
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
      showMsg(`Assigné à ${username}`);
      await fetchMetadata();
    } catch (err: any) {
      showMsg(err.response?.data?.detail || err.message, 'error');
    } finally {
      setActionLoading(false);
    }
  };

  const handleUnassign = async () => {
    try {
      setActionLoading(true);
      setShowAssignMenu(false);
      await apiClient.delete(`/findings/${findingId}/assign`);
      showMsg('Assignation retirée');
      await fetchMetadata();
    } catch (err: any) {
      showMsg(err.response?.data?.detail || err.message, 'error');
    } finally {
      setActionLoading(false);
    }
  };

  const handleStatus = async (status: string) => {
    try {
      setActionLoading(true);
      setShowStatusMenu(false);
      await apiClient.patch(`/findings/${findingId}/status`, { status });
      showMsg(`Statut mis à jour : ${STATUS_LABELS[status]}`);
      await fetchMetadata();
    } catch (err: any) {
      showMsg(err.response?.data?.detail || err.message, 'error');
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="fa-loading">
        <div className="fa-loading-spinner" />
        <span>Chargement...</span>
      </div>
    );
  }

  const currentStatus = metadata?.status || 'open';

  return (
    <div className="fa-root">

      {/* ── Message feedback ── */}
      {message && (
        <div className={`fa-message fa-message--${message.type}`}>
          {message.type === 'success' ? <IconCheck /> : <IconError />}
          {message.text}
        </div>
      )}

      <div className="fa-actions-row">

        {/* ── Épingler ── */}
        <button
          onClick={handlePin}
          disabled={actionLoading}
          className={`fa-btn${metadata?.is_pinned ? ' fa-btn--pinned' : ''}`}
        >
          <IconPin />
          {metadata?.is_pinned ? 'Désépingler' : 'Épingler'}
        </button>

        {/* ── Assigner ── */}
        <div className="fa-dropdown-wrap">
          <button
            onClick={() => { setShowAssignMenu(!showAssignMenu); setShowStatusMenu(false); }}
            disabled={actionLoading}
            className={`fa-btn${metadata?.assigned_to ? ' fa-btn--assigned' : ''}`}
          >
            <IconUser />
            {metadata?.assigned_to ? `Assigné : ${metadata.assigned_to}` : 'Assigner'}
            <IconChevron />
          </button>

          {showAssignMenu && (
            <div className="fa-dropdown">
              {metadata?.assigned_to && (
                <div
                  className="fa-dropdown-item fa-dropdown-item--danger"
                  onClick={handleUnassign}
                >
                  <IconRemove />
                  Retirer l'assignation
                </div>
              )}
              {users.map(user => (
                <div
                  key={user.id}
                  className="fa-dropdown-item"
                  onClick={() => handleAssign(user.id, user.username)}
                >
                  <IconUser />
                  <span className="fa-dropdown-username">{user.username}</span>
                  <span className="fa-dropdown-role">{user.role}</span>
                </div>
              ))}
              {users.length === 0 && (
                <div className="fa-dropdown-empty">
                  Aucun développeur disponible
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Statut ── */}
        <div className="fa-dropdown-wrap">
          <button
            onClick={() => { setShowStatusMenu(!showStatusMenu); setShowAssignMenu(false); }}
            disabled={actionLoading}
            className={`fa-btn fa-btn--status ${STATUS_CLASSES[currentStatus]}`}
          >
            <IconStatus />
            {STATUS_LABELS[currentStatus]}
            <IconChevron />
          </button>

          {showStatusMenu && (
            <div className="fa-dropdown">
              {Object.entries(STATUS_LABELS).map(([key, label]) => (
                <div
                  key={key}
                  className={`fa-dropdown-item fa-dropdown-item--status ${STATUS_CLASSES[key]}${currentStatus === key ? ' fa-dropdown-item--active' : ''}`}
                  onClick={() => handleStatus(key)}
                >
                  {currentStatus === key && <IconCheck />}
                  {label}
                </div>
              ))}
            </div>
          )}
        </div>

      </div>

      {/* ── Infos assignation ── */}
      {metadata?.assigned_to && (
        <div className="fa-meta-info">
          <span className="fa-meta-label">Assigné par</span>
          <span className="fa-meta-value">{metadata.assigned_by}</span>
          {metadata.assigned_at && (
            <span className="fa-meta-date">
              · {new Date(metadata.assigned_at).toLocaleDateString('fr-FR')}
            </span>
          )}
        </div>
      )}

      {/* ── Infos épinglage ── */}
      {metadata?.is_pinned && (
        <div className="fa-meta-info">
          <span className="fa-meta-label">Épinglé par</span>
          <span className="fa-meta-value fa-meta-value--pinned">{metadata.pinned_by}</span>
          {metadata.pinned_at && (
            <span className="fa-meta-date">
              · {new Date(metadata.pinned_at).toLocaleDateString('fr-FR')}
            </span>
          )}
        </div>
      )}

    </div>
  );
}