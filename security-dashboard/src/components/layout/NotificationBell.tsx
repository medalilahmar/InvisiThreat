import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../../auth/hooks/useAuth';
import {
  getNotifications,
  getUnreadCount,
  markAsRead,
  markAllAsRead,
} from '../../api/services/notifications';
import { Notification, NotificationType } from '../../types/notification';
import './NotificationBell.css';

// ─── Config par type ──────────────────────────────────────────────────────────
const typeConfig: Record<NotificationType, { emoji: string }> = {
  [NotificationType.NEW_USER]:         { emoji: '👤' },
  [NotificationType.LOGIN_FAILED]:     { emoji: '🔒' },
  [NotificationType.USER_BLOCKED]:     { emoji: '🚫' },
  [NotificationType.PENDING_REMINDER]: { emoji: '⏳' },
  [NotificationType.PROJECT_SYNC]:     { emoji: '🔄' },
};

const formatDate = (iso: string) =>
  new Date(iso).toLocaleString('fr-FR', {
    day: '2-digit', month: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });

const POLL_INTERVAL_MS = 30_000;

// ─── Icône cloche SVG ─────────────────────────────────────────────────────────
const BellIcon = () => (
  <svg className="notif-bell__icon" viewBox="0 0 24 24" aria-hidden="true">
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
    <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
  </svg>
);

// ─── Composant ────────────────────────────────────────────────────────────────
export default function NotificationBell() {
  const { user } = useAuth();

  const [unreadCount,   setUnreadCount]   = useState(0);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [open,          setOpen]          = useState(false);
  const [loading,       setLoading]       = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const isAdmin = user?.role === 'admin';

  // ── Polling ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isAdmin) return;
    const fetchCount = async () => {
      const count = await getUnreadCount();
      setUnreadCount(count);
    };
    fetchCount();
    const id = setInterval(fetchCount, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [isAdmin]);

  // ── Fermer en cliquant à l'extérieur ──────────────────────────────────────
  useEffect(() => {
    if (!isAdmin) return;
    const handler = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [isAdmin]);

  // ── Pas admin → rien ──────────────────────────────────────────────────────
  if (!isAdmin) return null;

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleOpen = async () => {
    const nextOpen = !open;
    setOpen(nextOpen);
    if (nextOpen) {
      setLoading(true);
      try {
        const data = await getNotifications(false, 0, 30);
        setNotifications(data);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleMarkRead = async (id: number) => {
    await markAsRead(id);
    setNotifications(prev =>
      prev.map(n => n.id === id ? { ...n, is_read: true } : n)
    );
    setUnreadCount(prev => Math.max(0, prev - 1));
  };

  const handleMarkAll = async () => {
    await markAllAsRead();
    setNotifications(prev => prev.map(n => ({ ...n, is_read: true })));
    setUnreadCount(0);
  };

  return (
    <div ref={dropdownRef} className="notif-bell">

      {/* ── Bouton cloche ──────────────────────────────────────────────────── */}
      <button
        onClick={handleOpen}
        title="Notifications"
        className={`notif-bell__btn ${open ? 'notif-bell__btn--active' : ''}`}
      >
        <BellIcon />
        {unreadCount > 0 && (
          <span className="notif-bell__badge">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {/* ── Dropdown ───────────────────────────────────────────────────────── */}
      {open && (
        <div className="notif-bell__dropdown">

          {/* En-tête */}
          <div className="notif-bell__header">
            <span className="notif-bell__header-title">
              Notifications
            </span>
            {unreadCount > 0 && (
              <button className="notif-bell__mark-all" onClick={handleMarkAll}>
                Tout marquer lu
              </button>
            )}
          </div>

          {/* Liste */}
          <div className="notif-bell__list">

            {loading && (
              <div className="notif-bell__loading">Chargement</div>
            )}

            {!loading && notifications.length === 0 && (
              <div className="notif-bell__empty">Aucune notification</div>
            )}

            {!loading && notifications.map(notif => {
              const cfg = typeConfig[notif.type] ?? { emoji: '📌' };
              const isUnread = !notif.is_read;
              return (
                <div
                  key={notif.id}
                  onClick={() => isUnread && handleMarkRead(notif.id)}
                  className={`notif-bell__item ${
                    isUnread
                      ? 'notif-bell__item--unread'
                      : 'notif-bell__item--read'
                  }`}
                >
                  <span className="notif-bell__emoji">{cfg.emoji}</span>

                  <div className="notif-bell__content">
                    <div className="notif-bell__title">{notif.title}</div>
                    <div className="notif-bell__message">{notif.message}</div>
                    <div className="notif-bell__date">{formatDate(notif.created_at)}</div>
                  </div>

                  {isUnread && <span className="notif-bell__dot" />}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}