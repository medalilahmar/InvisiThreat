import { apiClient } from '../client';
import { Notification, UnreadCountResponse } from '../../types/notification';

// ─── GET /admin/notifications/ ────────────────────────────────────────────────
export const getNotifications = async (
  unreadOnly = false,
  skip = 0,
  limit = 50
): Promise<Notification[]> => {
  const res = await apiClient.get('/admin/notifications/', {
    params: { unread_only: unreadOnly, skip, limit },
  });
  return res.data;
};

// ─── GET /admin/notifications/unread-count ────────────────────────────────────
// Retourne directement le nombre — 0 si erreur réseau (cloche silencieuse)
export const getUnreadCount = async (): Promise<number> => {
  try {
    const res = await apiClient.get<UnreadCountResponse>(
      '/admin/notifications/unread-count'
    );
    return res.data.count;
  } catch {
    return 0;
  }
};

// ─── PUT /admin/notifications/{id}/read ──────────────────────────────────────
export const markAsRead = async (id: number): Promise<void> => {
  await apiClient.put(`/admin/notifications/${id}/read`);
};

// ─── PUT /admin/notifications/read-all ───────────────────────────────────────
export const markAllAsRead = async (): Promise<void> => {
  await apiClient.put('/admin/notifications/read-all');
};


// ── TOUS LES UTILISATEURS ──────────────────────────────────────────
export const getMyUnreadCount = async (): Promise<number> => {
  const res = await apiClient.get('/admin/notifications/me/unread-count');
  return res.data.count;
};

export const getMyNotifications = async (
  unreadOnly = false, skip = 0, limit = 30
): Promise<Notification[]> => {
  const res = await apiClient.get('/admin/notifications/me', {
    params: { unread_only: unreadOnly, skip, limit }
  });
  return res.data;
};

export const markMyAsRead = async (id: number): Promise<void> => {
  await apiClient.put(`/admin/notifications/me/${id}/read`);
};

export const markAllMyAsRead = async (): Promise<void> => {
  await apiClient.put('/admin/notifications/me/read-all');
};