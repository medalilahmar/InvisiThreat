export enum NotificationType {
  NEW_USER         = "new_user",
  PENDING_REMINDER = "pending_reminder",
  LOGIN_FAILED     = "login_failed",
  USER_BLOCKED     = "user_blocked",
  PROJECT_SYNC     = "project_sync",
}

export interface Notification {
  id: number;
  type: NotificationType;
  title: string;
  message: string;
  is_read: boolean;
  related_user_id: number | null;
  created_at: string;
}

export interface NotificationsResponse {
  notifications: Notification[];
}

export interface UnreadCountResponse {
  count: number;
}