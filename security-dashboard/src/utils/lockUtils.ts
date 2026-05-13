// ─── Retourne le statut lisible du verrou d'un compte ────────────────────────

export interface LockStatus {
  isLocked: boolean;
  label: string;           // Texte à afficher dans l'UI
  minutesLeft?: number;    // Présent si verrouillé
}

export const getLockStatus = (locked_until?: string | null): LockStatus => {
  if (!locked_until) {
    return { isLocked: false, label: 'Déverrouillé' };
  }

  const lockDate = new Date(locked_until);
  const now      = new Date();

  if (lockDate <= now) {
    return { isLocked: false, label: 'Déverrouillé' };
  }

  const diffMs      = lockDate.getTime() - now.getTime();
  const minutesLeft = Math.ceil(diffMs / 60000);
  const timeStr     = lockDate.toLocaleTimeString('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
  });

  return {
    isLocked: true,
    label: `Verrouillé jusqu'à ${timeStr} (${minutesLeft} min)`,
    minutesLeft,
  };
};

// ─── Formater last_login ──────────────────────────────────────────────────────

export const formatLastLogin = (last_login?: string | null): string => {
  if (!last_login) return 'Jamais connecté';
  return new Date(last_login).toLocaleString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};