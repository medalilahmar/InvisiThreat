import { useState } from "react";

interface AvatarUser {
  username:   string;
  role?:      string;
  avatar_url?: string | null;
}

const ROLE_COLORS: Record<string, string> = {
  admin:     'linear-gradient(135deg, #ef4444, #dc2626)',
  manager:   'linear-gradient(135deg, #eab308, #ca8a04)',
  analyst:   'linear-gradient(135deg, #6366f1, #4f46e5)',
  developer: 'linear-gradient(135deg, #22c55e, #16a34a)',
};

export function UserAvatar({
  user,
  size = 28,
}: {
  user: AvatarUser;
  size?: number;
}) {
  const [imgError, setImgError] = useState(false);
  const initials = user.username.slice(0, 2).toUpperCase();
  const base = {
    width:        size,
    height:       size,
    borderRadius: '50%',
    border:       '1px solid var(--border2)',
    flexShrink:   0 as const,
  };

  if (user.avatar_url && !imgError) {
    return (
      <img
        src={user.avatar_url}
        alt={user.username}
        style={{ ...base, objectFit: 'cover' as const }}
        onError={() => setImgError(true)}
      />
    );
  }

  return (
    <div style={{
      ...base,
      background:     ROLE_COLORS[user.role ?? ''] || ROLE_COLORS.developer,
      display:        'flex',
      alignItems:     'center',
      justifyContent: 'center',
      fontSize:       size * 0.38,
      fontWeight:     700,
      color:          '#fff',
      fontFamily:     'var(--font-display)',
    }}>
      {initials}
    </div>
  );
}