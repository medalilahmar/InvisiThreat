interface Member {
  id:         number;
  username:   string;
  email:      string;
  role:       string;
  job_title:  string | null;
  avatar_url: string | null;
}

interface Props {
  productName:   string;
  members:       Member[];
  loading:       boolean;
  error:         string | null;
  onClose:       () => void;
}

const ROLE_COLORS: Record<string, string> = {
  admin:     "rgba(255,71,87,0.2)",
  analyst:   "rgba(0,212,255,0.2)",
  developer: "rgba(46,213,115,0.2)",
};

const ROLE_TEXT: Record<string, string> = {
  admin:     "#ff4757",
  analyst:   "#00d4ff",
  developer: "#2ed573",
};

const ROLE_ICONS: Record<string, string> = {
  admin:     "👑",
  analyst:   "🔍",
  developer: "👤",
};

export default function MembersModal({
  productName,
  members,
  loading,
  error,
  onClose,
}: Props) {
  return (
    // Overlay
    <div
      onClick={onClose}
      style={{
        position:        "fixed",
        inset:           0,
        background:      "rgba(0,0,0,0.75)",
        display:         "flex",
        alignItems:      "center",
        justifyContent:  "center",
        zIndex:          1000,
        backdropFilter:  "blur(4px)",
      }}
    >
      {/* Modal Box */}
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background:   "#0d1117",
          border:       "1px solid rgba(0,212,255,0.25)",
          borderRadius: "16px",
          padding:      "28px",
          width:        "100%",
          maxWidth:     "480px",
          maxHeight:    "80vh",
          overflowY:    "auto",
          boxShadow:    "0 24px 64px rgba(0,0,0,0.6)",
        }}
      >
        {/* Header */}
        <div style={{
          display:        "flex",
          justifyContent: "space-between",
          alignItems:     "center",
          marginBottom:   "20px",
        }}>
          <div>
            <div style={{
              color:      "#00d4ff",
              fontSize:   "11px",
              fontWeight: 600,
              letterSpacing: "2px",
              marginBottom: "4px",
            }}>
              ÉQUIPE DU PROJET
            </div>
            <h3 style={{
              color:      "#ffffff",
              fontSize:   "18px",
              fontWeight: 700,
              margin:     0,
            }}>
              {productName}
            </h3>
          </div>

          <button
            onClick={onClose}
            style={{
              background:   "rgba(255,255,255,0.05)",
              border:       "1px solid rgba(255,255,255,0.1)",
              color:        "#aaa",
              borderRadius: "8px",
              width:        "32px",
              height:       "32px",
              cursor:       "pointer",
              fontSize:     "16px",
              display:      "flex",
              alignItems:   "center",
              justifyContent: "center",
            }}
          >
            ✕
          </button>
        </div>

        {/* Loading */}
        {loading && (
          <div style={{
            textAlign:  "center",
            color:      "#666",
            padding:    "40px 0",
          }}>
            <div style={{ fontSize: "24px", marginBottom: "12px" }}>⟳</div>
            Chargement des membres...
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div style={{
            textAlign:    "center",
            color:        "#ff4757",
            padding:      "40px 0",
            background:   "rgba(255,71,87,0.05)",
            borderRadius: "12px",
          }}>
            ⚠️ {error}
          </div>
        )}

        {/* Members List */}
        {!loading && !error && (
          <>
            {/* Count */}
            <div style={{
              color:        "#666",
              fontSize:     "13px",
              marginBottom: "16px",
            }}>
              {members.length} membre{members.length > 1 ? "s" : ""}
            </div>

            {/* Liste */}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
              {members.map(member => (
                <div
                  key={member.id}
                  style={{
                    display:      "flex",
                    alignItems:   "center",
                    gap:          "12px",
                    padding:      "12px 16px",
                    background:   "rgba(255,255,255,0.03)",
                    border:       "1px solid rgba(255,255,255,0.07)",
                    borderRadius: "10px",
                  }}
                >
                  {/* Avatar */}
                  <div style={{
                    width:          "40px",
                    height:         "40px",
                    borderRadius:   "50%",
                    background:     ROLE_COLORS[member.role] || "rgba(255,255,255,0.1)",
                    border:         `1.5px solid ${ROLE_TEXT[member.role] || "#666"}40`,
                    display:        "flex",
                    alignItems:     "center",
                    justifyContent: "center",
                    fontSize:       "16px",
                    fontWeight:     700,
                    color:          ROLE_TEXT[member.role] || "#fff",
                    flexShrink:     0,
                  }}>
                    {member.avatar_url
                      ? <img
                          src={member.avatar_url}
                          alt={member.username}
                          style={{ width: "40px", height: "40px", borderRadius: "50%" }}
                        />
                      : member.username[0].toUpperCase()
                    }
                  </div>

                  {/* Infos */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      display:    "flex",
                      alignItems: "center",
                      gap:        "8px",
                      marginBottom: "3px",
                    }}>
                      <span style={{
                        color:      "#ffffff",
                        fontWeight: 600,
                        fontSize:   "14px",
                      }}>
                        {member.username}
                      </span>

                      {/* Role Badge */}
                      <span style={{
                        fontSize:     "10px",
                        fontWeight:   600,
                        padding:      "2px 8px",
                        borderRadius: "20px",
                        background:   ROLE_COLORS[member.role] || "rgba(255,255,255,0.1)",
                        color:        ROLE_TEXT[member.role]   || "#aaa",
                        letterSpacing: "0.5px",
                      }}>
                        {ROLE_ICONS[member.role]} {member.role.toUpperCase()}
                      </span>
                    </div>

                    <div style={{
                      color:     "#666",
                      fontSize:  "12px",
                      overflow:  "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace:   "nowrap",
                    }}>
                      {member.email}
                      {member.job_title && (
                        <span style={{ color: "#555" }}> — {member.job_title}</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}

              {/* Empty */}
              {members.length === 0 && (
                <div style={{
                  textAlign:  "center",
                  color:      "#555",
                  padding:    "40px 0",
                  fontSize:   "14px",
                }}>
                  📭 Aucun membre dans ce projet
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}