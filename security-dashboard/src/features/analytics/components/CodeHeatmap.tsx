import { useState } from "react";
import { useHeatmap, FileNode } from "../hooks/useHeatmap";
import { useNavigate } from "react-router-dom";

interface Props {
  engagementId?: number;
  productName?:  string;
}

// Couleur selon le nombre de findings
function getRiskColor(node: FileNode["stats"]): {
  bg: string; border: string; text: string; label: string;
} {
  if (node.critical > 0) return {
    bg:     "var(--severity-critical-bg)",
    border: "var(--severity-critical-border)",
    text:   "var(--severity-critical)",
    label:  "CRITICAL"
  };
  if (node.high > 0) return {
    bg:     "var(--severity-high-bg)",
    border: "var(--severity-high-border)",
    text:   "var(--severity-high)",
    label:  "HIGH"
  };
  if (node.medium > 0) return {
    bg:     "var(--severity-medium-bg)",
    border: "var(--severity-medium-border)",
    text:   "var(--severity-medium)",
    label:  "MEDIUM"
  };
  if (node.low > 0) return {
    bg:     "var(--severity-low-bg)",
    border: "var(--severity-low-border)",
    text:   "var(--severity-low)",
    label:  "LOW"
  };
  return {
    bg:     "var(--bg3)",
    border: "var(--border)",
    text:   "var(--dimmed)",
    label:  "SECURE"
  };
}

// Composant récursif pour chaque nœud de l'arbre
function TreeNode({
  node,
  onFileClick,
}: {
  node:        FileNode;
  onFileClick: (path: string) => void;
}) {
  const [expanded, setExpanded] = useState(
    node.depth < 2  // Auto-expand les 2 premiers niveaux
  );

  const colors   = getRiskColor(node.stats);
  const hasKids  = node.children.length > 0;
  const isFile   = node.stats.is_file;

  return (
    <div style={{ marginLeft: node.depth === 0 ? 0 : "16px" }}>
      {/* Nœud */}
      <div
        onClick={() => {
          if (isFile) {
            onFileClick(node.stats.path);
          } else {
            setExpanded(!expanded);
          }
        }}
        style={{
          display:      "flex",
          alignItems:   "center",
          gap:          "8px",
          padding:      "6px 10px",
          marginBottom: "3px",
          background:   node.stats.total > 0 ? colors.bg : "transparent",
          border:       `1px solid ${node.stats.total > 0 ? colors.border : "transparent"}`,
          borderRadius: "6px",
          cursor:       "pointer",
          transition:   "all 0.15s",
        }}
      >
        {/* Icône */}
        <span style={{ fontSize: "14px", flexShrink: 0 }}>
          {isFile
            ? "📄"
            : expanded
            ? "📂"
            : "📁"
          }
        </span>

        {/* Nom */}
        <span style={{
          color:     node.stats.total > 0 ? colors.text : "var(--dimmed)",
          fontSize:  "13px",
          fontWeight: node.stats.total > 0 ? 600 : 400,
          flex:      1,
          overflow:  "hidden",
          textOverflow: "ellipsis",
          whiteSpace:   "nowrap",
        }}>
          {node.name}
        </span>

        {/* Badge findings */}
        {node.stats.total > 0 && (
          <div style={{ display: "flex", gap: "4px", flexShrink: 0 }}>
            {node.stats.critical > 0 && (
              <span style={{
                fontSize:     "10px",
                padding:      "1px 6px",
                borderRadius: "10px",
                background: "var(--severity-critical-bg)",
                color: "var(--severity-critical)",
                fontWeight:   700,
              }}>
                {node.stats.critical}C
              </span>
            )}
            {node.stats.high > 0 && (
              <span style={{
                fontSize:     "10px",
                padding:      "1px 6px",
                borderRadius: "10px",
                background: "var(--severity-high-bg)",
                color: "var(--severity-high)",
                fontWeight:   700,
              }}>
                {node.stats.high}H
              </span>
            )}
            {(node.stats.medium > 0 || node.stats.low > 0) && (
              <span style={{
                fontSize:     "10px",
                padding:      "1px 6px",
                borderRadius: "10px",
                background: "var(--bg4)",
                color: "var(--dimmed)",

              }}>
                {node.stats.medium + node.stats.low}
              </span>
            )}
          </div>
        )}

        {/* Flèche expand/collapse pour dossiers */}
        {!isFile && hasKids && (
          <span style={{ color: "var(--dimmed)", fontSize: "10px" }}>
            {expanded ? "▼" : "▶"}
          </span>
        )}
      </div>

      {/* Enfants */}
      {!isFile && expanded && hasKids && (
        <div>
          {node.children.map((child, i) => (
            <TreeNode
              key={i}
              node={child}
              onFileClick={onFileClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}


// Composant principal
export default function CodeHeatmap({ engagementId, productName }: Props) {
  const navigate = useNavigate();
  const { data, loading, error } = useHeatmap({ engagementId, productName });

  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [searchTerm,   setSearchTerm]   = useState("");

  const handleFileClick = (path: string) => {
    setSelectedFile(path);
    // Navigue vers la liste des findings filtrée par fichier
    navigate(`/findings?file_path=${encodeURIComponent(path)}`);
  };

  // Filtre les fichiers risqués selon la recherche
  const topFiles = data?.stats.top_risky_files.filter(f =>
    f.path.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  if (loading) {
    return (
      <div style={{
        display:        "flex",
        flexDirection:  "column",
        alignItems:     "center",
        justifyContent: "center",
        padding:        "60px",
        color:          "#666",
        gap:            "12px",
      }}>
        <div style={{
          width:  "32px",
          height: "32px",
          border: "2px solid var(--accent-muted)",
          borderTop: "2px solid var(--accent)",
          color: "var(--dimmed)",
          borderRadius: "50%",
          animation: "spin 1s linear infinite",
        }} />
        Génération de la heatmap...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        textAlign:    "center",
        color:        "var(--severity-critical)",
        padding:      "40px",
        background:   "var(--glass-danger)",
        borderRadius: "12px",
      }}>
        ⚠️ {error}
      </div>
    );
  }

  if (!data) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>

      {/* Stats globales */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(3, 1fr)",
        gap:     "12px",
      }}>
        {[
          { label: "Fichiers analysés", value: data.stats.total_files,    icon: "📁" },
          { label: "Findings total",    value: data.stats.total_findings,  icon: "🛡️" },
          { label: "Critiques",         value: data.stats.total_critical,  icon: "🔴" },
        ].map((stat, i) => (
          <div key={i} style={{
            background:   "var(--bg3)",
            border:       "1px solid var(--border)",
            borderRadius: "10px",
            padding:      "14px",
            textAlign:    "center",
          }}>
            <div style={{ fontSize: "20px", marginBottom: "6px" }}>{stat.icon}</div>
            <div style={{ color: "var(--text-strong)", fontSize: "22px", fontWeight: 700 }}>
              {stat.value}
            </div>
            <div style={{ color: "var(--dimmed)", fontSize: "11px" }}>{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Contenu principal : arbre + top fichiers */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 320px",
        gap:     "16px",
        alignItems: "start",
      }}>

        {/* Arbre des fichiers */}
        <div style={{
          background:   "var(--bg3)",
          border:       "1px solid var(--border)",
          borderRadius: "12px",
          padding:      "16px",
          maxHeight:    "600px",
          overflowY:    "auto",
        }}>
          <div style={{
            color:        "var(--accent)",
            fontSize:     "11px",
            letterSpacing: "2px",
            marginBottom: "14px",
          }}>
            ARBORESCENCE DU CODE
          </div>

          {data.tree.length === 0 ? (
            <div style={{ color: "var(--dimmed)", textAlign: "center", padding: "40px 0" }}>
              Aucun fichier avec findings
            </div>
          ) : (
            data.tree.map((node, i) => (
              <TreeNode
                key={i}
                node={node}
                onFileClick={handleFileClick}
              />
            ))
          )}
        </div>

        {/* Top fichiers risqués */}
        <div style={{
          background:   "var(--bg3)",
          border:       "1px solid var(--border)",
          borderRadius: "12px",
          padding:      "16px",
        }}>
          <div style={{
            color:        "var(--severity-critical)",
            fontSize:     "11px",
            letterSpacing: "2px",
            marginBottom: "14px",
          }}>
            🔥 TOP FICHIERS RISQUÉS
          </div>

          {/* Recherche */}
          <input
            type="text"
            placeholder="Filtrer..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            style={{
              width:        "100%",
              background:   "var(--bg4)",
              border:       "1px solid var(--border)",
              borderRadius: "6px",
              padding:      "6px 10px",
              color:        "var(--text)",
              fontSize:     "12px",
              marginBottom: "12px",
              outline:      "none",
              boxSizing:    "border-box",
            }}
          />

          {/* Liste */}
          <div style={{
            display:       "flex",
            flexDirection: "column",
            gap:           "8px",
            maxHeight:     "500px",
            overflowY:     "auto",
          }}>
            {topFiles.map((file, i) => (
              <div
                key={i}
                onClick={() => handleFileClick(file.path)}
                style={{
                  padding:      "10px 12px",
                  background:   "var(--bg4)",
                  border:       `1px solid ${file.critical > 0 ? "var(--severity-critical)" : "var(--border)"}`,
                  borderRadius: "8px",
                  cursor:       "pointer",
                }}
              >
                {/* Nom du fichier */}
                <div style={{
                  color:        "var(--text)",
                  fontSize:     "12px",
                  fontWeight:   600,
                  overflow:     "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace:   "nowrap",
                  marginBottom: "6px",
                }}>
                  📄 {file.path.split("/").pop()}
                </div>

                {/* Chemin court */}
                <div style={{
                  color:        "var(--dimmed)",
                  fontSize:     "10px",
                  overflow:     "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace:   "nowrap",
                  marginBottom: "8px",
                }}>
                  {file.path}
                </div>

                {/* Barre de risque */}
                <div style={{
                  height:       "3px",
                  background:   "var(--bg4)",
                  borderRadius: "2px",
                  marginBottom: "6px",
                }}>
                  <div style={{
                    height:       "3px",
                    width:        `${Math.min(file.total * 10, 100)}%`,
                    background:   file.critical > 0 ? "var(--severity-critical)" : "var(--severity-high)",
                    borderRadius: "2px",
                  }} />
                </div>

                {/* Badges */}
                <div style={{ display: "flex", gap: "4px" }}>
                  {file.critical > 0 && (
                    <span style={{
                      fontSize:     "10px",
                      padding:      "1px 6px",
                      borderRadius: "10px",
                      background:   "var(--severity-critical-bg)",
                      color:        "var(--severity-critical)",
                    }}>
                      {file.critical} critical
                    </span>
                  )}
                  <span style={{
                    fontSize:     "10px",
                    padding:      "1px 6px",
                    borderRadius: "10px",
                    background:   "var(--bg4)",
                    color:        "var(--dimmed)",
                  }}>
                    {file.total} total
                  </span>
                  {file.score > 0 && (
                    <span style={{
                      fontSize:     "10px",
                      padding:      "1px 6px",
                      borderRadius: "10px",
                      background:   "var(--severity-ia-bg)",
                      color:        "var(--severity-ia)",
                    }}>
                      IA {file.score}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Légende */}
      <div style={{
        display:        "flex",
        gap:            "20px",
        justifyContent: "center",
        flexWrap:       "wrap",
      }}>
        {[
          { color: "var(--severity-critical)", label: "Critical" },
          { color: "var(--severity-high)", label: "High"     },
          { color: "var(--severity-medium)", label: "Medium"   },
          { color: "var(--severity-low)", label: "Low"      },
          { color: "var(--severity-secure)", label: "Secure"   },
        ].map((item, i) => (
          <div key={i} style={{
            display:    "flex",
            alignItems: "center",
            gap:        "6px",
          }}>
            <div style={{
              width:        "10px",
              height:       "10px",
              borderRadius: "2px",
              background:   item.color,
            }} />
            <span style={{ color: "var(--dimmed)", fontSize: "12px" }}>
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}