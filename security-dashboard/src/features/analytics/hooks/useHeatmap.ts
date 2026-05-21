import { useState, useEffect } from "react";
import {apiClient } from "../../../api/client";

export interface FileNode {
  name:     string;
  depth:    number;
  stats: {
    total:     number;
    critical:  number;
    high:      number;
    medium:    number;
    low:       number;
    max_score: number;
    is_file:   boolean;
    path:      string;
  };
  children: FileNode[];
}

interface HeatmapData {
  tree:  FileNode[];
  stats: {
    total_files:    number;
    total_findings: number;
    total_critical: number;
    top_risky_files: {
      path:     string;
      total:    number;
      critical: number;
      score:    number;
    }[];
  };
}

export function useHeatmap(params?: {
  engagementId?: number;
  productName?:  string;
}) {
  const [data,    setData]    = useState<HeatmapData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState<string | null>(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        setLoading(true);
        setError(null);

        const queryParams: Record<string, string> = {};
        if (params?.engagementId) {
          queryParams.engagement_id = String(params.engagementId);
        }
        if (params?.productName) {
          queryParams.product_name = params.productName;
        }

        const res = await apiClient.get("/analytics/heatmap", {
          params: queryParams
        });
        setData(res.data);
      } catch (err: any) {
        setError(err.response?.data?.detail || "Erreur chargement heatmap");
      } finally {
        setLoading(false);
      }
    };

    fetch();
  }, [params?.engagementId, params?.productName]);

  return { data, loading, error };
}