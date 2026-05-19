import { useState } from "react";
import { apiClient } from "../../../api/client";

interface Member {
  id:         number;
  username:   string;
  email:      string;
  role:       string;
  job_title:  string | null;
  avatar_url: string | null;
  status:     string;
}

interface ProjectMembersData {
  project_id:    number;
  project_name:  string;
  members_count: number;
  members:       Member[];
}

export function useProjectMembers() {
  const [data,    setData]    = useState<ProjectMembersData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const fetchMembers = async (projectId: number) => {
    try {
      setLoading(true);
      setError(null);
      const res = await apiClient.get(`/projects/${projectId}/members`);
      setData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Erreur chargement membres");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setData(null);
    setError(null);
  };

  return { data, loading, error, fetchMembers, reset };
}