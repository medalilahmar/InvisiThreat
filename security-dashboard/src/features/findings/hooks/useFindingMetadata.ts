import { useState } from "react";
import { apiClient } from "../../../api/client";

export function useFindingMetadata(findingId: number) {
  const [loading, setLoading] = useState(false);

  const pin = async (data: {
    finding_title:  string;
    product_name?:  string;
    severity?:      string;
    ai_risk_score?: number;
  }) => {
    setLoading(true);
    try {
      await apiClient.post(`/findings/${findingId}/pin`, data);
      return true;
    } catch (err: any) {
      throw new Error(err.response?.data?.detail || "Erreur épinglage");
    } finally {
      setLoading(false);
    }
  };

  const unpin = async () => {
    setLoading(true);
    try {
      await apiClient.delete(`/findings/${findingId}/pin`);
      return true;
    } catch (err: any) {
      throw new Error(err.response?.data?.detail || "Erreur désépinglage");
    } finally {
      setLoading(false);
    }
  };

  const assign = async (data: {
    assigned_to_id: number;
    finding_title:  string;
    product_name?:  string;
    severity?:      string;
    ai_risk_score?: number;
  }) => {
    setLoading(true);
    try {
      await apiClient.post(`/findings/${findingId}/assign`, data);
      return true;
    } catch (err: any) {
      throw new Error(err.response?.data?.detail || "Erreur assignation");
    } finally {
      setLoading(false);
    }
  };

  const unassign = async () => {
    setLoading(true);
    try {
      await apiClient.delete(`/findings/${findingId}/assign`);
      return true;
    } catch (err: any) {
      throw new Error(err.response?.data?.detail || "Erreur désassignation");
    } finally {
      setLoading(false);
    }
  };

  const updateStatus = async (status: string) => {
    setLoading(true);
    try {
      await apiClient.patch(`/findings/${findingId}/status`, { status });
      return true;
    } catch (err: any) {
      throw new Error(err.response?.data?.detail || "Erreur statut");
    } finally {
      setLoading(false);
    }
  };

  return { loading, pin, unpin, assign, unassign, updateStatus };
}