import { useState, useEffect } from 'react';
import axios from 'axios';
import type { ModelMetrics, HealthStatus } from '../../../types/model';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8081';

export function useModelInfo() {
  const [modelInfo, setModelInfo] = useState<ModelMetrics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloading, setReloading] = useState(false);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);
      const [modelRes, healthRes] = await Promise.all([
        axios.get<ModelMetrics>(`${API_URL}/model/info`),
        axios.get<HealthStatus>(`${API_URL}/health`),
      ]);
      setModelInfo(modelRes.data);
      setHealth(healthRes.data);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erreur lors du chargement du modèle';
      setError(message);
      console.error('Model fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const reloadModel = async () => {
    try {
      setReloading(true);
      await axios.post(`${API_URL}/model/reload`);
      await fetchData();
      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Erreur lors du rechargement';
      setError(message);
      console.error('Reload error:', err);
      return false;
    } finally {
      setReloading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  return { modelInfo, health, loading, error, reloadModel, reloading, refetch: fetchData };
}