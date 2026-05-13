import { useState, useCallback } from 'react';
import { getSolution, LLMSolution } from '../../../api/services/explanations';

export const useSolution = () => {
  const [solution, setSolution] = useState<LLMSolution | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSolution = useCallback(async (findingId: number) => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await getSolution(findingId);
      setSolution(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Erreur lors de la génération de la solution');
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setSolution(null);
    setError(null);
    setLoading(false);
  }, []);

  return { solution, loading, error, fetchSolution, reset };
};