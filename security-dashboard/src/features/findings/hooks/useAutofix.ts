import { useState, useCallback } from 'react';
import {
  checkAutofixCapability,
  applyAutofix,
  AutoFixCapability,
  AutoFixResponse,
} from '../../../api/services/explanations';

export const useAutofix = () => {
  const [capability, setCapability] = useState<AutoFixCapability | null>(null);
  const [capabilityLoading, setCapabilityLoading] = useState(false);
  const [autofixResult, setAutofixResult] = useState<AutoFixResponse | null>(null);
  const [autofixLoading, setAutofixLoading] = useState(false);
  const [autofixError, setAutofixError] = useState<string | null>(null);

  const checkCapability = useCallback(async (findingId: number) => {
    setCapabilityLoading(true);
    try {
      const { data } = await checkAutofixCapability(findingId);
      setCapability(data);
    } catch (err) {
      console.error('Erreur vérification capacité autofix:', err);
    } finally {
      setCapabilityLoading(false);
    }
  }, []);

  const executeAutofix = useCallback(async (findingId: number) => {
    setAutofixLoading(true);
    setAutofixError(null);
    try {
      const { data } = await applyAutofix(findingId);
      setAutofixResult(data);
    } catch (err: any) {
      setAutofixError(err.response?.data?.detail || err.message || 'Échec de l\'autofix');
    } finally {
      setAutofixLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setCapability(null);
    setAutofixResult(null);
    setAutofixError(null);
  }, []);

  return {
    capability,
    capabilityLoading,
    autofixResult,
    autofixLoading,
    autofixError,
    checkCapability,
    executeAutofix,
    reset,
  };
};