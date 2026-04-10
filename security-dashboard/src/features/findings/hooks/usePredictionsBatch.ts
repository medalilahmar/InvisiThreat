import { useQuery } from '@tanstack/react-query';
import { predictionsApi } from '../../../api/services/predictions';

export const usePredictionsBatch = (findings: any[], options?: { enabled?: boolean }) => {
  return useQuery({
    queryKey: ['predictionsBatch', findings],
    queryFn: () => predictionsApi.predictBatch(findings).then(res => res.data),
    enabled: options?.enabled && !!findings?.length,
  });
};