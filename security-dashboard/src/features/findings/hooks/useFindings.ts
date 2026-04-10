import { useQuery } from '@tanstack/react-query';
import { findingsApi } from '../../../api/services/findings';

export const useFindings = (engagementId: number) => {
  return useQuery({
    queryKey: ['findings', engagementId],
    queryFn: () => findingsApi.getByEngagement(engagementId).then(res => res.data),
    enabled: !!engagementId,
  });
};