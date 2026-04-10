import { useQuery } from '@tanstack/react-query';
import { engagementsApi } from '../../../api/services/engagements';

export const useEngagements = (productId: number) => {
  return useQuery({
    queryKey: ['engagements', productId],
    queryFn: () => engagementsApi.getByProduct(productId).then(res => res.data),
    enabled: !!productId,
  });
};