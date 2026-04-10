import { useQuery } from '@tanstack/react-query';
import { productsApi } from '../../../api/services/products';

export const useProducts = () => {
  return useQuery({
    queryKey: ['products'],
    queryFn: () => productsApi.getAll().then(res => res.data),
  });
};