/**
 * useHashedParams.ts — InvisiThreat
 *
 * Hooks pour lire et décoder automatiquement les IDs hashés depuis l'URL.
 *
 * Couvre les 3 URLs à masquer :
 *   /engagements?productId=1       → /engagements?productId=aX9k2p
 *   /findings?engagementId=13      → /findings?engagementId=3kQpX7
 *   /findings/1485                 → /findings/mZ8nQr4x
 */

import { useParams, useSearchParams } from "react-router-dom";
import { decodeId } from "../utils/hashId";

/**
 * /findings/:id  →  décode le path param :id
 *
 * Utilisation dans FindingDetailPage :
 *   const { id } = useHashedPathParam();
 */
export function useHashedPathParam() {
  const { id: rawToken } = useParams<{ id: string }>();
  const id = rawToken ? decodeId(rawToken) : null;
  return { id, rawToken };
}

/**
 * /findings?engagementId=XXX  →  décode le query param engagementId
 *
 * Utilisation dans FindingsPage :
 *   const { engagementId } = useHashedEngagementId();
 */
export function useHashedEngagementId() {
  const [searchParams] = useSearchParams();
  const rawToken = searchParams.get("engagementId");
  const engagementId = rawToken ? decodeId(rawToken) : null;
  return { engagementId, rawToken };
}

/**
 * /engagements?productId=XXX  →  décode le query param productId
 *
 * Utilisation dans EngagementsPage :
 *   const { productId } = useHashedProductId();
 */
export function useHashedProductId() {
  const [searchParams] = useSearchParams();
  const rawToken = searchParams.get("productId");
  const productId = rawToken ? decodeId(rawToken) : null;
  return { productId, rawToken };
}