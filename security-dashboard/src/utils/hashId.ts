const SECRET = import.meta.env.VITE_ID_SECRET ?? "invisi_threat_s3cr3t_k3y_2026";

const BASE62_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

// --- Helpers base62 ---

function toBase62(n: number): string {
  if (n === 0) return "0";
  let result = "";
  let num = n;
  while (num > 0) {
    result = BASE62_CHARS[num % 62] + result;
    num = Math.floor(num / 62);
  }
  return result;
}

function fromBase62(s: string): number {
  let result = 0;
  for (const ch of s) {
    const idx = BASE62_CHARS.indexOf(ch);
    if (idx === -1) return -1; // caractère invalide
    result = result * 62 + idx;
  }
  return result;
}

// --- Dérivation d'une clé numérique depuis la chaîne secrète ---

function deriveKey(): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < SECRET.length; i++) {
    hash ^= SECRET.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0; // FNV prime, forcer uint32
  }
  return hash;
}

const KEY = deriveKey();

// --- Permutation réversible (Feistel simple 2 rounds) ---

function feistelEncode(id: number): number {
  // Travaille sur 32 bits, split en deux moitiés de 16 bits
  let lo = id & 0xffff;
  let hi = (id >>> 16) & 0xffff;

  const k1 = (KEY ^ 0xa5a5a5a5) & 0xffff;
  const k2 = (KEY ^ 0x5a5a5a5a) & 0xffff;

  // Round 1
  lo = (lo ^ ((hi * k1 + 0x9e37) & 0xffff)) & 0xffff;
  // Round 2
  hi = (hi ^ ((lo * k2 + 0x79b9) & 0xffff)) & 0xffff;

  return ((hi << 16) | lo) >>> 0;
}

function feistelDecode(encoded: number): number {
  let lo = encoded & 0xffff;
  let hi = (encoded >>> 16) & 0xffff;

  const k1 = (KEY ^ 0xa5a5a5a5) & 0xffff;
  const k2 = (KEY ^ 0x5a5a5a5a) & 0xffff;

  // Inverse Round 2
  hi = (hi ^ ((lo * k2 + 0x79b9) & 0xffff)) & 0xffff;
  // Inverse Round 1
  lo = (lo ^ ((hi * k1 + 0x9e37) & 0xffff)) & 0xffff;

  return ((hi << 16) | lo) >>> 0;
}

// --- API publique ---

/**
 * Encode un ID numérique en token URL-safe opaque.
 * @param id  ID numérique positif (ex: 852)
 * @returns   Token base62 (ex: "3kQpX7mZ")
 */
export function encodeId(id: number): string {
  if (!Number.isInteger(id) || id < 0) {
    throw new Error(`encodeId: id doit être un entier positif, reçu: ${id}`);
  }
  const permuted = feistelEncode(id);
  // Ajouter un checksum simple (derniers 2 chiffres de la somme)
  const checksum = ((id % 62) + 7) % 62;
  const combined = permuted * 62 + checksum;
  return toBase62(combined);
}

/**
 * Décode un token URL en ID numérique.
 * @param token  Token base62 (ex: "3kQpX7mZ")
 * @returns      ID original ou null si le token est invalide/corrompu
 */
export function decodeId(token: string): number | null {
  if (!token || typeof token !== "string") return null;

  const combined = fromBase62(token);
  if (combined === -1 || combined < 0) return null;

  const checksum = combined % 62;
  const permuted = Math.floor(combined / 62);

  const id = feistelDecode(permuted);

  // Vérifier le checksum
  const expectedChecksum = ((id % 62) + 7) % 62;
  if (checksum !== expectedChecksum) return null;

  return id;
}

/**
 * Encode un engagementId pour les query params.
 * Identique à encodeId, alias sémantique.
 */
export const encodeEngagementId = encodeId;

/**
 * Décode un engagementId depuis les query params.
 * Identique à decodeId, alias sémantique.
 */
export const decodeEngagementId = decodeId;