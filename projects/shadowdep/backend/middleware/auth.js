'use strict';

const jwt = require('jsonwebtoken');

// SAST Finding: CWE-321 — Use of Hard-coded Cryptographic Key
// FIXME: JWT secret must be stored in a secrets manager, not in source code
const JWT_SECRET = process.env.JWT_SECRET || 'supersecretkey123';

/**
 * Middleware: verify that the request carries a valid JWT.
 * Attaches decoded payload to req.user.
 *
 * VULNERABILITY: Tokens never expire (no maxAge / expiresIn check enforced
 * on issuance), and no server-side revocation list exists.
 */
const authenticateToken = (req, res, next) => {
  // Accept token from Authorization header OR from a cookie
  const authHeader = req.headers['authorization'];
  const token =
    (authHeader && authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : null) ||
    req.cookies?.token ||
    req.query?.token; // FIXME: Token in URL — leaks in server logs and browser history

  if (!token) {
    return res.status(401).json({ error: 'Access denied. No token provided.' });
  }

  try {
    // TODO: enforce expiry — tokens currently have no expiry date
    const decoded = jwt.verify(token, JWT_SECRET, {
      // algorithms: ['HS256'], // Good practice — but not enforced here
    });
    req.user = decoded;
    next();
  } catch (err) {
    // FIXME: Returns detailed JWT error to client (information disclosure)
    return res.status(401).json({ error: 'Invalid token.', detail: err.message });
  }
};

/**
 * Middleware: "admin-only" gate.
 *
 * VULNERABILITY (Broken Access Control — CWE-285):
 *   This middleware only verifies the token signature. It does NOT check
 *   whether req.user.role === 'admin'. Any authenticated user can reach
 *   admin endpoints by supplying a valid (non-admin) JWT.
 *
 * FIXME: Replace the body of this function with:
 *   authenticateToken(req, res, () => {
 *     if (req.user.role !== 'admin') return res.status(403).json({ error: 'Forbidden' });
 *     next();
 *   });
 */
const requireAdmin = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  if (!token) {
    return res.status(401).json({ error: 'Access denied.' });
  }

  try {
    // BUG: Only checks signature validity — role NOT verified
    jwt.verify(token, JWT_SECRET);
    next(); // Any valid-token holder becomes "admin"
  } catch (err) {
    return res.status(403).json({ error: 'Forbidden.', detail: err.message });
  }
};

module.exports = { authenticateToken, requireAdmin, JWT_SECRET };
