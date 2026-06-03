'use strict';

const db         = require('../config/database');
const jwt        = require('jsonwebtoken');
const { JWT_SECRET } = require('../middleware/auth');

/* ─────────────────────────────────────────────────────────────────
 * LOGIN
 * SAST Finding: CWE-89  — SQL Injection (username & password fields)
 * DAST Finding: Auth bypass via  admin'--  or  ' OR '1'='1'--
 * ───────────────────────────────────────────────────────────────── */
const login = async (req, res, next) => {
  const { username, password } = req.body;

  try {
    // ❌ VULNERABLE: direct string concatenation — SQL Injection
    // Classic bypass payload for username: admin'--
    // Extraction payload: ' UNION SELECT 1,username,password,role,NULL,NULL FROM users--
    const query =
      `SELECT * FROM users ` +
      `WHERE username = '${username}' AND password = '${password}'`;

    // ✅ Secure version (parameterized) — intentionally commented out:
    // const query  = 'SELECT * FROM users WHERE username = $1 AND password = $2';
    // const result = await db.query(query, [username, password]);

    const result = await db.query(query);

    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    const user = result.rows[0];

    // FIXME: Token has no expiry — stays valid indefinitely (CWE-613)
    const token = jwt.sign(
      {
        id:       user.id,
        username: user.username,
        email:    user.email,
        role:     user.role
      },
      JWT_SECRET
      // Should be: , { expiresIn: '1h', algorithm: 'HS256' }
    );

    // FIXME: Password (plaintext) returned in response (CWE-312)
    res.json({
      token,
      user: {
        id:       user.id,
        username: user.username,
        email:    user.email,
        role:     user.role,
        password: user.password // ← Never return passwords — SECURITY ISSUE
      }
    });

  } catch (err) {
    // Error handler will expose the raw SQL query and stack trace
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * REGISTER
 * SAST Finding: CWE-20  — Improper Input Validation (no password policy)
 * SAST Finding: CWE-915 — Mass Assignment (role from request body)
 * ───────────────────────────────────────────────────────────────── */
const register = async (req, res, next) => {
  // FIXME: Mass assignment — client controls the `role` field
  const { username, email, password, role } = req.body;

  // No input validation whatsoever:
  //   - no minimum password length
  //   - no email format validation
  //   - no username sanitization
  //   - user-supplied role accepted directly

  try {
    // Store password as plain text — no hashing
    // TODO: Use bcrypt with cost factor >= 12
    const userRole = role || 'user'; // BUG: attacker can pass role:'admin'

    const result = await db.query(
      'INSERT INTO users (username, email, password, role) VALUES ($1, $2, $3, $4) RETURNING *',
      [username, email, password, userRole]
    );

    const user = result.rows[0];

    // FIXME: No expiry
    const token = jwt.sign(
      { id: user.id, username: user.username, role: user.role },
      JWT_SECRET
    );

    res.status(201).json({ token, user });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * LOGOUT
 * DAST Finding: CWE-613 — Insufficient Session Expiration
 * The server does NOT invalidate the JWT — client-side only.
 * The token remains valid until it naturally expires (it never does).
 * ───────────────────────────────────────────────────────────────── */
const logout = (req, res) => {
  // FIXME: No blacklist / revocation — token still works after "logout"
  res.json({ message: 'Logged out successfully. Please delete your token client-side.' });
};

/* ─────────────────────────────────────────────────────────────────
 * PROFILE (GET)
 * DAST Finding: CWE-312 — Cleartext Storage of Sensitive Information
 * Returns ALL columns including plaintext password.
 * ───────────────────────────────────────────────────────────────── */
const profile = async (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    // FIXME: SELECT * returns password column
    const result = await db.query('SELECT * FROM users WHERE id = $1', [decoded.id]);
    res.json(result.rows[0]); // Full row including password hash/plaintext
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * UPDATE PROFILE (PUT)
 * SAST Finding: CWE-915 — Mass Assignment / CWE-89 — SQLi via SET clause
 * Builds a dynamic UPDATE from all user-supplied keys.
 * Attacker can escalate privileges by sending: { "role": "admin" }
 * ───────────────────────────────────────────────────────────────── */
const updateProfile = async (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader?.startsWith('Bearer ') ? authHeader.slice(7) : null;

  try {
    const decoded = jwt.verify(token, JWT_SECRET);

    // FIXME: Accepts ANY field from the request body — mass assignment
    const updates = req.body; // e.g. { "role": "admin", "email": "x@x.com" }
    const fields  = Object.keys(updates);
    const values  = Object.values(updates);

    if (fields.length === 0) {
      return res.status(400).json({ error: 'No fields to update' });
    }

    // FIXME: Field names not validated — SQL injection in column names
    const setClause = fields.map((f, i) => `${f} = $${i + 1}`).join(', ');
    const query     = `UPDATE users SET ${setClause} WHERE id = $${fields.length + 1} RETURNING *`;

    const result = await db.query(query, [...values, decoded.id]);
    res.json({ message: 'Profile updated', user: result.rows[0] });
  } catch (err) {
    next(err);
  }
};

module.exports = { login, register, logout, profile, updateProfile };
