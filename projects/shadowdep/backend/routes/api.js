'use strict';

const express = require('express');
const router  = express.Router();
const db      = require('../config/database');

/* ─────────────────────────────────────────────────────────────────
 * PUBLIC API — No authentication required on any endpoint below.
 * DAST Finding: CWE-284 — Improper Access Control
 * ───────────────────────────────────────────────────────────────── */

// GET /api/users → returns all users including password column — public!
// FIXME: This must be auth-gated and must never return the password field
router.get('/users', async (req, res, next) => {
  try {
    const result = await db.query(
      'SELECT id, username, email, password, role, created_at FROM users'
    );
    res.json(result.rows); // ← Password hashes / plaintext exposed to everyone
  } catch (err) {
    next(err);
  }
});

// GET /api/projects → returns all projects — public
router.get('/projects', async (req, res, next) => {
  try {
    const result = await db.query('SELECT * FROM projects ORDER BY created_at DESC');
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
});

// GET /api/info → leaks JWT secret, API keys, DB connection info
router.get('/info', (req, res) => {
  // FIXME: Never expose secrets via an API endpoint
  res.json({
    name:       'ShadowDep Public API',
    version:    '1.0.0',
    database:   `${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`,
    jwt_secret: process.env.JWT_SECRET,          // ← CRITICAL: leaks signing key
    api_key:    process.env.ADMIN_API_KEY,       // ← Leaks admin API key
    aws_key_id: process.env.AWS_ACCESS_KEY_ID,   // ← Leaks AWS key
    env:        process.env.NODE_ENV,
    debug:      process.env.DEBUG
  });
});

// POST /api/query → JSON-based query injection (CWE-89 variant)
// Attacker passes JSON filter that is string-interpolated into SQL
// e.g. { "filter": {"name":"test\' UNION SELECT...--"} }
router.post('/query', async (req, res, next) => {
  const { filter } = req.body;
  try {
    // FIXME: JSON.stringify output embedded directly in SQL — injection possible
    const filterStr = JSON.stringify(filter || {});
    const query     = `SELECT * FROM projects WHERE data @> '${filterStr}'::jsonb`;
    const result    = await db.query(query);
    res.json(result.rows);
  } catch (err) {
    next(err);
  }
});

// GET /api/user/:id → IDOR — any user can fetch any user record
router.get('/user/:id', async (req, res, next) => {
  try {
    // FIXME: No auth, no ownership check, returns password
    const result = await db.query('SELECT * FROM users WHERE id = $1', [req.params.id]);
    res.json(result.rows[0] || { error: 'User not found' });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
