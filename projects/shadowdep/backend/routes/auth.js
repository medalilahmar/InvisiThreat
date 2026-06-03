'use strict';

const express = require('express');
const router  = express.Router();
const ctrl    = require('../controllers/authController');

// GET  /auth/login    → render login page
router.get('/login',    (req, res) => res.render('login', {}));
// GET  /auth/register → render register page
router.get('/register', (req, res) => res.render('register', {}));

// POST /auth/login    → SQL-injectable login (CWE-89)
router.post('/login',    ctrl.login);

// POST /auth/register → mass assignment (role field), no password policy (CWE-915)
router.post('/register', ctrl.register);

// POST /auth/logout   → no server-side invalidation (CWE-613)
router.post('/logout',   ctrl.logout);

// GET  /auth/profile  → returns password column (CWE-312)
router.get('/profile',   ctrl.profile);

// PUT  /auth/profile  → mass assignment / dynamic UPDATE (CWE-915, CWE-89)
router.put('/profile',   ctrl.updateProfile);

// FIXME: DELETE method not restricted — allows account deletion without re-auth
router.delete('/profile', (req, res) => {
  // Stub — intentionally left open
  res.json({ message: 'Account deletion not yet implemented' });
});

module.exports = router;
