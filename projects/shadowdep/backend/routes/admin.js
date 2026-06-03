'use strict';

const express = require('express');
const router  = express.Router();
const { requireAdmin } = require('../middleware/auth');
const ctrl = require('../controllers/adminController');

// NOTE: requireAdmin only validates token signature, not role.
//       Any authenticated user can access ALL admin routes below.

// GET  /admin/panel → returns users (with passwords), projects, env vars
router.get('/panel', requireAdmin, ctrl.getPanel);

// GET  /admin/exec?cmd= → OS command injection (CWE-78)
router.get('/exec',  requireAdmin, ctrl.execCommand);

// POST /admin/users/role → privilege escalation (CWE-269)
router.post('/users/role', requireAdmin, ctrl.updateUserRole);

// GET  /admin/logs?file= → path traversal + command injection (CWE-22, CWE-78)
router.get('/logs',  requireAdmin, ctrl.readLogs);

// GET  /admin/config → exposes full config including secrets
router.get('/config', requireAdmin, (req, res) => {
  // FIXME: Exposes entire process environment
  res.json({
    env:    process.env,
    config: {
      jwt_secret:    process.env.JWT_SECRET,
      db_password:   process.env.DB_PASSWORD,
      admin_api_key: process.env.ADMIN_API_KEY,
      aws_key:       process.env.AWS_ACCESS_KEY_ID,
      aws_secret:    process.env.AWS_SECRET_ACCESS_KEY
    }
  });
});

module.exports = router;
