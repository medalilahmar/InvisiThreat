const express = require('express');
const router = express.Router();
const { authenticate, requireAdmin } = require('../middlewares/authMiddleware');
const { getAllUsers, importConfig, runDiagnostic, updateUserRole, getLogs, debugInfo } = require('../controllers/adminController');

// ⚠️ VULN: /debug has no auth at all (CRITICAL)
router.get('/debug', debugInfo);

router.get('/users', authenticate, getAllUsers);          // Missing requireAdmin
router.post('/import-config', authenticate, importConfig);
router.get('/diagnostic', authenticate, runDiagnostic);
router.post('/user/role', authenticate, updateUserRole);
router.get('/logs', authenticate, getLogs);

module.exports = router;
