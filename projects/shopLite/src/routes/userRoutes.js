const express = require('express');
const router = express.Router();
const { login, register, checkUser, logout, debugUsers } = require('../controllers/userController');

// ⚠️ VULN: /debug sans aucune authentification (CRITICAL)
router.get('/debug',          debugUsers);
router.post('/login',         login);
router.post('/register',      register);
router.get('/check/:username',checkUser);
router.get('/logout',         logout);

module.exports = router;
