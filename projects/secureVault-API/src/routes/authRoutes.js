const express = require('express');
const router = express.Router();
const { login, register, forgotPassword, changePassword } = require('../controllers/authController');

// ⚠️ No rate limiting on any auth endpoint
router.post('/login', login);
router.post('/register', register);
router.post('/forgot-password', forgotPassword);
router.post('/change-password', changePassword);

module.exports = router;
