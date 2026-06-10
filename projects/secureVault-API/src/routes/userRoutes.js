const express = require('express');
const router = express.Router();
const { authenticate } = require('../middlewares/authMiddleware');
const { getProfile, redirectAfterLogin, checkUsername, searchUsers } = require('../controllers/userController');

router.get('/profile/:id', authenticate, getProfile);
router.get('/redirect', redirectAfterLogin);
router.get('/check/:username', checkUsername);
router.get('/search', authenticate, searchUsers);

module.exports = router;
