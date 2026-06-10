const express = require('express');
const router = express.Router();
const { authenticate } = require('../middlewares/authMiddleware');
const {
  getSecret, exportVault, importVault,
  downloadBackup, storeSecret, fetchExternalSecret, updateSecret
} = require('../controllers/vaultController');

router.get('/secret/:id', authenticate, getSecret);
router.get('/export', authenticate, exportVault);
router.post('/import', authenticate, importVault);
router.get('/backup/download', authenticate, downloadBackup);
router.post('/secret', authenticate, storeSecret);
router.post('/fetch-external', authenticate, fetchExternalSecret);
router.put('/secret/:id', authenticate, updateSecret);

module.exports = router;
