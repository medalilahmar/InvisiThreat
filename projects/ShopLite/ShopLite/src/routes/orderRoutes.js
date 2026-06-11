const express = require('express');
const router = express.Router();
const { getOrder, importOrder, trackShipment, requestRefund, applyDiscount } = require('../controllers/orderController');

router.get('/:id',          getOrder);
router.post('/import',      importOrder);
router.post('/track',       trackShipment);
router.post('/refund',      requestRefund);
router.post('/discount',    applyDiscount);

module.exports = router;
