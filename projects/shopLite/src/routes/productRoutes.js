const express = require('express');
const router = express.Router();
const { getProduct, searchProducts, filterByCategory, getProductImage, createProduct } = require('../controllers/productController');

router.get('/',              searchProducts);
router.get('/search',        searchProducts);
router.get('/filter',        filterByCategory);
router.get('/image',         getProductImage);
router.get('/:id',           getProduct);
router.post('/',             createProduct);

module.exports = router;
