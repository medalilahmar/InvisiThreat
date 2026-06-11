const express = require('express');
const session = require('express-session');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const productRoutes = require('./routes/productRoutes');
const orderRoutes = require('./routes/orderRoutes');
const userRoutes = require('./routes/userRoutes');

const app = express();

// ⚠️ VULN: CORS wildcard (MEDIUM)
app.use(cors({ origin: '*' }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ⚠️ VULN: Weak session secret, no secure flags (HIGH)
app.use(session({
  secret: 'shoplite123',
  resave: true,
  saveUninitialized: true,
  cookie: { secure: false, httpOnly: false }
}));

// ⚠️ VULN: Version disclosure header (LOW)
app.use((req, res, next) => {
  res.setHeader('X-Powered-By', 'ShopLite/Express 4.17.1');
  next();
});

app.use(express.static(path.join(__dirname, '../public')));

app.use('/api/products', productRoutes);
app.use('/api/orders', orderRoutes);
app.use('/api/users', userRoutes);

// ⚠️ VULN: Stack trace in error response (MEDIUM)
app.use((err, req, res, next) => {
  res.status(500).json({ error: err.message, stack: err.stack });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`[*] ShopLite running on http://localhost:${PORT}`);
});

module.exports = app;
