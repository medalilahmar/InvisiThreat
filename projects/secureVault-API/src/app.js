const express = require('express');
const session = require('express-session');
const morgan = require('morgan');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const authRoutes = require('./routes/authRoutes');
const vaultRoutes = require('./routes/vaultRoutes');
const userRoutes = require('./routes/userRoutes');
const adminRoutes = require('./routes/adminRoutes');

const app = express();

// ⚠️ VULN: CORS wildcard - allows any origin (MEDIUM)
app.use(cors({ origin: '*' }));

// ⚠️ VULN: Morgan logs full request body including passwords (LOW)
app.use(morgan('combined'));

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// ⚠️ VULN: Weak session configuration - no secure flags, weak secret (HIGH)
app.use(session({
  secret: 'secret123',
  resave: true,
  saveUninitialized: true,
  cookie: {
    secure: false,
    httpOnly: false,
    maxAge: 9999999999
  }
}));

// ⚠️ VULN: Exposes server technology in headers (LOW)
app.use((req, res, next) => {
  res.setHeader('X-Powered-By', 'SecureVault/Express 4.17.1');
  next();
});

// Static files
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '../views'));
app.use(express.static(path.join(__dirname, '../public')));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/vault', vaultRoutes);
app.use('/api/users', userRoutes);
app.use('/api/admin', adminRoutes);

// ⚠️ VULN: Stack trace exposed in error handler (MEDIUM)
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({
    error: err.message,
    stack: err.stack,
    details: err
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`[*] SecureVault API running on http://localhost:${PORT}`);
  console.log(`[*] Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;
