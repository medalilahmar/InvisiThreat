'use strict';

const express      = require('express');
const bodyParser   = require('body-parser');
const cors         = require('cors');
const cookieParser = require('cookie-parser');
const morgan       = require('morgan');
const path         = require('path');
const serveIndex   = require('serve-index');
require('dotenv').config();

const authRoutes    = require('./routes/auth');
const projectRoutes = require('./routes/projects');
const adminRoutes   = require('./routes/admin');
const apiRoutes     = require('./routes/api');
const vulnerableRoutes = require('./routes/vulnerable');
const graphqlRoutes = require('./routes/graphql');
const errorHandler  = require('./middleware/errorHandler');

const app = express();

/* ─────────────────────────────────────────────────────────────────
 * SECURITY MISCONFIGURATION: CORS allows ALL origins
 * DAST Finding: CWE-942 — Permissive CORS Policy
 * FIXME: Replace '*' with specific trusted origin(s)
 * ───────────────────────────────────────────────────────────────── */
app.use(cors({
  origin:      '*',          // Allows ANY domain to call this API
  methods:     ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS', 'HEAD', 'TRACE'],
  allowedHeaders: ['*'],     // Accepts any header
  credentials: false         // Note: credentials can't combine with origin:'*'
}));

/* ─────────────────────────────────────────────────────────────────
 * SECURITY MISCONFIGURATION: No security headers
 * DAST Finding: Missing X-Frame-Options, CSP, HSTS, X-Content-Type-Options
 * FIXME: Use helmet() middleware:
 *   const helmet = require('helmet');
 *   app.use(helmet());
 * ───────────────────────────────────────────────────────────────── */

// Request logging — in production this leaks sensitive URL params to log files
app.use(morgan('combined'));

app.use(cookieParser());

// FIXME: Large payload limit allows DoS / memory exhaustion
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ extended: true, limit: '50mb' }));

/* ─────────────────────────────────────────────────────────────────
 * Template engine (EJS)
 * SSTI vector: user-controlled data rendered with <%- %> (unescaped)
 * ───────────────────────────────────────────────────────────────── */
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve frontend build (if present)
app.use(express.static(path.join(__dirname, 'public')));

/* ─────────────────────────────────────────────────────────────────
 * SECURITY MISCONFIGURATION: Directory listing enabled
 * DAST Finding: CWE-548 — Exposure of Information Through Directory Listing
 * Any visitor can browse /uploads and see all uploaded files.
 * This also enables downloading uploaded .php / .html / .sh files.
 * FIXME: Remove serve-index; restrict access to authenticated users
 * ───────────────────────────────────────────────────────────────── */
const uploadsPath = path.join(__dirname, 'uploads');
app.use('/uploads',
  express.static(uploadsPath, {
    dotfiles:   'allow',   // FIXME: Serves .htaccess, .env etc.
    extensions: false,
    index:      false
  }),
  serveIndex(uploadsPath, {
    icons:  true,
    hidden: true           // FIXME: Also lists hidden files
  })
);

/* ─────────────────────────────────────────────────────────────────
 * DEBUG MODE enabled in production
 * DAST Finding: Information disclosure via X-Powered-By header
 * ───────────────────────────────────────────────────────────────── */
// FIXME: app.disable('x-powered-by') — currently exposing "Express" header
app.set('debug', true);

/* ─────────────────────────────────────────────────────────────────
 * ROUTES
 * ───────────────────────────────────────────────────────────────── */
app.use('/auth',     authRoutes);
app.use('/projects', projectRoutes);
app.use('/admin',    adminRoutes);
app.use('/api',      apiRoutes);
app.use('/vulnerable', vulnerableRoutes);
app.use('/graphql',  graphqlRoutes);

// Public health-check — leaks internal info
app.get('/health', (req, res) => {
  res.json({
    status:      'ok',
    version:     '1.0.0',
    environment: process.env.NODE_ENV,
    debug:       process.env.DEBUG,
    database:    `${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME}`,
    node:        process.version,
    uptime:      process.uptime()
  });
});

// Robots.txt — lists sensitive paths (helps attackers enumerate endpoints)
app.get('/robots.txt', (req, res) => {
  res.type('text/plain');
  res.send([
    'User-agent: *',
    'Disallow: /admin/',
    'Disallow: /admin/panel',
    'Disallow: /admin/exec',
    'Disallow: /admin/config',
    'Disallow: /api/info',
    'Disallow: /uploads/',
    '# Internal dashboard — do not index'
  ].join('\n'));
});

/* ─────────────────────────────────────────────────────────────────
 * GLOBAL ERROR HANDLER — sends stack traces to client (CWE-209)
 * ───────────────────────────────────────────────────────────────── */
app.use(errorHandler);

/* ─────────────────────────────────────────────────────────────────
 * START SERVER
 * FIXME: HTTP only — no TLS/HTTPS (CWE-319 — Cleartext Transmission)
 * ───────────────────────────────────────────────────────────────── */
const PORT = process.env.PORT || 5001;
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🌑 ShadowDep backend running on http://0.0.0.0:${PORT}`);
  console.log(`   Environment : ${process.env.NODE_ENV}`);
  console.log(`   Debug mode  : ${process.env.DEBUG}`);
  console.log(`   DB          : ${process.env.DB_HOST}/${process.env.DB_NAME}`);
  console.log(`   JWT secret  : ${process.env.JWT_SECRET}`);  // FIXME: Never log secrets
  console.log(`   Uploads dir : ${uploadsPath} (directory listing ON)\n`);
});

module.exports = app;
