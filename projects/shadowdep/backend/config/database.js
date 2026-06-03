'use strict';

const { Pool } = require('pg');
require('dotenv').config();

// FIXME: Credentials should only come from env vars, never hardcoded as fallback
// SAST Finding: CWE-798 — Use of Hard-coded Credentials
const pool = new Pool({
  host:     process.env.DB_HOST     || 'localhost',
  port:     process.env.DB_PORT     || 5432,
  database: process.env.DB_NAME     || 'shadowdep',
  user:     process.env.DB_USER     || 'postgres',
  password: process.env.DB_PASSWORD || 'postgres123', // Hardcoded fallback — bad practice
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  // TODO: Enable SSL for production connections
  // ssl: { rejectUnauthorized: false }
});

pool.on('error', (err) => {
  // FIXME: Full error (including credentials info) logged to console
  console.error('Unexpected PostgreSQL pool error:', err);
});

// Test connection on startup and print connection info (info leak)
pool.connect((err, client, release) => {
  if (err) {
    console.error('DB connection failed:', err.message);
    return;
  }
  // FIXME: Logs DB host and user — information disclosure
  console.log(`Connected to PostgreSQL at ${process.env.DB_HOST}:${process.env.DB_PORT}/${process.env.DB_NAME} as ${process.env.DB_USER}`);
  release();
});

module.exports = pool;
