'use strict';

/**
 * Global Express error handler.
 *
 * SAST/DAST Finding: CWE-209 — Information Exposure Through an Error Message
 * FIXME: Never send stack traces or internal error details to the client.
 *        In production, return only a generic message and log internally.
 */
const errorHandler = (err, req, res, next) => {
  // Log full error server-side (appropriate)
  console.error('[ERROR]', new Date().toISOString(), err);

  const statusCode = err.status || err.statusCode || 500;

  // VULNERABLE: Full stack trace, internal paths and variable names
  // sent to client in ALL environments (including production).
  res.status(statusCode).json({
    error:   err.message,
    // FIXME: Remove the fields below before any production deployment
    stack:   err.stack,         // Leaks file paths, line numbers, module names
    details: err,               // May contain query text with injected payload
    path:    req.path,
    method:  req.method,
    body:    req.body,          // Echoes request body — may contain credentials
    headers: req.headers,       // Echoes all request headers incl. Authorization
    env:     process.env.NODE_ENV,
    version: process.version
  });
};

module.exports = errorHandler;
