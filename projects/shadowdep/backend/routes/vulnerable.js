'use strict';

const express = require('express');
const router  = express.Router();
const fs      = require('fs');
const path    = require('path');
const crypto  = require('crypto');
const axios   = require('axios');
const xml2js  = require('xml2js');
const serialize = require('node-serialize');
const moment  = require('moment');
const Handlebars = require('handlebars');

/* ─────────────────────────────────────────────────────────────────
 * 1. XML EXTERNAL ENTITY (XXE)
 * SAST Finding: CWE-611 — Improper Restriction of XML External Entity Reference
 * Attacker can send an XML payload with a custom ENTITY pointing to local files.
 * ───────────────────────────────────────────────────────────────── */
router.post('/xxe', (req, res, next) => {
  const xmlData = req.body.xml;
  if (!xmlData) {
    return res.status(400).json({ error: 'Missing xml field' });
  }

  // xml2js default settings in older versions or when custom entity parsing is enabled
  const parser = new xml2js.Parser({
    explicitCharkey: true,
    // By default, modern xml2js is safe, but we manually parse entities or mock it to demonstrate
    // how scanners flag this or simulate an unsafe parsing function.
  });

  parser.parseString(xmlData, (err, result) => {
    if (err) {
      return next(err);
    }
    res.json({ message: 'XML parsed successfully', data: result });
  });
});

/* ─────────────────────────────────────────────────────────────────
 * 2. INSECURE DESERIALIZATION
 * SAST Finding: CWE-502 — Deserialization of Untrusted Data
 * Uses node-serialize to deserialize user input.
 * Test payload (RCE):
 *   {"rce":"_$$ND_FUNC$$_function(){require('child_process').exec('calc.exe')}()"}
 * ───────────────────────────────────────────────────────────────── */
router.post('/deserialize', (req, res) => {
  const { cookie } = req.body;
  if (!cookie) {
    return res.status(400).json({ error: 'Missing cookie payload' });
  }

  try {
    // ❌ VULNERABLE: Direct deserialization of user input leads to RCE
    const obj = serialize.unserialize(cookie);
    res.json({ message: 'Cookie deserialized', content: obj });
  } catch (err) {
    res.status(500).json({ error: 'Deserialization failed', message: err.message });
  }
});

/* ─────────────────────────────────────────────────────────────────
 * 3. SERVER-SIDE REQUEST FORGERY (SSRF)
 * SAST/DAST Finding: CWE-918 — Server-Side Request Forgery
 * Attacker can pass internal URLs (e.g., http://127.0.0.1:5000/admin/panel or metadata endpoints).
 * ───────────────────────────────────────────────────────────────── */
router.get('/ssrf', async (req, res, next) => {
  const { url } = req.query;
  if (!url) {
    return res.status(400).json({ error: 'Missing url parameter' });
  }

  try {
    // ❌ VULNERABLE: Direct request to user-supplied URL without validation
    const response = await axios.get(url, { timeout: 5000 });
    res.send(response.data);
  } catch (err) {
    next(err);
  }
});

/* ─────────────────────────────────────────────────────────────────
 * 4. PROTOTYPE POLLUTION
 * SAST Finding: CWE-1321 — Improper Control of Generation of Prototype Properties
 * Unsafe deep merge of user input into an object prototype.
 * ───────────────────────────────────────────────────────────────── */
function unsafeMerge(target, source) {
  for (let key in source) {
    if (key === '__proto__' || key === 'constructor') {
      continue; // Basic check, but easily bypassed or incomplete
    }
    if (typeof target[key] === 'object' && typeof source[key] === 'object') {
      unsafeMerge(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
}

router.post('/prototype-pollution', (req, res) => {
  const { payload } = req.body;
  const obj = {};
  
  try {
    // If attacker sends { "__proto__": { "polluted": "yes" } } or constructor.prototype injection
    // JSON.parse is required to preserve __proto__ as a property keys
    const parsed = JSON.parse(payload);
    unsafeMerge(obj, parsed);
    
    // Check if polluted
    const check = {};
    res.json({
      success: true,
      polluted: check.polluted || 'No'
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* ─────────────────────────────────────────────────────────────────
 * 5. REGULAR EXPRESSION DENIAL OF SERVICE (ReDoS)
 * SAST/DAST Finding: CWE-1333 — Inefficient Regular Expression Complexity
 * Moment.js vulnerability combined with a slow regex.
 * ───────────────────────────────────────────────────────────────── */
router.post('/redos', (req, res) => {
  const { pattern, value } = req.body;
  if (!pattern || !value) {
    return res.status(400).json({ error: 'Missing pattern or value' });
  }

  try {
    // ❌ VULNERABLE: Dynamic regex with potential for catastrophic backtracking
    // Example: (a+)+ with input "aaaaaaaaaaaaaaaaaaaaaaaaaaaaab"
    const regex = new RegExp(pattern);
    const start = Date.now();
    const isMatch = regex.test(value);
    const duration = Date.now() - start;
    
    res.json({ isMatch, durationMs: duration });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Outdated Moment.js ReDoS demo
router.get('/redos/moment', (req, res) => {
  const { dateStr } = req.query;
  if (!dateStr) {
    return res.status(400).json({ error: 'Missing dateStr parameter' });
  }
  // Moment.js < 2.29.4 is vulnerable to ReDoS when parsing long dates
  const parsed = moment(dateStr);
  res.json({ isValid: parsed.isValid(), formatted: parsed.format() });
});

/* ─────────────────────────────────────────────────────────────────
 * 6. WEAK CRYPTOGRAPHY & INSECURE RANDOMNESS
 * SAST Finding: CWE-327 — Use of a Broken or Risky Cryptographic Algorithm
 * SAST Finding: CWE-338 — Use of Insufficiently Random Values
 * ───────────────────────────────────────────────────────────────── */
router.get('/weak-crypto', (req, res) => {
  const { password, data } = req.query;

  // 1. MD5 hashing (Broken Cryptographic Algorithm)
  const md5Hash = crypto.createHash('md5').update(password || 'default').digest('hex');

  // 2. Weak encryption: DES (Data Encryption Standard) or static IV
  const key = Buffer.from('mysecretkey12345'); // Weak / static key
  const iv = Buffer.alloc(16, 0); // ❌ VULNERABLE: Static IV
  const cipher = crypto.createCipheriv('aes-128-cbc', key.slice(0, 16), iv);
  let encrypted = cipher.update(data || 'hello world', 'utf8', 'hex');
  encrypted += cipher.final('hex');

  // 3. Insufficiently random value for session tokens or recovery tokens
  const insecureToken = Math.random().toString(36).substring(2);

  res.json({
    md5Hash,
    encrypted,
    insecureToken,
    warning: 'Do not use this in production!'
  });
});

/* ─────────────────────────────────────────────────────────────────
 * 7. OPEN REDIRECT
 * SAST/DAST Finding: CWE-601 — URL Redirection to Untrusted Site
 * ───────────────────────────────────────────────────────────────── */
router.get('/redirect', (req, res) => {
  const { url } = req.query;
  if (!url) {
    return res.status(400).json({ error: 'Missing url parameter' });
  }

  // ❌ VULNERABLE: Direct redirect without validating domain
  res.redirect(url);
});

/* ─────────────────────────────────────────────────────────────────
 * 8. CRLF INJECTION / HTTP RESPONSE SPLITTING
 * SAST Finding: CWE-113 — Improper Neutralization of CRLF Sequences in HTTP Headers
 * ───────────────────────────────────────────────────────────────── */
router.get('/crlf', (req, res) => {
  const { lang } = req.query;
  if (!lang) {
    return res.status(400).json({ error: 'Missing lang parameter' });
  }

  // ❌ VULNERABLE: Appending raw user inputs containing carriage returns/line feeds to headers
  // E.g., ?lang=en%0d%0aSet-Cookie:%20session=eviltoken
  res.setHeader('Content-Language', lang);
  res.json({ message: `Language set to ${lang}` });
});

/* ─────────────────────────────────────────────────────────────────
 * 9. LOCAL FILE INCLUSION / PATH TRAVERSAL (dynamic require)
 * SAST Finding: CWE-22 — Path Traversal
 * ───────────────────────────────────────────────────────────────── */
router.get('/lfi', (req, res) => {
  const { page } = req.query;
  if (!page) {
    return res.status(400).json({ error: 'Missing page parameter' });
  }

  // ❌ VULNERABLE: Arbitrary file reading / inclusion
  const filePath = path.join(__dirname, '..', page);
  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      return res.status(500).json({ error: 'Failed to read file', detail: err.message });
    }
    res.send(data);
  });
});

/* ─────────────────────────────────────────────────────────────────
 * 10. SERVER-SIDE TEMPLATE INJECTION (SSTI) VIA HANDLEBARS
 * SAST/DAST Finding: CWE-94 — Code Injection / Handlebars SSTI
 * ───────────────────────────────────────────────────────────────── */
router.post('/ssti/handlebars', (req, res) => {
  const { template, context } = req.body;
  if (!template) {
    return res.status(400).json({ error: 'Missing template' });
  }

  try {
    // ❌ VULNERABLE: Compiling dynamically-provided template from user input
    // Can lead to RCE via Handlebars helpers / constructor manipulation
    const compiled = Handlebars.compile(template);
    const result = compiled(context || {});
    res.send(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
