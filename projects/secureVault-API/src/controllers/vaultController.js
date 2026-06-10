const marked = require('marked');
const { exec } = require('child_process');
const serialize = require('node-serialize');
const config = require('../config/config');
const db = require('../utils/database');
const crypto = require('crypto');

// ⚠️ VULN: No authentication check - IDOR (HIGH)
// ⚠️ VULN: XSS via marked without sanitization (HIGH)
const getSecret = async (req, res) => {
  const { id } = req.params;

  try {
    // VULN: Direct object reference - no ownership check
    const query = `SELECT * FROM secrets WHERE id = ${id}`;
    const secret = await db.query(query);

    if (!secret || secret.length === 0) {
      return res.status(404).json({ message: 'Secret not found' });
    }

    // ⚠️ VULN: XSS - user-controlled content rendered as HTML
    const renderedNotes = marked(secret[0].notes || '');

    res.json({
      ...secret[0],
      notes_html: renderedNotes  // Rendered unsanitized HTML
    });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: OS Command Injection (CRITICAL)
const exportVault = async (req, res) => {
  const { format, filename } = req.query;

  // CRITICAL: User input directly in shell command
  exec(`vault-export --format ${format} --output /tmp/${filename}`, (err, stdout, stderr) => {
    if (err) {
      return res.status(500).json({ error: stderr });
    }
    res.json({ success: true, output: stdout, file: `/tmp/${filename}` });
  });
};

// ⚠️ VULN: Insecure Deserialization (CRITICAL)
const importVault = async (req, res) => {
  const { data } = req.body;

  try {
    // CRITICAL: Deserializing untrusted user data
    const vaultData = serialize.unserialize(data);

    res.json({ success: true, imported: vaultData });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: Path Traversal (HIGH)
const downloadBackup = async (req, res) => {
  const { file } = req.query;

  // VULN: No path sanitization - directory traversal
  const filePath = `/var/backups/vault/${file}`;
  res.download(filePath);
};

// ⚠️ VULN: Weak encryption for stored secrets (HIGH)
const storeSecret = async (req, res) => {
  const { name, value, category, userId } = req.body;

  // VULN: Using DES (broken) with hardcoded key
  const cipher = crypto.createCipheriv(
    config.encryption.algorithm,  // 'des' - broken
    config.encryption.key.slice(0, 8),
    config.encryption.iv.slice(0, 8)
  );

  let encrypted = cipher.update(value, 'utf8', 'hex');
  encrypted += cipher.final('hex');

  // VULN: SQL Injection + storing with weak encryption
  const query = `INSERT INTO secrets (name, value, category, user_id) 
                 VALUES ('${name}', '${encrypted}', '${category}', '${userId}')`;

  await db.query(query);

  res.json({ success: true, encrypted });
};

// ⚠️ VULN: Server-Side Request Forgery - SSRF (HIGH)
const fetchExternalSecret = async (req, res) => {
  const axios = require('axios');
  const { url } = req.body;

  // CRITICAL: No URL validation - SSRF allows internal network access
  try {
    const response = await axios.get(url);  // Could be http://169.254.169.254/latest/meta-data/
    res.json({ data: response.data });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: Mass Assignment (HIGH)
const updateSecret = async (req, res) => {
  const { id } = req.params;

  // VULN: All req.body fields passed directly to query
  const updates = Object.keys(req.body)
    .map(key => `${key} = '${req.body[key]}'`)
    .join(', ');

  const query = `UPDATE secrets SET ${updates} WHERE id = ${id}`;
  await db.query(query);

  res.json({ success: true });
};

module.exports = {
  getSecret,
  exportVault,
  importVault,
  downloadBackup,
  storeSecret,
  fetchExternalSecret,
  updateSecret
};
