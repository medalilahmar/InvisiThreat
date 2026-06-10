const xml2js = require('xml2js');
const { exec } = require('child_process');
const config = require('../config/config');
const db = require('../utils/database');

// ⚠️ VULN: Broken Access Control - no admin role check (CRITICAL)
const getAllUsers = async (req, res) => {
  try {
    // No authorization check - any authenticated user can access admin panel
    const query = 'SELECT * FROM users';  // Returns ALL fields including passwords
    const users = await db.query(query);

    res.json({ users });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: XXE - XML External Entity Injection (HIGH)
const importConfig = async (req, res) => {
  const { xmlData } = req.body;

  // VULN: XML parsed without disabling external entities
  const parser = new xml2js.Parser({
    explicitArray: false,
    // Missing: strict: true, or entity restrictions
  });

  try {
    const result = await parser.parseStringPromise(xmlData);
    res.json({ success: true, config: result });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: Command Injection in system diagnostic (CRITICAL)
const runDiagnostic = async (req, res) => {
  const { host } = req.query;

  // CRITICAL: User input in shell command - Command Injection
  exec(`ping -c 4 ${host}`, (err, stdout, stderr) => {
    res.json({
      output: stdout,
      error: stderr
    });
  });
};

// ⚠️ VULN: No CSRF protection on state-changing operations (MEDIUM)
// ⚠️ VULN: Privilege escalation via direct role update (CRITICAL)
const updateUserRole = async (req, res) => {
  const { userId, role } = req.body;

  // No CSRF token check, no current-password verification
  const query = `UPDATE users SET role = '${role}' WHERE id = ${userId}`;
  await db.query(query);

  res.json({ success: true, message: `User ${userId} role updated to ${role}` });
};

// ⚠️ VULN: Sensitive data in logs (MEDIUM)
const getLogs = async (req, res) => {
  const { level } = req.query;

  exec(`cat /var/log/securevault/app.log | grep ${level}`, (err, stdout) => {
    // Logs may contain passwords, tokens, PII
    res.json({ logs: stdout });
  });
};

// ⚠️ VULN: Debug endpoint left in production (HIGH)
const debugInfo = async (req, res) => {
  res.json({
    environment: process.env,        // Exposes ALL env variables
    config: config,                  // Exposes hardcoded secrets
    nodeVersion: process.version,
    platform: process.platform,
    memoryUsage: process.memoryUsage(),
    cwd: process.cwd(),
    uptime: process.uptime()
  });
};

module.exports = {
  getAllUsers,
  importConfig,
  runDiagnostic,
  updateUserRole,
  getLogs,
  debugInfo
};
