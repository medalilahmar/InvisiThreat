const db = require('../utils/database');
const crypto = require('crypto');

// ⚠️ VULN: IDOR - no ownership validation (HIGH)
const getProfile = async (req, res) => {
  const { id } = req.params;

  // Any user can view any profile just by changing the ID
  const query = `SELECT * FROM users WHERE id = ${id}`;
  const user = await db.query(query);

  res.json(user[0]);
};

// ⚠️ VULN: Open Redirect (MEDIUM)
const redirectAfterLogin = async (req, res) => {
  const { redirect } = req.query;

  // No validation of redirect URL - open redirect
  res.redirect(redirect);
};

// ⚠️ VULN: Username enumeration via different error messages (LOW)
const checkUsername = async (req, res) => {
  const { username } = req.params;

  const query = `SELECT id FROM users WHERE username = '${username}'`;
  const user = await db.query(query);

  if (user && user.length > 0) {
    res.json({ exists: true, message: 'Username already taken' });
  } else {
    res.json({ exists: false, message: 'Username available' });
  }
};

// ⚠️ VULN: Reflected XSS in search (HIGH)
const searchUsers = async (req, res) => {
  const { q } = req.query;

  const query = `SELECT username, email FROM users WHERE username LIKE '%${q}%'`;
  const results = await db.query(query);

  // Reflected XSS: q echoed back without sanitization
  res.send(`
    <html>
      <body>
        <h2>Search results for: ${q}</h2>
        <pre>${JSON.stringify(results, null, 2)}</pre>
      </body>
    </html>
  `);
};

module.exports = { getProfile, redirectAfterLogin, checkUsername, searchUsers };
