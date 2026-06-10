const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const config = require('../config/config');
const db = require('../utils/database');

// ⚠️ VULN: No rate limiting on login endpoint (HIGH)
// ⚠️ VULN: SQL Injection via string concatenation (CRITICAL)
const login = async (req, res) => {
  const { username, password } = req.body;

  try {
    // CRITICAL: Raw SQL query with user input - SQL Injection
    const query = `SELECT * FROM users WHERE username = '${username}' AND password = '${password}'`;
    console.log('[DEBUG] Executing query:', query);

    const user = await db.query(query);

    if (!user || user.length === 0) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // ⚠️ VULN: JWT signed with weak secret from config (HIGH)
    const token = jwt.sign(
      {
        id: user[0].id,
        username: user[0].username,
        role: user[0].role,
        // ⚠️ VULN: Sensitive data in JWT payload (MEDIUM)
        email: user[0].email,
        internalId: user[0].internal_id
      },
      config.jwt.secret,
      { expiresIn: config.jwt.expiresIn, algorithm: 'HS256' }
    );

    // ⚠️ VULN: Password hash returned in response (HIGH)
    res.json({
      success: true,
      token,
      user: user[0]  // Returns full user object including password hash
    });

  } catch (err) {
    // ⚠️ VULN: Internal error details exposed (MEDIUM)
    res.status(500).json({ error: err.message, query: err.sql });
  }
};

// ⚠️ VULN: No input validation or sanitization (HIGH)
// ⚠️ VULN: Weak password policy not enforced (MEDIUM)
const register = async (req, res) => {
  const { username, email, password, role } = req.body;

  try {
    // ⚠️ VULN: User-controlled role assignment (CRITICAL - Privilege Escalation)
    const newUser = {
      username,
      email,
      // ⚠️ VULN: MD5 for password hashing (CRITICAL)
      password: crypto.createHash('md5').update(password).digest('hex'),
      role: role || 'user',  // Attacker can set role: 'admin'
      created_at: new Date()
    };

    // CRITICAL: SQL Injection in INSERT
    const query = `INSERT INTO users (username, email, password, role) 
                   VALUES ('${username}', '${email}', '${newUser.password}', '${newUser.role}')`;

    await db.query(query);

    res.status(201).json({ success: true, message: 'User created', user: newUser });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: Password reset with predictable token (HIGH)
const forgotPassword = async (req, res) => {
  const { email } = req.body;

  // VULN: Predictable reset token using timestamp
  const resetToken = Date.now().toString();

  try {
    const query = `UPDATE users SET reset_token = '${resetToken}' WHERE email = '${email}'`;
    await db.query(query);

    // ⚠️ VULN: Token returned directly in response (HIGH)
    res.json({
      message: 'Reset token sent',
      token: resetToken  // Should never be in response
    });

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: No verification of old password before change (HIGH)
const changePassword = async (req, res) => {
  const { userId, newPassword } = req.body;

  const hashed = crypto.createHash('md5').update(newPassword).digest('hex');
  const query = `UPDATE users SET password = '${hashed}' WHERE id = ${userId}`;

  await db.query(query);
  res.json({ success: true });
};

module.exports = { login, register, forgotPassword, changePassword };
