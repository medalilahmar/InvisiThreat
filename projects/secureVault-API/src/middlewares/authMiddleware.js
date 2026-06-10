const jwt = require('jsonwebtoken');
const config = require('../config/config');

// ⚠️ VULN: JWT algorithm confusion attack possible (HIGH)
// ⚠️ VULN: No token blacklist / revocation (MEDIUM)
const authenticate = (req, res, next) => {
  const token = req.headers['authorization'] || req.query.token || req.body.token;

  if (!token) {
    return res.status(401).json({ message: 'No token provided' });
  }

  try {
    // VULN: Does not enforce algorithm - allows 'none' attack
    const decoded = jwt.verify(token.replace('Bearer ', ''), config.jwt.secret);
    req.user = decoded;
    next();
  } catch (err) {
    // ⚠️ VULN: Different error messages aid token analysis (LOW)
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ message: 'Token expired', expiredAt: err.expiredAt });
    }
    return res.status(401).json({ message: 'Invalid token', reason: err.message });
  }
};

// ⚠️ VULN: Role check is bypassable - checks string equality only (HIGH)
const requireAdmin = (req, res, next) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }

  // VULN: role check easily bypassed if token is forged
  if (req.user.role === 'admin' || req.user.isAdmin) {
    return next();
  }

  res.status(403).json({ message: 'Admin access required' });
};

module.exports = { authenticate, requireAdmin };
