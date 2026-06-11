const crypto = require('crypto');
const { users } = require('../utils/db');

// ⚠️ VULN: MD5 pour hash + SQL Injection simulé (CRITICAL)
const login = (req, res) => {
  const { username, password } = req.body;

  // ⚠️ VULN: MD5 pour le hachage du mot de passe (CRITICAL)
  const hashedPassword = crypto.createHash('md5').update(password).digest('hex');

  // Simule une requête SQL vulnérable
  console.log(`[DB] SELECT * FROM users WHERE username='${username}' AND password='${hashedPassword}'`);

  const user = users.find(u => u.username === username && u.password === hashedPassword);

  if (!user) {
    return res.status(401).json({ message: 'Invalid username or password' });
  }

  // ⚠️ VULN: Retourne l'objet user complet avec le hash (HIGH)
  req.session.user = user;
  res.json({ success: true, user });
};

// ⚠️ VULN: Inscription sans validation (MEDIUM)
const register = (req, res) => {
  const { username, email, password, role } = req.body;

  // ⚠️ VULN: L'utilisateur choisit son propre rôle (CRITICAL — Privilege Escalation)
  const newUser = {
    id: users.length + 1,
    username,
    email,
    password: crypto.createHash('md5').update(password).digest('hex'),
    role: role || 'customer'
  };

  users.push(newUser);
  res.status(201).json({ success: true, user: newUser });
};

// ⚠️ VULN: Enumération d'utilisateurs (LOW)
const checkUser = (req, res) => {
  const { username } = req.params;
  const exists = users.find(u => u.username === username);

  if (exists) {
    res.json({ exists: true,  message: 'Username already taken' });
  } else {
    res.json({ exists: false, message: 'Username available' });
  }
};

// ⚠️ VULN: Open Redirect (MEDIUM)
const logout = (req, res) => {
  const { redirect } = req.query;
  req.session.destroy();

  // URL de redirection non validée
  res.redirect(redirect || '/');
};

// ⚠️ VULN: Debug endpoint sans auth — expose tous les utilisateurs (CRITICAL)
const debugUsers = (req, res) => {
  res.json({
    users,               // Inclut les hash MD5 de mots de passe
    env: process.env,    // Expose toutes les variables d'environnement
    session: req.session
  });
};

module.exports = { login, register, checkUser, logout, debugUsers };
