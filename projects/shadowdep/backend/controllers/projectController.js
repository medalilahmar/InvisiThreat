'use strict';

const db      = require('../config/database');
const multer  = require('multer');
const path    = require('path');

/* ─────────────────────────────────────────────────────────────────
 * FILE UPLOAD STORAGE
 * SAST Finding: CWE-434 — Unrestricted Upload of File with Dangerous Type
 * ───────────────────────────────────────────────────────────────── */
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, '..', 'uploads')); // Stored in web-accessible folder
  },
  filename: (req, file, cb) => {
    // FIXME: Uses original filename — path traversal possible (e.g., ../../app.js)
    // FIXME: No sanitization of filename — stored as-is
    cb(null, file.originalname);
  }
});

const upload = multer({
  storage,
  // FIXME: No fileFilter — accepts any MIME type (PHP, HTML, EXE, etc.)
  // FIXME: No size limit — DoS via large file uploads
  // Secure version:
  // fileFilter: (req, file, cb) => {
  //   const allowed = ['image/jpeg','image/png','application/pdf'];
  //   cb(null, allowed.includes(file.mimetype));
  // },
  // limits: { fileSize: 5 * 1024 * 1024 }
});

/* ─────────────────────────────────────────────────────────────────
 * SEARCH PROJECTS
 * SAST Finding: CWE-89 — SQL Injection
 * DAST Finding: Reflected XSS (via EJS template) and SQLi
 *
 * Test payloads:
 *   ?q=' OR '1'='1
 *   ?q=' UNION SELECT id,username,password,role,NULL,NULL FROM users--
 *   ?q=<script>alert(document.cookie)</script>
 * ───────────────────────────────────────────────────────────────── */
const searchProjects = async (req, res, next) => {
  const { q = '' } = req.query;

  try {
    // ❌ VULNERABLE: Raw string concatenation — SQL Injection
    const query =
      "SELECT * FROM projects " +
      "WHERE name ILIKE '%" + q + "%' OR description ILIKE '%" + q + "%'";

    // ✅ Secure parameterized version (commented out for demo):
    // const query  = 'SELECT * FROM projects WHERE name ILIKE $1 OR description ILIKE $1';
    // const result = await db.query(query, [`%${q}%`]);

    const result = await db.query(query);

    // Reflected XSS: `q` is passed unescaped to EJS template
    // In search.ejs: <h1>Results for: <%- query %></h1>   ← <%- is unescaped
    res.render('search', { query: q, projects: result.rows });
  } catch (err) {
    next(err); // errorHandler will leak the SQL query in its response
  }
};

/* ─────────────────────────────────────────────────────────────────
 * GET ALL PROJECTS (renders dashboard)
 * ───────────────────────────────────────────────────────────────── */
const getAllProjects = async (req, res, next) => {
  try {
    const result = await db.query('SELECT * FROM projects ORDER BY created_at DESC');
    res.render('dashboard', { projects: result.rows, user: req.user || {} });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * GET SINGLE PROJECT
 * DAST Finding: CWE-639 — IDOR (Insecure Direct Object Reference)
 * DAST Finding: Stored XSS via project.description (rendered unescaped)
 *
 * Any user can read ANY project by enumerating IDs (sequential integers).
 * No ownership or permission check performed.
 * ───────────────────────────────────────────────────────────────── */
const getProject = async (req, res, next) => {
  const { id } = req.params;

  try {
    // FIXME: Missing ownership check. Secure version:
    // WHERE id = $1 AND user_id = $2  [req.params.id, req.user.id]
    const result = await db.query('SELECT * FROM projects WHERE id = $1', [id]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'Project not found' });
    }

    const project = result.rows[0];

    // SSTI demo: if ?name= param is provided, it is injected into template context
    // e.g. /projects/1?userInput=<%= process.env.JWT_SECRET %>
    const userInput = req.query.userInput; // FIXME: SSTI via EJS <%- userInput %>

    res.render('project', { project, user: req.user || {}, userInput });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * CREATE PROJECT
 * SAST Finding: CWE-915 — Mass Assignment (user_id from body)
 * ───────────────────────────────────────────────────────────────── */
const createProject = async (req, res, next) => {
  try {
    // FIXME: user_id accepted from body — attacker can create project on behalf of any user
    const { name, description, user_id, status, priority } = req.body;

    // BUG: Should always use req.user.id — never trust client-supplied user_id
    const effectiveUserId = user_id || (req.user && req.user.id);

    const result = await db.query(
      'INSERT INTO projects (name, description, user_id, status, priority) VALUES ($1, $2, $3, $4, $5) RETURNING *',
      [name, description, effectiveUserId, status || 'active', priority || 'medium']
    );

    res.status(201).json(result.rows[0]);
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * UPDATE PROJECT
 * DAST Finding: CWE-639 — IDOR (no ownership check)
 * ───────────────────────────────────────────────────────────────── */
const updateProject = async (req, res, next) => {
  const { id } = req.params;
  try {
    const { name, description, status, priority } = req.body;
    // FIXME: No check that req.user.id owns this project
    const result = await db.query(
      'UPDATE projects SET name=$1, description=$2, status=$3, priority=$4, updated_at=NOW() WHERE id=$5 RETURNING *',
      [name, description, status, priority, id]
    );
    res.json(result.rows[0]);
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * DELETE PROJECT
 * DAST Finding: CWE-639 — IDOR
 * ───────────────────────────────────────────────────────────────── */
const deleteProject = async (req, res, next) => {
  const { id } = req.params;
  try {
    // FIXME: No ownership check
    await db.query('DELETE FROM projects WHERE id = $1', [id]);
    res.json({ message: 'Project deleted' });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * FILE UPLOAD
 * SAST Finding: CWE-434 — Unrestricted file type
 * Files stored in /uploads which is served statically and directory-listed
 * ───────────────────────────────────────────────────────────────── */
const uploadAttachment = (req, res, next) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  // FIXME: Original filename used — contains no sanitization
  // Uploaded .php, .html, .sh files are directly accessible via /uploads/<filename>
  res.json({
    message:  'File uploaded successfully',
    filename: req.file.originalname,
    path:     `/uploads/${req.file.originalname}`,  // Publicly accessible URL
    mimetype: req.file.mimetype,
    size:     req.file.size
  });
};

module.exports = {
  searchProjects,
  getAllProjects,
  getProject,
  createProject,
  updateProject,
  deleteProject,
  uploadAttachment,
  upload
};
