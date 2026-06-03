'use strict';

const express = require('express');
const router  = express.Router();
const { authenticateToken } = require('../middleware/auth');
const ctrl = require('../controllers/projectController');

// GET  /projects/search?q= → SQL injection + reflected XSS (no auth required)
// FIXME: Public search endpoint with SQLi and XSS — should require auth + parameterized query
router.get('/search', ctrl.searchProjects);

// GET  /projects → list all projects (auth required)
router.get('/', authenticateToken, ctrl.getAllProjects);

// GET  /projects/:id → IDOR — no ownership check, stored XSS in template
router.get('/:id', ctrl.getProject); // FIXME: No auth guard on individual project

// POST /projects → create project — mass assignment (user_id from body)
router.post('/', authenticateToken, ctrl.createProject);

// PUT  /projects/:id → update — IDOR, no ownership check
router.put('/:id', authenticateToken, ctrl.updateProject);

// DELETE /projects/:id → delete — IDOR, no ownership check
router.delete('/:id', authenticateToken, ctrl.deleteProject);

// POST /projects/upload → unrestricted file upload, stored in web root
// FIXME: Must be declared BEFORE /:id to avoid route collision with multer
router.post('/upload', authenticateToken, ctrl.upload.single('file'), ctrl.uploadAttachment);

// FIXME: TRACE / OPTIONS methods not blocked (CORS allows all methods)
// curl -X TRACE http://localhost:5000/projects/
// curl -X OPTIONS http://localhost:5000/projects/

module.exports = router;
