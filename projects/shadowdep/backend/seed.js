'use strict';

/**
 * ShadowDep — Database Seeder
 *
 * Creates tables and inserts:
 *   - Default admin account: admin / admin   ← SECURITY MISCONFIGURATION
 *   - Sample user:           john  / password123
 *   - 5 sample projects
 *
 * FIXME: Default credentials must be changed before any deployment
 */

const db = require('./config/database');

async function seed() {
  console.log('🌱 Starting database seed...');

  try {
    /* ── Create tables ───────────────────────────────────────────── */
    await db.query(`
      CREATE TABLE IF NOT EXISTS users (
        id         SERIAL PRIMARY KEY,
        username   VARCHAR(255) UNIQUE NOT NULL,
        email      VARCHAR(255) UNIQUE NOT NULL,
        password   VARCHAR(255) NOT NULL,   -- stored as PLAINTEXT for demo
        role       VARCHAR(50) DEFAULT 'user',
        created_at TIMESTAMP DEFAULT NOW()
      )
    `);

    await db.query(`
      CREATE TABLE IF NOT EXISTS projects (
        id          SERIAL PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        description TEXT,
        user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
        status      VARCHAR(50)  DEFAULT 'active',
        priority    VARCHAR(50)  DEFAULT 'medium',
        data        JSONB        DEFAULT '{}',
        created_at  TIMESTAMP    DEFAULT NOW(),
        updated_at  TIMESTAMP    DEFAULT NOW()
      )
    `);

    /* ── Default admin account ───────────────────────────────────── */
    // SECURITY MISCONFIGURATION: weak, well-known credentials (CWE-521)
    // Username: admin   Password: admin
    await db.query(`
      INSERT INTO users (username, email, password, role)
      VALUES ('admin', 'admin@shadowdep.local', 'admin', 'admin')
      ON CONFLICT (username) DO NOTHING
    `);

    /* ── Regular user ────────────────────────────────────────────── */
    await db.query(`
      INSERT INTO users (username, email, password, role)
      VALUES ('john', 'john@shadowdep.local', 'password123', 'user')
      ON CONFLICT (username) DO NOTHING
    `);

    /* ── Another user (to demonstrate IDOR cross-user access) ────── */
    await db.query(`
      INSERT INTO users (username, email, password, role)
      VALUES ('alice', 'alice@shadowdep.local', 'alice123', 'user')
      ON CONFLICT (username) DO NOTHING
    `);

    /* ── Sample projects ─────────────────────────────────────────── */
    const projects = [
      {
        name:        'Alpha Project',
        description: 'Main product development initiative for Q1 2024. On track for delivery.',
        status:      'active',
        priority:    'high'
      },
      {
        name:        'Beta Testing Phase',
        description: 'QA and regression testing for all core features before public release.',
        status:      'active',
        priority:    'medium'
      },
      {
        name:        'Infrastructure Upgrade',
        description: 'Migration from on-premise to cloud infrastructure. Budget: $50,000.',
        status:      'pending',
        priority:    'critical'
      },
      {
        // Stored XSS payload already embedded (shows as a project with malicious description)
        name:        'Security Audit 2024',
        description: 'Annual penetration testing and security review. <b>Status: CONFIDENTIAL</b>',
        status:      'active',
        priority:    'high'
      },
      {
        name:        'Mobile App Development',
        description: 'Cross-platform iOS and Android application. Tech: React Native.',
        status:      'active',
        priority:    'medium'
      }
    ];

    for (const p of projects) {
      await db.query(
        `INSERT INTO projects (name, description, user_id, status, priority)
         VALUES ($1, $2, 1, $3, $4)`,
        [p.name, p.description, p.status, p.priority]
      );
    }

    /* ── Alice's private project (to demo IDOR) ──────────────────── */
    await db.query(
      `INSERT INTO projects (name, description, user_id, status, priority)
       VALUES ($1, $2, 3, $3, $4)`,
      [
        "Alice's Private Project",
        'This project belongs to Alice and should NOT be visible to other users. Salary budget: $120,000.',
        'active',
        'low'
      ]
    );

    console.log('\n✅ Database seeded successfully!');
    console.log('┌─────────────────────────────────────────────┐');
    console.log('│  Default credentials (CHANGE IMMEDIATELY!)  │');
    console.log('│  Admin : admin    / admin                   │');
    console.log('│  User  : john     / password123             │');
    console.log('│  User  : alice    / alice123                │');
    console.log('└─────────────────────────────────────────────┘\n');

    process.exit(0);
  } catch (err) {
    console.error('❌ Seeding failed:', err.message);
    console.error(err.stack);
    process.exit(1);
  }
}

seed();
