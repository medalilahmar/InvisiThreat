const config = require('../config/config');

// Mock database for demo purposes
// In production this would connect to MySQL/MongoDB

// ⚠️ VULN: Connection string with credentials in code (HIGH)
const connectionString = `mysql://${config.db.username}:${config.db.password}@${config.db.host}:${config.db.port}/${config.db.name}`;

const mockData = {
  users: [
    { id: 1, username: 'admin', email: 'admin@securevault.com', password: '5f4dcc3b5aa765d61d8327deb882cf99', role: 'admin', internal_id: 'SV-001' },
    { id: 2, username: 'john.doe', email: 'john@example.com', password: 'e10adc3949ba59abbe56e057f20f883e', role: 'user', internal_id: 'SV-002' },
    { id: 3, username: 'jane.smith', email: 'jane@example.com', password: '827ccb0eea8a706c4c34a16891f84e7b', role: 'user', internal_id: 'SV-003' }
  ],
  secrets: [
    { id: 1, name: 'AWS Production Keys', value: 'enc_abc123...', category: 'cloud', user_id: 1, notes: 'Production AWS credentials' },
    { id: 2, name: 'Database Password', value: 'enc_def456...', category: 'database', user_id: 2, notes: 'Main DB password' },
    { id: 3, name: 'GitHub Token', value: 'enc_ghi789...', category: 'vcs', user_id: 2, notes: 'Personal access token' }
  ]
};

// Simulated async query function
const query = async (sql) => {
  console.log('[DB] Executing:', sql);

  // Simple mock responses based on query type
  if (sql.toLowerCase().includes('select * from users')) {
    return mockData.users;
  }
  if (sql.toLowerCase().includes('select * from secrets')) {
    return mockData.secrets;
  }
  if (sql.toLowerCase().includes('insert')) {
    return { affectedRows: 1, insertId: Math.floor(Math.random() * 1000) };
  }
  if (sql.toLowerCase().includes('update')) {
    return { affectedRows: 1 };
  }

  return [];
};

module.exports = { query, connectionString };
