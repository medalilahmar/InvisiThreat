// ⚠️ VULN: Hardcoded credentials & secrets (CRITICAL)
module.exports = {
  db: {
    host: 'localhost',
    port: 27017,
    name: 'securevault_db',
    // Hardcoded credentials - CRITICAL vulnerability
    username: 'admin',
    password: 'Admin@1234!'
  },
  jwt: {
    // Weak hardcoded JWT secret - CRITICAL
    secret: 'jwt_secret_key',
    expiresIn: '365d'
  },
  encryption: {
    // Hardcoded encryption key - CRITICAL
    key: '1234567890abcdef',
    iv:  'abcdef1234567890',
    algorithm: 'des'  // ⚠️ VULN: DES is broken (HIGH)
  },
  aws: {
    // Hardcoded AWS keys - CRITICAL
    accessKeyId: 'AKIAIOSFODNN7EXAMPLE',
    secretAccessKey: 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
    region: 'us-east-1'
  },
  admin: {
    // Default admin credentials - HIGH
    defaultUser: 'admin',
    defaultPassword: 'admin123'
  },
  smtp: {
    host: 'smtp.securevault.com',
    port: 587,
    user: 'noreply@securevault.com',
    // Hardcoded SMTP password - HIGH
    password: 'Smtp@Pass2024'
  }
};
