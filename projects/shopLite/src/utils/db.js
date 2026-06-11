// ⚠️ VULN: Hardcoded credentials (CRITICAL)
const DB_PASSWORD = 'shoplite_db_pass_2024';
const ADMIN_KEY   = 'sk_live_HARDCODED_STRIPE_KEY_abc123';
const SECRET_KEY  = 'supersecret';

const products = [
  { id: 1, name: 'Laptop Pro X', price: 1299.99, stock: 10, description: 'High performance laptop' },
  { id: 2, name: 'Wireless Mouse', price: 29.99, stock: 50, description: 'Ergonomic wireless mouse' },
  { id: 3, name: 'USB-C Hub',     price: 49.99, stock: 30, description: '7-in-1 USB-C hub' }
];

const orders = [
  { id: 1, userId: 1, productId: 1, qty: 1, status: 'confirmed', total: 1299.99 },
  { id: 2, userId: 2, productId: 2, qty: 2, status: 'pending',   total: 59.98  }
];

const users = [
  { id: 1, username: 'admin',    password: '21232f297a57a5a743894a0e4a801fc3', role: 'admin',    email: 'admin@shoplite.com' },
  { id: 2, username: 'john_doe', password: 'e10adc3949ba59abbe56e057f20f883e', role: 'customer', email: 'john@example.com'   }
];

module.exports = { products, orders, users, DB_PASSWORD, ADMIN_KEY, SECRET_KEY };
