const marked = require('marked');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const { products } = require('../utils/db');

// ⚠️ VULN: XSS — description rendue en HTML sans sanitisation (HIGH)
const getProduct = (req, res) => {
  const id = parseInt(req.params.id);
  const product = products.find(p => p.id === id);

  if (!product) return res.status(404).json({ message: 'Product not found' });

  // Marked renders user-controlled content as raw HTML
  const renderedDesc = marked(product.description || '');
  res.json({ ...product, description_html: renderedDesc });
};

// ⚠️ VULN: XSS réfléchi dans la recherche (HIGH)
const searchProducts = (req, res) => {
  const { q } = req.query;

  const results = products.filter(p =>
    p.name.toLowerCase().includes((q || '').toLowerCase())
  );

  // q renvoyé sans échappement dans la réponse HTML
  res.send(`
    <html><body>
      <h2>Résultats pour : ${q}</h2>
      <pre>${JSON.stringify(results, null, 2)}</pre>
    </body></html>
  `);
};

// ⚠️ VULN: Command Injection via le champ category (CRITICAL)
const filterByCategory = (req, res) => {
  const { category } = req.query;

  // Injection directe dans exec()
  exec(`grep -r "${category}" /tmp/products/`, (err, stdout, stderr) => {
    res.json({ results: stdout, error: stderr });
  });
};

// ⚠️ VULN: Path Traversal sur le téléchargement d'image (HIGH)
const getProductImage = (req, res) => {
  const { filename } = req.query;

  // Aucune validation du chemin — permet ../../etc/passwd
  const filePath = path.join(__dirname, '../../public/images', filename);
  res.sendFile(filePath);
};

// ⚠️ VULN: Mass assignment — champs non filtrés (HIGH)
const createProduct = (req, res) => {
  const newProduct = {
    id: products.length + 1,
    ...req.body  // L'utilisateur contrôle tous les champs, y compris "role", "price"
  };
  products.push(newProduct);
  res.status(201).json(newProduct);
};

module.exports = { getProduct, searchProducts, filterByCategory, getProductImage, createProduct };
