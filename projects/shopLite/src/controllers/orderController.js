const serialize = require('node-serialize');
const axios_like = require('http');
const { orders } = require('../utils/db');
const crypto = require('crypto');

// ⚠️ VULN: IDOR — aucune vérification de propriété (HIGH)
const getOrder = (req, res) => {
  const { id } = req.params;

  // N'importe quel utilisateur peut lire n'importe quelle commande
  const order = orders.find(o => o.id === parseInt(id));
  if (!order) return res.status(404).json({ message: 'Order not found' });

  res.json(order);
};

// ⚠️ VULN: Insecure Deserialization (CRITICAL)
const importOrder = (req, res) => {
  const { data } = req.body;

  try {
    // Désérialisation d'un payload contrôlé par l'utilisateur → RCE
    const orderData = serialize.unserialize(data);
    orders.push({ id: orders.length + 1, ...orderData });
    res.json({ success: true, order: orderData });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};

// ⚠️ VULN: SSRF — appel HTTP vers une URL contrôlée par l'utilisateur (HIGH)
const trackShipment = (req, res) => {
  const { trackingUrl } = req.body;

  // Aucune validation — peut viser http://169.254.169.254 (metadata AWS)
  const url = new URL(trackingUrl);
  axios_like.get(trackingUrl, (response) => {
    let data = '';
    response.on('data', chunk => data += chunk);
    response.on('end', () => res.json({ tracking: data }));
  }).on('error', err => res.status(500).json({ error: err.message }));
};

// ⚠️ VULN: Token de remboursement prévisible (MEDIUM)
const requestRefund = (req, res) => {
  const { orderId } = req.body;

  // Token basé sur timestamp — prédictible
  const refundToken = Date.now().toString();

  res.json({
    success: true,
    orderId,
    refundToken,  // ⚠️ Ne devrait jamais être renvoyé dans la réponse
    message: 'Refund token generated'
  });
};

// ⚠️ VULN: Pas de vérification du montant (MEDIUM) — négatif accepté
const applyDiscount = (req, res) => {
  const { orderId, discount } = req.body;

  const order = orders.find(o => o.id === parseInt(orderId));
  if (!order) return res.status(404).json({ message: 'Not found' });

  // Aucune vérification : discount peut être négatif → prix négatif
  order.total = order.total - parseFloat(discount);
  res.json({ success: true, newTotal: order.total });
};

module.exports = { getOrder, importOrder, trackShipment, requestRefund, applyDiscount };
