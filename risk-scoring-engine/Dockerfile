FROM python:3.11-slim

WORKDIR /app

# ── Dépendances système ───────────────────────────────────────
# Ce layer ne change que si vous modifiez cette liste
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances Python (layer caché tant que requirements.txt ne change pas) ──
# IMPORTANT : copier UNIQUEMENT requirements.txt avant le reste du code.
# Docker invalide le cache d'un layer seulement si les fichiers
# copiés AVANT lui ont changé. Tant que requirements.txt est identique,
# le pip install ne sera JAMAIS réexécuté, même si src/ change.
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# ── Code source (copié EN DERNIER pour maximiser le cache) ────
# Ce layer est invalidé à chaque modification de src/
# mais il ne déclenche plus le pip install
COPY src/ ./src/

# ── Répertoires montés par les PVC Kubernetes ─────────────────
RUN mkdir -p /app/models \
             /app/data/raw \
             /app/data/processed \
             /app/logs

# ── Port ──────────────────────────────────────────────────────
EXPOSE 8081

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# ── Démarrage ─────────────────────────────────────────────────
CMD ["uvicorn", "src.api_simple:app", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "--workers", "2"]