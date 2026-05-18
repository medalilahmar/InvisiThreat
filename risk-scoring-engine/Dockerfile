FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source complet (tous les sous-modules)
COPY src/ ./src/

# Créer les répertoires montés par les PVC
RUN mkdir -p /app/models \
             /app/data/raw \
             /app/data/processed \
             /app/logs

# Port exposé
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Démarrage
CMD ["uvicorn", "src.api_simple:app", \
     "--host", "0.0.0.0", \
     "--port", "8081", \
     "--workers", "2"]