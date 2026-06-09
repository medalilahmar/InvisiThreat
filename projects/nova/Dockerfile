# Utilisation d'une image de base obsolète (SCA / Docker Lint Target)
FROM python:3.9-slim

# Mauvaise pratique : exécuter l'application sous l'utilisateur root
USER root

WORKDIR /app

COPY requirements.txt .

# Installation des paquets vulnérables
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposer le port de l'application
EXPOSE 5001

# Mauvaise pratique : Clé secrète configurée via variables d'environnement statiques dans le Dockerfile
ENV AWS_SECRET_KEY="supersecretinsecurevalue"

CMD ["python", "src/nova/app.py"]
