# ❌ VULNERABLE: Outdated base image with known vulnerabilities (CVEs)
FROM node:12.18.3

# ❌ SECURITY MISCONFIGURATION: Run everything as root
USER root

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --unsafe-perm

# Copy source code
COPY . .

# ❌ SECURITY MISCONFIGURATION: Hardcoded credential built into the image
ENV APP_SECRET_KEY="super-secret-build-key-do-not-expose"
ENV AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEDOCKER"

EXPOSE 5000

CMD ["node", "app.js"]
