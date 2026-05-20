FROM node:20-alpine AS builder

WORKDIR /app

# ── Layer cache npm ──────────────────────────────────────
# Ces deux lignes DOIVENT rester ensemble et stables
# pour que le cache soit réutilisé entre les builds
COPY package.json package-lock.json ./

RUN npm ci --ignore-scripts && \
    npm rebuild esbuild && \
    npm cache clean --force

# ── Code source ──────────────────────────────────────────
# Copié APRÈS npm ci pour ne pas invalider le cache npm
COPY . .

ARG VITE_API_URL=
ARG VITE_APP_NAME=InvisiThreat
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_APP_NAME=$VITE_APP_NAME

RUN npm run build

# ── Image finale ─────────────────────────────────────────
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx-spa.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]