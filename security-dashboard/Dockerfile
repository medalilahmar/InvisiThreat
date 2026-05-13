# ====================== BUILD STAGE ======================
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm ci --frozen-lockfile --production=false

COPY . .

ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL:-http://localhost:8081}

RUN npm run build

FROM nginx:alpine AS production

RUN apk add --no-cache curl && \
    rm -rf /var/cache/apk/* && \
    addgroup -g 101 -S nginx && \
    adduser -S -D -H -u 101 -G nginx nginx

RUN rm /etc/nginx/conf.d/default.conf

COPY nginx-spa.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

RUN chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 555 /usr/share/nginx/html

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

EXPOSE 80

USER nginx

CMD ["nginx", "-g", "daemon off;"]