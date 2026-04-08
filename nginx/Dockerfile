# nginx/Dockerfile
FROM nginx:alpine

# Устанавливаем curl для healthcheck (опционально)
RUN apk add --no-cache curl

# Копируем конфигурацию nginx
COPY nginx/nginx.conf /etc/nginx/nginx.conf
COPY nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf

# Копируем собранный фронтенд в стандартную директорию nginx
COPY client/dist/ /usr/share/nginx/html/

# Создаём директорию для логов (на случай, если не существует)
RUN mkdir -p /var/log/nginx && \
    touch /var/log/nginx/access.log /var/log/nginx/error.log && \
    chown -R nginx:nginx /var/log/nginx

# Переключаемся на не-root пользователя (безопасность)
USER nginx

EXPOSE 80

# Healthcheck для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]