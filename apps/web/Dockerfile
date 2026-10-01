# syntax=docker/dockerfile:1
# Web client image: build the SPA with Node, serve it with nginx.
#
# Two stages on purpose. The npm toolchain and node_modules are ~300MB and have
# no business in a serving image; nginx is a few MB and is what actually serves
# traffic.
#
#   docker build -f apps/web/Dockerfile -t son-web:dev .

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM node:22-bookworm-slim AS builder

WORKDIR /build

ENV npm_config_update_notifier=false

# Dependency layer first: package-lock.json plus every workspace manifest.
COPY package.json package-lock.json ./
COPY apps/web/package.json apps/web/package.json
COPY libs/ui-terminology/package.json libs/ui-terminology/package.json

RUN npm ci

COPY tsconfig.base.json* ./
COPY apps/web/ apps/web/
COPY libs/ui-terminology/ libs/ui-terminology/

RUN npm run build --workspace apps/web

# ---------------------------------------------------------------------------
# Runtime: nginx serving the built assets and proxying the API
# ---------------------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

RUN apk add --no-cache curl

COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY deploy/nginx/proxy_params_son.conf /etc/nginx/proxy_params_son.conf
COPY deploy/nginx/connection_upgrade.conf /etc/nginx/connection_upgrade.conf
COPY --from=builder /build/apps/web/dist /usr/share/nginx/html

# Validate the config at build time. A misplaced directive or a bad include
# otherwise only surfaces as a container that crash-loops on start.
#
# The upstream name is rewritten to loopback first: nginx resolves `server` names
# when it loads the config, and there is no `api` DNS record inside the build
# sandbox. The shipped config keeps the service name, which is what makes it
# survive the API container being recreated with a new IP.
RUN sed -i 's|server api:9969;|server 127.0.0.1:9969;|' /etc/nginx/conf.d/default.conf \
    && nginx -t \
    && sed -i 's|server 127.0.0.1:9969;|server api:9969;|' /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]