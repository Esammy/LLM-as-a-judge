# The dashboard ships separately from the Python services: a different runtime,
# a different release cadence, and no reason to rebuild the API because a
# stylesheet changed.

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app

# HOSTNAME is not cosmetic here. Next's standalone server.js binds to
# process.env.HOSTNAME, and Docker sets that to the container id - so without
# this the server listens on the container's own IP alone. Published ports
# still work, because docker-proxy forwards to that IP, which is exactly what
# makes the fault easy to miss: the site loads fine from the host while every
# localhost connection inside the container is refused, and the health check
# can never pass.
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000

RUN addgroup --system --gid 1001 nodejs \
 && adduser --system --uid 1001 --ingroup nodejs nextjs

# `output: "standalone"` means Next traced exactly the files it needs, so the
# runtime image carries a minimal server rather than the whole node_modules tree.
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
