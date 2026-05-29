---
id: cache.redis
kind: cache
provides: [cache, session_store]
env_vars: [REDIS_URL]
docker:
  service: redis
  image: redis:7-alpine
  ports: ["6379:6379"]
probe: redis_ping
docs: |
  Redis for cache and session storage.
---

# Capability: cache.redis

Test fixture body.
