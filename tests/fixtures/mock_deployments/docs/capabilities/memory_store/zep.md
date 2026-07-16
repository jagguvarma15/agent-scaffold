---
id: memory_store.zep
kind: memory_store
provides: [conversation_memory]
env_vars: [ZEP_API_KEY]
docker:
  service: zep
  image: ghcr.io/getzep/zep:latest-fixture
  ports: ["8010:8000"]
docs: |
  Zep conversation memory store.
---

# Capability: memory_store.zep

Test fixture body.
