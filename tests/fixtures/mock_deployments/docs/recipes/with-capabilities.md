---
status: blueprint
languages: [python]
capabilities:
  - cache.redis
  - vector_db.qdrant
  - host.vercel
  - vector_db.nonexistent
  - BAD_FORMAT
  - cache.redis
---

# With Capabilities

Test recipe declaring a mix of resolvable + unresolvable capability ids, plus a
malformed one (`BAD_FORMAT`) that should be warned + dropped during recipe
parsing, and a duplicate (`cache.redis` twice).
