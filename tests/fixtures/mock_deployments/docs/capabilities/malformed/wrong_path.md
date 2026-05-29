---
id: vector_db.qdrant
kind: vector_db
env_vars: [QDRANT_URL]
---

# Wrong path

This file's id (`vector_db.qdrant`) doesn't match its path (`malformed/wrong_path`).
Loader should reject this file and keep the canonical vector_db/qdrant.md.
