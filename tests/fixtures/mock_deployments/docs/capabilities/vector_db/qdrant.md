---
id: vector_db.qdrant
kind: vector_db
provides: [embeddings_store, collection_init]
env_vars: [QDRANT_URL, QDRANT_API_KEY]
docker:
  service: qdrant
  image: qdrant/qdrant:v1.12.0
  ports: ["6333:6333"]
  volumes: ["qdrant_data:/qdrant/storage"]
  environment:
    QDRANT__LOG_LEVEL: INFO
  healthcheck:
    test: ["CMD-SHELL", "wget -qO- http://localhost:6333/healthz || exit 1"]
probe: qdrant_collections
bootstrap_step: bootstrap_vector_db
emit_files: []
docs: |
  Qdrant vector DB for RAG retrieval.
---

# Capability: vector_db.qdrant

Test fixture body.

## Local setup

The docker fragment runs Qdrant on port 6333.
