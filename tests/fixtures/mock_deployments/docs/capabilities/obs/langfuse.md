---
id: obs.langfuse
kind: obs
provides: [tracing, llm_observability]
env_vars: [LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]
docker:
  service: langfuse
  image: langfuse/langfuse:2
  ports: ["3001:3000"]
probe: langfuse_health
bootstrap_step: bootstrap_langfuse
docs: |
  Langfuse self-hosted LLM observability (test fixture).
---

# Capability: obs.langfuse

Test fixture body.
