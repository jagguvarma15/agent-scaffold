---
id: obs.langsmith
kind: obs
provides: [tracing, llm_observability]
env_vars: [LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT, LANGCHAIN_ENDPOINT]
docker: null
probe: langsmith_workspace
bootstrap_step: bootstrap_langsmith
docs: |
  LangSmith hosted LLM observability (test fixture).
---

# Capability: obs.langsmith

Test fixture body.
