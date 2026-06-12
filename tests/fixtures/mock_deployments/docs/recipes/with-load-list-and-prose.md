---
status: Blueprint (design spec)
languages: [python]
load_list:
  - {path: ../patterns/react.md, required: true, cache_tier: hot}
  - {path: ../cross-cutting/logging-structured.md, required: false}
---

# With Load List And Prose

This recipe declares a structured `load_list:` AND its prose deliberately
mentions alias bait: the stack uses Qdrant for storage, and production
deployments care about rate limiting. With a load_list present, the alias
and cross-cutting prose scans must NOT fire — the author's declaration wins.
