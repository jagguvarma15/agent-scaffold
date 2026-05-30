---
id: frontend.nextjs-tiny
kind: frontend
env_vars: [NEXT_PUBLIC_AGENT_URL]
emit_files:
  - source: templates/nextjs-tiny/**
    dest: frontend/
---

# Tiny frontend fixture

Used by tests/test_capability_emit.py to exercise the glob copy path
without bloating the test tree with a real Next.js project.
