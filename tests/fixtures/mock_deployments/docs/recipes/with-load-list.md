---
status: Blueprint (validated)
languages: [python, typescript]
capabilities:
  - obs.langfuse
load_list:
  - {path: ../patterns/react.md, required: true}
  - {path: ../frameworks/pydantic-ai.md, required: true, when: "language == 'python'"}
  - {path: ../frameworks/vercel-ai-sdk.md, required: true, when: "language == 'typescript'"}
  - {path: ../cross-cutting/logging-structured.md, required: false}
  - {path: ../cross-cutting/observability.md, required: false, when: "capabilities contains 'obs.langfuse'"}
  - {path: ../cross-cutting/multi-tenancy.md, required: false, when: "capabilities contains 'multi-tenancy'"}
---

# With Load List

A recipe whose body intentionally mentions NOTHING — every loaded doc
comes from the structured `load_list:` block above. Exercises the D6-follow
integration in `context.assemble`.
