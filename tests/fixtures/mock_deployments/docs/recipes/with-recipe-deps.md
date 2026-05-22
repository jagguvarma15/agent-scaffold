---
status: blueprint
languages: [python, typescript]
recipe_dependencies:
  python:
    redis: ">=5.0.0"
    structlog: ">=24.1.0"
  typescript:
    ioredis: "^5.4.0"
---

# With Recipe Deps

Recipe that declares extra per-language dependencies.
