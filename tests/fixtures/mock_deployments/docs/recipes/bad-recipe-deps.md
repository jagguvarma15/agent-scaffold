---
status: blueprint
languages: [python]
recipe_dependencies:
  - not
  - a
  - dict
---

# Bad Recipe Deps

Recipe with malformed recipe_dependencies; should warn and be treated as empty.
