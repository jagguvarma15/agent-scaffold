---
status: blueprint
languages: [python]
required_files:
  - /etc/passwd
  - ../escape.txt
  - Dockerfile
---

# Bad Required Files

Recipe with a mix of invalid and valid required_files entries; the bad
ones should be warned + dropped.
