---
id: host.vercel-single
kind: host
env_vars: [VERCEL_TOKEN]
emit_files:
  - source: templates/vercel.json
    dest: vercel.json
deploy_configs:
  - target: vercel
    cli_cmd: "vercel deploy --prod"
---

# Single-file host fixture

Exercises the single-file emit path (no glob).
