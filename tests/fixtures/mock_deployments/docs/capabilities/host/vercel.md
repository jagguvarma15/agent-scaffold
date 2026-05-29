---
id: host.vercel
kind: host
env_vars: [VERCEL_TOKEN]
bootstrap_step: emit_deploy_configs
emit_files:
  - source: templates/vercel.json
    dest: vercel.json
deploy_configs:
  - target: vercel
    cli_cmd: "vercel deploy --prod"
    dashboard_url: "https://vercel.com/dashboard"
    config_file: vercel.json
---

# Capability: host.vercel

Test fixture: deploy hint.
