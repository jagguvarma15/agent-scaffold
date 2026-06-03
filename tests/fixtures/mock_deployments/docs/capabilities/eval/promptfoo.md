---
id: eval.promptfoo
kind: eval
provides: [eval_runs, golden_tests]
env_vars: []
bootstrap_step: bootstrap_evals
docs: |
  Promptfoo — Node-based eval runner that shells out to `npx promptfoo`.
---

# Capability: eval.promptfoo

Test fixture body. Promptfoo runs the recipe's eval suite from
`tests/eval/promptfooconfig.yaml`.
