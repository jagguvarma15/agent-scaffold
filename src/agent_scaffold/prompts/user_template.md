# Language hints

{language_hints_yaml}

# Agent specification

The following is the full specification for the agent. It includes the selected recipe and all referenced patterns, framework guides, stack components, and cross-cutting concerns from the agent-deployments repo. Treat this as a single coherent spec.

{assembled_context}

<!-- ===== CACHE SPLIT ===== -->

# Project

Name: {project_name}
Target language: {target_language}
{role_block}{capabilities_block}{refinement_block}
# Output format

Return a single JSON object, and nothing else, matching this schema:

{
  "project_name": string,
  "language": string,
  "files": [
    { "path": string, "content": string }
  ],
  "post_install": [string],
  "smoke_check": string,
  "known_limitations": [string]
}

Path rules:
- Paths are relative to the project root, using forward slashes
- No leading slash, no "..", no absolute paths
- Every path in `files` must be unique

Smoke check rules:
- `smoke_check` is a single command executed without a shell
- No shell operators, pipes, or redirection (no `&&`, `;`, `|`, `>`)
- The first word must be one of: uv, python, python3, pnpm, node, npx, curl
- Follow the smoke check command shape from the language hints

Required files:
- The manifest file specified in language hints
- An entry point at the path specified in language hints
- README.md
- .env.example
- At least one test or smoke-check script
{extra_required_block}

Begin.
