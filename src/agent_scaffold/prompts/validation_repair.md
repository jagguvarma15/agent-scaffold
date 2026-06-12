You are fixing validation failures in an AI agent project that was just generated. The project is already on disk; a validation command failed. Emit ONLY the files that must change to make the command pass.

# Output format (strict)

Return a single JSON object, and nothing else, matching this schema:

{
  "files": [
    { "path": string, "content": string }
  ]
}

Rules:
- Include ONLY files whose content must change to fix the failure (plus a new file only when the failure itself demands one, e.g. a missing `__init__.py`). Unchanged files must NOT appear.
- Each `content` is the COMPLETE new file body — not a diff, not a fragment, not an excerpt.
- Paths are relative to the project root, using forward slashes. No leading slash, no "..", no absolute paths. Every path must be unique.
- Keep the fix minimal: do not refactor, do not rename public identifiers, do not change behavior beyond what the failing command requires.
- The result must be lint-clean and consistent with the project's existing imports, type hints, and style.

# Language hints

{language_hints_yaml}

# Project recipe (the spec this project implements — for intent only)

{recipe_body}

# Project files on disk

{project_file_list}

# Failing command

`{failing_command}`

# Validation output

```
{validation_output}
```

# Files most likely implicated (current content on disk)

{implicated_files_block}

Begin.
