You are regenerating ONE file in an existing AI agent project. The CLI will replace the file on disk verbatim with whatever you emit. Faithfulness to the surrounding code and the stated reason are the two success criteria.

# Output format (strict)

Emit exactly ONE fenced code block whose contents become the entire new file. No prose before or after the fence. No JSON. No second block. No diff.

```{language_fence}
<entire new file content here>
```

# Project recipe

{recipe_body}

# Target file

Path: `{target_path}`

Current content:

```{language_fence}
{current_content}
```

# Neighbour files (context only — DO NOT regenerate these)

These files import from the target, or are imported by the target. Use them to keep identifiers, signatures, and call shapes consistent.

{neighbours_block}

# Reason for regeneration

{reason}

# Rules

- Preserve every public identifier (functions, classes, constants) that the neighbour files reference, unless the reason explicitly tells you to rename or remove it.
- Match the project's existing style — imports, type hints, docstring conventions, error handling — based on the neighbour files.
- Output ONLY the replacement file content inside a single fenced code block. No commentary, no JSON, no markdown around the fence.
- If you need to add a new import, add it; if you need to remove an unused one, remove it. The output must be lint-clean.
