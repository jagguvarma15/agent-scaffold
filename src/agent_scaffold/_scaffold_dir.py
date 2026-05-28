"""Single source of truth for the per-project ``.scaffold/`` directory name.

Every module that wants to write into the generated project's metadata
directory imports :data:`SCAFFOLD_DIR` from here rather than spelling
the literal. Previously ``".scaffold"`` was hardcoded in six places
(``manifest.py``, ``orchestrator.py``, ``template_snapshot.py``, two
inline sites in ``cli.py`` / ``steps/commit_push.py``, plus the
gitignore-defaults list in ``writer.py``); changing the name would have
required a grep + audit. Now it's one constant.

This is intentionally a tiny module with no imports — every other
module in the codebase can depend on it without fear of cycles.
"""

from __future__ import annotations

SCAFFOLD_DIR = ".scaffold"
"""Project-local metadata directory. Contains ``manifest.json``,
``state.json``, ``template-snapshots/``, and the in-progress journal."""
