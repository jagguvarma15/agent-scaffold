"""Best-effort import-neighbour discovery for ``regenerate``.

Per-file regen prompts the model with the target file plus a handful of
related files so it can preserve function signatures and identifiers that
other files depend on. "Related" is defined here as files the target imports
from or files that import from the target.

This is intentionally a heuristic. We use ``ast.parse`` for Python imports
(which is exact) and a regex for TypeScript/JavaScript (which is fuzzy). The
output is context-only — wrong neighbours cost a few tokens of prompt budget
but never affect correctness of the regenerated file.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_TS_IMPORT_RE = re.compile(r'import\s+(?:[^;]+?\s+from\s+)?["\']([^"\']+)["\']')

# Source directories to scan for neighbour candidates. Keeping the search
# scoped to ``src/`` and ``tests/`` avoids walking node_modules / .venv etc.
_SCAN_DIRS = ("src", "tests")


def _python_path_to_module(rel_path: str) -> str:
    """Convert e.g. ``src/demo_agent/main.py`` → ``demo_agent.main``."""
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _python_outgoing_modules(source: str) -> set[str]:
    """Extract every module name imported by ``source`` via ast.parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
    return modules


def _ts_outgoing_modules(source: str) -> set[str]:
    return {m.group(1) for m in _TS_IMPORT_RE.finditer(source)}


def _scan_files(project_dir: Path, suffixes: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for sub in _SCAN_DIRS:
        root = project_dir / sub
        if not root.is_dir():
            continue
        for ext in suffixes:
            found.extend(root.rglob(f"*{ext}"))
    return found


def _is_python(path: Path) -> bool:
    return path.suffix == ".py"


def _is_typescript(path: Path) -> bool:
    return path.suffix in {".ts", ".tsx", ".js", ".jsx"}


def discover_neighbours(
    project_dir: Path,
    target_rel: str,
    *,
    max_neighbours: int = 6,
) -> list[Path]:
    """Return up to ``max_neighbours`` files related to ``target_rel`` by imports.

    Includes both directions: files that ``target_rel`` imports from, and
    files that import from ``target_rel``. Results are stable-ordered
    (alphabetical by relative path) so the prompt is deterministic across
    runs.
    """
    target_abs = project_dir / target_rel
    if not target_abs.is_file():
        return []
    source = target_abs.read_text(encoding="utf-8")

    if _is_python(target_abs):
        outgoing = _python_outgoing_modules(source)
        candidates = _scan_files(project_dir, (".py",))
        target_module = _python_path_to_module(target_rel)
        path_to_module = {p: _python_path_to_module(str(p.relative_to(project_dir))) for p in candidates}
    elif _is_typescript(target_abs):
        outgoing = _ts_outgoing_modules(source)
        candidates = _scan_files(project_dir, (".ts", ".tsx", ".js", ".jsx"))
        target_module = target_rel  # TS imports use relative paths; coarse compare.
        path_to_module = {p: str(p.relative_to(project_dir)) for p in candidates}
    else:
        return []

    neighbours: set[Path] = set()
    for cand in candidates:
        if cand == target_abs:
            continue
        cand_text = cand.read_text(encoding="utf-8", errors="replace")
        # Incoming: ``cand`` mentions our module path.
        if target_module and target_module in cand_text:
            neighbours.add(cand)
            continue
        # Outgoing: target imports something whose module matches this file.
        cand_module = path_to_module.get(cand, "")
        if not cand_module:
            continue
        for module in outgoing:
            if module == cand_module or module.startswith(cand_module + "."):
                neighbours.add(cand)
                break

    ordered = sorted(neighbours, key=lambda p: str(p.relative_to(project_dir)))
    return ordered[:max_neighbours]
