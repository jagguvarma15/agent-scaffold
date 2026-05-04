"""Bundled agent-deployments docs for zero-config usage.

The docs/ directory is populated at build time by scripts/sync_deployments.sh.
When installed via PyPI/Homebrew, the bundled docs allow agent-forge to work
without requiring a separate agent-deployments clone.
"""

from __future__ import annotations

from pathlib import Path


def bundled_docs_path() -> Path:
    """Return the path to the bundled deployments root (contains docs/)."""
    return Path(__file__).parent
