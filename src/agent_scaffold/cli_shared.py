"""Shared CLI singletons used across ``cli.py`` and the ``cli_*`` sub-modules.

Lives in its own module so the extracted command groups (``cli_auth``,
``cli_doctor``, ``cli_secrets``) can import ``console`` without creating
an import cycle through ``cli.py``. The Console is constructed lazily —
import-only — so test fixtures that monkeypatch this attribute see a
single object every site shares.

This is intentionally tiny. New shared helpers belong here only when at
least two command modules need them and they would otherwise force the
new modules to import from ``cli.py``.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
