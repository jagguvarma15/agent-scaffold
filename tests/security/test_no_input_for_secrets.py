"""Rule 2 audit: secret-handling modules must never use ``input(`` for prompts.

``input()`` echoes characters to the terminal. ``getpass.getpass`` doesn't.
Calling ``input("Paste your API key: ")`` would leave the key in scrollback
and in any tee'd transcript.

This test scans the modules that touch secrets and asserts ``input(`` doesn't
appear in any function whose name suggests it prompts for a credential.
``cli.py`` is **excluded** from this scan because it correctly uses ``input``
for the non-secret commit/push confirmation prompts in ``commit_push.py``.
For the same reason ``commit_push.py`` is excluded too.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agent_scaffold"

# Modules whose code paths must never call ``input(`` for any reason.
_FORBIDDEN_INPUT_MODULES: tuple[Path, ...] = (
    SRC_ROOT / "auth.py",
    SRC_ROOT / "auth_browser.py",
    SRC_ROOT / "steps" / "wire_credentials.py",
)

# Bare ``input(`` call — anchored so ``builtins.input`` and ``input_box``
# don't false-positive. Comments / docstrings are allowed (rare edge); the
# pattern is conservative and module-scoped.
_INPUT_CALL = re.compile(r"(?<![A-Za-z0-9_.])input\(")


def test_secret_modules_never_use_input() -> None:
    violations: list[str] = []
    for module in _FORBIDDEN_INPUT_MODULES:
        text = module.read_text(encoding="utf-8")
        # Strip line comments before scanning — docstring matches are still
        # technically caught but no current docstring uses ``input(``.
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if _INPUT_CALL.search(code):
                violations.append(f"{module.relative_to(SRC_ROOT)}:{lineno}: {line.strip()}")
    assert not violations, (
        "input(...) forbidden in secret-handling modules — use getpass.getpass:\n  "
        + "\n  ".join(violations)
    )
