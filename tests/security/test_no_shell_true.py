"""Rule 4 audit: ``subprocess`` invocations must be ``shell=False`` + list-form args.

A single ``shell=True`` opens the door to command injection if any element
of the command string was derived from user input. A single string-form
argument (``subprocess.run("git status")``) makes the shell route the only
correct interpretation, which is the same trap with a different shape.

This test parses every ``.py`` file under ``src/agent_scaffold/`` and walks
the AST looking for the two patterns. The intent is a CI-blocking regression
net: if a future PR slips in ``shell=True`` or a string-form first arg, the
suite fails before merge.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "agent_scaffold"

# Subprocess functions whose first positional is the command to run.
_SUBPROCESS_FUNCS = frozenset({"run", "Popen", "call", "check_call", "check_output"})

# Allow-list of ``(relative_path, function_name)`` exempted from the shell=True
# rule. Each entry MUST be justified. Empty since the smoke tier moved to
# argv execution (``validator._smoke_argv`` gates the model-authored string
# and ``_run`` executes it with ``shell=False``) — there is no sanctioned
# shell-string execution left anywhere in ``src/``.
_SHELL_TRUE_EXEMPT: frozenset[tuple[str, str]] = frozenset()


def _shell_kwarg_is_true(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
            return True
    return False


def _first_arg_is_string_literal(call: ast.Call) -> bool:
    if not call.args:
        return False
    first = call.args[0]
    return isinstance(first, ast.Constant) and isinstance(first.value, str)


def _is_subprocess_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr in _SUBPROCESS_FUNCS:
        # subprocess.run / subprocess.Popen / ...
        if isinstance(fn.value, ast.Name) and fn.value.id == "subprocess":
            return True
    return False


def _enclosing_function(tree: ast.Module, lineno: int) -> str | None:
    """Walk the tree once to find which top-level function contains ``lineno``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(node, "end_lineno", node.lineno)
            if node.lineno <= lineno <= end:
                return node.name
    return None


def test_no_shell_true_in_src() -> None:
    violations: list[str] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (_is_subprocess_call(node) and _shell_kwarg_is_true(node)):
                continue
            rel = py_file.relative_to(SRC_ROOT)
            func = _enclosing_function(tree, node.lineno)
            if (rel.as_posix().split("/")[-1], func) in _SHELL_TRUE_EXEMPT:
                continue
            violations.append(f"{rel}:{node.lineno}")
    assert not violations, "shell=True is forbidden:\n  " + "\n  ".join(violations)


def test_no_string_form_subprocess_args() -> None:
    """First positional to subprocess.* must never be a string literal."""
    violations: list[str] = []
    for py_file in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if _is_subprocess_call(node) and _first_arg_is_string_literal(node):
                violations.append(f"{py_file.relative_to(SRC_ROOT)}:{node.lineno}")
    assert not violations, "string-form subprocess args forbidden:\n  " + "\n  ".join(violations)
