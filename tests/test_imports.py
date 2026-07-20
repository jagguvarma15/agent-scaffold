"""Unit tests for agent_scaffold.imports neighbour discovery."""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.imports import discover_neighbours


def _scaffold_python_project(tmp_path: Path) -> Path:
    """Build a minimal src-layout project with cross-file imports."""
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "__init__.py").touch()
    (src / "main.py").write_text(
        "from demo.helpers import h\nfrom demo.unused import never_imported\n",
        encoding="utf-8",
    )
    (src / "helpers.py").write_text("def h() -> int:\n    return 1\n", encoding="utf-8")
    (src / "unused.py").write_text("def never_imported() -> None:\n    pass\n", encoding="utf-8")
    (src / "consumer.py").write_text(
        "from demo.main import something\nprint('uses main')\n", encoding="utf-8"
    )
    (src / "unrelated.py").write_text("answer = 42\n", encoding="utf-8")
    return tmp_path


def test_discover_neighbours_python_includes_imported_and_importers(tmp_path: Path) -> None:
    project = _scaffold_python_project(tmp_path)
    rels = {str(p.relative_to(project)) for p in discover_neighbours(project, "src/demo/main.py")}
    # helpers + unused are imported by main; consumer imports main. unrelated.py
    # should NOT appear (no cross-reference either direction).
    assert "src/demo/helpers.py" in rels
    assert "src/demo/unused.py" in rels
    assert "src/demo/consumer.py" in rels
    assert "src/demo/unrelated.py" not in rels


def test_discover_neighbours_returns_empty_for_unknown_path(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    assert discover_neighbours(tmp_path, "nope.py") == []


def test_discover_neighbours_caps_results(tmp_path: Path) -> None:
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "target.py").write_text("x = 1\n", encoding="utf-8")
    # Generate many importers; the discovery should cap.
    for i in range(20):
        (src / f"caller_{i}.py").write_text("from demo.target import x\n", encoding="utf-8")
    neighbours = discover_neighbours(tmp_path, "src/demo/target.py", max_neighbours=5)
    assert len(neighbours) == 5


def test_discover_neighbours_skips_target_itself(tmp_path: Path) -> None:
    src = tmp_path / "src" / "demo"
    src.mkdir(parents=True)
    (src / "self_ref.py").write_text("# from demo.self_ref import foo\n", encoding="utf-8")
    out = discover_neighbours(tmp_path, "src/demo/self_ref.py")
    assert all(p.name != "self_ref.py" for p in out)


def test_discover_neighbours_typescript_uses_regex(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.ts").write_text("import { x } from './helpers';\n", encoding="utf-8")
    (src / "helpers.ts").write_text("export const x = 1;\n", encoding="utf-8")
    (src / "unused.ts").write_text("export const y = 2;\n", encoding="utf-8")
    out = {p.name for p in discover_neighbours(tmp_path, "src/main.ts")}
    assert "helpers.ts" in out
