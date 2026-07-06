"""Tier preset → resolve → emit: the deterministic substrate a tier lands.

These tests exercise the exact seam the CLI uses to turn a chosen tier into
on-disk files — ``expand_tier`` → ``tier_seed_ids`` → ``resolve`` →
``copy_capability_templates`` (cli.py) — with no LLM in the loop. They guard the
half of the value path that is fully deterministic: selecting a tier must emit
its ``core.*`` substrate, and the tiers must stack (``T1 ⊇ T0``) at the file
level.

The ``core.*`` capabilities are built as a small, hermetic fixture tree that
mirrors the real deployments emit contract (same ids, same ``emit_files``
source/dest → ``agent/prompts/`` · ``agent/io/`` · ``agent/tools/``). The
*correctness* of the real template payloads is guarded by the deployments
repo's own self-tests; here we prove only the wiring — that a chosen tier pulls
the right capabilities and their files actually land where the contract says.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import ResolvedStack, load_capabilities, resolve
from agent_scaffold.capability_emit import EmitResult, copy_capability_templates
from agent_scaffold.discovery import Recipe
from agent_scaffold.tiers import default_presets, expand_tier, tier_seed_ids

# Each core capability, as (path under docs/capabilities/, emit source glob,
# emit dest, {template filename: content}). The frontmatter is intentionally
# minimal — id / kind / emit_files are the only fields the loader requires —
# so the fixture stays faithful to the real emit contract without dragging in
# the full card/provides/probe surface.
_SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly."

_CORE_CAPABILITIES: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "core/prompts.md",
        "templates/prompts/**",
        "agent/prompts/",
        {
            "loader.py": (
                '"""Load editable prompt files (fixture mirror of core.prompts)."""\n\n'
                "from pathlib import Path\n\n\n"
                "def load_prompt(name: str) -> str:\n"
                '    return (Path(__file__).parent / f"{name}.txt").read_text(\n'
                '        encoding="utf-8"\n'
                "    ).strip()\n"
            ),
            "system.txt": f"{_SYSTEM_PROMPT}\n",
            "__init__.py": 'from .loader import load_prompt\n\n__all__ = ["load_prompt"]\n',
        },
    ),
    (
        "core/io_schema.md",
        "templates/io_schema/**",
        "agent/io/",
        {
            "schemas.py": (
                '"""Chat I/O schemas (fixture mirror of core.io_schema)."""\n\n'
                "from pydantic import BaseModel\n\n\n"
                "class ChatRequest(BaseModel):\n    message: str\n\n\n"
                "class ChatResponse(BaseModel):\n    reply: str\n"
            ),
            "__init__.py": (
                "from .schemas import ChatRequest, ChatResponse\n\n"
                '__all__ = ["ChatRequest", "ChatResponse"]\n'
            ),
        },
    ),
    (
        "core/tool_registry.md",
        "templates/tool_registry/**",
        "agent/tools/",
        {
            "registry.py": (
                '"""Typed tool registry (fixture mirror of core.tool_registry)."""\n\n\n'
                "class ToolRegistry:\n"
                "    def __init__(self) -> None:\n"
                "        self._tools: dict[str, object] = {}\n"
            ),
            "__init__.py": 'from .registry import ToolRegistry\n\n__all__ = ["ToolRegistry"]\n',
        },
    ),
)


def _write_core_capabilities(deployments_root: Path) -> Path:
    """Build a hermetic deployments tree carrying the T0/T1 ``core.*`` caps.

    Returns the deployments root (the dir ``load_capabilities`` is called with).
    ``core.spec`` is deliberately absent — it is emitted by the pipeline as
    ``.agent/spec.md``, not by an ``emit_files`` template, so it is not a
    catalog capability here (matching the real deployments catalog).
    """
    caps_dir = deployments_root / "docs" / "capabilities"
    for doc_rel, source, dest, templates in _CORE_CAPABILITIES:
        doc_path = caps_dir / doc_rel
        cap_id = doc_rel.removesuffix(".md").replace("/", ".")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(
            "---\n"
            f"id: {cap_id}\n"
            "kind: core\n"
            "emit_files:\n"
            f"  - source: {source}\n"
            f"    dest: {dest}\n"
            "---\n\n"
            f"# {cap_id} (fixture)\n",
            encoding="utf-8",
        )
        template_root = doc_path.parent / source.removesuffix("/**")
        template_root.mkdir(parents=True, exist_ok=True)
        for name, content in templates.items():
            (template_root / name).write_text(content, encoding="utf-8")
    return deployments_root


def _emit_tier(
    tier: str, deployments_root: Path, project_dir: Path
) -> tuple[ResolvedStack, EmitResult]:
    """Run the CLI's tier→emit seam for ``tier`` into ``project_dir``."""
    catalog = load_capabilities(deployments_root)
    seeds = tier_seed_ids(expand_tier(tier, default_presets()))
    recipe = Recipe(slug="chat", title="Chat", path=deployments_root / "recipe.md")
    stack = resolve(recipe, catalog, add_capabilities=seeds)
    project_dir.mkdir(parents=True, exist_ok=True)
    result = copy_capability_templates(
        stack=stack,
        capabilities_root=deployments_root / "docs" / "capabilities",
        project_dir=project_dir,
    )
    return stack, result


def _emitted_rel(result: EmitResult, project_dir: Path) -> set[str]:
    root = project_dir.resolve()
    return {p.relative_to(root).as_posix() for p in result.written}


def test_t0_tier_emits_prompts_and_io_substrate(tmp_path: Path) -> None:
    deployments = _write_core_capabilities(tmp_path / "deployments")
    project = tmp_path / "project"
    stack, result = _emit_tier("T0", deployments, project)

    # T0's emit-bearing capabilities resolved (not inert).
    resolved_ids = {cap.id for cap in stack.capabilities}
    assert {"core.prompts", "core.io_schema"} <= resolved_ids

    # The owned-prompts and schema-I/O substrate landed at the contract paths.
    assert (project / "agent" / "prompts" / "loader.py").is_file()
    assert (project / "agent" / "prompts" / "__init__.py").is_file()
    assert (project / "agent" / "io" / "schemas.py").is_file()

    # Content round-trips byte-for-byte through the atomic copier.
    assert (project / "agent" / "prompts" / "system.txt").read_text() == f"{_SYSTEM_PROMPT}\n"

    # The tool registry is a T1 capability — it must NOT appear at T0.
    assert not (project / "agent" / "tools").exists()
    assert "core.tool_registry" not in resolved_ids


def test_core_spec_stays_pipeline_emitted_not_catalog(tmp_path: Path) -> None:
    # core.spec is seeded by T0 but is emitted by the pipeline as .agent/spec.md,
    # not via a catalog emit_files template. It therefore resolves inertly — and
    # that inertness must never block the emit-bearing caps around it.
    deployments = _write_core_capabilities(tmp_path / "deployments")
    stack, result = _emit_tier("T0", deployments, tmp_path / "project")

    assert "core.spec" in stack.unresolved
    assert "core.spec" not in {cap.id for cap in stack.capabilities}
    # The real substrate still emitted despite core.spec being unresolved.
    assert any(p.name == "loader.py" for p in result.written)


def test_t1_tier_adds_tool_registry_over_t0(tmp_path: Path) -> None:
    deployments = _write_core_capabilities(tmp_path / "deployments")
    project = tmp_path / "project"
    stack, _ = _emit_tier("T1", deployments, project)

    resolved_ids = {cap.id for cap in stack.capabilities}
    assert {"core.prompts", "core.io_schema", "core.tool_registry"} <= resolved_ids

    # Everything T0 lands, plus the typed tool registry.
    assert (project / "agent" / "prompts" / "loader.py").is_file()
    assert (project / "agent" / "io" / "schemas.py").is_file()
    assert (project / "agent" / "tools" / "registry.py").is_file()
    assert (project / "agent" / "tools" / "__init__.py").is_file()


def test_t1_emit_is_superset_of_t0(tmp_path: Path) -> None:
    # The T1 ⊇ T0 tier invariant, asserted at the level that matters to a user:
    # the set of files a generated project actually receives.
    deployments = _write_core_capabilities(tmp_path / "deployments")
    p0, p1 = tmp_path / "p0", tmp_path / "p1"
    _, r0 = _emit_tier("T0", deployments, p0)
    _, r1 = _emit_tier("T1", deployments, p1)

    files0 = _emitted_rel(r0, p0)
    files1 = _emitted_rel(r1, p1)

    assert files0, "T0 should emit a non-empty substrate"
    assert files0 <= files1, "T1 must emit every file T0 does"
    added = files1 - files0
    assert added, "T1 must add files over T0"
    assert all(
        f.startswith("agent/tools/") for f in added
    ), f"T1's only additions over T0 should be the tool registry; got {sorted(added)}"
