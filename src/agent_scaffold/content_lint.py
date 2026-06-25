"""Consumer-side content lint for a resolved agent-deployments source.

This is the consumer mirror of the producer-side validator in agent-deployments'
``scripts/generate_catalog.py`` (``validate_recipe_references`` +
``report_content_warnings``). It runs the same drift rules against any resolved
deployments tree — including a forked or custom one — so the contract
"valid == lint passes" holds on both sides.

The rules are deliberately frontmatter-driven (not built on the typed
:mod:`discovery` / :mod:`capabilities` loaders) for two reasons:

1. The typed loaders are *lossy by design* — :class:`capabilities.Capability`
   types ``kind`` as a ``Literal``, so a capability with an out-of-set ``kind``
   is silently dropped rather than surfaced. A lint must *catch* that, so it
   parses raw frontmatter exactly like the producer does.
2. Several rules read fields the typed models don't carry (``runtime_modes``
   descriptions, ``docker.ports``), so frontmatter is the only faithful source.

What *is* shared — and pinned by ``tests/test_content_lint.py`` so producer and
consumer can't drift — are the canonical **constants**: the capability-kind set
(:data:`capabilities._KNOWN_KINDS`), the topology set (:class:`topology.Topology`),
and the lint-local :data:`ENTRY_POINT_BASENAMES` / :data:`ADVERTISED_PROVIDERS`
(mirrors of the same names in the deployments generator).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent_scaffold.capabilities import _KNOWN_KINDS
from agent_scaffold.discovery import _NON_RECIPE_STEMS, _parse_frontmatter
from agent_scaffold.topology import Topology

# --- Shared canonical constants (mirror agent-deployments) -------------------

#: Capability ``kind`` values the consumer recognizes. Identical to the
#: producer's ``VALID_CAPABILITY_KINDS``; pinned by the parity test.
VALID_CAPABILITY_KINDS: frozenset[str] = _KNOWN_KINDS

#: Canonical recipe ``topology`` values, derived from the shared enum.
VALID_TOPOLOGIES: frozenset[str] = frozenset(t.value for t in Topology)

#: Recognized backend entry-point basenames. Mirrors the producer constant and
#: the launch heuristics in ``steps/launch_backend.py``: a recipe that ships
#: application source must name one of these in ``required_files`` or run has
#: nothing the generation contract guaranteed.
ENTRY_POINT_BASENAMES: frozenset[str] = frozenset(
    {
        "main.py",
        "app.py",
        "server.py",
        "__main__.py",
        "index.ts",
        "index.js",
        "main.ts",
        "server.ts",
        "app.ts",
    }
)

#: Providers that, when named in a ``runtime_modes`` mode description, should be
#: backed by a matching capability id (substring) AND a recipe dependency
#: (substring). Keyed by the lowercase token to scan for. Mirrors the producer
#: constant. Used by the advisory advertisement-coherence check (warns only).
ADVERTISED_PROVIDERS: dict[str, tuple[str, str]] = {
    "qdrant": ("qdrant", "qdrant"),
    "chroma": ("chroma", "chroma"),
    "pgvector": ("pgvector", "pgvector"),
    "openai": ("embedding.openai", "openai"),
    "cohere": ("rerank.cohere", "cohere"),
    "zep": ("memory_store.zep", "zep"),
}

Severity = str  # "error" | "warn"


class ContentLintError(Exception):
    """Raised when the deployments source can't be linted at all (e.g. the
    ``docs/recipes/`` directory is missing)."""


@dataclass(frozen=True)
class Finding:
    """One lint result. ``error`` findings fail the lint; ``warn`` are advisory
    (coverage gaps, stale advertisements) and never fail it."""

    severity: Severity
    rule: str
    location: str
    message: str

    def format(self) -> str:
        tag = "error" if self.severity == "error" else "warn "
        return f"{tag} [{self.rule}] {self.location}: {self.message}"


# --- Helpers (mirror the producer) -------------------------------------------


def _host_port(binding: Any) -> str | None:
    """Host side of a ``"HOST:CONTAINER"`` docker port binding, or None for the
    container-only form."""
    if not isinstance(binding, str) or ":" not in binding:
        return None
    return binding.split(":", 1)[0].strip() or None


def _resolve_capability_stack(
    declared: list[str], cap_requires: dict[str, list[str]]
) -> list[str]:
    """Expand declared capability ids to include transitive ``requires`` deps —
    the full service set ``docker compose up`` brings online."""
    seen: set[str] = set()
    stack: list[str] = []
    queue = list(declared)
    while queue:
        cid = queue.pop()
        if cid in seen:
            continue
        seen.add(cid)
        stack.append(cid)
        for dep in cap_requires.get(cid, []):
            if dep not in seen:
                queue.append(dep)
    return stack


def _iter_frontmatter(
    directory: Path, deployments_path: Path, *, recursive: bool
) -> list[dict[str, Any]]:
    """Parse every ``*.md`` under ``directory`` to a frontmatter dict, skipping
    dotfiles + non-recipe stems. Each dict gets a repo-relative ``path`` and an
    absolute ``_abs`` Path (used for on-disk load_list resolution)."""
    out: list[dict[str, Any]] = []
    paths = directory.rglob("*.md") if recursive else directory.glob("*.md")
    for p in sorted(paths):
        if p.name.startswith(".") or p.stem.lower() in _NON_RECIPE_STEMS:
            continue
        if not p.is_file():
            continue
        fm, _ = _parse_frontmatter(p.read_text(encoding="utf-8"))
        if not fm:
            continue
        entry = dict(fm)
        entry["path"] = p.relative_to(deployments_path).as_posix()
        entry["_abs"] = p
        out.append(entry)
    return out


def _load_recipes(deployments_path: Path) -> list[dict[str, Any]]:
    return _iter_frontmatter(
        deployments_path / "docs" / "recipes", deployments_path, recursive=False
    )


def _load_capabilities(deployments_path: Path) -> list[dict[str, Any]]:
    caps_dir = deployments_path / "docs" / "capabilities"
    if not caps_dir.is_dir():
        return []
    # Mirror the producer's collect_capabilities: a capability file declares
    # both id + kind. Files missing either aren't capabilities.
    return [
        c
        for c in _iter_frontmatter(caps_dir, deployments_path, recursive=True)
        if "id" in c and "kind" in c
    ]


def _load_blueprint_patterns(deployments_path: Path) -> set[str] | None:
    """Blueprint pattern ids from the committed ``catalog.yaml`` (the consumer's
    view of the blueprints cohort). Returns None when no catalog is present, so
    the agent_pattern / orphan checks degrade to skipped rather than wrong."""
    catalog_path = deployments_path / "catalog.yaml"
    if not catalog_path.is_file():
        return None
    try:
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    return {e["id"] for e in (data.get("patterns") or []) if isinstance(e, dict) and "id" in e}


# --- Rules -------------------------------------------------------------------


def _lint_capabilities(capabilities: list[dict[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for c in capabilities:
        loc = c.get("path", "<unknown>")
        kind = c.get("kind")
        if kind is not None and kind not in VALID_CAPABILITY_KINDS:
            findings.append(
                Finding(
                    "error",
                    "capability-kind",
                    loc,
                    f"kind={kind!r} is not one of {sorted(VALID_CAPABILITY_KINDS)}",
                )
            )
        card = c.get("card")
        if not isinstance(card, dict):
            findings.append(
                Finding("error", "capability-card", loc, "missing required 'card' mapping")
            )
        else:
            for key in ("name", "description"):
                if not card.get(key):
                    findings.append(
                        Finding(
                            "error",
                            "capability-card",
                            loc,
                            f"card.{key} must be a non-empty string",
                        )
                    )
    return findings


def _lint_recipes(
    recipes: list[dict[str, Any]],
    cap_ids: set[str],
    cap_ports: dict[str, list[str]],
    cap_requires: dict[str, list[str]],
    pattern_ids: set[str] | None,
) -> list[Finding]:
    findings: list[Finding] = []
    for r in recipes:
        loc = r.get("path", "<unknown>")

        # Topology enum membership.
        topology = r.get("topology")
        if topology is not None and topology not in VALID_TOPOLOGIES:
            findings.append(
                Finding(
                    "error",
                    "topology",
                    loc,
                    f"topology={topology!r} is not one of {sorted(VALID_TOPOLOGIES)}",
                )
            )

        # Capability ids resolve to a discovered capability file.
        for cid in r.get("capabilities") or []:
            if cid not in cap_ids:
                findings.append(
                    Finding(
                        "error",
                        "capability-ref",
                        loc,
                        f"capabilities[] {cid!r} has no docs/capabilities/ entry",
                    )
                )

        # agent_pattern resolves to a blueprint cohort id (only when a catalog
        # is available to resolve against).
        ap = r.get("agent_pattern")
        if ap and pattern_ids is not None and ap not in pattern_ids:
            findings.append(
                Finding(
                    "error",
                    "agent-pattern",
                    loc,
                    f"agent_pattern={ap!r} resolves to no blueprint pattern",
                )
            )

        # required_files names a recognized entry point.
        required_files = r.get("required_files") or []
        if isinstance(required_files, list) and required_files:
            has_entry = any(
                isinstance(f, str) and f.rsplit("/", 1)[-1] in ENTRY_POINT_BASENAMES
                for f in required_files
            )
            if not has_entry:
                findings.append(
                    Finding(
                        "error",
                        "required-files-entry",
                        loc,
                        "required_files names no recognized entry point "
                        f"(one of {sorted(ENTRY_POINT_BASENAMES)}) — run cannot launch it",
                    )
                )

        # Host-port collisions across the resolved capability stack.
        declared = [c for c in (r.get("capabilities") or []) if c in cap_ids]
        stack = _resolve_capability_stack(declared, cap_requires)
        app_port = str((r.get("env_overrides") or {}).get("APP_PORT", "8000"))
        host_ports: dict[str, list[str]] = {app_port: ["app"]}
        for cid in stack:
            for binding in cap_ports.get(cid, []):
                hp = _host_port(binding)
                if hp is not None:
                    host_ports.setdefault(hp, []).append(cid)
        for hp, owners in sorted(host_ports.items()):
            if len(owners) > 1:
                findings.append(
                    Finding(
                        "error",
                        "port-collision",
                        loc,
                        f"host port {hp} is bound by multiple services in the "
                        f"resolved stack: {', '.join(sorted(owners))}",
                    )
                )

        # load_list links resolve on disk (fail closed, like the producer).
        recipe_dir = r["_abs"].parent if isinstance(r.get("_abs"), Path) else None
        for i, item in enumerate(r.get("load_list") or []):
            if not isinstance(item, dict):
                continue
            rel = item.get("path")
            if recipe_dir is not None and isinstance(rel, str) and rel:
                if not (recipe_dir / rel).resolve().exists():
                    findings.append(
                        Finding(
                            "error",
                            "load-list-link",
                            loc,
                            f"load_list[{i}].path {rel!r} does not resolve to a file on disk",
                        )
                    )
    return findings


def _lint_advisories(
    recipes: list[dict[str, Any]],
    frameworks: list[dict[str, Any]],
    pattern_ids: set[str] | None,
) -> list[Finding]:
    findings: list[Finding] = []

    # Advertisement coherence — a provider named in a runtime_modes description
    # should be backed by a capability + dependency.
    for r in recipes:
        loc = r.get("path", "<unknown>")
        caps = r.get("capabilities") or []
        deps = r.get("recipe_dependencies") or {}
        dep_names = [
            str(name).lower()
            for lang in deps.values()
            if isinstance(lang, dict)
            for name in lang
        ]
        rmodes = r.get("runtime_modes") or {}
        desc = " ".join(
            str(m.get("description", "")) for m in rmodes.values() if isinstance(m, dict)
        ).lower()
        for token, (cap_sub, dep_sub) in sorted(ADVERTISED_PROVIDERS.items()):
            if token not in desc:
                continue
            cap_backed = any(cap_sub in c for c in caps)
            dep_backed = any(dep_sub in d for d in dep_names)
            if not (cap_backed or dep_backed):
                findings.append(
                    Finding(
                        "warn",
                        "advertisement",
                        loc,
                        f"runtime_modes advertises {token!r} but no capability or "
                        f"dependency backs it",
                    )
                )

    # Orphan blueprint patterns — a pattern no recipe selects (only when a
    # catalog is available).
    if pattern_ids is not None:
        used = {r.get("agent_pattern") for r in recipes if r.get("agent_pattern")}
        for orphan in sorted(pattern_ids - used):
            findings.append(
                Finding(
                    "warn",
                    "orphan-pattern",
                    "catalog",
                    f"blueprint pattern {orphan!r} has no recipe (coverage gap)",
                )
            )

    # Orphan frameworks — a framework doc no recipe references in its load_list.
    referenced: set[str] = set()
    for r in recipes:
        for item in r.get("load_list") or []:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                rel = item["path"]
                if "/frameworks/" in rel:
                    referenced.add(rel.rsplit("/", 1)[-1])
    for fw in frameworks:
        basename = fw["_abs"].name if isinstance(fw.get("_abs"), Path) else None
        if basename and basename not in referenced:
            findings.append(
                Finding(
                    "warn",
                    "orphan-framework",
                    fw.get("path", basename),
                    f"framework {fw.get('id', basename)!r} is referenced by no recipe "
                    "load_list (coverage gap)",
                )
            )
    return findings


def lint_content(deployments_path: Path) -> list[Finding]:
    """Run every content-drift rule against a resolved deployments tree.

    Returns all findings (errors + advisory warnings). Raises
    :class:`ContentLintError` only when the tree has no ``docs/recipes/`` to
    lint at all.
    """
    recipes_dir = deployments_path / "docs" / "recipes"
    if not recipes_dir.is_dir():
        raise ContentLintError(
            f"no docs/recipes/ under {deployments_path} — is this an "
            "agent-deployments source?"
        )

    recipes = _load_recipes(deployments_path)
    capabilities = _load_capabilities(deployments_path)
    frameworks = _iter_frontmatter(
        deployments_path / "docs" / "frameworks", deployments_path, recursive=False
    ) if (deployments_path / "docs" / "frameworks").is_dir() else []
    pattern_ids = _load_blueprint_patterns(deployments_path)

    cap_ids = {c["id"] for c in capabilities if "id" in c}
    cap_ports = {
        c["id"]: list((c.get("docker") or {}).get("ports") or [])
        for c in capabilities
        if "id" in c
    }
    cap_requires = {c["id"]: list(c.get("requires") or []) for c in capabilities if "id" in c}

    findings: list[Finding] = []
    findings += _lint_capabilities(capabilities)
    findings += _lint_recipes(recipes, cap_ids, cap_ports, cap_requires, pattern_ids)
    findings += _lint_advisories(recipes, frameworks, pattern_ids)
    return findings


def errors(findings: list[Finding]) -> list[Finding]:
    """The error-severity subset — the findings that make the lint fail."""
    return [f for f in findings if f.severity == "error"]


def summarize(findings: list[Finding]) -> str:
    """One-line tally, e.g. ``"3 errors, 7 warnings"``."""
    n_err = sum(1 for f in findings if f.severity == "error")
    n_warn = len(findings) - n_err

    def _plural(n: int, word: str) -> str:
        return f"{n} {word}{'' if n == 1 else 's'}"

    return f"{_plural(n_err, 'error')}, {_plural(n_warn, 'warning')}"
