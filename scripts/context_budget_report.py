#!/usr/bin/env python3
"""Context-window budget report — per-recipe assembled-context token totals.

A repeatable measurement harness for the context-window-reduction work. It
mirrors the CLI's assemble path (catalog load + capability resolution +
``context.assemble``) against a local ``agent-deployments`` checkout and prints,
per recipe: the total assembled-context token estimate, the per-tier breakdown,
and the recipe-frontmatter token cost (the "drop frontmatter" lever). Re-run
before and after a change to quantify savings.

To see the *true* assembled size (not the budget-capped one) the harness sets a
very high ``max_context_tokens`` so nothing is dropped; per-doc truncation stays
at its real default.

Usage::

    uv run scripts/context_budget_report.py
    uv run scripts/context_budget_report.py --deployments ~/Desktop/agent-deployments
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from agent_scaffold.capabilities import load_capabilities, resolve
from agent_scaffold.catalog import load_catalog
from agent_scaffold.context import _estimate_tokens, _strip_frontmatter, assemble
from agent_scaffold.discovery import discover_recipes
from agent_scaffold.framework_versions import available_frameworks_for_language

# High enough that nothing is dropped — we want the true assembled size.
_NO_DROP_CAP = 1_000_000


def _pick_framework(deployments: Path, language: str) -> str:
    """Deterministic, representative framework for the recipe's language."""
    fws = available_frameworks_for_language(deployments, language)
    return fws[0] if fws else "none"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-recipe context-budget report.")
    ap.add_argument(
        "--deployments",
        type=Path,
        default=Path.home() / "Desktop" / "agent-deployments",
        help="Path to the agent-deployments checkout.",
    )
    ap.add_argument(
        "--blueprints",
        type=Path,
        default=Path.home() / "Desktop" / "agent-blueprints",
        help="Path to the agent-blueprints checkout (for blueprint-URL resolution).",
    )
    args = ap.parse_args()

    deployments: Path = args.deployments.expanduser().resolve()
    blueprints: Path = args.blueprints.expanduser().resolve()
    blueprints_path = blueprints if blueprints.is_dir() else None

    catalog_path = deployments / "catalog.yaml"
    with tempfile.TemporaryDirectory() as cache:
        catalog = load_catalog(url=str(catalog_path), cache_dir=Path(cache))

    cap_registry = load_capabilities(deployments)
    recipes = sorted(discover_recipes(deployments), key=lambda r: r.slug)

    print(
        f"{'recipe':<30}{'lang':<7}{'framework':<14}"
        f"{'tokens':>9}{'docs':>6}{'caps':>6}{'fm_tok':>8}"
    )
    print("-" * 80)

    total = total_fm = total_docs = counted = 0
    for recipe in recipes:
        language = recipe.languages[0] if recipe.languages else "python"
        framework = _pick_framework(deployments, language)
        stack = resolve(
            recipe,
            cap_registry,
            default_frontend=True,
            default_key_bootstrap=(language == "python"),
        )
        try:
            ctx = assemble(
                recipe,
                language,
                framework,
                deployments,
                catalog=catalog,
                blueprints_path=blueprints_path,
                resolved_stack=stack if stack.capabilities else None,
                max_context_tokens=_NO_DROP_CAP,
            )
        except Exception as exc:  # noqa: BLE001 — record and keep going
            print(f"{recipe.slug:<30}{language:<7}{framework:<14}{'ERR':>9}  {exc!s:.40}")
            continue

        recipe_text = recipe.path.read_text(encoding="utf-8")
        fm_tokens = _estimate_tokens(recipe_text) - _estimate_tokens(
            _strip_frontmatter(recipe_text)
        )
        n_docs = sum(t.docs for t in ctx.summary.tiers) if ctx.summary else 0
        n_caps = len(stack.capabilities)
        total += ctx.token_estimate
        total_fm += fm_tokens
        total_docs += n_docs
        counted += 1
        print(
            f"{recipe.slug:<30}{language:<7}{framework:<14}"
            f"{ctx.token_estimate:>9,}{n_docs:>6}{n_caps:>6}{fm_tokens:>8,}"
        )

    print("-" * 80)
    if counted:
        print(
            f"{'TOTAL (' + str(counted) + ' recipes)':<51}"
            f"{total:>9,}{total_docs:>6}{'':>6}{total_fm:>8,}"
        )
        print(
            f"{'MEAN':<51}{total // counted:>9,}"
            f"{total_docs // counted:>6}{'':>6}{total_fm // counted:>8,}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
