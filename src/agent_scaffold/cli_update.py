"""``agent-scaffold update`` flow extracted from ``cli.py``.

Copier-style template evolution: re-runs the recipe against an existing
generated project, three-way-merges template changes against the user's
edits, leaves conflict markers where hunks overlap, and finalises with
``agent-scaffold update --continue`` after resolution.

The ``@app.command`` decoration stays in ``cli.py`` so Typer discovers
``update`` at module load; the command body lives here as :func:`run`.
Private helpers (``_classify_update``, ``_render_update_plan``,
``_apply_update``, ``_continue_update``, ``_finalise_update``,
``_regenerate_for_update``, ``_decide_removals``, the in-progress JSON
helpers) are this module's private surface — tests import them by name
from here, not from ``cli``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from agent_scaffold._scaffold_dir import SCAFFOLD_DIR
from agent_scaffold.capabilities import load_capabilities
from agent_scaffold.capabilities import resolve as resolve_capabilities
from agent_scaffold.cli_interactive import _interactive_select
from agent_scaffold.cli_shared import console, prompt_to_raise_context_cap
from agent_scaffold.config import Config, ConfigError, load_config
from agent_scaffold.context import AssembledContext, ContextBudgetError, assemble
from agent_scaffold.contract import ContractParseError, parse
from agent_scaffold.discovery import DiscoveryError, discover_recipes
from agent_scaffold.generator import GenerationRequest, generate
from agent_scaffold.manifest import (
    Manifest,
    ManifestNotFoundError,
    UpdateEntry,
    build_file_entries,
    read_manifest,
    write_manifest,
)
from agent_scaffold.merge import has_unresolved_markers, three_way_merge
from agent_scaffold.sources import SourceConfigError, SourceFetchError, resolve_deployments
from agent_scaffold.template_snapshot import (
    cleanup_tempdir,
    compute_template_sha,
    has_snapshot,
    load_generation_snapshot,
    prune_snapshots,
    save_generation_snapshot,
)

UPDATE_IN_PROGRESS_FILENAME = "update.in-progress.json"


# ── classification ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _UpdateClassification:
    added: list[str]
    modified: list[str]
    conflicted: list[str]
    removed: list[str]
    binary_skipped: list[str]
    merge_results: dict[str, object]  # path -> MergeResult; object to avoid forward-ref


# ── in-progress JSON helpers ────────────────────────────────────────────────


def _in_progress_path(project_dir: Path) -> Path:
    return project_dir / SCAFFOLD_DIR / UPDATE_IN_PROGRESS_FILENAME


def _read_in_progress(project_dir: Path) -> dict[str, Any] | None:
    path = _in_progress_path(project_dir)
    if not path.is_file():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _write_in_progress(project_dir: Path, payload: dict[str, Any]) -> None:
    path = _in_progress_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _clear_in_progress(project_dir: Path) -> None:
    _in_progress_path(project_dir).unlink(missing_ok=True)


# ── classify / render / apply ───────────────────────────────────────────────


def _iter_base_files(base_dir: Path) -> list[str]:
    """Return file paths from a previously-extracted snapshot, relative to base_dir."""
    if not base_dir.is_dir():
        return []
    out: list[str] = []
    for path in base_dir.rglob("*"):
        if path.is_file():
            out.append(path.relative_to(base_dir).as_posix())
    return out


def _classify_update(
    base_dir: Path,
    project_dir: Path,
    fresh_files: dict[str, str],
    *,
    previous_files: list[str],
) -> _UpdateClassification:
    """Compute the 3-way classification per file.

    ``previous_files`` is the manifest's list of paths from the prior gen;
    used to identify removals (files that *were* templated but aren't now).
    """
    added: list[str] = []
    modified: list[str] = []
    conflicted: list[str] = []
    binary_skipped: list[str] = []
    merge_results: dict[str, Any] = {}

    previous_set = set(previous_files)
    base_lookup = {rel: (base_dir / rel) for rel in _iter_base_files(base_dir)}

    for rel, fresh_text in fresh_files.items():
        on_disk = project_dir / rel
        base_path = base_lookup.get(rel)
        if not on_disk.exists():
            added.append(rel)
            continue
        ours = on_disk.read_bytes()
        theirs = fresh_text.encode("utf-8")
        base_bytes = base_path.read_bytes() if base_path and base_path.is_file() else theirs
        merge = three_way_merge(base_bytes, ours, theirs)
        merge_results[rel] = merge
        if merge.binary:
            binary_skipped.append(rel)
            continue
        if ours.decode("utf-8", errors="replace") == merge.text:
            continue  # nothing actually changes — skip silently
        if merge.conflicted:
            conflicted.append(rel)
        else:
            modified.append(rel)

    removed = sorted(
        rel for rel in previous_set if rel not in fresh_files and (project_dir / rel).is_file()
    )
    return _UpdateClassification(
        added=sorted(added),
        modified=sorted(modified),
        conflicted=sorted(conflicted),
        removed=removed,
        binary_skipped=sorted(binary_skipped),
        merge_results=merge_results,
    )


def _render_update_plan(classification: _UpdateClassification) -> Panel:
    rows: list[str] = []
    rows.append(
        f"[green]Files added   ({len(classification.added)}):[/]  "
        + (", ".join(classification.added) or "[dim](none)[/]")
    )
    rows.append(
        f"[cyan]Files updated ({len(classification.modified)}):[/]  "
        + (", ".join(classification.modified) or "[dim](none)[/]")
    )
    if classification.conflicted:
        rows.append(
            f"[red]Conflicts     ({len(classification.conflicted)}):[/]  "
            + ", ".join(classification.conflicted)
            + "\n                     → conflict markers will be written"
        )
    else:
        rows.append("[red]Conflicts     (0):[/]")
    rows.append(
        f"[yellow]Files removed ({len(classification.removed)}):[/]  "
        + (", ".join(classification.removed) or "[dim](none)[/]")
    )
    if classification.binary_skipped:
        rows.append(
            f"[dim]Binary kept   ({len(classification.binary_skipped)}):[/] "
            + ", ".join(classification.binary_skipped)
        )
    return Panel("\n".join(rows), title="Update plan", expand=False)


def _apply_update(
    project_dir: Path,
    fresh_files: dict[str, str],
    classification: _UpdateClassification,
    *,
    remove_decisions: dict[str, bool],
) -> None:
    """Write the merged content + handle removals. No prompting here."""
    for rel in classification.added:
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fresh_files[rel], encoding="utf-8")
    for rel in classification.modified + classification.conflicted:
        merge = classification.merge_results[rel]
        target = project_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(merge.text, encoding="utf-8")  # type: ignore[attr-defined]
    for rel, remove in remove_decisions.items():
        if remove:
            (project_dir / rel).unlink(missing_ok=True)


# ── removals + finalise + regenerate ────────────────────────────────────────


def _decide_removals(removed: list[str], *, yes: bool) -> dict[str, bool]:
    """Per file, ask the user whether to remove it. ``--yes`` keeps everything.

    Default is **no**: removals are sticky — we'd rather over-keep than delete
    something the user values. Output dict maps path → ``True if remove``.
    """
    if not removed:
        return {}
    if yes:
        for rel in removed:
            console.print(
                f"[yellow]Warning:[/] {rel} is gone from the template; "
                "keeping (pass without --yes to confirm removal)."
            )
        return dict.fromkeys(removed, False)
    decisions: dict[str, bool] = {}
    import questionary

    for rel in removed:
        answer = questionary.confirm(
            f"File {rel} was in the template but is gone now. Remove from your project?",
            default=False,
        ).ask()
        decisions[rel] = bool(answer)
    return decisions


def _continue_update(project_dir: Path, manifest: Manifest) -> None:
    """``--continue`` path: verify markers cleared then finalise."""
    in_progress = _read_in_progress(project_dir)
    if in_progress is None:
        console.print("[red]No update in progress.[/] Run `agent-scaffold update` first.")
        raise typer.Exit(code=1)
    conflicted: list[str] = list(in_progress.get("conflicts", []))
    still_marked: list[tuple[str, int]] = []
    for rel in conflicted:
        target = project_dir / rel
        if not target.is_file():
            still_marked.append((rel, -1))
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        if has_unresolved_markers(text):
            from agent_scaffold.merge import count_unresolved_markers

            still_marked.append((rel, count_unresolved_markers(text)))
    if still_marked:
        console.print(
            "[red]Conflict markers still present in:[/]\n  "
            + "\n  ".join(f"{rel} ({n} marker line(s))" for rel, n in still_marked)
            + "\n\nResolve all `<<<<<<< / ======= / >>>>>>>` regions, then "
            "re-run `agent-scaffold update --continue`.\n"
            "If you want to abandon: "
            f"`rm {_in_progress_path(project_dir)} && git checkout .`"
        )
        raise typer.Exit(code=1)

    classification = _UpdateClassification(
        added=list(in_progress.get("files_added", [])),
        modified=list(in_progress.get("files_modified", [])),
        conflicted=conflicted,
        removed=list(in_progress.get("files_removed", [])),
        binary_skipped=[],
        merge_results={},
    )
    on_disk = {
        rel: (project_dir / rel).read_text(encoding="utf-8", errors="replace")
        for rel in {*classification.added, *classification.modified, *classification.conflicted}
        if (project_dir / rel).is_file()
    }
    _finalise_update(
        project_dir,
        manifest,
        str(in_progress["to_template_sha"]),
        classification,
        removed=classification.removed,
        generated_files=on_disk,
    )
    _clear_in_progress(project_dir)
    console.print("[green]Conflicts resolved. Update finalised.[/]")


def _finalise_update(
    project_dir: Path,
    manifest: Manifest,
    new_sha: str,
    classification: _UpdateClassification,
    *,
    removed: list[str],
    generated_files: dict[str, str],
) -> None:
    """Append the update history entry, save the new snapshot, prune LRU."""
    save_generation_snapshot(project_dir, new_sha, generated_files)
    prune_snapshots(project_dir)
    new_entry = UpdateEntry(
        timestamp=datetime.now(UTC).isoformat(),
        from_schema=manifest.schema_version,
        to_schema=manifest.schema_version,
        from_template_sha=manifest.template_snapshot_sha,
        to_template_sha=new_sha,
        model=manifest.model,
        files_added=classification.added,
        files_modified=classification.modified,
        files_removed=removed,
        files_conflicted=classification.conflicted,
    )
    updated_files: list[str] = list({f.path for f in manifest.files} | set(classification.added))
    for rel in removed:
        if rel in updated_files:
            updated_files.remove(rel)
    new_manifest = manifest.model_copy(
        update={
            "template_snapshot_sha": new_sha,
            "update_history": [*manifest.update_history, new_entry],
            "files": build_file_entries(project_dir, sorted(updated_files)),
        }
    )
    write_manifest(project_dir, new_manifest)


def _regenerate_for_update(
    manifest: Manifest, deployments: Path, cfg: Config
) -> dict[str, str] | None:
    """Re-run generation with the manifest's captured answers; return ``{rel: text}``.

    Returns ``None`` on a hard failure so the caller can keep the project
    untouched.
    """
    try:
        recipes = discover_recipes(deployments)
    except DiscoveryError as exc:
        console.print(f"[red]Discovery failed:[/] {exc}")
        return None
    recipe = next((r for r in recipes if r.slug == manifest.recipe), None)
    if recipe is None:
        console.print(f"[red]Recipe {manifest.recipe!r} not found in deployments.[/]")
        return None

    # _load_language_hints lives in cli.py; lazy import avoids a load-time
    # cycle (cli.py imports this module transitively through cmd_update).
    from agent_scaffold.cli import _load_language_hints

    language = manifest.language
    framework = manifest.framework
    hints = _load_language_hints(language)
    recipe_lang_deps = recipe.recipe_dependencies.get(language, {})
    if recipe_lang_deps:
        pinned = dict(hints.get("pinned_dependencies") or {})
        pinned.update(recipe_lang_deps)
        hints = {**hints, "pinned_dependencies": pinned}

    project_name = manifest.answers.get("project_name") or manifest.recipe
    catalog = load_capabilities(deployments)
    resolved_stack = resolve_capabilities(recipe, catalog)

    from agent_scaffold.catalog import load_catalog_for_config

    top_catalog = load_catalog_for_config(cfg)

    def _assemble_update(active_cfg: Config) -> AssembledContext:
        return assemble(
            recipe,
            language,
            framework,
            deployments,
            max_context_tokens=active_cfg.max_context_tokens,
            max_link_depth=active_cfg.max_link_depth,
            max_tokens_per_doc=active_cfg.max_tokens_per_doc,
            resolved_stack=resolved_stack if resolved_stack.capabilities else None,
            catalog=top_catalog,
        )

    try:
        assembled = _assemble_update(cfg)
    except ContextBudgetError as exc:
        bumped = prompt_to_raise_context_cap(console, exc)
        if bumped is None:
            return None
        new_cap, new_per_doc = bumped
        cfg = cfg.model_copy(
            update={"max_context_tokens": new_cap, "max_tokens_per_doc": new_per_doc}
        )
        try:
            assembled = _assemble_update(cfg)
        except ContextBudgetError as exc2:
            console.print(f"[red]Context assembly failed:[/] {exc2}")
            return None
    req = GenerationRequest(
        project_name=project_name,
        target_language=language,
        framework=framework,
        assembled_context=assembled,
        language_hints=hints,
        extra_required=list(recipe.required_files),
    )
    update_cfg = cfg.model_copy(update={"model": manifest.model})
    try:
        raw = generate(req, update_cfg)
    except Exception as exc:  # noqa: BLE001 — surface any model/network failure
        console.print(f"[red]Generation failed:[/] {exc}")
        return None
    try:
        result = parse(raw)
    except ContractParseError as exc:
        console.print(f"[red]Contract parse failed:[/] {exc.reason}")
        return None
    return {f.path.replace("\\", "/"): f.content for f in result.files}


# ── command body ────────────────────────────────────────────────────────────


def run(
    project_dir: Path,
    *,
    dry_run: bool,
    continue_: bool,
    yes: bool,
    debug: bool,
) -> None:
    """Body for ``cmd_update``. The ``@app.command`` lives in cli.py and
    delegates here.
    """
    # ``_print_source_status`` + ``_exit_on_source_config_error`` live in
    # cli.py. Lazy-import to avoid the module-load cycle with this module.
    from agent_scaffold.cli import _exit_on_source_config_error, _print_source_status

    del debug  # currently unused; reserved for future verbose flag
    project_dir = project_dir.expanduser().resolve()
    try:
        manifest = read_manifest(project_dir)
    except ManifestNotFoundError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if continue_:
        _continue_update(project_dir, manifest)
        return

    in_progress = _read_in_progress(project_dir)
    if in_progress is not None:
        console.print(
            "[yellow]Previous update in progress.[/] "
            "Pass --continue to resume after resolving markers, "
            f"or `rm {_in_progress_path(project_dir)}` to abandon."
        )
        raise typer.Exit(code=1)

    try:
        cfg = load_config()
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        dep_source = resolve_deployments(
            override=cfg.deployments_path,
            mode=cfg.deployments_source,
            cache_dir=cfg.cache_dir,
        )
    except SourceConfigError as exc:
        _exit_on_source_config_error(exc)
    except SourceFetchError as exc:
        console.print(f"[red]Source resolution error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    if dep_source.path is None:
        console.print("[red]Could not resolve deployments source.[/]")
        raise typer.Exit(code=1)
    deployments = dep_source.path
    _print_source_status("Deployments", dep_source)

    current_sha = compute_template_sha(deployments)
    if manifest.template_snapshot_sha == current_sha:
        console.print("[green]Template unchanged since last generation.[/] Nothing to update.")
        raise typer.Exit(code=0)

    bootstrap = not (
        manifest.template_snapshot_sha and has_snapshot(project_dir, manifest.template_snapshot_sha)
    )
    if bootstrap:
        console.print(
            "[yellow]No prior generation snapshot;[/] bootstrapping by snapshotting "
            "the current project files. Next update will produce real diffs."
        )
        on_disk_files = {
            f.path: (project_dir / f.path).read_text(encoding="utf-8", errors="replace")
            for f in manifest.files
            if (project_dir / f.path).is_file()
        }
        snap = save_generation_snapshot(project_dir, current_sha, on_disk_files)
        prune_snapshots(project_dir)
        manifest = manifest.model_copy(update={"template_snapshot_sha": snap.sha})
        write_manifest(project_dir, manifest)
        raise typer.Exit(code=0)

    base_tmp = load_generation_snapshot(project_dir, manifest.template_snapshot_sha or "")
    try:
        base_files_dir = base_tmp
        fresh_files = _regenerate_for_update(manifest, deployments, cfg)
        if fresh_files is None:
            console.print(
                "[red]Regeneration failed.[/] Original files untouched. "
                "Run with --debug to see the raw model response (if cached)."
            )
            raise typer.Exit(code=1)
        previous_paths = [f.path for f in manifest.files]
        classification = _classify_update(
            base_files_dir, project_dir, fresh_files, previous_files=previous_paths
        )
        console.print(_render_update_plan(classification))
        if dry_run:
            raise typer.Exit(code=0)
        if not yes:
            action = _interactive_select(
                "Apply?",
                choices=[
                    ("yes", "yes — apply the merge above"),
                    ("dry-run", "dry-run — print the plan only (no writes)"),
                    ("no", "no — abort without changes"),
                ],
                default="yes",
            )
            if action == "no":
                console.print("[yellow]Aborted.[/]")
                raise typer.Exit(code=0)
            if action == "dry-run":
                raise typer.Exit(code=0)

        remove_decisions = _decide_removals(classification.removed, yes=yes)
        _apply_update(project_dir, fresh_files, classification, remove_decisions=remove_decisions)

        if classification.conflicted:
            _write_in_progress(
                project_dir,
                {
                    "from_template_sha": manifest.template_snapshot_sha,
                    "to_template_sha": current_sha,
                    "from_schema": manifest.schema_version,
                    "to_schema": manifest.schema_version,
                    "conflicts": classification.conflicted,
                    "files_added": classification.added,
                    "files_modified": classification.modified,
                    "files_removed": [r for r, drop in remove_decisions.items() if drop],
                    "model": manifest.model,
                },
            )
            console.print(
                f"[yellow]{len(classification.conflicted)} conflict(s) require manual "
                "resolution.[/]\n  "
                + "\n  ".join(classification.conflicted)
                + "\n\nResolve markers, then run `agent-scaffold update --continue`."
            )
            raise typer.Exit(code=2)

        _finalise_update(
            project_dir,
            manifest,
            current_sha,
            classification,
            removed=[r for r, drop in remove_decisions.items() if drop],
            generated_files=fresh_files,
        )
        console.print(
            f"[green]Update complete.[/] "
            f"+{len(classification.added)} / ~{len(classification.modified)} / "
            f"-{sum(remove_decisions.values())} files."
        )
    finally:
        cleanup_tempdir(base_tmp)
