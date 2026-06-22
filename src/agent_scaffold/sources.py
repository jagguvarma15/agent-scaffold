"""Resolve where `agent-deployments` and `agent-blueprints` come from.

Bare `agent-scaffold new` should "just work" — historically the user had to
clone agent-deployments and either pass ``--deployments-path`` or set
``AGENT_SCAFFOLD_DEPLOYMENTS_PATH``. The bundled snapshot was a fallback
but quickly went stale between scaffold releases. Blueprints was never
fetched at all, so cross-repo URLs (``github.com/.../agent-blueprints/...``)
in deployments docs were silently dropped by the context assembler.

This module fetches both repos directly from GitHub on demand, caches them
locally keyed by commit SHA, and falls back gracefully when the network is
unavailable. The same primitives serve both repos; the only differences
are the offline fallback (bundled for deployments, skip for blueprints).

Cache layout::

    ~/.cache/agent-scaffold/
      deployments/
        abc1234.../          # extracted tarball
        HEAD.sha             # cached HEAD ref for short-circuit
        HEAD.etag            # last GitHub ETag for conditional GET
      blueprints/
        9876fed.../
        HEAD.sha
        HEAD.etag

Network calls go through :func:`urllib.request.urlopen` so tests can
monkeypatch the transport without touching GitHub.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Public type aliases the CLI uses for its --*-source flags.
# vX+1: bundled mode is no longer a valid deployments source — the bundled
# snapshot has been removed in favor of the catalog + on-disk fetch cache.
# bundled-fallback / bundled-explicit kinds stay in SourceKind so callers
# that construct ResolvedSource directly (a few tests, the lower-level
# resolve_source with a custom bundled_fallback path) keep type-checking.
DeploymentsMode = Literal["auto"]
BlueprintsMode = Literal["auto", "skip"]
SourceKind = Literal[
    "explicit-path",
    "env-path",
    "fetched",
    "cached",
    "bundled-fallback",
    "bundled-explicit",
    "skipped",
    "vendored",
]

# GitHub unauthenticated rate limit is 60 req/hr per IP. The conditional GET
# (If-None-Match) returns 304 without consuming quota when the ref is
# unchanged, but we also short-circuit if we hit HEAD within this window so
# back-to-back `agent-scaffold` invocations don't even round-trip.
_HEAD_REFRESH_SECONDS = 300

# How many cached extracted tarballs to keep before LRU-pruning. 3 covers
# "previous + current + one for safety" without bloating disk.
_MAX_CACHED_REVISIONS = 3

# Per-request timeout. Short enough to fail fast when offline, long enough
# that a slow link doesn't false-positive.
_NETWORK_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class RepoSpec:
    """Static identity of a repo we know how to fetch."""

    repo: str  # "owner/repo"
    branch: str
    cache_subdir: str  # name under ~/.cache/agent-scaffold/


DEPLOYMENTS_SPEC = RepoSpec("jagguvarma15/agent-deployments", "main", "deployments")
BLUEPRINTS_SPEC = RepoSpec("jagguvarma15/agent-blueprints", "main", "blueprints")


@dataclass(frozen=True)
class ResolvedSource:
    """A resolved source for use by the rest of the pipeline.

    ``path`` is ``None`` only when ``kind == "skipped"`` (blueprints with no
    fetch and no bundled fallback). All other kinds have a real directory.

    ``used_fallback`` + ``fallback_reason`` (S3) make the offline / bundled-
    fallback path queryable without parsing ``label``. Downstream surfaces
    (CLI status print, plan panel) check ``used_fallback`` to decide whether
    to highlight "you're on a stale snapshot" to the user.
    """

    spec: RepoSpec
    path: Path | None
    label: str
    kind: SourceKind
    commit_sha: str | None  # populated for kind in {fetched, cached}
    used_fallback: bool = False
    fallback_reason: str | None = None


class SourceFetchError(Exception):
    """Base class for source resolution failures.

    Two subclasses (S3) carry the failure mode so CLI handlers can branch:

    - :class:`SourceConfigError` — user input is wrong (bad path, bad mode,
      missing bundled snapshot when ``--source=bundled``). Exit; the user
      must fix their config.
    - :class:`SourceNetworkError` — transient (GitHub down, timeout, rate
      limit). Eaten internally by the fallback path unless no fallback
      exists; if it bubbles up, treat as recoverable and warn.

    Existing ``except SourceFetchError`` blocks keep working — both subclasses
    inherit, so the base catches either.
    """


class SourceConfigError(SourceFetchError):
    """User-input error: bad path, bad mode, missing required bundled snapshot.

    Exit with a red error message — fixing the input is the only path forward.
    """


class SourceNetworkError(SourceFetchError):
    """Transient fetch failure (GitHub unreachable, timeout, 5xx, rate limit).

    Caller decides whether to fall back. The auto-resolver catches this
    internally and returns a bundled / skipped ``ResolvedSource``; only
    surfaces to the user when no fallback is configured.
    """


# ---------------------------------------------------------------------------
# Public resolvers
# ---------------------------------------------------------------------------


def resolve_source(
    spec: RepoSpec,
    *,
    override: Path | None,
    mode: str,
    cache_dir: Path,
    bundled_fallback: Path | None,
    env: dict[str, str] | None = None,
) -> ResolvedSource:
    """Resolve a single repo to a local directory and a human-readable label.

    Resolution order:

    1. ``override`` (e.g. ``--deployments-path``) — explicit, never network.
    2. ``mode == "bundled"`` and a bundled copy exists — use it (deployments).
    3. ``mode == "skip"`` — return a ``skipped`` source with ``path=None``
       (blueprints opt-out).
    4. ``mode == "auto"`` — try GitHub fetch; on failure, fall back to
       bundled (if provided) or skip.

    Network failure during step 4 is non-fatal: we either fall back or
    return a skipped source with a label that surfaces why.
    """
    env_map = os.environ if env is None else env

    if override is not None:
        path = override.expanduser().resolve()
        if not path.is_dir():
            raise SourceConfigError(f"{spec.repo}: --path override does not exist: {path}")
        return ResolvedSource(
            spec=spec,
            path=path,
            label=f"local: {path}",
            kind="explicit-path",
            commit_sha=None,
        )

    # Env vars come in via the caller, not directly here, but we still honor
    # them as a path-like override for any caller who forgot to plumb them.
    env_key = f"AGENT_SCAFFOLD_{spec.cache_subdir.upper()}_PATH"
    env_val = env_map.get(env_key)
    if env_val:
        path = Path(env_val).expanduser().resolve()
        if not path.is_dir():
            raise SourceConfigError(f"{spec.repo}: ${env_key} does not exist: {path}")
        return ResolvedSource(
            spec=spec,
            path=path,
            label=f"env: {path}",
            kind="env-path",
            commit_sha=None,
        )

    if mode == "bundled":
        if bundled_fallback is None:
            raise SourceConfigError(f"{spec.repo}: --source=bundled requested but no bundled copy")
        return ResolvedSource(
            spec=spec,
            path=bundled_fallback,
            label=f"bundled (explicit): {bundled_fallback}",
            kind="bundled-explicit",
            commit_sha=None,
        )

    if mode == "skip":
        return ResolvedSource(
            spec=spec,
            path=None,
            label=f"skipped (explicit): {spec.repo}",
            kind="skipped",
            commit_sha=None,
        )

    if mode != "auto":
        raise SourceConfigError(f"{spec.repo}: unknown source mode {mode!r}")

    # mode == "auto": try GitHub, fall back as needed.
    try:
        path, sha, was_cached = _fetch_or_use_cache(spec, cache_dir)
        kind: SourceKind = "cached" if was_cached else "fetched"
        return ResolvedSource(
            spec=spec,
            path=path,
            label=f"github.com/{spec.repo} @ {sha[:7]} ({kind})",
            kind=kind,
            commit_sha=sha,
        )
    except SourceFetchError as exc:
        # Network down / GitHub unreachable / rate-limited. Fall back.
        # Promote the raw fetch error to SourceNetworkError so callers that
        # care about transient-vs-config can branch. (The auto-resolver
        # catches it here so the user only ever sees the fallback path.)
        reason = str(exc)
        if bundled_fallback is not None:
            return ResolvedSource(
                spec=spec,
                path=bundled_fallback,
                label=f"bundled (offline fallback: {reason})",
                kind="bundled-fallback",
                commit_sha=None,
                used_fallback=True,
                fallback_reason=reason,
            )
        return ResolvedSource(
            spec=spec,
            path=None,
            label=f"skipped (offline: {reason})",
            kind="skipped",
            commit_sha=None,
            used_fallback=True,
            fallback_reason=reason,
        )


def resolve_deployments(
    *,
    override: Path | None,
    mode: DeploymentsMode,
    cache_dir: Path,
    env: dict[str, str] | None = None,
) -> ResolvedSource:
    # ``bundled_fallback=None`` — the bundled snapshot has been removed.
    # If someone explicitly passes mode="bundled", they must provide their
    # own ``bundled_fallback`` path via the lower-level resolve_source.
    return resolve_source(
        DEPLOYMENTS_SPEC,
        override=override,
        mode=mode,
        cache_dir=cache_dir,
        bundled_fallback=None,
        env=env,
    )


VENDORED_BLUEPRINTS_SUBPATH = ("vendored", "blueprints")
"""Relative location of the vendored blueprints snapshot inside an
``agent-deployments`` checkout. Mirrors ``agent-deployments/vendir.yml``'s
declared ``directories[].path``. Kept here as a constant so any
scaffold-side caller resolving blueprints content lands at the same
canonical location without duplicating the literal."""


def resolve_blueprints(
    *,
    override: Path | None,
    mode: BlueprintsMode,
    cache_dir: Path,
    env: dict[str, str] | None = None,
    deployments_path: Path | None = None,
) -> ResolvedSource:
    """Resolve where blueprints content comes from.

    Resolution order:

    1. ``override`` / ``$AGENT_SCAFFOLD_BLUEPRINTS_PATH`` — explicit local path.
    2. ``deployments_path`` carries a ``vendored/blueprints/`` directory — use
       it. This is the standard release-driven path: agent-deployments
       vendors blueprints content via ``vendir`` and exposes it under
       ``vendored/blueprints/``; scaffold reads it directly without a
       separate GitHub fetch.
    3. ``mode == "skip"`` — return a skipped source.
    4. ``mode == "auto"`` — legacy fallback: fetch from GitHub. Kept for
       backward compat with deployments checkouts that pre-date the
       vendored layout. If GitHub is unreachable, falls back to skipped.

    ``deployments_path`` is the resolved deployments directory (output of
    :func:`resolve_deployments`); callers that haven't resolved it yet pass
    ``None`` to skip the vendored shortcut.
    """
    # 1. + 2.: explicit override always wins over the vendored shortcut.
    if override is None and deployments_path is not None:
        vendored = (
            deployments_path / VENDORED_BLUEPRINTS_SUBPATH[0] / VENDORED_BLUEPRINTS_SUBPATH[1]
        )
        if vendored.is_dir() and any(vendored.iterdir()):
            return ResolvedSource(
                spec=BLUEPRINTS_SPEC,
                path=vendored.resolve(),
                label=f"vendored (in deployments): {vendored}",
                kind="vendored",
                commit_sha=None,
            )

    # 3. + 4.: fall through to legacy resolution (override / env / fetch / skip).
    return resolve_source(
        BLUEPRINTS_SPEC,
        override=override,
        mode=mode,
        cache_dir=cache_dir,
        bundled_fallback=None,
        env=env,
    )


# ---------------------------------------------------------------------------
# GitHub HEAD probe (cheap, ETag-cached)
# ---------------------------------------------------------------------------


def _github_head_sha(spec: RepoSpec, cache_root: Path) -> str:
    """Return the SHA of HEAD on ``spec.branch``.

    Uses ``If-None-Match`` against the previous ETag so unchanged refs return
    304 without consuming a rate-limit slot. Also short-circuits if our cached
    HEAD.sha file is fresher than ``_HEAD_REFRESH_SECONDS`` ago — repeated
    runs of ``agent-scaffold`` within the same shell session don't need a
    network call each time.
    """
    head_sha_path = cache_root / "HEAD.sha"
    head_etag_path = cache_root / "HEAD.etag"
    if head_sha_path.is_file():
        age = time.time() - head_sha_path.stat().st_mtime
        if age < _HEAD_REFRESH_SECONDS:
            cached = head_sha_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached

    url = f"https://api.github.com/repos/{spec.repo}/commits/{spec.branch}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})  # noqa: S310 — hardcoded https api.github.com
    prior_etag = ""
    if head_etag_path.is_file():
        prior_etag = head_etag_path.read_text(encoding="utf-8").strip()
        if prior_etag:
            req.add_header("If-None-Match", prior_etag)

    try:
        with urllib.request.urlopen(req, timeout=_NETWORK_TIMEOUT_SECONDS) as resp:  # noqa: S310 — hardcoded https api.github.com
            payload = json.loads(resp.read().decode("utf-8"))
            etag = resp.headers.get("ETag", "")
            sha = payload.get("sha")
            if not isinstance(sha, str) or not sha:
                raise SourceFetchError(f"unexpected payload from {url}")
            cache_root.mkdir(parents=True, exist_ok=True)
            head_sha_path.write_text(sha, encoding="utf-8")
            if etag:
                head_etag_path.write_text(etag, encoding="utf-8")
            return sha
    except urllib.error.HTTPError as exc:
        if exc.code == 304 and head_sha_path.is_file():
            # ref unchanged — refresh mtime so the short-circuit holds.
            cached = head_sha_path.read_text(encoding="utf-8").strip()
            head_sha_path.touch()
            return cached
        raise SourceFetchError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SourceFetchError(f"{type(exc).__name__}: {exc}") from exc


# ---------------------------------------------------------------------------
# Tarball download + safe extract
# ---------------------------------------------------------------------------


def _fetch_or_use_cache(spec: RepoSpec, cache_dir: Path) -> tuple[Path, str, bool]:
    """Return ``(path, sha, was_cached)`` for the latest ``spec`` revision."""
    cache_root = cache_dir / spec.cache_subdir
    sha = _github_head_sha(spec, cache_root)
    target = cache_root / sha
    if target.is_dir() and any(target.iterdir()):
        return target, sha, True
    _download_and_extract(spec, sha, target)
    _gc_old_revisions(cache_root, keep=_MAX_CACHED_REVISIONS)
    return target, sha, False


def _download_and_extract(spec: RepoSpec, sha: str, dest_dir: Path) -> None:
    """Download a branch tarball and safe-extract into ``dest_dir``.

    GitHub serves branch tarballs at ``codeload.github.com/<repo>/tar.gz/<ref>``
    with a top-level directory named ``<repo-name>-<branch>/``. We strip that
    leading component so ``dest_dir`` contains the repo content directly.
    """
    url = f"https://codeload.github.com/{spec.repo}/tar.gz/refs/heads/{spec.branch}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Stream to a temp file we open separately so we don't fight NamedTemporaryFile's
    # context manager (which closes the handle the second the `with` exits).
    fd, tmp_name = tempfile.mkstemp(suffix=".tar.gz")
    tmp_path = Path(tmp_name)
    try:
        try:
            with (
                urllib.request.urlopen(url, timeout=_NETWORK_TIMEOUT_SECONDS) as resp,  # noqa: S310 — hardcoded https codeload.github.com
                os.fdopen(fd, "wb") as tmp_file,
            ):
                shutil.copyfileobj(resp, tmp_file)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SourceFetchError(f"download failed: {type(exc).__name__}: {exc}") from exc
        _safe_extract(tmp_path, dest_dir, strip_top_dir=True)
        # Tag with the SHA so a partial extract that fails mid-way is detectable.
        (dest_dir / ".sha").write_text(sha, encoding="utf-8")
    except Exception:
        # Clean up half-written cache on any failure.
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


def _safe_extract(tar_path: Path, dest_dir: Path, *, strip_top_dir: bool) -> None:
    """Extract ``tar_path`` into ``dest_dir`` with path-traversal protection.

    Validates every member before writing. Rejects:
    - absolute paths
    - any ``..`` component
    - symlinks / hardlinks pointing outside ``dest_dir``
    - device / FIFO entries

    We hand-validate rather than rely on Python 3.12's ``tarfile.data_filter``
    so this works on the project's stated 3.11 floor.
    """
    dest_dir = dest_dir.resolve()
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for member in tar.getmembers():
            if not (member.isfile() or member.isdir() or member.issym()):
                # Skip devices, fifos, hardlinks — defense in depth.
                continue
            name = member.name
            if strip_top_dir:
                # tarball top-level looks like "agent-deployments-main/"
                parts = name.split("/", 1)
                name = parts[1] if len(parts) == 2 else ""
                if not name:
                    continue
            if name.startswith("/") or ".." in Path(name).parts:
                raise SourceFetchError(f"unsafe tar member: {member.name!r}")
            target = (dest_dir / name).resolve()
            try:
                target.relative_to(dest_dir)
            except ValueError as exc:
                raise SourceFetchError(f"path escape: {member.name!r}") from exc
            if member.issym():
                # Resolve symlink target relative to its directory and ensure
                # it stays inside dest_dir.
                link_target = (target.parent / member.linkname).resolve()
                try:
                    link_target.relative_to(dest_dir)
                except ValueError as exc:
                    raise SourceFetchError(f"symlink escape: {member.name!r}") from exc
            # Safe — extract this member with rewritten name. filter="data" silences
            # the Python 3.14 DeprecationWarning and gives us belt-and-suspenders
            # protection on top of our own validation.
            member.name = name
            tar.extract(member, dest_dir, filter="data")


def _gc_old_revisions(cache_root: Path, *, keep: int) -> None:
    """LRU-prune extracted-revision directories, keeping the newest ``keep``."""
    if not cache_root.is_dir():
        return
    revisions = [p for p in cache_root.iterdir() if p.is_dir()]
    revisions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old in revisions[keep:]:
        shutil.rmtree(old, ignore_errors=True)
