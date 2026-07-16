"""Tests for ``agent_scaffold.sources``.

All HTTP is monkeypatched — tests must never hit github.com.
"""

from __future__ import annotations

import io
import json
import tarfile
import urllib.error
from pathlib import Path

import pytest

from agent_scaffold.sources import (
    BLUEPRINTS_SPEC,
    DEPLOYMENTS_SPEC,
    SourceFetchError,
    _gc_old_revisions,
    _github_head_sha,
    _latest_cached_revision,
    _safe_extract,
    resolve_source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal urlopen response good enough for ``json.load`` + headers + copyfileobj."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._buf = io.BytesIO(body)
        self.headers = headers or {}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self._buf.close()


def _make_tarball(
    tmp_dir: Path,
    *,
    top_dir: str = "agent-deployments-main",
    files: dict[str, str] | None = None,
) -> Path:
    """Build a tarball that mimics codeload.github.com's layout."""
    files = files or {"docs/recipes/foo.md": "# foo\n"}
    tar_path = tmp_dir / "fixture.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tar:
        # Top-level directory entry.
        info = tarfile.TarInfo(top_dir)
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        for rel, body in files.items():
            data = body.encode("utf-8")
            info = tarfile.TarInfo(f"{top_dir}/{rel}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return tar_path


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


# ---------------------------------------------------------------------------
# _github_head_sha
# ---------------------------------------------------------------------------


def test_github_head_sha_fresh_call(monkeypatch: pytest.MonkeyPatch, cache_dir: Path) -> None:
    """First call hits the API, stores SHA + ETag."""
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir
    cache_root.mkdir(parents=True)

    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResponse:
        captured["url"] = req.full_url  # type: ignore[attr-defined]
        captured["headers"] = dict(req.headers)  # type: ignore[attr-defined]
        return _FakeResponse(
            json.dumps({"sha": "abc1234deadbeef" * 2}).encode("utf-8"),
            headers={"ETag": 'W/"fake-etag"'},
        )

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fake_urlopen)
    sha = _github_head_sha(spec, cache_root)
    assert sha.startswith("abc1234")
    assert (cache_root / "HEAD.sha").read_text().startswith("abc1234")
    assert (cache_root / "HEAD.etag").read_text() == 'W/"fake-etag"'
    # The first call has no prior etag, so If-None-Match header is absent.
    assert "If-none-match" not in captured["headers"]  # type: ignore[index]


def test_github_head_sha_uses_etag_on_second_call(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
) -> None:
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir
    cache_root.mkdir(parents=True)
    (cache_root / "HEAD.etag").write_text('W/"saved"')
    (cache_root / "HEAD.sha").write_text("oldsha" * 8)
    # Force a refresh by aging the cached HEAD.sha mtime.
    import os as _os
    import time as _time

    past = _time.time() - 3600
    _os.utime(cache_root / "HEAD.sha", (past, past))

    captured: dict[str, object] = {}

    def fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResponse:
        captured["headers"] = dict(req.headers)  # type: ignore[attr-defined]
        return _FakeResponse(
            json.dumps({"sha": "newsha" * 8}).encode("utf-8"),
            headers={"ETag": 'W/"newetag"'},
        )

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fake_urlopen)
    sha = _github_head_sha(spec, cache_root)
    assert sha == "newsha" * 8
    assert captured["headers"]["If-none-match"] == 'W/"saved"'  # type: ignore[index]


def test_github_head_sha_304_returns_cached(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
) -> None:
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir
    cache_root.mkdir(parents=True)
    (cache_root / "HEAD.sha").write_text("cachedsha" * 5)
    # Age the file so we don't short-circuit.
    import os as _os
    import time as _time

    past = _time.time() - 3600
    _os.utime(cache_root / "HEAD.sha", (past, past))

    def fake_urlopen(req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.HTTPError(req.full_url, 304, "Not Modified", {}, None)  # type: ignore[attr-defined]

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fake_urlopen)
    sha = _github_head_sha(spec, cache_root)
    assert sha == "cachedsha" * 5


def test_github_head_sha_short_circuits_when_fresh(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
) -> None:
    """Recent cache → no network call at all."""
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir
    cache_root.mkdir(parents=True)
    (cache_root / "HEAD.sha").write_text("freshsha" * 5)

    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise AssertionError("should not be called")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    assert _github_head_sha(spec, cache_root) == "freshsha" * 5


def test_github_head_sha_network_error_raises_source_fetch_error(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
) -> None:
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir

    def fake_urlopen(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("dns failure")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(SourceFetchError, match="URLError"):
        _github_head_sha(spec, cache_root)


# ---------------------------------------------------------------------------
# _safe_extract
# ---------------------------------------------------------------------------


def test_safe_extract_strips_top_dir(tmp_path: Path) -> None:
    tar = _make_tarball(tmp_path, files={"docs/recipes/x.md": "x\n"})
    dest = tmp_path / "out"
    dest.mkdir()
    _safe_extract(tar, dest, strip_top_dir=True)
    assert (dest / "docs" / "recipes" / "x.md").read_text() == "x\n"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    tar_path = tmp_path / "evil.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tar:
        info = tarfile.TarInfo("agent-deployments-main")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        info = tarfile.TarInfo("agent-deployments-main/../../etc/passwd")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"pwn\n"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceFetchError, match="unsafe tar member"):
        _safe_extract(tar_path, dest, strip_top_dir=True)


def test_safe_extract_rejects_absolute_paths(tmp_path: Path) -> None:
    tar_path = tmp_path / "abs.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tar:
        info = tarfile.TarInfo("/etc/passwd")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"pwn\n"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceFetchError):
        _safe_extract(tar_path, dest, strip_top_dir=False)


def test_safe_extract_rejects_symlink_escape(tmp_path: Path) -> None:
    tar_path = tmp_path / "sym.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tar:
        info = tarfile.TarInfo("agent-deployments-main")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
        info = tarfile.TarInfo("agent-deployments-main/danger")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../../../etc/passwd"
        tar.addfile(info)
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(SourceFetchError, match="symlink escape"):
        _safe_extract(tar_path, dest, strip_top_dir=True)


# ---------------------------------------------------------------------------
# _gc_old_revisions
# ---------------------------------------------------------------------------


def test_gc_keeps_newest_n(tmp_path: Path) -> None:
    root = tmp_path / "deployments"
    root.mkdir()
    import os as _os
    import time as _time

    revs: list[Path] = []
    base = _time.time()
    for i, name in enumerate(["old", "older", "oldest", "newer", "newest"]):
        p = root / name
        p.mkdir()
        (p / "marker").write_text(name)
        # Spread mtimes so sort is unambiguous.
        ts = base - (10 - i) * 60
        _os.utime(p, (ts, ts))
        revs.append(p)

    _gc_old_revisions(root, keep=3)
    remaining = sorted(p.name for p in root.iterdir())
    # The three with the newest mtimes survive ("oldest" got a newer ts due
    # to the loop ordering — the test exercises mtime ordering, not naming).
    assert len(remaining) == 3


# ---------------------------------------------------------------------------
# resolve_source — precedence + fallbacks
# ---------------------------------------------------------------------------


def test_resolve_source_override_wins(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=local,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "explicit-path"
    assert resolved.path == local.resolve()
    assert resolved.commit_sha is None


def test_resolve_source_env_var_used_when_no_override(tmp_path: Path) -> None:
    local = tmp_path / "from-env"
    local.mkdir()
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={"AGENT_SCAFFOLD_DEPLOYMENTS_PATH": str(local)},
    )
    assert resolved.kind == "env-path"
    assert resolved.path == local.resolve()


def test_resolve_source_bundled_mode_explicit(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="bundled",
        cache_dir=tmp_path / "cache",
        bundled_fallback=bundled,
        env={},
    )
    assert resolved.kind == "bundled-explicit"
    assert resolved.path == bundled


def test_resolve_source_skip_mode_returns_none_path(tmp_path: Path) -> None:
    resolved = resolve_source(
        BLUEPRINTS_SPEC,
        override=None,
        mode="skip",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "skipped"
    assert resolved.path is None


def test_resolve_source_auto_falls_back_to_bundled_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "bundled"
    bundled.mkdir()

    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=bundled,
        env={},
    )
    assert resolved.kind == "bundled-fallback"
    assert resolved.path == bundled
    assert "offline" in resolved.label.lower()
    # S3: structured fallback fields so callers don't have to parse the label.
    assert resolved.used_fallback is True
    assert resolved.fallback_reason is not None
    assert "offline" in resolved.fallback_reason.lower()


def test_resolve_source_fresh_fetch_has_used_fallback_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S3: a healthy fetch must NOT report fallback state."""
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    fetched_dir = tmp_path / "fetched"
    fetched_dir.mkdir()

    def fake_fetch(
        spec,
        cache_root,  # type: ignore[no-untyped-def]
        *,
        refresh=False,
    ):
        return fetched_dir, "abc1234567" * 4, False, refresh  # path, sha, was_cached, checked

    monkeypatch.setattr("agent_scaffold.sources._fetch_or_use_cache", fake_fetch)
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=bundled,
        env={},
    )
    assert resolved.kind == "fetched"
    assert resolved.used_fallback is False
    assert resolved.fallback_reason is None


def test_resolve_source_bad_override_raises_source_config_error(tmp_path: Path) -> None:
    """S3: bad --path override is a config error, not a fetch error."""
    from agent_scaffold.sources import SourceConfigError

    with pytest.raises(SourceConfigError, match="does not exist"):
        resolve_source(
            DEPLOYMENTS_SPEC,
            override=tmp_path / "missing",
            mode="auto",
            cache_dir=tmp_path / "cache",
            bundled_fallback=None,
            env={},
        )


def test_resolve_source_unknown_mode_raises_source_config_error(tmp_path: Path) -> None:
    """S3: unknown mode is a config error so callers exit instead of retrying."""
    from agent_scaffold.sources import SourceConfigError

    with pytest.raises(SourceConfigError, match="unknown source mode"):
        resolve_source(
            DEPLOYMENTS_SPEC,
            override=None,
            mode="lemonparty",
            cache_dir=tmp_path / "cache",
            bundled_fallback=None,
            env={},
        )


def test_source_config_error_is_a_source_fetch_error() -> None:
    """Backward compat: existing ``except SourceFetchError`` clauses must keep
    catching SourceConfigError so callers that haven't migrated yet still work."""
    from agent_scaffold.sources import SourceConfigError, SourceFetchError, SourceNetworkError

    assert issubclass(SourceConfigError, SourceFetchError)
    assert issubclass(SourceNetworkError, SourceFetchError)


def test_resolve_source_auto_skips_blueprints_when_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    resolved = resolve_source(
        BLUEPRINTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,  # blueprints has no bundled copy
        env={},
    )
    assert resolved.kind == "skipped"
    assert resolved.path is None
    assert "offline" in resolved.label.lower()


def test_resolve_source_auto_cache_hit_no_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = DEPLOYMENTS_SPEC
    cache_root = tmp_path / "cache" / spec.cache_subdir
    sha = "abc1234deadbeef" * 2
    extracted = cache_root / sha
    (extracted / "docs" / "recipes").mkdir(parents=True)
    (extracted / "docs" / "recipes" / "x.md").write_text("# x\n")

    def head_only(req: object, timeout: float = 8.0) -> _FakeResponse:
        if "api.github.com" in req.full_url:  # type: ignore[attr-defined]
            return _FakeResponse(
                json.dumps({"sha": sha}).encode("utf-8"),
                headers={"ETag": 'W/"x"'},
            )
        raise AssertionError("download should not happen on cache hit")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", head_only)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "cached"
    assert resolved.commit_sha == sha
    assert resolved.path == extracted


def test_resolve_source_auto_full_fetch_extracts_tarball(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = DEPLOYMENTS_SPEC
    sha = "freshsha000000000000000000"
    # Build a fixture tarball, then have urlopen stream it back.
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    tar = _make_tarball(
        fixture_dir,
        top_dir=f"{spec.repo.split('/')[1]}-{spec.branch}",
        files={"docs/recipes/foo.md": "# foo\n"},
    )
    tar_bytes = tar.read_bytes()

    def urlopen(req: object, timeout: float = 8.0) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[attr-defined]
        if "api.github.com" in url:
            return _FakeResponse(
                json.dumps({"sha": sha}).encode("utf-8"),
                headers={"ETag": 'W/"x"'},
            )
        if "codeload.github.com" in url:
            return _FakeResponse(tar_bytes)
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", urlopen)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "fetched"
    assert resolved.commit_sha == sha
    assert resolved.path is not None
    assert (resolved.path / "docs" / "recipes" / "foo.md").read_text() == "# foo\n"


# ---------------------------------------------------------------------------
# Offline / rate-limited cache fallback (deployments has no bundled fallback)
# ---------------------------------------------------------------------------


def test_resolve_source_uses_stale_cache_when_github_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A populated cache + an unreachable GitHub (rate-limit / offline) must fall
    back to the cached revision instead of returning path=None — the
    'deployments source unavailable' brick when the HEAD-SHA probe fails."""
    cache = tmp_path / "cache"
    rev = cache / DEPLOYMENTS_SPEC.cache_subdir / "abc1234deadbeef"
    (rev / "docs" / "recipes").mkdir(parents=True)
    (rev / "docs" / "recipes" / "foo.md").write_text("# foo\n", encoding="utf-8")

    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=cache,
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "cached"
    assert resolved.path == rev


def test_resolve_source_cold_cache_offline_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No cache + offline genuinely fails — the one case we can't paper over."""

    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    resolved = resolve_source(
        DEPLOYMENTS_SPEC,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "skipped"
    assert resolved.path is None


def test_latest_cached_revision_picks_newest_nonempty(tmp_path: Path) -> None:
    import os

    root = tmp_path / "deployments"
    root.mkdir()
    (root / "HEAD.sha").write_text("x", encoding="utf-8")  # bookkeeping file — ignored
    (root / "emptyrev").mkdir()  # empty dir — ignored
    old = root / "oldsha"
    (old / "docs").mkdir(parents=True)
    (old / "docs" / "a.md").write_text("a", encoding="utf-8")
    new = root / "newsha"
    (new / "docs").mkdir(parents=True)
    (new / "docs" / "b.md").write_text("b", encoding="utf-8")
    os.utime(old, (1, 1))  # make `old` clearly older

    assert _latest_cached_revision(root) == new
    assert _latest_cached_revision(tmp_path / "does-not-exist") is None


# ---------------------------------------------------------------------------
# refresh=True — the REPL's startup sync
# ---------------------------------------------------------------------------


def test_github_head_sha_refresh_bypasses_ttl(
    monkeypatch: pytest.MonkeyPatch, cache_dir: Path
) -> None:
    """refresh=True must skip the TTL short-circuit and hit the API."""
    spec = DEPLOYMENTS_SPEC
    cache_root = cache_dir / spec.cache_subdir
    cache_root.mkdir(parents=True)
    (cache_root / "HEAD.sha").write_text("staleold" * 5)

    calls = {"n": 0}

    def fake_urlopen(_req: object, timeout: float = 8.0) -> _FakeResponse:
        calls["n"] += 1
        return _FakeResponse(
            json.dumps({"sha": "confirmed" * 5}).encode("utf-8"),
            headers={"ETag": 'W/"y"'},
        )

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fake_urlopen)
    assert _github_head_sha(spec, cache_root, refresh=True) == "confirmed" * 5
    assert calls["n"] == 1


def test_resolve_source_refresh_up_to_date_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """refresh + network-confirmed SHA + extracted tree -> 'up to date'."""
    spec = DEPLOYMENTS_SPEC
    cache_root = tmp_path / "cache" / spec.cache_subdir
    sha = "abc1234deadbeef" * 2
    extracted = cache_root / sha
    (extracted / "docs").mkdir(parents=True)
    (extracted / "docs" / "x.md").write_text("# x\n")

    def head_only(req: object, timeout: float = 8.0) -> _FakeResponse:
        if "api.github.com" in req.full_url:  # type: ignore[attr-defined]
            return _FakeResponse(
                json.dumps({"sha": sha}).encode("utf-8"), headers={"ETag": 'W/"x"'}
            )
        raise AssertionError("download should not happen")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", head_only)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
        refresh=True,
    )
    assert resolved.kind == "cached"
    assert "(up to date)" in resolved.label


def test_resolve_source_refresh_updated_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """refresh + new SHA + tarball download -> 'updated'."""
    spec = DEPLOYMENTS_SPEC
    sha = "freshsha000000000000000000"
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()
    tar = _make_tarball(
        fixture_dir,
        top_dir=f"{spec.repo.split('/')[1]}-{spec.branch}",
        files={"docs/recipes/foo.md": "# foo\n"},
    )
    tar_bytes = tar.read_bytes()

    def urlopen(req: object, timeout: float = 8.0) -> _FakeResponse:
        url = req.full_url if hasattr(req, "full_url") else str(req)  # type: ignore[attr-defined]
        if "api.github.com" in url:
            return _FakeResponse(
                json.dumps({"sha": sha}).encode("utf-8"), headers={"ETag": 'W/"x"'}
            )
        return _FakeResponse(tar_bytes)

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", urlopen)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
        refresh=True,
    )
    assert resolved.kind == "fetched"
    assert "(updated)" in resolved.label


def test_resolve_source_refresh_offline_keeps_cached_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """refresh that could not reach GitHub must not claim freshness."""
    spec = DEPLOYMENTS_SPEC
    cache_root = tmp_path / "cache" / spec.cache_subdir
    extracted = cache_root / ("oldsha" * 6)
    (extracted / "docs").mkdir(parents=True)
    (extracted / "docs" / "x.md").write_text("# x\n")

    def offline(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", offline)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
        refresh=True,
    )
    assert resolved.kind == "cached"
    assert "(cached)" in resolved.label
    assert "up to date" not in resolved.label


def test_resolve_source_default_no_refresh_keeps_legacy_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default calls (no refresh) label exactly as before."""
    spec = DEPLOYMENTS_SPEC
    cache_root = tmp_path / "cache" / spec.cache_subdir
    sha = "abc1234deadbeef" * 2
    extracted = cache_root / sha
    (extracted / "docs").mkdir(parents=True)
    (extracted / "docs" / "x.md").write_text("# x\n")
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "HEAD.sha").write_text(sha)

    def fail(_req: object, timeout: float = 8.0) -> _FakeResponse:
        raise AssertionError("TTL short-circuit should prevent network")

    monkeypatch.setattr("agent_scaffold.sources.urllib.request.urlopen", fail)
    resolved = resolve_source(
        spec,
        override=None,
        mode="auto",
        cache_dir=tmp_path / "cache",
        bundled_fallback=None,
        env={},
    )
    assert resolved.kind == "cached"
    assert "(cached)" in resolved.label
