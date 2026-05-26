"""Tests for ``agent_scaffold._filesec.secure_write`` + ``assert_secret_mode``."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from agent_scaffold._filesec import (
    MODE_PUBLIC,
    MODE_SECRET,
    FileSecurityError,
    assert_secret_mode,
    secure_write,
)


def test_secure_write_creates_file_with_mode_0600(tmp_path: Path) -> None:
    target = tmp_path / "secret.txt"
    secure_write(target, "hello")
    assert target.is_file()
    actual = stat.S_IMODE(target.stat().st_mode)
    assert actual == 0o600, f"expected 0600, got {oct(actual)}"
    assert target.read_text(encoding="utf-8") == "hello"


def test_secure_write_overrides_mode(tmp_path: Path) -> None:
    target = tmp_path / "public.json"
    secure_write(target, "{}", mode=MODE_PUBLIC)
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_secure_write_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o644)
    secure_write(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    # Mode is re-asserted on overwrite, even if the existing inode was 0o644.
    assert stat.S_IMODE(target.stat().st_mode) == MODE_SECRET


def test_secure_write_handles_bytes(tmp_path: Path) -> None:
    target = tmp_path / "binary"
    secure_write(target, b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"


def test_secure_write_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c.txt"
    secure_write(nested, "x")
    assert nested.is_file()


def test_secure_write_atomic_no_partial_on_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the write step raises, the destination must be unchanged."""
    target = tmp_path / "f.txt"
    target.write_text("original", encoding="utf-8")
    target.chmod(0o600)

    from agent_scaffold import _filesec

    real_write = _filesec.os.write

    def boom(*_args: object, **_kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(_filesec.os, "write", boom)
    with pytest.raises(OSError):
        secure_write(target, "new content")
    monkeypatch.setattr(_filesec.os, "write", real_write)
    # Original file untouched.
    assert target.read_text(encoding="utf-8") == "original"
    # Tempfile cleaned up.
    siblings = list(tmp_path.glob("f.txt.tmp.*"))
    assert siblings == [], f"left orphan tempfiles: {siblings}"


def test_assert_secret_mode_passes_for_0600(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    secure_write(target, "x")
    assert_secret_mode(target)  # no raise


def test_assert_secret_mode_raises_on_wider_mode(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    target.chmod(0o644)
    with pytest.raises(FileSecurityError):
        assert_secret_mode(target)


def test_assert_secret_mode_noop_for_missing_file(tmp_path: Path) -> None:
    assert_secret_mode(tmp_path / "nope")  # no raise
