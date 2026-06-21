"""Tests for ``agent_scaffold.auth_browser`` — local-server paste flow."""

from __future__ import annotations

import re
import threading
import time
import urllib.request
from typing import Any
from urllib.error import HTTPError

import pytest

from agent_scaffold import auth_browser


def _spin_up_in_thread() -> tuple[threading.Thread, list[str | None]]:
    """Run browser_paste_flow in a background thread and collect its return value."""
    captured: list[str | None] = [None]

    def runner() -> None:
        captured[0] = auth_browser.browser_paste_flow(timeout_seconds=5)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, captured


def _spin_up_with(**kwargs: Any) -> tuple[threading.Thread, list[str | None]]:
    """Like ``_spin_up_in_thread`` but forwards kwargs (label/hint/placeholder)."""
    captured: list[str | None] = [None]

    def runner() -> None:
        captured[0] = auth_browser.browser_paste_flow(timeout_seconds=5, **kwargs)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    return thread, captured


def _get_html(url: str) -> str:
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.read().decode("utf-8")


def _find_local_url(opened: list[str]) -> str:
    # webbrowser.open is patched to record the URL; first call wins.
    deadline = time.time() + 3
    while time.time() < deadline:
        if opened:
            return opened[0]
        time.sleep(0.02)
    raise AssertionError("server never opened a browser URL")


def _extract_csrf(url: str) -> str:
    """GET the form page and pull the CSRF token out of the hidden input."""
    with urllib.request.urlopen(url, timeout=2) as response:
        html = response.read().decode("utf-8")
    match = re.search(r'name="csrf"\s+value="([^"]+)"', html)
    if match is None:
        raise AssertionError(f"no csrf token in form page: {html[:200]}")
    return match.group(1)


def _post(url: str, csrf: str, api_key: str) -> tuple[int, str]:
    import urllib.parse

    body = urllib.parse.urlencode({"csrf": csrf, "api_key": api_key}).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/submit",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, response.read().decode("utf-8")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


@pytest.fixture
def captured_browser(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch webbrowser.open to record the URL instead of launching a browser."""
    opened: list[str] = []

    def fake_open(url: str, *args: Any, **kwargs: Any) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(auth_browser.webbrowser, "open", fake_open)
    return opened


def test_browser_paste_flow_happy_path(captured_browser: list[str]) -> None:
    thread, captured = _spin_up_in_thread()
    url = _find_local_url(captured_browser)
    csrf = _extract_csrf(url)
    status, body = _post(url, csrf, "sk-ant-pasted-key-123")
    assert status == 200
    assert "close this tab" in body.lower()
    thread.join(timeout=3)
    assert captured[0] == "sk-ant-pasted-key-123"


def test_browser_paste_flow_csrf_rejection(captured_browser: list[str]) -> None:
    thread, captured = _spin_up_in_thread()
    url = _find_local_url(captured_browser)
    _extract_csrf(url)  # warm the page; not used
    status, body = _post(url, csrf="wrong-csrf-token", api_key="sk-ant-malicious-1234")
    assert status == 403
    assert "rejected" in body.lower()
    # Submit a real one so the thread can exit cleanly.
    real_csrf = _extract_csrf(url)
    _post(url, real_csrf, "sk-ant-real-key1234")
    thread.join(timeout=3)
    assert captured[0] == "sk-ant-real-key1234"


def test_browser_paste_flow_empty_key_rejected(captured_browser: list[str]) -> None:
    thread, captured = _spin_up_in_thread()
    url = _find_local_url(captured_browser)
    csrf = _extract_csrf(url)
    status, _body = _post(url, csrf, "")
    assert status == 403
    real_csrf = _extract_csrf(url)
    _post(url, real_csrf, "sk-ant-real-key1234")
    thread.join(timeout=3)
    assert captured[0] == "sk-ant-real-key1234"


def test_browser_paste_flow_timeout_returns_none(
    monkeypatch: pytest.MonkeyPatch, captured_browser: list[str]
) -> None:
    """If nothing posts within the timeout, return None and tear down cleanly."""
    result = auth_browser.browser_paste_flow(timeout_seconds=1)
    assert result is None


def test_browser_paste_flow_falls_back_when_webbrowser_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When ``webbrowser.open`` returns False, the URL is still printed."""

    def cannot_open(url: str, *args: Any, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(auth_browser.webbrowser, "open", cannot_open)
    thread, captured = _spin_up_in_thread()
    # Wait for "Open this URL" prompt to appear in stdout.
    deadline = time.time() + 3
    url: str | None = None
    while time.time() < deadline and url is None:
        out = capsys.readouterr().out
        match = re.search(r"http://127\.0\.0\.1:\d+/", out)
        if match:
            url = match.group(0)
            break
        time.sleep(0.05)
    assert url is not None, "no URL printed when webbrowser.open returned False"
    csrf = _extract_csrf(url)
    _post(url, csrf, "sk-ant-fallback-1234")
    thread.join(timeout=3)
    assert captured[0] == "sk-ant-fallback-1234"


def test_browser_paste_flow_get_unknown_path_404(captured_browser: list[str]) -> None:
    thread, captured = _spin_up_in_thread()
    url = _find_local_url(captured_browser)
    request = urllib.request.Request(url.rstrip("/") + "/totally-not-here", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2):
            pytest.fail("expected 404")
    except HTTPError as exc:
        assert exc.code == 404
    # POSTing to a non-/submit path also 404s.
    body_request = urllib.request.Request(
        url.rstrip("/") + "/something",
        data=b"x=1",
        method="POST",
    )
    try:
        with urllib.request.urlopen(body_request, timeout=2):
            pytest.fail("expected 404")
    except HTTPError as exc:
        assert exc.code == 404
    # Final real submit so the server exits.
    csrf = _extract_csrf(url)
    _post(url, csrf, "sk-ant-final-key1234")
    thread.join(timeout=3)
    assert captured[0] == "sk-ant-final-key1234"
