"""Tests for :mod:`agent_scaffold._urlsec`.

The redirect handler is exercised directly (no sockets): urllib calls
``redirect_request`` with the original request and the new URL, and the
return value (or raised HTTPError) decides what happens next.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from email.message import Message

import pytest

from agent_scaffold._urlsec import SecureRedirectHandler, is_safe_etag


def _redirect(original: urllib.request.Request, newurl: str) -> urllib.request.Request | None:
    handler = SecureRedirectHandler()
    return handler.redirect_request(original, io.BytesIO(b""), 302, "Found", Message(), newurl)


def _request_with_auth(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Bearer secret-token")
    req.add_header("Accept", "application/vnd.github+json")
    return req


def test_http_redirect_target_is_refused() -> None:
    req = _request_with_auth("https://api.github.com/repos/x/tarball")
    with pytest.raises(urllib.error.HTTPError, match="non-https"):
        _redirect(req, "http://api.github.com/elsewhere")


def test_cross_host_redirect_strips_authorization() -> None:
    req = _request_with_auth("https://api.github.com/repos/x/tarball")
    new_req = _redirect(req, "https://codeload.github.com/x/tar.gz/main")
    assert new_req is not None
    header_names = {k.lower() for k in new_req.headers}
    assert "authorization" not in header_names
    # Non-credential headers survive the hop.
    assert "accept" in header_names


def test_same_host_redirect_keeps_authorization() -> None:
    req = _request_with_auth("https://api.github.com/repos/x/tarball")
    new_req = _redirect(req, "https://api.github.com/repos/x/tarball/refs")
    assert new_req is not None
    assert any(k.lower() == "authorization" for k in new_req.headers)


def test_is_safe_etag_accepts_normal_shapes() -> None:
    assert is_safe_etag('"abc123"')
    assert is_safe_etag('W/"weak-etag"')


def test_is_safe_etag_rejects_hostile_shapes() -> None:
    assert not is_safe_etag("")
    assert not is_safe_etag('"x"\r\nX-Injected: 1')
    assert not is_safe_etag('"\x00"')
    assert not is_safe_etag("e" * 300)
