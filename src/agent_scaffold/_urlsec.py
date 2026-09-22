"""Hardened HTTP fetch seam shared by catalog.py and sources.py.

One function -- :func:`urlopen` -- wrapping a private OpenerDirector whose
redirect handler (a) refuses any redirect target that is not https and
(b) drops the Authorization header when a redirect changes host. urllib's
default handler re-sends Authorization cross-host and follows https->http
downgrades; both are wrong for catalog/tarball fetches, which enforce https
on the initial URL precisely because their payloads steer docker pins and
executed commands, and which may carry a GitHub token.

Deliberately not installed globally (``urllib.request.install_opener``):
local probes and bootstrap steps legitimately talk plain http to sandbox
containers and must keep default redirect behavior.

Tests monkeypatch ``agent_scaffold._urlsec.urlopen``.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import IO, Any

_MAX_ETAG_LENGTH = 256


def is_safe_etag(value: str) -> bool:
    """True when an ETag is safe to persist and re-send as a request header.

    ``http.client`` already rejects embedded CR/LF at send time; validating
    here keeps a poisoned cache file from hard-failing every later fetch.
    """
    if not value or len(value) > _MAX_ETAG_LENGTH:
        return False
    return all(0x20 <= ord(ch) <= 0x7E for ch in value)


class SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirects must stay on https, and credentials must not change host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        target = urllib.parse.urlparse(newurl)
        if target.scheme != "https":
            raise urllib.error.HTTPError(
                newurl, code, f"refusing redirect to non-https URL: {newurl}", headers, fp
            )
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            old_host = urllib.parse.urlparse(req.full_url).hostname
            if old_host != target.hostname:
                new_req.headers = {
                    k: v for k, v in new_req.headers.items() if k.lower() != "authorization"
                }
        return new_req


_OPENER = urllib.request.build_opener(SecureRedirectHandler())


def urlopen(req: urllib.request.Request | str, *, timeout: float) -> Any:
    """Drop-in for ``urllib.request.urlopen`` with hardened redirects."""
    return _OPENER.open(req, timeout=timeout)
