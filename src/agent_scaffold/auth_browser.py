"""Browser-paste flow for `auth login`.

Opens the user's browser to a tiny local server, lets them paste their
Anthropic key into a form, returns the key to the CLI. CSRF-token-guarded
so a malicious page can't drive-by the listener.

Kept separate from ``auth.py`` so importing ``auth`` does not pull in
``http.server`` / ``socketserver`` / ``webbrowser`` (and so this module can
be tested in isolation).
"""

from __future__ import annotations

import html
import http.server
import logging
import secrets
import socket
import threading
import urllib.parse
import webbrowser
from dataclasses import dataclass

log = logging.getLogger(__name__)

ANTHROPIC_KEYS_URL = "https://console.anthropic.com/settings/keys"
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT_RANGE = range(53700, 53800)


_FORM_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>agent-scaffold — {label}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 640px;
         margin: 64px auto; padding: 0 16px; color: #222; }}
  textarea {{ width: 100%; min-height: 110px; font-family: ui-monospace, monospace;
              font-size: 14px; padding: 8px; box-sizing: border-box; }}
  button {{ margin-top: 12px; padding: 8px 18px; font-size: 14px; cursor: pointer; }}
  .hint {{ color: #666; font-size: 14px; }}
  code {{ background: #f3f3f3; padding: 1px 4px; border-radius: 3px; }}
</style></head>
<body>
  <h2>Paste your {label}</h2>
  <p class="hint">{hint_html}The {label} never leaves this machine — it goes from
    this form to your local <code>agent-scaffold</code> process and into your
    keychain (or a mode-0600 file).</p>
  <form method="POST" action="/submit">
    <input type="hidden" name="csrf" value="{csrf}">
    <textarea name="api_key" placeholder="{placeholder}" required autofocus></textarea>
    <br><button type="submit">Save</button>
  </form>
</body></html>
"""

_DONE_HTML = """<!doctype html>
<html><body style="font-family: system-ui, sans-serif; max-width: 560px;
margin: 80px auto; padding: 0 16px;">
<h2>Got it. You can close this tab.</h2>
<p>agent-scaffold is finishing the setup in your terminal.</p>
</body></html>
"""

_REJECTED_HTML = b"""<!doctype html>
<html><body style="font-family: system-ui, sans-serif;">
<h2>Rejected (bad CSRF token)</h2>
<p>This request did not originate from the form that agent-scaffold opened.</p>
</body></html>
"""


@dataclass
class _Capture:
    """Mutable container shared between the request handler and the CLI."""

    value: str | None = None
    event: threading.Event | None = None


def _pick_port() -> int:
    """First port in ``LOCAL_PORT_RANGE`` that ``bind()`` accepts.

    Falls back to letting the OS pick (port 0) if every fixed slot is taken,
    which sacrifices a stable bookmark URL but keeps the flow alive.
    """
    for port in LOCAL_PORT_RANGE:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((LOCAL_HOST, port))
            except OSError:
                continue
            return port
    return 0


def _build_handler(
    csrf: str,
    capture: _Capture,
    *,
    label: str,
    hint_html: str,
    placeholder: str,
) -> type[http.server.BaseHTTPRequestHandler]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:  # quiet stderr
            log.debug("auth-browser: " + fmt, *args)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if self.path == "/" or self.path.startswith("/?"):
                body = _FORM_HTML.format(
                    csrf=csrf, label=label, hint_html=hint_html, placeholder=placeholder
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            if self.path != "/submit":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            fields = urllib.parse.parse_qs(raw)
            posted_csrf = (fields.get("csrf") or [""])[0]
            posted_key = (fields.get("api_key") or [""])[0].strip()
            if not secrets.compare_digest(posted_csrf, csrf) or not posted_key:
                self.send_response(403)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_REJECTED_HTML)))
                self.end_headers()
                self.wfile.write(_REJECTED_HTML)
                return
            capture.value = posted_key
            done = _DONE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(done)))
            self.end_headers()
            self.wfile.write(done)
            if capture.event is not None:
                capture.event.set()

    return _Handler


def browser_available() -> bool:
    """True if a web browser can be launched (False when headless / no display).

    Lets a caller decide *before* spinning up the local server whether to use
    this flow or fall back to a terminal prompt. ``webbrowser.get()`` raises
    :class:`webbrowser.Error` when no usable browser is registered.
    """
    try:
        webbrowser.get()
    except webbrowser.Error:
        return False
    return True


def _build_hint_html(*, hint: str | None, hint_url: str | None) -> str:
    """The "where to get this credential" snippet for the form, HTML-escaped.

    ``hint_url`` renders as a clickable link (the Anthropic-key default);
    ``hint`` is a plain breadcrumb string (e.g. ``"smith.langchain.com → …"``).
    Returns ``""`` when neither is known so the form simply omits the line.
    """
    if hint_url:
        safe_url = html.escape(hint_url, quote=True)
        return (
            f'Need one? Open <a href="{safe_url}" target="_blank" '
            f'rel="noopener">{html.escape(hint_url)}</a> and copy it. '
        )
    if hint:
        return f"Where to get one: {html.escape(hint)}. "
    return ""


def browser_paste_flow(
    timeout_seconds: int = 300,
    *,
    label: str = "Anthropic API key",
    hint: str | None = None,
    hint_url: str | None = ANTHROPIC_KEYS_URL,
    placeholder: str = "sk-ant-...",
) -> str | None:
    """Open browser, capture a pasted credential, return it. ``None`` on timeout.

    ``label`` names the credential in the form (heading, title, privacy note);
    ``hint``/``hint_url`` tell the user where to obtain it. Defaults reproduce
    the Anthropic-key form used by ``auth login``.

    The HTTP server runs on a daemon thread; if the user closes the browser
    without submitting, the function returns ``None`` after ``timeout_seconds``
    and the daemon goes away with the process.
    """
    csrf = secrets.token_urlsafe(16)
    hint_html = _build_hint_html(hint=hint, hint_url=hint_url)
    port = _pick_port()
    capture = _Capture(event=threading.Event())
    handler_cls = _build_handler(
        csrf,
        capture,
        label=html.escape(label),
        hint_html=hint_html,
        placeholder=html.escape(placeholder, quote=True),
    )
    server = http.server.HTTPServer((LOCAL_HOST, port), handler_cls)
    actual_port = server.server_address[1]
    url = f"http://{LOCAL_HOST}:{actual_port}/"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    opened = False
    try:
        opened = webbrowser.open(url)
    except webbrowser.Error as exc:
        log.debug("webbrowser.open failed: %s", exc)
    if not opened:
        # Headless / no display — let the caller print the URL and the user
        # can open it from another machine on localhost-forwarded SSH.
        log.info("Open this URL in a browser: %s", url)
        print(f"Open this URL in a browser to paste your {label}:\n  {url}")

    try:
        assert capture.event is not None
        capture.event.wait(timeout=timeout_seconds)
    finally:
        server.shutdown()
        server.server_close()

    return capture.value


__all__ = [
    "ANTHROPIC_KEYS_URL",
    "LOCAL_PORT_RANGE",
    "browser_available",
    "browser_paste_flow",
]
