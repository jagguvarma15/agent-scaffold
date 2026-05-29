"""Service probes for recipe-declared external dependencies.

Each probe returns a :class:`CheckResult` describing whether a service is
reachable. The contract is intentionally narrow:

- Probes never throw. Every exception path returns a ``CheckResult``.
- Probes honor a timeout (default 5s).
- Probes are dependency-light: Redis uses a raw socket; Postgres falls back
  to a TCP check if ``psycopg`` is not installed; Kafka does the same with
  ``kafka-python``. The scaffold's job is to flag service health, not embed
  a full client matrix.

The registry at the bottom maps the recipe-frontmatter ``probe`` value
(e.g. ``"redis_ping"``) to a callable.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from agent_scaffold.doctor import CheckResult, CheckStatus

if TYPE_CHECKING:
    from agent_scaffold.discovery import ExternalService

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0

ProbeCallable = Callable[["ExternalService", float], CheckResult]


# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """Resolved address for a service probe."""

    source: str  # which env var (or "default_local") supplied the value
    raw: str  # the original URL/host string


def resolve_endpoint(svc: ExternalService) -> Endpoint | None:
    """Return the first usable address for ``svc``.

    Order: env vars listed on the service (first non-empty wins) → ``default_local``.
    """
    for env_var in svc.env_vars:
        value = os.environ.get(env_var, "").strip()
        if value:
            return Endpoint(source=env_var, raw=value)
    if svc.default_local:
        return Endpoint(source="default_local", raw=svc.default_local)
    return None


def _hostport_from_url(raw: str, default_port: int) -> tuple[str, int]:
    """Extract ``(host, port)`` from a URL or bare ``host:port`` / ``host`` string."""
    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or "localhost"
        port = parsed.port or default_port
        return host, port
    if ":" in raw:
        host, _, port_text = raw.partition(":")
        try:
            return host or "localhost", int(port_text)
        except ValueError:
            return host or "localhost", default_port
    return raw or "localhost", default_port


def _result(
    svc: ExternalService,
    status: CheckStatus,
    title: str,
    detail: str = "",
    fix_hint: str = "",
) -> CheckResult:
    explain_hint = f"agent-scaffold doctor --explain {svc.explain}" if svc.explain else ""
    if explain_hint and not fix_hint:
        fix_hint = explain_hint
    return CheckResult(
        id=f"service.{svc.id}",
        category="Recipe services",
        status=status,
        title=title,
        detail=detail,
        fix_hint=fix_hint,
        explain_topic=svc.explain,
    )


def _no_address(svc: ExternalService) -> CheckResult:
    env_hint = ", ".join(svc.env_vars) if svc.env_vars else "(no env vars declared)"
    return _result(
        svc,
        CheckStatus.SKIP if not svc.required else CheckStatus.FAIL,
        f"{svc.id}: no address resolvable",
        detail=f"set one of: {env_hint}",
        fix_hint=f"export {svc.env_vars[0]}=..." if svc.env_vars else "",
    )


# ---------------------------------------------------------------------------
# Anthropic — uses Q2's auth.load_key() resolution chain
# ---------------------------------------------------------------------------


def probe_anthropic_list_models(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe Anthropic by calling ``models.list(limit=1)`` with a short timeout."""
    try:
        from agent_scaffold.auth import load_key
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "anthropic: auth module unavailable",
            detail="install agent-scaffold with auth extras",
        )

    secret = load_key()
    if secret is None:
        return _result(
            svc,
            CheckStatus.FAIL,
            "anthropic: no API key resolvable",
            detail="checked ANTHROPIC_API_KEY, keyring, credentials file",
            fix_hint="agent-scaffold auth login",
        )

    try:
        import anthropic
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "anthropic: SDK not installed",
            detail="add `anthropic` to the project deps",
        )

    try:
        client = anthropic.Anthropic(api_key=secret.get_secret_value(), timeout=timeout)
        page = client.models.list(limit=1)
        count = len(list(page.data))
        return _result(
            svc,
            CheckStatus.OK,
            f"anthropic: API reachable ({count} model(s) visible)",
        )
    except anthropic.AuthenticationError:
        return _result(
            svc,
            CheckStatus.FAIL,
            "anthropic: 401 unauthorized",
            detail="key rejected; recreate in console or check workspace",
            fix_hint="agent-scaffold auth login",
        )
    except Exception as exc:  # noqa: BLE001 - we promise to never throw
        return _result(
            svc,
            CheckStatus.FAIL,
            "anthropic: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Redis — raw-socket PING; no client dependency
# ---------------------------------------------------------------------------


_REDIS_PING = b"*1\r\n$4\r\nPING\r\n"


def probe_redis_ping(svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> CheckResult:
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    host, port = _hostport_from_url(endpoint.raw, default_port=6379)
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.sendall(_REDIS_PING)
            reply = sock.recv(64)
    except (TimeoutError, OSError) as exc:
        return _result(
            svc,
            CheckStatus.FAIL,
            f"redis: connection failed ({host}:{port})",
            detail=str(exc),
            fix_hint="docker run -d -p 6379:6379 redis:7-alpine",
        )
    if reply.startswith(b"+PONG"):
        return _result(svc, CheckStatus.OK, f"redis: PING ok ({host}:{port})")
    # NOAUTH / WRONGPASS — server is up but credentials are wrong.
    if reply.startswith(b"-NOAUTH") or reply.startswith(b"-WRONGPASS"):
        return _result(
            svc,
            CheckStatus.FAIL,
            f"redis: auth required ({host}:{port})",
            detail=reply.decode("utf-8", errors="replace").strip(),
            fix_hint=f"set {svc.env_vars[0]} to include the password" if svc.env_vars else "",
        )
    return _result(
        svc,
        CheckStatus.FAIL,
        f"redis: unexpected response ({host}:{port})",
        detail=reply[:64].decode("utf-8", errors="replace").strip(),
    )


# ---------------------------------------------------------------------------
# Postgres — psycopg if available, else TCP-only
# ---------------------------------------------------------------------------


def probe_postgres_select_one(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)

    try:
        import psycopg
    except ImportError:
        psycopg = None

    if psycopg is None:
        # Fall back to a TCP-only check so we still surface "server is down".
        host, port = _hostport_from_url(endpoint.raw, default_port=5432)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except (TimeoutError, OSError) as exc:
            return _result(
                svc,
                CheckStatus.FAIL,
                f"postgres: TCP connect failed ({host}:{port})",
                detail=str(exc),
            )
        return _result(
            svc,
            CheckStatus.WARN,
            f"postgres: TCP-only ok ({host}:{port})",
            detail="psycopg not installed; cannot run SELECT 1 — install psycopg[binary] for the full probe",
        )

    try:
        with psycopg.connect(endpoint.raw, connect_timeout=int(timeout)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        if row and row[0] == 1:
            return _result(svc, CheckStatus.OK, "postgres: SELECT 1 ok")
        return _result(svc, CheckStatus.FAIL, "postgres: SELECT 1 returned unexpected row")
    except psycopg.OperationalError as exc:
        return _result(
            svc,
            CheckStatus.FAIL,
            "postgres: connection failed",
            detail=str(exc).splitlines()[0] if str(exc) else type(exc).__name__,
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "postgres: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Langfuse — HTTP /api/public/health
# ---------------------------------------------------------------------------


def probe_langfuse_health(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    base = endpoint.raw.rstrip("/")
    if not base.startswith("http"):
        base = f"https://{base}"
    url = f"{base}/api/public/health"
    try:
        import httpx
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "langfuse: httpx not available",
            detail="install httpx (ships transitively with anthropic)",
        )
    try:
        response = httpx.get(url, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return _result(
            svc,
            CheckStatus.FAIL,
            f"langfuse: cannot reach {base}",
            detail=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "langfuse: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if response.status_code == 200:
        return _result(svc, CheckStatus.OK, f"langfuse: {base} healthy")
    return _result(
        svc,
        CheckStatus.FAIL,
        f"langfuse: {base} returned {response.status_code}",
        detail=response.text[:200],
    )


# ---------------------------------------------------------------------------
# Kafka — kafka-python metadata if available, else TCP-only
# ---------------------------------------------------------------------------


def probe_kafka_metadata(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    host, port = _hostport_from_url(endpoint.raw, default_port=9092)

    try:
        from kafka import KafkaClient
    except ImportError:
        KafkaClient = None

    if KafkaClient is None:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except (TimeoutError, OSError) as exc:
            return _result(
                svc,
                CheckStatus.FAIL,
                f"kafka: TCP connect failed ({host}:{port})",
                detail=str(exc),
            )
        return _result(
            svc,
            CheckStatus.WARN,
            f"kafka: TCP-only ok ({host}:{port})",
            detail="kafka-python not installed; cannot fetch metadata — install kafka-python for the full probe",
        )

    try:
        client = KafkaClient(
            bootstrap_servers=f"{host}:{port}", request_timeout_ms=int(timeout * 1000)
        )
        client.close()
        return _result(svc, CheckStatus.OK, f"kafka: metadata ok ({host}:{port})")
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            f"kafka: probe failed ({host}:{port})",
            detail=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Qdrant — HTTP /collections
# ---------------------------------------------------------------------------


def probe_qdrant_collections(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe Qdrant via the REST collections endpoint.

    OK when the endpoint returns HTTP 200 with a ``result.collections`` list.
    FAIL on connection error or 4xx/5xx. Falls back to a TCP-only check when
    ``httpx`` isn't importable (it usually is via the anthropic SDK).
    """
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    base = endpoint.raw.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"
    url = f"{base}/collections"
    try:
        import httpx
    except ImportError:
        host, port = _hostport_from_url(endpoint.raw, default_port=6333)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except (TimeoutError, OSError) as exc:
            return _result(
                svc,
                CheckStatus.FAIL,
                f"qdrant: TCP connect failed ({host}:{port})",
                detail=str(exc),
            )
        return _result(
            svc,
            CheckStatus.WARN,
            f"qdrant: TCP-only ok ({host}:{port})",
            detail="httpx not installed; cannot fetch /collections",
        )
    headers: dict[str, str] = {}
    api_key = os.environ.get("QDRANT_API_KEY", "").strip()
    if api_key:
        headers["api-key"] = api_key
    try:
        response = httpx.get(url, timeout=timeout, headers=headers)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return _result(svc, CheckStatus.FAIL, f"qdrant: cannot reach {base}", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "qdrant: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if response.status_code != 200:
        return _result(
            svc,
            CheckStatus.FAIL,
            f"qdrant: {base} returned {response.status_code}",
            detail=response.text[:200],
        )
    try:
        body = response.json()
        collections = body.get("result", {}).get("collections", [])
        count = len(collections) if isinstance(collections, list) else 0
    except (ValueError, AttributeError, TypeError):
        return _result(
            svc,
            CheckStatus.WARN,
            f"qdrant: {base} ok (200) but body shape unexpected",
            detail=response.text[:120],
        )
    return _result(svc, CheckStatus.OK, f"qdrant: {count} collection(s)")


# ---------------------------------------------------------------------------
# Chroma — HTTP /api/v1/heartbeat
# ---------------------------------------------------------------------------


def probe_chroma_heartbeat(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe Chroma via the heartbeat endpoint.

    OK on HTTP 200 with the canonical ``nanosecond heartbeat`` JSON shape.
    """
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    base = endpoint.raw.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"
    url = f"{base}/api/v1/heartbeat"
    try:
        import httpx
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "chroma: httpx not available",
            detail="install httpx for the full probe",
        )
    try:
        response = httpx.get(url, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return _result(svc, CheckStatus.FAIL, f"chroma: cannot reach {base}", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "chroma: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if response.status_code == 200:
        return _result(svc, CheckStatus.OK, f"chroma: heartbeat ok ({base})")
    return _result(
        svc,
        CheckStatus.FAIL,
        f"chroma: {base} returned {response.status_code}",
        detail=response.text[:200],
    )


# ---------------------------------------------------------------------------
# Kafka topic list — extends kafka_metadata with topic enumeration
# ---------------------------------------------------------------------------


def probe_kafka_topic_list(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe Kafka by listing topics via ``KafkaAdminClient``.

    OK with the topic count in the title. Falls back to a TCP-only check
    when ``kafka-python`` isn't installed (same shape as :func:`probe_kafka_metadata`).
    """
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    host, port = _hostport_from_url(endpoint.raw, default_port=9092)
    try:
        from kafka import KafkaAdminClient
    except ImportError:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except (TimeoutError, OSError) as exc:
            return _result(
                svc,
                CheckStatus.FAIL,
                f"kafka: TCP connect failed ({host}:{port})",
                detail=str(exc),
            )
        return _result(
            svc,
            CheckStatus.WARN,
            f"kafka: TCP-only ok ({host}:{port})",
            detail="kafka-python not installed; cannot list topics",
        )
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=f"{host}:{port}",
            request_timeout_ms=int(timeout * 1000),
        )
        topics = admin.list_topics()
        admin.close()
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            f"kafka: list_topics failed ({host}:{port})",
            detail=f"{type(exc).__name__}: {exc}",
        )
    return _result(svc, CheckStatus.OK, f"kafka: {len(topics)} topic(s)")


# ---------------------------------------------------------------------------
# Grafana — HTTP /api/health
# ---------------------------------------------------------------------------


def probe_grafana_health(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe Grafana via the health endpoint.

    OK on HTTP 200 with ``database: ok`` in the response body.
    """
    endpoint = resolve_endpoint(svc)
    if endpoint is None:
        return _no_address(svc)
    base = endpoint.raw.rstrip("/")
    if not base.startswith("http"):
        base = f"http://{base}"
    url = f"{base}/api/health"
    try:
        import httpx
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "grafana: httpx not available",
            detail="install httpx for the full probe",
        )
    try:
        response = httpx.get(url, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        return _result(svc, CheckStatus.FAIL, f"grafana: cannot reach {base}", detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "grafana: probe failed",
            detail=f"{type(exc).__name__}: {exc}",
        )
    if response.status_code != 200:
        return _result(
            svc,
            CheckStatus.FAIL,
            f"grafana: {base} returned {response.status_code}",
            detail=response.text[:200],
        )
    try:
        body = response.json()
        db_state = body.get("database", "unknown") if isinstance(body, dict) else "unknown"
    except ValueError:
        db_state = "unknown"
    if db_state != "ok":
        return _result(
            svc,
            CheckStatus.WARN,
            f"grafana: {base} reachable but database={db_state}",
        )
    return _result(svc, CheckStatus.OK, f"grafana: {base} healthy")


# ---------------------------------------------------------------------------
# LangSmith — workspace info via SDK
# ---------------------------------------------------------------------------


def probe_langsmith_workspace(
    svc: ExternalService, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CheckResult:
    """Probe LangSmith via ``Client.info()``.

    SKIP when ``langsmith`` SDK isn't installed (it's optional). Reads
    ``LANGCHAIN_API_KEY`` from env directly — LangSmith doesn't accept the
    keyring resolution chain.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        return _result(
            svc,
            CheckStatus.SKIP if not svc.required else CheckStatus.FAIL,
            "langsmith: LANGCHAIN_API_KEY not set",
            fix_hint="export LANGCHAIN_API_KEY=ls__... (get one at https://smith.langchain.com/settings)",
        )
    try:
        from langsmith import Client
    except ImportError:
        return _result(
            svc,
            CheckStatus.SKIP,
            "langsmith: SDK not installed",
            detail='install via the "obs" extra: pip install "agent-scaffold-cli[obs]"',
        )
    try:
        client = Client(api_key=api_key)
        # Client.info() returns workspace metadata; any non-empty response is
        # enough to confirm the key is valid.
        info = client.info()
    except Exception as exc:  # noqa: BLE001
        return _result(
            svc,
            CheckStatus.FAIL,
            "langsmith: workspace info failed",
            detail=f"{type(exc).__name__}: {exc}",
            fix_hint="rotate LANGCHAIN_API_KEY in the LangSmith dashboard",
        )
    _ = timeout  # SDK manages its own timeouts
    workspace_name = ""
    if info is not None:
        # Different SDK versions return dict or pydantic model; coerce gently.
        if isinstance(info, dict):
            workspace_name = str(
                info.get("tenant_handle", "") or info.get("workspace_name", "") or ""
            )
        else:
            workspace_name = str(
                getattr(info, "tenant_handle", "") or getattr(info, "workspace_name", "") or ""
            )
    title = (
        f"langsmith: workspace {workspace_name!r}" if workspace_name else "langsmith: workspace ok"
    )
    return _result(svc, CheckStatus.OK, title)


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


PROBES: dict[str, ProbeCallable] = {
    "anthropic_list_models": probe_anthropic_list_models,
    "redis_ping": probe_redis_ping,
    "postgres_select_one": probe_postgres_select_one,
    "langfuse_health": probe_langfuse_health,
    "kafka_metadata": probe_kafka_metadata,
    # Capability-driven probes
    "qdrant_collections": probe_qdrant_collections,
    "chroma_heartbeat": probe_chroma_heartbeat,
    "kafka_topic_list": probe_kafka_topic_list,
    "grafana_health": probe_grafana_health,
    "langsmith_workspace": probe_langsmith_workspace,
}


def run_probe(
    svc: ExternalService,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    skip: bool = False,
) -> CheckResult:
    """Dispatch ``svc`` to its registered probe. Returns a ``CheckResult`` always."""
    if skip:
        return _result(
            svc,
            CheckStatus.SKIP,
            f"{svc.id}: probes disabled",
            detail="--no-probes is set",
        )
    if not svc.probe:
        return _result(
            svc,
            CheckStatus.SKIP,
            f"{svc.id}: no probe configured",
            detail="recipe declares the service without a probe name",
        )
    probe = PROBES.get(svc.probe)
    if probe is None:
        return _result(
            svc,
            CheckStatus.SKIP,
            f"{svc.id}: unknown probe {svc.probe!r}",
            detail=f"known probes: {', '.join(sorted(PROBES))}",
        )
    try:
        return probe(svc, timeout)
    except Exception as exc:  # noqa: BLE001 - last-resort safety net
        log.exception("probe %s raised an unexpected exception", svc.probe)
        return _result(
            svc,
            CheckStatus.FAIL,
            f"{svc.id}: probe crashed",
            detail=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "PROBES",
    "Endpoint",
    "ProbeCallable",
    "probe_anthropic_list_models",
    "probe_kafka_metadata",
    "probe_langfuse_health",
    "probe_postgres_select_one",
    "probe_redis_ping",
    "resolve_endpoint",
    "run_probe",
]
