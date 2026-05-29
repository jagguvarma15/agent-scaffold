"""``bootstrap_kafka`` step: create Kafka topics + Redis Streams consumer groups.

Despite the name, this step covers BOTH event-source capabilities that need
post-up bootstrap:

- ``queue.kafka``        → ``KafkaAdminClient.create_topics()``, idempotent
  via ``list_topics()`` first.
- ``queue.redis-streams`` → ``XGROUP CREATE`` per declared stream/group pair,
  swallowing ``BUSYGROUP`` errors.

Recipe drives the topic + stream list via two manifest answer fields
(populated at generation time, falls back to empty if absent):

- ``manifest.answers["kafka_topics"]``  — JSON list of
  ``{name, partitions, replication_factor}`` objects.
- ``manifest.answers["redis_streams"]`` — JSON list of
  ``{name, consumer_group}`` objects.

Skips cleanly when neither capability is on the recipe.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from agent_scaffold.orchestrator import (
    DetectionResult,
    StepContext,
    StepLog,
    StepProgress,
    StepResult,
    StepStatus,
    compute_fingerprint,
)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_PARTITIONS = 1
_DEFAULT_REPLICATION = 1


@dataclass
class BootstrapKafkaStep:
    """Create Kafka topics + Redis Stream consumer groups declared by the recipe."""

    id: str = "bootstrap_kafka"
    description: str = "Create Kafka topics / Redis Stream consumer groups"
    depends_on: tuple[str, ...] = ("docker_up",)
    timeout: float = _DEFAULT_TIMEOUT
    troubleshoot: dict[str, str] = field(
        default_factory=lambda: {
            "NoBrokersAvailable": (
                "kafka broker unreachable — `agent-scaffold up --retry docker_up`"
            ),
            "kafka-python": (
                'install the kafka extra: pip install "agent-scaffold-cli[kafka]"'
            ),
            "BUSYGROUP": (
                "consumer group already exists — re-run is safe (group reused)"
            ),
        }
    )

    # ---- detection ----------------------------------------------------

    def detect(self, ctx: StepContext) -> DetectionResult:
        kafka, streams = self._queue_capabilities(ctx)
        if not kafka and not streams:
            return DetectionResult(
                StepStatus.SKIPPED, reason="recipe declares no queue.* capability"
            )
        topics = _resolve_topics(ctx)
        stream_groups = _resolve_streams(ctx)
        if kafka and not topics:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="queue.kafka declared but no topics in manifest.answers",
            )
        if streams and not stream_groups:
            return DetectionResult(
                StepStatus.SKIPPED,
                reason="queue.redis-streams declared but no streams in manifest.answers",
            )
        return DetectionResult(
            StepStatus.PENDING,
            reason=f"create: {len(topics)} topic(s), {len(stream_groups)} stream group(s)",
        )

    # ---- apply --------------------------------------------------------

    def apply(self, ctx: StepContext) -> StepResult:
        kafka, streams = self._queue_capabilities(ctx)
        if not kafka and not streams:
            return StepResult(StepStatus.SKIPPED, detail="no queue.* capability")
        topics = _resolve_topics(ctx)
        stream_groups = _resolve_streams(ctx)
        summary: list[str] = []
        if kafka and topics:
            try:
                created = _create_kafka_topics(topics, ctx)
            except _BootstrapSkip as skip:
                summary.append(f"kafka skipped: {skip.reason}")
            except _BootstrapFail as fail:
                return StepResult(
                    StepStatus.FAILED, error=fail.reason, stderr_tail=fail.stderr_tail
                )
            else:
                summary.append(f"kafka: {created} topic(s) ensured")
        if streams and stream_groups:
            try:
                created = _create_stream_groups(stream_groups, ctx)
            except _BootstrapSkip as skip:
                summary.append(f"streams skipped: {skip.reason}")
            except _BootstrapFail as fail:
                return StepResult(
                    StepStatus.FAILED, error=fail.reason, stderr_tail=fail.stderr_tail
                )
            else:
                summary.append(f"redis-streams: {created} consumer group(s) ensured")
        if not summary:
            return StepResult(StepStatus.SKIPPED, detail="nothing to bootstrap")
        return StepResult(StepStatus.DONE, detail="; ".join(summary))

    # ---- fingerprint --------------------------------------------------

    def fingerprint(self, ctx: StepContext) -> str:
        kafka, streams = self._queue_capabilities(ctx)
        return compute_fingerprint(
            {
                "kafka": bool(kafka),
                "streams": bool(streams),
                "topics": _resolve_topics(ctx),
                "stream_groups": _resolve_streams(ctx),
            }
        )

    # ---- helpers ------------------------------------------------------

    def _queue_capabilities(self, ctx: StepContext) -> tuple[list[Any], list[Any]]:
        stack = ctx.resolved_stack
        if stack is None:
            return ([], [])
        queues = [c for c in stack.capabilities if c.kind == "queue"]
        kafka = [c for c in queues if c.id == "queue.kafka"]
        streams = [c for c in queues if c.id == "queue.redis-streams"]
        return (kafka, streams)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


class _BootstrapSkip(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BootstrapFail(Exception):
    def __init__(self, reason: str, stderr_tail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.stderr_tail = stderr_tail


def _resolve_topics(ctx: StepContext) -> list[dict[str, Any]]:
    raw = ctx.manifest.answers.get("kafka_topics", "") if ctx.manifest else ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.append(
            {
                "name": name,
                "partitions": int(entry.get("partitions") or _DEFAULT_PARTITIONS),
                "replication_factor": int(
                    entry.get("replication_factor") or _DEFAULT_REPLICATION
                ),
            }
        )
    return out


def _resolve_streams(ctx: StepContext) -> list[dict[str, str]]:
    raw = ctx.manifest.answers.get("redis_streams", "") if ctx.manifest else ""
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, str]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        group = entry.get("consumer_group")
        if not isinstance(name, str) or not isinstance(group, str):
            continue
        if not name or not group:
            continue
        out.append({"name": name, "consumer_group": group})
    return out


def _create_kafka_topics(topics: list[dict[str, Any]], ctx: StepContext) -> int:
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
    except ImportError as exc:
        raise _BootstrapSkip(
            'kafka-python not installed — pip install "agent-scaffold-cli[kafka]"'
        ) from exc
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    try:
        admin = KafkaAdminClient(
            bootstrap_servers=bootstrap,
            request_timeout_ms=int(_DEFAULT_TIMEOUT * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        raise _BootstrapFail(f"kafka: cannot connect to {bootstrap}: {exc}") from exc
    try:
        existing = set(admin.list_topics())
    except Exception as exc:  # noqa: BLE001
        admin.close()
        raise _BootstrapFail(f"kafka: list_topics failed: {exc}") from exc

    to_create = [
        NewTopic(
            name=t["name"],
            num_partitions=t["partitions"],
            replication_factor=t["replication_factor"],
        )
        for t in topics
        if t["name"] not in existing
    ]
    if not to_create:
        admin.close()
        return len(topics)
    try:
        admin.create_topics(to_create, validate_only=False)
    except TopicAlreadyExistsError:
        pass  # raced with another bootstrapper; ignore.
    except Exception as exc:  # noqa: BLE001
        admin.close()
        raise _BootstrapFail(
            f"kafka: create_topics failed: {exc}",
            stderr_tail=", ".join(t["name"] for t in to_create),
        ) from exc
    finally:
        admin.close()
    for t in to_create:
        ctx.emit(StepLog(step_id="bootstrap_kafka", line=f"kafka: created topic {t.name}"))
    return len(topics)


def _create_stream_groups(streams: list[dict[str, str]], ctx: StepContext) -> int:
    try:
        import redis
    except ImportError as exc:
        raise _BootstrapSkip(
            'redis-py not installed — pip install "agent-scaffold-cli[kafka]"'
        ) from exc
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        client = redis.Redis.from_url(url, socket_timeout=_DEFAULT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001
        raise _BootstrapFail(f"redis: cannot connect to {url}: {exc}") from exc
    created = 0
    for entry in streams:
        try:
            client.xgroup_create(
                name=entry["name"],
                groupname=entry["consumer_group"],
                id="$",
                mkstream=True,
            )
            created += 1
            ctx.emit(
                StepProgress(
                    step_id="bootstrap_kafka",
                    message=f"redis: created group {entry['consumer_group']} on {entry['name']}",
                )
            )
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" in str(exc):
                # Group already exists — that's the idempotent path.
                continue
            raise _BootstrapFail(
                f"redis xgroup_create({entry['name']!r}, {entry['consumer_group']!r}) failed: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise _BootstrapFail(
                f"redis xgroup_create({entry['name']!r}) failed: {exc}"
            ) from exc
    return len(streams)


__all__ = ["BootstrapKafkaStep"]
