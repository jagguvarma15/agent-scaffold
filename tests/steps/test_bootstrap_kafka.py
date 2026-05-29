"""Tests for ``agent_scaffold.steps.bootstrap_kafka`` (Phase 2)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from agent_scaffold.capabilities import Capability, ResolvedStack
from agent_scaffold.manifest import Manifest
from agent_scaffold.orchestrator import StepContext, StepStatus
from agent_scaffold.steps.bootstrap_kafka import BootstrapKafkaStep


def _cap(id_: str, kind: str, tmp_path: Path) -> Capability:
    return Capability(id=id_, kind=kind, path=tmp_path / f"{id_}.md", env_vars=[])


def _manifest(answers: dict[str, str]) -> Manifest:
    return Manifest(
        recipe="r",
        language="python",
        framework="none",
        model="m",
        generated_at="2026-01-01T00:00:00+00:00",
        answers=answers,
    )


def test_detect_skipped_without_queue_capability(
    ctx_factory: Callable[..., StepContext],
) -> None:
    result = BootstrapKafkaStep().detect(ctx_factory())
    assert result.status is StepStatus.SKIPPED


def test_detect_skipped_when_kafka_declared_but_no_topics(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.kafka", "queue", tmp_path)])
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest({}))
    result = BootstrapKafkaStep().detect(ctx)
    assert result.status is StepStatus.SKIPPED
    assert "no topics" in result.reason


def test_detect_pending_with_topics(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.kafka", "queue", tmp_path)])
    answers = {"kafka_topics": '[{"name": "orders", "partitions": 3}]'}
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest(answers))
    result = BootstrapKafkaStep().detect(ctx)
    assert result.status is StepStatus.PENDING
    assert "1 topic" in result.reason


def test_apply_creates_missing_kafka_topics(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.kafka", "queue", tmp_path)])
    answers = {"kafka_topics": '[{"name": "orders", "partitions": 3, "replication_factor": 1}]'}
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest(answers))

    created: list[Any] = []

    class FakeAdmin:
        def __init__(self, **_kw: Any) -> None:
            pass

        def list_topics(self) -> list[str]:
            return []

        def create_topics(self, topics: list[Any], **_kw: Any) -> None:
            created.extend(topics)

        def close(self) -> None:
            pass

    class FakeNewTopic:
        def __init__(self, name: str, num_partitions: int, replication_factor: int) -> None:
            self.name = name
            self.num_partitions = num_partitions
            self.replication_factor = replication_factor

    fake_admin_mod = type(
        "M", (), {"KafkaAdminClient": FakeAdmin, "NewTopic": FakeNewTopic}
    )
    fake_errors = type("M", (), {"TopicAlreadyExistsError": type("E", (Exception,), {})})
    sys = __import__("sys")
    monkeypatch.setitem(sys.modules, "kafka", type("M", (), {}))
    monkeypatch.setitem(sys.modules, "kafka.admin", fake_admin_mod)
    monkeypatch.setitem(sys.modules, "kafka.errors", fake_errors)

    result = BootstrapKafkaStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert [t.name for t in created] == ["orders"]
    assert "1 topic" in result.detail


def test_apply_kafka_idempotent_when_topic_exists(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.kafka", "queue", tmp_path)])
    answers = {"kafka_topics": '[{"name": "orders"}]'}
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest(answers))

    create_calls: list[Any] = []

    class FakeAdmin:
        def __init__(self, **_kw: Any) -> None:
            pass

        def list_topics(self) -> list[str]:
            return ["orders"]

        def create_topics(self, topics: list[Any], **_kw: Any) -> None:
            create_calls.append(topics)

        def close(self) -> None:
            pass

    sys = __import__("sys")
    fake_admin_mod = type(
        "M", (), {"KafkaAdminClient": FakeAdmin, "NewTopic": lambda **kw: kw}
    )
    monkeypatch.setitem(sys.modules, "kafka", type("M", (), {}))
    monkeypatch.setitem(sys.modules, "kafka.admin", fake_admin_mod)
    monkeypatch.setitem(
        sys.modules, "kafka.errors", type("M", (), {"TopicAlreadyExistsError": type("E", (Exception,), {})})
    )

    result = BootstrapKafkaStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert create_calls == []  # no create call when topic exists


def test_apply_redis_streams_creates_consumer_groups(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.redis-streams", "queue", tmp_path)])
    answers = {
        "redis_streams": '[{"name": "events", "consumer_group": "workers"}]'
    }
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest(answers))
    xgroups: list[tuple[str, str]] = []

    class _ResponseError(Exception):
        pass

    class _Exceptions:
        ResponseError = _ResponseError

    class FakeRedis:
        @classmethod
        def from_url(cls, url: str, **_kw: Any) -> FakeRedis:
            return cls()

        def xgroup_create(self, **kw: Any) -> None:
            xgroups.append((kw["name"], kw["groupname"]))

    sys = __import__("sys")
    monkeypatch.setitem(
        sys.modules,
        "redis",
        type("M", (), {"Redis": FakeRedis, "exceptions": _Exceptions}),
    )
    result = BootstrapKafkaStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert xgroups == [("events", "workers")]


def test_apply_redis_streams_handles_busygroup(
    ctx_factory: Callable[..., StepContext],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.redis-streams", "queue", tmp_path)])
    answers = {"redis_streams": '[{"name": "events", "consumer_group": "g"}]'}
    ctx = ctx_factory(resolved_stack=stack, manifest=_manifest(answers))

    class _ResponseError(Exception):
        pass

    class _Exceptions:
        ResponseError = _ResponseError

    class FakeRedis:
        @classmethod
        def from_url(cls, url: str, **_kw: Any) -> FakeRedis:
            return cls()

        def xgroup_create(self, **_kw: Any) -> None:
            raise _ResponseError("BUSYGROUP consumer group name already exists")

    sys = __import__("sys")
    monkeypatch.setitem(
        sys.modules,
        "redis",
        type("M", (), {"Redis": FakeRedis, "exceptions": _Exceptions}),
    )
    result = BootstrapKafkaStep().apply(ctx)
    assert result.status is StepStatus.DONE
    assert "1 consumer group(s)" in result.detail


def test_fingerprint_changes_when_topic_set_changes(
    ctx_factory: Callable[..., StepContext], tmp_path: Path
) -> None:
    stack = ResolvedStack(capabilities=[_cap("queue.kafka", "queue", tmp_path)])
    a = ctx_factory(resolved_stack=stack, manifest=_manifest({"kafka_topics": '[{"name": "a"}]'}))
    b = ctx_factory(resolved_stack=stack, manifest=_manifest({"kafka_topics": '[{"name": "b"}]'}))
    step = BootstrapKafkaStep()
    assert step.fingerprint(a) != step.fingerprint(b)
