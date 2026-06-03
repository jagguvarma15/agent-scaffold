"""Tests for :meth:`ResolvedStack.by_kind` and :data:`LAYER_ORDER`.

The grouping preserves declaration order within each kind. ``LAYER_ORDER`` is
a stable presentation hint consumers use when rendering layers (wizard,
report). It must reference every currently-known kind exactly once.
"""

from __future__ import annotations

from pathlib import Path

from agent_scaffold.capabilities import (
    LAYER_ORDER,
    Capability,
    ResolvedStack,
    _KNOWN_KINDS,
)


def _cap(cap_id: str, kind: str) -> Capability:
    return Capability(id=cap_id, kind=kind, path=Path(f"/tmp/{cap_id}.md"))  # type: ignore[arg-type]


def test_by_kind_groups_by_kind_field() -> None:
    stack = ResolvedStack(
        capabilities=[
            _cap("relational.postgres", "relational"),
            _cap("vector_db.qdrant", "vector_db"),
            _cap("vector_db.pgvector", "vector_db"),
            _cap("obs.langfuse", "obs"),
        ]
    )
    groups = stack.by_kind()
    assert list(groups["vector_db"]) == [
        _cap("vector_db.qdrant", "vector_db"),
        _cap("vector_db.pgvector", "vector_db"),
    ]
    assert [c.id for c in groups["relational"]] == ["relational.postgres"]
    assert [c.id for c in groups["obs"]] == ["obs.langfuse"]


def test_by_kind_preserves_within_kind_declaration_order() -> None:
    # Two of the same kind declared back-to-back must retain their order.
    stack = ResolvedStack(
        capabilities=[
            _cap("eval.promptfoo", "eval"),
            _cap("eval.ragas", "eval"),
            _cap("eval.deepeval", "eval"),
        ]
    )
    assert [c.id for c in stack.by_kind()["eval"]] == [
        "eval.promptfoo",
        "eval.ragas",
        "eval.deepeval",
    ]


def test_by_kind_omits_kinds_with_no_capabilities() -> None:
    stack = ResolvedStack(capabilities=[_cap("cache.redis", "cache")])
    groups = stack.by_kind()
    assert set(groups) == {"cache"}


def test_by_kind_handles_empty_stack() -> None:
    assert ResolvedStack().by_kind() == {}


def test_layer_order_covers_every_known_kind_exactly_once() -> None:
    # Phase 3 will add `tools` here alongside the Literal extension — this
    # invariant is what enforces the two stay in sync.
    assert set(LAYER_ORDER) == _KNOWN_KINDS
    assert len(LAYER_ORDER) == len(set(LAYER_ORDER))


def test_layer_order_places_persistence_before_presentation() -> None:
    # Sanity check on the chosen reading order: storage layers first,
    # then signal layers, then surfaces.
    order = list(LAYER_ORDER)
    assert order.index("relational") < order.index("obs")
    assert order.index("vector_db") < order.index("frontend")
    assert order.index("obs") < order.index("host")
