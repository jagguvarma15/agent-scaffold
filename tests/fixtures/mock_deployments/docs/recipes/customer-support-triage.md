---
status: validated
languages:
  - python
  - typescript
---

# Customer Support Triage

A small triage agent that routes incoming customer messages to specialist sub-agents.

## Composes

This recipe is built around the [ReAct pattern](../patterns/react.md) and uses
the [LangGraph framework](../frameworks/langgraph.md) for orchestration. Storage
is backed by [Qdrant](../stack/qdrant.md).

## Cross-cutting

This recipe also pulls in auth and logging concerns.

## System prompt

You are a customer support triage agent. Be brief and route accurately.
