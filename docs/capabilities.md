# Capabilities

A capability is one concrete option from the deployments catalog — a Redis cache, a Qdrant vector store, a Langfuse tracer, a Llama Guard classifier. Recipes declare a starting set; you add, remove, and re-host them per project. Browse everything live with `/stack` in the shell, grouped exactly as below.

## Layers

The catalog groups capabilities into layers; `/layer <layer> <ids...>` applies picks per layer and `/stack <id>` shows any option's detail card (env vars, docker service, probe, cost tier, provisioning time).

| Layer | Covers | Examples |
| --- | --- | --- |
| memory | relational, cache, vector store, memory store | `relational.postgres`, `cache.redis`, `vector_db.qdrant` |
| infrastructure | queues, durable execution | `queue.redis-streams`, `durable.temporal` |
| tools | live data, MCP, embeddings, reranking, sandboxes, guardrails | `embedding.openai`, `rerank.cohere`, `guardrail.injection-classifier` |
| observability | tracing and metrics backends | `obs.langsmith`, `obs.langfuse`, `obs.grafana-stack` |
| eval | evaluation harnesses | `eval.promptfoo` |
| interface | frontends | the default chat frontend |
| hosting | deploy targets | `host.vercel`, `host.fly` |
| auth | authentication | project auth options |
| core | always-included primitives | `core.spec`, `core.tool_registry`, `core.step_log` |

## Tiers: T0 to T4

A tier is a curated capability floor for how serious the project is. Each tier strictly contains the one below it (T4 includes T3 includes T2, and so on). Set it with `/tier T2`, the wizard's tier step, or `--tier` on `new`.

| Tier | Title | Adds |
| --- | --- | --- |
| T0 | Chat | Owned editable prompts + schema-validated I/O (`core.spec`, `core.prompts`, `core.io_schema`). |
| T1 | Tool agent | A typed tool registry with permission tiers and compact-error retry. |
| T2 | Workflow | A serializable step-log as state: pause, resume, retry, trace. |
| T3 | Production | An eval seam seeded from the spec plus structured tracing. |
| T4 | Enterprise | Production plus opt-in overlays: multi-agent, human-in-the-loop, durable execution, guardrails, observability. |

## Bundles

Bundles are named shortcuts for common capability groups — the wizard's RAG and guardrails steps expand them, and `new --bundle <name>` does the same non-interactively.

| Bundle | What it wires |
| --- | --- |
| `rag-simple` | Single-stage retrieval: pgvector on the existing postgres, OpenAI embeddings, top-k cosine into the prompt. |
| `rag-complex` | Hybrid dense plus keyword retrieval on Qdrant with Cohere reranking between retrieval and the LLM. |
| `guardrails-basic` | Input and output classification with Llama Guard before and after the agent loop. |

## Delivery: docker, cloud, or both

Every capability has a delivery mode, shown in `/stack`:

- **docker** — runs as a service in the generated `docker-compose.yml`; `up` starts and health-checks it.
- **cloud hosted** — a managed service (LangSmith, managed Redis or Postgres); nothing runs locally, `/connect <option>` captures and validates the credentials after generation.
- **docker + cloud override** — runs in docker by default, and a hosting override (`/observability langfuse cloud`, `--obs-hosting langfuse=cloud`, or a free-text refinement) keeps the capability but drops its container and wires the endpoint by credentials instead.

## How picks become code

The resolved stack is not a label — it changes what gets generated. Selected capabilities pull their stack docs into the [context](ecosystem.md#how-the-arms-funnel-into-plan-and-generate), add their services to the compose file, declare their env vars in `.env.example`, register their provisioning steps with `up`, and appear in the project manifest so `update`, `status`, and `connect` know what the project runs on. The `/plan` panel shows the final stack with delivery modes before you spend a token.
