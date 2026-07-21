# Interactive shell

`agent-scaffold scaffold` (or bare `scaffold`) opens a persistent REPL. Make selections with slash commands, refine the plan with free text, see the cost estimate, generate when you're ready, then start the next project — all without re-launching the CLI.

```
scaffold › /recipe restaurant-rebooking
scaffold › /language python
scaffold › /framework langgraph
scaffold › /name demo
scaffold › /plan
  [renders the generation plan with token + cost estimate]
scaffold › swap to sonnet and skip the smoke test
  ✓ applied refinement
  Δ model: claude-opus-4-8 → claude-sonnet-5
  Δ steps: -smoke_test
scaffold › /generate
  [runs the generation pipeline]
scaffold › /exit
```

Type `/help` inside the shell for the full command list, or see the [REPL reference](../reference/repl.md).

## Free-text refinement

Free-text input ("use Sonnet, add Redis") is interpreted by a tiny Haiku call (~$0.002) into a typed patch over the plan. Run `/help refine` for the full list of accepted refinement keys — they're also listed in the [REPL reference](../reference/repl.md#free-text-refinement-keys).

Command and slug completion is fuzzy — `/observ` + Tab reaches `/observability` and a mistyped `/genrate` still suggests `/generate`. Unknown commands and capability ids get a "did you mean" hint, and a partial `/stack <query>` or `/recipe <query>` filters to matching rows (`/stack qdr` narrows to `vector_db.qdrant`).

## The /new wizard

The `/new` wizard walks the mandatory picks first (recipe, language, framework, name, destination), then one optional-features menu: RAG, Observability, Guardrails, More layers. Only the features you check get their own step — Enter with nothing checked goes straight to the plan.

The RAG step offers `simple` (single-stage retrieval on pgvector plus embeddings) or `complex` (hybrid search plus reranking), expanded from the catalog's published bundles; `custom` opens the full layer walk. The observability step asks where the backend runs when it supports both modes (`/observability langfuse cloud` mirrors it) — cloud keeps the capability but drops its compose service and wires the endpoint by credentials. The same presets work non-interactively:

```bash
agent-scaffold new --bundle rag-simple --obs-hosting langfuse=cloud
```

## Browsing the catalog

While composing, `/stack` browses the entire capability catalog grouped by layer — delivery (docker, cloud hosted, or docker with a cloud override), cost tier, and provisioning time per option, with your current picks marked. `/stack <id>` shows a detail card (description, env vars, connect handle); `/layer <layer> <ids...>` applies picks across memory, infrastructure, tools, observability, eval, interface, hosting, and auth.

## After generation

The shell stays useful: `/up` brings the stack up, `/status` checks readiness, `/connect <option>` wires a cloud hosted integration (LangSmith, managed Redis/Postgres), and `/down` tears the stack back down.

## Resume work

Selections autosave to a named draft as you go (at most 3 are kept; `/draft list` lists them, `/draft load <name>` resumes one). Once a project generates, its draft is retired — from then on `/open <dir>` (alias `/load`), or launching with `scaffold <dir>`, attaches the shell to the generated project so `/up`, `/connect`, and `/status` work on it. Loading a draft whose destination was already generated attaches to the project instead of rehydrating the stale selections.

## Startup sync

Each shell launch checks GitHub for newer deployments/blueprints content before the banner (the banner label reads "up to date" or "updated"); pass `--no-sync` to skip the check and start from the cache. If the check fails, the banner says so and the shell serves the cached trees — `/sync` retries in-session. See [Content sources](../guides/content-sources.md) for the full resolution and cache story.
