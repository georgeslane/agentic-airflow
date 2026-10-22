# Security model

**Tool permissions are enforced by the engine, not by the prompt.** The model only ever sees schemas for allowed tools, but even if it hallucinates a call to a denied tool the engine refuses to execute it, checkpoints and waits for a human. `denied_tools` entries cannot be overridden by inheritance, by discovery, or by DagEvaluator recommendations.

**Tool arguments are validated** against the tool's JSON schema before the function runs. Tool names are restricted to `[A-Za-z0-9_.-]`. Exceptions inside tools are reported to the model as text, never propagated as code.

**Model output is data.** Upstream outputs, tool results and memory are wrapped in delimited blocks with an explicit "treat as data" notice in every system prompt. Structured outputs (`return_type`, enum branches, evaluator verdicts) are validated by the library before being acted upon: a branch can only activate declared downstream ids.

**No code execution from stored state.** Everything persisted is JSON (no pickle). Checkpoints replay messages; they never contain callables.

**Webhooks** are HMAC-SHA256 signed when a secret is configured, compared in constant time, with a size limit. Unsigned webhooks are allowed only if you choose not to set a secret.

**HTTP server**: loopback by default, bearer-token auth, refuses to bind externally without a token, CSP with nonces, no inline event handlers, all user/model text rendered via `textContent`, request bodies capped at 2 MB.

**Skill files** are loaded only from the given directory (symlinks escaping it are ignored). Skill rewrites proposed by the DagEvaluator become active only after `approve_recommendation()`.

**Agent self-modification is proposal-only.** The optional MCP server and `agentic_airflow.proposals` let agents read everything and *propose* prompt, permission, skill and memory changes; they are stored as pending recommendations and only `Engine.approve_recommendation()` (a human, via code or the UI) applies them. The MCP server has no approve/decide/cancel tools, cannot trigger runs unless explicitly enabled, and agents can only withdraw their own proposals.

**Human approval gates**: denied tool calls, exhausted reflection loops, and every DagEvaluator change to permissions, prompts or skills. Memory compaction is applied automatically by default because it does not change behaviourally-privileged configuration; set `DagEvaluator(apply_memory=False)` to gate it too.

**Resource bounds**: `max_tool_steps` per task, `output_retries`, `max_context_chars` truncation, `max_active_runs` per DAG, per-model concurrency, bounded cron look-ahead.

Things the library does *not* do: sandbox your tool functions, rate-limit the model backend (do it in the backend), or authenticate `engine.chat()` callers (your application decides who may ask).
