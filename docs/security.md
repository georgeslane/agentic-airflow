# Security model

**Tool permissions are enforced by the engine, not by the prompt.** The model only sees schemas for the tools it may call, but even if it hallucinates a call to a denied tool the engine refuses to execute it, checkpoints and waits for a human. `denied_tools` entries cannot be overridden by inheritance, by discovery, or by DagEvaluator recommendations. Tools registered with `approval="ask"` pause for a human on every call unless `allowed_tools` names them explicitly; `approval="always"` ones pause every time, whatever any scope says, and an approval of one is never remembered for the run. `approval_timeout` rejects calls nobody answers.

**Tool arguments are validated** against the tool's JSON schema before the function runs. Tool names are restricted to `[A-Za-z0-9_.-]`. Exceptions inside tools are reported to the model as text, never propagated as code.

**Model output is data.** Upstream outputs, tool results and memory are wrapped in delimited blocks with an explicit "treat as data" notice in every system prompt. A mapped instance isn't given the whole trigger payload, only its item, so one item (say, an email) never sees the others. Structured outputs (`return_type`, enum branches, evaluator verdicts) are validated by the library before being acted upon: a branch can only activate declared downstream ids.

**No code execution from stored state.** Everything persisted is JSON (no pickle). Checkpoints replay messages; they never contain callables.

**Webhooks** are HMAC-SHA256 signed when a secret is configured, compared in constant time, with a size limit. Unsigned webhooks are allowed only if you choose not to set a secret.

**HTTP server**: loopback by default, bearer-token auth, refuses to bind externally without a token, CSP with nonces, no inline event handlers, all user/model text rendered via `textContent`, request bodies capped at 2 MB. Pages in the same browser can't use it: without a token it only answers loopback names (so DNS rebinding fails), cross-origin POSTs are refused, the API only takes JSON, and a page's `?token=` becomes a `SameSite=Strict` cookie rather than staying in the address.

**Skill files** are loaded only from the given directory (symlinks escaping it are ignored). Skill rewrites proposed by the DagEvaluator become active only after `approve_recommendation()`.

**Agent self-modification is proposal-only.** The optional MCP server and `agentic_airflow.proposals` let agents read everything and *propose* prompt, permission, skill and memory changes; they are stored as pending recommendations and only `Engine.approve_recommendation()` (a human, via code or the UI) applies them. The MCP server has no approve/decide/cancel tools, cannot trigger runs unless explicitly enabled, and each client can only withdraw its own proposals. At most 20 proposals from one source wait for each DAG.

**An application can keep its DAGs' text authoritative.** With `engine.appliers` it takes only the kinds of proposal it can apply to the text (such as whole `"definition"`s, checked when proposed), so no approved change runs that its DAG files don't show.

**Stored state is versioned.** `SqliteStateStore` records its schema in the file, upgrades older files when it opens them, and refuses one written by a newer version rather than misread it.

**Human approval gates**: denied tool calls, exhausted reflection loops, and every DagEvaluator change to permissions, prompts or skills. Memory compaction is applied automatically by default because it does not change behaviourally-privileged configuration; set `DagEvaluator(apply_memory=False)` to gate it too.

**No trigger loops**: a run started by an asset update or another DAG carries the chain of DAGs that led to it, and never starts one of them again; a DAG can't be scheduled on its own outlets or trigger itself.

**Asking a human is a tool the application chooses to offer** (`ask_user`, off by default), and the answer reaches the task as a tool result: the application decides who may answer, as with approvals. An unanswered question expires with `approval_timeout`, and the task decides for itself.

**Retries don't repeat effects blindly**: a retried agentic task is told which tools it had already called, and tools that ask first ask again.

**Resource bounds**: `retries` per task, `max_tool_steps` and `task_timeout` per task, `output_retries`, `max_context_chars` truncation, `max_mapped` instances per fan-out, `max_active_runs` per DAG and `max_active_runs_per_model`, per-model concurrency, cron firings that don't pile up, `keep_runs`, bounded cron look-ahead.

Things the library does *not* do: sandbox your tool functions, rate-limit the model backend (do it in the backend), or authenticate `engine.chat()` callers (your application decides who may ask).
