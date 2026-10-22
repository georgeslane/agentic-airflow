# Operators & configuration reference

All operators are nodes of a `Dag`. Build them inside a `with Dag(...)` block (or pass `dag=` / `group=`) and wire dependencies with `>>` / `<<` and lists:

```python
read >> classify >> [reply, archive] >> summary   # Airflow syntax
summary << archive                               # reverse works too
```

## Common parameters (every operator)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | required | Unique within the DAG. Pattern `[A-Za-z][A-Za-z0-9_.:-]*`. Must not collide with mapped ids of another task (`draft_0`). |
| `context` | operator \| id \| list \| `None` | `None` | Which ancestors' outputs feed this node. `None` = direct upstream dependencies (branches are transparent, see below). `[]` = nothing. Explicit entries must be ancestors. |
| `fan_out` | `bool` | `False` (`True` for `AgenticTask(return_type="list")`) | If the node returns a `list`, each element spawns one mapped instance of every immediately downstream node. |
| `follow_mapping` | `bool` | `False` (`True` for `AgenticEvaluator`) | When an upstream node is mapped (but not fanning out further), run this node once per upstream instance instead of joining. |
| `trigger_rule` | `"none_failed"` \| `"all_done"` | `"none_failed"` | `none_failed`: skip when all upstream lanes were skipped, fail (`UPSTREAM_FAILED`) when any upstream failed. `all_done`: run regardless (used by the DagEvaluator; missing inputs are simply absent). |
| `on_complete` | callable `(engine, run, instance)` | `None` | Hook after success (sync or async). |

### Data flow rules

1. A task's input is a dict keyed by upstream task id. `Task`/`Branch` functions receive those values as positional arguments in `context` order (missing/skipped upstreams are passed as `None`); agentic tasks get them rendered into the prompt as delimited data blocks.
2. **Branches are transparent.** A node downstream of a `Branch`/`AgenticBranch` receives what the branch received (the routed data), not the selected id. Use `context=[branch]` explicitly to receive the selection.
3. **Mapping (fan-out).** If an upstream node has `fan_out=True` and an instance returns a list of N items, the downstream node gets N instances `<id>_0 … <id>_N-1`, each receiving element `i` under the upstream's key. Nested fan-outs produce `<id>_i_j`. A non-list output from a `fan_out` node passes through unmapped.
4. **Fan-in (join).** A node downstream of a mapped node receives an index-ordered dict `{"draft_0": …, "draft_1": …}` (skipped instances omitted) unless it sets `follow_mapping=True`, in which case it is mapped 1:1 with the upstream lanes. Other upstreams are broadcast to every lane; an upstream that shares a lane prefix (e.g. a sibling mapped from the same source) is matched by lane.
5. **Empty fan-out** (`[]`) marks the downstream node and its exclusive descendants `SKIPPED`; a join that still has other successful inputs runs.
6. At most one direct upstream of a node may have `fan_out=True` (validated).

### `ctx` injection for Python callables

If a `Task`/`Branch` function declares a `ctx` parameter (or `**kwargs`) it receives a `TaskContext` with `run_id`, `dag_id`, `task_id`, `instance_id`, `map_path`, `attempt`, `memory`, `seed`, `trigger` (`{"source", "payload", ...}`), `upstream`, `retry_history`, `run_log` (the `dag_context`) and `engine`.

## `Task(id, fn, *, context=None, fan_out=False, follow_mapping=False, trigger_rule=..., on_complete=None)`

Deterministic Python wrapper. `fn` may be sync (runs on the CPU worker pool, never blocking the event loop) or async.

## `Branch(id, fn, *, context=None, follow_mapping=False)`

`fn` returns:

* `bool` → requires exactly two downstream nodes declared as `[true_task, false_task]`;
* `str` → the id of the downstream task to activate;
* `list[str]` → several downstream ids;
* `None` → nothing selected (all downstream skipped).

Unselected downstream nodes and their *exclusive* descendants are `SKIPPED`. Join nodes run once all active (non-skipped) upstream lanes finish. Invalid selections fail the branch with a descriptive error.

## `AgenticTask(id, prompt, **options)`

| Parameter | Default | Notes |
|---|---|---|
| `model` | inherit | Model id handed to the backend. |
| `allowed_tools` / `denied_tools` / `hidden_tools` | inherit / additive / additive | See [tool governance](#tool-governance). |
| `skills` / `inherit_skills` | `None` / `True` | Skill names rendered into the system prompt. |
| `thinking` | inherit (`"off"`) | `"on"`/`"off"` → `GenerationRequest.thinking`. |
| `return_type` | `"string"` | `"string" \| "list" \| "json" \| "image" \| "file"`. Validated by `validation.coerce_output`; one corrective retry (configurable with `EngineConfig.output_retries`) before failing. |
| `json_schema` | `None` | Optional schema enforced for `json`/`list` outputs and passed to the backend for constrained decoding. |
| `fan_out` | `True` if `return_type="list"` | Set `False` to pass the whole list to one downstream instance. |
| `verbosity` | inherit (`0`) | `0` silent, `1` tool calls + status transitions to the chat sink, `2` plus reasoning and full responses. |
| `max_tool_steps` | inherit (`10`) | Cap on inference steps that request tools. When spent, tools are withdrawn and the model is asked for a final answer; a further tool request fails the task. |
| `include_run_log` / `include_dag_definition` | `False` | Render `dag_context` / the DAG definition into the prompt (used by the DagEvaluator). |
| `system_prompt` | `None` | Extra task-level guidelines appended to the DAG objective. |

Prompt composition: system = DAG objective + task guidelines + skills + tool guidance + output contract + data notice. User message = task prompt + memory + trigger payload + upstream outputs (delimited `<<<BEGIN id>>> … <<<END id>>>` blocks) + retry history. Outputs longer than `EngineConfig.max_context_chars` are truncated with a marker.

### Tool governance

Resolution per task (`tools.resolve_permissions`):

* `allowed_tools=None` inherits the parent's list; at the top, `None` means every registered tool.
* `denied_tools` accumulate across scopes and always win (also over human-approved `allow` overrides).
* `hidden_tools` accumulate; a hidden tool is allowed but its schema is not injected until the agent calls the built-in `search_tools` tool. Tools registered with `hidden=True` are hidden everywhere by default.
* Anything registered but not allowed is **denied**: calling it serialises the conversation, puts the instance in `AWAITING_TOOL_APPROVAL`, releases the model slot and creates an `Approval`. `engine.approve(id, remember=True)` executes it (and allows it for the rest of the run); `engine.reject(id, reason)` injects a denial message so the agent adjusts.
* Unknown (unregistered) tool names produce an error message to the model, not a pause.
* Tool arguments are validated against the tool's JSON schema before execution.

## `AgenticBranch(id, prompt, *, multi_select=False, **AgenticTask options)`

The downstream ids are passed to the backend as an `enum` output contract (`OutputSpec(kind="enum", choices=[...])`) and listed in the prompt. Returns one id, or a list when `multi_select=True`. Skipping semantics are identical to `Branch`.

## `AgenticEvaluator(id, prompt, *, target=None, max_retries=3, on_exhausted="pause_for_user", follow_mapping=True, **AgenticTask options)`

Output contract (enforced JSON schema): `{"rerun": bool, "reason": str}`.

* `target` defaults to the single upstream node (must be a direct upstream, not another evaluator).
* `rerun=True` → the target instance is reset and re-executed with its **original** input plus `retry_history = [{"attempt", "output", "reason"}, …]`; the evaluator then runs again. Verdicts are recorded in `dag_context`.
* `rerun=False` → the evaluator's own output becomes the **target's validated output**, so downstream tasks see the artefact, not the critique.
* After `max_retries` reruns: `pass_latest` passes the last output, `fail_dag` fails the task (and the run), `pause_for_user` creates an `evaluator_exhausted` approval. Resolve it with `engine.decide(id, "pass" | "retry" | "fail")`.
* With a mapped target the evaluator is mapped 1:1 (`follow_mapping=True`), so each lane has its own loop.

## `TaskGroup(id, *, model, allowed_tools, denied_tools, hidden_tools, skills, inherit_skills, thinking, max_tool_steps)`

Scoped container and context manager. Nestable. A group built outside a DAG (e.g. the `DagEvaluator`) is attached with `group.attach(dag)`.

## `Dag(id, prompt, *, context=None, schedule=None, auto_evaluate=True, description="", tags=(), max_active_runs=1, **TaskGroup options + verbosity)`

* `prompt` is the global objective rendered into every agentic system prompt and audited by the DagEvaluator.
* `context` (dict or str) is static seed context merged with the persisted memory (`{**seed, **memory}`; a string seed is stored under `"seed"`).
* `schedule` accepts cron strings, `CronTrigger(expr, timezone)`, `DagTrigger(dag_id, on=("success",))`, `WebhookTrigger(name, secret=None, max_bytes=1_000_000)`.
* `auto_evaluate=True` attaches the DagEvaluator at `engine.register()` time (skipped with a warning if the engine has no backend). Pass `engine.register(dag, evaluator=DagEvaluator(model="judge"))` to customise it.
* `max_active_runs` queues additional runs of the same DAG.

## Configuration cascade (`config.py`)

`EngineConfig` (system scope) → `Dag` → `TaskGroup` chain → task. `EngineConfig` adds engine knobs: `model_concurrency: {model: n}`, `default_concurrency=1`, `cpu_workers=4`, `max_context_chars=60_000`, `output_retries=1`, `persist_every_step=True`. Inspect the result for any task with `engine.resolved_config(dag_id, task_id)`.
