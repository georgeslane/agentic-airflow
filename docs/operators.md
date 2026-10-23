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
| `trigger_rule` | `"none_failed"` \| `"all_done"` | `"none_failed"` | `none_failed`: skip when all upstream lanes were skipped, fail (`UPSTREAM_FAILED`) when any upstream failed; with `follow_mapping`, a failed lane of a mapped upstream fails only the lane that follows it. `all_done`: run regardless (used by the DagEvaluator; missing inputs are simply absent). |
| `on_complete` | callable `(engine, run, instance)` | `None` | Hook after success (sync or async), awaited before the instance counts as done. |
| `outlets` | list of `Asset` or asset names | `[]` | Assets the node updates once all its instances have finished, if it produced something: DAGs with `Asset(name)` in their `schedule` run with what it produced (see [Assets](#assets-outlets)). Not allowed on branches. |
| `skip_if` | callable `(engine, run) -> str \| None` | `None` | Checked when the node is about to be scheduled: a returned reason skips it (and, by the trigger rules, what depends only on it). |

### Data flow rules

1. A task's input is a dict keyed by upstream task id. `Task`/`Branch` functions receive those values as positional arguments in `context` order (missing/skipped upstreams are passed as `None`); agentic tasks get them rendered into the prompt as delimited data blocks.
2. **Branches are transparent.** A node downstream of a `Branch`/`AgenticBranch` receives what the branch received (the routed data), not the selected id. Use `context=[branch]` explicitly to receive the selection.
3. **Mapping (fan-out).** If an upstream node has `fan_out=True` and an instance returns a list of N items, the downstream node gets N instances `<id>_0 … <id>_N-1`, each receiving element `i` under the upstream's key. Nested fan-outs produce `<id>_i_j`. A non-list output from a `fan_out` node passes through unmapped.
4. **Fan-in (join).** A node downstream of a mapped node receives an index-ordered dict `{"draft_0": …, "draft_1": …}` (skipped instances omitted) unless it sets `follow_mapping=True`, in which case it is mapped 1:1 with the upstream lanes. Other upstreams are broadcast to every lane; an upstream that shares a lane prefix (e.g. a sibling mapped from the same source) is matched by lane.
5. **Empty fan-out** (`[]`) marks the downstream node and its exclusive descendants `SKIPPED`; a join that still has other successful inputs runs.
6. At most one direct upstream of a node may have `fan_out=True` (validated).
7. A fan-out list longer than `EngineConfig.max_mapped` (100) is cut short.
8. **The trigger payload** is rendered into the prompts of unmapped instances only: a mapped instance works on its item, so one item's lane never sees the others through the payload.

### Ready-made functions

* `trigger_items` (for `Task("items", fn=trigger_items, fan_out=True)`): maps the tasks after it over what came with whatever started the run: the payload if it's a list, a dict payload's `"items"`, nothing if there was none, otherwise the payload as one item.
* `found_nothing` (for `skip_if=`): skips a task when none of the DAG's own tasks found anything: every one that ran returned nothing (`None`, blank text, `[]`, `{}`), branches selected nothing, and nothing failed.

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
| `prompt` | required | A string, or (from application code) `fn(engine, run) -> str`, written when the task starts. |
| `validator` | `None` | `fn(value)`: checks the answer once it matches `return_type`; raising `OutputValidationError` sends its message back to the model (within `output_retries`); returning a value replaces the answer. |

Prompt composition: system = DAG objective + task guidelines + skills + tool guidance + output contract + data notice. User message = memory + trigger payload (unmapped instances only) + upstream outputs (delimited `<<<BEGIN id>>> … <<<END id>>>` blocks) + retry history + workflow definition + run log + branch choices, then **the task prompt last**: models follow instructions after long context best, and tasks given the same data share a prefix a model server can cache. Outputs longer than `EngineConfig.max_context_chars` are truncated with a marker; the run log gives each entry an equal share of it, one line each.

### Tool governance

Resolution per task (`tools.resolve_permissions`):

* `allowed_tools=None` inherits the parent's list; at the top, `None` means every registered tool. Names in any of the lists may be shell-style patterns (`"news_*"`).
* `denied_tools` accumulate across scopes and always win (also over human-approved `allow` overrides).
* A tool registered with `approval="ask"` is **gated** (offered to the model, but each call pauses for approval) unless some scope's `allowed_tools` (or an `allow` override) names it; `allowed_tools=None` doesn't count as naming it. One with `approval="always"` is gated whatever `allowed_tools` says, and approving it with `remember=True` doesn't allow it for the rest of the run.
* `hidden_tools` accumulate; a hidden tool is allowed but its schema is not injected until the agent calls the built-in `search_tools` tool. Tools registered with `hidden=True` are hidden everywhere by default.
* Anything registered but not allowed is **denied**: calling it serialises the conversation, puts the instance in `AWAITING_TOOL_APPROVAL`, releases the model slot and creates an `Approval`. `engine.approve(id, remember=True)` executes it (and allows it for the rest of the run); `engine.reject(id, reason)` injects a denial message so the agent adjusts.
* Unknown (unregistered) tool names produce an error message to the model, not a pause.
* Tool arguments are validated against the tool's JSON schema before execution.

## `AgenticBranch(id, prompt, *, multi_select=False, allow_none=False, **AgenticTask options)`

The downstream ids are passed to the backend as an `enum` output contract (`OutputSpec(kind="enum", choices=[...])`) and listed in the prompt. Returns one id, or a list when `multi_select=True`. An answer that isn't exactly a choice but names exactly one of them as a word (``I'd say `apply` ``) counts as that choice. Skipping semantics are identical to `Branch`.

`allow_none=True` adds the choice `"none"` (`NO_BRANCH`), which selects nothing, so every downstream task is skipped. A branch with one downstream task is then a yes or no: `arrived = AgenticBranch("arrived", prompt="Has the parcel come?", allow_none=True); arrived >> tell`. No downstream task may be called `none`.

## `AgenticEvaluator(id, prompt, *, target=None, max_retries=3, on_exhausted="pause_for_user", follow_mapping=True, **AgenticTask options)`

Output contract (enforced JSON schema): `{"rerun": bool, "reason": str}`.

* `target` defaults to the single upstream node (must be a direct upstream, not another evaluator). Tasks that use the target's output go after the evaluator, not after the target (validated): they'd otherwise run on the first output, before a rerun.
* `rerun=True` → the target instance is reset and re-executed with its **original** input plus `retry_history = [{"attempt", "output", "reason"}, …]`; the evaluator then runs again. Verdicts are recorded in `dag_context`.
* `rerun=False` → the evaluator's own output becomes the **target's validated output**, so downstream tasks see the artefact, not the critique.
* After `max_retries` reruns: `pass_latest` passes the last output, `fail_dag` fails the task (and the run), `pause_for_user` creates an `evaluator_exhausted` approval. Resolve it with `engine.decide(id, "pass" | "retry" | "fail")`.
* With a mapped target the evaluator is mapped 1:1 (`follow_mapping=True`), so each lane has its own loop.

## `TaskGroup(id, *, model, allowed_tools, denied_tools, hidden_tools, skills, inherit_skills, thinking, max_tool_steps)`

Scoped container and context manager. Nestable. A group built outside a DAG (e.g. the `DagEvaluator`) is attached with `group.attach(dag)`. As in Airflow, a group takes part in `>>` and `<<`: before something it stands for its last tasks (those with nothing after them in the group), after something for its first: `src >> each_email >> report`.

## `Dag(id, prompt, *, context=None, schedule=None, auto_evaluate=True, description="", tags=(), max_active_runs=1, **TaskGroup options + verbosity)`

* `prompt` is the global objective rendered into every agentic system prompt and audited by the DagEvaluator.
* `context` (dict or str) is static seed context merged with the persisted memory (`{**seed, **memory}`; a string seed is stored under `"seed"`).
* `schedule` accepts cron strings, `CronTrigger(expr, timezone)`, `DagTrigger(dag_id, on=("success",))`, `WebhookTrigger(name, secret=None, max_bytes=1_000_000)`.
* `auto_evaluate=True` attaches the DagEvaluator at `engine.register()` time (skipped with a warning if the engine has no backend). Pass `engine.register(dag, evaluator=DagEvaluator(model="judge"))` to customise it.
* `max_active_runs` queues additional runs of the same DAG.

### Assets (outlets)

```python
with Dag("finder", prompt="Find new jobs.", schedule=["@hourly"]):
    AgenticTask("jobs", prompt="List new jobs.", return_type="list", fan_out=False, outlets=[Asset("jobs")])

with Dag("applier", prompt="Apply for jobs.", schedule=[Asset("jobs")]):
    jobs = Task("jobs", fn=trigger_items, fan_out=True)
    jobs >> AgenticTask("apply", prompt="Apply for this job.")
```

Once all of a node's instances have finished, each of its `outlets` is updated with what it produced: its output, or for a mapped node the list of its instances' outputs (lists joined into one). A node that produced nothing (no success, or only `None`, blank text, `[]`, `{}`) updates nothing. The runs it starts have `trigger = {"source": "asset", "asset", "payload", "chain", "from_dag", "from_task", "from_run"}`. `engine.update_asset(name, payload)` updates one from outside.

`chain` lists the DAGs whose runs led to the update: a run never starts one of them again, so DAGs can't trigger each other in a circle (this holds for `DagTrigger` too). A DAG can't be scheduled on one of its own outlets.

## Configuration cascade (`config.py`)

`EngineConfig` (system scope) → `Dag` → `TaskGroup` chain → task. `EngineConfig` adds engine knobs: `model_concurrency: {model: n}`, `default_concurrency=1`, `cpu_workers=4`, `max_context_chars=60_000`, `output_retries=1`, `persist_every_step=True`, `atomic="task"` (or `"step"`), `max_active_runs_per_model=1`, `approval_timeout=None`, `task_timeout=None`, `max_mapped=100`, `keep_runs=None` (see `config.EngineConfig`). Inspect the result for any task with `engine.resolved_config(dag_id, task_id)`.
