# Architecture

```
agentic_airflow/
  operators.py   Task, Branch, AgenticTask, AgenticBranch, AgenticEvaluator (+ >> syntax)
  dag.py         Dag, TaskGroup (context managers, validation, topological order)
  config.py      ScopeConfig / EngineConfig / resolve_config (System→Dag→Group→Task cascade)
  tools.py       Tool, ToolRegistry, three-state permissions, search_tools
  skills.py      SkillStore (versioned prompt fragments)
  triggers.py    CronTrigger (own 5-field parser), DagTrigger, WebhookTrigger (HMAC)
  models.py      GenerationRequest/Response, Message, ToolCall, OutputSpec, ModelBackend protocol, ScriptedBackend
  validation.py  mini JSON-schema validator + return_type coercion
  run.py         DagRun, TaskInstance, statuses, RunLogEntry (dag_context), Approval, Event
  priority.py    per-model priority queues (tier 0 user, tier 1 aging-FIFO DAG work)
  agent_loop.py  one agentic instance: prompt building, step loop, tool execution, checkpoints
  helpers.py     ready-made functions: trigger_items, found_nothing
  engine.py      Engine: registration, triggers, run driver, node expansion, HITL, chat, recommendations
  evaluator.py   DagEvaluator group
  proposals.py   human-gated proposals: making them, and applying each kind (engine.appliers)
  state.py       StateStore protocol, InMemoryStateStore, SqliteStateStore
  ui/            graph JSON + layout, self-contained HTML renderer, stdlib HTTP server
  backends/      OpenAI-compatible and Anthropic adapters
```

## Run lifecycle

1. `engine.trigger()` creates a `DagRun` (`PENDING`) with `memory = {**dag.context, **persisted_memory}` and the trigger payload, persists it and starts a driver task if the DAG's `max_active_runs` and the engine's `max_active_runs_per_model` allow. Waiting runs start first come, first served; a run counts against its DAG's model (the DAG-level resolved model) from when it starts until it finishes, paused or not. A run keeps the `Dag` object it started with, so `register(dag, replace=True)` never changes a run halfway; runs still waiting start with the new definition, and `unregister()` cancels them.
2. The driver loop: **expand** ready nodes → **launch** ready instances → **finish** nodes whose instances are all terminal (emit `node.finished`, update their outlets' assets) → wait for a wake-up (an instance finished, an approval was resolved) → repeat until every node is expanded and every instance is terminal.
3. A node is *expanded* once all its upstream nodes are expanded and all their instances are terminal. Expansion creates `TaskInstance`s (one per lane) with `depends_on` and `bindings` that describe how to assemble the input at launch time (`("output", inst)`, `("item", inst, i)`, `("fanin", [insts])`). Binding at launch time (not expansion) is what makes evaluator reruns see fresh outputs.
4. Instances are launched as asyncio tasks when their `depends_on` are terminal. `Task`/`Branch` run on the CPU pool; agentic instances run `agent_loop.run_agentic_instance`.
5. Terminal statuses: `SUCCESS`, `FAILED`, `SKIPPED`, `UPSTREAM_FAILED`. Waiting statuses: `AWAITING_TOOL_APPROVAL`, `AWAITING_USER`. When nothing is active and something is waiting, the run is `PAUSED` (and `wait()` returns); resolving the approval wakes the driver.
6. Final run status: `FAILED` if any instance of the DAG's own tasks failed (not the evaluator's), else `SUCCESS`; `CANCELLED` via `engine.cancel()`. DagTriggers of other DAGs fire; waiting runs start. The finished run leaves `engine.runs` (it's read back from the state store when asked for), and runs beyond `keep_runs` are deleted.

## Expansion algorithm (per node)

```
if trigger_rule == none_failed and any upstream instance failed → single UPSTREAM_FAILED instance
if skip_if(engine, run) gives a reason → single SKIPPED instance
lanes =
  if an upstream fan_out node produced lists  → one lane per (upstream instance, item): path = up.path + (i,)
  elif follow_mapping and an upstream is mapped → one lane per upstream lane path (deepest level)
  else → one lane with path ()
for each lane:
  for each upstream u: pick the instance of u whose map_path is the longest prefix of the lane path
     found   → bind ("output", it) [or ("item", it, idx) when u fans out and the lane is deeper]
     missing → bind ("fanin", successful instances of u)
  branch upstream whose picked instance did not select this node → lane SKIPPED
  all upstream lanes skipped (none_failed) → lane SKIPPED
no lanes → single SKIPPED placeholder ("empty fan-out")
instance id = node.id + "_" + "_".join(path)
```

## The agentic step loop

```
checkpoint = instance.checkpoint or {messages: [user prompt], tool_steps: 0, discovered: []}
acquire a slot on the model's queue (priority 1, key = run.created_at)      ┐ atomic="task":
loop:                                                                       │ the whole task
  if task_timeout has passed since the slot was acquired → fail             │
  response = backend.generate(request)   (atomic="step": slot held here only)
  if response.tool_calls:                                                   │
      for each call: allowed/hidden-discovered → execute;                   │
                     gated/denied → checkpoint, release, pause (AWAITING_TOOL_APPROVAL)
      tool_steps += 1; if tool_steps == max_tool_steps: ask for a final answer with no tools
  else:                                                                     │
      validate against return_type; on failure send feedback once; then succeed/fail
  checkpoint (persisted when persist_every_step)                            ┘
```

With `atomic="task"` (the default) a task is never interrupted: a live user request waits for it to finish, then goes before the next task. With `atomic="step"` the slot is held only while the backend generates, so users cut in between steps and tools run without holding the model. Either way a task waiting for approval holds nothing.

## Priority queue

`ModelQueueSet` keeps one `ModelQueue` per model id with `concurrency` slots (default 1 for local models). Waiters are ordered by `(priority, order_key, arrival)`. `engine.user_slot(model)` and `engine.chat()` acquire at priority 0 with key 0, so live user requests form a first-come heap that runs as soon as the current task (or step) finishes, before any DAG work. DAG tasks use priority 1 with the run's creation time as key, which implements "longest waiting DAG run first".

## Persistence

Every state change calls `state.save_run(run.to_dict())` (JSON). `state.delete_run()` deletes a run with its approvals (`keep_runs`). Approvals, recommendations, memory, overrides and skills have their own collections. `SqliteStateStore` keeps everything in one `documents` table (WAL mode). `resume()` reloads a run, turns `RUNNING/QUEUED` instances back into `PENDING` (they restart from their last checkpoint messages) and re-queues the run. `engine.stop()` detaches drivers without cancelling.

## DagEvaluator

`engine.register()` attaches the five-task group to the leaves (`trigger_rule="all_done"`, no tools, `include_run_log=True`). Each task's `on_complete` hook turns its JSON output into `Recommendation` documents (`kind` = `topology` | `tools` | `skills`, status `pending`), applies compacted memory (when `apply_memory=True`) and stores the executive summary on the run. `engine.approve_recommendation()` applies each with `engine.appliers[kind]` (`proposals.py`): tool overrides (`{"allow", "hide", "deny"}` per DAG, consulted by `resolve_permissions`), new skill versions, or prompt overrides (`overrides["prompts"][task_id]`, consulted when the prompt is built). An application can replace or remove these, and add its own kinds, such as `"definition"` for DAGs read from text. Structural changes are reported for manual follow-up because the DAG is defined in code.
