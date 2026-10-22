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
  engine.py      Engine: registration, triggers, run driver, node expansion, HITL, chat, recommendations
  evaluator.py   DagEvaluator group + recommendation application
  state.py       StateStore protocol, InMemoryStateStore, SqliteStateStore
  ui/            graph JSON + layout, self-contained HTML renderer, stdlib HTTP server
  backends/      OpenAI-compatible and Anthropic adapters
```

## Run lifecycle

1. `engine.trigger()` creates a `DagRun` (`PENDING`) with `memory = {**dag.context, **persisted_memory}` and the trigger payload, persists it and starts a driver task if `max_active_runs` allows.
2. The driver loop: **expand** ready nodes → **launch** ready instances → wait for a wake-up (an instance finished, an approval was resolved) → repeat until every node is expanded and every instance is terminal.
3. A node is *expanded* once all its upstream nodes are expanded and all their instances are terminal. Expansion creates `TaskInstance`s (one per lane) with `depends_on` and `bindings` that describe how to assemble the input at launch time (`("output", inst)`, `("item", inst, i)`, `("fanin", [insts])`). Binding at launch time (not expansion) is what makes evaluator reruns see fresh outputs.
4. Instances are launched as asyncio tasks when their `depends_on` are terminal. `Task`/`Branch` run on the CPU pool; agentic instances run `agent_loop.run_agentic_instance`.
5. Terminal statuses: `SUCCESS`, `FAILED`, `SKIPPED`, `UPSTREAM_FAILED`. Waiting statuses: `AWAITING_TOOL_APPROVAL`, `AWAITING_USER`. When nothing is active and something is waiting, the run is `PAUSED` (and `wait()` returns); resolving the approval wakes the driver.
6. Final run status: `FAILED` if any instance failed, else `SUCCESS`; `CANCELLED` via `engine.cancel()`. DagTriggers of other DAGs fire; queued runs of the same DAG start.

## Expansion algorithm (per node)

```
if trigger_rule == none_failed and any upstream instance failed → single UPSTREAM_FAILED instance
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
loop:
  acquire a slot on the model's queue (priority 1, key = run.created_at)      ┐ one atomic
  response = backend.generate(request)                                        │ inference
  release slot                                                                ┘ step
  if response.tool_calls:
      for each call: allowed/hidden-discovered → execute; denied → checkpoint, pause (AWAITING_TOOL_APPROVAL)
      tool_steps += 1; if tool_steps == max_tool_steps: ask for a final answer with no tools
  else:
      validate against return_type; on failure send feedback once; then succeed/fail
  checkpoint (persisted when persist_every_step)
```

Tool execution happens *after* the model slot is released, so a slow tool never blocks other tasks or user chat waiting for the model; interruption still only happens at step boundaries, which keeps the conversation state consistent.

## Priority queue

`ModelQueueSet` keeps one `ModelQueue` per model id with `concurrency` slots (default 1 for local models). Waiters are ordered by `(priority, order_key, arrival)`. `engine.chat()` acquires at priority 0 with key 0, so it runs right after the current step finishes. DAG steps use priority 1 with the run's creation time as key, which implements "longest waiting DAG run first".

## Persistence

Every state change calls `state.save_run(run.to_dict())` (JSON). Approvals, recommendations, memory, overrides and skills have their own collections. `SqliteStateStore` keeps everything in one `documents` table (WAL mode). `resume()` reloads a run, turns `RUNNING/QUEUED` instances back into `PENDING` (they restart from their last checkpoint messages) and re-queues the run. `engine.stop()` detaches drivers without cancelling.

## DagEvaluator

`engine.register()` attaches the five-task group to the leaves (`trigger_rule="all_done"`, no tools, `include_run_log=True`). Each task's `on_complete` hook turns its JSON output into `Recommendation` documents (`kind` = `topology` | `tools` | `skills`, status `pending`), applies compacted memory (when `apply_memory=True`) and stores the executive summary on the run. `engine.approve_recommendation()` applies: tool overrides (`{"allow", "hide", "deny"}` per DAG, consulted by `resolve_permissions`), new skill versions, or prompt overrides (`overrides["prompts"][task_id]`, consulted when the prompt is built). Structural changes are reported for manual follow-up because the DAG is defined in code.
