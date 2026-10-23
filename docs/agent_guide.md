# Agent guide: integrating with agentic-airflow

This page is written for an AI agent (or a developer in a hurry) that needs to use or extend the library. Read `README.md` first; this is the operational cheat-sheet.

## Mental model in five lines

1. A `Dag` is a static graph of operators; an `Engine` turns it into `DagRun`s made of `TaskInstance`s (mapped nodes have several).
2. `Task`/`Branch` are Python; `AgenticTask`/`AgenticBranch`/`AgenticEvaluator` call your `ModelBackend.generate()` once per inference step.
3. Data flows along `>>` edges (branches transparent); `fan_out` lists create lanes; joins get dicts; `follow_mapping` stays per-lane.
4. Permissions (`allowed/denied/hidden_tools`) are resolved per task and enforced by the engine; denied calls pause for approval.
5. Everything is persisted as JSON through a `StateStore`; `resume_interrupted()` continues after a crash.

## Minimal integration checklist

- [ ] Implement `async def generate(request: GenerationRequest) -> GenerationResponse` (one model call; return `tool_calls` or a final answer). See `docs/plugging_in.md`.
- [ ] Register tools in a `ToolRegistry` (`@tool`). Mark rarely-used ones `hidden=True`.
- [ ] Build the DAG inside `with Dag(...)`; set `model=` somewhere in the cascade (or `EngineConfig(model=...)`).
- [ ] `engine.register(dag)`; `async with engine:`; `await engine.run(dag_id)` or `engine.trigger()` + `engine.wait()`.
- [ ] Handle pauses: `engine.list_approvals()` → `approve/reject/decide`.
- [ ] Review DagEvaluator output: `run.summary`, `engine.list_recommendations()` → `approve_recommendation()`.
- [ ] For production: `SqliteStateStore`, `UIServer(token=...)`, `engine.resume_interrupted()` at startup.

## Where to look

| Need | File / symbol |
|---|---|
| Add a parameter to all operators | `operators.BaseOperator.__init__`, `config.ScopeConfig` if it cascades |
| Change how prompts are built | `agent_loop.build_system_prompt`, `build_user_prompt` |
| Change mapping/join semantics | `engine.Engine._expand_node` (see `docs/architecture.md` for the algorithm) |
| Add a new human decision type | `engine.Engine._create_approval` / `decide` |
| Add a trigger type | `triggers.py` + handling in `engine.Engine` (`_cron_loop`, `deliver_webhook`, `update_asset`, `_fire_dag_triggers`) |
| Change how runs queue | `engine.Engine._start_queued_runs` |
| Add persistence backend | implement `state.StateStore` (see `InMemoryStateStore`) |
| Add an API route | `ui/server.py::_Handler._api_get/_api_post` |
| Change the DagEvaluator prompts/schemas | `evaluator.py` |

## Invariants to preserve

* `generate()` is called with a complete conversation every time; never mutate `request.messages`.
* An instance id is `task_id` or `task_id_<i>[_<j>…]`; task ids are validated not to collide with these.
* Terminal statuses never change except through `reset_for_retry()` by an evaluator rerun or a user `retry` decision.
* A started run uses the `Dag` it started with (`engine.dag_for(run)`), never one registered since.
* `engine.runs` holds unfinished runs only; finished ones are read back from the state store.
* `denied` always wins; `run.approved_tools` only widens for the current run.
* Everything stored must be JSON-serialisable (`DagRun.to_dict()` is the contract).
* Tests in `tests/` cover each invariant; run `python -m pytest` after any change.

## Reading a run quickly

```python
run = engine.get_run(run_id)
{i.instance_id: (i.status.value, i.reason or i.error) for i in run.instances.values()}
run.dag_context          # structured log
run.outputs()            # successful outputs by instance id
engine.graph(dag_id, run_id)   # UI JSON
```
