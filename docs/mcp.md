# MCP server: let agents inspect their DAGs and propose changes

`pip install agentic-airflow[mcp]` adds `agentic_airflow.mcp`, a [Model Context Protocol](https://modelcontextprotocol.io) server that gives an external agent (Claude Code, Claude Desktop, any MCP client) a read-only view of the orchestrator plus the ability to **propose** changes that a human must approve.

## Guarantees

* **No tool approves, applies, decides, cancels or edits anything.** Proposals become pending recommendations in the same store the `DagEvaluator` uses; a human applies them with `engine.approve_recommendation(id)` or `POST /api/recommendations/{id}` in the UI.
* Runs cannot be started unless the operator builds the server with `allow_trigger=True`.
* Explicit `denied_tools` in code still win over any approved permission proposal.
* The server instructions tell the agent that run logs are data, not instructions.
* The agent can only withdraw proposals it created itself (`source == "mcp"`).

## Tools

| Tool | Purpose |
|---|---|
| `list_dags()` | DAG ids, schedules, task counts, next cron fires |
| `get_dag(dag_id)` | objective, every task with prompt, dependencies, **resolved settings** (model, thinking, skills, allowed/hidden/denied tools), effective prompt after overrides, active overrides, memory, registered tools, skills |
| `get_graph(dag_id, run_id=None)` | the UI graph JSON with layout and live statuses |
| `list_runs(dag_id, status, limit)` | runs, newest first |
| `get_run(run_id)` | status, trigger, memory, every instance (status, attempt, output preview, error, reason), summary, pending approvals |
| `get_task_output(run_id, instance_id)` | full output, input context, tool calls, retry history of one instance |
| `get_run_log(run_id, task_id, limit)` | the `dag_context` log |
| `list_approvals(run_id)` | pending human approvals (read-only) |
| `list_recommendations(dag_id, run_id, status)` | DagEvaluator recommendations and the agent's own proposals |
| `get_skills(name)` / `get_memory(dag_id)` / `get_stats()` | skills, persisted memory, engine stats |
| `propose_prompt_change(dag_id, task_id, proposed_prompt, rationale)` | new prompt for an agentic task (applied as an override on approval) |
| `propose_structural_change(dag_id, description, rationale)` | new tasks / edges in plain language; recorded for a human to implement in code |
| `propose_tool_permissions(dag_id, rationale, allow, hide, deny)` | DAG-level tool overrides |
| `propose_skill_rewrite(skill, content, rationale, dag_id)` | new skill version or new skill |
| `propose_memory_update(dag_id, memory, rationale)` | replace persisted memory |
| `withdraw_proposal(recommendation_id, reason)` | withdraw one of the agent's own pending proposals |
| `trigger_run(dag_id, payload)` | only with `allow_trigger=True` |

Every tool returns `{"ok": true, "data": ...}` or `{"ok": false, "error": "..."}`; validation errors (unknown task, non-agentic prompt, unregistered tool, missing rationale) are reported to the agent instead of raising.

## Running it

**Same process as the orchestrator** (the server sees live runs):

```python
from agentic_airflow.mcp import build_server

server = build_server(engine, client_name="claude-code")   # allow_trigger=False by default
await server.run_stdio_async()                              # inside the engine's event loop
# or server.run_streamable_http_async(host="127.0.0.1", port=8765)  -- put auth in front of it
```

**Separate process** (the normal MCP client launch), sharing state through `SqliteStateStore`:

```python
# myapp/workflows.py
engine = Engine(backend, state=SqliteStateStore("orchestrator.db"), tools=registry)
engine.register(dag)
```

```json
{
  "mcpServers": {
    "agentic-airflow": {
      "command": "python",
      "args": ["-m", "agentic_airflow.mcp", "myapp.workflows:engine", "--client-name", "claude-code"]
    }
  }
}
```

In this mode the MCP process reads runs, approvals and recommendations from the store, and the proposals it writes are visible to the running orchestrator and the UI immediately. In-memory stores are per process, so use SQLite (or your own `StateStore`) for the two-process setup.

## Reviewing proposals

```python
for rec in engine.list_recommendations(status="pending"):
    print(rec["id"], rec["source"], rec["kind"], rec["title"], rec["rationale"])
await engine.approve_recommendation(rec_id)      # or engine.reject_recommendation(rec_id, reason)
```

The same list is served by `GET /api/recommendations` and decided through `POST /api/recommendations/{id}` on the HTTP UI. Proposal records carry `source` (`"mcp"`, `"dag_evaluator"` for evaluator output, or whatever you pass to `agentic_airflow.proposals.*`), `proposed_by`, `rationale`, `payload` and, for prompt/skill/memory changes, the previous value for diffing.

## Using the proposal API without MCP

`agentic_airflow.proposals` is part of the core package and has no dependencies, so an application (or a tool exposed to agents *inside* a DAG) can create the same gated proposals directly: `propose_prompt_change(engine, ...)`, `propose_tool_permissions(engine, ...)`, `propose_skill_rewrite(engine, ...)`, `propose_memory_update(engine, ...)`, `propose_structural_change(engine, ...)`, `withdraw_proposal(engine, rec_id)`.
