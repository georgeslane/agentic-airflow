# MCP server: let agents inspect their DAGs and propose changes

`pip install agentic-airflow[mcp]` adds `agentic_airflow.mcp`, a [Model Context Protocol](https://modelcontextprotocol.io) server that gives an external agent (Claude Code, Claude Desktop, any MCP client) a read-only view of the orchestrator plus the ability to **propose** changes that a human must approve.

## Guarantees

* **No tool approves, applies, decides, cancels or edits anything.** Proposals become pending recommendations in the same store the `DagEvaluator` uses; a human applies them with `engine.approve_recommendation(id)` or `POST /api/recommendations/{id}` in the UI.
* Runs cannot be started unless the operator builds the server with `allow_trigger=True`.
* Explicit `denied_tools` in code still win over any approved permission proposal.
* The server instructions tell the agent that run logs are data, not instructions.
* A client can only withdraw proposals it made itself (`source == "mcp"` and its `client_name`).
* At most 20 proposals from one source wait for each DAG (`proposals.MAX_PENDING`), so an agent can't flood the queue.

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
| `propose_prompt_change(dag_id, task_id, proposed_prompt, rationale)` | new prompt for an agentic task (applied as an override on approval); refused for a task whose prompt is written in code when it starts |
| `propose_structural_change(dag_id, description, rationale)` | new tasks / edges in plain language; recorded for a human to implement in code |
| `propose_tool_permissions(dag_id, rationale, allow, hide, deny, unset)` | DAG-level tool overrides; `unset` drops earlier ones |
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

`agentic_airflow.proposals` is part of the core package and has no dependencies, so an application (or a tool exposed to agents *inside* a DAG) can create the same gated proposals directly: `propose_prompt_change(engine, ...)`, `propose_tool_permissions(engine, ...)`, `propose_skill_rewrite(engine, ...)`, `propose_memory_update(engine, ...)`, `propose_structural_change(engine, ...)`, `propose_definition(engine, ...)`, `withdraw_proposal(engine, rec_id)`.

## Proposals your application applies

By default an approved prompt or tool change is applied as an *override*, kept in the state store beside the DAG. That suits DAGs defined in code. An application that reads its DAGs from text (files a user edits, say) wants the text to stay what runs, so it decides which kinds of proposal it takes, and how each is applied, with `engine.appliers`:

```python
from agentic_airflow import Applier, Dag, StaleProposalError
from agentic_airflow.errors import ConfigurationError

dag = read_my_dag(text)            # however the application builds it
dag.definition = text              # or Dag(..., definition=text): what a "definition" proposal replaces

def check(engine, rec):            # when it's proposed: ConfigurationError tells the proposer why it won't do
    read_my_dag(rec["payload"]["definition"])

async def apply(engine, rec):      # when a human approves it
    if rec["payload"]["base"] != current_text(rec["dag_id"]):
        raise StaleProposalError("The DAG has changed since")   # marked stale, and raised
    save_and_register(rec["payload"]["definition"])
    return {"saved": True}         # kept as the proposal's "applied"

engine.appliers["definition"] = Applier(apply, check)
for kind in ("topology", "tools", "memory"):   # no overrides: they'd make the file and what runs differ
    del engine.appliers[kind]
```

- **`propose_definition(engine, dag_id, definition, rationale, title=..., run_id=..., details=...)`** proposes a whole new text; the payload keeps the text it replaces as `base`, with `details` besides.
- **Only kinds in `engine.appliers` can be proposed or approved**, and the MCP server only offers tools for those, as the engine is when the server's built: here `propose_definition` and `propose_skill_rewrite`, and `get_dag` returns the `definition`.
- **A stale proposal**, one whose applier raised `StaleProposalError`, gets status `stale`, with the reason.
