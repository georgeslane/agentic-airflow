# Graph UI & HTTP API

The UI is intentionally small: a JSON description of the DAG (with live statuses) and a self-contained HTML page that draws it as an SVG, colours nodes by status, and shows a side panel with instances, outputs and errors. No external assets, dark/light aware, phone-width friendly.

## JSON graph

```python
graph = engine.graph("inbox")                 # definition only
graph = engine.graph("inbox", run.run_id)     # with statuses, instances, outputs, run summary
```

Shape:

```json
{
  "dag": {"id": "...", "prompt": "...", "schedule": [...], "tags": [...]},
  "nodes": [{"id", "kind", "agentic", "group", "fan_out", "follow_mapping", "trigger_rule", "prompt",
             "status", "instances": [{"id", "status", "map_path", "attempt", "steps", "tool_calls", "output", "error", "reason", "selected", "approval_id"}],
             "position": {"x", "y", "layer"}}],
  "edges": [{"source", "target"}],
  "groups": [...],
  "layout": {"width", "height"},
  "run": {"run_id", "status", "created_at", "started_at", "finished_at", "trigger", "error", "summary", "log_entries"}
}
```

`ui.graph.layout_graph(nodes, edges)` gives layered coordinates (longest-path layering, barycentric ordering) if you render in your own front-end.

## HTML

```python
from agentic_airflow.ui import render_html, render_embed
html = render_html(graph, refresh_url="/api/runs/<id>/graph")   # full page, polls when a run is live
fragment = render_embed(graph, container_id="my-graph")          # <div> + <style> + <script> to drop into your app
```

All data is JSON-embedded with `</` escaped and rendered via `textContent`; the full page sets a nonce-based Content-Security-Policy.

## HTTP server (optional)

```python
from agentic_airflow.ui.server import UIServer
server = UIServer(engine, host="127.0.0.1", port=8800, token="change-me").start()
```

The server runs in a daemon thread and proxies calls onto the engine's event loop. Routes:

| Method & path | Purpose |
|---|---|
| `GET /` | index of DAGs and recent runs |
| `GET /dags/{dag_id}[?run=<run_id>]` | graph page (auto-refreshing while the run is live) |
| `GET /api/dags`, `GET /api/dags/{id}`, `GET /api/dags/{id}/graph` | definitions |
| `GET /api/runs?dag_id=&limit=`, `GET /api/runs/{id}`, `/graph`, `/log` | run state, graph, `dag_context` |
| `POST /api/runs` `{"dag_id", "payload"}` | trigger a run (202) |
| `POST /api/runs/{id}/cancel` | cancel |
| `GET /api/approvals?status=pending|all`, `POST /api/approvals/{id}` `{"decision": "approve"|"reject"|"pass"|"retry"|"fail", "remember", "reason"}` | human in the loop |
| `GET /api/recommendations`, `POST /api/recommendations/{id}` `{"decision": "approve"|"reject"}` | DagEvaluator gate |
| `POST /api/chat` `{"question", "run_id"}` | live user chat |
| `POST /webhooks/{name}` (`X-Signature-256: sha256=<hmac>`) | webhook delivery (202) |
| `GET /api/stats`, `GET /healthz` | diagnostics |

Authentication: when `token` is set, every route except webhooks requires `Authorization: Bearer <token>` (or `X-Api-Token`); HTML pages also accept `?token=` so they can be opened in a browser. Binding to a non-loopback address without a token raises unless `allow_insecure=True`. Responses carry `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` and a CSP.
