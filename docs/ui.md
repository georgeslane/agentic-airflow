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
| `GET /api/approvals?status=pending|all`, `POST /api/approvals/{id}` `{"decision": "approve"|"reject"|"answer"|"dismiss"|"pass"|"retry"|"fail", "remember", "reason", "answer"}` | human in the loop |
| `GET /api/recommendations`, `POST /api/recommendations/{id}` `{"decision": "approve"|"reject"}` | DagEvaluator gate |
| `POST /api/chat` `{"question", "run_id"}` | live user chat |
| `POST /webhooks/{name}` (`X-Signature-256: sha256=<hmac>`) | webhook delivery (202) |
| `GET /api/stats`, `GET /healthz` | diagnostics |

Authentication: when `token` is set, every route except webhooks requires `Authorization: Bearer <token>` (or `X-Api-Token`). To open the pages in a browser, add `?token=` once: the server answers with a `HttpOnly`, `SameSite=Strict` cookie, derived from the token, and sends the browser to the same address without it, so the token isn't left in the address bar, the history or a log. The API doesn't take `?token=`. Binding to a non-loopback address without a token raises unless `allow_insecure=True`. Responses carry `X-Content-Type-Options: nosniff`, `Cache-Control: no-store` and a CSP.

Other pages open in the same browser can't use the server:

* **Without a token**, it only answers requests addressed to a loopback name (`127.0.0.1`, `localhost`, `::1`) or the host it was bound to; pass `allowed_hosts=[...]` for others. A page on another site that rebinds its own name to `127.0.0.1` is refused.
* **A POST from another origin** (its `Origin` header isn't this server) is refused with 403, webhooks included.
* **The API only takes JSON**, with `Content-Type: application/json` (415 otherwise), which a page on another site can't send without asking first, and the server never agrees to.
