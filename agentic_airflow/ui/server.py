"""Optional stdlib HTTP server exposing the graph UI, a JSON API and webhooks.

.. code-block:: python

    server = UIServer(engine, host="127.0.0.1", port=8800, token="secret")
    server.start()            # background thread; engine must be running in an event loop

Routes (``Authorization: Bearer <token>`` required for everything except
webhooks when a token is configured; pages also accept ``?token=``):

* ``GET  /``                              - index of DAGs and recent runs
* ``GET  /dags/{dag_id}[?run=<run_id>]``  - graph page (auto-refreshing)
* ``GET  /api/dags``                      - DAG list
* ``GET  /api/dags/{dag_id}/graph``       - graph JSON (definition)
* ``GET  /api/runs?dag_id=``              - runs
* ``GET  /api/runs/{run_id}``             - run JSON (full state)
* ``GET  /api/runs/{run_id}/graph``       - graph JSON with live status
* ``POST /api/runs``  ``{"dag_id": ..., "payload": ...}``  - trigger
* ``POST /api/runs/{run_id}/cancel``
* ``GET  /api/approvals`` / ``POST /api/approvals/{id}`` ``{"decision": "approve"|"reject"|..., "remember": bool, "reason": str}``
* ``GET  /api/recommendations`` / ``POST /api/recommendations/{id}`` ``{"decision": "approve"|"reject"}``
* ``POST /api/chat`` ``{"question": ..., "run_id": ...}``
* ``POST /webhooks/{name}``               - webhook delivery (HMAC via ``X-Signature-256``)
* ``GET  /api/stats``

Binding to a non-loopback host without a token raises unless
``allow_insecure=True`` is passed explicitly.
"""

from __future__ import annotations

import asyncio
import hmac
import html
import json
import logging
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from ..errors import AgenticAirflowError, WebhookAuthenticationError
from .html import render_html

log = logging.getLogger("agentic_airflow.ui")

MAX_BODY = 2_000_000


class UIServer:
    def __init__(self, engine: Any, *, host: str = "127.0.0.1", port: int = 8800, token: str | None = None, allow_insecure: bool = False, refresh_ms: int = 2000):
        if host not in ("127.0.0.1", "localhost", "::1") and not token and not allow_insecure:
            raise AgenticAirflowError("Refusing to bind a non-loopback address without a token (pass allow_insecure=True to override)")
        self.engine = engine
        self.host = host
        self.port = port
        self.token = token
        self.refresh_ms = refresh_ms
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "UIServer":
        server = self
        handler = type("Handler", (_Handler,), {"server_ref": server})
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="agentic-airflow-ui", daemon=True)
        self._thread.start()
        log.info("UI listening on http://%s:%s", self.host, self.port)
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- engine access from server threads ---------------------------------
    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn`` on the engine's event loop thread and wait for the result."""
        loop = self.engine.loop
        if loop is None or not loop.is_running():
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result

        async def runner() -> Any:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return asyncio.run_coroutine_threadsafe(runner(), loop).result(timeout=60)

    def authorised(self, headers: Any, query: dict[str, list[str]]) -> bool:
        if not self.token:
            return True
        auth = headers.get("Authorization", "")
        candidate = auth[7:] if auth.lower().startswith("bearer ") else headers.get("X-Api-Token", "")
        if not candidate and "token" in query:
            candidate = query["token"][0]
        return hmac.compare_digest(candidate, self.token)


class _Handler(BaseHTTPRequestHandler):
    server_ref: UIServer
    server_version = "agentic-airflow"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        log.debug("%s - " + fmt, self.address_string(), *args)

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, default=str).encode(), "application/json; charset=utf-8")

    def _html(self, text: str, status: int = 200) -> None:
        self._send(status, text.encode(), "text/html; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        return self.rfile.read(length) if length else b""

    def _route(self) -> tuple[list[str], dict[str, list[str]]]:
        parts = urlsplit(self.path)
        segments = [s for s in parts.path.split("/") if s]
        return segments, parse_qs(parts.query)

    # -- GET -----------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        srv = self.server_ref
        segments, query = self._route()
        try:
            if not srv.authorised(self.headers, query):
                return self._error(401, "unauthorised")
            engine = srv.engine
            token_q = f"?token={query['token'][0]}" if "token" in query else ""
            if not segments:
                return self._html(self._index_page(token_q))
            if segments[0] == "dags" and len(segments) == 2:
                dag_id = segments[1]
                if dag_id not in engine.dags:
                    return self._error(404, "unknown dag")
                run_id = query.get("run", [None])[0]
                graph = srv.call(engine.graph, dag_id, run_id)
                refresh = f"/api/runs/{run_id}/graph" if run_id else None
                token = query.get("token", [None])[0]
                return self._html(render_html(graph, refresh_url=refresh, refresh_ms=srv.refresh_ms, token=token))
            if segments[0] == "api":
                return self._api_get(segments[1:], query)
            if segments[0] == "healthz":
                return self._json({"ok": True})
            return self._error(404, "not found")
        except AgenticAirflowError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("GET failed")
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _api_get(self, segs: list[str], query: dict[str, list[str]]) -> None:
        srv, engine = self.server_ref, self.server_ref.engine
        if segs == ["dags"]:
            return self._json([{"id": d.id, "description": d.description, "schedule": [s.to_dict() for s in d.schedule], "tasks": len(d.tasks)} for d in engine.dags.values()])
        if len(segs) == 3 and segs[0] == "dags" and segs[2] == "graph":
            if segs[1] not in engine.dags:
                return self._error(404, "unknown dag")
            return self._json(srv.call(engine.graph, segs[1]))
        if len(segs) == 2 and segs[0] == "dags":
            if segs[1] not in engine.dags:
                return self._error(404, "unknown dag")
            return self._json(engine.dags[segs[1]].to_dict())
        if segs == ["runs"]:
            dag_id = query.get("dag_id", [None])[0]
            limit = int(query.get("limit", ["50"])[0])
            runs = srv.call(engine.list_runs, dag_id, None, min(limit, 500))
            return self._json([{"run_id": r.run_id, "dag_id": r.dag_id, "status": r.status.value, "created_at": r.created_at, "finished_at": r.finished_at} for r in runs])
        if len(segs) >= 2 and segs[0] == "runs":
            run = srv.call(engine.get_run, segs[1])
            if len(segs) == 3 and segs[2] == "graph":
                return self._json(srv.call(engine.graph, run.dag_id, run.run_id))
            if len(segs) == 3 and segs[2] == "log":
                return self._json(run.dag_context)
            return self._json(run.to_dict())
        if segs == ["approvals"]:
            status = query.get("status", ["pending"])[0]
            return self._json([a.to_dict() for a in srv.call(engine.list_approvals, None if status == "all" else status)])
        if segs == ["recommendations"]:
            return self._json(srv.call(engine.list_recommendations, query.get("dag_id", [None])[0], query.get("run_id", [None])[0], query.get("status", [None])[0]))
        if segs == ["stats"]:
            return self._json(srv.call(engine.stats))
        return self._error(404, "not found")

    # -- POST ----------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        srv = self.server_ref
        segments, query = self._route()
        try:
            raw = self._body()
            if segments[:1] == ["webhooks"] and len(segments) == 2:
                signature = self.headers.get("X-Signature-256") or self.headers.get("X-Hub-Signature-256")
                try:
                    runs = srv.call(srv.engine.deliver_webhook, segments[1], None, raw_body=raw, signature=signature)
                except WebhookAuthenticationError as exc:
                    return self._error(401, str(exc))
                return self._json({"runs": [r.run_id for r in runs]}, 202)
            if not srv.authorised(self.headers, query):
                return self._error(401, "unauthorised")
            data = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(data, dict):
                return self._error(400, "JSON object expected")
            return self._api_post(segments, data)
        except json.JSONDecodeError:
            return self._error(400, "invalid JSON")
        except AgenticAirflowError as exc:
            return self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("POST failed")
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def _api_post(self, segs: list[str], data: dict[str, Any]) -> None:
        srv, engine = self.server_ref, self.server_ref.engine
        if segs == ["api", "runs"]:
            dag_id = data.get("dag_id")
            if dag_id not in engine.dags:
                return self._error(404, "unknown dag")
            run = srv.call(engine.trigger, dag_id, data.get("payload"), source="api")
            return self._json({"run_id": run.run_id, "status": run.status.value}, 202)
        if len(segs) == 4 and segs[:2] == ["api", "runs"] and segs[3] == "cancel":
            run = srv.call(engine.cancel, segs[2])
            return self._json({"run_id": run.run_id, "status": run.status.value})
        if len(segs) == 3 and segs[:2] == ["api", "approvals"]:
            decision = str(data.get("decision") or "")
            extra = {k: v for k, v in data.items() if k in ("remember", "reason", "note")}
            approval = srv.call(engine.decide, segs[2], decision, **extra)
            return self._json(approval.to_dict())
        if len(segs) == 3 and segs[:2] == ["api", "recommendations"]:
            decision = str(data.get("decision") or "")
            if decision == "approve":
                return self._json(srv.call(engine.approve_recommendation, segs[2]))
            if decision == "reject":
                return self._json(srv.call(engine.reject_recommendation, segs[2], str(data.get("reason") or "")))
            return self._error(400, "decision must be 'approve' or 'reject'")
        if segs == ["api", "chat"]:
            question = str(data.get("question") or "").strip()
            if not question:
                return self._error(400, "question required")
            answer = srv.call(engine.chat, question, run_id=data.get("run_id"), model=data.get("model"))
            return self._json({"answer": answer})
        return self._error(404, "not found")

    # -- pages ---------------------------------------------------------------
    def _index_page(self, token_q: str) -> str:
        srv, engine = self.server_ref, self.server_ref.engine
        runs = srv.call(engine.list_runs, None, None, 30)
        rows = "".join(
            f"<tr><td><a href=\"/dags/{html.escape(r.dag_id)}?run={html.escape(r.run_id)}{html.escape('&' + token_q[1:]) if token_q else ''}\">{html.escape(r.run_id)}</a></td>"
            f"<td>{html.escape(r.dag_id)}</td><td>{html.escape(r.status.value)}</td></tr>"
            for r in runs
        )
        dags = "".join(f"<li><a href=\"/dags/{html.escape(d.id)}{html.escape(token_q)}\">{html.escape(d.id)}</a> <small>{html.escape(d.description or '')}</small></li>" for d in engine.dags.values())
        nonce = secrets.token_urlsafe(8)
        return (
            "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'nonce-{nonce}'; base-uri 'none'\">"
            f"<title>agentic-airflow</title><style nonce=\"{nonce}\">body{{font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1c1c1e;background:#f7f7f8}}"
            "table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:left}a{color:#3b6fd6}"
            "@media (prefers-color-scheme:dark){body{background:#141417;color:#ececf1}td,th{border-color:#333}a{color:#8db0ff}}</style></head><body>"
            f"<h1>agentic-airflow</h1><h2>DAGs</h2><ul>{dags or '<li>none registered</li>'}</ul>"
            f"<h2>Recent runs</h2><table><tr><th>run</th><th>dag</th><th>status</th></tr>{rows or '<tr><td colspan=3>no runs yet</td></tr>'}</table></body></html>"
        )
