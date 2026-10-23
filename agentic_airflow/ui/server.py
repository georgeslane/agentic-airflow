"""Optional stdlib HTTP server exposing the graph UI, a JSON API and webhooks.

.. code-block:: python

    server = UIServer(engine, host="127.0.0.1", port=8800, token="secret")
    server.start()            # background thread; engine must be running in an event loop

Routes (``Authorization: Bearer <token>`` required for everything except
webhooks when a token is configured; pages also accept ``?token=``, which is
moved into a ``SameSite=Strict`` cookie and taken out of the address):

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

Other web pages open in the same browser can't use the server: without a token
it only answers requests addressed to a loopback name or the host it was bound
to (``allowed_hosts`` adds more), so a DNS-rebound name is refused; a POST from
another origin is refused; and the API only takes JSON, which a page on another
site can't send without the server's agreement.
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
from http.cookies import SimpleCookie
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs, urlencode, urlsplit

from ..errors import AgenticAirflowError, WebhookAuthenticationError
from .html import render_html

log = logging.getLogger("agentic_airflow.ui")

MAX_BODY = 2_000_000
LOOPBACK = ("127.0.0.1", "localhost", "::1")
COOKIE = "agentic_airflow_token"


class UIServer:
    def __init__(self, engine: Any, *, host: str = "127.0.0.1", port: int = 8800, token: str | None = None, allow_insecure: bool = False, refresh_ms: int = 2000, allowed_hosts: Iterable[str] = ()):
        if host not in LOOPBACK and not token and not allow_insecure:
            raise AgenticAirflowError("Refusing to bind a non-loopback address without a token (pass allow_insecure=True to override)")
        self.engine = engine
        self.host = host
        self.port = port
        self.token = token
        self.refresh_ms = refresh_ms
        self.allowed_hosts = {h.lower() for h in (*LOOPBACK, host, *allowed_hosts)}
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

    def authorised(self, headers: Any) -> bool:
        """The bearer token, the X-Api-Token header or the page's cookie; anything without a token."""
        if not self.token:
            return True
        auth = headers.get("Authorization", "")
        candidate = auth[7:] if auth.lower().startswith("bearer ") else headers.get("X-Api-Token", "")
        if candidate:
            return hmac.compare_digest(candidate.encode(), self.token.encode())
        cookie = SimpleCookie(headers.get("Cookie", ""))
        return COOKIE in cookie and hmac.compare_digest(cookie[COOKIE].value, self.cookie_value())

    def cookie_value(self) -> str:
        """What the pages' cookie holds: derived from the token, so the token itself isn't kept in the browser."""
        return hmac.new((self.token or "").encode(), b"agentic-airflow page cookie", "sha256").hexdigest()

    def host_allowed(self, headers: Any) -> bool:
        """Whether a request is addressed to this server, by a name it answers to. With a token anything is: a
        DNS-rebound page can't send the token."""
        if self.token:
            return True
        return _hostname(headers.get("Host", "")) in self.allowed_hosts

    @staticmethod
    def same_origin(headers: Any) -> bool:
        """Whether a request isn't from another site's page: browsers send an Origin with every POST, and other
        clients send none."""
        origin = headers.get("Origin")
        if origin is None:
            return True
        return urlsplit(origin).netloc.lower() == headers.get("Host", "").lower()


def _hostname(host: str) -> str:
    """The name in a Host header, without its port: "[::1]:8800" -> "::1"."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


class _Handler(BaseHTTPRequestHandler):
    server_ref: UIServer
    server_version = "agentic-airflow"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default logging
        log.debug("%s - " + fmt, self.address_string(), *args)

    # -- helpers -----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
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
    def _with_cookie(self, query: dict[str, list[str]]) -> None:
        """A page asked for with ?token=: keep the token in a cookie only this site's pages send, and ask for the
        page again without it, so it isn't left in the address, the history or a log."""
        srv = self.server_ref
        if not srv.token or not hmac.compare_digest(query["token"][0].encode(), srv.token.encode()):
            return self._error(401, "unauthorised")
        rest = {k: v for k, v in query.items() if k != "token"}
        location = urlsplit(self.path).path + (f"?{urlencode(rest, doseq=True)}" if rest else "")
        cookie = f"{COOKIE}={srv.cookie_value()}; Path=/; HttpOnly; SameSite=Strict"
        self._send(303, b"", "text/plain", {"Location": location, "Set-Cookie": cookie})

    def do_GET(self) -> None:  # noqa: N802
        srv = self.server_ref
        segments, query = self._route()
        try:
            if not srv.host_allowed(self.headers):
                return self._error(403, "unknown host")
            if "token" in query and segments[:1] != ["api"]:
                return self._with_cookie(query)
            if not srv.authorised(self.headers):
                return self._error(401, "unauthorised")
            engine = srv.engine
            if not segments:
                return self._html(self._index_page())
            if segments[0] == "dags" and len(segments) == 2:
                dag_id = segments[1]
                if dag_id not in engine.dags:
                    return self._error(404, "unknown dag")
                run_id = query.get("run", [None])[0]
                graph = srv.call(engine.graph, dag_id, run_id)
                refresh = f"/api/runs/{run_id}/graph" if run_id else None  # its cookie goes with the page's requests
                return self._html(render_html(graph, refresh_url=refresh, refresh_ms=srv.refresh_ms))
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
        segments, _ = self._route()
        try:
            if not srv.host_allowed(self.headers):
                return self._error(403, "unknown host")
            if not srv.same_origin(self.headers):
                return self._error(403, "cross-origin request refused")
            raw = self._body()
            if segments[:1] == ["webhooks"] and len(segments) == 2:
                signature = self.headers.get("X-Signature-256") or self.headers.get("X-Hub-Signature-256")
                try:
                    runs = srv.call(srv.engine.deliver_webhook, segments[1], None, raw_body=raw, signature=signature)
                except WebhookAuthenticationError as exc:
                    return self._error(401, str(exc))
                return self._json({"runs": [r.run_id for r in runs]}, 202)
            if not srv.authorised(self.headers):
                return self._error(401, "unauthorised")
            if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
                return self._error(415, "send JSON, with Content-Type: application/json")
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
    def _index_page(self) -> str:
        srv, engine = self.server_ref, self.server_ref.engine
        runs = srv.call(engine.list_runs, None, None, 30)
        rows = "".join(
            f"<tr><td><a href=\"/dags/{html.escape(r.dag_id)}?run={html.escape(r.run_id)}\">{html.escape(r.run_id)}</a></td>"
            f"<td>{html.escape(r.dag_id)}</td><td>{html.escape(r.status.value)}</td></tr>"
            for r in runs
        )
        dags = "".join(f"<li><a href=\"/dags/{html.escape(d.id)}\">{html.escape(d.id)}</a> <small>{html.escape(d.description or '')}</small></li>" for d in engine.dags.values())
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
