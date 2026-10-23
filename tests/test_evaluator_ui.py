import json
import urllib.request

import pytest

from agentic_airflow import AgenticTask, Dag, DagEvaluator, RunStatus, Task
from agentic_airflow.ui import render_embed, render_html
from agentic_airflow.ui.graph import build_graph, layout_graph


def evaluator_responses(req):
    kind = req.metadata["task_id"].rsplit(".", 1)[-1]
    if kind == "workflow_critique":
        return {"output": {"score": 6, "assessment": "ok", "prompt_changes": [{"task_id": "work", "proposed": "Better prompt", "rationale": "clearer"}], "structural_changes": ["add a verification task"]}}
    if kind == "tool_audit":
        return {"output": {"hide": ["send_email"], "allow": ["delete_file"], "deny": [], "rationale": "unused"}}
    if kind == "skill_rewriter":
        return {"output": {"skills": [{"name": "Writing", "content": "Be concise.", "rationale": "verbose"}]}}
    if kind == "context_compactor":
        return {"output": {"memory": {"processed_ids": [1, 2]}, "discarded": ["search results"]}}
    return "Executive summary text"


@pytest.mark.asyncio
async def test_dag_evaluator_recommendations_memory_and_approval_gate(make_engine, backend):
    with Dag("ev", prompt="Do the work") as dag:  # auto_evaluate defaults to True
        Task("src", fn=lambda: "data") >> AgenticTask("work", prompt="Work on it")
    backend.add("work", "worked")
    backend.default = evaluator_responses
    engine = make_engine()
    engine.skills.register("Writing", "Write a lot.")
    engine.register(dag)
    assert dag.evaluator_group is not None and len(dag.tasks) == 7
    assert dag.get("dag_evaluator.workflow_critique").upstream_ids == ["work"]
    async with engine:
        run = await engine.run("ev", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    assert run.summary == "Executive summary text"
    # memory compaction applied automatically; the rest gated on approval
    assert engine.get_memory("ev") == {"processed_ids": [1, 2]}
    recs = engine.list_recommendations(dag_id="ev")
    assert sorted(r["kind"] for r in recs) == ["skills", "tools", "topology"]
    assert all(r["status"] == "pending" for r in recs)
    assert engine.overrides_for("ev") == {} and engine.skills.get("Writing").content == "Write a lot."
    critique_req = next(r for r in backend.requests if r.metadata["task_id"].endswith("workflow_critique"))
    assert "Run log" in critique_req.messages[0].content and "Workflow definition" in critique_req.messages[0].content
    assert critique_req.tools == []  # evaluator tasks get no tools

    tools_rec = next(r for r in recs if r["kind"] == "tools")
    await engine.approve_recommendation(tools_rec["id"])
    assert engine.overrides_for("ev") == {"allow": ["delete_file"], "hide": ["send_email"], "deny": []}
    assert engine.resolved_config("ev", "work")["tools"]["hidden"] == ["send_email", "web_search"]
    skills_rec = next(r for r in recs if r["kind"] == "skills")
    await engine.approve_recommendation(skills_rec["id"])
    assert engine.skills.get("Writing").content == "Be concise." and engine.skills.get("Writing").current == 2
    topo = next(r for r in recs if r["kind"] == "topology")
    result = await engine.approve_recommendation(topo["id"])
    assert result["applied"]["prompts_updated"] == ["work"] and result["applied"]["manual_follow_up"] == ["add a verification task"]
    assert engine.overrides_for("ev")["prompts"] == {"work": "Better prompt"}
    await engine.reject_recommendation(topo["id"]) if False else None
    with pytest.raises(Exception):
        await engine.approve_recommendation(topo["id"])  # already decided
    # subsequent runs use the approved prompt override
    backend.add("work", "worked again")
    async with engine:
        run2 = await engine.run("ev", timeout=5)
    work_req = [r for r in backend.requests if r.metadata["task_id"] == "work"][-1]
    assert work_req.messages[0].content.endswith("# Task\nBetter prompt")
    assert run2.memory == {"processed_ids": [1, 2]}


@pytest.mark.asyncio
async def test_evaluator_runs_even_when_dag_fails_and_custom_group(make_engine, backend):
    def boom():
        raise RuntimeError("x")

    with Dag("evfail", prompt="p") as dag:
        Task("boom", fn=boom)
    backend.default = evaluator_responses
    engine = make_engine()
    engine.register(dag, evaluator=DagEvaluator(model="judge", apply_memory=False))
    async with engine:
        run = await engine.run("evfail", timeout=5)
    assert run.status == RunStatus.FAILED
    assert run.instances["dag_evaluator.executive_summary"].status.value == "success"
    assert all(r.model == "judge" for r in backend.requests)
    assert engine.get_memory("evfail") == {}
    assert any(r["kind"] == "memory" for r in engine.list_recommendations(run_id=run.run_id))


def test_layout_and_graph_json(make_engine):
    with Dag("g", prompt="p", auto_evaluate=False) as dag:
        a = Task("a", fn=lambda: 1)
        b = Task("b", fn=lambda: 1)
        c = Task("c", fn=lambda: 1)
        d = Task("d", fn=lambda: 1)
        a >> [b, c] >> d
    pos = layout_graph(["a", "b", "c", "d"], [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    assert pos["a"]["layer"] == 0 and pos["b"]["layer"] == pos["c"]["layer"] == 1 and pos["d"]["layer"] == 2
    assert pos["b"]["y"] != pos["c"]["y"]
    graph = build_graph(dag)
    assert [n["id"] for n in graph["nodes"]] == ["a", "b", "c", "d"]
    assert {(e["source"], e["target"]) for e in graph["edges"]} == {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")}
    html = render_html(graph)
    assert "<!doctype html>" in html and "Content-Security-Policy" in html and "nonce" in html
    embed = render_embed({"dag": {"id": "x</script><script>alert(1)", "prompt": "", "schedule": []}, "nodes": [], "edges": [], "groups": [], "layout": {"width": 0, "height": 0}}, container_id="c")
    assert "</script><script>alert" not in embed and "<\\/script>" in embed


@pytest.mark.asyncio
async def test_http_server_routes(make_engine, backend):
    from agentic_airflow import WebhookTrigger
    from agentic_airflow.ui.server import UIServer

    with Dag("srv", prompt="p", auto_evaluate=False, schedule=[WebhookTrigger("in", secret="k")]) as dag:
        Task("a", fn=lambda ctx: ctx.trigger["payload"])
    engine = make_engine()
    engine.register(dag)
    async with engine:
        server = UIServer(engine, port=0, token="tok").start()
        base = server.url

        def call(path, data=None, token="tok", headers=None):
            hdrs = {"Authorization": f"Bearer {token}"} if token else {}
            if isinstance(data, dict):
                hdrs["Content-Type"] = "application/json"
            hdrs.update(headers or {})
            body = json.dumps(data).encode() if isinstance(data, dict) else data
            req = urllib.request.Request(base + path, data=body, headers=hdrs, method="POST" if body is not None else "GET")
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()

        import asyncio

        def run_calls():
            out = {}
            out["unauth"] = call("/api/dags", token=None)[0]
            out["dags"] = json.loads(call("/api/dags")[1])
            out["page"] = call("/dags/srv", token="tok")
            out["trigger"] = json.loads(call("/api/runs", {"dag_id": "srv", "payload": {"k": 1}})[1])
            hook = dag.schedule[0]
            raw = b'{"from": "hook"}'
            out["hook_bad"] = call("/webhooks/in", raw, token=None, headers={"X-Signature-256": "nope"})[0]
            out["hook"] = json.loads(call("/webhooks/in", raw, token=None, headers={"X-Signature-256": "sha256=" + hook.sign(raw)})[1])
            out["missing"] = call("/api/dags/nope/graph")[0]
            return out

        out = await asyncio.get_running_loop().run_in_executor(None, run_calls)
        run = await engine.wait(out["trigger"]["run_id"], timeout=5)
        run_hook = await engine.wait(out["hook"]["runs"][0], timeout=5)
        graph = await asyncio.get_running_loop().run_in_executor(None, lambda: json.loads(call(f"/api/runs/{run.run_id}/graph")[1]))
        server.stop()
    assert out["unauth"] == 401 and out["missing"] == 404 and out["hook_bad"] == 401
    assert out["dags"][0]["id"] == "srv"
    assert out["page"][0] == 200 and b"<!doctype html>" in out["page"][1]
    assert run.instances["a"].output == {"k": 1} and run_hook.instances["a"].output == {"from": "hook"}
    assert graph["run"]["status"] == "success" and graph["nodes"][0]["status"] == "success"


def test_server_refuses_insecure_bind(make_engine):
    from agentic_airflow.errors import AgenticAirflowError
    from agentic_airflow.ui.server import UIServer

    with pytest.raises(AgenticAirflowError):
        UIServer(make_engine(), host="0.0.0.0", port=0)


def _raw(server, method, path, headers=None, body=b""):
    """A request as a browser might send it, without following redirects."""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request(method, path, body=body or None, headers=headers or {})
    resp = conn.getresponse()
    out = resp.status, dict(resp.getheaders()), resp.read()
    conn.close()
    return out


@pytest.mark.asyncio
async def test_other_sites_pages_cant_use_the_server(make_engine):
    import asyncio

    from agentic_airflow.ui.server import COOKIE, UIServer

    with Dag("srv", prompt="p", auto_evaluate=False) as dag:
        Task("a", fn=lambda ctx: "x")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        open_ = UIServer(engine, port=0).start()  # no token: loopback only
        locked = UIServer(engine, port=0, token="tok").start()
        here = f"127.0.0.1:{open_.port}"
        body = b'{"dag_id": "srv"}'

        def calls():
            out = {}
            post = lambda headers: _raw(open_, "POST", "/api/runs", {"Host": here, **headers}, body)[0]  # noqa: E731
            out["json"] = post({"Content-Type": "application/json"})
            out["same_origin"] = post({"Content-Type": "application/json", "Origin": f"http://{here}"})
            out["other_origin"] = post({"Content-Type": "application/json", "Origin": "https://evil.example"})
            out["text"] = post({"Content-Type": "text/plain"})  # what a form or a simple fetch sends
            out["rebound"] = _raw(open_, "POST", "/api/runs", {"Host": "evil.example", "Content-Type": "application/json"}, body)[0]
            out["rebound_get"] = _raw(open_, "GET", "/api/runs", {"Host": f"evil.example:{open_.port}"})[0]
            out["localhost"] = _raw(open_, "GET", "/api/dags", {"Host": f"localhost:{open_.port}"})[0]
            out["ipv6"] = _raw(open_, "GET", "/api/dags", {"Host": f"[::1]:{open_.port}"})[0]
            # With a token, ?token= on a page becomes a cookie, and the address loses it.
            status, headers, _ = _raw(locked, "GET", "/dags/srv?run=r1&token=tok")
            out["redirect"] = status, headers.get("Location"), headers.get("Set-Cookie", "")
            out["bad_token"] = _raw(locked, "GET", "/?token=nope")[0]
            cookie = headers.get("Set-Cookie", "").split(";")[0]
            out["with_cookie"] = _raw(locked, "GET", "/api/dags", {"Cookie": cookie})[0]
            out["wrong_cookie"] = _raw(locked, "GET", "/api/dags", {"Cookie": f"{COOKIE}=tok"})[0]
            out["api_query"] = _raw(locked, "GET", "/api/dags?token=tok")[0]  # the API takes a header, not ?token=
            return out

        out = await asyncio.get_running_loop().run_in_executor(None, calls)
        open_.stop()
        locked.stop()
    assert out["json"] == 202 and out["same_origin"] == 202 and out["localhost"] == 200 and out["ipv6"] == 200
    assert out["other_origin"] == 403 and out["text"] == 415 and out["rebound"] == 403 and out["rebound_get"] == 403
    status, location, cookie = out["redirect"]
    assert status == 303 and location == "/dags/srv?run=r1" and "tok" not in location
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "=tok;" not in cookie
    assert out["bad_token"] == 401 and out["with_cookie"] == 200 and out["wrong_cookie"] == 401 and out["api_query"] == 401
    assert len([r for r in engine.list_runs() if r.trigger.get("source") == "api"]) == 2
