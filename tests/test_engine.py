import asyncio

import pytest

from agentic_airflow import (
    AgenticBranch,
    AgenticEvaluator,
    AgenticTask,
    Branch,
    Dag,
    EngineConfig,
    GenerationResponse,
    RunStatus,
    SqliteStateStore,
    Task,
    TaskStatus,
    ToolCall,
)


def statuses(run):
    return {i.instance_id: i.status.value for i in run.instances.values()}


# ---------------------------------------------------------------------------
# Deterministic tasks, mapping, fan-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_linear_python_dag_passes_outputs(make_engine):
    with Dag("lin", prompt="p", auto_evaluate=False) as dag:
        a = Task("a", fn=lambda: 2)
        b = Task("b", fn=lambda x: x * 3)
        c = Task("c", fn=lambda y, ctx: (y, ctx.task_id, ctx.map_path, sorted(ctx.upstream)))
        a >> b >> c
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("lin", timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert run.instances["c"].output == (6, "c", (), ["b"])
    assert [e.task_id for e in run.log] == ["a", "b", "c"]
    assert run.log[1].input_context == {"upstream": {"a": 2}, "retry_history": []}


@pytest.mark.asyncio
async def test_fan_out_and_fan_in(make_engine):
    with Dag("map", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: ["x", "y", "z"], fan_out=True)
        work = Task("work", fn=lambda item: item.upper())
        join = Task("join", fn=lambda d: d)
        src >> work >> join
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("map", timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert [i for i in run.node_instances["work"]] == ["work_0", "work_1", "work_2"]
    assert run.instances["join"].output == {"work_0": "X", "work_1": "Y", "work_2": "Z"}
    assert run.instances["work_1"].map_path == (1,)


@pytest.mark.asyncio
async def test_empty_fan_out_skips_chain_but_join_still_runs_with_other_inputs(make_engine):
    with Dag("empty", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: [], fan_out=True)
        work = Task("work", fn=lambda item: item)
        after = Task("after", fn=lambda x: x)
        other = Task("other", fn=lambda: "o")
        join = Task("join", fn=lambda a, b: (a, b))
        src >> work >> after
        [after, other] >> join
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("empty", timeout=5)
    assert run.status == RunStatus.SUCCESS
    s = statuses(run)
    assert s["work"] == "skipped" and s["after"] == "skipped" and s["join"] == "success"
    assert run.instances["join"].output == (None, "o")


@pytest.mark.asyncio
async def test_follow_mapping_and_nested_fan_out(make_engine):
    with Dag("nested", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: [1, 2], fan_out=True)
        per = Task("per", fn=lambda n: [n * 10, n * 10 + 1], fan_out=True)
        leaf = Task("leaf", fn=lambda v, ctx: (v, ctx.map_path))
        other = Task("other", fn=lambda n: n * 100)          # mapped by src, scalar output
        sibling = Task("sibling", fn=lambda n: n + 1, follow_mapping=True)  # 1:1 with other_i
        joined = Task("joined", fn=lambda d: d)               # default: fan-in over other
        join = Task("join", fn=lambda leaves, sibs: (leaves, sibs))
        src >> per >> leaf
        src >> other >> [sibling, joined]
        [leaf, sibling] >> join
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("nested", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    assert run.node_instances["leaf"] == ["leaf_0_0", "leaf_0_1", "leaf_1_0", "leaf_1_1"]
    assert run.instances["leaf_1_0"].output == (20, (1, 0))
    assert run.node_instances["sibling"] == ["sibling_0", "sibling_1"]
    assert run.instances["sibling_1"].output == 201
    assert run.instances["joined"].output == {"other_0": 100, "other_1": 200}
    assert run.instances["join"].output[1] == {"sibling_0": 101, "sibling_1": 201}
    assert set(run.instances["join"].output[0]) == {"leaf_0_0", "leaf_0_1", "leaf_1_0", "leaf_1_1"}


@pytest.mark.asyncio
async def test_failure_propagates_and_all_done_rule_runs(make_engine):
    def boom():
        raise ValueError("kaboom")

    with Dag("fail", prompt="p", auto_evaluate=False) as dag:
        a = Task("a", fn=boom)
        b = Task("b", fn=lambda x: x)
        cleanup = Task("cleanup", fn=lambda ctx: len(ctx.run_log), trigger_rule="all_done", context=[])
        a >> b >> cleanup
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("fail", timeout=5)
    assert run.status == RunStatus.FAILED
    s = statuses(run)
    assert s == {"a": "failed", "b": "upstream_failed", "cleanup": "success"}
    assert "kaboom" in run.instances["a"].error


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bool_branch_skips_exclusive_chain_and_join_waits_for_active(make_engine):
    with Dag("br", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: {"urgent": True})
        br = Branch("is_urgent", fn=lambda d: d["urgent"])
        yes = Task("yes", fn=lambda d: f"yes:{d['urgent']}")
        no = Task("no", fn=lambda d: "no")
        no_child = Task("no_child", fn=lambda x: x)
        join = Task("join", fn=lambda a, b: (a, b))
        src >> br >> [yes, no]
        no >> no_child
        [yes, no_child] >> join
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("br", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    s = statuses(run)
    assert s["yes"] == "success" and s["no"] == "skipped" and s["no_child"] == "skipped" and s["join"] == "success"
    assert run.instances["yes"].output == "yes:True"  # branch is transparent for data
    assert run.instances["join"].output == ("yes:True", None)
    assert run.instances["is_urgent"].selected == ["yes"]


@pytest.mark.asyncio
async def test_branch_by_id_list_and_errors(make_engine):
    with Dag("br2", prompt="p", auto_evaluate=False) as dag:
        br = Branch("route", fn=lambda: ["b", "c"])
        a, b, c = (Task(x, fn=lambda: x) for x in "abc")
        br >> [a, b, c]
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("br2", timeout=5)
    assert statuses(run) == {"route": "success", "a": "skipped", "b": "success", "c": "success"}

    with Dag("br3", prompt="p", auto_evaluate=False) as dag3:
        br = Branch("route", fn=lambda: "nope")
        br >> Task("a", fn=lambda: 1)
    engine.register(dag3)
    async with engine:
        run = await engine.run("br3", timeout=5)
    assert run.status == RunStatus.FAILED and "unknown downstream" in run.instances["route"].error


@pytest.mark.asyncio
async def test_agentic_branch_uses_enum_contract(make_engine, backend):
    with Dag("abr", prompt="Objective", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: "an urgent email")
        br = AgenticBranch("classify", prompt="Pick a branch")
        reply = AgenticTask("reply", prompt="Reply")
        archive = Task("archive", fn=lambda e: e)
        src >> br >> [reply, archive]
    backend.add("classify", "reply")
    backend.add("reply", "drafted")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("abr", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    req = next(r for r in backend.requests if r.metadata["task_id"] == "classify")
    assert req.output.kind == "enum" and req.output.choices == ["reply", "archive"]
    assert "Objective" in req.system and "an urgent email" in req.messages[0].content
    assert statuses(run)["archive"] == "skipped" and run.instances["reply"].output == "drafted"


# ---------------------------------------------------------------------------
# Agentic tasks: tools, permissions, approvals
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_loop_hidden_discovery_and_denied_pause(make_engine, backend):
    with Dag("tools", prompt="p", auto_evaluate=False, allowed_tools=["send_email", "web_search"]) as dag:
        AgenticTask("agent", prompt="do it")
    seen = []

    def step(req):
        seen.append([t.name for t in req.tools])
        n = req.metadata["step"]
        if n == 1:
            return GenerationResponse(tool_calls=[ToolCall(name="search_tools", arguments={"query": "search web"})])
        if n == 2:
            return GenerationResponse(tool_calls=[ToolCall(name="web_search", arguments={"query": "q"}), ToolCall(name="delete_file", arguments={"path": "/x"})])
        return "final answer"

    backend.add("agent", step, step, step)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("tools", timeout=5)
        assert run.status == RunStatus.PAUSED
        inst = run.instances["agent"]
        assert inst.status == TaskStatus.AWAITING_TOOL_APPROVAL
        assert seen[0] == ["send_email", "search_tools"]          # hidden tool not injected
        assert seen[1] == ["send_email", "web_search"]            # discovered after search_tools
        assert inst.checkpoint["pending"]["index"] == 1
        approvals = engine.list_approvals()
        assert len(approvals) == 1 and approvals[0].payload["tool_call"]["name"] == "delete_file"
        assert approvals[0].payload["reason"] == "denied_tool"
        await engine.reject(approvals[0].approval_id, reason="too risky")
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    tool_names = [(c["name"], c["status"]) for c in inst.tool_calls]
    assert tool_names == [("search_tools", "ok"), ("web_search", "ok"), ("delete_file", "rejected")]
    # the model was told about the rejection
    final_req = backend.requests[-1]
    assert any("denied by the user" in m.content for m in final_req.messages if m.role == "tool")
    assert inst.output == "final answer"
    assert run.log[-1].tool_calls[-1]["status"] == "rejected"


@pytest.mark.asyncio
async def test_approve_with_remember_executes_tool(make_engine, backend):
    with Dag("appr", prompt="p", auto_evaluate=False, allowed_tools=[]) as dag:
        AgenticTask("agent", prompt="do it")

    def step(req):
        n = req.metadata["step"]
        if n <= 2:
            return GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})])
        return "done"

    backend.add("agent", step, step, step)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("appr", timeout=5)
        assert run.status == RunStatus.PAUSED
        apr = engine.list_approvals()[0]
        await engine.approve(apr.approval_id, remember=True)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    inst = run.instances["agent"]
    assert [c["status"] for c in inst.tool_calls] == ["approved", "ok"]  # second call needed no approval
    assert run.approved_tools == ["send_email"]
    assert engine.list_approvals(status="approved")[0].approval_id == apr.approval_id


@pytest.mark.asyncio
async def test_parallel_branches_continue_while_one_is_paused(make_engine, backend):
    with Dag("par", prompt="p", auto_evaluate=False, allowed_tools=[]) as dag:
        paused = AgenticTask("paused", prompt="x")
        free = AgenticTask("free", prompt="y")
        tail = Task("tail", fn=lambda a, b: (a, b))
        [paused, free] >> tail
    backend.add("paused", GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})]), "p-done")
    backend.add("free", "f-done")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("par", timeout=5)
        assert run.status == RunStatus.PAUSED
        assert run.instances["free"].status == TaskStatus.SUCCESS
        assert "tail" not in run.instances  # not expanded until both upstreams finish
        await engine.approve(engine.list_approvals()[0].approval_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.instances["tail"].output == ("p-done", "f-done")


@pytest.mark.asyncio
async def test_max_tool_steps_and_output_validation_retry(make_engine, backend):
    with Dag("limits", prompt="p", auto_evaluate=False, max_tool_steps=2) as dag:
        AgenticTask("loop", prompt="x")
        AgenticTask("json", prompt="y", return_type="json")
    loop_call = GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})])
    backend.add("loop", loop_call, loop_call, loop_call, loop_call)
    backend.add("json", "not json at all", '{"ok": true}')
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("limits", timeout=5)
    assert run.status == RunStatus.FAILED
    assert "max_tool_steps" in run.instances["loop"].error
    assert run.instances["json"].output == {"ok": True}
    json_reqs = [r for r in backend.requests if r.metadata["task_id"] == "json"]
    assert len(json_reqs) == 2 and "rejected" in json_reqs[1].messages[-1].content
    loop_reqs = [r for r in backend.requests if r.metadata["task_id"] == "loop"]
    assert loop_reqs[-1].tools == []  # tools withdrawn once the budget is spent


@pytest.mark.asyncio
async def test_missing_model_and_backend_errors(backend, tools):
    from agentic_airflow import Engine

    with Dag("nomodel", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("a", prompt="x")
    engine = Engine(backend, tools=tools)
    engine.register(dag)
    async with engine:
        run = await engine.run("nomodel", timeout=5)
    assert run.status == RunStatus.FAILED and "no model resolved" in run.instances["a"].error
    engine.close()


# ---------------------------------------------------------------------------
# Evaluator loop
# ---------------------------------------------------------------------------

def _draft_dag(on_exhausted="pause_for_user", max_retries=1):
    with Dag("eval", prompt="p", auto_evaluate=False) as dag:
        draft = AgenticTask("draft", prompt="write")
        review = AgenticEvaluator("review", prompt="check", max_retries=max_retries, on_exhausted=on_exhausted)
        send = Task("send", fn=lambda text: f"sent:{text}")
        draft >> review >> send
    return dag


@pytest.mark.asyncio
async def test_evaluator_rerun_with_history_then_pass(make_engine, backend):
    backend.add("draft", "v1", "v2")
    backend.add("review", {"output": {"rerun": True, "reason": "too short"}}, {"output": {"rerun": False, "reason": "fine"}})
    engine = make_engine()
    engine.register(_draft_dag())
    async with engine:
        run = await engine.run("eval", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    draft = run.instances["draft"]
    assert draft.attempt == 2 and draft.retry_history == [{"attempt": 1, "output": "v1", "reason": "too short"}]
    second = [r for r in backend.requests if r.metadata["task_id"] == "draft"][1]
    assert "too short" in second.messages[0].content and "Attempt 1" in second.messages[0].content
    assert run.instances["review"].output == "v2"  # evaluator passes the validated output
    assert run.instances["send"].output == "sent:v2"
    assert [(e.task_id, e.status) for e in run.log][:3] == [("draft", "success"), ("review", "running"), ("draft", "success")]


@pytest.mark.asyncio
async def test_evaluator_exhausted_pass_latest_and_fail_dag(make_engine, backend):
    backend.add("draft", "v1", "v2")
    backend.add("review", {"output": {"rerun": True, "reason": "bad"}}, {"output": {"rerun": True, "reason": "still bad"}})
    engine = make_engine()
    engine.register(_draft_dag("pass_latest"))
    async with engine:
        run = await engine.run("eval", timeout=5)
    assert run.status == RunStatus.SUCCESS and run.instances["send"].output == "sent:v2"
    assert "exhausted" in run.instances["review"].reason

    backend.add("draft", "v1", "v2")
    backend.add("review", {"output": {"rerun": True, "reason": "bad"}}, {"output": {"rerun": True, "reason": "still bad"}})
    engine.unregister("eval")
    engine.register(_draft_dag("fail_dag"))
    async with engine:
        run = await engine.run("eval", timeout=5)
    assert run.status == RunStatus.FAILED and "still bad" in run.instances["review"].error


@pytest.mark.asyncio
async def test_evaluator_exhausted_pause_for_user_decisions(make_engine, backend):
    backend.add("draft", "v1", "v2", "v3")
    backend.add("review", {"output": {"rerun": True, "reason": "bad"}}, {"output": {"rerun": True, "reason": "bad2"}}, {"output": {"rerun": False, "reason": "ok"}})
    engine = make_engine()
    engine.register(_draft_dag("pause_for_user"))
    async with engine:
        run = await engine.run("eval", timeout=5)
        assert run.status == RunStatus.PAUSED
        apr = engine.list_approvals()[0]
        assert apr.kind == "evaluator_exhausted" and apr.payload["latest_output"] == "v2"
        await engine.decide(apr.approval_id, "retry")
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    assert run.instances["draft"].attempt == 3 and run.instances["send"].output == "sent:v3"


@pytest.mark.asyncio
async def test_evaluator_per_item_in_fan_out(make_engine, backend):
    with Dag("evmap", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: ["a", "b"], fan_out=True)
        draft = AgenticTask("draft", prompt="write")
        review = AgenticEvaluator("review", prompt="check", max_retries=1, on_exhausted="pass_latest")
        summary = Task("summary", fn=lambda d: d)
        src >> draft >> review >> summary
    backend.add("draft", lambda r: f"draft-{r.messages[0].content[-20:]}", lambda r: "draft-b-v1", lambda r: "draft-b-v2")
    backend.add("review", lambda r: {"output": {"rerun": "draft-b-v1" in r.messages[0].content, "reason": "x"}}, lambda r: {"output": {"rerun": "draft-b-v1" in r.messages[0].content, "reason": "x"}}, {"output": {"rerun": False, "reason": "ok"}})
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("evmap", timeout=5)
    assert run.status == RunStatus.SUCCESS, run.error
    assert run.node_instances["review"] == ["review_0", "review_1"]
    assert set(run.instances["summary"].output) == {"review_0", "review_1"}


# ---------------------------------------------------------------------------
# Priority queue / chat preemption
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_preempts_between_steps(make_engine, backend):
    order = []
    gate = asyncio.Event()

    async def slow_step(req):
        order.append(("dag", req.metadata["step"]))
        gate.set()
        await asyncio.sleep(0.05)
        if req.metadata["step"] < 3:
            return GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})])
        return "done"

    def chat(req):
        order.append(("chat", req.metadata["priority"]))
        return "answer"

    backend.add("agent", slow_step, slow_step, slow_step)
    backend.add(None, chat)
    with Dag("pre", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("agent", prompt="x")
    engine = make_engine(config=EngineConfig(model="test-model", atomic="step"))
    engine.register(dag)
    async with engine:
        run = await engine.trigger("pre")
        await gate.wait()
        answer = await engine.chat("what is happening?", run_id=run.run_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert answer == "answer"
    assert order[0] == ("dag", 1) and ("chat", 0) in order
    assert order.index(("chat", 0)) < order.index(("dag", 3))  # answered before the DAG finished
    assert run.chat_log[0]["answer"] == "answer"
    assert engine.queues.get("test-model").stats()["completed_steps"] == 4


@pytest.mark.asyncio
async def test_chat_waits_for_the_running_task_then_goes_before_the_next(make_engine, backend):
    order = []
    started = asyncio.Event()

    async def first_step(req):
        order.append(("first", req.metadata["step"]))
        started.set()
        await asyncio.sleep(0.05)
        if req.metadata["step"] < 3:
            return GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})])
        return "done"

    def second(req):
        order.append(("second", req.metadata["step"]))
        return "done"

    def chat(req):
        order.append(("chat", req.metadata["priority"]))
        return "answer"

    backend.add("first", first_step, first_step, first_step)
    backend.add("second", second)
    backend.add(None, chat)
    with Dag("atomic", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("first", prompt="x", allowed_tools=["send_email"]) >> AgenticTask("second", prompt="y")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.trigger("atomic")
        await started.wait()
        answer = await engine.chat("what is happening?", run_id=run.run_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert answer == "answer" and run.status == RunStatus.SUCCESS
    # The running task kept the model through all three of its steps; the chat went before the next task.
    assert order == [("first", 1), ("first", 2), ("first", 3), ("chat", 0), ("second", 1)]


@pytest.mark.asyncio
async def test_model_queue_orders_by_priority_then_age():
    from agentic_airflow.priority import DAG_PRIORITY, USER_PRIORITY, ModelQueue

    q = ModelQueue("m", concurrency=1)
    await q.acquire(DAG_PRIORITY, 100)
    done = []

    async def waiter(name, prio, key):
        await q.acquire(prio, key)
        done.append(name)
        q.release()

    tasks = [asyncio.create_task(waiter("old-dag", DAG_PRIORITY, 1)), asyncio.create_task(waiter("new-dag", DAG_PRIORITY, 50)), asyncio.create_task(waiter("user", USER_PRIORITY, 999))]
    await asyncio.sleep(0.01)
    assert q.waiting == 3
    q.release()
    await asyncio.gather(*tasks)
    assert done == ["user", "old-dag", "new-dag"]
    with pytest.raises(RuntimeError):
        q.release()


# ---------------------------------------------------------------------------
# Triggers, persistence, resume, cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dag_trigger_and_webhook(make_engine):
    from agentic_airflow import DagTrigger, WebhookTrigger, WebhookAuthenticationError, TriggerError

    with Dag("first", prompt="p", auto_evaluate=False, schedule=[WebhookTrigger("hook", secret="s")]) as d1:
        Task("a", fn=lambda ctx: ctx.trigger["payload"])
    with Dag("second", prompt="p", auto_evaluate=False, schedule=[DagTrigger("first")]) as d2:
        Task("b", fn=lambda ctx: ctx.trigger["payload"]["outputs"])
    engine = make_engine()
    engine.register(d1)
    engine.register(d2)
    hook = d1.schedule[0]
    body = b'{"n": 7}'
    async with engine:
        with pytest.raises(WebhookAuthenticationError):
            await engine.deliver_webhook("hook", raw_body=body, signature="bad")
        with pytest.raises(TriggerError):
            await engine.deliver_webhook("missing", raw_body=body)
        runs = await engine.deliver_webhook("hook", raw_body=body, signature=hook.sign(body))
        run1 = await engine.wait(runs[0].run_id, timeout=5)
        assert run1.instances["a"].output == {"n": 7} and run1.trigger["source"] == "webhook"
        for _ in range(50):
            seconds = engine.list_runs(dag_id="second")
            if seconds and seconds[0].is_terminal:
                break
            await asyncio.sleep(0.02)
    assert seconds[0].instances["b"].output == {"a": {"n": 7}}
    assert seconds[0].trigger["source"] == "dag"
    assert engine.webhook_names() == {"hook": ["first"]}


@pytest.mark.asyncio
async def test_cron_scheduling_registers_next_fire(make_engine):
    with Dag("cron", prompt="p", auto_evaluate=False, schedule=["0 9 * * 1-5", "@hourly"]) as dag:
        Task("a", fn=lambda: 1)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        fires = engine.next_cron_fires()
    assert len(fires["cron"]) == 2


@pytest.mark.asyncio
async def test_sqlite_persistence_memory_and_resume(tmp_path, backend, tools):
    from agentic_airflow import Engine

    store = SqliteStateStore(tmp_path / "state.db")
    with Dag("persist", prompt="p", auto_evaluate=False, context={"seed": 1}, allowed_tools=[]) as dag:
        a = Task("a", fn=lambda ctx: ctx.memory)
        agent = AgenticTask("agent", prompt="x", context=[])
        a >> agent
    backend.add("agent", GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "a", "body": "b"})]), "finished")
    engine = Engine(backend, config=EngineConfig(model="m"), state=store, tools=tools)
    engine.set_memory("persist", {"remembered": True})
    engine.register(dag)
    async with engine:
        run = await engine.run("persist", timeout=5)
        assert run.status == RunStatus.PAUSED
        assert run.instances["a"].output == {"seed": 1, "remembered": True}
        apr_id = engine.list_approvals()[0].approval_id
    engine.close()

    # a fresh engine (new process) resumes from the store
    engine2 = Engine(backend, config=EngineConfig(model="m"), state=store, tools=tools)
    engine2.register(dag)
    async with engine2:
        assert [a.approval_id for a in engine2.list_approvals()] == [apr_id]
        resumed = await engine2.resume_interrupted()
        assert [r.run_id for r in resumed] == [run.run_id]
        await engine2.approve(apr_id)
        run2 = await engine2.wait(run.run_id, timeout=5)
    assert run2.status == RunStatus.SUCCESS, run2.error
    assert run2.instances["agent"].output == "finished"
    assert store.get_run(run.run_id)["status"] == "success"
    assert [r["run_id"] for r in store.list_runs(dag_id="persist")] == [run.run_id]
    engine2.close()
    store.close()


def test_sqlite_store_keeps_its_schema_version(tmp_path, monkeypatch):
    import sqlite3

    from agentic_airflow import state as state_module
    from agentic_airflow.errors import ConfigurationError
    from agentic_airflow.run import DagRun

    db = tmp_path / "state.db"
    old = sqlite3.connect(db)  # a file from before versions: user_version 0
    old.execute("CREATE TABLE documents (collection TEXT NOT NULL, key TEXT NOT NULL, scope TEXT, scope2 TEXT, status TEXT, created_at REAL, updated_at REAL NOT NULL, value TEXT NOT NULL, PRIMARY KEY (collection, key))")
    old.close()
    store = SqliteStateStore(db)
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == state_module.SCHEMA
    # a run kept by another version: a log entry with a field this one doesn't know, and one without some it does
    run = {"run_id": "r", "dag_id": "d", "log": [{"timestamp": 1.0, "task_id": "a", "instance_id": "a", "status": "success", "later": 1}, {"task_id": "b"}]}
    store.save_run(run)
    assert [e.task_id for e in DagRun.from_dict(store.get_run("r")).log] == ["a", "b"]
    store.close()

    # an upgrade step runs once, in order; a file from a newer version is refused, not misread
    ran = []
    monkeypatch.setattr(state_module, "SCHEMA", 2)
    monkeypatch.setitem(state_module.MIGRATIONS, 2, lambda conn: ran.append(2))
    SqliteStateStore(db).close()
    SqliteStateStore(db).close()
    assert ran == [2]
    monkeypatch.setattr(state_module, "SCHEMA", 1)
    with pytest.raises(ConfigurationError, match="newer agentic-airflow"):
        SqliteStateStore(db)


@pytest.mark.asyncio
async def test_cancel_run(make_engine):
    started = asyncio.Event()

    async def slow():
        started.set()
        await asyncio.sleep(10)

    with Dag("cancel", prompt="p", auto_evaluate=False) as dag:
        Task("slow", fn=slow)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.trigger("cancel")
        await started.wait()
        await engine.cancel(run.run_id)
    assert run.status == RunStatus.CANCELLED and run.instances["slow"].error == "cancelled"


@pytest.mark.asyncio
async def test_max_active_runs_queues_runs(make_engine):
    release = asyncio.Event()

    async def wait_for_release():
        await release.wait()
        return 1

    with Dag("serial", prompt="p", auto_evaluate=False, max_active_runs=1) as dag:
        Task("w", fn=wait_for_release)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        r1 = await engine.trigger("serial")
        r2 = await engine.trigger("serial")
        await asyncio.sleep(0.02)
        assert r1.status == RunStatus.RUNNING and r2.status == RunStatus.PENDING
        release.set()
        await engine.wait(r1.run_id, timeout=5)
        await engine.wait(r2.run_id, timeout=5)
    assert r2.status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_events_and_chat_sink_respect_verbosity(make_engine, backend):
    sink, all_events = [], []
    with Dag("verb", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("quiet", prompt="x", verbosity=0)
        AgenticTask("loud", prompt="y", verbosity=2)
    backend.add("quiet", GenerationResponse(text="q", reasoning="thinking..."))
    backend.add("loud", GenerationResponse(text="l", reasoning="thinking..."))
    engine = make_engine(chat_sink=lambda e: sink.append((e.task_id, e.type)))
    engine.subscribe(lambda e: all_events.append(e.type))
    engine.register(dag)
    async with engine:
        await engine.run("verb", timeout=5)
    assert all(t == "loud" for t, _ in sink)
    assert {"task.started", "reasoning", "response", "task.finished"} <= {e for _, e in sink}
    assert "run.finished" in all_events and "task.finished" in all_events


@pytest.mark.asyncio
async def test_resolved_config_and_stats(make_engine):
    with Dag("cfg", prompt="p", auto_evaluate=False, skills=["A"], hidden_tools=["send_email"]) as dag:
        AgenticTask("t", prompt="x", denied_tools=["delete_file"], thinking="on")
    engine = make_engine()
    engine.register(dag)
    data = engine.resolved_config("cfg", "t")
    assert data["model"] == "test-model" and data["thinking"] is True and data["skills"] == ["A"]
    assert data["tools"] == {"allowed": [], "gated": [], "hidden": ["send_email", "web_search"], "denied": ["delete_file"]}
    assert engine.stats()["dags"] == ["cfg"]
