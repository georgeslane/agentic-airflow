"""Tools that need a human's OK, the context passed to tools, and how long a task and an approval may take."""

import asyncio

import pytest

from agentic_airflow import (
    AgenticTask,
    Dag,
    EngineConfig,
    GenerationResponse,
    RunStatus,
    TaskStatus,
    ToolCall,
    ToolRegistry,
    resolve_permissions,
    tool,
)
from agentic_airflow.config import ScopeConfig, resolve_config


def _registry():
    registry = ToolRegistry()

    @tool(description="Read the inbox")
    def read_inbox() -> str:
        return "3 emails"

    @tool(description="Send an email", approval="ask")
    def send_email(to: str) -> str:
        return f"sent to {to}"

    @tool(description="Push to GitHub", approval="always")
    def push(branch: str) -> str:
        return f"pushed {branch}"

    for t in (read_inbox, send_email, push):
        registry.register(t)
    return registry


def _perms(**scope):
    return resolve_permissions(resolve_config([ScopeConfig(**scope)]), _registry())


def test_tools_that_ask_are_offered_but_gated_unless_named():
    perms = _perms()
    assert perms.allowed == {"read_inbox"}
    assert perms.gated == {"send_email", "push"} and perms.visible == {"read_inbox", "send_email", "push"}
    assert perms.state("send_email") == "gated"
    named = _perms(allowed_tools=["read_inbox", "send_email", "push"])
    assert named.allowed == {"read_inbox", "send_email"}  # "always" asks whatever allowed_tools says
    assert named.gated == {"push"} and named.always == {"push"}
    narrow = _perms(allowed_tools=["send_email"])
    assert narrow.allowed == {"send_email"} and narrow.denied == {"read_inbox"}
    denied = _perms(denied_tools=["push"])
    assert "push" not in denied.visible and denied.state("push") == "denied"


def test_tool_lists_take_patterns():
    perms = _perms(allowed_tools=["read_*", "send_*"], hidden_tools=["send*"])
    assert perms.allowed == {"read_inbox", "send_email"} and perms.hidden == {"send_email"}
    assert _perms(denied_tools=["*"]).visible == frozenset()


def test_ctx_parameter_is_not_part_of_the_schema():
    @tool
    def where(query: str, ctx) -> str:
        return ""

    assert where.parameters["properties"] == {"query": {"type": "string"}} and where.wants_ctx


@pytest.mark.asyncio
async def test_gated_tool_pauses_then_runs_once_approved(make_engine, backend):
    backend.add(
        "mail",
        GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "ann"}, id="c1")]),
        "done",
    )
    with Dag("gated", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("mail", prompt="Email Ann.")
    engine = make_engine(tools=_registry())
    engine.register(dag)
    async with engine:
        run = await engine.run("gated", timeout=5)
        assert run.status == RunStatus.PAUSED
        request = backend.requests[0]
        assert {t.name for t in request.tools} == {"read_inbox", "send_email", "push"}
        assert "need a human's approval" in request.system
        [approval] = engine.list_approvals()
        assert approval.payload["reason"] == "requires_approval"
        await engine.approve(approval.approval_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert run.instances["mail"].tool_calls[0]["result"] == "sent to ann"


@pytest.mark.asyncio
async def test_always_tools_ask_every_time_even_when_remembered(make_engine, backend):
    call = lambda i: GenerationResponse(tool_calls=[ToolCall(name="push", arguments={"branch": "b"}, id=f"c{i}")])  # noqa: E731
    backend.add("ship", call(1), call(2), "done")
    with Dag("always", prompt="p", auto_evaluate=False, allowed_tools=["push", "read_inbox"]) as dag:
        AgenticTask("ship", prompt="Push twice.")
    engine = make_engine(tools=_registry())
    engine.register(dag)
    async with engine:
        run = await engine.run("always", timeout=5)
        await engine.approve(engine.list_approvals()[0].approval_id, remember=True)
        run = await engine.wait(run.run_id, timeout=5)
        assert run.status == RunStatus.PAUSED  # remembering doesn't count for an "always" tool
        assert run.approved_tools == []
        await engine.approve(engine.list_approvals()[0].approval_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_tools_get_who_called_them(make_engine, backend):
    seen = []
    registry = ToolRegistry()

    @tool
    async def note(text: str, ctx) -> str:
        seen.append((text, ctx.dag_id, ctx.task_id, ctx.instance_id, ctx.run is not None))
        return "ok"

    registry.register(note)
    backend.add("t", GenerationResponse(tool_calls=[ToolCall(name="note", arguments={"text": "hi"})]), "done")
    with Dag("ctx", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("t", prompt="x")
    engine = make_engine(tools=registry)
    engine.register(dag)
    async with engine:
        run = await engine.run("ctx", timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert seen == [("hi", "ctx", "t", "t", True)]


@pytest.mark.asyncio
async def test_unanswered_approval_is_rejected_after_the_timeout(make_engine, backend):
    backend.add(
        "mail",
        GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "ann"}, id="c1")]),
        lambda req: "gave up" if "No answer within" in req.messages[-1].content else "?",
    )
    with Dag("expire", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("mail", prompt="Email Ann.")
    engine = make_engine(tools=_registry(), config=EngineConfig(model="test-model", approval_timeout=0.2))
    engine.register(dag)
    async with engine:
        run = await engine.run("expire", timeout=5)
        assert run.status == RunStatus.PAUSED
        run = await engine.wait(run.run_id, timeout=5, stop_on_pause=False)
    assert run.status == RunStatus.SUCCESS and run.instances["mail"].output == "gave up"
    assert run.instances["mail"].tool_calls[0]["status"] == "rejected"


@pytest.mark.asyncio
async def test_task_timeout_fails_a_task_that_works_too_long(make_engine, backend):
    async def slow(req):
        await asyncio.sleep(0.15)
        return GenerationResponse(tool_calls=[ToolCall(name="read_inbox", arguments={})])

    backend.add("slow", slow, slow, slow)
    with Dag("timeout", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("slow", prompt="x")
    engine = make_engine(tools=_registry(), config=EngineConfig(model="test-model", task_timeout=0.2))
    engine.register(dag)
    async with engine:
        run = await engine.run("timeout", timeout=5)
    assert run.status == RunStatus.FAILED
    assert run.instances["slow"].status == TaskStatus.FAILED and "task_timeout" in run.instances["slow"].error


@pytest.mark.asyncio
async def test_tool_list_only_grows_as_tools_are_discovered(make_engine, backend):
    registry = _registry()

    @tool(description="Find news about a topic", hidden=True)
    def news(topic: str) -> str:
        return "news"

    registry.register(news)
    backend.add(
        "t",
        GenerationResponse(tool_calls=[ToolCall(name="search_tools", arguments={"query": "news"})]),
        "done",
    )
    with Dag("order", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("t", prompt="x")
    engine = make_engine(tools=registry)
    engine.register(dag)
    async with engine:
        await engine.run("order", timeout=5)
    first, second = ([t.name for t in r.tools] for r in backend.requests)
    assert first == ["push", "read_inbox", "send_email", "search_tools"]
    assert second == ["push", "read_inbox", "send_email", "news"]  # the new one goes on the end
