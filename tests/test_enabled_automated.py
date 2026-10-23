"""enabled and automated: the tools a task can use at all, and which of those that ask first run without asking."""

import pytest
from test_tool_approval import _registry

from agentic_airflow import (
    AgenticTask,
    Dag,
    GenerationResponse,
    RunStatus,
    TaskGroup,
    ToolCall,
    resolve_permissions,
)
from agentic_airflow.config import ScopeConfig, resolve_config


def _perms(*scopes):
    return resolve_permissions(resolve_config([ScopeConfig(**s) for s in scopes]), _registry())


def test_a_task_can_use_only_the_tools_enabled_for_it():
    none = _perms({"enabled": []})
    assert none.offered == frozenset() and none.refuse_denied
    reading = _perms({"enabled": ["read_*"]})
    assert reading.allowed == {"read_inbox"} and reading.denied == {"send_email", "push"}  # it doesn't ask first
    every = _perms({"enabled": "*"})
    assert every.allowed == {"read_inbox"} and every.gated == {"send_email", "push"} and every.always == {"push"}


def test_automated_tools_run_without_asking_but_automating_one_doesnt_enable_it():
    assert _perms({"enabled": "*", "automated": ["send_email"]}).allowed == {"read_inbox", "send_email"}
    everything = _perms({"enabled": "*", "automated": "*"})
    assert everything.allowed == {"read_inbox", "send_email"} and everything.gated == {"push"}  # push always asks
    assert _perms({"enabled": [], "automated": "*"}).offered == frozenset()


def test_the_nearest_list_decides():
    dag, task = {"enabled": "*", "automated": ["send_email"]}, {"enabled": ["send_email"]}
    perms = _perms(dag, task)
    assert perms.allowed == {"send_email"} and perms.denied == {"read_inbox", "push"}  # its own list, the DAG's automated
    assert _perms(dag, {"automated": []}).gated == {"send_email", "push"}


def test_a_tool_tagged_explicit_is_enabled_only_by_its_name():
    from agentic_airflow import tool

    registry = _registry()

    @tool(description="For one task alone", tags=("explicit",))
    def propose(text: str) -> str:
        return text

    registry.register(propose)
    scope = lambda enabled: resolve_config([ScopeConfig(enabled=enabled)])  # noqa: E731
    assert "propose" not in resolve_permissions(scope("*"), registry).offered
    assert "propose" not in resolve_permissions(scope(["prop*"]), registry).offered
    assert "propose" in resolve_permissions(scope(["propose"]), registry).offered


def test_hidden_tools_are_found_only_among_those_enabled():
    perms = _perms({"enabled": ["send_email"], "hidden_tools": ["*"]})
    assert perms.hidden == {"send_email"} and perms.visible == frozenset()


def test_a_list_is_a_list_or_star():
    assert ScopeConfig(enabled="*").enabled == ["*"]
    with pytest.raises(TypeError):
        ScopeConfig(automated="send_email")


def test_without_enabled_permissions_are_as_they_were():
    assert not _perms({"allowed_tools": ["read_inbox"]}).refuse_denied


@pytest.mark.asyncio
async def test_calling_a_tool_that_isnt_enabled_fails_and_the_task_carries_on(make_engine, backend):
    backend.add(
        "notify",
        GenerationResponse(tool_calls=[ToolCall(name="read_inbox", arguments={}, id="c1")]),
        "Nothing new.",
    )
    with Dag("checks", prompt="p", auto_evaluate=False, enabled=["read_inbox"]) as dag:
        AgenticTask("notify", prompt="Tell me what's new.", enabled=[])
    engine = make_engine(tools=_registry())
    engine.register(dag)
    async with engine:
        run = await engine.run("checks", timeout=5)
    assert run.status == RunStatus.SUCCESS and engine.list_approvals() == []  # no one was asked
    [call] = run.instances["notify"].tool_calls
    assert call["status"] == "error" and call["result"].startswith("Error: read_inbox isn't one of the tools")
    assert not backend.requests[0].tools  # it was offered none


@pytest.mark.asyncio
async def test_approving_a_tool_for_the_run_counts_only_where_its_enabled(make_engine, backend):
    send = lambda i: GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "ann"}, id=f"c{i}")])  # noqa: E731
    backend.add("first", send(1), "sent")
    backend.add("second", send(2), "tried")
    with Dag("mail", prompt="p", auto_evaluate=False, enabled=[]) as dag:
        with TaskGroup("g"):
            first = AgenticTask("first", prompt="Email Ann.", enabled=["send_email"])
            second = AgenticTask("second", prompt="Report.")
            first >> second
    engine = make_engine(tools=_registry())
    engine.register(dag)
    async with engine:
        run = await engine.run("mail", timeout=5)
        await engine.approve(engine.list_approvals()[0].approval_id, remember=True)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS and run.approved_tools == ["send_email"]
    assert run.instances["first"].tool_calls[0]["result"] == "sent to ann"
    assert run.instances["second"].tool_calls[0]["status"] == "error"  # remembered, but not one of its tools
