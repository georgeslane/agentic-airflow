"""Tasks tried again after they fail, cron firings caught up after the engine was stopped, and tasks that ask a
human a question."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agentic_airflow import (
    AgenticTask,
    Dag,
    EngineConfig,
    GenerationResponse,
    PermanentError,
    RunStatus,
    SqliteStateStore,
    Task,
    TaskGroup,
    TaskStatus,
    ToolCall,
)
from agentic_airflow.config import ScopeConfig, resolve_config
from agentic_airflow.engine import CRON_KEY

# -- retries -----------------------------------------------------------------------------------------------


def _raises(exc):
    def respond(request):
        raise exc

    return respond


def test_retries_cascade_like_other_settings():
    assert resolve_config([ScopeConfig()]).retries == 0 and resolve_config([ScopeConfig()]).retry_delay == 60.0
    cfg = resolve_config([ScopeConfig(retries=2, retry_delay=5), ScopeConfig(retries=0), ScopeConfig(ask_user=True)])
    assert (cfg.retries, cfg.retry_delay, cfg.ask_user) == (0, 5.0, True)
    for bad in ({"retries": -1}, {"retries": True}, {"retry_delay": -1}, {"ask_user": "yes"}):
        with pytest.raises(ValueError):
            ScopeConfig(**bad)


@pytest.mark.asyncio
async def test_a_failed_task_is_tried_again_and_told_why(make_engine, backend):
    calls = {"fn": 0}

    def flaky():
        calls["fn"] += 1
        if calls["fn"] == 1:
            raise ConnectionError("the site timed out")
        return "fetched"

    with Dag("retry", prompt="p", auto_evaluate=False, retries=1, retry_delay=0) as dag:
        fetch = Task("fetch", fn=flaky)
        sum_up = AgenticTask("sum_up", prompt="Sum it up.")
        fetch >> sum_up
    backend.add("sum_up", _raises(RuntimeError("the model fell over")), "Summed up.")
    engine = make_engine()
    engine.register(dag)
    events = []
    engine.subscribe(lambda e: events.append(e.type))
    async with engine:
        run = await engine.run("retry", timeout=5)
    assert run.status == RunStatus.SUCCESS
    fetch_, sum_ = run.instances["fetch"], run.instances["sum_up"]
    assert fetch_.attempt == 2 and fetch_.output == "fetched"
    assert fetch_.retry_history == [
        {"attempt": 1, "output": None, "reason": "ConnectionError: the site timed out", "failed": True, "tool_calls": []}
    ]
    assert sum_.attempt == 2 and sum_.output == "Summed up."
    retried = backend.requests[-1].messages[0].content
    assert "## Previous attempts failed" in retried and "### Attempt 1 failed\nRuntimeError: the model fell over" in retried
    assert events.count("task.retrying") == 2 and "task.failed" not in events
    # The run log has each failed attempt, then the one that worked.
    assert [(e["task_id"], e["status"], e["attempt"]) for e in run.dag_context] == [
        ("fetch", "failed", 1), ("fetch", "success", 2), ("sum_up", "failed", 1), ("sum_up", "success", 2)
    ]


@pytest.mark.asyncio
async def test_a_retry_is_told_which_tools_it_had_called(make_engine, backend):
    with Dag("tools", prompt="p", auto_evaluate=False, allowed_tools=["send_email"], retries=1, retry_delay=0) as dag:
        AgenticTask("mail", prompt="Email Ann.")
    backend.add(
        "mail",
        GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "ann", "body": "hi"})]),
        _raises(RuntimeError("lost the connection")),
        "Done.",
    )
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("tools", timeout=5)
    assert run.status == RunStatus.SUCCESS and run.instances["mail"].retry_history[0]["tool_calls"] == ["send_email"]
    assert "Before failing it had called: send_email." in backend.requests[-1].messages[0].content


@pytest.mark.asyncio
async def test_failures_that_cant_be_helped_and_tasks_without_retries_fail_at_once(make_engine, backend):
    def broken():
        raise PermanentError("the server's gone")

    with Dag("once", prompt="p", auto_evaluate=False, retry_delay=0) as dag:
        with TaskGroup("g", retries=3):
            Task("permanent", fn=broken)
        Task("plain", fn=lambda: 1 / 0)  # no retries: the default
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("once", timeout=5)
    assert run.status == RunStatus.FAILED
    assert run.instances["permanent"].attempt == 1 and run.instances["plain"].attempt == 1


@pytest.mark.asyncio
async def test_cancelling_a_run_waiting_to_retry_stops_it(make_engine, backend):
    with Dag("slow", prompt="p", auto_evaluate=False, retries=1, retry_delay=30) as dag:
        Task("fails", fn=lambda: 1 / 0)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.trigger("slow")
        for _ in range(100):
            if run.instances.get("fails") and run.instances["fails"].attempt == 2:
                break
            await asyncio.sleep(0.01)
        assert run.instances["fails"].status == TaskStatus.QUEUED  # waiting to go again
        await engine.cancel(run.run_id)
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.CANCELLED and run.instances["fails"].error == "cancelled"


# -- catching up --------------------------------------------------------------------------------------------


def _due(engine, dag_id, expression, when):
    engine.state.set_meta(CRON_KEY, {dag_id: {expression: when.isoformat()}})


@pytest.mark.asyncio
async def test_a_missed_cron_firing_runs_once_when_the_engine_starts(tmp_path, make_engine, backend):
    backend.default = "ok"
    db = tmp_path / "s.db"
    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    store = SqliteStateStore(db)
    store.set_meta(CRON_KEY, {"caught": {"*/15 * * * *": hour_ago.isoformat()}, "skipped": {"*/15 * * * *": hour_ago.isoformat()}})
    store.close()
    engine = make_engine(state=SqliteStateStore(db))
    with Dag("caught", prompt="p", auto_evaluate=False, catchup=True, schedule=["*/15 * * * *"]) as caught:
        AgenticTask("a", prompt="x")
    with Dag("skipped", prompt="p", auto_evaluate=False, schedule=["*/15 * * * *"]) as skipped:
        AgenticTask("a", prompt="x")
    with Dag("new", prompt="p", auto_evaluate=False, catchup=True, schedule=["*/15 * * * *"]) as new:
        AgenticTask("a", prompt="x")
    for dag in (caught, skipped, new):
        engine.register(dag)
    async with engine:
        for _ in range(100):
            if engine.list_runs("caught"):
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        runs = engine.list_runs()
    assert [r.dag_id for r in runs] == ["caught"]  # once, though four firings were missed
    trigger = runs[0].trigger
    assert trigger["catchup"] is True and trigger["source"] == "cron"
    missed = datetime.fromisoformat(trigger["scheduled_for"])
    assert datetime.now(timezone.utc) - timedelta(minutes=15) < missed <= datetime.now(timezone.utc)  # the latest
    # What came due is kept, so the next start doesn't catch up again; a new DAG starts from now.
    due = engine.state.get_meta(CRON_KEY)
    assert due["caught"]["*/15 * * * *"] == trigger["scheduled_for"] and "*/15 * * * *" in due["new"]
    engine.close()

    again = make_engine(state=SqliteStateStore(db))
    again.register(caught)
    async with again:
        await asyncio.sleep(0.2)
        assert len(again.list_runs("caught")) == 1
    again.unregister("caught")
    assert "caught" not in again.state.get_meta(CRON_KEY)


@pytest.mark.asyncio
async def test_firings_due_while_paused_arent_caught_up(make_engine, backend):
    engine = make_engine()
    with Dag("paused", prompt="p", auto_evaluate=False, catchup=True, schedule=["* * * * *"]) as dag:
        AgenticTask("a", prompt="x")
    engine.register(dag)
    _due(engine, "paused", "* * * * *", datetime.now(timezone.utc) - timedelta(minutes=5))
    engine.pause("paused")
    async with engine:
        await asyncio.sleep(0.2)
    assert engine.list_runs() == []
    assert datetime.fromisoformat(engine.state.get_meta(CRON_KEY)["paused"]["* * * * *"]) > datetime.now(timezone.utc) - timedelta(minutes=5)


# -- asking --------------------------------------------------------------------------------------------------


def _asks(question="Apply to Acme?", choices=("Yes", "No")):
    return GenerationResponse(tool_calls=[ToolCall(name="ask_user", arguments={"question": question, "choices": list(choices)})])


@pytest.mark.asyncio
async def test_a_task_asks_a_question_and_carries_on_with_the_answer(make_engine, backend):
    with Dag("jobs", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("apply", prompt="Apply if I want to.", ask_user=True)
        AgenticTask("other", prompt="x")
    backend.add("apply", _asks(), "Applied to Acme.")
    backend.add("other", "ok")
    engine = make_engine()
    engine.register(dag)
    events = []
    engine.subscribe(lambda e: events.append(e))
    async with engine:
        run = await engine.trigger("jobs")
        for _ in range(200):
            if engine.list_approvals():
                break
            await asyncio.sleep(0.01)
        [question] = engine.list_approvals()
        assert question.kind == "question" and question.payload["question"] == "Apply to Acme?"
        assert question.payload["choices"] == ["Yes", "No"]
        assert run.instances["apply"].status == TaskStatus.AWAITING_USER
        asked = next(e for e in events if e.type == "task.awaiting_user")
        assert asked.payload["approval_id"] == question.approval_id
        with pytest.raises(Exception, match="An answer is needed"):
            await engine.answer(question.approval_id, " ")
        await engine.answer(question.approval_id, "Yes, but mention Python")
        run = await engine.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.SUCCESS and run.instances["apply"].output == "Applied to Acme."
    told = [m for m in backend.requests[-1].messages if m.role == "tool"]
    assert told[-1].content == "The user answered: Yes, but mention Python"
    assert run.instances["apply"].tool_calls[-1]["status"] == "answered"
    assert engine.get_approval(question.approval_id).status == "answered"
    offered = {t.name for r in backend.requests if r.metadata["task_id"] == "apply" for t in r.tools}
    other = {t.name for r in backend.requests if r.metadata["task_id"] == "other" for t in r.tools}
    assert "ask_user" in offered and "ask_user" not in other  # only where it's allowed


@pytest.mark.asyncio
async def test_an_unanswered_question_expires_and_the_task_decides_for_itself(make_engine, backend):
    with Dag("jobs", prompt="p", auto_evaluate=False, ask_user=True) as dag:
        AgenticTask("apply", prompt="Apply if I want to.")
    backend.add("apply", _asks(choices=()), "I didn't apply.")
    engine = make_engine(config=EngineConfig(model="test-model", approval_timeout=0.2))
    engine.register(dag)
    async with engine:
        run = await engine.trigger("jobs")
        run = await engine.wait(run.run_id, timeout=5, stop_on_pause=False)
    assert run.status == RunStatus.SUCCESS
    told = [m for m in backend.requests[-1].messages if m.role == "tool"][-1].content
    assert told.startswith("No answer: No answer within 0.2 seconds. Decide for yourself")
    assert run.instances["apply"].tool_calls[-1]["status"] == "unanswered"


@pytest.mark.asyncio
async def test_a_task_without_ask_user_cant_ask(make_engine, backend):
    with Dag("jobs", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("apply", prompt="Apply.")
    backend.add("apply", _asks(), "Decided myself.")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("jobs", timeout=5)
    assert run.status == RunStatus.SUCCESS and engine.list_approvals(status=None) == []
    told = [m for m in backend.requests[-1].messages if m.role == "tool"][-1].content
    assert told.startswith("Error: ask_user isn't available to this task")
