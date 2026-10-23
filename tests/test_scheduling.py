"""How runs queue for a model, pausing, cron coalescing, DAG definitions changing mid-run, and pruning."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agentic_airflow import AgenticTask, Dag, EngineConfig, RunStatus, SqliteStateStore, Task


def _dag(dag_id, model=None, **kwargs):
    with Dag(dag_id, prompt="p", auto_evaluate=False, model=model, **kwargs) as dag:
        AgenticTask("work", prompt="x")
    return dag


@pytest.mark.asyncio
async def test_runs_on_the_same_model_queue_first_come_first_served(make_engine, backend):
    order = []

    async def answer(req):
        name = req.metadata["dag_id"]
        order.append(f"{name} start")
        await asyncio.sleep(0.05)
        order.append(f"{name} end")
        return "ok"

    backend.default = answer
    engine = make_engine()
    for dag_id in ("a", "b"):
        engine.register(_dag(dag_id))
    engine.register(_dag("c", model="other-model"))
    async with engine:
        first = await engine.trigger("a")
        second = await engine.trigger("b")
        third = await engine.trigger("c")
        assert second.status == RunStatus.PENDING  # waits for "a", on the same model
        assert third.status == RunStatus.RUNNING  # another model: it goes straight away
        for run in (first, second, third):
            await engine.wait(run.run_id, timeout=5)
    assert order.index("a end") < order.index("b start")
    assert order.index("c start") < order.index("a end")


@pytest.mark.asyncio
async def test_a_paused_dag_only_runs_when_triggered_by_hand(make_engine, backend):
    backend.default = "ok"
    engine = make_engine()
    engine.register(_dag("p"))
    with Dag("listener", prompt="p", auto_evaluate=False, schedule=["@hourly"]) as dag:
        AgenticTask("work", prompt="x")
    engine.register(dag)
    async with engine:
        engine.pause("listener")
        assert engine.is_paused("listener")
        assert await engine.update_asset("anything") == []
        run = await engine.run("listener", timeout=5)  # by hand: still runs
        assert run.status == RunStatus.SUCCESS
        engine.unpause("listener")
        assert not engine.is_paused("listener")


@pytest.mark.asyncio
async def test_pausing_cancels_runs_waiting_to_start_but_not_manual_ones(make_engine, backend):
    gate = asyncio.Event()

    async def hold(req):
        await gate.wait()
        return "ok"

    backend.default = hold
    engine = make_engine()
    engine.register(_dag("busy"))
    engine.register(_dag("q"))
    async with engine:
        busy = await engine.trigger("busy")
        queued = await engine.trigger("q", source="webhook")
        manual = await engine.trigger("q")
        engine.pause("q")
        assert engine.get_run(queued.run_id).status == RunStatus.CANCELLED
        gate.set()
        await engine.wait(busy.run_id, timeout=5)
        assert (await engine.wait(manual.run_id, timeout=5)).status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_cron_doesnt_queue_a_run_while_one_is_still_waiting(make_engine, backend):
    gate = asyncio.Event()

    async def hold(req):
        await gate.wait()
        return "ok"

    backend.default = hold
    engine = make_engine()
    engine.register(_dag("busy"))
    with Dag("hourly", prompt="p", auto_evaluate=False, schedule=["@hourly"]) as dag:
        AgenticTask("work", prompt="x")
    engine.register(dag)
    async with engine:
        await engine.trigger("busy")  # holds the model, so "hourly" waits
        past = datetime.now(timezone.utc) - timedelta(minutes=1)
        for _ in range(3):  # three firings while the first is still waiting
            engine._cron_next[("hourly", 0)] = past
            engine._tick.set()
            await asyncio.sleep(0.1)
        waiting = [r for r in engine.runs.values() if r.dag_id == "hourly"]
        assert len(waiting) == 1 and waiting[0].status == RunStatus.PENDING
        gate.set()


@pytest.mark.asyncio
async def test_a_run_keeps_its_dag_when_it_changes_and_waiting_runs_get_the_new_one(make_engine, backend):
    gate = asyncio.Event()

    async def first(req):
        await gate.wait()
        return "old"

    backend.add("work", first)
    backend.add("new_work", "new")
    engine = make_engine()
    engine.register(_dag("d"))
    async with engine:
        running = await engine.trigger("d")
        await asyncio.sleep(0.05)
        waiting = await engine.trigger("d")
        with Dag("d", prompt="p", auto_evaluate=False) as changed:
            AgenticTask("new_work", prompt="y")
        engine.register(changed, replace=True)
        gate.set()
        running = await engine.wait(running.run_id, timeout=5)
        waiting = await engine.wait(waiting.run_id, timeout=5)
    assert running.status == RunStatus.SUCCESS and running.outputs() == {"work": "old"}
    assert waiting.status == RunStatus.SUCCESS and waiting.outputs() == {"new_work": "new"}


@pytest.mark.asyncio
async def test_unregistering_cancels_runs_waiting_to_start(make_engine, backend):
    gate = asyncio.Event()

    async def hold(req):
        await gate.wait()
        return "ok"

    backend.default = hold
    engine = make_engine()
    engine.register(_dag("busy"))
    engine.register(_dag("gone"))
    async with engine:
        await engine.trigger("busy")
        run = await engine.trigger("gone")
        engine.unregister("gone")
        run = engine.get_run(run.run_id)
        assert run.status == RunStatus.CANCELLED and "unregistered" in run.error
        gate.set()


@pytest.mark.asyncio
async def test_resuming_a_run_whose_dag_changed_fails_it(tmp_path, make_engine, backend):
    gate = asyncio.Event()

    async def hold(req):
        await gate.wait()
        return "ok"

    backend.default = hold
    db = tmp_path / "s.db"
    engine = make_engine(state=SqliteStateStore(db))
    engine.register(_dag("d"))
    async with engine:
        run = await engine.trigger("d")
        await asyncio.sleep(0.05)
    # Restart with a DAG that has other tasks.
    engine2 = make_engine(state=SqliteStateStore(db))
    with Dag("d", prompt="p", auto_evaluate=False) as changed:
        AgenticTask("other", prompt="y")
    engine2.register(changed)
    async with engine2:
        await engine2.resume_interrupted()
        run = await engine2.wait(run.run_id, timeout=5)
    assert run.status == RunStatus.FAILED and "changed" in run.error


@pytest.mark.asyncio
async def test_old_runs_are_pruned(make_engine, backend):
    backend.default = "ok"
    engine = make_engine(config=EngineConfig(model="test-model", keep_runs=2))
    with Dag("many", prompt="p", auto_evaluate=False) as dag:
        Task("t", fn=lambda: 1)
    engine.register(dag)
    async with engine:
        ids = [(await engine.run("many", timeout=5)).run_id for _ in range(4)]
    kept = [r.run_id for r in engine.list_runs("many")]
    assert sorted(kept) == sorted(ids[2:])
    assert engine.runs == {}  # finished runs aren't kept in memory


@pytest.mark.asyncio
async def test_user_slot_waits_for_the_running_task(make_engine, backend):
    order = []
    started = asyncio.Event()

    async def task(req):
        started.set()
        await asyncio.sleep(0.05)
        order.append("task")
        return "ok"

    backend.add("work", task)
    engine = make_engine()
    engine.register(_dag("u"))
    async with engine:
        run = await engine.trigger("u")
        await started.wait()
        async with engine.user_slot("test-model"):
            order.append("user")
        await engine.wait(run.run_id, timeout=5)
    assert order == ["task", "user"]


@pytest.mark.asyncio
async def test_cancelling_a_run_that_has_only_just_started_cancels_it(make_engine, backend):
    gate = asyncio.Event()

    async def hold(req):
        await gate.wait()
        return "ok"

    backend.default = hold
    engine = make_engine()
    engine.register(_dag("d"))
    async with engine:
        first = await engine.trigger("d")
        second = await engine.trigger("d")  # waits for the first
        await asyncio.sleep(0.05)
        await engine.cancel(first.run_id)  # the second starts...
        await engine.cancel(second.run_id)  # ...and is cancelled before its driver has run
        assert engine.get_run(second.run_id).status == RunStatus.CANCELLED
        assert engine.runs == {}
        gate.set()
