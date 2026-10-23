"""Assets (outlets), mapping over what started a run, branches that select nothing, and skip_if."""

import pytest

from agentic_airflow import (
    AgenticBranch,
    AgenticEvaluator,
    AgenticTask,
    Asset,
    Dag,
    DagEvaluator,
    DagTrigger,
    DagValidationError,
    EngineConfig,
    RunStatus,
    Task,
    TriggerError,
    found_nothing,
    trigger_items,
)


@pytest.mark.asyncio
async def test_outlets_start_the_dags_scheduled_on_the_asset_with_what_was_produced(make_engine, backend):
    backend.add("find", '["job a", "job b"]')
    backend.default = lambda req: f"applied: {req.messages[0].content.split('<<<BEGIN items>>>')[1].split('<<<END')[0].strip()}"
    with Dag("finder", prompt="p", auto_evaluate=False) as finder:
        AgenticTask("find", prompt="Find jobs.", return_type="list", fan_out=False, outlets=[Asset("jobs")])
    with Dag("applier", prompt="p", auto_evaluate=False, schedule=[Asset("jobs")]) as applier:
        items = Task("items", fn=trigger_items, fan_out=True)
        items >> AgenticTask("apply", prompt="Apply.")
    engine = make_engine()
    engine.register(finder)
    engine.register(applier)
    events = []
    engine.subscribe(lambda e: events.append(e.type))
    async with engine:
        await engine.run("finder", timeout=5)
        [run] = [r for r in engine.list_runs("applier")] or [None]
        run = await engine.wait(run.run_id, timeout=5)
    assert run.trigger["source"] == "asset" and run.trigger["asset"] == "jobs"
    assert run.trigger["payload"] == ["job a", "job b"] and run.trigger["chain"] == ["finder"]
    assert run.trigger["from_task"] == "find"
    assert run.outputs()["apply_0"] == 'applied: job a' and run.outputs()["apply_1"] == 'applied: job b'
    assert "node.finished" in events and "asset.updated" in events


@pytest.mark.asyncio
async def test_a_mapped_node_updates_its_asset_with_the_list_of_outputs_and_nothing_starts_nothing(make_engine, backend):
    backend.default = "ok"
    with Dag("producer", prompt="p", auto_evaluate=False) as producer:
        src = Task("src", fn=lambda ctx: ctx.trigger["payload"], fan_out=True)
        src >> Task("each", fn=lambda item: f"<{item}>", outlets=["done"])
    with Dag("consumer", prompt="p", auto_evaluate=False, schedule=[Asset("done")]) as consumer:
        Task("got", fn=lambda ctx: ctx.trigger["payload"])
    engine = make_engine()
    engine.register(producer)
    engine.register(consumer)
    async with engine:
        await engine.run("producer", ["x", "y"], timeout=5)
        [run] = engine.list_runs("consumer")
        run = await engine.wait(run.run_id, timeout=5)
        assert run.outputs()["got"] == ["<x>", "<y>"]
        await engine.run("producer", [], timeout=5)  # nothing produced: nothing started
        assert len(engine.list_runs("consumer")) == 1


@pytest.mark.asyncio
async def test_assets_never_go_round_in_a_circle(make_engine, backend):
    with Dag("a", prompt="p", auto_evaluate=False, schedule=[Asset("from-b")]) as a:
        Task("t", fn=lambda: "a", outlets=["from-a"])
    with Dag("b", prompt="p", auto_evaluate=False, schedule=[Asset("from-a")]) as b:
        Task("t", fn=lambda: "b", outlets=["from-b"])
    engine = make_engine()
    engine.register(a)
    engine.register(b)
    async with engine:
        await engine.run("a", timeout=5)
        [run_b] = engine.list_runs("b")
        await engine.wait(run_b.run_id, timeout=5)
    assert len(engine.list_runs("a")) == 1  # b's update of from-b doesn't start a again
    assert engine.get_run(run_b.run_id).trigger["chain"] == ["a"]


@pytest.mark.asyncio
async def test_dag_triggers_never_go_round_in_a_circle(make_engine, backend):
    with Dag("a", prompt="p", auto_evaluate=False, schedule=[DagTrigger("b")]) as a:
        Task("t", fn=lambda: "a")
    with Dag("b", prompt="p", auto_evaluate=False, schedule=[DagTrigger("a")]) as b:
        Task("t", fn=lambda: "b")
    engine = make_engine()
    engine.register(a)
    engine.register(b)
    async with engine:
        await engine.run("a", timeout=5)
        for _ in range(20):
            runs = engine.list_runs()
            if len(runs) == 2 and all(r.is_terminal for r in runs):
                break
            import asyncio

            await asyncio.sleep(0.05)
    assert sorted(r.dag_id for r in engine.list_runs()) == ["a", "b"]


def test_a_dag_cant_be_scheduled_on_its_own_outlet(make_engine):
    with Dag("self", prompt="p", auto_evaluate=False, schedule=[Asset("x")]) as dag:
        Task("t", fn=lambda: 1, outlets=[Asset("x")])
    with pytest.raises(TriggerError, match="cannot trigger itself"):
        make_engine().register(dag)


def test_branches_cant_have_outlets_and_none_is_reserved():
    with Dag("b", prompt="p", auto_evaluate=False) as dag:
        AgenticBranch("pick", prompt="x", outlets=["y"]) >> AgenticTask("go", prompt="g")
    with pytest.raises(DagValidationError, match="can't have outlets"):
        dag.validate()
    with Dag("n", prompt="p", auto_evaluate=False) as dag:
        AgenticBranch("pick", prompt="x", allow_none=True) >> AgenticTask("none", prompt="g")
    with pytest.raises(DagValidationError, match="'none'"):
        dag.validate()


def test_tasks_using_an_evaluated_output_must_come_after_the_evaluator():
    with Dag("e", prompt="p", auto_evaluate=False) as dag:
        draft = AgenticTask("draft", prompt="d")
        review = AgenticEvaluator("review", prompt="r")
        send = AgenticTask("send", prompt="s")
        draft >> [review, send]
    with pytest.raises(DagValidationError, match="must come after 'review'"):
        dag.validate()


def test_trigger_items():
    from types import SimpleNamespace as NS

    assert trigger_items(NS(trigger={"payload": ["a", "b"]})) == ["a", "b"]
    assert trigger_items(NS(trigger={"payload": {"text": "t", "items": ["a"]}})) == ["a"]
    assert trigger_items(NS(trigger={"payload": "one"})) == ["one"]
    assert trigger_items(NS(trigger={"payload": None})) == [] and trigger_items(NS(trigger={})) == []
    assert trigger_items(NS(trigger={"payload": "  "})) == []


@pytest.mark.asyncio
async def test_fan_out_is_capped_at_max_mapped(make_engine):
    with Dag("cap", prompt="p", auto_evaluate=False) as dag:
        Task("src", fn=lambda: list(range(10)), fan_out=True) >> Task("each", fn=lambda i: i)
    engine = make_engine(config=EngineConfig(model="test-model", max_mapped=3))
    engine.register(dag)
    async with engine:
        run = await engine.run("cap", timeout=5)
    assert run.node_instances["each"] == ["each_0", "each_1", "each_2"]


@pytest.mark.asyncio
async def test_mapped_tasks_arent_given_the_whole_trigger_payload(make_engine, backend):
    backend.default = "ok"
    with Dag("lanes", prompt="p", auto_evaluate=False) as dag:
        items = Task("items", fn=trigger_items, fan_out=True)
        each = AgenticTask("each", prompt="Do it.")
        report = AgenticTask("report", prompt="Report.")
        items >> each >> report
    engine = make_engine()
    engine.register(dag)
    async with engine:
        await engine.run("lanes", ["secret-a", "secret-b"], timeout=5)
    prompts = {r.metadata["instance_id"]: r.messages[0].content for r in backend.requests}
    assert "Trigger payload" not in prompts["each_0"] and "secret-b" not in prompts["each_0"]
    assert "Trigger payload" in prompts["report"]


@pytest.mark.asyncio
async def test_a_branch_that_allows_none_can_select_nothing(make_engine, backend):
    backend.add("arrived", "none")
    with Dag("parcel", prompt="p", auto_evaluate=False) as dag:
        arrived = AgenticBranch("arrived", prompt="Has it arrived?", allow_none=True)
        arrived >> AgenticTask("tell", prompt="Tell me.")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("parcel", timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert run.instances["arrived"].selected == [] and run.instances["tell"].status.value == "skipped"
    assert backend.requests[0].output.choices == ["tell", "none"]


@pytest.mark.asyncio
async def test_skip_if_and_the_evaluator_skipping_a_run_that_found_nothing(make_engine, backend):
    backend.add("arrived", "none", "tell")
    backend.add("tell", "It's here.")
    backend.default = '{"score": 5, "assessment": "", "prompt_changes": [], "structural_changes": [], "hide": [], "allow": [], "skills": [], "memory": {}}'
    with Dag("parcel", prompt="p") as dag:
        arrived = AgenticBranch("arrived", prompt="Has it arrived?", allow_none=True)
        arrived >> AgenticTask("tell", prompt="Tell me.")
    engine = make_engine()
    engine.register(dag, evaluator=DagEvaluator(skip_when_idle=True))
    async with engine:
        quiet = await engine.run("parcel", timeout=5)
        backend.add("dag_evaluator.executive_summary", "Your parcel arrived.")
        news = await engine.run("parcel", timeout=5)
    evaluator = [t for t in quiet.node_instances if t.startswith("dag_evaluator.")]
    assert len(evaluator) == 5
    assert all(quiet.instances[t].status.value == "skipped" for t in evaluator)
    assert quiet.instances["dag_evaluator.executive_summary"].reason == "the run found nothing"
    assert news.instances["dag_evaluator.executive_summary"].status.value == "success"
    assert news.summary == "Your parcel arrived."
    assert found_nothing(engine, news) is None


@pytest.mark.asyncio
async def test_on_complete_finishes_before_the_task_counts_as_done(make_engine):
    async def mark(engine, run, inst):
        run.memory["marked"] = inst.output

    with Dag("hook", prompt="p", auto_evaluate=False) as dag:
        a = Task("a", fn=lambda: "x", on_complete=mark)
        b = Task("b", fn=lambda x, ctx: ctx.memory.get("marked"))
        a >> b
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("hook", timeout=5)
    assert run.outputs()["b"] == "x"


@pytest.mark.asyncio
async def test_prompts_can_be_written_for_each_run_and_answers_checked(make_engine, backend):
    from agentic_airflow import OutputValidationError

    def check(value):
        if "python" not in value:
            raise OutputValidationError("Give the file in a python block.")
        return value.upper()

    backend.add("review", "no code", "here: python")
    with Dag("dyn", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("review", prompt=lambda engine, run: f"Review run {run.trigger['payload']}.", validator=check)
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("dyn", "#7", timeout=5)
    assert run.outputs()["review"] == "HERE: PYTHON"
    first, second = backend.requests
    assert first.messages[0].content.endswith("# Task\nReview run #7.")
    assert "Give the file in a python block." in second.messages[-1].content
    assert dag.get("review").describe()["prompt"] == "(written when the task starts)"


def test_a_branch_answer_with_words_around_the_choice_is_understood():
    from agentic_airflow import GenerationResponse, OutputSpec, OutputValidationError
    from agentic_airflow.validation import coerce_output

    spec = OutputSpec(kind="enum", choices=["apply", "skip", "none"])
    assert coerce_output(GenerationResponse(text="I'd say `apply`, it fits."), spec) == "apply"
    with pytest.raises(OutputValidationError):
        coerce_output(GenerationResponse(text="apply or skip?"), spec)


def test_the_run_log_gives_each_entry_its_share():
    from agentic_airflow.agent_loop import render_run_log

    log = [
        {"instance_id": f"t_{i}", "status": "success", "attempt": 1, "output": "x" * 5000,
         "tool_calls": [{"name": "fetch", "status": "ok"}]}
        for i in range(4)
    ]
    text = render_run_log(log, 2000)
    assert text.count("\n") == 3 and "t_3 (success, attempt 1); tools: fetch (ok): xxx" in text
    assert len(text) < 2200


@pytest.mark.asyncio
async def test_an_engine_can_share_an_applications_queues(make_engine):
    from agentic_airflow.priority import ModelQueueSet

    queues = ModelQueueSet()
    engine = make_engine(queues=queues)
    assert engine.queues is queues


@pytest.mark.asyncio
async def test_an_evaluator_task_failing_doesnt_fail_the_run(make_engine, backend):
    backend.add("work", "done")
    backend.default = "not json"  # every evaluator task fails to parse
    with Dag("ev_fails", prompt="p") as dag:
        AgenticTask("work", prompt="x")
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("ev_fails", timeout=5)
    assert run.status == RunStatus.SUCCESS
    assert run.instances["dag_evaluator.workflow_critique"].status.value == "failed"


@pytest.mark.asyncio
async def test_a_failed_lane_fails_only_the_lanes_that_follow_it(make_engine):
    def check(item):
        if item == "bad":
            raise ValueError("no")
        return item

    with Dag("lanes_fail", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: ["good", "bad"], fan_out=True)
        first = Task("first", fn=check)
        then = Task("then", fn=lambda x: f"then {x}", follow_mapping=True)
        join = Task("join", fn=lambda d: d)
        src >> first >> then >> join
    engine = make_engine()
    engine.register(dag)
    async with engine:
        run = await engine.run("lanes_fail", timeout=5)
    s = {i: inst.status.value for i, inst in run.instances.items()}
    assert s["first_1"] == "failed" and s["then_1"] == "upstream_failed"
    assert s["then_0"] == "success" and run.outputs()["then_0"] == "then good"  # the good lane carries on
    assert s["join"] == "upstream_failed" and run.status == RunStatus.FAILED  # a join over a failed lane doesn't run
