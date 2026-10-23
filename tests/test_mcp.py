import pytest

pytest.importorskip("mcp")

from agentic_airflow import AgenticTask, Dag, RunStatus, Task  # noqa: E402
from agentic_airflow.mcp import build_server  # noqa: E402
from agentic_airflow.proposals import propose_tool_permissions, withdraw_proposal  # noqa: E402

FORBIDDEN = {"approve", "reject", "decide", "apply", "cancel", "set_memory", "set_overrides"}


async def call(_server, _tool, **args):
    result = await _server.call_tool(_tool, args)
    return result.structured_content


@pytest.fixture
async def engine_with_run(make_engine, backend):
    with Dag("insp", prompt="Objective", auto_evaluate=False, allowed_tools=["send_email"]) as dag:
        src = Task("src", fn=lambda: ["a", "b"], fan_out=True)
        work = AgenticTask("work", prompt="Work on it", skills=["S"])
        src >> work
    backend.default = lambda req: "done:" + req.metadata["instance_id"]
    engine = make_engine()
    engine.skills.register("S", "Be careful.")
    engine.register(dag)
    async with engine:
        run = await engine.run("insp", timeout=5)
    assert run.status == RunStatus.SUCCESS
    return engine, run


@pytest.mark.asyncio
async def test_server_exposes_only_inspection_and_proposal_tools(engine_with_run):
    engine, _ = engine_with_run
    tools = {t.name for t in await build_server(engine).list_tools()}
    assert "trigger_run" not in tools
    assert not any(any(word in name for word in FORBIDDEN) for name in tools)
    assert {"get_dag", "get_run", "get_task_output", "get_run_log", "propose_prompt_change", "withdraw_proposal"} <= tools
    tools_with_trigger = {t.name for t in await build_server(engine, allow_trigger=True).list_tools()}
    assert "trigger_run" in tools_with_trigger


@pytest.mark.asyncio
async def test_inspection_tools(engine_with_run):
    engine, run = engine_with_run
    server = build_server(engine, output_chars=6)
    dags = await call(server, "list_dags")
    assert dags["ok"] and dags["data"][0]["dag_id"] == "insp"
    dag = (await call(server, "get_dag", dag_id="insp"))["data"]
    work = next(t for t in dag["tasks"] if t["id"] == "work")
    assert work["effective_prompt"] == "Work on it"
    assert work["resolved"]["tools"]["allowed"] == ["send_email"] and work["resolved"]["skills"] == ["S"]
    assert dag["skills"] == ["S"] and dag["memory"] == {}
    runs = (await call(server, "list_runs", dag_id="insp"))["data"]
    assert runs[0]["run_id"] == run.run_id and runs[0]["status"] == "success"
    got = (await call(server, "get_run", run_id=run.run_id))["data"]
    inst = next(i for i in got["instances"] if i["instance_id"] == "work_1")
    assert inst["output"].startswith("done:w") and "truncated" in inst["output"]
    full = (await call(server, "get_task_output", run_id=run.run_id, instance_id="work_1"))["data"]
    assert full["output"] == "done:work_1" and full["input_context"]["upstream"] == {"src": "b"}
    log = (await call(server, "get_run_log", run_id=run.run_id, task_id="work"))["data"]
    assert [e["instance_id"] for e in log] == ["work_0", "work_1"]
    assert (await call(server, "get_graph", dag_id="insp", run_id=run.run_id))["data"]["run"]["status"] == "success"
    assert (await call(server, "get_skills", name="S"))["data"]["versions"][0]["content"] == "Be careful."
    assert (await call(server, "get_stats"))["data"]["dags"] == ["insp"]
    missing = await call(server, "get_task_output", run_id=run.run_id, instance_id="nope")
    assert missing["ok"] is False and "Unknown instance" in missing["error"]
    unknown_dag = await call(server, "get_dag", dag_id="nope")
    assert unknown_dag["ok"] is False


@pytest.mark.asyncio
async def test_proposals_are_pending_until_a_human_approves(engine_with_run):
    engine, run = engine_with_run
    server = build_server(engine, client_name="claude-code")

    prompt = await call(server, "propose_prompt_change", dag_id="insp", task_id="work", proposed_prompt="Work carefully.", rationale="Outputs were sloppy.")
    assert prompt["ok"] and prompt["data"]["status"] == "pending" and prompt["data"]["proposed_by"] == "claude-code"
    bad = await call(server, "propose_prompt_change", dag_id="insp", task_id="src", proposed_prompt="x", rationale="r")
    assert bad["ok"] is False and "not agentic" in bad["error"]
    no_rationale = await call(server, "propose_prompt_change", dag_id="insp", task_id="work", proposed_prompt="x", rationale="  ")
    assert no_rationale["ok"] is False

    perms = await call(server, "propose_tool_permissions", dag_id="insp", rationale="unused", hide=["send_email"], allow=["web_search"])
    assert perms["ok"]
    unknown_tool = await call(server, "propose_tool_permissions", dag_id="insp", rationale="r", allow=["nope"])
    assert unknown_tool["ok"] is False and "Unknown tool" in unknown_tool["error"]
    skill = await call(server, "propose_skill_rewrite", skill="S", content="Be very careful.", rationale="too vague")
    assert skill["ok"] and skill["data"]["payload"]["previous"] == "Be careful."
    same = await call(server, "propose_skill_rewrite", skill="S", content="Be careful.", rationale="noop")
    assert same["ok"] is False
    memory = await call(server, "propose_memory_update", dag_id="insp", memory={"seen": ["a", "b"]}, rationale="track")
    assert memory["ok"]
    structural = await call(server, "propose_structural_change", dag_id="insp", description="Add a review step after work", rationale="quality")
    assert structural["ok"]

    # nothing changed yet
    assert engine.overrides_for("insp") == {}
    assert engine.skills.get("S").content == "Be careful."
    assert engine.get_memory("insp") == {}
    pending = (await call(server, "list_recommendations", dag_id="insp", status="pending"))["data"]
    assert len(pending) == 4 and all(r["source"] == "mcp" for r in pending)

    # the agent can withdraw its own proposal, but not one from another source
    withdrawn = await call(server, "withdraw_proposal", recommendation_id=structural["data"]["id"], reason="changed my mind")
    assert withdrawn["ok"] and withdrawn["data"]["status"] == "withdrawn"
    foreign = propose_tool_permissions(engine, "insp", deny=["delete_file"], rationale="human says so", source="ui")
    refused = await call(server, "withdraw_proposal", recommendation_id=foreign["id"])
    assert refused["ok"] is False and "cannot be withdrawn" in refused["error"]
    withdraw_proposal(engine, foreign["id"])  # no source filter: allowed
    with pytest.raises(Exception):
        withdraw_proposal(engine, foreign["id"])  # already withdrawn

    # human approval applies them through the existing gate
    await engine.approve_recommendation(prompt["data"]["id"])
    await engine.approve_recommendation(perms["data"]["id"])
    await engine.approve_recommendation(skill["data"]["id"])
    await engine.approve_recommendation(memory["data"]["id"])
    assert engine.overrides_for("insp")["prompts"] == {"work": "Work carefully."}
    assert engine.overrides_for("insp")["hide"] == ["send_email"] and engine.overrides_for("insp")["allow"] == ["web_search"]
    assert engine.skills.get("S").content == "Be very careful."
    assert engine.get_memory("insp") == {"seen": ["a", "b"]}
    dag = (await call(server, "get_dag", dag_id="insp"))["data"]
    assert next(t for t in dag["tasks"] if t["id"] == "work")["effective_prompt"] == "Work carefully."
    assert dag["tasks"][1]["resolved"]["tools"]["hidden"] == ["send_email", "web_search"]


@pytest.mark.asyncio
async def test_trigger_tool_when_enabled(engine_with_run, backend):
    engine, _ = engine_with_run
    server = build_server(engine, allow_trigger=True)
    async with engine:
        res = await call(server, "trigger_run", dag_id="insp", payload={"x": 1})
        assert res["ok"]
        run = await engine.wait(res["data"]["run_id"], timeout=5)
    assert run.status == RunStatus.SUCCESS and run.trigger["source"] == "mcp" and run.trigger["payload"] == {"x": 1}


def test_cli_loads_engine_target(tmp_path, monkeypatch):
    from agentic_airflow.mcp.__main__ import load_engine

    mod = tmp_path / "myapp_engine_mod.py"
    mod.write_text(
        "from agentic_airflow import Engine, Dag, Task\n"
        "def make():\n"
        "    e = Engine()\n"
        "    with Dag('d', prompt='p', auto_evaluate=False) as dag:\n"
        "        Task('a', fn=lambda: 1)\n"
        "    e.register(dag)\n"
        "    return e\n"
        "engine = make()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert "d" in load_engine("myapp_engine_mod:engine").dags
    assert "d" in load_engine("myapp_engine_mod:make").dags
    with pytest.raises(SystemExit):
        load_engine("myapp_engine_mod")


@pytest.mark.asyncio
async def test_prompts_written_when_a_task_starts(make_engine):
    with Dag("dyn", prompt="p", auto_evaluate=False) as dag:
        AgenticTask("written", prompt=lambda engine, run: "Written now.")
    engine = make_engine()
    engine.register(dag)
    server = build_server(engine)
    got = await call(server, "get_dag", dag_id="dyn")
    assert got["ok"] and got["data"]["tasks"][0]["effective_prompt"] == "(written when the task starts)"
    refused = await call(server, "propose_prompt_change", dag_id="dyn", task_id="written", proposed_prompt="x", rationale="r")
    assert refused["ok"] is False and "writes its prompt when it starts" in refused["error"]


@pytest.mark.asyncio
async def test_clients_withdraw_only_their_own_and_cant_flood_the_queue(engine_with_run):
    from agentic_airflow.proposals import MAX_PENDING

    engine, _ = engine_with_run
    mine, other = build_server(engine, client_name="mine"), build_server(engine, client_name="other")
    made = await call(mine, "propose_structural_change", dag_id="insp", description="Add a step", rationale="r")
    refused = await call(other, "withdraw_proposal", recommendation_id=made["data"]["id"])
    assert refused["ok"] is False and "another client" in refused["error"]
    assert (await call(mine, "withdraw_proposal", recommendation_id=made["data"]["id"]))["ok"]
    for n in range(MAX_PENDING):
        assert (await call(mine, "propose_structural_change", dag_id="insp", description=f"Step {n}", rationale="r"))["ok"]
    flood = await call(other, "propose_structural_change", dag_id="insp", description="One more", rationale="r")
    assert flood["ok"] is False and "waiting for 'insp' already" in flood["error"]


@pytest.mark.asyncio
async def test_tool_overrides_can_be_taken_back(engine_with_run):
    engine, _ = engine_with_run
    server = build_server(engine)
    allowed = await call(server, "propose_tool_permissions", dag_id="insp", rationale="r", allow=["web_search"], deny=["delete_file"])
    await engine.approve_recommendation(allowed["data"]["id"])
    nothing = await call(server, "propose_tool_permissions", dag_id="insp", rationale="r", unset=["send_email"])
    assert nothing["ok"] is False and "No override" in nothing["error"]
    back = await call(server, "propose_tool_permissions", dag_id="insp", rationale="r", unset=["web_search"])
    await engine.approve_recommendation(back["data"]["id"])
    assert engine.overrides_for("insp")["allow"] == [] and engine.overrides_for("insp")["deny"] == ["delete_file"]
    skill = await call(server, "propose_skill_rewrite", skill="S", content="New.", rationale="r")
    await engine.approve_recommendation(skill["data"]["id"])
    assert engine.skills.get("S").to_dict()["versions"][-1]["note"].startswith(f"Approved proposal {skill['data']['id']} by mcp")


def _reads_definitions(engine, saved):
    """An application whose DAGs are read from text: it takes whole definitions, applied by saving them, and no
    overrides."""
    from agentic_airflow import Applier, StaleProposalError
    from agentic_airflow.errors import ConfigurationError

    def check(engine, rec):
        if "Dag(" not in rec["payload"]["definition"]:
            raise ConfigurationError("That isn't a DAG file: line 1")

    async def apply(engine, rec):
        dag = engine.dags[rec["dag_id"]]
        if rec["payload"]["base"] != dag.definition:
            raise StaleProposalError("The DAG has changed since")
        saved.append(rec["payload"]["definition"])
        dag.definition = rec["payload"]["definition"]
        return {"saved": True}

    engine.appliers["definition"] = Applier(apply, check)
    for kind in ("topology", "tools", "memory"):
        del engine.appliers[kind]


@pytest.mark.asyncio
async def test_an_application_applies_whole_definitions(make_engine):
    from agentic_airflow import ApprovalError, StaleProposalError
    from agentic_airflow.errors import ConfigurationError
    from agentic_airflow.proposals import propose_definition, propose_prompt_change

    text = "with Dag('filed', prompt='p'):\n    AgenticTask('a', prompt='x')\n"
    with Dag("filed", prompt="p", auto_evaluate=False, definition=text) as dag:
        AgenticTask("a", prompt="x")
    engine = make_engine()
    engine.register(dag)

    # An engine takes no definitions unless its application says how to apply them.
    with pytest.raises(ConfigurationError, match="doesn't take 'definition'"):
        propose_definition(engine, "filed", text + "# more\n", "r")
    old = propose_prompt_change(engine, "filed", "a", "y", "r")  # made while it took them

    saved = []
    _reads_definitions(engine, saved)
    server = build_server(engine, client_name="claude-code")
    tools = {t.name for t in await server.list_tools()}
    assert "propose_definition" in tools and "propose_skill_rewrite" in tools
    assert not {"propose_prompt_change", "propose_tool_permissions", "propose_memory_update"} & tools
    assert (await call(server, "get_dag", dag_id="filed"))["data"]["definition"] == text

    bad = await call(server, "propose_definition", dag_id="filed", definition="print('hi')", rationale="r")
    assert bad["ok"] is False and "isn't a DAG file" in bad["error"]
    same = await call(server, "propose_definition", dag_id="filed", definition=text.replace("\n", " \n"), rationale="r")
    assert same["ok"] is False and "already" in same["error"]
    with pytest.raises(ConfigurationError, match="doesn't take 'topology'"):
        propose_prompt_change(engine, "filed", "a", "y", "r")
    with pytest.raises(ApprovalError, match="doesn't apply 'topology'"):
        await engine.approve_recommendation(old["id"])

    new = text + "    AgenticTask('b', prompt='y')\n"
    first = await call(server, "propose_definition", dag_id="filed", definition=new, rationale="Add b")
    second = propose_definition(engine, "filed", new + "# and\n", "r", title="Tools", run_id="run_1", details={"what": "tools"})
    assert first["ok"] and first["data"]["payload"]["base"] == text and first["data"]["kind"] == "definition"
    assert second["run_id"] == "run_1" and second["payload"]["what"] == "tools" and second["title"] == "Tools"
    approved = await engine.approve_recommendation(first["data"]["id"])
    assert approved["status"] == "approved" and approved["applied"] == {"saved": True} and saved == [new]

    # The other was proposed against the DAG as it was: it's stale now, and said so.
    with pytest.raises(StaleProposalError):
        await engine.approve_recommendation(second["id"])
    assert engine.state.get_recommendation(second["id"])["status"] == "stale"
    engine.state.delete_recommendation(second["id"])
    assert engine.state.get_recommendation(second["id"]) is None


def test_recommendations_can_be_deleted_from_sqlite(tmp_path):
    from agentic_airflow import SqliteStateStore

    store = SqliteStateStore(tmp_path / "s.db")
    store.save_recommendation({"id": "rec_1", "dag_id": "d", "run_id": None, "status": "pending"})
    store.delete_recommendation("rec_1")
    assert store.get_recommendation("rec_1") is None and store.list_recommendations() == []
    store.close()
