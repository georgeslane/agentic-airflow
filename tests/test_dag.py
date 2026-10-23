import pytest

from agentic_airflow import AgenticBranch, AgenticEvaluator, AgenticTask, Branch, Dag, DagValidationError, Task, TaskGroup


def noop(*a, **k):
    return None


def test_dependency_syntax_and_context_resolution():
    with Dag("d", prompt="p") as dag:
        a = Task("a", fn=noop)
        b = Task("b", fn=noop)
        c = Task("c", fn=noop)
        d = Task("d_", fn=noop, context=["a"])
        a >> [b, c] >> d
    dag.validate()
    assert a.downstream_ids == ["b", "c"]
    assert d.upstream_ids == ["b", "c"]
    assert [o.id for o in d.effective_context_ops()] == ["a"]
    assert [o.id for o in dag.topological_order()] == ["a", "b", "c", "d_"]
    assert [r.id for r in dag.roots()] == ["a"] and [l.id for l in dag.leaves()] == ["d_"]


def test_lshift_and_chains():
    with Dag("d", prompt="p") as dag:
        a = Task("a", fn=noop)
        b = Task("b", fn=noop)
        c = Task("c", fn=noop)
        c << b << a
    dag.validate()
    assert [o.id for o in dag.topological_order()] == ["a", "b", "c"]


def test_validation_errors():
    with Dag("d", prompt="p") as dag:
        a = Task("a", fn=noop)
        b = Task("b", fn=noop)
        a >> b >> a
    with pytest.raises(DagValidationError, match="cycle"):
        dag.validate()

    with pytest.raises(DagValidationError):
        with Dag("d2", prompt="p"):
            Task("a", fn=noop)
            Task("a", fn=noop)

    with Dag("d3", prompt="p") as dag3:
        a = Task("a", fn=noop)
        b = Task("b", fn=noop, context=[a])
    with pytest.raises(DagValidationError, match="not an upstream ancestor"):
        dag3.validate()

    with Dag("d4", prompt="p") as dag4:
        a = Task("a", fn=noop, fan_out=True)
        b = Task("b", fn=noop, fan_out=True)
        c = Task("c", fn=noop)
        [a, b] >> c
    with pytest.raises(DagValidationError, match="fan_out"):
        dag4.validate()

    with Dag("d5", prompt="p") as dag5:
        a = Task("a", fn=noop)
        b = Task("a_0", fn=noop)
        a >> b
    with pytest.raises(DagValidationError, match="collides"):
        dag5.validate()

    with pytest.raises(DagValidationError):
        Task("bad id", fn=noop)
    with pytest.raises(DagValidationError):
        Dag("x", prompt="")


def test_evaluator_target_resolution():
    with Dag("d", prompt="p") as dag:
        a = AgenticTask("a", prompt="x")
        e = AgenticEvaluator("e", prompt="check")
        a >> e
    dag.validate()
    assert e.target is a and e.follow_mapping is True and e.return_type == "json"

    with Dag("d2", prompt="p") as dag2:
        a = AgenticTask("a", prompt="x")
        b = AgenticTask("b", prompt="x")
        e = AgenticEvaluator("e", prompt="check")
        [a, b] >> e
    with pytest.raises(DagValidationError, match="exactly one upstream"):
        dag2.validate()


def test_task_groups_cascade_and_nesting():
    with Dag("d", prompt="p", model="dag-model", skills=["Base"]) as dag:
        with TaskGroup("g", skills=["G"], allowed_tools=["x"]) as g:
            t1 = AgenticTask("t1", prompt="x")
            with TaskGroup("inner", model="inner-model") as inner:
                t2 = AgenticTask("t2", prompt="x", skills=["T"])
        t3 = AgenticTask("t3", prompt="x")
    dag.validate()
    assert t1.group is g and t2.group is inner and inner.parent is g
    chain = dag.scopes_for(t2)
    assert [c.model for c in chain] == ["dag-model", None, "inner-model", None]
    assert g.path() == ["g"] and inner.path() == ["g", "inner"]
    assert t3.group is None or t3.group.is_dag
    desc = dag.to_dict()
    assert {t["id"] for t in desc["tasks"]} == {"t1", "t2", "t3"}
    assert desc["groups"][0]["groups"][0]["id"] == "inner"


def test_branch_requires_downstream_and_agentic_branch_defaults():
    with Dag("d", prompt="p") as dag:
        Branch("b", fn=noop)
    with pytest.raises(DagValidationError, match="no downstream"):
        dag.validate()
    br = AgenticBranch("ab", prompt="choose", multi_select=True)
    assert br.is_branch and br.fan_out is False
    assert AgenticTask("l", prompt="x", return_type="list").fan_out is True
    assert AgenticTask("l2", prompt="x", return_type="list", fan_out=False).fan_out is False


def test_a_task_group_stands_for_its_first_or_last_tasks():
    from agentic_airflow import AgenticTask, Dag, Task, TaskGroup

    with Dag("g", prompt="p", auto_evaluate=False) as dag:
        src = Task("src", fn=lambda: [1], fan_out=True)
        with TaskGroup("each") as each:
            a = AgenticTask("a", prompt="a")
            b = AgenticTask("b", prompt="b")
            c = AgenticTask("c", prompt="c")
            a >> [b, c]
        report = AgenticTask("report", prompt="r")
        src >> each >> report
        other = Task("other", fn=lambda: 1)
        [other, each] >> AgenticTask("after", prompt="x")
        AgenticTask("before", prompt="y") >> [each]
    dag.validate()
    assert a.upstream_ids == ["src", "before"]
    assert sorted(report.upstream_ids) == ["b", "c"]
    assert sorted(dag.get("after").upstream_ids) == ["b", "c", "other"]
