import pytest

from agentic_airflow import ScopeConfig, ToolRegistry, resolve_config, resolve_permissions, tool


def test_standard_parameters_inherit_and_override():
    cfg = resolve_config([
        ScopeConfig(model="sys", thinking="off", verbosity=0),
        ScopeConfig(model="dag", thinking="on"),
        ScopeConfig(),  # group inherits
        ScopeConfig(verbosity=2),
    ])
    assert cfg.model == "dag"
    assert cfg.thinking is True
    assert cfg.verbosity == 2
    assert cfg.max_tool_steps == 10


def test_skills_are_additive_unless_isolated():
    cfg = resolve_config([ScopeConfig(skills=["A"]), ScopeConfig(skills=["B", "A"]), ScopeConfig(skills=["C"])])
    assert cfg.skills == ("A", "B", "C")
    cfg = resolve_config([ScopeConfig(skills=["A"]), ScopeConfig(skills=["B"], inherit_skills=False)])
    assert cfg.skills == ("B",)


def test_allowed_inherits_and_denied_accumulates():
    cfg = resolve_config([ScopeConfig(allowed_tools=["a", "b", "c"], denied_tools=["c"]), ScopeConfig(denied_tools=["b"]), ScopeConfig()])
    assert cfg.allowed_tools == frozenset({"a", "b", "c"})
    assert cfg.denied_tools == frozenset({"b", "c"})
    cfg2 = resolve_config([ScopeConfig(allowed_tools=["a"]), ScopeConfig(allowed_tools=["x"])])
    assert cfg2.allowed_tools == frozenset({"x"})


def test_permission_resolution_three_states():
    reg = ToolRegistry()
    reg.register(tool(lambda: 1, name="a"))
    reg.register(tool(lambda: 1, name="b", hidden=True))
    reg.register(tool(lambda: 1, name="c"))
    reg.register(tool(lambda: 1, name="d"))
    cfg = resolve_config([ScopeConfig(allowed_tools=["a", "b", "c", "zzz"], denied_tools=["c"])])
    perms = resolve_permissions(cfg, reg)
    assert perms.visible == {"a"}
    assert perms.hidden == {"b"}
    assert perms.denied == {"c", "d"}
    assert perms.state("c") == "denied"
    assert perms.state("b") == "hidden"
    assert perms.state("nope", registered=False) == "unknown"
    # denied always wins, even over an approved override
    perms2 = resolve_permissions(cfg, reg, overrides={"allow": ["c", "d"], "hide": ["a"]})
    assert "c" not in perms2.allowed and "d" in perms2.allowed and "a" in perms2.hidden


def test_no_allowed_list_means_every_registered_tool():
    reg = ToolRegistry([tool(lambda: 1, name="a"), tool(lambda: 1, name="b")])
    perms = resolve_permissions(resolve_config([ScopeConfig(denied_tools=["b"])]), reg)
    assert perms.allowed == {"a"} and perms.denied == {"b"}


def test_scope_config_validation():
    with pytest.raises(ValueError):
        ScopeConfig(thinking="maybe")
    with pytest.raises(TypeError):
        ScopeConfig(skills="Research")
