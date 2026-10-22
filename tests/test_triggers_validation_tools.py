from datetime import datetime, timezone

import pytest

from agentic_airflow import (
    CronTrigger,
    DagTrigger,
    GenerationResponse,
    OutputSpec,
    OutputValidationError,
    Tool,
    ToolError,
    ToolRegistry,
    TriggerError,
    WebhookAuthenticationError,
    WebhookTrigger,
    tool,
)
from agentic_airflow.triggers import CronExpression, normalise_schedule
from agentic_airflow.validation import coerce_output, schema_errors


def test_cron_next_after_weekdays_at_nine():
    cron = CronExpression("0 9 * * 1-5")
    sat = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)  # Saturday
    nxt = cron.next_after(sat)
    assert nxt == datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc)  # Monday


def test_cron_steps_lists_and_names():
    cron = CronExpression("*/15 8-10 1,15 jan,jul sun")
    assert cron.minutes == {0, 15, 30, 45}
    assert cron.hours == {8, 9, 10}
    assert cron.months == {1, 7}
    assert cron.dows == {0}
    nxt = cron.next_after(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc))
    assert nxt.month == 1 and nxt.hour == 8 and nxt.minute == 0
    assert nxt.day == 1 or nxt.weekday() == 6  # dom OR dow semantics


def test_cron_aliases_and_errors():
    assert CronExpression("@hourly").minutes == {0}
    with pytest.raises(TriggerError):
        CronExpression("0 9 * *")
    with pytest.raises(TriggerError):
        CronExpression("61 * * * *")


def test_cron_trigger_timezone():
    trig = CronTrigger("0 9 * * *", timezone="America/New_York")
    after = datetime(2026, 7, 1, 0, 0, tzinfo=timezone.utc)
    assert trig.next_fire(after) == datetime(2026, 7, 1, 13, 0, tzinfo=timezone.utc)
    with pytest.raises(TriggerError):
        CronTrigger("0 9 * * *", timezone="Mars/Olympus")


def test_normalise_schedule():
    sched = normalise_schedule(["0 9 * * *", DagTrigger("other"), WebhookTrigger("hook")])
    assert [t.kind for t in sched] == ["cron", "dag", "webhook"]
    assert normalise_schedule(None) == []
    with pytest.raises(TriggerError):
        normalise_schedule([42])


def test_webhook_signature():
    hook = WebhookTrigger("hook", secret="s3cret")
    body = b'{"a": 1}'
    sig = hook.sign(body)
    hook.verify(body, sig)
    hook.verify(body, "sha256=" + sig.upper())
    with pytest.raises(WebhookAuthenticationError):
        hook.verify(body, "deadbeef")
    with pytest.raises(WebhookAuthenticationError):
        hook.verify(body, None)
    with pytest.raises(WebhookAuthenticationError):
        hook.verify(b"x" * 2_000_000, sig)
    WebhookTrigger("open").verify(body, None)  # unsigned hooks accept anything


# --- validation ---------------------------------------------------------------

def test_schema_errors():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}, "tags": {"type": "array", "items": {"type": "string"}}}, "required": ["n"], "additionalProperties": False}
    assert schema_errors({"n": 2, "tags": ["a"]}, schema) == []
    errs = schema_errors({"n": 0, "tags": [1], "x": 1}, schema)
    assert any("minimum" in e or ">= 1" in e for e in errs)
    assert any("tags[0]" in e for e in errs)
    assert any("unexpected property" in e for e in errs)
    assert schema_errors({}, schema)[0].endswith("missing required property 'n'")
    assert schema_errors(True, {"type": "integer"})  # bool is not int


def test_coerce_output_kinds(tmp_path):
    assert coerce_output(GenerationResponse(text="hi"), OutputSpec("string")) == "hi"
    assert coerce_output(GenerationResponse(text='```json\n["a", "b"]\n```'), OutputSpec("list")) == ["a", "b"]
    assert coerce_output(GenerationResponse(text='Here: {"items": [1, 2]}'), OutputSpec("list")) == [1, 2]
    assert coerce_output(GenerationResponse(text='{"rerun": true, "reason": "x"}'), OutputSpec("json")) == {"rerun": True, "reason": "x"}
    with pytest.raises(OutputValidationError):
        coerce_output(GenerationResponse(text="not json"), OutputSpec("json"))
    with pytest.raises(OutputValidationError):
        coerce_output(GenerationResponse(text='{"rerun": "yes"}'), OutputSpec("json", json_schema={"type": "object", "properties": {"rerun": {"type": "boolean"}}, "required": ["rerun"]}))
    assert coerce_output(GenerationResponse(text=" 'b' "), OutputSpec("enum", choices=["a", "b"])) == "b"
    assert coerce_output(GenerationResponse(text='["a", "b", "a"]'), OutputSpec("enum", choices=["a", "b"], multi=True)) == ["a", "b"]
    with pytest.raises(OutputValidationError):
        coerce_output(GenerationResponse(text="c"), OutputSpec("enum", choices=["a", "b"]))
    f = tmp_path / "out.txt"
    f.write_text("x")
    assert coerce_output(GenerationResponse(text=str(f)), OutputSpec("file"))["path"] == str(f)
    with pytest.raises(OutputValidationError):
        coerce_output(GenerationResponse(text="/nonexistent/file"), OutputSpec("file"))
    img = coerce_output(GenerationResponse(output={"mime": "image/png", "data_base64": "aGVsbG8="}), OutputSpec("image"))
    assert img["kind"] == "image"
    assert coerce_output(GenerationResponse(output=b"\x89PNG"), OutputSpec("image"))["data_base64"]


# --- tools ----------------------------------------------------------------------

def test_tool_schema_inference_and_invocation():
    @tool(description="Add numbers")
    def add(a: int, b: int = 2, tags: list[str] | None = None) -> int:
        return a + b

    assert add.parameters["required"] == ["a"]
    assert add.parameters["properties"]["a"] == {"type": "integer"}
    assert add.parameters["properties"]["b"]["default"] == 2
    assert add.validate_arguments({"a": "x"})
    assert not add.validate_arguments({"a": 1})
    assert add.description == "Add numbers"


@pytest.mark.asyncio
async def test_tool_invoke_validates_and_runs_async():
    calls = []

    async def fetch(url: str) -> str:
        calls.append(url)
        return "ok"

    t = Tool(name="fetch", fn=fetch, parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})
    assert await t.invoke({"url": "u"}) == "ok"
    with pytest.raises(ToolError):
        await t.invoke({})
    with pytest.raises(ToolError):
        Tool(name="bad name!", fn=fetch)


def test_registry_search_and_reserved_name():
    reg = ToolRegistry()
    reg.register(tool(lambda q: q, name="web_search", description="Search the web", hidden=True))
    reg.register(tool(lambda: 1, name="calendar", description="Read calendar events"))
    assert [t.name for t in reg.search("search web")] == ["web_search"]
    assert [t.name for t in reg.search("events", within=["calendar"])] == ["calendar"]
    with pytest.raises(ToolError):
        reg.register(tool(lambda: 1, name="search_tools"))
