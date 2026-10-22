"""End-to-end demo that runs offline with a scripted fake model.

It exercises fan-out, an agentic branch, a per-item reflection loop, a denied
tool call that pauses for approval, the DagEvaluator, memory and the graph UI.

    python examples/email_triage.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentic_airflow import (
    AgenticBranch,
    AgenticEvaluator,
    AgenticTask,
    Dag,
    Engine,
    EngineConfig,
    GenerationResponse,
    ScriptedBackend,
    SqliteStateStore,
    Task,
    ToolCall,
    ToolRegistry,
    tool,
)
from agentic_airflow.ui import render_html

INBOX = [
    {"id": 1, "from": "cfo@example.com", "subject": "URGENT: invoice overdue", "body": "Please confirm payment today."},
    {"id": 2, "from": "news@example.com", "subject": "Weekly newsletter", "body": "Top stories..."},
    {"id": 3, "from": "alice@example.com", "subject": "Lunch?", "body": "Are you free Thursday?"},
]


# --- tools -------------------------------------------------------------------
registry = ToolRegistry()


@tool(description="Look up a contact in the CRM")
def crm_lookup(email: str) -> dict:
    return {"email": email, "vip": email.startswith("cfo")}


@tool(description="Search the web", hidden=True)  # on-demand: discovered via search_tools
def web_search(query: str) -> list[str]:
    return [f"result about {query}"]


@tool(description="Send an email on the user's behalf")
def send_email(to: str, body: str) -> str:
    return f"sent to {to}"


for t in (crm_lookup, web_search, send_email):
    registry.register(t)


# --- deterministic tasks -------------------------------------------------------
def read_emails(ctx):
    since = (ctx.trigger.get("payload") or {}).get("since", "forever")
    print(f"[read_emails] fetching inbox since {since}; memory={ctx.memory}")
    return INBOX


def archive(email):
    return f"archived #{email['id']}"


# --- the fake model ------------------------------------------------------------
def fake_model(req):
    """Plays the LLM. Branch on task id and inference step."""
    if req.metadata.get("kind") == "user_chat":
        return "I archived the weekly newsletter (#2) because it did not need a reply."
    task = req.metadata["task_id"]
    step = req.metadata["step"]
    prompt = req.messages[0].content
    if task == "classify":
        return "draft_reply" if "URGENT" in prompt or "Lunch" in prompt else "archive"
    if task == "draft_reply":
        if step == 1 and "cfo@" in prompt:
            return GenerationResponse(tool_calls=[ToolCall(name="crm_lookup", arguments={"email": "cfo@example.com"})])
        if step == 2 and "cfo@" in prompt:  # denied tool -> pauses for approval
            return GenerationResponse(tool_calls=[ToolCall(name="send_email", arguments={"to": "cfo@example.com", "body": "Paid."})])
        attempt = req.metadata["attempt"]
        return f"Reply draft v{attempt}: thanks for your message, I will follow up." + (" (expanded)" if attempt > 1 else "")
    if task == "review":
        rerun = "v1" in prompt and req.metadata["instance_id"].endswith("_2")  # one rewrite of the lunch reply
        return {"output": {"rerun": rerun, "reason": "Too generic; mention Thursday." if rerun else "Looks good."}}
    if task == "daily_summary":
        return "Replied to the CFO and Alice, archived the newsletter."
    kind = task.rsplit(".", 1)[-1]
    if kind == "workflow_critique":
        return {"output": {"score": 8, "assessment": "Goal met.", "prompt_changes": [{"task_id": "draft_reply", "proposed": "Draft a concise, friendly reply that answers every question in the email.", "rationale": "First drafts were generic."}], "structural_changes": []}}
    if kind == "tool_audit":
        return {"output": {"hide": ["crm_lookup"], "allow": ["send_email"], "deny": [], "rationale": "send_email needed approval every time."}}
    if kind == "skill_rewriter":
        return {"output": {"skills": []}}
    if kind == "context_compactor":
        return {"output": {"memory": {"replied_to": [1, 3], "archived": [2]}, "discarded": ["email bodies"]}}
    return "Executive summary: 3 emails processed, 2 replies drafted, 1 archived. Two recommendations await approval."


async def main() -> None:
    with Dag(
        "inbox",
        prompt="Keep the inbox at zero. Never send money-related confirmations without a human.",
        context={"owner": "George"},
        schedule=["0 9 * * 1-5"],
        allowed_tools=["crm_lookup", "web_search"],  # send_email is denied -> approval
        model="fake-model",
    ) as dag:
        read = Task("read_emails", fn=read_emails, fan_out=True)
        classify = AgenticBranch("classify", prompt="Decide whether this email needs a reply or can be archived.")
        reply = AgenticTask("draft_reply", prompt="Draft a reply to this email.", follow_mapping=True, verbosity=1)
        review = AgenticEvaluator("review", prompt="Reject generic replies.", max_retries=2, on_exhausted="pass_latest")
        arch = Task("archive", fn=archive, follow_mapping=True)
        summary = AgenticTask("daily_summary", prompt="Summarise today's inbox work.", context=[review, arch])
        read >> classify >> [reply, arch]
        reply >> review >> summary
        arch >> summary

    backend = ScriptedBackend(default=fake_model)
    db = Path(__file__).with_name("email_triage.db")
    db.unlink(missing_ok=True)
    engine = Engine(
        backend,
        config=EngineConfig(model="fake-model"),
        tools=registry,
        state=SqliteStateStore(db),
        chat_sink=lambda e: print(f"  [chat:{e.task_id}] {e.type} {json.dumps(e.payload)[:80]}"),
    )
    engine.subscribe(lambda e: e.type.startswith("approval") and print(f"  [event] {e.type} {e.payload.get('kind')}"))
    engine.register(dag)

    async with engine:
        run = await engine.run("inbox", payload={"since": "yesterday"})
        print(f"\nrun {run.run_id}: {run.status.value}")
        for apr in engine.list_approvals():
            call = apr.payload["tool_call"]
            print(f"  approval needed: {call['name']}({call['arguments']}) -> approving")
            await engine.approve(apr.approval_id, remember=True)
        run = await engine.wait(run.run_id)
        print(f"run {run.run_id}: {run.status.value}\n")
        for inst in run.instances.values():
            print(f"  {inst.instance_id:35s} {inst.status.value:10s} attempt={inst.attempt} {str(inst.output)[:60]}")
        print("\nsummary:", run.summary)
        print("memory now:", engine.get_memory("inbox"))
        for rec in engine.list_recommendations(run_id=run.run_id):
            print(f"recommendation {rec['id']} [{rec['kind']}] {rec['title']} -> {rec['status']}")
        print("\nchat:", await engine.chat("What did you do with the newsletter?", run_id=run.run_id))
        out = Path(__file__).with_name("run_graph.html")
        out.write_text(render_html(engine.graph("inbox", run.run_id)))
        print(f"\ngraph written to {out}")
    engine.close()
    engine.state.close()
    db.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
