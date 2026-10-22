# agentic-airflow

Airflow-inspired workflow orchestration for LLM agents, in a single dependency-free Python package.

It combines deterministic DAG scheduling, dynamic fan-out/fan-in task mapping, hierarchical tool and skill governance, bounded reflection loops, user-first compute preemption, human-in-the-loop pauses and an automated post-run self-optimisation pipeline (`DagEvaluator`). It ships with a lightweight graph UI (self-contained HTML + JSON API) that plugs into any application.

```python
from agentic_airflow import Dag, Task, AgenticTask, AgenticBranch, AgenticEvaluator, Engine, EngineConfig

with Dag("inbox", prompt="Keep my inbox at zero without sending anything embarrassing.",
         schedule=["0 9 * * 1-5"], allowed_tools=["search", "send_email"]) as dag:
    read      = Task("read_emails", fn=fetch_unread, fan_out=True)          # list -> one lane per email
    classify  = AgenticBranch("classify", prompt="Route this email.")        # picks a downstream id
    reply     = AgenticTask("draft_reply", prompt="Draft a reply.", follow_mapping=True)
    review    = AgenticEvaluator("review", prompt="Reject replies that are rude or vague.")
    archive   = Task("archive", fn=archive_email, follow_mapping=True)
    summary   = AgenticTask("daily_summary", prompt="Summarise what happened.", context=[review, archive])

    read >> classify >> [reply, archive]
    reply >> review >> summary
    archive >> summary

engine = Engine(backend=my_backend, config=EngineConfig(model="qwen-3.6-25B-A3B"))
engine.register(dag)

async with engine:                       # starts the cron scheduler
    run = await engine.run("inbox")      # returns on completion or when a human is needed
    print(run.status, run.summary)
```

`my_backend` is anything with `async def generate(request) -> GenerationResponse`: see [Plugging in your agent framework](docs/plugging_in.md). Two adapters are included: an OpenAI-compatible HTTP backend (vLLM, Ollama, llama.cpp, OpenAI...) and an Anthropic SDK backend.

## Install

```bash
pip install -e .            # core: zero runtime dependencies, Python >= 3.10
pip install -e .[anthropic] # optional Claude backend
pip install -e .[mcp]       # optional MCP server for agent self-inspection
pip install -e .[dev]       # pytest
python -m pytest
```

## What you get

| Area | Highlights |
|---|---|
| **Operators** | `Task`, `Branch` (deterministic, run on a CPU thread pool), `AgenticTask`, `AgenticBranch`, `AgenticEvaluator` (LLM). Airflow `>>` syntax. |
| **Config cascade** | System → Dag → TaskGroup → Task. `model`/`thinking`/`verbosity` inherit; skills are additive (`inherit_skills=False` to isolate); `denied_tools` always beat `allowed_tools`. |
| **Three-state tools** | allowed (schema injected), hidden/on-demand (discoverable via the built-in `search_tools`), denied (the task checkpoints and pauses for approval). |
| **Dynamic mapping** | `fan_out=True` + list output → `task_0 … task_N-1`; joins receive an index-ordered dict; empty lists skip the chain; nested mapping works. |
| **Reflection loops** | `AgenticEvaluator` reruns its target with `retry_history`, bounded by `max_retries` and `on_exhausted`. |
| **Engine** | asyncio, per-model priority queues (live chat = tier 0, DAG work = aging FIFO), one inference step per slot so users preempt at atomic boundaries, checkpoint after every step. |
| **Human in the loop** | Denied tool calls and exhausted evaluators pause only the affected lane; parallel work continues. `approve()`, `reject()`, `decide()`. |
| **Triggers** | cron strings (5-field, timezone aware, no dependency), `DagTrigger` (on another DAG's completion), `WebhookTrigger` (HMAC-SHA256 signed). |
| **DagEvaluator** | Five-task group auto-attached to terminal nodes: critique, tool audit, skill rewriter, memory compactor, executive summary. Recommendations are staged and only applied after `approve_recommendation()`. |
| **Persistence** | `InMemoryStateStore` or `SqliteStateStore` (stdlib). Runs, checkpoints, approvals, memory, skill versions and overrides survive restarts; `resume_interrupted()` picks up where a process died. |
| **UI** | `engine.graph()` JSON, `render_html()` self-contained page, `render_embed()` fragment, optional `UIServer` (stdlib HTTP, bearer token, CSP). |
| **MCP server** (`[mcp]` extra) | Lets an external agent inspect its DAGs (settings, runs, outputs, logs) and *propose* prompt, permission, skill and memory changes that a human must approve. No approve/apply tools exist on the server. |

## Documentation

* [Operators & configuration reference](docs/operators.md): every parameter, data-flow and mapping rules.
* [Plugging in your agent framework](docs/plugging_in.md): backends, tools, skills, events, chat sink, state stores.
* [Architecture](docs/architecture.md): how a run executes, statuses, the expansion algorithm, the priority queue.
* [Graph UI & HTTP API](docs/ui.md).
* [MCP server](docs/mcp.md): agent self-inspection and human-gated proposals.
* [Security model](docs/security.md).
* [Spec deviations & design decisions](docs/spec_deviations.md): where the implementation adjusts the original spec and why.
* [Agent guide](docs/agent_guide.md): a compact orientation for another AI agent that needs to integrate with or extend the library.

## Examples

* [`examples/email_triage.py`](examples/email_triage.py): the full feature set with a scripted fake model; runs offline and writes `run_graph.html`.
* [`examples/serve_ui.py`](examples/serve_ui.py): starts the engine plus the HTTP UI with a demo DAG.
* [`examples/local_model.py`](examples/local_model.py): wiring a local OpenAI-compatible model (vLLM/Ollama) with real tools.

## Status

v0.1.0. The core engine is covered by the test-suite in `tests/` (mapping, branching, evaluator loops, approvals, preemption, persistence/resume, triggers, DagEvaluator, UI and server). The Anthropic/OpenAI-compatible adapters are thin and best verified against your own endpoint.
