# Plugging in your agent framework

The engine never talks to a model directly. Four seams let you integrate anything:

1. **`ModelBackend`**: one inference step.
2. **`ToolRegistry`**: the functions your agents may call.
3. **`SkillStore`**: reusable prompt fragments.
4. **Events / chat sink / state store**: observability and persistence.

## 1. The model backend

```python
from agentic_airflow import GenerationRequest, GenerationResponse, ToolCall

class MyBackend:
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # request.model      - resolved model id
        # request.system     - system prompt (objective, skills, tool guidance, output contract)
        # request.messages   - list[Message]: user / assistant(tool_calls) / tool(tool_call_id)
        # request.tools      - list[ToolSpec] (name, description, parameters JSON schema)
        # request.output     - OutputSpec(kind, json_schema, choices, multi) for constrained decoding
        # request.thinking   - bool
        # request.metadata   - dag_id, run_id, task_id, instance_id, attempt, step, priority (0 = live user)
        reply = await my_framework.complete(...)
        if reply.wants_tools:
            return GenerationResponse(tool_calls=[ToolCall(name=c.name, arguments=c.args, id=c.id) for c in reply.calls])
        return GenerationResponse(text=reply.text, output=reply.parsed_json_or_None, reasoning=reply.thoughts, usage={"input_tokens": 12})
```

Rules:

* Perform **exactly one** model call per `generate`. The engine runs the tool loop itself so it can enforce permissions, checkpoint between steps and let user requests preempt. Do not run your framework's own internal tool loop inside `generate`.
* Return `tool_calls` to request tools; otherwise the step is final and the engine validates `output` (preferred) or `text` against `request.output`.
* It must be safe to call concurrently (the engine runs parallel branches and multiple DAGs).
* Raise an exception to fail the task; the error text is recorded on the instance.
* `request.metadata` says which run, task and instance it's for, so a backend can add its own context to `request.system`.

Helpers: `FunctionBackend(fn)` wraps a plain sync/async callable returning a `GenerationResponse`, a string, or a dict. `ScriptedBackend` queues canned responses per task id for tests.

Included adapters:

* `agentic_airflow.backends.openai_compatible.OpenAICompatibleBackend(base_url, api_key=None, thinking_param=..., model_map=...)` for any `/chat/completions` server (vLLM, Ollama, llama.cpp, LM Studio, OpenAI). Stdlib only.
* `agentic_airflow.backends.anthropic.AnthropicBackend(client=None, max_tokens=16000, effort=None, fallbacks="default")` using the `anthropic` SDK (`pip install agentic-airflow[anthropic]`). Structured outputs are used for JSON/enum contracts; adaptive thinking is sent when `thinking="on"`; server-side refusal fallbacks are enabled by default (pass `fallbacks=None` to opt out).

## 2. Tools

```python
from agentic_airflow import ToolRegistry, tool

registry = ToolRegistry()

@tool(description="Search the web", hidden=True)          # hidden = on-demand by default
def web_search(query: str, limit: int = 5) -> list[str]:
    ...

@tool                                                      # schema inferred from the signature & docstring
async def send_email(to: str, body: str) -> str:
    """Send an email."""

registry.register(web_search)
registry.register(send_email)
registry.register(my_plain_function, name="fetch", description="HTTP GET", parameters={...})  # explicit schema
engine = Engine(backend, tools=registry)
```

Sync tools run on the engine's thread pool; async tools are awaited. Arguments are validated against the schema before execution; validation errors and exceptions are returned to the model as `Error: …` tool results (the task does not crash). Results are serialised to JSON for the model and truncated at `max_context_chars`.

```python
@tool(description="Send an email", approval="ask")      # offered, but each call waits for a human's OK
async def send_email(to: str, body: str, ctx) -> str:   # ctx: a ToolContext (run_id, dag_id, task_id, instance_id, run, engine)
    ...
```

`approval="ask"` tools are offered to the model, and each call pauses the task for approval unless the task's `allowed_tools` names the tool; `approval="always"` ones always pause. A function with a `ctx` parameter gets a `ToolContext` saying which run, task and instance called it (`ctx` isn't part of the tool's schema).

If your framework already owns tool execution, expose its tools through the registry (thin wrappers) rather than executing them inside `generate`; otherwise the permission model and approvals cannot work.

## 3. Skills

```python
engine.skills.register("Research", "Cite sources. Prefer primary documents.")
engine.skills.load_directory("skills/")   # one <name>.md per skill
```

Skills referenced by `skills=[...]` at any scope are rendered into the system prompt as `### Skill: <name> (v<n>)` blocks. Versions are kept; `DagEvaluator` rewrites become new versions only after approval.

## 4. Observability

```python
engine.subscribe(lambda event: log.info("%s %s %s", event.type, event.task_id, event.payload))   # everything
engine = Engine(backend, chat_sink=my_chat.post)   # receives only what each task's verbosity allows
```

Event types: `run.created|started|paused|resumed|finished|detached|summary`, `node.finished` (all of a node's instances have finished), `task.started|finished|failed|skipped|upstream_failed|rerun_requested|awaiting_tool_approval|awaiting_user`, `tool_call`, `reasoning`, `response`, `approval.requested|resolved`, `recommendation.created|approved|rejected`, `memory.updated`, `asset.updated`, `chat.answered`. Each `Event` carries `run_id`, `dag_id`, `task_id`, `instance_id`, `payload`, `timestamp`.

`run.dag_context` is the append-only structured log (`{timestamp, task_id, instance_id, status, input_context, tool_calls, output, error, attempt}`), available during and after the run and used by `engine.chat()` and the DagEvaluator.

## 5. Live user chat

```python
answer = await engine.chat("Which emails did you archive?", run_id=run.run_id)
```

The question is answered from the run logs at priority 0 on the model's queue: it executes as soon as the running task finishes (or, with `atomic="step"`, the running inference step), before any queued DAG work. Pass `history=[Message(...)]` for multi-turn chat.

If your application has its own conversation with the same model, hold the model the same way, so it neither interrupts a DAG task nor waits behind queued ones:

```python
async with engine.user_slot("qwen-3.6"):
    reply = await my_chat_model.answer(message)
```

## 6. Human in the loop

```python
for approval in engine.list_approvals():
    if approval.kind == "tool_call":
        await engine.approve(approval.approval_id, remember=True)   # or engine.reject(id, reason="...")
    elif approval.kind == "evaluator_exhausted":
        await engine.decide(approval.approval_id, "pass")           # "retry" | "fail"
```

`engine.run()`/`engine.wait()` return when the run finishes **or pauses** (`stop_on_pause=True`), so a script can resolve approvals and `wait()` again. Approvals are persisted and survive restarts.

## 7. Persistence and recovery

```python
engine = Engine(backend, state=SqliteStateStore("orchestrator.db"))
...
await engine.resume_interrupted()   # after a crash/restart: continues non-terminal runs from checkpoints
```

Implement the `StateStore` protocol (plain JSON documents) to use another database. `engine.stop()` detaches from running runs without cancelling them; `engine.cancel(run_id)` cancels explicitly.

## 8. Triggers

```python
await engine.trigger("inbox", payload={"since": "yesterday"})            # manual / API
await engine.deliver_webhook("gh_push", raw_body=body, signature=sig)  # WebhookTrigger("gh_push", secret=...)
```

Cron triggers fire from the scheduler loop started by `engine.start()` / `async with engine`. Missed schedules (while the process was down) are not back-filled, and a cron firing doesn't queue another run while one of the DAG's is still waiting to start. `DagTrigger("other")` runs are triggered with payload `{"triggered_by", "dag_id", "status", "outputs"}`. `Asset("name")` runs start when a task with the asset in its `outlets` produces something, or on `await engine.update_asset("name", payload)`.

`engine.pause(dag_id)` stops everything but `trigger()` (a manual run) from starting a DAG, and survives restarts; `engine.unpause(dag_id)` undoes it.
