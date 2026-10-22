"""Wire a local OpenAI-compatible model (vLLM, Ollama, llama.cpp, LM Studio).

    OPENAI_BASE_URL=http://localhost:11434/v1 MODEL=qwen3:8b python examples/local_model.py

Requires a running server; not executed by the test-suite.
"""

from __future__ import annotations

import asyncio
import os

from agentic_airflow import AgenticEvaluator, AgenticTask, Dag, Engine, EngineConfig, SqliteStateStore, Task, ToolRegistry, tool
from agentic_airflow.backends.openai_compatible import OpenAICompatibleBackend

registry = ToolRegistry()


@tool(description="Return the current date in ISO format")
def today() -> str:
    import datetime

    return datetime.date.today().isoformat()


@tool(description="Read a local text file", hidden=True)
def read_file(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()[:4000]


registry.register(today)
registry.register(read_file)


async def main() -> None:
    backend = OpenAICompatibleBackend(
        os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        # vLLM/Qwen style thinking toggle; adjust for your server or drop it.
        thinking_param=lambda on: {"chat_template_kwargs": {"enable_thinking": on}},
    )
    engine = Engine(
        backend,
        config=EngineConfig(model=os.environ.get("MODEL", "qwen3:8b"), model_concurrency={}, default_concurrency=1),
        tools=registry,
        state=SqliteStateStore("local_model.db"),
        chat_sink=lambda e: print(f"[{e.task_id}] {e.type}: {e.payload}"),
    )
    engine.skills.register("Brevity", "Answer in at most three sentences.")

    with Dag("brief", prompt="Produce a short, dated status brief.", skills=["Brevity"], allowed_tools=["today", "read_file"], verbosity=1) as dag:
        seed = Task("seed", fn=lambda: ["asyncio", "sqlite"], fan_out=True)
        write = AgenticTask("write", prompt="Write two sentences about this Python topic, prefixed with today's date (use the tool).", thinking="on")
        review = AgenticEvaluator("review", prompt="Reject if the date is missing or the text exceeds three sentences.", max_retries=2, on_exhausted="pass_latest")
        combine = AgenticTask("combine", prompt="Merge the reviewed snippets into one brief.")
        seed >> write >> review >> combine
    engine.register(dag)

    async with engine:
        run = await engine.run("brief", timeout=600)
        print(run.status, run.instances["combine"].output)
        print("summary:", run.summary)
    engine.close()


if __name__ == "__main__":
    asyncio.run(main())
