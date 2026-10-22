"""Run the engine with the HTTP UI and a demo DAG driven by a fake model.

    python examples/serve_ui.py           # then open the printed URL

Trigger more runs with:  curl -X POST -H "Authorization: Bearer demo" \
    -d '{"dag_id": "demo"}' http://127.0.0.1:8800/api/runs
"""

from __future__ import annotations

import asyncio
import random

from agentic_airflow import AgenticEvaluator, AgenticTask, Dag, Engine, EngineConfig, ScriptedBackend, Task
from agentic_airflow.ui.server import UIServer


async def slow_model(req):
    await asyncio.sleep(random.uniform(0.3, 1.0))  # pretend to think
    task = req.metadata["task_id"]
    if task == "review":
        return {"output": {"rerun": req.metadata["attempt"] == 1, "reason": "Add more detail."}}
    if req.output.kind == "json":
        return {"output": {"score": 7, "assessment": "fine", "prompt_changes": [], "structural_changes": [], "hide": [], "allow": [], "rationale": "", "skills": [], "memory": {"runs": req.metadata["run_id"]}}}
    return f"{task} output (attempt {req.metadata['attempt']})"


async def main() -> None:
    with Dag("demo", prompt="Research a topic and write a short brief.", model="demo-model") as dag:
        topics = Task("topics", fn=lambda: ["agents", "schedulers", "graphs"], fan_out=True)
        research = AgenticTask("research", prompt="Research this topic.")
        draft = AgenticTask("draft", prompt="Write a brief.", follow_mapping=True)
        review = AgenticEvaluator("review", prompt="Check the brief.", max_retries=1, on_exhausted="pass_latest")
        publish = AgenticTask("publish", prompt="Combine the briefs.")
        topics >> research >> draft >> review >> publish

    engine = Engine(ScriptedBackend(default=slow_model), config=EngineConfig(model="demo-model"))
    engine.register(dag)
    async with engine:
        server = UIServer(engine, port=8800, token="demo").start()
        run = await engine.trigger("demo")
        print(f"UI: {server.url}/dags/demo?run={run.run_id}&token=demo")
        print(f"API: {server.url}/api/runs/{run.run_id}/graph  (Authorization: Bearer demo)")
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            server.stop()


if __name__ == "__main__":
    asyncio.run(main())
