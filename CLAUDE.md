# Working on agentic-airflow

* Core package: `agentic_airflow/` (no runtime dependencies; Python >= 3.10). Keep it that way: optional integrations go under `agentic_airflow/backends/` behind optional extras.
* Run tests with `python -m pytest` (asyncio mode is auto). Lint with `python -m pyflakes agentic_airflow tests`.
* Start with `docs/agent_guide.md` for orientation; `docs/architecture.md` explains the run driver and the node expansion algorithm; `docs/spec_deviations.md` records deliberate departures from the original spec. Update those docs when you change behaviour.
* All persisted state must stay JSON-serialisable (`DagRun.to_dict()` / `from_dict()`).
* Security rules in `docs/security.md` are load-bearing: permission checks live in the engine, never only in prompts.
