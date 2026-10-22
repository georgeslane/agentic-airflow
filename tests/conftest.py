import pytest

from agentic_airflow import Engine, EngineConfig, ScriptedBackend, ToolRegistry, tool


@pytest.fixture
def backend():
    return ScriptedBackend()


@pytest.fixture
def tools():
    registry = ToolRegistry()

    @tool(description="Send an email to someone")
    def send_email(to: str, body: str) -> str:
        return f"sent to {to}"

    @tool(description="Search the web for information", hidden=True)
    def web_search(query: str, limit: int = 3) -> list:
        return [f"result for {query}"] * limit

    @tool(description="Delete a file (dangerous)")
    def delete_file(path: str) -> str:
        return f"deleted {path}"

    registry.register(send_email)
    registry.register(web_search)
    registry.register(delete_file)
    return registry


@pytest.fixture
def make_engine(backend, tools):
    engines = []

    def _make(**kwargs):
        kwargs.setdefault("config", EngineConfig(model="test-model"))
        kwargs.setdefault("tools", tools)
        engine = Engine(backend, **kwargs)
        engines.append(engine)
        return engine

    yield _make
    for e in engines:
        e.close()
