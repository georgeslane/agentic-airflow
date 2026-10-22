"""Context-manager stack used by ``with Dag(...)`` / ``with TaskGroup(...)``."""

from __future__ import annotations

import contextvars
from typing import Any

_stack: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar("agentic_airflow_scope_stack", default=())


def push(scope: Any) -> contextvars.Token:
    return _stack.set((*_stack.get(), scope))


def pop(token: contextvars.Token) -> None:
    _stack.reset(token)


def current_group() -> Any | None:
    """The innermost active Dag or TaskGroup, or None."""
    stack = _stack.get()
    return stack[-1] if stack else None


def current_dag() -> Any | None:
    for scope in reversed(_stack.get()):
        dag = getattr(scope, "dag", None)
        if dag is not None:
            return dag
        if getattr(scope, "is_dag", False):
            return scope
    return None
