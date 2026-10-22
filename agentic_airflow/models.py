"""Model-facing data types and the :class:`ModelBackend` plug-in protocol.

This module is the seam between agentic-airflow and *your* agent framework.
The engine never talks to an LLM directly; it builds a
:class:`GenerationRequest` describing **one inference step** and hands it to
whatever object you pass as ``backend=`` to :class:`~agentic_airflow.engine.Engine`.
The backend returns a :class:`GenerationResponse` containing either a final
answer or a list of tool calls. The engine executes the tools, enforces
permissions, checkpoints, and calls the backend again.

Implementing a backend is therefore a single ``async def generate(request)``
method. See :mod:`agentic_airflow.backends` for ready-made adapters and
``docs/plugging_in.md`` for a walkthrough.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]
OutputKind = Literal["string", "list", "json", "enum", "image", "file"]


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: _new_id("call"))

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": dict(self.arguments)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        return cls(
            name=str(data["name"]),
            arguments=dict(data.get("arguments") or {}),
            id=str(data.get("id") or _new_id("call")),
        )


@dataclass
class Message:
    """One entry of the conversation held by an agentic task.

    * ``role="tool"`` messages carry a tool result and reference the
      originating call through ``tool_call_id``.
    * ``role="assistant"`` messages may carry ``tool_calls``.
    * ``reasoning`` holds chain-of-thought text when the backend exposes it.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.reasoning is not None:
            data["reasoning"] = self.reasoning
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=str(data.get("content") or ""),
            tool_calls=[ToolCall.from_dict(c) for c in data.get("tool_calls") or []],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
            reasoning=data.get("reasoning"),
        )


@dataclass
class ToolSpec:
    """A JSON-schema description of a tool, as injected into the model context."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class OutputSpec:
    """Describes the structured-output contract the engine expects back.

    * ``kind="string"``: free text (``response.text``).
    * ``kind="list"``: a JSON array (``response.output`` or JSON in ``text``).
    * ``kind="json"``: a JSON object; ``json_schema`` may constrain it.
    * ``kind="enum"``: one of ``choices`` (or several if ``multi``).
    * ``kind="image"`` / ``"file"``: a binary payload or path, see
      :func:`agentic_airflow.validation.coerce_output`.

    Backends that support constrained decoding should use ``json_schema`` /
    ``choices``; backends that do not can simply prompt and let the engine
    validate.
    """

    kind: OutputKind = "string"
    json_schema: dict[str, Any] | None = None
    choices: list[str] | None = None
    multi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "json_schema": self.json_schema,
            "choices": list(self.choices) if self.choices is not None else None,
            "multi": self.multi,
        }

    def instructions(self) -> str:
        """Human-readable contract appended to the system prompt."""
        if self.kind == "string":
            return "Respond with your final answer as plain text."
        if self.kind == "list":
            return "Respond with ONLY a JSON array (e.g. [\"item 1\", \"item 2\"]). No prose."
        if self.kind == "json":
            if self.json_schema:
                import json

                return (
                    "Respond with ONLY a JSON object that conforms to this JSON schema. No prose.\n"
                    + json.dumps(self.json_schema, indent=2)
                )
            return "Respond with ONLY a JSON object. No prose."
        if self.kind == "enum":
            opts = ", ".join(repr(c) for c in (self.choices or []))
            if self.multi:
                return (
                    "Respond with ONLY a JSON array containing one or more of these exact values: "
                    f"{opts}."
                )
            return f"Respond with ONLY one of these exact values and nothing else: {opts}."
        if self.kind == "image":
            return (
                "Respond with an image. Return it as a JSON object "
                '{"mime": "<mime type>", "data_base64": "<base64>"} or a local file path.'
            )
        if self.kind == "file":
            return 'Respond with ONLY the path of the produced file, or a JSON object {"path": "..."}.'
        return ""


@dataclass
class GenerationRequest:
    """Everything a backend needs to perform exactly one inference step."""

    model: str
    system: str
    messages: list[Message]
    tools: list[ToolSpec] = field(default_factory=list)
    output: OutputSpec = field(default_factory=OutputSpec)
    thinking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    """Routing/observability info: ``dag_id``, ``run_id``, ``task_id``,
    ``instance_id``, ``attempt``, ``step``, ``priority`` (0 = live user)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system": self.system,
            "messages": [m.to_dict() for m in self.messages],
            "tools": [t.to_dict() for t in self.tools],
            "output": self.output.to_dict(),
            "thinking": self.thinking,
            "metadata": dict(self.metadata),
        }


@dataclass
class GenerationResponse:
    """The result of one inference step.

    Exactly one of these should be meaningful:

    * ``tool_calls`` non-empty: the engine executes them and loops.
    * otherwise the step is final: ``output`` (already-structured value) takes
      precedence over ``text`` when validating against the task's
      ``return_type``.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    output: Any = None
    reasoning: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "output": self.output,
            "reasoning": self.reasoning,
            "usage": dict(self.usage),
        }


@runtime_checkable
class ModelBackend(Protocol):
    """The plug-in interface for LLM providers / agent frameworks.

    ``generate`` must perform **one** model call (not an internal tool loop:
    the engine owns the loop so it can enforce permissions and checkpoint
    between steps). It must be safe to call concurrently.
    """

    async def generate(self, request: GenerationRequest) -> GenerationResponse:  # pragma: no cover - protocol
        ...


class FunctionBackend:
    """Adapt a plain (sync or async) callable into a :class:`ModelBackend`.

    The callable receives the :class:`GenerationRequest` and returns either a
    :class:`GenerationResponse`, a ``str`` (treated as final text) or a
    ``dict`` with ``GenerationResponse`` fields.
    """

    def __init__(self, fn: Callable[[GenerationRequest], Any]):
        self._fn = fn

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        result = self._fn(request)
        if inspect.isawaitable(result):
            result = await result
        return coerce_response(result)


def coerce_response(result: Any) -> GenerationResponse:
    """Normalise loosely-typed backend results into a :class:`GenerationResponse`."""
    if isinstance(result, GenerationResponse):
        return result
    if result is None:
        return GenerationResponse(text="")
    if isinstance(result, str):
        return GenerationResponse(text=result)
    if isinstance(result, dict):
        return GenerationResponse(
            text=str(result.get("text") or ""),
            tool_calls=[
                c if isinstance(c, ToolCall) else ToolCall.from_dict(c)
                for c in (result.get("tool_calls") or [])
            ],
            output=result.get("output"),
            reasoning=result.get("reasoning"),
            usage=dict(result.get("usage") or {}),
            raw=result.get("raw"),
        )
    return GenerationResponse(output=result)


class ScriptedBackend:
    """A deterministic backend for tests and examples.

    Responses are queued per task id (``metadata["task_id"]``) or globally.
    Each queued item may be a :class:`GenerationResponse`, a ``str``, a
    ``dict`` or a callable ``fn(request) -> response``. Every request is
    recorded in :attr:`requests` so tests can assert on prompts, tools, etc.

    When a queue is exhausted the ``default`` response is used (a callable or
    value); if no default is set an error is raised, which surfaces as a task
    failure - handy for catching unexpected model calls in tests.
    """

    def __init__(self, default: Any = None):
        self._by_task: dict[str, list[Any]] = {}
        self._global: list[Any] = []
        self.default = default
        self.requests: list[GenerationRequest] = []
        self._lock = asyncio.Lock()

    def add(self, task_id: str | None, *responses: Any) -> "ScriptedBackend":
        """Queue responses for ``task_id`` (or globally when ``task_id`` is None)."""
        target = self._global if task_id is None else self._by_task.setdefault(task_id, [])
        target.extend(responses)
        return self

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        async with self._lock:
            self.requests.append(request)
            task_id = request.metadata.get("task_id")
            queue = self._by_task.get(task_id or "")
            if queue:
                item = queue.pop(0)
            elif self._global:
                item = self._global.pop(0)
            elif self.default is not None:
                item = self.default
            else:
                raise RuntimeError(
                    f"ScriptedBackend has no response queued for task {task_id!r}"
                )
        if callable(item) and not isinstance(item, GenerationResponse):
            item = item(request)
            if inspect.isawaitable(item):
                item = await item
        return coerce_response(item)
