"""Backend for Claude models via the official ``anthropic`` SDK.

Install with ``pip install agentic-airflow[anthropic]``. One engine inference
step maps to one ``messages.create`` call: tool calls come back as
:class:`~agentic_airflow.models.ToolCall` objects and the engine runs its own
permission-aware loop.

.. code-block:: python

    from agentic_airflow.backends.anthropic import AnthropicBackend
    engine = Engine(AnthropicBackend(), config=EngineConfig(model="claude-opus-5-5"))

Notes
-----
* ``thinking=True`` sends adaptive thinking (``{"type": "adaptive"}``); when
  ``thinking`` is off the parameter is omitted so the model's default applies
  (recent Claude models cannot disable thinking; use ``effort`` to lower it).
* ``return_type="json"``/enum contracts are enforced with structured outputs
  (``output_config.format``) when no tools are offered in that step.
* Server-side refusal fallbacks are enabled by default (``fallbacks="default"``
  with the matching beta header); pass ``fallbacks=None`` to disable.
"""

from __future__ import annotations

import json
from typing import Any

from ..errors import AgenticAirflowError
from ..models import GenerationRequest, GenerationResponse, ToolCall

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicBackend:
    def __init__(
        self,
        client: Any | None = None,
        *,
        max_tokens: int = 16000,
        effort: str | None = None,
        thinking_display: str | None = None,
        fallbacks: Any = "default",
        extra: dict[str, Any] | None = None,
    ):
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover
                raise AgenticAirflowError("pip install anthropic (or agentic-airflow[anthropic]) to use AnthropicBackend") from exc
            client = AsyncAnthropic()
        self.client = client
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking_display = thinking_display
        self.fallbacks = fallbacks
        self.extra = dict(extra or {})

    def build_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush_results() -> None:
            if pending_results:
                messages.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in request.messages:
            if m.role == "tool":
                pending_results.append({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content})
                continue
            flush_results()
            if m.role == "assistant":
                content: list[dict[str, Any]] = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for c in m.tool_calls:
                    content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif m.role == "system":
                messages.append({"role": "user", "content": m.content})
            else:
                messages.append({"role": "user", "content": m.content})
        flush_results()

        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": messages,
        }
        if request.tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools
            ]
        if request.thinking:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if self.thinking_display:
                thinking["display"] = self.thinking_display
            kwargs["thinking"] = thinking
        output_config: dict[str, Any] = {}
        if self.effort:
            output_config["effort"] = self.effort
        spec = request.output
        if not request.tools:
            schema = None
            if spec.kind == "json" and spec.json_schema:
                schema = spec.json_schema
            elif spec.kind == "enum" and spec.choices:
                item = {"type": "string", "enum": list(spec.choices)}
                schema = {"type": "object", "properties": {"selection": {"type": "array", "items": item} if spec.multi else item}, "required": ["selection"], "additionalProperties": False}
            if schema is not None:
                output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            kwargs["output_config"] = output_config
        if self.fallbacks is not None:
            kwargs["fallbacks"] = self.fallbacks
            kwargs["betas"] = [FALLBACK_BETA]
        kwargs.update(self.extra)
        return kwargs

    def parse(self, response: Any, request: GenerationRequest) -> GenerationResponse:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise AgenticAirflowError(f"Model refused the request ({getattr(details, 'category', None)}): {getattr(details, 'explanation', '')}")
        text_parts: list[str] = []
        reasoning: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking" and getattr(block, "thinking", ""):
                reasoning.append(block.thinking)
            elif btype == "tool_use":
                tool_calls.append(ToolCall(name=block.name, arguments=dict(block.input or {}), id=block.id))
        text = "".join(text_parts)
        output: Any = None
        spec = request.output
        if not tool_calls and spec.kind == "enum" and text.strip():
            try:
                output = json.loads(text).get("selection")
            except (ValueError, AttributeError):
                output = None
        usage = getattr(response, "usage", None)
        usage_dict = {}
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            value = getattr(usage, key, None)
            if isinstance(value, int):
                usage_dict[key] = value
        return GenerationResponse(text=text, tool_calls=tool_calls, output=output, reasoning="\n".join(reasoning) or None, usage=usage_dict, raw=response)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        kwargs = self.build_kwargs(request)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        response = await api.create(**kwargs)
        return self.parse(response, request)
