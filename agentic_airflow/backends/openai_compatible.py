"""Backend for OpenAI-style ``/v1/chat/completions`` endpoints (stdlib only).

Works with vLLM, Ollama, llama.cpp server, LM Studio, OpenRouter, OpenAI and
anything else speaking the chat-completions dialect - the natural choice for
local models such as the ``qwen-3.6-25B-A3B`` example in the spec.

.. code-block:: python

    backend = OpenAICompatibleBackend("http://localhost:8000/v1", api_key=None)
    engine = Engine(backend, config=EngineConfig(model="qwen-3.6-25B-A3B"))

Requests run in a worker thread (``asyncio.to_thread``) so the event loop is
never blocked. Structured outputs use ``response_format`` (``json_schema`` /
``json_object``) when ``json_mode`` is on; the engine still validates every
response, so servers that ignore ``response_format`` work too.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import urllib.error
import urllib.request
from typing import Any, Callable

from ..errors import AgenticAirflowError
from ..models import GenerationRequest, GenerationResponse, ToolCall


class BackendError(AgenticAirflowError):
    """The remote endpoint returned an error."""


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int | None = None,
        json_mode: bool = True,
        thinking_param: Callable[[bool], dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        model_map: dict[str, str] | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ):
        """
        :param base_url: e.g. ``http://localhost:11434/v1``.
        :param thinking_param: callable mapping the engine's ``thinking`` flag to
            extra request fields (servers differ: e.g. ``lambda on: {"chat_template_kwargs": {"enable_thinking": on}}``
            for vLLM/Qwen, ``lambda on: {"reasoning": {"effort": "high" if on else "low"}}`` for others).
        :param model_map: translate engine model ids to server model names.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.thinking_param = thinking_param
        self.extra_body = dict(extra_body or {})
        self.extra_headers = dict(extra_headers or {})
        self.model_map = dict(model_map or {})
        self.ssl_context = ssl_context

    # -- request building ----------------------------------------------------
    def build_body(self, request: GenerationRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": request.system}]
        for m in request.messages:
            if m.role == "tool":
                messages.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                        for c in m.tool_calls
                    ]
                messages.append(entry)
            else:
                messages.append({"role": m.role, "content": m.content})
        body: dict[str, Any] = {
            "model": self.model_map.get(request.model, request.model),
            "messages": messages,
        }
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in request.tools
            ]
            body["tool_choice"] = "auto"
        spec = request.output
        if self.json_mode and not request.tools:
            if spec.kind == "json" and spec.json_schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": spec.json_schema}}
            elif spec.kind == "enum" and spec.choices:
                if spec.multi:
                    schema = {"type": "array", "items": {"type": "string", "enum": list(spec.choices)}}
                else:
                    schema = {"type": "string", "enum": list(spec.choices)}
                body["response_format"] = {"type": "json_schema", "json_schema": {"name": "selection", "schema": schema}}
            elif spec.kind in ("json", "list"):
                body["response_format"] = {"type": "json_object"} if spec.kind == "json" else {
                    "type": "json_schema",
                    "json_schema": {"name": "items", "schema": spec.json_schema or {"type": "array"}},
                }
        if self.thinking_param is not None:
            body.update(self.thinking_param(request.thinking))
        body.update(self.extra_body)
        return body

    # -- transport -------------------------------------------------------------
    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise BackendError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"Could not reach {self.base_url}: {exc.reason}") from exc

    def parse(self, payload: dict[str, Any]) -> GenerationResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise BackendError(f"No choices in response: {json.dumps(payload)[:500]}")
        message = choices[0].get("message") or {}
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            tool_calls.append(ToolCall(name=str(fn.get("name") or ""), arguments=args if isinstance(args, dict) else {"value": args}, id=str(call.get("id") or "")))
        usage = payload.get("usage") or {}
        return GenerationResponse(
            text=message.get("content") or "",
            tool_calls=tool_calls,
            reasoning=message.get("reasoning_content") or message.get("reasoning"),
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
            raw=payload,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        body = self.build_body(request)
        payload = await asyncio.to_thread(self._post, body)
        return self.parse(payload)
