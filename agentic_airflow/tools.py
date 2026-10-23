"""Tool registry, permission resolution and the three-state visibility model.

A :class:`Tool` wraps a Python callable with a JSON-schema description. The
:class:`ToolRegistry` is the *system* list of tools; permission resolution
(:func:`resolve_permissions`) turns a task's resolved configuration into the
three sets from spec §1:

* **allowed / visible** - schema injected into the prompt, executes directly;
* **hidden / on-demand** - allowed, but not injected until the agent calls the
  built-in ``search_tools`` tool;
* **gated** - tools registered with ``approval="ask"`` (unless ``allowed_tools``
  names them) or ``approval="always"``: injected, but every call pauses the
  task for human approval;
* **denied** - explicitly denied or outside ``allowed_tools``. Invocation
  pauses the task for human approval.
"""

from __future__ import annotations

import asyncio
import fnmatch
import functools
import inspect
import json
import re
import typing
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from .config import ResolvedConfig
from .errors import ToolError
from .models import ToolSpec
from .validation import schema_errors

TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SEARCH_TOOL_NAME = "search_tools"
EXPLICIT = "explicit"  # a tool's tag: enabled only by its name, never by a pattern (resolve_permissions)

ToolState = Literal["allowed", "hidden", "gated", "denied", "unknown"]
Approval = Literal["none", "ask", "always"]

_PY_TO_JSON = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    list: {"type": "array"},
    dict: {"type": "object"},
    type(None): {"type": "null"},
}


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation in _PY_TO_JSON:
        return dict(_PY_TO_JSON[annotation])
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (list, typing.List):
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _schema_for_annotation(args[0])
        return schema
    if origin in (dict, typing.Dict):
        return {"type": "object"}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _schema_for_annotation(non_none[0])
            if schema.get("type"):
                schema["type"] = [schema["type"], "null"]
            return schema
        return {"anyOf": [_schema_for_annotation(a) for a in args]}
    return {}


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Infer a JSON-schema ``parameters`` object from a function signature."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - forward refs etc.
        hints = {}
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) or name == "ctx":
            continue
        props[name] = _schema_for_annotation(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            props[name]["default"] = param.default if _jsonable(param.default) else None
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class ToolContext:
    """Passed to tools whose function declares a ``ctx`` parameter: who is calling."""

    run_id: str
    dag_id: str
    task_id: str
    instance_id: str
    map_path: tuple[int, ...] = ()
    attempt: int = 1
    engine: Any = None
    run: Any = None


@dataclass
class Tool:
    """A callable exposed to agents.

    ``fn`` may be sync or async. Sync functions are executed on the engine's
    CPU worker pool so they never block the event loop. ``hidden=True`` makes
    the tool on-demand by default (discoverable via ``search_tools``).

    ``approval`` says whether a call needs a human's OK: ``"none"`` (default);
    ``"ask"``, unless the task's ``allowed_tools`` names the tool explicitly;
    or ``"always"``, whatever ``allowed_tools`` says. A tool that needs it is
    still offered to the model; calling it pauses the task for approval.

    A function with a ``ctx`` parameter receives a :class:`ToolContext`.
    """

    name: str
    fn: Callable[..., Any]
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    hidden: bool = False
    tags: tuple[str, ...] = ()
    approval: Approval = "none"

    def __post_init__(self) -> None:
        if not TOOL_NAME_RE.match(self.name):
            raise ToolError(f"Invalid tool name {self.name!r}")
        if not isinstance(self.parameters, dict):
            raise ToolError(f"Tool {self.name!r}: parameters must be a JSON-schema dict")
        if self.approval not in ("none", "ask", "always"):
            raise ToolError(f"Tool {self.name!r}: approval must be 'none', 'ask' or 'always'")
        try:
            self.wants_ctx = "ctx" in inspect.signature(self.fn).parameters
        except (TypeError, ValueError):
            self.wants_ctx = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def validate_arguments(self, arguments: Any) -> list[str]:
        if not isinstance(arguments, dict):
            return ["arguments must be a JSON object"]
        return schema_errors(arguments, self.parameters)

    async def invoke(
        self, arguments: dict[str, Any], *, executor: Executor | None = None, ctx: ToolContext | None = None
    ) -> Any:
        """Validate and execute the tool, returning its raw result."""
        errs = self.validate_arguments(arguments)
        if errs:
            raise ToolError(f"Invalid arguments for {self.name}: " + "; ".join(errs))
        kwargs = dict(arguments)
        if self.wants_ctx:
            kwargs["ctx"] = ctx
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(executor, functools.partial(self.fn, **kwargs))
        if inspect.isawaitable(result):
            result = await result
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "hidden": self.hidden,
            "tags": list(self.tags),
            "approval": self.approval,
        }


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
    hidden: bool = False,
    tags: Iterable[str] = (),
    approval: Approval = "none",
) -> Any:
    """Decorator turning a function into a :class:`Tool`.

    .. code-block:: python

        @tool(description="Search the web")
        def search(query: str, limit: int = 5) -> list[str]: ...

    The decorated object *is* a :class:`Tool`; call ``search.fn(...)`` to use
    the underlying function directly.
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            fn=f,
            description=(description if description is not None else (inspect.getdoc(f) or "")).strip(),
            parameters=parameters if parameters is not None else schema_from_signature(f),
            hidden=hidden,
            tags=tuple(tags),
            approval=approval,
        )

    return wrap(fn) if fn is not None else wrap


class ToolRegistry:
    """System-wide collection of tools.

    Tools are registered one by one, or come from providers: functions an
    application adds with :meth:`add_provider`, called whenever the registry
    is consulted, for tool sets that change (say, an MCP server's). A
    registered tool wins over a provider's of the same name.
    """

    def __init__(self, tools: Iterable[Tool | Callable[..., Any]] = ()):
        self._tools: dict[str, Tool] = {}
        self._providers: list[Callable[[], Iterable[Tool]]] = []
        for t in tools:
            self.register(t)

    def register(self, item: Tool | Callable[..., Any], **kwargs: Any) -> Tool:
        t = item if isinstance(item, Tool) else tool(item, **kwargs)
        if t.name == SEARCH_TOOL_NAME:
            raise ToolError(f"{SEARCH_TOOL_NAME!r} is reserved for the built-in registry lookup tool")
        self._tools[t.name] = t
        return t

    def add_provider(self, provider: Callable[[], Iterable[Tool]]) -> None:
        self._providers.append(provider)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def _all(self) -> dict[str, Tool]:
        if not self._providers:
            return self._tools
        found: dict[str, Tool] = {}
        for provider in self._providers:
            for t in provider():
                if t.name != SEARCH_TOOL_NAME:
                    found.setdefault(t.name, t)
        return {**found, **self._tools}

    def get(self, name: str) -> Tool | None:
        return self._all().get(name)

    def names(self) -> list[str]:
        return list(self._all())

    def __contains__(self, name: object) -> bool:
        return name in self._all()

    def __len__(self) -> int:
        return len(self._all())

    def __iter__(self):
        return iter(self._all().values())

    def hidden_by_default(self) -> set[str]:
        return {t.name for t in self._all().values() if t.hidden}

    def specs(self, names: Iterable[str]) -> list[ToolSpec]:
        tools = self._all()
        return [tools[n].spec() for n in names if n in tools]

    def search(self, query: str, within: Iterable[str] | None = None, limit: int = 10) -> list[Tool]:
        """Case-insensitive keyword search over names, descriptions and tags."""
        tools = self._all()
        words = [w for w in re.split(r"\W+", query.lower()) if w]
        pool = [tools[n] for n in (within if within is not None else tools) if n in tools]
        scored: list[tuple[int, Tool]] = []
        for t in pool:
            hay = f"{t.name} {t.description} {' '.join(t.tags)}".lower()
            score = sum(hay.count(w) for w in words) if words else 1
            if score:
                scored.append((score, t))
        scored.sort(key=lambda s: (-s[0], s[1].name))
        return [t for _, t in scored[:limit]]


@dataclass(frozen=True)
class ToolPermissions:
    """Effective permissions for one task run."""

    allowed: frozenset[str]
    """Tools the task may execute without approval (includes hidden ones)."""
    hidden: frozenset[str]
    """Subset of ``allowed | gated`` that is on-demand (not injected until discovered)."""
    denied: frozenset[str]
    """Registered tools the task may not call without approval; never offered."""
    gated: frozenset[str] = frozenset()
    """Tools offered to the task, each call of which pauses for approval."""
    always: frozenset[str] = frozenset()
    """Subset of ``gated`` whose approval can't be remembered for the rest of a run."""
    refuse_denied: bool = False
    """A call to a denied tool fails, saying so, rather than pausing for approval: it isn't one of the task's
    tools at all (``enabled``)."""

    @property
    def offered(self) -> frozenset[str]:
        """Tools the task may call, with or without approval: what it's offered or can discover."""
        return self.allowed | self.gated

    @property
    def visible(self) -> frozenset[str]:
        return self.offered - self.hidden

    def state(self, name: str, registered: bool = True) -> ToolState:
        if name in self.allowed:
            return "hidden" if name in self.hidden else "allowed"
        if name in self.gated:
            return "gated"
        if registered or name in self.denied:
            return "denied"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": sorted(self.allowed - self.hidden),
            "gated": sorted(self.gated - self.hidden),
            "hidden": sorted(self.hidden),
            "denied": sorted(self.denied),
        }


def resolve_permissions(
    config: ResolvedConfig,
    registry: ToolRegistry,
    overrides: dict[str, Any] | None = None,
) -> ToolPermissions:
    """Compute :class:`ToolPermissions` for a resolved task configuration.

    ``overrides`` is the persisted, human-approved adjustment produced by the
    :class:`~agentic_airflow.evaluator.DagEvaluator` (``{"allow": [...],
    "hide": [...], "deny": [...]}``). Explicit ``denied_tools`` from the
    configuration cascade still win over an ``allow`` override.

    Names in the lists may be shell-style patterns (``"news_*"``), matched
    against the registered tools, so they cover tools that come and go.

    With an ``allowed_tools`` list, the tools that ask first and aren't named
    on it are hidden: found with ``search_tools`` when the task needs one, and
    still asking, rather than offered in every prompt.
    """
    tools = {t.name: t for t in registry}  # once: a provider's tools are made again each time they're listed
    registered = set(tools)
    overrides = overrides or {}

    def matching(patterns: Iterable[str]) -> set[str]:
        return {name for p in patterns for name in fnmatch.filter(registered, p)}

    if config.enabled is not None:
        return _enabled(config, tools, matching, overrides)

    named = matching([*(config.allowed_tools or []), *(overrides.get("allow") or [])])
    denied_explicit = matching([*config.denied_tools, *(overrides.get("deny") or [])])
    allowed: set[str] = set()
    gated: set[str] = set()
    always: set[str] = set()
    for name in registered - denied_explicit:
        approval = tools[name].approval
        if approval == "always":
            gated.add(name)
            always.add(name)
        elif approval == "ask":
            (allowed if name in named else gated).add(name)
        elif config.allowed_tools is None or name in named:
            allowed.add(name)
    denied = registered - allowed - gated
    hidden = (matching(config.hidden_tools) | {n for n, t in tools.items() if t.hidden} | matching(overrides.get("hide") or [])) & (
        allowed | gated
    )
    if config.allowed_tools is not None:  # a list is what the task needs: others that ask are found when needed
        hidden |= gated - named
    return ToolPermissions(
        allowed=frozenset(allowed),
        hidden=frozenset(hidden),
        denied=frozenset(denied),
        gated=frozenset(gated),
        always=frozenset(always),
    )


def _enabled(
    config: ResolvedConfig, tools: dict[str, Tool], matching: Callable[[Iterable[str]], set[str]], overrides: dict[str, Any]
) -> ToolPermissions:
    """Permissions once a scope sets ``enabled``: a task can use only the tools it names, less ``denied_tools``,
    and nothing else is offered, found or run, even with approval. Of those, a tool that asks first runs without
    asking if ``automated`` names it (or an ``allow`` override does); one that always asks always does; the rest
    run without asking. Automating a tool doesn't enable it. A tool tagged ``"explicit"`` is enabled only by its
    name, never by a pattern such as ``"*"``: one for a few tasks alone."""
    allow = matching(overrides.get("allow") or [])
    enabled = {n for n in matching(config.enabled or []) if EXPLICIT not in tools[n].tags or n in (config.enabled or [])}
    usable = (enabled | allow) - matching([*config.denied_tools, *(overrides.get("deny") or [])])
    automated = matching(config.automated) | allow
    allowed: set[str] = set()
    gated: set[str] = set()
    always: set[str] = set()
    for name in usable:
        approval = tools[name].approval
        if approval == "always":
            gated.add(name)
            always.add(name)
        elif approval == "ask" and name not in automated:
            gated.add(name)
        else:
            allowed.add(name)
    hidden = (matching(config.hidden_tools) | {n for n, t in tools.items() if t.hidden} | matching(overrides.get("hide") or [])) & usable
    return ToolPermissions(
        allowed=frozenset(allowed),
        hidden=frozenset(hidden),
        denied=frozenset(set(tools) - usable),
        gated=frozenset(gated),
        always=frozenset(always),
        refuse_denied=True,
    )


ASK_TOOL_NAME = "ask_user"
MAX_CHOICES = 10


def ask_user_spec() -> ToolSpec:
    """Schema of the built-in tool for asking a human, offered to tasks with ``ask_user=True``."""
    return ToolSpec(
        name=ASK_TOOL_NAME,
        description=(
            "Ask the user a question, and wait for their answer: only for a decision that's theirs to make, or "
            "something only they know, not what you can find out or decide yourself. Give choices when the answer "
            "is one of a few; they can still answer in their own words. If they don't answer in time, you're told, "
            "and decide for yourself."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question, complete in itself: they may see it hours later."},
                "choices": {"type": "array", "items": {"type": "string"}, "description": f"Up to {MAX_CHOICES} answers to offer."},
            },
            "required": ["question"],
        },
    )


def search_tools_spec() -> ToolSpec:
    """Schema of the built-in registry lookup tool offered when hidden tools exist."""
    return ToolSpec(
        name=SEARCH_TOOL_NAME,
        description=(
            "Search the tool registry for additional tools that are available but not "
            "currently loaded. Returns matching tools and makes them callable."
        ),
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords describing the capability you need."}},
            "required": ["query"],
        },
    )


def format_tool_result(result: Any, max_chars: int = 0) -> str:
    """Serialise a tool result for the model (JSON when possible)."""
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(result)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} characters]"
    return text
