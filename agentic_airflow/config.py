"""Configuration cascade: System -> Dag -> TaskGroup -> Task.

Resolution rules (see spec §1):

* **Standard parameters** (``model``, ``thinking``, ``verbosity``,
  ``max_tool_steps``): the most specific non-``None`` value wins.
* **Skills**: additive (union, order preserved) unless a scope sets
  ``inherit_skills=False``, in which case the cascade restarts at that scope.
* **Tool permissions**: ``allowed_tools=None`` inherits the parent's list;
  ``denied_tools`` accumulate across scopes and always win over any
  ``allowed_tools`` entry; ``hidden_tools`` accumulate and mark allowed tools
  as on-demand (not injected into the prompt until discovered).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable, Literal, Sequence

Thinking = Literal["on", "off"]


@dataclass
class ScopeConfig:
    """Configuration parameters that can be set at any scope."""

    model: str | None = None
    thinking: Thinking | None = None
    verbosity: int | None = None
    skills: list[str] | None = None
    inherit_skills: bool = True
    allowed_tools: list[str] | None = None
    denied_tools: list[str] | None = None
    hidden_tools: list[str] | None = None
    max_tool_steps: int | None = None

    def __post_init__(self) -> None:
        if self.thinking not in (None, "on", "off"):
            raise ValueError("thinking must be 'on', 'off' or None")
        if self.verbosity is not None and self.verbosity not in (0, 1, 2):
            raise ValueError("verbosity must be 0, 1 or 2")
        if self.max_tool_steps is not None and self.max_tool_steps < 0:
            raise ValueError("max_tool_steps must be >= 0")
        for name in ("skills", "allowed_tools", "denied_tools", "hidden_tools"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, str):
                    raise TypeError(f"{name} must be a list of strings, not a string")
                setattr(self, name, [str(v) for v in value])

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ResolvedConfig:
    """The fully-resolved configuration for one task."""

    model: str | None
    thinking: bool
    verbosity: int
    skills: tuple[str, ...]
    allowed_tools: frozenset[str] | None
    """``None`` means "every registered tool" (nothing restricted it)."""
    denied_tools: frozenset[str]
    hidden_tools: frozenset[str]
    max_tool_steps: int

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "thinking": self.thinking,
            "verbosity": self.verbosity,
            "skills": list(self.skills),
            "allowed_tools": sorted(self.allowed_tools) if self.allowed_tools is not None else None,
            "denied_tools": sorted(self.denied_tools),
            "hidden_tools": sorted(self.hidden_tools),
            "max_tool_steps": self.max_tool_steps,
        }


DEFAULT_MAX_TOOL_STEPS = 10


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def resolve_config(scopes: Sequence[ScopeConfig]) -> ResolvedConfig:
    """Resolve a cascade of scopes ordered from broadest to most specific."""
    model: str | None = None
    thinking: Thinking | None = None
    verbosity: int | None = None
    max_tool_steps: int | None = None
    skills: list[str] = []
    allowed: list[str] | None = None
    denied: list[str] = []
    hidden: list[str] = []

    for scope in scopes:
        if scope.model is not None:
            model = scope.model
        if scope.thinking is not None:
            thinking = scope.thinking
        if scope.verbosity is not None:
            verbosity = scope.verbosity
        if scope.max_tool_steps is not None:
            max_tool_steps = scope.max_tool_steps
        if not scope.inherit_skills:
            skills = list(scope.skills or [])
        elif scope.skills:
            skills = _dedupe([*skills, *scope.skills])
        if scope.allowed_tools is not None:
            allowed = list(scope.allowed_tools)
        if scope.denied_tools:
            denied = _dedupe([*denied, *scope.denied_tools])
        if scope.hidden_tools:
            hidden = _dedupe([*hidden, *scope.hidden_tools])

    return ResolvedConfig(
        model=model,
        thinking=(thinking == "on"),
        verbosity=verbosity if verbosity is not None else 0,
        skills=tuple(skills),
        allowed_tools=frozenset(allowed) if allowed is not None else None,
        denied_tools=frozenset(denied),
        hidden_tools=frozenset(hidden),
        max_tool_steps=max_tool_steps if max_tool_steps is not None else DEFAULT_MAX_TOOL_STEPS,
    )


@dataclass
class EngineConfig(ScopeConfig):
    """System-level defaults plus engine tuning knobs.

    Inherits every :class:`ScopeConfig` parameter (these act as the *System*
    scope of the cascade) and adds:

    * ``model_concurrency``: per-model number of simultaneous inference steps
      (default ``1``, typical for local models). Keys are model ids.
    * ``default_concurrency``: used for models absent from ``model_concurrency``.
    * ``cpu_workers``: size of the thread pool for :class:`Task`/:class:`Branch`
      callables and synchronous tools.
    * ``max_context_chars``: upstream outputs longer than this are truncated
      (with a marker) when rendered into a prompt. ``0`` disables truncation.
    * ``output_retries``: how many times a malformed final answer is sent back
      to the model with corrective feedback before the task fails.
    * ``persist_every_step``: checkpoint the run after every inference step
      (safer) rather than only on task boundaries (faster).
    * ``atomic``: what holds a model slot. ``"task"`` (default): an agentic
      task keeps its slot from its first inference step to its answer, so
      live user requests wait for the running task and then go before the
      next one. ``"step"``: the slot is held for one inference step at a time,
      so users cut in between steps.
    * ``max_active_runs_per_model``: how many runs may be active (running or
      paused) at once among the DAGs whose resolved model is the same. Runs
      beyond it wait in a first-come, first-served queue. ``None``: no limit.
    * ``approval_timeout``: seconds a human decision may stay pending before
      it is resolved for them (a tool call is rejected, an exhausted evaluator
      fails). ``None``: they wait forever.
    * ``task_timeout``: seconds an agentic task may work, counted from when it
      gets its model slot (time paused for approval does not count). ``None``:
      no limit.
    * ``max_mapped``: the most mapped instances one fan-out list creates; longer
      lists are cut short.
    * ``keep_runs``: how many finished runs of each DAG are kept in the state
      store; older ones are deleted. ``None``: all of them.
    """

    model_concurrency: dict[str, int] = field(default_factory=dict)
    default_concurrency: int = 1
    cpu_workers: int = 4
    max_context_chars: int = 60_000
    output_retries: int = 1
    persist_every_step: bool = True
    atomic: Literal["task", "step"] = "task"
    max_active_runs_per_model: int | None = 1
    approval_timeout: float | None = None
    task_timeout: float | None = None
    max_mapped: int = 100
    keep_runs: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.atomic not in ("task", "step"):
            raise ValueError("atomic must be 'task' or 'step'")
        if self.max_active_runs_per_model is not None and self.max_active_runs_per_model < 1:
            raise ValueError("max_active_runs_per_model must be >= 1 or None")
        if self.max_mapped < 1:
            raise ValueError("max_mapped must be >= 1")
        if self.keep_runs is not None and self.keep_runs < 1:
            raise ValueError("keep_runs must be >= 1 or None")
