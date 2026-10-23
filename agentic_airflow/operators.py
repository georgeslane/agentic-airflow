"""Operators: the node types of a DAG.

* :class:`Task` - deterministic Python function (no LLM).
* :class:`Branch` - deterministic router.
* :class:`AgenticTask` - one LLM task (prompt + optional tool loop).
* :class:`AgenticBranch` - LLM router constrained to downstream task ids.
* :class:`AgenticEvaluator` - reflection loop that can re-run its upstream.

Dependencies use Airflow syntax: ``a >> b >> [c, d] >> e``. Data flow is
controlled by ``context=``; when omitted a task receives the outputs of its
direct upstream dependencies.
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, ClassVar, Iterable, Literal, Sequence

from . import _context
from .config import ScopeConfig, Thinking
from .errors import ConfigurationError, DagValidationError
from .models import OutputSpec
from .triggers import outlet_names

TASK_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
TriggerRule = Literal["none_failed", "all_done"]
ReturnType = Literal["string", "list", "json", "image", "file"]
OnExhausted = Literal["pause_for_user", "pass_latest", "fail_dag"]
NO_BRANCH = "none"  # what an AgenticBranch with allow_none=True answers to select nothing

EVALUATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rerun": {"type": "boolean", "description": "true if the output must be regenerated"},
        "reason": {
            "type": "string",
            "description": "Specific critique explaining why the output failed validation and how to fix it.",
        },
    },
    "required": ["rerun", "reason"],
    "additionalProperties": False,
}

OperatorRef = "BaseOperator | str"


class BaseOperator:
    """Common behaviour of every DAG node.

    Parameters shared by all operators:

    * ``id``: unique within the DAG (``[A-Za-z][A-Za-z0-9_.:-]*``).
    * ``context``: upstream operator(s) (or ids) whose outputs feed this node.
      ``None`` = direct upstream dependencies; ``[]`` = nothing.
    * ``fan_out``: when the node returns a ``list``, each element spawns one
      mapped instance of every immediately downstream node.
    * ``follow_mapping``: when an upstream node is mapped, run this node once
      per upstream instance (``this_i`` consumes ``upstream_i``) instead of
      joining all instances into one call. Evaluators default to ``True``.
    * ``trigger_rule``: ``"none_failed"`` (default - skip if every upstream was
      skipped, fail if any upstream failed) or ``"all_done"`` (run whatever
      happened upstream; used by the DagEvaluator).
    * ``on_complete``: optional callback ``fn(engine, run, instance)`` (sync or
      async) invoked after the node succeeds, before it counts as done.
    * ``outlets``: assets (:class:`~agentic_airflow.triggers.Asset` or their
      names) the node updates once all its instances have finished, if it
      produced something: DAGs scheduled on them run with what it produced.
    * ``skip_if``: optional ``fn(engine, run) -> str | None`` checked when the
      node is about to be scheduled: a returned reason skips it.
    * ``retries`` / ``retry_delay``: how many times the node is run again after
      it fails, and the seconds between (inherited from the group, DAG and
      engine when ``None``; default no retries, 60 seconds). A failure raised as
      :class:`~agentic_airflow.errors.PermanentError` isn't retried.
    """

    kind: ClassVar[str] = "task"
    is_agentic: ClassVar[bool] = False
    is_branch: ClassVar[bool] = False
    is_evaluator: ClassVar[bool] = False

    def __init__(
        self,
        id: str,
        *,
        context: "BaseOperator | str | Sequence[BaseOperator | str] | None" = None,
        fan_out: bool = False,
        follow_mapping: bool = False,
        trigger_rule: TriggerRule = "none_failed",
        on_complete: Callable[..., Any] | None = None,
        outlets: Sequence[Any] | None = None,
        skip_if: Callable[..., str | None] | None = None,
        dag: Any | None = None,
        group: Any | None = None,
        scope: ScopeConfig | None = None,
    ):
        if not isinstance(id, str) or not TASK_ID_RE.match(id):
            raise DagValidationError(
                f"Invalid task id {id!r}: must match {TASK_ID_RE.pattern}"
            )
        if trigger_rule not in ("none_failed", "all_done"):
            raise ConfigurationError(f"Unknown trigger_rule {trigger_rule!r}")
        self.id = id
        self.scope = scope or ScopeConfig()
        self.fan_out = bool(fan_out)
        self.follow_mapping = bool(follow_mapping)
        self.trigger_rule: TriggerRule = trigger_rule
        self.on_complete = on_complete
        self.outlets: list[str] = outlet_names(outlets)
        self.skip_if = skip_if
        self.upstream: list[BaseOperator] = []
        self.downstream: list[BaseOperator] = []
        self._context_refs: list[BaseOperator | str] | None
        if context is None:
            self._context_refs = None
        elif isinstance(context, (BaseOperator, str)):
            self._context_refs = [context]
        else:
            self._context_refs = list(context)
        self.context_ops: list[BaseOperator] | None = None  # resolved by Dag.validate()
        self.dag: Any | None = None
        self.group: Any | None = None
        owner = group or _context.current_group()
        if owner is None and dag is not None:
            owner = dag
        if owner is not None:
            owner.add(self)

    # -- dependency syntax -------------------------------------------------
    def set_downstream(self, other: "BaseOperator | Iterable[BaseOperator]") -> None:
        for node in _as_nodes(other):
            if node is self:
                raise DagValidationError(f"Task {self.id!r} cannot depend on itself")
            if self.dag is not None and node.dag is not None and self.dag is not node.dag:
                raise DagValidationError(
                    f"Cannot link {self.id!r} and {node.id!r}: they belong to different DAGs"
                )
            if node not in self.downstream:
                self.downstream.append(node)
            if self not in node.upstream:
                node.upstream.append(self)

    def set_upstream(self, other: "BaseOperator | Iterable[BaseOperator]") -> None:
        for node in _as_nodes(other, last=True):
            node.set_downstream(self)

    def __rshift__(self, other):
        self.set_downstream(other)
        return other

    def __rrshift__(self, other):
        self.set_upstream(other)
        return self

    def __lshift__(self, other):
        self.set_upstream(other)
        return other

    def __rlshift__(self, other):
        self.set_downstream(other)
        return self

    # -- helpers ------------------------------------------------------------
    @property
    def downstream_ids(self) -> list[str]:
        return [n.id for n in self.downstream]

    @property
    def upstream_ids(self) -> list[str]:
        return [n.id for n in self.upstream]

    def effective_context_ops(self) -> list["BaseOperator"]:
        """Operators whose outputs feed this node.

        Explicit ``context=`` wins. Otherwise the direct upstream dependencies
        are used, with **branch nodes made transparent**: a task downstream of
        ``Branch``/``AgenticBranch`` receives the data the branch routed (the
        branch's own context) rather than the selected task id. Pass
        ``context=[branch]`` explicitly to receive the selection itself.
        """
        if self.context_ops is not None:
            return list(self.context_ops)
        out: list[BaseOperator] = []
        for u in self.upstream:
            sources = u.effective_context_ops() if u.is_branch else [u]
            for s in sources:
                if s not in out:
                    out.append(s)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "agentic": self.is_agentic,
            "fan_out": self.fan_out,
            "follow_mapping": self.follow_mapping,
            "trigger_rule": self.trigger_rule,
            "outlets": list(self.outlets),
            "group": self.group.id if self.group is not None and getattr(self.group, "is_dag", False) is False else None,
            "upstream": self.upstream_ids,
            "downstream": self.downstream_ids,
            "context": [op.id for op in self.effective_context_ops()],
            "config": self.scope.to_dict(),
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id}>"


def _as_nodes(other: Any, last: bool = False) -> list[BaseOperator]:
    """The operators ``other`` stands for in a dependency: a task group stands for its first tasks, or with
    ``last`` its last."""
    if isinstance(other, BaseOperator):
        return [other]
    if hasattr(other, "roots") and hasattr(other, "leaves") and not getattr(other, "is_dag", False):  # TaskGroup
        return other._ends(last=last)
    if isinstance(other, (list, tuple, set)):
        nodes = []
        for item in other:
            nodes.extend(_as_nodes(item, last) if not isinstance(item, BaseOperator) else [item])
        return nodes
    raise DagValidationError(f"Cannot create a dependency on {other!r}")


class Task(BaseOperator):
    """Wrap a Python callable as a DAG node (no LLM compute).

    ``fn`` receives the outputs of its context operators as positional
    arguments (in declaration order) and, if its signature has a ``ctx``
    parameter, a :class:`~agentic_airflow.run.TaskContext`. Async functions
    are awaited; sync functions run on the CPU worker pool.
    """

    kind = "task"

    def __init__(
        self,
        id: str,
        fn: Callable[..., Any],
        *,
        context: Any = None,
        fan_out: bool = False,
        follow_mapping: bool = False,
        trigger_rule: TriggerRule = "none_failed",
        on_complete: Callable[..., Any] | None = None,
        outlets: Sequence[Any] | None = None,
        skip_if: Callable[..., str | None] | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        dag: Any | None = None,
        group: Any | None = None,
    ):
        if not callable(fn):
            raise ConfigurationError(f"Task {id!r}: fn must be callable")
        super().__init__(
            id,
            context=context,
            fan_out=fan_out,
            follow_mapping=follow_mapping,
            trigger_rule=trigger_rule,
            on_complete=on_complete,
            outlets=outlets,
            skip_if=skip_if,
            dag=dag,
            group=group,
            scope=ScopeConfig(retries=retries, retry_delay=retry_delay),
        )
        self.fn = fn
        self.wants_ctx = _accepts_ctx(fn)


class Branch(Task):
    """Deterministic router.

    ``fn`` returns a ``bool`` (requires exactly two downstream nodes declared
    as ``[true_task, false_task]``), a downstream task id, or a list of ids.
    Unselected downstream nodes and their exclusive descendants are SKIPPED.
    """

    kind = "branch"
    is_branch = True

    def __init__(self, id: str, fn: Callable[..., Any], *, context: Any = None, follow_mapping: bool = False, dag: Any | None = None, group: Any | None = None, on_complete: Callable[..., Any] | None = None, skip_if: Callable[..., str | None] | None = None, retries: int | None = None, retry_delay: float | None = None):
        super().__init__(id, fn, context=context, follow_mapping=follow_mapping, dag=dag, group=group, on_complete=on_complete, skip_if=skip_if, retries=retries, retry_delay=retry_delay)


def _accepts_ctx(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "ctx" in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


class AgenticTask(BaseOperator):
    """Core LLM execution node.

    See spec §2.3. Notable behaviours:

    * ``return_type`` is enforced by :func:`agentic_airflow.validation.coerce_output`;
      a malformed answer is sent back to the model once with corrective
      feedback before the task fails.
    * ``fan_out`` defaults to ``True`` when ``return_type="list"``.
    * ``json_schema`` optionally constrains ``return_type="json"``/``"list"``.
    * ``include_run_log=True`` renders the run's event log (``dag_context``)
      into the prompt and ``include_dag_definition=True`` renders the DAG
      topology/prompts (both used by the DagEvaluator).
    * ``system_prompt`` is appended to the DAG-level objective for this task only.
    * ``prompt`` may be a function ``fn(engine, run) -> str``, written when the
      task starts, for prompts that depend on the run (as an application's own
      end-of-run tasks might).
    * ``ask_user=True`` offers the task the ``ask_user`` tool: it asks a human a
      question, with optional choices, and the task waits (without holding its
      model) until :meth:`Engine.answer` answers it, or it expires.
    * ``validator`` checks the answer once it matches ``return_type``: it raises
      :class:`~agentic_airflow.errors.OutputValidationError` to have the model
      try again with its message (as with a malformed answer), and returns the
      value to use, or ``None`` to keep it.
    """

    kind = "agentic_task"
    is_agentic = True

    def __init__(
        self,
        id: str,
        prompt: str | Callable[..., str],
        *,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        hidden_tools: list[str] | None = None,
        skills: list[str] | None = None,
        inherit_skills: bool = True,
        thinking: Thinking | None = None,
        return_type: ReturnType | None = "string",
        json_schema: dict[str, Any] | None = None,
        fan_out: bool | None = None,
        follow_mapping: bool = False,
        context: Any = None,
        verbosity: int | None = None,
        max_tool_steps: int | None = None,
        include_run_log: bool = False,
        include_dag_definition: bool = False,
        system_prompt: str | None = None,
        validator: Callable[[Any], Any] | None = None,
        trigger_rule: TriggerRule = "none_failed",
        on_complete: Callable[..., Any] | None = None,
        outlets: Sequence[Any] | None = None,
        skip_if: Callable[..., str | None] | None = None,
        retries: int | None = None,
        retry_delay: float | None = None,
        ask_user: bool | None = None,
        dag: Any | None = None,
        group: Any | None = None,
    ):
        if not callable(prompt) and (not isinstance(prompt, str) or not prompt.strip()):
            raise ConfigurationError(f"AgenticTask {id!r}: prompt is required")
        return_type = return_type or "string"
        if return_type not in ("string", "list", "json", "image", "file"):
            raise ConfigurationError(f"AgenticTask {id!r}: unsupported return_type {return_type!r}")
        if fan_out is None:
            fan_out = return_type == "list"
        scope = ScopeConfig(
            model=model,
            thinking=thinking,
            verbosity=verbosity,
            skills=skills,
            inherit_skills=inherit_skills,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            hidden_tools=hidden_tools,
            max_tool_steps=max_tool_steps,
            retries=retries,
            retry_delay=retry_delay,
            ask_user=ask_user,
        )
        super().__init__(
            id,
            context=context,
            fan_out=fan_out,
            follow_mapping=follow_mapping,
            trigger_rule=trigger_rule,
            on_complete=on_complete,
            outlets=outlets,
            skip_if=skip_if,
            dag=dag,
            group=group,
            scope=scope,
        )
        self.prompt = prompt
        self.return_type: ReturnType = return_type
        self.json_schema = json_schema
        self.include_run_log = include_run_log
        self.include_dag_definition = include_dag_definition
        self.system_prompt = system_prompt
        self.validator = validator

    @property
    def prompt_text(self) -> str:
        """The prompt, or what stands for one written when the task starts."""
        return self.prompt if isinstance(self.prompt, str) else "(written when the task starts)"

    def render_prompt(self, engine: Any, run: Any) -> str:
        return self.prompt(engine, run) if callable(self.prompt) else self.prompt

    def output_spec(self, **_: Any) -> OutputSpec:
        return OutputSpec(kind=self.return_type, json_schema=self.json_schema)

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({"prompt": self.prompt_text, "return_type": self.return_type, "include_run_log": self.include_run_log})
        return data


class AgenticBranch(AgenticTask):
    """Semantic router: the model picks downstream task id(s).

    The downstream ids are passed to the backend as a constrained ``enum``.
    ``multi_select=True`` lets the model activate several branches.
    ``allow_none=True`` adds the choice ``"none"``, which selects nothing, so
    every downstream task is skipped: a branch with a single downstream task is
    then a yes or no.
    """

    kind = "agentic_branch"
    is_branch = True

    def __init__(self, id: str, prompt: str, *, multi_select: bool = False, allow_none: bool = False, **kwargs: Any):
        kwargs.pop("return_type", None)
        kwargs.pop("fan_out", None)
        super().__init__(id, prompt, return_type="string", fan_out=False, **kwargs)
        self.multi_select = bool(multi_select)
        self.allow_none = bool(allow_none)

    def choices(self) -> list[str]:
        return [*self.downstream_ids, *([NO_BRANCH] if self.allow_none else [])]

    def output_spec(self, choices: Sequence[str] | None = None, **_: Any) -> OutputSpec:
        return OutputSpec(kind="enum", choices=list(choices or self.choices()), multi=self.multi_select)

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["multi_select"] = self.multi_select
        data["allow_none"] = self.allow_none
        return data


class AgenticEvaluator(AgenticTask):
    """Reflection node: critiques the target's output and may force a re-run.

    Returns ``{"rerun": bool, "reason": str}``. On ``rerun`` the target runs
    again with its original context plus ``retry_history``; otherwise the
    evaluator passes the **target's** output downstream (so the next task sees
    the validated artefact, not the critique).

    ``target`` defaults to the single upstream node. ``max_retries`` bounds the
    loop; ``on_exhausted`` picks what happens afterwards.
    """

    kind = "agentic_evaluator"
    is_evaluator = True

    def __init__(
        self,
        id: str,
        prompt: str,
        *,
        target: "BaseOperator | str | None" = None,
        max_retries: int = 3,
        on_exhausted: OnExhausted = "pause_for_user",
        follow_mapping: bool = True,
        **kwargs: Any,
    ):
        if max_retries < 0:
            raise ConfigurationError(f"AgenticEvaluator {id!r}: max_retries must be >= 0")
        if on_exhausted not in ("pause_for_user", "pass_latest", "fail_dag"):
            raise ConfigurationError(f"AgenticEvaluator {id!r}: unknown on_exhausted {on_exhausted!r}")
        kwargs.pop("return_type", None)
        kwargs.pop("fan_out", None)
        kwargs.pop("json_schema", None)
        super().__init__(id, prompt, return_type="json", json_schema=EVALUATOR_SCHEMA, fan_out=False, follow_mapping=follow_mapping, **kwargs)
        self._target_ref = target
        self.target: BaseOperator | None = None  # resolved by Dag.validate()
        self.max_retries = int(max_retries)
        self.on_exhausted: OnExhausted = on_exhausted

    def output_spec(self, **_: Any) -> OutputSpec:
        return OutputSpec(kind="json", json_schema=EVALUATOR_SCHEMA)

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({
            "target": self.target.id if self.target else None,
            "max_retries": self.max_retries,
            "on_exhausted": self.on_exhausted,
        })
        return data
