"""``Dag`` and ``TaskGroup`` containers."""

from __future__ import annotations

import re
from typing import Any, Iterable

from . import _context
from .config import ScopeConfig, Thinking
from .errors import DagValidationError
from .operators import NO_BRANCH, AgenticBranch, AgenticEvaluator, BaseOperator, Branch
from .triggers import Trigger, normalise_schedule

DAG_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_MAP_SUFFIX_RE = re.compile(r"^(?P<base>.+?)(?:_\d+)+$")


class TaskGroup:
    """Scoped organisational container applying shared configuration.

    Accepts ``id``, ``model``, ``allowed_tools``, ``denied_tools``,
    ``hidden_tools``, ``skills``, ``inherit_skills``, ``thinking`` and
    ``max_tool_steps``. Use as a context manager::

        with TaskGroup("research", skills=["Research"]):
            a = AgenticTask("search", prompt="...")

    Groups can be nested and reused (e.g. the built-in DagEvaluator). As in
    Airflow, a group can be used with ``>>`` and ``<<``: before something it
    stands for its last tasks (those with nothing after them in the group),
    after something for its first.
    """

    is_dag = False

    def __init__(
        self,
        id: str,
        *,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        hidden_tools: list[str] | None = None,
        skills: list[str] | None = None,
        inherit_skills: bool = True,
        thinking: Thinking | None = None,
        max_tool_steps: int | None = None,
        dag: "Dag | None" = None,
        parent: "TaskGroup | None" = None,
    ):
        if not DAG_ID_RE.match(id):
            raise DagValidationError(f"Invalid group id {id!r}")
        self.id = id
        self.scope = ScopeConfig(
            model=model,
            thinking=thinking,
            skills=skills,
            inherit_skills=inherit_skills,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            hidden_tools=hidden_tools,
            max_tool_steps=max_tool_steps,
        )
        self.tasks: list[BaseOperator] = []
        self.groups: list[TaskGroup] = []
        self._token = None
        owner = parent or _context.current_group()
        self.parent: TaskGroup | None = None
        self.dag: Dag | None = dag
        if owner is not None:
            if getattr(owner, "is_dag", False):
                self.dag = self.dag or owner  # type: ignore[assignment]
            else:
                self.parent = owner
                self.dag = self.dag or owner.dag
        if self.parent is not None:
            self.parent.groups.append(self)
        if self.dag is not None:
            self.dag._register_group(self)

    def __enter__(self) -> "TaskGroup":
        self._token = _context.push(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _context.pop(self._token)
            self._token = None

    def add(self, op: BaseOperator) -> None:
        op.group = self
        self.tasks.append(op)
        if self.dag is not None:
            self.dag.add_task(op)

    def attach(self, dag: "Dag") -> None:
        """Attach a group built outside a ``with Dag`` block to ``dag``."""
        if self.dag is not None and self.dag is not dag:
            raise DagValidationError(f"TaskGroup {self.id!r} already belongs to DAG {self.dag.id!r}")
        self.dag = dag
        dag._register_group(self)
        for op in self.tasks:
            dag.add_task(op)
        for g in self.groups:
            g.attach(dag)

    def all_tasks(self) -> list[BaseOperator]:
        out = list(self.tasks)
        for g in self.groups:
            out.extend(g.all_tasks())
        return out

    # -- dependency syntax: the group's first and last tasks ------------------
    def roots(self) -> list[BaseOperator]:
        """Its tasks with nothing before them in the group."""
        inside = self.all_tasks()
        return [t for t in inside if not any(u in inside for u in t.upstream)]

    def leaves(self) -> list[BaseOperator]:
        """Its tasks with nothing after them in the group."""
        inside = self.all_tasks()
        return [t for t in inside if not any(d in inside for d in t.downstream)]

    def _ends(self, last: bool) -> list[BaseOperator]:
        ends = self.leaves() if last else self.roots()
        if not ends:
            raise DagValidationError(f"TaskGroup {self.id!r} has no tasks to depend on")
        return ends

    def __rshift__(self, other: Any) -> Any:
        for t in self._ends(last=True):
            t.set_downstream(_first_tasks(other))
        return other

    def __rrshift__(self, other: Any) -> "TaskGroup":
        for t in self._ends(last=False):
            t.set_upstream(_last_tasks(other))
        return self

    def __lshift__(self, other: Any) -> Any:
        for t in self._ends(last=False):
            t.set_upstream(_last_tasks(other))
        return other

    def __rlshift__(self, other: Any) -> "TaskGroup":
        for t in self._ends(last=True):
            t.set_downstream(_first_tasks(other))
        return self

    def scope_chain(self) -> list[ScopeConfig]:
        chain: list[ScopeConfig] = []
        node: TaskGroup | None = self
        while node is not None:
            chain.append(node.scope)
            node = node.parent
        chain.reverse()
        return chain

    def path(self) -> list[str]:
        node: TaskGroup | None = self
        ids: list[str] = []
        while node is not None:
            ids.append(node.id)
            node = node.parent
        return list(reversed(ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent": self.parent.id if self.parent else None,
            "tasks": [t.id for t in self.tasks],
            "groups": [g.to_dict() for g in self.groups],
            "config": self.scope.to_dict(),
        }


def _first_tasks(other: Any) -> list[BaseOperator]:
    items = other if isinstance(other, (list, tuple, set)) else [other]
    return [t for item in items for t in (item.roots() if isinstance(item, TaskGroup) else [item])]


def _last_tasks(other: Any) -> list[BaseOperator]:
    items = other if isinstance(other, (list, tuple, set)) else [other]
    return [t for item in items for t in (item.leaves() if isinstance(item, TaskGroup) else [item])]


class Dag:
    """Top-level workflow container (spec §2.7).

    Parameters:

    * ``id`` / ``prompt``: identity and the global objective injected into
      every agentic task's system prompt.
    * ``context``: static seed context (``dict`` or ``str``) merged with the
      persisted memory from previous runs.
    * ``schedule``: cron strings, :class:`CronTrigger`, :class:`DagTrigger`,
      :class:`WebhookTrigger` entries.
    * ``auto_evaluate``: attach the :class:`~agentic_airflow.evaluator.DagEvaluator`
      group to the terminal nodes when the DAG is registered with an engine.
    * all :class:`TaskGroup` parameters (plus ``verbosity``) as DAG defaults.
    """

    is_dag = True

    def __init__(
        self,
        id: str,
        prompt: str,
        *,
        context: dict[str, Any] | str | None = None,
        schedule: Iterable[str | Trigger] | str | Trigger | None = None,
        auto_evaluate: bool = True,
        description: str = "",
        tags: Iterable[str] = (),
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
        hidden_tools: list[str] | None = None,
        skills: list[str] | None = None,
        inherit_skills: bool = True,
        thinking: Thinking | None = None,
        verbosity: int | None = None,
        max_tool_steps: int | None = None,
        max_active_runs: int = 1,
    ):
        if not DAG_ID_RE.match(id):
            raise DagValidationError(f"Invalid dag id {id!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DagValidationError(f"Dag {id!r}: prompt is required")
        if max_active_runs < 1:
            raise DagValidationError("max_active_runs must be >= 1")
        self.id = id
        self.prompt = prompt
        self.context = context
        self.schedule: list[Trigger] = normalise_schedule(schedule)
        self.auto_evaluate = auto_evaluate
        self.description = description
        self.tags = tuple(tags)
        self.max_active_runs = max_active_runs
        self.scope = ScopeConfig(
            model=model,
            thinking=thinking,
            verbosity=verbosity,
            skills=skills,
            inherit_skills=inherit_skills,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            hidden_tools=hidden_tools,
            max_tool_steps=max_tool_steps,
        )
        self.tasks: dict[str, BaseOperator] = {}
        self.groups: list[TaskGroup] = []
        self.evaluator_group: TaskGroup | None = None
        self._token = None
        self._validated = False

    # -- construction -------------------------------------------------------
    def __enter__(self) -> "Dag":
        self._token = _context.push(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _context.pop(self._token)
            self._token = None

    def add(self, op: BaseOperator) -> None:
        """Register a task directly under the DAG (no group)."""
        self.add_task(op)

    def add_task(self, op: BaseOperator) -> None:
        if op.dag is not None and op.dag is not self:
            raise DagValidationError(f"Task {op.id!r} already belongs to DAG {op.dag.id!r}")
        existing = self.tasks.get(op.id)
        if existing is not None and existing is not op:
            raise DagValidationError(f"Duplicate task id {op.id!r} in DAG {self.id!r}")
        op.dag = self
        self.tasks[op.id] = op
        self._validated = False

    def _register_group(self, group: TaskGroup) -> None:
        if group not in self.groups:
            self.groups.append(group)
        self._validated = False

    def get(self, task_id: str) -> BaseOperator:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise DagValidationError(f"Unknown task {task_id!r} in DAG {self.id!r}") from None

    def __contains__(self, task_id: object) -> bool:
        return task_id in self.tasks

    def __iter__(self):
        return iter(self.tasks.values())

    # -- graph queries ------------------------------------------------------
    def roots(self) -> list[BaseOperator]:
        return [t for t in self.tasks.values() if not t.upstream]

    def leaves(self) -> list[BaseOperator]:
        return [t for t in self.tasks.values() if not t.downstream]

    def own_tasks(self) -> list[BaseOperator]:
        """The tasks of the DAG itself: all but those of its evaluator group."""
        group = self.evaluator_group
        return [t for t in self.tasks.values() if group is None or t.group is not group]

    def outlets(self) -> list[str]:
        """The assets the DAG's tasks update."""
        return list(dict.fromkeys(name for t in self.tasks.values() for name in t.outlets))

    def topological_order(self) -> list[BaseOperator]:
        indeg = {t.id: len(t.upstream) for t in self.tasks.values()}
        ready = [t for t in self.tasks.values() if indeg[t.id] == 0]
        order: list[BaseOperator] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for d in node.downstream:
                indeg[d.id] -= 1
                if indeg[d.id] == 0:
                    ready.append(d)
        if len(order) != len(self.tasks):
            cyclic = sorted(t for t, n in indeg.items() if n > 0)
            raise DagValidationError(f"DAG {self.id!r} contains a cycle involving {cyclic}")
        return order

    def ancestors(self, op: BaseOperator) -> set[str]:
        seen: set[str] = set()
        stack = list(op.upstream)
        while stack:
            node = stack.pop()
            if node.id in seen:
                continue
            seen.add(node.id)
            stack.extend(node.upstream)
        return seen

    def scopes_for(self, op: BaseOperator) -> list[ScopeConfig]:
        """Scope cascade for ``op`` excluding the system scope: dag, groups..., task."""
        chain: list[ScopeConfig] = [self.scope]
        if op.group is not None and not getattr(op.group, "is_dag", False):
            chain.extend(op.group.scope_chain())
        chain.append(op.scope)
        return chain

    # -- validation ---------------------------------------------------------
    def validate(self) -> "Dag":
        """Validate structure; raises :class:`DagValidationError`."""
        order = self.topological_order()
        ids = set(self.tasks)
        for op in order:
            for n in (*op.upstream, *op.downstream):
                if n.dag is not self:
                    raise DagValidationError(f"Task {n.id!r} linked to {op.id!r} is not part of DAG {self.id!r}")
            m = _MAP_SUFFIX_RE.match(op.id)
            if m and m.group("base") in ids:
                raise DagValidationError(
                    f"Task id {op.id!r} collides with mapped instance ids of {m.group('base')!r}"
                )
            # context references
            if op._context_refs is not None:
                resolved: list[BaseOperator] = []
                ancestors = self.ancestors(op)
                for ref in op._context_refs:
                    node = self.get(ref) if isinstance(ref, str) else ref
                    if node.id not in ancestors:
                        raise DagValidationError(
                            f"Task {op.id!r}: context task {node.id!r} is not an upstream ancestor"
                        )
                    resolved.append(node)
                op.context_ops = resolved
            fan_sources = [u for u in op.upstream if u.fan_out]
            if len(fan_sources) > 1:
                raise DagValidationError(
                    f"Task {op.id!r} has more than one fan_out upstream ({[u.id for u in fan_sources]}); "
                    "only one upstream may drive dynamic mapping"
                )
            if isinstance(op, AgenticEvaluator):
                ref = op._target_ref
                if ref is None:
                    if len(op.upstream) != 1:
                        raise DagValidationError(
                            f"AgenticEvaluator {op.id!r} must have exactly one upstream or an explicit target="
                        )
                    op.target = op.upstream[0]
                else:
                    op.target = self.get(ref) if isinstance(ref, str) else ref
                    if op.target not in op.upstream:
                        raise DagValidationError(
                            f"AgenticEvaluator {op.id!r}: target {op.target.id!r} must be a direct upstream"
                        )
                if op.target.is_evaluator:
                    raise DagValidationError(f"AgenticEvaluator {op.id!r} cannot evaluate another evaluator")
                others = [d.id for d in op.target.downstream if not d.is_evaluator]
                if others:
                    # They would run on the first output, before the evaluator could have it redone.
                    raise DagValidationError(
                        f"Task {op.target.id!r} is evaluated by {op.id!r}, so the tasks that use its output "
                        f"({others}) must come after {op.id!r}, not after {op.target.id!r}"
                    )
            if isinstance(op, (Branch, AgenticBranch)):
                if not op.downstream:
                    raise DagValidationError(f"Branch {op.id!r} has no downstream tasks to select from")
                if op.outlets:
                    raise DagValidationError(
                        f"Branch {op.id!r} can't have outlets: give them to the tasks it selects"
                    )
                if isinstance(op, AgenticBranch) and op.allow_none and NO_BRANCH in op.downstream_ids:
                    raise DagValidationError(
                        f"Branch {op.id!r} allows none, so no task after it can be called {NO_BRANCH!r}"
                    )
        self._validated = True
        return self

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "description": self.description,
            "tags": list(self.tags),
            "schedule": [t.to_dict() for t in self.schedule],
            "auto_evaluate": self.auto_evaluate,
            "config": self.scope.to_dict(),
            "tasks": [t.describe() for t in self.topological_order()],
            "groups": [g.to_dict() for g in self.groups if g.parent is None],
        }

    def __repr__(self) -> str:
        return f"<Dag {self.id} tasks={len(self.tasks)}>"
