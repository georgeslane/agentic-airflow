"""Run-time records: :class:`DagRun`, :class:`TaskInstance` and the event log."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    UPSTREAM_FAILED = "upstream_failed"
    AWAITING_TOOL_APPROVAL = "awaiting_tool_approval"
    AWAITING_USER = "awaiting_user"


TERMINAL_STATUSES = frozenset({TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.UPSTREAM_FAILED})
WAITING_STATUSES = frozenset({TaskStatus.AWAITING_TOOL_APPROVAL, TaskStatus.AWAITING_USER})
ACTIVE_STATUSES = frozenset({TaskStatus.QUEUED, TaskStatus.RUNNING})


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


RUN_TERMINAL = frozenset({RunStatus.SUCCESS, RunStatus.FAILED, RunStatus.CANCELLED})


def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"


@dataclass
class TaskInstance:
    """One execution unit of a node (mapped nodes have several).

    ``bindings`` describe how the input context is assembled at launch time:
    ``{upstream_task_id: ("output", inst_id) | ("item", inst_id, index) |
    ("fanin", [inst_ids])}``.
    """

    instance_id: str
    task_id: str
    map_path: tuple[int, ...] = ()
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 1
    depends_on: list[str] = field(default_factory=list)
    bindings: dict[str, Any] = field(default_factory=dict)
    input_context: dict[str, Any] | None = None
    output: Any = None
    error: str | None = None
    reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retry_history: list[dict[str, Any]] = field(default_factory=list)
    selected: list[str] | None = None
    verdict: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    approval_id: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    steps: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    extra_retries: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def is_waiting(self) -> bool:
        return self.status in WAITING_STATUSES

    def reset_for_retry(self) -> None:
        self.status = TaskStatus.PENDING
        self.output = None
        self.error = None
        self.reason = None
        self.checkpoint = None
        self.approval_id = None
        self.tool_calls = []
        self.selected = None
        self.verdict = None
        self.started_at = None
        self.finished_at = None
        self.steps = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "map_path": list(self.map_path),
            "status": self.status.value,
            "attempt": self.attempt,
            "depends_on": list(self.depends_on),
            "bindings": self.bindings,
            "input_context": self.input_context,
            "output": self.output,
            "error": self.error,
            "reason": self.reason,
            "tool_calls": self.tool_calls,
            "retry_history": self.retry_history,
            "selected": self.selected,
            "verdict": self.verdict,
            "checkpoint": self.checkpoint,
            "approval_id": self.approval_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": self.steps,
            "usage": self.usage,
            "extra_retries": self.extra_retries,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskInstance":
        inst = cls(
            instance_id=d["instance_id"],
            task_id=d["task_id"],
            map_path=tuple(d.get("map_path") or ()),
            status=TaskStatus(d.get("status", "pending")),
            attempt=int(d.get("attempt", 1)),
            depends_on=list(d.get("depends_on") or []),
            bindings=_restore_bindings(d.get("bindings") or {}),
            input_context=d.get("input_context"),
            output=d.get("output"),
            error=d.get("error"),
            reason=d.get("reason"),
            tool_calls=list(d.get("tool_calls") or []),
            retry_history=list(d.get("retry_history") or []),
            selected=d.get("selected"),
            verdict=d.get("verdict"),
            checkpoint=d.get("checkpoint"),
            approval_id=d.get("approval_id"),
            created_at=float(d.get("created_at") or time.time()),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            steps=int(d.get("steps") or 0),
            usage=dict(d.get("usage") or {}),
            extra_retries=int(d.get("extra_retries") or 0),
        )
        return inst


def _restore_bindings(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        out[key] = tuple(value) if isinstance(value, list) and value and isinstance(value[0], str) and value[0] in ("output", "item", "fanin") else value
    return out


@dataclass
class RunLogEntry:
    """One entry of the append-only ``dag_context`` log."""

    timestamp: float
    task_id: str
    instance_id: str
    status: str
    input_context: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    output: Any = None
    error: str | None = None
    attempt: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunLogEntry":
        """An entry as stored, by any version: fields it doesn't know are left out, missing ones take defaults."""
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known.setdefault("timestamp", 0.0)
        for name in ("task_id", "instance_id", "status"):
            known.setdefault(name, "")
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "status": self.status,
            "input_context": self.input_context,
            "tool_calls": self.tool_calls,
            "output": self.output,
            "error": self.error,
            "attempt": self.attempt,
        }


@dataclass
class DagRun:
    """State of one execution of a DAG."""

    run_id: str
    dag_id: str
    status: RunStatus = RunStatus.PENDING
    trigger: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    seed: Any = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    instances: dict[str, TaskInstance] = field(default_factory=dict)
    node_instances: dict[str, list[str]] = field(default_factory=dict)
    expanded: list[str] = field(default_factory=list)
    done_nodes: list[str] = field(default_factory=list)
    """Nodes whose instances have all finished (``node.finished`` emitted, outlets updated)."""
    log: list[RunLogEntry] = field(default_factory=list)
    chat_log: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    approved_tools: list[str] = field(default_factory=list)
    """Tools approved with ``remember=True`` for the rest of this run."""
    summary: str | None = None
    """Executive summary produced by the DagEvaluator (if attached)."""

    @property
    def is_terminal(self) -> bool:
        return self.status in RUN_TERMINAL

    @property
    def dag_context(self) -> list[dict[str, Any]]:
        """The structured, append-only run log (spec §4)."""
        return [e.to_dict() for e in self.log]

    def instances_for(self, task_id: str) -> list[TaskInstance]:
        return [self.instances[i] for i in self.node_instances.get(task_id, [])]

    def outputs(self) -> dict[str, Any]:
        """Outputs of successful instances keyed by instance id."""
        return {i.instance_id: i.output for i in self.instances.values() if i.status == TaskStatus.SUCCESS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "status": self.status.value,
            "trigger": self.trigger,
            "memory": self.memory,
            "seed": self.seed,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "instances": {k: v.to_dict() for k, v in self.instances.items()},
            "node_instances": self.node_instances,
            "expanded": list(self.expanded),
            "done_nodes": list(self.done_nodes),
            "log": [e.to_dict() for e in self.log],
            "chat_log": self.chat_log,
            "error": self.error,
            "approved_tools": list(self.approved_tools),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DagRun":
        run = cls(
            run_id=d["run_id"],
            dag_id=d["dag_id"],
            status=RunStatus(d.get("status", "pending")),
            trigger=dict(d.get("trigger") or {}),
            memory=dict(d.get("memory") or {}),
            seed=d.get("seed"),
            created_at=float(d.get("created_at") or time.time()),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            instances={k: TaskInstance.from_dict(v) for k, v in (d.get("instances") or {}).items()},
            node_instances={k: list(v) for k, v in (d.get("node_instances") or {}).items()},
            expanded=list(d.get("expanded") or []),
            done_nodes=list(d.get("done_nodes") or []),
            log=[RunLogEntry.from_dict(e) for e in d.get("log") or []],
            chat_log=list(d.get("chat_log") or []),
            error=d.get("error"),
            approved_tools=list(d.get("approved_tools") or []),
            summary=d.get("summary"),
        )
        return run


@dataclass
class TaskContext:
    """Injected into ``Task``/``Branch`` callables that declare a ``ctx`` parameter."""

    run_id: str
    dag_id: str
    task_id: str
    instance_id: str
    map_path: tuple[int, ...]
    attempt: int
    memory: dict[str, Any]
    seed: Any
    trigger: dict[str, Any]
    upstream: dict[str, Any]
    retry_history: list[dict[str, Any]]
    run_log: list[dict[str, Any]]
    engine: Any = None


@dataclass
class Approval:
    """A pending human decision (tool approval or exhausted evaluator)."""

    approval_id: str
    run_id: str
    dag_id: str
    instance_id: str
    task_id: str
    kind: str  # "tool_call" | "evaluator_exhausted"
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending | approved | rejected | resolved
    decision: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "instance_id": self.instance_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "status": self.status,
            "decision": self.decision,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Approval":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})  # type: ignore[arg-type]


@dataclass
class Event:
    """Observability event emitted by the engine."""

    type: str
    run_id: str | None = None
    dag_id: str | None = None
    task_id: str | None = None
    instance_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "run_id": self.run_id,
            "dag_id": self.dag_id,
            "task_id": self.task_id,
            "instance_id": self.instance_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
