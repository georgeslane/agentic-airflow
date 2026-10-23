"""Ready-made functions for DAGs.

* :func:`trigger_items` - for ``Task("items", fn=trigger_items, fan_out=True)``:
  map the tasks after it over what came with whatever started the run (a
  webhook, an asset update, another DAG...).
* :func:`found_nothing` - for ``skip_if=``: skip a task (such as the
  DagEvaluator's) when the run found nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .run import TaskStatus

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine
    from .run import DagRun, TaskContext


def trigger_items(ctx: "TaskContext") -> list[Any]:
    """The items that came with what started the run: its payload, if that's a list; the payload's ``"items"``,
    if it's a dict with a list there; nothing, if there was no payload; otherwise the payload as one item."""
    payload = (ctx.trigger or {}).get("payload")
    if payload is None or (isinstance(payload, str) and not payload.strip()):
        return []
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return list(payload["items"])
    return [payload]


def found_nothing(engine: "Engine", run: "DagRun") -> str | None:
    """A ``skip_if`` check: a reason to skip, if none of the DAG's own tasks found anything - every one that ran
    returned nothing, an empty list, or a branch selected nothing - and none failed. ``None`` otherwise."""
    dag = engine.dag_for(run)
    for task in dag.own_tasks():
        for inst in run.instances_for(task.id):
            if inst.status in (TaskStatus.FAILED, TaskStatus.UPSTREAM_FAILED):
                return None
            if inst.status != TaskStatus.SUCCESS:
                continue
            if task.is_branch:
                if inst.selected:
                    return None
            elif not is_empty(inst.output):
                return None
    return "the run found nothing"


def is_empty(value: Any) -> bool:
    """Nothing: None, blank text, or an empty list or dict."""
    return value is None or (isinstance(value, str) and not value.strip()) or (isinstance(value, (list, dict)) and not value)
