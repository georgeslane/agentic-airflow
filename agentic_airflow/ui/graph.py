"""Graph JSON + a dependency-free layered layout for rendering DAGs."""

from __future__ import annotations

import json
from typing import Any

from ..dag import Dag
from ..run import DagRun, TaskStatus

STATUS_ORDER = [
    TaskStatus.FAILED, TaskStatus.UPSTREAM_FAILED, TaskStatus.AWAITING_TOOL_APPROVAL, TaskStatus.AWAITING_USER,
    TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.PENDING, TaskStatus.SUCCESS, TaskStatus.SKIPPED,
]


def _preview(value: Any, limit: int = 400) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def aggregate_status(statuses: list[TaskStatus]) -> str:
    if not statuses:
        return "pending"
    if all(s == TaskStatus.SUCCESS for s in statuses):
        return "success"
    if all(s == TaskStatus.SKIPPED for s in statuses):
        return "skipped"
    if all(s in (TaskStatus.SUCCESS, TaskStatus.SKIPPED) for s in statuses):
        return "success"
    for s in STATUS_ORDER:
        if s in statuses:
            return s.value
    return statuses[0].value


def layout_graph(nodes: list[str], edges: list[tuple[str, str]], *, dx: float = 220.0, dy: float = 90.0) -> dict[str, dict[str, float]]:
    """Longest-path layering + barycentric ordering. Returns ``{id: {x, y, layer}}``."""
    preds: dict[str, list[str]] = {n: [] for n in nodes}
    succs: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in preds and b in preds:
            succs[a].append(b)
            preds[b].append(a)
    layer: dict[str, int] = {}
    remaining = set(nodes)
    while remaining:
        progressed = False
        for n in list(remaining):
            if all(p in layer for p in preds[n]):
                layer[n] = max((layer[p] + 1 for p in preds[n]), default=0)
                remaining.discard(n)
                progressed = True
        if not progressed:  # cycle guard
            for n in remaining:
                layer[n] = 0
            break
    layers: dict[int, list[str]] = {}
    for n in nodes:
        layers.setdefault(layer[n], []).append(n)
    order = {n: i for i, n in enumerate(nodes)}
    for _ in range(4):
        for li in sorted(layers):
            def key(n: str) -> float:
                ps = [order[p] for p in preds[n] if p in order]
                return (sum(ps) / len(ps)) if ps else order[n]
            layers[li].sort(key=key)
            for i, n in enumerate(layers[li]):
                order[n] = i
        for li in sorted(layers, reverse=True):
            def key2(n: str) -> float:
                ss = [order[s] for s in succs[n] if s in order]
                return (sum(ss) / len(ss)) if ss else order[n]
            layers[li].sort(key=key2)
            for i, n in enumerate(layers[li]):
                order[n] = i
    max_height = max((len(v) for v in layers.values()), default=1)
    positions: dict[str, dict[str, float]] = {}
    for li, items in layers.items():
        offset = (max_height - len(items)) * dy / 2
        for i, n in enumerate(items):
            positions[n] = {"x": li * dx, "y": offset + i * dy, "layer": li}
    return positions


def build_graph(dag: Dag, run: DagRun | None = None) -> dict[str, Any]:
    """Serialise a DAG (and optionally a run's live state) for the UI."""
    order = dag.topological_order()
    edges = [(u.id, t.id) for t in order for u in t.upstream]
    positions = layout_graph([t.id for t in order], edges)
    nodes = []
    for t in order:
        insts = run.instances_for(t.id) if run else []
        node = {
            "id": t.id,
            "kind": t.kind,
            "agentic": t.is_agentic,
            "group": t.group.id if t.group is not None and not getattr(t.group, "is_dag", False) else None,
            "evaluator_group": dag.evaluator_group is not None and t.group is dag.evaluator_group,
            "fan_out": t.fan_out,
            "follow_mapping": t.follow_mapping,
            "trigger_rule": t.trigger_rule,
            "prompt": getattr(t, "prompt", None),
            "return_type": getattr(t, "return_type", None),
            "config": t.scope.to_dict(),
            "status": aggregate_status([i.status for i in insts]) if run else None,
            "instances": [
                {
                    "id": i.instance_id,
                    "status": i.status.value,
                    "map_path": list(i.map_path),
                    "attempt": i.attempt,
                    "started_at": i.started_at,
                    "finished_at": i.finished_at,
                    "steps": i.steps,
                    "tool_calls": len(i.tool_calls),
                    "output": _preview(i.output),
                    "error": i.error,
                    "reason": i.reason,
                    "selected": i.selected,
                    "approval_id": i.approval_id,
                }
                for i in insts
            ],
            "position": positions[t.id],
        }
        nodes.append(node)
    graph: dict[str, Any] = {
        "dag": {
            "id": dag.id,
            "prompt": dag.prompt,
            "description": dag.description,
            "schedule": [s.to_dict() for s in dag.schedule],
            "tags": list(dag.tags),
        },
        "nodes": nodes,
        "edges": [{"source": a, "target": b} for a, b in edges],
        "groups": [g.to_dict() for g in dag.groups if g.parent is None],
        "layout": {"width": max((p["x"] for p in positions.values()), default=0) + 220, "height": max((p["y"] for p in positions.values()), default=0) + 90},
    }
    if run is not None:
        graph["run"] = {
            "run_id": run.run_id,
            "status": run.status.value,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "trigger": run.trigger,
            "error": run.error,
            "summary": run.summary,
            "log_entries": len(run.log),
        }
    return graph
