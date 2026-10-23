"""``DagEvaluator``: the built-in post-run self-optimisation pipeline (spec §6).

Five agentic tasks appended to the terminal nodes of a DAG::

    [workflow_critique] ──┐
    [tool_audit]        ──┼──> [executive_summary]
    [skill_rewriter]    ──┤
    [context_compactor] ──┘

Tasks 1-3 produce *recommendations* that are persisted with status
``pending``; nothing changes until :meth:`Engine.approve_recommendation` is
called (the safety invariant). Task 4 compacts the run log into durable memory
for future runs (applied automatically unless ``apply_memory=False``). Task 5
writes the user-facing summary onto the run.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from .dag import Dag, TaskGroup
from .helpers import found_nothing
from .operators import AgenticTask
from .run import DagRun, Event, TaskInstance

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine

EVALUATOR_GROUP_ID = "dag_evaluator"

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "assessment": {"type": "string"},
        "prompt_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "current": {"type": "string"},
                    "proposed": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["task_id", "proposed", "rationale"],
            },
        },
        "structural_changes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["score", "assessment", "prompt_changes", "structural_changes"],
}

TOOL_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "hide": {"type": "array", "items": {"type": "string"}},
        "allow": {"type": "array", "items": {"type": "string"}},
        "deny": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["hide", "allow", "rationale"],
}

SKILL_SCHEMA = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["name", "content", "rationale"],
            },
        }
    },
    "required": ["skills"],
}

COMPACTOR_SCHEMA = {
    "type": "object",
    "properties": {"memory": {"type": "object"}, "discarded": {"type": "array", "items": {"type": "string"}}},
    "required": ["memory"],
}


def _rec(engine: "Engine", run: DagRun, kind: str, title: str, payload: dict[str, Any], source_task: str) -> dict[str, Any]:
    rec = {
        "id": f"rec_{uuid.uuid4().hex[:12]}",
        "run_id": run.run_id,
        "dag_id": run.dag_id,
        "kind": kind,
        "title": title,
        "payload": payload,
        "source_task": source_task,
        "status": "pending",
        "created_at": time.time(),
        "decided_at": None,
    }
    engine.state.save_recommendation(rec)
    engine._emit(Event("recommendation.created", run.run_id, run.dag_id, source_task, payload=rec))
    return rec


def _on_critique(engine: "Engine", run: DagRun, inst: TaskInstance) -> None:
    out = inst.output or {}
    changes = [c for c in out.get("prompt_changes") or [] if c.get("task_id") and c.get("proposed")]
    structural = out.get("structural_changes") or []
    if changes or structural:
        _rec(engine, run, "topology", "Workflow & prompt changes", {
            "score": out.get("score"),
            "assessment": out.get("assessment"),
            "prompt_changes": changes,
            "structural_changes": structural,
        }, inst.task_id)


def _on_tool_audit(engine: "Engine", run: DagRun, inst: TaskInstance) -> None:
    out = inst.output or {}
    registered = set(engine.tools.names())
    hide = [t for t in out.get("hide") or [] if t in registered]
    allow = [t for t in out.get("allow") or [] if t in registered]
    deny = [t for t in out.get("deny") or [] if t in registered]
    if hide or allow or deny:
        _rec(engine, run, "tools", "Tool permission changes", {"hide": hide, "allow": allow, "deny": deny, "rationale": out.get("rationale")}, inst.task_id)


def _on_skill_rewrite(engine: "Engine", run: DagRun, inst: TaskInstance) -> None:
    out = inst.output or {}
    for item in out.get("skills") or []:
        name, content = item.get("name"), item.get("content")
        if not name or not content:
            continue
        existing = engine.skills.get(name)
        if existing is not None and existing.content.strip() == str(content).strip():
            continue
        _rec(engine, run, "skills", f"Rewrite skill {name!r}", {
            "skill": name, "content": content, "rationale": item.get("rationale"),
            "previous": existing.content if existing else None,
        }, inst.task_id)


def _make_on_compact(apply_memory: bool):
    def _on_compact(engine: "Engine", run: DagRun, inst: TaskInstance) -> None:
        out = inst.output or {}
        memory = out.get("memory")
        if not isinstance(memory, dict):
            return
        if apply_memory:
            previous = engine.get_memory(run.dag_id)
            engine.set_memory(run.dag_id, memory)
            engine.state.set_meta(f"memory_history:{run.dag_id}:{run.run_id}", {"previous": previous, "run_id": run.run_id, "at": time.time()})
        else:
            _rec(engine, run, "memory", "Compacted memory", {"memory": memory, "discarded": out.get("discarded")}, inst.task_id)

    return _on_compact


def _on_summary(engine: "Engine", run: DagRun, inst: TaskInstance) -> None:
    run.summary = str(inst.output or "")
    pending = engine.list_recommendations(run_id=run.run_id, status="pending")
    engine._emit(Event("run.summary", run.run_id, run.dag_id, inst.task_id, payload={"summary": run.summary, "pending_recommendations": [r["id"] for r in pending]}))


def DagEvaluator(
    id: str = EVALUATOR_GROUP_ID,
    *,
    model: str | None = None,
    skills: list[str] | None = None,
    thinking: Any = None,
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    apply_memory: bool = True,
    max_tool_steps: int = 0,
    skip_when_idle: bool = False,
) -> TaskGroup:
    """Build the five-task evaluator group (not yet attached to a DAG).

    By default evaluator tasks get **no tools** (``allowed_tools=[]``,
    ``max_tool_steps=0``) so they cannot act on the world while auditing.
    ``skip_when_idle=True`` skips all five when the run found nothing
    (:func:`~agentic_airflow.helpers.found_nothing`), so a DAG that checks for
    something often costs little when there's nothing new.
    """
    group = TaskGroup(
        id, model=model, skills=skills, thinking=thinking,
        allowed_tools=allowed_tools if allowed_tools is not None else [],
        denied_tools=denied_tools, max_tool_steps=max_tool_steps, dag=None, parent=None,
    )
    group.dag = None
    skip_if = found_nothing if skip_when_idle else None
    common = dict(context=[], trigger_rule="all_done", include_run_log=True, group=group, return_type="json", skip_if=skip_if)
    critique = AgenticTask(
        f"{id}.workflow_critique",
        prompt=(
            "You are auditing a completed workflow run. Compare the run log against the workflow objective "
            "and the workflow definition. Score how effectively the goal was achieved (0-10), explain the main "
            "strengths and failures, and propose concrete improvements: rewritten task prompts (keep task_id, give "
            "the full proposed prompt) and any structural changes (new tasks, removed tasks, re-ordered dependencies) "
            "described in plain language. Only propose changes with clear evidence in the log."
        ),
        json_schema=CRITIQUE_SCHEMA, include_dag_definition=True, on_complete=_on_critique, **common,
    )
    tool_audit = AgenticTask(
        f"{id}.tool_audit",
        prompt=(
            "Audit every tool call in the run log. Recommend: (1) `hide` - tools that were available but unused and "
            "should become on-demand to save context tokens; (2) `allow` - tools that required manual approval during "
            "the run and should be permitted next time; (3) `deny` - tools that caused harm or waste and should be "
            "blocked. Use exact tool names from the log/definition. Be conservative: an empty list is a valid answer."
        ),
        json_schema=TOOL_AUDIT_SCHEMA, include_dag_definition=True, on_complete=_on_tool_audit, **common,
    )
    skill_rewriter = AgenticTask(
        f"{id}.skill_rewriter",
        prompt=(
            "Review the skills used by the tasks in this run together with any failures, rejected outputs or "
            "evaluator reruns in the run log. For each skill whose guidance contributed to a problem or could have "
            "prevented one, draft an improved full replacement text. Return an empty list if no skill needs changes."
        ),
        json_schema=SKILL_SCHEMA, include_dag_definition=True, on_complete=_on_skill_rewrite, **common,
    )
    compactor = AgenticTask(
        f"{id}.context_compactor",
        prompt=(
            "Consolidate memory for future runs of this workflow. Start from the current memory (if any) and the run "
            "log, and produce a compact JSON object containing only durable information that future runs or the user "
            "will need (identifiers, decisions, state of ongoing work, preferences). Discard transient intermediate "
            "results such as raw search output. List the kinds of information you discarded."
        ),
        json_schema=COMPACTOR_SCHEMA, on_complete=_make_on_compact(apply_memory), **common,
    )
    summary = AgenticTask(
        f"{id}.executive_summary",
        prompt=(
            "Write a concise, user-facing executive summary of this workflow run: what it accomplished, what went "
            "wrong, and the key recommendations produced by the critique, tool audit and skill review (listed in the "
            "upstream outputs). Finish with a short bullet list of the recommendations awaiting the user's approval."
        ),
        context=[critique, tool_audit, skill_rewriter, compactor], trigger_rule="all_done", return_type="string",
        on_complete=_on_summary, group=group, skip_if=skip_if,
    )
    for t in (critique, tool_audit, skill_rewriter, compactor):
        t >> summary
    return group


def attach_dag_evaluator(dag: Dag, group: TaskGroup) -> TaskGroup:
    """Attach ``group`` to ``dag`` so it runs after every terminal node."""
    leaves = [t for t in dag.leaves() if t.group is not group]
    group.attach(dag)
    dag.evaluator_group = group
    entry = [t for t in group.tasks if not any(u in group.tasks for u in t.upstream)]
    for leaf in leaves:
        leaf.set_downstream(entry)
    return group


def apply_recommendation(engine: "Engine", rec: dict[str, Any]) -> dict[str, Any]:
    """Apply an approved recommendation. Returns a description of what changed."""
    kind = rec["kind"]
    payload = rec.get("payload") or {}
    dag_id = rec["dag_id"]
    if kind == "tools":
        overrides = dict(engine.overrides_for(dag_id))
        for key in ("allow", "hide", "deny"):
            current = list(overrides.get(key) or [])
            for name in payload.get(key) or []:
                if name not in current:
                    current.append(name)
            overrides[key] = current
        # consistency: an allowed tool is no longer denied/hidden by override and vice-versa
        overrides["deny"] = [t for t in overrides["deny"] if t not in (payload.get("allow") or [])]
        overrides["allow"] = [t for t in overrides["allow"] if t not in (payload.get("deny") or [])]
        engine.set_overrides(dag_id, overrides)
        return {"overrides": overrides}
    if kind == "skills":
        skill = engine.skills.register(payload["skill"], payload["content"], note=f"DagEvaluator rewrite from run {rec.get('run_id')}")
        return {"skill": skill.name, "version": skill.current}
    if kind == "topology":
        overrides = dict(engine.overrides_for(dag_id))
        prompts = dict(overrides.get("prompts") or {})
        dag = engine.dags.get(dag_id)
        applied, manual = [], list(payload.get("structural_changes") or [])
        for change in payload.get("prompt_changes") or []:
            task_id = change.get("task_id")
            if dag is not None and task_id in dag.tasks and dag.get(task_id).is_agentic:
                prompts[task_id] = change["proposed"]
                applied.append(task_id)
            else:
                manual.append(f"prompt change for unknown/non-agentic task {task_id!r}")
        overrides["prompts"] = prompts
        engine.set_overrides(dag_id, overrides)
        return {"prompts_updated": applied, "manual_follow_up": manual}
    if kind == "memory":
        engine.set_memory(dag_id, payload.get("memory") or {})
        return {"memory_keys": sorted(payload.get("memory") or {})}
    return {"note": f"recommendation kind {kind!r} is advisory"}
