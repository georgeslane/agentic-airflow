"""Human-gated change proposals.

Anything (an agent via the MCP server, the DagEvaluator, your own code) can
*propose* a change to a DAG's prompts, tool permissions, skills or memory.
Proposals are stored as recommendations with status ``pending`` and take
effect only when a human calls :meth:`Engine.approve_recommendation` (or
clicks approve in the UI). Nothing in this module applies a change.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from .errors import ApprovalError, ConfigurationError
from .run import Event
from .skills import SKILL_NAME_RE

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Engine

MAX_RATIONALE = 4000
MAX_PROMPT = 50_000
MAX_PENDING = 20  # from one source, for one DAG: withdraw or decide some before proposing more


def _store(engine: "Engine", *, dag_id: str | None, kind: str, title: str, payload: dict[str, Any], rationale: str, source: str, proposed_by: str | None) -> dict[str, Any]:
    if not isinstance(rationale, str) or not rationale.strip():
        raise ConfigurationError("A rationale is required for every proposal")
    waiting = [r for r in engine.state.list_recommendations(dag_id=dag_id, status="pending") if r.get("source") == source]
    if len(waiting) >= MAX_PENDING:
        raise ConfigurationError(f"{len(waiting)} proposals from {source!r} are waiting for {dag_id!r} already: withdraw some, or wait for a human to decide them")
    rec = {
        "id": f"rec_{uuid.uuid4().hex[:12]}",
        "run_id": None,
        "dag_id": dag_id,
        "kind": kind,
        "title": title,
        "payload": payload,
        "rationale": rationale.strip()[:MAX_RATIONALE],
        "source": source,
        "proposed_by": proposed_by,
        "source_task": None,
        "status": "pending",
        "created_at": time.time(),
        "decided_at": None,
    }
    engine.state.save_recommendation(rec)
    engine._emit(Event("recommendation.created", None, dag_id, payload=rec))
    return rec


def propose_prompt_change(engine: "Engine", dag_id: str, task_id: str, proposed_prompt: str, rationale: str, *, source: str = "api", proposed_by: str | None = None) -> dict[str, Any]:
    """Propose a new prompt for an agentic task (applied as a persisted override on approval)."""
    dag = engine._dag(dag_id)
    op = dag.get(task_id)
    if not op.is_agentic:
        raise ConfigurationError(f"Task {task_id!r} is not agentic; only agentic prompts can be changed")
    if not isinstance(op.prompt, str):
        raise ConfigurationError(f"Task {task_id!r} writes its prompt when it starts, in code; describe the change with propose_structural_change instead")
    if not isinstance(proposed_prompt, str) or not proposed_prompt.strip():
        raise ConfigurationError("proposed_prompt must be a non-empty string")
    if len(proposed_prompt) > MAX_PROMPT:
        raise ConfigurationError(f"proposed_prompt exceeds {MAX_PROMPT} characters")
    current = (engine.overrides_for(dag_id).get("prompts") or {}).get(task_id) or op.prompt
    payload = {
        "prompt_changes": [{"task_id": task_id, "current": current, "proposed": proposed_prompt, "rationale": rationale}],
        "structural_changes": [],
    }
    return _store(engine, dag_id=dag_id, kind="topology", title=f"Prompt change for {task_id!r}", payload=payload, rationale=rationale, source=source, proposed_by=proposed_by)


def propose_structural_change(engine: "Engine", dag_id: str, description: str, rationale: str, *, source: str = "api", proposed_by: str | None = None) -> dict[str, Any]:
    """Propose a change to tasks/dependencies. Advisory: DAGs are code, so a human applies it."""
    engine._dag(dag_id)
    if not isinstance(description, str) or not description.strip():
        raise ConfigurationError("description must be a non-empty string")
    payload = {"prompt_changes": [], "structural_changes": [description.strip()[:MAX_PROMPT]]}
    return _store(engine, dag_id=dag_id, kind="topology", title="Structural change (manual follow-up)", payload=payload, rationale=rationale, source=source, proposed_by=proposed_by)


def propose_tool_permissions(engine: "Engine", dag_id: str, *, allow: list[str] | None = None, hide: list[str] | None = None, deny: list[str] | None = None, unset: list[str] | None = None, rationale: str, source: str = "api", proposed_by: str | None = None) -> dict[str, Any]:
    """Propose DAG-level tool permission overrides: allow, hide or deny tools, or ``unset`` earlier overrides of
    them, so the DAG's own settings apply again."""
    engine._dag(dag_id)
    registered = set(engine.tools.names())
    lists = {"allow": list(allow or []), "hide": list(hide or []), "deny": list(deny or [])}
    unset = list(unset or [])
    unknown = sorted({t for names in lists.values() for t in names} - registered)
    if unknown:
        raise ConfigurationError(f"Unknown tool(s) {unknown}; registered tools: {sorted(registered)}")
    if not any(lists.values()) and not unset:
        raise ConfigurationError("Provide at least one of allow, hide, deny or unset")
    overrides = engine.overrides_for(dag_id)
    overridden = {t for key in ("allow", "hide", "deny") for t in overrides.get(key) or []}
    if not_overridden := sorted(set(unset) - overridden):
        raise ConfigurationError(f"No override to unset for {not_overridden}")
    if both := sorted(set(unset) & {t for names in lists.values() for t in names}):
        raise ConfigurationError(f"Tools cannot be both set and unset: {both}")
    overlap = set(lists["allow"]) & set(lists["deny"])
    if overlap:
        raise ConfigurationError(f"Tools cannot be both allowed and denied: {sorted(overlap)}")
    payload = {**lists, "unset": unset, "rationale": rationale}
    return _store(engine, dag_id=dag_id, kind="tools", title="Tool permission changes", payload=payload, rationale=rationale, source=source, proposed_by=proposed_by)


def propose_skill_rewrite(engine: "Engine", skill: str, content: str, rationale: str, *, dag_id: str | None = None, source: str = "api", proposed_by: str | None = None) -> dict[str, Any]:
    """Propose a new version of a skill (or a brand-new skill)."""
    if not SKILL_NAME_RE.match(skill or ""):
        raise ConfigurationError(f"Invalid skill name {skill!r}")
    if not isinstance(content, str) or not content.strip():
        raise ConfigurationError("content must be a non-empty string")
    if len(content) > MAX_PROMPT:
        raise ConfigurationError(f"content exceeds {MAX_PROMPT} characters")
    existing = engine.skills.get(skill)
    if existing is not None and existing.content.strip() == content.strip():
        raise ConfigurationError(f"Skill {skill!r} already has this content")
    payload = {"skill": skill, "content": content, "rationale": rationale, "previous": existing.content if existing else None}
    return _store(engine, dag_id=dag_id, kind="skills", title=f"Rewrite skill {skill!r}" if existing else f"New skill {skill!r}", payload=payload, rationale=rationale, source=source, proposed_by=proposed_by)


def propose_memory_update(engine: "Engine", dag_id: str, memory: dict[str, Any], rationale: str, *, source: str = "api", proposed_by: str | None = None) -> dict[str, Any]:
    """Propose replacing a DAG's persisted memory."""
    engine._dag(dag_id)
    if not isinstance(memory, dict):
        raise ConfigurationError("memory must be a JSON object")
    payload = {"memory": memory, "previous": engine.get_memory(dag_id)}
    return _store(engine, dag_id=dag_id, kind="memory", title="Memory update", payload=payload, rationale=rationale, source=source, proposed_by=proposed_by)


def withdraw_proposal(engine: "Engine", rec_id: str, *, source: str | None = None, proposed_by: str | None = None, reason: str = "") -> dict[str, Any]:
    """Withdraw a pending proposal. With ``source`` set, only proposals from that source can be withdrawn, and with
    ``proposed_by`` too, only those it made."""
    rec = engine.state.get_recommendation(rec_id)
    if rec is None:
        raise ApprovalError(f"Unknown proposal {rec_id!r}")
    if rec.get("status") != "pending":
        raise ApprovalError(f"Proposal {rec_id!r} is already {rec.get('status')}")
    if source is not None and rec.get("source") != source:
        raise ApprovalError(f"Proposal {rec_id!r} was not created by {source!r}; it cannot be withdrawn here")
    if source is not None and rec.get("proposed_by") != proposed_by:
        raise ApprovalError(f"Proposal {rec_id!r} was made by another client; it cannot be withdrawn here")
    rec["status"] = "withdrawn"
    rec["decided_at"] = time.time()
    rec["reason"] = reason
    engine.state.save_recommendation(rec)
    engine._emit(Event("recommendation.withdrawn", rec.get("run_id"), rec.get("dag_id"), payload=rec))
    return rec
