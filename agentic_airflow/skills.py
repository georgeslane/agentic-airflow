"""Skill store: named, versioned prompt fragments injected into system prompts.

Skills are plain text/markdown documents (``skills=['Investment', 'Research']``).
They can be registered in code, loaded from a directory of ``*.md`` files, and
are versioned so the :class:`~agentic_airflow.evaluator.DagEvaluator` can
propose rewrites that only take effect once a human approves them.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigurationError

SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9_. -]{1,80}$")


@dataclass
class SkillVersion:
    version: int
    content: str
    created_at: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "content": self.content, "created_at": self.created_at, "note": self.note}


@dataclass
class Skill:
    name: str
    versions: list[SkillVersion] = field(default_factory=list)
    current: int = 0

    @property
    def content(self) -> str:
        for v in self.versions:
            if v.version == self.current:
                return v.content
        return self.versions[-1].content if self.versions else ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "current": self.current, "versions": [v.to_dict() for v in self.versions]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        s = cls(name=data["name"], current=int(data.get("current") or 0))
        s.versions = [SkillVersion(**v) for v in data.get("versions") or []]
        return s


class SkillStore:
    """In-memory registry of skills with optional persistence hooks.

    Pass ``state=`` (any :class:`~agentic_airflow.state.StateStore`) to persist
    versions across restarts; the engine wires this automatically.
    """

    def __init__(self, state: Any | None = None):
        self._skills: dict[str, Skill] = {}
        self._state = state
        if state is not None:
            for data in state.list_skills():
                skill = Skill.from_dict(data)
                self._skills[skill.name] = skill

    def _persist(self, skill: Skill) -> None:
        if self._state is not None:
            self._state.save_skill(skill.to_dict())

    def register(self, name: str, content: str, note: str = "", *, activate: bool = True) -> Skill:
        """Register a skill or add a new version to an existing one."""
        if not SKILL_NAME_RE.match(name):
            raise ConfigurationError(f"Invalid skill name {name!r}")
        skill = self._skills.get(name)
        if skill is None:
            skill = Skill(name=name)
            self._skills[name] = skill
        if skill.versions and skill.content == content and activate:
            return skill
        version = (skill.versions[-1].version + 1) if skill.versions else 1
        skill.versions.append(SkillVersion(version=version, content=content, created_at=time.time(), note=note))
        if activate or not skill.current:
            skill.current = version
        self._persist(skill)
        return skill

    def activate(self, name: str, version: int) -> Skill:
        skill = self.require(name)
        if version not in {v.version for v in skill.versions}:
            raise ConfigurationError(f"Skill {name!r} has no version {version}")
        skill.current = version
        self._persist(skill)
        return skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def require(self, name: str) -> Skill:
        skill = self._skills.get(name)
        if skill is None:
            raise ConfigurationError(f"Unknown skill {name!r}")
        return skill

    def names(self) -> list[str]:
        return list(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def load_directory(self, path: str | os.PathLike[str], pattern: str = "*.md") -> list[str]:
        """Load ``<name>.md`` files from a directory (non-recursive)."""
        root = Path(path).resolve()
        if not root.is_dir():
            raise ConfigurationError(f"Skill directory {root} does not exist")
        loaded = []
        for file in sorted(root.glob(pattern)):
            if not file.is_file() or file.resolve().parent != root:
                continue  # ignore symlinks escaping the directory
            self.register(file.stem, file.read_text(encoding="utf-8"), note=f"loaded from {file.name}")
            loaded.append(file.stem)
        return loaded

    def render(self, names: Iterable[str], *, strict: bool = False) -> str:
        """Render skills as a markdown block for a system prompt."""
        blocks = []
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                if strict:
                    raise ConfigurationError(f"Unknown skill {name!r}")
                continue
            blocks.append(f"### Skill: {name} (v{skill.current})\n{skill.content.strip()}")
        return "\n\n".join(blocks)
