"""Persistence: :class:`StateStore` protocol with in-memory and SQLite backends.

Everything stored is plain JSON (no pickle), so stores are safe to share and
inspect. The SQLite store uses WAL mode and a single connection guarded by a
lock, which is plenty for a lightweight orchestrator and keeps the dependency
footprint at zero.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable, Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    """Minimal persistence contract. All values are JSON-serialisable dicts."""

    # runs
    def save_run(self, run: dict[str, Any]) -> None: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def list_runs(self, dag_id: str | None = None, status: str | Iterable[str] | None = None, limit: int = 50) -> list[dict[str, Any]]: ...
    def delete_run(self, run_id: str) -> None: ...
    # memory + overrides (per dag)
    def get_memory(self, dag_id: str) -> dict[str, Any]: ...
    def set_memory(self, dag_id: str, memory: dict[str, Any]) -> None: ...
    def get_overrides(self, dag_id: str) -> dict[str, Any]: ...
    def set_overrides(self, dag_id: str, overrides: dict[str, Any]) -> None: ...
    # skills
    def save_skill(self, skill: dict[str, Any]) -> None: ...
    def list_skills(self) -> list[dict[str, Any]]: ...
    # approvals + recommendations
    def save_approval(self, approval: dict[str, Any]) -> None: ...
    def get_approval(self, approval_id: str) -> dict[str, Any] | None: ...
    def list_approvals(self, status: str | None = None, run_id: str | None = None) -> list[dict[str, Any]]: ...
    def save_recommendation(self, rec: dict[str, Any]) -> None: ...
    def get_recommendation(self, rec_id: str) -> dict[str, Any] | None: ...
    def list_recommendations(self, dag_id: str | None = None, run_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]: ...
    # trigger bookkeeping
    def get_meta(self, key: str) -> dict[str, Any] | None: ...
    def set_meta(self, key: str, value: dict[str, Any]) -> None: ...


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_default)


def _default(obj: Any) -> Any:
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _roundtrip(value: Any) -> Any:
    """Deep-copy via JSON so callers can't mutate stored state."""
    return json.loads(_json_dumps(value))


class InMemoryStateStore:
    """Non-persistent store (default). Good for tests and ephemeral agents."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def _put(self, collection: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._docs.setdefault(collection, {})[key] = _roundtrip(value)

    def _get(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._docs.get(collection, {}).get(key)
            return _roundtrip(value) if value is not None else None

    def _all(self, collection: str) -> list[dict[str, Any]]:
        with self._lock:
            return [_roundtrip(v) for v in self._docs.get(collection, {}).values()]

    # runs
    def save_run(self, run: dict[str, Any]) -> None:
        self._put("runs", run["run_id"], run)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._get("runs", run_id)

    def delete_run(self, run_id: str) -> None:
        """Delete a run and its approvals."""
        with self._lock:
            self._docs.get("runs", {}).pop(run_id, None)
            approvals = self._docs.get("approvals", {})
            for key in [k for k, a in approvals.items() if a.get("run_id") == run_id]:
                approvals.pop(key)

    def list_runs(self, dag_id=None, status=None, limit=50):
        statuses = {status} if isinstance(status, str) else (set(status) if status else None)
        runs = [
            r for r in self._all("runs")
            if (dag_id is None or r.get("dag_id") == dag_id) and (statuses is None or r.get("status") in statuses)
        ]
        runs.sort(key=lambda r: r.get("created_at", 0), reverse=True)
        return runs[:limit]

    # memory / overrides
    def get_memory(self, dag_id):
        return self._get("memory", dag_id) or {}

    def set_memory(self, dag_id, memory):
        self._put("memory", dag_id, memory)

    def get_overrides(self, dag_id):
        return self._get("overrides", dag_id) or {}

    def set_overrides(self, dag_id, overrides):
        self._put("overrides", dag_id, overrides)

    # skills
    def save_skill(self, skill):
        self._put("skills", skill["name"], skill)

    def list_skills(self):
        return self._all("skills")

    # approvals
    def save_approval(self, approval):
        self._put("approvals", approval["approval_id"], approval)

    def get_approval(self, approval_id):
        return self._get("approvals", approval_id)

    def list_approvals(self, status=None, run_id=None):
        items = [
            a for a in self._all("approvals")
            if (status is None or a.get("status") == status) and (run_id is None or a.get("run_id") == run_id)
        ]
        items.sort(key=lambda a: a.get("created_at", 0))
        return items

    # recommendations
    def save_recommendation(self, rec):
        self._put("recommendations", rec["id"], rec)

    def get_recommendation(self, rec_id):
        return self._get("recommendations", rec_id)

    def list_recommendations(self, dag_id=None, run_id=None, status=None):
        items = [
            r for r in self._all("recommendations")
            if (dag_id is None or r.get("dag_id") == dag_id)
            and (run_id is None or r.get("run_id") == run_id)
            and (status is None or r.get("status") == status)
        ]
        items.sort(key=lambda r: r.get("created_at", 0))
        return items

    # meta
    def get_meta(self, key):
        return self._get("meta", key)

    def set_meta(self, key, value):
        self._put("meta", key, value)


class SqliteStateStore(InMemoryStateStore):
    """Durable store backed by a single SQLite file (stdlib ``sqlite3``).

    Layout: one ``documents`` table ``(collection, key, scope, status, value,
    updated_at)``. ``scope`` holds the dag id / run id used for filtering so
    listings never deserialise unrelated rows.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self._lock = threading.RLock()
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                collection TEXT NOT NULL,
                key TEXT NOT NULL,
                scope TEXT,
                scope2 TEXT,
                status TEXT,
                created_at REAL,
                updated_at REAL NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (collection, key)
            )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(collection, scope, status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_scope2 ON documents(collection, scope2)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _write(self, collection: str, key: str, value: dict[str, Any], scope: str | None = None, scope2: str | None = None, status: str | None = None) -> None:
        created = value.get("created_at") if isinstance(value, dict) else None
        with self._lock:
            self._conn.execute(
                """INSERT INTO documents(collection, key, scope, scope2, status, created_at, updated_at, value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(collection, key) DO UPDATE SET
                     scope=excluded.scope, scope2=excluded.scope2, status=excluded.status,
                     created_at=COALESCE(documents.created_at, excluded.created_at),
                     updated_at=excluded.updated_at, value=excluded.value""",
                (collection, key, scope, scope2, status, created if isinstance(created, (int, float)) else time.time(), time.time(), _json_dumps(value)),
            )

    def _read(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM documents WHERE collection=? AND key=?", (collection, key)).fetchone()
        return json.loads(row[0]) if row else None

    def _query(self, collection: str, scope=None, scope2=None, statuses=None, limit=None, newest_first=False) -> list[dict[str, Any]]:
        sql = "SELECT value FROM documents WHERE collection=?"
        params: list[Any] = [collection]
        if scope is not None:
            sql += " AND scope=?"
            params.append(scope)
        if scope2 is not None:
            sql += " AND scope2=?"
            params.append(scope2)
        if statuses:
            sql += " AND status IN (%s)" % ",".join("?" * len(statuses))
            params.extend(statuses)
        sql += " ORDER BY created_at %s" % ("DESC" if newest_first else "ASC")
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r[0]) for r in rows]

    # runs
    def save_run(self, run):
        self._write("runs", run["run_id"], run, scope=run.get("dag_id"), status=run.get("status"))

    def get_run(self, run_id):
        return self._read("runs", run_id)

    def delete_run(self, run_id):
        with self._lock:
            self._conn.execute("DELETE FROM documents WHERE collection='runs' AND key=?", (run_id,))
            self._conn.execute("DELETE FROM documents WHERE collection='approvals' AND scope=?", (run_id,))

    def list_runs(self, dag_id=None, status=None, limit=50):
        statuses = [status] if isinstance(status, str) else (list(status) if status else None)
        return self._query("runs", scope=dag_id, statuses=statuses, limit=limit, newest_first=True)

    def get_memory(self, dag_id):
        return self._read("memory", dag_id) or {}

    def set_memory(self, dag_id, memory):
        self._write("memory", dag_id, memory, scope=dag_id)

    def get_overrides(self, dag_id):
        return self._read("overrides", dag_id) or {}

    def set_overrides(self, dag_id, overrides):
        self._write("overrides", dag_id, overrides, scope=dag_id)

    def save_skill(self, skill):
        self._write("skills", skill["name"], skill)

    def list_skills(self):
        return self._query("skills")

    def save_approval(self, approval):
        self._write("approvals", approval["approval_id"], approval, scope=approval.get("run_id"), status=approval.get("status"))

    def get_approval(self, approval_id):
        return self._read("approvals", approval_id)

    def list_approvals(self, status=None, run_id=None):
        return self._query("approvals", scope=run_id, statuses=[status] if status else None)

    def save_recommendation(self, rec):
        self._write("recommendations", rec["id"], rec, scope=rec.get("dag_id"), scope2=rec.get("run_id"), status=rec.get("status"))

    def get_recommendation(self, rec_id):
        return self._read("recommendations", rec_id)

    def list_recommendations(self, dag_id=None, run_id=None, status=None):
        return self._query("recommendations", scope=dag_id, scope2=run_id, statuses=[status] if status else None)

    def get_meta(self, key):
        return self._read("meta", key)

    def set_meta(self, key, value):
        self._write("meta", key, value)
