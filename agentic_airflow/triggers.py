"""Triggers that start DAG runs: cron schedules, other DAGs and webhooks.

``Dag(schedule=[...])`` accepts plain cron strings (converted to
:class:`CronTrigger`), :class:`DagTrigger` and :class:`WebhookTrigger`
instances.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from .errors import TriggerError, WebhookAuthenticationError

# ---------------------------------------------------------------------------
# Cron parsing (5-field Unix cron, no external dependency)
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_DAYS = {d: i for i, d in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))}
_ALIASES = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}


def _parse_field(text: str, lo: int, hi: int, names: dict[str, int] | None = None) -> tuple[set[int], bool]:
    """Parse one cron field into a set of ints. Returns (values, is_wildcard)."""
    values: set[int] = set()
    wildcard = False
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            raise TriggerError(f"Empty cron field component in {text!r}")
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            if not step_s.isdigit() or int(step_s) < 1:
                raise TriggerError(f"Invalid cron step {step_s!r}")
            step = int(step_s)
        if part == "*":
            start, end = lo, hi
            if step == 1:
                wildcard = True
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = _parse_value(a, names, lo, hi), _parse_value(b, names, lo, hi)
            if start > end:
                raise TriggerError(f"Invalid cron range {part!r}")
        else:
            start = end = _parse_value(part, names, lo, hi)
            if step != 1:
                end = hi
        values.update(range(start, end + 1, step))
    return values, wildcard


def _parse_value(token: str, names: dict[str, int] | None, lo: int, hi: int) -> int:
    if names and token in names:
        return names[token]
    if not token.isdigit():
        raise TriggerError(f"Invalid cron value {token!r}")
    value = int(token)
    if value < lo or value > hi:
        raise TriggerError(f"Cron value {value} out of range {lo}-{hi}")
    return value


class CronExpression:
    """A parsed 5-field cron expression with ``next_after`` computation."""

    def __init__(self, expression: str):
        expr = _ALIASES.get(expression.strip().lower(), expression.strip())
        parts = expr.split()
        if len(parts) != 5:
            raise TriggerError(f"Cron expression must have 5 fields, got {expression!r}")
        self.expression = expression
        self.minutes, _ = _parse_field(parts[0], 0, 59)
        self.hours, _ = _parse_field(parts[1], 0, 23)
        self.days, self.dom_wild = _parse_field(parts[2], 1, 31)
        self.months, _ = _parse_field(parts[3], 1, 12, _MONTHS)
        dows, self.dow_wild = _parse_field(parts[4], 0, 7, _DAYS)
        self.dows = {0 if d == 7 else d for d in dows}

    def _day_matches(self, dt: datetime) -> bool:
        dom = dt.day in self.days
        dow = ((dt.weekday() + 1) % 7) in self.dows  # python Monday=0 -> cron Sunday=0
        if self.dom_wild and self.dow_wild:
            return True
        if self.dom_wild:
            return dow
        if self.dow_wild:
            return dom
        return dom or dow  # standard Unix semantics: OR when both restricted

    def matches(self, dt: datetime) -> bool:
        return (
            dt.minute in self.minutes
            and dt.hour in self.hours
            and dt.month in self.months
            and self._day_matches(dt)
        )

    def next_after(self, after: datetime, limit_years: int = 5) -> datetime:
        """Return the first matching time strictly after ``after``."""
        dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        horizon = after + timedelta(days=366 * limit_years)
        while dt <= horizon:
            if dt.month not in self.months:
                year, month = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
                dt = dt.replace(year=year, month=month, day=1, hour=0, minute=0)
                continue
            if not self._day_matches(dt):
                dt = (dt + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if dt.hour not in self.hours:
                dt = (dt + timedelta(hours=1)).replace(minute=0)
                continue
            if dt.minute not in self.minutes:
                dt = dt + timedelta(minutes=1)
                continue
            return dt
        raise TriggerError(f"Cron expression {self.expression!r} never fires within {limit_years} years")


# ---------------------------------------------------------------------------
# Trigger types
# ---------------------------------------------------------------------------


@dataclass
class Trigger:
    """Base class for schedule entries."""

    kind: str = field(init=False, default="trigger")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass
class CronTrigger(Trigger):
    """Fire on a cron schedule, evaluated in ``timezone`` (IANA name)."""

    expression: str = "0 * * * *"
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        self.kind = "cron"
        self.cron = CronExpression(self.expression)
        try:
            self.tzinfo = ZoneInfo(self.timezone)
        except Exception as exc:  # noqa: BLE001
            raise TriggerError(f"Unknown timezone {self.timezone!r}") from exc

    def next_fire(self, after: datetime | None = None) -> datetime:
        """Next fire time as an aware UTC datetime."""
        now = after or datetime.now(timezone.utc)
        local = now.astimezone(self.tzinfo)
        return self.cron.next_after(local).astimezone(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "cron", "expression": self.expression, "timezone": self.timezone}


@dataclass
class DagTrigger(Trigger):
    """Fire when another DAG's run finishes with one of the ``on`` statuses."""

    dag_id: str = ""
    on: Sequence[str] = ("success",)

    def __post_init__(self) -> None:
        self.kind = "dag"
        if not self.dag_id:
            raise TriggerError("DagTrigger requires a dag_id")
        self.on = tuple(s.lower() for s in self.on)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "dag", "dag_id": self.dag_id, "on": list(self.on)}


_WEBHOOK_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


@dataclass
class WebhookTrigger(Trigger):
    """Fire when a payload is delivered to the named webhook.

    When ``secret`` is set, deliveries must carry an HMAC-SHA256 hex digest of
    the raw body (``X-Signature-256: sha256=<hex>`` on the HTTP server) and are
    rejected otherwise. ``max_bytes`` bounds the accepted payload size.
    """

    name: str = ""
    secret: str | None = None
    max_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        self.kind = "webhook"
        if not _WEBHOOK_NAME_RE.match(self.name or ""):
            raise TriggerError(f"Invalid webhook name {self.name!r}")

    def sign(self, raw_body: bytes) -> str:
        if not self.secret:
            raise TriggerError("Webhook has no secret configured")
        return hmac.new(self.secret.encode(), raw_body, hashlib.sha256).hexdigest()

    def verify(self, raw_body: bytes | None, signature: str | None) -> None:
        """Raise :class:`WebhookAuthenticationError` on failure."""
        if raw_body is not None and len(raw_body) > self.max_bytes:
            raise WebhookAuthenticationError("Webhook payload too large")
        if not self.secret:
            return
        if raw_body is None or not signature:
            raise WebhookAuthenticationError("Missing webhook signature")
        expected = self.sign(raw_body)
        given = signature.strip()
        if given.lower().startswith("sha256="):
            given = given[7:]
        if not hmac.compare_digest(expected, given.lower()):
            raise WebhookAuthenticationError("Invalid webhook signature")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "webhook", "name": self.name, "signed": bool(self.secret)}


def normalise_schedule(schedule: Iterable[Any] | Any | None) -> list[Trigger]:
    """Accept strings / Trigger instances (or a single one) and return Triggers."""
    if schedule is None:
        return []
    if isinstance(schedule, (str, Trigger)):
        schedule = [schedule]
    out: list[Trigger] = []
    for item in schedule:
        if isinstance(item, Trigger):
            out.append(item)
        elif isinstance(item, str):
            out.append(CronTrigger(item))
        else:
            raise TriggerError(f"Unsupported schedule entry {item!r}")
    return out


# Short aliases matching the spec's vocabulary.
Cron = CronTrigger
Webhook = WebhookTrigger
