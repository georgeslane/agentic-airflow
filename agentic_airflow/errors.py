"""Exception hierarchy for agentic-airflow.

Every exception raised by the library derives from :class:`AgenticAirflowError`
so integrators can catch a single base class.
"""

from __future__ import annotations


class AgenticAirflowError(Exception):
    """Base class for all library errors."""


class DagValidationError(AgenticAirflowError):
    """The DAG definition is invalid (duplicate ids, cycles, bad references...)."""


class ConfigurationError(AgenticAirflowError):
    """The engine or an operator is misconfigured."""


class OutputValidationError(AgenticAirflowError):
    """A model response did not satisfy the task's ``return_type`` contract.

    The message is written so it can be fed straight back to the model as
    corrective feedback.
    """


class ToolError(AgenticAirflowError):
    """A tool could not be resolved, validated or executed."""


class ToolPermissionError(ToolError):
    """The agent attempted to call a tool it is not permitted to use."""


class SchemaValidationError(AgenticAirflowError):
    """A value did not conform to a JSON schema."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


class TriggerError(AgenticAirflowError):
    """A trigger (cron / webhook / dag) is invalid or a delivery was rejected."""


class WebhookAuthenticationError(TriggerError):
    """A webhook delivery failed signature verification."""


class RunError(AgenticAirflowError):
    """Problems with a DAG run (unknown id, invalid state transition...)."""


class ApprovalError(AgenticAirflowError):
    """An approval / decision could not be applied."""
