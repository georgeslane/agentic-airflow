"""agentic-airflow: Airflow-inspired orchestration for LLM agents.

Quick start::

    from agentic_airflow import Dag, Task, AgenticTask, Engine, EngineConfig

    with Dag("inbox", prompt="Triage my inbox", schedule=["0 9 * * 1-5"]) as dag:
        read = Task("read_emails", fn=fetch_emails, fan_out=True)
        draft = AgenticTask("draft_reply", prompt="Draft a reply to this email.")
        summary = AgenticTask("summary", prompt="Summarise the drafted replies.")
        read >> draft >> summary

    engine = Engine(backend=my_backend, config=EngineConfig(model="qwen-3.6"))
    engine.register(dag)
"""

from .config import EngineConfig, ResolvedConfig, ScopeConfig, resolve_config
from .dag import Dag, TaskGroup
from .engine import Engine
from .errors import (
    AgenticAirflowError,
    ApprovalError,
    ConfigurationError,
    DagValidationError,
    OutputValidationError,
    PermanentError,
    RunError,
    SchemaValidationError,
    StaleProposalError,
    ToolError,
    ToolPermissionError,
    TriggerError,
    WebhookAuthenticationError,
)
from .evaluator import DagEvaluator, attach_dag_evaluator
from .helpers import found_nothing, trigger_items
from .models import (
    FunctionBackend,
    GenerationRequest,
    GenerationResponse,
    Message,
    ModelBackend,
    OutputSpec,
    ScriptedBackend,
    ToolCall,
    ToolSpec,
)
from .operators import NO_BRANCH, AgenticBranch, AgenticEvaluator, AgenticTask, Branch, Task
from .proposals import Applier
from .run import Approval, DagRun, Event, RunStatus, TaskContext, TaskInstance, TaskStatus
from .skills import SkillStore
from .state import InMemoryStateStore, SqliteStateStore, StateStore
from .tools import Tool, ToolContext, ToolPermissions, ToolRegistry, resolve_permissions, tool
from .triggers import Asset, AssetTrigger, Cron, CronTrigger, DagTrigger, Trigger, Webhook, WebhookTrigger

__version__ = "0.1.0"

__all__ = [
    "AgenticAirflowError", "ApprovalError", "ConfigurationError", "DagValidationError", "OutputValidationError",
    "PermanentError", "RunError", "SchemaValidationError", "StaleProposalError", "ToolError", "ToolPermissionError", "TriggerError", "WebhookAuthenticationError",
    "Dag", "TaskGroup", "Task", "Branch", "AgenticTask", "AgenticBranch", "AgenticEvaluator", "NO_BRANCH",
    "DagEvaluator", "attach_dag_evaluator", "trigger_items", "found_nothing", "Applier",
    "Engine", "EngineConfig", "ScopeConfig", "ResolvedConfig", "resolve_config",
    "ModelBackend", "FunctionBackend", "ScriptedBackend", "GenerationRequest", "GenerationResponse", "Message", "ToolCall", "ToolSpec", "OutputSpec",
    "Tool", "ToolContext", "ToolRegistry", "ToolPermissions", "tool", "resolve_permissions", "SkillStore",
    "StateStore", "InMemoryStateStore", "SqliteStateStore",
    "DagRun", "TaskInstance", "TaskStatus", "RunStatus", "TaskContext", "Approval", "Event",
    "Trigger", "Cron", "CronTrigger", "DagTrigger", "Webhook", "WebhookTrigger", "Asset", "AssetTrigger",
    "__version__",
]
