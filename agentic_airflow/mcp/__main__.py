"""Run the MCP server over stdio.

    python -m agentic_airflow.mcp myapp.workflows:engine [--allow-trigger] [--client-name NAME]

The target is ``module:attribute`` where the attribute is an
:class:`~agentic_airflow.engine.Engine` or a zero-argument callable returning
one. Point it at the same persistent state store as your running orchestrator.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from . import build_server


def load_engine(target: str):
    if ":" not in target:
        raise SystemExit("target must be module:attribute, e.g. myapp.workflows:engine")
    module_name, attr = target.split(":", 1)
    sys.path.insert(0, "")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    if callable(obj) and not hasattr(obj, "dags"):
        obj = obj()
    if not hasattr(obj, "dags") or not hasattr(obj, "state"):
        raise SystemExit(f"{target} is not an agentic_airflow Engine")
    return obj


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m agentic_airflow.mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="module:attribute resolving to an Engine (or a factory)")
    parser.add_argument("--allow-trigger", action="store_true", help="expose trigger_run (starts runs that execute tools)")
    parser.add_argument("--client-name", default=None, help="recorded as proposed_by on proposals")
    parser.add_argument("--output-chars", type=int, default=4000, help="truncate outputs in listings (0 = unlimited)")
    args = parser.parse_args(argv)
    engine = load_engine(args.target)
    server = build_server(engine, allow_trigger=args.allow_trigger, client_name=args.client_name, output_chars=args.output_chars)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
