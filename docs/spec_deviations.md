# Spec deviations & design decisions

The implementation follows `agentic_workflow_orchestrator_specification.md` closely. These are the places where it adjusts or clarifies the spec, and why.

| Topic | Spec | Implementation | Why |
|---|---|---|---|
| Default `context` | "explicit upstream references" | `context=None` means the direct upstream dependencies; `[]` means nothing | Without it `a >> b` would hand `b` no data, and the fan-out rule ("instance `i` receives element `i`") implies implicit data flow anyway. |
| Branch data flow | not specified | branches are transparent: downstream nodes receive what the branch received; `context=[branch]` yields the selection | `read >> classify >> [reply, archive]` must give `reply` the email, not the string `"reply"`. |
| Downstream of a mapped node | joins (fan-in dict) | same default, plus `follow_mapping=True` to stay mapped 1:1; evaluators default to it | Per-item pipelines (draft_i → review_i → send_i) are the norm for agents; the evaluator loop only makes sense per instance. |
| Nested fan-out | not covered | supported, ids `task_i_j`, joins collect all leaves | Natural generalisation of the index path. |
| Evaluator output | "validated output ... passed to the next downstream task" | the evaluator instance's output *is* the target output; verdicts are stored in `instance.verdict` and the run log | Keeps `>>` chains simple. |
| Atomic unit | generation + synchronous tool execution | the model slot is held only during generation; tools execute right after, before the next step | A slow tool should not block other tasks or user chat waiting for the model. Interruption still happens only at step boundaries. |
| `max_tool_steps` | hard cap | at the cap the tools are withdrawn and the model is asked for a final answer; a further tool request fails the task | Gives the task one chance to finish gracefully instead of failing with work done. |
| `denied_tools` default | "all registered tools not in allowed" | identical; denied status also covers overrides so a DagEvaluator `allow` recommendation can never beat an explicit deny | Safety. |
| Hidden tools | configured manually or via DagEvaluator | `hidden_tools` at every scope (additive), `Tool(hidden=True)` registry default, `hide` overrides; discovery via the built-in `search_tools` tool | The spec's "discoverable if the agent queries the registry" needs a concrete mechanism. |
| Unknown tool names | not covered | error message to the model (no pause) | Nothing to approve: the tool does not exist. |
| DagEvaluator memory | "No modifications ... take effect until approved" lists structure, permissions, skills | compacted memory is applied automatically (reversible, previous memory kept in meta); `apply_memory=False` gates it | Otherwise memory never evolves without a human in the loop for every run. |
| Structural DAG changes | staged diff | prompt changes are applied as persisted overrides on approval; new tasks/edges are reported as manual follow-ups | DAGs are defined in code; rewriting source is out of scope. |
| Cron catch-up | not covered | missed firings while the process was down are not back-filled | Predictable for agent workloads; trigger manually if needed. |
| Images/files | return types | validated as `{"kind": "image", mime/data_base64 | path | url}` / `{"kind": "file", "path"}` | A concrete, serialisable contract. |
| `auto_evaluate` without a backend | always attach | attaches only when the engine has a backend (warning otherwise) | Pure-Python DAGs should not fail. |
| `engine.stop()` | not covered | detaches from running runs (resumable); `cancel_runs=True` or `cancel()` to cancel | Graceful restarts. |

Everything else (configuration cascade, three-state visibility, operator parameters, fan-out ids, fan-in dict, two-tier context, priority tiers, non-blocking pause, DagEvaluator tasks and approval gate) is implemented as specified.
