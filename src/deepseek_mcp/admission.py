"""Admission policy, overlap notices, and pending-recovery signals (P4).

Keep out of ``job_manager`` so the manager stays within its source budget, and
never import ``job_manager`` from here (that would be circular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .agent_catalog import MUTATION_TOOLS
from .budget_limits import (
    DEFAULT_SAME_WORKSPACE_WRITERS,
    SAME_WORKSPACE_WRITERS_EXCLUSIVE,
)
from .job_listing import LEASE_EXCLUSIVE, lease_conflict_message, task_preview
from .transaction_journal import TransactionJournalError, pending_total
from .transaction_recovery import (
    TransactionRecoveryError,
    pending_summary,
    require_no_pending,
)

MAX_OVERLAP_LINES = 8
MAX_SIGNAL_CHARS = 1200


@dataclass(frozen=True)
class AdmissionPlan:
    """Resolved per-admission decision handed back to the job manager."""

    shared: bool
    recovery_check: Any
    conflict: str | None
    extras: dict[str, Any]


def writer_mode(config: object) -> str:
    mode = getattr(config, "same_workspace_writers", DEFAULT_SAME_WORKSPACE_WRITERS)
    if mode == SAME_WORKSPACE_WRITERS_EXCLUSIVE:
        return SAME_WORKSPACE_WRITERS_EXCLUSIVE
    return DEFAULT_SAME_WORKSPACE_WRITERS


def _shared(mode: str, capability: str, allowed_tools: Any) -> bool:
    if mode != SAME_WORKSPACE_WRITERS_EXCLUSIVE:
        return True
    return capability != "coding" and not MUTATION_TOOLS.intersection(allowed_tools)


def _identity_running(running: Mapping[str, Any], identity: str) -> dict[str, Any]:
    return {
        job_id: job for job_id, job in running.items()
        if getattr(job, "workspace_identity", "") == identity
    }


def overlap_notice(running: Mapping[str, Any], identity: str, capability: str) -> str | None:
    """Bounded notice when a coding job joins an identity already coding."""
    if capability != "coding":
        return None
    existing = [
        job for job in _identity_running(running, identity).values()
        if getattr(job, "capability", "") == "coding"
    ]
    if not existing:
        return None
    lines = [
        f"{job.job_id} agent={getattr(job, 'agent', '') or 'coding'} "
        f"{task_preview(getattr(job, 'task_preview', '') or getattr(job, 'task', ''))}"
        for job in existing[:MAX_OVERLAP_LINES]
    ]
    remaining = len(existing) - len(lines)
    if remaining > 0:
        lines.append(f"[truncated: {remaining} more]")
    return (
        "other coding agents are already editing this workspace; overlap notice:\n"
        + "\n".join(lines)
        + "\nuse disjoint scopes or create_deepseek_worktree for isolation"
    )


def _pending_signal(config: Any) -> dict[str, Any] | None:
    try:
        summary = pending_summary(config)
    except (TransactionRecoveryError, TransactionJournalError, OSError):
        return None
    return summary if summary.get("count") else None


def admission_plan(config: Any, current_mode: str | None, running: Mapping[str, Any]) -> AdmissionPlan:
    """Resolve shared/recovery/conflict/extras for one admission."""
    identity = getattr(config, "expected_workspace_identity", "") or ""
    mode = writer_mode(config)
    capability = getattr(config, "delegation_capability", "coding")
    conflict = None
    recovery = None
    extras: dict[str, Any] = {}
    # An EX entry must stay single-writer even if the incoming call is allow-mode.
    if current_mode is not None and (
        mode == SAME_WORKSPACE_WRITERS_EXCLUSIVE or current_mode == LEASE_EXCLUSIVE
    ):
        conflict = lease_conflict_message(
            _identity_running(running, identity), current_mode, capability
        )
    if mode == SAME_WORKSPACE_WRITERS_EXCLUSIVE:
        recovery = require_no_pending
    else:
        # Journal reads here are bounded (<=128 records), matching the old
        # exclusive gate's cost; cache per identity at drain transitions if it
        # ever shows up in profiles.
        notice = overlap_notice(running, identity, capability)
        if notice is not None:
            extras["overlap_notice"] = notice
        summary = _pending_signal(config)
        if summary is not None:
            extras["pending_recovery"] = summary
    return AdmissionPlan(
        _shared(mode, capability, getattr(config, "allowed_tools", ())),
        recovery,
        conflict,
        extras,
    )


def format_sync_result(result: dict[str, Any]) -> str:
    """Format a sync delegation result plus bounded admission signals."""
    text = (
        f"{result['final_message']}\n\n"
        f"---\n"
        f"[deepseek-mcp] {result['turns_used']} turns, "
        f"{result['tool_calls']} tool calls, "
        f"{result['tokens']['total']} tokens, "
        f"{result['duration_seconds']}s"
    )
    notice = result.get("overlap_notice")
    if notice:
        text += f"\n\n[deepseek-mcp] {str(notice)[:MAX_SIGNAL_CHARS]}"
    pending = result.get("pending_recovery")
    if isinstance(pending, dict) and pending.get("count"):
        text += (
            f"\n\n[deepseek-mcp] {pending['count']} pending mutation records "
            "await recovery (see get_deepseek_recovery)"
        )
    return text


def pending_badges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach bounded per-identity pending counts to listing rows (fail-open)."""
    counts: dict[str, int] = {}
    for row in rows:
        identity = row.get("workspace_identity")
        if isinstance(identity, str) and identity and identity not in counts:
            counts[identity] = _safe_pending_total(identity)
    for row in rows:
        identity = row.get("workspace_identity")
        count = counts.get(identity) if isinstance(identity, str) else None
        if count:
            row["pending_recovery"] = count
    return rows


def _safe_pending_total(identity: str) -> int:
    try:
        return pending_total(identity)
    except Exception:
        return 0
