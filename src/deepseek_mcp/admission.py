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
from .job_listing import lease_conflict_message, task_preview
from .transaction_journal import TransactionJournalError, pending_total
from .transaction_recovery import (
    TransactionRecoveryError,
    pending_summary,
    require_no_pending,
)

MAX_OVERLAP_LINES = 8


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
    if mode == SAME_WORKSPACE_WRITERS_EXCLUSIVE:
        if current_mode is not None:
            conflict = lease_conflict_message(
                _identity_running(running, identity), current_mode, capability
            )
        recovery = require_no_pending
    else:
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
