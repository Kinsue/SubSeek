"""Bounded text rendering and terminal-outcome helpers for DeepSeek jobs.

Pure helpers so ``job_manager`` and ``server`` stay small and free of circular
imports: this module intentionally never imports ``job_manager``. Records are
read structurally (``getattr``) so any object with the ``JobRecord`` shape
works. Outcome helpers only depend on ``mutation_outcome``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .mutation_outcome import mutation_failure_message, records_from_result

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
TASK_PREVIEW_CHARS = 80
MAX_LIST_LINES = 64
MAX_LIST_ROWS = MAX_LIST_LINES - 2
TABLE_HEADER = "job_id | status | capability | task | started | age_s | tokens"
CANCELLED_ERROR = "DeepSeek job cancelled by parent agent"


@dataclass(frozen=True)
class JobOutcome:
    """Terminal outcome produced by one background agent run."""

    status: str
    result: dict[str, Any] | None
    error: str | None
    preserve_mutation_error: bool = False


def apply_cancelled_outcome(job: Any, outcome: JobOutcome) -> None:
    """Force a cancellation terminal state while preserving mutation evidence."""
    records = records_from_result(outcome.result)
    job.cancel_event.set()
    job.status = "cancelled"
    job.result = None
    if records:
        job.error = mutation_failure_message(records, CANCELLED_ERROR)
    elif outcome.preserve_mutation_error:
        job.error = outcome.error
    else:
        job.error = outcome.error if outcome.status == "cancelled" else CANCELLED_ERROR


def workspace_mismatch_message(
    lease_identity: str | None, requested_identity: str | None
) -> str:
    """Fail-closed message for a cached lease that no longer matches the request."""
    return (
        "workspace changed while DeepSeek jobs are running: lease workspace "
        f"identity {lease_identity} does not match requested workspace identity "
        f"{requested_identity}; wait for running jobs to drain before delegating "
        "to another workspace"
    )


def task_preview(task: object) -> str:
    """Return a single-line, bounded task preview."""
    if not isinstance(task, str):
        return ""
    return " ".join(task.split())[:TASK_PREVIEW_CHARS]


def _preview_of(job: object) -> str:
    preview = getattr(job, "task_preview", "") or getattr(job, "task", "")
    return task_preview(preview)


def _sync_marker(job_id: object) -> str:
    return " (sync)" if str(job_id).startswith("sync-") else ""


def _token_total(job: object) -> int | None:
    result = getattr(job, "result", None)
    if not isinstance(result, dict):
        return None
    tokens = result.get("tokens")
    if not isinstance(tokens, dict):
        return None
    total = tokens.get("total")
    if isinstance(total, bool) or not isinstance(total, int):
        return None
    return total


def row_from_job(job: object) -> dict[str, Any]:
    """Project a job record to plain, JSON-safe listing data."""
    job_id = getattr(job, "job_id", "")
    return {
        "job_id": job_id,
        "sync": str(job_id).startswith("sync-"),
        "status": getattr(job, "status", "queued"),
        "capability": getattr(job, "capability", "coding"),
        "task_preview": _preview_of(job),
        "created_at": getattr(job, "created_at", 0.0) or 0.0,
        "started_at": getattr(job, "started_at", None),
        "finished_at": getattr(job, "finished_at", None),
        "tokens_total": _token_total(job),
    }


def list_rows(records: Iterable[object], status: str = "") -> list[dict[str, Any]]:
    """Filter (exact status match) and sort rows: running first, newest first."""
    rows = [row_from_job(record) for record in records]
    if status:
        rows = [row for row in rows if row["status"] == status]
    rows.sort(key=lambda row: (row["status"] != "running", -row["created_at"]))
    return rows


def _pool_line(job_id: str, job: object) -> str:
    return (
        f"{job_id}{_sync_marker(job_id)} {getattr(job, 'status', 'running')} "
        f"{getattr(job, 'capability', 'coding')} {_preview_of(job)}"
    )


def full_pool_message(limit: int, running: Mapping[str, Any]) -> str:
    """Build the bounded pool-full rejection; one line per running job."""
    header = f"DeepSeek pool is full (max_parallel_agents={limit})"
    entries = list(running.items())
    lines = [_pool_line(job_id, job) for job_id, job in entries[:MAX_LIST_ROWS]]
    if not lines:
        return header
    message = f"{header}; running:\n" + "\n".join(lines)
    remaining = len(entries) - len(lines)
    if remaining > 0:
        message += f"\n[truncated: {remaining} more running jobs]"
    return message


def _started_iso(started_at: object) -> str:
    if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
        return "-"
    return datetime.fromtimestamp(started_at, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _age_seconds(now: float, started_at: object) -> str:
    if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
        return "-"
    return str(max(0, int(now - started_at)))


def _format_row(row: dict[str, Any], now: float) -> str:
    total = row["tokens_total"]
    token_text = f"tokens={total}" if total is not None else "-"
    marker = " (sync)" if row.get("sync") else ""
    return (
        f"{row['job_id']}{marker} | {row['status']} | {row['capability']} | "
        f"{row['task_preview']} | {_started_iso(row['started_at'])} | "
        f"{_age_seconds(now, row['started_at'])} | {token_text}"
    )


def format_jobs_table(
    rows: Iterable[dict[str, Any]], now: float | None = None
) -> str:
    """Render a bounded text table (<= MAX_LIST_LINES lines)."""
    moment = time.time() if now is None else now
    materialized = list(rows)
    lines = [TABLE_HEADER]
    shown = materialized[:MAX_LIST_ROWS]
    lines.extend(_format_row(row, moment) for row in shown)
    remaining = len(materialized) - len(shown)
    if remaining > 0:
        lines.append(f"[truncated: {remaining} more jobs]")
    elif not materialized:
        lines.append("(no jobs)")
    return "\n".join(lines)


def prune_records(jobs: dict[str, Any], max_retained: int) -> None:
    """Drop oldest terminal records once the retained set is exceeded."""
    terminal = [
        job for job in jobs.values()
        if getattr(job, "status", "") in TERMINAL_STATES
        and not getattr(job, "usage_recording", False)
    ]
    if len(terminal) < max_retained:
        return
    terminal.sort(
        key=lambda job: getattr(job, "finished_at", None)
        or getattr(job, "created_at", 0.0)
    )
    for job in terminal[: len(terminal) - max_retained + 1]:
        jobs.pop(job.job_id, None)
