"""Bounded text rendering for the DeepSeek agent work list and pool errors.

Pure data-in/text-out helpers so ``job_manager`` and ``server`` stay small and
free of circular imports: this module intentionally never imports
``job_manager``. Records are read structurally (``getattr``) so any object with
the ``JobRecord`` shape works.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
TASK_PREVIEW_CHARS = 80
MAX_LIST_LINES = 64
MAX_LIST_ROWS = MAX_LIST_LINES - 2
TABLE_HEADER = "job_id | status | capability | agent | task | started | age_s | tokens"
LEASE_SHARED = "shared"
LEASE_EXCLUSIVE = "exclusive"


def task_preview(task: object) -> str:
    """Return a single-line, bounded task preview."""
    if not isinstance(task, str):
        return ""
    return " ".join(task.split())[:TASK_PREVIEW_CHARS]


def _preview_of(job: object) -> str:
    preview = getattr(job, "task_preview", "") or getattr(job, "task", "")
    return task_preview(preview)


def _agent_of(job: object) -> str:
    agent = getattr(job, "agent", "")
    if isinstance(agent, str) and agent:
        return agent
    return str(getattr(job, "capability", "coding"))


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
        "agent": _agent_of(job),
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
        f"{getattr(job, 'capability', 'coding')} agent={_agent_of(job)} "
        f"{_preview_of(job)}"
    )


def _listing_lines(running: Mapping[str, Any]) -> tuple[list[str], int]:
    entries = list(running.items())
    lines = [_pool_line(job_id, job) for job_id, job in entries[:MAX_LIST_ROWS]]
    return lines, len(entries) - len(lines)


def full_pool_message(limit: int, running: Mapping[str, Any]) -> str:
    """Build the bounded pool-full rejection; one line per running job."""
    header = f"DeepSeek pool is full (max_parallel_agents={limit})"
    lines, remaining = _listing_lines(running)
    if not lines:
        return header
    message = f"{header}; running:\n" + "\n".join(lines)
    if remaining > 0:
        message += f"\n[truncated: {remaining} more running jobs]"
    return message


def _blocked_message(blocker: str, running: Mapping[str, Any]) -> str:
    lines, remaining = _listing_lines(running)
    message = f"{blocker}; {len(running)} jobs still running:"
    if lines:
        message += "\n" + "\n".join(lines)
    if remaining > 0:
        message += f"\n[truncated: {remaining} more running jobs]"
    return message


def lease_conflict_message(
    running: Mapping[str, Any], lease_mode: str, capability: str
) -> str | None:
    """Return a bounded rejection when the requested lease mode conflicts."""
    if capability == "coding":
        return _blocked_message(
            "coding delegation requires the exclusive workspace lease", running
        )
    if lease_mode == LEASE_EXCLUSIVE:
        return _blocked_message(
            "readonly delegation cannot share the exclusive workspace lease", running
        )
    return None


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
        f"agent={row['agent']} | {row['task_preview']} | "
        f"{_started_iso(row['started_at'])} | "
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
