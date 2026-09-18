"""Recovery access to durable mutation intents (internal journal flock only).

P4 drops the workspace execution lease from recovery: the journal's own flock is
the only lock. Running-job guards replace the lease as the safety mechanism.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, _load_data, _load_workspace
from .transaction_journal import (
    TransactionJournalError,
    acknowledge,
    pending_attribution,
    pending_records,
)

PENDING_WARNING_THRESHOLD = 96


class TransactionRecoveryError(RuntimeError):
    """Recovery state is unavailable or has not been acknowledged."""


class TransactionRecoveryRequired(TransactionRecoveryError):
    """A prior workspace mutation must be reviewed before delegation."""


def load_recovery_config() -> Config:
    """Load only the workspace boundary; provider credentials are unnecessary."""
    data = _load_data()
    return Config(api_key="", workspace=_load_workspace(data), allowed_tools=[])


def require_no_pending(config: Config) -> None:
    """Fail closed while the caller owns the workspace execution lease."""
    records = _pending(config)
    if not records:
        return
    identifiers = ", ".join(str(record["transaction_id"]) for record in records)
    raise TransactionRecoveryRequired(
        "workspace has unacknowledged mutation transactions "
        f"({identifiers}); call get_deepseek_recovery, verify the files, then "
        "call acknowledge_deepseek_mutations; DO NOT RETRY"
    )


def query_with_lease(
    config: Config, lock_directory: Path | None = None,
) -> list[dict[str, object]]:
    """List pending records; the journal flock is the only lock (no lease)."""
    return _pending(config)


def acknowledge_with_lease(
    config: Config, transaction_ids: list[str],
    lock_directory: Path | None = None, *, status_resolver=None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Acknowledge records, refusing ids owned by a still-running job."""
    _reject_running(config, transaction_ids, status_resolver)
    try:
        removed = acknowledge(config, transaction_ids)
    except TransactionJournalError:
        raise TransactionRecoveryError(
            "transaction acknowledgement failed safely"
        ) from None
    return removed, _pending(config)


def pending_snapshot(config: Config, status_resolver=None) -> dict[str, object]:
    """Group pending records by job id and annotate live job status."""
    records = _pending(config)
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        job_id = str(record.get("job_id") or "unknown")
        groups.setdefault(job_id, []).append(record)
    by_job = {
        job_id: {
            "status": _job_status(job_id, status_resolver),
            "count": len(items),
            "records": items,
        }
        for job_id, items in groups.items()
    }
    payload: dict[str, object] = {
        "pending": records, "count": len(records), "by_job": by_job,
    }
    _add_warning(payload, len(records))
    return payload


def pending_summary(config: Config) -> dict[str, object]:
    """Non-blocking pending counts by job for admission/listing signals."""
    try:
        groups = pending_attribution(config)
    except (TransactionJournalError, OSError):
        raise TransactionRecoveryError(
            "transaction recovery journal is unavailable; DO NOT RETRY"
        ) from None
    summary: dict[str, object] = {"count": sum(groups.values()), "by_job": groups}
    _add_warning(summary, sum(groups.values()))
    return summary


def _reject_running(config: Config, ids, status_resolver) -> None:
    if status_resolver is None:
        return
    if isinstance(ids, (str, bytes)):
        return
    try:
        requested = {value for value in ids}
    except TypeError:
        return
    running: set[str] = set()
    for record in _pending(config):
        job_id = record.get("job_id")
        if (
            record.get("transaction_id") in requested
            and isinstance(job_id, str) and job_id
            and status_resolver(job_id) == "running"
        ):
            running.add(job_id)
    if running:
        raise TransactionRecoveryError(
            "cannot acknowledge mutation intents while their job is still running "
            f"({', '.join(sorted(running))}); wait for terminal state"
        )


def _add_warning(payload: dict[str, object], count: int) -> None:
    if count >= PENDING_WARNING_THRESHOLD:
        payload["warning"] = (
            f"{count} pending mutation records; acknowledge promptly before the "
            "128-record journal ceiling is reached"
        )


def _job_status(job_id: str, status_resolver) -> str:
    if job_id == "unknown" or job_id.startswith("sync-") or status_resolver is None:
        return "unknown"
    try:
        status = status_resolver(job_id)
    except Exception:
        return "unknown"
    return status if isinstance(status, str) and status else "unknown"


def _pending(config: Config) -> list[dict[str, object]]:
    try:
        return pending_records(config)
    except (TransactionJournalError, OSError):
        raise TransactionRecoveryError(
            "transaction recovery journal is unavailable; DO NOT RETRY"
        ) from None
