"""Terminal-outcome and lease-conflict vocabulary for DeepSeek jobs.

Pure helpers kept out of ``job_manager`` so the manager stays within its source
budget and module concerns stay separated. This module never imports
``job_manager``; records are handled structurally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mutation_outcome import mutation_failure_message, records_from_result

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
