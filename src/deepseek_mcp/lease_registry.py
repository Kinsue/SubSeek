"""Per-workspace execution lease registry for the DeepSeek job pool.

One entry per workspace identity, each with a running-job count and lease mode.
Same-identity conflicts keep the P2 exclusive/shared rules; different
identities are independent, so parallel coding jobs on separate workspaces
(git worktrees) may run concurrently. Cross-process exclusivity per workspace
is unchanged (the underlying flock is per identity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .execution_lock import WorkspaceExecutionLease
from .job_listing import LEASE_EXCLUSIVE, LEASE_SHARED


@dataclass
class _Entry:
    lease: WorkspaceExecutionLease
    mode: str
    count: int = 1


class LeaseRegistry:
    """Track at most one lease per workspace identity with a running count."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def is_idle(self) -> bool:
        return not self._entries

    def running_count(self, identity: str) -> int:
        entry = self._entries.get(identity)
        return entry.count if entry is not None else 0

    def mode(self, identity: str) -> str | None:
        entry = self._entries.get(identity)
        return entry.mode if entry is not None else None

    def lease_for(self, identity: str) -> WorkspaceExecutionLease | None:
        entry = self._entries.get(identity)
        return entry.lease if entry is not None else None

    def acquire(
        self,
        identity: str,
        config: Any,
        shared: bool,
        acquire_lease: Callable[..., WorkspaceExecutionLease],
        recovery_check: Callable[[Any], None],
        release_lease: Callable[[WorkspaceExecutionLease], None],
    ) -> WorkspaceExecutionLease:
        """Admit one job for ``identity``, running the recovery gate each time."""
        entry = self._entries.get(identity)
        if entry is not None:
            recovery_check(config)
            entry.count += 1
            return entry.lease
        lease = acquire_lease(config, shared)
        try:
            recovery_check(config)
        except BaseException:
            release_lease(lease)
            raise
        mode = LEASE_SHARED if shared else LEASE_EXCLUSIVE
        self._entries[identity] = _Entry(lease, mode)
        return lease

    def release(
        self, identity: str, release_lease: Callable[[WorkspaceExecutionLease], None]
    ) -> None:
        """Release one job; drop and release the lease when the count hits zero."""
        entry = self._entries.get(identity)
        if entry is None:
            return
        entry.count -= 1
        if entry.count > 0:
            return
        self._entries.pop(identity, None)
        release_lease(entry.lease)
