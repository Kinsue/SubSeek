# Multi-Agent Parallelism Design

Status: accepted (owner decisions 2026-09-18) · Phases: P1 scheduler + work
list → P2 read/write lock split → P3 agent catalog + worktree isolation →
P4 same-workspace parallel writers (mainstream alignment).

## P4 — Same-workspace parallel writers (owner decision, opencode-aligned)

Owner choice: align with the mainstream model (opencode / Claude Code
subagents) — same-workspace concurrent CODING jobs are allowed; safety comes
from advisory prompts + git, not mutual exclusion. The strict P2/P3 behavior
remains available as an opt-in.

- **Lease model**: coding jobs take the SHARED lease per workspace identity
  (same as readonly). Exclusive admission disappears; cross-process parallel
  writers on one workspace are admitted. The lease survives as the liveness
  anchor and for the worktree-remove EX probe (still refuses while any job
  holds SH on that identity). The P3a mutation-tools-force-EX hardening
  applies ONLY in exclusive mode — in allow mode every coding toolset
  contains mutation tools, so keeping the clause unconditionally would
  nullify the phase.
- **Recovery semantics** (amended per design gate):
  - Journal records gain job attribution: `job_id`, `agent`, `started_at`
    (pid optional). The job_id reaches the tool-child journal write path via
    the tool settings payload (the same channel as `delegation_capability`),
    including the sync path.
  - Admission is no longer hard-blocked by pending mutation transactions.
  - `get_deepseek_recovery` runs under the internal journal flock only (no
    workspace-lease acquisition), groups pending records by job, and
    annotates each group with the job's current status resolved from the
    live manager; unresolvable ids (restarted server) and `sync-*` ids
    annotate as terminal-unknown.
  - `acknowledge_deepseek_mutations` runs under the internal journal flock
    only and REJECTS ids belonging to still-running jobs — acknowledging a
    live writer's intents races its outcome report. This running-job guard
    replaces the lease as the safety mechanism.
- **Pending-recovery signals at admission/listing level** (not only in
  results): the coding admission response on an identity with pending
  records carries a non-blocking pending-recovery summary (counts by job);
  `list_deepseek_jobs` badges identities with unacknowledged records. A
  crashed background job whose result is never polled must still be visible.
- **Advisory layer**: the coding child system prompt gains a concurrency
  advisory including edit-conflict retry semantics — when an Edit fails its
  expected-identity check because another writer changed the file, re-read
  and re-apply; never force or loop blindly. The overlap notice is
  admission-response-only (transient, never stored on JobRecord) and
  carries the existing job's id, agent, and task preview plus the worktree
  suggestion.
- **Config**: `same_workspace_writers: "allow" (default) | "exclusive"`.
  `"exclusive"` restores the P2/P3 single-writer-per-workspace behavior (EX
  lease for coding, pending-recovery admission gate, mutation-tools
  hardening). Mode is per process; flock arbitrates cross-mode contention
  fail-closed; an exclusive-mode process may starve under continuous SH
  churn (documented best-effort, consistent with reject-with-listing).
- **Journal capacity**: the 128-records-per-identity ceiling is reached much
  faster with N writers and host-delayed acknowledgement; recovery and
  admission payloads warn at >=96 pending records and docs state the
  host's obligation to acknowledge promptly in allow mode.
- **Windows**: allow mode grants in-process parallelism only (exclusive-open
  lease semantics); cross-process sharing stays POSIX-only.
- **Unchanged**: pool/capacity, budgets, agent catalog, worktree tools,
  readonly semantics, list tool, readonly-agents-cannot-mutate rule.
- **Release note obligation**: the default flip changes behavior for hosts
  that relied on JobBusy-induced serialization of fanned-out coding tasks —
  flag loudly in release notes.


## Goals

1. Run multiple DeepSeek sub-agent jobs concurrently inside one MCP server
   process (bounded, configurable pool).
2. Expose an agent work list: enumerate jobs, states, usage — hosts can fan
   out and reconcile.
3. Preserve the repo's safety posture: cross-process workspace single-flight
   (P1), read/write lock split (P2), filesystem isolation for parallel
   writers via git worktrees (P3).

Non-goals: unbounded breadth, background-job completion push (MCP host
notification support is uneven; pull stays), nested sub-agents spawned by
sub-agents.

## Current state (why only single-agent today)

- `DeepSeekJobManager` is a hard single-slot scheduler: `_active_job_id` +
  `_sync_active` under `_ensure_slot_available_locked`
  (`src/deepseek_mcp/job_manager.py:462-473`); a second `start_deepseek`
  returns `JobBusy` immediately.
- Workspace execution lease is exclusive per workspace, cross-process
  (`src/deepseek_mcp/execution_lock.py:302-305`, fcntl flock keyed by
  workspace dev:inode), held for the whole delegation; recovery tools
  contend for the same lease.
- MCP surface is per-job verbs only (`server.py`): no list/enumeration tool.
- Workspace binds globally (config/env/cwd), not per job.
- Already per-job (reusable): job thread + `JobRecord` (cancel, steering
  mailbox), per-request provider subprocess, per-call tool subprocess with
  `start_new_session`, per-call `transaction_id`, per-run budgets
  (`token_budget.py`, `budget_limits.py`).

## Patterns borrowed from opencode (anomalyco/opencode)

- Per-unit serialization, cross-unit concurrency (their
  `SessionRunCoordinator`; our per-job threads under a bounded pool).
- Work list / job board as first-class read surface; keep pull-based status
  (they removed `task_status` for push injection — TUI-centric; MCP hosts
  vary).
- Agent catalog as data with model-facing descriptions (P3).
- Cap what must be capped (pool size, depth=1); breadth bounded by explicit
  config, not hardcoded.
- Isolation via worktrees for parallel writers; recovery over mutual
  exclusion only where filesystem separation exists (P3). We deliberately do
  NOT adopt their no-lock same-tree concurrency.

## Owner decisions

| Decision | Choice |
|---|---|
| Pool-full policy | Reject immediately; error lists running jobs (id/status/capability/task preview). No queue. |
| Same-workspace parallel writers | P3: forbidden by default. P4 (supersedes): allowed by default (shared lease, advisory prompts + git); `same_workspace_writers=exclusive` restores single-writer; worktrees remain the guaranteed-isolation path. |
| Concurrency | `max_parallel_agents` default **16**, positive integer, **no upper bound** (provider is the authority for its own limits). |

## Architecture

### P1 — Scheduler pool + work list

- `job_manager`: `_active_job_id`/`_sync_active` → `_running: dict[job_id ->
  JobRecord]`. `start()` and `run_sync()` each occupy one pool slot; reject
  when `len(_running) >= max_parallel_agents` with a `JobBusy` message
  embedding the running-jobs listing.
- **Lease refcounting**: the workspace lease is what makes this safe. Today
  every job acquires it exclusively. In P1 the manager acquires the lease
  once on first concurrent job and releases it when the pool drains to zero
  (refcount under the manager lock). Cross-process exclusivity is unchanged;
  in-process parallelism becomes possible without weakening it.
- Config: new key `max_parallel_agents` (default 16, positive int, no cap) +
  env `DEEPSEEK_MAX_PARALLEL_AGENTS` (env wins over file; empty = unset).
- New MCP tool `list_deepseek_jobs(status: str = "")`: compact bounded table —
  `job_id | status | capability | model | task[:80] | started | age`; terminal
  jobs include total tokens from the result payload when present.

### P2 — Read/write lock split

- `execution_lock`: readonly jobs take `LOCK_SH`, coding jobs take `LOCK_EX`.
  Multiple readonly agents analyze one workspace concurrently; any coding job
  still excludes everything (per workspace, cross-process).
- Sync readonly delegations enter the pool like any job.

### P3 — Agent catalog + per-job workspace/worktree

- `agents` config section: `{id, description, model, allowed_tools,
  capability}`; built-ins `coding`/`readonly` remain defaults. `start_*` /
  `delegate_*` gain an `agent` parameter; tool descriptions embed the catalog
  (opencode selection pattern).
- Per-job `workspace` argument; optional `deepseek_worktree_create/remove`
  helpers creating `subseek/<job_id>` branches for parallel writers; merge
  stays a host/git concern.
- Job records gain `agent` id + per-job workspace identity; transaction
  journal already scopes by workspace identity.

## Test strategy

- P1: N concurrent jobs complete (fake provider/tool paths as in existing
  tests); N+1th rejected with listing; lease acquired once, released at
  drain-zero; sync+async mixed occupancy; `max_parallel_agents` config/env
  validation (default/override/positive-int rejection); `list_deepseek_jobs`
  shape under running/terminal mix.
- P2: two readonly jobs overlap; readonly+coding conflict cross-process;
  lease modes verified via lock introspection.
- P3: catalog selection, per-job workspace isolation, worktree
  create/remove lifecycle.

## References

- Reconnaissance: `.slim/deepwork/multi-agent-parallel.md` (session state).
- opencode: `packages/core/src/background-job.ts`,
  `packages/core/src/session/run-coordinator.ts`,
  `packages/opencode/src/tool/task.ts`, `agent/subagent-permissions.ts`,
  `worktree/index.ts` (docs: opencode.ai/docs/agents, /docs/server).
