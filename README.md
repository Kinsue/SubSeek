# deepseek-as-subagent

**English** · [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/PsChina/deepseek-as-subagent?style=social)](https://github.com/PsChina/deepseek-as-subagent)
[![Glama MCP server](https://glama.ai/mcp/servers/PsChina/deepseek-as-subagent/badges/score.svg)](https://glama.ai/mcp/servers/PsChina/deepseek-as-subagent)
[![MCP](https://img.shields.io/badge/protocol-MCP-purple)](https://modelcontextprotocol.io/)
[![Mentioned in Awesome MCP Servers](https://awesome.re/mentioned-badge.svg)](https://github.com/punkpeye/awesome-mcp-servers)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)](https://github.com/PsChina/deepseek-as-subagent)

> Run DeepSeek as a **real sub-agent** inside Claude Code / Codex CLI — not just an LLM endpoint.
> The host agent keeps the main conversation, planning, judgment, and verification.
> DeepSeek gets its own agent loop for execution-heavy work.
> Coding APIs use workspace-scoped writes and bounded trusted-host Bash; separate read-only APIs provide pure file analysis without command execution.

### Full coding delegation

```text
       Claude / Codex (main agent)
         ├─ ordinary coding → delegate_to_deepseek
         └─ coding task whose direction may change
                            → start_deepseek → job_id
                                               ├─ send_deepseek_message(job_id, ...)
                                               ├─ get_deepseek_status(job_id)
                                               ├─ cancel_deepseek(job_id)
                                               └─ get_deepseek_result(job_id)
         ▼
       DeepSeek coding sub-agent
         │  Read / Write / Edit / Bash / Glob / Grep / NotebookEdit
         │  autonomously reads, modifies, runs, and tests in the workspace
         ▼
       Result returns to the host
       Host verifies representative changes / tests
```

Coding Bash runs on the trusted host with `cwd=workspace`; it is bounded and
credential-isolated, but it is not an OS sandbox.

### Read-only analysis delegation

```text
       Claude / Codex (main agent)
         ├─ ordinary read-only analysis → delegate_to_deepseek_readonly
         └─ read-only analysis whose direction may change
                            → start_deepseek_readonly → job_id
                                                        ├─ send_deepseek_message(job_id, ...)
                                                        ├─ get_deepseek_status(job_id)
                                                        ├─ cancel_deepseek(job_id)
                                                        └─ get_deepseek_result(job_id)
         ▼
       DeepSeek read-only sub-agent
         │  Read / Glob / Grep
         │  autonomously reads, searches, reviews, and performs static analysis
         ▼
       Analysis returns to the host
       Host verifies the conclusion
```

## Quick start

```bash
git clone https://github.com/PsChina/deepseek-as-subagent.git
cd deepseek-as-subagent
# Inspect install.sh and requirements.lock, then:
./install.sh
```

Python 3.10–3.12 must already be installed. The installer never pipes a remote
bootstrap script into a shell. It installs the exact, hash-verified dependency
set in `requirements.lock`, registers the MCP server with Claude Code, deploys
protected generation copies of the skill + `/ds` slash command. It does not
modify shell startup files. Helper deployment is best-effort after the core MCP
registration commits; a foreign destination is preserved and reported.

After install, edit `~/.deepseek-mcp/config.json` to paste your DeepSeek API
key on POSIX, or set `DEEPSEEK_API_KEY` on Windows (get one at
[platform.deepseek.com](https://platform.deepseek.com)). Then
run `claude` and try `/ds inspect this workspace and summarize its structure`.

To upgrade, fetch and inspect an explicit tag or commit, then re-run the local
installer. Coding always uses `trusted_host`; read-only APIs need neither Bash
nor Docker/Podman. For
Codex or other MCP clients, see [Install](#install) below.

## How is this different from existing DeepSeek MCP servers?

Most `deepseek-mcp-server` projects expose DeepSeek as a **single LLM call** (`create_chat_completion`, `create_anthropic_message`). The host has to read every file itself and feed content into the prompt — DeepSeek only saves the "thinking" cost, not the "reading/writing" cost.

This project gives DeepSeek **its own agent loop**: tool dispatch, file I/O,
optional command execution for coding, and multi-turn reasoning against the
configured workspace. The host hands off a complete logical unit and gets a
result back. Token savings are end-to-end.

## What's in the box

- **MCP server** (Python, stdio transport)
- **Coding and read-only delegation**: `delegate_to_deepseek` / `delegate_to_deepseek_readonly`
- **Steerable background jobs**: `start_deepseek` / `start_deepseek_readonly` plus shared controls
- **Flash / Pro model routing**: host chooses a stable profile; users control the actual provider model IDs in config
- **Local DeepSeek agent loop** (`agent_loop.py`) with OpenAI-compatible function calling
- **Fixed capability APIs**: coding gets Read / Write / Edit / Bash / Glob / Grep / NotebookEdit; read-only gets Read / Glob / Grep
- **Bash execution**: bounded credential-isolated trusted-host commands through the tool-child boundary
- **Workspace path boundary** for file tools, with outbound symlinks rejected
- **Cross-process execution lease** so at most one coding execution runs against the same workspace at a time; readonly executions may share a lease and run in parallel (POSIX)
- **Crash-safe mutation journal for Write / Edit / NotebookEdit** with recovery query, file verification, and exact acknowledgement before another delegation; trusted-host Bash changes are not journaled
- **Explicit network retry policy** with OpenAI SDK internal retries disabled to avoid nested retry amplification in proxy/TLS-timeout environments
- **Claude Code skill + `/ds` command** for delegation policy and forced delegation

## Compatibility

The four delegation entry points accept one additive optional argument,
`model="flash" | "pro"`. Existing calls that omit it remain valid and now default
to the Flash profile. Background-job and recovery tools remain additive.
Mutation-capable legacy hosts must adopt the recovery query/verify/ack handshake
before starting another delegation; read-only use needs no change.
Clients should not parse health/error text byte-for-byte because diagnostics are now more specific. Provider calls still
use DeepSeek's [OpenAI-compatible Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/).
Local Python module signatures are implementation details rather than a stable
public API.

## Install

### Claude Code (default)

```bash
git clone https://github.com/PsChina/deepseek-as-subagent
cd deepseek-as-subagent
./install.sh
```

Then edit `~/.deepseek-mcp/config.json` on POSIX, or set
`DEEPSEEK_API_KEY` on Windows.

### Codex CLI

```bash
git clone https://github.com/PsChina/deepseek-as-subagent
cd deepseek-as-subagent
bash adapters/codex/install.sh
```

See [adapters/codex/README.md](adapters/codex/README.md) for the Codex-specific install, delegation policy, and background-job workflow.

The Claude and Codex installers build a fresh isolated runtime, validate its
configuration and MCP protocol, and only then switch the host registration.
They keep the active generation plus one previous generation for recovery. Any
manual runtime must stay outside a delegated workspace when file-mutation tools
are enabled; unsafe layouts are rejected at startup.
Both installers serialize install/uninstall transactions. A hard-killed
installer intentionally leaves an empty fail-closed lock that must be removed
only after confirming no installer is running.

### Cursor / Cline / Claude Desktop / other MCP clients

The MCP server itself is client-agnostic. Install `requirements.lock` with
`pip --require-hashes`, install this project with dependency resolution disabled,
then point your client's MCP config at the generated `deepseek-mcp` entrypoint.

## Usage

Choose capability for the task's **entire expected lifecycle** first. Use
read-only only when every expected step is static file analysis with Read, Glob,
and Grep—no command execution. If any step might need Bash, tests, builds,
lint, Git, program execution, dependency work, workspace mutation, or is not
clearly read-only, choose coding.

### Simple delegation

Use a synchronous API when the task can run to completion without mid-flight
intervention. The MCP request remains open until DeepSeek finishes:

- `delegate_to_deepseek(task, context, model="flash")` for coding, Bash, tests,
  or any task that might write the workspace.
- `delegate_to_deepseek_readonly(task, context, model="flash")` for static file
  analysis only.

`model` is optional and accepts only `flash` or `pro`. Omit it for normal work;
select `pro` explicitly for difficult debugging, architecture-level reasoning,
or when Flash has already proved insufficient. The host never passes a provider
model ID directly.

### Steerable background delegation

For longer tasks that may need new instructions or cancellation, choose the
matching background API, then use the same controls for either job type:

```text
start_deepseek(task, context, model="flash") / start_deepseek_readonly(task, context, model="flash") -> job_id
send_deepseek_message(job_id, message)
get_deepseek_status(job_id)
cancel_deepseek(job_id)
get_deepseek_result(job_id)
```

Either `start_*` API returns quickly while the DeepSeek agent continues in a
background worker. Steering changes only the task instruction: it cannot change
the job's fixed tools, Bash availability, or selected model profile. Cancellation
wakes retry backoff and promptly terminates an in-flight provider or local-tool
subprocess.

If a readonly job later needs a command or workspace mutation, cancel or finish
it, then create a new coding job with `start_deepseek`; steering cannot upgrade
the existing readonly job.

If a steering message arrives after DeepSeek has planned tool calls but before a not-yet-executed tool runs, the stale tool call is skipped and DeepSeek re-plans from the new parent instruction.

At most one **coding** DeepSeek execution per canonical workspace may run at a time — including executions started by separate MCP server processes — while readonly executions may share a lease and run in parallel (POSIX). This lease coordinates DeepSeek MCP executions only; it cannot prevent the host agent, IDE, user, or another local process from changing the workspace. While a coding background job is running, the host should steer, query, or cancel that job rather than independently mutate the same workspace, then resume host-side edits after the job reaches a terminal state. Background job IDs and results are session-scoped; collect the result before closing the host session.

### Mutation recovery

Mutations committed through `Write`, `Edit`, and `NotebookEdit` are journaled
before commit. Trusted-host Bash runs outside this transaction journal and may
modify workspace files directly; those changes are not represented by
`get_deepseek_recovery`. After an interrupted coding run in which Bash may have
executed, inspect the workspace independently before continuing or retrying work.
After a result reports journaled mutations—or after cancellation, disconnection,
or MCP restart—run:

```text
get_deepseek_recovery()
# verify every reported file
acknowledge_deepseek_mutations(transaction_ids)
```

New delegation fails closed until the exact reviewed IDs are acknowledged.
Recovery works without a valid DeepSeek API credential and never deletes or
rolls back workspace files.

### Claude Code helpers

- `delegate_to_deepseek` / `delegate_to_deepseek_readonly` — Claude selects the
  matching fixed capability and Flash/Pro profile
- `/ds <task>` — force synchronous coding delegation
- `DEEPSEEK_MODE=off claude` — start one session with DeepSeek disabled

## When delegation actually saves money

The delegation decision should happen **before the host reads large amounts of source**. If the host reads first and then delegates, both agents pay the repository-reading cost.

Sweet spot:
- ✅ Multi-file implementation / mechanical refactors / test generation
- ✅ Large data + simple processing (log scan, file conversion, ETL)
- ✅ Tasks that may benefit from a cheap independent execution loop
- ❌ Tiny edits where orchestration overhead dominates
- ❌ Cross-domain architecture / ambiguous root-cause analysis / security-sensitive judgment

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  Claude Code / Codex CLI (main agent)                           │
│    ↓ stdio (MCP protocol, local)                                │
│  deepseek-as-subagent (Python MCP process)                      │
│    ├─ synchronous delegate                                      │
│    └─ steerable background job manager                          │
│         ↓                                                       │
│       DeepSeek agent loop + selected fixed-capability tools     │
│    ↓ HTTPS                                                      │
│  api.deepseek.com                                               │
└─────────────────────────────────────────────────────────────────┘
```

No third-party proxy or cloud relay is introduced by this project. Delegated prompts and tool/file outputs selected by the agent are sent to the configured DeepSeek-compatible API, so only delegate data that endpoint is permitted to receive.

## Configuration

`~/.deepseek-mcp/config.json`:

```json
{
  "api_key": "sk-...",
  "flash": "deepseek-v4-flash",
  "flash_reasoning_effort": "high",
  "pro": "deepseek-v4-pro",
  "pro_reasoning_effort": "high",
  "_reasoning_effort_options": ["none", "low", "high", "max"],
  "max_turns": 50,
  "max_run_seconds": 18000,
  "allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "NotebookEdit"]
}
```

Optional budget keys bound tokens, tool calls, and mutation bytes per run;
see [Budgets](#budgets).

`flash` and `pro` are the provider model IDs behind the two stable MCP routing
profiles. You can change these strings when DeepSeek publishes a new model
revision, or when a compatible endpoint uses different model names, without
changing how Claude/Codex calls the MCP tools. The public tool argument remains
only `model="flash"` or `model="pro"`.

`flash_reasoning_effort` and `pro_reasoning_effort` accept `none`, `low`, `high`,
or `max`. `none` disables thinking; the other values explicitly enable thinking
at that effort. `_reasoning_effort_options` is only an in-file hint and is ignored
at runtime. If an effort field is absent, deepseek-mcp leaves thinking controls
unspecified for that slot so the provider's existing default applies; this keeps
older configs and OpenAI-compatible gateways compatible. New installer-generated
configs explicitly set both slots to `high`.

For upgrade compatibility, a legacy single `model` field is still accepted when
`flash` and `pro` are absent; its value is used for both slots. Do not combine
legacy `model` with the new `flash` / `pro` fields.

`allowed_tools` is retained for configuration compatibility and validation. It
does not select capabilities for a delegation: each MCP API applies its own
fixed profile after configuration is loaded.

`max_run_seconds` is the wall-clock limit for one delegated run. Its default is
18,000 seconds (5 hours), it may be increased explicitly, and its absolute
accepted maximum is 172,800 seconds (48 hours). Individual provider requests
remain bounded to 180 seconds within that run budget.
For synchronous delegation, the MCP client's tool timeout must be at least the
configured run limit plus cleanup grace; Codex installs with an 18,060-second
default (five hours plus 60 seconds).

### Budgets

Six budget keys bound what one delegated run may consume. All are optional;
defaults match the previous built-in limits, and each accepts any positive
integer. There are no client-side hard caps — values beyond the active
model's limits pass local validation and fail at the provider with a 4xx;
the provider is the final authority.

| key | default | bounds |
|---|---|---|
| `max_total_tokens_per_run` | 1,000,000 | tokens one run may consume (latest prompt + cumulative completion) |
| `max_history_tokens` | 98,304 | conversation history sent to the provider, in tokens |
| `max_output_tokens_per_request` | 16,384 | completion tokens per provider request |
| `max_tool_calls_per_turn` | 32 | tool calls in one turn |
| `max_tool_calls_per_run` | 128 | tool calls across one run |
| `max_mutation_bytes_per_run` | 67,108,864 (64 MiB) | bytes written through mutating tools |

Token accounting is usage-based: the run budget counts the provider-reported
prompt tokens of the latest request plus cumulative completion tokens, so long
sessions no longer trip the budget early. Before the first provider response —
and for the history check until the next response arrives — sizes are
conservatively estimated from message bytes (~4 bytes/token), so dense-content
tasks near the caps may still be rejected by the provider rather than locally.
A 12 MiB internal guard on the encoded conversation history remains fixed.

Cross-key rules: `max_output_tokens_per_request` must not exceed
`max_total_tokens_per_run`, and `max_tool_calls_per_turn` must not exceed
`max_tool_calls_per_run`.

`max_input_token` and `max_output_token` are accepted as aliases for
`max_history_tokens` and `max_output_tokens_per_request` respectively; an
alias and its canonical key must not both be set.

Each key can be overridden at runtime with a `DEEPSEEK_MAX_*` environment
variable of the same name (for example `DEEPSEEK_MAX_TOOL_CALLS_PER_RUN=256`);
an empty variable counts as unset, and the environment wins over the file.
`max_turns` and `max_run_seconds` remain file-only.

Budget errors are transparent: they report the used amount, the configured
limit, and the config key involved; they reach the host agent instead of a
generic failure string, and the effective budget set is logged once at startup.

### Parallel agents

Multiple sub-agent jobs run concurrently inside one server process. The pool
size is set by `max_parallel_agents` (default 16, any positive integer; env
`DEEPSEEK_MAX_PARALLEL_AGENTS`, empty counts as unset, environment wins over
the file). The limit applies at admission: when the pool is full, a new job
is rejected immediately with a listing of the running jobs (id, status,
capability, task preview) instead of queuing. Running jobs are never evicted,
so a lowered limit takes effect as the pool drains.

`list_deepseek_jobs(status="")` returns the agent work list — running jobs
first, then newest first — with job id, status, capability, task preview,
age, and total tokens for finished jobs. Entries marked `(sync)` are
in-flight synchronous delegations: they appear in listings but do not accept
per-job messages.

Workspace access is lease-governed per workspace. By default
(`same_workspace_writers: "allow"`) every job — coding included — takes a
shared cross-process lease, so multiple agents, in this or another process,
can work on one workspace concurrently: the mainstream model. Safety comes
from advisory prompts (the coding agent is told to scope edits to its task
and to re-read and re-apply when an edit conflicts with another writer),
per-job mutation attribution, and git. Set `same_workspace_writers:
"exclusive"` to restore strict single-writer semantics: a coding job then
requires the workspace's exclusive lease, is admitted only when no other
job runs against that workspace, and admissions are gated on pending
mutation recovery. Worktrees remain the guaranteed-isolation path for
parallel writers. Shared leases are POSIX-only; on Windows every delegation
takes the exclusive lease.

Mutation journaling carries per-job attribution: `get_deepseek_recovery`
groups pending records by job with each job's current status (acknowledging
a still-running job's records is refused), admission responses and
`list_deepseek_jobs` surface pending-recovery counts, and a warning appears
at 96+ pending records. Acknowledge promptly in allow mode.

#### Agent catalog

Agents are data. The four delegation tools accept an optional `agent`
argument (empty selects the API's built-in: `coding` for the coding APIs,
`readonly` for the read-only APIs) and an optional `workspace` argument
(empty selects the configured default) so a job can target another workspace
— or a git worktree — directly.

Custom agents extend the built-ins via the `agents` config key:

```json
"agents": [
  {
    "id": "reviewer",
    "description": "Reviews changes without editing",
    "capability": "readonly",
    "model": "deepseek-v4-pro",
    "allowed_tools": ["Read", "Glob", "Grep"]
  }
]
```

`id` is a lowercase slug, `capability` is `coding` or `readonly`,
`allowed_tools` defaults to the capability's toolset, and an agent whose
tools include mutation tools must declare `coding` capability. Model
precedence: the explicit `model` tool argument wins, then the agent's model
(applied only when the tool argument is left at its default), then the
config default. Selecting an agent whose capability contradicts the API is
rejected naming the agent, its capability, and the API.

#### Worktrees

`create_deepseek_worktree(name)` creates a git worktree of the configured
workspace at `<workspace>/../.subseek-worktrees/<name>` on branch
`subseek/<name>` and returns its absolute path. Pass that path as a job's
`workspace` argument to run coding agents in parallel on isolated trees;
merging stays a host-side git concern. `remove_deepseek_worktree(name,
force=False)` removes the worktree (refusing dirty trees unless `force`) and
prunes. Removal is refused while any DeepSeek execution — in this or another
process — is running against that worktree.

**Workspace root** auto-follows the directory where you launch the host client.
To lock it to a fixed path regardless of cwd, add `"workspace": "/abs/path"`
to the config. It is the file-tool path boundary and the working directory for
coding Bash; it is not an OS sandbox for trusted-host Bash.

`delegate_to_deepseek` and `start_deepseek` always use full coding tools and
bounded `trusted_host` Bash. `delegate_to_deepseek_readonly` and
`start_deepseek_readonly` always use only Read/Glob/Grep and never expose Bash.
The selected API—not a task argument or model request—freezes that capability
for the job lifetime.
See [SECURITY.md](SECURITY.md) for boundaries and platform limitations.

Override at runtime with env vars: `DEEPSEEK_API_KEY`, `DEEPSEEK_WORKSPACE`, `DEEPSEEK_MODE=off`, plus the per-key `DEEPSEEK_MAX_*` budget overrides (see [Budgets](#budgets)).

## Uninstall

Claude Code: `./uninstall.sh`. Codex: `bash adapters/codex/uninstall.sh`.

Each uninstaller removes only its owned host registration. Neither deletes your
projects, DeepSeek config/API key, logs, or account.

## License

MIT