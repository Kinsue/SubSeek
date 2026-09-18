"""Git worktree helpers for parallel coding on isolated sibling workspaces.

Same-workspace parallel writes stay forbidden; parallel coders get their own
worktree instead. Every git command runs without a shell, with a timeout and
bounded, trimmed output.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .execution_lock import WorkspaceLockError, workspace_identity
from .safety import is_unsafe_workspace_root
from .transaction_recovery import load_recovery_config

WORKTREE_SLUG = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
WORKTREE_PARENT = ".subseek-worktrees"
GIT_TIMEOUT_SECONDS = 60
MAX_GIT_STDERR = 2000


class WorktreeError(RuntimeError):
    """A worktree helper could not complete safely."""


def validate_name(name: object) -> str:
    if not isinstance(name, str) or not WORKTREE_SLUG.match(name):
        raise WorktreeError("name must be a slug matching ^[a-z][a-z0-9-]{1,31}$")
    return name


def worktree_path(workspace: Path, name: str) -> Path:
    """Resolve the sibling worktree path, refusing anything inside the workspace."""
    root = (workspace.parent / WORKTREE_PARENT).resolve()
    candidate = (root / name).resolve()
    if candidate == workspace or workspace in candidate.parents:
        raise WorktreeError("worktree path must not be inside the configured workspace")
    return candidate


def _run_git(args: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(cwd), shell=False, check=False,
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        raise WorktreeError("git is not available") from None
    except subprocess.TimeoutExpired:
        raise WorktreeError("git command timed out") from None
    except OSError as error:
        raise WorktreeError(f"git could not run: {error}") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        detail = detail[-MAX_GIT_STDERR:].replace("\n", " ")
        raise WorktreeError(f"git {args[0]} failed: {detail or 'unknown git error'}")
    return completed.stdout


def _require_git_repo(workspace: Path) -> None:
    output = _run_git(["rev-parse", "--is-inside-work-tree"], workspace)
    if output.strip() != "true":
        raise WorktreeError("configured workspace is not a git repository")


def create_worktree(name: object, manager: Any = None, config: Any = None) -> dict:
    """Create ``<workspace>/../.subseek-worktrees/<name>`` on ``subseek/<name>``."""
    try:
        slug = validate_name(name)
        active = config or load_recovery_config()
        workspace = active.workspace
        _require_git_repo(workspace)
        path = worktree_path(workspace, slug)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if is_unsafe_workspace_root(path.parent):
            raise WorktreeError(
                "worktree directory is a protected or overly broad host path"
            )
        if path.exists():
            raise WorktreeError(f"worktree already exists: {path}")
        branch = f"subseek/{slug}"
        _run_git(["worktree", "add", "-b", branch, str(path)], workspace)
        return {"ok": True, "path": str(path), "branch": branch}
    except WorktreeError as error:
        return {"ok": False, "error": str(error)}
    except (OSError, RuntimeError) as error:
        return {"ok": False, "error": f"worktree create failed: {error}"}


def remove_worktree(
    name: object, manager: Any = None, config: Any = None, force: bool = False
) -> dict:
    """Remove a DeepSeek-created worktree, refusing while a job holds it."""
    try:
        slug = validate_name(name)
        active = config or load_recovery_config()
        workspace = active.workspace
        path = worktree_path(workspace, slug)
        if not path.exists():
            raise WorktreeError(f"worktree does not exist: {path}")
        if manager is not None:
            try:
                identity = workspace_identity(path).hex()
            except (OSError, WorkspaceLockError):
                raise WorktreeError("worktree path cannot be identified") from None
            if manager.identity_is_busy(identity):
                raise WorktreeError(
                    f"workspace is busy with a running DeepSeek job: {path}"
                )
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))
        _run_git(args, workspace)
        _run_git(["worktree", "prune"], workspace)
        return {"ok": True, "path": str(path)}
    except WorktreeError as error:
        return {"ok": False, "error": str(error)}
    except (OSError, RuntimeError) as error:
        return {"ok": False, "error": f"worktree remove failed: {error}"}


def build_tools(manager: Any) -> tuple:
    """Build the two MCP worktree tools bound to one job manager."""

    def create_deepseek_worktree(name: str) -> str:
        """Create a git worktree sibling for a parallel coding job.

        Returns JSON {ok, path, branch}. Pass the absolute path as a delegation
        `workspace` argument and merge the branch with git when done.
        """
        return json.dumps(create_worktree(name, manager))

    def remove_deepseek_worktree(name: str, force: bool = False) -> str:
        """Remove a DeepSeek-created git worktree (force discards local changes)."""
        return json.dumps(remove_worktree(name, manager, force=force))

    return create_deepseek_worktree, remove_deepseek_worktree
