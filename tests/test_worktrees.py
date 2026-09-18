from __future__ import annotations

import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_mcp.config import Config
from deepseek_mcp.job_manager import DeepSeekJobManager
from deepseek_mcp.worktrees import create_worktree, remove_worktree, worktree_path


def _result(message: str = "done") -> dict:
    return {
        "final_message": message,
        "turns_used": 1,
        "tokens": {"prompt": 1, "completion": 1, "total": 2},
        "tool_calls": 0,
        "duration_seconds": 0.01,
    }


class WorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is not available")
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        self.workspace = self.root / "repo"
        self.workspace.mkdir()
        self._git("init")
        self._git("config", "user.email", "test@example.invalid")
        self._git("config", "user.name", "Test")
        (self.workspace / "README.md").write_text("hello\n", encoding="utf-8")
        self._git("add", "README.md")
        self._git("commit", "-m", "initial")
        self.config = Config("sk-test", self.workspace, allowed_tools=["Read"])

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            ["git", *args], cwd=self.workspace, capture_output=True, text=True
        )
        if completed.returncode != 0:
            self.fail(f"git {args} failed: {completed.stderr.strip()}")
        return completed

    def test_create_worktree_returns_sibling_path_and_branch(self) -> None:
        result = create_worktree("job-one", config=self.config)

        self.assertTrue(result["ok"], result)
        path = Path(result["path"])
        self.assertEqual(result["branch"], "subseek/job-one")
        self.assertTrue(path.is_dir())
        self.assertFalse(path.is_relative_to(self.workspace))
        self.assertTrue(path.is_relative_to(self.root))
        branches = self._git("branch", "--list", "subseek/job-one").stdout
        self.assertIn("subseek/job-one", branches)

    def test_create_duplicate_name_is_rejected(self) -> None:
        self.assertTrue(create_worktree("job-one", config=self.config)["ok"])

        result = create_worktree("job-one", config=self.config)

        self.assertFalse(result["ok"])
        self.assertIn("already exists", result["error"])

    def test_invalid_slug_and_non_repo_are_rejected(self) -> None:
        self.assertFalse(create_worktree("Bad Name", config=self.config)["ok"])

        plain = self.root / "plain"
        plain.mkdir()
        config = Config("sk-test", plain, allowed_tools=["Read"])
        result = create_worktree("job-two", config=config)
        self.assertFalse(result["ok"])
        self.assertIn("not a git repository", result["error"])

    def test_remove_dirty_requires_force(self) -> None:
        created = create_worktree("job-three", config=self.config)
        path = Path(created["path"])
        (path / "scratch.txt").write_text("dirty", encoding="utf-8")

        failed = remove_worktree("job-three", config=self.config)
        self.assertFalse(failed["ok"])

        forced = remove_worktree("job-three", config=self.config, force=True)
        self.assertTrue(forced["ok"], forced)
        self.assertFalse(path.exists())

    def test_remove_refuses_while_a_job_runs_on_the_worktree(self) -> None:
        created = create_worktree("job-four", config=self.config)
        path = Path(created["path"])
        manager = DeepSeekJobManager(lock_directory=self.root / "locks")
        worker_config = Config("sk-test", path, allowed_tools=["Read"])
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("work", "", worker_config)
            self.assertTrue(started.wait(1.0))
            refused = remove_worktree("job-four", manager=manager, config=self.config)
            self.assertFalse(refused["ok"])
            self.assertIn("busy", refused["error"])
            release.set()
            self.assertTrue(manager.wait_for_terminal(job["job_id"], 2.0))

        removed = remove_worktree(
            "job-four", manager=manager, config=self.config, force=True
        )
        self.assertTrue(removed["ok"], removed)

    def test_worktree_path_is_never_inside_the_workspace(self) -> None:
        path = worktree_path(self.workspace, "job-five")
        self.assertEqual(path, (self.root / ".subseek-worktrees" / "job-five").resolve())
        self.assertFalse(path.is_relative_to(self.workspace))


if __name__ == "__main__":
    unittest.main()
