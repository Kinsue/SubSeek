from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_mcp.agent_loop import AgentLoopCancelled
from deepseek_mcp.config import DEFAULT_ALLOWED_TOOLS, Config
from deepseek_mcp.job_listing import (
    LEASE_EXCLUSIVE,
    LEASE_SHARED,
    MAX_LIST_LINES,
    MAX_LIST_ROWS,
    TABLE_HEADER,
    format_jobs_table,
    full_pool_message,
)
from deepseek_mcp.job_manager import (
    MAX_CONTEXT_BYTES,
    MAX_COMBINED_TASK_BYTES,
    MAX_QUEUED_MESSAGES,
    MAX_QUEUED_MESSAGE_BYTES,
    MAX_RETAINED_JOBS,
    MAX_STEERING_MESSAGE_BYTES,
    MAX_TASK_BYTES,
    DeepSeekJobManager,
    JobBusy,
    JobError,
    JobRecord,
)
from deepseek_mcp.lease_registry import LeaseRegistry
from deepseek_mcp.transaction_recovery import TransactionRecoveryError


def _result(message: str = "done") -> dict:
    return {
        "final_message": message,
        "turns_used": 1,
        "tokens": {"prompt": 1, "completion": 1, "total": 2},
        "tool_calls": 0,
        "duration_seconds": 0.01,
    }


class JobManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        self.lock_directory = root / "locks"

    def _manager(self) -> DeepSeekJobManager:
        return DeepSeekJobManager(lock_directory=self.lock_directory)

    def _config(self, workspace: Path | None = None, **overrides) -> Config:
        capability = overrides.get("delegation_capability", "coding")
        overrides.setdefault(
            "allowed_tools",
            ["Read", "Glob", "Grep"] if capability == "readonly"
            else list(DEFAULT_ALLOWED_TOOLS),
        )
        return Config(
            api_key="sk-test", workspace=workspace or self.workspace, **overrides
        )

    def _assert_terminal(self, manager: DeepSeekJobManager, job_id: str) -> dict:
        self.assertTrue(manager.wait_for_terminal(job_id, 2.0))
        return manager.status(job_id)

    def _mode(self, manager: DeepSeekJobManager, config: Config) -> str | None:
        return manager._leases.mode(config.expected_workspace_identity or "")

    def _seed_terminal_jobs(self, manager: DeepSeekJobManager) -> list[str]:
        job_ids = []
        for index in range(MAX_RETAINED_JOBS):
            job_id = f"retained-{index}"
            manager._jobs[job_id] = JobRecord(
                job_id=job_id,
                task="",
                context="",
                task_length=index,
                status="completed",
                finished_at=float(index),
                result=_result(job_id),
            )
            job_ids.append(job_id)
        return job_ids

    def _assert_results_retained(
        self,
        manager: DeepSeekJobManager,
        job_ids: list[str],
    ) -> None:
        for job_id in job_ids:
            payload = manager.result(job_id)
            self.assertTrue(payload["ready"], job_id)
            self.assertEqual(payload["result"]["final_message"], job_id)

    def test_background_job_accepts_steering_and_completes(self) -> None:
        manager = self._manager()
        started = threading.Event()
        allow_poll = threading.Event()
        got_message = threading.Event()
        captured: list[str] = []

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if not allow_poll.wait(2.0):
                raise AssertionError("test did not release steering poll")
            captured.extend(kwargs["control_poll"]())
            got_message.set()
            kwargs["control_finalize"]()
            return _result()

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("test task", "", self._config())
            self.assertTrue(started.wait(1.0))
            queued = manager.send_message(job["job_id"], "change direction")
            self.assertTrue(queued["message_queued"])
            allow_poll.set()
            self.assertTrue(got_message.wait(1.0))
            self._assert_terminal(manager, job["job_id"])

        self.assertEqual(captured, ["change direction"])
        result = manager.result(job["job_id"])
        self.assertTrue(result["ready"])
        self.assertEqual(result["result"]["final_message"], "done")

        usage = manager.claim_usage_record(job["job_id"])
        self.assertIsNotNone(usage)
        assert usage is not None
        task_length, usage_result = usage
        self.assertEqual(task_length, len("test task"))
        self.assertEqual(usage_result["tokens"]["total"], 2)
        self.assertIsNone(manager.claim_usage_record(job["job_id"]))
        manager.finish_usage_record(job["job_id"], False)
        self.assertIsNotNone(manager.claim_usage_record(job["job_id"]))
        manager.finish_usage_record(job["job_id"], True)
        self.assertIsNone(manager.claim_usage_record(job["job_id"]))

    def test_cancel_is_cooperative_and_reaches_cancelled(self) -> None:
        manager = self._manager()
        started = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if kwargs["cancel_signal"].wait(2.0):
                raise AgentLoopCancelled("cancelled in test")
            return _result("unexpected")

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("long task", "", self._config())
            self.assertTrue(started.wait(1.0))
            manager.send_message(job["job_id"], "private queued direction")
            response = manager.cancel(job["job_id"])
            self.assertTrue(response["cancel_accepted"])
            self.assertTrue(response["cancel_requested"])
            status = self._assert_terminal(manager, job["job_id"])

        self.assertEqual(status["status"], "cancelled")
        self.assertFalse(status["accepting_messages"])
        self.assertEqual(status["queued_messages"], 0)

    def test_unexpected_failure_does_not_expose_raw_exception_text(self) -> None:
        manager = self._manager()
        marker = "private-provider-body"
        with (
            patch(
                "deepseek_mcp.job_manager.run_agent",
                side_effect=RuntimeError(marker),
            ),
            patch("deepseek_mcp.job_manager.logger.error") as logged,
        ):
            job = manager.start("failure", "", self._config())
            status = self._assert_terminal(manager, job["job_id"])

        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["error"], "unexpected internal failure")
        self.assertNotIn(marker, str(logged.call_args_list))

    def test_cancel_accepted_before_final_commit_wins_atomically(self) -> None:
        manager = self._manager()
        ready_to_return = threading.Event()
        release_return = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            ready_to_return.set()
            if not release_return.wait(2.0):
                raise AssertionError("test did not release final response")
            return _result()

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("final race", "", self._config())
            self.assertTrue(ready_to_return.wait(1.0))
            cancelled = manager.cancel(job["job_id"])
            self.assertTrue(cancelled["cancel_accepted"])
            release_return.set()
            status = self._assert_terminal(manager, job["job_id"])

        self.assertEqual(status["status"], "cancelled")
        self.assertTrue(status["cancel_requested"])
        self.assertIsNone(manager.result(job["job_id"])["result"])

    def test_late_cancel_preserves_completed_mutation_transactions(self) -> None:
        manager = self._manager()
        ready_to_return = threading.Event()
        release_return = threading.Event()
        result = _result()
        result["mutations"] = [{
            "transaction_id": "a" * 32,
            "tool": "NotebookEdit",
            "status": "committed",
        }]

        def fake_run_agent(task, config, **kwargs):
            ready_to_return.set()
            if not release_return.wait(2.0):
                raise AssertionError("test did not release mutation result")
            return result

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("mutation race", "", self._config())
            self.assertTrue(ready_to_return.wait(1.0))
            manager.cancel(job["job_id"])
            release_return.set()
            status = self._assert_terminal(manager, job["job_id"])

        self.assertEqual(status["status"], "cancelled")
        self.assertIn("DO NOT RETRY", status["error"])
        self.assertIn("a" * 32, status["error"])

    def test_cancel_after_terminal_is_not_accepted(self) -> None:
        manager = self._manager()

        with patch("deepseek_mcp.job_manager.run_agent", return_value=_result()):
            job = manager.start("quick", "", self._config())
            self._assert_terminal(manager, job["job_id"])

        response = manager.cancel(job["job_id"])
        self.assertFalse(response["cancel_accepted"])
        self.assertFalse(response["cancel_requested"])
        self.assertEqual(response["status"], "completed")

    def test_only_one_background_job_can_run(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=1)
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if not release.wait(2.0):
                raise AssertionError("test did not release job")
            return _result()

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config)
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy):
                manager.start("second", "", config)
            with self.assertRaises(JobBusy):
                manager.run_sync("sync", config)
            release.set()
            self._assert_terminal(manager, first["job_id"])

    def test_second_manager_cannot_run_same_workspace(self) -> None:
        first_manager = self._manager()
        second_manager = self._manager()
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if not release.wait(2.0):
                raise AssertionError("test did not release job")
            return _result()

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = first_manager.start(
                "first", "", self._config(same_workspace_writers="exclusive")
            )
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy):
                second_manager.run_sync(
                    "same workspace", self._config(same_workspace_writers="exclusive")
                )
            release.set()
            self._assert_terminal(first_manager, first["job_id"])

    def test_sync_execution_blocks_background_start(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=1)
        started = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if not release.wait(2.0):
                raise AssertionError("test did not release sync run")
            return _result()

        def run_sync() -> None:
            try:
                manager.run_sync("sync", config)
            except BaseException as error:
                errors.append(error)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            thread = threading.Thread(target=run_sync)
            thread.start()
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy):
                manager.start("background", "", config)
            release.set()
            thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_busy_start_does_not_prune_terminal_results(self) -> None:
        manager = self._manager()
        retained = self._seed_terminal_jobs(manager)
        active = JobRecord(
            job_id="active",
            task="active",
            context="",
            task_length=6,
            status="running",
        )
        manager._jobs[active.job_id] = active
        manager._running[active.job_id] = active

        with self.assertRaises(JobBusy):
            manager.start("rejected", "", self._config(max_parallel_agents=1))

        self._assert_results_retained(manager, retained)

    def test_busy_workspace_lease_does_not_prune_terminal_results(self) -> None:
        manager = self._manager()
        retained = self._seed_terminal_jobs(manager)

        with patch.object(
            manager,
            "_acquire_workspace_lease_locked",
            side_effect=JobBusy("workspace busy"),
        ):
            with self.assertRaises(JobBusy):
                manager.start("rejected", "", self._config())

        self._assert_results_retained(manager, retained)

    def test_result_and_usage_claim_are_atomic_against_pruning_start(self) -> None:
        manager = self._manager()
        retained = self._seed_terminal_jobs(manager)
        oldest = retained[0]
        payload_started = threading.Event()
        release_payload = threading.Event()
        collected = []
        start_results = []
        original_payload = manager._result_payload_locked

        def delayed_payload(job: JobRecord) -> dict:
            payload = original_payload(job)
            payload_started.set()
            if not release_payload.wait(2):
                raise AssertionError("test did not release result claim")
            return payload

        def collect() -> None:
            collected.append(manager.result_with_usage_claim(oldest))

        def start_new() -> None:
            start_results.append(manager.start("new", "", self._config()))

        with (
            patch.object(manager, "_result_payload_locked", side_effect=delayed_payload),
            patch("deepseek_mcp.job_manager.run_agent", return_value=_result("new")),
        ):
            collector = threading.Thread(target=collect)
            starter = threading.Thread(target=start_new)
            collector.start()
            self.assertTrue(payload_started.wait(1))
            starter.start()
            release_payload.set()
            collector.join(2)
            starter.join(2)
            self._assert_terminal(manager, start_results[0]["job_id"])

        self.assertFalse(collector.is_alive())
        self.assertFalse(starter.is_alive())
        payload, usage = collected[0]
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["result"]["final_message"], oldest)
        self.assertIsNotNone(usage)
        self.assertEqual(len(start_results), 1)

    def test_thread_start_failure_rolls_back_job_and_workspace_lease(self) -> None:
        manager = self._manager()
        retained = self._seed_terminal_jobs(manager)

        with patch.object(
            threading.Thread,
            "start",
            side_effect=RuntimeError("thread unavailable"),
        ):
            with self.assertRaisesRegex(JobError, "failed to start"):
                manager.start("first", "", self._config())

        self._assert_results_retained(manager, retained)

        with patch("deepseek_mcp.job_manager.run_agent", return_value=_result()):
            second = manager.start("second", "", self._config())
            status = self._assert_terminal(manager, second["job_id"])

        self.assertEqual(status["status"], "completed")

    def test_finalize_closes_mailbox_atomically(self) -> None:
        job = JobRecord(job_id="abc", task="x", context="", task_length=1)
        self.assertEqual(job.drain_messages(finalize_if_empty=True), [])
        with self.assertRaises(JobError):
            job.queue_message("too late")

    def test_task_context_and_sync_inputs_are_bounded(self) -> None:
        manager = self._manager()
        with self.assertRaisesRegex(JobError, "task exceeds"):
            manager.start("x" * (MAX_TASK_BYTES + 1), "", self._config())
        with self.assertRaisesRegex(JobError, "context exceeds"):
            manager.start("task", "x" * (MAX_CONTEXT_BYTES + 1), self._config())
        with self.assertRaisesRegex(JobError, "task exceeds"):
            manager.run_sync("x" * (MAX_COMBINED_TASK_BYTES + 1), self._config())

    def test_steering_queue_count_and_byte_limits_are_atomic(self) -> None:
        job = JobRecord(job_id="bounded", task="x", context="", task_length=1)
        for _ in range(MAX_QUEUED_MESSAGES):
            job.queue_message("x")
        with self.assertRaisesRegex(JobError, "too many"):
            job.queue_message("one too many")
        self.assertEqual(job.snapshot()["queued_messages"], MAX_QUEUED_MESSAGES)

        byte_limited = JobRecord(
            job_id="byte-bounded", task="x", context="", task_length=1
        )
        chunk = "x" * MAX_STEERING_MESSAGE_BYTES
        for _ in range(MAX_QUEUED_MESSAGE_BYTES // MAX_STEERING_MESSAGE_BYTES):
            byte_limited.queue_message(chunk)
        with self.assertRaisesRegex(JobError, "byte limit"):
            byte_limited.queue_message("x")
        self.assertEqual(
            byte_limited.snapshot()["queued_messages"],
            MAX_QUEUED_MESSAGE_BYTES // MAX_STEERING_MESSAGE_BYTES,
        )

    def test_single_steering_message_limit_and_close_clears_content(self) -> None:
        job = JobRecord(job_id="single", task="x", context="", task_length=1)
        with self.assertRaisesRegex(JobError, "message exceeds"):
            job.queue_message("x" * (MAX_STEERING_MESSAGE_BYTES + 1))
        job.queue_message("private")
        job.close_messages()
        self.assertEqual(job.snapshot()["queued_messages"], 0)
        self.assertEqual(job._queued_message_bytes, 0)

    def test_pool_runs_concurrent_jobs_up_to_limit(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly"
        )
        barrier = threading.Barrier(2, timeout=2.0)

        def fake_run_agent(task, config, **kwargs):
            barrier.wait()
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config)
            second = manager.start("second", "", config)
            self.assertTrue(manager.wait_for_terminal(first["job_id"], 2.0))
            self.assertTrue(manager.wait_for_terminal(second["job_id"], 2.0))

        self.assertEqual(manager.status(first["job_id"])["status"], "completed")
        self.assertEqual(manager.status(second["job_id"])["status"], "completed")

    def test_pool_full_rejection_lists_running_jobs(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly"
        )
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first task", "", config)
            second = manager.start("second task", "", config)
            with self.assertRaises(JobBusy) as raised:
                manager.start("third task", "", config)
            release.set()
            self.assertTrue(manager.wait_for_terminal(first["job_id"], 2.0))
            self.assertTrue(manager.wait_for_terminal(second["job_id"], 2.0))

        message = str(raised.exception)
        self.assertIn(first["job_id"], message)
        self.assertIn(second["job_id"], message)
        self.assertIn("max_parallel_agents=2", message)

    def test_workspace_lease_is_acquired_once_and_released_at_drain(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly"
        )
        release = threading.Event()
        acquired: list[object] = []
        released: list[object] = []
        original_acquire = manager._acquire_workspace_lease_locked
        original_release = manager._release_workspace_lease_locked

        def counting_acquire(cfg, shared=False):
            lease = original_acquire(cfg, shared)
            acquired.append(lease)
            return lease

        def counting_release(lease):
            released.append(lease)
            original_release(lease)

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with (
            patch.object(
                manager,
                "_acquire_workspace_lease_locked",
                side_effect=counting_acquire,
            ),
            patch.object(
                manager,
                "_release_workspace_lease_locked",
                side_effect=counting_release,
            ),
            patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent),
        ):
            first = manager.start("first", "", config)
            second = manager.start("second", "", config)
            self.assertEqual(len(acquired), 1)
            self.assertEqual(released, [])
            release.set()
            self.assertTrue(manager.wait_for_terminal(first["job_id"], 2.0))
            self.assertTrue(manager.wait_for_terminal(second["job_id"], 2.0))

        self.assertEqual(len(acquired), 1)
        self.assertEqual(len(released), 1)

    def test_sync_delegation_occupies_a_pool_slot(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly"
        )
        started = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        def run_sync() -> None:
            try:
                manager.run_sync("sync", config)
            except BaseException as error:
                errors.append(error)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            thread = threading.Thread(target=run_sync)
            thread.start()
            self.assertTrue(started.wait(1.0))
            async_job = manager.start("async task", "", config)
            with self.assertRaises(JobBusy):
                manager.start("overflow", "", config)
            release.set()
            thread.join(1.0)
            self.assertTrue(manager.wait_for_terminal(async_job["job_id"], 2.0))

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])

    def test_job_listing_shape_filter_and_bounds(self) -> None:
        manager = self._manager()
        running_release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            if task.startswith("slow"):
                running_release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            completed = manager.start("fast task", "", self._config())
            self.assertTrue(manager.wait_for_terminal(completed["job_id"], 2.0))
            running = manager.start("slow task " + "x" * 200, "", self._config())
            slow_id = running["job_id"]
            all_rows = manager.list_jobs()
            running_rows = manager.list_jobs("running")
            running_release.set()
            self.assertTrue(manager.wait_for_terminal(slow_id, 2.0))

        self.assertEqual([row["job_id"] for row in running_rows], [slow_id])
        text = format_jobs_table(all_rows)
        self.assertIn(TABLE_HEADER, text)
        self.assertIn(completed["job_id"], text)
        self.assertIn(slow_id, text)
        self.assertIn("tokens=2", text)
        self.assertLess(text.index(slow_id), text.index(completed["job_id"]))
        self.assertNotIn("x" * 81, text)

        for index in range(70):
            manager._jobs[f"bulk-{index}"] = JobRecord(
                job_id=f"bulk-{index}",
                task="bulk",
                context="",
                task_length=4,
                status="completed",
                created_at=float(index),
                finished_at=float(index),
                result=_result("bulk"),
            )
        bounded = format_jobs_table(manager.list_jobs())
        self.assertLessEqual(len(bounded.splitlines()), MAX_LIST_LINES)
        self.assertIn("[truncated:", bounded)

    def test_parallel_coding_jobs_on_distinct_workspaces_run_concurrently(self) -> None:
        manager = self._manager()
        release = threading.Event()
        barrier = threading.Barrier(2, timeout=2.0)

        def fake_run_agent(task, config, **kwargs):
            barrier.wait()
            release.wait(2.0)
            return _result(task)

        other = self.workspace.parent / "other"
        other.mkdir()
        config_a = self._config(
            max_parallel_agents=2, same_workspace_writers="exclusive"
        )
        config_b = self._config(
            other, max_parallel_agents=2, same_workspace_writers="exclusive"
        )
        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config_a)
            second = manager.start("second", "", config_b)
            self.assertEqual(self._mode(manager, config_a), LEASE_EXCLUSIVE)
            self.assertEqual(self._mode(manager, config_b), LEASE_EXCLUSIVE)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self._assert_terminal(manager, second["job_id"])

        self.assertTrue(manager._leases.is_idle())

    def test_same_workspace_coding_conflicts_beyond_capacity(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=4, same_workspace_writers="exclusive"
        )
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("write a", "", config)
            with self.assertRaises(JobBusy) as raised:
                manager.start("write b", "", config)
            release.set()
            self._assert_terminal(manager, first["job_id"])

        self.assertIn("exclusive workspace lease", str(raised.exception))

    def test_per_workspace_drain_keeps_other_lease_held(self) -> None:
        manager = self._manager()
        other = self.workspace.parent / "other"
        other.mkdir()
        releases = {True: threading.Event(), False: threading.Event()}

        def fake_run_agent(task, config, **kwargs):
            releases[task == "a"].wait(2.0)
            return _result(task)

        config_a = self._config(
            max_parallel_agents=2, same_workspace_writers="exclusive"
        )
        config_b = self._config(
            other, max_parallel_agents=2, same_workspace_writers="exclusive"
        )
        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("a", "", config_a)
            second = manager.start("b", "", config_b)
            releases[True].set()
            self._assert_terminal(manager, first["job_id"])
            self.assertIsNone(self._mode(manager, config_a))
            self.assertEqual(self._mode(manager, config_b), LEASE_EXCLUSIVE)
            releases[False].set()
            self._assert_terminal(manager, second["job_id"])

        self.assertTrue(manager._leases.is_idle())

    def test_admission_rechecks_pending_transactions_with_cached_lease(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly",
            same_workspace_writers="exclusive",
        )
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config)
            self.assertEqual(self._mode(manager, config), LEASE_SHARED)
            with patch(
                "deepseek_mcp.admission.require_no_pending",
                side_effect=TransactionRecoveryError("unacknowledged transactions"),
            ):
                with self.assertRaisesRegex(JobError, "unacknowledged"):
                    manager.start("second", "", config)
                with self.assertRaisesRegex(JobError, "unacknowledged"):
                    manager.run_sync("second", config)
            admitted = manager.start("second", "", config)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self._assert_terminal(manager, admitted["job_id"])

    def test_run_sync_failure_releases_slot_and_lease(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=1)

        with patch(
            "deepseek_mcp.job_manager.run_agent", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                manager.run_sync("fail", config)

        self.assertEqual(manager._running, {})
        self.assertTrue(manager._leases.is_idle())

        with patch("deepseek_mcp.job_manager.run_agent", return_value=_result()):
            job = manager.start("after", "", config)
            self._assert_terminal(manager, job["job_id"])

    def test_cancel_drain_releases_workspace_lease(self) -> None:
        manager = self._manager()
        config = self._config()
        started = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            if kwargs["cancel_signal"].wait(2.0):
                raise AgentLoopCancelled("cancelled in test")
            return _result("unexpected")

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("cancel me", "", config)
            self.assertTrue(started.wait(1.0))
            manager.cancel(job["job_id"])
            status = self._assert_terminal(manager, job["job_id"])

        self.assertEqual(status["status"], "cancelled")
        self.assertEqual(manager._running, {})
        self.assertTrue(manager._leases.is_idle())

    def test_full_pool_message_is_bounded(self) -> None:
        class _Rec:
            status = "running"
            capability = "coding"
            task_preview = "x"
            task = ""

        running = {f"job-{index}": _Rec() for index in range(100)}
        message = full_pool_message(100, running)

        self.assertIn("max_parallel_agents=100", message)
        self.assertIn("job-0", message)
        self.assertNotIn(f"job-{MAX_LIST_ROWS}", message)
        self.assertIn(
            f"[truncated: {100 - MAX_LIST_ROWS} more running jobs]", message
        )
        self.assertLessEqual(len(message.splitlines()), MAX_LIST_LINES)

    def test_sync_slots_are_marked_in_listing_and_pool_message(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=1)
        started = threading.Event()
        release = threading.Event()
        errors: list[BaseException] = []

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        def run_sync() -> None:
            try:
                manager.run_sync("sync task", config)
            except BaseException as error:
                errors.append(error)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            thread = threading.Thread(target=run_sync)
            thread.start()
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy) as raised:
                manager.start("overflow", "", config)
            rows = manager.list_jobs()
            release.set()
            thread.join(1.0)

        self.assertIn("(sync)", str(raised.exception))
        self.assertIn("(sync)", format_jobs_table(rows))
        self.assertEqual(errors, [])

    def test_coding_admission_while_readonly_running_is_rejected(self) -> None:
        manager = self._manager()
        readonly = self._config(
            delegation_capability="readonly", same_workspace_writers="exclusive"
        )
        coding = self._config(same_workspace_writers="exclusive")
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            readonly_job = manager.start("read", "", readonly)
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy) as raised:
                manager.start("write", "", coding)
            release.set()
            self._assert_terminal(manager, readonly_job["job_id"])

        message = str(raised.exception)
        self.assertIn("exclusive workspace lease", message)
        self.assertIn("jobs still running", message)
        self.assertIn(readonly_job["job_id"], message)
        self.assertIn("readonly", message)

    def test_readonly_admission_while_coding_running_is_rejected(self) -> None:
        manager = self._manager()
        coding = self._config(same_workspace_writers="exclusive")
        readonly = self._config(
            delegation_capability="readonly", same_workspace_writers="exclusive"
        )
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            coding_job = manager.start("write", "", coding)
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy) as raised:
                manager.start("read", "", readonly)
            release.set()
            self._assert_terminal(manager, coding_job["job_id"])

        message = str(raised.exception)
        self.assertIn("cannot share the exclusive workspace lease", message)
        self.assertIn("jobs still running", message)
        self.assertIn(coding_job["job_id"], message)

    def test_lease_mode_transitions_only_via_full_drain(self) -> None:
        manager = self._manager()
        readonly = self._config(
            delegation_capability="readonly", same_workspace_writers="exclusive"
        )
        coding = self._config(same_workspace_writers="exclusive")
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("read", "", readonly)
            self.assertEqual(self._mode(manager, readonly), LEASE_SHARED)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self.assertIsNone(self._mode(manager, readonly))

            release.clear()
            second = manager.start("write", "", coding)
            self.assertEqual(self._mode(manager, coding), LEASE_EXCLUSIVE)
            release.set()
            self._assert_terminal(manager, second["job_id"])
            self.assertIsNone(self._mode(manager, coding))

            release.clear()
            third = manager.start("read again", "", readonly)
            self.assertEqual(self._mode(manager, readonly), LEASE_SHARED)
            release.set()
            self._assert_terminal(manager, third["job_id"])
            self.assertIsNone(self._mode(manager, readonly))

    def test_readonly_leases_coexist_across_managers_but_exclude_coding(self) -> None:
        first = self._manager()
        second = self._manager()
        readonly = self._config(
            delegation_capability="readonly", same_workspace_writers="exclusive"
        )
        coding = self._config(same_workspace_writers="exclusive")
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            held = first.start("read a", "", readonly)
            peered = second.start("read b", "", readonly)
            with self.assertRaises(JobBusy):
                second.start("write b", "", coding)
            third = self._manager()
            with self.assertRaises(JobBusy):
                third.start("write c", "", coding)
            release.set()
            self._assert_terminal(first, held["job_id"])
            self._assert_terminal(second, peered["job_id"])

    def test_job_listing_reports_resolved_agent_id(self) -> None:
        manager = self._manager()
        config = self._config(active_agent="reviewer")

        with patch("deepseek_mcp.job_manager.run_agent", return_value=_result()):
            job = manager.start("review task", "", config)
            self._assert_terminal(manager, job["job_id"])
            rows = manager.list_jobs()

        self.assertEqual(rows[0]["agent"], "reviewer")
        self.assertIn("agent=reviewer", format_jobs_table(rows))

    def test_job_listing_reports_workspace_label(self) -> None:
        manager = self._manager()
        other = self.workspace.parent / "review-wt"
        other.mkdir()
        config = self._config(other)

        with patch("deepseek_mcp.job_manager.run_agent", return_value=_result()):
            job = manager.start("review task", "", config)
            self._assert_terminal(manager, job["job_id"])
            rows = manager.list_jobs()

        self.assertEqual(rows[0]["workspace"], "review-wt")
        self.assertIn("workspace=review-wt", format_jobs_table(rows))

    def test_blocked_listing_includes_agent_id(self) -> None:
        manager = self._manager()
        readonly = self._config(
            delegation_capability="readonly", active_agent="reviewer",
            same_workspace_writers="exclusive",
        )
        coding = self._config(same_workspace_writers="exclusive")
        started = threading.Event()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            started.set()
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("read", "", readonly)
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy) as raised:
                manager.start("write", "", coding)
            release.set()
            self._assert_terminal(manager, job["job_id"])

        self.assertIn("agent=reviewer", str(raised.exception))

    def test_readonly_with_mutation_tools_still_takes_exclusive_lease(self) -> None:
        manager = self._manager()
        config = self._config(same_workspace_writers="exclusive")
        # Bypass Config validation to exercise manager-level defense in depth.
        config.delegation_capability = "readonly"
        config.allowed_tools = ["Read", "Write"]
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("mutate", "", config)
            self.assertEqual(self._mode(manager, config), LEASE_EXCLUSIVE)
            release.set()
            self._assert_terminal(manager, job["job_id"])

        self.assertIsNone(self._mode(manager, config))

    def test_registry_releases_new_lease_when_recovery_gate_fails(self) -> None:
        registry = LeaseRegistry()
        released: list[object] = []

        def acquire_lease(config, shared=False):
            return object()

        def recovery_gate(config):
            raise TransactionRecoveryError("unacknowledged transactions")

        with self.assertRaises(TransactionRecoveryError):
            registry.acquire(
                "identity-1", object(), False, acquire_lease, recovery_gate,
                released.append,
            )

        self.assertTrue(registry.is_idle())
        self.assertEqual(registry.running_count("identity-1"), 0)
        self.assertEqual(len(released), 1)

        lease = registry.acquire(
            "identity-1", object(), False, acquire_lease,
            lambda config: None, released.append,
        )
        self.assertIsNotNone(lease)
        self.assertEqual(registry.running_count("identity-1"), 1)

    def test_allow_mode_parallel_coding_on_same_workspace(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=2)  # default allow mode
        barrier = threading.Barrier(2, timeout=2.0)
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            barrier.wait()
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("a", "", config)
            second = manager.start("b", "", config)
            self.assertEqual(self._mode(manager, config), LEASE_SHARED)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self._assert_terminal(manager, second["job_id"])

        self.assertTrue(manager._leases.is_idle())

    def test_allow_mode_overlap_notice_on_second_coding_job(self) -> None:
        manager = self._manager()
        config = self._config(max_parallel_agents=2)
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first task", "", config)
            self.assertNotIn("overlap_notice", first)
            second = manager.start("second task", "", config)
            notice = second["overlap_notice"]
            self.assertIn(first["job_id"], notice)
            self.assertIn("agent=", notice)
            self.assertIn("create_deepseek_worktree", notice)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self._assert_terminal(manager, second["job_id"])

    def test_allow_mode_pending_recovery_signal(self) -> None:
        manager = self._manager()
        config = self._config()
        summary = {"count": 2, "by_job": {"job-x": 2}}

        with (
            patch("deepseek_mcp.admission.pending_summary", return_value=summary),
            patch("deepseek_mcp.job_manager.run_agent", return_value=_result()),
        ):
            job = manager.start("t", "", config)
            self._assert_terminal(manager, job["job_id"])

        self.assertEqual(job["pending_recovery"], summary)

        with (
            patch("deepseek_mcp.admission.pending_summary", return_value={"count": 0}),
            patch("deepseek_mcp.job_manager.run_agent", return_value=_result()),
        ):
            quiet = manager.start("t2", "", config)
            self._assert_terminal(manager, quiet["job_id"])

        self.assertNotIn("pending_recovery", quiet)

    def test_listing_shows_pending_recovery_badge(self) -> None:
        manager = self._manager()
        config = self._config()

        with (
            patch("deepseek_mcp.admission.pending_total", return_value=3),
            patch("deepseek_mcp.job_manager.run_agent", return_value=_result()),
        ):
            job = manager.start("t", "", config)
            self._assert_terminal(manager, job["job_id"])
            text = format_jobs_table(manager.list_jobs())

        self.assertIn("pending=3", text)

    def test_job_attribution_flows_from_manager_to_tool_settings(self) -> None:
        from deepseek_mcp import tool_child, tool_process

        manager = self._manager()
        config = self._config()
        captured: dict = {}

        def fake_run_agent(task, config, **kwargs):
            captured["job_id"] = config.job_id
            captured["agent"] = config.active_agent
            captured["started_at"] = config.job_started_at
            return _result()

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            job = manager.start("t", "", config)
            self._assert_terminal(manager, job["job_id"])

        self.assertEqual(captured["job_id"], job["job_id"])
        self.assertEqual(captured["agent"], "coding")
        self.assertIsNotNone(captured["started_at"])

        sync_config = self._config()
        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            manager.run_sync("s", sync_config)
        self.assertTrue(captured["job_id"].startswith("sync-"))

        payload = tool_process._config_payload(config)
        self.assertEqual(payload["job_id"], job["job_id"])
        self.assertEqual(payload["active_agent"], "coding")
        child = tool_child._config(payload)
        self.assertEqual(child.job_id, job["job_id"])
        self.assertEqual(child.job_started_at, config.job_started_at)


if __name__ == "__main__":
    unittest.main()
