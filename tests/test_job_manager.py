from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from deepseek_mcp.agent_loop import AgentLoopCancelled
from deepseek_mcp.config import Config
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
        return Config(
            api_key="sk-test", workspace=workspace or self.workspace, **overrides
        )

    def _assert_terminal(self, manager: DeepSeekJobManager, job_id: str) -> dict:
        self.assertTrue(manager.wait_for_terminal(job_id, 2.0))
        return manager.status(job_id)

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
            first = first_manager.start("first", "", self._config())
            self.assertTrue(started.wait(1.0))
            with self.assertRaises(JobBusy):
                second_manager.run_sync("same workspace", self._config())
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

    def test_workspace_change_while_pool_running_is_rejected(self) -> None:
        manager = self._manager()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        other = self.workspace.parent / "other"
        other.mkdir()
        config_a = self._config(delegation_capability="readonly")
        config_b = self._config(other, delegation_capability="readonly")
        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config_a)
            with self.assertRaisesRegex(JobError, "workspace") as raised:
                manager.start("second", "", config_b)
            message = str(raised.exception)
            self.assertIn(config_a.expected_workspace_identity, message)
            self.assertIn(config_b.expected_workspace_identity, message)
            with self.assertRaisesRegex(JobError, "workspace"):
                manager.run_sync("second", config_b)
            same_workspace = manager.start("third", "", config_a)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self._assert_terminal(manager, same_workspace["job_id"])

    def test_admission_rechecks_pending_transactions_with_cached_lease(self) -> None:
        manager = self._manager()
        config = self._config(
            max_parallel_agents=2, delegation_capability="readonly"
        )
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("first", "", config)
            self.assertEqual(manager._lease_mode, LEASE_SHARED)
            with patch(
                "deepseek_mcp.job_manager.require_no_pending",
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
        self.assertIsNone(manager._lease)
        self.assertIsNone(manager._lease_identity)
        self.assertIsNone(manager._lease_mode)

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
        self.assertIsNone(manager._lease)
        self.assertIsNone(manager._lease_identity)
        self.assertIsNone(manager._lease_mode)

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
        readonly = self._config(delegation_capability="readonly")
        coding = self._config()
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
        coding = self._config()
        readonly = self._config(delegation_capability="readonly")
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
        readonly = self._config(delegation_capability="readonly")
        coding = self._config()
        release = threading.Event()

        def fake_run_agent(task, config, **kwargs):
            release.wait(2.0)
            return _result(task)

        with patch("deepseek_mcp.job_manager.run_agent", side_effect=fake_run_agent):
            first = manager.start("read", "", readonly)
            self.assertEqual(manager._lease_mode, LEASE_SHARED)
            release.set()
            self._assert_terminal(manager, first["job_id"])
            self.assertIsNone(manager._lease_mode)

            release.clear()
            second = manager.start("write", "", coding)
            self.assertEqual(manager._lease_mode, LEASE_EXCLUSIVE)
            release.set()
            self._assert_terminal(manager, second["job_id"])
            self.assertIsNone(manager._lease_mode)

            release.clear()
            third = manager.start("read again", "", readonly)
            self.assertEqual(manager._lease_mode, LEASE_SHARED)
            release.set()
            self._assert_terminal(manager, third["job_id"])
            self.assertIsNone(manager._lease_mode)

    def test_readonly_leases_coexist_across_managers_but_exclude_coding(self) -> None:
        first = self._manager()
        second = self._manager()
        readonly = self._config(delegation_capability="readonly")
        coding = self._config()
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


if __name__ == "__main__":
    unittest.main()
