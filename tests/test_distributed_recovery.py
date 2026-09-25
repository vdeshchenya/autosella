"""Failure injection for durable Redis reconciliation and bounded recovery."""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_fakeredis

from distributed_validate.client import RemoteOptimizationClient
from distributed_validate.protocol import (
    load_next_optimization_task, queue_key, store_optimization_result, worker_key,
)
from distributed_validate.recovery import evaluate_cases


CASES = [{"mol_name": "a", "baseline": {}, "max_steps": 100}]
SPEC = {"source": "def minimize(): pass"}


def success(name="a", stop_reason="force_call_limit"):
    return {"result": {"mol_name": name, "stop_reason": stop_reason},
            "error": None, "error_kind": None}


@pytest.fixture
def client():
    import fakeredis

    class Client(RemoteOptimizationClient):
        def __init__(self):
            self._redis = fakeredis.FakeStrictRedis()
            self.poll_interval_seconds = 0
            self.submissions = []
            self.on_submit = None

        def reconnect(self):
            self._redis.ping()

        def submit(self, task, task_id=None):
            task_id = super().submit(task, task_id=task_id)
            self.submissions.append(task_id)
            if self.on_submit:
                self.on_submit(task_id, task)
            return task_id

    return Client()


def evaluate(client, path, cases=CASES, **kwargs):
    # Atomic state fsync can exceed a subsecond budget under parallel test load.
    # Tests that inspect pending tasks explicitly use timeout_seconds=0.
    kwargs.setdefault("timeout_seconds", 5.0)
    return evaluate_cases(client, cases, SPEC, path, backoff_seconds=(0, 0),
                          poll_interval_seconds=0, **kwargs)


def read_state(path):
    return json.loads(path.read_text())


def test_uncertain_submit_reconciles_same_id_before_replay(client, tmp_path):
    from redis.exceptions import ConnectionError

    path = tmp_path / "state.json"

    def committed_but_acknowledgement_lost(task_id, task):
        assert read_state(path)["records"][0]["attempts"][0]["task_id"] == task_id
        store_optimization_result(client._redis, task_id, success())
        raise ConnectionError("response lost after transaction committed")

    client.on_submit = committed_but_acknowledgement_lost
    outcome = evaluate(client, path)
    assert outcome["status"] == "complete"
    assert len(client.submissions) == 1
    assert len(outcome["results"]) == 1
    assert outcome["recovery_events"][0]["cause"] == "transport_failure"


def test_read_connection_loss_recovers_existing_result(client, tmp_path):
    from redis.exceptions import ConnectionError

    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    task_id = client.submissions[0]
    store_optimization_result(client._redis, task_id, success())
    real_read = client.get_result
    calls = []

    def broken_once(task_id):
        calls.append(task_id)
        if len(calls) == 1:
            raise ConnectionError("read connection lost")
        return real_read(task_id)

    client.get_result = broken_once
    outcome = evaluate(client, path)
    assert outcome["status"] == "complete"
    assert client.submissions == [task_id]
    assert client.get_result(task_id) == success()


def test_existing_running_task_is_never_resubmitted(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    task_id, _ = load_next_optimization_task(client._redis, ["xtb"], worker_id="alive")
    outcome = evaluate(client, path, timeout_seconds=0)
    assert outcome["pending_reason"] == "waiting_for_existing_tasks"
    assert client.submissions == [task_id]
    assert read_state(path)["records"][0]["attempts"][0]["state"] == "running"


def test_expired_lease_is_ambiguous_and_not_resubmitted(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    task_id, _ = load_next_optimization_task(client._redis, ["xtb"], worker_id="gone")
    client._redis.delete(worker_key("gone"))
    outcome = evaluate(client, path)
    assert outcome["status"] == "pending"
    assert outcome["pending_reason"] == "worker_state_requires_diagnosis"
    assert client.submissions == [task_id]


def test_proven_lost_queued_task_retries_only_affected_molecule(client, tmp_path):
    path = tmp_path / "state.json"
    cases = CASES + [{"mol_name": "b", "baseline": {}, "max_steps": 100}]
    evaluate(client, path, cases, timeout_seconds=0)
    first, lost = client.submissions
    store_optimization_result(client._redis, first, success("a"))
    client._redis.lrem(queue_key("xtb"), 0, lost)
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, success(task["mol_name"]))
    outcome = evaluate(client, path, cases)
    assert outcome["status"] == "complete"
    assert [item["mol_name"] for item in outcome["results"]] == ["a", "b"]
    records = read_state(path)["records"]
    assert len(records[0]["attempts"]) == 1
    assert len(records[1]["attempts"]) == 2
    assert len(client.submissions) == 3


def test_successes_are_durable_and_resume_deduplicates(client, tmp_path):
    path = tmp_path / "state.json"
    cases = CASES + [{"mol_name": "b", "baseline": {}, "max_steps": 100}]
    evaluate(client, path, cases, timeout_seconds=0)
    first, second = client.submissions
    store_optimization_result(client._redis, first, success("a"))
    pending = evaluate(client, path, cases, timeout_seconds=0)
    assert pending["status"] == "pending" and pending["results"] == []
    assert read_state(path)["records"][0]["payload"] == success("a")
    # Once downloaded, success is retained even if Redis subsequently loses it.
    from distributed_validate.protocol import delete_task_artifacts
    delete_task_artifacts(client._redis, first)
    store_optimization_result(client._redis, second, success("b"))
    complete = evaluate(client, path, cases)
    assert [item["mol_name"] for item in complete["results"]] == ["a", "b"]
    assert evaluate(client, path, cases)["results"] == complete["results"]
    assert len(client.submissions) == 2


def test_repeated_candidate_worker_failure_stops_after_three_attempts(client, tmp_path):
    client.on_submit = lambda tid, task: store_optimization_result(
        client._redis, tid, {"result": None, "error": "child exited -9",
                             "error_kind": "infrastructure_error"})
    path = tmp_path / "state.json"
    outcome = evaluate(client, path)
    assert outcome["pending_reason"] == "worker_attempts_exhausted_requires_diagnosis"
    assert outcome["results"] == [] and outcome["num_errors"] == 0
    assert len(client.submissions) == 3
    assert len(read_state(path)["records"][0]["attempts"]) == 3
    assert len([event for event in outcome["recovery_events"]
                if event["cause"] == "worker_infrastructure_failure"]) == 3
    evaluate(client, path)
    assert len(client.submissions) == 3


@pytest.mark.parametrize("stop_reason", ["force_call_limit", "time_limit", "converged"])
def test_completed_scientific_stops_are_not_retried(client, tmp_path, stop_reason):
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, success(stop_reason=stop_reason))
    outcome = evaluate(client, tmp_path / "state.json")
    assert outcome["status"] == "complete"
    assert outcome["results"][0]["stop_reason"] == stop_reason
    assert len(client.submissions) == 1


def test_optimizer_exception_is_terminal_candidate_evidence(client, tmp_path):
    client.on_submit = lambda tid, task: store_optimization_result(
        client._redis, tid, {"result": None, "error": "bad optimizer shape",
                             "error_kind": "optimizer_error"})
    outcome = evaluate(client, tmp_path / "state.json")
    assert outcome["status"] == "complete" and outcome["num_errors"] == 1
    assert outcome["errors"] == [("a", "bad optimizer shape")]
    assert len(client.submissions) == 1


def memory_limit_failure(kind="infrastructure_error"):
    return {"result": None, "error": "worker_guard_rss: task exceeded RSS limit",
            "error_kind": kind, "worker_guard": {"kind": "worker_guard_rss"},
            "diagnostics": {"availability": "stored", "artifact_id": "b" * 64}}


@pytest.mark.parametrize("kind", ["infrastructure_error", "optimizer_error"])
def test_memory_limit_from_old_or_new_worker_is_not_retried(client, tmp_path, kind):
    payload = memory_limit_failure(kind)
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, payload)
    result = evaluate(client, tmp_path / "state.json")
    assert result["status"] == "complete" and result["num_errors"] == 1
    assert result["results"] == []
    assert result["failure_details"][0]["error_kind"] == "optimizer_error"
    assert result["failure_details"][0]["diagnostics"] == payload["diagnostics"]
    assert len(client.submissions) == 1


def test_exhausted_legacy_memory_failure_recovers_without_redis_or_reexecution(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    state = read_state(path)
    record = state["records"][0]
    record["attempts"] = [{"task_id": f"old-{i}", "state": "infrastructure_failed",
                           "failure": memory_limit_failure()} for i in range(3)]
    state["pending_reason"] = "worker_attempts_exhausted_requires_diagnosis"
    path.write_text(json.dumps(state))
    result = evaluate(client, path)
    assert result["status"] == "complete" and result["num_errors"] == 1
    assert result["results"] == [] and len(client.submissions) == 1
    saved = read_state(path)["records"][0]
    assert saved["payload"]["error_kind"] == "optimizer_error"
    assert saved["attempts"][0]["failure"]["error_kind"] == "infrastructure_error"
    assert len(saved["attempts"]) == 3
    assert result["recovery_events"][-1]["cause"] == "memory_limit_reclassified_as_optimizer_error"
    assert evaluate(client, path)["errors"] == result["errors"]


@pytest.mark.parametrize("guard", [None, {"kind": "worker_child_exit"}])
def test_error_text_alone_does_not_reclassify_infrastructure_failure(client, tmp_path, guard):
    payload = memory_limit_failure()
    payload["worker_guard"] = guard
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, payload)
    result = evaluate(client, tmp_path / "state.json")
    assert result["status"] == "pending" and result["num_errors"] == 0
    assert len(client.submissions) == 3


def test_failure_receipts_survive_recovery_without_becoming_measurements(client, tmp_path):
    receipt = {"availability": "upload_pending", "artifact_id": "a" * 64,
               "summary": {"n_steps": 2, "n_completed_steps": 1}}
    client.on_submit = lambda tid, task: store_optimization_result(
        client._redis, tid, {"result": None, "error": "SCF failure", "error_kind": "optimizer_error",
                             "worker_guard": {"kind": "child_exception", "traceback": "trace\n" * 3000},
                             "diagnostics": receipt})
    path = tmp_path / "state.json"
    outcome = evaluate(client, path)
    detail = outcome["failure_details"][0]
    assert outcome["status"] == "complete" and outcome["results"] == []
    assert outcome["errors"] == [("a", "SCF failure")]
    assert detail["diagnostics"] == receipt
    assert detail["task_id"] == client.submissions[0]
    assert len(detail["traceback"]) == 4096 and detail["traceback_truncated"]
    assert evaluate(client, path)["failure_details"] == outcome["failure_details"]
    assert len(client.submissions) == 1


def test_pending_infrastructure_details_do_not_change_error_accounting(client, tmp_path):
    client.on_submit = lambda tid, task: store_optimization_result(
        client._redis, tid, {"result": None, "error": "worker lost", "error_kind": "infrastructure_error",
                             "diagnostics": {"availability": "unavailable", "reason": "quota"}})
    result = evaluate(client, tmp_path / "state.json")
    assert result["status"] == "pending" and result["num_errors"] == 0
    assert result["results"] == result["errors"] == []
    assert len(result["failure_details"]) == 1
    assert result["failure_details"][0]["error_kind"] == "infrastructure_error"
    assert result["failure_details"][0]["task_id"] == client.submissions[-1]


def test_redis_unavailable_leaves_durable_pending_with_bounded_attempts(tmp_path):
    from distributed_validate.client import InfrastructureUnavailable
    attempts = []

    def unavailable():
        attempts.append(1)
        raise InfrastructureUnavailable("Redis cannot be reached")

    path = tmp_path / "state.json"
    outcome = evaluate(unavailable, path)
    assert outcome["status"] == "pending" and outcome["results"] == []
    assert outcome["pending_reason"] == "infrastructure_unavailable"
    assert len(attempts) == 3
    assert len(read_state(path)["records"][0]["attempts"]) == 1


def test_absent_redis_state_after_submission_does_not_duplicate_possible_worker(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    client._redis.flushall()
    outcome = evaluate(client, path)
    assert outcome["pending_reason"] == "task_state_missing_requires_diagnosis"
    assert len(client.submissions) == 1


def test_pause_withdraw_and_resume_use_owned_existing_id(client, tmp_path):
    path = tmp_path / "state.json"
    metadata = {"run_id": "run", "evaluation_id": "eval"}
    evaluate(client, path, timeout_seconds=0, task_metadata=metadata)
    tid = client.submissions[0]
    assert not client.withdraw_queued_task(tid, run_id="other", evaluation_id="eval")
    assert client.withdraw_queued_task(tid, **metadata)
    outcome = evaluate(client, path, task_metadata=metadata, should_continue=lambda: False)
    assert outcome["pending_reason"] == "paused_or_cancelled"
    assert client._redis.llen(queue_key("xtb")) == 0
    evaluate(client, path, timeout_seconds=0, task_metadata=metadata)
    assert client.task_status(tid)["state"] == "queued"
    assert client.submissions == [tid]


def test_state_identity_prevents_reusing_results_for_different_candidate(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    with pytest.raises(ValueError, match="does not match"):
        evaluate_cases(client, CASES, {"source": "different"}, path)


@pytest.mark.parametrize("payload", [{}, {"result": None, "error": None},
    {"result": {"mol_name": "wrong"}, "error": None},
    {"result": None, "error": None, "error_kind": "infrastructure_error"}, "bad",
    {"result": {"mol_name": "a"}, "error": "also failed", "error_kind": "optimizer_error"}])
def test_malformed_result_is_terminal_error_not_retry(client, tmp_path, payload):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    # Bypass the result publisher for non-dict corrupt payload simulation.
    original_read = client.get_result
    client.get_result = lambda tid: payload if tid in client.submissions else original_read(tid)
    outcome = evaluate(client, path, timeout_seconds=0)
    assert outcome["status"] == "complete" and outcome["num_errors"] == 1
    assert outcome["results"] == []
    assert "Malformed worker result" in outcome["errors"][0][1]
    assert len(client.submissions) == 1


def test_previously_persisted_result_plus_error_becomes_one_terminal_error(client, tmp_path):
    path = tmp_path / "state.json"
    evaluate(client, path, timeout_seconds=0)
    state = read_state(path)
    state["records"][0]["payload"] = {"result": {"mol_name": "a"}, "error": "also failed"}
    path.write_text(json.dumps(state))
    outcome = evaluate(client, path, timeout_seconds=0)
    assert outcome["status"] == "complete"
    assert outcome["num_errors"] == 1 and outcome["results"] == []
    assert read_state(path)["records"][0]["payload"]["result"] is None


def test_uncertain_submit_of_still_queued_task_does_not_requeue(client, tmp_path):
    from redis.exceptions import ConnectionError

    def response_lost(tid, task):
        raise ConnectionError("submit response lost")

    client.on_submit = response_lost
    outcome = evaluate(client, tmp_path / "state.json", timeout_seconds=0)
    assert outcome["status"] == "pending"
    assert len(client.submissions) == 1
    assert client._redis.llen(queue_key("xtb")) == 1


def test_coordinator_interrupt_after_submit_resumes_durable_id(client, tmp_path):
    path = tmp_path / "state.json"

    def interrupt(tid, task):
        raise KeyboardInterrupt()

    client.on_submit = interrupt
    with pytest.raises(KeyboardInterrupt):
        evaluate(client, path)
    record = read_state(path)["records"][0]
    assert record["attempts"][0]["state"] == "submitting"
    task_id = client.submissions[0]
    client.on_submit = None
    store_optimization_result(client._redis, task_id, success())
    outcome = evaluate(client, path)
    assert outcome["status"] == "complete"
    assert client.submissions == [task_id]


def test_resume_after_initial_outage_retains_original_task_id(client, tmp_path):
    from distributed_validate.client import InfrastructureUnavailable
    path = tmp_path / "state.json"

    def unavailable():
        raise InfrastructureUnavailable("offline")

    assert evaluate(unavailable, path)["status"] == "pending"
    task_id = read_state(path)["records"][0]["attempts"][0]["task_id"]
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, success())
    outcome = evaluate(client, path)
    assert outcome["status"] == "complete"
    assert client.submissions == [task_id]
    assert outcome["transport_retries"] == 2
    assert outcome["infrastructure_retries"] == 0


@pytest.mark.parametrize("phase", ["submitted", "withdrawn", "uncertain_submit"])
def test_resume_after_retention_expiry_requires_diagnosis_without_resubmission(client, tmp_path, monkeypatch, phase):
    import time
    from distributed_validate.protocol import TASK_TTL_SECONDS

    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    path = tmp_path / "state.json"
    metadata = {"run_id": "run", "evaluation_id": "eval"}
    if phase == "uncertain_submit":
        def interrupt(tid, task):
            raise KeyboardInterrupt()
        client.on_submit = interrupt
        with pytest.raises(KeyboardInterrupt):
            evaluate(client, path, task_metadata=metadata)
    else:
        evaluate(client, path, timeout_seconds=0, task_metadata=metadata)
    task_id = client.submissions[0]
    if phase == "withdrawn":
        assert client.withdraw_queued_task(task_id, **metadata)
    clock[0] += TASK_TTL_SECONDS + 1
    client.on_submit = None
    outcome = evaluate(client, path, timeout_seconds=0, task_metadata=metadata)
    assert outcome["status"] == "pending"
    assert outcome["pending_reason"] == "task_state_missing_requires_diagnosis"
    assert client.submissions == [task_id]
    assert len(read_state(path)["records"][0]["attempts"]) == 1


def test_persisted_complete_result_survives_redis_retention_expiry(client, tmp_path, monkeypatch):
    import time
    from distributed_validate.protocol import TASK_TTL_SECONDS

    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    path = tmp_path / "state.json"
    client.on_submit = lambda tid, task: store_optimization_result(client._redis, tid, success())
    assert evaluate(client, path)["status"] == "complete"
    task_id = client.submissions[0]
    clock[0] += TASK_TTL_SECONDS + 1
    assert client.get_result(task_id) is None
    outcome = evaluate(client, path)
    assert outcome["status"] == "complete"
    assert outcome["results"] == [success()["result"]]
    assert client.submissions == [task_id]
