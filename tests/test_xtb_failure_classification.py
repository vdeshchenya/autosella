"""Exercise actual backend classification through worker and recovery boundaries."""
from __future__ import annotations

from subprocess import CompletedProcess

import pytest

import xtb_molecular_system as xtb
from distributed_validate import worker
from distributed_validate.recovery import evaluate_cases


# Exact messages from the retained 2026-09-08 worker diagnostic.
SCF_STDOUT = (
    "   *** convergence criteria cannot be satisfied within 250 iterations ***\n"
    "-1- scf: Self consistent charge iterator did not converge\n"
)


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("failure, expected_kind, expected_attempts", [
    ("scf_nonconvergence", "optimizer_error", 1),
    ("signal_exit", "infrastructure_error", 3),
    ("unrecognized_exit", "infrastructure_error", 3),
    ("missing_executable", "infrastructure_error", 3),
    ("io_error", "infrastructure_error", 3),
])
def test_backend_failure_controls_worker_classification_and_retries(
    monkeypatch, water_xyz_path, tmp_path, verbose,
    failure, expected_kind, expected_attempts,
):
    monkeypatch.setattr(xtb, "_resolve_xtb_binary", lambda: "/test/xtb")
    system = xtb.XTBMolecularSystem(str(water_xyz_path), verbose=verbose)

    def failed_xtb(command, **kwargs):
        if failure == "missing_executable":
            raise FileNotFoundError("missing xTB executable")
        if failure == "io_error":
            raise PermissionError("xTB executable is not accessible")
        if failure == "signal_exit":
            return CompletedProcess(command, -9, SCF_STDOUT, "killed\n")
        if failure == "unrecognized_exit":
            return CompletedProcess(command, 1, "unrecognized calculator failure\n", "")
        return CompletedProcess(command, 1, SCF_STDOUT, "abnormal termination of xtb\n")

    monkeypatch.setattr(xtb.subprocess, "run", failed_xtb)
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        return positions, 1

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    case = {
        "mol_name": "water",
        "max_steps": 5,
        "baseline": {
            "xyz_path": str(water_xyz_path), "initial_energy": 0.0,
            "final_energy": -1.0, "improvement": 1.0, "n_steps": 10,
        },
    }

    class ImmediateWorkerClient:
        """Local response fixture; no Redis connection or live task mutation."""
        def __init__(self):
            self.results = {}
            self.submissions = []

        def get_result(self, task_id):
            return self.results.get(task_id)

        def task_status(self, task_id):
            return {"state": "finished" if task_id in self.results else "missing"}

        def submit(self, task, task_id=None):
            self.submissions.append(task_id)
            self.results[task_id] = worker._run_optimization_task_guarded(
                task, task_timeout_seconds=2.0, max_task_rss_bytes=0,
                guard_poll_seconds=0.01, kill_grace_seconds=0.02,
            )
            return task_id

    client = ImmediateWorkerClient()
    outcome = evaluate_cases(
        client, [case], {"module_name": "unused_test_optimizer"},
        tmp_path / "recovery.json", timeout_seconds=5.0,
        backoff_seconds=(0, 0), poll_interval_seconds=0,
    )
    assert len(client.submissions) == expected_attempts
    assert all(result["result"] is None for result in client.results.values())
    assert {result["error_kind"] for result in client.results.values()} == {expected_kind}
    if expected_kind == "optimizer_error":
        assert outcome["status"] == "complete"
        assert outcome["num_errors"] == 1
        assert not any(event["cause"] == "worker_infrastructure_failure"
                       for event in outcome["recovery_events"])
    else:
        assert outcome["status"] == "pending"
        assert outcome["num_errors"] == 0
        assert outcome["pending_reason"] == "worker_attempts_exhausted_requires_diagnosis"
