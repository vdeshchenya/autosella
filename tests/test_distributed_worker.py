"""Tests for distributed_validate.worker._run_optimization_task."""

from __future__ import annotations

import sys
import time
import types

import numpy as np
import pytest

from distributed_validate import worker
import xtb_molecular_system as actual_xtb_backend


@pytest.fixture
def fake_xtb_backend(monkeypatch):
    """Inject a fake xtb_molecular_system into sys.modules."""
    module = types.ModuleType("xtb_molecular_system")

    class FakeSystem:
        def __init__(self, xyz_path, num_threads=1, **kw):
            self.atomic_numbers = np.array([8, 1, 1], dtype=int)
            self.initial_positions = np.array(
                [[0.0, 0.0, 0.0], [0.0757, 0.0586, 0.0], [-0.0757, 0.0586, 0.0]],
                dtype=float,
            )

        def compute(self, positions):
            pos = np.asarray(positions, dtype=float)
            return 0.5 * float(np.sum(pos**2)), -pos.copy()

    module.MolecularSystem = FakeSystem
    monkeypatch.setitem(sys.modules, "xtb_molecular_system", module)
    # Also clear any cached systems from previous tests
    worker._SYSTEM_CACHE.clear()
    worker._OPTIMIZER_CACHE.clear()
    return FakeSystem


@pytest.fixture
def stub_minimize_func(monkeypatch):
    """Bypass optimizer loading; return a canned minimize_func."""
    def fake_minimize(positions, atomic_numbers, calc, max_force_calls, converged):
        final_positions = positions * 0.5
        calc(final_positions)
        return final_positions, 1  # always make it better

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: fake_minimize)


def _task(xyz_path="/tmp/x.xyz", n_atoms=3):
    return {
        "mode": "xtb",
        "mol_name": "test_mol",
        "baseline": {
            "n_steps": 10,
            "improvement": 100.0,
            "initial_energy": 0.0757**2 + 0.0586**2,
            "final_energy": 0.0757**2 + 0.0586**2 - 100.0,
            "n_atoms": n_atoms,
            "xyz_path": xyz_path,
        },
        "max_steps": 50,
        "optimizer_spec": {"kind": "stub"},
        "num_threads": 1,
    }


def test_run_task_returns_schema(fake_xtb_backend, stub_minimize_func):
    out = worker._run_optimization_task(_task())
    assert out["error"] is None
    result = out["result"]
    assert set(result.keys()) == {
        "mol_name",
        "converged",
        "max_steps",
        "n_steps",
        "n_completed_steps",
        "stop_reason",
        "final_energy",
        "final_positions",
        "rel_energy",
        "energy_delta_kcal_mol",
        "rel_steps",
    }
    assert result["mol_name"] == "test_mol"
    assert result["n_steps"] == 1
    assert result["max_steps"] == 50
    assert result["rel_steps"] == 1 / 10


def test_run_task_nonfinite_positions_raises(fake_xtb_backend, monkeypatch):
    """Non-finite returned coordinates are rejected before geometry binding."""
    def nan_minimize(positions, atomic_numbers, calc, max_force_calls, converged):
        calc(positions)
        return np.full_like(positions, np.nan), 1

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: nan_minimize)

    with pytest.raises(ValueError, match="Final positions are not finite"):
        worker._run_optimization_task(_task())


def test_run_task_nonfinite_last_calc_energy_raises(
    fake_xtb_backend, monkeypatch
):
    """The recorded final calc energy must be finite."""
    system = fake_xtb_backend("/tmp/x.xyz")
    compute_calls = 0
    original_compute = system.compute

    def nonfinite_compute(positions):
        nonlocal compute_calls
        compute_calls += 1
        energy, forces = original_compute(positions)
        if compute_calls == 1:
            energy = float("nan")
        return energy, forces

    system.compute = nonfinite_compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def nan_energy_minimize(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        return positions.copy(), 1

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: nan_energy_minimize
    )

    with pytest.raises(ValueError, match="Calculation energy is not a finite scalar"):
        worker._run_optimization_task(_task())


def test_run_task_bad_shape_raises(fake_xtb_backend, monkeypatch):
    """Wrong-shape result → ValueError propagates (catch is in start_worker, not here)."""
    def bad_shape(positions, atomic_numbers, calc, max_force_calls, converged):
        calc(positions)
        return np.zeros((5, 3)), 1  # wrong atom count

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: bad_shape)

    with pytest.raises(ValueError, match="Wrong number of atoms"):
        worker._run_optimization_task(_task())


def test_system_cache_reuse(fake_xtb_backend, stub_minimize_func):
    """Two tasks with the same xyz_path should hit the system cache."""
    t1 = _task(xyz_path="/tmp/same.xyz")
    t2 = _task(xyz_path="/tmp/same.xyz")

    worker._run_optimization_task(t1)
    worker._run_optimization_task(t2)

    # Only one system entry in the cache for this path
    keys_for_this_xyz = [k for k in worker._SYSTEM_CACHE if k[1] == "/tmp/same.xyz"]
    assert len(keys_for_this_xyz) == 1


def test_guarded_task_times_out(monkeypatch):
    def slow_task(task):
        time.sleep(5)
        return {"result": {"mol_name": "late"}, "error": None}

    monkeypatch.setattr(worker, "_run_optimization_task", slow_task)

    out = worker._run_optimization_task_guarded(
        _task(),
        task_timeout_seconds=0.2,
        max_task_rss_bytes=0,
        guard_poll_seconds=0.05,
        kill_grace_seconds=0.05,
    )

    assert out["result"] is None
    assert out["error"].startswith("worker_guard_timeout:")
    assert out["error_kind"] == "infrastructure_error"
    assert out["worker_guard"]["kind"] == "worker_guard_timeout"


def test_guarded_task_reports_child_exception(monkeypatch):
    def boom(task):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(worker, "_run_optimization_task", boom)

    out = worker._run_optimization_task_guarded(
        _task(),
        task_timeout_seconds=5.0,
        max_task_rss_bytes=0,
        guard_poll_seconds=0.05,
        kill_grace_seconds=0.05,
    )

    assert out["result"] is None
    assert out["error"] == "backend exploded"
    assert out["worker_guard"]["kind"] == "child_exception"


def _guard(task):
    return worker._run_optimization_task_guarded(
        task, task_timeout_seconds=0.25, max_task_rss_bytes=0,
        guard_poll_seconds=0.01, kill_grace_seconds=0.02,
    )


def test_timeout_scores_last_completed_geometry(fake_xtb_backend, monkeypatch):
    def evaluate_then_hang(positions, atomic_numbers, calc, **kwargs):
        calc(positions * 0.5)
        time.sleep(5)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: evaluate_then_hang)
    out = _guard(_task())
    result = out["result"]
    assert out["error"] is out["error_kind"] is None
    assert result["stop_reason"] == "time_limit"
    assert result["converged"] is False
    assert result["n_steps"] == result["n_completed_steps"] == 1
    system = fake_xtb_backend("/tmp/x.xyz")
    final_pos = system.initial_positions * 0.5
    np.testing.assert_array_equal(result["final_positions"], final_pos)
    assert result["final_energy"] == system.compute(final_pos)[0]
    assert result["rel_energy"] > 0


def test_timeout_during_next_compute_preserves_completed_state_and_actual_calls(
    fake_xtb_backend, monkeypatch
):
    system = fake_xtb_backend("/tmp/x.xyz")
    actual_compute = system.compute
    calls = 0

    def hanging_second_compute(positions):
        nonlocal calls
        calls += 1
        if calls == 2:
            time.sleep(5)
        return actual_compute(positions)

    system.compute = hanging_second_compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions * 0.5)
        calc(positions * 0.25)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    result = out["result"]
    assert out["error"] is None
    assert result["n_steps"] == 2
    assert result["n_completed_steps"] == 1
    assert result["rel_steps"] == 0.2
    np.testing.assert_array_equal(result["final_positions"], system.initial_positions * 0.5)
    assert result["final_energy"] == actual_compute(system.initial_positions * 0.5)[0]


@pytest.mark.parametrize("bad_output", [
    None, (np.zeros((3, 3)),), (np.zeros((3, 3)), 0, "extra"),
    (np.full((3, 3), np.inf), 1), (np.zeros((2, 3)), 1),
])
def test_malformed_optimizer_return_invalidates_good_calculation(
    fake_xtb_backend, monkeypatch, bad_output
):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        return bad_output

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"


@pytest.mark.parametrize("fault", ["energy_nan", "energy_shape", "forces_nan", "forces_shape", "tuple_shape"])
def test_invalid_calculation_output_cannot_be_hidden_by_timeout(
    fake_xtb_backend, monkeypatch, fault
):
    system = fake_xtb_backend("/tmp/x.xyz")
    compute = system.compute
    calls = 0

    def invalid_second_compute(positions):
        nonlocal calls
        calls += 1
        if calls == 1:
            return compute(positions)
        energy, forces = compute(positions)
        return {
            "energy_nan": (np.nan, forces),
            "energy_shape": ([energy], forces),
            "forces_nan": (energy, np.full_like(forces, np.nan)),
            "forces_shape": (energy, forces[:1]),
            "tuple_shape": (energy,),
        }[fault]

    system.compute = invalid_second_compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        try:
            calc(positions * 0.5)
        except Exception:
            time.sleep(5)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"


def test_optimizer_exception_after_valid_calc_is_not_salvaged(fake_xtb_backend, monkeypatch):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        raise RuntimeError("optimizer exploded")

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert "optimizer exploded" in out["error"]


def test_candidate_memory_error_is_numerical_failure(fake_xtb_backend, monkeypatch):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        raise MemoryError("allocation failed")

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert "numerical_memory_error" in out["error"]


def test_child_loss_does_not_score_partial_checkpoint(fake_xtb_backend, monkeypatch):
    import os

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        os._exit(23)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "infrastructure_error"
    assert out["worker_guard"]["exitcode"] == 23


def test_backend_memory_error_cannot_be_hidden_by_candidate(fake_xtb_backend, monkeypatch):
    def no_memory(self, positions):
        raise MemoryError("backend allocation failed")

    def minimize(positions, atomic_numbers, calc, **kwargs):
        try:
            calc(positions)
        except Exception:
            pass
        return positions, 1

    monkeypatch.setattr(fake_xtb_backend, "compute", no_memory)
    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert "numerical_memory_error" in out["error"]


def test_parent_drains_large_child_result_before_exit(monkeypatch):
    large_result = {"result": {"final_positions": [[1.0, 2.0, 3.0]] * 10000}, "error": None}
    monkeypatch.setattr(worker, "_run_optimization_task", lambda task: large_result)
    out = worker._run_optimization_task_guarded(
        _task(), task_timeout_seconds=5, max_task_rss_bytes=0,
        guard_poll_seconds=0.01, kill_grace_seconds=0.02,
    )
    assert out == large_result


def test_disabled_guards_still_isolate_optimizer_and_send_heartbeat(monkeypatch):
    import os

    monkeypatch.setattr(worker, "_run_optimization_task", lambda task: {"pid": os.getpid()})
    heartbeats = []
    out = worker._run_optimization_task_guarded(
        _task(), task_timeout_seconds=0, max_task_rss_bytes=0,
        guard_poll_seconds=0.01, heartbeat=lambda: heartbeats.append(True),
    )
    assert out["pid"] != os.getpid()
    assert heartbeats


@pytest.mark.parametrize("coordinates", [np.full((3, 3), np.nan), np.zeros((2, 3))])
def test_bad_requested_geometry_invalidates_even_if_optimizer_catches_it(
    fake_xtb_backend, monkeypatch, coordinates
):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        try:
            calc(coordinates)
        except Exception:
            pass
        return positions, 1

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"


def test_rss_guard_is_terminal_candidate_failure(monkeypatch):
    def slow_task(task):
        time.sleep(5)

    monkeypatch.setattr(worker, "_run_optimization_task", slow_task)
    monkeypatch.setattr(worker, "_process_group_rss_bytes", lambda pid: 1000)
    out = worker._run_optimization_task_guarded(
        _task(), task_timeout_seconds=5, max_task_rss_bytes=100,
        guard_poll_seconds=0.01, kill_grace_seconds=0.02,
    )
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert out["worker_guard"]["kind"] == "worker_guard_rss"


def test_timeout_during_convergence_keeps_the_complete_calculation(
    fake_xtb_backend, stub_minimize_func, monkeypatch
):
    monkeypatch.setattr(worker, "update_convergence_state", lambda *args: time.sleep(5))
    out = _guard(_task())
    assert out["error"] is None
    assert out["result"]["stop_reason"] == "time_limit"
    assert out["result"]["n_steps"] == out["result"]["n_completed_steps"] == 1
    system = fake_xtb_backend("/tmp/x.xyz")
    np.testing.assert_array_equal(out["result"]["final_positions"], system.initial_positions * 0.5)


@pytest.mark.parametrize("failure", ["nonfinite", "malformed_gradient"])
def test_real_xtb_numerical_failure_invalidates_saved_state_on_timeout(
    monkeypatch, water_xyz_path, failure
):
    from pathlib import Path
    from subprocess import CompletedProcess

    monkeypatch.setattr(actual_xtb_backend, "_resolve_xtb_binary", lambda: "/test/xtb")
    system = actual_xtb_backend.XTBMolecularSystem(str(water_xyz_path))
    calls = 0

    def fake_xtb_process(command, *, cwd, **kwargs):
        nonlocal calls
        calls += 1
        energy = "nan" if calls == 2 and failure == "nonfinite" else "-1.0"
        rows = 1 if calls == 2 and failure == "malformed_gradient" else 3
        gradient = "$grad\nSCF energy = " + energy + "\n" + "0 0 0\n" * rows + "$end\n"
        Path(cwd, "gradient").write_text(gradient)
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(actual_xtb_backend.subprocess, "run", fake_xtb_process)
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        try:
            calc(positions * 0.5)
        except ValueError:
            time.sleep(5)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task(xyz_path=str(water_xyz_path)))
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert out["error"].startswith("backend_numerical_error:")
    assert out["worker_guard"]["kind"] == "worker_guard_timeout"


def test_calculator_plain_value_error_invalidates_result(fake_xtb_backend, monkeypatch):
    system = fake_xtb_backend("/tmp/x.xyz")

    def invalid_gradient(positions):
        raise ValueError("expected 3 gradient rows, found 1")

    system.compute = invalid_gradient
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _guard(_task())
    assert out["result"] is None
    assert out["error_kind"] == "optimizer_error"
    assert "expected 3 gradient rows" in out["error"]


def _diagnostic_guard(task, **overrides):
    options = dict(task_timeout_seconds=2, max_task_rss_bytes=0,
                   guard_poll_seconds=0.01, kill_grace_seconds=0.02)
    options.update(overrides)
    return worker._run_optimization_task_guarded(
        {**task, "_capture_diagnostics": True}, **options
    )


def _scientific_payload(payload):
    return {key: payload[key] for key in ("result", "error", "error_kind")}


def test_diagnostics_preserve_exact_results_calls_and_convergence_metrics(
    fake_xtb_backend, monkeypatch
):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        final = positions * 0.8
        calc(final)
        return final, 2

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    plain = worker._run_optimization_task(_task())
    traced = _diagnostic_guard(_task())
    assert _scientific_payload(traced) == _scientific_payload(plain)
    bundle = traced["_diagnostic_bundle"]
    assert bundle["attempted_calls"] == bundle["completed_calls"] == 2
    assert [frame["call_index"] for frame in bundle["frames"]] == [1, 2]
    assert len(bundle["scalar_rows"]) == 2
    first, second = bundle["scalar_rows"]
    assert first["energy_change_eh"] is None
    assert first["max_displacement_bohr"] is None
    assert second["energy_change_eh"] > 0
    assert second["max_displacement_bohr"] > 0
    assert second["max_force_eh_bohr"] > 0
    assert second["stage"] == 3
    np.testing.assert_array_equal(bundle["frames"][-1]["forces"],
                                  -np.array(traced["result"]["final_positions"]))


def test_converged_success_discards_diagnostics(fake_xtb_backend, monkeypatch):
    # Mock contract test, not a candidate optimizer or convergence shortcut.
    def completed_mock(positions, atomic_numbers, calc, converged, **kwargs):
        zeros = np.zeros_like(positions)
        calc(zeros)
        assert not converged()
        calc(zeros)
        assert converged()
        return zeros, 2

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: completed_mock)
    plain = worker._run_optimization_task(_task())
    traced = _diagnostic_guard(_task())
    assert _scientific_payload(traced) == _scientific_payload(plain)
    assert traced["result"]["converged"] is True
    assert "_diagnostic_bundle" not in traced


def test_first_backend_failure_captures_attempt_before_call(fake_xtb_backend, monkeypatch):
    system = fake_xtb_backend("/tmp/x.xyz")

    def fail(positions):
        raise ValueError("mock SCF failure")

    system.compute = fail
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)
    monkeypatch.setattr(worker, "_get_minimize_func",
                        lambda spec: lambda positions, atomic_numbers, calc, **kw: calc(positions * 0.5))
    traced = _diagnostic_guard(_task())
    bundle = traced["_diagnostic_bundle"]
    assert traced["result"] is None
    assert bundle["attempted_calls"] == 1
    assert bundle["completed_calls"] == 0
    assert bundle["frames"] == []
    assert bundle["last_attempt"]["call_index"] == 1
    np.testing.assert_array_equal(bundle["last_attempt"]["positions"], system.initial_positions * 0.5)
    assert "mock SCF failure" in bundle["exception"]["traceback"]
    assert "raise ValueError" in bundle["exception"]["traceback"]


@pytest.mark.parametrize("coordinates", [np.full((3, 3), np.nan), np.zeros((2, 3))])
def test_bad_first_input_is_diagnostic_without_counting_a_backend_call(
    fake_xtb_backend, monkeypatch, coordinates
):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(coordinates)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _diagnostic_guard(_task())
    bundle = out["_diagnostic_bundle"]
    assert bundle["attempted_calls"] == bundle["completed_calls"] == 0
    assert bundle["invalid_input"]["backend_call_started"] is False
    assert bundle["invalid_input"]["shape"] == list(coordinates.shape)
    if np.isnan(coordinates).any():
        assert bundle["invalid_input"]["values_flat"] == [None] * 9


@pytest.mark.parametrize("caught_failure", [False, True])
def test_timeout_diagnostic_keeps_pending_input_and_prior_force_frame(
    fake_xtb_backend, monkeypatch, caught_failure
):
    system = fake_xtb_backend("/tmp/x.xyz")
    original = system.compute
    count = 0

    def compute(positions):
        nonlocal count
        count += 1
        if count == 2:
            if caught_failure:
                raise ValueError("caught mock SCF failure")
            time.sleep(5)
        return original(positions)

    system.compute = compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        try:
            calc(positions * 0.5)
        except ValueError:
            time.sleep(5)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    out = _diagnostic_guard(_task(), task_timeout_seconds=0.25)
    bundle = out["_diagnostic_bundle"]
    assert bundle["attempted_calls"] == 2
    assert bundle["completed_calls"] == 1
    assert len(bundle["frames"]) == 1
    assert bundle["scalar_rows"][-1]["stage"] == 1
    np.testing.assert_array_equal(bundle["last_attempt"]["positions"], system.initial_positions * 0.5)
    if caught_failure:
        assert out["result"] is None
        assert "caught mock SCF failure" in bundle["exception"]["traceback"]
    else:
        assert out["result"]["stop_reason"] == "time_limit"
        assert out["result"]["n_steps"] == 2


@pytest.mark.parametrize("loss", ["crash", "memory_error", "rss"])
def test_child_loss_preserves_bounded_diagnostics_without_salvaging_score(
    fake_xtb_backend, monkeypatch, loss
):
    import multiprocessing as mp
    import os
    ready = mp.get_context("fork" if hasattr(os, "fork") else "spawn").Event()

    def minimize(positions, atomic_numbers, calc, **kwargs):
        calc(positions)
        ready.set()
        if loss == "crash":
            os._exit(23)
        if loss == "memory_error":
            raise MemoryError("mock allocation failed")
        time.sleep(5)

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    monkeypatch.setattr(worker, "_process_group_rss_bytes", lambda pid: 1000 if ready.is_set() else 0)
    out = _diagnostic_guard(_task(), max_task_rss_bytes=100 if loss == "rss" else 0)
    assert out["result"] is None
    assert out["error_kind"] == ("infrastructure_error" if loss == "crash" else "optimizer_error")
    bundle = out["_diagnostic_bundle"]
    assert bundle["completed_calls"] == 1
    assert len(bundle["frames"]) == 1
    if loss == "memory_error":
        assert "mock allocation failed" in bundle["exception"]["traceback"]


def test_source_traceback_resolves_exact_hashed_candidate_lines(fake_xtb_backend):
    import hashlib
    source = (
        "def minimize_func(positions, atomic_numbers, calc, **kwargs):\n"
        "    calc(positions)\n"
        "    raise RuntimeError('exact source failure')\n"
    )
    task = {**_task(), "optimizer_spec": {"kind": "source", "source": source}}
    out = _diagnostic_guard(task)
    bundle = out["_diagnostic_bundle"]
    digest = hashlib.sha256(source.encode()).hexdigest()
    assert bundle["source_sha256"] == digest
    assert bundle["source_map"] == {f"<_optimizer_source_{digest}>": digest}
    assert "line 3, in minimize_func" in bundle["exception"]["traceback"]
    assert "raise RuntimeError('exact source failure')" in bundle["exception"]["traceback"]


def test_diagnostic_io_error_does_not_change_scientific_result(
    fake_xtb_backend, stub_minimize_func, monkeypatch
):
    plain = worker._run_optimization_task(_task())

    def unavailable(*args):
        raise OSError("mock full diagnostic disk")

    monkeypatch.setattr(worker.TaskCapture, "attempted", unavailable)
    traced = _diagnostic_guard(_task())
    assert _scientific_payload(traced) == _scientific_payload(plain)
    assert traced["_diagnostic_bundle"]["capture_incomplete"]


def test_force_limit_tracing_preserves_exact_count_and_result(fake_xtb_backend, monkeypatch):
    def minimize(positions, atomic_numbers, calc, **kwargs):
        for step in range(10):
            calc(positions * (1 - 0.01 * step))

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: minimize)
    task = {**_task(), "max_steps": 3}
    plain = worker._run_optimization_task(task)
    traced = _diagnostic_guard(task)
    assert _scientific_payload(traced) == _scientific_payload(plain)
    assert traced["result"]["stop_reason"] == "force_call_limit"
    assert traced["_diagnostic_bundle"]["attempted_calls"] == 3
    assert traced["_diagnostic_bundle"]["completed_calls"] == 3


def test_timeout_on_first_force_call_still_has_input(fake_xtb_backend, monkeypatch):
    system = fake_xtb_backend("/tmp/x.xyz")
    system.compute = lambda positions: time.sleep(5)
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)
    monkeypatch.setattr(worker, "_get_minimize_func",
                        lambda spec: lambda positions, atomic_numbers, calc, **kw: calc(positions))
    out = _diagnostic_guard(_task(), task_timeout_seconds=0.25)
    assert out["result"] is None
    bundle = out["_diagnostic_bundle"]
    assert bundle["attempted_calls"] == 1
    assert bundle["completed_calls"] == 0
    assert bundle["frames"] == []
    np.testing.assert_array_equal(bundle["last_attempt"]["positions"], system.initial_positions)


def test_parent_upload_failure_still_stores_science_and_retries_when_busy(monkeypatch):
    from distributed_validate import diagnostic_upload
    calls = []
    stored = []
    fake_redis = types.SimpleNamespace(
        Redis=lambda **kwargs: types.SimpleNamespace(ping=lambda: True),
        exceptions=types.SimpleNamespace(RedisError=RuntimeError),
    )
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setattr(worker, "setup_logging", lambda *args, **kwargs: worker.logger)
    monkeypatch.setattr(worker, "_preflight_backend_import", lambda *args: None)
    monkeypatch.setattr(worker, "_cap_in_process_threads", lambda *args: None)
    monkeypatch.setattr(worker, "describe_optimizer_spec", lambda *args: "mock")
    monkeypatch.setattr(worker, "load_next_optimization_task",
                        lambda *args, **kwargs: ("task123", {**_task(), "_capture_diagnostics": True}))
    monkeypatch.setattr(diagnostic_upload, "capture_enabled", lambda: False)

    def retry():
        calls.append("retry")
        raise RuntimeError("mock unavailable collector")

    def guarded(task, **kwargs):
        assert task["task_id"] == "task123"
        assert task["_capture_diagnostics"] is False  # untrusted task flag overridden
        return {"result": {"n_steps": 1}, "error": None, "error_kind": None,
                "_diagnostic_bundle": {"frames": ["private"]}}

    monkeypatch.setattr(diagnostic_upload, "retry_pending", retry)
    monkeypatch.setattr(worker, "_run_optimization_task_guarded", guarded)
    monkeypatch.setattr(worker, "store_optimization_result", lambda conn, task_id, result: stored.append(result))
    monkeypatch.setattr(worker, "finish_worker_task", lambda *args: calls.append("finish"))

    def bad_publish(*args):
        raise AttributeError("mock malformed collector response")

    monkeypatch.setattr(diagnostic_upload, "publish", bad_publish)
    worker.start_worker("unused", 123, ["xtb"], max_tasks=1)
    assert len(stored) == 1
    assert stored[0]["result"] == {"n_steps": 1}
    assert stored[0]["error"] is None
    assert stored[0]["diagnostics"]["availability"] == "unavailable"
    assert "_diagnostic_bundle" not in stored[0]
    assert calls == ["retry", "finish", "retry"]


def test_parent_strips_capture_even_if_publisher_forgets():
    payload = {"result": {"n_steps": 3}, "_diagnostic_bundle": {"frames": []}}
    uploader = types.SimpleNamespace(publish=lambda task, result: result)
    assert worker._publish_task_diagnostics(uploader, {}, payload) == {"result": {"n_steps": 3}}


def test_collector_configuration_stays_in_parent(monkeypatch):
    import os
    keys = ["GIGAOPT_DIAGNOSTICS_" + suffix for suffix in ("URL", "TOKEN_FILE", "SPOOL_DIR")]
    for key in keys:
        monkeypatch.setenv(key, "parent-only-test-value")
    monkeypatch.setattr(worker, "_run_optimization_task", lambda task: {
        "config_visible": [key for key in keys if key in os.environ]})
    result = _diagnostic_guard(_task())
    assert result["config_visible"] == []
    assert all(os.environ[key] == "parent-only-test-value" for key in keys)
