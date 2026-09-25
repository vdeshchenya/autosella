"""Extra tests for distributed_validate.worker — covers helpers that the
main test_distributed_worker.py does not touch directly."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from distributed_validate import worker


# ── _backend_module_name ────────────────────────────────────────────────────

def test_backend_module_name_xtb():
    assert worker._backend_module_name("xtb") == "xtb_molecular_system"


def test_backend_module_name_ff_is_rejected():
    with pytest.raises(ValueError, match="Invalid mode"):
        worker._backend_module_name("ff")


def test_backend_module_name_invalid_raises():
    with pytest.raises(ValueError, match="Invalid mode"):
        worker._backend_module_name("mm")


# ── _preflight_backend_import ───────────────────────────────────────────────

def test_preflight_with_importable_backend(monkeypatch):
    """Successful preflight returns None (doesn't raise)."""
    module = types.ModuleType("xtb_molecular_system")
    module._resolve_xtb_binary = lambda: "/usr/bin/xtb"
    monkeypatch.setitem(sys.modules, "xtb_molecular_system", module)
    worker._preflight_backend_import("xtb")


def test_preflight_wraps_import_failure_with_context(monkeypatch):
    """If backend import fails, preflight raises RuntimeError mentioning the
    module name and sys.executable (helps debug interpreter mismatches)."""
    import importlib

    real_import = importlib.import_module

    def faux_import(name, *args, **kwargs):
        if name == "xtb_molecular_system":
            raise ImportError("forced failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", faux_import)

    with pytest.raises(RuntimeError, match="xtb_molecular_system"):
        worker._preflight_backend_import("xtb")


# ── _get_minimize_func caches by spec key ───────────────────────────────────

def test_get_minimize_func_caches(monkeypatch):
    worker._OPTIMIZER_CACHE.clear()

    call_count = {"n": 0}

    def fake_load(spec):
        call_count["n"] += 1
        return lambda *a, **k: None

    monkeypatch.setattr(worker, "load_minimize_func", fake_load)

    spec = {"module_name": "sample_optimizer"}
    fn1 = worker._get_minimize_func(spec)
    fn2 = worker._get_minimize_func(spec)
    # load_minimize_func called only once → cache reused
    assert call_count["n"] == 1
    assert fn1 is fn2


def test_get_minimize_func_different_specs(monkeypatch):
    worker._OPTIMIZER_CACHE.clear()

    call_count = {"n": 0}
    sentinels = []

    def fake_load(spec):
        call_count["n"] += 1
        sent = object()
        sentinels.append(sent)
        return sent

    monkeypatch.setattr(worker, "load_minimize_func", fake_load)

    worker._get_minimize_func({"module_name": "foo.bar"})
    worker._get_minimize_func({"module_name": "baz.qux"})
    # Two different spec keys → cache miss twice
    assert call_count["n"] == 2


def test_run_task_rejects_legacy_ff_before_backend_setup(monkeypatch):
    def forbidden_setup(**kwargs):
        raise AssertionError("legacy backend must not be imported")

    monkeypatch.setattr(worker, "_get_system", forbidden_setup)
    with pytest.raises(ValueError, match="Invalid mode"):
        worker._run_optimization_task({"mode": "ff"})


# ── _build_system invalid mode ──────────────────────────────────────────────

def test_build_system_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid mode"):
        worker._build_system("bogus", {"xyz_path": "/x"}, num_threads=1)


# ── budget enforcement in the distributed path ────────────────────────────

@pytest.fixture
def fake_xtb_backend(monkeypatch):
    """xTB backend stub with a quadratic potential."""
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
    worker._SYSTEM_CACHE.clear()
    worker._OPTIMIZER_CACHE.clear()
    return FakeSystem


def _xtb_task_with_budget(max_steps, xyz_path="/tmp/x.xyz"):
    return {
        "mode": "xtb",
        "mol_name": "budget_mol",
        "baseline": {
            "n_steps": 10,
            "improvement": 100.0,
            "initial_energy": 0.0757**2 + 0.0586**2,
            "final_energy": 0.0757**2 + 0.0586**2 - 100.0,
            "n_atoms": 3,
            "xyz_path": xyz_path,
        },
        "max_steps": max_steps,
        "optimizer_spec": {"kind": "stub"},
        "num_threads": 1,
    }


def test_run_task_budget_exhaustion_scores_last_calculation(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    system = fake_xtb_backend(str(water_xyz_path))
    actual_calls = []
    compute = system.compute

    def counted_compute(positions):
        actual_calls.append(positions.copy())
        return compute(positions)

    system.compute = counted_compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def over_budget(positions, atomic_numbers, calc, max_force_calls, converged):
        # Ordinary optimizer exception recovery must not swallow harness stops.
        try:
            for index in range(100):
                calc(positions * (index + 1))
        except Exception:
            raise AssertionError("budget stop was exposed as an optimizer error")
        return positions, 100

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: over_budget)
    out = worker._run_optimization_task(
        _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
    )
    assert out["error"] is None
    result = out["result"]
    assert result["stop_reason"] == "force_call_limit"
    assert result["converged"] is False
    assert result["n_steps"] == result["n_completed_steps"] == len(actual_calls) == 5
    np.testing.assert_array_equal(result["final_positions"], actual_calls[-1])
    assert result["final_energy"] == compute(actual_calls[-1])[0]


def test_run_task_budget_exact_is_ok(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    """Calling calc() exactly max_steps times succeeds."""
    def respectful(positions, atomic_numbers, calc, max_force_calls, converged):
        for _ in range(max_force_calls):
            calc(positions)
        return positions, max_force_calls

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: respectful)

    out = worker._run_optimization_task(
        _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
    )
    assert out["error"] is None
    assert out["result"]["n_steps"] == 5


def test_run_task_preserves_negative_relative_energy(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    """Ending above the starting energy must be penalized, not clipped to zero."""

    def uphill(positions, atomic_numbers, calc, max_force_calls, converged):
        final_positions = positions * 2.0
        calc(final_positions)
        return final_positions, 1

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: uphill)

    out = worker._run_optimization_task(
        _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
    )
    assert out["error"] is None
    assert out["result"]["rel_energy"] < 0.0


def test_run_task_rejects_force_call_underreporting(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def underreport(positions, atomic_numbers, calc, max_force_calls, converged):
        calc(positions)
        calc(positions)
        return positions, 1

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: underreport)

    with pytest.raises(
        RuntimeError,
        match=r"^force_call_count_mismatch:measured=2:reported=1$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_rejects_force_call_overreporting(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def overreport(positions, atomic_numbers, calc, max_force_calls, converged):
        calc(positions)
        calc(positions)
        return positions, 3

    monkeypatch.setattr(worker, "_get_minimize_func", lambda spec: overreport)

    with pytest.raises(
        RuntimeError,
        match=r"^force_call_count_mismatch:measured=2:reported=3$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_rejects_noninteger_force_call_report(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def noninteger_report(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        return positions, 1.0

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: noninteger_report
    )

    with pytest.raises(
        RuntimeError,
        match=r"^force_call_count_mismatch:reported_count_is_not_an_integer$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_rejects_boolean_force_call_report(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def boolean_report(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        return positions.copy(), True

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: boolean_report
    )

    with pytest.raises(
        RuntimeError,
        match=r"^force_call_count_mismatch:reported_count_is_not_an_integer$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_rejects_numpy_boolean_force_call_report(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def numpy_boolean_report(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        return positions.copy(), np.bool_(True)

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: numpy_boolean_report
    )

    with pytest.raises(
        RuntimeError,
        match=r"^force_call_count_mismatch:reported_count_is_not_an_integer$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_accepts_numpy_integer_force_call_report(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def numpy_integer_report(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        return positions.copy(), np.int64(1)

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: numpy_integer_report
    )

    out = worker._run_optimization_task(
        _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
    )
    assert out["error"] is None
    assert out["result"]["n_steps"] == 1


def test_run_task_rejects_returned_geometry_not_last_evaluated(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def mismatched_geometry(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        calc(positions)
        returned = positions.copy()
        returned[0, 0] += np.finfo(float).eps
        return returned, 1

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: mismatched_geometry
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"^returned_geometry_mismatch:"
            r"last_evaluated_geometry_required$"
        ),
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_rejects_return_without_successful_force_call(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    def no_force_call(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        return positions.copy(), 0

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: no_force_call
    )

    with pytest.raises(
        RuntimeError,
        match=r"^returned_geometry_mismatch:no_successful_force_call$",
    ):
        worker._run_optimization_task(
            _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
        )


def test_run_task_reuses_last_calc_energy_without_hidden_compute(
    fake_xtb_backend, monkeypatch, water_xyz_path
):
    system = fake_xtb_backend(str(water_xyz_path))
    compute_calls = 0
    original_compute = system.compute

    def counted_compute(positions):
        nonlocal compute_calls
        compute_calls += 1
        if compute_calls > 1:
            raise AssertionError("unexpected uncounted final compute")
        return original_compute(positions)

    system.compute = counted_compute
    monkeypatch.setattr(worker, "_get_system", lambda **kwargs: system)

    def evaluated_return(
        positions, atomic_numbers, calc, max_force_calls, converged
    ):
        final_positions = positions * 0.5
        calc(final_positions)
        return final_positions.copy(), 1

    monkeypatch.setattr(
        worker, "_get_minimize_func", lambda spec: evaluated_return
    )

    out = worker._run_optimization_task(
        _xtb_task_with_budget(max_steps=5, xyz_path=str(water_xyz_path))
    )
    assert out["error"] is None
    assert compute_calls == 1


def test_build_system_selects_gfn2_explicitly(monkeypatch):
    seen = {}
    module = types.ModuleType("xtb_molecular_system")

    def make_system(path, **kwargs):
        seen.update(path=path, **kwargs)
        return object()

    module.MolecularSystem = make_system
    monkeypatch.setitem(sys.modules, "xtb_molecular_system", module)
    worker._build_system("xtb", {"xyz_path": "/prepared/start.xyz"}, 3)
    assert seen == {"path": "/prepared/start.xyz", "method": "GFN2-xTB", "num_threads": 3}
