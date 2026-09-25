"""Tests for validate.score_results -- the promotion gate -- and the run log.

score_results() is the whole scoring contract: it decides `fitness` and
`is_valid` from the per-molecule worker results. These tests drive it directly
rather than through a wrapper, so a failure points at the gate itself.
"""

from __future__ import annotations

import json

import pytest

import validate as vd


def _result(**overrides):
    """One molecule's worker result; a valid, baseline-matching run by default."""
    base = {
        "mol_name": "mol",
        "n_steps": 50,
        "max_steps": 50,
        "rel_steps": 1.0,
        "rel_energy": 1.0,
        "energy_delta_kcal_mol": 0.0,
        "converged": True,
    }
    base.update(overrides)
    return base


# ── sentinel paths: no measurement worth reporting ──────────────────────────

def test_empty_results_returns_sentinel():
    out = vd.score_results([], num_errors=0)
    assert out["fitness"] == vd.INVALID_FITNESS
    assert out["mean_rel_steps"] == vd.INVALID_FITNESS
    assert out["mean_rel_energy"] == -1.0
    assert out["max_final_energy_delta_kcal_mol"] == -1.0
    assert out["converged"] == 0.0
    assert out["is_valid"] == 0


def test_any_errors_returns_sentinel():
    out = vd.score_results([_result(n_steps=5, rel_steps=0.1)], num_errors=3)
    assert out["mean_rel_steps"] == vd.INVALID_FITNESS
    assert out["is_valid"] == 0


# ── aggregation ─────────────────────────────────────────────────────────────

def test_success_aggregation():
    out = vd.score_results(
        [
            _result(n_steps=40, rel_steps=0.8, energy_delta_kcal_mol=-0.25),
            _result(n_steps=50, rel_steps=1.0, energy_delta_kcal_mol=0.5),
        ],
        num_errors=0,
    )
    assert abs(out["mean_rel_steps"] - 0.9) < 1e-9
    assert abs(out["mean_rel_energy"] - 1.0) < 1e-9
    assert abs(out["max_final_energy_delta_kcal_mol"] - 0.5) < 1e-9
    assert abs(out["converged"] - 1.0) < 1e-9
    assert out["is_valid"] == 1


# ── the energy gate ─────────────────────────────────────────────────────────

def test_is_invalid_with_low_mean_rel_energy():
    """mean_rel_energy < 1.0 (under-relaxed on average) is invalid."""
    out = vd.score_results([_result(rel_energy=0.5, energy_delta_kcal_mol=0.999)], num_errors=0)
    assert out["is_valid"] == 0
    assert out["invalid_reason"] == "energy_below_baseline"
    # Honest fitness: no errors, inside budget, converged -- so mean_rel_steps is
    # a real measurement and is reported. `is_valid`, not the fitness value, is
    # the promotion gate.
    assert out["fitness"] == 1.0
    assert out["mean_rel_steps"] == 1.0


def test_energy_delta_does_not_gate():
    """A large per-molecule energy_delta stays valid while mean_rel_energy >= 1.0."""
    out = vd.score_results([_result(energy_delta_kcal_mol=5.0)], num_errors=0)
    assert out["is_valid"] == 1
    assert abs(out["max_final_energy_delta_kcal_mol"] - 5.0) < 1e-9


def test_float_tolerance_admits_the_unmodified_baseline():
    """Summing 250 exact 1.0s lands a hair under 1.0; that must stay valid."""
    out = vd.score_results([_result(rel_energy=1.0 - 4e-16) for _ in range(250)], num_errors=0)
    assert out["is_valid"] == 1


def test_tolerance_does_not_admit_real_under_relaxation():
    out = vd.score_results([_result(rel_energy=1.0 - 1e-6)], num_errors=0)
    assert out["is_valid"] == 0


# ── budget and convergence guards ───────────────────────────────────────────

def test_exceeded_budget_is_invalid():
    out = vd.score_results([_result(n_steps=100, max_steps=50, rel_steps=2.0)], num_errors=0)
    assert out["is_valid"] == 0
    assert out["invalid_reason"] == "internal_error"


def test_early_stop_without_convergence_is_diagnostic():
    out = vd.score_results([_result(n_steps=10, rel_steps=0.2, converged=False)], 0)
    assert out["is_valid"] == 1
    assert out["fitness"] == 0.2
    assert out["converged_fraction"] == 0


def test_budget_exhaustion_scores_last_state():
    out = vd.score_results([_result(converged=False, stop_reason="force_call_limit")], 0)
    assert out["is_valid"] == 1
    assert out["fitness"] == 1
    assert out["stop_reason_counts"] == {"force_call_limit": 1}


def test_converged_on_the_final_allowed_call_is_valid():
    out = vd.score_results([_result()], num_errors=0)
    assert out["is_valid"] == 1
    assert out["fitness"] == 1.0


def test_partial_convergence_does_not_gate():
    out = vd.score_results([_result(), _result(converged=False)], 0)
    assert out["is_valid"] == 1
    assert out["converged_count"] == 1
    assert out["converged_fraction"] == .5


@pytest.mark.parametrize("field,value", [("rel_steps", float("nan")), ("rel_energy", float("inf")),
                                         ("n_steps", -1), ("n_steps", 2.5)])
def test_malformed_metrics_invalidate(field, value):
    out = vd.score_results([_result(**{field: value})], 0)
    assert out["is_valid"] == 0
    assert out["validity_reason"] == "internal_error"


@pytest.mark.parametrize("result", [None, [], _result(stop_reason=[]), _result(converged="false")])
def test_malformed_diagnostics_invalidate_without_crashing(result):
    out = vd.score_results([result], 0)
    assert out["is_valid"] == 0
    assert out["validity_reason"] == "internal_error"


def test_malformed_complete_result_produces_serializable_terminal_record():
    out = vd.build_evaluation_record({"status": "complete", "results": [_result(rel_energy=float("nan"))],
        "num_errors": 0, "errors": []}, split="train", molecule_count=1)
    assert out["status"] == "complete"
    assert out["is_valid"] == 0
    assert out["internal_error_count"] == 1
    assert out["results"] == []
    json.dumps(out, allow_nan=False)


def test_pending_has_no_scientific_score():
    out = vd.build_evaluation_record({"status": "pending", "results": [], "num_errors": 0,
        "errors": [], "pending_reason": "redis_unavailable"}, split="train", molecule_count=457)
    assert out["is_valid"] is None
    assert out["fitness"] is None
    assert out["pending_reason"] == "redis_unavailable"


def test_failure_details_do_not_change_score_or_coverage():
    raw = {"status": "complete", "results": [_result()], "num_errors": 0, "errors": []}
    details = [{"mol_name": "mol", "diagnostics": {"availability": "unavailable"}}]
    before = vd.build_evaluation_record(raw, split="train", molecule_count=1)
    after = vd.build_evaluation_record({**raw, "failure_details": details}, split="train", molecule_count=1)
    assert after["failure_details"] == details
    for key in before:
        if key not in {"timestamp", "failure_details"}:
            assert after[key] == before[key]


def test_partial_coverage_cannot_be_scored():
    out = vd.build_evaluation_record({"status": "complete", "results": [_result()], "num_errors": 0,
        "errors": []}, split="train", molecule_count=2)
    assert out["status"] == "pending"
    assert out["fitness"] is None


def test_duplicate_results_cannot_be_scored():
    out = vd.build_evaluation_record({"status": "complete", "results": [_result(), _result()], "num_errors": 0,
        "errors": []}, split="train", molecule_count=2)
    assert out["status"] == "pending"
    assert out["fitness"] is None


# ── the run log the agent reads ─────────────────────────────────────────────

def test_append_run_log_writes_a_parseable_block(tmp_path):
    log_path = tmp_path / "run.log"
    results = [_result(mol_name="mol_a", n_steps=40, rel_steps=0.8)]
    score = vd.score_results(results, num_errors=0)
    vd._append_run_log(
        log_path,
        optimizer_description="pickled:deadbeef",
        split="valid",
        num_molecules=1,
        duration_s=12.3,
        results=results,
        errors=[],
        score=score,
    )
    content = log_path.read_text()
    # program.md tells the agent to locate its cycle by the `split:` line and
    # read metrics from the trailing JSON object.
    assert "split: valid" in content
    assert "mol_a" in content
    payload = json.loads(content.strip().splitlines()[-1])
    assert payload["mean_rel_steps"] == score["mean_rel_steps"]
    assert payload["is_valid"] == score["is_valid"]


# ── candidate serialization (what main() ships to the workers) ──────────────

def test_serialize_program_minimize_func_round_trips(tmp_path):
    import cloudpickle

    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        "def minimize_func(pos, atomic_numbers, calc, max_force_calls, converged):\n"
        "    return pos, 0\n",
        encoding="utf-8",
    )
    payload = vd.serialize_program_minimize_func(program_path)
    assert isinstance(payload, bytes) and payload

    # Workers reconstruct it by value, with no import of the candidate module.
    minimize_func = cloudpickle.loads(payload)
    assert minimize_func("pos", None, None, 1, None) == ("pos", 0)


def test_serialize_program_minimize_func_rejects_missing_callable(tmp_path):
    program_path = tmp_path / "candidate.py"
    program_path.write_text("not_minimize_func = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must define callable minimize_func"):
        vd.serialize_program_minimize_func(program_path)
