"""Production startup needs reference reproduction, beyond candidate validity."""
import copy
import hashlib
import json

import pytest

from isolated_runs import runner


def receipt(split="train"):
    return {
        "status": "complete", "split": split, "run_id": "fresh-run",
        "candidate_commit": "a" * 40, "source_sha256": "b" * 64,
        "program_id": "b" * 64, "timestamp": "2026-09-10T00:00:00Z",
        "evaluation_id": "fresh-" + split, "release_id": "prepared-v1",
        "is_valid": 1, "internal_error_count": 0,
        "mean_rel_steps": 1.0, "mean_rel_energy": 1.0,
        "molecule_count": 2, "num_results": 2, "num_errors": 0, "errors": [],
        "results": [
            {"mol_name": "synthetic-a", "rel_steps": 1.0, "rel_energy": 1.0},
            {"mol_name": "synthetic-b", "rel_steps": 1.0, "rel_energy": 1.0},
        ],
    }


def test_equal_means_do_not_hide_opposite_molecular_discrepancies():
    for field, delta in (("rel_steps", .1), ("rel_energy", .01)):
        result = receipt()
        result["results"][0][field] -= delta
        result["results"][1][field] += delta
        assert sum(row[field] for row in result["results"]) / 2 == 1.0
        with pytest.raises(runner.GateError, match="synthetic-a"):
            runner.check_anchor_reproduction(result, "train")


@pytest.mark.parametrize("change", [
    {"results": []}, {"molecule_count": 3}, {"num_results": 1},
    {"num_errors": 1}, {"errors": [["synthetic-a", "failure"]]},
    {"results": [{"mol_name": "synthetic-a", "rel_steps": 1, "rel_energy": 1}] * 2},
    {"results": [None, None]}, {"mean_rel_steps": .999999999999},
    {"mean_rel_energy": 1.0001}, {"mean_rel_energy": float("nan")},
])
def test_incomplete_inconsistent_or_mismatched_receipts_block_startup(change):
    with pytest.raises(runner.GateError, match="starting anchor/reference mismatch"):
        runner.check_anchor_reproduction({**receipt(), **change}, "train")


@pytest.mark.parametrize("value", [None, True, "1", float("nan"), float("inf"), .99, 1.01])
@pytest.mark.parametrize("field", ["rel_steps", "rel_energy"])
def test_bad_molecular_ratios_fail_closed(value, field):
    result = receipt()
    result["results"][0][field] = value
    with pytest.raises(runner.GateError):
        runner.check_anchor_reproduction(result, "train")


@pytest.mark.parametrize("offset", [-.5e-9, 0, .5e-9])
def test_energy_tolerance_is_symmetric_but_call_count_match_is_exact(offset):
    result = receipt()
    result["mean_rel_energy"] += offset
    for row in result["results"]:
        row["rel_energy"] += offset
    runner.check_anchor_reproduction(result, "train")
    result["results"][0]["rel_steps"] += 1e-12
    with pytest.raises(runner.GateError, match="force-call ratio"):
        runner.check_anchor_reproduction(result, "train")


@pytest.mark.parametrize("purpose", ["research", "verification"])
@pytest.mark.parametrize("mismatched_split", ["train", "valid"])
def test_both_observations_survive_mismatch_without_recording_false_anchor(
        tmp_path, monkeypatch, purpose, mismatched_split):
    program = tmp_path / "algo.py"
    program.write_text("# synthetic starting optimizer\n")
    source_hash = hashlib.sha256(program.read_bytes()).hexdigest()
    config = {"run_id": "fresh-run", "initial_commit": "a" * 40,
              "purpose": purpose, "expected_release_id": "prepared-v1"}
    requests = []

    def fake_evaluate(_client, **kwargs):
        split = kwargs["split"]
        requests.append(split)
        result = receipt(split)
        result.update(source_sha256=source_hash, program_id=source_hash)
        if split == mismatched_split:
            result["results"][0]["rel_steps"] = .8
            result["mean_rel_steps"] = .9
        runner.atomic_json(kwargs["output_json"], result)
        return copy.deepcopy(result)

    monkeypatch.setattr(runner, "evaluate", fake_evaluate)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        pytest.fail("mismatched or verification observations cannot create an anchor"))
    if purpose == "research":
        with pytest.raises(runner.GateError, match="reference mismatch"):
            runner.prepare_starting_anchors(object(), config, {}, program=program, workspace=tmp_path)
    else:
        anchors = runner.prepare_starting_anchors(object(), config, {}, program=program, workspace=tmp_path)
        assert anchors[mismatched_split]["mean_rel_steps"] == .9
    assert requests == ["train", "valid"]
    for split in requests:
        assert json.loads((tmp_path / "anchors" / f"{split}.json").read_text())["status"] == "complete"
