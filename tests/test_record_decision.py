"""Decision records use complete evaluator evidence and preserve strict split gates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.record_decision import build_record, read_evaluation, write_record
from scripts.score_log import load_last_json


CANDIDATE = "a" * 40
ANCHOR = "b" * 40


def evaluation(split="train", commit=CANDIDATE, **updates):
    result = dict(
        timestamp="2026-09-07T12:00:00Z", evaluation_id=f"eval-{split}-{commit[:7]}",
        program_id=f"program-{commit}", candidate_commit=commit, split=split,
        release_id="fixture-gfn2", status="complete", mean_rel_steps=0.8,
        mean_rel_energy=1.0, converged_count=1, molecule_count=2,
        converged_fraction=0.5, internal_error_count=0, infrastructure_retries=0,
        stop_reason_counts={"converged": 1, "force_call_limit": 1},
        is_valid=1, validity_reason="valid", evaluation_log=f"{split}.log",
        evaluation_json=f"{split}-{commit}.json",
    )
    result.update(updates)
    return result


def record(**updates):
    arguments = dict(
        cycle=2, candidate_commit=CANDIDATE, anchor_commit=ANCHOR,
        description="replace curvature transport", train=evaluation(),
        validation=evaluation("valid"),
        anchor_train=evaluation(commit=ANCHOR, mean_rel_steps=1),
        anchor_valid=evaluation("valid", commit=ANCHOR, mean_rel_steps=1),
    )
    arguments.update(updates)
    return build_record(**arguments)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


@pytest.mark.parametrize("train_steps,valid_steps,decision", [
    (0.9999, 0.9998, "keep"),  # Train admits exactly 1e-4.
    (0.9998, 0.9999, "non_generalizable"),  # Valid must exceed 1e-4.
    (0.99990001, 0.8, "discard"),
    (0.8, 1.0, "non_generalizable"),
])
def test_original_train_and_valid_boundaries(train_steps, valid_steps, decision):
    result = record(train=evaluation(mean_rel_steps=train_steps),
                    validation=evaluation("valid", mean_rel_steps=valid_steps))
    assert result["decision"] == decision


@pytest.mark.parametrize("energy,errors,is_valid,expected", [
    (0.999999999, 0, 1, "keep"),
    (0.9999999989, 0, 1, "invalid"),
    (1.1, 1, 1, "invalid"),
    (1.1, 0, 0, "invalid"),
])
def test_validity_requires_energy_and_zero_internal_errors(energy, errors, is_valid, expected):
    assert record(train=evaluation(mean_rel_energy=energy, internal_error_count=errors,
                                   is_valid=is_valid))["decision"] == expected


def test_convergence_diagnostic_does_not_gate_acceptance():
    result = record(train=evaluation(converged_count=0, converged_fraction=0,
                                    stop_reason_counts={"force_call_limit": 1, "time_limit": 1}))
    assert result["decision"] == "keep"
    assert result["train_converged_fraction"] == 0
    assert json.loads(result["train_stop_reason_counts"])["time_limit"] == 1


@pytest.mark.parametrize("which", ["train", "valid", "missing_valid"])
def test_pending_is_unscored_and_never_rejected(tmp_path, which):
    updates = {"train": evaluation(status="pending")} if which == "train" else {
        "validation": None if which == "missing_valid" else evaluation("valid", status="pending")}
    result = write_record(tmp_path, record(**updates))
    assert result["decision"] == "pending"
    assert result["accepted"] == ""
    assert not (tmp_path / "non_generalizable.tsv").exists()
    assert not (tmp_path / "generalizable.tsv").exists()


def test_pending_can_resolve_same_cycle_without_duplicate_decision(tmp_path):
    pending = record(validation=evaluation("valid", status="pending"))
    write_record(tmp_path, pending)
    write_record(tmp_path, pending)
    completed = record()
    write_record(tmp_path, completed)
    write_record(tmp_path, completed)
    assert [r["decision"] for r in rows(tmp_path / "results.tsv")] == ["pending", "keep"]
    assert len(rows(tmp_path / "generalizable.tsv")) == 1


def test_rejection_reason_preserves_both_valid_failures(tmp_path):
    result = record(validation=evaluation("valid", is_valid=0, mean_rel_energy=0.9,
                                          mean_rel_steps=1.1, validity_reason="insufficient energy"))
    write_record(tmp_path, result)
    assert "valid split invalid" in result["validity_reason"]
    assert "improvement is not greater" in result["validity_reason"]
    assert len(rows(tmp_path / "non_generalizable.tsv")) == 1


def test_completed_cycle_cannot_be_rewritten(tmp_path):
    write_record(tmp_path, record())
    with pytest.raises(ValueError, match="different completed decision"):
        write_record(tmp_path, record(validation=evaluation("valid", mean_rel_steps=1)))
    assert len(rows(tmp_path / "results.tsv")) == 1


@pytest.mark.parametrize("updates,match", [
    ({"anchor_train": None}, "Both train and valid anchor"),
    ({"anchor_valid": evaluation("valid", commit=ANCHOR, status="pending")}, "complete and valid"),
    ({"anchor_valid": evaluation("valid", commit=ANCHOR, is_valid=0)}, "complete and valid"),
    ({"validation": evaluation("valid", program_id="other")}, "same candidate program"),
    ({"validation": evaluation("valid", release_id="other")}, "same release"),
])
def test_refuses_missing_invalid_or_mismatched_anchors_and_provenance(updates, match):
    with pytest.raises(ValueError, match=match):
        record(**updates)


def test_starting_anchor_has_no_improvement_claim(tmp_path):
    result = record(cycle=1, anchor_commit=CANDIDATE, anchor_train=None, anchor_valid=None,
                    starting_anchor=True)
    assert result["decision"] == "anchor"
    assert result["train_improvement_vs_anchor"] == ""
    write_record(tmp_path, result)
    assert rows(tmp_path / "generalizable.tsv")[0]["decision"] == "anchor"


@pytest.mark.parametrize("updates,match", [
    ({"candidate_commit": ANCHOR}, "commit does not match"),
    ({"split": "test"}, "expected split"),
    ({"evaluation_id": None}, "missing evaluation_id"),
    ({"mean_rel_steps": None}, "Missing or non-numeric"),
    ({"mean_rel_energy": float("nan")}, "Non-finite"),
    ({"status": "failed"}, "unsupported evaluation status"),
    ({"stop_reason_counts": None}, "missing stop_reason_counts"),
])
def test_malformed_evidence_is_not_a_scientific_rejection(tmp_path, updates, match):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(evaluation(**updates)))
    with pytest.raises(ValueError, match=match):
        read_evaluation(path, "train", CANDIDATE)
    assert not (tmp_path / "results.tsv").exists()


def test_reader_accepts_document_and_append_only_log(tmp_path):
    path = tmp_path / "run.log"
    path.write_text(json.dumps(evaluation(), indent=2))
    assert load_last_json(path)["mean_rel_steps"] == 0.8
    path.write_text("human log\n" + json.dumps(evaluation(mean_rel_steps=0.7)) + "\n")
    assert load_last_json(path)["mean_rel_steps"] == 0.7


def test_reader_uses_evaluator_dataset_release_id_and_pending_null_metrics(tmp_path):
    payload = evaluation(status="pending", mean_rel_steps=None, mean_rel_energy=None,
                         converged_count=None, converged_fraction=None, is_valid=None)
    payload["dataset_release_id"] = payload.pop("release_id")
    path = tmp_path / "pending.json"
    path.write_text(json.dumps(payload))
    result = read_evaluation(path, "train", CANDIDATE)
    assert result["release_id"] == "fixture-gfn2"
    assert record(train=result)["decision"] == "pending"
    write_record(tmp_path, record(train=result))
    write_record(tmp_path, record(train=result))
    assert len(rows(tmp_path / "results.tsv")) == 1
    assert rows(tmp_path / "results.tsv")[0]["train_is_valid"] == ""


def test_last_generalizable_keep_controls_next_anchor(tmp_path):
    write_record(tmp_path, record())
    with pytest.raises(ValueError, match="last generalizable keep"):
        write_record(tmp_path, record(cycle=3))
    assert len(rows(tmp_path / "results.tsv")) == 1


def test_changed_evaluation_of_same_anchor_is_not_silently_substituted(tmp_path):
    write_record(tmp_path, record())
    changed_anchor = evaluation(evaluation_id="unrelated-evaluation")
    with pytest.raises(ValueError, match="Anchor evaluation must match"):
        write_record(tmp_path, record(cycle=3, anchor_commit=CANDIDATE,
                                     anchor_train=changed_anchor,
                                     anchor_valid=evaluation("valid")))


def test_cli_derives_numbers_without_metric_arguments(tmp_path):
    arguments = []
    for option, payload in (("train", evaluation()), ("valid", evaluation("valid")),
                            ("anchor-train", evaluation(commit=ANCHOR, mean_rel_steps=1)),
                            ("anchor-valid", evaluation("valid", commit=ANCHOR, mean_rel_steps=1))):
        path = tmp_path / f"{option}.json"
        path.write_text(json.dumps(payload))
        arguments.extend([f"--{option}-json", str(path)])
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/record_decision.py"),
               "--cycle", "2", "--candidate-commit", CANDIDATE, "--anchor-commit", ANCHOR,
               "--description", "curvature transport, new update", "--output-dir", str(tmp_path),
               *arguments]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    row = json.loads(result.stdout)
    assert row["decision"] == "keep"
    assert row["train_improvement_vs_anchor"] == "0.2"
    assert rows(tmp_path / "results.tsv")[0]["description"] == "curvature transport, new update"
