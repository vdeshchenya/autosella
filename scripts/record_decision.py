#!/usr/bin/env python
"""Generate append-only cycle and generalization records from evaluator JSON.

No metric arguments are accepted: values and provenance come from the evaluator.
Pending infrastructure evaluations are unscored and never enter either decision
table. Repeating a command is idempotent; a later completed evaluation can append
a resolution for the same pending cycle.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import json
from pathlib import Path
from typing import Any

try:
    from .score_log import load_last_json
except ImportError:  # Direct script execution.
    from score_log import load_last_json


SIGNIFICANT_CHANGE = Decimal("0.0001")
ENERGY_MINIMUM = Decimal("0.999999999")  # Existing 1.0 - 1e-9 gate.
SPLIT_FIELDS = (
    "timestamp", "mean_rel_steps", "improvement_vs_anchor", "mean_rel_energy",
    "max_final_energy_delta_kcal_mol", "duration_s",
    "converged_count", "molecule_count", "converged_fraction",
    "internal_error_count", "infrastructure_retries", "is_valid",
    "validity_reason", "stop_reason_counts", "evaluation_id", "program_id",
    "evaluation_log", "evaluation_json",
)
FIELDS = (
    "timestamp", "cycle", "candidate_commit", "anchor_commit", "release_id",
    "decision", "accepted", "validity_reason",
    *(f"{split}_{field}" for split in ("train", "valid") for field in SPLIT_FIELDS),
    "anchor_train_mean_rel_steps", "anchor_valid_mean_rel_steps",
    "anchor_train_evaluation_json", "anchor_valid_evaluation_json",
    "anchor_train_evaluation_id", "anchor_valid_evaluation_id", "description",
)


def number(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Missing or non-numeric {name}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"Non-finite {name}: {value!r}")
    return result


def same_commit(left: str, right: str) -> bool:
    return min(len(left), len(right)) >= 7 and (left.startswith(right) or right.startswith(left))


def read_evaluation(path: Path | None, split: str, commit: str) -> dict | None:
    if path is None:
        return None
    evaluation = dict(load_last_json(path))
    if "dataset_release_id" in evaluation:
        if evaluation.get("release_id", evaluation["dataset_release_id"]) != evaluation["dataset_release_id"]:
            raise ValueError(f"{path}: conflicting release identifiers")
        evaluation["release_id"] = evaluation["dataset_release_id"]
    if evaluation.get("split") != split:
        raise ValueError(f"{path}: expected split {split!r}")
    actual_commit = str(evaluation.get("candidate_commit", ""))
    if not same_commit(actual_commit, commit):
        raise ValueError(f"{path}: candidate commit does not match {commit}")
    for key in ("evaluation_id", "program_id", "release_id", "timestamp", "status"):
        if not evaluation.get(key):
            raise ValueError(f"{path}: missing {key}")
    if evaluation["status"] not in ("complete", "pending"):
        raise ValueError(f"{path}: unsupported evaluation status {evaluation['status']!r}")
    evaluation["evaluation_json"] = str(path.resolve())
    if evaluation["status"] == "complete":
        # Required metrics must exist even for invalid complete evaluations;
        # malformed output is never converted into a scientific rejection.
        for key in ("mean_rel_steps", "mean_rel_energy", "internal_error_count",
                    "infrastructure_retries", "converged_count", "molecule_count",
                    "converged_fraction", "is_valid"):
            number(evaluation.get(key), key)
        for key in ("internal_error_count", "infrastructure_retries", "converged_count", "molecule_count"):
            value = number(evaluation[key], key)
            if value < 0 or value != value.to_integral_value():
                raise ValueError(f"{path}: {key} must be a nonnegative integer")
            evaluation[key] = int(value)
        if evaluation["molecule_count"] <= 0:
            raise ValueError(f"{path}: complete evaluation has no molecules")
        if not 0 <= evaluation["converged_count"] <= evaluation["molecule_count"]:
            raise ValueError(f"{path}: inconsistent convergence counts")
        if not 0 <= number(evaluation["converged_fraction"], "converged_fraction") <= 1:
            raise ValueError(f"{path}: convergence fraction is outside [0, 1]")
        if evaluation["is_valid"] not in (0, 1):
            raise ValueError(f"{path}: is_valid must be 0 or 1")
        if not isinstance(evaluation.get("stop_reason_counts"), dict):
            raise ValueError(f"{path}: missing stop_reason_counts")
    return evaluation


def valid(evaluation: dict) -> bool:
    return (evaluation["status"] == "complete"
            and evaluation["is_valid"] == 1
            and number(evaluation["mean_rel_energy"], "mean_rel_energy") >= ENERGY_MINIMUM
            and evaluation["internal_error_count"] == 0)


def improvement(anchor: dict, candidate: dict) -> Decimal:
    return (number(anchor["mean_rel_steps"], "anchor mean_rel_steps")
            - number(candidate["mean_rel_steps"], "candidate mean_rel_steps"))


def build_record(*, cycle: int, candidate_commit: str, anchor_commit: str,
                 train: dict, validation: dict | None, anchor_train: dict | None,
                 anchor_valid: dict | None, description: str,
                 starting_anchor: bool = False) -> dict:
    if cycle < 1:
        raise ValueError("cycle must be positive")
    evaluations = [item for item in (train, validation, anchor_train, anchor_valid) if item]
    if len({item["release_id"] for item in evaluations}) != 1:
        raise ValueError("Candidate and anchor evaluations must use the same release")
    if validation and train["program_id"] != validation["program_id"]:
        raise ValueError("Train and valid must evaluate the same candidate program")
    if anchor_train and anchor_valid and anchor_train["program_id"] != anchor_valid["program_id"]:
        raise ValueError("Train and valid anchors must evaluate the same program")
    if starting_anchor:
        if cycle != 1 or not same_commit(candidate_commit, anchor_commit):
            raise ValueError("Starting anchor requires cycle 1 and matching candidate/anchor commits")
        if anchor_train or anchor_valid:
            raise ValueError("Starting anchor must not supply separate anchor evaluations")
    else:
        if not anchor_train or not anchor_valid:
            raise ValueError("Both train and valid anchor JSON files are required")
        if not valid(anchor_train) or not valid(anchor_valid):
            raise ValueError("Both anchor evaluations must be complete and valid")

    row = {name: "" for name in FIELDS}
    row.update(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               cycle=cycle, candidate_commit=candidate_commit, anchor_commit=anchor_commit,
               release_id=train["release_id"], description=description)
    for split, evaluation, anchor in (("train", train, anchor_train),
                                      ("valid", validation, anchor_valid)):
        if evaluation:
            for field in SPLIT_FIELDS:
                value = evaluation.get(field, "")
                if field == "stop_reason_counts" and isinstance(value, dict):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"))
                row[f"{split}_{field}"] = value
            if anchor and evaluation["status"] == "complete":
                row[f"{split}_improvement_vs_anchor"] = str(improvement(anchor, evaluation))
        if anchor:
            row[f"anchor_{split}_mean_rel_steps"] = anchor["mean_rel_steps"]
            row[f"anchor_{split}_evaluation_json"] = anchor["evaluation_json"]
            row[f"anchor_{split}_evaluation_id"] = anchor["evaluation_id"]

    def decide(decision: str, reason: str) -> dict:
        row.update(decision=decision, validity_reason=reason,
                   accepted="" if decision == "pending" else int(decision in ("anchor", "keep")))
        return row

    if train["status"] != "complete":
        return decide("pending", "train evaluation pending; unscored")
    if not valid(train):
        return decide("invalid", "train invalid: " + str(train.get("validity_reason") or "energy/internal-error gate"))
    if not starting_anchor and improvement(anchor_train, train) < SIGNIFICANT_CHANGE:
        return decide("discard", "train improvement below 1e-4")
    if not validation or validation["status"] != "complete":
        return decide("pending", "valid evaluation pending; provisional train result is unscored")
    if not valid(validation):
        reason = "valid split invalid: " + str(validation.get("validity_reason") or "energy/internal-error gate")
        if not starting_anchor and improvement(anchor_valid, validation) <= SIGNIFICANT_CHANGE:
            reason += "; valid improvement is not greater than 1e-4"
        return decide("invalid" if starting_anchor else "non_generalizable",
                      reason)
    if starting_anchor:
        return decide("anchor", "complete valid starting train/valid anchor; no improvement claim")
    if improvement(anchor_valid, validation) <= SIGNIFICANT_CHANGE:
        return decide("non_generalizable", "valid improvement is not greater than 1e-4")
    return decide("keep", "train improvement >= 1e-4; valid improvement > 1e-4; both evaluations valid")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"{path}: incompatible table schema; refusing to mix records")
        return list(reader)


def append_row(path: Path, row: dict) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_record(output_dir: Path, row: dict) -> dict:
    """Serialize appends, reject contradictory completed cycles, and allow pending recovery."""
    # csv serializes None as an empty cell; normalize before the idempotence
    # comparison as evaluator pending records deliberately contain null metrics.
    row = {key: "" if value is None else value for key, value in row.items()}
    if any("\t" in str(value) or "\n" in str(value) or "\r" in str(value) for value in row.values()):
        raise ValueError("TSV values must be single-line and contain no tabs")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Lock an existing artifact itself; no mutable sidecar state is needed.
    results_path = output_dir / "results.tsv"
    with results_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        paths = {name: output_dir / f"{name}.tsv" for name in
                 ("results", "generalizable", "non_generalizable")}
        tables = {name: read_rows(path) if path.stat().st_size else []
                  for name, path in paths.items() if path.exists()}
        comparable = lambda item: {key: str(value) for key, value in item.items() if key != "timestamp"}
        matches = [item for item in tables.get("results", []) if item["cycle"] == str(row["cycle"])]
        for previous in matches:
            if previous["candidate_commit"] != row["candidate_commit"]:
                raise ValueError("Cycle already belongs to a different candidate")
            if comparable(previous) == comparable(row):
                row = previous
                break
            if previous["decision"] != "pending":
                raise ValueError("Cycle already has a different completed decision")
        else:
            keeps = tables.get("generalizable", [])
            if keeps:
                last_keep = keeps[-1]
                if not same_commit(str(row["anchor_commit"]), last_keep["candidate_commit"]):
                    raise ValueError("Anchor must be the last generalizable keep")
                for split in ("train", "valid"):
                    if row[f"anchor_{split}_evaluation_id"] != last_keep[f"{split}_evaluation_id"]:
                        raise ValueError("Anchor evaluation must match the last generalizable keep")
            append_row(results_path, row)
        target = ("generalizable" if row["decision"] in ("anchor", "keep") else
                  "non_generalizable" if row["decision"] == "non_generalizable" else None)
        if target:
            existing = [item for item in tables.get(target, []) if item["cycle"] == str(row["cycle"])]
            if existing and comparable(existing[0]) != comparable(row):
                raise ValueError(f"Conflicting {target} record")
            if not existing:
                append_row(paths[target], row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--anchor-commit", required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--valid-json", type=Path)
    parser.add_argument("--anchor-train-json", type=Path)
    parser.add_argument("--anchor-valid-json", type=Path)
    parser.add_argument("--description", required=True)
    parser.add_argument("--starting-anchor", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    try:
        row = build_record(
            cycle=args.cycle, candidate_commit=args.candidate_commit,
            anchor_commit=args.anchor_commit, description=args.description,
            starting_anchor=args.starting_anchor,
            train=read_evaluation(args.train_json, "train", args.candidate_commit),
            validation=read_evaluation(args.valid_json, "valid", args.candidate_commit),
            anchor_train=read_evaluation(args.anchor_train_json, "train", args.anchor_commit),
            anchor_valid=read_evaluation(args.anchor_valid_json, "valid", args.anchor_commit),
        )
        print(json.dumps(write_record(args.output_dir, row), sort_keys=True))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
