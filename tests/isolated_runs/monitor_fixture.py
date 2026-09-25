"""Synthetic UI fixture only; never run against an operator's real run root.

Usage: PYTHONPATH=. python tests/isolated_runs/monitor_fixture.py /new/private/root
Creates a new root exclusively; does not launch a server or any experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.record_decision import build_record, write_record


def evaluation(split, commit, cost=1.0, **updates):
    value = dict(timestamp="2026-09-08T12:00:00Z", evaluation_id=f"fixture-{split}-{commit[:8]}",
        program_id=f"program-{commit}", candidate_commit=commit, split=split, release_id="synthetic-fixture-only",
        status="complete", mean_rel_steps=cost, mean_rel_energy=1.0, converged_count=8, molecule_count=10,
        converged_fraction=.8, internal_error_count=0, infrastructure_retries=0,
        stop_reason_counts={"converged": 8, "force_call_limit": 2}, is_valid=1, validity_reason="valid",
        evaluation_log=f"fixture-{split}.log", evaluation_json=f"fixture-{split}-{commit}.json")
    value.update(updates)
    return value


def fixture_records():
    anchor_commit = "b" * 40
    anchor_train, anchor_valid = evaluation("train", anchor_commit), evaluation("valid", anchor_commit)
    rows = [build_record(cycle=1, candidate_commit=anchor_commit, anchor_commit=anchor_commit,
        train=anchor_train, validation=anchor_valid, anchor_train=None, anchor_valid=None,
        description="SYNTHETIC FIXTURE: starting anchor, no improvement claim", starting_anchor=True)]
    for cycle, commit, train_args, valid_args in [
        (2, "c" * 40, {"cost": .92}, {"cost": .96}),
        (3, "d" * 40, {"cost": .921}, None),
        (4, "e" * 40, {"cost": .85, "is_valid": 0, "mean_rel_energy": .97, "validity_reason": "energy_below_baseline"}, None),
        (5, "f" * 40, {"cost": .86}, {"cost": .97}),
        (6, "a" * 40, {"cost": .82}, {"cost": .9, "is_valid": 0, "mean_rel_energy": .95, "validity_reason": "energy_below_baseline"}),
        (7, "1" * 40, {"cost": 1000, "mean_rel_energy": -1, "is_valid": 0, "internal_error_count": 2,
                            "validity_reason": "internal_error"}, None),
        (8, "2" * 40, {"cost": .84}, {"cost": .9}),
        (9, "3" * 40, {"cost": .82}, {"cost": None, "status": "pending", "mean_rel_energy": None,
                 "is_valid": None, "converged_count": None, "converged_fraction": None}),
    ]:
        train = evaluation("train", commit, **train_args)
        valid = evaluation("valid", commit, **valid_args) if valid_args else None
        record = build_record(cycle=cycle, candidate_commit=commit, anchor_commit=anchor_commit,
            train=train, validation=valid, anchor_train=anchor_train, anchor_valid=anchor_valid,
            description=f"SYNTHETIC FIXTURE cycle {cycle}: example only, not research evidence")
        rows.append(record)
        if record["decision"] == "keep":
            anchor_commit, anchor_train, anchor_valid = commit, train, valid
    return rows


def create_fixture(root):
    root = Path(root)
    root.mkdir(mode=0o700)  # Exclusive: do not modify an existing run root.
    path = root / "synthetic-progress-fixture"
    path.mkdir(mode=0o700)
    for part in ("control", "workspace", "secrets", "home"):
        (path / part).mkdir(mode=0o700)
    state = {"schema": 1, "name": path.name, "run_id": path.name + "-" + "a" * 16,
             "status": "stopped", "purpose": "research", "initial_commit": "b" * 40,
             "created_at": "2026-09-08T12:00:00Z", "updated_at": "2026-09-08T12:00:00Z"}
    (path / "control/state.json").write_text(json.dumps(state))
    (path / "control/runtime.json").write_text(json.dumps({"provider": "fixture", "model": "synthetic-example", "effort": "none"}))
    (path / "workspace/run.log").write_text("SYNTHETIC UI FIXTURE ONLY. No experiments or workers were launched.\n")
    for row in fixture_records():
        write_record(path / "workspace", row)
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(create_fixture(parser.parse_args().root))
