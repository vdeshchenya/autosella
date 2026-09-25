"""Cycle reservations survive restarts and use the existing evidence gates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.cycle import begin, record, recover, status
from scripts.record_decision import build_record, read_evaluation, read_rows, write_record


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cycle.py"
ANCHOR = "a" * 40
CANDIDATE = "b" * 40


def evidence(root, cycle, split, commit, *, steps=1, program_id=None, **updates):
    path = root / "evaluation_results" / f"cycle-{cycle}-{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = program_id or hashlib.sha256(commit.encode()).hexdigest()
    payload = dict(
        timestamp="2026-09-10T12:00:00Z", evaluation_id=f"eval-{cycle}-{split}",
        program_id=digest, source_sha256=digest, candidate_commit=commit,
        split=split, release_id="cycle-fixture", status="complete",
        mean_rel_steps=steps, mean_rel_energy=1, converged_count=2,
        molecule_count=2, converged_fraction=1, internal_error_count=0,
        infrastructure_retries=0, stop_reason_counts={"converged": 2},
        is_valid=1, validity_reason="valid", evaluation_log=str(path),
    )
    payload.update(updates)
    path.write_text(json.dumps(payload))
    return read_evaluation(path, split, commit)


def starting_anchor(root, commit=ANCHOR, program_id=None):
    return write_record(root, build_record(
        cycle=1, candidate_commit=commit, anchor_commit=commit,
        train=evidence(root, 1, "train", commit, program_id=program_id),
        validation=evidence(root, 1, "valid", commit, program_id=program_id),
        anchor_train=None, anchor_valid=None, description="starting anchor", starting_anchor=True,
    ))


def decision(root, cycle=2, commit=CANDIDATE, *, train_steps=0.8, valid_steps=0.8,
             pending=False, program_id=None):
    anchor = read_rows(root / "generalizable.tsv")[-1]
    return write_record(root, build_record(
        cycle=cycle, candidate_commit=commit, anchor_commit=anchor["candidate_commit"],
        train=evidence(root, cycle, "train", commit, steps=train_steps, program_id=program_id),
        validation=evidence(root, cycle, "valid", commit, steps=valid_steps, program_id=program_id,
                            status="pending" if pending else "complete"),
        anchor_train=read_evaluation(Path(anchor["train_evaluation_json"]), "train", anchor["candidate_commit"]),
        anchor_valid=read_evaluation(Path(anchor["valid_evaluation_json"]), "valid", anchor["candidate_commit"]),
        description="candidate",
    ))


def command(root, *args, check=True):
    return subprocess.run([sys.executable, str(SCRIPT), *args, "--output-dir", str(root)],
                          capture_output=True, text=True, check=check)


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                          check=True).stdout.strip()


@pytest.fixture
def candidate_repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "fixture@example.invalid")
    git(tmp_path, "config", "user.name", "Cycle fixture")
    (tmp_path / "algo.py").write_text("# starting optimizer\n")
    git(tmp_path, "add", "algo.py")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "starting optimizer")
    anchor = git(tmp_path, "rev-parse", "HEAD")
    starting_anchor(tmp_path, anchor,
                    hashlib.sha256((tmp_path / "algo.py").read_bytes()).hexdigest())
    begin(tmp_path)
    (tmp_path / "algo.py").write_text("# candidate optimizer\n")
    git(tmp_path, "add", "algo.py")
    git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "cycle 2: candidate")
    candidate = git(tmp_path, "rev-parse", "HEAD")
    digest = hashlib.sha256((tmp_path / "algo.py").read_bytes()).hexdigest()
    return tmp_path, candidate, digest


def test_begin_survives_restart_before_any_record(tmp_path):
    starting_anchor(tmp_path)
    initial = json.loads(command(tmp_path, "begin").stdout)
    saved = (tmp_path / "cycle.json").read_bytes()
    assert initial["cycle"] == 2
    assert initial["status"] == "allocated"
    assert initial["decision"] is None
    assert initial["train_json"] == str(tmp_path / "evaluation_results/cycle-2-train.json")
    assert command(tmp_path, "begin", "--number").stdout == "2\n"
    assert json.loads(command(tmp_path, "status").stdout) == initial
    assert (tmp_path / "cycle.json").read_bytes() == saved
    assert len(read_rows(tmp_path / "results.tsv")) == 1


def test_pending_completion_then_next_cycle(tmp_path):
    starting_anchor(tmp_path)
    begin(tmp_path)
    decision(tmp_path, pending=True)
    assert begin(tmp_path)["cycle"] == 2
    assert status(tmp_path)["status"] == "pending"
    assert status(tmp_path)["candidate_commit"] == CANDIDATE
    decision(tmp_path)
    completed = status(tmp_path)
    assert completed["cycle"] == 2
    assert completed["status"] == "complete"
    assert completed["decision"] == "keep"
    assert completed["anchor_commit"] == ANCHOR
    allocated = begin(tmp_path)
    assert allocated["cycle"] == 3
    assert allocated["anchor_commit"] == CANDIDATE
    assert allocated["anchor_valid_json"] == str(tmp_path / "evaluation_results/cycle-2-valid.json")


def test_pending_without_sidecar_is_recovered(tmp_path):
    starting_anchor(tmp_path)
    decision(tmp_path, pending=True)
    recovered = begin(tmp_path)
    assert recovered["cycle"] == 2
    assert recovered["status"] == "pending"
    assert recovered["candidate_commit"] == CANDIDATE
    assert (tmp_path / "cycle.json").exists()


def test_concurrent_begin_reserves_one_cycle(tmp_path):
    starting_anchor(tmp_path)
    processes = [subprocess.Popen(
        [sys.executable, str(SCRIPT), "begin", "--number", "--output-dir", str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(8)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stderr
        assert stdout == "2\n"
    assert status(tmp_path)["cycle"] == 2
    assert len(read_rows(tmp_path / "results.tsv")) == 1


def test_missing_starting_champion_blocks_allocation(tmp_path):
    with pytest.raises(ValueError, match="Missing complete starting champion"):
        begin(tmp_path)
    assert not (tmp_path / "cycle.json").exists()


def test_status_does_not_allocate(tmp_path):
    starting_anchor(tmp_path)
    with pytest.raises(ValueError, match="No cycle allocated"):
        status(tmp_path)
    assert not (tmp_path / "cycle.json").exists()


def test_champion_is_last_keep_not_last_candidate(tmp_path):
    starting_anchor(tmp_path)
    begin(tmp_path)
    decision(tmp_path, train_steps=1)
    allocated = begin(tmp_path)
    assert allocated["cycle"] == 3
    assert allocated["anchor_commit"] == ANCHOR
    decision(tmp_path, cycle=3, commit="c" * 40)
    allocated = begin(tmp_path)
    assert allocated["cycle"] == 4
    assert allocated["anchor_commit"] == "c" * 40
    assert allocated["anchor_train_json"].endswith("cycle-3-train.json")


def test_multiple_pending_cycles_refuse_to_skip(tmp_path):
    starting_anchor(tmp_path)
    begin(tmp_path)
    decision(tmp_path, pending=True)
    decision(tmp_path, cycle=3, commit="c" * 40, pending=True)
    saved = (tmp_path / "cycle.json").read_bytes()
    with pytest.raises(ValueError, match="Multiple unresolved"):
        begin(tmp_path)
    assert (tmp_path / "cycle.json").read_bytes() == saved


def test_new_reservation_cannot_skip_pending_cycle(tmp_path):
    starting_anchor(tmp_path)
    begin(tmp_path)
    decision(tmp_path, pending=True)
    saved = json.loads((tmp_path / "cycle.json").read_text())
    saved.update(cycle=3, train_json=str(tmp_path / "evaluation_results/cycle-3-train.json"),
                 valid_json=str(tmp_path / "evaluation_results/cycle-3-valid.json"))
    (tmp_path / "cycle.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="skip an unresolved cycle"):
        begin(tmp_path)


def test_partial_decision_write_requires_recovery(tmp_path):
    starting_anchor(tmp_path)
    begin(tmp_path)
    original = (tmp_path / "generalizable.tsv").read_bytes()
    decision(tmp_path)
    (tmp_path / "generalizable.tsv").write_bytes(original)
    with pytest.raises(ValueError, match="inconsistent; recover the decision write"):
        begin(tmp_path)


@pytest.mark.parametrize("valid_steps, table", [(.8, "generalizable"), (1, "non_generalizable")])
def test_recover_replays_interrupted_secondary_append(tmp_path, valid_steps, table):
    starting_anchor(tmp_path)
    begin(tmp_path)
    path = tmp_path / f"{table}.tsv"
    before = path.read_bytes() if path.exists() else b""
    decision(tmp_path, valid_steps=valid_steps)
    completed = path.read_bytes()
    results = (tmp_path / "results.tsv").read_bytes()
    reservation = (tmp_path / "cycle.json").read_bytes()
    path.write_bytes(before)  # Interruption after results.tsv, before the second append.
    for operation in (begin, status, lambda root: record(root, description="candidate")):
        with pytest.raises(ValueError, match="inconsistent; recover the decision write"):
            operation(tmp_path)
    assert (tmp_path / "cycle.json").read_bytes() == reservation
    assert path.read_bytes() == before
    result = json.loads(command(tmp_path, "recover").stdout)
    assert result == {"recovered_cycles": [2], "repaired_tables": {table: 1}}
    assert path.read_bytes() == completed
    assert (tmp_path / "results.tsv").read_bytes() == results
    assert (tmp_path / "cycle.json").read_bytes() == reservation
    assert recover(tmp_path) == {"recovered_cycles": [], "repaired_tables": {}}
    assert status(tmp_path)["status"] == "complete"
    assert begin(tmp_path)["cycle"] == 3


def test_recover_starting_anchor_secondary_append(tmp_path):
    starting_anchor(tmp_path)
    expected = (tmp_path / "generalizable.tsv").read_bytes()
    (tmp_path / "generalizable.tsv").unlink()
    assert recover(tmp_path)["recovered_cycles"] == [1]
    assert (tmp_path / "generalizable.tsv").read_bytes() == expected
    assert begin(tmp_path)["cycle"] == 2


def test_recover_validates_all_tables_before_any_repair(tmp_path):
    starting_anchor(tmp_path)
    starting = (tmp_path / "generalizable.tsv").read_bytes()
    decision(tmp_path)
    decision(tmp_path, cycle=3, commit="c" * 40, train_steps=.6, valid_steps=1)
    (tmp_path / "generalizable.tsv").write_bytes(starting)
    path = tmp_path / "non_generalizable.tsv"
    path.write_text(path.read_text().replace("\tcandidate\n", "\tconflicting description\n"))
    before = {name: (tmp_path / f"{name}.tsv").read_bytes()
              for name in ("results", "generalizable", "non_generalizable")}
    with pytest.raises(ValueError, match="non_generalizable.tsv are inconsistent"):
        recover(tmp_path)
    assert before == {name: (tmp_path / f"{name}.tsv").read_bytes() for name in before}


def test_recover_does_not_reorder_or_duplicate_existing_secondary_rows(tmp_path):
    starting_anchor(tmp_path)
    decision(tmp_path)
    path = tmp_path / "generalizable.tsv"
    lines = path.read_text().splitlines(keepends=True)
    for bad in (lines[:1] + lines[2:], lines + lines[2:]):
        path.write_text("".join(bad))
        with pytest.raises(ValueError, match="generalizable.tsv are inconsistent"):
            recover(tmp_path)
        assert path.read_text() == "".join(bad)


def test_invalid_champion_evidence_blocks_allocation(tmp_path):
    starting_anchor(tmp_path)
    path = tmp_path / "evaluation_results/cycle-1-valid.json"
    payload = json.loads(path.read_text())
    payload["status"] = "pending"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="complete and valid"):
        begin(tmp_path)


def test_record_cli_uses_reservation_and_committed_source(candidate_repo):
    root, candidate, digest = candidate_repo
    for split in ("train", "valid"):
        evidence(root, 2, split, candidate, steps=0.8, program_id=digest)
    result = json.loads(command(root, "record", "--description", "candidate").stdout)
    assert result["decision"] == "keep"
    assert result["cycle"] == 2
    assert result["candidate_commit"] == candidate
    assert status(root)["status"] == "complete"
    command(root, "record", "--description", "candidate")
    assert len(read_rows(root / "results.tsv")) == 2


def test_record_missing_valid_is_pending_then_resolves(candidate_repo):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=0.8, program_id=digest)
    assert record(root, description="candidate")["decision"] == "pending"
    assert begin(root)["cycle"] == 2
    evidence(root, 2, "valid", candidate, steps=0.8, program_id=digest)
    assert record(root, description="candidate")["decision"] == "keep"
    assert [row["decision"] for row in read_rows(root / "results.tsv")] == ["anchor", "pending", "keep"]
    assert begin(root)["cycle"] == 3


def test_record_train_failure_without_valid_is_completed(candidate_repo):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=1, program_id=digest)
    assert record(root, description="candidate")["decision"] == "discard"
    assert begin(root)["cycle"] == 3


def test_pending_rejects_new_commit(candidate_repo):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=0.8, program_id=digest)
    record(root, description="candidate")
    (root / "algo.py").write_text("# another candidate\n")
    git(root, "add", "algo.py")
    git(root, "-c", "commit.gpgsign=false", "commit", "-qm", "another candidate")
    with pytest.raises(ValueError, match="different candidate; restore that commit"):
        record(root, description="candidate")
    assert begin(root)["cycle"] == 2


def test_record_rejects_uncommitted_source_change(candidate_repo):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=0.8, program_id=digest)
    (root / "algo.py").write_text("# uncommitted edit\n")
    with pytest.raises(ValueError, match="differs from HEAD"):
        record(root, description="candidate")
    assert len(read_rows(root / "results.tsv")) == 1


@pytest.mark.parametrize("changed", ["program_id", "source_sha256", "candidate_commit"])
def test_record_rejects_mismatched_evidence(candidate_repo, changed):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=0.8, program_id=digest)
    path = root / "evaluation_results/cycle-2-train.json"
    payload = json.loads(path.read_text())
    payload[changed] = "d" * (40 if changed == "candidate_commit" else 64)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match|commit does not match"):
        record(root, description="candidate")
    assert len(read_rows(root / "results.tsv")) == 1


def test_record_refuses_stale_valid_even_if_train_fails(candidate_repo):
    root, candidate, digest = candidate_repo
    evidence(root, 2, "train", candidate, steps=1, program_id=digest)
    evidence(root, 2, "valid", "d" * 40, steps=0.8)
    with pytest.raises(ValueError, match="commit does not match"):
        record(root, description="candidate")
    assert len(read_rows(root / "results.tsv")) == 1
