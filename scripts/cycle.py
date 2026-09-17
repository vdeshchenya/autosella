#!/usr/bin/env python
"""Reserve or resume a research cycle, and record its evaluator evidence.

``begin`` allocates before editing and is idempotent until the reserved cycle
has a completed decision. ``status`` never advances it. ``record`` delegates
all scoring and table writes to record_decision, using the reservation and HEAD.
``recover`` replays completed records whose secondary-table append was interrupted.
The uncommitted cycle.json survives process restarts; results.tsv supplies the
same interprocess lock used by the decision writer.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

try:
    from .record_decision import (build_record, read_evaluation, read_rows,
                                  same_commit, valid, write_record)
except ImportError:  # Direct script execution.
    from record_decision import (build_record, read_evaluation, read_rows,
                                 same_commit, valid, write_record)


DECISIONS = {"anchor", "keep", "discard", "invalid", "non_generalizable", "pending"}
RESERVATION_FIELDS = ("cycle", "anchor_commit", "anchor_train_json",
                      "anchor_valid_json", "train_json", "valid_json")


def _cycle(value) -> int:
    if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 1:
        raise ValueError(f"Invalid cycle number: {value!r}")
    return int(value)


def _comparable(row: dict) -> dict:
    return {key: str(value) for key, value in row.items() if key != "timestamp"}


def _rows(output_dir: Path, name: str) -> list[dict]:
    path = output_dir / f"{name}.tsv"
    return read_rows(path) if path.exists() and path.stat().st_size else []


def _secondary_tables(latest: dict[int, dict]) -> dict[str, list[dict]]:
    ordered = [latest[number] for number in sorted(latest)]
    return {"generalizable": [row for row in ordered if row["decision"] in {"anchor", "keep"}],
            "non_generalizable": [row for row in ordered if row["decision"] == "non_generalizable"]}


def _history(output_dir: Path, *, allow_missing_secondary: bool = False) -> tuple[dict[int, dict], list[dict]]:
    latest = {}
    for row in _rows(output_dir, "results"):
        cycle = _cycle(row["cycle"])
        if row["decision"] not in DECISIONS:
            raise ValueError(f"Cycle {cycle}: unknown decision")
        previous = latest.get(cycle)
        if previous:
            for field in ("candidate_commit", "anchor_commit", "release_id",
                          "anchor_train_evaluation_id", "anchor_valid_evaluation_id"):
                if previous[field] != row[field]:
                    raise ValueError(f"Cycle {cycle}: inconsistent {field}")
            if previous["decision"] != "pending" and _comparable(previous) != _comparable(row):
                raise ValueError(f"Cycle {cycle}: conflicting completed decisions")
        latest[cycle] = row

    secondary = _secondary_tables(latest)
    keeps = secondary["generalizable"]
    if (not keeps or not latest.get(1) or latest[1]["decision"] != "anchor"
            or keeps[0]["decision"] != "anchor" or keeps[0]["cycle"] != "1"):
        raise ValueError("Missing complete starting champion (cycle 1); finish the starting anchor first")
    for name, expected in secondary.items():
        actual = [_comparable(row) for row in _rows(output_dir, name)]
        expected = [_comparable(row) for row in expected]
        if actual != expected:
            if not allow_missing_secondary or actual != expected[:len(actual)]:
                raise ValueError(f"results.tsv and {name}.tsv are inconsistent; recover the decision write first")
    if any(row["decision"] == "anchor" for row in keeps[1:]):
        raise ValueError("Only cycle 1 may be a starting anchor")
    unresolved = [number for number, row in latest.items() if row["decision"] == "pending"]
    if len(unresolved) > 1 or (unresolved and unresolved[0] != max(latest)):
        raise ValueError("Multiple unresolved cycles or a pending cycle before newer decisions; recover history first")
    _anchor_evidence(output_dir, keeps[-1])
    return latest, keeps


def _path(output_dir: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else output_dir / path).resolve()


def _anchor_evidence(output_dir: Path, row: dict) -> tuple[dict, dict]:
    evaluations = []
    for split in ("train", "valid"):
        evidence_path = row[f"{split}_evaluation_json"]
        if not evidence_path:
            raise ValueError(f"Champion is missing {split} evidence")
        evaluation = read_evaluation(_path(output_dir, evidence_path), split, row["candidate_commit"])
        if not valid(evaluation):
            raise ValueError(f"Champion {split} evidence must be complete and valid")
        for field in ("evaluation_id", "program_id"):
            if evaluation[field] != row[f"{split}_{field}"]:
                raise ValueError(f"Champion {split} {field} does not match its recorded evidence")
        if evaluation["release_id"] != row["release_id"]:
            raise ValueError("Champion evidence release does not match its record")
        evaluations.append(evaluation)
    if evaluations[0]["program_id"] != evaluations[1]["program_id"]:
        raise ValueError("Champion train and valid must evaluate the same program")
    return tuple(evaluations)


def _reservation(output_dir: Path, cycle: int, champion: dict, pending: dict | None = None) -> dict:
    result = dict(cycle=cycle, anchor_commit=champion["candidate_commit"])
    for split in ("train", "valid"):
        result[f"anchor_{split}_json"] = str(_path(output_dir, champion[f"{split}_evaluation_json"]))
        existing = pending.get(f"{split}_evaluation_json") if pending else None
        result[f"{split}_json"] = str(_path(
            output_dir, existing or f"evaluation_results/cycle-{cycle}-{split}.json"))
    return result


def _current(output_dir: Path, latest: dict[int, dict], keeps: list[dict],
             saved: dict | None = None) -> dict | None:
    if saved is None:
        path = output_dir / "cycle.json"
        if not path.exists():
            return None
        saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or set(saved) != set(RESERVATION_FIELDS):
        raise ValueError("cycle.json has an incompatible reservation schema")
    cycle = _cycle(saved["cycle"])
    if not isinstance(saved["cycle"], int):
        raise ValueError("cycle.json cycle must be an integer")
    if cycle <= 1 or cycle < max(latest) or cycle > max(latest) + 1:
        raise ValueError("cycle.json disagrees with recorded cycle numbers; recover state before proceeding")
    for field in RESERVATION_FIELDS[1:]:
        if not isinstance(saved[field], str) or not saved[field]:
            raise ValueError(f"cycle.json is missing {field}")
    previous_keeps = [row for row in keeps if _cycle(row["cycle"]) < cycle]
    if not previous_keeps:
        raise ValueError("Allocated cycle has no preceding champion")
    champion = previous_keeps[-1]
    row = latest.get(cycle)
    expected = _reservation(output_dir, cycle, champion, row)
    if (not same_commit(saved["anchor_commit"], expected["anchor_commit"])
            or any(_path(output_dir, saved[field]) != _path(output_dir, expected[field])
                   for field in RESERVATION_FIELDS[2:])):
        raise ValueError("cycle.json anchor or evaluation paths disagree with recorded history")
    if row:
        if not same_commit(row["anchor_commit"], champion["candidate_commit"]):
            raise ValueError("Allocated cycle uses a different anchor from its preceding champion")
        for split in ("train", "valid"):
            if row[f"anchor_{split}_evaluation_id"] != champion[f"{split}_evaluation_id"]:
                raise ValueError("Allocated cycle anchor evidence disagrees with its preceding champion")
    elif any(item["decision"] == "pending" for item in latest.values()):
        raise ValueError("cycle.json would skip an unresolved cycle")
    return {**saved, "status": "allocated" if row is None else
            ("pending" if row["decision"] == "pending" else "complete"),
            "decision": row["decision"] if row else None,
            "candidate_commit": row["candidate_commit"] if row else None}


def _save(output_dir: Path, reservation: dict) -> None:
    descriptor, filename = tempfile.mkstemp(prefix="cycle-", suffix=".tmp", dir=output_dir)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(reservation, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_dir / "cycle.json")
    finally:
        temporary.unlink(missing_ok=True)


def begin(output_dir: Path = Path(".")) -> dict:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.tsv").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest, keeps = _history(output_dir)
        current = _current(output_dir, latest, keeps)
        if current and current["status"] != "complete":
            return current
        pending = next((row for row in latest.values() if row["decision"] == "pending"), None)
        cycle = _cycle(pending["cycle"]) if pending else max(latest) + 1
        reservation = _reservation(output_dir, cycle, keeps[-1], pending)
        current = _current(output_dir, latest, keeps, reservation)
        _save(output_dir, reservation)
        return current


def status(output_dir: Path = Path(".")) -> dict:
    output_dir = output_dir.resolve()
    with (output_dir / "results.tsv").open(encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        latest, keeps = _history(output_dir)
        current = _current(output_dir, latest, keeps)
        if current is None:
            raise ValueError("No cycle allocated; run `python scripts/cycle.py begin` first")
        return current


def recover(output_dir: Path = Path(".")) -> dict:
    """Repair only missing secondary-table suffixes by replaying exact results.

    Validate every table before writing any row. Release the read lock before
    invoking the sole writer, which independently locks and checks each replay.
    Repetition is harmless, including interruption during recovery itself.
    """
    output_dir = output_dir.resolve()
    with (output_dir / "results.tsv").open(encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        latest, _ = _history(output_dir, allow_missing_secondary=True)
        missing = {name: expected[len(_rows(output_dir, name)):]
                   for name, expected in _secondary_tables(latest).items()}
    for row in sorted((row for rows in missing.values() for row in rows),
                      key=lambda row: _cycle(row["cycle"])):
        write_record(output_dir, row)
    return {"recovered_cycles": sorted(_cycle(row["cycle"])
                                      for rows in missing.values() for row in rows),
            "repaired_tables": {name: len(rows) for name, rows in missing.items() if rows}}


def _git(output_dir: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=output_dir, capture_output=True)
    if result.returncode:
        raise ValueError("Cannot inspect candidate git state: " + result.stderr.decode().strip())
    return result.stdout


def record(output_dir: Path = Path("."), *, description: str) -> dict:
    output_dir = output_dir.resolve()
    current = status(output_dir)
    head = _git(output_dir, "rev-parse", "HEAD").decode().strip()
    candidate = current["candidate_commit"] or head
    if not same_commit(candidate, head):
        raise ValueError("Reserved cycle already belongs to a different candidate; restore that commit and resume its evidence")
    source = output_dir / "algo.py"
    if source.read_bytes() != _git(output_dir, "show", "HEAD:./algo.py"):
        raise ValueError("algo.py differs from HEAD; record the committed candidate without further source edits")
    digest = hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
    evaluations = {}
    for split in ("train", "valid"):
        path = _path(output_dir, current[f"{split}_json"])
        evaluation = read_evaluation(path if split == "train" or path.exists() else None,
                                     split, candidate)
        if evaluation and (evaluation["program_id"] != digest
                           or evaluation.get("source_sha256", digest) != digest):
            raise ValueError(f"{split} evidence does not match the committed algo.py source")
        evaluations[split] = evaluation
    row = build_record(
        cycle=current["cycle"], candidate_commit=candidate, anchor_commit=current["anchor_commit"],
        description=description, train=evaluations["train"], validation=evaluations["valid"],
        anchor_train=read_evaluation(_path(output_dir, current["anchor_train_json"]), "train", current["anchor_commit"]),
        anchor_valid=read_evaluation(_path(output_dir, current["anchor_valid_json"]), "valid", current["anchor_commit"]),
    )
    # write_record takes the same results.tsv lock itself and remains the sole
    # table writer. An allocation cannot advance until this cycle completes.
    return write_record(output_dir, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("begin", "status", "record", "recover"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--output-dir", type=Path, default=Path("."))
        if command == "begin":
            subparser.add_argument("--number", action="store_true", help="Print only the allocated cycle number")
        elif command == "record":
            subparser.add_argument("--description", required=True)
    args = parser.parse_args()
    try:
        result = record(args.output_dir, description=args.description) if args.command == "record" else globals()[args.command](args.output_dir)
        print(result["cycle"] if getattr(args, "number", False) else json.dumps(result, sort_keys=True))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
