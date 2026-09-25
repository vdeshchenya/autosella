#!/usr/bin/env python3
"""Generate a private, deterministic SYNTHETIC dashboard demonstration.

Nothing here runs an optimizer, model, worker, Git command, or server. All
numbers and identifiers are invented UI examples, never research evidence.
Creation requires a new root. Appends require this generator's demo marker.

    python tests/isolated_runs/generate_monitor_demo.py create
    python tests/isolated_runs/generate_monitor_demo.py verify /private/demo/root
    python tests/isolated_runs/generate_monitor_demo.py append-demo-cycle /private/demo/root --run demo-astra
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monitor_fixture import evaluation as fixture_evaluation
from isolated_runs.monitor import Monitor
from isolated_runs.monitor_progress import TABLE_NAMES, build_progress, parse_table
from scripts.record_decision import build_record, read_evaluation, write_record

NOTICE = "SYNTHETIC DEMO: generated UI examples only; no model, optimizer, or worker was run."
MARKER = "SYNTHETIC_DEMO.json"
GENERATOR = "tests/isolated_runs/generate_monitor_demo.py"
RELEASE = "synthetic-dashboard-demo-20260908-v1"
START = datetime(2026, 9, 8, 3, 0, tzinfo=timezone.utc)
MODELS = {"demo-astra": ("codex", "gpt-6-astra", 84),
          "demo-opus": ("claude", "claude-opus-5", 76)}
EFFORTS = {"demo-astra": "high", "demo-opus": "max"}
SCENARIOS = ("no_improvement", "keep", "no_improvement", "generalization_reject",
             "train_energy_reject", "keep", "no_improvement", "numerical_error",
             "valid_energy_reject", "keep", "generalization_reject", "no_improvement")


def dump(path: Path, value) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def identity(seed: int, name: str, kind: str, cycle: int, length=64) -> str:
    return hashlib.sha256(f"SYNTHETIC:{seed}:{name}:{kind}:{cycle}".encode()).hexdigest()[:length]


def cycle_time(cycle: int, offset=0) -> str:
    return (START + timedelta(minutes=(cycle - 1) * 6, seconds=offset)).strftime("%Y-%m-%dT%H:%M:%SZ")


def generated_progress(value: dict) -> dict:
    """Invent consistent outcome counts for the explicitly synthetic snapshot."""
    total = value["molecule_count"]
    completed = total if value["status"] == "complete" else total * 3 // 5
    return {"availability": "available", "total": total, "completed": completed,
        "pending": total - completed, "attempts": completed, "infrastructure_retries": 0,
        "updated_at": value["timestamp"], "age_seconds": None, "stale": None,
        "synthetic": True, "synthetic_notice": NOTICE}


def evaluation(path: Path, context: dict, cycle: int, split: str, cost: float | None,
               *, scenario="keep", pending=False) -> dict:
    """Reuse the fixture schema and persist clearly identified generated JSON."""
    name, seed = context["name"], context["seed"]
    commit = identity(seed, name, "candidate", cycle, 40)
    count = 100 if split == "train" else 40
    rng = random.Random(identity(seed, name, split, cycle))
    errors = 2 if scenario == "numerical_error" else 0
    energy_reject = scenario == f"{split}_energy_reject"
    valid = not errors and not energy_reject
    converged = int(count * rng.uniform(.82, .98)) if not errors else 0
    status = "pending" if pending else "complete"
    json_path = path / "workspace/synthetic-evaluations" / f"cycle-{cycle:03d}-{split}-{status}.json"
    value = fixture_evaluation(split, commit, cost=cost,
        timestamp=cycle_time(cycle, 30 if split == "train" else 120),
        evaluation_id=identity(seed, name, f"evaluation-{split}", cycle),
        program_id="synthetic-program-" + identity(seed, name, "program", cycle),
        run_id=context["run_id"], release_id=RELEASE, status=status,
        mean_rel_energy=(-1 if errors else .998 if energy_reject else 1.000002),
        max_final_energy_delta_kcal_mol=None if errors else .03 if energy_reject else -.000002,
        duration_s=round(rng.uniform(80, 300), 3),
        converged_count=converged, molecule_count=count, converged_fraction=converged / count,
        internal_error_count=errors, infrastructure_retries=0,
        stop_reason_counts={"converged": converged, "force_call_limit": count - converged - errors,
                            "internal_error": errors},
        is_valid=int(valid), validity_reason="internal_error" if errors else "energy_below_baseline" if energy_reject else "valid",
        evaluation_log=str(path / "workspace/logs/SYNTHETIC-evaluations.log"),
        evaluation_json=str(json_path), synthetic=True, synthetic_notice=NOTICE)
    if pending:
        for key in ("mean_rel_steps", "mean_rel_energy", "max_final_energy_delta_kcal_mol", "duration_s",
                    "converged_count", "converged_fraction", "is_valid"):
            value[key] = None
        value.update(stop_reason_counts={}, validity_reason="SYNTHETIC waiting-for-append demonstration")
    value["progress"] = generated_progress(value)
    dump(json_path, value)
    # The real evaluator reader verifies every completed JSON before the writer.
    return read_evaluation(json_path, split, commit)


def generate_cycle(path: Path, context: dict, cycle: int, scenario: str) -> dict:
    rng = random.Random(identity(context["seed"], context["name"], "metrics", cycle))
    starting = cycle == 1
    anchor = context.get("anchor")
    anchor_train, anchor_valid = (None, None) if starting else (anchor["train"], anchor["valid"])
    train_cost, valid_cost = (1.0, 1.03) if starting else (
        anchor_train["mean_rel_steps"], anchor_valid["mean_rel_steps"])
    if not starting:
        # Deliberate example shapes, not predictions about either named model.
        train_cost -= rng.uniform(.007, .017)
        valid_cost -= rng.uniform(.005, .016)
    if scenario == "no_improvement":
        train_cost = anchor_train["mean_rel_steps"] + rng.uniform(.0005, .055)
    elif scenario == "generalization_reject":
        valid_cost = anchor_valid["mean_rel_steps"] + rng.uniform(.001, .04)
    elif scenario == "numerical_error":
        train_cost = 1000
    train = evaluation(path, context, cycle, "train", round(train_cost, 6), scenario=scenario)
    validation = None
    if scenario not in {"no_improvement", "train_energy_reject", "numerical_error"}:
        validation = evaluation(path, context, cycle, "valid", round(valid_cost, 6),
                                scenario=scenario, pending=scenario == "pending")
    commit = train["candidate_commit"]
    description = f"SYNTHETIC DEMO {context['name']} cycle {cycle}: {scenario.replace('_', ' ')}; generated example, not research evidence"
    row = build_record(cycle=cycle, candidate_commit=commit,
        anchor_commit=commit if starting else anchor["commit"],
        train=train, validation=validation, anchor_train=anchor_train, anchor_valid=anchor_valid,
        description=description, starting_anchor=starting)
    row["timestamp"] = cycle_time(cycle, 150 if scenario != "pending" else 90)
    write_record(path / "workspace", row)
    if row["decision"] in {"anchor", "keep"}:
        context["anchor"] = {"commit": commit, "train": train, "valid": validation}
    if scenario == "pending":
        context["pending_cycle"] = cycle
    context["last_cycle"] = cycle
    context["last_scenario"] = scenario
    for filename in TABLE_NAMES:
        target = path / "workspace" / f"{filename}.tsv"
        if target.exists():
            target.chmod(0o600)
    for relative in ("workspace/run.log", "workspace/logs/SYNTHETIC-evaluations.log",
                     "workspace/.ralph/logs/SYNTHETIC-ralph-demo.log"):
        target = path / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(f"{cycle_time(cycle, 150)} {NOTICE} Cycle {cycle}: {row['decision']}; {scenario}.\n")
        target.chmod(0o600)
    return row


def save_metadata(path: Path, context: dict) -> None:
    updated = cycle_time(context["last_cycle"], 150)
    state = {"schema": 1, "name": context["name"], "run_id": context["run_id"],
        "status": "stopped", "purpose": "research", "synthetic": True, "synthetic_notice": NOTICE,
        "initial_commit": identity(context["seed"], context["name"], "candidate", 1, 40),
        "created_at": cycle_time(1), "updated_at": updated}
    dump(path / "control/state.json", state)
    dump(path / "control/demo.json", context)
    dump(path / "workspace/.run-ready.json", {"run_id": context["run_id"], "status": "ready",
        "synthetic": True, "synthetic_notice": NOTICE})
    dump(path / "workspace/.ralph/ralph-loop.state.json", {"active": False,
        "iteration": context["last_cycle"], "synthetic": True, "synthetic_notice": NOTICE,
        "reason": "Generated saved-state example; no Ralph process exists for this demo."})
    # Saved receipts are generated data, with no endpoint or executable attached.
    pending = context["pending_cycle"]
    receipts = {}
    for split in ("train", "valid"):
        evaluation_status = "complete" if split == "train" else "pending"
        value = json.loads((path / "workspace/synthetic-evaluations" /
            f"cycle-{pending:03d}-{split}-{evaluation_status}.json").read_text())
        value["progress"] = generated_progress(value)
        receipts[value["evaluation_id"]] = {"request_id": f"synthetic-cycle-{pending}-{split}",
            "evaluation_id": value["evaluation_id"], "receipt": value}
    dump(path / "workspace/.evaluation_state/client.json", {"evaluations": receipts,
        "synthetic": True, "synthetic_notice": NOTICE})


def create_demo(root: Path, seed=20260908) -> dict:
    root = root.expanduser().absolute()
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.mkdir(mode=0o700)  # Exclusive; never reuse a real or previous run root.
    marker = {"schema": 1, "synthetic": True, "generator": GENERATOR, "seed": seed,
        "notice": NOTICE, "runs": list(MODELS), "release_id": RELEASE,
        "model_labels": "Display examples only; model names do not imply these models produced the data.",
        "candidate_ids": "Deterministic generated identifiers, not real Git commits.",
        "timestamps": "Fixed synthetic timeline beginning 2026-09-08T03:00:00Z."}
    dump(root / MARKER, marker)
    for name, (provider, model, cycles) in MODELS.items():
        path = root / name
        path.mkdir(mode=0o700)
        for part in ("control", "workspace", "home", "secrets"):
            (path / part).mkdir(mode=0o700)
        context = {"synthetic": True, "name": name, "seed": seed,
            "run_id": name + "-" + identity(seed, name, "run", 0, 16)}
        dump(path / "control/runtime.json", {"provider": provider, "model": model,
            "effort": EFFORTS[name], "synthetic": True, "synthetic_notice": NOTICE})
        for cycle in range(1, cycles + 1):
            scenario = "anchor" if cycle == 1 else "pending" if cycle == cycles else SCENARIOS[(cycle - 2 + (3 if name == "demo-opus" else 0)) % len(SCENARIOS)]
            generate_cycle(path, context, cycle, scenario)
            if cycle == 1:
                for split in ("train", "valid"):
                    dump(path / f"workspace/anchors/{split}.json", context["anchor"][split])
        save_metadata(path, context)
    return verify_demo(root)


def load_demo(root: Path) -> dict:
    if root.is_symlink():
        raise ValueError("Demo root must not be a symlink")
    marker = json.loads((root / MARKER).read_text())
    if (marker.get("schema") != 1 or marker.get("synthetic") is not True
            or marker.get("generator") != GENERATOR or marker.get("runs") != list(MODELS)):
        raise ValueError("Refusing to modify a root without this generator's exact SYNTHETIC DEMO marker")
    return marker


def append_demo_cycle(root: Path, name: str) -> dict:
    """Resolve one pending example as a keep and append one new pending cycle."""
    load_demo(root)
    path = root / name
    context = json.loads((path / "control/demo.json").read_text())
    if context.get("synthetic") is not True or context.get("name") != name:
        raise ValueError("Not a generated synthetic run")
    pending = context["pending_cycle"]
    generate_cycle(path, context, pending, "keep")
    generate_cycle(path, context, pending + 1, "pending")
    save_metadata(path, context)
    return verify_demo(root)


def verify_demo(root: Path) -> dict:
    """Check schema, strict projection, anchor metadata, and truthful log labels."""
    load_demo(root)
    summaries = {}
    # Explicit stub: verification performs no Docker command or network request.
    def no_container(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "SYNTHETIC demo has no container")
    monitor = Monitor(root, docker=no_container)
    try:
        assert monitor.names() == sorted(MODELS)
        for name in MODELS:
            path = root / name
            state = json.loads((path / "control/state.json").read_text())
            tables = {table: parse_table((path / f"workspace/{table}.tsv").read_bytes())
                      for table in TABLE_NAMES}
            progress = build_progress(tables, initial_commit=state["initial_commit"])
            assert progress["state"] == "available" and progress["warnings"] == [], progress
            assert progress["points"][-1]["status"] == "pending"
            assert progress["counts"]["pending"] == 1
            assert {"accepted", "no_improvement", "validity_reject", "generalization_reject", "error", "anchor"} <= progress["counts"].keys()
            assert all("SYNTHETIC DEMO" in item["description"] for item in progress["points"])
            detail = monitor.detail(name)
            assert detail["anchors_ready"] is True and detail["anchors_state"] == "ready", detail
            assert detail["lifecycle"] == "stopped" and detail["container"]["running"] is False
            assert detail["research_progress"] == progress
            assert "SYNTHETIC DEMO" in monitor.log(name, "run-log")
            assert "SYNTHETIC DEMO" in monitor.log(name, "ralph/SYNTHETIC-ralph-demo.log")
            summaries[name] = {"cycles": len(progress["points"]), "records": progress["record_count"],
                "champions_including_anchor": len(progress["champions"]), "counts": progress["counts"],
                "warnings": progress["warnings"], "pending_cycle": progress["points"][-1]["cycle"],
                "anchors_ready": detail["anchors_ready"]}
    finally:
        monitor.close()
    return {"root": str(root), "synthetic": True, "seed": load_demo(root)["seed"], "runs": summaries}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Create a new private demo root exclusively")
    create.add_argument("root", nargs="?", type=Path)
    create.add_argument("--seed", type=int, default=20260908)
    append = sub.add_parser("append-demo-cycle", help="Resolve final pending demo cycle and append another")
    append.add_argument("root", type=Path)
    append.add_argument("--run", choices=MODELS, default="demo-astra")
    verify = sub.add_parser("verify", help="Validate generated data without external commands")
    verify.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.command == "create":
        root = args.root or Path.home() / ".local/share/gigaopt" / f"dashboard-demo-20260908-{uuid.uuid4().hex[:8]}"
        output = create_demo(root, args.seed)
    elif args.command == "append-demo-cycle":
        output = append_demo_cycle(args.root.expanduser().absolute(), args.run)
    else:
        output = verify_demo(args.root.expanduser().absolute())
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
