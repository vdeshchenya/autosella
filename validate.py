from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from distributed_validate.client import RemoteOptimizationClient
from distributed_validate.recovery import evaluate_cases, evaluation_identity
from distributed_validate.optimizer import (
    describe_optimizer_spec,
    load_minimize_func,
    normalize_optimizer_spec,
)
from utils import HARTREE_TO_KJ, MAX_FORCE_CALLS

REPO_ROOT = Path(__file__).resolve().parent
DATASET_RELEASE_ID = "main-f59f2b89a-gfn2-v1"
DEFAULT_MOLECULES_DIR = REPO_ROOT / "molecules" / "releases" / DATASET_RELEASE_ID

# Exact thermochemical calorie conversion used for energy diagnostics.
KCAL_TO_KJ = 4.184
# Validity is gated on the AGGREGATE relative energy: a candidate is valid only if, on
# average, it relaxes at least as deep as the baseline (mean_rel_energy >= 1.0). Because
# rel_energy = (candidate energy drop) / (baseline energy drop), any under-relaxed geometry
# (the failure mode behind displacement-cap / "convergence polishing" gaming) drives
# rel_energy below 1.0 and fails the gate. max_final_energy_delta_kcal_mol is still computed
# and reported, but as a diagnostic only — it no longer gates validity.
ENERGY_VALIDITY_MIN_MEAN_REL_ENERGY = 1.0
# Preserve the existing aggregate floating-point tolerance. Reproduction of a
# newly published reference release must be checked independently.
ENERGY_VALIDITY_TOL = 1e-9
INVALID_FITNESS = 1000.0


def _load_module_from_path(module_name: str, path: str | os.PathLike[str]):
    module_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load program from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def serialize_program_minimize_func(program_path: str | os.PathLike[str]) -> bytes:
    try:
        import cloudpickle
    except ImportError as exc:
        raise RuntimeError("cloudpickle is required to serialize algo.py for workers") from exc

    module_path = Path(program_path).resolve()
    module_name = f"_optimizer_candidate_{abs(hash(str(module_path))):x}"
    module = _load_module_from_path(module_name, module_path)
    minimize_fn = getattr(module, "minimize_func", None)
    if minimize_fn is None or not callable(minimize_fn):
        raise RuntimeError(f"{program_path} must define callable minimize_func")

    if hasattr(cloudpickle, "register_pickle_by_value"):
        cloudpickle.register_pickle_by_value(module)
    return cloudpickle.dumps(minimize_fn)


# Operator split registry: (reference manifest, XYZ directory), relative to the
# prepared release. Research gateway permissions restrict access to train/valid.
_SPLITS: dict[str, tuple[str, str]] = {
    "train": ("train_XTB.json", "xyz"),
    "valid": ("valid_XTB.json", "xyz"),
    "test": ("evaluation/test/ood_XTB.json", "evaluation/test/xyz"),
    "dipeptides": (
        "evaluation/dipeptides/dipeptides_XTB.json",
        "evaluation/dipeptides/xyz",
    ),
}

SPLIT_NAMES = tuple(_SPLITS)


def discover_xtb_molecules(molecules_dir: str | os.PathLike[str], split: str) -> tuple[list[str], dict]:
    molecules_path = Path(molecules_dir)
    try:
        manifest_rel, xyz_rel = _SPLITS[split]
    except KeyError:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {', '.join(SPLIT_NAMES)}"
        ) from None

    baselines_path = molecules_path / manifest_rel
    xyz_dir = molecules_path / xyz_rel
    if not baselines_path.is_file():
        raise FileNotFoundError(
            f"No baseline manifest for split {split!r} at {baselines_path}. "
            "The Sella/GFN2-xTB baseline for this split has not been generated "
            "yet; prepare matching reference metrics on main and publish a new "
            "dataset release before evaluating this split."
        )
    with open(baselines_path, "r", encoding="utf-8") as file_obj:
        baselines_raw = json.load(file_obj)

    mol_cost: dict[str, float] = {}
    mol_names: list[str] = []
    baselines: dict[str, dict] = {}
    for mol_name, metrics in baselines_raw.items():
        if Path(mol_name).name != mol_name:
            raise ValueError(f"Invalid molecule key: {mol_name!r}")
        if not all(np.isfinite(metrics[key]) for key in ("initial_energy", "final_energy")):
            raise ValueError(f"Nonfinite reference energy for {mol_name}")
        if metrics["force_calls"] <= 0 or metrics["initial_energy"] <= metrics["final_energy"]:
            raise ValueError(f"Unusable reference metrics for {mol_name}")
        mol_names.append(mol_name)
        # The Sella baseline manifests record the force-call count as
        # `force_calls`; the harness has always called the same quantity
        # `n_steps` (see program.md), so map it on read
        # rather than renaming the metric everywhere.
        xyz_path = xyz_dir / f"{mol_name}.xyz"
        # Immutable releases use the exact reference key as the XYZ stem.
        # Keep suffix inference only for legacy, unpackaged evaluation inputs.
        if (not (molecules_path / "release.json").is_file()
                and not xyz_path.is_file() and not mol_name.endswith("_mm")):
            xyz_path = xyz_dir / f"{mol_name}_mm.xyz"
        baselines[mol_name] = {
            "n_steps": metrics["force_calls"],
            "improvement": (metrics["initial_energy"] - metrics["final_energy"]) * HARTREE_TO_KJ,
            "n_atoms": metrics["n_atoms"],
            "initial_energy": metrics["initial_energy"] * HARTREE_TO_KJ,
            "final_energy": metrics["final_energy"] * HARTREE_TO_KJ,
            "method": "GFN2-xTB",
            "xyz_path": str(xyz_path),
        }
        mol_cost[mol_name] = metrics["force_calls"] * metrics["n_atoms"]

    molecules = sorted(mol_names, key=lambda name: mol_cost[name], reverse=True)
    return molecules, baselines


def _invalid_score(reason: str) -> dict:
    return {
        "fitness": INVALID_FITNESS,
        "mean_rel_steps": INVALID_FITNESS,
        "mean_rel_energy": -1.0,
        "max_final_energy_delta_kcal_mol": -1.0,
        "converged": -1.0,
        "is_valid": 0,
        "lower_is_better": True,
        "invalid_reason": reason,
    }


def score_results(results: list[dict], num_errors: int) -> dict:
    """Score a complete evaluation; convergence is a numerical diagnostic."""
    diagnostic_records = [r for r in results if isinstance(r, dict)
                          and isinstance(r.get("converged"), (bool, np.bool_))
                          and isinstance(r.get("mol_name"), str)
                          and isinstance(r.get("stop_reason", "unknown"), str)]
    malformed_count = len(results) - len(diagnostic_records)
    diagnostics = {
        "converged_count": sum(bool(r["converged"]) for r in diagnostic_records),
        "internal_error_count": num_errors + malformed_count,
        "stop_reason_counts": dict(Counter(r.get("stop_reason", "unknown") for r in diagnostic_records)),
    }
    denominator = len(results) + num_errors
    fraction = diagnostics["converged_count"] / denominator if denominator else 0.0
    diagnostics.update(converged_fraction=fraction, molecule_count=denominator)
    reason = ""
    if not results or num_errors or malformed_count:
        reason = "internal_error" if num_errors or malformed_count else "empty_evaluation"
    else:
        try:
            for r in results:
                if not all(np.isfinite(r[k]) for k in ("n_steps", "max_steps", "rel_steps", "rel_energy", "energy_delta_kcal_mol")):
                    raise ValueError("nonfinite result")
                if isinstance(r["n_steps"], (bool, np.bool_)) or not 0 < r["n_steps"] <= r["max_steps"] or r["n_steps"] != int(r["n_steps"]) or r["rel_steps"] <= 0:
                    raise ValueError("malformed force-call accounting")
                if "final_energy" in r and not np.isfinite(r["final_energy"]):
                    raise ValueError("nonfinite final energy")
                if "final_positions" in r:
                    positions = np.asarray(r["final_positions"], dtype=float)
                    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
                        raise ValueError("malformed final geometry")
        except (KeyError, TypeError, ValueError):
            reason = "internal_error"
            diagnostics["internal_error_count"] += 1
    if reason:
        score = _invalid_score(reason)
        score.update(diagnostics, validity_reason=reason)
        score["converged"] = fraction
        return score
    mean_rel_steps = float(np.mean([r["rel_steps"] for r in results]))
    mean_rel_energy = float(np.mean([r["rel_energy"] for r in results]))
    if not np.isfinite(mean_rel_steps) or not np.isfinite(mean_rel_energy):
        score = _invalid_score("internal_error")
        diagnostics["internal_error_count"] += 1
        score.update(diagnostics, validity_reason="internal_error", converged=fraction)
        return score
    valid = int(mean_rel_energy >= ENERGY_VALIDITY_MIN_MEAN_REL_ENERGY - ENERGY_VALIDITY_TOL)
    reason = "" if valid else "energy_below_baseline"
    return {
        **diagnostics,
        "fitness": mean_rel_steps,
        "mean_rel_steps": mean_rel_steps,
        "mean_rel_energy": mean_rel_energy,
        "max_final_energy_delta_kcal_mol": float(max(r["energy_delta_kcal_mol"] for r in results)),
        "converged": fraction,
        "is_valid": valid,
        "lower_is_better": True,
        "invalid_reason": reason,
        "validity_reason": reason,
    }


def build_evaluation_record(result: dict, *, split: str, molecule_count: int,
                            duration_s: float = 0.0, **metadata) -> dict:
    """One numerical record shared by CLI, gateway, and acceptance writers."""
    # Reject malformed measurements once, before coverage checks or JSON output.
    # Raw NaN/Inf values must not prevent the invalid evaluation being recorded.
    if result.get("status", "complete") == "complete":
        clean_results, malformed = [], []
        for index, measurement in enumerate(result["results"]):
            if score_results([measurement], 0)["validity_reason"] == "internal_error":
                name = measurement.get("mol_name", f"result_{index}") if isinstance(measurement, dict) else f"result_{index}"
                malformed.append((str(name), "malformed_worker_measurement: " + repr(measurement)[:2000]))
            else:
                clean_results.append(measurement)
        if malformed:
            result = {**result, "results": clean_results,
                      "errors": list(result["errors"]) + malformed,
                      "num_errors": result["num_errors"] + len(malformed)}
    status = result.get("status", "complete")
    if status == "complete" and len(result["results"]) + result["num_errors"] != molecule_count:
        status = "pending"
        result = {**result, "pending_reason": "incomplete_molecule_coverage"}
    if status == "complete":
        names = [r["mol_name"] for r in result["results"]]
        if len(names) != len(set(names)):
            status = "pending"
            result = {**result, "pending_reason": "duplicate_molecule_results"}
    if status == "complete":
        score = score_results(result["results"], result["num_errors"])
    else:
        score = {k: None for k in ("fitness", "mean_rel_steps", "mean_rel_energy",
                  "max_final_energy_delta_kcal_mol", "converged", "converged_fraction",
                  "converged_count", "is_valid")}
        score.update(invalid_reason="", validity_reason="infrastructure_pending",
                     internal_error_count=result.get("num_errors", 0), stop_reason_counts={})
    return {
        **metadata, **score, "status": status, "split": split,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset_release_id": metadata.get("dataset_release_id", DATASET_RELEASE_ID),
        "method": "GFN2-xTB", "duration_s": duration_s,
        "molecule_count": molecule_count, "num_results": len(result["results"]),
        "num_errors": result["num_errors"],
        "infrastructure_retries": result.get("infrastructure_retries", 0),
        "recovery_events": result.get("recovery_events", []),
        "recovery_state_path": result.get("recovery_state_path"),
        "pending_reason": result.get("pending_reason", ""),
        "results": result["results"], "errors": result["errors"],
        "failure_details": result.get("failure_details", []),
    }


class Evaluator:
    def __init__(
        self,
        molecules_dir: str | os.PathLike[str] = DEFAULT_MOLECULES_DIR,
        split: str = "train",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        poll_interval_seconds: float = 0.1,
        task_timeout_xtb: float = 3600.0,
    ):
        self.molecules_dir = str(molecules_dir)
        self.split = split
        self.molecules, self.baselines = discover_xtb_molecules(self.molecules_dir, split)
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.poll_interval_seconds = poll_interval_seconds
        self.task_timeout_xtb = task_timeout_xtb

    def evaluate(self, optimizer: Callable | str | dict[str, Any], verbose: bool = False,
                 *, state_path=None, task_metadata=None, should_continue=None) -> dict:
        optimizer_spec = normalize_optimizer_spec(optimizer)
        # Source specifications stay inert here; only the guarded worker loads code.
        cases = [{"mol_name": name, "baseline": self.baselines[name],
                  "max_steps": MAX_FORCE_CALLS} for name in self.molecules]
        if not cases:
            return {"status": "complete", "results": [], "num_errors": 0, "errors": []}
        if state_path is None:
            identity = evaluation_identity(cases, optimizer_spec)
            state_path = REPO_ROOT / ".evaluation_state" / f"{self.split}-{identity}.json"
        factory = lambda: RemoteOptimizationClient(
            redis_host=self.redis_host, redis_port=self.redis_port,
            redis_db=self.redis_db,
            poll_interval_seconds=self.poll_interval_seconds)
        return evaluate_cases(factory, cases, optimizer_spec, state_path,
                              timeout_seconds=self.task_timeout_xtb,
                              task_metadata=task_metadata, should_continue=should_continue)


def _append_run_log(
    log_path: str | os.PathLike[str],
    *,
    optimizer_description: str,
    split: str,
    num_molecules: int,
    duration_s: float,
    results: list[dict],
    errors: list[tuple[str, str]],
    score: dict,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    sorted_results = sorted(results, key=lambda item: item["mol_name"])
    with open(log_path, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"=== {timestamp} ===\n")
        file_obj.write(f"optimizer: {optimizer_description}\n")
        file_obj.write("mode: xtb\n")
        file_obj.write(f"split: {split}\n")
        file_obj.write(f"molecules: {num_molecules}\n")
        file_obj.write(f"duration: {duration_s:.1f}s\n")
        file_obj.write(f"results: {len(results)}\n")
        file_obj.write(f"errors: {len(errors)}\n")
        for mol_name, message in errors:
            file_obj.write(f"error: {mol_name}: {message}\n")

        if score.get("status") == "pending" or not results:
            file_obj.write(f"outcome: {score.get('status', 'complete')}\n")
            file_obj.write(json.dumps(score, sort_keys=True) + "\n")
        else:
            file_obj.write(
                "molecule\tn_steps\tmax_steps\trel_steps\trel_energy\t"
                "energy_delta_kcal_mol\tconv\tstop_reason\n"
            )
            for result in sorted_results:
                file_obj.write(
                    f"{result['mol_name']}\t{result['n_steps']}\t{result['max_steps']}\t"
                    f"{result['rel_steps']:.6f}\t{result['rel_energy']:.6f}\t"
                    f"{result['energy_delta_kcal_mol']:.6f}\t"
                    f"{int(result['converged'])}\t{result.get('stop_reason', 'unknown')}\n"
                )
            file_obj.write(f"mean_rel_steps = {score['mean_rel_steps']:.6f}\n")
            file_obj.write(f"mean_rel_energy = {score['mean_rel_energy']:.6f}\n")
            file_obj.write(
                "max_final_energy_delta_kcal_mol = "
                f"{score['max_final_energy_delta_kcal_mol']:.6f}\n"
            )
            file_obj.write(f"converged = {score['converged']:.4f}\n")
            file_obj.write(f"is_valid = {score['is_valid']}\n")
            file_obj.write(json.dumps(score, sort_keys=True) + "\n")
        file_obj.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the editable optimizer.")
    parser.add_argument("--program", default="algo.py", help="Optimizer file to evaluate")
    parser.add_argument("--molecules-dir", default=str(DEFAULT_MOLECULES_DIR))
    parser.add_argument("--split", choices=list(SPLIT_NAMES), default="train")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    parser.add_argument("--run-log", default="run.log")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--state-path")
    parser.add_argument("--output-json", default="evaluation.json")
    args = parser.parse_args()

    optimizer_spec = normalize_optimizer_spec({"kind": "source", "source": Path(args.program).read_text()})
    evaluator = Evaluator(
        molecules_dir=args.molecules_dir,
        split=args.split,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db,
    )
    start_time = time.time()
    result = evaluator.evaluate(optimizer_spec, verbose=args.verbose, state_path=args.state_path)
    duration_s = time.time() - start_time
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(args.program).resolve().parent,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    source_sha = hashlib.sha256(Path(args.program).read_bytes()).hexdigest()
    score = build_evaluation_record(result, split=args.split, molecule_count=len(evaluator.molecules),
                                    duration_s=duration_s, candidate_commit=commit,
                                    program_id=source_sha, evaluation_id=Path(result.get("recovery_state_path") or args.output_json).stem,
                                    evaluation_log=str(Path(args.run_log).resolve()))
    Path(args.output_json).write_text(json.dumps(score, sort_keys=True, allow_nan=False) + "\n")

    _append_run_log(
        args.run_log,
        optimizer_description=describe_optimizer_spec(optimizer_spec),
        split=args.split,
        num_molecules=len(evaluator.molecules),
        duration_s=duration_s,
        results=score["results"],
        errors=score["errors"],
        score=score,
    )

    with open(REPO_ROOT / "validate_debug.log", "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps({k: v for k, v in score.items() if k != "results"}, sort_keys=True) + "\n")
    print(json.dumps(score, sort_keys=True, allow_nan=False))
    if score["status"] == "pending":
        raise SystemExit(75)



if __name__ == "__main__":
    main()
