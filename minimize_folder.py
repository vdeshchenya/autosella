#!/usr/bin/env python
"""
Minimize a folder of XYZ files on the xTB or force-field potential.

For each input ``name.xyz`` this writes:
  output_dir/name_final.xyz
  output_dir/name_info.txt
and records the full dataset in ``output_dir/summary.tsv``.
"""

from __future__ import annotations

import os

for _var, _val in (
    ("OMP_NUM_THREADS", "1"),
    ("OMP_THREAD_LIMIT", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OMP_DYNAMIC", "FALSE"),
    ("MKL_DYNAMIC", "FALSE"),
    ("JAX_PLATFORMS", "cpu"),
):
    os.environ.setdefault(_var, _val)

import argparse
import importlib.util
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convergence import init_convergence_state, update_convergence_state
from utils import HARTREE_TO_KJ, MAX_FORCE_CALLS, read_xyz_charge, write_xyz


KCAL_TO_KJ = 4.184
POTENTIALS = ("xtb", "ff")

# Both GFN potentials use the GeomeTRIC/Gaussian-style test on energy, gradient and displacement.
CONVERGENCE_MODE = "geometric"

SUMMARY_COLUMNS = (
    "mol", "charge", "status", "converged", "n_calls",
    "initial_kcal_mol", "final_kcal_mol", "delta_kcal_mol",
    "elapsed_s", "elapsed_s_per_step",
)


class _BudgetExhausted(Exception):
    pass


def _hartree(energy_kj: float) -> float:
    return float(energy_kj) / HARTREE_TO_KJ


def _relative_display_path(path: str | Path) -> str:
    """Return a path relative to the invocation directory for metadata."""
    return os.path.relpath(Path(path).expanduser().resolve(), Path.cwd().resolve())


# ── loading ───────────────────────────────────────────────────────────────────

def _load_minimize_func(minimizer_path: str | None) -> tuple[Callable, str]:
    path = Path(minimizer_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"minimizer file does not exist: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = f"custom_minimizer_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load minimizer module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    minimize_func = getattr(module, "minimize_func", None)
    if not callable(minimize_func):
        raise AttributeError(f"{path} does not define a callable minimize_func")

    return minimize_func, _relative_display_path(path)


def _build_system(potential: str, xyz_path: Path, input_dir: Path, xtb_method: str):
    """Return (system, extra_info) for one molecule."""
    if potential == "xtb":
        from xtb_molecular_system import XTBMolecularSystem

        system = XTBMolecularSystem(str(xyz_path), method=xtb_method, num_threads=1)
        return system, {"method": xtb_method, "accuracy": system.accuracy}

    from ff_molecular_system import FFMolecularSystem

    system = FFMolecularSystem(str(xyz_path), num_threads=1)
    return system, {"method": system.method, "topology_sha256": system.topology_sha256}


# ── input collection ──────────────────────────────────────────────────────────

def _collect_xyz_files(xyz_folder: Path) -> list[Path]:
    """Return every top-level XYZ file in deterministic filename order."""
    return sorted(path for path in xyz_folder.glob("*.xyz") if path.is_file())


# ── one molecule ──────────────────────────────────────────────────────────────

def _write_info(path: Path, info: dict) -> None:
    path.write_text("".join(f"{key}: {value}\n" for key, value in info.items()))


def run_one(
    xyz_path: Path,
    input_dir: Path,
    output_dir: Path,
    potential: str,
    minimize_func: Callable,
    minimizer_label: str,
    xtb_method: str,
    max_force_calls: int = MAX_FORCE_CALLS,
) -> dict:
    mol_id = xyz_path.stem
    final_xyz = output_dir / f"{mol_id}_final.xyz"
    info_txt = output_dir / f"{mol_id}_info.txt"

    start_time = time.time()
    system, extra = _build_system(potential, xyz_path, input_dir, xtb_method)
    atomic_numbers = system.atomic_numbers
    initial_pos = system.initial_positions.copy()
    mol_charge = system.charge

    state = init_convergence_state(CONVERGENCE_MODE, initial_pos)
    ever_converged = False
    n_calls = 0
    final_pos = initial_pos.copy()
    status = "ok"

    def calc(pos):
        nonlocal n_calls, ever_converged, final_pos
        if n_calls >= max_force_calls:
            raise _BudgetExhausted
        pos_arr = np.asarray(pos, dtype=float)
        energy, forces = system.compute(pos_arr)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise RuntimeError(f"{potential} returned non-finite energy or forces")
        n_calls += 1
        final_pos = pos_arr.copy()
        if update_convergence_state(state, pos_arr, energy, forces):
            ever_converged = True
        return energy, forces

    initial_energy_kj = system.compute(initial_pos)[0]

    try:
        returned_pos, returned_calls = minimize_func(
            initial_pos.copy(),
            atomic_numbers,
            calc,
            max_force_calls=max_force_calls,
            converged=lambda: ever_converged,
        )
        if isinstance(returned_calls, bool):
            raise RuntimeError("optimizer returned a boolean force-call count")
        returned_calls_int = int(returned_calls)
        if returned_calls_int != returned_calls:
            raise RuntimeError(f"optimizer returned non-integral force_calls={returned_calls!r}")
        if returned_calls_int != n_calls:
            raise RuntimeError(
                f"force-call mismatch: counted={n_calls}, returned={returned_calls_int}"
            )
        returned_pos_arr = np.asarray(returned_pos, dtype=float)
        if returned_pos_arr.shape != initial_pos.shape or not np.all(
            np.isfinite(returned_pos_arr)
        ):
            raise ValueError("optimizer returned invalid final positions")
        final_pos = returned_pos_arr
    except _BudgetExhausted:
        status = "budget_exhausted"

    if not n_calls:
        raise RuntimeError("optimizer made zero force calls")

    final_energy_kj = system.compute(final_pos)[0]
    if not np.isfinite(initial_energy_kj) or not np.isfinite(final_energy_kj):
        raise ValueError(f"{potential} returned non-finite scoring energy")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(atomic_numbers, final_pos, str(final_xyz), mol_charge)

    elapsed_seconds = time.time() - start_time
    elapsed_seconds_per_step = elapsed_seconds / max(1, n_calls)
    info = {
        "molecule": mol_id,
        "input_xyz": _relative_display_path(xyz_path),
        "final_xyz": _relative_display_path(final_xyz),
        "potential": potential,
        **extra,
        "minimizer": minimizer_label,
        "charge": mol_charge,
        "n_atoms": int(len(atomic_numbers)),
        "max_force_calls": max_force_calls,
        "n_calls": n_calls,
        "converged": ever_converged,
        "status": status,
        "initial_energy_kj_mol": f"{initial_energy_kj:.12f}",
        "final_energy_kj_mol": f"{final_energy_kj:.12f}",
        "delta_energy_kj_mol": f"{final_energy_kj - initial_energy_kj:.12f}",
    }
    if potential == "xtb":
        info.update(
            {
                "initial_energy_hartree": f"{_hartree(initial_energy_kj):.12f}",
                "final_energy_hartree": f"{_hartree(final_energy_kj):.12f}",
                "delta_energy_hartree": f"{_hartree(final_energy_kj - initial_energy_kj):.12f}",
            }
        )
    info["elapsed_seconds"] = f"{elapsed_seconds:.1f}"
    info["elapsed_seconds_per_step"] = f"{elapsed_seconds_per_step:.1f}"
    _write_info(info_txt, info)
    return info


# ── worker plumbing ───────────────────────────────────────────────────────────

_WORKER: dict = {}


def _init_worker(config: dict) -> None:
    minimize_func, minimizer_label = _load_minimize_func(config["minimizer_path"])
    _WORKER.update(config)
    _WORKER["minimize_func"] = minimize_func
    _WORKER["minimizer_label"] = minimizer_label


def _work(xyz_path_str: str) -> tuple[str, dict | None, str | None]:
    xyz_path = Path(xyz_path_str)
    output_dir = Path(_WORKER["output_dir"])
    try:
        info = run_one(
            xyz_path=xyz_path,
            input_dir=Path(_WORKER["input_dir"]),
            output_dir=output_dir,
            potential=_WORKER["potential"],
            minimize_func=_WORKER["minimize_func"],
            minimizer_label=_WORKER["minimizer_label"],
            xtb_method=_WORKER["xtb_method"],
            max_force_calls=_WORKER["max_force_calls"],
        )
        return xyz_path.stem, info, None
    except Exception:
        err_text = traceback.format_exc()
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_info(
            output_dir / f"{xyz_path.stem}_info.txt",
            {
                "molecule": xyz_path.stem,
                "input_xyz": _relative_display_path(xyz_path),
                "potential": _WORKER["potential"],
                "minimizer": _WORKER.get("minimizer_label", "?"),
                "status": "failed",
                "error": err_text,
            },
        )
        return xyz_path.stem, None, err_text


# ── summary I/O ───────────────────────────────────────────────────────────────

def _summary_row(mol_id: str, info: dict | None) -> str:
    if info is None:
        return "\t".join([mol_id, "nan", "failed", "False", "0",
                          "nan", "nan", "nan", "nan", "nan"]) + "\n"
    return "\t".join(
        [
            mol_id,
            str(info["charge"]),
            info["status"],
            str(info["converged"]),
            str(info["n_calls"]),
            f"{float(info['initial_energy_kj_mol']) / KCAL_TO_KJ:.12f}",
            f"{float(info['final_energy_kj_mol']) / KCAL_TO_KJ:.12f}",
            f"{float(info['delta_energy_kj_mol']) / KCAL_TO_KJ:.12f}",
            info["elapsed_seconds"],
            info["elapsed_seconds_per_step"],
        ]
    ) + "\n"


def _rewrite_sorted(summary_path: Path, header_lines: list[str]) -> None:
    """Dedupe by molecule (last row wins) and rewrite in sorted order."""
    if not summary_path.exists():
        return
    with summary_path.open() as fh:
        lines = fh.readlines()
    comments = [ln for ln in lines if ln.startswith("#")] or header_lines[:-1]
    body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    if not body:
        return
    column_line = body[0] if body[0].startswith("mol\t") else header_lines[-1]
    rows = {}
    for line in body:
        if line.startswith("mol\t"):
            continue
        rows[line.split("\t", 1)[0]] = line
    tmp = summary_path.with_suffix(".tsv.tmp")
    with tmp.open("w") as fh:
        fh.writelines(comments)
        fh.write(column_line)
        for mol_id in sorted(rows):
            fh.write(rows[mol_id])
    os.replace(tmp, summary_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimize all .xyz files in a folder on the xTB or FF potential."
    )
    parser.add_argument("xyz_folder", help="Folder containing .xyz files")
    parser.add_argument(
        "--potential",
        required=True,
        choices=POTENTIALS,
        help="'xtb' (GFN2-xTB) or 'ff' (GFN-FF), via the xtb binary",
    )
    parser.add_argument("-o", "--output-dir", default="output",
                        help="Directory for *_final.xyz, *_info.txt and summary.tsv")
    parser.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="Worker processes (default: cpu_count - 2)")
    parser.add_argument("--max-force-calls", type=int, default=MAX_FORCE_CALLS,
                        help=f"Force-evaluation budget per molecule (default: {MAX_FORCE_CALLS})")
    parser.add_argument(
        "--minimizer-path",
        default="minimizers/sella_baseline.py",
        help=(
            "Path to a Python file defining minimize_func. "
            "Defaults to minimizers/sella_baseline.py."
        ),
    )
    parser.add_argument("--xtb-method", default="GFN2-xTB",
                        choices=["GFN2-xTB", "GFN1-xTB"],
                        help="xTB Hamiltonian; ignored for --potential ff")
    args = parser.parse_args()
    if args.max_force_calls < 1 or args.procs < 1:
        parser.error("--max-force-calls and --procs must be positive integers")

    xyz_folder = Path(args.xyz_folder)
    output_dir = Path(args.output_dir)
    xyz_files = _collect_xyz_files(xyz_folder)

    if not xyz_files:
        print(f"No .xyz files found in {xyz_folder}", flush=True)
        return 0

    for xyz_path in xyz_files:
        read_xyz_charge(str(xyz_path))

    # Fail fast on a bad minimizer path before spawning the pool.
    _, minimizer_label = _load_minimize_func(args.minimizer_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.tsv"
    for pattern in ("*_final.xyz", "*_info.txt"):
        for generated_path in output_dir.glob(pattern):
            generated_path.unlink()
    summary_path.unlink(missing_ok=True)

    header_lines = [
        f"# potential: {args.potential}\n",
        f"# minimizer: {minimizer_label}\n",
        f"# inputs: {xyz_folder}\n",
        f"# max_force_calls: {args.max_force_calls}\n",
        "# energy_unit: kcal/mol\n",
        "\t".join(SUMMARY_COLUMNS) + "\n",
    ]

    print(
        f"Minimizing {len(xyz_files)} molecule(s) on the {args.potential} potential "
        f"with {args.procs} process(es) "
        f"Minimizer: {minimizer_label}",
        flush=True,
    )

    config = {
        "minimizer_path": args.minimizer_path,
        "potential": args.potential,
        "input_dir": str(xyz_folder),
        "output_dir": str(output_dir),
        "xtb_method": args.xtb_method,
        "max_force_calls": args.max_force_calls,
    }

    successes = failures = 0
    start = time.time()

    with summary_path.open("w") as summary:
        summary.writelines(header_lines)
        summary.flush()

        if xyz_files:
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.procs, initializer=_init_worker, initargs=(config,)) as pool:
                iterator = pool.imap_unordered(
                    _work, [str(p) for p in xyz_files], chunksize=1
                )
                for index, (mol_id, info, err) in enumerate(iterator, 1):
                    summary.write(_summary_row(mol_id, info))
                    summary.flush()
                    os.fsync(summary.fileno())
                    if err is None:
                        successes += 1
                    else:
                        failures += 1
                        print(f"  FAIL {mol_id}: {err.strip().splitlines()[-1]}", flush=True)
                    if index % 25 == 0 or index == len(xyz_files):
                        rate = index / max(1e-9, time.time() - start)
                        remaining = (len(xyz_files) - index) / max(1e-9, rate)
                        print(
                            f"  {index}/{len(xyz_files)}  {time.time() - start:.0f}s elapsed, "
                            f"~{remaining:.0f}s left, {failures} failed",
                            flush=True,
                        )

    _rewrite_sorted(summary_path, header_lines)

    print(
        f"Finished: {successes} completed, {failures} failed "
        f"in {time.time() - start:.0f}s. Summary: {summary_path}",
        flush=True,
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
