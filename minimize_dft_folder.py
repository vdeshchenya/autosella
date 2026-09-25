#!/usr/bin/env python
"""
Minimize a folder of XYZ files with DFT forces from ORCA/OPI.

For each input ``name.xyz`` this writes:
  output_dir/name_final.xyz
  output_dir/name_info.txt
and records the full dataset in ``output_dir/summary.tsv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convergence import init_convergence_state, update_convergence_state
from dft_molecular_system import DFTMolecularSystem
from utils import HARTREE_TO_KJ, MAX_FORCE_CALLS, read_xyz_charge, write_xyz


MULTIPLICITY = 1


class _BudgetExhausted(Exception):
    pass


def _hartree(energy_kj: float) -> float:
    return float(energy_kj) / HARTREE_TO_KJ


def _relative_display_path(path: str | Path) -> str:
    """Return a path relative to the invocation directory for metadata."""
    return os.path.relpath(Path(path).expanduser().resolve(), Path.cwd().resolve())


def _write_info(path: Path, info: dict) -> None:
    lines = []
    for key, value in info.items():
        lines.append(f"{key}: {value}\n")
    path.write_text("".join(lines))


def _load_minimize_func(minimizer_path: str | None) -> tuple[Callable, str]:
    path = Path(minimizer_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"minimizer file does not exist: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = f"custom_minimizer_{path.stem}_{abs(hash(path))}"
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


def _collect_xyz_files(xyz_folder: Path) -> list[Path]:
    """Return every top-level XYZ file in deterministic filename order."""
    return sorted(path for path in xyz_folder.glob("*.xyz") if path.is_file())


def _electron_count(atomic_numbers: np.ndarray, charge: int) -> int:
    return int(np.sum(atomic_numbers)) - int(charge)


def run_one(
    xyz_path: Path,
    output_dir: Path,
    minimize_func: Callable,
    minimizer_label: str,
    ncores: int,
    max_force_calls: int = MAX_FORCE_CALLS,
) -> dict:
    mol_id = xyz_path.stem
    final_xyz = output_dir / f"{mol_id}_final.xyz"
    info_txt = output_dir / f"{mol_id}_info.txt"

    start_time = time.time()
    system = DFTMolecularSystem(
        str(xyz_path),
        multiplicity=MULTIPLICITY,
        ncores=ncores,
    )
    atomic_numbers = system.atomic_numbers
    initial_pos = system.initial_positions.copy()
    mol_charge = system.charge
    electron_count = _electron_count(atomic_numbers, mol_charge)
    if electron_count % 2 != (MULTIPLICITY - 1) % 2:
        raise RuntimeError(
            f"charge/multiplicity mismatch for {mol_id}: charge={mol_charge}, "
            f"multiplicity={MULTIPLICITY}, electrons={electron_count}"
        )

    state = init_convergence_state("geometric", initial_pos)
    energies_kj: list[float] = []
    ever_converged = False
    n_calls = 0
    final_pos = initial_pos.copy()
    status = "ok"

    def calc(pos):
        nonlocal n_calls, ever_converged, final_pos
        if n_calls >= max_force_calls:
            raise _BudgetExhausted
        pos_arr = np.asarray(pos, dtype=float)
        n_calls += 1
        energy, forces = system.compute(pos_arr)
        if not np.isfinite(energy):
            raise RuntimeError(system.last_error or "DFT calculation failed")
        final_pos = pos_arr.copy()
        energies_kj.append(float(energy))
        if update_convergence_state(state, pos_arr, energy, forces):
            ever_converged = True
        return energy, forces

    def check_converged() -> bool:
        return ever_converged

    try:
        final_pos, optimizer_calls = minimize_func(
            initial_pos,
            atomic_numbers,
            calc,
            max_force_calls=max_force_calls,
            converged=check_converged,
        )
        if optimizer_calls != n_calls:
            raise RuntimeError(
                "force-call count mismatch: "
                f"evaluator recorded {n_calls}, optimizer reported {optimizer_calls}"
            )
    except _BudgetExhausted:
        status = "budget_exhausted"

    if not energies_kj:
        raise RuntimeError("optimizer made zero force calls")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(atomic_numbers, final_pos, str(final_xyz), mol_charge)

    initial_energy_kj = energies_kj[0]
    final_energy_kj = energies_kj[-1]
    elapsed_seconds = time.time() - start_time
    elapsed_seconds_per_step = elapsed_seconds / max(1, n_calls)
    info = {
        "molecule": mol_id,
        "input_xyz": _relative_display_path(xyz_path),
        "final_xyz": _relative_display_path(final_xyz),
        "method": "r2SCAN-3c",
        "basis": "r2SCAN-3c default",
        "task": "ENGRAD",
        "scf": "TightSCF",
        "minimizer": minimizer_label,
        "charge": mol_charge,
        "multiplicity": MULTIPLICITY,
        "ncores": ncores,
        "max_force_calls": max_force_calls,
        "n_calls": n_calls,
        "converged": ever_converged,
        "status": status,
        "initial_energy_kj_mol": f"{initial_energy_kj:.12f}",
        "final_energy_kj_mol": f"{final_energy_kj:.12f}",
        "delta_energy_kj_mol": f"{final_energy_kj - initial_energy_kj:.12f}",
        "initial_energy_hartree": f"{_hartree(initial_energy_kj):.12f}",
        "final_energy_hartree": f"{_hartree(final_energy_kj):.12f}",
        "delta_energy_hartree": f"{_hartree(final_energy_kj - initial_energy_kj):.12f}",
        "elapsed_seconds": f"{elapsed_seconds:.1f}",
        "elapsed_seconds_per_step": f"{elapsed_seconds_per_step:.1f}",
    }
    _write_info(info_txt, info)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Minimize all .xyz files in a folder using ORCA/OPI DFT forces."
    )
    parser.add_argument("xyz_folder", help="Folder containing .xyz files")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="output",
        help="Directory for *_final.xyz and *_info.txt files (default: output)",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=psutil.cpu_count(logical=False) or 1,
        help=(
            "ORCA cores per energy-and-gradient calculation "
            "(default: physical CPU count, falling back to 1)"
        ),
    )
    parser.add_argument(
        "--max-force-calls",
        type=int,
        default=MAX_FORCE_CALLS,
        help=f"Force-evaluation budget per molecule (default: {MAX_FORCE_CALLS})",
    )
    parser.add_argument(
        "--minimizer-path",
        default="minimizers/sella_baseline.py",
        help=(
            "Path to a Python file defining minimize_func. "
            "Defaults to minimizers/sella_baseline.py."
        ),
    )
    args = parser.parse_args()
    if args.ncores < 1:
        parser.error("--ncores must be at least 1")
    if args.max_force_calls < 1:
        parser.error("--max-force-calls must be at least 1")

    xyz_folder = Path(args.xyz_folder)
    output_dir = Path(args.output_dir)
    minimize_func, minimizer_label = _load_minimize_func(args.minimizer_path)
    xyz_files = _collect_xyz_files(xyz_folder)

    if not xyz_files:
        print(f"No .xyz files found in {xyz_folder}", flush=True)
        return 0

    for xyz_path in xyz_files:
        read_xyz_charge(str(xyz_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.tsv"
    for pattern in ("*_final.xyz", "*_info.txt"):
        for generated_path in output_dir.glob(pattern):
            generated_path.unlink()
    summary_path.unlink(missing_ok=True)

    print(
        f"Minimizing {len(xyz_files)} molecule(s), one at a time, "
        f"with {args.ncores} ORCA core(s) per force call. "
        f"Minimizer: {minimizer_label}",
        flush=True,
    )

    successes = 0
    failures = 0
    with summary_path.open("w") as summary:
        summary.write(f"# minimizer: {minimizer_label}\n")
        summary.write(
            "mol\tcharge\tstatus\tconverged\tn_calls\tinitial_Eh\tfinal_Eh\tdelta_Eh\t"
            "elapsed_s\telapsed_s_per_step\n"
        )
        for idx, xyz_path in enumerate(xyz_files, 1):
            mol_id = xyz_path.stem
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{idx}/{len(xyz_files)} {timestamp}] {mol_id}", flush=True)
            try:
                info = run_one(
                    xyz_path=xyz_path,
                    output_dir=output_dir,
                    minimize_func=minimize_func,
                    minimizer_label=minimizer_label,
                    ncores=args.ncores,
                    max_force_calls=args.max_force_calls,
                )
                successes += 1
                summary.write(
                    f"{mol_id}\t{info['charge']}\t{info['status']}\t{info['converged']}\t"
                    f"{info['n_calls']}\t{info['initial_energy_hartree']}\t"
                    f"{info['final_energy_hartree']}\t{info['delta_energy_hartree']}\t"
                    f"{info['elapsed_seconds']}\t{info['elapsed_seconds_per_step']}\n"
                )
                summary.flush()
                os.fsync(summary.fileno())
                print(
                    f"  done: converged={info['converged']} "
                    f"calls={info['n_calls']}",
                    flush=True,
                )
            except Exception:
                failures += 1
                err_path = output_dir / f"{mol_id}_info.txt"
                err_text = traceback.format_exc()
                _write_info(
                    err_path,
                    {
                        "molecule": mol_id,
                        "input_xyz": _relative_display_path(xyz_path),
                        "minimizer": minimizer_label,
                        "status": "failed",
                        "error": err_text,
                    },
                )
                summary.write(
                    f"{mol_id}\tnan\tfailed\tFalse\t0\tnan\tnan\tnan\tnan\tnan\n"
                )
                summary.flush()
                os.fsync(summary.fileno())
                print(f"  FAIL {mol_id}: see {err_path}", flush=True)

    print(
        f"Finished: {successes} succeeded, {failures} failed. "
        f"Summary: {summary_path}",
        flush=True,
    )
    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
