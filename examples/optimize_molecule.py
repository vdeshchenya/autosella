#!/usr/bin/env python3
"""Optimize a charge-tagged XYZ with AutoSella and GFN2-xTB.

Run from the repository root, for example:

    python examples/optimize_molecule.py examples/mol.xyz --optimizer g
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molecules.convergence import (
    init_convergence_state,
    update_convergence_state,
)
from molecules.utils import MAX_FORCE_CALLS, write_xyz
from molecules.xtb_molecular_system import XTBMolecularSystem

OPTIMIZER_MODULES = {
    "g": "minimizers.autosella_g",
    "a": "minimizers.autosella_a",
    "f": "minimizers.autosella_f",
    "sella": "minimizers.sella_wrapper",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xyz",
        type=Path,
        help="Input XYZ with `charge=<integer>` on its second line.",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZER_MODULES),
        default="g",
        help="Optimizer: g (default), a, f, or the Sella reference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output XYZ path (default: <input>_optimized.xyz).",
    )
    parser.add_argument(
        "--max-force-calls",
        type=int,
        default=MAX_FORCE_CALLS,
        help=f"Hard force-call budget (default: {MAX_FORCE_CALLS}).",
    )
    parser.add_argument(
        "--xtb-accuracy",
        type=float,
        default=0.2,
        help="xTB SCF accuracy parameter (default: 0.2).",
    )
    parser.add_argument(
        "--xtb-threads",
        type=int,
        default=1,
        help="Threads used by each xTB calculation (default: 1).",
    )
    parser.add_argument(
        "--verbose-xtb",
        action="store_true",
        help="Show xTB output for every force call.",
    )
    return parser.parse_args()


def optimize(args: argparse.Namespace) -> Path:
    if args.max_force_calls < 1:
        raise ValueError("--max-force-calls must be positive")

    xyz_path = args.xyz.resolve()
    if not xyz_path.is_file():
        raise FileNotFoundError(xyz_path)

    output_path = args.output
    if output_path is None:
        output_path = xyz_path.with_name(f"{xyz_path.stem}_optimized.xyz")
    else:
        output_path = output_path.resolve()

    module = importlib.import_module(OPTIMIZER_MODULES[args.optimizer])
    minimize_func = module.entrypoint()

    system = XTBMolecularSystem(
        str(xyz_path),
        method="GFN2-xTB",
        accuracy=args.xtb_accuracy,
        verbose=args.verbose_xtb,
        num_threads=args.xtb_threads,
    )
    initial_positions = system.initial_positions.copy()
    state = init_convergence_state("geometric", initial_positions)
    energies_kj_mol: list[float] = []

    def calc(positions_nm: np.ndarray) -> tuple[float, np.ndarray]:
        energy, forces = system.compute(positions_nm)
        energies_kj_mol.append(float(energy))
        update_convergence_state(state, positions_nm, energy, forces)
        return energy, forces

    def converged() -> bool:
        return state.converged

    final_positions, force_calls = minimize_func(
        initial_positions,
        system.atomic_numbers,
        calc,
        max_force_calls=args.max_force_calls,
        converged=converged,
    )
    if not energies_kj_mol:
        raise RuntimeError("optimizer completed without an energy/force call")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_xyz(
        system.atomic_numbers,
        final_positions,
        str(output_path),
        charge=system.charge,
    )

    improvement = energies_kj_mol[0] - energies_kj_mol[-1]
    print(f"optimizer: {args.optimizer}")
    print(f"converged: {state.converged}")
    print(f"force calls: {force_calls}/{args.max_force_calls}")
    print(f"energy improvement: {improvement/4.184:.2f} kcal/mol")
    print(f"optimized geometry: {output_path}")
    return output_path


def main() -> None:
    optimize(parse_args())


if __name__ == "__main__":
    main()
