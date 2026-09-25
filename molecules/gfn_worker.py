"""Sella/GFN2-xTB candidate optimization."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

IMPROVEMENT_MIN = 0.005  # Hartree

# Apply before importing numerical libraries, also in spawned workers.
for _name, _value in (
    ("OMP_NUM_THREADS", "1"),
    ("OMP_THREAD_LIMIT", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("BLIS_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("OMP_DYNAMIC", "FALSE"),
    ("MKL_DYNAMIC", "FALSE"),
    ("OMP_WAIT_POLICY", "PASSIVE"),
    ("GOMP_SPINCOUNT", "0"),
    ("JAX_PLATFORMS", "cpu"),
    ("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"),
):
    os.environ.setdefault(_name, _value)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def default_worker_count():
    import psutil

    workers = int(os.environ.get("XTB_WORKERS", psutil.cpu_count(logical=False) or 1))
    if workers < 1:
        raise ValueError("XTB_WORKERS must be positive")
    return workers


def check_runtime():
    """Fail before selecting candidates when the calculation runtime is missing."""
    binary = os.environ.get("XTB_BIN", "xtb")
    if shutil.which(binary) is None:
        raise FileNotFoundError(f"xTB executable not found: {binary}")
    from molecules.xtb_molecular_system import XTBMolecularSystem
    from minimizers.sella_wrapper import minimize_func


def run_candidate(candidate, work_dir):
    """Optimize a source conformer with Sella/GFN2-xTB under 200 force calls.

    candidate supplies numbers, positions (nm), charge, mapped smiles, and
    xyz_file. Baseline energies (Hartree), convergence, and the saved final
    geometry refer to the last evaluation, including when the force budget
    is exhausted. Reject bond mismatches not resolved by relative covalent-radius
    distances, a diameter above twice the input value, or an energy decrease below
    IMPROVEMENT_MIN Hartree.
    """

    run_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=work_dir))
    input_path = run_dir / candidate["xyz_file"]
    stage = "prepare"
    try:
        import numpy as np
        from rdkit import Chem
        from molecules.utils import HARTREE_TO_KJ, MAX_FORCE_CALLS, write_xyz
        from molecules.pipeline_helpers import check_connectivity, check_atom_separation

        numbers = np.asarray(candidate["numbers"], dtype=int)
        positions = np.asarray(candidate["positions"], dtype=float)
        charge = candidate["charge"]
        smiles = candidate["smiles"]
        params = Chem.SmilesParserParams()
        params.removeHs = False
        molecule = Chem.MolFromSmiles(smiles, params)
        if molecule is None:
            raise ValueError("Invalid reference mapped SMILES")
        atoms = sorted(molecule.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())
        if [atom.GetAtomMapNum() for atom in atoms] != list(range(1, len(numbers) + 1)):
            raise ValueError("Reference atom maps do not match XYZ indices")
        if [atom.GetAtomicNum() for atom in atoms] != list(numbers):
            raise ValueError("Reference elements do not match XYZ atom order")
        molecule = Chem.RenumberAtoms(molecule, [atom.GetIdx() for atom in atoms])
        if Chem.GetFormalCharge(molecule) != charge:
            raise ValueError(f"Charge mismatch in {candidate['xyz_file']}")
        spin = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()) + 1
        if spin != 1:
            print(f"WARNING: {input_path.name} has spin multiplicity {spin}", flush=True)
        reference_bonds = Chem.GetAdjacencyMatrix(molecule)

        write_xyz(numbers, positions, str(input_path), charge)

        stage = "xtb"
        from minimizers.sella_wrapper import minimize_func
        from molecules.convergence import init_convergence_state, update_convergence_state
        from molecules.xtb_molecular_system import XTBMolecularSystem

        final_path = run_dir / f"{input_path.stem}_final.xyz"
        system = XTBMolecularSystem(
            str(input_path),
            method="GFN2-xTB",
            num_threads=1,
        )
        initial_pos = system.initial_positions.copy()
        state = init_convergence_state("geometric", initial_pos)
        energies_kj = []
        force_calls = 0
        final_pos = initial_pos.copy()

        def calc(pos):
            nonlocal force_calls, final_pos
            if force_calls >= MAX_FORCE_CALLS:
                raise RuntimeError("Sella exceeded the force-call budget")
            pos_arr = np.asarray(pos, dtype=float)
            if not np.isfinite(pos_arr).all():
                raise ValueError("Sella returned non-finite coordinates")
            energy, forces = system.compute(pos_arr)
            force_calls += 1
            if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
                raise RuntimeError("GFN2-xTB returned non-finite energy or forces")
            energies_kj.append(float(energy))
            final_pos = pos_arr.copy()
            update_convergence_state(state, pos_arr, energy, forces)
            return energy, forces

        _, wrapper_force_calls = minimize_func(
            initial_pos,
            system.atomic_numbers,
            calc,
            max_force_calls=MAX_FORCE_CALLS,
            converged=lambda: state.converged,
        )
        if wrapper_force_calls != force_calls:
            raise RuntimeError(
                f"Sella force-call accounting mismatch: wrapper={wrapper_force_calls}, callback={force_calls}"
            )
        if not energies_kj:
            raise RuntimeError("Sella made zero force calls")
        stage = "minimized conformation connectivity"
        check_connectivity(system.atomic_numbers, final_pos, reference_bonds)
        stage = "minimized conformation atom separation"
        check_atom_separation(initial_pos, final_pos)
        baseline = {
            "n_atoms": len(system.atomic_numbers),
            "initial_energy": energies_kj[0] / HARTREE_TO_KJ,
            "final_energy": energies_kj[-1] / HARTREE_TO_KJ,
            "converged": bool(state.converged),
            "force_calls": force_calls,
        }
        stage = "energy"
        drop = baseline["initial_energy"] - baseline["final_energy"]
        if drop < IMPROVEMENT_MIN:
            raise ValueError(
                f"insufficient energy improvement ({drop:.8g} Ha < {IMPROVEMENT_MIN} Ha)"
            )
        stage = "save"
        write_xyz(system.atomic_numbers, final_pos, str(final_path), system.charge)
        return {
            "status": "ok",
            "baseline": baseline,
            "charge": charge,
            "input_path": str(input_path),
            "final_path": str(final_path),
            "work_dir": str(run_dir),
        }
    except Exception as exc:
        lines = str(exc).splitlines()
        reason = lines[0] if lines else type(exc).__name__
        if stage == "xtb" and reason.startswith("xtb "):
            reason = "error"
        report = {
            "status": "fatal" if isinstance(exc, (OSError, ImportError)) else "error",
            "error": reason,
        }
    shutil.rmtree(run_dir)
    report["stage"] = stage
    return report
