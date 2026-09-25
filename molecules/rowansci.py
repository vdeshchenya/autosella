"""Add reproducible coordinate noise to the fixed RowanSci inputs, then optimize."""

import concurrent.futures as cf
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from molecules import gfn_worker
from molecules import pipeline_helpers as helpers
from molecules.utils import ANGSTROM_TO_NM, parse_xyz

NOISE_MEAN_ANGSTROM = 0.02
NOISE_SEED = 42


def perturb_positions(positions):
    """Add centered random noise with a fixed mean displacement per atom."""
    import numpy as np

    direction = np.random.default_rng(NOISE_SEED).normal(size=positions.shape)
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction, axis=1).mean()
    return positions + NOISE_MEAN_ANGSTROM * ANGSTROM_TO_NM * direction


def build_rowansci():
    from rdkit import Chem
    from xyz2mol import xyz2mol

    inputs = helpers.ensure_rowansci_inputs()
    output_dir = helpers.MOLECULES_DIR
    gfn_worker.check_runtime()
    streams = []
    for path in inputs:
        numbers, positions = parse_xyz(str(path))
        charge = 0
        molecule = xyz2mol(numbers.tolist(), positions / ANGSTROM_TO_NM, charge=charge)[0]
        for index, atom in enumerate(molecule.GetAtoms(), 1):
            atom.SetAtomMapNum(index)
        noisy_positions = perturb_positions(positions)
        print(
            f"Perturbed {path.name}: mean displacement {NOISE_MEAN_ANGSTROM:.4f} A",
            flush=True,
        )
        candidate = {
            "numbers": numbers,
            "positions": noisy_positions,
            "charge": charge,
            "smiles": Chem.MolToSmiles(molecule),
            "xyz_file": path.name,
        }
        streams.append((path.stem, [candidate]))
    baselines = {}
    selected = []
    workers = gfn_worker.default_worker_count()
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = helpers.select_candidates(
            streams, output_dir / "data" / "tmp", pool=pool, workers=workers
        )
    for candidate, result in results:
        key = helpers.publish_candidate(output_dir, candidate, result)
        baselines[key] = result["baseline"]
        selected.append(
            {
                "xyz_file": candidate["xyz_file"],
                "baseline_key": key,
                "charge": result["charge"],
            }
        )
    helpers.write_json(
        output_dir / "data" / "rowansci_selected_inputs.json",
        {
            "source": {
                key: value for key, value in helpers.ROWANSCI_SOURCE.items() if key != "files"
            },
            "selected_inputs": sorted(selected, key=lambda item: item["xyz_file"]),
            "noise": {
                "seed": NOISE_SEED,
                "mean_displacement_angstrom": NOISE_MEAN_ANGSTROM,
            },
        },
    )
    helpers.merge_baselines(output_dir, baselines)
    return baselines


if __name__ == "__main__":
    build_rowansci()
