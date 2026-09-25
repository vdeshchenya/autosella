"""Select one accepted Sella/GFN2-xTB conformer per fixed SPICE dipeptide."""

import concurrent.futures as cf
import multiprocessing
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MOLECULES_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(MOLECULES_DIR.parent))
from molecules import pipeline_helpers as helpers
from molecules import gfn_worker

N_DIPEPTIDES = 676


def build_dipeptides():
    import h5py

    inputs = helpers.ensure_spice_inputs()
    hdf5_path = inputs["spice_hdf5"]
    output_dir = BASE_DIR
    gfn_worker.check_runtime()
    with h5py.File(hdf5_path, "r") as h5:
        systems = {
            group: int(h5[group]["conformations"].shape[0])
            for group in sorted(h5)
            if re.fullmatch(r"[a-z]+-[a-z]+", group)
        }
    if len(systems) != N_DIPEPTIDES:
        raise RuntimeError(
            f"Expected {N_DIPEPTIDES} SPICE dipeptides, found {len(systems)}"
        )
    streams = (
        (group, helpers.spice_candidates(hdf5_path, group, n_conformers))
        for group, n_conformers in systems.items()
    )
    selection = {}
    workers = gfn_worker.default_worker_count()
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        results = helpers.select_candidates(
            streams, output_dir / "data" / "tmp", pool=pool, workers=workers
        )
    for candidate, result in results:
        helpers.publish_candidate(output_dir, candidate, result, save_final=False)
        selection[candidate["hdf5_group"]] = candidate["conformer_index"]
    helpers.write_json(output_dir / "selected_conformers.json", selection)
    missing = sorted(systems.keys() - selection.keys())
    print(f"Parameterized {len(selection)}/{len(systems)} dipeptides", flush=True)
    print(f"Not parameterized: {', '.join(missing) if missing else 'none'}", flush=True)
    return selection


if __name__ == "__main__":
    build_dipeptides()
