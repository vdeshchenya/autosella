"""Prepare a balanced random SPICE 2.0.1 amino-acid--ligand evaluation set."""

import concurrent.futures as cf
import multiprocessing
import random
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MOLECULES_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(MOLECULES_DIR.parent))

from molecules import pipeline_helpers as helpers
from molecules import gfn_worker

AMINO_ACIDS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU",
    "GLY", "HIS", "ILE", "LEU", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
SYSTEMS_PER_AMINO_ACID = 25
SELECTION_SEED = 42
SPICE_SUBSET = "SPICE Amino Acid Ligand v1.0"
GROUP_PATTERN = (
    r"[0-9A-Z]{3} "
    r"(?:ALA|ARG|ASN|ASP|CYS|GLN|GLU|GLY|HIS|ILE|LEU|MET|PHE|PRO|"
    r"SER|THR|TRP|TYR|VAL)"
)
EXPECTED_SYSTEMS = 79967
EXPECTED_CONFORMATIONS = 194174


def xyz_filename(group):
    """Return a shell-friendly filename for an amino-acid--ligand group."""
    filename = f"{group.replace(' ', '_')}.xyz"
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe SPICE group name: {group!r}")
    return filename


def discover_systems(h5):
    """Find and validate the release-pinned amino-acid--ligand systems."""
    pattern = re.compile(GROUP_PATTERN)
    systems = []
    unexpected_subsets = []
    for group in sorted(name for name in h5 if pattern.fullmatch(name)):
        record = h5[group]
        subset = record["subset"].asstr()[0]
        if subset != SPICE_SUBSET:
            unexpected_subsets.append((group, subset))
            continue
        systems.append((group, int(record["conformations"].shape[0])))

    if unexpected_subsets:
        examples = ", ".join(
            f"{group!r}: {subset!r}" for group, subset in unexpected_subsets[:5]
        )
        raise RuntimeError(
            f"Unexpected SPICE subset labels for amino acid ligand pairs: {examples}"
        )

    system_count = len(systems)
    conformation_count = sum(count for _, count in systems)
    if system_count != EXPECTED_SYSTEMS:
        raise RuntimeError(
            f"Expected {EXPECTED_SYSTEMS} amino acid ligand pairs systems, "
            f"found {system_count}"
        )
    if conformation_count != EXPECTED_CONFORMATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFORMATIONS} amino acid ligand pairs "
            f"conformations, found {conformation_count}"
        )

    filenames = [xyz_filename(group) for group, _ in systems]
    if len(set(filenames)) != len(filenames):
        raise RuntimeError("Filename collision in the amino acid ligand pairs subset")
    return systems


def balanced_pair_pools(
    systems,
    amino_acids=AMINO_ACIDS,
    systems_per_amino_acid=SYSTEMS_PER_AMINO_ACID,
    seed=SELECTION_SEED,
):
    """Return reproducibly shuffled reserve candidates for every residue."""
    by_amino_acid = defaultdict(list)
    for system in systems:
        group, _ = system
        ligand, amino_acid = group.split(" ", 1)
        by_amino_acid[amino_acid].append((ligand, system))

    rng = random.Random(seed)
    pools = {}
    for amino_acid in amino_acids:
        candidates = by_amino_acid[amino_acid][:]
        rng.shuffle(candidates)
        pool = [system for _, system in candidates]
        if len(pool) < systems_per_amino_acid:
            raise RuntimeError(
                f"Could only find {len(pool)}/{systems_per_amino_acid} "
                f"systems for {amino_acid}"
            )
        pools[amino_acid] = pool
    return pools


def allocate_unique_pairs(reserves, needed, used_ligands, amino_acids=AMINO_ACIDS):
    """Allocate a deterministic batch without reusing ligand identities."""
    batch = []
    targets = {amino_acid: needed[amino_acid] for amino_acid in amino_acids}
    remaining = targets.copy()
    while any(remaining.values()):
        for amino_acid in amino_acids:
            if remaining[amino_acid] == 0:
                continue
            while reserves[amino_acid]:
                system = reserves[amino_acid].popleft()
                ligand = system[0].split(" ", 1)[0]
                if ligand in used_ligands:
                    continue
                used_ligands.add(ligand)
                batch.append((amino_acid, system))
                remaining[amino_acid] -= 1
                break
            else:
                allocated = targets[amino_acid] - remaining[amino_acid]
                raise RuntimeError(
                    f"Could only allocate {allocated}/{targets[amino_acid]} "
                    f"additional unique systems for {amino_acid}"
                )
    return batch


def prepare_balanced_pairs(
    pair_pools,
    hdf5_path,
    amino_acids=AMINO_ACIDS,
    systems_per_amino_acid=SYSTEMS_PER_AMINO_ACID,
):
    """Fill residue quotas with unique ligands that pass the main acceptance checks."""
    reserves = {
        amino_acid: deque(pair_pools[amino_acid]) for amino_acid in amino_acids
    }
    needed = {amino_acid: systems_per_amino_acid for amino_acid in amino_acids}
    used_ligands = set()
    selected = []
    workers = gfn_worker.default_worker_count()
    with cf.ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        while any(needed.values()):
            pairs = allocate_unique_pairs(
                reserves, needed, used_ligands, amino_acids=amino_acids
            )
            streams = (
                (group, helpers.spice_candidates(
                    hdf5_path, group, n_conformers, xyz_file=xyz_filename(group)
                ))
                for _, (group, n_conformers) in pairs
            )
            results = helpers.select_candidates(
                streams, BASE_DIR / "data" / "tmp", pool=pool, workers=workers
            )
            for candidate, result in results:
                helpers.publish_candidate(BASE_DIR, candidate, result, save_final=False)
                group = candidate["hdf5_group"]
                needed[group.split(" ", 1)[1]] -= 1
                selected.append({
                    "hdf5_group": group,
                    "conformer_index": candidate["conformer_index"],
                    "xyz_file": candidate["xyz_file"],
                })

    selected.sort(key=lambda item: item["hdf5_group"])
    return selected


def build_amino_acid_ligand_pairs():
    import h5py

    hdf5_path = helpers.ensure_spice_inputs()["spice_hdf5"]
    gfn_worker.check_runtime()
    with h5py.File(hdf5_path, "r") as h5:
        source_systems = discover_systems(h5)
    pair_pools = balanced_pair_pools(source_systems)
    selected = prepare_balanced_pairs(pair_pools, hdf5_path)

    manifest = {
        "subset": SPICE_SUBSET,
        "source_system_count": len(source_systems),
        "source_conformation_count": sum(count for _, count in source_systems),
        "selected_conformer_count": len(selected),
        "selected_conformers": selected,
        "selection": {
            "method": "globally_unique_stratified_random",
            "seed": SELECTION_SEED,
            "systems_per_amino_acid": SYSTEMS_PER_AMINO_ACID,
            "amino_acid_count": len(AMINO_ACIDS),
            "unique_ligands_across_amino_acids": True,
        },
    }
    helpers.write_json(BASE_DIR / "selected_conformers.json", manifest)
    print(
        f"Saved {len(selected)} GFN2-xTB-accepted amino acid ligand pair conformers",
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    build_amino_acid_ligand_pairs()
