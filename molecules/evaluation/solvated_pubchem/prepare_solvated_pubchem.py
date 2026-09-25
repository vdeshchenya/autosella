"""Prepare a diverse 200-system SPICE 2.0.1 solvated PubChem evaluation set."""

import re
import sys
import time
from pathlib import Path

from rdkit import Chem, DataStructs

BASE_DIR = Path(__file__).resolve().parent
MOLECULES_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(MOLECULES_DIR.parent))

from molecules import gfn_worker
from molecules import pipeline_helpers as helpers
from molecules.evaluation.test.prepare_ood_pubchem import (
    fingerprint,
    load_reference_smiles,
    molecule_from_smiles,
)

N_SELECTED = 100
N_WATERS = 20
SPICE_SUBSET = "SPICE Solvated PubChem Set 1 v1.0"
GROUP_PATTERN = r"solvated [0-9]+"
EXPECTED_SYSTEMS = 1397
EXPECTED_CONFORMATIONS = 13934


def xyz_filename(group):
    """Return a shell-friendly filename for a solvated SPICE group."""
    filename = f"{group.replace(' ', '_')}.xyz"
    if Path(filename).name != filename:
        raise ValueError(f"Unsafe SPICE group name: {group!r}")
    return filename


def discover_systems(h5):
    """Find and validate the release-pinned solvated PubChem systems."""
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
            f"Unexpected SPICE subset labels for solvated PubChem: {examples}"
        )

    system_count = len(systems)
    conformation_count = sum(count for _, count in systems)
    if system_count != EXPECTED_SYSTEMS:
        raise RuntimeError(
            f"Expected {EXPECTED_SYSTEMS} solvated PubChem systems, "
            f"found {system_count}"
        )
    if conformation_count != EXPECTED_CONFORMATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_CONFORMATIONS} solvated PubChem conformations, "
            f"found {conformation_count}"
        )

    filenames = [xyz_filename(group) for group, _ in systems]
    if len(set(filenames)) != len(filenames):
        raise RuntimeError("Filename collision in the solvated PubChem subset")
    return systems


def _is_water_fragment(fragment):
    if fragment.GetNumAtoms() != 1:
        return False
    atom = fragment.GetAtomWithIdx(0)
    return (
        atom.GetAtomicNum() == 8
        and atom.GetFormalCharge() == 0
        and atom.GetTotalNumHs() == 2
    )


def solute_smiles(mapped_system_smiles):
    """Return canonical unmapped SMILES for the PubChem fragment only."""
    system = molecule_from_smiles(mapped_system_smiles)
    fragments = Chem.GetMolFrags(system, asMols=True, sanitizeFrags=True)
    waters = [fragment for fragment in fragments if _is_water_fragment(fragment)]
    solutes = [fragment for fragment in fragments if not _is_water_fragment(fragment)]
    if len(waters) != N_WATERS or len(solutes) != 1:
        raise ValueError(
            f"Expected one solute and {N_WATERS} waters, found "
            f"{len(solutes)} solutes and {len(waters)} waters"
        )
    return helpers.canonical_unmapped_smiles(Chem.MolToSmiles(solutes[0]))


def load_solvated_queries(hdf5_path, systems, reference_smiles):
    import h5py

    queries = []
    seen_smiles = set(reference_smiles)
    skipped_duplicates = 0
    invalid = []
    with h5py.File(hdf5_path, "r") as h5:
        for group, n_conformers in sorted(
            systems,
            key=lambda item: int(item[0].removeprefix("solvated ")),
        ):
            mapped_smiles = h5[group]["smiles"].asstr()[0]
            try:
                smiles = solute_smiles(mapped_smiles)
            except ValueError:
                invalid.append(group)
                continue
            if smiles in seen_smiles:
                skipped_duplicates += 1
                continue
            seen_smiles.add(smiles)
            queries.append(
                {
                    "group": group,
                    "smiles": smiles,
                    "mapped_smiles": mapped_smiles,
                    "n_conformers": n_conformers,
                    "fingerprint": fingerprint(molecule_from_smiles(smiles)),
                }
            )
    if invalid:
        print(f"Skipped invalid solvated PubChem groups: {invalid}", flush=True)
    print(
        f"Eligible solvated PubChem systems: {len(queries)} "
        f"({skipped_duplicates} duplicate reference/query SMILES skipped)",
        flush=True,
    )
    return queries


def prepare_query(query, hdf5_path, work_dir):
    return helpers.select_first_candidate(
        helpers.spice_candidates(
            hdf5_path,
            query["group"],
            query["n_conformers"],
            xyz_file=xyz_filename(query["group"]),
        ),
        work_dir,
    )


def build_solvated_pubchem():
    from molecules import utils
    utils.MAX_FORCE_CALLS = 500

    import h5py

    started = time.time()
    hdf5_path = helpers.ensure_spice_inputs()["spice_hdf5"]
    gfn_worker.check_runtime()
    with h5py.File(hdf5_path, "r") as h5:
        systems = discover_systems(h5)

    reference_smiles = load_reference_smiles()
    reference_fps = [
        fingerprint(molecule_from_smiles(smiles))
        for smiles in sorted(reference_smiles)
    ]
    queries = load_solvated_queries(hdf5_path, systems, reference_smiles)
    maximum_similarity = [
        max(DataStructs.BulkTanimotoSimilarity(query["fingerprint"], reference_fps))
        for query in queries
    ]

    work_dir = BASE_DIR / "data" / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    while queries and len(selected) < N_SELECTED:
        index = min(
            range(len(queries)),
            key=lambda i: (
                maximum_similarity[i],
                int(queries[i]["group"].removeprefix("solvated ")),
            ),
        )
        query = queries.pop(index)
        score = maximum_similarity.pop(index)
        prepared = prepare_query(query, hdf5_path, work_dir)
        if prepared is None:
            continue

        candidate, result = prepared
        helpers.publish_candidate(BASE_DIR, candidate, result, save_final=False)
        selected.append(
            {
                "hdf5_group": candidate["hdf5_group"],
                "conformer_index": candidate["conformer_index"],
                "xyz_file": candidate["xyz_file"],
                "mapped_smiles": candidate["smiles"],
                "smiles": query["smiles"],
            }
        )

        similarities = DataStructs.BulkTanimotoSimilarity(
            query["fingerprint"], [item["fingerprint"] for item in queries]
        )
        maximum_similarity = [
            max(previous, similarity)
            for previous, similarity in zip(maximum_similarity, similarities)
        ]
        print(
            f"Accepted {query['group']} ({len(selected)}/{N_SELECTED}), "
            f"similarity {score:.4f}",
            flush=True,
        )

    if len(selected) != N_SELECTED:
        raise RuntimeError(f"Selection exhausted: accepted {len(selected)}/{N_SELECTED}")
    if len({item["smiles"] for item in selected}) != len(selected):
        raise RuntimeError("Solvated PubChem selection contains duplicate solutes")

    manifest = {
        "subset": SPICE_SUBSET,
        "source_system_count": len(systems),
        "source_conformation_count": sum(count for _, count in systems),
        "selected_conformer_count": len(selected),
        "selected_conformers": selected,
        "selection": {
            "method": "reference_aware_ecfp4_farthest_first",
            "target": N_SELECTED,
            "fingerprinted_fragment": "pubchem_solute",
        },
    }
    helpers.write_json(BASE_DIR / "selected_conformers.json", manifest)
    print(f"Completed in {time.time() - started:.0f}s", flush=True)
    return manifest


if __name__ == "__main__":
    build_solvated_pubchem()
