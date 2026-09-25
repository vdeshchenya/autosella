"""Select and optimize exactly 475 diverse SPICE PubChem molecules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from molecules import gfn_worker
from molecules import pipeline_helpers as helpers

FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 1024


def load_pubchem_records(hdf5_path, *, excluded_smiles=()):
    """Load unique numeric SPICE groups in deterministic ID order."""
    import h5py
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FINGERPRINT_RADIUS, fpSize=FINGERPRINT_BITS
    )
    records = []
    seen_smiles = set(excluded_smiles)
    with h5py.File(hdf5_path, "r") as h5:
        for group in sorted((key for key in h5 if key.isdigit()), key=int):
            record = h5[group]
            mapped_smiles = record["smiles"].asstr()[0]
            try:
                smiles = helpers.canonical_unmapped_smiles(mapped_smiles)
            except ValueError:
                print(f"Skipping PubChem group {group}: invalid SMILES", flush=True)
                continue
            if smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            molecule = Chem.MolFromSmiles(smiles)
            records.append(
                {
                    "group": group,
                    "n_conformers": int(record["conformations"].shape[0]),
                    "mapped_smiles": mapped_smiles,
                    "smiles": smiles,
                    "fingerprint": fingerprint_generator.GetFingerprint(
                        Chem.AddHs(Chem.RemoveHs(molecule))
                    ),
                }
            )
    return records


def select_pubchem(hdf5_path, output_dir, records, target, reference_fps=(), save_final=True):
    """Select accepted molecules farthest from references and previous successes."""
    from rdkit import DataStructs

    work_dir = Path(output_dir) / "data" / "tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    records = list(records)
    maximum_similarity = None
    if reference_fps:
        maximum_similarity = [
            max(DataStructs.BulkTanimotoSimilarity(record["fingerprint"], reference_fps))
            for record in records
        ]
    selected = []
    baselines = {}

    print(f"Selecting from {len(records)} SPICE PubChem molecules", flush=True)
    while records and len(selected) < target:
        index = 0
        if maximum_similarity is not None:
            index = min(
                range(len(records)),
                key=lambda i: (maximum_similarity[i], int(records[i]["group"])),
            )
        record = records.pop(index)
        if maximum_similarity is not None:
            maximum_similarity.pop(index)

        prepared = helpers.select_first_candidate(
            helpers.spice_candidates(hdf5_path, record["group"], record["n_conformers"]),
            work_dir,
        )
        if prepared is None:
            continue
        candidate, result = prepared

        key = helpers.publish_candidate(output_dir, candidate, result, save_final=save_final)
        baselines[key] = result["baseline"]
        selected.append({
            "hdf5_group": candidate["hdf5_group"],
            "conformer_index": candidate["conformer_index"],
            "xyz_file": candidate["xyz_file"],
            "mapped_smiles": candidate["smiles"],
            "smiles": record["smiles"],
        })

        similarities = DataStructs.BulkTanimotoSimilarity(
            record["fingerprint"], [item["fingerprint"] for item in records]
        )
        if maximum_similarity is None:
            maximum_similarity = list(similarities)
        else:
            maximum_similarity = [
                max(previous, similarity)
                for previous, similarity in zip(maximum_similarity, similarities)
            ]
        print(
            f"Accepted {record['group']} ({len(selected)}/{target})",
            flush=True,
        )

    if len(selected) != target:
        raise RuntimeError(f"Selection exhausted: accepted {len(selected)}/{target}")
    if len({item["smiles"] for item in selected}) != len(selected):
        raise RuntimeError("PubChem selection contains duplicate canonical SMILES")

    return selected, baselines


def build_pubchem():
    hdf5_path = helpers.ensure_spice_inputs()["spice_hdf5"]
    gfn_worker.check_runtime()
    output_dir = helpers.MOLECULES_DIR
    target = helpers.N_TRAIN_PUBCHEM + helpers.N_VALID_PUBCHEM
    records = load_pubchem_records(hdf5_path)
    selected, baselines = select_pubchem(hdf5_path, output_dir, records, target)

    helpers.write_json(
        output_dir / "data" / "pubchem_selected_conformers.json",
        {
            "pubchem_system_count": len(selected),
            "selected_conformer_count": len(selected),
            "selected_conformers": selected,
        },
    )
    helpers.merge_baselines(output_dir, baselines)
    return selected


if __name__ == "__main__":
    build_pubchem()
