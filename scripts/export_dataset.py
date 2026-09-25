"""Export an audited train/validation release from Git without running preparation.

The source revision pins the settings described below. Supporting a new source
revision requires reviewing those declarations and adding an audited profile.
Only the explicit reference/selection allowlist and referenced XYZ files are
read into the payload; no directory is copied from the source checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

SOURCE_COMMIT = "f59f2b89ae6dcdb154a9dc27ce9f825c8c38d1d0"
RELEASE_ID = "main-f59f2b89a-gfn2-v1"
SOURCE_PROFILES = {
    SOURCE_COMMIT: {
        "release_id": RELEASE_ID, "max_conformers": 10,
        "opaque_convergence": True, "raw_inputs": True,
        "pubchem_selection": {
            "canonical_smiles_deduplication": True,
            "fingerprint": "hydrogen-normalized 1024-bit ECFP4",
            "ranking": "Greedy farthest-first against accepted molecules only; start with the smallest group ID that passes preparation",
            "conformer_order": "Deterministic hash order; sequential attempts retain the first passing conformer",
            "target_accepted_groups": 475,
        },
    },
}
PAYLOAD_MANIFESTS = (
    "train_XTB.json", "valid_XTB.json", "data/baseline_XTB.json",
    "data/pubchem_selected_conformers.json", "data/des370k_selected_dimers.json",
    "data/rowansci_selected_inputs.json",
)
SOURCE_SETTINGS = (
    "molecules/gfn_worker.py", "molecules/pipeline_helpers.py",
    "molecules/pubchem.py", "molecules/des370k.py", "molecules/rowansci.py",
    "molecules/train_valid_split.py", "molecules/convergence.py",
    "molecules/utils.py", "molecules/xtb_molecular_system.py",
    "minimizers/sella_wrapper.py", "environment.yml",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def read_source_files(repo: Path, commit: str, names: set[str]) -> dict[str, bytes]:
    # Read Git blobs directly: archive attributes must not omit or substitute
    # bytes, and working-tree changes must not affect the release.
    tree = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-rz", "--full-tree", commit, "--", *sorted(names)]
    )
    objects = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        details, name = entry.decode().split("\t", 1)
        mode, kind, object_id = details.split()
        if name not in names or kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"Unexpected source file: {name}")
        objects[name] = object_id
    if set(objects) != names:
        raise ValueError("Source revision has missing payload files")
    contents = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input="".join(object_id + "\n" for object_id in objects.values()).encode(),
    )
    stream, result = io.BytesIO(contents), {}
    for name, expected_id in objects.items():
        object_id, kind, size = stream.readline().decode().split()
        if object_id != expected_id or kind != "blob":
            raise ValueError(f"Unexpected source object: {name}")
        result[name] = stream.read(int(size))
        if len(result[name]) != int(size) or stream.read(1) != b"\n":
            raise ValueError(f"Truncated source object: {name}")
    if stream.read():
        raise ValueError("Unexpected extra source objects")
    return result


def source_family(key: str) -> str:
    if key.startswith("des370k_"):
        return "des370k"
    return "pubchem" if re.fullmatch(r"\d+(?:_mm)?", key) else "rowansci"


def build_release(repo: Path, commit: str = SOURCE_COMMIT) -> dict[str, bytes]:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}"],
        text=True,
    ).strip()
    if commit not in SOURCE_PROFILES:
        raise ValueError("Source revision has no audited settings profile")
    profile = SOURCE_PROFILES[commit]
    source = read_source_files(repo, commit, {"molecules/" + name for name in PAYLOAD_MANIFESTS})
    payload = {name.removeprefix("molecules/"): data for name, data in source.items()}
    references = {split: json.loads(payload[f"{split}_XTB.json"]) for split in ("train", "valid")}
    train, valid = references["train"], references["valid"]
    if set(train) & set(valid):
        raise ValueError("Train and validation references overlap")
    if json.loads(payload["data/baseline_XTB.json"]) != {**train, **valid}:
        raise ValueError("Combined baseline differs from the train/validation union")
    keys = set(train) | set(valid)
    key_pattern = r"[A-Za-z0-9_-]+" if profile.get("raw_inputs") else r"[A-Za-z0-9_-]+_mm"
    if any(not re.fullmatch(key_pattern, key) for key in keys):
        raise ValueError("Invalid molecule key")
    geometries = read_source_files(repo, commit, {f"molecules/xyz/{key}.xyz" for key in keys})
    payload.update({name.removeprefix("molecules/"): data for name, data in geometries.items()})
    geometry_metadata = {}
    for name, data in sorted(payload.items()):
        if not name.endswith(".xyz"):
            continue
        lines = data.decode().splitlines()
        charge = re.fullmatch(r"charge=([+-]?\d+)", lines[1])
        n_atoms = int(lines[0])
        if not charge or n_atoms < 1 or len(lines) != n_atoms + 2:
            raise ValueError(f"Invalid prepared geometry: {name}")
        geometry_metadata[name] = {"charge": int(charge[1]), "n_atoms": n_atoms}
    payload["geometry_metadata.json"] = json_bytes(geometry_metadata)
    hashes = {name: sha256(data) for name, data in sorted(payload.items())}
    payload["sha256.json"] = json_bytes(hashes)
    settings = read_source_files(repo, commit, set(SOURCE_SETTINGS))
    manifest = {
        "schema_version": 1,
        "dataset_release_id": profile["release_id"],
        "source_commit": commit,
        "source_repository": "opt_problem",
        "scope": "train_valid_only",
        "payload_policy": "Prepared GFN-FF geometries and frozen train/validation selection/reference data exported byte-for-byte from the source commit. geometry_metadata.json and release/checksum manifests are packaging metadata.",
        "units": {
            "coordinates": "angstrom", "charge": "elementary_charge",
            "reference_energy": "hartree", "reference_energy_kind": "total_molecular_energy",
            "runtime_energy": "kJ/mol", "runtime_force": "kJ/(mol nm)",
        },
        "preparation": {
            "potential": "GFN-FF", "optimizer": "Sella", "optimizer_version_declared": "2.5.0",
            "internal_coordinates": True, "order": 0, "max_force_calls": 200,
            "selection_seed": 42, "max_conformers_per_system": profile["max_conformers"],
            "filename_suffix": "_mm means minimized; selected conformer indices are metadata",
        },
        "reference": {
            "potential": "GFN2-xTB", "xtb_version_declared": "6.6.1", "accuracy": 0.2,
            "num_threads": 1, "optimizer": "Sella", "optimizer_version_declared": "2.5.0",
            "internal_coordinates": True, "order": 0, "max_force_calls": 200,
            "jax_enable_x64": True, "energy_kind": "total_molecular_energy", "energy_unit": "hartree",
            "initial_energy_geometry": "The supplied GFN-FF-preoptimized XYZ",
            "final_energy_geometry": "Last evaluated GFN2-xTB optimization geometry, not exported",
            "acceptance_min_energy_decrease_hartree": 0.005,
            "reference_convergence_required_for_selection": False,
            "convergence": {
                "type": "geometric", "energy_change_hartree": 1e-6,
                "max_force_hartree_per_bohr": 0.00045, "rms_force_hartree_per_bohr": 0.0003,
                "max_aligned_displacement_angstrom": 0.0018,
                "rms_aligned_displacement_angstrom": 0.0012,
            },
        },
        "datasets": {
            split: {
                "count": len(entries), "reference_manifest": f"{split}_XTB.json", "xyz_dir": "xyz",
                "source_counts": dict(Counter(source_family(key) for key in entries)),
                "nonconverged_keys": sorted(key for key, value in entries.items() if not value["converged"]),
            }
            for split, entries in references.items()
        },
        "selection_manifests": {
            "pubchem": "data/pubchem_selected_conformers.json",
            "des370k": "data/des370k_selected_dimers.json",
            "rowansci": "data/rowansci_selected_inputs.json",
        },
        "geometry_metadata": "geometry_metadata.json", "checksums_file": "sha256.json",
        "checksums_sha256": sha256(payload["sha256.json"]),
        "source_settings_sha256": {name: sha256(data) for name, data in sorted(settings.items())},
        "provenance_limitations": [
            "Settings describe the source code and declared package versions at source_commit; the source artifacts have no execution-environment or executable-hash record.",
            "No baseline was recalculated for this export. Exact numerical reproduction on a newly resolved runtime environment has not been validated.",
        ],
    }
    if profile.get("raw_inputs"):
        manifest["payload_policy"] = "Source optimization inputs and frozen train/validation selection/reference data exported byte-for-byte from the source commit. RowanSci inputs include the declared noise. geometry_metadata.json and release/checksum manifests are packaging metadata."
        manifest["preparation"] = {
            "potential": None, "preoptimization": False,
            "input_geometry": "SPICE source conformers; RowanSci source coordinates with reproducible centered noise",
            "selection_seed": 42, "max_conformers_per_system": 10,
            "filename_suffix": "No method suffix; filename stem is the baseline key",
            "rowansci_noise": {"seed": 42, "mean_displacement_angstrom": 0.02},
        }
        manifest["reference"].update({
            "xtb_version_declared": "6.7.1",
            "initial_energy_geometry": "The supplied source/noisy XYZ; no GFN-FF preoptimization",
            "final_geometry_acceptance": {
                "maximum_diameter_ratio": 2.0,
                "connectivity": "Final RDKit connectivity versus mapped source SMILES; missing bonds allowed only for g < 1, extra bonds only for g > 1, with g = distance / sum of RDKit covalent radii",
                "checks_during_trajectory": False,
            },
        })
    if "pubchem_selection" in profile:
        manifest["preparation"]["pubchem_selection"] = profile["pubchem_selection"]
    if profile.get("opaque_convergence"):
        manifest["reference"]["convergence"] = {
            "type": "geometric", "implementation_source": "molecules/convergence.py",
        }
    payload["release.json"] = json_bytes(manifest)
    return payload


def export(repo: Path, destination: Path, commit: str = SOURCE_COMMIT) -> Path:
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Destination exists; immutable releases cannot be overwritten")
    payload = build_release(repo, commit)
    if destination.name != json.loads(payload["release.json"])["dataset_release_id"]:
        raise ValueError("Destination must end in the dataset release ID")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".dataset-export-", dir=destination.parent))
    created = False
    try:
        for name, data in payload.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        # mkdir reserves the release identity exclusively; never replace an
        # existing release, even if another exporter raced this one.
        destination.mkdir()
        created = True
        for path in staging.iterdir():
            os.rename(path, destination / path.name)
    except BaseException:
        if created:
            shutil.rmtree(destination)
        raise
    finally:
        shutil.rmtree(staging)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    args = parser.parse_args()
    result = export(args.repo, args.destination, args.source_commit)
    print(json.dumps({"path": str(result), "release_sha256": sha256((result / "release.json").read_bytes())}))
