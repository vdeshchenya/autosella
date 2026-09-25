"""Validate the frozen prepared release without downloading or recalculating it."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pytest

from utils import MAX_FORCE_CALLS, SYMBOL_TO_Z, parse_xyz, read_xyz_charge

MOLECULES = Path(__file__).resolve().parent.parent / "molecules"
RELEASE_ID = "main-f59f2b89a-gfn2-v1"
RELEASE = MOLECULES / "releases" / RELEASE_ID
SOURCE_COMMIT = "f59f2b89ae6dcdb154a9dc27ce9f825c8c38d1d0"
RELEASE_SHA256 = "43f467330820f1b0da694cbf37047d7c59c8a071c70867b7190359792576ed81"
NONCONVERGED = {"train": set(), "valid": set()}
COUNTS = {
    "train": {"pubchem": 225, "des370k": 219, "rowansci": 25},
    "valid": {"pubchem": 250, "des370k": 215},
}


def load_json(relative):
    return json.loads((RELEASE / relative).read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_release_identity_units_and_settings():
    # This anchor protects release metadata as well as its payload checksum map.
    assert sha256(RELEASE / "release.json") == RELEASE_SHA256
    release = load_json("release.json")
    assert release["schema_version"] == 1
    assert release["dataset_release_id"] == RELEASE_ID
    assert release["source_commit"] == SOURCE_COMMIT
    assert release["units"]["coordinates"] == "angstrom"
    assert release["units"]["reference_energy"] == "hartree"
    assert release["units"]["reference_energy_kind"] == "total_molecular_energy"
    assert release["preparation"]["potential"] is None
    assert release["preparation"]["preoptimization"] is False
    assert release["reference"]["xtb_version_declared"] == "6.7.1"
    assert release["preparation"]["rowansci_noise"] == {"seed": 42, "mean_displacement_angstrom": 0.02}
    assert release["preparation"]["max_conformers_per_system"] == 10
    assert release["preparation"]["pubchem_selection"]["canonical_smiles_deduplication"] is True
    reference = release["reference"]
    assert reference["potential"] == "GFN2-xTB"
    assert reference["energy_kind"] == "total_molecular_energy"
    assert reference["energy_unit"] == "hartree"
    assert reference["accuracy"] == 0.2
    assert reference["max_force_calls"] == MAX_FORCE_CALLS == 200
    assert reference["reference_convergence_required_for_selection"] is False
    assert reference["jax_enable_x64"] is True
    assert reference["convergence"] == {
        "type": "geometric", "implementation_source": "molecules/convergence.py",
    }


def test_every_payload_file_matches_frozen_hash():
    manifest = load_json("release.json")
    checksum_path = RELEASE / manifest["checksums_file"]
    assert sha256(checksum_path) == manifest["checksums_sha256"]
    hashes = json.loads(checksum_path.read_text())
    actual = {
        p.relative_to(RELEASE).as_posix()
        for p in RELEASE.rglob("*")
        if p.is_file() and p.name not in {"release.json", "sha256.json"}
    }
    assert actual == set(hashes)
    assert len(hashes) == 941
    for name, expected in hashes.items():
        path = RELEASE / name
        assert not path.is_symlink(), name
        assert path.resolve().is_relative_to(RELEASE.resolve()), name
        assert sha256(path) == expected, name


def test_every_xyz_has_explicit_charge_and_matching_atoms():
    metadata = load_json("geometry_metadata.json")
    actual = {p.relative_to(RELEASE).as_posix() for p in RELEASE.rglob("*.xyz")}
    assert set(metadata) == actual
    assert len(actual) == 934
    for name, expected in metadata.items():
        path = RELEASE / name
        assert not path.name.endswith("_mm.xyz")
        lines = path.read_text().splitlines()
        assert re.fullmatch(r"charge=[+-]?\d+", lines[1]), name
        assert len(lines) == int(lines[0]) + 2, name
        assert all(len(line.split()) == 4 for line in lines[2:]), name
        numbers, positions = parse_xyz(str(path))
        assert len(numbers) == expected["n_atoms"] == int(lines[0]), name
        assert positions.shape == (len(numbers), 3), name
        assert np.isfinite(positions).all(), name
        assert read_xyz_charge(str(path)) == expected["charge"], name


def _source(key):
    if key.startswith("des370k_"):
        return "des370k"
    if re.fullmatch(r"\d+", key):
        return "pubchem"
    return "rowansci"


@pytest.mark.parametrize("split", ["train", "valid"])
def test_reference_schema_counts_naming_and_retained_nonconvergence(split):
    manifest = load_json("release.json")["datasets"][split]
    references = load_json(manifest["reference_manifest"])
    assert len(references) == manifest["count"] == sum(COUNTS[split].values())
    assert Counter(_source(key) for key in references) == COUNTS[split]
    assert manifest["source_counts"] == COUNTS[split]
    nonconverged = {key for key, value in references.items() if not value["converged"]}
    assert nonconverged == NONCONVERGED[split] == set(manifest["nonconverged_keys"])
    for key, entry in references.items():
        assert not key.endswith("_mm")
        assert set(entry) == {"n_atoms", "initial_energy", "final_energy", "force_calls", "converged"}
        path = RELEASE / manifest["xyz_dir"] / f"{key}.xyz"
        numbers, _ = parse_xyz(str(path))
        assert type(entry["n_atoms"]) is int and entry["n_atoms"] == len(numbers)
        assert type(entry["force_calls"]) is int
        assert 0 < entry["force_calls"] <= MAX_FORCE_CALLS
        assert type(entry["converged"]) is bool
        if key in nonconverged:
            assert entry["force_calls"] == MAX_FORCE_CALLS
        assert math.isfinite(entry["initial_energy"])
        assert math.isfinite(entry["final_energy"])
        assert entry["initial_energy"] - entry["final_energy"] >= 0.005
    # The hash anchor preserves the original total-Hartree values exactly.
    # Do not apply the previous isolated-atom/formation-energy bounds here.


def test_combined_references_and_xyz_are_exact_split_union():
    train = load_json("train_XTB.json")
    valid = load_json("valid_XTB.json")
    assert not set(train) & set(valid)
    assert load_json("data/baseline_XTB.json") == {**train, **valid}
    assert {p.stem for p in (RELEASE / "xyz").glob("*.xyz")} == set(train) | set(valid)


def _check_mapped_atoms_and_charge(record, xyz_dir):
    """Read only explicit mapped atoms; avoid a preparation-toolkit dependency."""
    smiles = record.get("mapped_smiles", record.get("dimer_mapped_smiles"))
    atoms = {}
    charge = 0
    for token in re.findall(r"\[([^]]+)\]", smiles):
        atom = re.fullmatch(r"(\d*)([A-Z][a-z]?|[bcnops])([^:]*):(\d+)", token)
        assert atom is not None, token
        atom_id = int(atom[4])
        assert atom_id not in atoms, token
        atoms[atom_id] = SYMBOL_TO_Z[atom[2].capitalize()]
        formal_charge = re.search(r"([+-]+)(\d*)", atom[3])
        if formal_charge:
            sign = 1 if formal_charge[1][0] == "+" else -1
            charge += sign * (int(formal_charge[2]) if formal_charge[2] else len(formal_charge[1]))
    assert set(atoms) == set(range(1, len(atoms) + 1))
    name = Path(record["xyz_file"]).stem + ".xyz"
    xyz_path = RELEASE / xyz_dir / name
    numbers, _ = parse_xyz(str(xyz_path))
    assert list(numbers) == [atoms[index] for index in sorted(atoms)], name
    assert read_xyz_charge(str(xyz_path)) == charge, name
    if "charge" in record:
        assert record["charge"] == charge
    assert type(record["conformer_index"]) is int and record["conformer_index"] >= 0


def test_main_selection_metadata_membership_and_charges():
    train = load_json("train_XTB.json")
    valid = load_json("valid_XTB.json")
    pubchem = load_json("data/pubchem_selected_conformers.json")
    records = pubchem["selected_conformers"]
    assert pubchem["selected_conformer_count"] == len(records) == 475
    ids = [record["hdf5_group"] for record in records]
    assert len(set(ids)) == 475
    assert len({record["smiles"] for record in records}) == 475
    for split, selected in [(train, records[:225]), (valid, records[225:])]:
        assert {key for key in split if _source(key) == "pubchem"} == {
            record["hdf5_group"] for record in selected
        }
    for record in records:
        assert record["xyz_file"] == record["hdf5_group"] + ".xyz"
        _check_mapped_atoms_and_charge(record, "xyz")
    des = load_json("data/des370k_selected_dimers.json")
    assert des["selected_dimer_count"] == len(des["selected_dimers"]) == 434
    for split, references in [("train", train), ("valid", valid)]:
        selected = [record for record in des["selected_dimers"] if record["split"] == split]
        assert len(selected) == COUNTS[split]["des370k"]
        assert {Path(record["xyz_file"]).stem for record in selected} == {
            key for key in references if key.startswith("des370k_")
        }
    for record in des["selected_dimers"]:
        assert Path(record["xyz_file"]).stem.endswith("__" + record["split"])
        assert 0 <= record["conformer_index"] < record["n_conformers"]
        _check_mapped_atoms_and_charge(record, "xyz")
    rowansci = load_json("data/rowansci_selected_inputs.json")["selected_inputs"]
    assert len(rowansci) == 25
    assert {record["baseline_key"] for record in rowansci} == {
        key for key in train if _source(key) == "rowansci"
    }
    for record in rowansci:
        key = record["baseline_key"]
        assert key == Path(record["xyz_file"]).stem
        assert read_xyz_charge(str(RELEASE / "xyz" / f"{key}.xyz")) == record["charge"]


def test_release_contains_only_train_and_validation_payloads():
    release = load_json("release.json")
    assert release["scope"] == "train_valid_only"
    assert set(release["datasets"]) == {"train", "valid"}
    expected = {
        "train_XTB.json", "valid_XTB.json", "geometry_metadata.json",
        "data/baseline_XTB.json", "data/pubchem_selected_conformers.json",
        "data/des370k_selected_dimers.json", "data/rowansci_selected_inputs.json",
    }
    expected |= {"xyz/" + key + ".xyz" for key in load_json("data/baseline_XTB.json")}
    assert set(load_json("sha256.json")) == expected
    assert set(load_json("geometry_metadata.json")) == {name for name in expected if name.endswith(".xyz")}
    assert not (RELEASE / "evaluation").exists()
    assert all("evaluation/" not in name for name in release["source_settings_sha256"])


@pytest.mark.parametrize("release_id,source_commit,release_hash,inventory_hash", [
    (
        "main-d288fbec4-gfn2-v1", "d288fbec457a4d38b01d0b101058bb3b80f1b5b0",
        "e0a28a199ce865e288e6e7579bc913f1349cfd74851d6d0ddf82309dab6741a2",
        "bbb089a898b3013f74edff2e66e35a61d828ddf1de58f889be7d1125cb1c3c16",
    ),
    (
        "main-f475a971a-gfn2-v1", "f475a971a906f945b8e987d97cf7902193ba58a8",
        "dfd9464979c33c11ce34f32ad29a640de1169119fcc82ba51386ee655c96e79f",
        "2fcceab290d3718d886181a9e206b606ccbb893fb8cdf0e10f9de229a0a81019",
    ),
])
def test_historical_release_remains_byte_for_byte_unchanged(release_id, source_commit, release_hash, inventory_hash):
    # Preserve the release used by earlier recorded evidence without making it
    # the active dataset or copying its contents into the current release.
    from scripts.install_dataset import verify

    old = MOLECULES / "releases" / release_id
    if not old.is_dir():
        pytest.skip("Historical operator release is not included in this branch")
    assert sha256(old / "release.json") == release_hash
    manifest = verify(old)
    assert manifest["source_commit"] == source_commit
    inventory = "\n".join(
        f"{path.relative_to(old)} {sha256(path)}"
        for path in sorted(old.rglob("*")) if path.is_file()
    )
    assert hashlib.sha256(inventory.encode()).hexdigest() == inventory_hash


def test_molecules_contains_prepared_release_without_pipeline_or_mm_assets():
    files = [p for p in MOLECULES.rglob("*") if p.is_file()]
    assert not any(p.suffix in {".py", ".sh", ".xml", ".npy", ".log"} for p in files)
    assert not any(p.name in {"log.txt", "forcefield.json"} for p in files)
    assert not any("xyz_init" in p.parts or "openmm" in p.parts for p in files)
