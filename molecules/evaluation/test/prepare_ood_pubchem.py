"""Build a diverse 500-molecule OOD PubChem test set."""

import json
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from xyz2mol import read_xyz_file, xyz2mol

BASE_DIR = Path(__file__).resolve().parent
MOLECULES_DIR = BASE_DIR.parents[1]
sys.path.insert(0, str(MOLECULES_DIR.parent))
from molecules import gfn_worker, pubchem
from molecules import pipeline_helpers as helpers
from molecules.utils import read_xyz_charge

N_SELECTED = 500
FINGERPRINT = AllChem.GetMorganGenerator(radius=2, fpSize=1024)


def fingerprint(molecule):
    molecule = Chem.AddHs(Chem.RemoveHs(molecule))
    return FINGERPRINT.GetFingerprint(molecule)


def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return molecule


def molecule_from_xyz(path):
    declared_charge = read_xyz_charge(str(path))
    numbers, charge, coordinates = read_xyz_file(str(path))
    if charge != declared_charge:
        raise ValueError(f"XYZ charge parser mismatch for {path}")
    molecule = xyz2mol(numbers, coordinates, charge=charge)[0]
    Chem.SanitizeMol(molecule)
    return molecule


def load_reference_smiles():
    pubchem = json.loads(
        (MOLECULES_DIR / "data" / "pubchem_selected_conformers.json").read_text()
    )
    pubchem_smiles = {
        item["hdf5_group"]: item["smiles"]
        for item in pubchem["selected_conformers"]
    }
    des = json.loads(
        (MOLECULES_DIR / "data" / "des370k_selected_dimers.json").read_text()
    )
    des_keys = {
        Path(item["xyz_file"]).stem for item in des["selected_dimers"]
    }
    reference = {
        monomer["smiles"]
        for item in des["selected_dimers"]
        for monomer in item["monomers"]
    }

    for split in ("train_XTB.json", "valid_XTB.json"):
        for key in json.loads((MOLECULES_DIR / split).read_text()):
            seed = key
            if seed in pubchem_smiles:
                reference.add(pubchem_smiles[seed])
            elif key not in des_keys:
                molecule = molecule_from_xyz(MOLECULES_DIR / "xyz" / f"{seed}.xyz")
                reference.add(helpers.canonical_unmapped_smiles(Chem.MolToSmiles(molecule)))
    return reference


def build_pubchem():
    started = time.time()
    hdf5_path = helpers.ensure_spice_inputs()["spice_hdf5"]
    gfn_worker.check_runtime()
    reference_smiles = load_reference_smiles()
    reference_fps = [
        fingerprint(molecule_from_smiles(smiles))
        for smiles in sorted(reference_smiles)
    ]
    records = pubchem.load_pubchem_records(
        hdf5_path, excluded_smiles=reference_smiles
    )
    selected, _ = pubchem.select_pubchem(
        hdf5_path, BASE_DIR, records, N_SELECTED,
        reference_fps=reference_fps, save_final=False,
    )

    helpers.write_json(
        BASE_DIR / "pubchem_selected_conformers.json",
        {
            "pubchem_system_count": len(selected),
            "selected_conformer_count": len(selected),
            "selected_conformers": selected,
        },
    )
    print(f"Completed in {time.time() - started:.0f}s", flush=True)
    return selected


if __name__ == "__main__":
    build_pubchem()
