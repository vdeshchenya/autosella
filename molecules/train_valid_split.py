"""Write the train/validation split from the accepted dataset manifests."""

import json
import sys
from pathlib import Path

MOLECULES_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOLECULES_DIR.parent))
from molecules import pipeline_helpers as helpers


def _load_selection(data_dir, filename, field):
    return json.loads((data_dir / filename).read_text())[field]


def _baseline_key(item):
    return Path(item["xyz_file"]).stem


def build_split():
    """Partition accepted baselines according to their selection manifests."""
    data_dir = MOLECULES_DIR / "data"
    baselines = json.loads((data_dir / "baseline_XTB.json").read_text())
    pubchem_items = _load_selection(
        data_dir, "pubchem_selected_conformers.json", "selected_conformers"
    )
    des_items = _load_selection(
        data_dir, "des370k_selected_dimers.json", "selected_dimers"
    )
    rowansci_items = _load_selection(
        data_dir, "rowansci_selected_inputs.json", "selected_inputs"
    )

    pubchem_keys = [_baseline_key(item) for item in pubchem_items]
    des_keys = [_baseline_key(item) for item in des_items]
    rowansci_keys = [item["baseline_key"] for item in rowansci_items]

    required_pubchem = helpers.N_TRAIN_PUBCHEM + helpers.N_VALID_PUBCHEM
    if len(pubchem_keys) != required_pubchem:
        raise RuntimeError(
            f"Need exactly {required_pubchem} PubChem molecules, "
            f"found {len(pubchem_keys)}"
        )

    manifest_keys = pubchem_keys + des_keys + rowansci_keys
    if len(manifest_keys) != len(set(manifest_keys)):
        raise RuntimeError("Duplicate baseline keys in selection manifests")
    if set(manifest_keys) != set(baselines):
        raise RuntimeError(
            "Baseline and selection manifests differ: "
            f"missing baselines {sorted(set(manifest_keys) - set(baselines))}; "
            f"unknown baselines {sorted(set(baselines) - set(manifest_keys))}"
        )
    if any(item["split"] not in {"train", "valid"} for item in des_items):
        raise RuntimeError("DES370K entries must use the train or valid split")

    des_keys_by_split = {
        split: [
            _baseline_key(item)
            for item in des_items
            if item["split"] == split
        ]
        for split in ("train", "valid")
    }
    train_keys = (
        pubchem_keys[: helpers.N_TRAIN_PUBCHEM]
        + des_keys_by_split["train"]
        + rowansci_keys
    )
    valid_keys = (
        pubchem_keys[helpers.N_TRAIN_PUBCHEM :]
        + des_keys_by_split["valid"]
    )
    if set(train_keys) & set(valid_keys):
        raise RuntimeError("Train and validation splits overlap")

    train = {key: baselines[key] for key in train_keys}
    valid = {key: baselines[key] for key in valid_keys}
    helpers.write_json(MOLECULES_DIR / "train_XTB.json", train)
    helpers.write_json(MOLECULES_DIR / "valid_XTB.json", valid)

    print(f"train_XTB.json: {len(train)} molecules")
    print(f"valid_XTB.json: {len(valid)} molecules")

    return train, valid


if __name__ == "__main__":
    build_split()
