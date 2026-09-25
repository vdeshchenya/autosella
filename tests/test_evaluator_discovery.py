"""Tests for validate.discover_xtb_molecules and Evaluator construction.

The evaluator is xTB-only and split-driven: it loads `{split}_XTB.json` from the
molecules dir, where the Sella baseline manifests record the per-molecule
force-call count as `force_calls`.
"""

from __future__ import annotations

import json
import os

import pytest

from validate import Evaluator, discover_xtb_molecules
from utils import HARTREE_TO_KJ


def test_discover_xtb_returns_all_molecules(molecules_layout_dir):
    molecules, baselines = discover_xtb_molecules(str(molecules_layout_dir), "train")
    assert set(molecules) == {"12345_0", "abemaciclib"}
    assert set(baselines.keys()) == {"12345_0", "abemaciclib"}


def test_discover_xtb_reads_the_requested_split(molecules_layout_dir):
    molecules, baselines = discover_xtb_molecules(str(molecules_layout_dir), "valid")
    assert set(molecules) == {"99999_1"}
    assert baselines["99999_1"]["n_steps"] == 9


def test_discover_xtb_unknown_split_raises(molecules_layout_dir):
    with pytest.raises(ValueError, match="Unknown split"):
        discover_xtb_molecules(str(molecules_layout_dir), "nope")


def test_known_split_without_a_manifest_raises_actionably(molecules_layout_dir):
    """`test` maps to the OOD set, whose Sella baseline may not exist yet.

    That must be a clear "not generated" error, not a bare KeyError or a
    silent empty split.
    """
    with pytest.raises(FileNotFoundError) as excinfo:
        discover_xtb_molecules(str(molecules_layout_dir), "test")
    message = str(excinfo.value)
    assert "ood_XTB.json" in message
    assert "main" in message and "new dataset release" in message


def test_split_registry_covers_the_cli_choices():
    from validate import SPLIT_NAMES

    assert SPLIT_NAMES == ("train", "valid", "test", "dipeptides")


def test_ood_split_reads_from_the_evaluation_tree(tmp_path):
    """The OOD split resolves XYZ paths under evaluation/test, not molecules/xyz."""
    (tmp_path / "evaluation" / "test" / "xyz").mkdir(parents=True)
    (tmp_path / "evaluation" / "test" / "ood_XTB.json").write_text(
        json.dumps(
            {
                "555_1": {
                    "n_atoms": 4,
                    "initial_energy": -3.0,
                    "final_energy": -3.1,
                    "force_calls": 11,
                    "converged": True,
                }
            }
        )
    )
    molecules, baselines = discover_xtb_molecules(str(tmp_path), "test")
    assert molecules == ["555_1"]
    assert baselines["555_1"]["xyz_path"] == os.path.join(
        str(tmp_path), "evaluation", "test", "xyz", "555_1_mm.xyz"
    )


def test_discover_xtb_baseline_schema(molecules_layout_dir):
    _, baselines = discover_xtb_molecules(str(molecules_layout_dir), "train")
    for entry in baselines.values():
        assert set(entry.keys()) == {"n_steps", "improvement", "n_atoms", "xyz_path", "initial_energy", "final_energy", "method"}


def test_discover_xtb_maps_force_calls_onto_n_steps(molecules_layout_dir):
    """The manifest key is `force_calls`; the harness metric stays `n_steps`."""
    raw = json.loads((molecules_layout_dir / "train_XTB.json").read_text())
    _, baselines = discover_xtb_molecules(str(molecules_layout_dir), "train")
    for mol_name, entry in raw.items():
        assert "n_steps" not in entry
        assert baselines[mol_name]["n_steps"] == entry["force_calls"]


def test_discover_xtb_hartree_to_kj(molecules_layout_dir):
    _, baselines = discover_xtb_molecules(str(molecules_layout_dir), "train")
    # 12345_0: initial=-10.3510, final=-10.3770 -> improvement in Hartree = 0.026
    expected = (-10.3510 - (-10.3770)) * HARTREE_TO_KJ
    assert abs(baselines["12345_0"]["improvement"] - expected) < 1e-6


def test_discover_xtb_xyz_path(molecules_layout_dir):
    _, baselines = discover_xtb_molecules(str(molecules_layout_dir), "train")
    assert baselines["12345_0"]["xyz_path"] == os.path.join(
        str(molecules_layout_dir), "xyz", "12345_0_mm.xyz"
    )
    assert baselines["abemaciclib"]["xyz_path"] == os.path.join(
        str(molecules_layout_dir), "xyz", "abemaciclib_mm.xyz"
    )
    assert os.path.exists(baselines["12345_0"]["xyz_path"])
    assert os.path.exists(baselines["abemaciclib"]["xyz_path"])


def test_discover_xtb_sort_order(molecules_layout_dir):
    molecules, _ = discover_xtb_molecules(str(molecules_layout_dir), "train")
    # abemaciclib: force_calls=24 x n_atoms=69 = 1656
    # 12345_0:     force_calls=12 x n_atoms=3  = 36
    # Descending cost -> abemaciclib first
    assert molecules[0] == "abemaciclib"
    assert molecules[1] == "12345_0"


def test_evaluator_init_xtb(molecules_layout_dir):
    ev = Evaluator(molecules_dir=str(molecules_layout_dir), split="train")
    assert len(ev.molecules) == 2
    assert "abemaciclib" in ev.baselines


def test_evaluator_defaults_to_train_split(molecules_layout_dir):
    ev = Evaluator(molecules_dir=str(molecules_layout_dir))
    assert ev.split == "train"
    assert set(ev.molecules) == {"12345_0", "abemaciclib"}


@pytest.mark.parametrize("split,count", [("train", 469), ("valid", 465)])
def test_evaluator_default_release_loads_current_prepared_splits(split, count):
    from pathlib import Path
    from validate import DATASET_RELEASE_ID, DEFAULT_MOLECULES_DIR

    assert DATASET_RELEASE_ID == "main-f59f2b89a-gfn2-v1"
    assert DEFAULT_MOLECULES_DIR.name == DATASET_RELEASE_ID
    evaluator = Evaluator(split=split)
    assert len(evaluator.molecules) == count
    for entry in evaluator.baselines.values():
        assert Path(entry["xyz_path"]).is_file()
        assert Path(entry["xyz_path"]).parent == DEFAULT_MOLECULES_DIR / "xyz"


def test_prepared_keys_do_not_double_append_suffix(tmp_path):
    (tmp_path / "train_XTB.json").write_text(json.dumps({"charged_mm": {
        "n_atoms": 1, "initial_energy": -2, "final_energy": -2.1,
        "force_calls": 200, "converged": False}}))
    names, baselines = discover_xtb_molecules(tmp_path, "train")
    assert names == ["charged_mm"]
    assert baselines["charged_mm"]["xyz_path"].endswith("/xyz/charged_mm.xyz")
    assert baselines["charged_mm"]["initial_energy"] == -2 * HARTREE_TO_KJ
    assert baselines["charged_mm"]["method"] == "GFN2-xTB"


def test_raw_release_uses_exact_key_even_if_legacy_geometry_exists(tmp_path):
    (tmp_path / 'release.json').write_text('{}')
    (tmp_path / 'xyz').mkdir()
    (tmp_path / 'xyz' / '123_mm.xyz').write_text('legacy')
    (tmp_path / 'train_XTB.json').write_text(json.dumps({'123': {
        'n_atoms': 1, 'initial_energy': -2, 'final_energy': -2.1,
        'force_calls': 10, 'converged': True}}))
    _, baselines = discover_xtb_molecules(tmp_path, 'train')
    # Missing release inputs must not silently resolve to a different geometry.
    assert baselines['123']['xyz_path'] == str(tmp_path / 'xyz' / '123.xyz')


def test_unpacked_raw_geometry_takes_precedence_over_legacy_suffix(tmp_path):
    (tmp_path / 'xyz').mkdir()
    for name in ('123.xyz', '123_mm.xyz'):
        (tmp_path / 'xyz' / name).write_text(name)
    (tmp_path / 'train_XTB.json').write_text(json.dumps({'123': {
        'n_atoms': 1, 'initial_energy': -2, 'final_energy': -2.1,
        'force_calls': 10, 'converged': True}}))
    _, baselines = discover_xtb_molecules(tmp_path, 'train')
    assert baselines['123']['xyz_path'] == str(tmp_path / 'xyz' / '123.xyz')
