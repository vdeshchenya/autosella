"""Edge-case tests for utils.parse_xyz / write_xyz — pure Python."""

from __future__ import annotations

import numpy as np
import pytest

from utils import ANGSTROM_TO_NM, parse_xyz, read_xyz_charge, write_xyz


# ── parse_xyz respects count header; ignores trailing content ──────────────

def test_parse_xyz_ignores_trailing_lines(tmp_path):
    content = (
        "2\n"
        "header\n"
        "O 0.0 0.0 0.0\n"
        "H 0.1 0.0 0.0\n"
        "this line should be ignored\n"
        "so should this\n"
    )
    path = tmp_path / "trailing.xyz"
    path.write_text(content)

    z, pos = parse_xyz(str(path))
    assert list(z) == [8, 1]
    assert pos.shape == (2, 3)


def test_parse_xyz_raises_when_fewer_lines_than_count(tmp_path):
    """Header declares more atoms than coordinate lines → ValueError.

    Guards against silent truncation of malformed XYZ files.
    """
    content = "3\nheader\nO 0.0 0.0 0.0\n"
    path = tmp_path / "short.xyz"
    path.write_text(content)

    with pytest.raises(ValueError, match="declares 3 atoms"):
        parse_xyz(str(path))


def test_parse_xyz_raises_when_zero_coord_lines(tmp_path):
    """Degenerate case: header says 5 atoms but no coords at all."""
    content = "5\nheader\n"
    path = tmp_path / "empty.xyz"
    path.write_text(content)

    with pytest.raises(ValueError, match="declares 5 atoms"):
        parse_xyz(str(path))


def test_parse_xyz_converts_angstrom_to_nm(tmp_path):
    """Ensure the Å → nm conversion actually happens (not the identity)."""
    content = "1\n\nH 1.0 2.0 3.0\n"
    path = tmp_path / "one.xyz"
    path.write_text(content)

    z, pos = parse_xyz(str(path))
    assert np.allclose(pos[0], [0.1, 0.2, 0.3])
    # ANGSTROM_TO_NM is applied once
    assert np.allclose(pos[0], np.array([1.0, 2.0, 3.0]) * ANGSTROM_TO_NM)


def test_parse_xyz_accepts_extra_whitespace(tmp_path):
    """Extra whitespace around numbers shouldn't break parsing."""
    content = "1\n   comment line   \n   H    1.0   2.0   3.0   \n"
    path = tmp_path / "spacey.xyz"
    path.write_text(content)

    z, pos = parse_xyz(str(path))
    assert list(z) == [1]
    assert np.allclose(pos[0], [0.1, 0.2, 0.3])


# ── write_xyz preserves column format ──────────────────────────────────────

def test_write_xyz_includes_atom_count_and_charge_comment(tmp_path):
    z = np.array([1, 8], dtype=int)
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
    out = tmp_path / "pair.xyz"
    write_xyz(z, pos, str(out), charge=0)

    lines = out.read_text().splitlines()
    assert int(lines[0]) == 2          # atom count on first line
    assert lines[1] == "charge=0"      # charge tag on the comment line
    assert lines[2].startswith("H ")
    assert lines[3].startswith("O ")


def test_write_xyz_accepts_list_inputs(tmp_path):
    """write_xyz converts via asarray, so it should accept plain lists."""
    z = [1, 8]
    pos = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    out = tmp_path / "lists.xyz"
    write_xyz(z, pos, str(out), charge=0)

    z_back, pos_back = parse_xyz(str(out))
    assert list(z_back) == z
    assert np.allclose(pos_back, pos)


@pytest.mark.parametrize("atomic_number", [0, -1, 119])
def test_write_xyz_unknown_element_raises(tmp_path, atomic_number):
    """Writing an atom with no symbol mapping is rejected (not silent)."""
    out = tmp_path / "bogus.xyz"
    z = np.array([atomic_number], dtype=int)
    pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="unsupported atomic numbers"):
        write_xyz(z, pos, str(out), charge=0)


def test_write_xyz_round_trip_supports_elements_outside_legacy_subset(tmp_path):
    z = np.array([2, 26, 30, 34, 54, 79, 118], dtype=int)
    pos = np.arange(3 * len(z), dtype=float).reshape(-1, 3) / 100
    out = tmp_path / "expanded_elements.xyz"
    write_xyz(z, pos, str(out), charge=-2)

    parsed_z, parsed_pos = parse_xyz(str(out))
    np.testing.assert_array_equal(parsed_z, z)
    np.testing.assert_allclose(parsed_pos, pos, atol=1e-9)
    assert read_xyz_charge(str(out)) == -2


# ── molecular charge round-trip ──────────────────────────────────────────────
# The charge lives on the XYZ comment line and is passed to xtb via --chrg.
# Losing it silently changes the potential, so it is a hard requirement.

@pytest.mark.parametrize("charge", [0, 1, -1, 2, -2, -8])
def test_charge_round_trips_through_write_and_read(tmp_path, charge):
    z = np.array([8, 1], dtype=int)
    pos = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float)
    out = tmp_path / f"ion_{charge}.xyz"
    write_xyz(z, pos, str(out), charge=charge)
    assert read_xyz_charge(str(out)) == charge


def test_read_xyz_charge_rejects_untagged_comment(tmp_path):
    """A blank/legacy comment line must fail loudly, not default to neutral."""
    out = tmp_path / "legacy.xyz"
    out.write_text("1\n\nO   0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="charge="):
        read_xyz_charge(str(out))


def test_read_xyz_charge_rejects_extra_text(tmp_path):
    out = tmp_path / "chatty.xyz"
    out.write_text("1\ncharge=1 energy=-5.0\nO   0.0 0.0 0.0\n")
    with pytest.raises(ValueError, match="charge="):
        read_xyz_charge(str(out))


def test_write_xyz_rejects_bool_charge(tmp_path):
    """bool is an int subclass; True must not silently mean charge=+1."""
    z = np.array([8], dtype=int)
    pos = np.array([[0.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(TypeError):
        write_xyz(z, pos, str(tmp_path / "b.xyz"), charge=True)


def test_write_xyz_rejects_non_finite_positions(tmp_path):
    z = np.array([8], dtype=int)
    pos = np.array([[0.0, np.nan, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="NaN or Inf"):
        write_xyz(z, pos, str(tmp_path / "nan.xyz"), charge=0)


def test_every_shipped_xyz_declares_a_charge():
    """Every molecule the workers can be handed must be charge-tagged."""
    import pathlib

    releases_dir = pathlib.Path(__file__).resolve().parent.parent / "molecules" / "releases"
    files = sorted(releases_dir.rglob("*.xyz"))
    assert files, f"no released XYZ files under {releases_dir}"
    for path in files:
        read_xyz_charge(str(path))  # raises if the tag is missing/malformed
