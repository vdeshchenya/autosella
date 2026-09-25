"""Tests for utils.py — pure Python, runs everywhere."""

from __future__ import annotations

import numpy as np
import pytest

from utils import (
    ANGSTROM_TO_NM,
    BOHR_PER_NM,
    EV_TO_HARTREE,
    EV_TO_KJ,
    FORCE_CONV,
    HARTREE_TO_KJ,
    SYMBOL_TO_Z,
    Z_TO_SYMBOL,
    parse_xyz,
    write_xyz,
)


# ── constants ────────────────────────────────────────────────────────────────

def test_hartree_to_kj_value():
    assert abs(HARTREE_TO_KJ - 2625.4996394799) < 1e-6


def test_bohr_per_nm_inverse():
    assert abs(BOHR_PER_NM * 0.0529177210544 - 1.0) < 1e-12


def test_angstrom_to_nm():
    assert ANGSTROM_TO_NM == 0.1


def test_force_conv_derivation():
    assert FORCE_CONV == HARTREE_TO_KJ * BOHR_PER_NM


def test_ev_to_kj_derivation():
    assert EV_TO_KJ == HARTREE_TO_KJ * EV_TO_HARTREE


# ── element maps ─────────────────────────────────────────────────────────────

def test_element_map_inverse():
    for sym, z in SYMBOL_TO_Z.items():
        if z > 0:
            assert Z_TO_SYMBOL[z] == sym


def test_element_map_unique_values():
    values = list(SYMBOL_TO_Z.values())
    assert len(values) == len(set(values))
    assert all(isinstance(v, int) and v >= 0 for v in values)
    assert SYMBOL_TO_Z["X"] == 0  # ASE's dummy symbol is excluded from XYZ writing.
    assert 0 not in Z_TO_SYMBOL


def test_element_map_contains_common_elements():
    for sym in ["H", "C", "N", "O", "F", "Cl", "Br", "I"]:
        assert sym in SYMBOL_TO_Z


def test_element_map_covers_periodic_table():
    assert set(Z_TO_SYMBOL) == set(range(1, 119))
    assert SYMBOL_TO_Z["Zn"] == 30
    assert SYMBOL_TO_Z["Se"] == 34
    assert SYMBOL_TO_Z["Og"] == 118


# ── parse_xyz ────────────────────────────────────────────────────────────────

def test_parse_xyz_water(water_xyz_path):
    atomic_numbers, positions_nm = parse_xyz(str(water_xyz_path))

    assert atomic_numbers.shape == (3,)
    assert atomic_numbers.dtype == np.int_ or atomic_numbers.dtype == int or np.issubdtype(atomic_numbers.dtype, np.integer)
    assert list(atomic_numbers) == [8, 1, 1]

    assert positions_nm.shape == (3, 3)
    assert positions_nm.dtype == float

    # Origin atom at (0, 0, 0), converted from Å to nm (still 0)
    assert np.allclose(positions_nm[0], [0.0, 0.0, 0.0])

    # First H at (0.757 Å, 0.586 Å, 0) → (0.0757, 0.0586, 0) nm
    assert np.allclose(positions_nm[1], [0.0757, 0.0586, 0.0], atol=1e-9)


def test_parse_xyz_unknown_symbol(tmp_path):
    bogus = tmp_path / "bogus.xyz"
    bogus.write_text("1\ncomment\nZz 0.0 0.0 0.0\n")
    with pytest.raises(KeyError):
        parse_xyz(str(bogus))


# ── write_xyz round-trip ─────────────────────────────────────────────────────

def test_write_parse_round_trip(tmp_path):
    z = np.array([6, 1, 1, 1, 1], dtype=int)  # methane
    pos_nm = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0635, 0.0635, 0.0635],
            [-0.0635, -0.0635, 0.0635],
            [0.0635, -0.0635, -0.0635],
            [-0.0635, 0.0635, -0.0635],
        ],
        dtype=float,
    )
    out = tmp_path / "methane.xyz"
    write_xyz(z, pos_nm, str(out), charge=0)

    z_back, pos_back = parse_xyz(str(out))
    assert list(z_back) == list(z)
    # Write uses %15.8f; round-trip precision is ~1e-9 nm
    assert np.allclose(pos_back, pos_nm, atol=1e-8)


def test_write_xyz_real_file_round_trip(water_xyz_path, tmp_path):
    z, pos = parse_xyz(str(water_xyz_path))
    out = tmp_path / "water_copy.xyz"
    write_xyz(z, pos, str(out), charge=0)
    z2, pos2 = parse_xyz(str(out))
    assert list(z2) == list(z)
    assert np.allclose(pos2, pos, atol=1e-8)
