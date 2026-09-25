"""Shared constants, element tables, and I/O helpers.

Single source of truth for unit conversions, element symbol/number maps, and
XYZ read/write. Import from here instead of redefining in each module.
"""

from __future__ import annotations

import operator
import re

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

MAX_FORCE_CALLS = 200


# ── unit conversions ──────────────────────────────────────────────────────────

HARTREE_TO_KJ = 2625.4996394799              # kJ/mol per Hartree
BOHR_PER_NM = 1.0 / 0.0529177210544          # Bohr per nm (~18.8973)
ANGSTROM_TO_NM = 0.1                         # nm per Å
EV_TO_HARTREE = 1.0 / 27.211386245988
EV_TO_KJ = HARTREE_TO_KJ * EV_TO_HARTREE
FORCE_CONV = HARTREE_TO_KJ * BOHR_PER_NM     # Eh/a₀ → kJ/(mol·nm)


# ── element maps ──────────────────────────────────────────────────────────────

SYMBOL_TO_Z: dict[str, int] = dict(atomic_numbers)
Z_TO_SYMBOL: dict[int, str] = {
    z: symbol for z, symbol in enumerate(chemical_symbols) if z > 0
}


# ── XYZ I/O ───────────────────────────────────────────────────────────────────

_XYZ_CHARGE_PATTERN = re.compile(r"charge=([+-]?\d+)")


def read_xyz_charge(path: str) -> int:
    """Read the required integer molecular charge from an XYZ comment line."""
    try:
        with open(path) as fh:
            next(fh)
            comment = next(fh).strip()
    except (OSError, StopIteration) as exc:
        raise ValueError(
            f"XYZ at {path!r} must have `charge=<integer>` on its second line"
        ) from exc

    match = _XYZ_CHARGE_PATTERN.fullmatch(comment)
    if match is None:
        raise ValueError(
            f"XYZ at {path!r} must have exactly one `charge=<integer>` "
            "declaration on its second line"
        )
    return int(match.group(1))


def parse_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an XYZ file.

    Returns
    -------
    atomic_numbers: (N,) int array
    positions_nm:   (N, 3) float array, converted from Ångström to nm

    Raises
    ------
    ValueError
        If the header declares N atoms but the file has fewer than N
        coordinate lines.
    """
    with open(path) as fh:
        lines = fh.readlines()
    n = int(lines[0].strip())
    coord_lines = lines[2 : 2 + n]
    if len(coord_lines) != n:
        raise ValueError(
            f"XYZ at {path!r}: header declares {n} atoms but file has "
            f"{len(coord_lines)} coordinate line(s)"
        )
    z: list[int] = []
    coords: list[list[float]] = []
    for ln in coord_lines:
        parts = ln.split()
        z.append(SYMBOL_TO_Z[parts[0]])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return (
        np.asarray(z, dtype=int),
        np.asarray(coords, dtype=float) * ANGSTROM_TO_NM,
    )


def write_xyz(atomic_numbers, positions_nm, path: str, charge: int) -> None:
    """Write a charge-tagged XYZ. Positions are converted from nm to Ångström."""
    if isinstance(charge, bool):
        raise TypeError("charge must be an integer, not bool")
    try:
        integer_charge = operator.index(charge)
    except TypeError as exc:
        raise TypeError("charge must be an integer") from exc

    atomic_numbers = np.asarray(atomic_numbers, dtype=int)
    if atomic_numbers.ndim != 1:
        raise ValueError(
            f"atomic_numbers must have shape (N,), got {atomic_numbers.shape}"
        )

    positions_nm = np.asarray(positions_nm, dtype=float)
    expected_shape = (len(atomic_numbers), 3)
    if positions_nm.shape != expected_shape:
        raise ValueError(
            f"positions_nm must have shape {expected_shape}, got {positions_nm.shape}"
        )
    if not np.all(np.isfinite(positions_nm)):
        raise ValueError("positions_nm contains NaN or Inf")

    unsupported = sorted(set(atomic_numbers.tolist()) - set(Z_TO_SYMBOL))
    if unsupported:
        raise ValueError(f"unsupported atomic numbers: {unsupported}")

    pos_ang = positions_nm / ANGSTROM_TO_NM
    with open(path, "w") as fh:
        fh.write(f"{len(atomic_numbers)}\ncharge={integer_charge}\n")
        for z, (x, y, z_) in zip(atomic_numbers, pos_ang):
            fh.write(f"{Z_TO_SYMBOL[int(z)]:<2} {x:15.8f} {y:15.8f} {z_:15.8f}\n")
