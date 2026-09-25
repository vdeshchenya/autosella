"""
XTB-based molecular system for energy minimization.

Energies and forces are computed with GFN2-xTB or GFN1-xTB by invoking
the standalone ``xtb`` 6.7.1 binary as a subprocess. We avoid the
``xtb-python`` bindings so the only runtime requirement is the ``xtb``
executable on $PATH (override with $XTB_BIN).
Atomic numbers and the starting geometry are loaded from an XYZ file.
Units: energy in kJ/mol, forces in kJ/(mol·nm).
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from utils import (
    ANGSTROM_TO_NM,
    FORCE_CONV,
    HARTREE_TO_KJ,
    Z_TO_SYMBOL,
    parse_xyz,
    read_xyz_charge,
)


_METHOD_TO_GFN: dict[str, str] = {
    "GFN2-xTB": "2",
    "GFN1-xTB": "1",
}


def _resolve_xtb_binary() -> str:
    explicit = os.environ.get("XTB_BIN")
    if explicit:
        return explicit
    found = shutil.which("xtb")
    if not found:
        raise RuntimeError(
            "xtb binary not found on PATH (set $XTB_BIN to override)."
        )
    return found


def _parse_gradient_file(path: Path, n_atoms: int) -> tuple[float, np.ndarray]:
    """Parse a Turbomole-format ``gradient`` file written by ``xtb --grad``.

    Returns (energy_hartree, gradient_eh_per_bohr) with gradient shape (N, 3).
    """
    text = path.read_text()
    # Fortran 'D' exponent → Python 'E'
    text = re.sub(r"([0-9.])[Dd]([+-]?\d)", r"\1E\2", text)

    energy_match = re.search(r"SCF energy\s*=\s*(\S+)", text)
    if not energy_match:
        raise ValueError(f"could not find SCF energy in {path}")
    energy = float(energy_match.group(1))

    # gradient is the last N 3-float lines before $end
    grad_rows: list[list[float]] = []
    for line in reversed(text.splitlines()):
        s = line.strip()
        if not s or s.startswith("$"):
            continue
        parts = s.split()
        if len(parts) != 3:
            continue
        try:
            grad_rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
        if len(grad_rows) == n_atoms:
            break

    if len(grad_rows) != n_atoms:
        raise ValueError(
            f"expected {n_atoms} gradient rows in {path}, found {len(grad_rows)}"
        )

    grad_rows.reverse()
    return energy, np.asarray(grad_rows, dtype=float)


def _write_xyz(
    atomic_numbers: np.ndarray,
    positions_ang: np.ndarray,
    path: Path,
    charge: int,
) -> None:
    n = len(atomic_numbers)
    lines = [f"{n}\n", f"charge={charge}\n"]
    for z, (x, y, z_) in zip(atomic_numbers, positions_ang):
        lines.append(f"{Z_TO_SYMBOL[int(z)]:<2} {x:20.10f} {y:20.10f} {z_:20.10f}\n")
    path.write_text("".join(lines))


def _captured_output(completed: subprocess.CompletedProcess) -> str:
    """Return both output streams; xTB splits useful errors across them."""
    parts = []
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout.strip()}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr.strip()}")
    return "\n".join(parts)


def _xtb_environment(num_threads: int) -> dict[str, str]:
    """Return an xTB environment with all common thread pools capped."""
    env = os.environ.copy()
    threads = str(max(1, int(num_threads)))
    for var in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[var] = threads
    env["OMP_DYNAMIC"] = "FALSE"
    env["MKL_DYNAMIC"] = "FALSE"
    return env


# ── main class ────────────────────────────────────────────────────────────────


class XTBMolecularSystem:
    """
    xTB-based molecular system for energy and force calculations.

    Atomic numbers and starting positions are parsed from an XYZ file.

    Args:
        xyz_path:  Path to an XYZ file. Element symbols + Ångström coords;
            positions are converted to nm on load.
        method:    xTB method — "GFN2-xTB" (default) or "GFN1-xTB".
        accuracy:  SCF convergence threshold (default 0.2). Increase (e.g. 3.0)
            to trade accuracy for speed in optimizer inner loops.
        verbose:   if False, xtb stdout/stderr are discarded.
        num_threads: threads to use for xTB calculations.
    """

    def __init__(
        self,
        xyz_path: str,
        method: str = "GFN2-xTB",
        accuracy: float = 0.2,
        verbose: bool = False,
        num_threads: int = 1,
    ):
        if method not in _METHOD_TO_GFN:
            raise ValueError(
                f"Unknown xTB method {method!r}; expected one of "
                f"{sorted(_METHOD_TO_GFN)}"
            )
        self._charge = read_xyz_charge(xyz_path)
        self._atomic_numbers, self._initial_positions = parse_xyz(xyz_path)
        self.method = method
        self._gfn = _METHOD_TO_GFN[method]
        self.accuracy = accuracy
        self.verbose = verbose
        self.num_threads = max(1, int(num_threads))
        self._xtb_bin = _resolve_xtb_binary()

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers

    @property
    def initial_positions(self) -> np.ndarray:
        return self._initial_positions

    @property
    def charge(self) -> int:
        return self._charge

    def compute(self, positions, **kwargs) -> tuple[float, np.ndarray]:
        """
        Compute energy and forces by shelling out to the xtb binary.

        Args:
            positions: array-like (N, 3) in nm.

        Returns:
            (energy, forces): energy in kJ/mol, forces in kJ/(mol·nm).
        """
        positions_arr = np.asarray(positions, dtype=float)
        if positions_arr.shape != self._initial_positions.shape:
            raise ValueError(
                f"positions must have shape {self._initial_positions.shape}, "
                f"got {positions_arr.shape}"
            )
        if not np.all(np.isfinite(positions_arr)):
            raise ValueError("positions contain NaN or Inf")

        positions_ang = positions_arr / ANGSTROM_TO_NM

        with tempfile.TemporaryDirectory(prefix="xtb_") as tmp_str:
            tmp = Path(tmp_str)
            xyz_path = tmp / "input.xyz"
            _write_xyz(
                self._atomic_numbers,
                positions_ang,
                xyz_path,
                self._charge,
            )

            threads = str(self.num_threads)
            env = _xtb_environment(self.num_threads)

            cmd = [self._xtb_bin, str(xyz_path)]
            cmd.extend(("--gfn", self._gfn))
            cmd.extend([
                "--grad",
                "--acc", f"{self.accuracy:g}",
                "--chrg", str(self._charge),
                "--parallel", threads,
            ])
            if not self.verbose:
                cmd.append("--silent")

            completed = subprocess.run(
                cmd,
                cwd=tmp,
                env=env,
                stdout=None if self.verbose else subprocess.PIPE,
                stderr=None if self.verbose else subprocess.PIPE,
                text=not self.verbose,
                check=False,
            )
            if completed.returncode != 0:
                details = ""
                if not self.verbose:
                    details = _captured_output(completed)
                    if len(details) > 4000:
                        details = details[-4000:]
                suffix = f": {details}" if details else ""
                raise RuntimeError(
                    f"xtb exited with status {completed.returncode}{suffix}"
                )

            gradient_path = tmp / "gradient"
            if not gradient_path.is_file():
                raise RuntimeError(f"xtb did not produce a gradient file in {tmp}")

            energy_hartree, grad_eh_bohr = _parse_gradient_file(
                gradient_path, len(self._atomic_numbers)
            )

        energy = energy_hartree * HARTREE_TO_KJ
        forces = -grad_eh_bohr * FORCE_CONV
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise RuntimeError("xtb produced non-finite energy or forces")
        return energy, np.asarray(forces, dtype=float)


# Alias so callers can do: from xtb_molecular_system import MolecularSystem
MolecularSystem = XTBMolecularSystem
