"""
ORCA/OPI-based molecular system for DFT energy and force calculations.

Each ``compute`` call creates a temporary ORCA working directory, writes the
current geometry, runs a r2SCAN-3c TightSCF ENGRAD single point through OPI, and
parses ORCA's ``.engrad`` file.

Units match the other MolecularSystem implementations:
energy in kJ/mol, forces in kJ/(mol*nm), positions in nm.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import tempfile
from pathlib import Path

import numpy as np

from opi.core import Calculator
from opi.input.simple_keywords import Dft, Scf, Task
from opi.input.structures.structure import Structure

from utils import FORCE_CONV, HARTREE_TO_KJ, parse_xyz, read_xyz_charge, write_xyz


def _expand_shell_vars(value: str, env: dict[str, str]) -> str:
    pattern = re.compile(r"\$(\w+)|\$\{([^}:]+)(?::\+([^}]*))?\}")

    def replace(match: re.Match[str]) -> str:
        bare, braced, alt = match.groups()
        name = bare or braced
        current = env.get(name, "")
        if alt is not None:
            return alt if current else ""
        return current

    previous = None
    expanded = value
    for _ in range(10):
        if expanded == previous:
            break
        previous = expanded
        expanded = pattern.sub(replace, expanded)
    return expanded


def _source_simple_env_file(path: Path) -> None:
    """Apply simple ``export NAME=value`` lines from a shell env file."""
    if not path.is_file():
        return

    env = os.environ.copy()
    env.setdefault("ORCA_DIR", str(path.resolve().parent))

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("export "):
            continue
        assignment = stripped[len("export ") :].strip()
        if "=" not in assignment:
            continue
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not name:
            continue
        try:
            parts = shlex.split(raw_value)
        except ValueError:
            continue
        value = " ".join(parts)
        expanded = _expand_shell_vars(value, env)
        os.environ[name] = expanded
        env[name] = expanded

    if os.environ.get("ORCA_BINARY") and not os.environ.get("OPI_ORCA"):
        os.environ["OPI_ORCA"] = os.environ["ORCA_BINARY"]


def _as_orca_executable(path_str: str) -> Path | None:
    path = Path(path_str).expanduser()
    if path.is_dir():
        path = path / "orca"
    if path.is_file() and os.access(path, os.X_OK):
        return path
    return None


def _configure_opi_orca_path() -> None:
    """Configure OPI from explicit environment variables or ``PATH``.

    ``ORCA_ENV_FILE`` is optional and is useful on clusters whose ORCA module
    ships a small shell file with PATH/library exports.  No machine-specific
    location is assumed by this repository.
    """
    env_file = os.environ.get("ORCA_ENV_FILE")
    if env_file:
        env_path = Path(env_file).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError(
                f"ORCA_ENV_FILE does not exist or is not a file: {env_path}"
            )
        _source_simple_env_file(env_path)

    if os.environ.get("OPI_ORCA"):
        return

    candidates: list[str] = []
    for env_var in ("ORCA_BINARY", "ORCA_BIN", "ORCA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(value)

    found = shutil.which("orca")
    if found:
        candidates.append(found)

    run_script = Path(__file__).resolve().with_name("orca_run.sh")
    if run_script.is_file():
        for line in run_script.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                parts = shlex.split(stripped)
            except ValueError:
                continue
            if parts:
                candidates.append(parts[0])
                break

    for candidate in candidates:
        executable = _as_orca_executable(candidate)
        if executable is not None:
            os.environ["OPI_ORCA"] = str(executable)
            return

    raise RuntimeError(
        "ORCA executable not found. Set OPI_ORCA or ORCA_BINARY, add orca "
        "to PATH, or point ORCA_ENV_FILE to your local ORCA environment file."
    )


def _parse_engrad(path: Path, n_atoms: int) -> tuple[float, np.ndarray]:
    """Parse ORCA ``.engrad`` energy and Cartesian gradient."""
    values: list[float] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values.append(float(stripped.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue

    if len(values) < 2 + 3 * n_atoms:
        raise ValueError(
            f"expected at least {2 + 3 * n_atoms} numeric values in {path}, "
            f"found {len(values)}"
        )

    parsed_n_atoms = int(round(values[0]))
    if parsed_n_atoms != n_atoms:
        raise ValueError(
            f"{path} reports {parsed_n_atoms} atoms, expected {n_atoms}"
        )

    energy_hartree = float(values[1])
    gradient = np.asarray(values[2 : 2 + 3 * n_atoms], dtype=float).reshape(n_atoms, 3)
    if not np.isfinite(energy_hartree):
        raise ValueError(f"non-finite energy in {path}")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"non-finite Cartesian gradient in {path}")
    return energy_hartree, gradient


class DFTMolecularSystem:
    """
    ORCA/OPI-based molecular system for energy and force calculations.

    Args:
        xyz_path: Path to an XYZ file. Coordinates are read in Angstrom and
            converted to nm. The molecular charge is read from an exact
            ``charge=<integer>`` declaration on the second line.
        multiplicity: Spin multiplicity.
        ncores: ORCA cores for each single-point calculation.
    """

    def __init__(
        self,
        xyz_path: str,
        multiplicity: int = 1,
        ncores: int = 48,
    ):
        self._charge = read_xyz_charge(xyz_path)
        self._atomic_numbers, self._initial_positions = parse_xyz(xyz_path)
        self.multiplicity = int(multiplicity)
        self.ncores = max(1, int(ncores))
        self.last_error: str | None = None

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
        Compute energy and forces using ORCA ENGRAD through OPI.

        Args:
            positions: array-like (N, 3) in nm.

        Returns:
            (energy, forces): energy in kJ/mol, forces in kJ/(mol*nm).
        """
        del kwargs
        positions_arr = np.asarray(positions, dtype=float)
        if positions_arr.shape != self._initial_positions.shape:
            self.last_error = (
                f"position shape {positions_arr.shape} does not match "
                f"{self._initial_positions.shape}"
            )
            return float("inf"), np.zeros_like(positions_arr, dtype=float)

        try:
            _configure_opi_orca_path()
            with tempfile.TemporaryDirectory(prefix="orca_opi_") as tmp_str:
                tmp = Path(tmp_str)
                xyz_path = tmp / "input.xyz"
                write_xyz(
                    self._atomic_numbers,
                    positions_arr,
                    str(xyz_path),
                    self._charge,
                )

                env_threads = str(self.ncores)
                for var in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    os.environ[var] = env_threads
                os.environ["OMP_DYNAMIC"] = "FALSE"
                os.environ["MKL_DYNAMIC"] = "FALSE"

                structure = Structure.from_xyz(xyz_path)
                structure.charge = self._charge
                structure.multiplicity = self.multiplicity

                try:
                    calc = Calculator(
                        basename="orca", working_dir=tmp, version_check=False
                    )
                except TypeError:
                    calc = Calculator(basename="orca", working_dir=tmp)
                calc.structure = structure
                calc.input.add_simple_keywords(
                    Dft.R2SCAN_3C,
                    Task.ENGRAD,
                    Scf.NOAUTOSTART,
                    Scf.TIGHTSCF,
                )
                calc.input.ncores = self.ncores

                calc.write_input()
                calc.run()

                output = calc.get_output()
                if not output.terminated_normally():
                    raise RuntimeError(f"ORCA did not terminate normally in {tmp}")
                output.parse()
                if hasattr(output, "scf_converged") and not output.scf_converged():
                    raise RuntimeError(f"ORCA SCF did not converge in {tmp}")

                engrad_path = tmp / "orca.engrad"
                if not engrad_path.is_file():
                    raise RuntimeError(f"ORCA did not produce {engrad_path}")
                energy_hartree, grad_eh_bohr = _parse_engrad(
                    engrad_path, len(self._atomic_numbers)
                )

            self.last_error = None
            return (
                energy_hartree * HARTREE_TO_KJ,
                -grad_eh_bohr * FORCE_CONV,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return float("inf"), np.zeros_like(positions_arr, dtype=float)


# Alias so callers can do: from dft_molecular_system import MolecularSystem
MolecularSystem = DFTMolecularSystem
