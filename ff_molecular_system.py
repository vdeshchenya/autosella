"""GFN-FF energies and forces through the standalone xTB binary.

One topology is generated from the original XYZ with ``xtb --gfnff`` and reused
for every evaluation, including scoring and optimizer-prepared geometries.
Each subprocess gets its own temporary directory and the same topology bytes.
Units: positions in nm, energies in kJ/mol, forces in kJ/(mol*nm).
"""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from utils import ANGSTROM_TO_NM, FORCE_CONV, HARTREE_TO_KJ, parse_xyz, read_xyz_charge
from xtb_molecular_system import (
    _captured_output,
    _parse_gradient_file,
    _resolve_xtb_binary,
    _write_xyz,
    _xtb_environment,
)


_TOPOLOGY_READ_MESSAGE = "GFN-FF topology read from file successfully!"


class FFMolecularSystem:
    """Evaluate GFN-FF with a topology fixed to the original input geometry.

    Args:
        xyz_path: Charge-tagged XYZ file, with coordinates in Angstrom.
        verbose: Print xTB output, which is otherwise captured for error checks.
        num_threads: Threads per calculation; defaults to one.

    Construction generates the topology once. A calculation that fails to read
    it or replaces it raises an error instead of silently changing the model.
    The topology is private to this molecule and requires no OpenMM assets.
    """

    def __init__(self, xyz_path: str, verbose: bool = False, num_threads: int = 1):
        self._charge = read_xyz_charge(xyz_path)
        self._atomic_numbers, self._initial_positions = parse_xyz(xyz_path)
        if not np.all(np.isfinite(self._initial_positions)):
            raise ValueError("initial positions contain NaN or Inf")
        self.method = "GFN-FF"
        self.verbose = verbose
        self.num_threads = max(1, int(num_threads))
        self._xtb_bin = _resolve_xtb_binary()
        self._topology = self._generate_topology()
        self.topology_sha256 = hashlib.sha256(self._topology).hexdigest()

    @property
    def atomic_numbers(self) -> np.ndarray:
        return self._atomic_numbers

    @property
    def initial_positions(self) -> np.ndarray:
        return self._initial_positions

    @property
    def charge(self) -> int:
        return self._charge

    def _run(self, arguments: list[str], directory: Path) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            [self._xtb_bin, *arguments],
            cwd=directory,
            env=_xtb_environment(self.num_threads),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if self.verbose:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            details = _captured_output(completed)[-4000:]
            raise RuntimeError(
                f"GFN-FF xTB exited with status {completed.returncode}: {details}"
            )
        return completed

    def _generate_topology(self) -> bytes:
        with tempfile.TemporaryDirectory(prefix="gfnff_topology_") as tmp_str:
            tmp = Path(tmp_str)
            _write_xyz(
                self._atomic_numbers,
                self._initial_positions / ANGSTROM_TO_NM,
                tmp / "input.xyz",
                self._charge,
            )
            # Use the main driver: in xTB 6.7.1, `xtb topo --chrg` can
            # generate a neutral topology even for a charged input. The main
            # GFN-FF driver honors --chrg when constructing the topology.
            completed = self._run(
                ["input.xyz", "--gfnff", "--norestart",
                 "--chrg", str(self._charge), "--parallel", str(self.num_threads)],
                tmp,
            )
            topology_path = tmp / "gfnff_topo"
            if not topology_path.is_file() or topology_path.stat().st_size == 0:
                details = _captured_output(completed)[-4000:]
                raise RuntimeError(f"xTB did not produce a GFN-FF topology: {details}")
            return topology_path.read_bytes()

    def compute(self, positions, **kwargs) -> tuple[float, np.ndarray]:
        """Return energy and forces using the original input's topology."""
        del kwargs
        positions_arr = np.asarray(positions, dtype=float)
        if positions_arr.shape != self._initial_positions.shape:
            raise ValueError(
                f"positions must have shape {self._initial_positions.shape}, "
                f"got {positions_arr.shape}"
            )
        if not np.all(np.isfinite(positions_arr)):
            raise ValueError("positions contain NaN or Inf")

        with tempfile.TemporaryDirectory(prefix="gfnff_") as tmp_str:
            tmp = Path(tmp_str)
            _write_xyz(
                self._atomic_numbers,
                positions_arr / ANGSTROM_TO_NM,
                tmp / "input.xyz",
                self._charge,
            )
            topology_path = tmp / "gfnff_topo"
            topology_path.write_bytes(self._topology)
            completed = self._run(
                ["input.xyz", "--gfnff", "--grad", "--restart",
                 "--chrg", str(self._charge), "--parallel", str(self.num_threads)],
                tmp,
            )
            # xTB can exit successfully after rejecting and regenerating a
            # topology. Require positive confirmation that the supplied one was
            # loaded, and check that it was not replaced during the calculation.
            output = completed.stdout + completed.stderr
            if _TOPOLOGY_READ_MESSAGE not in output:
                details = _captured_output(completed)[-4000:]
                raise RuntimeError(f"GFN-FF did not reuse the original topology: {details}")
            if not topology_path.is_file() or topology_path.read_bytes() != self._topology:
                raise RuntimeError("GFN-FF changed the original topology")
            gradient_path = tmp / "gradient"
            if not gradient_path.is_file():
                raise RuntimeError(f"GFN-FF did not produce a gradient file in {tmp}")
            energy_hartree, grad_eh_bohr = _parse_gradient_file(
                gradient_path, len(self._atomic_numbers)
            )

        energy = energy_hartree * HARTREE_TO_KJ
        forces = -grad_eh_bohr * FORCE_CONV
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)):
            raise RuntimeError("GFN-FF produced non-finite energy or forces")
        return energy, np.asarray(forces, dtype=float)


MolecularSystem = FFMolecularSystem
