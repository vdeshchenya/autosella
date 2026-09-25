from __future__ import division
from pathlib import Path
import warnings
from typing import Callable
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from molecules.utils import ANGSTROM_TO_NM, EV_TO_KJ

warnings.filterwarnings("ignore", category=FutureWarning, module="ase")
from sella import Sella

import jax
_cache_dir = Path.home() / ".cache" / "sella" / "jax_cache"
_cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_enable_x64", True)
try:
    jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except (AttributeError, ValueError):
    pass


class _ForceCallBudgetExhausted(Exception):
    """Raised before a calculator call would exceed the force-call budget."""


class _WrappedCalc(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, calc_func, max_force_calls, **kwargs):
        super().__init__(**kwargs)
        if max_force_calls < 1:
            raise ValueError("max_force_calls must be positive")
        self.calc_func = calc_func
        self.max_force_calls = max_force_calls
        self.force_calls = 0
        self.last_positions_nm = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)
        if self.force_calls >= self.max_force_calls:
            raise _ForceCallBudgetExhausted
        pos_nm = self.atoms.get_positions() * ANGSTROM_TO_NM
        energy_kj, forces_kj_nm = self.calc_func(pos_nm)
        self.force_calls += 1
        self.last_positions_nm = pos_nm.copy()
        self.results["energy"] = energy_kj / EV_TO_KJ
        self.results["forces"] = np.array(forces_kj_nm) / EV_TO_KJ * ANGSTROM_TO_NM


def minimize_func(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    calc: Callable,
    max_force_calls: int,
    converged: Callable,
) -> tuple[np.ndarray, int]:
    pos_ang = np.array(positions) / ANGSTROM_TO_NM
    atoms = Atoms(numbers=atomic_numbers, positions=pos_ang)
    wrapper = _WrappedCalc(calc, max_force_calls=max_force_calls)
    atoms.calc = wrapper
    opt = Sella(atoms, internal=True, order=0, logfile=None)
    try:
        for _ in opt.irun(fmax=0, steps=max_force_calls - 1):
            if converged():
                break
    except _ForceCallBudgetExhausted:
        pass
    final_pos_nm = wrapper.last_positions_nm
    if final_pos_nm is None:
        raise RuntimeError("Sella made zero force calls")
    return final_pos_nm, wrapper.force_calls


def entrypoint():
    return minimize_func
