"""A minimal importable optimizer, used only as a test fixture.

`distributed_validate/optimizer.py` can load a candidate either from
cloudpickled bytes (what the autoresearch loop ships) or from a module path.
Exercising the module-path branch needs *some* importable module exposing
`minimize_func` and `entrypoint()`; this is that module, and nothing more.
It is deliberately not a research starting point -- runs start from algo.py.
"""

from __future__ import annotations

import numpy as np


def minimize_func(positions, atomic_numbers, calc, max_force_calls, converged):
    """One steepest-descent step, then stop. Correct shape, no ambitions."""
    pos = np.asarray(positions, dtype=float).copy()
    _, forces = calc(pos)
    pos = pos + 1e-4 * np.asarray(forces, dtype=float)
    return pos, 1


def entrypoint():
    return minimize_func
