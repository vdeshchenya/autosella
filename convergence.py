from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import ANGSTROM_TO_NM, BOHR_PER_NM, FORCE_CONV, HARTREE_TO_KJ

# ── GeomeTRIC-style thresholds ─────────────────────────────
# https://geometric.readthedocs.io/en/latest/how-it-works.html#equation-rmsd

GAU_TOL_E = 1.0e-6  # Eh
GAU_TOL_RMS_G = 3.0e-4  # Eh/Bohr
GAU_TOL_MAX_G = 4.5e-4  # Eh/Bohr
GAU_TOL_RMS_D = 1.2e-3 * ANGSTROM_TO_NM * BOHR_PER_NM  # A -> bohr
GAU_TOL_MAX_D = 1.8e-3 * ANGSTROM_TO_NM * BOHR_PER_NM  # A -> bohr

OPENMM_TOL_RMSF = 10.0  # kJ/mol/nm


# Horn's unit-quaternion alignment for corresponding, equally weighted atoms.

def _rotation_matrix(moving, target):
    """Return the proper rotation bringing centered ``moving`` onto ``target``."""
    covariance = moving.T @ target
    trace = np.trace(covariance)
    identity = np.eye(3)
    antisymmetric = np.array(
        [
            covariance[1, 2] - covariance[2, 1],
            covariance[2, 0] - covariance[0, 2],
            covariance[0, 1] - covariance[1, 0],
        ]
    )
    quaternion_matrix = np.block([
        [trace, antisymmetric[None, :]],
        [antisymmetric[:, None], covariance + covariance.T - trace * identity],
    ])

    _, eigenvectors = np.linalg.eigh(quaternion_matrix)
    scalar, vector = eigenvectors[0, -1], eigenvectors[1:, -1]
    cross_product = np.cross(identity, vector)
    return (
        (scalar**2 - vector @ vector) * identity
        + 2 * np.outer(vector, vector)
        + 2 * scalar * cross_product
    )


@dataclass
class ConvergenceState:
    criterion: str
    position_norm: float
    prev_pos: np.ndarray | None = None
    prev_energy: float | None = None
    converged: bool = False


def init_convergence_state(criterion: str, initial_pos: np.ndarray) -> ConvergenceState:
    pos = np.asarray(initial_pos, dtype=float)
    position_norm = float(max(1.0, np.sqrt(3 * np.square(pos).mean())))
    return ConvergenceState(criterion=criterion, position_norm=position_norm)


def _calc_displacement_bohr(
    Xnew: np.ndarray, Xold: np.ndarray, align: bool = True
) -> tuple[float, float]:
    """Return maximum and RMS per-atom displacement after rigid alignment."""
    old_centered = Xold - Xold.mean(axis=0)
    new_centered = Xnew - Xnew.mean(axis=0)
    if align:
        rotation = _rotation_matrix(new_centered, old_centered)
        new_centered = new_centered @ rotation.T

    displacement = np.linalg.norm(new_centered - old_centered, axis=1)
    return float(displacement.max()), float(np.sqrt(np.mean(displacement**2)))


def update_geometric_convergence(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    pos_arr = np.asarray(pos, dtype=float).copy()
    forces_arr = np.asarray(forces, dtype=float) / FORCE_CONV  # (N, 3) in Eh/Bohr

    atom_force_norms = np.sqrt(np.sum(forces_arr ** 2, axis=1))
    g_max_eh = float(np.max(atom_force_norms))
    g_rms_eh = float(np.sqrt(np.mean(atom_force_norms ** 2)))

    if state.prev_pos is not None:
        pos_bohr_new = pos_arr * BOHR_PER_NM
        pos_bohr_old = state.prev_pos * BOHR_PER_NM
        d_max_bohr, d_rms_bohr = _calc_displacement_bohr(pos_bohr_new, pos_bohr_old)
    else:
        d_max_bohr = float("inf")
        d_rms_bohr = float("inf")

    state.converged = False
    if state.prev_energy is not None:
        state.converged = (
            abs(energy - state.prev_energy) / HARTREE_TO_KJ < GAU_TOL_E
            and g_max_eh < GAU_TOL_MAX_G
            and g_rms_eh < GAU_TOL_RMS_G
            and d_max_bohr < GAU_TOL_MAX_D
            and d_rms_bohr < GAU_TOL_RMS_D
        )

    state.prev_pos = pos_arr
    state.prev_energy = float(energy)
    return state.converged


def update_openmm_convergence(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    del energy
    pos_arr = np.asarray(pos, dtype=float)
    forces_arr = np.asarray(forces, dtype=float)
    state.converged = (
        np.linalg.norm(forces_arr) / max(1.0, np.linalg.norm(pos_arr))
        < OPENMM_TOL_RMSF / state.position_norm
    )
    return state.converged


def update_convergence_state(
    state: ConvergenceState,
    pos: np.ndarray,
    energy: float,
    forces: np.ndarray,
) -> bool:
    if state.criterion == "geometric":
        return update_geometric_convergence(state, pos, energy, forces)
    if state.criterion == "openmm":
        return update_openmm_convergence(state, pos, energy, forces)
    raise ValueError(f"Invalid convergence criterion: {state.criterion}")


def is_converged(state: ConvergenceState) -> bool:
    return state.converged
