"""Additional edge cases for convergence.py."""

from __future__ import annotations

import numpy as np
import pytest

from convergence import (
    ConvergenceState,
    _calc_displacement_bohr,
    init_convergence_state,
    is_converged,
    update_convergence_state,
    update_xtb_convergence,
)


@pytest.fixture
def pos():
    return np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        dtype=float,
    )


# ── is_converged just reads the flag ─────────────────────────────────────

def test_is_converged_reflects_state():
    """is_converged is a pure getter — it does not re-evaluate criteria."""
    state = ConvergenceState(mode="xtb", converged=False)
    assert is_converged(state) is False
    state.converged = True
    assert is_converged(state) is True


# ── xtb: after converging, a regression should flip converged back to False ─

def test_xtb_can_un_converge(pos):
    """Each call evaluates freshly: convergence doesn't latch once set."""
    state = init_convergence_state("xtb", pos)
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    # Next call: tiny change → True
    update_xtb_convergence(state, pos + 1e-12, -100.000001, np.ones_like(pos) * 1e-10)
    assert state.converged is True

    # Now: large forces at a new step → False again
    update_xtb_convergence(state, pos + 2e-12, -100.0, np.ones_like(pos) * 100.0)
    assert state.converged is False


def test_xtb_large_energy_change_not_converged(pos):
    """Tiny forces and displacement but big ΔE → still not converged."""
    state = init_convergence_state("xtb", pos)
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    # ΔE = 0.1 kJ/mol = 3.8e-5 Hartree > GAU_TOL_E (1e-6 Ha)
    tiny_forces = np.ones_like(pos) * 1e-10
    result = update_xtb_convergence(state, pos + 1e-12, -100.1, tiny_forces)
    assert result is False


def test_xtb_large_rms_gradient_not_converged(pos):
    """g_max below threshold, but RMS over atoms above threshold.

    Easiest way to test: use forces of uniform magnitude so max==rms.
    """
    state = init_convergence_state("xtb", pos)
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))

    # Uniform atom norms between RMS (3e-4) and max (4.5e-4) thresholds.
    from utils import FORCE_CONV
    forces = np.zeros_like(pos)
    forces[:, 0] = 3.75e-4 * FORCE_CONV
    result = update_xtb_convergence(state, pos + 1e-12, -100.0, forces)
    assert result is False


# ── update_convergence_state dispatches to the right updater ───────────────

def test_dispatch_xtb_updates_prev_pos(pos):
    state = init_convergence_state("xtb", pos)
    assert state.prev_pos is None
    update_convergence_state(state, pos, -100.0, np.zeros_like(pos))
    assert state.prev_pos is not None
    np.testing.assert_allclose(state.prev_pos, pos)


# ── init_convergence_state accepts list inputs ─────────────────────────────

def test_init_convergence_accepts_list_positions():
    """The initializer accepts the optimizer's Python-list positions."""
    state = init_convergence_state("xtb", [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    assert state.mode == "xtb"
    assert state.prev_pos is None


@pytest.mark.parametrize("n_atoms", [1, 2, 3, 12])
def test_aligned_displacement_removes_rigid_rotation_and_translation(n_atoms):
    """Cover degenerate atom/linear geometries and a nonplanar geometry."""
    rng = np.random.default_rng(1941)
    original = rng.normal(size=(n_atoms, 3))
    axis = np.array([1.0, -2.0, 3.0])
    axis /= np.linalg.norm(axis)
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    angle = 1.23
    rotation = (np.eye(3) * np.cos(angle)
                + (1 - np.cos(angle)) * np.outer(axis, axis)
                + np.sin(angle) * cross)
    transformed = original @ rotation.T + np.array([3.0, -4.0, 2.0])

    maximum, rms = _calc_displacement_bohr(transformed, original)
    assert maximum < 1e-12
    assert rms < 1e-12


def test_aligned_displacement_preserves_internal_deformation():
    original = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    stretched = 1.025 * original + np.array([2.0, 3.0, 4.0])
    maximum, rms = _calc_displacement_bohr(stretched, original)
    assert maximum == pytest.approx(0.025)
    assert rms == pytest.approx(0.025)


def test_rigid_alignment_does_not_remove_molecular_reflection():
    original = np.array([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
    ])
    reflected = original * np.array([-1.0, 1.0, 1.0])
    maximum, rms = _calc_displacement_bohr(reflected, original)
    assert maximum > 1.0
    assert rms > 1.0


def test_displacement_without_alignment_keeps_rotation():
    original = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    rotated = original[:, [1, 0, 2]] + np.array([2.0, 3.0, 4.0])
    maximum, rms = _calc_displacement_bohr(rotated, original, align=False)
    assert maximum == pytest.approx(np.sqrt(2.0))
    assert rms == pytest.approx(np.sqrt(2.0))
