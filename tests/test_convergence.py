"""Tests for convergence.py — pure Python."""

from __future__ import annotations

import numpy as np
import pytest

from convergence import (
    ConvergenceState,
    init_convergence_state,
    is_converged,
    update_convergence_state,
    update_xtb_convergence,
)


@pytest.fixture
def pos():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=float,
    )


# ── init_convergence_state ───────────────────────────────────────────────────

def test_init_xtb(pos):
    state = init_convergence_state("xtb", pos)
    assert isinstance(state, ConvergenceState)
    assert state.mode == "xtb"
    assert state.prev_pos is None
    assert state.prev_energy is None
    assert state.converged is False


@pytest.mark.parametrize("mode", ["ff", "both", "xyz"])
def test_init_rejects_unsupported_mode(pos, mode):
    with pytest.raises(ValueError, match="Invalid mode"):
        init_convergence_state(mode, pos)


def test_update_convergence_invalid_mode(pos):
    state = ConvergenceState(mode="xyz")
    with pytest.raises(ValueError):
        update_convergence_state(state, pos, 0.0, np.zeros_like(pos))


def test_is_converged_fresh_state(pos):
    state = init_convergence_state("xtb", pos)
    assert bool(is_converged(state)) is False


# ── xTB convergence ──────────────────────────────────────────────────────────

def test_xtb_first_call_not_converged(pos):
    state = init_convergence_state("xtb", pos)
    result = update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    assert bool(result) is False
    assert bool(is_converged(state)) is False
    assert state.prev_energy == -100.0


def test_xtb_tiny_change_converges(pos):
    state = init_convergence_state("xtb", pos)
    # Step 1: seed
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    # Step 2: tiny change in everything
    new_pos = pos + 1e-12
    tiny_forces = np.ones_like(pos) * 1e-10
    result = update_xtb_convergence(state, new_pos, -100.000001, tiny_forces)
    assert bool(result) is True


def test_xtb_large_gradient_never_converges(pos):
    state = init_convergence_state("xtb", pos)
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    # Big forces → g_max will exceed 3e-4 Eh/Bohr
    big_forces = np.ones_like(pos) * 10.0
    result = update_xtb_convergence(state, pos + 1e-12, -100.0, big_forces)
    assert bool(result) is False


def test_xtb_large_displacement_not_converged(pos):
    state = init_convergence_state("xtb", pos)
    update_xtb_convergence(state, pos, -100.0, np.zeros_like(pos))
    # Tiny forces + tiny energy change, but a big NON-RIGID displacement.
    # GAU convergence measures RMSD-aligned displacement, so a uniform shift
    # would align away — move a single atom instead (survives alignment).
    tiny_forces = np.ones_like(pos) * 1e-10
    far_pos = pos.copy()
    far_pos[0] += 0.1  # one atom by 0.1 nm ≈ 1.89 Bohr >> displacement tol
    result = update_xtb_convergence(state, far_pos, -100.0, tiny_forces)
    assert bool(result) is False


# ── dispatch via update_convergence_state ────────────────────────────────────

def test_dispatch_xtb(pos):
    state = init_convergence_state("xtb", pos)
    update_convergence_state(state, pos, -100.0, np.zeros_like(pos))
    assert state.prev_energy == -100.0


# ── edge case: single-atom "molecule" ───────────────────────────────────────

def test_init_and_update_with_single_atom():
    """Convergence machinery must not break on N=1 (degenerate but valid)."""
    pos = np.array([[0.5, 0.5, 0.5]], dtype=float)

    state_xtb = init_convergence_state("xtb", pos)
    update_xtb_convergence(state_xtb, pos, -1.0, np.zeros_like(pos))
    # Second call: identical geometry → all deltas zero → should converge
    result = update_xtb_convergence(state_xtb, pos, -1.0, np.zeros_like(pos))
    assert bool(result) is True
