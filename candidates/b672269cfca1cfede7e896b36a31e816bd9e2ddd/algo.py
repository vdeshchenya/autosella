"""Self-contained Sella minimiser (order=0, internal coordinates).

Vendored from the `sella` package (2.5.0), restricted to the code path that
`Sella(atoms, internal=True, order=0)` + `irun(fmax=0)` actually executes.
Numerically identical to `molecules/sella_wrapper.py`; no `sella` import needed,
so the module can be cloudpickled by value to the validation workers.

Entry point: minimize_func(positions_nm, atomic_numbers, calc, max_force_calls, converged)
"""

import os as _os

import sys as _sys

import os

import numpy as np

from scipy.linalg import eigh as _cpu_eigh

from scipy.linalg import eigh, lstsq, solve

from typing import List

from itertools import product

from scipy.sparse.linalg import LinearOperator

from typing import (
    Tuple, Callable, Iterator, Union, TypeVar, Optional, List, Dict, Type
)

from itertools import (
    product,
    combinations,
    combinations_with_replacement as cwr
)

from functools import partialmethod

import warnings

from scipy import sparse

from scipy.linalg import svdvals

from ase import Atom, Atoms, units

from ase.cell import Cell

from ase.geometry import complete_cell, minkowski_reduce

from ase.data import covalent_radii

from ase.constraints import (
    FixConstraint, FixAtoms, FixCom, FixBondLengths, FixCartesian, FixInternals
)

import jax.numpy as jnp

from jax import jit, grad, jacfwd, jacrev, vmap, jvp, device_get

import logging

from typing import Union, Callable

from scipy.linalg import eigh, expm, expm_frechet, logm, polar, qr, solve_triangular

from scipy.integrate import LSODA

from ase import Atoms

from ase.build import niggli_reduce

from ase.utils import basestring

from ase.visualize import view

from ase.calculators.singlepoint import SinglePointCalculator

from ase.io.trajectory import Trajectory

from typing import Optional, Tuple, Type, List

from scipy.linalg import eigh

from typing import Optional, Union, List

import inspect

from time import localtime, strftime

from typing import Union, Callable, Optional

from ase.optimize.optimize import Optimizer

from ase.calculators.calculator import Calculator, all_changes

import jax as _jax

_jax.config.update("jax_enable_x64", True)

try:
    _cache = _os.path.expanduser("~/.cache/sella/jax_cache")
    _os.makedirs(_cache, exist_ok=True)
    _jax.config.update("jax_compilation_cache_dir", _cache)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except (AttributeError, ValueError, OSError):
    pass


def rayleigh_ritz(*args, **kwargs):
    raise NotImplementedError(
        'rayleigh_ritz belongs to the order>0 saddle-search path, which this\n'
        'minimisation-only build does not include.')

# (sella/_gpu.py contributed only the CPU fallbacks; the torch offload layer it
#  wrapped is unreachable here and has been collapsed into direct scipy/numpy calls.)












def symmetrize_Y2(S, Y):
    _, nvecs = S.shape
    dY = np.zeros_like(Y)
    YTS = Y.T @ S
    dYTS = np.zeros_like(YTS)
    STS = S.T @ S
    for i in range(1, nvecs):
        RHS = np.linalg.lstsq(STS[:i, :i],
                              YTS[i, :i].T - YTS[:i, i] - dYTS[:i, i],
                              rcond=None)[0]
        dY[:, i] = -S[:, :i] @ RHS
        dYTS[i, :] = -STS[:, :i] @ RHS
    return dY

def symmetrize_Y(S, Y, symm):
    if symm is None or S.shape[1] == 1:
        return Y
    elif symm == 0:
        return Y + S @ lstsq(S.T @ S, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 1:
        return Y + Y @ lstsq(S.T @ Y, np.tril(S.T @ Y - Y.T @ S, -1).T)[0]
    elif symm == 2:
        return Y + symmetrize_Y2(S, Y)
    else:  # pragma: no cover
        raise ValueError("Unknown symmetrization method {}".format(symm))

def update_H(B, S, Y, method='TS-BFGS', symm=2, lams=None, vecs=None):
    """Quasi-Newton update."""
    if len(S.shape) == 1:
        if np.linalg.norm(S) < 1e-8:
            return B
        S = S[:, np.newaxis]
    if len(Y.shape) == 1:
        Y = Y[:, np.newaxis]

    Ytilde = symmetrize_Y(S, Y, symm)

    if B is None:
        # Approximate B as a scaled identity matrix, where the
        # scalar is the average Ritz value from S.T @ Y
        thetas, _ = eigh(S.T @ Ytilde)
        # Guard against zero eigenvalues which would give log(0) = -Inf
        thetas_abs = np.abs(thetas)
        thetas_abs = np.maximum(thetas_abs, 1e-12)
        lam0 = np.exp(np.average(np.log(thetas_abs)))
        d, _ = S.shape
        B = lam0 * np.eye(d)

    if lams is None or vecs is None:
        lams, vecs = eigh(B)

    if method == 'BFGS_auto':
        # Default to TS-BFGS, and only use BFGS if B and S.T @ Y are
        # both positive definite
        method = 'TS-BFGS'
        if lams is not None and np.all(lams > 0):
            lams_STY, vecs_STY = eigh(S.T @ Ytilde, S.T @ S)
            if np.all(lams_STY > 0):
                method = 'BFGS'

    if method == 'BFGS':
        Bplus = _MS_BFGS(B, S, Ytilde)
    elif method == 'TS-BFGS':
        Bplus = _MS_TS_BFGS(B, S, Ytilde, lams, vecs)
    elif method == 'PSB':
        Bplus = _MS_PSB(B, S, Ytilde)
    elif method == 'DFP':
        Bplus = _MS_DFP(B, S, Ytilde)
    elif method == 'SR1':
        Bplus = _MS_SR1(B, S, Ytilde)
    elif method == 'Greenstadt':
        Bplus = _MS_Greenstadt(B, S, Ytilde)
    else:  # pragma: no cover
        raise ValueError('Unknown update method {}'.format(method))

    Bplus += B
    # Symmetrize to clean up floating-point roundoff. The MS_* updates above
    # are mathematically symmetric, so any asymmetry is at machine precision;
    # (B + B.T) / 2 is faster than the tril-based approach and gives the same
    # result up to ~1e-16.
    Bplus = (Bplus + Bplus.T) * 0.5

    return Bplus

def _MS_BFGS(B, S, Y):
    return Y @ solve(Y.T @ S, Y.T) - B @ S @ solve(S.T @ B @ S, S.T @ B)

def _MS_TS_BFGS(B, S, Y, lams, vecs):
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    X2 = S.T @ absBS @ absBS.T
    U = lstsq((X1 + X2) @ S, X1 + X2)[0].T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T

def _MS_PSB(B, S, Y):
    J = Y - B @ S
    U = solve(S.T @ S, S.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T

def _MS_DFP(B, S, Y):
    J = Y - B @ S
    U = solve(S.T @ Y, Y.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T

def _MS_SR1(B, S, Y):
    YBS = Y - B @ S
    return YBS @ solve(YBS.T @ S, YBS.T)

def _MS_Greenstadt(B, S, Y):
    J = Y - B @ S
    MS = B @ S
    U = solve(S.T @ MS, MS.T).T
    UJT = U @ J.T
    return (UJT + UJT.T) - U @ (J.T @ S) @ U.T


class NumericalHessian(LinearOperator):
    dtype = np.dtype('float64')

    def __init__(self, func, x0, g0, eta, threepoint=False, Uproj=None):
        self.func = func
        self.x0 = x0.copy()
        self.g0 = g0.copy()
        self.eta = eta
        self.threepoint = threepoint
        self.calls = 0
        self.Uproj = Uproj

        self.ntrue = len(self.x0)

        if self.Uproj is not None:
            ntrue, n = self.Uproj.shape
            assert ntrue == self.ntrue
        else:
            n = self.ntrue

        super().__init__(self.dtype, (n, n))

        self.Vs = np.empty((self.ntrue, 0), dtype=self.dtype)
        self.AVs = np.empty((self.ntrue, 0), dtype=self.dtype)


    def __add__(self, other):
        return MatrixSum(self, other)


class MatrixSum(LinearOperator):
    def __init__(self, *matrices):
        # This makes sure that if matrices of different dtypes are
        # provided, we use the most general type for the sum.

        # For example, if two matrices are provided with the detypes
        # np.int64 and np.float64, then this MatrixSum object will be
        # np.float64.
        dtype = sorted([mat.dtype for mat in matrices], reverse=True)[0]
        super().__init__(dtype, matrices[0].shape)

        mnum = None
        self.matrices = []
        for matrix in matrices:
            assert matrix.dtype <= self.dtype
            assert matrix.shape == self.shape, (matrix.shape, self.shape)
            if isinstance(matrix, np.ndarray):
                if mnum is None:
                    mnum = np.zeros(self.shape, dtype=self.dtype)
                mnum += matrix
            else:
                self.matrices.append(matrix)

        if mnum is not None:
            self.matrices.append(mnum)



    def __add__(self, other):
        return MatrixSum(*self.matrices, other)

class ApproximateHessian(LinearOperator):
    def __init__(
        self,
        dim: int,
        ncart: int,
        B0: np.ndarray = None,
        update_method: str = 'TS-BFGS',
        symm: int = 2,
        initialized: bool = False,
    ) -> None:
        """A wrapper object for the approximate Hessian matrix."""
        self.dim = dim
        self.ncart = ncart
        super().__init__(np.float64, (dim, dim))
        self.update_method = update_method
        self.symm = symm
        self.initialized = initialized
        # Lazy eigendecomposition: only compute when needed
        self._evals = None
        self._evecs = None
        self._eigen_computed = False
        self.set_B(B0)

    def _ensure_eigen_computed(self):
        """Compute the eigendecomposition of B once, on first access."""
        if self._eigen_computed or self.B is None:
            return
        self._evals, self._evecs = _cpu_eigh(self.B)
        self._eigen_computed = True

    @property
    def evals(self):
        """Lazily compute eigenvalues on first access."""
        self._ensure_eigen_computed()
        return self._evals

    @evals.setter
    def evals(self, value):
        self._evals = value
        if value is None:
            self._eigen_computed = False

    @property
    def evecs(self):
        """Lazily compute eigenvectors on first access."""
        self._ensure_eigen_computed()
        return self._evecs

    @evecs.setter
    def evecs(self, value):
        self._evecs = value
        if value is None:
            self._eigen_computed = False

    def set_B(self, target):
        if target is None:
            self.B = None
            self._evals = None
            self._evecs = None
            self._eigen_computed = False
            self.initialized = False
            return
        elif np.isscalar(target):
            target = target * np.eye(self.dim)
        else:
            self.initialized = True
        assert target.shape == self.shape
        self.B = target
        # Mark eigendecomposition as stale - will recompute on next access
        self._eigen_computed = False

    def update(self, dx, dg):
        """Perform a quasi-Newton update on B"""
        if self.B is None:
            B = np.zeros(self.shape, dtype=self.dtype)
        else:
            B = self.B.copy()
        if not self.initialized:
            self.initialized = True
            dx_cart = dx[:self.ncart]
            dg_cart = dg[:self.ncart]
            B[:self.ncart, :self.ncart] = update_H(
                None, dx_cart, dg_cart, method=self.update_method,
                symm=self.symm, lams=None, vecs=None
            )
            self.set_B(B)
            return

        lams, vecs = self.evals, self.evecs
        self.set_B(update_H(B, dx, dg, method=self.update_method,
                            symm=self.symm, lams=lams, vecs=vecs))

    def project(self, U):
        """Project B into the subspace defined by U."""
        m, n = U.shape
        assert m == self.dim

        if self.B is None:
            Bproj = None
        else:
            Bproj = U.T @ self.B @ U

        return ApproximateHessian(n, 0, Bproj, self.update_method,
                                  self.symm)

    def asarray(self):
        if self.B is not None:
            return self.B
        return np.eye(self.dim)

    def _matvec(self, v):
        if self.B is None:
            return v
        return self.B @ v




    def __add__(self, other):
        initialized = self.initialized
        if isinstance(other, ApproximateHessian):
            initialized = initialized and other.initialized
            other = other.B
        if not self.initialized or other is None:
            tot = None
            initialized = False
        else:
            tot = self.B + other
        return ApproximateHessian(
            self.dim, self.ncart, tot, self.update_method, self.symm,
            initialized=initialized,
        )

class SparseInternalHessian(LinearOperator):
    dtype = np.float64

    def __init__(
        self,
        natoms: int,
        indices: List[int],
        vals: np.ndarray,
    ) -> None:
        self.natoms = natoms
        super().__init__(self.dtype, (3 * natoms, 3 * natoms))
        self.indices = np.asarray(indices)
        self.vals = np.asarray(vals)

    def asarray(self) -> np.ndarray:
        H = np.zeros((self.natoms, self.natoms, 3, 3))
        idx = self.indices
        n = len(idx)
        if n == 0:
            return H.transpose(0, 2, 1, 3).reshape(self.shape)

        # Create meshgrid of all (a, b) pairs and compute linear indices
        idx_a, idx_b = np.meshgrid(idx, idx, indexing='ij')
        linear_idx = idx_a * self.natoms + idx_b  # (n, n) linear indices

        # H is (natoms, natoms, 3, 3) so H_flat[a*natoms+b] = H[a, b, :, :]
        H_flat = H.reshape(self.natoms * self.natoms, 3, 3)
        # vals has shape (n, 3, n, 3) - transpose to (n, n, 3, 3) before reshaping
        vals_flat = self.vals.transpose(0, 2, 1, 3).reshape(n * n, 3, 3)

        # Vectorized accumulation
        np.add.at(H_flat, linear_idx.ravel(), vals_flat)

        # Transpose back to (natoms, 3, natoms, 3) and reshape
        return H.transpose(0, 2, 1, 3).reshape(self.shape)



class SparseInternalHessiansSkeleton:
    """Index-only data for SparseInternalHessians.

    Holds the per-size groupings, atom-index arrays, and pre-computed flat
    indices used by ldot/rdot. These derive only from the per-coord
    ``indices`` and the global ``natoms`` — neither depends on the
    coordinate Hessian *values* or atomic positions, so the skeleton can
    be cached across optimizer steps and reused as long as the active
    set of internal coordinates is unchanged.
    """

    def __init__(self, hessians: List[SparseInternalHessian], natoms: int):
        self.natoms = natoms
        self.n_hess = len(hessians)

        # Group hessians by size (number of atoms involved).
        by_size = {}
        for i, h in enumerate(hessians):
            n = len(h.indices)
            if n not in by_size:
                by_size[n] = {'orig_idx': [], 'indices': []}
            by_size[n]['orig_idx'].append(i)
            by_size[n]['indices'].append(h.indices)

        i_idx, j_idx = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
        i_flat = i_idx.ravel()
        j_flat = j_idx.ravel()

        # rdot only needs orig_idx + per-coord atom indices; vals are
        # filled in per call by SparseInternalHessians.
        self.rdot_meta = {}
        # ldot also needs the precomputed flat 1D scatter index.
        self.ldot_meta = {}

        for size, data in by_size.items():
            orig_idx = np.array(data['orig_idx'])
            indices = np.array(data['indices'])  # (batch, size)
            batch = len(orig_idx)

            self.rdot_meta[size] = {
                'orig_idx': orig_idx,
                'indices': indices,
            }

            n_pairs = size * size
            a_local, b_local = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
            a_local = a_local.ravel()
            b_local = b_local.ravel()

            row_atoms = indices[:, a_local]  # (batch, size*size)
            col_atoms = indices[:, b_local]
            row_atoms = np.repeat(row_atoms, 9, axis=1)  # (batch, size*size*9)
            col_atoms = np.repeat(col_atoms, 9, axis=1)
            i_full = np.tile(i_flat, (batch, n_pairs))
            j_full = np.tile(j_flat, (batch, n_pairs))

            # Pre-compute the flat 1D index used by the bincount-based ldot.
            # M is (natoms, 3, natoms, 3); flat index =
            # ((row*3 + i)*natoms + col)*3 + j.
            linear_idx = ((row_atoms.ravel() * 3 + i_full.ravel())
                          * natoms + col_atoms.ravel()) * 3 + j_full.ravel()

            self.ldot_meta[size] = {
                'orig_idx': orig_idx,
                'linear_idx': linear_idx,
                'batch': batch,
                'n_pairs': n_pairs,
            }

class SparseInternalHessians:
    def __init__(
        self,
        hessians: List[SparseInternalHessian],
        ndof: int,
        skeleton: 'SparseInternalHessiansSkeleton' = None,
    ):
        self.hessians = hessians
        self.natoms = ndof // 3
        self.shape = (len(self.hessians), ndof, ndof)

        if skeleton is None:
            skeleton = SparseInternalHessiansSkeleton(hessians, self.natoms)
        elif skeleton.n_hess != len(hessians) or skeleton.natoms != self.natoms:
            raise ValueError(
                "skeleton was built for a different (n_hess, natoms); "
                f"got skeleton ({skeleton.n_hess}, {skeleton.natoms}) vs "
                f"this ({len(hessians)}, {self.natoms})"
            )
        self._skeleton = skeleton
        self._build_value_views()

    def _build_value_views(self):
        """Stitch fresh per-coord vals onto the cached skeleton.

        ``vals_flat`` is rebuilt on every call (it tracks atomic positions
        via the per-coord Hessian values), but the index arrays are reused
        from the skeleton.
        """
        hessians = self.hessians
        self._batched_rdot = {}
        self._batched_ldot = {}
        for size, meta in self._skeleton.rdot_meta.items():
            orig_idx = meta['orig_idx']
            # Stack the current Hessian values for this size group.
            vals = np.array([hessians[i].vals for i in orig_idx])
            self._batched_rdot[size] = {
                'orig_idx': orig_idx,
                'indices': meta['indices'],
                'vals': vals,
            }
            ldot_meta = self._skeleton.ldot_meta[size]
            # Reorder vals: (batch, size, 3, size, 3) -> (batch, size*size*9)
            vals_reordered = vals.transpose(0, 1, 3, 2, 4)
            vals_flat = vals_reordered.reshape(ldot_meta['batch'], -1)
            self._batched_ldot[size] = {
                'orig_idx': orig_idx,
                'vals_flat': vals_flat,
                'linear_idx': ldot_meta['linear_idx'],
            }

    def asarray(self) -> np.ndarray:
        return np.array([hess.asarray() for hess in self.hessians])


    def ldot(self, v: np.ndarray) -> np.ndarray:
        """Vectorized left dot: v^T @ D -> (ndof, ndof) matrix.

        Uses np.bincount on a precomputed flat 1D index instead of np.add.at
        on a 4D index. bincount handles duplicate indices in vectorized C
        code, while np.add.at falls back to a Python-level loop for repeats.
        On NACJAF (120 atoms, 72 constraints) this is ~2.5× faster.
        """
        n_dof = self.natoms * 3
        M_flat = np.zeros(n_dof * n_dof)

        for size, data in self._batched_ldot.items():
            weights = v[data['orig_idx']]
            weighted = (data['vals_flat'] * weights[:, None]).ravel()
            M_flat += np.bincount(data['linear_idx'], weights=weighted,
                                  minlength=n_dof * n_dof)

        return M_flat.reshape((n_dof, n_dof))



class LightAtoms:
    """Lightweight wrapper providing positions and cell without Atoms overhead."""
    __slots__ = ('positions', 'cell')

    def __init__(self, positions: np.ndarray, cell: np.ndarray) -> None:
        self.positions = positions
        self.cell = cell

def _bond_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Bond length: pos shape (2, 3), tvec shape (1, 3)"""
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])

def _angle_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Angle value: pos shape (3, 3), tvec shape (2, 3)"""
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)

def _dihedral_value(pos: jnp.ndarray, tvec: jnp.ndarray) -> float:
    """Dihedral angle: pos shape (4, 3), tvec shape (3, 3)"""
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)

_bond_grad_batched = jit(vmap(grad(_bond_value, argnums=0), in_axes=(0, 0)))

_angle_grad_batched = jit(vmap(grad(_angle_value, argnums=0), in_axes=(0, 0)))

_dihedral_grad_batched = jit(vmap(grad(_dihedral_value, argnums=0), in_axes=(0, 0)))

_bond_value_batched = jit(vmap(_bond_value, in_axes=(0, 0)))

_angle_value_batched = jit(vmap(_angle_value, in_axes=(0, 0)))

_dihedral_value_batched = jit(vmap(_dihedral_value, in_axes=(0, 0)))

_bond_hess_batched = jit(vmap(jacfwd(grad(_bond_value, argnums=0), argnums=0), in_axes=(0, 0)))

_angle_hess_batched = jit(vmap(jacfwd(grad(_angle_value, argnums=0), argnums=0), in_axes=(0, 0)))

_dihedral_hess_batched = jit(vmap(jacfwd(grad(_dihedral_value, argnums=0), argnums=0), in_axes=(0, 0)))

def _bond_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single bond without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_bond_value, argnums=0), primals, tangents)
    return hvp_result

def _angle_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single angle without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_angle_value, argnums=0), primals, tangents)
    return hvp_result

def _dihedral_hvp_single(pos: jnp.ndarray, tvec: jnp.ndarray, tangent: jnp.ndarray) -> jnp.ndarray:
    """Compute Hessian @ tangent for a single dihedral without forming the Hessian."""
    primals = (pos, tvec)
    tangents = (tangent, jnp.zeros_like(tvec))
    _, hvp_result = jvp(grad(_dihedral_value, argnums=0), primals, tangents)
    return hvp_result

_bond_hvp_batched = jit(vmap(_bond_hvp_single, in_axes=(0, 0, 0)))

_angle_hvp_batched = jit(vmap(_angle_hvp_single, in_axes=(0, 0, 0)))

_dihedral_hvp_batched = jit(vmap(_dihedral_hvp_single, in_axes=(0, 0, 0)))

def _bond_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Bond length with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (1, 3) @ (3, 3) -> (1, 3)
    return jnp.linalg.norm(pos[1] - pos[0] + tvec[0])

def _angle_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (2, 3) @ (3, 3) -> (2, 3)
    dx1 = -(pos[1] - pos[0] + tvec[0])
    dx2 = pos[2] - pos[1] + tvec[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)

def _dihedral_with_cell(pos: jnp.ndarray, ncvec: jnp.ndarray, cell: jnp.ndarray) -> float:
    """Dihedral angle with cell as explicit parameter for autodiff."""
    tvec = ncvec @ cell  # (3, 3) @ (3, 3) -> (3, 3)
    dx1 = pos[1] - pos[0] + tvec[0]
    dx2 = pos[2] - pos[1] + tvec[1]
    dx3 = pos[3] - pos[2] + tvec[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)

_bond_cell_grad_single = jit(grad(_bond_with_cell, argnums=2))

_angle_cell_grad_single = jit(grad(_angle_with_cell, argnums=2))

_dihedral_cell_grad_single = jit(grad(_dihedral_with_cell, argnums=2))

_bond_cell_grad_batched = jit(vmap(_bond_cell_grad_single, in_axes=(0, 0, None)))

_angle_cell_grad_batched = jit(vmap(_angle_cell_grad_single, in_axes=(0, 0, None)))

_dihedral_cell_grad_batched = jit(vmap(_dihedral_cell_grad_single, in_axes=(0, 0, None)))

BLOCK_SIZE = 64

IVec = Tuple[int, int, int]

class NoValidInternalError(ValueError):
    pass

class DuplicateInternalError(ValueError):
    pass

class DuplicateConstraintError(DuplicateInternalError):
    pass

def _gradient(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(grad(func, argnums=0))

def _hessian(
    func: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    return jit(jacfwd(jacrev(func, argnums=0), argnums=0))

class Coordinate:
    nindices = None
    kwargs = None

    def __init__(
        self,
        indices: Tuple[int, ...],
    ) -> None:
        if self.nindices is not None:
            assert len(indices) == self.nindices
        self.indices = np.array(indices, dtype=np.int32)
        self.kwargs = dict()

    def reverse(self) -> 'Coordinate':
        raise NotImplementedError

    def __eq__(self, other: 'Coordinate') -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if len(self.indices) != len(other.indices):
            return False
        if np.all(self.indices == other.indices):
            return True
        return False

    def __add__(self, other: 'Coordinate') -> 'Coordinate':
        raise NotImplementedError

    def split(self) -> Tuple['Coordinate', 'Coordinate']:
        raise NotImplementedError

    def __repr__(self) -> str:
        out = [f'indices={self.indices}']
        out += [f'{key}={val}' for key, val in self.kwargs.items()]
        str_out = ', '.join(out)
        return f'{self.__class__.__name__}({str_out})'

    @staticmethod
    def _eval0(pos: jnp.ndarray, **kwargs) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(pos: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        return float(self._eval0(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        return np.array(self._eval1(
            atoms.positions[self.indices], **self.kwargs
        ))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        return np.array(self._eval2(
            atoms.positions[self.indices], **self.kwargs
        ))




class Internal(Coordinate):
    union = None
    diff = None

    def __init__(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None
    ) -> None:
        Coordinate.__init__(self, indices)

        if self.nindices is not None:
            if ncvecs is None:
                ncvecs = np.zeros((self.nindices - 1, 3), dtype=np.int32)
            else:
                ncvecs = np.asarray(ncvecs).reshape((self.nindices - 1, 3))
        else:
            if ncvecs is not None:
                raise ValueError(
                    "{} does not support ncvecs"
                    .format(self.__class__.__name__)
                )
            ncvecs = np.empty((0, 3), dtype=np.int32)
        self.kwargs['ncvecs'] = ncvecs

    def reverse(self) -> 'Internal':
        return self.__class__(self.indices[::-1], -self.kwargs['ncvecs'][::-1])

    def __eq__(self, other: object) -> bool:
        if not Coordinate.__eq__(self, other):
            return False
        srev = self.reverse()
        if not Coordinate.__eq__(srev, other):
            return False
        if np.all(self.kwargs['ncvecs'] == other.kwargs['ncvecs']):
            return True
        if np.all(srev.kwargs['ncvecs'] == other.kwargs['ncvecs']):
            return True
        return False

    def __add__(self, other: object) -> 'Internal':
        if self.union is None:
            return NotImplemented
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self == other:
            raise NoValidInternalError(
                'Cannot add {} object to itself.'
                .format(self.__class__.__name__)
            )

        for s, o in product([self, self.reverse()], [other, other.reverse()]):
            if (
                np.all(s.indices[1:] == o.indices[:-1])
                and np.all(s.kwargs['ncvecs'][1:] == o.kwargs['ncvecs'][:-1])
            ):
                new_indices = [*s.indices, o.indices[-1]]
                new_ncvecs = [*s.kwargs['ncvecs'], o.kwargs['ncvecs'][-1]]
                return self.union(new_indices, new_ncvecs)
        raise NoValidInternalError(
            '{} indices do not overlap!'.format(self.__class__.__name__)
        )

    def split(self) -> Tuple['Internal', 'Internal']:
        if self.diff is None:
            raise RuntimeError(
                "Don't know how to split a {}!".format(self.__class__.__name__)
            )
        return (
            self.diff(self.indices[:-1], self.kwargs['ncvecs'][:-1]),
            self.diff(self.indices[1:], self.kwargs['ncvecs'][1:])
        )

    @staticmethod
    def _eval0(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def _eval1(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    @staticmethod
    def _eval2(
        pos: jnp.ndarray, tvecs: jnp.ndarray
    ) -> jnp.ndarray:
        raise NotImplementedError

    def calc(self, atoms: Atoms) -> float:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return float(self._eval0(atoms.positions[self.indices], tvecs))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval1(atoms.positions[self.indices], tvecs))

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        tvecs = jnp.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        return np.array(self._eval2(atoms.positions[self.indices], tvecs))

    @staticmethod
    def _eval_cell_grad(
        pos: jnp.ndarray, ncvecs: jnp.ndarray, cell: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute gradient of coordinate with respect to cell matrix.

        Must be overridden in subclasses (Bond, Angle, Dihedral).
        Returns shape (3, 3) for d(coord)/d(cell).
        """
        raise NotImplementedError


def _translation(
    pos: jnp.ndarray,
    dim: int,
) -> float:
    return pos[:, dim].mean()

class Translation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        dim: int,
    ) -> None:
        Coordinate.__init__(self, indices)
        self.kwargs['dim'] = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['dim'] != other.kwargs['dim']:
            return False
        if set(self.indices) != set(other.indices):
            return False
        return True

    _eval0 = staticmethod(jit(_translation))
    _eval1 = staticmethod(_gradient(_translation))
    _eval2 = staticmethod(_hessian(_translation))

def _rotation_hessian_np(pos, axis, refpos, q_stable=None):
    """Closed-form Hessian of the rotation coordinate w.r.t. positions.

    Uses an analytic eigenvector second derivative that handles degenerate
    eigenvalues (linear molecules) via the Moore-Penrose pseudoinverse,
    avoiding the NaN that JAX autodiff produces in that case.

    Parameters
    ----------
    pos : ndarray (N, 3)
    axis : int (0, 1, or 2)
    refpos : ndarray (N, 3), already centered
    q_stable : ndarray (4,), optional stabilized quaternion

    Returns
    -------
    hessian : ndarray (N, 3, N, 3)
    """
    return _rotation_hessian_single(
        np.asarray(pos, dtype=np.float64),
        axis,
        np.asarray(refpos, dtype=np.float64),
        q_stable=q_stable,
    )

def _build_F_matrix_np(dx, refpos):
    """Build the 4x4 quaternion F-matrix in numpy.

    Parameters
    ----------
    dx : ndarray (N, 3), centered positions (pos - centroid)
    refpos : ndarray (N, 3), centered reference positions
    """
    R = dx.T @ refpos
    Rtr = np.trace(R)
    Ftop = np.array([R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0]])
    F = np.empty((4, 4))
    F[0, 0] = Rtr
    F[0, 1:] = Ftop
    F[1:, 0] = Ftop
    F[1:, 1:] = -Rtr * np.eye(3) + R + R.T
    return F

def _stabilize_quaternion(F, q_prev):
    """Compute branch-stable quaternion from F-matrix eigendecomposition.

    Projects q_prev onto the top eigenspace of F and normalizes.
    For non-degenerate cases (1D top eigenspace), this is equivalent
    to picking the rightmost eigenvector with consistent sign.
    For degenerate cases (2-atom / linear fragments with 2D+ top
    eigenspace), this picks the linear combination closest to q_prev,
    ensuring continuity across geometry steps.
    """
    ws, vecs = np.linalg.eigh(F)
    return _stabilize_quaternion_from_eigh(ws, vecs, q_prev)

def _stabilize_quaternion_from_eigh(ws, vecs, q_prev):
    """Compute branch-stable quaternion from pre-computed eigendecomposition."""
    if q_prev is None:
        q_prev = np.array([1.0, 0.0, 0.0, 0.0])
    top_mask = (ws[-1] - ws) < 1e-10
    top_vecs = vecs[:, top_mask]
    coeffs = top_vecs.T @ q_prev
    q = top_vecs @ coeffs
    norm = np.linalg.norm(q)
    if norm < 1e-14:
        q = vecs[:, -1].copy()
    else:
        q /= norm
    if q[0] < 0:
        q = -q
    return q

def _asinc_np(x):
    """Inverse sinc: arccos(x) / sqrt(1 - x^2), with Taylor branch near x=1."""
    if x < 0.97:
        return np.arccos(x) / np.sqrt(1.0 - x * x)
    y = x - 1.0
    return (1.0 - y / 3 + 2 * y**2 / 15 - 2 * y**3 / 35
            + 8 * y**4 / 315 - 8 * y**5 / 693 + 16 * y**6 / 3003
            - 16 * y**7 / 6435 + 128 * y**8 / 109395
            - 128 * y**9 / 230945)

def _expmap_np(q):
    """Convert unit quaternion to rotation vector (3,)."""
    a = _asinc_np(q[0])
    return 2.0 * q[1:4] * a

def _rotation_3axis_jacobian_np(pos, refpos, q):
    """Jacobian of all 3 rotation values w.r.t. positions, using quaternion q.

    Parameters
    ----------
    pos    : (N, 3)
    refpos : (N, 3), already centered
    q      : (4,), stabilized quaternion

    Returns
    -------
    jac : (3, N, 3) — Jacobian[axis, atom, xyz]
    """
    N = len(pos)
    dx = pos - pos.mean(0)
    F = _build_F_matrix_np(dx, refpos)
    ws, vecs = np.linalg.eigh(F)

    c = q
    gaps = ws - ws[-1]
    safe_inv = np.where(np.abs(gaps) > 1e-14,
                        1.0 / np.where(np.abs(gaps) > 1e-14, gaps, 1.0),
                        0.0)

    Prefpos = refpos  # refpos is already centered at construction
    dFc = _apply_dF(Prefpos, c, N)  # (N, 3, 4)
    dFc_flat = dFc.reshape(N * 3, 4)
    dc_flat = -(vecs @ (safe_inv[:, None] * (vecs.T @ dFc_flat.T))).T  # (N*3, 4)

    q0 = c[0]
    asinc_val = _asinc_np(q0)
    if abs(q0 - 1.0) < 1e-8:
        y = q0 - 1.0
        dasinc = -1.0 / 3 + 4 * y / 15
    elif abs(q0) < 1.0 - 1e-12:
        s2 = 1 - q0**2
        s = np.sqrt(s2)
        ac = np.arccos(q0)
        dasinc = -1.0 / s2 + q0 * ac / (s * s2)
    else:
        dasinc = 0.0

    jac = np.zeros((3, N, 3))
    for k in range(3):
        a = k + 1
        jac_flat = 2 * (dc_flat[:, a] * asinc_val + c[a] * dasinc * dc_flat[:, 0])
        jac[k] = jac_flat.reshape(N, 3)
    return jac

def _apply_dF(Prefpos, vec, N):
    """Compute dF_{k,d} @ vec for all (k,d), batched over fragments.

    Prefpos : (B, N, 3) or (N, 3)
    vec     : (B, 4) or (4,)

    Returns : (B, N, 3, 4) or (N, 3, 4)
    """
    single = Prefpos.ndim == 2
    if single:
        Prefpos = Prefpos[None]
        vec = vec[None]
    B = Prefpos.shape[0]

    v0 = vec[:, 0]        # (B,)
    v3 = vec[:, 1:]       # (B, 3)
    Pv3 = np.einsum('bni,bi->bn', Prefpos, v3)  # (B, N) = Prefpos @ v3

    result = np.zeros((B, N, 3, 4))
    for d in range(3):
        dRtr = Prefpos[:, :, d]  # (B, N)
        # dFtop for this d
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        # dR[d1,d2]-dR[d2,d1] etc., with dR[i,j]=Prefpos[k,j]*delta_{i,d}
        dFtop = np.zeros((B, N, 3))
        # Antisymmetric part of dR: dR[i,j]-dR[j,i]
        # Only nonzero entries: dR[d,j] = Pref[k,j], dR[j,d] = 0 for j!=d
        # So: component 0 = dR[1,2]-dR[2,1]:
        #   if d==1: Pref[k,2]; if d==2: -Pref[k,1]; else 0
        # Simpler pattern: cross-product-like
        dFtop[:, :, d1] = -Prefpos[:, :, d2]
        dFtop[:, :, d2] = Prefpos[:, :, d1]
        # dFtop[d] = 0 (already)

        # result[:, :, d, 0] = dRtr * v0 + dFtop @ v3
        result[:, :, d, 0] = dRtr * v0 + np.einsum('bni,bi->bn', dFtop, v3)

        # result[:, :, d, 1:] = dFtop * v0 + (-dRtr*I + dR + dR.T) @ v3
        for i_ax in range(3):
            val = -dRtr * v3[:, i_ax, None]  # (B, N) — broadcast
            val = val.squeeze(-1) if val.ndim > 2 else val
            # Correction: val shape should be (B, N)
            val = -dRtr * v3[:, i_ax:i_ax+1]  # (B, 1) broadcast with (B, N) -> (B, N)
            if i_ax == d:
                val = val + Pv3  # (B, N)
            val = val + Prefpos[:, :, i_ax] * v3[:, d:d+1]  # (B, N)
            result[:, :, d, 1 + i_ax] = dFtop[:, :, i_ax] * v0[:, None] + val

    if single:
        return result[0]
    return result

def _rotation_hessian_single(pos, axis, refpos, q_stable=None):
    """Closed-form Hessian for a single rotation on a single fragment.

    pos    : (N, 3)
    axis   : int
    refpos : (N, 3), already centered
    q_stable : (4,), optional stabilized quaternion

    Returns (N, 3, N, 3)
    """
    N = len(pos)
    a = axis + 1

    # F-matrix
    dx = pos - pos.mean(0)
    F = _build_F_matrix_np(dx, refpos)

    # Eigendecomposition + safe pseudoinverse
    ws, vecs = np.linalg.eigh(F)
    if q_stable is not None:
        c = q_stable
    else:
        c = vecs[:, -1]
        if c[0] < 0:
            c = -c
    gaps = ws - ws[-1]
    safe_inv = np.where(np.abs(gaps) > 1e-14, 1.0 / np.where(np.abs(gaps) > 1e-14, gaps, 1.0), 0.0)

    def M_inv_mat(mat):
        return vecs @ (safe_inv[:, None] * (vecs.T @ mat))

    # Prefpos and dFc
    P = np.eye(N) - 1.0 / N
    Prefpos = P @ refpos  # (N, 3)
    dFc = _apply_dF(Prefpos, c, N)  # (N, 3, 4)
    dFc_flat = dFc.reshape(N * 3, 4)

    dE_flat = dFc_flat @ c  # (N*3,)
    dc_flat = -M_inv_mat(dFc_flat.T).T  # (N*3, 4)

    # asinc derivatives
    q0 = c[0]
    qa = c[a]
    if abs(q0 - 1.0) < 1e-8:
        y = q0 - 1.0
        asinc_val = 1 - y / 3 + 2 * y**2 / 15
        dasinc = -1.0 / 3 + 4 * y / 15
        d2asinc = 4.0 / 15
    elif abs(q0) < 1.0 - 1e-12:
        s2 = 1 - q0**2
        s = np.sqrt(s2)
        ac = np.arccos(q0)
        asinc_val = ac / s
        dasinc = -1.0 / s2 + q0 * ac / (s * s2)
        d2asinc = (3 * q0 / s2 - (1 + 2 * q0**2) * ac / (s * s2)) * (-1.0 / s2)
    else:
        asinc_val = np.pi / 2 if q0 > 0 else -np.pi / 2
        dasinc = 0.0
        d2asinc = 0.0

    df_dq = np.zeros(4)
    df_dq[0] = 2 * qa * dasinc
    df_dq[a] = 2 * asinc_val

    d2f_dq2 = np.zeros((4, 4))
    d2f_dq2[0, 0] = 2 * qa * d2asinc
    d2f_dq2[0, a] = 2 * dasinc
    d2f_dq2[a, 0] = 2 * dasinc

    # Term 1: quadratic in first derivatives
    hess_flat = dc_flat @ d2f_dq2 @ dc_flat.T

    # Term 2: df_dq contracted with d2c
    w = vecs @ (safe_inv * (vecs.T @ df_dq))
    wc = w @ c
    w_dc = dc_flat @ w
    fdq_c = df_dq @ c

    dFw = _apply_dF(Prefpos, w, N)
    dFw_flat = dFw.reshape(N * 3, 4)
    wdFdc = dFw_flat @ dc_flat.T

    d2E_mat = 2 * dFc_flat @ dc_flat.T
    dc_dot = dc_flat @ dc_flat.T

    term2 = (dE_flat[:, None] * w_dc[None, :]
             + dE_flat[None, :] * w_dc[:, None]
             + d2E_mat * wc
             - wdFdc - wdFdc.T
             - fdq_c * dc_dot)

    hess_flat += term2
    return hess_flat.reshape(N, 3, N, 3)

def _rotation_hvp_closed(pos, axis, refpos, tangent, q_stable=None):
    """HVP for a single rotation using the closed-form Hessian."""
    hess = _rotation_hessian_single(pos, axis, refpos, q_stable=q_stable)
    return np.einsum('aibj,bj->ai', hess, tangent)

def _build_dF_vec_batched(Pref, vec, n_batch, nr):
    """Compute dF_{k,d} @ vec for all (k,d) in a batched fragment group.

    Parameters
    ----------
    Pref : (n_batch, nr, 3), centered reference positions
    vec  : (n_batch, 4), quaternion-space vector

    Returns
    -------
    dF_vec : (n_batch, nr*3, 4)
    """
    v0 = vec[:, 0:1]       # (n_batch, 1)
    v3 = vec[:, 1:]         # (n_batch, 3)
    Pv3 = np.squeeze(Pref @ v3[:, :, None], -1)  # (n_batch, nr)

    result = np.empty((n_batch, nr, 3, 4))
    for d in range(3):
        d1 = (d + 1) % 3
        d2 = (d + 2) % 3
        dRtr = Pref[:, :, d]  # (n_batch, nr)

        dFtop_d1 = -Pref[:, :, d2]  # (n_batch, nr)
        dFtop_d2 = Pref[:, :, d1]   # (n_batch, nr)

        result[:, :, d, 0] = (dRtr * v0
                              + dFtop_d1 * v3[:, d1:d1+1]
                              + dFtop_d2 * v3[:, d2:d2+1])

        vd = v3[:, d:d+1]  # (n_batch, 1)
        for i_ax in range(3):
            val = -dRtr * v3[:, i_ax:i_ax+1]
            if i_ax == d:
                val = val + Pv3
            val = val + Pref[:, :, i_ax] * vd
            if i_ax == d1:
                dFtop_iax = dFtop_d1
            elif i_ax == d2:
                dFtop_iax = dFtop_d2
            else:
                dFtop_iax = 0.0
            result[:, :, d, 1 + i_ax] = dFtop_iax * v0 + val

    return result.reshape(n_batch, nr * 3, 4)

def _rotation_3axis_hvp_batched_closed(pos_pad, ref_pad, mask, v_pad,
                                       q_stable_all=None,
                                       ws_all=None, vecs_all=None):
    """Batched HVP for multiple fragments using closed-form Hessians.

    Parameters
    ----------
    pos_pad : (B, N_max, 3)
    ref_pad : (B, N_max, 3)
    mask : (B, N_max)
    v_pad : (B, N_max, 3)
    q_stable_all : (B, 4), optional stabilized quaternions per fragment
    ws_all : (B, 4), optional cached eigenvalues per fragment
    vecs_all : (B, 4, 4), optional cached eigenvectors per fragment

    Returns
    -------
    hvp : (B, 3, N_max, 3)
    """
    B, N_max, _ = pos_pad.shape
    n_real = np.sum(mask, axis=1).astype(int)
    hvp = np.zeros((B, 3, N_max, 3))

    size_groups = {}
    for fi in range(B):
        nr = n_real[fi]
        size_groups.setdefault(nr, []).append(fi)

    for nr, frag_indices in size_groups.items():
        n_batch = len(frag_indices)
        idx = np.array(frag_indices)

        pos_group = pos_pad[idx, :nr]    # (n_batch, nr, 3)
        ref_group = ref_pad[idx, :nr]    # (n_batch, nr, 3)
        v_group = v_pad[idx, :nr]        # (n_batch, nr, 3)

        if ws_all is not None and vecs_all is not None:
            ws = ws_all[idx]
            vecs = vecs_all[idx]
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]
        else:
            dx = pos_group - pos_group.mean(axis=1, keepdims=True)
            R = np.matmul(dx.swapaxes(1, 2), ref_group)  # (n_batch, 3, 3)
            Rtr = np.trace(R, axis1=1, axis2=2)
            Ftop = np.stack([
                R[:, 1, 2] - R[:, 2, 1],
                R[:, 2, 0] - R[:, 0, 2],
                R[:, 0, 1] - R[:, 1, 0],
            ], axis=1)
            F = np.zeros((n_batch, 4, 4))
            F[:, 0, 0] = Rtr
            F[:, 0, 1:] = Ftop
            F[:, 1:, 0] = Ftop
            for i in range(3):
                F[:, 1+i, 1+i] = -Rtr
            F[:, 1:, 1:] += R + R.transpose(0, 2, 1)
            ws, vecs = np.linalg.eigh(F)
            if q_stable_all is not None:
                c = q_stable_all[idx]
            else:
                c = vecs[:, :, -1]
                sign = np.where(c[:, 0] >= 0, 1.0, -1.0)
                c *= sign[:, None]

        gaps = ws - ws[:, -1:]
        safe_inv = np.where(
            np.abs(gaps) > 1e-14,
            1.0 / np.where(np.abs(gaps) > 1e-14, gaps, 1.0),
            0.0,
        )

        # refpos is already centered at construction
        Pref = ref_group

        # dFc: dF @ c for all (k,d)
        dFc_flat = _build_dF_vec_batched(Pref, c, n_batch, nr)  # (n_batch, M, 4)
        M = nr * 3

        dE_flat = np.squeeze(dFc_flat @ c[:, :, None], -1)  # (n_batch, M)
        # dc_flat = -vecs @ (safe_inv * (vecs^T @ dFc_flat^T))^T
        proj = np.matmul(dFc_flat, vecs)  # (n_batch, M, 4)
        dc_flat = -np.matmul(proj * safe_inv[:, None, :], vecs.swapaxes(1, 2))  # (n_batch, M, 4)

        # Axis-independent computations (hoisted from axis loop)
        v_flat = v_group.reshape(n_batch, M)
        dc_v = np.squeeze(dc_flat.swapaxes(1, 2) @ v_flat[:, :, None], -1)  # (n_batch, 4)
        dE_v = (dE_flat * v_flat).sum(axis=1)  # (n_batch,)
        d2E_v = 2 * np.squeeze(dFc_flat @ dc_v[:, :, None], -1)  # (n_batch, M)
        dc_dot_v = np.squeeze(dc_flat @ dc_v[:, :, None], -1)  # (n_batch, M)

        q0 = c[:, 0]
        s2 = np.maximum(1 - q0**2, 1e-30)
        s = np.sqrt(s2)
        ac = np.arccos(np.clip(q0, -1+1e-15, 1-1e-15))
        near_one = np.abs(q0 - 1.0) < 1e-8
        y = q0 - 1.0
        asinc_val = np.where(near_one, 1 - y/3 + 2*y**2/15, ac/s)
        dasinc = np.where(near_one, -1.0/3 + 4*y/15, -1.0/s2 + q0*ac/(s*s2))
        d2asinc = np.where(near_one, 4.0/15,
                           (3*q0/s2 - (1+2*q0**2)*ac/(s*s2)) * (-1.0/s2))

        for axis in range(3):
            a = axis + 1
            qa = c[:, a]

            df_dq = np.zeros((n_batch, 4))
            df_dq[:, 0] = 2 * qa * dasinc
            df_dq[:, a] = 2 * asinc_val

            d2f_dq2 = np.zeros((n_batch, 4, 4))
            d2f_dq2[:, 0, 0] = 2 * qa * d2asinc
            d2f_dq2[:, 0, a] = 2 * dasinc
            d2f_dq2[:, a, 0] = 2 * dasinc

            # term1: dc @ d2f @ dc^T @ v = dc @ d2f @ dc_v
            t1_hvp = np.squeeze(
                dc_flat @ (d2f_dq2 @ dc_v[:, :, None]), -1
            )  # (n_batch, M)

            # term2: w = M_inv(df_dq)
            proj_w = np.squeeze(vecs.swapaxes(1, 2) @ df_dq[:, :, None], -1)
            w = np.squeeze(vecs @ (safe_inv * proj_w)[:, :, None], -1)  # (n_batch, 4)
            wc = (w * c).sum(axis=1)  # (n_batch,)
            w_dc = np.squeeze(dc_flat @ w[:, :, None], -1)  # (n_batch, M)
            fdq_c = (df_dq * c).sum(axis=1)  # (n_batch,)

            # dFw: dF @ w for all (k,d)
            dFw_flat = _build_dF_vec_batched(Pref, w, n_batch, nr)  # (n_batch, M, 4)

            w_dc_v = (w_dc * v_flat).sum(axis=1)  # (n_batch,)

            # wdFdc @ v = dFw_flat @ dc_v
            wdFdc_v = np.squeeze(dFw_flat @ dc_v[:, :, None], -1)

            # wdFdc^T @ v = dc_flat @ (dFw_flat^T @ v)
            dFw_v = np.squeeze(dFw_flat.swapaxes(1, 2) @ v_flat[:, :, None], -1)
            wdFdcT_v = np.squeeze(dc_flat @ dFw_v[:, :, None], -1)

            t2_hvp = (dE_flat * w_dc_v[:, None]
                      + dE_v[:, None] * w_dc
                      + wc[:, None] * d2E_v
                      - wdFdc_v - wdFdcT_v
                      - fdq_c[:, None] * dc_dot_v)

            hvp_axis = (t1_hvp + t2_hvp).reshape(n_batch, nr, 3)
            hvp[idx, axis, :nr, :] = hvp_axis

    return hvp

def _rotation_3axis_hvp(pos, refpos, mask, v):
    """HVP for one fragment, all 3 axes at once.

    Returns shape (3, N, 3) — the directional derivative of the
    Jacobian (3, N, 3) along v (N, 3).
    """
    primals = (pos,)
    tangents = (v,)
    _, hvp = jvp(
        lambda p: jacfwd(_rotation_3axis_masked, argnums=0)(p, refpos, mask),
        primals, tangents
    )
    return hvp

_rotation_3axis_hvp_batched_jit = jit(
    vmap(_rotation_3axis_hvp, in_axes=(0, 0, 0, 0))
)

class Rotation(Coordinate):
    def __init__(
        self,
        indices: Tuple[int, ...],
        axis: int,
        refpos: np.ndarray,
    ) -> None:
        assert len(indices) >= 2
        Coordinate.__init__(self, indices)
        self.kwargs['axis'] = axis
        self.kwargs['refpos'] = refpos.copy() - refpos.mean(0)
        self.q_prev = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.kwargs['axis'] != other.kwargs['axis']:
            return False
        if len(self.indices) != len(other.indices):
            return False
        if set(self.indices) != set(other.indices):
            return False
        if not np.allclose(self.kwargs['refpos'], other.kwargs['refpos']):
            return False
        return True

    def calc(self, atoms: Atoms) -> float:
        pos = np.asarray(atoms.positions[self.indices], dtype=np.float64)
        dx = pos - pos.mean(0)
        refpos = self.kwargs['refpos']
        F = _build_F_matrix_np(dx, refpos)
        q = _stabilize_quaternion(F, self.q_prev)
        self.q_prev = q
        axis = self.kwargs['axis']
        return float(2.0 * q[axis + 1] * _asinc_np(q[0]))

    def calc_gradient(self, atoms: Atoms) -> np.ndarray:
        pos = np.asarray(atoms.positions[self.indices], dtype=np.float64)
        refpos = self.kwargs['refpos']
        jac = _rotation_3axis_jacobian_np(pos, refpos, self.q_prev)
        return jac[self.kwargs['axis']]

    def calc_hessian(self, atoms: Atoms) -> jnp.ndarray:
        return _rotation_hessian_np(
            atoms.positions[self.indices],
            self.kwargs['axis'],
            self.kwargs['refpos'],
            q_stable=self.q_prev,
        )

def _bond(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    return jnp.linalg.norm(
        pos[1] - pos[0] + tvecs[0]
    )

class Bond(Internal):
    nindices = 2
    _eval0 = staticmethod(jit(_bond))
    _eval1 = staticmethod(_gradient(_bond))
    _eval2 = staticmethod(_hessian(_bond))
    _eval_cell_grad = staticmethod(_bond_cell_grad_single)

    def calc_vec(self, atoms: Atoms) -> np.ndarray:
        tvecs = np.asarray(
            self.kwargs['ncvecs'] @ atoms.cell, dtype=np.float64
        )
        i, j = self.indices
        return atoms.positions[j] - atoms.positions[i] + tvecs[0]

def _angle(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = -(pos[1] - pos[0] + tvecs[0])
    dx2 = pos[2] - pos[1] + tvecs[1]
    cos_angle = dx1 @ dx2 / (jnp.linalg.norm(dx1) * jnp.linalg.norm(dx2))
    # Clamp to avoid NaN from arccos due to floating-point errors
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    return jnp.arccos(cos_angle)

class Angle(Internal):
    nindices = 3
    _eval0 = staticmethod(jit(_angle))
    _eval1 = staticmethod(_gradient(_angle))
    _eval2 = staticmethod(_hessian(_angle))
    _eval_cell_grad = staticmethod(_angle_cell_grad_single)

def _dihedral(
    pos: jnp.ndarray,
    tvecs: jnp.ndarray
) -> float:
    dx1 = pos[1] - pos[0] + tvecs[0]
    dx2 = pos[2] - pos[1] + tvecs[1]
    dx3 = pos[3] - pos[2] + tvecs[2]
    numer = dx2 @ jnp.cross(jnp.cross(dx1, dx2), jnp.cross(dx2, dx3))
    denom = jnp.linalg.norm(dx2) * jnp.cross(dx1, dx2) @ jnp.cross(dx2, dx3)
    return jnp.arctan2(numer, denom)

class Dihedral(Internal):
    nindices = 4
    _eval0 = staticmethod(jit(_dihedral))
    _eval1 = staticmethod(_gradient(_dihedral))
    _eval2 = staticmethod(_hessian(_dihedral))
    _eval_cell_grad = staticmethod(_dihedral_cell_grad_single)

Bond.union = Angle

Angle.union = Dihedral

Angle.diff = Bond

Dihedral.diff = Angle

class BaseInternals:
    _names = (
        'translations', 'bonds', 'angles', 'dihedrals', 'other', 'rotations'
    )

    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None
    ) -> None:
        self.atoms = atoms

        self._lastpos = None
        self._cache = dict()
        self._cache_version = 0

        if dummies is None:
            if dinds is not None:
                raise ValueError('"dinds" provided, but no "dummies"!')
            dummies = Atoms()
            dinds = -np.ones(len(self.atoms), dtype=np.int32)
        else:
            if dinds is None:
                raise ValueError('"dummies" provided, but no "dinds"!')
            ndum = len(dummies)
            ndind = np.sum(dinds >= 0)
            if ndum != ndind:
                raise ValueError(
                    '{} dummy atoms were provided, but only {} dummy indices!'
                    .format(ndum, ndind)
                )
        self.dummies = dummies
        self.dinds = dinds

        # Cache atom count (doesn't change during optimization)
        self._natoms = len(atoms)

        self.internals = {key: [] for key in self._names}
        self._internals_set = {key: set() for key in self._names}
        self._active = {key: [] for key in self._names}
        self.cell = None
        self.rcell = None
        self._rcell_reciprocal_T = None
        self.op = None
        self._hessian_skeleton = None

        # Batched arrays for vectorized computation (built lazily)
        self._batched_arrays_valid = False

        # Lazy caches.
        self._tvecs_cache = None  # set to {'cell_hash': ..., 'tvecs': ...} on first build
        self._hvp_buf = None  # reusable buffer for hessian_rdot output

    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def ndummies(self) -> int:
        return len(self.dummies)

    @property
    def ndof(self) -> int:
        return 3 * (self._natoms + len(self.dummies))

    @property
    def ntrans(self) -> int:
        return sum(self._active['translations'])

    @property
    def nbonds(self) -> int:
        return sum(self._active['bonds'])

    @property
    def nangles(self) -> int:
        return sum(self._active['angles'])

    @property
    def ndihedrals(self) -> int:
        return sum(self._active['dihedrals'])

    @property
    def nother(self) -> int:
        return sum(self._active['other'])

    @property
    def nrotations(self) -> int:
        return sum(self._active['rotations'])

    @property
    def _active_mask(self) -> List[bool]:
        active = []
        for name in self._names:
            active += self._active[name]
        return active

    @property
    def _active_indices(self) -> List[int]:
        return [idx for idx, active in enumerate(self._active_mask) if active]

    @property
    def nint(self) -> int:
        return len(self._active_indices)

    @property
    def all_positions(self) -> np.ndarray:
        """Get combined positions without creating an Atoms object.

        Cached on ``self._cache['all_positions']`` so repeated reads
        within a single position evaluation reuse the same vstack.
        ``_cache_check`` clears the cache whenever positions change.
        """
        if self.ndummies == 0:
            return self.atoms.positions
        cached = self._cache.get('all_positions')
        if cached is not None:
            return cached
        merged = np.vstack([self.atoms.positions, self.dummies.positions])
        self._cache['all_positions'] = merged
        return merged

    @property
    def all_atoms(self) -> Atoms:
        return self.atoms + self.dummies

    @property
    def light_atoms(self) -> LightAtoms:
        """Get lightweight atoms-like object for coordinate calculations."""
        cell = self.atoms.cell.array
        return LightAtoms(self.all_positions, cell)

    def _cache_check(self) -> None:
        # we are comparing the current atomic positions to what they were
        # the last time a property was calculated. These positions are floats,
        # but we use a strict equality check to compare to avoid subtle bugs
        # that might occur during fine-resolution geodesic steps.
        if self.ndummies == 0:
            current_pos = self.atoms.positions
        else:
            current_pos = np.vstack([self.atoms.positions, self.dummies.positions])
        if (
            self._lastpos is None
            or np.any(current_pos != self._lastpos)
        ):
            self._cache = dict()
            self._lastpos = current_pos.copy()
            self._cache_version += 1
        # Park the freshly-merged positions in the cache so the next
        # all_positions access doesn't redo the vstack.
        if self.ndummies > 0:
            self._cache.setdefault('all_positions', self._lastpos)

    def _build_batched_arrays(self) -> None:
        """Build batched index arrays for vectorized computation.

        Arrays are padded to multiples of BLOCK_SIZE for GPU/SIMD efficiency.
        Masks are stored to filter results back to actual sizes.
        """
        if self._batched_arrays_valid:
            return

        def pad_to_block(n: int) -> int:
            """Round up to nearest multiple of BLOCK_SIZE."""
            return ((n + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

        # Build arrays for bonds
        bonds = self.internals['bonds']
        n_bonds = len(bonds)
        if n_bonds > 0:
            n_bonds_padded = pad_to_block(n_bonds)
            # Original (unpadded) arrays for indexing
            self._bond_indices = np.array([b.indices for b in bonds], dtype=np.int32)
            self._bond_ncvecs = np.array(
                [b.kwargs['ncvecs'] for b in bonds], dtype=np.int32
            )
            # Padded arrays for batch computation
            self._bond_indices_padded = np.zeros((n_bonds_padded, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.zeros((n_bonds_padded, 1, 3), dtype=np.int32)
            self._bond_indices_padded[:n_bonds] = self._bond_indices
            self._bond_ncvecs_padded[:n_bonds] = self._bond_ncvecs
            self._bond_mask = np.zeros(n_bonds_padded, dtype=np.float64)
            self._bond_mask[:n_bonds] = 1.0
            self._n_bonds_actual = n_bonds
        else:
            self._bond_indices = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_indices_padded = np.empty((0, 2), dtype=np.int32)
            self._bond_ncvecs_padded = np.empty((0, 1, 3), dtype=np.int32)
            self._bond_mask = np.empty(0, dtype=np.float64)
            self._n_bonds_actual = 0

        # Build arrays for angles
        angles = self.internals['angles']
        n_angles = len(angles)
        if n_angles > 0:
            n_angles_padded = pad_to_block(n_angles)
            self._angle_indices = np.array([a.indices for a in angles], dtype=np.int32)
            self._angle_ncvecs = np.array(
                [a.kwargs['ncvecs'] for a in angles], dtype=np.int32
            )
            self._angle_indices_padded = np.zeros((n_angles_padded, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.zeros((n_angles_padded, 2, 3), dtype=np.int32)
            self._angle_indices_padded[:n_angles] = self._angle_indices
            self._angle_ncvecs_padded[:n_angles] = self._angle_ncvecs
            self._angle_mask = np.zeros(n_angles_padded, dtype=np.float64)
            self._angle_mask[:n_angles] = 1.0
            self._n_angles_actual = n_angles
        else:
            self._angle_indices = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_indices_padded = np.empty((0, 3), dtype=np.int32)
            self._angle_ncvecs_padded = np.empty((0, 2, 3), dtype=np.int32)
            self._angle_mask = np.empty(0, dtype=np.float64)
            self._n_angles_actual = 0

        # Build arrays for dihedrals
        dihedrals = self.internals['dihedrals']
        n_dihedrals = len(dihedrals)
        if n_dihedrals > 0:
            n_dihedrals_padded = pad_to_block(n_dihedrals)
            self._dihedral_indices = np.array(
                [d.indices for d in dihedrals], dtype=np.int32
            )
            self._dihedral_ncvecs = np.array(
                [d.kwargs['ncvecs'] for d in dihedrals], dtype=np.int32
            )
            self._dihedral_indices_padded = np.zeros((n_dihedrals_padded, 4), dtype=np.int32)
            self._dihedral_ncvecs_padded = np.zeros((n_dihedrals_padded, 3, 3), dtype=np.int32)
            self._dihedral_indices_padded[:n_dihedrals] = self._dihedral_indices
            self._dihedral_ncvecs_padded[:n_dihedrals] = self._dihedral_ncvecs
            self._dihedral_mask = np.zeros(n_dihedrals_padded, dtype=np.float64)
            self._dihedral_mask[:n_dihedrals] = 1.0
            self._n_dihedrals_actual = n_dihedrals
        else:
            self._dihedral_indices = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_indices_padded = np.empty((0, 4), dtype=np.int32)
            self._dihedral_ncvecs_padded = np.empty((0, 3, 3), dtype=np.int32)
            self._dihedral_mask = np.empty(0, dtype=np.float64)
            self._n_dihedrals_actual = 0

        # Precompute flat column indices for direct scatter in hessian_rdot.
        # For bond (a,b), the non-zero columns in the (ndof,) output are
        # [3a, 3a+1, 3a+2, 3b, 3b+1, 3b+2].  Analogous for angles (9 cols)
        # and dihedrals (12 cols).  These are topology-dependent and
        # invalidated together with the rest of the batched arrays.
        offsets = np.arange(3)
        if self._n_bonds_actual > 0:
            bi = self._bond_indices  # (n_bonds, 2)
            self._bond_flat_cols = np.concatenate([
                bi[:, k:k+1] * 3 + offsets for k in range(2)
            ], axis=1)  # (n_bonds, 6)
        else:
            self._bond_flat_cols = np.empty((0, 6), dtype=np.intp)

        if self._n_angles_actual > 0:
            ai = self._angle_indices  # (n_angles, 3)
            self._angle_flat_cols = np.concatenate([
                ai[:, k:k+1] * 3 + offsets for k in range(3)
            ], axis=1)  # (n_angles, 9)
        else:
            self._angle_flat_cols = np.empty((0, 9), dtype=np.intp)

        if self._n_dihedrals_actual > 0:
            di = self._dihedral_indices  # (n_dihedrals, 4)
            self._dihedral_flat_cols = np.concatenate([
                di[:, k:k+1] * 3 + offsets for k in range(4)
            ], axis=1)  # (n_dihedrals, 12)
        else:
            self._dihedral_flat_cols = np.empty((0, 12), dtype=np.intp)

        # Build CSR structure for sparse hessian_rdot output.
        # Bonds/angles/dihedrals have fixed nnz per row (6/9/12).
        # Translations have zero rows. Rotations/other are dense (ndof cols).
        ndof = self.ndof
        n_trans = len(self.internals['translations'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])
        n_active = (n_trans + self._n_bonds_actual + self._n_angles_actual
                    + self._n_dihedrals_actual + n_other + n_rot)

        col_blocks = []
        nnz_per_row = []

        # Translations: zero rows
        for _ in range(n_trans):
            nnz_per_row.append(0)

        # Bonds: 6 nnz per row
        if self._n_bonds_actual > 0:
            col_blocks.append(self._bond_flat_cols.ravel())
            nnz_per_row.extend([6] * self._n_bonds_actual)

        # Angles: 9 nnz per row
        if self._n_angles_actual > 0:
            col_blocks.append(self._angle_flat_cols.ravel())
            nnz_per_row.extend([9] * self._n_angles_actual)

        # Dihedrals: 12 nnz per row
        if self._n_dihedrals_actual > 0:
            col_blocks.append(self._dihedral_flat_cols.ravel())
            nnz_per_row.extend([12] * self._n_dihedrals_actual)

        # Other/rotations: dense rows (ndof cols each)
        for _ in range(n_other + n_rot):
            col_blocks.append(np.arange(ndof))
            nnz_per_row.append(ndof)

        self._csr_indptr = np.zeros(n_active + 1, dtype=np.int32)
        np.cumsum(nnz_per_row, out=self._csr_indptr[1:])
        self._csr_indices = np.concatenate(col_blocks).astype(np.int32) if col_blocks else np.empty(0, dtype=np.int32)
        self._csr_data = np.zeros(len(self._csr_indices), dtype=np.float64)
        self._csr_n_active = n_active
        # Precompute data offset for each section
        self._csr_bond_offset = n_trans * 0  # bonds start after translations (0 nnz)
        self._csr_angle_offset = self._csr_bond_offset + self._n_bonds_actual * 6
        self._csr_dih_offset = self._csr_angle_offset + self._n_angles_actual * 9
        self._csr_other_offset = self._csr_dih_offset + self._n_dihedrals_actual * 12

        self._batched_arrays_valid = True

    def _get_cached_tvecs(self, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Get cached translation vectors for cell, computing if necessary.

        The tvecs (ncvecs @ cell) are constant for a given cell, so we cache
        them to avoid redundant matrix multiplications during ODE integration.

        Returns both unpadded tvecs (for indexing) and padded tvecs (for batch ops).
        """
        cell_hash = cell.tobytes()
        if self._tvecs_cache is not None and self._tvecs_cache['cell_hash'] == cell_hash:
            return self._tvecs_cache['tvecs']

        self._build_batched_arrays()
        tvecs = {}

        # Unpadded tvecs (for result indexing)
        if len(self._bond_indices) > 0:
            tvecs['bonds'] = self._bond_ncvecs @ cell
        else:
            tvecs['bonds'] = np.empty((0, 1, 3), dtype=np.float64)

        if len(self._angle_indices) > 0:
            tvecs['angles'] = self._angle_ncvecs @ cell
        else:
            tvecs['angles'] = np.empty((0, 2, 3), dtype=np.float64)

        if len(self._dihedral_indices) > 0:
            tvecs['dihedrals'] = self._dihedral_ncvecs @ cell
        else:
            tvecs['dihedrals'] = np.empty((0, 3, 3), dtype=np.float64)

        # Padded tvecs (for GPU-efficient batch computation)
        if len(self._bond_indices_padded) > 0:
            tvecs['bonds_padded'] = self._bond_ncvecs_padded @ cell
        else:
            tvecs['bonds_padded'] = np.empty((0, 1, 3), dtype=np.float64)

        if len(self._angle_indices_padded) > 0:
            tvecs['angles_padded'] = self._angle_ncvecs_padded @ cell
        else:
            tvecs['angles_padded'] = np.empty((0, 2, 3), dtype=np.float64)

        if len(self._dihedral_indices_padded) > 0:
            tvecs['dihedrals_padded'] = self._dihedral_ncvecs_padded @ cell
        else:
            tvecs['dihedrals_padded'] = np.empty((0, 3, 3), dtype=np.float64)

        self._tvecs_cache = {'cell_hash': cell_hash, 'tvecs': tvecs}
        return tvecs


    def _compute_batched_values(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all internal coordinate values using vectorized operations.

        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays for consistent JAX shapes
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]  # (n_padded, 2, 3)
            values_padded = np.asarray(device_get(_bond_value_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = values_padded[:self._n_bonds_actual]
        else:
            result['bonds'] = np.empty(0)

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]  # (n_padded, 3, 3)
            values_padded = np.asarray(device_get(_angle_value_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = values_padded[:self._n_angles_actual]
        else:
            result['angles'] = np.empty(0)

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]  # (n_padded, 4, 3)
            values_padded = np.asarray(device_get(_dihedral_value_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = values_padded[:self._n_dihedrals_actual]
        else:
            result['dihedrals'] = np.empty(0)

        return result

    def _compute_batched_gradients(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate gradients using vectorized operations.

        Returns dict mapping coord type to (indices, gradients) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]  # (n_padded, 2, 3)
            grads_padded = np.asarray(device_get(_bond_grad_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = (self._bond_indices, grads_padded[:self._n_bonds_actual])
        else:
            result['bonds'] = (np.empty((0, 2), dtype=np.int32), np.empty((0, 2, 3)))

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]
            grads_padded = np.asarray(device_get(_angle_grad_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = (self._angle_indices, grads_padded[:self._n_angles_actual])
        else:
            result['angles'] = (np.empty((0, 3), dtype=np.int32), np.empty((0, 3, 3)))

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]
            grads_padded = np.asarray(device_get(_dihedral_grad_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = (self._dihedral_indices, grads_padded[:self._n_dihedrals_actual])
        else:
            result['dihedrals'] = (np.empty((0, 4), dtype=np.int32), np.empty((0, 4, 3)))

        return result

    def _compute_batched_hessians(self, positions: np.ndarray, cell: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Compute all internal coordinate hessians using vectorized operations.

        Returns dict mapping coord type to (indices, hessians) tuples.
        Uses padded arrays for GPU/SIMD efficiency, then slices to actual size.
        """
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)
        result = {}

        # Bonds - use padded arrays
        if self._n_bonds_actual > 0:
            bond_pos = positions[self._bond_indices_padded]
            hess_padded = np.asarray(device_get(_bond_hess_batched(bond_pos, tvecs['bonds_padded'])))
            result['bonds'] = (self._bond_indices, hess_padded[:self._n_bonds_actual])
        else:
            result['bonds'] = (np.empty((0, 2), dtype=np.int32), np.empty((0, 2, 3, 2, 3)))

        # Angles
        if self._n_angles_actual > 0:
            angle_pos = positions[self._angle_indices_padded]
            hess_padded = np.asarray(device_get(_angle_hess_batched(angle_pos, tvecs['angles_padded'])))
            result['angles'] = (self._angle_indices, hess_padded[:self._n_angles_actual])
        else:
            result['angles'] = (np.empty((0, 3), dtype=np.int32), np.empty((0, 3, 3, 3, 3)))

        # Dihedrals
        if self._n_dihedrals_actual > 0:
            dihedral_pos = positions[self._dihedral_indices_padded]
            hess_padded = np.asarray(device_get(_dihedral_hess_batched(dihedral_pos, tvecs['dihedrals_padded'])))
            result['dihedrals'] = (self._dihedral_indices, hess_padded[:self._n_dihedrals_actual])
        else:
            result['dihedrals'] = (np.empty((0, 4), dtype=np.int32), np.empty((0, 4, 3, 4, 3)))

        return result


    def copy(self) -> 'BaseInternals':
        raise NotImplementedError

    def calc(self) -> np.ndarray:
        """Calculates the internal coordinate vector using vectorized operations."""
        self._cache_check()
        if 'coords' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_vals = self._compute_batched_values(positions, cell)

            # Build full coords list in order
            all_coords = []

            # Translations (not batched - usually few) - use lightweight atoms
            atoms = self.light_atoms
            for coord in self.internals['translations']:
                all_coords.append(coord.calc(atoms))

            # Bonds (batched)
            all_coords.extend(batched_vals['bonds'].tolist())

            # Angles (batched)
            all_coords.extend(batched_vals['angles'].tolist())

            # Dihedrals (batched)
            all_coords.extend(batched_vals['dihedrals'].tolist())

            # Other (not batched - heterogeneous)
            for coord in self.internals['other']:
                all_coords.append(coord.calc(atoms))

            # Rotations (batched if all 3 axes present per fragment)
            rot_vals = self._batched_rotation_values(positions)
            if rot_vals is None:
                for coord in self.internals['rotations']:
                    all_coords.append(coord.calc(atoms))
            else:
                all_coords.extend(rot_vals)

            self._cache['coords'] = np.array(all_coords)

        return np.array([
            x for x, a in zip(self._cache['coords'], self._active_mask) if a
        ])

    def jacobian(self) -> np.ndarray:
        """Calculates the internal coordinate Jacobian matrix using vectorized operations."""
        self._cache_check()

        # If a fully-built B was cached, return it directly. The cache is
        # invalidated by _cache_check whenever positions change, and the active
        # mask is stable within a single position evaluation.
        cached_B = self._cache.get('jacobian_B')
        if cached_B is not None:
            return cached_B

        if 'jacobian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_grads = self._compute_batched_gradients(positions, cell)

            # Non-batched coords use lightweight atoms
            atoms = self.light_atoms
            trans_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                          for coord in self.internals['translations']]
            other_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_gradients(positions)
            if rot_data is None:
                rot_data = [(coord.indices, np.array(coord.calc_gradient(atoms)))
                            for coord in self.internals['rotations']]

            self._cache['jacobian_batched'] = batched_grads
            self._cache['jacobian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['jacobian'] = object()

        # Get cached data
        batched = self._cache['jacobian_batched']
        trans_data, other_data, rot_data = self._cache['jacobian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = active_mask[start:start+n_bonds]
        start += n_bonds
        angles_active = active_mask[start:start+n_angles]
        start += n_angles
        dihedrals_active = active_mask[start:start+n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_active = sum(active_mask)
        n_atoms = self.natoms + self.ndummies
        B = np.zeros((n_active, n_atoms, 3))
        row = 0

        # Translations (not batched)
        for i, (idx, jac) in enumerate(trans_data):
            if trans_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        # Bonds (batched) - vectorized scatter
        bond_indices, bond_grads = batched['bonds']
        bonds_active_arr = np.array(bonds_active, dtype=bool)
        n_active_bonds = bonds_active_arr.sum()
        if n_active_bonds > 0:
            active_bond_idx = bond_indices[bonds_active_arr]
            active_bond_grads = bond_grads[bonds_active_arr]
            # Vectorized scatter: replace loop with advanced indexing
            rows_idx = np.arange(row, row + n_active_bonds)[:, None]
            B[rows_idx, active_bond_idx] = active_bond_grads
            row += n_active_bonds

        # Angles (batched) - vectorized scatter
        angle_indices, angle_grads = batched['angles']
        angles_active_arr = np.array(angles_active, dtype=bool)
        n_active_angles = angles_active_arr.sum()
        if n_active_angles > 0:
            active_angle_idx = angle_indices[angles_active_arr]
            active_angle_grads = angle_grads[angles_active_arr]
            # Vectorized scatter
            rows_idx = np.arange(row, row + n_active_angles)[:, None]
            B[rows_idx, active_angle_idx] = active_angle_grads
            row += n_active_angles

        # Dihedrals (batched) - vectorized scatter
        dihedral_indices, dihedral_grads = batched['dihedrals']
        dihedrals_active_arr = np.array(dihedrals_active, dtype=bool)
        n_active_dihedrals = dihedrals_active_arr.sum()
        if n_active_dihedrals > 0:
            active_dih_idx = dihedral_indices[dihedrals_active_arr]
            active_dih_grads = dihedral_grads[dihedrals_active_arr]
            # Vectorized scatter
            rows_idx = np.arange(row, row + n_active_dihedrals)[:, None]
            B[rows_idx, active_dih_idx] = active_dih_grads
            row += n_active_dihedrals

        # Other (not batched)
        for i, (idx, jac) in enumerate(other_data):
            if other_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        # Rotations (not batched)
        for i, (idx, jac) in enumerate(rot_data):
            if rot_active[i]:
                np.add.at(B, (row, idx), jac)
                row += 1

        result = B.reshape((n_active, 3 * n_atoms))
        self._cache['jacobian_B'] = result
        return result


    def _rotation_padded_inputs(self, positions: np.ndarray):
        """Build padded (pos, refpos, mask) batches grouped by fragment.

        Returns ``(pos_pad, ref_pad, mask, frag_indices, frag_axis_slots,
        valid)`` where:
          pos_pad/ref_pad shape (n_frags, N_max, 3),
          mask shape (n_frags, N_max),
          frag_indices: list of np.array per fragment,
          frag_axis_slots: list of [axis0_idx, axis1_idx, axis2_idx]
            per fragment (each entry is the original Rotation index),
          valid: True if all fragments have all 3 axes.
        Cached per geometry on ``self._cache``.
        """
        cached = self._cache.get('rotation_pad')
        if cached is not None:
            return cached
        rotations = self.internals['rotations']
        if not rotations:
            out = (None, None, None, [], [], True)
            self._cache['rotation_pad'] = out
            return out
        groups = {}
        for i, r in enumerate(rotations):
            key = (tuple(r.indices), r.kwargs['refpos'].tobytes())
            slot = groups.setdefault(key, [None, None, None])
            slot[r.kwargs['axis']] = i
        if any(None in slot for slot in groups.values()):
            out = (None, None, None, [], [], False)
            self._cache['rotation_pad'] = out
            return out
        n_frags = len(groups)
        n_max = max(len(r.indices) for r in rotations)
        pos_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        ref_pad = np.zeros((n_frags, n_max, 3), dtype=np.float64)
        mask = np.zeros((n_frags, n_max), dtype=np.float64)
        frag_indices = []
        frag_axis_slots = []
        for fi, slot in enumerate(groups.values()):
            r0 = rotations[slot[0]]
            n = len(r0.indices)
            pos_pad[fi, :n] = positions[r0.indices]
            ref_pad[fi, :n] = r0.kwargs['refpos']
            mask[fi, :n] = 1.0
            frag_indices.append(np.asarray(r0.indices))
            frag_axis_slots.append(slot)
        out = (pos_pad, ref_pad, mask, frag_indices, frag_axis_slots, True)
        self._cache['rotation_pad'] = out
        return out

    def _get_stabilized_quaternions(self, positions: np.ndarray):
        """Return cached stabilized quaternions, recomputing if needed.

        Returns a list of (4,) numpy arrays, one per fragment, or None
        if the batched path is invalid.  Also caches per-fragment
        eigenvalues/eigenvectors in ``self._cache['stabilized_q_eigh']``
        for reuse in the HVP path.
        """
        cached = self._cache.get('stabilized_q')
        if cached is not None:
            return cached
        rotations = self.internals['rotations']
        if not rotations:
            self._cache['stabilized_q'] = []
            self._cache['stabilized_q_eigh'] = (None, None)
            return []
        pos_pad, ref_pad, mask, frag_indices, slots, valid = (
            self._rotation_padded_inputs(positions)
        )
        if not valid:
            self._cache['stabilized_q'] = None
            self._cache['stabilized_q_eigh'] = (None, None)
            return None
        n_frags = len(slots)
        ws_list = []
        vecs_list = []
        qs = []
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            pos_frag = pos_pad[fi, :n]
            ref_frag = ref_pad[fi, :n]
            dx = pos_frag - pos_frag.mean(0)
            F = _build_F_matrix_np(dx, ref_frag)
            q_prev = rotations[slot[0]].q_prev
            ws_i, vecs_i = np.linalg.eigh(F)
            ws_list.append(ws_i)
            vecs_list.append(vecs_i)
            q = _stabilize_quaternion_from_eigh(ws_i, vecs_i, q_prev)
            for axis in range(3):
                rotations[slot[axis]].q_prev = q
            qs.append(q)
        self._cache['stabilized_q'] = qs
        self._cache['stabilized_q_eigh'] = (
            np.array(ws_list), np.array(vecs_list)
        )
        return qs

    def _batched_rotation_values(self, positions: np.ndarray):
        """Per-Rotation values with projective quaternion stabilization.

        Returns a length-N_rotations list of floats in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return None
        _, _, _, _, slots, _ = self._rotation_padded_inputs(positions)
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            vals = _expmap_np(qs[fi])
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = float(vals[axis])
        return out

    def _batched_rotation_gradients(self, positions: np.ndarray):
        """Per-Rotation gradients using stabilized quaternion.

        Returns a list of ``(indices, grad)`` tuples in original order,
        or None when the heterogeneous fall-back is required.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return None
        pos_pad, ref_pad, _, frag_indices, slots, _ = (
            self._rotation_padded_inputs(positions)
        )
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            pos_frag = pos_pad[fi, :n]
            ref_frag = ref_pad[fi, :n]
            jac = _rotation_3axis_jacobian_np(pos_frag, ref_frag, qs[fi])
            for axis, rot_idx in enumerate(slot):
                out[rot_idx] = (frag_indices[fi], jac[axis])
        return out

    def _batched_rotation_hessians(self, positions: np.ndarray):
        """Compute per-Rotation Hessians using stabilized quaternion.

        Returns a list of ``(indices, hess)`` tuples in the original
        per-Rotation order.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return []
        qs = self._get_stabilized_quaternions(positions)
        if qs is None:
            return [(r.indices, np.array(r.calc_hessian(
                self.light_atoms))) for r in rotations]
        pos_pad, ref_pad, _, frag_indices, slots, _ = (
            self._rotation_padded_inputs(positions)
        )
        out = [None] * len(rotations)
        for fi, slot in enumerate(slots):
            n = len(frag_indices[fi])
            pos_frag = np.asarray(pos_pad[fi, :n], dtype=np.float64)
            ref_frag = np.asarray(ref_pad[fi, :n], dtype=np.float64)
            for axis, rot_idx in enumerate(slot):
                h = _rotation_hessian_single(pos_frag, axis, ref_frag,
                                             q_stable=qs[fi])
                out[rot_idx] = (frag_indices[fi], h)
        return out

    def _get_hessian_skeleton(self, hessians):
        """Return a cached SparseInternalHessiansSkeleton for ``hessians``.

        The skeleton holds index-derived data (per-size groupings, scatter
        indices) that depend only on which coordinates exist and which
        atom indices they touch — not on positions or Hessian values. We
        invalidate by total coord count + active mask, which jointly
        cover the mutation paths: ``add_dummy_to_internals`` /
        ``find_all_*`` / ``check_for_bad_internals`` regenerations grow
        ``self.internals``, while ``apply_inequalities`` /
        ``validate_inequalities`` flip ``self._active``.
        """
        key = (len(hessians), self.natoms + self.ndummies,
               tuple(self._active_mask))
        cached = self._hessian_skeleton
        if cached is not None and cached[0] == key:
            return cached[1]
        skeleton = SparseInternalHessiansSkeleton(hessians,
                                                  self.natoms + self.ndummies)
        self._hessian_skeleton = (key, skeleton)
        return skeleton

    def hessian(self) -> np.ndarray:
        """Calculates the Hessian matrix for each internal coordinate using vectorized operations."""
        self._cache_check()

        # Return cached SparseInternalHessians object if available
        if 'hessian_result' in self._cache:
            return self._cache['hessian_result']

        if 'hessian' not in self._cache:
            positions = self.all_positions
            cell = self.atoms.cell.array

            # Use vectorized computation for bonds, angles, dihedrals
            batched_hess = self._compute_batched_hessians(positions, cell)

            # Non-batched coords use lightweight atoms. Translation hessians are
            # identically zero (translations are linear in positions), so cache
            # one zero array per (n,) and reuse — avoids 24+ JAX calls per
            # hessian rebuild on systems with TRICs.
            atoms = self.light_atoms
            trans_data = []
            zero_cache = {}
            for coord in self.internals['translations']:
                n = len(coord.indices)
                z = zero_cache.get(n)
                if z is None:
                    z = np.zeros((n, 3, n, 3))
                    zero_cache[n] = z
                trans_data.append((coord.indices, z))
            other_data = [(coord.indices, np.array(coord.calc_hessian(atoms)))
                          for coord in self.internals['other']]
            rot_data = self._batched_rotation_hessians(positions)

            self._cache['hessian_batched'] = batched_hess
            self._cache['hessian_nonbatched'] = (trans_data, other_data, rot_data)
            # Store a unique object (not a singleton) for cache identity
            self._cache['hessian'] = object()

        # Get cached data
        batched = self._cache['hessian_batched']
        trans_data, other_data, rot_data = self._cache['hessian_nonbatched']

        # Get counts for each type
        n_trans = len(trans_data)
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(other_data)
        n_rot = len(rot_data)

        # Build active masks per type
        active_mask = self._active_mask
        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = active_mask[start:start+n_bonds]
        start += n_bonds
        angles_active = active_mask[start:start+n_angles]
        start += n_angles
        dihedrals_active = active_mask[start:start+n_dihedrals]
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_atoms = self.natoms + self.ndummies
        hessians = []

        # Translations (not batched). Hessian rows are stored in cached
        # nonbatched data; SparseInternalHessian only reads .vals so views are
        # safe.
        for i, (idx, hess) in enumerate(trans_data):
            if trans_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        # Bonds (batched). Fancy indexing already returns a fresh array; per-row
        # views into it are read-only consumers, so no per-coord copy is needed.
        bond_indices, bond_hess = batched['bonds']
        bonds_active_arr = np.asarray(bonds_active, dtype=bool)
        if bonds_active_arr.any():
            active_bond_idx = bond_indices[bonds_active_arr]
            active_bond_hess = bond_hess[bonds_active_arr]
            for i in range(len(active_bond_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_bond_idx[i], active_bond_hess[i]))

        # Angles (batched)
        angle_indices, angle_hess = batched['angles']
        angles_active_arr = np.asarray(angles_active, dtype=bool)
        if angles_active_arr.any():
            active_angle_idx = angle_indices[angles_active_arr]
            active_angle_hess = angle_hess[angles_active_arr]
            for i in range(len(active_angle_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_angle_idx[i], active_angle_hess[i]))

        # Dihedrals (batched)
        dihedral_indices, dihedral_hess = batched['dihedrals']
        dihedrals_active_arr = np.asarray(dihedrals_active, dtype=bool)
        if dihedrals_active_arr.any():
            active_dih_idx = dihedral_indices[dihedrals_active_arr]
            active_dih_hess = dihedral_hess[dihedrals_active_arr]
            for i in range(len(active_dih_idx)):
                hessians.append(SparseInternalHessian(n_atoms, active_dih_idx[i], active_dih_hess[i]))

        # Other (not batched)
        for i, (idx, hess) in enumerate(other_data):
            if other_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        # Rotations (not batched)
        for i, (idx, hess) in enumerate(rot_data):
            if rot_active[i]:
                hessians.append(SparseInternalHessian(n_atoms, idx, hess))

        result = SparseInternalHessians(hessians, self.ndof,
                                        skeleton=self._get_hessian_skeleton(hessians))
        self._cache['hessian_result'] = result
        return result

    def hessian_rdot(self, v: np.ndarray):
        """Compute Hessian @ v for all internal coordinates using direct HVP.

        This computes the same result as hessian().rdot(v) but uses forward-over-reverse
        mode autodiff (jvp(grad(f))) to compute Hessian-vector products directly,
        avoiding the O(n²) cost of materializing full Hessian matrices.

        Args:
            v: Vector of shape (ndof,) to multiply with each coordinate's Hessian

        Returns:
            Sparse CSR matrix of shape (n_active_coords, ndof) where each row
            is H_i @ v. Returns dense ndarray as fallback when not all
            coordinates are active.
        """
        self._cache_check()
        positions = self.all_positions
        cell = self.atoms.cell.array
        self._build_batched_arrays()
        tvecs = self._get_cached_tvecs(cell)

        # Reshape v for easy indexing
        v_atoms = v.reshape((-1, 3))  # (n_atoms, 3)
        n_atoms = self.natoms + self.ndummies
        ndof = self.ndof  # Cache to avoid repeated property lookups

        # Get active mask and counts
        active_mask = self._active_mask
        n_trans = len(self.internals['translations'])
        n_bonds = len(self.internals['bonds'])
        n_angles = len(self.internals['angles'])
        n_dihedrals = len(self.internals['dihedrals'])
        n_other = len(self.internals['other'])
        n_rot = len(self.internals['rotations'])

        start = 0
        trans_active = active_mask[start:start+n_trans]
        start += n_trans
        bonds_active = np.array(active_mask[start:start+n_bonds], dtype=bool)
        start += n_bonds
        angles_active = np.array(active_mask[start:start+n_angles], dtype=bool)
        start += n_angles
        dihedrals_active = np.array(active_mask[start:start+n_dihedrals], dtype=bool)
        start += n_dihedrals
        other_active = active_mask[start:start+n_other]
        start += n_other
        rot_active = active_mask[start:start+n_rot]

        n_active = sum(active_mask)

        # Fast path: when all coords are active, use pre-built CSR structure
        use_sparse = (n_active == self._csr_n_active)

        if use_sparse:
            data = self._csr_data
            data[:] = 0
        else:
            if (self._hvp_buf is None
                    or self._hvp_buf.shape != (n_active, ndof)):
                self._hvp_buf = np.zeros((n_active, ndof))
            out = self._hvp_buf
            out[:] = 0

        row = 0  # Current write position in output

        # Translations - Hessian is zero
        n_active_trans = sum(trans_active)
        # out[row:row+n_active_trans] is already zero from the clear
        row += n_active_trans

        # Launch all JAX HVP computations, deferring device_get
        # This allows JAX to pipeline the computations before we block on transfer

        bond_jax_result = None
        bond_active_idx = None
        if bonds_active.any() and self._n_bonds_actual > 0:
            if bonds_active.all():
                bond_pos = positions[self._bond_indices_padded]
                bond_tvecs = tvecs['bonds_padded']
                v_sub = v_atoms[self._bond_indices_padded]
                bond_jax_result = _bond_hvp_batched(bond_pos, bond_tvecs, v_sub)
                bond_active_idx = self._bond_indices
            else:
                bond_active_idx = self._bond_indices[bonds_active]
                bond_pos = positions[bond_active_idx]
                bond_tvecs = tvecs['bonds'][bonds_active]
                v_sub = v_atoms[bond_active_idx]
                bond_jax_result = _bond_hvp_batched(bond_pos, bond_tvecs, v_sub)

        angle_jax_result = None
        angle_active_idx = None
        if angles_active.any() and self._n_angles_actual > 0:
            if angles_active.all():
                angle_pos = positions[self._angle_indices_padded]
                angle_tvecs = tvecs['angles_padded']
                v_sub = v_atoms[self._angle_indices_padded]
                angle_jax_result = _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)
                angle_active_idx = self._angle_indices
            else:
                angle_active_idx = self._angle_indices[angles_active]
                angle_pos = positions[angle_active_idx]
                angle_tvecs = tvecs['angles'][angles_active]
                v_sub = v_atoms[angle_active_idx]
                angle_jax_result = _angle_hvp_batched(angle_pos, angle_tvecs, v_sub)

        dih_jax_result = None
        dih_active_idx = None
        if dihedrals_active.any() and self._n_dihedrals_actual > 0:
            if dihedrals_active.all():
                dih_pos = positions[self._dihedral_indices_padded]
                dih_tvecs = tvecs['dihedrals_padded']
                v_sub = v_atoms[self._dihedral_indices_padded]
                dih_jax_result = _dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub)
                dih_active_idx = self._dihedral_indices
            else:
                dih_active_idx = self._dihedral_indices[dihedrals_active]
                dih_pos = positions[dih_active_idx]
                dih_tvecs = tvecs['dihedrals'][dihedrals_active]
                v_sub = v_atoms[dih_active_idx]
                dih_jax_result = _dihedral_hvp_batched(dih_pos, dih_tvecs, v_sub)

        # Compute rotation HVPs using closed-form Hessian (handles
        # degenerate eigenvalues for linear/near-linear fragments).
        rot_closed_results = []
        rot_batched_slots = None
        rot_batched_frag_indices = None
        rot_batched_hvp = None
        all_rot_active = bool(np.asarray(rot_active, dtype=bool).all())
        if all_rot_active and self.internals['rotations']:
            pos_pad, ref_pad, mask, frag_indices, slots, valid = (
                self._rotation_padded_inputs(positions)
            )
        else:
            valid = False
        if valid:
            qs = self._get_stabilized_quaternions(positions)
            q_stable_all = np.array(qs) if qs is not None else None
            cached_eigh = self._cache.get('stabilized_q_eigh', (None, None))
            ws_cached, vecs_cached = cached_eigh
            n_max = mask.shape[1]
            v_pad = np.zeros((len(frag_indices), n_max, 3), dtype=np.float64)
            for fi, fi_idx in enumerate(frag_indices):
                v_pad[fi, :len(fi_idx)] = v_atoms[fi_idx]
            rot_batched_hvp = _rotation_3axis_hvp_batched_closed(
                pos_pad, ref_pad, mask, v_pad,
                q_stable_all=q_stable_all,
                ws_all=ws_cached, vecs_all=vecs_cached,
            )
            rot_batched_slots = slots
            rot_batched_frag_indices = frag_indices
        else:
            for i, coord in enumerate(self.internals['rotations']):
                if rot_active[i]:
                    idx = np.array(coord.indices)
                    pos = positions[idx]
                    v_sub = v_atoms[idx]
                    axis = coord.kwargs['axis']
                    refpos = coord.kwargs['refpos']
                    hvp = _rotation_hvp_closed(pos, axis, refpos, v_sub,
                                               q_stable=coord.q_prev)
                    rot_closed_results.append((hvp, idx))

        # Now collect results with device_get and scatter into output

        if bond_jax_result is not None:
            hvp = np.asarray(device_get(bond_jax_result))
            if bonds_active.all():
                hvp = hvp[:self._n_bonds_actual]
            n_coords = self._n_bonds_actual if bonds_active.all() else int(bonds_active.sum())
            if use_sparse:
                off = self._csr_bond_offset
                data[off:off + n_coords * 6] = hvp.reshape(-1)
            else:
                flat_cols = self._bond_flat_cols if bonds_active.all() else self._bond_flat_cols[bonds_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        if angle_jax_result is not None:
            hvp = np.asarray(device_get(angle_jax_result))
            if angles_active.all():
                hvp = hvp[:self._n_angles_actual]
            n_coords = self._n_angles_actual if angles_active.all() else int(angles_active.sum())
            if use_sparse:
                off = self._csr_angle_offset
                data[off:off + n_coords * 9] = hvp.reshape(-1)
            else:
                flat_cols = self._angle_flat_cols if angles_active.all() else self._angle_flat_cols[angles_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        if dih_jax_result is not None:
            hvp = np.asarray(device_get(dih_jax_result))
            if dihedrals_active.all():
                hvp = hvp[:self._n_dihedrals_actual]
            n_coords = self._n_dihedrals_actual if dihedrals_active.all() else int(dihedrals_active.sum())
            if use_sparse:
                off = self._csr_dih_offset
                data[off:off + n_coords * 12] = hvp.reshape(-1)
            else:
                flat_cols = self._dihedral_flat_cols if dihedrals_active.all() else self._dihedral_flat_cols[dihedrals_active]
                out[row:row+n_coords, :] = 0
                out[np.arange(row, row+n_coords)[:, None], flat_cols] = hvp.reshape(n_coords, -1)
            row += n_coords

        # Other - use existing hessian computation (typically few coords, loop is fine)
        atoms = self.light_atoms
        off = self._csr_other_offset if use_sparse else 0
        for i, coord in enumerate(self.internals['other']):
            if other_active[i]:
                hess = np.array(coord.calc_hessian(atoms))
                idx = np.array(coord.indices)
                v_sub = v_atoms[idx]
                hvp = np.einsum('aibj,bj->ai', hess, v_sub)
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1

        # Rotations - collect results from closed-form Hessian (no NaN
        # for degenerate eigenvalues)
        if rot_batched_hvp is not None:
            hvp_padded = rot_batched_hvp
            # hvp_padded.shape == (n_frags, 3, N_max, 3)
            ordered = [None] * len(self.internals['rotations'])
            for fi, slot in enumerate(rot_batched_slots):
                n = len(rot_batched_frag_indices[fi])
                for axis, rot_idx in enumerate(slot):
                    ordered[rot_idx] = (
                        hvp_padded[fi, axis, :n, :],
                        rot_batched_frag_indices[fi],
                    )
            for i, coord in enumerate(self.internals['rotations']):
                if not rot_active[i]:
                    continue
                hvp, idx = ordered[i]
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1
        else:
            for hvp, idx in rot_closed_results:
                if use_sparse:
                    dense_row = np.zeros(ndof)
                    dense_row.reshape((-1, 3))[idx] = hvp
                    data[off:off + ndof] = dense_row
                    off += ndof
                else:
                    out_row = out[row].reshape((-1, 3))
                    out_row[idx] = hvp
                row += 1

        if use_sparse:
            return sparse.csr_matrix(
                (data, self._csr_indices, self._csr_indptr),
                shape=(self._csr_n_active, ndof), copy=False,
            )
        return out[:row]

    def wrap(self, vec: np.ndarray) -> np.ndarray:
        """Wraps an internal coord. displacement vector into a valid domain."""
        start = 0
        for name in self._names:
            n = len(self.internals[name])
            if name == 'dihedrals':
                vec[start:start + n] = (vec[start:start + n] + np.pi) % (2 * np.pi) - np.pi
            elif name == 'rotations' and n > 0:
                self._wrap_rotation_diff(vec, start)
            start += n
        return vec

    def _wrap_rotation_diff(self, vec, rot_start):
        """Wrap rotation coordinate differences along rotation axis.

        The exponential map has period 2π along the rotation axis
        direction. For each fragment's 3 rotation components, find the
        minimum-image difference by adding/subtracting 2π * v̂.
        """
        rotations = self.internals['rotations']
        if not rotations:
            return
        # Group rotations by fragment (same indices and refpos)
        groups = {}
        for i, r in enumerate(rotations):
            key = (tuple(r.indices), r.kwargs['refpos'].tobytes())
            groups.setdefault(key, []).append(i)

        for key, indices in groups.items():
            if len(indices) != 3:
                continue
            # Get the 3-component rotation difference vector
            idx = [rot_start + i for i in indices]
            v = vec[idx].copy()
            vnorm = np.linalg.norm(v)
            if vnorm < 1e-10:
                continue
            vh = v / vnorm
            # Try adding/subtracting 2π along v̂ to minimize |v|
            best_v = v.copy()
            best_d2 = np.dot(v, v)
            for direction in [1, -1]:
                vt = v.copy()
                while True:
                    vt += direction * 2 * np.pi * vh
                    d2 = np.dot(vt, vt)
                    if d2 >= best_d2:
                        break
                    best_v = vt.copy()
                    best_d2 = d2
            vec[idx] = best_v

    def __iter__(self) -> Iterator[Coordinate]:
        for name in self._names:
            for coord in self.internals[name]:
                yield coord

    def _get_neighbors(self, dx: np.ndarray) -> Iterator[np.ndarray]:
        pbc = self.atoms.pbc
        if self.cell is None or not np.allclose(self.cell, self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
            self._rcell_reciprocal_T = self.rcell.reciprocal().T
        dx_sc = dx @ self._rcell_reciprocal_T
        offset = np.zeros(3, dtype=np.int32)
        for _ in range(2):
            offset += pbc * ((dx_sc - offset) // 1.).astype(np.int32)

        for ts in product(*[np.arange(-1 * p, p + 1) for p in pbc]):
            yield (np.array(ts) - offset) @ self.op

    def _find_mic(self, indices: Tuple[int, ...]) -> np.ndarray:
        ncvecs = np.zeros((len(indices) - 1, 3), dtype=np.int32)
        if not np.any(self.atoms.pbc):
            return ncvecs

        pos = self.all_positions
        dxs = np.array([
            pos[i] - pos[j] for i, j in zip(indices[1:], indices[:-1])
        ])

        for dx, ncvec in zip(dxs, ncvecs):
            vlen = np.inf
            for neighbor in self._get_neighbors(dx):
                trial = np.linalg.norm(dx + neighbor @ self.atoms.cell)
                if trial < vlen:
                    vlen = trial
                    ncvec[:] = neighbor
        return ncvecs

    def _get_ncvecs(
        self,
        indices: Tuple[int, ...],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None
    ) -> np.ndarray:
        if ncvecs is None:
            if mic is None or not mic:
                return np.zeros((len(indices) - 1, 3), dtype=np.int32)
            else:
                return self._find_mic(indices)
        else:
            if mic:
                raise ValueError(
                    "Minimum image convention (mic) requested, but explicit "
                    "periodic vectors (ncvecs) were also provided! These "
                    "keyword arguments are mutually exclusive."
                )
            return np.asarray(
                ncvecs,
                dtype=np.int32
            ).reshape((len(indices) - 1, 3))


    def add_dummy_to_internals(
        self,
        idx: int
    ) -> None:
        didx = self.dinds[idx]
        assert didx >= 0
        npos = len(self.all_positions)
        for i, trans in enumerate(self.internals['translations']):
            if idx in trans.indices and didx not in trans.indices:
                new_indices = (*trans.indices, didx)
                new_trans = Translation(new_indices, trans.kwargs['dim'])
                self.internals['translations'][i] = new_trans

        for i, rot in enumerate(self.internals['rotations']):
            if idx in rot.indices and didx not in rot.indices:
                new_indices = np.array((*rot.indices, didx), dtype=np.int32)
                if np.all(new_indices < npos):
                    new_rot = Rotation(
                        new_indices, rot.kwargs['axis'],
                        self.all_positions[new_indices]
                    )
                    self.internals['rotations'][i] = new_rot



class Constraints(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        dinds: np.ndarray = None,
        ignore_rotation: bool = True,
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self._targets = {key: [] for key in self._names}
        self._kind = {key: [] for key in self._names}
        self.ignore_rotation = ignore_rotation
        for ase_cons in atoms.constraints:
            self.merge_ase_constraint(ase_cons)

    def copy(self) -> 'Constraints':
        new = self.__class__(
            self.atoms, self.dummies, self.dinds, self.ignore_rotation
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new._targets[name] = self._targets[name].copy()
            new._active[name] = self._active[name].copy()
            new._kind[name] = self._kind[name].copy()
        return new

    @property
    def targets(self) -> np.ndarray:
        vec = []
        for key in self._names:
            vec += self._targets[key]
        return np.array(vec, dtype=np.float64)[self._active_indices]

    def residual(self) -> np.ndarray:
        """Calculates the constraint residual vector."""
        res = self.wrap(self.calc() - self.targets)
        if self.ignore_rotation and self.nrotations:
            res[-self.nrotations:] = 0.
        return res

    def has_inequalities(self) -> bool:
        """Check if any inequality constraints (lt/gt) exist."""
        for name in self._names:
            for kind in self._kind[name]:
                if kind in ('lt', 'gt'):
                    return True
        return False

    def disable_satisfied_inequalities(self) -> None:
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if kind == 'lt' and coord.calc(self.all_atoms) <= target:
                    active = False
                elif kind == 'gt' and coord.calc(self.all_atoms) >= target:
                    active = False
                else:
                    active = True
                self._active[name][i] = active

    def validate_inequalities(self) -> bool:
        all_valid = True
        for name in self._names:
            for i, (coord, kind, target) in enumerate(zip(
                self.internals[name], self._kind[name], self._targets[name]
            )):
                if self._active[name][i]:
                    continue
                if kind == 'lt' and coord.calc(self.all_atoms) > target:
                    self._active[name][i] = True
                    all_valid = False
                elif kind == 'gt' and coord.calc(self.all_atoms) < target:
                    self._active[name][i] = True
                    all_valid = False
        return all_valid

    def fix_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.asarray(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.fix_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_positions[indices]
            )
        try:
            _ = self.internals['rotations'].index(new)
        except ValueError:
            self.internals['rotations'].append(new)
            self._targets['rotations'].append(0.)
            self._active['rotations'].append(True)
            self._kind['rotations'].append('eq')
        else:
            raise DuplicateConstraintError(
                "This rotation has already been constrained!"
            )

    def fix_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None,
        target: float = None,
        replace_ok: bool = True,
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Translation'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            if np.isscalar(index):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                if target is not None:
                    raise ValueError(
                        '"target" keyword requires explicit "dim"!'
                    )
                for dim in range(3):
                    self.fix_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if target is None:
            target = new.calc(self.all_atoms)
        try:
            idx = self.internals['translations'].index(new)
        except ValueError:
            self.internals['translations'].append(new)
            self._targets['translations'].append(target)
            self._active['translations'].append(True)
            self._kind['translations'].append('eq')
        else:
            if replace_ok:
                self._targets['translations'][idx] = target
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets['translations'][idx])
            )

    def _fix_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        conv: float,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
        target: float = None,
        comparator: str = 'eq',
        replace_ok: bool = True,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        if target is None:
            target = new.calc(self.all_atoms)
        else:
            target *= conv
        try:
            idx = self.internals[name].index(new)
        except ValueError:
            self.internals[name].append(new)
            self._targets[name].append(target)
            self._active[name].append(True)
            self._kind[name].append(comparator)
        else:
            if replace_ok:
                self._targets[name][idx] = target
                self._kind[name][idx] = comparator
                return
            raise DuplicateConstraintError(
                "Coordinate {} is already fixed to target {}"
                .format(new, self._targets[name][idx] / conv)
            )

    fix_bond = partialmethod(_fix_internal, Bond, 'bonds', 1.)
    fix_angle = partialmethod(_fix_internal, Angle, 'angles', np.pi / 180.)
    fix_dihedral = partialmethod(
        _fix_internal, Dihedral, 'dihedrals', np.pi / 180.
    )


    def merge_ase_constraint(self, ase_cons: FixConstraint) -> None:
        if isinstance(ase_cons, FixAtoms):
            for index in ase_cons.index:
                try:
                    self.fix_translation(index)
                except DuplicateConstraintError:
                    pass
        elif isinstance(ase_cons, FixCom):
            try:
                self.fix_translation()
            except DuplicateConstraintError:
                pass
        elif isinstance(ase_cons, FixBondLengths):
            for i, indices in enumerate(ase_cons.pairs):
                if ase_cons.bondlengths is None:
                    target = None
                else:
                    target = ase_cons.bondlengths[i]
                try:
                    self.fix_bond(indices, mic=True, target=target)
                except DuplicateConstraintError:
                    pass
            return
        elif isinstance(ase_cons, FixCartesian):
            for dim, relaxed in enumerate(ase_cons.mask):
                if relaxed:
                    continue
                try:
                    self.fix_translation(ase_cons.a, dim=dim)
                except DuplicateConstraintError:
                    pass
        elif isinstance(ase_cons, FixInternals):
            for ase_cons_list, adder in zip(
                (ase_cons.bonds, ase_cons.angles, ase_cons.dihedrals),
                (self.fix_bond, self.fix_angle, self.fix_dihedral),
            ):
                for target, indices in ase_cons_list:
                    try:
                        adder(indices, target=target)
                    except DuplicateInternalError:
                        pass
            if ase_cons.bondcombos:
                raise RuntimeError(
                    "Sella currently does not support combination constraints."
                )
        else:
            raise RuntimeError(
                "Sella does not currently implement the ASE {} Constraint "
                "class.".format(ase_cons.__class__.__name__)
            )

# Reference radii of the model-Hessian stiffness formulas (the Almlof and
# Fischer-Almlof exponentials in _h0_bond, _h0_angle, _h0_dihedral, the
# torsion-class bond order and the non-local pair term): the covalent radii,
# except that the group-1/2 metals use their Shannon ionic radius (CN 6).
# The Almlof stretch curvature Ab exp(-Bb (r - r_ref)) is that of a covalent
# single bond (0.36 Ha/Bohr^2) at r = r_ref.  The Cordero "covalent" radius of
# an s-block metal is calibrated on its ionic contacts (Na 1.66 + O 0.66 =
# 2.32 A is the Na+...O distance, Li 1.28 + C 0.76 = 2.04 ~ the Li+...C(pi)
# distance), so every metal-ligand contact -- and every metal-ligand "bond"
# of a connected system -- gets 0.1-0.4 Ha/Bohr^2, 5-15x the curvature of an
# ion-dipole or cation-pi contact (0.02-0.07 Ha/Bohr^2 from the 200-500 cm^-1
# M+...OH2 stretches of hydrated alkali and alkaline-earth ions); the cage of
# radial pair terms around a cation on a pi face is then also ~10x too stiff
# sideways, and the tracked pair term re-imposes it at every geometry.  With
# the ionic radius the same formula gives 0.03-0.06 Ha/Bohr^2 at the observed
# contact distances.  Halide anions are the opposite case (their covalent
# radius is the C-X one; Cl-...H at 2.1 A is already 0.8 A outside r_ref) and
# keep the covalent radius.  The bond criterion of find_all_bonds
# (connectivity, fragments, the start displacement) keeps the covalent radii,
# so the coordinate systems are unchanged.
_STIFFNESS_RADII = np.array(covalent_radii, dtype=np.float64)
for _z, _r in ((3, 0.76), (11, 1.02), (19, 1.38), (37, 1.52), (55, 1.67),
               (4, 0.45), (12, 0.72), (20, 1.00), (38, 1.18), (56, 1.35)):
    _STIFFNESS_RADII[_z] = _r
del _z, _r

# Row factors of the Almlof stretch curvature for the covalent bonds of the
# p-block elements beyond the second row (Al-Ar, Ga-Kr, In-Xe, Tl-Rn).  The
# single exponential gives every bond 0.36 Ha/Bohr^2 = 5.6 mdyn/A at
# r = r_ref, which is right for the first-row bonds (C-C 4.5, C-N 5.0-5.5,
# C-O 5.0-5.4, C-F 5.9, C-H 5.0, N-H 6.4, O-H 7.8 mdyn/A), but the bonds of
# the heavier p-block elements are soft for their length (diffuse valence
# shells): C-Si 2.9, C-P 3.0, C-S 3.1, C-Cl 3.4, C-Br 2.9, C-I 2.3, Si-Cl
# 3.0, P-Cl 2.5, S-S 2.5, Cl-Cl 3.2, Br-Br 2.5, I-I 1.7 mdyn/A -- 1.6-5x
# below the exponential, the homonuclear ones also sitting inside r_ref
# where it grows.  One factor per element row, applied once per atom of
# the bond (product), reproduces the C-X values (row 3: 3.2-3.3, row 4:
# 3.0, row 5: 2.4 mdyn/A) and leaves the X-X and H-X bonds 10-30 % soft
# (S-S 2.3, Cl-Cl 2.3, Br-Br 2.2, I-I 1.5, Si-Cl 2.8, P-Cl 2.3, S-H 3.3,
# Si-H 2.6, H-Cl 4.1, H-Br 4.1, H-I 3.3; Si-F 5.3, Si-O 5.4, S=O 9.1, C=S
# 6.7 against 6.4, 5-6, 10.0, 7.5).  The s-block keeps the ionic-radius
# treatment above, and the d-block is left alone.  The factor applies to
# the bonds of connected systems only: the cost of a molecular complex is
# set by its intermolecular coordinates, and the intramolecular paths of
# its fragments (and the basins they reach) stay as they are.
_STIFFNESS_SCALE = np.ones(len(covalent_radii), dtype=np.float64)
for (_z0, _z1), _f in (((13, 18), 0.58), ((31, 36), 0.50), ((49, 54), 0.42),
                       ((81, 86), 0.35)):
    _STIFFNESS_SCALE[_z0:_z1 + 1] = _f
del _z0, _z1, _f

class Internals(BaseInternals):
    def __init__(
        self,
        atoms: Atoms,
        dummies: Atoms = None,
        atol: float = 15.,
        dinds: np.ndarray = None,
        cons: Constraints = None,
        allow_fragments: bool = False
    ) -> None:
        BaseInternals.__init__(self, atoms, dummies, dinds)
        self.atol = atol * np.pi / 180.
        self.forbidden = {key: [] for key in self._names}
        if cons is None:
            cons = Constraints(self.atoms, self.dummies, self.dinds)
        else:
            if (
                (dummies is not None and dummies is not cons.dummies)
                or (dinds is not None and dinds is not cons.dinds)
            ):
                raise RuntimeError(
                    "Constraints has inconsistent dummy atom definitions!"
                )
            self.dummies = cons.dummies
            self.dinds = cons.dinds
        self.cons = cons

        for kind, adder in zip(self._names, (
            self.add_translation, self.add_bond, self.add_angle,
            self.add_dihedral, self.add_other, self.add_rotation
        )):
            for coord in self.cons.internals[kind]:
                adder(coord)
        self.allow_fragments = allow_fragments
        self.fragment_atom_groups = None

    def copy(self) -> 'Internals':
        new = self.__class__(
            self.atoms,
            self.dummies,
            self.atol * 180. / np.pi,
            self.dinds,
            self.cons.copy(),
            self.allow_fragments,
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new._internals_set[name] = self._internals_set[name].copy()
            new.forbidden[name] = self.forbidden[name].copy()
            new._active[name] = self._active[name].copy()
        return new

    def shadow_copy(self) -> 'Internals':
        """A copy with its own Atoms objects (no calculator attached) whose
        positions can be set freely, sharing the coordinate lists and the
        covalent graph: InternalPES evaluates the analytic part of the
        model Hessian at interior points of a step with it.  The fragment
        rotations get their own Rotation objects (same indices, axis and
        reference), because evaluating one updates its quaternion-branch
        state; sync_rotation_state copies that state over before use."""
        atoms = self.atoms.copy()
        atoms.calc = None
        dummies = self.dummies.copy()
        new = self.__class__(
            atoms,
            dummies,
            self.atol * 180. / np.pi,
            self.dinds,
            None,
            self.allow_fragments,
        )
        for name in self._names:
            new.internals[name] = self.internals[name].copy()
            new._internals_set[name] = self._internals_set[name].copy()
            new.forbidden[name] = self.forbidden[name].copy()
            new._active[name] = self._active[name].copy()
        rotations = []
        for rot in self.internals['rotations']:
            new_rot = Rotation(rot.indices, rot.kwargs['axis'],
                               rot.kwargs['refpos'])
            new_rot.kwargs['refpos'] = rot.kwargs['refpos'].copy()
            rotations.append(new_rot)
        new.internals['rotations'] = rotations
        new.sync_rotation_state(self)
        new.fragment_atom_groups = self.fragment_atom_groups
        return new

    def sync_rotation_state(self, src: 'Internals') -> None:
        """Copy the quaternion-branch state of the fragment rotations of
        src (a shadow_copy source with the same rotation list)."""
        mine = self.internals['rotations']
        theirs = src.internals['rotations']
        if len(mine) != len(theirs):
            return
        for rot, srot in zip(mine, theirs):
            q = getattr(srot, 'q_prev', None)
            rot.q_prev = None if q is None else np.array(q, copy=True)

    def symmetry_maps(self, tol):
        """Point-group operations of the current geometry (per-atom rmsd of
        the mapped geometry within tol) as maps of the extended (atoms +
        dummies) Cartesian displacements, for the symmetry images of the
        secant pairs: a list of (perm, R, dperm, dflip, dcentre) with
        u'_i = R u_{perm[i]} for the atoms and, for the dummy k of the
        linear centre dcentre[k], u'_k = R u_{dperm[k]} when the operation
        maps the dummy of the image centre onto the dummy itself, or
        u'_k = 2 R u_{perm[dcentre[k]]} - R u_{dperm[k]} when it maps it
        onto its inversion through the centre (the dummy is placed
        perpendicular to the chain, so the inverted dummy satisfies the
        same bond and angle constraints, and the inversion through the
        centre turns the mirrored dummy path into a path in the frame of
        the current dummy).  An operation that maps a dummy elsewhere (a
        chain on a rotation axis, whose dummy has no symmetry-related
        azimuth) is dropped, as is one that maps a linear centre onto a
        centre without a dummy.  Multi-fragment systems are treated alike:
        the operations are those of the whole complex (an ion on the
        mirror plane of its partner, a planar hydrogen-bonded pair, a
        homodimer with a C2 or an inversion centre), and the fragment
        translation and rotation coordinates are functions of the
        Cartesian positions like every other internal, so the image map
        through Cartesian space (InternalPES._symmetry_images, with the
        current Jacobian and its pseudo-inverse) applies to them
        unchanged.  A lone-atom fragment enters the permutation search
        like any atom."""
        if self.internals['other']:
            return []
        nat = self.natoms
        if nat < 3:
            return []
        pos = self.atoms.positions
        ops = _symmetry_operations(self.atoms.numbers, pos, tol)
        if not ops:
            return []
        ndum = self.ndummies
        dpos = self.dummies.positions if ndum else None
        dinds = self.dinds
        dcentre = -np.ones(ndum, dtype=np.int64)
        for j in range(nat):
            k = int(dinds[j])
            if k >= 0:
                dcentre[k - nat] = j
        maps = []
        for perm, det, R, c in ops:
            dperm = -np.ones(ndum, dtype=np.int64)
            dflip = np.ones(ndum, dtype=np.float64)
            ok = True
            for k in range(ndum):
                j = int(dcentre[k])
                if j < 0:
                    ok = False
                    break
                kk = int(dinds[perm[j]])
                if kk < 0:
                    ok = False
                    break
                xk = dpos[k]
                r = R @ (dpos[kk - nat] - c) + c
                if np.linalg.norm(r - xk) <= 3.0 * tol:
                    pass
                elif np.linalg.norm(r - (2.0 * pos[j] - xk)) <= 3.0 * tol:
                    dflip[k] = -1.0
                else:
                    ok = False
                    break
                dperm[k] = kk - nat
            if not ok:
                continue
            maps.append((perm, R, dperm, dflip, dcentre))
        return maps

    def add_rotation(
        self,
        indices: Union[Tuple[int, ...], Rotation] = None,
        axis: int = None,
    ) -> None:
        if isinstance(indices, Rotation):
            if axis is not None:
                raise ValueError(
                    "'axis' keyword cannot be used with explicit Rotation"
                )
            new = indices
        else:
            if indices is None:
                indices = np.arange(len(self.all_atoms), dtype=np.int32)
            indices = np.array(indices, dtype=np.int32)
            if axis is None:
                for axis in range(3):
                    self.add_rotation(indices, axis)
                return
            new = Rotation(
                indices,
                axis,
                self.all_positions[indices]
            )
        if (
            new in self.internals['rotations']
            or new in self.forbidden['rotations']
        ):
            raise DuplicateInternalError
        self.internals['rotations'].append(new)
        self._active['rotations'].append(True)

    def add_translation(
        self,
        index: Union[int, Tuple[int, ...], Translation] = None,
        dim: int = None
    ) -> None:
        if isinstance(index, Translation):
            if dim is not None:
                raise ValueError(
                    '"dim" keyword cannot be used with explicit Cart'
                )
            new = index
        else:
            if index is None:
                index = np.arange(len(self.all_atoms), dtype=np.int32)
            elif isinstance(index, int):
                index = np.array((index,), dtype=np.int32)
            if dim is None:
                for dim in range(3):
                    self.add_translation(index, dim=dim)
                return
            new = Translation(index, dim)
        if (
            new in self.internals['translations']
            or new in self.forbidden['translations']
        ):
            raise DuplicateInternalError
        self.internals['translations'].append(new)
        self._active['translations'].append(True)

    def _add_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        key = (tuple(new.indices), tuple(map(tuple, new.kwargs['ncvecs'])))
        if (
            key in self._internals_set[name]
            or new in self.forbidden[name]
        ):
            raise DuplicateInternalError
        self.internals[name].append(new)
        self._internals_set[name].add(key)
        self._active[name].append(True)

    add_bond = partialmethod(_add_internal, Bond, 'bonds')
    add_angle = partialmethod(_add_internal, Angle, 'angles')
    add_dihedral = partialmethod(_add_internal, Dihedral, 'dihedrals')

    def add_other(
        self,
        coord: Coordinate,
    ) -> None:
        try:
            self.internals['other'].index(coord)
        except ValueError:
            self.internals['other'].append(coord)
            self._active['other'].append(True)
        else:
            raise DuplicateInternalError()


    def _forbid_internal(
        self,
        kind: TypeVar('Coordinate', bound=Coordinate),
        name: str,
        indices: Union[Tuple[int, ...], Coordinate],
        ncvecs: Tuple[IVec, ...] = None,
        mic: bool = None,
    ) -> None:
        if isinstance(indices, kind):
            if ncvecs is not None or mic is not None:
                raise ValueError(
                    '"ncvecs" and "mic" keywords cannot be used '
                    'with explicit {}'.format(kind.__name__)
                )
            new = indices
        else:
            ncvecs = self._get_ncvecs(indices, ncvecs, mic)
            new = kind(indices, ncvecs=ncvecs)
        try:
            self.forbidden[name].remove(new)
        except ValueError:
            pass
        if new not in self.forbidden[name]:
            self.forbidden[name].append(new)

    forbid_bond = partialmethod(_forbid_internal, Bond, 'bonds')
    forbid_angle = partialmethod(_forbid_internal, Angle, 'angles')
    forbid_dihedral = partialmethod(_forbid_internal, Dihedral, 'dihedrals')

    @staticmethod
    def flood_fill(
        index: int,
        nbonds: np.ndarray,
        c10y: np.ndarray,
        labels: np.ndarray,
        label: int
    ) -> None:
        for j in c10y[index, :nbonds[index]]:
            if labels[j] != label:
                labels[j] = label
                Internals.flood_fill(j, nbonds, c10y, labels, label)

    def _find_bonds_vectorized(self, labels, scale, rcov):
        """Vectorized bond search across all candidate atom pairs.

        Returns a list of (i, j, ts) tuples for bonds that pass the
        distance threshold, where ts is the integer translation vector.
        """
        natoms = self.natoms
        pos = self.atoms.positions
        cell = self.atoms.cell.array
        pbc = self.atoms.pbc

        # Ensure cell/rcell/op are cached
        if self.cell is None or not np.allclose(self.cell, self.atoms.cell):
            self.cell = self.atoms.cell.array.copy()
            rcell, self.op = minkowski_reduce(
                complete_cell(self.cell), pbc=pbc
            )
            self.rcell = Cell(rcell)
            self._rcell_reciprocal_T = self.rcell.reciprocal().T

        # 1. Generate all candidate pairs (i <= j)
        ii, jj = np.triu_indices(natoms, k=0)
        # Skip pairs in the same labeled fragment
        same_frag = (labels[ii] == labels[jj]) & (labels[ii] != -1)
        keep = ~same_frag
        ii, jj = ii[keep], jj[keep]

        if len(ii) == 0:
            return []

        # 2. All pairwise displacements
        dx = pos[jj] - pos[ii]  # (n_pairs, 3)

        # 3. Pair-dependent offsets (vectorized _get_neighbors logic)
        dx_sc = dx @ self._rcell_reciprocal_T
        offset = np.zeros(dx_sc.shape, dtype=np.int32)
        for _ in range(2):
            offset += (pbc * ((dx_sc - offset) // 1.)).astype(np.int32)

        # 4. Base translation vectors from PBC dimensions
        ranges = [np.arange(-1 * p, p + 1) for p in pbc]
        base_ts = np.array(
            list(product(*ranges)), dtype=np.int32
        )  # (n_ts, 3)

        # 5. Shifted translations and Cartesian vectors
        shifted = base_ts[None, :, :] - offset[:, None, :]  # (n_pairs, n_ts, 3)
        tvecs_cart = (shifted @ self.op) @ cell  # (n_pairs, n_ts, 3)

        # 6. Distances
        dists = np.linalg.norm(
            dx[:, None, :] + tvecs_cart, axis=2
        )  # (n_pairs, n_ts)

        # 7. Covalent radius threshold
        thresholds = scale * (rcov[ii] + rcov[jj])
        bond_mask = dists <= thresholds[:, None]

        # 8. Exclude self-bonds (i==j) with zero translation
        self_bond = (ii == jj)
        zero_ts = np.all(shifted @ self.op == 0, axis=2)
        bond_mask &= ~(self_bond[:, None] & zero_ts)

        # 9. Collect hits
        pair_idx, ts_idx = np.nonzero(bond_mask)
        op = self.op
        results = []
        for k in range(len(pair_idx)):
            p = pair_idx[k]
            t = ts_idx[k]
            ts = (shifted[p, t] @ op).astype(np.int32)
            results.append((int(ii[p]), int(jj[p]), ts))
        return results

    def _wrap_fragment_positions(self, group, cumshifts):
        """Shift atom positions so fragment atoms are contiguous across PBC.

        BFS from first atom in group, using bond ncvecs to bring each
        bonded neighbor into the same periodic image. Accumulates shifts
        along bond chains so molecules spanning multiple cell boundaries
        are fully contracted. Records cumulative shifts in cumshifts dict
        for subsequent ncvec correction.
        """
        group_set = set(group)
        cell = np.asarray(self.atoms.cell)

        adj = {i: [] for i in group}
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i in group_set and j in group_set:
                ncvec = bond.kwargs['ncvecs'][0]
                adj[i].append((j, ncvec))
                adj[j].append((i, -ncvec))

        anchor = group[0]
        cumshifts[anchor] = np.zeros(3, dtype=int)
        queue = [anchor]
        while queue:
            i = queue.pop(0)
            for j, ncvec in adj[i]:
                if j in cumshifts:
                    continue
                cumshifts[j] = ncvec + cumshifts[i]
                self.atoms.positions[j] += cumshifts[j] @ cell
                queue.append(j)

    def find_all_bonds(
        self,
        nbond_cart_thr: int = 6,
        max_bonds: int = 20,
        scale: float = 1.25,
    ) -> None:
        rcov = covalent_radii[self.atoms.numbers]
        nbonds = np.zeros(self.natoms, dtype=np.int32)
        labels = -np.ones(self.natoms, dtype=np.int32)
        c10y = -np.ones((self.natoms, max_bonds), dtype=np.int32)

        for bond in self.internals['bonds']:
            i, j = bond.indices
            c10y[i, nbonds[i]] = j
            nbonds[i] += 1
            c10y[j, nbonds[j]] = i
            nbonds[j] += 1

        first_run = True
        while True:
            # use flood fill algorithm to count the number of disconnected
            # fragments
            nlabels = 0
            labels[:] = -1
            for i in range(self.natoms):
                if labels[i] == -1:
                    labels[i] = nlabels
                    self.flood_fill(i, nbonds, c10y, labels, nlabels)
                    nlabels += 1
            # if there is only one fragment, then the internal coordinates
            # are complete, and we can stop
            if nlabels == 1:
                break

            # Remove labels from atoms with no bonding partners.
            # This must happen BEFORE the allow_fragments break, otherwise
            # single atoms will retain fragment labels and cause rotation ICs
            # to be incorrectly added to single-atom groups.
            labels[nbonds == 0] = -1

            if self.allow_fragments and not first_run:
                break

            candidates = self._find_bonds_vectorized(
                labels, scale, rcov
            )
            for i, j, ts in candidates:
                try:
                    self.add_bond((i, j), ts)
                except DuplicateInternalError:
                    continue
                if nbonds[i] < max_bonds and nbonds[j] < max_bonds:
                    c10y[i, nbonds[i]] = j
                    nbonds[i] += 1
                    c10y[j, nbonds[j]] = i
                    nbonds[j] += 1
            first_run = False
            scale *= 1.05

        if self.allow_fragments and nlabels != 1:
            assert nlabels > 1
            groups = [[] for _ in range(nlabels)]
            for i, label in enumerate(labels):
                if label == -1:
                    # A lone atom not bonded to anything else
                    self.add_translation(i)
                else:
                    groups[label].append(i)
            cumshifts = {}
            self.fragment_atom_groups = []
            for group in groups:
                if not group:
                    continue
                self._wrap_fragment_positions(group, cumshifts)
                self.fragment_atom_groups.append(np.array(group, dtype=np.int32))
                self.add_translation(group)
                if len(group) >= 2:
                    self.add_rotation(group)

            # Update bond ncvecs to match the new wrapped positions.
            # ncvec_new = ncvec_old - cumshift[j] + cumshift[i]
            zero = np.zeros(3, dtype=int)
            for bond in self.internals['bonds']:
                i, j = bond.indices
                shift_i = cumshifts.get(i, zero)
                shift_j = cumshifts.get(j, zero)
                if np.any(shift_i != 0) or np.any(shift_j != 0):
                    bond.kwargs['ncvecs'] = np.array(
                        [bond.kwargs['ncvecs'][0] - shift_j + shift_i]
                    )

    def find_all_angles(
        self,
    ) -> None:
        bonds = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i < self.natoms:
                bonds[i].append(bond)
            if j < self.natoms:
                bonds[j].append(bond.reverse())

        for j, jbonds in enumerate(bonds):
            linear = []
            for b1, b2 in combinations(jbonds, 2):
                new = b1 + b2
                assert new.indices[1] == j, new.indices
                if self.atol < new.calc(self.atoms) < np.pi - self.atol:
                    try:
                        self.add_angle(new)
                    except DuplicateInternalError:
                        pass
                else:
                    self.forbid_angle(new)
                    linear.append((b1, b2))
            if linear:
                if len(jbonds) == 2:
                    # Add a dummy atom to an atom center with only 2 bonds
                    # sort bonds from shortest to longest to ensure
                    # permutational invariance
                    b1, b2 = sorted(jbonds, key=lambda x: x.calc(self.atoms))
                    # First try to take the cross product of the two bond
                    # vectors. These two vectors are close to collinear, and
                    # may be exactly collinear, so there's a backup strategy
                    # if this results in the zero-vector.
                    if self.dinds[j] < 0:
                        self.dinds[j] = self.natoms + self.ndummies
                        dx1 = -b1.calc_vec(self.atoms)
                        dx1 /= np.linalg.norm(dx1)
                        dx2 = b2.calc_vec(self.atoms)
                        dx2 /= np.linalg.norm(dx2)
                        dpos = np.cross(dx1, dx2)
                        dpos_norm = np.linalg.norm(dpos)
                        if dpos_norm < 1e-4:
                            # the aforementioned backup strategy
                            # pick the cartesian basis vector that is maximally
                            # orthogonal with the shorter of the two
                            # displacement vectors.
                            # note: this is not rotationally invariant, but
                            # there's not much we can do about that
                            dim = np.argmin(np.abs(dx1))
                            dpos[:] = 0.
                            dpos[dim] = 1.
                            dpos -= dx1 * (dpos @ dx1)
                            dpos /= np.linalg.norm(dpos)
                        else:
                            dpos /= dpos_norm
                        # Add the dummy atom
                        dpos += self.atoms.positions[j]
                        self.dummies += Atom('X', dpos)
                        self._batched_arrays_valid = False
                        self._cache.pop('all_positions', None)
                    # Create and fix dummy bond
                    dbond = Bond((j, self.dinds[j]))
                    self.cons.fix_bond(dbond, replace_ok=False)
                    self.add_bond(dbond)
                    # Fix one dummy angle (only one — for linear O1-C-O2
                    # the angles O1-C-dummy and O2-C-dummy are supplementary,
                    # so constraining both over-constrains real atoms)
                    dangle1 = b1 + dbond
                    self.cons.fix_angle(dangle1, replace_ok=False)
                    dangle2 = b2 + dbond
                    # Fix the improper dihedral and update relevant internals
                    if b2.indices[1] == j:
                        b2 = b2.reverse()
                    dbond2 = Bond(
                        (self.dinds[j], b2.indices[1]), b2.kwargs['ncvecs']
                    )
                    dangle3 = dbond + dbond2
                    ddihedral = dangle1 + dangle3
                    self.add_dihedral(ddihedral)
                    self.add_dummy_to_internals(j)
                    self.cons.add_dummy_to_internals(j)
                    # Add relevant angles
                    for b1 in jbonds:
                        new = b1 + dbond
                        assert new.indices[1] == j
                        angle = new.calc(self.all_atoms)
                        if self.atol < angle < np.pi - self.atol:
                            try:
                                self.add_angle(new)
                            except DuplicateInternalError:
                                pass
                        else:
                            self.forbid_angle(new)
                else:
                    for b1, b2 in linear:
                        for b3 in jbonds:
                            if b3 in (b1, b2):
                                continue
                            indices = (
                                b1.indices[1], j, b3.indices[1], b2.indices[1]
                            )
                            ncvecs = (
                                -b1.kwargs['ncvecs'][0],
                                b3.kwargs['ncvecs'][0],
                                b2.kwargs['ncvecs'][0] - b3.kwargs['ncvecs'][0]
                            )
                            try:
                                self.add_dihedral(indices, ncvecs)
                            except DuplicateInternalError:
                                pass
                            break
                        else:
                            raise RuntimeError(
                                "Unable to find improper dihedral to replace "
                                "linear angle!"
                            )

    def find_all_dihedrals(self) -> None:
        # First, find proper dihedrals from angle combinations.
        # Group angles by their bond edges so we only try pairs that
        # share a bond (required for __add__ to succeed).
        edge_to_angles = {}
        for angle in self.internals['angles']:
            i, j, k = angle.indices
            for edge_key in ((min(i, j), max(i, j)), (min(j, k), max(j, k))):
                edge_to_angles.setdefault(edge_key, []).append(angle)

        seen_pairs = set()
        for angles_on_edge in edge_to_angles.values():
            for a1, a2 in combinations(angles_on_edge, 2):
                pair_key = (id(a1), id(a2))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                try:
                    new = a1 + a2
                except NoValidInternalError:
                    continue
                # this is a dihedral that has the same exact atom as both
                # the first and last atom.
                if (
                    new.indices[0] == new.indices[3]
                    and np.all(
                        np.sum(new.kwargs['ncvecs'], axis=0)
                        == np.array((0, 0, 0))
                    )
                ):
                    continue
                try:
                    self.add_dihedral(new)
                except DuplicateInternalError:
                    continue

        # Second, add improper dihedrals for atoms with 3 or 4 neighbors that don't
        # have any proper dihedral passing through them. This is needed because:
        # 1. At planar geometries, bond/angle derivatives vanish for out-of-plane motion
        # 2. Even starting non-planar, the geometry may planarize during optimization
        # 3. Improper dihedrals capture the out-of-plane (umbrella) mode


        # Note this does add some redundancy to the internals but it also makes it
        # so that the Jacobian is well-conditioned in the case of planar systems,
        # such as nitrate.
        #
        # We only add impropers when no proper dihedral exists through the atom,
        # which avoids excessive unnecessary additional internals.

        # First, find which atoms have proper dihedrals through them
        dihedral_centers = set()
        for d, a in zip(self.internals['dihedrals'], self._active['dihedrals']):
            if a:
                # Positions 1 and 2 are the "central" atoms of a dihedral
                dihedral_centers.add(int(d.indices[1]))
                dihedral_centers.add(int(d.indices[2]))

        # Build neighbor list
        neighbors = [[] for _ in range(self.natoms)]
        for bond in self.internals['bonds']:
            i, j = bond.indices
            if i < self.natoms:
                neighbors[i].append((int(j), bond.kwargs['ncvecs'][0]))
            if j < self.natoms:
                neighbors[j].append((int(i), -bond.kwargs['ncvecs'][0]))

        for center in range(self.natoms):
            # Consider atoms with 3 or 4 neighbors that lack proper dihedrals.
            # - 3 neighbors: at planar geometries (e.g., NO3, sp2 carbons), the
            #   3 angles sum to 360°, creating linear dependency.
            # - 4 neighbors: at square planar geometries (e.g., Pt(II)), the
            #   4 cis angles sum to 360°, similar issue. For tetrahedral, the
            #   improper is redundant but harmless (pseudo-inverse handles it).
            # - 5+ neighbors: rare, and typically have proper dihedrals anyway.
            if len(neighbors[center]) not in (3, 4):
                continue

            # Skip if this atom already has proper dihedrals through it
            if center in dihedral_centers:
                continue

            # Add improper dihedral: neighbors[0]-center-neighbors[1]-neighbors[2]
            n0, ncvec0 = neighbors[center][0]
            n1, ncvec1 = neighbors[center][1]
            n2, ncvec2 = neighbors[center][2]
            # Improper dihedral indices: (n0, center, n1, n2)
            # The ncvecs connect consecutive atoms in the dihedral
            imp_ncvecs = (
                -ncvec0,  # from n0 to center
                ncvec1,   # from center to n1
                ncvec2 - ncvec1,  # from n1 to n2
            )
            try:
                self.add_dihedral((n0, center, n1, n2), imp_ncvecs)
            except DuplicateInternalError:
                pass

    def validate_basis(self) -> None:
        jac = self.jacobian()
        S = svdvals(jac)
        ndeloc = np.sum(S > 1e-8)

        # If TRICs (translations/rotations) are present, they span the full
        # 3N DOF. Otherwise, 6 DOF are removed for global translation/rotation.
        has_trics = (len(self.internals['translations']) > 0 or
                     len(self.internals['rotations']) > 0)
        if has_trics:
            ndof = 3 * (self.natoms + self.ndummies)
        else:
            ntot = self.natoms + self.ndummies
            has_periodic_bonds = any(
                np.any(bond.kwargs['ncvecs'] != 0)
                for bond in self.internals['bonds']
            )
            if has_periodic_bonds:
                ndof = 3 * ntot
            elif ntot <= 1:
                ndof = 0
            elif ntot == 2:
                ndof = 1
            else:
                ndof = 3 * ntot - 6

        if ndeloc != ndof:
            warnings.warn(
                f'{ndeloc} coords found! Expected {ndof}.'
            )

    def check_for_bad_internals(self) -> Optional[Dict[str, List[Coordinate]]]:
        """Check for angles that are too close to 0 or pi (linear).

        Uses vectorized computation for efficiency.
        """
        bad = {'bonds': [], 'angles': []}

        angles = self.internals['angles']
        if not angles:
            return None

        # Use vectorized computation to check all angles at once
        # Use padded arrays for consistent JAX shapes (avoids recompilation)
        self._build_batched_arrays()
        if self._n_angles_actual > 0:
            positions = self.all_positions
            cell = self.atoms.cell.array
            tvecs = self._get_cached_tvecs(cell)
            angle_pos = positions[self._angle_indices_padded]
            angle_vals_padded = np.asarray(_angle_value_batched(angle_pos, tvecs['angles_padded']))
            angle_vals = angle_vals_padded[:self._n_angles_actual]

            # Find bad angles
            bad_mask = ~((self.atol < angle_vals) & (angle_vals < np.pi - self.atol))
            if np.any(bad_mask):
                bad_indices = np.where(bad_mask)[0]
                for idx in bad_indices:
                    bad['angles'].append(angles[idx])

        for ints in bad.values():
            if ints:
                return bad
        return None

    def _h0_bond(
        self,
        bond: Bond,
        Ab: float = 0.3601,
        Bb: float = 1.944,
    ) -> float:
        idx = np.asarray(bond.indices, dtype=np.int32)
        numbers = self.all_atoms.numbers[idx]
        rcov = _STIFFNESS_RADII[numbers].sum()
        rij = bond.calc(self.all_atoms)
        h0 = Ab * np.exp(-Bb * (rij - rcov) / units.Bohr)
        if (self.ntrans + self.nrotations) == 0 and numbers.min() > 0:
            # Connected system, bond between real atoms: the row factors of
            # the heavier p-block elements (see _STIFFNESS_SCALE).
            h0 *= _STIFFNESS_SCALE[numbers].prod()
        return h0 * units.Hartree / units.Bohr**2

    def _h0_stretch_diagonal(self) -> Optional[np.ndarray]:
        """The Almlof stretch guesses of the bonds at the current geometry
        as a vector over the internal coordinates (zero elsewhere).

        The exponential of _h0_bond is the local curvature-length relation
        of a bond (Badger's rule: d ln k / dr = -3 / (r - d_ij), i.e.
        -3.3 to -4.3 per Angstrom for first-row bonds, against Bb = 3.7),
        so a bond that relaxes by 0.1 A during the run is 30-45 % stiffer
        or softer at the end than the start-geometry guess says.  InternalPES
        moves this part of the model with the geometry and transports the
        secant pairs with it (see InternalPES._track_analytic_model).
        """
        h = np.zeros(self.nint, dtype=np.float64)
        idx = len(self.internals['translations'])
        bonds = self.internals['bonds']
        if idx + len(bonds) > self.nint:
            return None
        for bond in bonds:
            h[idx] = self._h0_bond(bond)
            idx += 1
        return h

    def _h0_angle(
        self,
        angle: Angle,
        Aa: float = 0.089,
        Ba: float = 0.11,
        Ca: float = 0.44,
        Da: float = -0.42,
        valence_scale: float = 0.5,
    ) -> float:
        bab, bbc = angle.split()
        idxab = np.asarray(bab.indices, dtype=np.int32)
        idxbc = np.asarray(bbc.indices, dtype=np.int32)
        rcovab = _STIFFNESS_RADII[self.all_atoms.numbers[idxab]].sum()
        rcovbc = _STIFFNESS_RADII[self.all_atoms.numbers[idxbc]].sum()
        rab = bab.calc(self.all_atoms)
        rbc = bbc.calc(self.all_atoms)
        h0 = (
            Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc) / units.Bohr)
            / (rcovab * rcovbc / units.Bohr**2)**Da
        )
        # The Fischer-Almlof bend formula gives 0.28 (H-C-H), 0.32 (H-C-C),
        # 0.35 (C-C-C), 0.30 (C-O-H) and 0.40 (C-C=O) Ha/rad^2, about twice
        # the valence force-field constants for the same primitive angles
        # (methane 0.53, ethane/propane 0.65-1.0, water/alcohols 0.75,
        # carbonyls ~1.0 mdyn A/rad^2 = 0.12-0.23 Ha/rad^2, the range of
        # Schlegel's 0.16/0.25 bend guesses).  The model energy of a
        # redundant primitive set is a valence force field, so the primitive
        # diagonal should match the valence constants; halve the guess.
        return valence_scale * h0 * units.Hartree

    def _h0_dihedral(
        self,
        dihedral: Dihedral,
        nbonds: np.ndarray,
        At: float = 0.0015,
        Bt: float = 14.0,
        Ct: float = 2.85,
        Dt: float = 0.57,
        Et: float = 4.00,
        proper: bool = True,
    ) -> float:
        _, bbc = dihedral.split()[0].split()
        idx = np.asarray(bbc.indices, dtype=np.int32)
        rcovbc = _STIFFNESS_RADII[self.all_atoms.numbers[idx]].sum()
        rbc = bbc.calc(self.all_atoms)
        L = nbonds[idx].sum() - 2
        bo = np.exp(-Ct * (rbc - rcovbc) / units.Bohr)
        h0 = (
            At + Bt * L**Dt * bo
            / (rbc * rcovbc / units.Bohr**2)**Et
        )
        if not proper:
            # Improper dihedral (umbrella mode at a centre without proper
            # dihedrals): not a rotation about its "central" bond, so it keeps
            # the plain Fischer-Almlof value.
            return h0 * units.Hartree
        # Bond-order factor: the same exponential the Fischer-Almlof formula
        # uses once.  Applied a second time it makes short (double, aromatic,
        # amide, ester) bonds ~2-3x stiffer than sp3 single bonds of the same
        # atoms, which together with the 1/sqrt(n) redundancy sharing in
        # guess_hessian keeps those torsions at the original per-dihedral
        # stiffness while sp3-sp3 rotations are softened towards their
        # physical ~0.02 Ha/rad^2.
        return h0 * bo * units.Hartree

    def _h0_linear_bend(self, centre: int, k_bend: float = 0.10,
                        k_bend_h: float = 0.05) -> float:
        """Diagonal guess (eV/rad^2) for the coordinates that describe the
        bending of a near-linear two-bonded centre A-centre-C through its
        dummy atom X: the dummy angles A-centre-X and C-centre-X (bend
        towards X) and the dummy dihedral A-centre-X-C (bend in the plane
        perpendicular to X, which is the plane of the initial A-centre-C
        bend), both with unit Jacobian with respect to the bend angle.
        Linear-bend force constants are far below the Fischer-Almlof bend
        value (0.31-0.37 Ha/rad^2 for these angles) and below the 0.5
        Ha/rad^2 Sella assigns to dummy dihedrals: C-C#C and C-C#N bends
        are 0.28-0.35 mdyn A/rad^2 = 0.06-0.08 Ha/rad^2, cumulene and azide
        bends (CO2 0.57, allene 0.55, HN3 0.5 mdyn A/rad^2) 0.11-0.13, and
        the D-H...A bend of a proton-shared hydrogen bond 0.02-0.05.
        k_bend covers the sp centres, k_bend_h the hydrogen centres; both
        sit at the stiff end of their range because a quasi-Newton step
        along a mode whose model stiffness is r times the true one leaves
        the fraction |1 - 1/r| of the error, so a model twice too stiff
        (0.5 per step) beats one twice too soft (no progress)."""
        z = int(self.atoms.numbers[centre])
        return (k_bend_h if z == 1 else k_bend) * units.Hartree

    def _torsion_centre_types(self, adj: List[List[int]]) -> List[str]:
        """Local hybridisation label of every real atom for the torsional
        guess, from the element and the covalent neighbour count alone:
        'pi' for a planar sp2 centre (C or B with three neighbours, N with
        two neighbours, N with three neighbours one of which is a terminal
        O or S, i.e. nitro/N-oxide), 'lp' for a lone-pair heteroatom that
        conjugates with a pi centre (N with three neighbours, O, S or Se
        with two), and 'sigma' for everything else (sp3 carbon and silicon,
        ammonium nitrogen, sulfonyl/phosphoryl centres, sp centres, terminal
        atoms)."""
        numbers = np.asarray(self.atoms.numbers)
        types = []
        for i in range(self.natoms):
            z = int(numbers[i])
            n = len(adj[i])
            label = 'sigma'
            if z in (5, 6) and n == 3:
                label = 'pi'
            elif z == 7:
                if n == 2:
                    label = 'pi'
                elif n == 3:
                    label = 'lp'
                    for j in adj[i]:
                        if int(numbers[j]) in (8, 16) and len(adj[j]) == 1:
                            label = 'pi'
                            break
            elif z in (8, 16, 34) and n == 2:
                label = 'lp'
            types.append(label)
        return types

    @staticmethod
    def _in_small_ring(b: int, c: int, adj: List[List[int]],
                       ring_max: int) -> bool:
        """True when the bond b-c closes a ring of at most ring_max atoms,
        i.e. c is reachable from b within ring_max - 1 steps without using
        the bond itself."""
        seen = {b}
        front = [b]
        for _ in range(ring_max - 1):
            new = []
            for u_ in front:
                for v in adj[u_]:
                    if u_ == b and v == c:
                        continue
                    if v == c:
                        return True
                    if v not in seen:
                        seen.add(v)
                        new.append(v)
            front = new
            if not front:
                break
        return False

    def _torsion_class_factor(
        self,
        b: int,
        c: int,
        types: List[str],
        adj: List[List[int]],
        bo: float,
        s_pi_sigma: float = 0.6,
        s_pi_lp: float = 0.45,
        s_carbonyl_lp: float = 0.7,
        s_pi_pi: float = 0.65,
        bo_lo: float = 1.5,
        bo_hi: float = 2.2,
        ring_max: int = 8,
    ) -> float:
        """Class-resolved scale of the Fischer-Almlof torsional guess for the
        rotation about the acyclic bond b-c.

        The Fischer-Almlof torsional constant, once shared over the n
        redundant dihedrals of a bond (1/sqrt(n) in guess_hessian), puts the
        rotational stiffness sum_d h_d of an sp3-sp3 bond (n = 9) at
        0.02-0.03 Ha/rad^2, the physical value of ethane-like rotors
        (V3 = 3 kcal/mol gives 9/2 V3 = 0.021), and it keeps double and
        aromatic bonds stiff through the bond-order factor.  Single bonds
        to a planar sp2 centre have far fewer dihedrals (n = 2-6), so the
        same sharing leaves them 2-3.5x above their rotational barriers
        (curvature n^2 V_n / 2 of the leading Fourier term): sp2-sp3 links
        (propene V3 = 2.0, acetaldehyde 1.2, ethylbenzene V2 ~ 1.3 kcal/mol;
        0.004-0.014 Ha/rad^2 against a 0.025 guess), sp2 centre to a
        lone-pair heteroatom (anisole/phenol V2 = 3-3.5, aniline ~4-5,
        methyl vinyl ether ~4; 0.010-0.017 against 0.034-0.049), sp2-sp2
        single bonds (styrene V2 ~ 3, butadiene/enones/aryl ketones 5-8,
        biphenyl ~2; 0.008-0.025 against 0.023-0.033) and carbonyl to a
        lone-pair heteroatom (esters/acids 10-13, amides/carbamates 15-20
        kcal/mol; 0.035-0.06 against 0.045-0.095).  Each class is scaled to
        the stiff end of its physical range (a model that is too soft by 2x
        makes no progress along the mode at all, one that is too stiff by
        2x still halves the error per step), and the scale fades back to 1
        with the bond-order factor between bo_lo and bo_hi so that acyclic
        double bonds (bo ~2.8), amidinium/guanidinium and other strongly
        conjugated links keep the full constant.  Bonds in rings of up to
        ring_max atoms are left alone: their dihedrals describe ring
        puckering, not a rotation, and the 1/sqrt(n) evidence shows ring
        torsions want the full constant.
        """
        tb, tc = types[b], types[c]
        if 'pi' not in (tb, tc):
            return 1.0
        if 'sigma' in (tb, tc):
            s = s_pi_sigma
        elif tb == 'pi' and tc == 'pi':
            s = s_pi_pi
        else:
            # pi centre bonded to a lone-pair heteroatom: carbonyl-like
            # centres (a terminal O or S neighbour) conjugate strongly.
            p = b if tb == 'pi' else c
            numbers = np.asarray(self.atoms.numbers)
            carbonyl = any(
                int(numbers[j]) in (8, 16) and len(adj[j]) == 1
                for j in adj[p]
            )
            s = s_carbonyl_lp if carbonyl else s_pi_lp
        if self._in_small_ring(b, c, adj, ring_max):
            return 1.0
        w = min(1.0, max(0.0, (bo - bo_lo) / (bo_hi - bo_lo)))
        return s + (1.0 - s) * w

    def _h0_fragment(
        self,
        coord: Coordinate,
        kind: str,
        h0_min: float = 1e-3,
    ) -> float:
        """Diagonal floor of the guess curvature for a fragment translation
        (eV/Angstrom^2) or rotation (eV/rad^2) coordinate.

        The curvature that holds the fragments together is not a property
        of one rigid-body coordinate: a single contact i...j couples the
        translations and rotations of both fragments (its Gauss-Newton
        block is k_ij grad r_ij grad r_ij^T over all twelve coordinates), and
        the contacts change completely while two fragments approach and
        reorient.  It is therefore carried by the geometry-tracked pair
        term of _h0_nonlocal_contacts (inter-fragment pairs included), which
        InternalPES moves along with the geometry before every secant
        update.  The rigid-body coordinates themselves only get the floor
        h0_min (Hartree): far-apart or dispersion-bound fragments are soft
        (steps limited by the trust radius, not by a stiff guess) and the
        pair term makes ion pairs and hydrogen bonds stiff along the
        contact-compressing combinations only.
        """
        return h0_min * units.Hartree

    def _covalent_graph(self) -> Tuple[List[List[int]], np.ndarray]:
        """Adjacency lists and fragment labels (connected components) of
        the covalent graph of the real atoms.  The graph of an Internals
        object is fixed after construction, so the result is cached; a
        rebuild creates a new object."""
        natoms = self.natoms
        bonds = tuple(sorted(
            (int(bond.indices[0]), int(bond.indices[1]))
            for bond in self.internals['bonds']
            if int(bond.indices[0]) < natoms and int(bond.indices[1]) < natoms
        ))
        key = (natoms, bonds)
        cache = getattr(self, '_covalent_graph_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        adj = [[] for _ in range(natoms)]
        for i, j in bonds:
            adj[i].append(j)
            adj[j].append(i)
        # Fragment labels (connected components of the covalent graph).
        label = -np.ones(natoms, dtype=np.int32)
        nlabels = 0
        for i in range(natoms):
            if label[i] >= 0:
                continue
            label[i] = nlabels
            stack = [i]
            while stack:
                u_ = stack.pop()
                for v in adj[u_]:
                    if label[v] < 0:
                        label[v] = nlabels
                        stack.append(v)
            nlabels += 1
        result = (adj, label)
        self._covalent_graph_cache = (key, result)
        return result

    def _nonlocal_pairs(
        self,
        min_path: int,
        vicinal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Atom pairs of the same covalent fragment at graph distance
        >= min_path, plus every pair of atoms in different fragments
        (upper triangle, real atoms only); with vicinal=True also the 1-4
        pairs (graph distance 3) of every fragment.  Returns (ii, jj, far)
        with far marking the pairs at graph distance >= min_path or in
        different fragments (the non-local contacts proper).  The covalent
        graph of an Internals object is fixed after construction, so the
        result is cached; a rebuild creates a new object."""
        natoms = self.natoms
        adj, label = self._covalent_graph()
        key = (natoms, min_path, vicinal, tuple(map(tuple, adj)))
        cache = getattr(self, '_nonlocal_pairs_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        # Graph distance from every atom, capped at min_path (a pair at
        # distance >= min_path keeps the value min_path).
        dist = np.full((natoms, natoms), min_path, dtype=np.int32)
        for i in range(natoms):
            dist[i, i] = 0
            front = [i]
            for depth in range(1, min_path):
                new = []
                for u_ in front:
                    for v in adj[u_]:
                        if dist[i, v] > depth:
                            dist[i, v] = depth
                            new.append(v)
                front = new
                if not front:
                    break
        ii, jj = np.triu_indices(natoms, k=1)
        # Atoms in different fragments are at infinite graph distance (the
        # search never reaches them, so dist keeps the cap): every
        # inter-fragment pair qualifies.
        far = (label[ii] != label[jj]) | (dist[ii, jj] >= min_path)
        keep = far
        if vicinal:
            keep = keep | (dist[ii, jj] == 3)
        result = (ii[keep], jj[keep], far[keep])
        self._nonlocal_pairs_cache = (key, result)
        return result

    def _h0_nonlocal_contacts(
        self,
        Ab: float = 0.3601,
        Bb: float = 1.944,
        min_path: int = 5,
        max_excess: float = 3.0,
        Binv: Optional[np.ndarray] = None,
        vicinal: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Curvature of the non-local contacts -- intramolecular and
        inter-fragment -- and, for connected systems, of the vicinal (1-4)
        pairs, expressed in the internal coordinates (nint x nint, same
        units as the diagonal guess: eV/Angstrom^2 for bonds and fragment
        translations, eV/rad^2 for angles, dihedrals and fragment
        rotations).

        Bonds, angles and dihedrals carry the 1-2, 1-3 and 1-4 curvature,
        and the Fischer-Almlof torsional constant is fitted to rotational
        barriers, so it also stands for the 1-5 through-space repulsion
        around the central bond.  Nothing in the redundant set describes
        the contacts that hold a folded chain or an intramolecular hydrogen
        bond together (atoms five or more bonds apart that sit 2-4 A from
        each other), nor the contacts that hold the fragments of a
        molecular complex together, so the model is too soft along the
        torsional and rigid-body combinations that compress them.
        Following Lindh's all-pair model Hessian, every such pair (same
        fragment at graph distance >= min_path, or different fragments;
        r < r_cov,i + r_cov,j + max_excess) is treated as a weak pseudo-bond
        with the Almlof stretch curvature k_ij(r) = Ab exp(-Bb (r - r_cov))
        (at a van der Waals contact this is ~2-3e-4 Ha/Bohr^2, the
        Lennard-Jones curvature at the minimum; for an H...O contact at
        1.9 A it is ~0.012 Ha/Bohr^2), and the Cartesian quadratic form
        sum_ij k_ij (u_ij . (dx_j - dx_i))^2 is expressed in the internal
        coordinates through the pseudo-inverse Jacobian: H = D^T K D with
        D_pq = u_ij . (Binv_j - Binv_i)_q.  A rigid motion of a fragment
        leaves its internal distances unchanged, so the intramolecular
        pairs contribute nothing to the translation/rotation coordinates;
        the inter-fragment pairs give them their complete Gauss-Newton
        block (stretch along the contact, coupled to the librations of both
        fragments and to the internal coordinates that move the contact
        atoms).  The hydrogen-bond contacts X-H...Y -- inter-fragment
        ones with any donor, intramolecular ones with a heteroatom donor
        -- and the inter-fragment X-H...pi contacts of heteroatom donors
        also carry the bending curvature of _h0_contact_bends (the
        sideways stiffness of a directional contact, which a radial pair
        term does not have); the twist about a contact axis and the
        sliding of dispersion-bound fragments stay at the diagonal floor
        of _h0_fragment.

        The vicinal pairs (graph distance 3, the atoms a and d of every
        dihedral a-b-c-d) get the same radial term when vicinal is true
        (default: connected systems).  The torsional constant of a single
        bond is the through-space repulsion of the substituents on its two
        ends, and the Fischer-Almlof fit to first-row barriers (ethane 2.9
        kcal/mol, 0.021 Ha/rad^2 of rotational stiffness) knows only the
        hydrogen and first-row substituents of its training set: the bond
        stays at the ethane value whatever sits on it, whereas the
        barriers grow with the substituents' size (CH3-CCl3 ~5.5, CF3-CF3
        3.9, C2Cl6 10-15 kcal/mol; butane's anti well is 20 % stiffer
        than ethane's).  The pair term supplies exactly that dependence,
        k_ad (dr_ad/dphi)^2 summed over the 1-4 pairs of the bond: nothing
        for the in-plane pairs of a planar (conjugated) rotor, 5-10 % on an
        alkyl rotor (ethane +0.001, butane +0.003 Ha/rad^2 -- within the
        force-field decomposition of the barrier into explicit torsion and
        1-4 van der Waals terms), 0.010 Ha/rad^2 on CH3-CCl3 (0.023 ->
        0.033, near the barrier value 0.04) and 0.04 on C2Cl6 (0.022 ->
        0.065 against ~0.1).
        Being geometry dependent, it also follows the compression of a 1-4
        contact during a rotor's relaxation, which the phase-blind torsional
        constant cannot.  It adds a few per cent to the bends a-b-c and
        b-c-d and under 3 % to the b-c stretch.  The vicinal pairs are
        radial only: an X-H...Y pair three bonds apart (the syn hydroxyl of
        a carboxylic acid, an amide N-H against its own carbonyl) is not a
        hydrogen bond and gets no contact bends.  Multi-fragment systems
        keep the non-local pairs alone, so that the intramolecular paths of
        their fragments (and the basins they reach) stay as they are.
        """
        natoms = self.natoms
        pos = np.asarray(self.atoms.positions, dtype=np.float64)
        numbers = np.asarray(self.atoms.numbers)
        if vicinal is None:
            vicinal = (self.ntrans + self.nrotations) == 0
        ii, jj, far = self._nonlocal_pairs(min_path, vicinal)
        if len(ii) == 0:
            return None
        dvec = pos[jj] - pos[ii]
        r = np.linalg.norm(dvec, axis=1)
        rcov = _STIFFNESS_RADII[numbers[ii]] + _STIFFNESS_RADII[numbers[jj]]
        close = r < rcov + max_excess
        ii = ii[close]
        jj = jj[close]
        far = far[close]
        if len(ii) == 0:
            return None
        dvec = dvec[close]
        r = np.maximum(r[close], 1e-8)
        rcov = rcov[close]
        u = dvec / r[:, np.newaxis]
        k = Ab * np.exp(-Bb * (r - rcov) / units.Bohr)
        k *= units.Hartree / units.Bohr**2
        if Binv is None:
            B = self.jacobian()
            if B.size == 0:
                return None
            Binv = np.linalg.pinv(B, rcond=1e-6)
        elif Binv.size == 0:
            return None
        Binv = Binv.reshape((-1, 3, Binv.shape[-1]))[:natoms]
        D = np.einsum('pk,pkq->pq', u, Binv[jj] - Binv[ii])
        H = (D * k[:, np.newaxis]).T @ D
        # The contact bends belong to the non-local pairs only.
        if np.all(far):
            Hb = self._h0_contact_bends(ii, jj, r, rcov, pos, numbers, Binv,
                                        Bb=Bb)
        elif np.any(far):
            Hb = self._h0_contact_bends(ii[far], jj[far], r[far], rcov[far],
                                        pos, numbers, Binv, Bb=Bb)
        else:
            Hb = None
        if Hb is not None:
            H = H + Hb
        return H

    def _h0_contact_bends(
        self,
        ii: np.ndarray,
        jj: np.ndarray,
        r: np.ndarray,
        rcov: np.ndarray,
        pos: np.ndarray,
        numbers: np.ndarray,
        Binv: np.ndarray,
        Bb: float = 1.944,
        Aphi_h: float = 0.5,
        Aphi_acc: float = 0.15,
        bo_cap: float = 0.05,
        bo_cap_weak: float = 0.01,
        bo_min: float = 1e-4,
        alpha_w: float = 0.5,
        hb_elements: Tuple[int, ...] = (7, 8, 9, 16, 17, 35, 53),
    ) -> Optional[np.ndarray]:
        """Bending curvature of the hydrogen-bond contacts X-H...Y (Y an
        acceptor element of hb_elements) among the non-local pairs --
        inter-fragment contacts with any donor, intramolecular contacts
        (same fragment, graph distance >= min_path) with a heteroatom
        donor X of hb_elements -- in the internal coordinates (nint x
        nint, eV units), evaluated at the current geometry like the pair
        term it complements.

        A radial pair term k_ij u u^T resists only the compression of the
        contact; the sideways motions of a hydrogen bond -- the donor
        swinging its hydrogen off the H...Y axis and the acceptor turning
        its lone pair away from the hydrogen -- are what holds the
        librations of the two fragments, and they sit at the floor of
        _h0_fragment otherwise.  Inside a molecule the same motions are
        the torsions and bends of the pseudo-ring closed by an
        intramolecular hydrogen bond (O-H...O=C, N-H...O, O-H...N): the
        valence guess of those coordinates is that of the free rotor
        (~0.01 Ha/rad^2 for a hydroxyl or amine torsion) while the
        hydrogen bond adds 0.01-0.03 Ha/rad^2 of curvature that acts
        sideways to the H...Y axis, which the radial pair term does not
        see -- a distinct error class of the model that costs the
        quasi-Newton endgame its own steps for every such contact.  The
        intramolecular C-H...Y contacts are left out: their bending
        stiffness (< 0.005 Ha/rad^2) is small against the covalent
        torsional and bending curvature that already spans those motions,
        whereas between fragments it stands against the floor alone.
        Following Lindh's model, every angle
        around the contact gets a bending curvature proportional to the
        bond-order factors of its two arms, k_phi = A_phi rho_bond
        min(rho_contact, cap), with the Almlof exponential of the pair term
        as rho.  The contact factor is capped at the value of an
        equilibrium hydrogen bond of that donor type (bo_cap for N/O/S/
        halogen donors, bo_cap_weak for C-H donors): a compressed contact
        relaxes to equilibrium during the run and its sideways stiffness
        is set by the electrostatics, not by the repulsive wall.  Two
        prefactors, chosen from the water dimer's harmonic intermolecular
        frequencies: at the hydrogen (angle X-H...Y) the reference geometry
        is linear and the potential depends on the deviation from
        linearity alone, so both perpendicular components carry the same
        curvature (isotropic linear-bend form) with A_phi = 0.5 Ha/rad^2
        (donor in-plane/out-of-plane bends at 350/600 cm^-1 give k_phi ~
        0.015 Ha/rad^2 for rho_contact ~ 0.027); at the acceptor (angles
        H...Y-Z) the reference angle is bent, the true angle derivative is
        used (a precession about the contact leaves the angle unchanged)
        with Lindh's A_phi = 0.15 Ha/rad^2 (acceptor wag/twist at 120-150
        cm^-1), and the isotropic form is blended in smoothly for
        near-linear acceptor angles (weight exp(-((pi - theta)/alpha_w)^2))
        so that the term stays continuous through linearity.  The
        Gauss-Newton form sum k_phi (grad theta)(grad theta)^T is mapped to
        the internal coordinates through the pseudo-inverse Jacobian
        exactly like the pair term.  Dispersion and steric contacts
        (C...C, C-H...C, H...H) and metal-cation contacts are radial and
        get no bending term.

        An acceptor with two or more covalent neighbours gets both of its
        angular degrees of freedom from the bends H...Y-Z: the precession
        of the hydrogen about one Y-Z axis changes the angle to the other
        neighbour.  An acceptor with a single neighbour -- the carbonyl,
        carboxylate or nitro oxygen, Y=Z with Z trigonal -- has only one
        such bend, and the precession about the Y=Z axis is invisible to
        every term above (it changes neither the contact length nor the
        angles at H and Y); physically it is the stiffer of the two
        acceptor motions, because the sp2 lone pairs lie in the plane of
        Z's substituents and a hydrogen bond above that plane costs a
        substantial part of the interaction energy (1-3 kcal/mol at 90
        degrees, i.e. k ~ 0.005-0.01 Ha/rad^2).  Such acceptors get the
        out-of-plane term k_oop s^2 / 2 with s = sin of the angle between
        Y->H and the plane (W1, Z, W2), k_oop = A_phi,acc rho_YZ
        min(rho_contact, cap) times the planarity of Z (squared cosine of
        the angle between Z->Y and the plane, so that a pyramidal centre
        such as a sulfoxide sulfur, whose oxygen lone pairs are nearly
        cylindrical, gets little of it); with the same prefactor as the
        acceptor bends the acceptor-side angular curvature becomes
        isotropic, as it effectively is for two-neighbour acceptors.  The
        gradient of s involves the hydrogen, the acceptor and the three
        atoms that define the plane, so the term also holds the rotation
        of the acceptor molecule about its own Y=Z axis.  Acceptors whose
        neighbour is linear (nitrile) or tetrahedral (sulfonyl,
        phosphoryl) have nearly cylindrical lone-pair distributions and
        keep the precession free.

        X-H...pi contacts between fragments (a heteroatom donor pointing
        its hydrogen at a double bond or an aromatic ring) are directional
        too: the hydrogen sits over the pi face, 2.0-2.8 A from the
        nearest unsaturated carbons, and the pair terms to those carbons --
        all inclined by only 15-30 degrees from the common normal -- give
        the sideways motions of the hydrogen (the donor tilting, the ring
        or double bond sliding under it) almost no curvature, so they stay
        at the floor although their librations lie at 100-300 cm^-1.  A
        hydrogen of a heteroatom donor whose closest non-hydrogen atom of
        another fragment (by r - r_cov) is an unsaturated carbon (two or
        three covalent neighbours) is taken as a pi donor towards that
        fragment, and every such carbon of the fragment within range gets
        the same two kinds of bend as a heteroatom acceptor: the isotropic
        bend at the hydrogen and the true-angle bends H...C-Z at the
        carbon, whose two or three arms 120 degrees apart cover the tilt
        in every direction.  The contact factor uses the weak cap
        (bo_cap_weak): the pi bond is a diffuse acceptor, and the two
        carbons of a double bond within reach sum to the libration
        stiffness (0.003-0.01 Ha/rad^2) that the water-dimer calibration
        of the caps gives per single heteroatom contact.  An aromatic face
        puts six carbons at the same distance, but it binds a donor only
        1.3-2 times more strongly than a double bond, not three times, so
        the contact factors of one (hydrogen, fragment) face are scaled to
        a sum of at most twice the largest of them -- a double bond's
        worth.  A hydrogen whose closest partner is a heteroatom keeps the
        heteroatom terms only, so hydrogen-bonded contacts are unchanged.
        """
        adj, label = self._covalent_graph()
        inter = label[ii] != label[jj]
        bo = np.exp(-Bb * (r - rcov) / units.Bohr)
        sel = np.where(bo >= bo_min)[0]
        if len(sel) == 0:
            return None
        hb_elem = np.isin(numbers, hb_elements)
        # (hydrogen, fragment label) pairs for which the hydrogen's closest
        # non-hydrogen atom of that fragment is an unsaturated carbon.
        pi_donor = set()
        pi_scale = {}
        if np.any(inter):
            excess = r - rcov
            nearest = {}
            for p in np.where(inter)[0]:
                i, j = int(ii[p]), int(jj[p])
                for h, y in ((i, j), (j, i)):
                    if numbers[h] != 1 or numbers[y] == 1:
                        continue
                    key = (h, int(label[y]))
                    if key not in nearest or excess[p] < nearest[key][0]:
                        nearest[key] = (float(excess[p]), y)
            for key, (_, y) in nearest.items():
                if numbers[y] == 6 and 2 <= len(adj[y]) <= 3:
                    pi_donor.add(key)
            # Per face, the contact factors of the unsaturated carbons in
            # range sum to at most twice the largest one.
            face = {}
            for p in sel:
                if not inter[p]:
                    continue
                i, j = int(ii[p]), int(jj[p])
                for h, y in ((i, j), (j, i)):
                    key = (h, int(label[y]))
                    if (key in pi_donor and numbers[y] == 6
                            and 2 <= len(adj[y]) <= 3):
                        face.setdefault(key, []).append(
                            min(float(bo[p]), bo_cap_weak))
            for key, cs in face.items():
                pi_scale[key] = min(1., 2. * max(cs) / sum(cs))
        V, A, B, W, ISO = [], [], [], [], []
        OOP, WO = [], []
        for p in sel:
            i, j = int(ii[p]), int(jj[p])
            for h, y in ((i, j), (j, i)):
                if numbers[h] != 1 or not adj[h]:
                    continue
                strong = any(hb_elem[kk] for kk in adj[h])
                scale = 1.
                if hb_elem[y]:
                    if not (strong or inter[p]):
                        # Intramolecular C-H...Y contact: no bending term.
                        continue
                    cap = bo_cap if strong else bo_cap_weak
                elif (strong and inter[p] and numbers[y] == 6
                      and 2 <= len(adj[y]) <= 3
                      and (h, int(label[y])) in pi_donor):
                    # X-H...pi contact with an unsaturated carbon.
                    cap = bo_cap_weak
                    scale = pi_scale[(h, int(label[y]))]
                else:
                    continue
                c = min(float(bo[p]), cap) * scale
                # Bend at the hydrogen: X-H...Y, linear reference.
                for x in adj[h]:
                    V.append(h)
                    A.append(x)
                    B.append(y)
                    W.append(Aphi_h * c)
                    ISO.append(True)
                # Bends at the acceptor: H...Y-Z, bent reference.
                for z in adj[y]:
                    V.append(y)
                    A.append(z)
                    B.append(h)
                    W.append(Aphi_acc * c)
                    ISO.append(False)
                # Lone-pair plane of a single-neighbour acceptor on a
                # trigonal centre (Y=Z with Z bonded to Y, W1, W2): the
                # out-of-plane wag of the hydrogen relative to the plane
                # (W1, Z, W2).
                if len(adj[y]) == 1 and len(adj[adj[y][0]]) == 3:
                    z = adj[y][0]
                    ws = [w for w in adj[z] if w != y]
                    if len(ws) == 2:
                        OOP.append((h, y, z, ws[0], ws[1]))
                        WO.append(Aphi_acc * c)
        if not V:
            return None
        V = np.asarray(V, dtype=np.int32)
        A = np.asarray(A, dtype=np.int32)
        B = np.asarray(B, dtype=np.int32)
        W = np.asarray(W, dtype=np.float64)
        ISO = np.asarray(ISO, dtype=bool)
        rcov_all = _STIFFNESS_RADII[numbers]
        a = pos[A] - pos[V]
        b = pos[B] - pos[V]
        ra = np.maximum(np.linalg.norm(a, axis=1), 1e-8)
        rb = np.maximum(np.linalg.norm(b, axis=1), 1e-8)
        ah = a / ra[:, np.newaxis]
        bh = b / rb[:, np.newaxis]
        bo_bond = np.exp(-Bb * (ra - rcov_all[A] - rcov_all[V]) / units.Bohr)
        W *= np.minimum(bo_bond, 1.) * units.Hartree
        cos_t = np.clip(np.sum(ah * bh, axis=1), -1., 1.)
        sin_t = np.sqrt(np.maximum(1. - cos_t**2, 0.))
        theta = np.arccos(cos_t)
        s = np.maximum(sin_t, 1e-8)[:, np.newaxis]
        # In-plane bend directions at A (perpendicular to a) and at B
        # (perpendicular to b); for a linear angle any direction
        # perpendicular to the axis serves for both.
        e1 = (cos_t[:, np.newaxis] * ah - bh) / s
        e2 = (cos_t[:, np.newaxis] * bh - ah) / s
        lin = sin_t < 1e-6
        if np.any(lin):
            nlin = int(lin.sum())
            ref = np.zeros((nlin, 3), dtype=np.float64)
            ref[np.arange(nlin), np.argmin(np.abs(ah[lin]), axis=1)] = 1.
            perp = np.cross(ah[lin], ref)
            perp /= np.maximum(np.linalg.norm(perp, axis=1),
                               1e-8)[:, np.newaxis]
            e1[lin] = perp
            e2[lin] = perp
        # Out-of-plane direction, orthogonal to the axis and to e1.
        pn = np.cross(ah, e1)
        pn /= np.maximum(np.linalg.norm(pn, axis=1), 1e-8)[:, np.newaxis]
        f = np.where(ISO, 1., np.exp(-((np.pi - theta) / alpha_w)**2))
        dA = e1 / ra[:, np.newaxis]
        dB = e2 / rb[:, np.newaxis]
        dth = (np.einsum('pk,pkq->pq', dA, Binv[A])
               + np.einsum('pk,pkq->pq', dB, Binv[B])
               - np.einsum('pk,pkq->pq', dA + dB, Binv[V]))
        H = (dth * W[:, np.newaxis]).T @ dth
        wA = pn / ra[:, np.newaxis]
        wB = pn / rb[:, np.newaxis]
        dpo = (np.einsum('pk,pkq->pq', wA, Binv[A])
               + np.einsum('pk,pkq->pq', wB, Binv[B])
               - np.einsum('pk,pkq->pq', wA + wB, Binv[V]))
        H += (dpo * (W * f)[:, np.newaxis]).T @ dpo
        if OOP:
            Hoop = self._h0_lone_pair_plane(np.asarray(OOP, dtype=np.int32),
                                            np.asarray(WO, dtype=np.float64),
                                            pos, rcov_all, Binv, Bb)
            if Hoop is not None:
                H += Hoop
        return H

    @staticmethod
    def _h0_lone_pair_plane(
        OOP: np.ndarray,
        WO: np.ndarray,
        pos: np.ndarray,
        rcov_all: np.ndarray,
        Binv: np.ndarray,
        Bb: float,
    ) -> Optional[np.ndarray]:
        """Gauss-Newton curvature sum k_oop (grad s)(grad s)^T of the
        out-of-plane coordinate s = u_YH . n of the hydrogen-bond contacts
        listed in OOP (rows h, y, z, w1, w2; see _h0_contact_bends), with
        n the unit normal of the plane (W1, Z, W2) and u_YH the unit
        vector from the acceptor to the hydrogen, in the internal
        coordinates (nint x nint, eV units).  WO holds A_phi
        min(rho_contact, cap) per row; the bond-order factor of Y-Z and
        the planarity of Z are applied here.  s is the sine of the angle
        between Y->H and the plane, so it is regular for every geometry
        and its gradient is bounded; the planarity factor is the squared
        cosine of the angle between Z->Y and the plane."""
        h, y, z, w1, w2 = (OOP[:, k] for k in range(5))
        u = pos[h] - pos[y]
        ru = np.maximum(np.linalg.norm(u, axis=1), 1e-8)
        u = u / ru[:, np.newaxis]
        a = pos[w1] - pos[z]
        b = pos[w2] - pos[z]
        n = np.cross(a, b)
        nn = np.linalg.norm(n, axis=1)
        ok = nn > 1e-6
        if not np.any(ok):
            return None
        if not np.all(ok):
            h, y, z, w1, w2 = h[ok], y[ok], z[ok], w1[ok], w2[ok]
            u, ru, a, b, n, nn = u[ok], ru[ok], a[ok], b[ok], n[ok], nn[ok]
            WO = WO[ok]
        nh = n / nn[:, np.newaxis]
        s = np.sum(u * nh, axis=1)
        v = pos[y] - pos[z]
        rv = np.maximum(np.linalg.norm(v, axis=1), 1e-8)
        planarity = 1. - (np.sum(v * nh, axis=1) / rv)**2
        bo_yz = np.exp(-Bb * (rv - rcov_all[y] - rcov_all[z]) / units.Bohr)
        k = WO * planarity * np.minimum(bo_yz, 1.) * units.Hartree
        # d s / d r_H = (n - s u) / r_YH; the plane atoms enter through
        # d n_hat = (I - n_hat n_hat^T) d n / |n| with n = a x b.
        gH = (nh - s[:, np.newaxis] * u) / ru[:, np.newaxis]
        q = (u - s[:, np.newaxis] * nh) / nn[:, np.newaxis]
        gW1 = np.cross(b, q)
        gW2 = np.cross(q, a)
        ds = (np.einsum('pk,pkq->pq', gH, Binv[h])
              - np.einsum('pk,pkq->pq', gH, Binv[y])
              + np.einsum('pk,pkq->pq', gW1, Binv[w1])
              + np.einsum('pk,pkq->pq', gW2, Binv[w2])
              - np.einsum('pk,pkq->pq', gW1 + gW2, Binv[z]))
        return (ds * k[:, np.newaxis]).T @ ds

    def guess_hessian(self, h0cart=70.) -> np.ndarray:
        nbonds = np.zeros(len(self.all_atoms), dtype=np.int32)
        h0 = np.zeros(self.nint, dtype=np.float64)
        idx = 0
        for trans in self.internals['translations']:
            if self.allow_fragments:
                h0[idx] = self._h0_fragment(trans, 'translation')
            else:
                h0[idx] = h0cart
            idx += 1
        for bond in self.internals['bonds']:
            h0[idx] = self._h0_bond(bond)
            idx += 1
            # count number of bonds per atom for dihedral later
            i, j = bond.indices
            nbonds[i] += 1
            nbonds[j] += 1
        dummy_set = set(range(self.natoms, self.natoms + self.ndummies))
        # Connected systems get the class-resolved soft-mode guesses (the
        # linear-bend constant below, the rotatable-bond torsion classes).
        # Multi-fragment systems keep the unscaled model: their cost is set
        # by the intermolecular coordinates, and the intramolecular paths of
        # the fragments (and with them the basins they reach) stay as before.
        connected = (self.ntrans + self.nrotations) == 0
        for angle in self.internals['angles']:
            if connected and any(j in dummy_set for j in angle.indices):
                # Dummy angle A-j-X or C-j-X of a near-linear centre j.
                h0[idx] = self._h0_linear_bend(int(angle.indices[1]))
            else:
                h0[idx] = self._h0_angle(angle)
            idx += 1
        bonded = set()
        for bond in self.internals['bonds']:
            i, j = bond.indices
            bonded.add(frozenset((int(i), int(j))))

        def is_proper(dihedral):
            # A proper dihedral a-b-c-d has d bonded to c and measures the
            # rotation about b-c.  The improper dihedrals find_all_dihedrals
            # adds at three-/four-neighbour centres without proper dihedrals
            # (n0, centre, n1, n2) have n2 bonded to the centre instead; they
            # describe the umbrella mode and are treated like the original.
            a, b, c, d = (int(j) for j in dihedral.indices)
            return frozenset((c, d)) in bonded

        # The Fischer-Almlof torsional constant describes the rotation about
        # the central bond as a whole.  A redundant set carries n dihedrals
        # around the same bond, and a rigid rotation changes all of them
        # together, so assigning the full constant to every dihedral makes
        # that rotation n times too stiff.  Share the constant by scaling each
        # dihedral with 1/sqrt(n) (a compromise between the per-dihedral value
        # and an exact 1/n split, which would be too soft for double bonds).
        ndih = {}
        for dihedral in self.internals['dihedrals']:
            if any(j in dummy_set for j in dihedral.indices):
                continue
            if not is_proper(dihedral):
                continue
            key = frozenset(int(j) for j in dihedral.indices[1:3])
            ndih[key] = ndih.get(key, 0) + 1
        # Class-resolved scale of the rotatable-bond torsions (see
        # _torsion_class_factor), from the covalent graph of the real atoms.
        scale_torsions = connected
        adj = [[] for _ in range(self.natoms)]
        for pair in bonded:
            i, j = tuple(pair)
            if i < self.natoms and j < self.natoms:
                adj[i].append(j)
                adj[j].append(i)
        types = self._torsion_centre_types(adj) if scale_torsions else None
        numbers = np.asarray(self.atoms.numbers)
        positions = np.asarray(self.atoms.positions, dtype=np.float64)
        tfac = {}
        for dihedral in self.internals['dihedrals']:
            if any(j in dummy_set for j in dihedral.indices):
                a, b, c, d = (int(j) for j in dihedral.indices)
                if connected and (b in dummy_set or c in dummy_set):
                    # The linear-bend dihedral A-j-X-C built in
                    # find_all_angles (dummy at an inner position; the
                    # dihedrals with a terminal dummy are the azimuths of
                    # the neighbouring substituents about the linear axis
                    # relative to X and keep the original value).
                    h0[idx] = self._h0_linear_bend(b if c in dummy_set
                                                   else c)
                else:
                    h0[idx] = 0.5 * units.Hartree
            elif not is_proper(dihedral):
                h0[idx] = self._h0_dihedral(dihedral, nbonds, proper=False)
            else:
                key = frozenset(int(j) for j in dihedral.indices[1:3])
                if key not in tfac:
                    fac = 1.0
                    if scale_torsions:
                        b, c = (int(j) for j in dihedral.indices[1:3])
                        rbc = np.linalg.norm(positions[c] - positions[b])
                        rcovbc = (_STIFFNESS_RADII[numbers[b]]
                                  + _STIFFNESS_RADII[numbers[c]])
                        bo = np.exp(-2.85 * (rbc - rcovbc) / units.Bohr)
                        fac = self._torsion_class_factor(
                            b, c, types, adj, float(bo))
                    tfac[key] = fac
                h0[idx] = (tfac[key] * self._h0_dihedral(dihedral, nbonds)
                           / np.sqrt(ndih[key]))
            idx += 1
        for rot in self.internals['rotations']:
            if self.allow_fragments:
                h0[idx] = self._h0_fragment(rot, 'rotation')
            else:
                h0[idx] = h0cart
            idx += 1
        H0 = np.diag(np.abs(h0))
        # Non-local contact curvature (folded chains, intramolecular
        # hydrogen bonds, and the contacts between the fragments of a
        # complex): a positive semi-definite pair term in the internal
        # coordinates, see _h0_nonlocal_contacts.  The term is kept so that
        # InternalPES can move it along with the geometry.
        Hnb = self._h0_nonlocal_contacts()
        if Hnb is not None and Hnb.shape != H0.shape:
            Hnb = None
        self._h0_nonlocal_last = Hnb
        if Hnb is not None:
            H0 = H0 + Hnb
        return H0

logger = logging.getLogger(__name__)

class _LRU2:
    """2-entry LRU cache keyed by state hash (bytes).

    Two entries match the optimization step cycle, which alternates between
    pre-ODE (post-cell-change) and post-ODE positions.
    """

    __slots__ = ('_entries', '_next')

    def __init__(self):
        self._entries = [None, None]
        self._next = 0

    def get(self, key):
        for entry in self._entries:
            if entry is not None and entry[0] == key:
                return entry[1]
        return None

    def put(self, key, value):
        for entry in self._entries:
            if entry is not None and entry[0] == key:
                return
        self._entries[self._next] = (key, value)
        self._next = 1 - self._next

def _split_cons_subspace(drdxnred, tol_factor=1e-6):
    """Split (n_int) into Ucons (rowspace of drdxnred) and Ufree (its complement).

    Replaces ``np.linalg.svd(drdxnred)`` (which materializes a full
    n_int×n_int V matrix) with rank-revealing QR on ``drdxnred.T``.
    For an (m, n) drdxnred with m << n, this is roughly half the cost of
    the SVD path and returns the same orthonormal subspaces (column order
    differs but the spans match — every downstream consumer is column-
    permutation-invariant).

    Returns ``(Ucons, Ufree)`` of shapes (n, ncons) and (n, n - ncons).
    """
    Q, R, _ = qr(drdxnred.T, mode='full', pivoting=True, check_finite=False)
    diag = np.abs(np.diag(R))
    if diag.size and diag[0] > 0:
        ncons = int(np.sum(diag > tol_factor * diag[0]))
    else:
        ncons = 0
    return Q[:, :ncons], Q[:, ncons:]


# Symmetry images of the secant pairs (InternalPES._symmetry_images).  The
# energy is exactly invariant under permutations of identical nuclei and
# under every orthogonal map of the coordinates, so at a geometry that is
# itself (nearly) invariant under such an operation every measured secant
# pair has an exact image pair, and the multi-secant update learns as many
# directions per force call as the point group has operations.
_SYM_TOL = 0.05        # A, per-atom rmsd of R.P.x - x below which an operation is used
_SYM_STEP_FRAC = 0.5   # ... or when it is below this fraction of the rms step just taken
_SYM_MAX_PERM = 48     # permutations kept per search
_SYM_MAX_NODES = 20000  # backtracking nodes per search


def _isometric_permutations(numbers, pos, tol, max_perm=_SYM_MAX_PERM,
                            max_nodes=_SYM_MAX_NODES):
    """Permutations P of same-element atoms with |d(i,k) - d(Pi,Pk)| < tol
    for every pair of atoms (the distance-preserving relabelings, i.e.
    the candidate point-group operations), by backtracking over the
    atoms with distance pruning; heavy atoms with the most distinct
    distance spectra are assigned first.  The identity is included.
    The search stops after max_perm permutations or max_nodes nodes
    (every permutation returned is a genuine isometry)."""
    n = len(numbers)
    if n < 2:
        return []
    numbers = np.asarray(numbers)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    order = sorted(range(n), key=lambda i: (numbers[i] <= 1, -float(np.std(d[i]))))
    order = np.array(order, dtype=np.int64)
    cands = [np.flatnonzero(numbers == numbers[i]) for i in range(n)]
    perm = -np.ones(n, dtype=np.int64)
    used = np.zeros(n, dtype=bool)
    sols = []
    nodes = [0]

    def bt(k):
        if len(sols) >= max_perm or nodes[0] > max_nodes:
            return
        if k == n:
            sols.append(perm.copy())
            return
        i = order[k]
        done = order[:k]
        di = d[i, done]
        pdone = perm[done]
        for j in cands[i]:
            if used[j]:
                continue
            nodes[0] += 1
            if k and np.max(np.abs(d[j, pdone] - di)) >= tol:
                continue
            perm[i] = j
            used[j] = True
            bt(k + 1)
            perm[i] = -1
            used[j] = False

    bt(0)
    return sols


def _symmetry_operations(numbers, pos, tol_r, tol_d=None):
    """Point-group operations of the geometry pos within a per-atom rmsd
    tol_r: list of (perm, det, R, c) with R (pos[perm] - c) + c ~ pos, det
    = +-1 the determinant of the orthogonal map R (Kabsch fit in both
    determinant branches: a planar molecule realises a permutation both
    as a rotation and as a reflection, and the identity permutation as
    the reflection through the molecular plane).  The trivial operation
    is excluded."""
    n = len(numbers)
    if n < 3:
        return []
    if tol_d is None:
        tol_d = 3.0 * tol_r
    c = pos.mean(axis=0)
    Q = pos - c
    ops = []
    ident = np.arange(n)
    for perm in _isometric_permutations(numbers, pos, tol_d):
        P = Q[perm]
        try:
            U, _, Vt = np.linalg.svd(P.T @ Q)
        except np.linalg.LinAlgError:
            continue
        d0 = 1.0 if np.linalg.det(U @ Vt) >= 0 else -1.0
        trivial = bool(np.all(perm == ident))
        for det in (1.0, -1.0):
            if trivial and det > 0:
                continue
            D = np.array([1.0, 1.0, det * d0])
            R = (U * D) @ Vt
            R = R.T
            resid = P @ R.T - Q
            rmsd = float(np.sqrt(np.mean(np.sum(resid * resid, axis=1))))
            if rmsd <= tol_r:
                ops.append((perm, det, R, c))
    return ops


class PES:
    n_cell_dof = 0

    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eigensolver: str = 'jd0',
        trajectory: Union[str, Trajectory] = None,
        eta: float = 1e-4,
        v0: np.ndarray = None,
        proj_trans: bool = None,
        proj_rot: bool = None,
        hessian_function: Callable[[Atoms], np.ndarray] = None,
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals['translations']:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass

        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
        self.eigensolver = eigensolver

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, 'w', self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = None

        self.eta = eta
        self.v0 = v0

        self.neval = 0
        self.curr = dict(
            x=None,
            f=None,
            g=None,
        )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        self.savepoint = dict(apos=None, dpos=None)
        self.first_diag = True

        self.hessian_function = hessian_function

        self._basis_cache = _LRU2()

        # Limited-memory multi-secant update: the last secant_memory
        # (dx, dg) pairs of this PES object are imposed simultaneously
        # (B S = Y~ with the symmetrised Y of update_H), so the model becomes
        # exact on the span of the recent steps instead of only along the
        # newest one. The history dies with the PES object, i.e. whenever
        # the internals are rebuilt and the Hessian guess is reset.
        self.secant_memory = 4
        self.secant_dep_tol = 0.3
        self.secant_cons_tol = 0.25
        self._secant_pairs = []

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def _state_hash(self) -> bytes:
        """Hash of all state that affects cached computations."""
        h = self.atoms.positions.tobytes()
        cell = self.atoms.cell
        if cell is not None and cell.any():
            h += cell.array.tobytes()
        return h

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint['apos']
        dpos = self.savepoint['dpos']
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    def close(self):
        """Close any open file handles (e.g., trajectory file)."""
        if self.traj is not None:
            self.traj.close()
            self.traj = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures trajectory is closed."""
        self.close()
        return False

    # Position getter/setter
    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return diff, diff, self.curr.get('g', np.zeros_like(diff))

    def get_x(self):
        return self.apos.ravel().copy()

    # Hessian getter/setter
    def get_H(self):
        return self.H

    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(
            self.dim, self.ncart, target, *args, **kwargs
        )

    # Hessian of the constraints
    def get_Hc(self):
        if self.curr['L'] is None:
            raise RuntimeError(
                "PES.get_Hc() called with L=None. "
                f"curr_g_is_none={self.curr.get('g') is None}, "
                f"curr_f_is_none={self.curr.get('f') is None}."
            )
        return self.cons.hessian().ldot(self.curr['L'])

    # Hessian of the Lagrangian

    def get_HL_projected(self, U):
        """Projected Hessian of the Lagrangian: ApproximateHessian(U.T @ HL @ U).

        Equivalent to ``self.get_HL().project(U)`` but skips constructing the
        full (dim, dim) HL matrix and the intermediate ApproximateHessian.
        """
        H = self.get_H()
        H_B = H.B
        if H_B is None:
            Bproj = None
        else:
            UtHU = U.T @ H_B @ U
            # Skip the constraint projection entirely when there are no
            # constraints — Hc is allocated as a (dim, dim) zero block in
            # CellInternalPES, so the matmul would just churn through ~N^3
            # zeros (and force a 44 MB GPU upload at 400 atoms).
            L = self.curr.get('L')
            if L is not None and L.size > 0:
                Hc = self.get_Hc()
                Bproj = UtHU - U.T @ Hc @ U
            else:
                Bproj = UtHU
        n = U.shape[1]
        return ApproximateHessian(n, 0, Bproj, self.H.update_method, self.H.symm)

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    def get_drdx(self):
        return self.cons.jacobian()

    def _calc_basis(self):
        state_hash = self._state_hash()
        cached = self._basis_cache.get(state_hash)
        if cached is not None:
            return cached

        drdx = self.get_drdx()
        Ucons, Ufree = _split_cons_subspace(drdx)
        Unred = np.eye(self.dim)
        result = (drdx, Ucons, Unred, Ufree)

        self._basis_cache.put(state_hash, result)
        return result

    def write_traj(self):
        if self.traj is not None:
            self.traj.write()

    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.write_traj()
        return f, g

    def _calc_eg(self, x):
        self.save()
        self.set_x(x)

        f, g = self.eval()

        self.restore()
        return f, g

    def get_scons(self):
        """Returns displacement vector for linear constraint correction."""
        Ucons = self.get_Ucons()

        scons = -Ucons @ np.linalg.lstsq(
            self.get_drdx() @ Ucons,
            self.get_res(),
            rcond=None,
        )[0]
        return scons

    def _update(self, feval=True):
        state = self._state_hash()
        new_point = True
        if self.curr['x'] is not None and state == self.curr.get('state_hash'):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False
        x = self.get_x()
        basis = self._calc_basis()

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['state_hash'] = state
        self.curr['f'] = f
        self.curr['g'] = g
        self._update_basis(basis)
        return True

    def _update_basis(self, basis=None):
        if basis is None:
            basis = self._calc_basis()
        drdx, Ucons, Unred, Ufree = basis
        self.curr['drdx'] = drdx
        self.curr['Ucons'] = Ucons
        self.curr['Unred'] = Unred
        self.curr['Ufree'] = Ufree

        if self.curr['g'] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr['g'], rcond=None)[0]

        self.curr['L'] = L

    def _update_H(self, dx, dg, T_now=None, tbar=None, images=None):
        if self.last['x'] is None or self.last['g'] is None:
            return
        if self.secant_memory <= 1 or not self.H.initialized:
            if T_now is not None and tbar is not None:
                dg = np.asarray(dg, dtype=np.float64) + (T_now @ dx - tbar)
            self.H.update(dx, dg)
            return
        S, Y = self._collect_secant_pairs(dx, dg, T_now, tbar, images)
        if S is None:
            return
        self.H.update(S, Y)

    def _collect_secant_pairs(self, dx, dg, T_now=None, tbar=None,
                              images=None):
        """Return (S, Y) column matrices of the recent secant pairs.

        The newest pair is column 0 (symmetrize_Y2 keeps column 0 exact and
        adjusts later columns to make S.T @ Y symmetric). Columns are
        normalised by the step length so that pairs from the approach phase
        and from the endgame enter with the same weight, and older steps
        that are (nearly) linearly dependent on newer ones are dropped so
        that the m x m secant system of _MS_TS_BFGS stays well conditioned.

        Symmetry images: images is a list of maps (s, y, tb) -> (s', y',
        tb') of a pair under the point-group operations of the current
        geometry (InternalPES._symmetry_images).  The energy is invariant
        under them, so every stored pair has an exact image pair -- the
        pair the same steps would have produced from the mirrored /
        permuted start -- and the update learns up to |G| directions per
        force call.  The images of a pair follow it in the candidate order
        (a real pair always precedes its images and the older pairs),
        subject to the same dependence and consistency filters; the
        transport term tbar is mapped alike.

        Secant transport: a pair measured over the step x_j -> x_j+1 gives
        the path-averaged Hessian, y_j = H_avg s_j.  When the model has an
        analytic geometry-dependent part T(x) (InternalPES: the stretch
        diagonal of the bonds and the non-local contact block), the pair
        is moved to the current point x with that part,
        y_j -> y_j + (T(x) - T_avg,j) s_j, T_avg,j the path average of T
        over the segment (InternalPES._analytic_model_average),
        so that the secant conditions imposed on the shifted model describe
        the local curvature at x instead of the average over each old
        segment (which the update would otherwise re-impose along the
        sampled directions, undoing the shift of the model there).  tbar is
        T_avg s for the new pair; T_now is T(x); both None disables it.
        """
        dx = np.asarray(dx, dtype=np.float64)
        dg = np.asarray(dg, dtype=np.float64)
        nrm = np.linalg.norm(dx)
        if not np.isfinite(nrm) or nrm < 1e-8 or not np.all(np.isfinite(dg)):
            return None, None
        if tbar is not None:
            tbar = np.asarray(tbar, dtype=np.float64) / nrm
            if not np.all(np.isfinite(tbar)):
                tbar = None
        self._secant_pairs.insert(0, (dx / nrm, dg / nrm, tbar))
        del self._secant_pairs[self.secant_memory:]
        n = len(dx)
        candidates = []
        for s, y, tb in self._secant_pairs:
            candidates.append((s, y, tb))
            if not images or len(s) != n:
                continue
            for image in images:
                candidates.append(image(s, y, tb))
        basis = []
        S_cols = []
        Y_cols = []
        for s, y, tb in candidates:
            if tb is not None and T_now is not None:
                y = y + (T_now @ s - tb)
            r = s.copy()
            for q in basis:
                r -= (q @ r) * q
            rn = np.linalg.norm(r)
            if rn < self.secant_dep_tol:
                continue
            # Quadratic consistency with every newer pair kept so far: a
            # symmetric Hessian requires s_new.y_old == s_old.y_new. Pairs
            # that violate it (strongly anharmonic soft modes, or a step
            # taken before the coordinates rebuilt their gradient transport)
            # are dropped instead of being averaged in by symmetrize_Y2.
            c_old = abs(s @ y)
            consistent = True
            for s_new, y_new in zip(S_cols, Y_cols):
                c_new = abs(s_new @ y_new)
                a = s_new @ y
                b = s @ y_new
                scale = np.sqrt(c_old * c_new) + 1e-12
                if abs(a - b) > self.secant_cons_tol * scale:
                    consistent = False
                    break
            if not consistent:
                continue
            basis.append(r / rn)
            S_cols.append(s)
            Y_cols.append(y)
        return np.column_stack(S_cols), np.column_stack(Y_cols)

    def get_f(self):
        self._update()
        return self.curr['f']

    def get_g(self) -> np.ndarray:
        self._update()
        return self.curr['g'].copy()

    def get_Unred(self):
        self._update(False)
        return self.curr['Unred']

    def get_Ufree(self):
        self._update(False)
        return self.curr['Ufree']

    def get_Ucons(self):
        self._update(False)
        return self.curr['Ucons']

    def diag(self, gamma=0.1, threepoint=False, maxiter=None):
        if self.curr['f'] is None:
            self._update(feval=True)

        Ufree = self.get_Ufree()
        nfree = Ufree.shape[1]

        # If there are no free DOF, there's nothing to diagonalize
        if nfree == 0:
            return

        P = self.get_HL_projected(Ufree)
        P_is_none = P.B is None

        # Determine initial guess vector
        if P_is_none or self.first_diag:
            v0 = self.v0 if self.v0 is not None else self.get_g() @ Ufree
            # If v0 is near-zero, let rayleigh_ritz choose its own initial guess
            if v0 is not None and np.linalg.norm(v0) < 1e-12:
                v0 = None
        else:
            v0 = None

        # Convert P to array
        P = np.eye(nfree) if P_is_none else P.asarray()

        Hproj = NumericalHessian(self._calc_eg, self.get_x(), self.get_g(),
                                 self.eta, threepoint, Ufree)
        Hc = self.get_Hc()
        rayleigh_ritz(Hproj - Ufree.T @ Hc @ Ufree, gamma, P, v0=v0,
                      method=self.eigensolver,
                      maxiter=maxiter)

        # Extract eigensolver iterates
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Re-calculate Ritz vectors
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2) - Vs.T @ Hc @ Vs
        _, X = eigh(Atilde)

        # Rotate Vs and AVs into X
        Vs = Vs @ X
        AVs = AVs @ X

        # Update the approximate Hessian
        self.H.update(Vs, AVs)

        self.first_diag = False

    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        return -(Ufree @ (Ufree.T @ g)).reshape((-1, 3))

    def converged(self, fmax, cmax=1e-5):
        fmax1 = np.linalg.norm(self.get_projected_forces(), axis=1).max()
        cmax1 = np.linalg.norm(self.get_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def wrap_dx(self, dx):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + (dx.T @ H @ dx) / 2.

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()

        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        if df_pred is None or abs(df_pred) < 1e-14:
            ratio = None
        else:
            ratio = df_actual / df_pred

        self._update_H(dx_final, dg_actual)

        if diag:
            if self.hessian_function is not None:
                self.calculate_hessian()
            else:
                self.diag(**diag_kwargs)

        return ratio

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self.hessian_function(self.atoms))

class InternalPES(PES):
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        iterative_stepper: int = 0,
        auto_find_internals: bool = True,
        **kwargs
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()

        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs
        )

        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            Q, _ = qr(B, mode='economic')
            P = Q @ Q.T
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
            # The non-local contact term of the guess is geometry
            # dependent; remember it so that _update_H can move it along
            # with the atoms (see _track_nonlocal_contacts).
            self._nb_prev = getattr(self.int, '_h0_nonlocal_last', None)
            self._track_nb = True
        else:
            self.set_H(H0, initialized=True)
            self._nb_prev = None
            self._track_nb = False

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

        self._pinv_cache = _LRU2()
        self._qr_cache = _LRU2()
        self._Hc_cache = _LRU2()

        # The whole analytic part of the model (stretch diagonal and
        # contact block) follows the geometry and the secant pairs are
        # transported with it, see _track_analytic_model.
        self._transport = bool(self._track_nb)
        self._an_prev = self._analytic_model() if self._transport else None
        self._pos_prev = None
        self._shadow = None
        if self._transport:
            self._pos_prev = (self.atoms.positions.copy(),
                              self.dummies.positions.copy())
        # Geometry at the previous update, for the tolerance of the
        # symmetry images (see _symmetry_images).
        self._sym_pos_prev = self.atoms.positions.copy()

    dpos = property(lambda self: self.dummies.positions.copy())

    def _state_hash(self) -> bytes:
        h = super()._state_hash()
        h += self.dummies.positions.tobytes()
        return h

    # =========================================================================
    # Cache optimization: Store and reuse Jacobian QR and pseudo-inverse
    # =========================================================================

    def _get_jacobian_qr(self):
        """Get cached economy QR of internal Jacobian.

        Returns (Q, R) from np.linalg.qr(B, mode='reduced').
        Q (m, n) is the orthonormal basis for range(B) (= Unred).
        R (n, n) upper triangular, with R^{-1} replacing V S^{-1} from SVD.

        Shared between _get_Binv and _calc_basis so the Jacobian is
        decomposed at most once per geometry.  ~2x faster than SVD.
        Falls back to SVD if the Jacobian is rank-deficient.
        """
        state_hash = self._state_hash()
        cached = self._qr_cache.get(state_hash)
        if cached is not None:
            return cached

        B = self.int.jacobian()
        Q, R = np.linalg.qr(B, mode='reduced')

        # Check for rank deficiency via R diagonal
        rdiag = np.abs(np.diag(R))
        if len(rdiag) > 0 and rdiag.min() < 1e-6 * rdiag.max():
            # Rank-deficient: fall back to SVD for safe truncation
            Ui, Si, VTi = np.linalg.svd(B, full_matrices=False)
            nnred = np.sum(Si > 1e-6)
            Q = Ui[:, :nnred]
            R = np.diag(Si[:nnred]) @ VTi[:nnred]

            # Pre-compute Binv from SVD factors and cache it, since the
            # non-square R can't be used with solve_triangular in _get_Binv
            Siinv = np.diag(1.0 / Si[:nnred])
            Binv = VTi[:nnred].T @ Siinv @ Ui[:, :nnred].T
            self._pinv_cache.put(state_hash, Binv)

        self._qr_cache.put(state_hash, (Q, R))
        return Q, R

    def _get_Binv(self):
        """Get cached pseudo-inverse of internal Jacobian.

        Computes Binv = R^{-1} Q^T from the shared QR cache,
        using a triangular solve instead of a full SVD.
        """
        state_hash = self._state_hash()
        cached = self._pinv_cache.get(state_hash)
        if cached is not None:
            return cached

        Q, R = self._get_jacobian_qr()
        if R.size == 0:
            ncart = 3 * len(self.atoms) + (3 * len(self.dummies) if self.dummies else 0)
            Binv = np.empty((ncart, 0))
        elif R.shape[0] == R.shape[1]:
            Binv = solve_triangular(R, Q.T, check_finite=False)
        else:
            # Non-square R from rank-deficient SVD fallback — Binv should
            # already have been cached by _get_jacobian_qr, but recompute
            # as a safety net (e.g., if the 2-entry cache evicted it).
            B = self.int.jacobian()
            Binv = np.linalg.pinv(B)

        self._pinv_cache.put(state_hash, Binv)
        return Binv

    # =========================================================================
    # Iterative stepper with improved convergence checking
    # =========================================================================
    # Uses Newton-Raphson iteration with robust convergence detection:
    # - Strict absolute tolerance (1e-8) for convergence
    # - Divergence detection (2x initial error)
    # - Stagnation detection (3 consecutive iterations without progress)
    # - Final verification pass before accepting solution
    # Falls back to ODE integrator on failure.
    # =========================================================================

    def _set_x_iterative(self, target, max_iter=20):
        """Fast iterative stepper for internal coordinate updates.

        Uses Newton-Raphson iteration to update Cartesian positions to match
        target internal coordinates. Returns None if convergence fails.
        """
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        x0 = self.get_x()
        dx_initial = target - x0

        # Get initial gradient in Cartesian space
        g0 = self._get_Binv() @ self.curr.get('g', np.zeros_like(dx_initial))

        rms_prev = np.inf
        initial_rms = None
        pos_first = None
        dpos_first = None
        stagnation_count = 0

        for iteration in range(max_iter):
            residual = self.wrap_dx(target - self.get_x())
            rms = np.linalg.norm(residual) / np.sqrt(len(residual))

            if initial_rms is None:
                initial_rms = rms

            # Converged
            if rms < 1e-8:
                break

            # Check for divergence (getting significantly worse)
            if rms > initial_rms * 2.0:
                # Diverging, restore and fall back
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return None

            # Check for stagnation (after first few iterations)
            if iteration > 3:
                if rms > rms_prev * 0.95:
                    stagnation_count += 1
                    if stagnation_count >= 3:
                        # Stagnating, give up if we haven't made progress
                        if rms > initial_rms * 0.5:
                            self.atoms.positions = pos0
                            self.dummies.positions = dpos0
                            return None
                        break  # Accept partial convergence
                else:
                    stagnation_count = 0

            rms_prev = rms

            # Newton step
            dx = np.linalg.lstsq(
                self.int.jacobian(),
                residual,
                rcond=None,
            )[0].reshape((-1, 3))

            # Update positions
            self.atoms.positions += dx[:len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms):]

            # Save first iteration result as fallback
            if pos_first is None:
                pos_first = self.atoms.positions.copy()
                dpos_first = self.dummies.positions.copy()

            # Check for bad internals during iteration
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                # Restore and return None to trigger ODE fallback
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                self.bad_int = None
                return None

        # After loop, verify we actually converged well enough
        final_residual = self.wrap_dx(target - self.get_x())
        final_rms = np.linalg.norm(final_residual) / np.sqrt(len(dx_initial))
        if final_rms > 1e-6:
            # Didn't converge well enough, fall back to ODE
            self.atoms.positions = pos0
            self.dummies.positions = dpos0
            return None

        dx_final = self.get_x() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    def _set_x_ode(self, target):
        """ODE-based stepper for internal coordinate updates.

        Uses LSODA to integrate the geodesic equation for reliable convergence
        on large or ill-conditioned steps.
        """
        dx = self.wrap_dx(target - self.get_x())
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                view(self.atoms + self.dummies)
                raise RuntimeError("Geometry update ODE is taking too long "
                                   "to converge!")

        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")

        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    # Position getter/setter
    def set_x(self, target):
        """Update internal coordinates to target values.

        Uses fast iterative stepper by default, with ODE fallback for robustness.
        """
        if self.iterative_stepper:
            res = self._set_x_iterative(target)
            if res is not None:
                q_after_ode = self.int.calc().copy()
                proj_moved = self._project_to_constraints()
                dx_initial, dx_final_ode, g_final = res
                dx_final = self._add_proj_delta(dx_final_ode, q_after_ode,
                                                proj_moved)
                return dx_initial, dx_final, g_final
        # Fall back to ODE solver
        res = self._set_x_ode(target)
        q_after_ode = self.int.calc().copy()
        proj_moved = self._project_to_constraints()
        dx_initial, dx_final_ode, g_final = res
        dx_final = self._add_proj_delta(dx_final_ode, q_after_ode, proj_moved)
        return dx_initial, dx_final, g_final

    def _add_proj_delta(self, dx_int_final, q_after_ode, proj_moved):
        """Combine ODE-tangent dx with the projection's IC delta.

        ``dx_int_final`` is the tangent-integrated displacement returned
        by the ODE/iterative stepper — what BFGS expects as the secant.
        If the projection then nudged atoms, that extra motion is
        captured as ``delta_proj = int.calc() - q_after_ode`` (raw IC
        difference, with dihedrals wrapped to (-π, π] for safety).
        BFGS sees the sum so its `s = dx` matches `dg = g_after - g_before`.
        """
        if not proj_moved:
            return dx_int_final
        delta_proj = self.int.calc() - q_after_ode
        dih_start = (self.int.ntrans + self.int.nbonds
                     + self.int.nangles)
        dih_end = dih_start + self.int.ndihedrals
        if dih_end > dih_start:
            delta_proj[dih_start:dih_end] = (
                (delta_proj[dih_start:dih_end] + np.pi)
                % (2 * np.pi) - np.pi
            )
        return dx_int_final + delta_proj

    def _project_to_constraints(self, target_tol=1e-7, max_iter=8,
                                safety_limit=0.05):
        """Newton projection onto the constraint manifold (IC null-space).

        Drives ``cons.residual()`` to zero with corrections that, to
        first order, do not change any *free* internal coordinate.
        This avoids the failure mode of a Cartesian min-norm projection,
        which would tilt the dummy atom in directions that couple back
        into the free improper-dihedral bending coordinate.

        Algorithm (one Newton iteration, repeated):

            r       = cons.residual()                          # (ncons,)
            drdx    = d(cons) / d(int_coords)                   # (ncons, n_int)
            Ucons   = IC-space basis spanned by constraints     # (n_int, ncons')
            s       = lstsq(drdx @ Ucons, -r)                   # min-norm in Ucons
            dq_int  = Ucons @ s                                 # IC-space step
            dx_cart = Binv @ dq_int                             # back to Cartesian

        Because ``dq_int`` lives entirely in ``Ucons`` (orthogonal to
        ``Ufree`` in the IC inner product), every free internal — including
        the improper dihedral that parametrizes a linear-bend — is
        unchanged to first order. The Cartesian step is the
        minimum-norm representative of that IC-space step (``Binv`` is
        the pseudoinverse), so real atoms only move when the
        constraint *requires* it (e.g. ``FixBondLengths``).

        ``safety_limit`` caps ``|dx_cart|_inf`` per iteration. If the
        Newton step would exceed it, we bail and accept the partial
        residual — the alternative (damped re-iteration) was tested
        and found to *increase* opt step counts (~+30%) on the tier1+2
        benchmarks because the partial corrections accumulate as
        Hessian noise. Bailing leaves the projection as a strict
        improvement: it can only help, never hurt.
        """
        if self.cons.residual().size == 0:
            return False

        n_real = 3 * len(self.atoms)
        n_dummy = 3 * len(self.dummies)
        moved = False

        for _ in range(max_iter):
            r = self.cons.residual()
            if np.linalg.norm(r, ord=np.inf) < target_tol:
                return moved

            # _compute_basis_int returns (drdx, Ucons, Unred, Ufree) for the
            # internal-only block (no cell DOF). drdx is ncons × n_int in
            # IC space; Ucons is n_int × ncons'.
            drdx, Ucons, _, _ = self._compute_basis_int()
            if Ucons.shape[1] == 0:
                return moved  # no constraint subspace — nothing to project

            s, *_ = np.linalg.lstsq(drdx @ Ucons, -r, rcond=None)
            dq_int = Ucons @ s                       # IC-space step (n_int,)
            dx = self._get_Binv() @ dq_int            # Cartesian (n_cart,)

            if np.linalg.norm(dx, ord=np.inf) > safety_limit:
                return moved  # would override optimizer's step — bail

            self.atoms.positions += dx[:n_real].reshape(-1, 3)
            if n_dummy > 0:
                self.dummies.positions += dx[n_real:n_real + n_dummy].reshape(-1, 3)
            moved = True

        return moved

    def get_x(self):
        x = self.int.calc()
        if self.curr['x'] is not None:
            dih_start = (self.int.ntrans + self.int.nbonds
                         + self.int.nangles)
            dih_end = dih_start + self.int.ndihedrals
            if dih_end > dih_start:
                dx = x[dih_start:dih_end] - self.curr['x'][dih_start:dih_end]
                x[dih_start:dih_end] = (
                    self.curr['x'][dih_start:dih_end]
                    + (dx + np.pi) % (2 * np.pi) - np.pi
                )
        return x

    # Hessian of the constraints
    def _compute_Hc_int(self):
        """Compute the internal-coords-only constraint Hessian (uncached)."""
        if self.curr['L'] is None:
            raise RuntimeError(
                "InternalPES.get_Hc() called with L=None. "
                f"curr_g_is_none={self.curr.get('g') is None}, "
                f"curr_f_is_none={self.curr.get('f') is None}."
            )

        # No constraints → L is empty → Hc is identically zero. Skip the
        # expensive ldot/matmul chain (~95ms at 400 atoms).
        Binv_int = self._get_Binv()
        n_dof = Binv_int.shape[1]
        if self.curr['L'].size == 0:
            return np.zeros((n_dof, n_dof))

        D_cons = self.cons.hessian().ldot(self.curr['L'])
        B_cons = self.cons.jacobian()
        L_int = self.curr['L'] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        return Binv_int.T @ (D_cons - D_int) @ Binv_int

    def get_Hc(self):
        # Subclasses (CellInternalPES) cache the cell-extended form themselves
        # and call _compute_Hc_int directly, so we only cache here when this
        # *is* the runtime class.
        state_hash = self._state_hash()
        cached = self._Hc_cache.get(state_hash)
        if cached is not None:
            return cached

        Hc = self._compute_Hc_int()
        self._Hc_cache.put(state_hash, Hc)
        return Hc

    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ self._get_Binv()

    def _compute_basis_int(self):
        """Compute the internal-coords-only basis (uncached, fast path).

        Uses the cached jacobian QR factors. Subclasses (CellInternalPES) call
        this to obtain the internal block, then add their own cell extension
        and cache the combined result themselves.
        """
        cons = self.cons
        Q, R = self._get_jacobian_qr()
        Unred = Q

        n_int = Q.shape[0]
        cons_jac = cons.jacobian()
        if cons_jac.shape[0] == 0:
            # No constraints: all non-redundant DOF are free
            drdx = np.zeros((0, n_int))
            Ucons = np.zeros((n_int, 0))
            Ufree = Unred
        else:
            if R.shape[0] == R.shape[1]:
                # Full rank: cons_jac @ R^{-1} via triangular solve
                drdxnred = solve_triangular(
                    R.T, cons_jac.T, lower=True, check_finite=False
                ).T
            else:
                # Rank-deficient (SVD fallback in _get_jacobian_qr)
                Binv = self._get_Binv()
                drdxnred = cons_jac @ (Binv @ Q)
            drdx = drdxnred @ Q.T
            Vcons, Vfree = _split_cons_subspace(drdxnred)
            Ucons = Unred @ Vcons
            Ufree = Unred @ Vfree
        return drdx, Ucons, Unred, Ufree

    def _calc_basis(self, internal=None, cons=None):
        # If custom internal/cons provided, bypass cache (used by refine paths)
        if internal is not None or cons is not None:
            if internal is None:
                internal = self.int
            if cons is None:
                cons = self.cons
            B = internal.jacobian()
            Ui, Si, VTi = np.linalg.svd(B, full_matrices=False)
            nnred = np.sum(Si > 1e-6)
            Unred = Ui[:, :nnred]
            Vnred = VTi[:nnred].T
            Siinv = np.diag(1 / Si[:nnred])
            cons_jac = cons.jacobian()
            n_int = B.shape[0]
            if cons_jac.shape[0] == 0:
                # No constraints: all non-redundant DOF are free
                drdx = np.zeros((0, n_int))
                Ucons = np.zeros((n_int, 0))
                Ufree = Unred
            else:
                drdxnred = cons_jac @ Vnred @ Siinv
                drdx = drdxnred @ Unred.T
                Vcons, Vfree = _split_cons_subspace(drdxnred)
                Ucons = Unred @ Vcons
                Ufree = Unred @ Vfree
            return drdx, Ucons, Unred, Ufree

        # Subclasses (CellInternalPES) cache the cell-extended form themselves
        # and call _compute_basis_int directly, so we only cache here when
        # this *is* the runtime class.
        state_hash = self._state_hash()
        cached = self._basis_cache.get(state_hash)
        if cached is not None:
            return cached

        result = self._compute_basis_int()
        self._basis_cache.put(state_hash, result)
        return result

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = self._get_Binv()
        return f, g_cart @ Binv[:len(g_cart)]


    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.

    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        # Use cached jacobian from curr if available
        if 'B' in self.curr and self.curr['B'] is not None:
            B = self.curr['B']
        else:
            B = self.int.jacobian()
        return -(Ufree @ (Ufree.T @ g) @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        # Use direct HVP computation instead of forming full Hessians.
        # Batch the two D_rdot @ vector products into one (D_rdot @ matrix)
        # matmul, then one Binv @ matrix matmul, halving the matmul count.
        D_rdot = self.int.hessian_rdot(dxdt)
        Binv = self._ode_Binv
        rhs = np.column_stack((dxdt, g))     # (ndof, 2)
        out = -Binv @ (D_rdot @ rhs)          # (ndof, 2)
        dydt[1] = out[:, 0]
        dydt[2] = out[:, 1]

        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        return ratio

    def _track_nonlocal_contacts(self) -> None:
        """Move the analytic non-local contact term of the model Hessian
        to the current geometry: H <- H + A(x) - A(x_prev) (the fallback of
        _update_H when the full transport of _track_analytic_model is off).

        The guess Hessian is diag(h0) + A(x0), where A is the contact
        curvature of _h0_nonlocal_contacts.  The quasi-Newton updates learn
        along the steps taken, but a molecule that folds during the run
        forms contacts (and intramolecular hydrogen bonds) that did not
        exist at x0, and the fragments of a complex that start apart form
        their hydrogen bonds or ion pair only after several steps of
        approach and reorientation; that curvature acts in directions the
        secant pairs have not sampled.  Replacing A(x_prev) by A(x) before
        every secant update keeps the analytic part of the model at the
        current geometry (including the change of frame through the
        current pseudo-inverse Jacobian) while the learned correction is
        kept; the subsequent update re-imposes the secant conditions on the
        shifted matrix.  Near the minimum the geometry hardly changes and
        the shift vanishes.
        """
        if not getattr(self, '_track_nb', False) or self.H.B is None:
            return
        try:
            Hnb = self.int._h0_nonlocal_contacts(Binv=self._get_Binv())
        except (ValueError, np.linalg.LinAlgError):
            return
        prev = self._nb_prev
        shape = self.H.B.shape
        if Hnb is not None and Hnb.shape != shape:
            return
        if prev is not None and prev.shape != shape:
            self._nb_prev = Hnb
            return
        if Hnb is None and prev is None:
            return
        dA = np.zeros(shape, dtype=np.float64)
        if Hnb is not None:
            dA += Hnb
        if prev is not None:
            dA -= prev
        if not np.all(np.isfinite(dA)):
            return
        self.H.set_B(self.H.B + dA)
        self._nb_prev = Hnb

    @staticmethod
    def _analytic_model_from(int_obj, Q, Binv) -> Optional[np.ndarray]:
        """T(x) of the Internals object int_obj at its current positions,
        given the orthonormal range basis Q and the pseudo-inverse Binv of
        its Jacobian (see _analytic_model)."""
        if Q is None or Binv is None:
            return None
        try:
            Hnb = int_obj._h0_nonlocal_contacts(Binv=Binv)
        except (ValueError, np.linalg.LinAlgError):
            return None
        d = int_obj._h0_stretch_diagonal()
        if d is None or Q.ndim != 2 or Q.shape[0] != d.shape[0]:
            return None
        T = Q @ ((Q.T * d) @ Q) @ Q.T
        if Hnb is not None and Hnb.shape == T.shape:
            T = T + Hnb
        if not np.all(np.isfinite(T)):
            return None
        return T

    def _analytic_model(self) -> Optional[np.ndarray]:
        """The geometry-dependent analytic part T(x) of the model Hessian
        at the current geometry (nint x nint): the Almlof stretch diagonal
        of the bonds, projected like the guess (P diag(d) P with P = Q Q^T
        the projector onto the range of the Jacobian), plus the non-local
        contact block of _h0_nonlocal_contacts.
        """
        try:
            Binv = self._get_Binv()
            Q, _ = self._get_jacobian_qr()
        except (ValueError, np.linalg.LinAlgError):
            return None
        return self._analytic_model_from(self.int, Q, Binv)

    @staticmethod
    def _model_frame(B):
        """(Q, Binv) of a Jacobian B without the caches: the reduced QR
        (SVD truncation when rank deficient), as _get_jacobian_qr and
        _get_Binv do for the current geometry."""
        if B.ndim != 2 or B.size == 0:
            return None, None
        Q, R = np.linalg.qr(B, mode='reduced')
        rdiag = np.abs(np.diag(R))
        if len(rdiag) > 0 and rdiag.min() < 1e-6 * rdiag.max():
            Ui, Si, VTi = np.linalg.svd(B, full_matrices=False)
            nnred = int(np.sum(Si > 1e-6))
            if nnred == 0:
                return None, None
            Q = Ui[:, :nnred]
            Binv = VTi[:nnred].T @ (Ui[:, :nnred] / Si[:nnred]).T
        elif R.shape[0] == R.shape[1]:
            Binv = solve_triangular(R, Q.T, check_finite=False)
        else:
            Binv = np.linalg.pinv(B, rcond=1e-6)
        return Q, Binv

    def _analytic_model_at(self, pos_prev, dpos_prev, frac) -> Optional[np.ndarray]:
        """T on the Cartesian straight line from the previous geometry
        (pos_prev, dpos_prev: atoms and dummies) to the current one, at the
        fraction frac of the step, evaluated with a shadow copy of the
        internal coordinates."""
        pos_now = self.atoms.positions
        dpos_now = self.dummies.positions
        if pos_prev.shape != pos_now.shape or dpos_prev.shape != dpos_now.shape:
            return None
        if self._shadow is None:
            self._shadow = self.int.shadow_copy()
        shadow = self._shadow
        if (len(shadow.atoms) != len(pos_now)
                or len(shadow.dummies) != len(dpos_now)):
            return None
        shadow.sync_rotation_state(self.int)
        shadow.atoms.positions[:] = pos_prev + frac * (pos_now - pos_prev)
        if len(dpos_now):
            shadow.dummies.positions[:] = dpos_prev + frac * (dpos_now - dpos_prev)
        try:
            Q, Binv = self._model_frame(shadow.jacobian())
        except (ValueError, np.linalg.LinAlgError):
            return None
        return self._analytic_model_from(shadow, Q, Binv)

    def _analytic_model_average(self, T_prev, T_now, pos_prev, dpos_prev):
        """Path average of T over the step just taken: composite Simpson
        on the Cartesian straight line, with enough panels (at most four)
        that no panel spans more than 0.4 A of atomic displacement, i.e.
        at most 0.8 A of any distance -- b dr <= 3 for the exponent
        b = 3.7/A of the stretch and contact terms, where Simpson's error
        is below 1 % of the larger end value.  Falls back to the endpoint
        average when an interior point is unavailable.
        """
        pos_now = self.atoms.positions
        if pos_prev.shape != pos_now.shape:
            return 0.5 * (T_prev + T_now)
        dmax = float(np.max(np.linalg.norm(pos_now - pos_prev, axis=1))) if len(pos_now) else 0.
        npanel = int(min(4, max(1, np.ceil(dmax / 0.4))))
        acc = T_prev + T_now
        for i in range(1, 2 * npanel):
            T_i = self._analytic_model_at(pos_prev, dpos_prev, i / (2. * npanel))
            if T_i is None or T_i.shape != T_now.shape:
                return 0.5 * (T_prev + T_now)
            acc = acc + (4.0 if i % 2 else 2.0) * T_i
        return acc / (6.0 * npanel)

    def _track_analytic_model(self, dx):
        """Move the analytic part T(x) of the model Hessian to the current
        geometry, H <- H + T(x) - T(x_prev), and return (T(x), tbar) for
        the transport of the secant pairs (tbar = T_avg dx, the analytic
        contribution averaged over the step just taken).

        _track_nonlocal_contacts does this for the contact block alone.
        Here the stretch diagonal moves too: the Almlof exponential is the
        local curvature-length relation of a bond (Badger's rule), so after
        a bond has relaxed by 0.1 A the start-geometry constant is 30-45 %
        off; the secant pairs, being path averages, lag behind the local
        curvature by half the change per step.  Shifting the model alone
        would not help along the sampled directions, because the update
        re-imposes the measured (average) secants there; PES._update_H
        therefore also moves every stored pair with T, y_j <- y_j +
        (T(x) - T_avg,j) s_j, so that the secant conditions describe the
        curvature at x.  Where the geometry (bond lengths, contacts, frame)
        does not change between two points, T does not either and the
        update is the plain multi-secant one.

        T_avg is Simpson's rule over the step, (T(x_prev) + 4 T(x_mid) +
        T(x)) / 6 with x_mid the Cartesian midpoint (composite, in panels
        of at most 0.4 A of atomic displacement, see _analytic_model_average):
        the terms of T are exponentials of distances, and a contact or bond
        that opens or closes by 0.5-1 A in one step (the approach of two
        fragments, early steps of a folding chain, a dissociating bond)
        changes its k by a factor 6-40, for which the trapezoid
        (T(x_prev) + T(x)) / 2 overstates the path average by 0.1-0.25 of
        the larger end value -- more than the smaller end value itself --
        and would leave the transported pair too soft or even of the wrong
        sign along the step; Simpson is within 1-2 % of the exact average
        of an exponential over any such step.  The endpoint average is kept
        when the interior points cannot be evaluated.

        Multi-fragment systems: the inter-fragment contact block (radial
        pairs, hydrogen-bond bends, lone-pair-plane terms) grows 2-6 times
        while the fragments approach and settle, and the rigid-body
        coordinates are sampled by every step, so the stored pairs there
        are exactly the ones whose averages the model shift alone cannot
        correct; the transport gives the endgame the contact curvature of
        the current geometry instead of the approach-phase average.
        """
        if self.H.B is None:
            return None, None
        T_now = self._analytic_model()
        shape = self.H.B.shape
        if T_now is None or T_now.shape != shape:
            return None, None
        T_prev = self._an_prev
        pos_prev = self._pos_prev
        self._an_prev = T_now
        self._pos_prev = (self.atoms.positions.copy(),
                          self.dummies.positions.copy())
        if T_prev is None or T_prev.shape != shape:
            return None, None
        dA = T_now - T_prev
        if not np.all(np.isfinite(dA)):
            return None, None
        self.H.set_B(self.H.B + dA)
        dx = np.asarray(dx, dtype=np.float64)
        if pos_prev is not None:
            T_avg = self._analytic_model_average(T_prev, T_now, *pos_prev)
        else:
            T_avg = 0.5 * (T_prev + T_now)
        tbar = T_avg @ dx
        return T_now, tbar

    def _symmetry_images(self):
        """Image maps of the secant pairs under the point-group operations
        of the current geometry (Internals.symmetry_maps): a list of
        functions (s, y, tb) -> (s', y', tb').  The energy is invariant
        under the permutation of identical nuclei and the orthogonal map
        of an operation, so at a geometry invariant under it every pair
        measured along the last steps has the exact image the same steps
        would have produced from the mirrored / permuted start.  The image
        is formed through Cartesian space with the Jacobian of the current
        geometry: s' = B G (B^+ s) for the displacement and y' = B^+T G
        (B^T y) for the gradient change (atoms only, the dummies carry no
        force), which is the signed permutation of the coordinates when
        the coordinate set is closed under the operation and the correct
        linear combination when it is not (a single improper dihedral at
        a symmetric centre).  An operation is used when the mapped
        geometry is within max(_SYM_TOL, _SYM_STEP_FRAC times the rms
        atomic step just taken) of the actual one: at _SYM_TOL = 0.05 A the
        curvature error of an image (2 a eps with the stretch
        anharmonicity a ~ 2 /A) is about 20 % for the stiffest coordinates
        and a few per cent for the bends, below the guess error of any
        direction the real pairs have not sampled, and an image displaced
        by less than half the step is at least as accurate as the real
        pair itself, whose secant averages the Hessian over the whole
        step.  Only the images of the stored real pairs are formed; the
        memory of real pairs is unchanged."""
        pos_now = self.atoms.positions
        prev = self._sym_pos_prev
        self._sym_pos_prev = pos_now.copy()
        step_rms = 0.0
        if prev is not None and prev.shape == pos_now.shape:
            step_rms = float(np.sqrt(np.mean(np.sum((pos_now - prev) ** 2,
                                                   axis=1))))
        tol = max(_SYM_TOL, _SYM_STEP_FRAC * step_rms)
        try:
            maps = self.int.symmetry_maps(tol)
        except (ValueError, IndexError, np.linalg.LinAlgError):
            return []
        if not maps:
            return []
        B = self.int.jacobian()
        Binv = self._get_Binv()
        nat = len(self.atoms)
        ncart = 3 * nat
        if B.shape[1] != Binv.shape[0] or Binv.shape[0] < ncart:
            return []
        Ba = B[:, :ncart]
        Binva = Binv[:ncart]

        def make(perm, R, dperm, dflip, dcentre):
            Rt = R.T
            inv = dflip < 0

            def disp(s):
                u = (Binv @ s).reshape((-1, 3))
                va = u[:nat][perm] @ Rt
                if len(dperm):
                    vd = u[nat:][dperm] @ Rt
                    if np.any(inv):
                        vd[inv] = 2.0 * va[dcentre[inv]] - vd[inv]
                    v = np.concatenate([va, vd])
                else:
                    v = va
                return B @ v.ravel()

            def cov(w):
                wa = (w @ Ba).reshape((-1, 3))
                return (wa[perm] @ Rt).ravel() @ Binva

            def image(s, y, tb):
                return disp(s), cov(y), (None if tb is None else cov(tb))

            return image

        return [make(*m) for m in maps]

    def _update_H(self, dx, dg):
        images = self._symmetry_images()
        if getattr(self, '_transport', False):
            T_now, tbar = self._track_analytic_model(dx)
            PES._update_H(self, dx, dg, T_now, tbar, images)
            return
        self._track_nonlocal_contacts()
        PES._update_H(self, dx, dg, images=images)

    def write_traj(self):
        if self.traj is not None:
            energy = self.atoms.calc.results['energy']
            forces = np.zeros((len(self.atoms) + len(self.dummies), 3))
            forces[:len(self.atoms)] = self.atoms.calc.results['forces']
            atoms_tmp = self.atoms + self.dummies
            atoms_tmp.calc = SinglePointCalculator(atoms_tmp, energy=energy,
                                                   forces=forces)
            self.traj.write(atoms_tmp)

    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return

        B = self.int.jacobian()
        Binv = self._get_Binv()  # Use cached version instead of recomputing
        self.curr.update(B=B, Binv=Binv)
        return True

    def _convert_cartesian_hessian_to_internal(
        self,
        Hcart: np.ndarray,
    ) -> np.ndarray:
        ncart = 3 * len(self.atoms)
        # Get Jacobian and calculate redundant and non-redundant spaces
        B = self.int.jacobian()[:, :ncart]
        Ui, Si, VTi = np.linalg.svd(B, full_matrices=True)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Ured = Ui[:, nnred:]

        # Calculate inverse Jacobian in non-redundant space
        Bnred_inv = VTi[:nnred].T @ np.diag(1 / Si[:nnred])

        # Convert Cartesian Hessian to non-redundant internal Hessian
        Hcart_coupled = self.int.hessian().ldot(self.get_g())[:ncart, :ncart]
        Hcart_corr = Hcart - Hcart_coupled
        Hnred = Bnred_inv.T @ Hcart_corr @ Bnred_inv

        # Find eigenvalues of non-redundant internal Hessian
        lnred, _ = np.linalg.eigh(Hnred)

        # The redundant part of the Hessian will be initialized to the
        # geometric mean of the non-redundant eigenvalues
        lnred_mean = np.exp(np.log(np.abs(lnred)).mean())

        # finish reconstructing redundant internal Hessian
        return Unred @ Hnred @ Unred.T + lnred_mean * Ured @ Ured.T


    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self._convert_cartesian_hessian_to_internal(
            self.hessian_function(self.atoms)
        ))

class BaseStepper:
    alpha0: Optional[float] = None
    alphamin: Optional[float] = None
    alphamax: Optional[float] = None
    # Whether the step size increases or decreases with increasing alpha
    slope: Optional[float] = None
    # Whether get_s is smooth enough for Newton to converge reliably
    newton_safe: bool = True
    synonyms: List[str] = []

    def __init__(
        self,
        g: np.ndarray,
        H: ApproximateHessian,
        order: int = 0,
        d1: Optional[np.ndarray] = None,
    ) -> None:
        self.g = g
        self.H = H
        self.order = order
        self.d1 = d1
        self._stepper_init()

    @classmethod
    def match(cls, name: str) -> bool:
        return name in cls.synonyms

    def _stepper_init(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError  # pragma: no cover

class NaiveStepper(BaseStepper):
    synonyms = []  # No synonyms, we don't want someone using this accidentally
    alpha0 = 0.5
    alphamin = 0.
    alphamax = 1.
    slope = 1.

    def __init__(self, dx: np.ndarray) -> None:
        self.dx = dx

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        return alpha * self.dx, self.dx

class QuasiNewton(BaseStepper):
    alpha0 = 0.
    alphamin = 0.
    alphamax = np.inf
    slope = -1
    synonyms = [
        'qn',
        'quasi-newton',
        'quasi newton',
        'quasi-newton',
        'newton',
        'mmf',
        'minimum mode following',
        'minimum-mode following',
        'dimer',
    ]

    def _stepper_init(self) -> None:
        # Get eigenvalues and eigenvectors from the Hessian
        # If not already computed, compute them now
        if self.H.evals is None:
            H_array = self.H.asarray()
            self.H.evals, self.H.evecs = eigh(H_array)

        self.L = np.abs(self.H.evals)
        self.L[:self.order] *= -1

        self.V = self.H.evecs
        self.Vg = self.V.T @ self.g

        self.ones = np.ones_like(self.L)
        self.ones[:self.order] = -1

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        denom = self.L + alpha * self.ones
        sproj = self.Vg / denom
        s = -self.V @ sproj
        dsda = self.V @ (sproj / denom)
        return s, dsda

class RationalFunctionOptimization(BaseStepper):
    alpha0 = 1.
    alphamin = 0.
    alphamax = 1.
    slope = 1.
    newton_safe = False
    synonyms = ['rfo', 'rational function optimization']

    def _stepper_init(self) -> None:
        self.A = np.block([
            [self.H.asarray(), self.g[:, np.newaxis]],
            [self.g, 0]
        ])

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        A = self.A * alpha
        A[:-1, :-1] *= alpha
        L, V = eigh(A)

        # Regularize denominator to avoid division by near-zero eigenvector component
        denom = V[-1, self.order]
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        s = V[:-1, self.order] * alpha / denom

        dAda = self.A.copy()
        dAda[:-1, :-1] *= 2 * alpha

        V1 = np.delete(V, self.order, 1)
        L1 = np.delete(L, self.order)

        # Regularize eigenvalue differences: clamp small values while preserving sign
        L_diff = L1 - L[self.order]
        L_diff = np.where(L_diff >= 0,
                         np.maximum(L_diff, 1e-12),
                         np.minimum(L_diff, -1e-12))
        # Reassociate to do two matvecs (V1.T @ dAda is otherwise a (k-1, k)
        # matmul that costs ~25× more for the same final vector result).
        dVda = V1 @ ((V1.T @ (dAda @ V[:, self.order])) / L_diff)

        dsda = (V[:-1, self.order] / denom
                + (alpha / denom) * dVda[:-1]
                - (V[:-1, self.order] * alpha / denom**2) * dVda[-1])
        return s, dsda

class PartitionedRationalFunctionOptimization(RationalFunctionOptimization):
    synonyms = ['prfo', 'p-rfo', 'partitioned rational function optimization']

    def _stepper_init(self) -> None:
        self.Vmax = self.H.evecs[:, :self.order]
        self.Vmin = self.H.evecs[:, self.order:]

        self.max = RationalFunctionOptimization(
            self.Vmax.T @ self.g,
            self.H.project(self.Vmax),
            order=self.Vmax.shape[1],
        )

        self.min = RationalFunctionOptimization(
            self.Vmin.T @ self.g,
            self.H.project(self.Vmin),
            order=0,
        )

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        smax, dsmaxda = self.max.get_s(alpha)
        smin, dsminda = self.min.get_s(alpha)

        s = self.Vmax @ smax + self.Vmin @ smin
        dsda = self.Vmax @ dsmaxda + self.Vmin @ dsminda
        return s, dsda

_all_steppers = [
    QuasiNewton,
    RationalFunctionOptimization,
    PartitionedRationalFunctionOptimization,
]

def get_stepper(name: str) -> Type[BaseStepper]:
    for stepper in _all_steppers:
        if stepper.match(name):
            return stepper
    raise ValueError("Unknown stepper name: {}".format(name))

class BaseRestrictedStep:
    synonyms: List[str] = []

    def __init__(
        self,
        pes: Union[PES, InternalPES],
        order: int,
        delta: float,
        method: str = 'qn',
        tol: float = None,
        maxiter: int = 1000,
        d1: Optional[np.ndarray] = None,
        W: Optional[np.ndarray] = None,
    ):
        self.pes = pes
        self.delta = delta
        self.d1 = d1
        g0 = self.pes.get_g()

        # W defaults to the identity, in which case Ufree.T @ W == Ufree.T.
        # Skip allocating an n_dof x n_dof eye for the (very common)
        # default case to avoid quadratic zero-fill on large systems.
        self._W_is_identity = (W is None)

        self.scons = self.pes.get_scons()
        # TODO: Should this be HL instead of H?
        g = g0 + self.pes.get_H() @ self.scons

        if inspect.isclass(method) and issubclass(method, BaseStepper):
            stepper = method
        else:
            stepper = get_stepper(method.lower())

        if self.cons(self.scons) - self.delta > 1e-8:
            self.P = self.pes.get_Unred().T
            dx = self.P @ self.scons
            self.stepper = NaiveStepper(dx)
            self.scons[:] *= 0
        else:
            if self._W_is_identity:
                self.P = self.pes.get_Ufree().T
            else:
                self.P = self.pes.get_Ufree().T @ W
            d1 = self.d1
            if d1 is not None:
                d1 = np.linalg.lstsq(self.P.T, d1, rcond=None)[0]
            self.stepper = stepper(
                self.P @ g,
                self.pes.get_HL_projected(self.P.T),
                order,
                d1=d1,
            )

        if tol is None:
            tol = 1e-10 if self.stepper.newton_safe else 1e-15
        self.tol = tol
        self.maxiter = maxiter

    def cons(self, s, dsda=None):
        raise NotImplementedError

    def eval(self, alpha):
        s, dsda = self.stepper.get_s(alpha)
        stot = self.P.T @ s + self.scons
        val, dval = self.cons(stot, self.P.T @ dsda)
        return stot, val, dval

    def get_s(self):
        alpha = self.stepper.alpha0

        s, val, dval = self.eval(alpha)
        if val < self.delta:
            assert val > 0.
            return s, val
        err = val - self.delta

        lower = self.stepper.alphamin
        upper = self.stepper.alphamax

        for niter in range(self.maxiter):
            if abs(err) <= self.tol:
                break

            if np.nextafter(lower, upper) >= upper:
                break

            if err * self.stepper.slope > 0:
                upper = alpha
            else:
                lower = alpha

            a1 = alpha - err / dval
            if np.isnan(a1) or a1 <= lower or a1 >= upper or (
                niter > 4 and not self.stepper.newton_safe
            ):
                a2 = (lower + upper) / 2.
                if np.isinf(a2):
                    alpha = alpha + max(1, 0.5 * alpha) * np.sign(a2)
                else:
                    alpha = a2
            else:
                alpha = a1

            s, val, dval = self.eval(alpha)
            err = val - self.delta
        else:
            raise RuntimeError("Restricted step failed to converge!")

        assert val > 0
        return s, self.delta

    @classmethod
    def match(cls, name):
        return name in cls.synonyms

class TrustRegion(BaseRestrictedStep):
    synonyms = [
        'tr',
        'trust region',
        'trust-region',
        'trust radius',
        'trust-radius',
    ]

    def cons(self, s, dsda=None):
        val = np.linalg.norm(s)
        if dsda is None:
            return val

        dval = dsda @ s / max(val, 1e-12)
        return val, dval

class RestrictedAtomicStep(BaseRestrictedStep):
    synonyms = ['ras', 'restricted atomic step']

    def __init__(self, pes, *args, **kwargs):
        if pes.int is not None:
            raise ValueError(
                "Internal coordinates are not compatible with "
                f"the {self.__class__.__name__} trust region method."
            )
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        s_mat = s.reshape((-1, 3))
        s_norms = np.linalg.norm(s_mat, axis=1)
        index = np.argmax(s_norms)
        val = s_norms[index]

        if dsda is None:
            return val

        dsda_mat = dsda.reshape((-1, 3))
        dval = dsda_mat[index] @ s_mat[index] / max(val, 1e-12)
        return val, dval

class MaxInternalStep(BaseRestrictedStep):
    synonyms = ['mis', 'max internal step']

    def __init__(
        self, pes, *args, wx=1., wb=1., wa=1., wd=1., wo=1., wc=1.,
        cart_ratio=None, **kwargs
    ):
        if pes.int is None:
            raise ValueError(
                f"Internal coordinates are required for the "
                f"{self.__class__.__name__} trust region method"
            )
        self.wx = wx
        self.wb = wb
        self.wa = wa
        self.wd = wd
        self.wo = wo
        self.wc = wc  # Weight for cell DOF
        # Optional Cartesian bound: the largest linearised atomic
        # displacement of the step (Binv @ s, real atoms only) counts as
        # a step of size |dx_max| / cart_ratio, so it may not exceed
        # cart_ratio * delta (A). None disables the bound.
        self.cart_ratio = cart_ratio
        self._ncart_atoms = 3 * len(pes.atoms)
        self._weights_cache = None
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        w = self._get_weights()
        assert len(w) == len(s)

        sw = np.abs(s * w)
        idx = np.argmax(np.abs(sw))
        val = sw[idx]
        if dsda is not None:
            dval = np.sign(s[idx]) * dsda[idx] * w[idx]

        if self.cart_ratio is not None and len(s) > 0:
            # Linearised Cartesian displacement of the (non-redundant)
            # internal step; the step lies in range(B), so Binv @ s is
            # the minimum-norm Cartesian move realising it (no net
            # translation/rotation).
            Binv = self.pes._get_Binv()[:self._ncart_atoms]
            nint = Binv.shape[1]
            dx = (Binv @ s[:nint]).reshape((-1, 3))
            dx_norms = np.linalg.norm(dx, axis=1)
            ia = np.argmax(dx_norms)
            val_c = dx_norms[ia] / self.cart_ratio
            if val_c > val:
                val = val_c
                if dsda is not None:
                    ddx = (Binv @ dsda[:nint]).reshape((-1, 3))
                    dval = (ddx[ia] @ dx[ia]
                            / max(dx_norms[ia], 1e-12) / self.cart_ratio)

        if dsda is None:
            return val
        return val, dval

    def _get_weights(self):
        """Build the per-DOF weight vector. Cached against
        (counts, weights, n_cell_dof) so the np.array construction
        only runs once per restricted-step instance."""
        cached = self._weights_cache
        n_cell_dof = self.pes.n_cell_dof
        key = (
            self.pes.int.ntrans, self.pes.int.nbonds,
            self.pes.int.nangles, self.pes.int.ndihedrals,
            self.pes.int.nother, self.pes.int.nrotations,
            n_cell_dof,
        )
        if cached is not None and cached[0] == key:
            return cached[1]
        w = np.array(
            [self.wx] * self.pes.int.ntrans
            + [self.wb] * self.pes.int.nbonds
            + [self.wa] * self.pes.int.nangles
            + [self.wd] * self.pes.int.ndihedrals
            + [self.wo] * self.pes.int.nother
            + [self.wx] * self.pes.int.nrotations
        )
        if n_cell_dof > 0:
            w = np.concatenate([w, [self.wc] * n_cell_dof])
        self._weights_cache = (key, w)
        return w

_all_restricted_step = [TrustRegion, RestrictedAtomicStep, MaxInternalStep]

def get_restricted_step(name):
    for rs in _all_restricted_step:
        if rs.match(name):
            return rs
    raise ValueError("Unknown restricted step name: {}".format(name))

logger = logging.getLogger(__name__)

_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=4./3.,
        rho_dec=100,
        # Minimisation: growth-only textbook policy (Nocedal & Wright
        # ch. 4) -- after a step whose energy drop was at least 75 % of the
        # model's prediction, including drops *larger* than predicted, the
        # radius grows x1.5 up to a cap (max internal component, A or rad):
        # delta_max_mol for connected (single-fragment) systems,
        # delta_max_tr when the internal set carries fragment translation/
        # rotation coordinates (soft, anharmonic intermolecular surfaces
        # tolerate smaller steps). A failed step (rho outside (0.01, 100))
        # halves the radius (x0.9 was matched to the x1.15 growth of the
        # saddle-search window policy and needs ~15 failures to return
        # from the cap to delta0).
        sigma_inc_mol=1.5,
        delta_max_mol=0.5,
        delta_max_tr=0.25,
        sigma_dec_mol=0.5,
        # Initial radius for connected systems. Most connected molecules of
        # the sets start far from their minimum (max atomic displacement
        # over the run >= 1 A for 40-60 % of them, RMSD ~0.9 A), so their
        # first quasi-Newton steps are trust-limited; from 0.1 the x1.5
        # growth needs four accepted steps to reach the 0.5 cap, from 0.25
        # two. Multi-fragment systems keep delta0 (their first steps are
        # limited by the fragment-coordinate model, not by the radius).
        delta0_mol=0.25,
        # Cartesian bound of the internal trust region for connected
        # systems: the linearised atomic displacement of the step
        # (Binv @ s, real atoms) may not exceed cart_ratio_mol * delta
        # (A). The max-internal-component measure is blind to lever arms:
        # a 0.25 rad twist of a chain of dihedrals swings a distal group by
        # several A (compounding along the chain), far beyond the region
        # where the quadratic model knows about non-bonded contacts.
        cart_ratio_mol=2.0,
        method='qn',
        eig=False
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method='prfo',
        eig=True
    )
)

class Sella(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: bool = None,
        logfile: str = '-',
        trajectory: Union[str, Trajectory] = None,
        master: bool = None,
        delta0: float = None,
        sigma_inc: float = None,
        sigma_dec: float = None,
        rho_dec: float = None,
        rho_inc: float = None,
        order: int = 1,
        eig: bool = None,
        eta: float = 1e-4,
        method: str = None,
        gamma: float = 0.1,
        threepoint: bool = False,
        constraints: Constraints = None,
        constraints_tol: float = 1e-5,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        append_trajectory: bool = False,
        rs: str = None,
        nsteps_per_diag: int = 3,
        diag_every_n: Optional[int] = None,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        smax: float = None,
        allow_fragments: bool = False,
        niggli: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        **kwargs
    ):
        """Initialize Sella optimizer.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to optimize.
        optimize_cell : bool, optional
            If True, optimize unit cell parameters along with atomic positions.
            Requires order=0. Default is False.
        cell_mask : ndarray, optional
            Boolean mask of shape (3, 3) indicating which cell DOF are free.
            Default is all True (full cell optimization).
        exp_cell_factor : float, optional
            Scaling factor for cell parameterization. Default is number of atoms.
        scalar_pressure : float, optional
            External pressure in eV/Å³ for cell optimization. Default is 0.
        smax : float, optional
            Maximum stress tolerance for convergence when optimize_cell=True.
            If None, uses fmax.
        allow_fragments : bool, optional
            If True, allow disconnected molecular fragments when using internal
            coordinates. Adds translation and rotation coordinates (TRICs) for
            each fragment. Useful for molecular crystals. Default is False.
        niggli : bool, optional
            If True, apply Niggli reduction during cell optimization when cell
            angles deviate more than 30 deg from 90 deg. This remaps to the
            most compact unit cell and resets the Hessian cell block.
            Default is False.
        refine_initial_hessian : bool or int, optional
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only (2 * n_cell_dof force calls)
            - 2: Also refine translation/rotation blocks for molecular crystals
              (adds 2 * n_tric force calls, where n_tric = n_fragments * 6)
            - 3: Refine full internal Hessian (2 * n_internal force calls, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        # Validate cell optimization parameters
        self.optimize_cell = optimize_cell
        self.allow_fragments = allow_fragments
        self.niggli = niggli
        self.smax = smax
        if optimize_cell:
            if order != 0:
                raise ValueError(
                    "Cell optimization is only supported for minima (order=0), "
                    f"got order={order}."
                )
            if not np.any(atoms.pbc):
                raise ValueError(
                    "Cell optimization requires periodic boundary conditions. "
                    "Set atoms.pbc = True for periodic systems."
                )

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)
            # Register trajectory for cleanup when close() is called
            self.closelater(trajectory)

        asetraj = None
        self.peskwargs = kwargs.copy()
        self.user_internal = internal
        self.initialize_pes(
            atoms,
            trajectory,
            order,
            eta,
            constraints,
            v0,
            internal,
            hessian_function,
            optimize_cell=optimize_cell,
            cell_mask=cell_mask,
            exp_cell_factor=exp_cell_factor,
            scalar_pressure=scalar_pressure,
            allow_fragments=allow_fragments,
            refine_initial_hessian=refine_initial_hessian,
            save_hessian=save_hessian,
            **kwargs
        )

        if rs is None:
            rs = 'mis' if internal else 'ras'
        self.rs = get_restricted_step(rs)
        Optimizer.__init__(self, atoms, restart=restart,
                           logfile=logfile, trajectory=asetraj,
                           master=master)

        if delta0 is None:
            delta0 = default['delta0']
            if (order == 0 and 'delta0_mol' in default
                    and not self._has_tr_internals()):
                delta0 = default['delta0_mol']
        if rs in ['mis', 'ras']:
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]
        self.delta_cell = delta0

        self.sigma_inc = sigma_inc if sigma_inc is not None else default['sigma_inc']
        self.sigma_dec = sigma_dec if sigma_dec is not None else default['sigma_dec']
        self.rho_inc = rho_inc if rho_inc is not None else default['rho_inc']
        self.rho_dec = rho_dec if rho_dec is not None else default['rho_dec']
        self.sigma_inc_mol = default.get('sigma_inc_mol', self.sigma_inc)
        self.delta_max_mol = default.get('delta_max_mol', np.inf)
        self.delta_max_tr = default.get('delta_max_tr', self.delta_max_mol)
        self.sigma_dec_mol = default.get('sigma_dec_mol', self.sigma_dec)
        self.cart_ratio_mol = default.get('cart_ratio_mol', None)
        self.method = method if method is not None else default['method']
        self.eig = eig if eig is not None else default['eig']

        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.diagkwargs = dict(gamma=gamma, threepoint=threepoint)
        self.rho = 1.

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.
        self.nsteps_per_diag = nsteps_per_diag

        # Set by run() / first converged() call.
        self.fmax = None
        self._last_converged = None
        self.nsteps_since_diag = 0
        self.diag_every_n = np.inf if diag_every_n is None else diag_every_n

    def initialize_pes(
        self,
        atoms: Atoms,
        trajectory: str = None,
        order: int = 1,
        eta: float = 1e-4,
        constraints: Constraints = None,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        allow_fragments: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        **kwargs
    ):
        if internal:
            if isinstance(internal, Internals):
                auto_find_internals = False
                if constraints is not None:
                    raise ValueError(
                        "Internals object and Constraint object cannot both "
                        "be provided to Sella. Instead, you must pass the "
                        "Constraints object to the constructor of the "
                        "Internals object."
                    )
            else:
                auto_find_internals = True
                internal = Internals(
                    atoms, cons=constraints, allow_fragments=allow_fragments,
                )
            self.internal = internal.copy()
            self.constraints = None

            if optimize_cell:
                raise NotImplementedError(
                    'Cell optimisation is not part of this minimisation-only build; '
                    'minimize_func always builds a non-periodic Atoms object.')
            else:
                self.pes = InternalPES(
                    atoms,
                    internals=internal,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    auto_find_internals=auto_find_internals,
                    hessian_function=hessian_function,
                    **kwargs
                )
        else:
            self.internal = None
            if constraints is None:
                constraints = Constraints(atoms)
            self.constraints = constraints
            if optimize_cell:
                raise NotImplementedError(
                    'Cell optimisation is not part of this minimisation-only build; '
                    'minimize_func always builds a non-periodic Atoms object.')
            else:
                self.pes = PES(
                atoms,
                constraints=constraints,
                trajectory=trajectory,
                eta=eta,
                v0=v0,
                hessian_function=hessian_function,
                **kwargs
            )
        self.trajectory = self.pes.traj

    def _has_tr_internals(self):
        """True when the internal coordinate set currently contains
        fragment translation/rotation coordinates (multi-fragment system)."""
        if not self.internal or getattr(self.pes, 'int', None) is None:
            return False
        return (self.pes.int.ntrans + self.pes.int.nrotations) > 0

    def _predict_step(self):
        if not self.initialized:
            self.pes.get_g()
            if self.eig:
                if self.pes.hessian_function is not None:
                    self.pes.calculate_hessian()
                else:
                    self.pes.diag(**self.diagkwargs)
                self.nsteps_since_diag = -1
            self.initialized = True

        self.pes.cons.disable_satisfied_inequalities()
        self.pes._update_basis()
        self.pes.save()
        x0 = self.pes.get_x()

        rs_kwargs = {}
        if self.optimize_cell and isinstance(self.rs, type) and issubclass(
            self.rs, MaxInternalStep
        ):
            rs_kwargs['wc'] = self.delta / self.delta_cell
        if (self.ord == 0 and self.cart_ratio_mol is not None
                and isinstance(self.rs, type)
                and issubclass(self.rs, MaxInternalStep)
                and not self._has_tr_internals()):
            # Connected systems: also bound the linearised atomic
            # displacement of the internal step (see cart_ratio_mol).
            rs_kwargs['cart_ratio'] = self.cart_ratio_mol

        if self.pes.cons.has_inequalities():
            all_valid = False
            while not all_valid:
                s, smag = self.rs(
                    self.pes, self.ord, self.delta, method=self.method,
                    **rs_kwargs
                ).get_s()
                self.pes.set_x(x0 + s)
                all_valid = self.pes.cons.validate_inequalities()
                self.pes._update_basis()
                self.pes.restore()
            self.pes._update_basis()
        else:
            s, smag = self.rs(
                self.pes, self.ord, self.delta, method=self.method,
                **rs_kwargs
            ).get_s()

        return s, smag

    def step(self):
        s, smag = self._predict_step()

        # Determine if we need to call the eigensolver, then step
        if self.nsteps_since_diag >= self.diag_every_n:
            ev = True
        elif self.eig and self.nsteps_since_diag >= self.nsteps_per_diag:
            if self.pes.H.evals is None:
                ev = True
            else:
                Unred = self.pes.get_Unred()
                ev = (self.pes.get_HL_projected(Unred)
                                       .evals[:self.ord] > 0).any()
        else:
            ev = False

        if ev:
            self.nsteps_since_diag = 0
        else:
            self.nsteps_since_diag += 1

        rho = self.pes.kick(s, ev, **self.diagkwargs)

        # Check for bad internals, and if found, reset PES object.
        # This skips the trust radius update.
        if self.internal and self.pes.int.check_for_bad_internals():
            if False:
                cell_mask = self.pes.cell_mask
                exp_cell_factor = self.pes.exp_cell_factor
                scalar_pressure = self.pes.scalar_pressure
            else:
                cell_mask = None
                exp_cell_factor = None
                scalar_pressure = 0.0
            self.initialize_pes(
                atoms=self.pes.atoms,
                trajectory=self.pes.traj,
                order=self.ord,
                eta=self.pes.eta,
                constraints=self.constraints,
                v0=None,  # TODO: use leftmost eigenvector from old H
                internal=self.user_internal,
                hessian_function=self.pes.hessian_function,
                optimize_cell=self.optimize_cell,
                cell_mask=cell_mask,
                exp_cell_factor=exp_cell_factor,
                scalar_pressure=scalar_pressure,
                allow_fragments=self.allow_fragments,
            )
            self.initialized = False
            self.rho = 1
            return

        # Update trust radius
        if rho is not None:
            if self.optimize_cell and False:
                n_int = self.pes.n_internal
                smag_int = np.max(np.abs(s[:n_int])) if n_int > 0 else 0
                smag_cell = np.max(np.abs(s[n_int:])) if len(s) > n_int else 0
            else:
                smag_int = smag
                smag_cell = 0

            if rho < 1./self.rho_dec or rho > self.rho_dec:
                sigma_dec = self.sigma_dec
                if self.ord == 0:
                    # Minimisation: faster shrink matched to the growth-only
                    # policy below.
                    sigma_dec = self.sigma_dec_mol
                self.delta = max(smag_int * sigma_dec, self.delta_min)
                if smag_cell > 0:
                    self.delta_cell = max(self.delta_cell * self.sigma_dec,
                                          self.delta_min)
            elif self.ord == 0 and rho > 1./self.rho_inc:
                # Minimisation: growth-only policy -- any step that realised
                # >= 75 % of the predicted drop (or more than predicted)
                # earns a larger radius, capped per system type.
                delta_max = (self.delta_max_tr if self._has_tr_internals()
                             else self.delta_max_mol)
                self.delta = min(max(self.sigma_inc_mol * smag_int, self.delta),
                                 delta_max)
                if smag_cell > 0:
                    self.delta_cell = max(self.sigma_inc * smag_cell,
                                          self.delta_cell)
            elif 1./self.rho_inc < rho < self.rho_inc:
                self.delta = max(self.sigma_inc * smag_int, self.delta)
                if smag_cell > 0:
                    self.delta_cell = max(self.sigma_inc * smag_cell,
                                          self.delta_cell)
            self.rho = rho
        else:
            self.rho = 1.

        # Apply Niggli reduction if cell becomes too skewed
        if self.optimize_cell and self.niggli and self.pes.maybe_niggli_reduce():
            logger.info("Applied Niggli reduction to reduce cell skewness")
            self.initialized = False
            self.rho = 1.

    def gradient_converged(self, gradient=None):
        return self.converged()

    def converged(self, forces=None):
        # fmax may still be None if converged() is called before run()
        fmax = self.fmax if self.fmax is not None else 0.05  # Default threshold
        if self.optimize_cell:
            smax = self.smax if self.smax is not None else fmax
            result = self.pes.converged(fmax, smax=smax)
            self._last_converged = result
            return result[0]
        result = self.pes.converged(fmax)
        self._last_converged = result
        return result[0]

    def log(self, forces=None):
        if self.logfile is None:
            return
        if self.optimize_cell:
            smax = self.smax if self.smax is not None else self.fmax
            result = self._last_converged
            if result is None or len(result) != 4:
                result = self.pes.converged(self.fmax, smax=smax)
            _, fmax, cmax, smax_actual = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s} {:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "smax", "cmax", "rtrust",
                                           "strust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, smax_actual,
                                       cmax, self.delta, self.delta_cell,
                                       self.rho))
        else:
            result = self._last_converged
            if result is None or len(result) != 3:
                result = self.pes.converged(self.fmax)
            _, fmax, cmax = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "cmax", "rtrust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, cmax,
                                       self.delta, self.rho))
        try:
            self.logfile.flush()
        except (AttributeError, TypeError):
            pass


from scipy.optimize import minimize as _scipy_minimize

_ANGSTROM_TO_NM = 0.1

# MUST equal utils.EV_TO_KJ bit-for-bit. utils builds it as
# HARTREE_TO_KJ * (1.0 / 27.211386245988); molecules/sella_wrapper.py imports it
# from there, and this file (which cannot import utils -- it is cloudpickled to
# the workers by value) has to reproduce the same double.
#
# It previously held the literal 96.4853321233100184, which is 201 ULP away
# (relative 3e-14). Since every energy and force handed to Sella is divided by
# this constant, that offset perturbed the entire trajectory: Sella's
# trust-region step is chaotic at the ULP level, so the two implementations
# diverged on 2 of 250 train molecules despite being the same algorithm.
_EV_TO_KJ = 2625.4996394799 * (1.0 / 27.211386245988)

class _WrappedCalc(Calculator):
    """Adapts the harness calc(pos_nm) -> (E_kJ/mol, F_kJ/mol/nm) to an ASE calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, calc_func, **kwargs):
        super().__init__(**kwargs)
        self.calc_func = calc_func
        self.call_count = 0
        self.last_positions_nm = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = ["energy", "forces"]
        super().calculate(atoms, properties, system_changes)
        pos_nm = self.atoms.get_positions() * _ANGSTROM_TO_NM
        energy_kj, forces_kj_nm = self.calc_func(pos_nm)
        self.call_count += 1
        self.last_positions_nm = pos_nm.copy()
        self.results["energy"] = energy_kj / _EV_TO_KJ
        self.results["forces"] = np.array(forces_kj_nm) / _EV_TO_KJ * _ANGSTROM_TO_NM

# Symmetry-breaking start displacement for multi-fragment systems: rigid-body
# translation (A) and rotation (rad) applied to every fragment of the input
# geometry before the first force call.
_SYMMETRY_BREAK_TRANS = 0.02
_SYMMETRY_BREAK_ROT = 0.02
_SYMMETRY_BREAK_SEED = 0


def _start_fragments(numbers, pos_ang, scale=1.25):
    """Fragments of the input geometry under the bond criterion of Sella's
    first bond pass (Internals.find_all_bonds: d_ij <= scale * (r_cov,i +
    r_cov,j)), as index arrays; a lone atom forms a one-atom fragment."""
    natoms = len(numbers)
    rcov = covalent_radii[np.asarray(numbers)]
    d = np.linalg.norm(pos_ang[:, None, :] - pos_ang[None, :, :], axis=2)
    bonded = d <= scale * (rcov[:, None] + rcov[None, :])
    np.fill_diagonal(bonded, False)
    label = -np.ones(natoms, dtype=int)
    groups = []
    for i in range(natoms):
        if label[i] >= 0:
            continue
        label[i] = len(groups)
        stack = [i]
        members = [i]
        while stack:
            k = stack.pop()
            for j in np.flatnonzero(bonded[k]):
                if label[j] < 0:
                    label[j] = label[i]
                    stack.append(j)
                    members.append(j)
        groups.append(np.array(sorted(members), dtype=int))
    return groups


def _break_start_symmetry(numbers, pos_ang):
    """Displace every fragment of a multi-fragment start as a rigid body by
    _SYMMETRY_BREAK_TRANS A along a random direction and (fragments of two
    or more atoms) _SYMMETRY_BREAK_ROT rad about a random axis through its
    centroid; directions come from a fixed-seed generator, so the
    displacement is a deterministic function of the fragment count only.

    Rationale: the input geometries of non-covalent complexes frequently
    lie on a symmetry element (an ion on the bisector of two equivalent
    donors, a molecule in the mirror plane of its partner). Along a
    symmetry-breaking mode the gradient is then identically zero, every
    quasi-Newton step preserves the symmetry, and the optimizer converges
    onto the symmetric stationary point even when it is a saddle: the
    force along a soft antisymmetric mode, |k|*delta, stays below the
    convergence threshold for any displacement delta the run reaches, so
    the saddle passes every force and displacement test. A small
    displacement off the symmetry element is the standard remedy for
    relaxations from high-symmetry starts (cf. ASE Atoms.rattle); it gives
    the unstable mode a finite seed that the descent amplifies by
    ~(1 + |k|/lambda_model) per step, while a stable mode absorbs it in one
    or two steps at negligible energy cost (~0.5*k*delta^2 < 1e-3 eV).
    Connected systems are returned unchanged (one fragment: their internal
    coordinates carry no such rigid-body symmetry element)."""
    groups = _start_fragments(numbers, pos_ang)
    if len(groups) < 2:
        return pos_ang
    rng = np.random.RandomState(_SYMMETRY_BREAK_SEED)
    pos = np.array(pos_ang, dtype=float, copy=True)
    for group in groups:
        t = rng.normal(size=3)
        t *= _SYMMETRY_BREAK_TRANS / np.linalg.norm(t)
        if len(group) >= 2:
            w = rng.normal(size=3)
            w *= _SYMMETRY_BREAK_ROT / np.linalg.norm(w)
            skew = np.array([[0., -w[2], w[1]],
                             [w[2], 0., -w[0]],
                             [-w[1], w[0], 0.]])
            rot = expm(skew)
            centroid = pos[group].mean(axis=0)
            pos[group] = (pos[group] - centroid) @ rot.T + centroid
        pos[group] += t
    return pos


# Rigid-body pre-relaxation ("docking") of neutral multi-fragment starts on a
# classical intermolecular surrogate.
#
# Rationale: when the start of a non-covalent complex lies far from its
# minimum (fragments an angstrom or two too far apart, or mis-oriented), the
# quasi-Newton walk covers ~0.1 A per force call -- the fragment modes are
# soft, their model curvature is a floor, and the trust radius caps the
# fragment step at 0.25 A -- so ten or more force calls are spent on the
# approach before the endgame even begins. A point-charge + 12-6 potential
# knows the shape of that approach without any force call: it is relaxed in
# the rigid-body coordinates of every fragment (intramolecular geometry
# frozen), and the quasi-Newton optimisation starts from the relaxed pose,
# which then only has to be corrected by the surrogate's error. The
# surrogate is trusted only where the first force call confirms it: the
# rigid-body projections of its force field and of the true one must point
# the same way (cosine >= _DOCK_MIN_COS), the relaxation must move the pose
# by at least _DOCK_MIN_RMSD (a shorter approach is cheaper for the
# quasi-Newton steps than the extra force call the docked pose costs), and
# the docked pose must lower the true energy, otherwise the run continues
# from the original start (its evaluation is cached, no force call is
# repeated). Ionic complexes are not docked: a non-polarisable point-charge
# model is unreliable for ions (charge transfer, polarisation), so a start
# with a fragment carrying a net formal charge (valence rules on the bond
# graph) or an odd electron count keeps the plain quasi-Newton path.
_DOCK_MIN_COS = 0.5
_DOCK_MIN_RMSD = 0.5      # A, all-atom rmsd between the start and the docked pose
_DOCK_MAX_MOVE = 6.0      # A, largest atomic displacement accepted (runaway guard)
_DOCK_CHARGE_PER_EN = 0.22  # e per unit Pauling electronegativity difference per bond
_DOCK_BOND_ORDER_LENGTH = 0.20  # A of bond shortening per extra bond order
_DOCK_COULOMB = 332.0637  # kcal/mol A e^-2
_DOCK_POLAR_H_EN = 2.57   # H bonded to an atom at least this electronegative (N, O, S,
                          # halogens; not C at 2.55) is a hydrogen-bond donor: no 12-6 term

# Pauling electronegativities.
_PAULING_EN = {1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
               9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19,
               16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 31: 1.81, 32: 2.01,
               33: 2.18, 34: 2.55, 35: 2.96, 37: 0.82, 38: 0.95, 49: 1.78,
               50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 55: 0.79, 56: 0.89,
               81: 1.62, 82: 2.33, 83: 2.02}
# UFF 12-6 parameters (Rappe et al. 1992): minimum distance x_i (A) and well
# depth D_i (kcal/mol); combined geometrically.
_UFF_LJ = {1: (2.886, 0.044), 2: (2.362, 0.056), 3: (2.451, 0.025),
           4: (2.745, 0.085), 5: (4.083, 0.180), 6: (3.851, 0.105),
           7: (3.660, 0.069), 8: (3.500, 0.060), 9: (3.364, 0.050),
           10: (3.243, 0.042), 11: (2.983, 0.030), 12: (3.021, 0.111),
           13: (4.499, 0.505), 14: (4.295, 0.402), 15: (4.147, 0.305),
           16: (4.035, 0.274), 17: (3.947, 0.227), 18: (3.868, 0.185),
           19: (3.812, 0.035), 20: (3.399, 0.238), 31: (4.383, 0.415),
           32: (4.280, 0.379), 33: (4.230, 0.309), 34: (4.205, 0.291),
           35: (4.189, 0.251), 36: (4.141, 0.220), 37: (4.114, 0.040),
           38: (3.641, 0.235), 49: (4.463, 0.599), 50: (4.392, 0.567),
           51: (4.420, 0.449), 52: (4.470, 0.398), 53: (4.009, 0.339),
           54: (4.404, 0.332), 55: (4.517, 0.045), 56: (3.703, 0.364),
           81: (4.347, 0.680), 82: (4.297, 0.663), 83: (4.370, 0.518)}
_ALKALI = (3, 11, 19, 37, 55)
_ALKALINE_EARTH = (4, 12, 20, 38, 56)
_HALOGENS = (9, 17, 35, 53)


def _dock_formal_charges(numbers, adj, dist):
    """Formal charges from the bond graph by valence rules (no bond orders
    except a C-O / C-S length test): alkali +1, alkaline earth +2, lone
    halide -1, four-coordinate N / three-coordinate O +1, sulfonium and
    phosphonium +1, amidinium / guanidinium / imidazolium +1 on the central
    carbon, alkoxide / phenolate / thiolate / hydroxide -1, the terminal O of
    carboxylate, carbonate, nitrate, nitrite, sulfinate, sulfonate, sulfate,
    phosphonate, phosphate and oxo-halide anions share the group charge,
    N-oxides carry N+ O-, four-coordinate B / Al -1. Anything else is
    neutral. Returns an array (e) that sums to the (approximate) net charge."""
    numbers = np.asarray(numbers)
    natoms = len(numbers)
    q = np.zeros(natoms)
    deg = np.array([len(a) for a in adj])

    def terminal(i, z):
        return [j for j in adj[i] if numbers[j] == z and deg[j] == 1]

    for i in range(natoms):
        z = numbers[i]
        d = deg[i]
        if z in _ALKALI:
            q[i] += 1.0
        elif z in _ALKALINE_EARTH:
            q[i] += 2.0
        elif z in _HALOGENS and d == 0:
            q[i] -= 1.0
        elif z == 7 and d == 4:
            q[i] += 1.0
        elif z == 8 and d == 3:
            q[i] += 1.0
        elif z == 16 and d == 3 and all(numbers[j] in (1, 6) for j in adj[i]):
            q[i] += 1.0
        elif z == 15 and d == 4 and all(numbers[j] in (1, 6) for j in adj[i]):
            q[i] += 1.0
        elif z in (5, 13) and d == 4:
            q[i] -= 1.0
        elif z == 6 and d == 3:
            # amidinium / imidazolium (two three-coordinate amine N, third
            # neighbour C or H) and guanidinium (three); nitro N excluded
            amine = [j for j in adj[i] if numbers[j] == 7 and deg[j] == 3
                     and not terminal(j, 8)]
            others = [j for j in adj[i] if j not in amine]
            if len(amine) == 3 or (len(amine) == 2 and numbers[others[0]] in (1, 6)):
                q[i] += 1.0
        elif z in (8, 16) and d == 1:
            x = adj[i][0]
            zx = numbers[x]
            dx = deg[x]
            k = len(terminal(x, z))
            if zx == 1:
                q[i] -= 1.0                       # hydroxide, hydrosulfide
            elif zx == 6:
                if k >= 2:
                    q[i] -= (k - 1.0) / k          # carboxylate, carbonate
                elif dx == 4:
                    q[i] -= 1.0                   # alkoxide, thiolate
                elif dx == 3:
                    # C=O 1.20-1.25 A and C=S 1.60-1.68 A even in a
                    # perturbed start; C-O(-) 1.26-1.30, C-S(-) 1.72-1.76
                    single = 1.30 if z == 8 else 1.75
                    if dist[i, x] > single:
                        q[i] -= 1.0               # phenolate, enolate, thiophenolate
            elif zx == 7:
                if dx == 3 and k == 3:
                    q[i] -= 2.0 / 3.0             # nitrate (N+ below)
                    q[x] += 1.0 / 3.0
                elif dx == 3 and k == 2:
                    q[i] -= 0.5                   # nitro: N+ O- O=
                    q[x] += 0.5
                elif dx == 3 and k == 1:
                    q[i] -= 1.0                   # amine / pyridine N-oxide
                    q[x] += 1.0
                elif dx == 4 and k == 1:
                    q[i] -= 1.0                   # N-oxide of a four-coordinate N (+1 above)
                elif dx == 2 and k == 2:
                    q[i] -= 0.5                   # nitrite
            elif zx == 16:
                if dx == 4 and k >= 3:
                    q[i] -= (k - 2.0) / k          # sulfonate, sulfate
                elif dx == 3 and k >= 2:
                    q[i] -= (k - 1.0) / k          # sulfinate, sulfite
            elif zx == 15:
                if k >= 2:
                    q[i] -= (k - 1.0) / k          # phosphonate, phosphate
            elif zx in _HALOGENS:
                q[i] -= 1.0 / k                   # hypochlorite ... perchlorate
    return q


def _dock_surrogate_terms(numbers, pos, groups):
    """Inter-fragment pair terms of the surrogate: indices, charge products
    and 12-6 parameters. Returns None when a fragment carries a net formal
    charge or an odd electron count (open shell or unrecognised ion)."""
    numbers = np.asarray(numbers)
    natoms = len(numbers)
    rcov = covalent_radii[numbers]
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    bonded = dist <= 1.25 * (rcov[:, None] + rcov[None, :])
    np.fill_diagonal(bonded, False)
    adj = [list(np.flatnonzero(bonded[i])) for i in range(natoms)]
    formal = _dock_formal_charges(numbers, adj, dist)
    for group in groups:
        net = formal[group].sum()
        if abs(net) > 1e-6 or (int(numbers[group].sum()) - int(round(net))) % 2:
            return None
    # Partial charges: formal charge plus a bond-polarity transfer of
    # _DOCK_CHARGE_PER_EN e per unit electronegativity difference and bond
    # order (order 1 + shortening below the covalent-radius sum in units of
    # _DOCK_BOND_ORDER_LENGTH, at most 3).
    chi = np.array([_PAULING_EN.get(int(z), 2.0) for z in numbers])
    q = formal.copy()
    for i in range(natoms):
        for j in adj[i]:
            order = 1.0 + min(2.0, max(0.0, (rcov[i] + rcov[j] - dist[i, j]) / _DOCK_BOND_ORDER_LENGTH))
            q[i] += _DOCK_CHARGE_PER_EN * order * (chi[j] - chi[i])
    lj_x = np.array([_UFF_LJ.get(int(z), (4.0, 0.2))[0] for z in numbers])
    lj_d = np.array([_UFF_LJ.get(int(z), (4.0, 0.2))[1] for z in numbers])
    for i in range(natoms):
        # a polar hydrogen has no 12-6 term (hydrogen bonds close on the
        # heavy-atom repulsion, as in the point-charge water models)
        if numbers[i] == 1 and adj[i] and chi[adj[i][0]] >= _DOCK_POLAR_H_EN:
            lj_d[i] = 0.0
    label = np.empty(natoms, dtype=int)
    for k, group in enumerate(groups):
        label[group] = k
    ii, jj = np.triu_indices(natoms, 1)
    keep = label[ii] != label[jj]
    ii = ii[keep]
    jj = jj[keep]
    return (ii, jj, _DOCK_COULOMB * q[ii] * q[jj], np.sqrt(lj_d[ii] * lj_d[jj]),
            np.sqrt(lj_x[ii] * lj_x[jj]))


def _dock_energy_gradient(pos, terms):
    """Surrogate energy (kcal/mol) and its Cartesian gradient (kcal/mol/A)."""
    ii, jj, qq, dd, xx = terms
    vec = pos[ii] - pos[jj]
    r = np.maximum(np.linalg.norm(vec, axis=1), 0.1)
    s6 = (xx / r) ** 6
    energy = np.sum(qq / r + dd * (s6 * s6 - 2.0 * s6))
    dedr = -qq / r ** 2 - 12.0 * dd * (s6 * s6 - s6) / r
    fvec = (dedr / r)[:, None] * vec
    grad = np.zeros_like(pos)
    np.add.at(grad, ii, fvec)
    np.add.at(grad, jj, -fvec)
    return energy, grad


def _dock_rigid_field(pos, groups, field):
    """Least-squares projection of a per-atom vector field onto the rigid-body
    motions (unit-weight translation + rotation) of every fragment."""
    out = np.zeros_like(field)
    for group in groups:
        r = pos[group] - pos[group].mean(axis=0)
        f = field[group]
        inertia = np.sum(r * r) * np.eye(3) - r.T @ r
        omega = np.linalg.lstsq(inertia, np.cross(r, f).sum(axis=0), rcond=None)[0]
        out[group] = f.sum(axis=0) / len(group) + np.cross(omega, r)
    return out


def _dock_rotation(omega):
    """Rotation matrix exp([omega]x) and the right Jacobian J_r of SO(3),
    d(exp([omega + delta]x)) = exp([omega]x) exp([J_r delta]x) + O(delta^2)."""
    theta = np.linalg.norm(omega)
    K = np.array([[0.0, -omega[2], omega[1]],
                  [omega[2], 0.0, -omega[0]],
                  [-omega[1], omega[0], 0.0]])
    if theta < 1e-6:
        a, b, c = 1.0 - theta ** 2 / 6.0, 0.5 - theta ** 2 / 24.0, 1.0 / 6.0 - theta ** 2 / 120.0
    else:
        a = np.sin(theta) / theta
        b = (1.0 - np.cos(theta)) / theta ** 2
        c = (theta - np.sin(theta)) / theta ** 3
    K2 = K @ K
    return np.eye(3) + a * K + b * K2, np.eye(3) - b * K + c * K2


def _dock_relax(pos0, groups, terms):
    """Minimise the surrogate over the rigid-body coordinates of the fragments
    (translation t and rotation vector omega about the start centroid, the
    latter scaled by the radius of gyration so that all parameters are
    lengths). Returns the relaxed positions."""
    cents = [pos0[g].mean(axis=0) for g in groups]
    rgs = [np.sqrt(np.mean(np.sum((pos0[g] - c) ** 2, axis=1))) for g, c in zip(groups, cents)]

    def place(params):
        pos = np.array(pos0, dtype=float, copy=True)
        rots = []
        for k, (g, c, rg) in enumerate(zip(groups, cents, rgs)):
            t = params[6 * k:6 * k + 3]
            if rg > 1e-8:
                omega = params[6 * k + 3:6 * k + 6] / rg
                R, J = _dock_rotation(omega)
                pos[g] = (pos0[g] - c) @ R.T + c + t
            else:
                R, J = np.eye(3), np.eye(3)
                pos[g] = pos0[g] + t
            rots.append((R, J))
        return pos, rots

    def fun(params):
        pos, rots = place(params)
        energy, grad = _dock_energy_gradient(pos, terms)
        dparams = np.zeros_like(params)
        for k, (g, c, rg) in enumerate(zip(groups, cents, rgs)):
            t = params[6 * k:6 * k + 3]
            dparams[6 * k:6 * k + 3] = grad[g].sum(axis=0)
            if rg > 1e-8:
                R, J = rots[k]
                torque = np.cross(pos[g] - c - t, grad[g]).sum(axis=0)
                dparams[6 * k + 3:6 * k + 6] = J.T @ (R.T @ torque) / rg
        return energy, dparams

    res = _scipy_minimize(fun, np.zeros(6 * len(groups)), jac=True, method='L-BFGS-B',
                          options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-5, 'ftol': 1e-12})
    return place(res.x)[0]


def _dock_start(atoms, wrapper):
    """Dock a neutral multi-fragment start (see the note above). Uses one
    force call for the gate and one for the docked pose; when the docked pose
    is rejected, the start evaluation is re-installed in the calculator cache
    so that the optimizer's first evaluation costs no call."""
    pos0 = atoms.get_positions()
    groups = _start_fragments(atoms.numbers, pos0)
    if len(groups) < 2:
        return
    terms = _dock_surrogate_terms(atoms.numbers, pos0, groups)
    if terms is None:
        return
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    _, g_sur = _dock_energy_gradient(pos0, terms)
    a = _dock_rigid_field(pos0, groups, f0).ravel()
    b = _dock_rigid_field(pos0, groups, -g_sur).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0.0 or nb <= 0.0 or a @ b < _DOCK_MIN_COS * na * nb:
        return
    pos1 = _dock_relax(pos0, groups, terms)
    move = pos1 - pos0
    if not np.all(np.isfinite(pos1)):
        return
    if np.max(np.linalg.norm(move, axis=1)) > _DOCK_MAX_MOVE:
        return
    if np.sqrt(np.mean(np.sum(move ** 2, axis=1))) < _DOCK_MIN_RMSD:
        return
    atoms.positions = pos1
    e1 = atoms.get_potential_energy()
    if e1 < e0:
        return
    atoms.positions = pos0
    wrapper.atoms = atoms.copy()
    wrapper.results = {"energy": e0, "forces": f0}


def minimize_func(positions, atomic_numbers, calc, max_force_calls, converged):
    pos_ang = np.array(positions) / _ANGSTROM_TO_NM
    # Multi-fragment starts leave their symmetry element before the first
    # force call (see _break_start_symmetry); connected systems unchanged.
    pos_ang = _break_start_symmetry(atomic_numbers, pos_ang)
    atoms = Atoms(numbers=atomic_numbers, positions=pos_ang)
    wrapper = _WrappedCalc(calc)
    atoms.calc = wrapper
    # Neutral multi-fragment starts are docked on the classical surrogate
    # first (see _dock_start); connected systems and ionic complexes are
    # untouched and make no force call here.
    _dock_start(atoms, wrapper)
    # allow_fragments=True: disconnected fragments (e.g. non-covalent dimers)
    # get explicit centroid translation + rotation internals (TRIC-style,
    # Wang & Song 2016) instead of being stitched together by long
    # inter-fragment pseudo-bonds found by inflating the covalent radii. Their
    # curvature comes from the geometry-tracked inter-fragment contact term
    # (Internals._h0_nonlocal_contacts) on top of a diagonal floor
    # (Internals._h0_fragment).
    opt = Sella(atoms, internal=True, order=0, logfile=None,
                allow_fragments=True)
    # The optimizer's first evaluation is served from the calculator cache
    # when the start was already costed by _dock_start, so the step budget is
    # reduced by the calls made so far beyond that one.
    for _ in opt.irun(fmax=0, steps=max_force_calls - max(1, wrapper.call_count)):
        if converged():
            break
    # Return the last geometry that was actually EVALUATED, not whatever the
    # Atoms object happens to hold. distributed_validate/worker.py rejects a run
    # whose returned geometry is not the last evaluated one
    # (returned_geometry_mismatch:last_evaluated_geometry_required), and after
    # irun() exits the Atoms may sit on a proposed step that was never costed.
    # These coincide today, so reading atoms.get_positions() passes -- but only
    # by luck, and any edit to the stepping loop can break that silently.
    final_pos_nm = wrapper.last_positions_nm
    if final_pos_nm is None:
        raise RuntimeError("optimizer made zero force calls")
    return (final_pos_nm, wrapper.call_count)



def entrypoint():
    """Harness contract: distributed_validate/optimizer.py loads this by name
    (`entrypoint_name="entrypoint"`, `use_entrypoint=True`)."""
    return minimize_func
