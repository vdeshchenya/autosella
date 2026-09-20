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

def update_H(B, S, Y, method='TS-BFGS', symm=2, lams=None, vecs=None,
             metric_diagonal=None):
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
        Bplus = _MS_TS_BFGS(B, S, Ytilde, lams, vecs, metric_diagonal)
        if S.shape[1] == 1:
            step = S[:, 0]
            residual = Ytilde[:, 0] - B @ step
            snorm = np.linalg.norm(step)
            enorm = np.linalg.norm(residual)
            if snorm > 1e-8 and enorm > 1e-12:
                unit_residual = residual / enorm
                cosine = float(np.clip(unit_residual @ (step / snorm), -1.0, 1.0))
                # Weighting SR1 by cosine squared cancels its singular
                # residual-dot-step denominator before evaluation.
                Bplus = ((1.0 - cosine**2) * Bplus
                         + cosine * (enorm / snorm)
                         * np.outer(unit_residual, unit_residual))
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

def _MS_TS_BFGS(B, S, Y, lams, vecs, metric_diagonal=None):
    J = Y - B @ S
    X1 = S.T @ Y @ Y.T
    absBS = vecs @ (np.abs(lams[:, np.newaxis]) * (vecs.T @ S))
    if (metric_diagonal is not None and lams.size and lams[0] < 0
            and metric_diagonal.shape == (B.shape[0],)
            and np.all(np.isfinite(metric_diagonal))
            and np.all(metric_diagonal > 0)):
        try:
            root = np.sqrt(metric_diagonal)
            reduced = B / root[:, None] / root[None, :]
            if np.all(np.isfinite(reduced)):
                values, vectors = eigh((reduced + reduced.T) / 2.0)
                mapped = root[:, None] * vectors
                physical = mapped @ (np.abs(values)[:, None] * (mapped.T @ S))
                if np.all(np.isfinite(physical)):
                    absBS = physical
        except np.linalg.LinAlgError:
            pass
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
        self.set_B(update_H(
            B, dx, dg, method=self.update_method, symm=self.symm,
            lams=lams, vecs=vecs,
            metric_diagonal=getattr(self, 'curvature_metric_diagonal', None)))

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

def _transverse_bend(pos, component):
    left = pos[0] - pos[1]
    right = pos[2] - pos[1]
    axis = pos[2] - pos[0]
    axis = axis / jnp.linalg.norm(axis)
    first = jnp.cross(axis, pos[3] - pos[1])
    first = first / jnp.linalg.norm(first)
    second = jnp.cross(axis, first)
    direction = (1 - component) * first + component * second
    return (left / jnp.linalg.norm(left)
            + right / jnp.linalg.norm(right)) @ direction


class TransverseBend(Coordinate):
    """Rigid-motion-invariant bends in a molecular four-atom frame."""
    nindices = 4
    _eval0 = staticmethod(jit(_transverse_bend))
    _eval1 = staticmethod(_gradient(_transverse_bend))
    _eval2 = staticmethod(_hessian(_transverse_bend))

    def __init__(self, indices, component):
        Coordinate.__init__(self, indices)
        self.kwargs.update(component=component)

    def __eq__(self, other):
        if not isinstance(other, TransverseBend):
            return NotImplemented
        return (np.array_equal(self.indices, other.indices)
                and self.kwargs['component'] == other.kwargs['component'])


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

        direct_bends = (not self.atoms.pbc.any() and self.ndummies == 0
                        and self.cons.nint == 0)
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
                if (direct_bends and len(jbonds) == 2
                        and (linear[0][0] + linear[0][1]).calc(self.atoms)
                        >= np.pi - self.atol):
                    endpoints = sorted(int(b.indices[1]) for b in jbonds)
                    # An actual intrafragment atom makes both bend scalars
                    # invariant under global translation and rotation.
                    connected = {j}
                    frontier = [j]
                    while frontier:
                        current = frontier.pop()
                        for bond in bonds[current]:
                            neighbor = int(bond.indices[1])
                            if 0 <= neighbor < self.natoms and neighbor not in connected:
                                connected.add(neighbor)
                                frontier.append(neighbor)
                    options = sorted(connected - {j, *endpoints})
                    reference_atom = None
                    if options:
                        xyz = self.atoms.positions
                        axis = xyz[endpoints[1]] - xyz[endpoints[0]]
                        axis /= np.linalg.norm(axis)
                        vectors = xyz[options] - xyz[j]
                        lengths2 = np.sum(vectors * vectors, axis=1)
                        transverse2 = np.sum(np.cross(axis, vectors)**2, axis=1)
                        sine2 = np.divide(transverse2, lengths2,
                                          out=np.zeros_like(lengths2),
                                          where=lengths2 > 0.0)
                        best = int(np.argmax(sine2))
                        if sine2[best] > np.sin(self.atol)**2:
                            reference_atom = options[best]
                    if reference_atom is not None:
                        for component in range(2):
                            try:
                                self.add_other(TransverseBend(
                                    (endpoints[0], j, endpoints[1], reference_atom),
                                    component))
                            except DuplicateInternalError:
                                pass
                        continue
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
        # Preserve relative twist across a chain whose interior bends are
        # represented by transverse components rather than dummy atoms.
        linear_centers = {int(c.indices[1]) for c in self.internals['other']
                          if isinstance(c, TransverseBend)}
        if linear_centers:
            neighbors = [set() for _ in range(self.natoms)]
            for bond in self.internals['bonds']:
                a, b = map(int, bond.indices)
                if a < self.natoms and b < self.natoms:
                    neighbors[a].add(b)
                    neighbors[b].add(a)
            paths = set()
            for center in sorted(linear_centers):
                if len(neighbors[center]) != 2:
                    continue
                left, right = sorted(neighbors[center])
                path = [left, center, right]
                for reverse in (False, True):
                    if reverse:
                        path.reverse()
                    while path[-1] in linear_centers:
                        onward = neighbors[path[-1]] - set(path)
                        if len(onward) != 1:
                            break
                        path.append(next(iter(onward)))
                paths.add(min(tuple(path), tuple(reversed(path))))
            for path in sorted(paths):
                left, right = path[0], path[-1]
                for a in sorted(neighbors[left] - set(path)):
                    for d in sorted(neighbors[right] - set(path)):
                        if a == d:
                            continue
                        first = Angle((a, left, right)).calc(self.atoms)
                        second = Angle((left, right, d)).calc(self.atoms)
                        if not (self.atol < first < np.pi - self.atol
                                and self.atol < second < np.pi - self.atol):
                            continue
                        try:
                            self.add_dihedral((a, left, right, d))
                        except DuplicateInternalError:
                            pass
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
        rcov = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rij = bond.calc(self.all_atoms)
        if np.all(idx < self.natoms):
            # Wittbrodt--Schlegel period-pair Badger parameters, in Bohr.
            offsets = (
                (-0.2573, 0.3401, 0.6937, 0.7126, 0.8355, 0.9491),
                (0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190),
                (0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185),
                (0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206),
                (0.8355, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110),
                (0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110),
            )
            numbers = self.all_atoms.numbers[idx]
            periods = np.searchsorted((2, 10, 18, 36, 54), numbers)
            gap = rij / units.Bohr - offsets[periods[0]][periods[1]]
            if np.all(numbers > 0) and gap > 1e-8:
                return 1.734 / gap**3 * units.Hartree / units.Bohr**2
        h0 = Ab * np.exp(-Bb * (rij - rcov) / units.Bohr)
        return h0 * units.Hartree / units.Bohr**2

    def _h0_angle(
        self,
        angle: Angle,
        Aa: float = 0.089,
        Ba: float = 0.11,
        Ca: float = 0.44,
        Da: float = -0.42,
    ) -> float:
        indices = np.asarray(angle.indices, dtype=np.int32)
        if np.all(indices < self.natoms):
            numbers = self.all_atoms.numbers[indices]
            if np.all(numbers > 0):
                # Published Schlegel real-angle prior, Hartree/radian².
                return (0.160 if numbers[0] == 1 or numbers[2] == 1
                        else 0.250) * units.Hartree
        for dummy in indices[indices >= self.natoms]:
            model = getattr(self, '_linear_bend_priors', {}).get(int(dummy))
            if model is not None:
                center, adjacent, stiffness = model
                if (indices[1] == center
                        and set(indices) <= adjacent | {center, int(dummy)}):
                    return stiffness
        bab, bbc = angle.split()
        idxab = np.asarray(bab.indices, dtype=np.int32)
        idxbc = np.asarray(bbc.indices, dtype=np.int32)
        rcovab = covalent_radii[self.all_atoms.numbers[idxab]].sum()
        rcovbc = covalent_radii[self.all_atoms.numbers[idxbc]].sum()
        rab = bab.calc(self.all_atoms)
        rbc = bbc.calc(self.all_atoms)
        h0 = (
            Aa + Ba * np.exp(-Ca * (rab + rbc - rcovab - rcovbc) / units.Bohr)
            / (rcovab * rcovbc / units.Bohr**2)**Da
        )
        return h0 * units.Hartree

    def _h0_dihedral(
        self,
        dihedral: Dihedral,
        nbonds: np.ndarray,
        At: float = 0.0015,
        Bt: float = 14.0,
        Ct: float = 2.85,
        Dt: float = 0.57,
        Et: float = 4.00,
    ) -> float:
        _, bbc = dihedral.split()[0].split()
        idx = np.asarray(bbc.indices, dtype=np.int32)
        rcovbc = covalent_radii[self.all_atoms.numbers[idx]].sum()
        rbc = bbc.calc(self.all_atoms)
        indices = np.asarray(dihedral.indices, dtype=np.int32)
        if (np.all(indices < self.natoms)
                and np.all(self.all_atoms.numbers[indices] > 0)):
            # OptKing SCHLEGEL branch, using distances in Bohr.
            radius, distance = rcovbc / units.Bohr, rbc / units.Bohr
            slope = 0.0 if distance > radius + 0.0023 / 0.07 else 0.07
            return (0.0023 - slope * (distance - radius)) * units.Hartree
        L = nbonds[idx].sum() - 2
        h0 = (
            At + Bt * L**Dt * np.exp(-Ct * (rbc - rcovbc) / units.Bohr)
            / (rbc * rcovbc / units.Bohr**2)**Et
        )
        return h0 * units.Hartree

    def guess_hessian(self, h0cart=70.) -> np.ndarray:
        self._linear_bend_priors = {}
        self._linear_bend_torsions = set()
        if not self.atoms.pbc.any():
            neighbors = [set() for _ in range(self.natoms)]
            for bond in self.internals['bonds']:
                first, second = bond.indices
                if first < self.natoms and second < self.natoms:
                    neighbors[first].add(second)
                    neighbors[second].add(first)
            for center, dummy in enumerate(self.dinds):
                adjacent = neighbors[center]
                if dummy < self.natoms or len(adjacent) != 2:
                    continue
                outer = sorted(adjacent)
                edges = self.atoms.positions[outer] - self.atoms.positions[center]
                lengths = np.linalg.norm(edges, axis=1)
                if np.any(lengths < 1e-8) or edges[0] @ edges[1] >= 0:
                    continue
                numbers = self.atoms.numbers[outer]
                stiffness = (0.160 if 1 in numbers else 0.250) * units.Hartree
                self._linear_bend_priors[int(dummy)] = (center, adjacent, stiffness)
        nbonds = np.zeros(len(self.all_atoms), dtype=np.int32)
        h0 = np.zeros(self.nint, dtype=np.float64)
        h0_tr = 0.005 * units.Hartree
        idx = 0
        for trans in self.internals['translations']:
            h0[idx] = h0_tr if self.allow_fragments else h0cart
            idx += 1
        for bond in self.internals['bonds']:
            h0[idx] = self._h0_bond(bond)
            idx += 1
            # count number of bonds per atom for dihedral later
            i, j = bond.indices
            nbonds[i] += 1
            nbonds[j] += 1
        for angle in self.internals['angles']:
            h0[idx] = self._h0_angle(angle)
            idx += 1
        dummy_set = set(range(self.natoms, self.natoms + self.ndummies))
        for dihedral in self.internals['dihedrals']:
            if any(j in dummy_set for j in dihedral.indices):
                h0[idx] = 0.5 * units.Hartree
                for dummy in dummy_set.intersection(dihedral.indices):
                    model = self._linear_bend_priors.get(dummy)
                    if model is not None:
                        center, adjacent, stiffness = model
                        if (set(dihedral.indices[1:3]) == {center, dummy}
                                and {dihedral.indices[0], dihedral.indices[3]} == adjacent):
                            h0[idx] = stiffness
                            self._linear_bend_torsions.add(tuple(dihedral.indices))
            else:
                h0[idx] = self._h0_dihedral(dihedral, nbonds)
            idx += 1
        for coord in self.internals['other']:
            if isinstance(coord, TransverseBend):
                outer = self.atoms.numbers[coord.indices[[0, 2]]]
                h0[idx] = (0.160 if 1 in outer else 0.250) * units.Hartree
            else:
                h0[idx] = h0cart
            idx += 1
        for rot in self.internals['rotations']:
            h0[idx] = h0_tr if self.allow_fragments else h0cart
            idx += 1
        return np.diag(np.abs(h0))

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

    def _update_H(self, dx, dg):
        if self.last['x'] is None or self.last['g'] is None:
            return
        self.H.update(dx, dg)

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

class _BadgerBondHessians:
    def __init__(self, base, jacobian, first, second):
        self.base = base
        self.jacobian = jacobian
        self.first = first
        self.second = second
        self.shape = base.shape

    def ldot(self, weights):
        return (self.base.ldot(weights * self.first)
                + self.jacobian.T @ ((weights * self.second)[:, None]
                                     * self.jacobian))

    def asarray(self):
        return (self.first[:, None, None] * self.base.asarray()
                + self.second[:, None, None] * self.jacobian[:, :, None]
                * self.jacobian[:, None, :])


class _BadgerBondInternals:
    """Arc length of the accepted inverse-cubic bond stiffness metric."""
    def __init__(self, base):
        self.base = base
        offsets = (
            (-0.2573, 0.3401, 0.6937, 0.7126, 0.8355, 0.9491),
            (0.3401, 0.9652, 1.2843, 1.4725, 1.6549, 1.7190),
            (0.6937, 1.2843, 1.6925, 1.8238, 2.1164, 2.3185),
            (0.7126, 1.4725, 1.8238, 2.0203, 2.2137, 2.5206),
            (0.8355, 1.6549, 2.1164, 2.2137, 2.3718, 2.5110),
            (0.9491, 1.7190, 2.3185, 2.5206, 2.5110, 2.5110),
        )
        raw = base.calc()
        rows, origins = [], []
        row = base.ntrans
        if not base.atoms.pbc.any():
            for bond, active in zip(base.internals['bonds'], base._active['bonds']):
                if not active:
                    continue
                if all(i < base.natoms for i in bond.indices):
                    numbers = base.atoms.numbers[list(bond.indices)]
                    periods = np.searchsorted((2, 10, 18, 36, 54), numbers)
                    origin = offsets[periods[0]][periods[1]] * units.Bohr
                    if np.all(numbers > 0) and (raw[row] - origin) / units.Bohr > 1e-8:
                        rows.append(row)
                        origins.append(origin)
                row += 1
        self.rows = np.asarray(rows, dtype=int)
        self.origins = np.asarray(origins)
        self.reference = raw[self.rows].copy()
        self.gaps = self.reference - self.origins

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'base'), name)

    def _current_gaps(self, raw):
        gap = raw[self.rows] - self.origins
        if np.any(gap <= 0):
            raise ValueError('Bond leaves the positive Badger coordinate domain')
        return gap

    def _derivatives(self):
        raw = self.base.calc()
        gap = self._current_gaps(raw)
        first, second = np.ones_like(raw), np.zeros_like(raw)
        first[self.rows] = (self.gaps / gap)**1.5
        second[self.rows] = -1.5 * first[self.rows] / gap
        return first, second

    def calc(self):
        raw = self.base.calc().copy()
        gap = self._current_gaps(raw)
        # expm1 avoids cancellation for near-reference late optimization steps.
        raw[self.rows] = (self.reference - 2.0 * self.gaps
                          * np.expm1(-0.5 * np.log(gap / self.gaps)))
        return raw

    def jacobian(self):
        first, _ = self._derivatives()
        return first[:, None] * self.base.jacobian()

    def hessian_rdot(self, vector):
        first, second = self._derivatives()
        jacobian = self.base.jacobian()
        raw = self.base.hessian_rdot(vector)
        correction = (second * (jacobian @ vector))[:, None] * jacobian
        if sparse.issparse(raw):
            return (raw.multiply(first[:, None]).tocsr()
                    + sparse.csr_matrix(correction))
        return first[:, None] * raw + correction

    def hessian(self):
        first, second = self._derivatives()
        return _BadgerBondHessians(self.base.hessian(), self.base.jacobian(),
                                  first, second)

    def guess_hessian(self, h0cart=70.):
        first, _ = self._derivatives()
        return self.base.guess_hessian(h0cart) / np.outer(first, first)


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

        self.int = (_BadgerBondInternals(new_int) if H0 is None else new_int)
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        self._fit_blocks = None
        self._fit_pairs = []
        self._curvature_metric_diagonal = None
        if H0 is None:
            # Keep physical blocks and a bounded correlation for an early fit.
            B = self.int.jacobian()
            Q, _ = qr(B, mode='economic')
            P = Q @ Q.T
            diagonal = np.diag(self.int.guess_hessian())
            if np.all(np.isfinite(diagonal)) and np.all(diagonal > 0):
                self._curvature_metric_diagonal = diagonal.copy()
            labels = np.array(
                [3] * self.int.ntrans + [0] * self.int.nbonds
                + [1] * self.int.nangles + [2] * self.int.ndihedrals
                + [3] * self.int.nother + [3] * self.int.nrotations)
            row = self.int.ntrans + self.int.nbonds + self.int.nangles
            for torsion, active in zip(self.int.internals['dihedrals'],
                                       self.int._active['dihedrals']):
                if active:
                    if tuple(torsion.indices) in self.int._linear_bend_torsions:
                        labels[row] = 1
                    row += 1
            for coord, active in zip(self.int.internals['other'],
                                     self.int._active['other']):
                if active:
                    if isinstance(coord, TransverseBend):
                        labels[row] = 1
                    row += 1
            self._fit_blocks = [
                (P * (diagonal * (labels == kind))) @ P
                for kind in range(4) if np.any(labels == kind)
            ]
            pi_correction = self._conjugated_bond_curvature(P, diagonal)
            if pi_correction is not None:
                self._fit_blocks[0] += pi_correction
            auxiliary_start = len(self._fit_blocks)
            auxiliary = self._auxiliary_curvature(B)
            self._fit_blocks.extend(auxiliary)
            out_of_plane = self._out_of_plane_curvature(B)
            if out_of_plane is not None:
                self._fit_blocks.append(out_of_plane)
            H0 = sum(self._fit_blocks, np.zeros_like(P))
            coupling = self._stretch_bend_curvature(P, diagonal)
            if coupling is not None:
                self._fit_blocks.append(coupling)
            count = len(self._fit_blocks)
            ridge = np.full(count, 0.25)
            if auxiliary:
                ridge[auxiliary_start:auxiliary_start + len(auxiliary)] /= len(auxiliary)
            self._fit_gram = np.diag(ridge)
            self._fit_rhs = ridge.copy()
            self._fit_lower = np.full(count, 0.5)
            self._fit_upper = np.full(count, 2.0)
            if coupling is not None:
                self._fit_rhs[-1] = 0.0
                self._fit_lower[-1], self._fit_upper[-1] = -1.0, 1.0
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

        self._pinv_cache = _LRU2()
        self._qr_cache = _LRU2()
        self._Hc_cache = _LRU2()

    def _conjugated_bond_curvature(self, projector, diagonal):
        """HSSH reference bond correlations with Badger marginal stiffness."""
        if self.atoms.pbc.any():
            return None
        numbers = self.atoms.numbers
        count = len(numbers)
        neighbors = [set() for _ in range(count)]
        bond_rows = {}
        row = self.int.ntrans
        for bond, active in zip(self.int.internals['bonds'], self.int._active['bonds']):
            i, j = bond.indices
            if 0 <= i < count and 0 <= j < count:
                neighbors[i].add(j)
                neighbors[j].add(i)
                if active:
                    bond_rows[tuple(sorted((i, j)))] = row
            if active:
                row += 1
        # One neutral p electron per three-coordinate carbon; heteroatom
        # orbital energies/occupations are not represented by this model.
        pi_atoms = {i for i in range(count) if numbers[i] == 6
                    and len(neighbors[i]) == 3
                    and all(numbers[j] in (1, 6) for j in neighbors[i])}
        positions = self.atoms.positions
        orbital_axes = {}
        for i in sorted(pi_atoms):
            directions = positions[sorted(neighbors[i])] - positions[i]
            lengths = np.linalg.norm(directions, axis=1)
            if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
                continue
            directions = directions / lengths[:, None]
            axis = np.cross(directions[1] - directions[0],
                            directions[2] - directions[0])
            norm = np.linalg.norm(axis)
            if not np.isfinite(norm) or norm <= 64.0 * np.finfo(float).eps:
                continue
            orbital_axes[i] = axis / norm
        unseen = set(pi_atoms)
        correction = np.zeros_like(projector)
        used = False
        while unseen:
            component = []
            todo = [min(unseen)]
            unseen.remove(todo[0])
            while todo:
                i = todo.pop()
                component.append(i)
                for j in sorted(neighbors[i] & unseen):
                    unseen.remove(j)
                    todo.append(j)
            component.sort()
            n = len(component)
            if n < 2 or n % 2:
                continue
            index = {atom: k for k, atom in enumerate(component)}
            edges = [(i, j, bond_rows[(i, j)]) for i in component
                     for j in sorted(neighbors[i]) if i < j and j in index
                     and (i, j) in bond_rows]
            if not edges:
                continue
            if any(i not in orbital_axes for i in component):
                continue
            # Slater-Koster pp-pi projection; pp-sigma is outside this model.
            overlaps = []
            hamiltonian = np.zeros((n, n))
            for i, j, _ in edges:
                axis = positions[j] - positions[i]
                axis = axis / np.linalg.norm(axis)
                ni, nj = orbital_axes[i], orbital_axes[j]
                overlap = ni @ nj - (ni @ axis) * (nj @ axis)
                overlaps.append(overlap)
                hamiltonian[index[i], index[j]] = -overlap
                hamiltonian[index[j], index[i]] = -overlap
            energies, orbitals = eigh(hamiltonian)
            occupied = n // 2
            gaps = energies[occupied:][None, :] - energies[:occupied, None]
            tolerance = 64.0 * np.finfo(float).eps * n * max(1.0, np.max(np.abs(energies)))
            if np.min(gaps) <= tolerance:
                continue
            amplitudes = np.array([
                (np.outer(orbitals[index[i], :occupied], orbitals[index[j], occupied:])
                 + np.outer(orbitals[index[j], :occupied], orbitals[index[i], occupied:])
                 ).ravel() for i, j, _ in edges])
            # Freeze the orbital geometry in the radial response:
            # beta_b(R) = overlap_b * (-1 + dR / y).
            amplitudes *= np.asarray(overlaps)[:, None]
            weighted = amplitudes / np.sqrt(gaps.ravel())
            # Divide the electronic response plus harmonic sigma springs by
            # spring curvature 2/(x*y), x=.189 A and y=.2756 A.
            relative = np.eye(len(edges)) - (2.0 * 0.189 / 0.2756) * (weighted @ weighted.T)
            eigenvalues = eigh(relative, eigvals_only=True)
            if np.min(eigenvalues) <= 64.0 * np.finfo(float).eps * len(edges):
                continue
            marginal = np.sqrt(np.diag(relative))
            correlation = relative / marginal[:, None] / marginal[None, :]
            rows = np.array([r for _, _, r in edges], dtype=int)
            mapping = projector[:, rows] * np.sqrt(diagonal[rows])
            correction += mapping @ (correlation - np.eye(len(edges))) @ mapping.T
            used = True
        if not used:
            return None
        return 0.5 * (correction + correction.T)

    def _stretch_bend_curvature(self, projector, diagonal):
        """Bounded signed correlation on the real bond-angle incidence graph."""
        if self.atoms.pbc.any():
            return None
        natoms = len(self.atoms)
        bonds = {}
        for index, bond in enumerate(self.int.internals['bonds']):
            if all(i < natoms for i in bond.indices):
                bonds[tuple(sorted(bond.indices))] = self.int.ntrans + index
        adjacency = np.zeros_like(projector)
        angle_start = self.int.ntrans + self.int.nbonds
        for index, angle in enumerate(self.int.internals['angles']):
            i, j, k = angle.indices
            if max(i, j, k) >= natoms:
                continue
            angle_index = angle_start + index
            for pair in ((i, j), (j, k)):
                bond_index = bonds.get(tuple(sorted(pair)))
                if bond_index is not None:
                    adjacency[bond_index, angle_index] = 1.0
                    adjacency[angle_index, bond_index] = 1.0
        degree = np.sum(adjacency, axis=1)
        connected = degree > 0
        if not np.any(connected):
            return None
        normalization = np.zeros_like(degree)
        normalization[connected] = 1.0 / np.sqrt(degree[connected])
        adjacency *= normalization[:, None] * normalization[None, :]
        scaled = projector * np.sqrt(diagonal)
        block = 0.25 * scaled @ adjacency @ scaled.T
        block = 0.5 * (block + block.T)
        return block if np.all(np.isfinite(block)) else None

    def _auxiliary_curvature(self, jacobian):
        """Map across-angle and other nearby springs into separate fit blocks."""
        positions = self.atoms.positions
        radii = covalent_radii[self.atoms.numbers]
        first, second = np.triu_indices(len(positions), 1)
        differences = positions[first] - positions[second]
        distances = np.linalg.norm(differences, axis=1)
        references = radii[first] + radii[second]
        bonded = {tuple(sorted(bond.indices)) for bond in self.int.internals['bonds']}
        parent = list(range(len(positions)))
        neighbors = [set() for _ in positions]

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for i, j in bonded:
            if i < len(positions) and j < len(positions):
                parent[root(i)] = root(j)
                neighbors[i].add(j)
                neighbors[j].add(i)
        labels = [root(i) for i in range(len(positions))]
        keep = np.array([(int(i), int(j)) not in bonded and labels[i] == labels[j]
                         for i, j in zip(first, second)], dtype=bool)
        keep &= (distances > 1e-8) & (distances < 2.5 * references)
        first, second = first[keep], second[keep]
        if first.size == 0:
            return []
        across_angle = np.array([bool(neighbors[i] & neighbors[j])
                                 for i, j in zip(first, second)], dtype=bool)
        distances, references = distances[keep], references[keep]
        directions = differences[keep] / distances[:, None]
        pair_jacobian = np.zeros((len(first), jacobian.shape[1]))
        rows = np.arange(len(first))[:, None]
        axes = np.arange(3)
        pair_jacobian[rows, 3 * first[:, None] + axes] = directions
        pair_jacobian[rows, 3 * second[:, None] + axes] = -directions
        stiffness = (0.3601 * np.exp(-1.944 * (distances - references) / units.Bohr)
                     * units.Hartree / units.Bohr**2)
        mapped = (np.sqrt(stiffness)[:, None] * pair_jacobian
                  ) @ np.linalg.pinv(jacobian, rcond=1e-6)
        return [mapped[mask].T @ mapped[mask]
                for mask in (across_angle, ~across_angle) if np.any(mask)]

    dpos = property(lambda self: self.dummies.positions.copy())

    def _out_of_plane_curvature(self, jacobian):
        """Permutation-averaged Fischer out-of-plane angular curvature."""
        if self.atoms.pbc.any():
            return None
        from itertools import permutations
        positions = self.atoms.positions
        radii = covalent_radii[self.atoms.numbers]
        neighbors = [set() for _ in positions]
        for bond in self.int.internals['bonds']:
            i, j = bond.indices
            if i < len(positions) and j < len(positions):
                neighbors[i].add(j)
                neighbors[j].add(i)
        rows, stiffness = [], []
        cosine_limit = np.cos(self.int.atol)
        for center, adjacent in enumerate(neighbors):
            if len(adjacent) != 3:
                continue
            local_rows, local_stiffness = [], []
            for first, second, third in permutations(sorted(adjacent)):
                outer = [first, second, third]
                edges = positions[outer] - positions[center]
                lengths = np.linalg.norm(edges, axis=1)
                if np.any(lengths < 1e-8):
                    continue
                u, v, w = edges / lengths[:, None]
                plane_cosine = float(v @ w)
                if abs(plane_cosine) >= cosine_limit:
                    continue
                normal = np.cross(v, w)
                plane_sine = np.linalg.norm(normal)
                sine = float(u @ normal) / plane_sine
                cosine_squared = 1.0 - sine * sine
                if cosine_squared <= 1e-12:
                    continue
                cosine = np.sqrt(cosine_squared)
                gradients = np.array([
                    normal / plane_sine - sine * u,
                    np.cross(w, u) / plane_sine
                    - sine * (v - plane_cosine * w) / plane_sine**2,
                    np.cross(u, v) / plane_sine
                    - sine * (w - plane_cosine * v) / plane_sine**2,
                ]) / (lengths[:, None] * cosine)
                row = np.zeros(jacobian.shape[1])
                for atom, gradient in zip(outer, gradients):
                    row[3 * atom:3 * atom + 3] = gradient
                row[3 * center:3 * center + 3] = -gradients.sum(axis=0)
                reference = (radii[center] + radii[outer]) / units.Bohr
                value = (0.0025 + 0.0061 * (reference[1] * reference[2])**0.80
                         * cosine_squared**2
                         * np.exp(-3.0 * (lengths[0] / units.Bohr - reference[0])))
                local_rows.append(row)
                local_stiffness.append(value * units.Hartree)
            if local_rows:
                rows.extend(local_rows)
                stiffness.extend(np.asarray(local_stiffness) / len(local_rows))
        if not rows:
            return None
        mapped = ((np.sqrt(stiffness)[:, None] * np.asarray(rows))
                  @ np.linalg.pinv(jacobian, rcond=1e-6))
        block = mapped.T @ mapped
        return block if np.all(np.isfinite(block)) else None

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
        """Backtrack realized nonbonded collisions without requesting forces."""
        self._collision_step_fraction = 1.0
        if self.atoms.pbc.any():
            return self._set_x_unchecked(target)
        positions = self.atoms.positions.copy()
        dummy_positions = self.dummies.positions.copy()
        q0 = self.get_x().copy()
        displacement = self.wrap_dx(target - q0)
        first, second = np.triu_indices(len(positions), 1)
        bonded = {tuple(sorted(bond.indices))
                  for bond in self.int.internals['bonds']}
        keep = np.array([(int(i), int(j)) not in bonded
                         for i, j in zip(first, second)], dtype=bool)
        first, second = first[keep], second[keep]
        radii = covalent_radii[self.atoms.numbers]
        distances = np.linalg.norm(positions[first] - positions[second], axis=1)
        # Match the existing topology detector, allowing gradual new contacts.
        separated = distances > 1.25 * (radii[first] + radii[second])
        first, second = first[separated], second[separated]
        minimum = radii[first] + radii[second]
        curr, last, bad_int = self.curr.copy(), self.last.copy(), self.bad_int
        for attempt in range(25):
            if attempt:
                self.atoms.positions = positions.copy()
                self.dummies.positions = dummy_positions.copy()
                self.curr, self.last = curr.copy(), last.copy()
                self.bad_int = bad_int
            trial_target = target if attempt == 0 else q0 + (0.5**attempt) * displacement
            result = self._set_x_unchecked(trial_target)
            candidate = self.atoms.positions
            distances = np.linalg.norm(candidate[first] - candidate[second], axis=1)
            if np.all(distances >= minimum):
                self._collision_step_fraction = 0.5**attempt
                return result
        self.atoms.positions = positions.copy()
        self.dummies.positions = dummy_positions.copy()
        self.curr, self.last, self.bad_int = curr, last, bad_int
        raise RuntimeError("No collision-free internal step found")

    def _set_x_unchecked(self, target):
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
        # Include the changing Badger row metric in the path connection.
        Binv = self._get_Binv()
        rhs = np.column_stack((dxdt, g))     # (ndof, 2)
        out = -Binv @ (D_rdot @ rhs)          # (ndof, 2)
        dydt[1] = out[:, 0]
        dydt[2] = out[:, 1]

        return dydt.ravel()

    def _update_H(self, dx, dg):
        if self.last['x'] is None or self.last['g'] is None:
            return
        if (self._fit_blocks is not None and len(self._fit_pairs) < 6
                and np.linalg.norm(dx) > 1e-8 and np.linalg.norm(dg) > 1e-10):
            # Equalize secant weights so the first large bond correction
            # does not dominate later angle and collective-mode information.
            scale = np.linalg.norm(dg)
            design = np.column_stack([block @ dx for block in self._fit_blocks]) / scale
            target = dg / scale
            self._fit_gram += design.T @ design
            self._fit_rhs += design.T @ target
            weights = np.clip(solve(self._fit_gram, self._fit_rhs),
                              self._fit_lower, self._fit_upper)
            self._fit_pairs.append((dx.copy(), dg.copy()))
            model = sum((weight * block for weight, block
                         in zip(weights, self._fit_blocks)),
                        np.zeros_like(self.H.B))
            # Replay all observed secants after refitting the starting model,
            # preserving learned off-diagonal information exactly as updates.
            for old_dx, old_dg in self._fit_pairs:
                model = update_H(
                    model, old_dx, old_dg, method=self.H.update_method,
                    symm=self.H.symm,
                    metric_diagonal=self._curvature_metric_diagonal)
            self.H.set_B(model)
            return
        self.H.curvature_metric_diagonal = self._curvature_metric_diagonal
        PES._update_H(self, dx, dg)

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        return ratio

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

    def set_curvature_metric(self, metric):
        # Leave the native positive-spectrum branch exactly unchanged.
        if self.order != 0 or not self.H.evals.size or self.H.evals[0] >= 0:
            return
        try:
            factor = np.linalg.cholesky((metric + metric.T) / 2.0)
            reduced = solve_triangular(factor, self.H.asarray(), lower=True)
            reduced = solve_triangular(factor, reduced.T, lower=True).T
            if not np.all(np.isfinite(reduced)):
                return
            values, vectors = eigh((reduced + reduced.T) / 2.0)
            mapped = factor @ vectors
            positive = (mapped * np.abs(values)) @ mapped.T
            if not np.all(np.isfinite(positive)):
                return
            values, vectors = eigh((positive + positive.T) / 2.0)
        except np.linalg.LinAlgError:
            return
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
            return
        self.L = np.abs(values)
        self.V = vectors
        self.Vg = self.V.T @ self.g

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

        metric_diagonal = getattr(self.pes, '_curvature_metric_diagonal', None)
        if (isinstance(self.stepper, QuasiNewton) and order == 0
                and self.stepper.H.evals.size and self.stepper.H.evals[0] < 0
                and self._W_is_identity and metric_diagonal is not None
                and metric_diagonal.size == self.P.shape[1]):
            metric = (self.P * metric_diagonal) @ self.P.T
            self.stepper.set_curvature_metric(metric)

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
        self, pes, *args, wx=1., wb=1., wa=1., wd=1., wo=1., wc=1., **kwargs
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
        self._weights_cache = None
        BaseRestrictedStep.__init__(self, pes, *args, **kwargs)

    def cons(self, s, dsda=None):
        w = self._get_weights()
        assert len(w) == len(s)

        sw = np.abs(s * w)
        idx = np.argmax(np.abs(sw))
        val = sw[idx]

        if dsda is None:
            return val
        return val, np.sign(s[idx]) * dsda[idx] * w[idx]

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
        row = (self.pes.int.ntrans + self.pes.int.nbonds
               + self.pes.int.nangles + self.pes.int.ndihedrals)
        for coord, active in zip(self.pes.int.internals['other'],
                                 self.pes.int._active['other']):
            if active:
                if isinstance(coord, TransverseBend):
                    w[row] = self.wa
                row += 1
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

# D3 reference data: s-dftd3, LGPL-3.0-or-later.
# https://github.com/dftd3/simple-dftd3/tree/41d5a07b98ce15e97bec7a1815869725f6c7b0c2
# Data redistributed without warranty under GNU LGPL v3 or later:
# https://www.gnu.org/licenses/lgpl-3.0.html
# Covalent radii: mctc-lib, Apache-2.0; Pyykko/Atsumi (2009), metal radii -10%.
# https://github.com/grimme-lab/mctc-lib/tree/5901f69c61834c8f034bf32383da0e0621071000
# https://www.apache.org/licenses/LICENSE-2.0
# C6 text retains the upstream Fortran order (first reference index fastest).
_D3_CN_REFERENCES = np.array([
    +0.9118, +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9865, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9808, +1.9697, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9706, +1.9441, +2.9128, +4.5856, -1.0000, -1.0000,
    +0.0000, +0.9868, +1.9985, +2.9987, +3.9844, -1.0000, -1.0000,
    +0.0000, +0.9944, +2.0143, +2.9903, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9925, +1.9887, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9982, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9684, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9628, +1.9496, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9648, +1.9311, +2.9146, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9507, +1.9435, +2.9407, +3.8677, -1.0000, -1.0000,
    +0.0000, +0.9947, +2.0102, +2.9859, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9948, +1.9903, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9972, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9767, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9831, +1.9349, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8627, +2.8999, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8299, +3.8675, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9138, +2.9110, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8269, 10.6191, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.6406, +9.8849, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.6483, +9.1376, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.7149, +2.9263, +7.7785, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.7937, +6.5458, +6.2918, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9576, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9419, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9601, +1.9315, +2.9233, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9434, +1.9447, +2.9186, +3.8972, -1.0000, -1.0000,
    +0.0000, +0.9889, +1.9793, +2.9709, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9901, +1.9812, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9974, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9738, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9801, +1.9143, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9153, +2.8903, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9355, +3.9106, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9545, +2.9225, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9420, 11.0556, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.6682, +9.5402, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8584, +8.8895, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9003, +2.9696, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8630, +5.7095, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9679, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9539, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9633, +1.9378, +2.9353, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9514, +1.9505, +2.9259, +3.9123, -1.0000, -1.0000,
    +0.0000, +0.9749, +1.9523, +2.9315, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9811, +1.9639, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9968, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9909, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9797, +1.8467, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9373, +2.9175, -1.0000, -1.0000, -1.0000, -1.0000,
    +2.7991, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9425, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9455, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9413, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9300, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8286, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.8732, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9086, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.8965, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9242, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9282, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9246, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.8482, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +2.9219, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9254, +3.8840, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9459, +2.8988, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9292, 10.9153, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8104, +9.8054, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8858, +9.1527, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.8648, +2.9424, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9188, +6.6669, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9846, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +1.9896, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9267, +1.9302, +2.9420, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9383, +1.9356, +2.9081, +3.9098, -1.0000, -1.0000,
    +0.0000, +0.9820, +1.9655, +2.9500, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9815, +1.9639, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9954, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9705, -1.0000, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9661, +1.9251, -1.0000, -1.0000, -1.0000, -1.0000,
    +0.0000, +0.9802, +1.9445, +2.9070, +3.8174, +4.6723, +5.5599,
    +0.0000, +0.9847, +1.9560, +2.9302, +3.8997, -1.0000, -1.0000,
    +0.0000, +0.9647, +1.9079, +2.9037, +3.8711, +4.9094, +4.5318,
    +0.0000, +0.9766, +2.8888, +3.9129, +4.1181, +5.9187, -1.0000,
    +0.0000, +0.9838, +1.9499, +2.9159, +3.9358, +4.9069, +5.9005,
    +0.0000, +0.9537, +1.9439, +2.9323, +3.9441, +4.9192, +5.8888,
    +0.0000, +0.9163, +1.8563, +2.8823, +4.8005, +5.7794, -1.0000,
    +0.0000, +0.9762, +1.9288, +2.8929, +3.8167, +4.7478, +5.6866,
    +0.0000, +0.9705, +1.9511, +2.9262, +3.9342, -1.0000, -1.0000,
    +0.0000, +0.9581, +1.9123, +2.9327, +3.9105, +5.8285, -1.0000,
    +0.0000, +0.9346, +1.8816, +2.9075, +3.8705, +4.8131, +5.7244,
    +0.0000, +0.9500, +1.9165, +2.9377, +3.8956, +4.8540, +5.8160,
    +0.0000, +0.9710, +1.9564, +2.9515, +3.9353, -1.0000, -1.0000,
    +0.0000, +0.9722, +1.9605, +2.9452, +3.9296, +4.2582, +4.5511,
    +0.0000, +0.9569, +1.9215, +2.8958, +3.7644, +4.6808, +5.5939,
]).reshape(103, 7)

_D3_COVALENT_RADII = np.array([
    0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71,
    0.63, 0.64, 0.67, 1.40, 1.25, 1.13, 1.04,
    1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33,
    1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99,
    1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14,
    1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24,
    1.15, 1.13, 1.13, 1.08, 1.15, 1.23, 1.28,
    1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
    1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51,
    1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
    1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11,
    1.12, 1.13, 1.32, 1.30, 1.30, 1.36, 1.31,
    1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52,
    1.53, 1.54, 1.55, 1.49, 1.49, 1.51, 1.51,
    1.48, 1.50, 1.56, 1.58, 1.45, 1.41, 1.34,
    1.29, 1.27, 1.21, 1.16, 1.15, 1.09, 1.22,
    1.36, 1.43, 1.46, 1.58, 1.48, 1.57,
])

_D3_R4_OVER_R2 = np.array([
    8.0589, 3.4698, 29.0974, 14.8517, 11.8799, 7.8715, 5.5588,
    4.7566, 3.8025, 3.1036, 26.1552, 17.2304, 17.7210, 12.7442,
    9.5361, 8.1652, 6.7463, 5.6004, 29.2012, 22.3934, 19.0598,
    16.8590, 15.4023, 12.5589, 13.4788, 12.2309, 11.2809, 10.5569,
    10.1428, 9.4907, 13.4606, 10.8544, 8.9386, 8.1350, 7.1251,
    6.1971, 30.0162, 24.4103, 20.3537, 17.4780, 13.5528, 11.8451,
    11.0355, 10.1997, 9.5414, 9.0061, 8.6417, 8.9975, 14.0834,
    11.8333, 10.0179, 9.3844, 8.4110, 7.5152, 32.7622, 27.5708,
    23.1671, 21.6003, 20.9615, 20.4562, 20.1010, 19.7475, 19.4828,
    15.6013, 19.2362, 17.4717, 17.8321, 17.4237, 17.1954, 17.1631,
    14.5716, 15.8758, 13.8989, 12.4834, 11.4421, 10.2671, 8.3549,
    7.8496, 7.3278, 7.4820, 13.5124, 11.6554, 10.0959, 9.7340,
    8.8584, 8.0125, 29.8135, 26.3157, 19.1885, 15.8542, 16.1305,
    15.6161, 15.1226, 16.1576, 14.6510, 14.7178, 13.9108, 13.5623,
    13.2326, 12.9189, 12.6133, 12.3142, 14.8326, 12.3771, 10.6378,
    9.3638, 8.2297, 7.5667, 6.9456, 6.3946, 5.9159, 5.4929,
    6.7286, 6.5144, 10.9169, 10.3600, 9.4723, 8.6641,
])

_D3_C6_ENCODED = (
    'c-qvRS#}#ua)r0T74YN`k;DEs=J;-c6jaQj?QRGtJhUbMEP<#g#yix{R9gSA&tCidNT2D)|876(pFVTj9&x4rmCrP)pZ>Uh<}&)_'
    '&w~HuPxn#Z;;a7lzdAmr+4#uzS##UpnHI;?nLmnsChylj=Tbhir|~g9d++aki*j!D)6DIojPhA>+P~u8KO7#nabEuoibu&T`$rz1'
    '*1Ub>67j3%pV2LU*3`#GO`m<V_EFNOr8=*D&+Ky)i*w4Ho5y3ATN!Wr2->vImgh%xp0PNs`2TyVZFAmnrnS_MoFW@Vod0O;v!rzG'
    'yH&h@+Gz2uNBgYi@t0gbXX`&-qCZDe{~YmFevEjGH}CB2-|MIDaZ#GR{;BK{8LaYj_EI}<oJ}tu^Rrv$kCJA&3n-tZN8wRwoM<k*'
    'cgZ8J<cKdG?`eEW1f_h@nE9Dz_c*0U&=C?pi;wovKATmGKWZNN?Owt%YSej=(@Kq~61OTI#z=21$E_&ybB;LQ95<=mN19nbTQ5-?'
    '_W18LUPp^d5s$C+&s6L9uy{_pU9)J+EefpJ<Me8rd5Ov-=ZO7z#7h@{#>4s1d9!l9yYDlq%Sb9S>XjMi7fF#PzNJquaqB%E0WW`E'
    '>)v=1xGGjt{_waOGvb`3sCIl%8Cf@f-4%!^nmb{HSCrdmai8Nt#CcfM1s@TQXS*v1zK@G{wDH78HQnNR#qAz(KE^*Ef1Zx7kNY|-'
    '&Ob->&G~zdP*92H^Uv{LzxM0nz7C7$(c(3Wr?8FmC(eC+yRTpS^>JUvMTAK;|M!<SbCXlyt*^5Z$l~>=@z#%$qbsWMM*KN8qFf_>'
    '4+>dVENmpKc*N#hpHb#D^Hp5Fh{SP5J#o__!N#N9;+D;L65_pX{r!sD7LhCNRgO2*JVIJzo8D7onMgC!;$a!_tmV5dD-KNYWIK7d'
    'tw%a;c@&8eDe6Z=ijw`Bl~zviK188A<IY(mP=b#~`t3PhhdlIl-9M@kv(xlAuF#j2kEl|{Nb#0N4;C>YLU+15+9OyJ`J)1hayeeg'
    'c+f|-j~XFw#^LeOjgepXV8>aOK8=E~Pi|nG#fYZHBZ$;^0wS1<d3Qc>pW{8$jrKAlgvImFxYGoc$XGRQ{M~yRA2TB*Mr4X~rpPr%'
    ')!ADk)Wq4}4@bB<W|N<f98E<u7g5wSLQ>Q^nWCqaKY69btw>h*rM(m#)_N3-CH}cbgo}2w_6PJ_HsfuMYtiD?Mu`%?HePt@Zobvz'
    'EXv)h@O4-`3B}?;io?8cc>Mpgi&27?yX0{9X*w)Et+oi89G5s59?$&eB))#@-N)%Lk3K9eb%_VA-j&F=FZcCZzdr5jun4&EpZf!X'
    'V9!w#L@c!Ubjq3>F|Z$tn26_|qvlpuISA{K4C4Qd`#di^UW#ZQRZCpqMmZ88F=Ar$lB2fwoL!`NpLol2;`NVniq0`Hq#$1ZeH)Ay'
    '-i*i8X4PYwqS55ljHk3*_m)L0ut+&kL{Sv^<bkUcKZz#m{!sqgh!@qkDnl`?@{mR}YS#bGT5<Y)#@iWhUoG7zvWi7wjQ7h*@fXhn'
    '2vddc-z)WU`HWyYxOGk)c^Kc!6h&T~W3uQ#$VY^c8bw<(JDzhi%|j7~?wNBgL}U7MD~EO{9uHz-yfQX4nRIIRh&R2YzrRCiYZwv7'
    'bdPHn(Vc%8nP-g0yjYPKBH1?1D!!?PRgu-4-yEI9-_2$m(;10yOq6rv5?Ymrt2G+azrQ&3Kj|xeHp=S7XViXil5x_9V<TKd39Dmr'
    'RRE42X;Sfy*8WcX@$VxO=n^EVs0hw8N70a#4(EfWDsdBsuT&ZdJid<#fcQSD@cugX&oQ}hO%i!E^S%6dNp2tWeaIvg4~82y4>Pu}'
    'Uv>MKuS24=^pX1$f_abM+E$|xcd&1MXC$PE5)l`pw$CNfTH^7!^e`|UK_e?gy&k1XyoG(f?PN1*@Y$l`YCYc5xc*LImwqk`qPTJH'
    'V!Y3h#s*K}nAhGbL`9qn$tLd9Z0f#yDt)*9F@kx7uuS5jRQ9sD1Q8o1bs#IC=#+^;WxPHwS7tcY39BpavGd74zkj6=)fOe*IASRw'
    'd;CnK5pxrt7e9|s)Hu)DHt!@cbPmPW>P)F3R$KWw<t>z)Q!%y<=BDuoJ+cHPxy_=uKl8dRw%IjAIF6*#d2*vHAAIe(najtwotp9v'
    '`f}sj_TfLT5@B~TOOA$%F^YeV-I$+$?vdr==ec`US?&+4Uicc$q9wjYn+ok?H)_x7>jvMa^)>d{WPTDviq|?*fUgmE8fg=>>7XJ8'
    '^7GE|C2E@foUVf3o&5H@SA)~aZd38+=aJGg9guJD(Cy!U{hSVo2>0^v73CqLMk>L5kfGH#n!oGsccUa5@zhsFGyQForEW`vkr`!t'
    'l>Dic{kc$rw<vyzh`qkS-H9N@PA@XUXeaY%2198o@ki0VKc|SDg*GE=!{nDZH?>Z*P=))p-+D1~txcJ%q%zOW&B*!$P(S&ctYKSU'
    'LH=aJ7U8Qo>8E@~(t7L`j_XgC#CNPGs1FuIvnwG*bJEYwBU22<c~n2S9wf9e>Y6W|d`yi(iQJSHcILrZAc_-R#-39@10hs8?}+Tv'
    'PQRj4{<8I;hB@1=I7`O3x{T1CHam*n%Ltd>J5C0N+Ve&;IOF9!X`AAor&8#wyj}zqL+O;YmF_1vmfMdaepb8wrkWlwmtQjZwDcYr'
    'tooT#@wZOUw^6F~nt$24ebzeO1COKx_RHRYHte^(LS~E6uQNWU!TN2SoyWe7ygQ$sOaCowcJIIbJhy*P-;+?g{f3B(uDt#BwY-)2'
    '8=_};uD@&=B4qvp^z%xdzkaV$?AP!0+hvJVko?)nr=|AGdTlT?I{$HsKK?r6JG%b-%K)~`_lV{#YMj~LKk>A9e5$a5{S1yWc+;b*'
    '(E{Jg^%GE>6Tk`r>$@-$PNPw6opmBG3i3$XzdN-$DAzb;2giK<SI9?H2@W_R(r5IS`=`qIA#1bjBd$UlC*5%Q9BILH@<ocl%J9CZ'
    'C2gP6PtX9R$5(EQqNA^~bdn8=034CBX0-j;&)QpXWp%LKF5Uekr!!M7+I;6yA6zEYHG_>iTR~0jn|W|IQ8h$NN!je=TM8u`r{d=g'
    'SFSTY$?HeUWVn-0O3}&`6UHjH2SXAvI0qhK7i8jOdJ)y`6g0#ZC1hWcFlUTQ=P4rB$-6KIV6wKhL*;p0eLQrjG|dmk%=s)s<gw7F'
    '<)qT>YWx>Asa*sd(;`t=yaqLYo>f>5h=VDjeV#qj*W!J!Yu9S2cYE)?n5HKlJ&=d-8e47kWNUi+Axw{W;jdqx#A_lxMm%<NW>Vni'
    's6XGD1^^*dB1cXogSDCY;=G)Q6;Vv)`DrX*Gend(E|Nz?_z{WqqxA&);v??I=MIuvERkP&oOz_`<XyauKGMa3AwWC@X9LP7WN8vk'
    '_U9Sal;ddxl^h*IA;@RA2fk@7^T=^AK<q~}ssngGi176Kv1oDl42d@mTHIL08EEvMGpZ6l0E^X~<_BV+fxy~qpyM^20966D_YqYZ'
    'bgSKUVecPRysDC#q88{7B{yYQhU?Jd1&W*>)si6Cqdo9Nif1)4-k>^-FU{SY>9f|&U?Y1R$-W`Je2Y3Aeq76aY|bto_&Qjgqy^TZ'
    '5S?R3wJYLzOt?E;Fw8zTrQnEv-wKqMC|L)YxIPq!koqUP!!Z{;y(tQp&2jN=TU6geM28Bh-|sm#cZCvr#vadDZ!mMBT>E)%hd?cb'
    't5b6{)z&{t^5=y0_-X)Ao6yQrX!N?-Gc{o50N@5Bs|p}A?S@_g`~#d#3$I9mYVGw!N8F&y_l=^zK$ePUyw=SgEMC~oEvOx~lifh8'
    '*WTVl72bL@j1jmp)p!A{Sl^c~KL@ap**`iAWfrp*rx>m9CIvYKP?X|h-hx*rEM8pCA-*4dYQ!%G@7#KHK;YCiSj6um%<~Pj7{PDX'
    '@PB{A*8wX{T|i5uf|W~Z^D+zGcs?WA88vS5Lo#Rpf^}t9OK275pXQ9|x>GxtL!}7L>=4%g%+b3?vD@Y-Z}}ER$;uWX3Ju6WmD~Vb'
    'kHjD68tM6vcZ&$=@OzmK2(TT=B;MR6#&5|NX1_#^|7=w}4P<z#M_C9l4|L$ruH|s9QS*0e@D&(vG141iQN%gNBhU`jV{VbqL3d{O'
    'eE@i<=5mP`J(+ox_TmmL?o$+9h29L{dQ|FsO{V3YLdXyEI@BPjZhMbA4lk&ZRXHBPh$C~8!qTi$`}YWYt%(-O1a7K|IuwF4f)`Wk'
    '=sFzg*-$o9ab$<2mQm&qEiF0`A7er(?<sCpBfZzhym*wmY1Va`Rp*-~prFim%g1LUla3P<2^i8*NQ*LB(iGvn=fk*+Ip07yg;D8@'
    'w`P=8VZ}XY^Dy1ciO?BIU^b1OAeW}msZ;AX>|F=Ip}6WD{7KZzjiIB)iRn=g8l4z5ZjBy>qP*p}VGbtwFnpZ=4nfOAcJ5KQM>%Qm'
    'J|fKben@*!;}1AeQ5BaIuSSDH77eMxpgjZ;6cHBrHhQKE%~S+pXr}cE{O`!HiMe__n_X2ekYX{y;;yvgT(L(E=Qb3Uuh9p4T`C2&'
    '9<CNGDV*VmK~^K8k_0={dk32L+H~S9mEc&As39=t)#^rapAkL(E=&bUG@WA+2`-Z$sHyG3*}w?W(1~fsT%^59Gg^46s=w*jY7ogs'
    'yrppN<H2)OLn=HyBJ81et)M_7qDEnD^oLpLz4h5d@^3!vy47_=Qfq@VBrL$tOyXGgHL<FPLej{=L{E$3`+86TeM0<;cG24`hq0Gx'
    '1W23|sUlIAcwt%Dp~3OmX3;4?jGa*2)<X2C^zNKr>731BclLc)BvLR(uCsR#i^U}^R1d}c?bGfKBecObAuHM8{J($c_osay#<VaV'
    'q<{ZH!TSO{OW=X*!|s&nISZUoHEd7mj&EXO8j5r^*Q{dzK8-6IU8&+&e|63F3-D|Rv)?L3MT@d1>CdRaO~ZG$_d`Dz^HY=_421~f'
    '8V%ViG|G&H(4cfrB;mWKmAW%gENOR4ftz8a!}4at$;VNzmR_zo-YI>el~?tGMJ&!L)W<}?vGPd5#3DR<Yj+!Wzj3vqZ^<x$kOxAa'
    'CsQObqvR^&em5e?d>X6(EuHETPJ<Lk-xej8QK+|9t~xnQr*h8Gc*U7zfj<&Dml+8XV6ebkdcS*%`AM0Oiw(A%<bfb|*Cn-;2G;^G'
    'z_@#tGsHt_*hqv5W?)=|5&erEmPnz}amshIC?F<rmH5vQWd<|hhEQ45ua&1Rio<-b_ZSz`qu`5E%Ct(vN2FAHLlJS%y5#hL0ak{F'
    'T}NMcGl{|w<tgPcb$gH68kI>ecgqGXGz`)Y8B_R~C>tV#bCM-0cUqXZoPQ$7B&Y-Mi0c?BmY)|y2D_A*jN;wK9r6<8Jy!nfxFPW>'
    '##<esk-DnmyH!&po6cY*F3JQ1!FY&7?sSQ3%p6aAeL#7{R#G>EF=LPXuih!5O{Zy^yhHPTXdV}|eQ0dZ7x6gWmv&n=JN(7F3W<+H'
    'RibfGA`CZScGL@S(j><F#->T0weefq;AFMr?JRvOY!~C#)PEbaB6db{pR^*?UBA1t@Nk}`xFRNPvT@%Q4B8qmOUun}QnQAb#A*Rr'
    'yQ`h~d5SBD_(bV-2?E7)LJ6iyO_;UUWhw)8y5^4!<}9Tojcc7tV}boazQYkJv#f>uO>HN_g-ai!OHOB|am|XMN}xs<9Ej31J8+R|'
    'V%)NZwbkEG_0(tFF><}r#?i4-=QnQ%w~zoCDidDr^&ASRu_6tX(MdA2H6r2=mZ@5sGt2~3=NHA%M4tL=rm{#6DndLRGPPg|6<L-N'
    '7iQdp>_L<Ih<COkkOjdNuG^SholZj$tEkJr+gwtKWU{U<ly$^DG92=Cp*kzGOw`Qzbr*`#P|}5!O$gK0$j2mIrWG1MA)AF%;N47x'
    'XQF|A=8)!kBO)UYNxj6shKmQ@AzI11%P~lEXwpR7lunz~HN@hHn&_2kD(-g!k^*-hQUV~!f*u(qTM}JAYJ*f%<sd!$k7-vmBLvxz'
    'R|DXHDob3HF_?Hn`p%h-2}=IF8xq_dOD1Pv4nZ=qR*9~u6$-K$Z`WOJ{`0BdhiW2(ngbF!W2`ayoOh^(Ki}Z%W516rh?1$U8ToIO'
    'rtfe2^%cH9_WMw9D7{yE!0?>Kxe4Ypm!^osJLAj6dG;g7D>Hl5d&WE2D4v>TXZgZHB(Q=Qs4%vtxQjm%fO1|8e^V-abp`hInn$Wv'
    '^pkR<<~v7iqpXLmQBC(Hw4CAJP2yYvGTN>?&ji5LYb8dPh%~%r@m@?y%*-Y_9Xh+rjJHYe1J$;+I_QCn8{nvMq(ae7H1I9)sg4Nz'
    'MB_AiC{l1que~rde-&vlwl}Hwbq+;Jvj8RX{JDUdRqEk!e^K<LDe7!uw0)TeY!*<UVK+b<LngI@(Xy~NJQBv)*W46<2NSc-fP4xT'
    'HPYdU>ol$Ll0|$qICZQ{$fga&oAQ{Kj#TYQB!U^`Io%wrLPmmi_!W!*5E36;#pux3<e<ee(oZ}m@T@2SSj$Xy1GG^k{tYtn^+2<W'
    '(<;ts^>`2?yf<VQP#A@>ci8v2WNg%5MGTjQs@_0ne)k~3t@s{&bgkbemzmBJneOO3dpp35rMS#_A?1xwb3&ocmq7Ia=jjM*#yjEC'
    '@Gd%+Ir3_#4!a_k@5Q9M?M!S8VeohoA!fBFEFgf@rQ7diH-%Cl1PvO`d3Y{ZwZ%aKPIdheY0&7G4qBelnFi_t5IAb?%A}I@1H-Ks'
    'vl00&X-Ks)a)R?`L4FYR%A@uP3REOLoGj>oE6N5_id3z(>ta0$1y7TNFwn`u*y*NiINwwH%@96J8ZIEx$DSwRm}bGSgpP}LRN_in'
    'gHSV+QX2V6dQ;9g__~|DD%+G4EgA|gq*{vhG=Ln9pAujt@`}TZqk2y4_~=VAXknq!{Qyad?k29AbimN1Vqwth7owX=;l+<NjaQLk'
    'CP|iPS!RTO^eAXQB3t{9ozsLLxqiy&BkM+$ZYbB3qmCxBA~b{W#Cb#_;IA<kU7JKw|9+D48j58SnjWCY1aXyoVMXi-d=uKCiY^VE'
    '_#DmDOhp{!EZRb}|BX;%1?pXGXVF%H`i8={2+VvS!WB#!z`rI01tB{ny)`aMjUUI495w*rOwsh3rT~((r{l5%$d4D#)_%&Uxu<qD'
    'QBBNMa@GfNB=XE_g35P}I@VP08Y^BF3GqN2pgVsKOAG-J8yzTQ6+xoxsA78eX<vu&0=X3T>QZj|`@y;W((h0EK8#MkjVFu<iJ5FE'
    'mY{4GFQ;!L8){wxjMzd@2N8k##aD;GEprmb$fOreZS}sgriAg1YKn$bNyzSaylkV4NXLbi;Up{Yr!#E2Hf$YK<U|dGFq%<l6a{6R'
    '&}7n-Mk9!D@Rst1+stnUnvl_U4Q47)Uk^buUO!u5Zjdo00Q@*I?PON;*xbyC{l(Nx&2@e#lx|m6!-N-{^pe2k+GdDEzX(LWX;@R)'
    '?2NwIFKYD`u}>~L?ldg5JkX(y$s1fnHi-Dfu=m^ro(|p<Okb36IK=T1q|Fe44b^RBC{qYvNgI-y^J`e9bx2Kl^^hCzhLAo(i>t(O'
    'cq>hN+;c<U$Gp)sWB0Yx+koKU2(f=rUpQnAl>27GiQ~q!COl_Zy;|By?=UuXft#rNWHV87ebO=_>Izw$G>30r_A(UF5cYU7<mLf$'
    'BbqTU8?7f+Yk+;LLV%QxlU+c$ShiLIKvvhruh+H}B#v~tFi4<e>AP~fHcv%SSGY$#lIYi4eP053rRPO%pa5@&oR;w+DchN$%VH0P'
    'O8g6EhdH-)@w*kTI$XQ5W@lZ%A=`O#t=*IMI*g;Vu3%$*Vg^}A(#D)da>-JEe*KdxaAsoV5>4H3+Fb)+zkB&me227Miio8}S0I;-'
    'yep;rm~7*i7w{&l7w#MNFdmNqMIqY_%>uNl&L<I-=-Zs}98Fg(Z8>_D9jj4|O%h(La^tHyoUQr^XX~Bf8AmH;Ro29i-W<x<8uBOw'
    '+|nCwbun3gUaEvnluM$;$ZJgDy+06Gz@);2=_1t%jf=j@tUCg;7|wI?)d$Eh(rDj|ki~Td{H2{GjT*LjyAm~GMgatr1rzo4FbuaR'
    'bv|<BX$#^e(g_X#JB{N%M^x?!4|su6zSCR(`i$=*^2!DC0?h-O3Q{uma#G%Fy`!|%E!*YOQ_hQnCJpyl-ZssCjgKO)w6%NeCv|l1'
    '%ek_{cQ4l`Ec5Ejp#ekOE1PFp#c9cWP556YQ{@KKgVhntxpa-f0=6|=##zsA85A}qs~sc46OK7vTCAJt@AC->&)Iw~JYgh_aiZcw'
    'tP$0yJlf1aGEb4H89mgKAl-m+;L!;`<iz<|24P7`6m|wA7yNt)j+J_1AdYb&U?W6-A3d`&iDsB=6;DnT#@bvqGp(GMD#5`Ye8o8B'
    'dLj!MDNhzqEYRu`?U|{&w2W&@Z{@5}tIJ)qjuYh>ESRMy0QB}Ec6AsU;89NZVyAM4QGg}cboFN`hy0g>dC$wxpp5$FxegkWJd2QM'
    's3*jroNuz^ke25tZ*Y&WeMF3>B*=g#!3};%hHINUF!nS3e*BftCK2)dDjvsq5)tKQhy2Z%y~0Rq8qmkNXg5kvNt-NY2WJ_b*{?C&'
    '1XEP(iK5Ziun)|{p>O8V&jdqdv7XiSPd}3trOjsdEWOP$byY#dKxYClFV#*jh{)qyko;(zIUP=9WnX-sQ~go2f#no}`Qkj&>@?<e'
    'Mc&peosc}ij18uMoL5Uvtm|MYX-tkbfuB93@v|sIn#2{#ne|;>+jbM)Ew`%_r@;6ckiyzd!pGOortE?J4cdv%@v9|5P5vI!R)GHX'
    't&{kQk&(#l7j2AI7A)*7P6i@q`K8Q|7|AcD2BNv->n!{`fJr+)9~A6<$saE4V*71bW$l-T*dVt31Bqp<*v1^cVKI9f=fBrP$ou8z'
    '6feIB*g<6b2ahz8AHVI2BsTC%gN2E7{e_;v0ER8{Q};*t)B7*XTo9nYbP32AU1thePBc63a6+w4dyiO$?0dKOPe9R+RQtteCc!3s'
    'll%j5>$eJ+ovQz$T|Gs=xK35UT50+(AYocSZO+bXbV&5Sq||B9L&lSf(8Rv8|EeR9(l5?%v~Cfqe>?9CRKtEdZ%d{BgBqwUL*erS'
    'q6BYVe^)r`zsOWiX+k}>MVgqU<zK@@{mn&dM4meHO?gpeH|sBXHGUqg`FlvlgTb&?-aZv|s9FNrpBOVQ2^|w)&g=~sBn;@JIyDe2'
    'e{UO%Qjsu#v+WYxKH2JcG#8nz5=f11`TCFL?1(&~I56q8Px`GE31c{H79g~Cl38k>pC)(;hzB5pm(Zoax&xHTkc4MDS;w&-k%v8^'
    'Mkk9jBpou`mn?y^6EI;x1_{c0q^fZ?-H@IG%&9SfQa$!<u5tZKlOihoniFUPG7G-gw%$*`00nK;8W#=8zLSRoeSjvV(QIe-)u;`%'
    ';o8DmeQe`U#5zbqpqG4R0X63mT0+3I=Cj2EEVeiYWQImQ*{rag2L4>veE)gnaDg=>p6)!``czv)Son46WX+ca<%vhmfDK%uDq&FG'
    'tcvkkPuN`O>uTlPd7rNzTEX9sCWWFh+kEf*7j4#O%W^@%u;Ro5et@bdSYP3RRo<Xy23z5ye;BNXc4F)p`K(8lgawMPUOgF#b~aV)'
    'OvL%@7coIZk`5$g6pqCf(sti<B1lErssjY($=w?hz)D%FMlGH6&aJS`1Q3lOO=(Xw%4+~(#*)k*<peM_#NC0+D{0gd_|L%943|F)'
    '9!5H8h<Gywa;s7D&3-cAYqYfyc$ZR6pO0<5+W>bfJhtsQJ_~^bwHAsgV4-|s0Nl_`U=CTTJPz}T*cZblDY;VQJP}^tgwqOTl#G7n'
    ')94lP10*(?@QqVnL8Ns-fFh}Wm2%wxu84Bi<Og*EAY<*EVb{CFX5N4dQjh{$#j}<1-s?-7QEhrc?-eVpsDb<0DDL)C%xF_d*B5VU'
    'AkGIs%Q-MPLWgG5ExyW7$D9dok}lL}ZPykqtP`gcxjqhQt0Q-6@9R#T8v?Q&i7-S26I>LBHkLO=bymeljVcCF1_c;IK`&0&?yLa-'
    '4ksMYv4DxnLu%qb!Z}2#5iJrY8l{OzLqF=ayy<<hsYZc<#=rOFLg;h8{_!<Ot11eLDsnamDCW<3vxcf2N;3M0s6bF=kuuUwWv`~j'
    'Rb1SyY2xZ520wRPH53$4+vt(xC8e0+zi(<|gcK7ZvH&?M<nSqBy~mY%WpJ;t2f_;i-kdxT$7TT_;X=gEFb5cu?~gWn*-RdbJZ)hi'
    'HOQgn?7ot3Mh6hQnqDtx%>6d6+0xhm&&LH>>l@VU&=d0q0vM#ptvd%+5EB<>l5<nrFm6e9BY(x4GR0Nj`8M*rlztm%p&`oXf5)Z;'
    'f|%g_DC|er5um?639FU`A(tdL#_Bg$^6d%XrpnYA4nJOJ=qcOI-V;e~Oa*Y1jP`b!Okbd9n>U@IP6wUG5_aAwOCWqQ7FWV0lWdPt'
    'Wcz24ykL-2q|64r6AnRnL#5+C!6EgCPD8{;94;E(ZH>2nqn==x+eE$Y8KrAzau6HK#aUQc2@-eS84qRw%WBY5cZk=VGD8&kZlS0l'
    'GkQ?Q8!C}U)i^pbb|^-{peUlmT%aw87y|($LcLhFai>fMpPSNlA~X+zxnoQPuYwaEzR5YgutdfrCk_>JiP?kurh%n^8xVes@)UJ*'
    'M(iU$$;i5+Jc%W3i_$;RNYsnaQg>xk7p)|st+T$I(sGRoU^n_|i0q@(?0gtP8szplHf!~U4HrmBk89T1x5)UXY_um_ej-@LBu6gO'
    'ML2KyfeT^XFKoq)T-k0`g?WdhiMtt{E#mcdW5ODapqpi(V?lt~1xBlhx`IuM+3wK9?!*z@W1=+{r~Y)4_4slWR)N(hN2!3&L`u-C'
    'HyMc~TJ44bQeA#(n4@e_`JDCuS%sf$qntqQtd0vv0#<G!k*VN)O%S6^ij&yX-&mo*cb6RbOpBObM!$AW<Hc)`@JqqUhbbA`$gAvv'
    '{<Gl{JkE-^UbKjWVxUIIHEf!Z0V4<h<`#E&&YKW_R3z8ND3t174dED#O*<>k0&HX>mGON@CY{K_M<I9<8NWZ|>wt=8QMK2nVq_!W'
    'c!?z%?HQ?~ZRAyu;ozRDF>?KB^Wu!ZN|!oaSX#L^AkmoK7gL)N0gd(xkD#c5z`Ge4-bdR9PPXt24w^w)i>APPzV;i~a$nvo1<zPS'
    'Z7BP8lM%y-GtZSPCFQh6C1W$TKs(6lP3xd7t%GubK@=t)$?`(ZWE?CJvne}?6RhAySFpiGs0<0c^bO5{+G$F!v)*>3NS(ADdNM0b'
    'F;s%O;QWzW-KpXr)ILO2>r@@k(o!!E47?zSd@&J;lZ<%U&S-^oXGVesI%1T;P3W4rshm;$u_ht-K?Y(0weuYTZ6eFd!Gq+Q`P5yb'
    '&s?lpM0s52_zGyiKX5O%%-xVbM3gMz)ZppZY1qi28=o}{cOx7y$k+t|N$4=J?iPK}==>R-Mn6$<%!Q$km*v!4?Mbmh?C~1?DY=oS'
    '3gzLkk!eIka#rch$2*<%=h(af7_u>8bNLX<QLSD;e1<_Q9Tswo^5Mc#V)MP4+_P;dxKF9d&v9`yDixtji11Bq60I0-Ja+PoR~*iT'
    'sjNDO9(6;<@q(r%eJCz!Jul5_YZqtkVEOdv^ulzM9Zxva!G8wFQErp|+7VCIoz2T;>Zc@2l*14Y3d(}M;0-BVU|l(22cT1d!|G&~'
    'OQSz`5Ua>;@L`)4KY6IrmIUE$zE+ffX<`#el#(TShq2iTA{W4+%Z+p{slN0i>Re;)9_a+3eIFB{eN2xw6ErX<jf$c0m-vF(1g2E`'
    'rf;BiXE8=jVL-Qp={I>3w>n3q2g!Q0DGQ|JvDpRTelVC&XhFuM1pLw;$)hoZxt9N1gep~<Xu9si94rch>WpL1gkYyFv)aw8Z7fi6'
    'nc;w?>Mse^4TRF<wIqFh!+sOVdU>QirQRG^EvQy5%Y2lXknfD1YH#rbP=F)-B^~%kzqM{aYZWAZMqvX#!G=36Z&B;7N2hCRyCgz@'
    'wz99osMIz$hDUyawp)tba+qfEW98x_3v~`%3p_I!V&in&-j@wh9FrcfVsnh;5<MB0%*|k>OPFrpBEq65s;Mdm6}$M(hm_PO%Wj+!'
    '<Mr>-9BMnY&V2JHZfMlt6w(kJY8yG}hJbn>O-A;#!d#VR7gzvhk3z!u&aDaU6jg3Sz_9^se;rVZVNW+CX*|+qZu9*S-$&F;>tL_v'
    ')p=d2|A_P?`D{jqwh^bHkwz21$V#j=<~25S|I1j-D-5ig!;}zS>3!o(&KQ4U{ATPUu>Pi&Eon0$VvJp>p`^$wvyqnnHcq|b|DyL_'
    'Lc*m>+%ZZSgi7lsMH?d+^b$$@Riohq+HF%HF=<MY5i4jfF*ax9I>zf0f7OaYO4l=@2sS<}kkY1jqee(W+R91H7$$7KBPz&gNy^FS'
    '?<OT(7M5)%Io;7t6Ac^*mr1$T>Q+)9YU~1mxyOSA3(WHa6_(j+i+h$e{l*`&;QhNHSitfcjcr_}@F~kL(rh<dYja7ofJ_~cz5sFU'
    'UIL)gg)V6zPt9;GE=1o=;R?-Se1ZhH@sZ5GOB4JDJw48t6LS%#UE>ws$v#U8j?M-Sw+=-2WjBVadh;FBO&$?GIv%6DPs~i~U4*vy'
    'AeU)mdlF7Lt9*#ZnrgNtP1)wIqr-=+np`~K*E*|)P2&ZD54@X+lhFWmWqbV#kMz@wgp0{NAO-eRd9y+179fhwI<iDFgH>+2`cfqN'
    'b|f9E^I*zr+1%fFkG1$0J)W^5VL!*LFKtHYo{%49xWL8+Ky|5%{qwbFixbQAK^;MNh7pXmRlXZIv4X({ssf{IUq+981HUz(v=00h'
    'D5(Z2j9beE%H3fYO~-3gi6YoWM5?8--GM5xsRzg95K@_MGe6i3!(8BbtmnHt5*e`pp51VZH8~P=CSq-ZX3ngw_a_0mz$6wNmLtH%'
    '4W}Q?QXAc1ogx5DJl}`(sk?k1Bgt;CPEl2v3vgjeMqehFV<WKWS^DI0DEZbkYty5#nXzeK+9YRRv9)CsrIPkxBR1w{P(CyW#&jDT'
    'X5Q`I;=IZ7dcEUDSB=^t{o8ikYyF0(i7VXo@VB~~#!;^#Si&hzF)10h*7=+)g@i$Aw{cjet&rhl+YQd3NcghkoD7>8$I0UD=Q<y2'
    'Vb-1#l=u|uZtT+d^G9q{KbVWV_EgJPa_-j1__|in?i_7%W>pQ%;ERiU^KecC@qzFcS>m8UGTQr2*y6R3v)mN_5u1CwI(Gs}jo^$a'
    'D}0`Uzh@Qv^V+v8+Ap}9@y;<<Yk0$wXnWgQ9u>ta{MiK%$`Q^{kldu;21h08$G8X?%qF-tykIXN#FDMjgvVEE^~M{C(2EMy=B8o-'
    '4*#gK5GxcNz>IfjlE$vcC_W=^_>_ApV3BbFzZ{zUh?db$i2+)mYUIV8|9Kj>BPtP={97|@ludo`{Tbg!aI>#)!iF0ZJEN!zk~~!L'
    '*WG{((f+Oy`bYvm<EnY>HAWSxM@m)(6eOQ?pKsIRB<b*!_7RmGKHd0`FkCEkOB{4oQUHLwGe$GU_Sd$$bqHgKq^Z(mxgFm=Qc2%z'
    '9cgG(yR;>Wn9<AF;oC{IBgwyuO*EuYNF6O*V4}z)ilY;p&4E2VLohEJmMlZofw46Au>vWRlUm=zB}kVU?l;J{jHxMeIzgv*IJG&~'
    'gwVj?#eeio`HPBC1Uj_1;HwVwiXQhOVIvQn6U10zLZ0PNy+H!&ZjXr0C=Yd3?r2wf-}np&(^#XX2Q6;#S@Gp}BK_HxAhu+v;&agd'
    '+3A5xa4SBJh8cVu1*tvIFj*!M6tc`MvdQ>&_RYjtZgZMuI+aDm`L~s=8#GqM5!<7A=wb#%_!zrcVqC+7%_XSCh$>A3*e5yikvl7C'
    'mSJ0Tn;?p9V}A?1DPj0)zzrtbu+3NO+Kp9qU!qtW^Va_QQCtTSA1@#eTNI8IQ9BC_$DO#j{F%hd!cpexd!PnGaxh7xi9~;<kN4Hh'
    '${>nr79<q+Iy|VsTG?iUD$;+d@=fgnj#ow>uwA>x7@c4qU5m+V*>O{gAlhsPg&0d#KOz%&*SO1Q)$N2uU|TL#11w-L1rrRZUVxs{'
    'yHxfW%tMXjFJ#lSX*k;?`a+(BK<g1X*v?wsq*>PhsI2{Gv@=blfWDK+p;~E8B)l0{PY5V@>`%fmI?E)|X@!f3<%?v%>Sp5Yaq|k+'
    'jir;08iT^QXAi^+!6-3`HJITOgr6g{xfzLBm#g~XU?!=7SY7nUpOY>~LD~>A=57lk*3LBE{-c(sEW>s1bdraYO2uw_X_@_={MXv5'
    'Lz(*3RCgm#G_*m#6^4pZF==F~&D@RWrONcPk-1H4aZCQ3IA1Z|m3kIW1+KM&uf8KnqS*-7W6cMrv@CkrG!YcCY1SL*1hE?2z~&Xu'
    'deBUm)kq@NQ`gdVTHZ;KH0GgQJ<~b~w)fqy9nRTKSmK7Fv7%O}uylW82l^sB0f8!R6kD&T;v*jQU0v8xhh@liB%%+62n_vh_1dj&'
    'ZV9Kf=tjH5e<BulLGX4kh7oTn+AWOBqAH1ij)D1RK<KFJVL=ES+W@J;?60$5psboqPve^~2<;e)My80zbnf<969S@XiXj9GK-#j~'
    '%Q)t<43MLZ5Va5RNWFsbc0~Bk8aFBKT3Ri=v&7lqw>=VQ&RW~yKaX<6P=nC7>&pgEt|q^Mu3*vH1q`S%$e!rlO*sB2wT6@+tZjH5'
    '-$!u2vSA7e->mPJn_r*teFP!>RL%9tvUHgn3m@OL8{A{s7!?r142ixjXWeNRA4IA~B5g)Y(wvPX<N0k<$}sz4$i?e8%_O8ArEcGx'
    'dK^KGCpN4{Cy3Q!_lw^#Tj|3ksb*;mm1{F?)>HHW4Fa@C6=HNpsW)ccDH25PaRld!Eub0foh#Tz(son?1+mtxQhQUCz7v_tfE}7<'
    'ebf4y8uqd=L;z!X6W|A>7kOwqfkU@19uO4%JUT!yCDV88-l!i-<oM|S2K0}(GwG3$i(U5|f>(m$8mJ$^y|L{$G$RIt7MpV*D5$!d'
    '8dhNCS=42;ZW^SWO*JJ?+7>x#(9YmvU*J=0@#u-G1u_9SCOqID*v<h)G!)yZ8gM&ktk`zLZpi#O&M2N2lcZCGwhc{bvXoxq(!u8t'
    '1^PxCOv?70cZP{qm^pfSlB6$pk7=PCvMu_SyrFy1XAYFxAa2AL!<#W}kX`8hFtma&z+f3XdMdBS0#X>KXgdvw$SJ7e$h+wR+<~!%'
    'rAT1op?z_Iwm2m!!D^pNC@YHA{=RNVIWfR!UIax`JqOp%b}E+$HD$~$n7VYzA|ZD!&QFn6JD6U^oTbR|iMMK_A007{<yE{!P|h?U'
    'v<iJ}lZTm^=R7nt%0pB_aJhbkcaIZBNSZuAm&TaWcd%w>5eeX`dOQV%K`7A8-2)9!uosYuc=9whpMt4vK9$Bb5_*#uOVjj;C1iaP'
    '`qE^Dp++mM$v5-7kv8)@ri25l_|m03c;mP394hfznpK0)hP6)7+B)rEK}-ongJ7x}DZZ~#Z1|3C0;{bqxthqpb;;WPW|8^--&6wt'
    'fs72QI_^b>P`6vm7xanIEP#B$dJ^bf-%;3MRTfl_fz{?IMlG|`O$y~@14&l&R7$#2Zi`PEAwo)HjZQt7VDsRwv=bg<gqVc3;H_bl'
    'Xm~nq1UU*Xx_CcBs@L-QEYs`)!#6~2+7bgANs7!8NZcR|Kr>D02Jvc+#qKE=pS)=sBs>WGp(YvWz9Uk;P5OnS+Yl3&j2OBUuhLFx'
    '(gZ4?`Ar(YWFfU@vWrWLpza{(0!aVLLedef?K^IBB-DsN?i!b-E`g4%*jGool4W3qa#&YI(yqWX%n7_8{J};zt??z}{exuWTcyte'
    'hNvb0rqsX4S?!2(rm)4Z4f7ibMSuyrfP3{M#5(dyh$}=rpu2_#(&i-63b87{9R6v>`N+3KfC}1zaYqGV%ayspsC0BHUXTURx;b-*'
    '8ZTzry|5U#$Lsx3RE9AlP9QaRY=)d}P77U|_$lQ7xB}*nt$IhL-Qc!XNkT%i-<`$x5$wznXJt!TZPR~$f5!I_S)iZ`EWT9mQ!@ew'
    'SqRj&IoT-j49uA^IVsHl@j|$`3@v>fAWGUeu6ewCR~QHW$M53IsE@XIe;S%gO|`kV4TOoBZ6>+4;1fHAm$Y*6O{4KrunL4aHGx2~'
    '-;UI|E8dAMkZDlr!GfK7F7GTUh03s-RAQr?ikQ~N1;#FAh>mL#CmjZPD4|raiLR)L4JjH%(JLiN5i5V5D>^DX^BKVrLs=yrm5uQj'
    '<-WS+2wu5njp$n4aoB62<q;7)IyVPJSLOpNXJj}N7A#q$$>=nt*O>Rb`zcqY=vo^)AO6GGc}K|;Gve65hp9>|G2WB9Rkd^=7V#K4'
    '(kTeX?!;v#zdDLIjgH~AYA61{`8X>%X3sEounBT>j5fU}h~re^&c?a7k|mnb+uW6xXrqsBF-vBzZBsH%o|Jf|kR_iS2c_pphUWud'
    '8dp8iA}U*^0sKyw4pCBBfZ1>jP-okTM<N9p*TR^aLvty-SFaEGvOLSw#!wvC=%Glub7uI&Ly2K4R>#;vocU>9CTQbS%}DL08-{Mt'
    '^mpd2_`tfW#+gMt;x0k{nmhV>9k+7srYP5>iZRN?Pru{ZgojaOIrHV~*?rS*G@(`I)DAjjmFmzBHgp2`yoqHDqpu{*ZK?iWfGi1d'
    'G4(pu+}A<;Vj!D0HX=?)GSIYWahI{$m@;>gpJvMn7BR(morjj3mTZSlEKp&Ztk&w7OFL&varigyXhPh%Nv?A8cxsk6w@o_GA4Z+b'
    ')P`nI@nmcc%!p)uX;SFfXW~&d#9(IWvc6&_o8)-GjyG)rGeg;g`*@c49Xj4c!ix_a5XW<xi>Uu~!Co_5!QA)$8v_(Jll>$yeKwOj'
    'eOqAXsob9ainb+6I@vZF1Q064#brucF$v^hXyr-nS|HRi{v(vmp4-wL&zqND<aRwRRAgN?ZW!il`gEiz61bC$Jb0ya2Y+lApoz<8'
    '*jY88Lc%CsOrCDLD!XvY`eLj}n;?nfbag#N@Bs|BBW4xm;5hl^^u|fJV~K;@>HzzMrnV=b!aG;a45*DT+yT4z0)LNu?X%AW4C~Oh'
    'I~y=ScSX+An_ex!qD3_ZeotHxxJ!y58<;7CL=>!;N(}@gvltYmT%fYTVF8qy15}}ls&xE29*Sx&4MH*fKvP%`dEMZJV0ZYG1amaM'
    'sH-7%=YD}rK~BuETY;EmYe$UrO|)6Ng~!BWefhjwQ;sI~Y*YylYV1S1X@ERwU7R^C;IG75r{Q9t?}(iP`0j3-UJYZu0_V5LH(t4;'
    ')3xu`0nD5kpyTeL)TN=s1;~;j=1kf0HJ<1a{ZJd5nip5Y0!EM7Tu1E(Ds>&(&2-cwLNZP;7G>PpiVJuO4A@{cOp;TE+c>IBb=JKo'
    ')jA^D4IxFAw)Ez`{QivZBbagwYp>6{AfTbcA@j~YeJqo~4YjmO#|-2*=>EZm<Kli_8Nl>MmiHvnaXT}ae~14yb}S?pkSm39mS9nA'
    'r^J*X?He%Mj^nZPN2g!ho{n7UlI%yQN7Xo3hPf%R*$XsO!1X@CG-CE0t@uu&6hlEI4WQQ3y><EgTmap%dX1$c(J-uHq8VXvGZ-G2'
    'f)YO<1H`Q#NE!KF9ik#3mk}kiDrtf!j&`0v8CK~OPoLep^LTCgV13dQR2Qj68r$~3%tR+rEs_MJq(}gu7I#2-bnoNso$|33&O0ty'
    '-3lml7@hcVXHJXxciFlPO*DSHlo=RI&bgUJ^R7jdMlQ9>NF@$Sk1I8CCFoHRqcwtkq)YZ;?B>XZwOONImoXfAlUs41M)4IX7_Bxy'
    '_8a9zL<$jB0*wq19U5xw&*@}o24*A>tQg_VF+}K_G0{RoJCNdxpdq8SxwVYNQI4G<1YV}JaxH(k6a%N1Ntg1Bsk$G#A=A1V@wFq6'
    'M;dkj)Ypap>6~1m_l*r4=)!Jq%twk^W{8hO0y06eD2g^CIk=78n6StVsZCAqdac(6jJpP5a8cQ%Wt!)B?dHyL!`Qmn&AhjwAP<5z'
    'rR_+}`HZPiW;8m!V)XC}*q#D3V*`%xU3A!aDKWK8y^Se#l4Qh0J0CFuxo>X96qP((1?vJ-U6wn5?~JDCfePA1Qi{iACQHukNt~2A'
    '>Y`{<BK9UV0Kt%VB!IGH@H9pos;N3jWY@Og6hPrgm(XN7ih}&6H<D!sqIw|0;%>B6ndG*&8K@e7uo>>+w>4hK2l`JA3Dh>psy2!q'
    'My2i>tPJBAlOzKQklesZ>fNM#tV=EH(jv<-7jw^d-ZX`5&`k}oP0{0`o+wY!WMJr9n9+dg9(|g~aD77vL3w&m=Exg~0-PCl-(+3A'
    'PE`U8V~IscF)UvIaLjNl^X@Aeu~?zh%UeRF;MG|WkZQ7Vkg(ge)zxp-V(6~b><;O#`+VR08}AU^X#p3598y`s(d}T69&Ypn4C}5H'
    'Q)J|Hbvs6y5LC+>21bBjEF&+rfk#HCy3`^}Kux5^QGwrC$l-FBAr&%TjCXM5;WjUz7cV|1&a4c%6Lp6Y#wMi>&=riqy)BL_FXa|s'
    '04_JGw5Eh%2sXjODXCq-`jxf}-f5i9n0PQkA=&CrAuowdEX1rSCNyZhl=?P_C$g4MkY#4=eIjYS{ss$RK~=-xx)tQ{GWEr7#Z0FT'
    'q)*hl%;sAXRCB*#t~dZwibw_tK->isL>!b|&ZZGnp<69d58R9uYJ=$jt;}xHjg0cQYSVh;i}1TLR_qH3U>DfyG|dfD6#%G6@<W2R'
    'yPH;k>GkEvAYXDv9%xrEVM+@~MP5B%4VghTyFltP8MU!?gqUuN$dGsTLEi_U8AL}<KCXFF%6)&t_Yq`lPj&_SETEVej37DjK{ZGN'
    'pBXA0Y2OZx0O37TSc}V)gpTj+y3^!n`Xma0EP^xo1?hty0hLmDPj*?y%9U!ig3#4X$$+YDF?hQlX0kkfW8Nf|8<f;6&8p~YCj?Tz'
    'F6y#ItjXf}CYZlvp}B@c?HFtx;POSYV4MowBP+3otJ<>;H9GmQRm|9P#s3jXCZT>6NxP3J)rvmswL2~0;2Xu9iKw&GUM7wxsB3sV'
    '?A-DKJH^Y3&kI`A$)hNW{$QdBc?dSlE2tRd^=r$n&7#hZ(rUEByh}>eVDM>LneHIQN>`47ryqKLBDp7g7!oSKSw>uV2T}>n@tBs1'
    'yfMH(#5pGvkkCN%=odP`%mc@ry@F>6$D|0<+*~J3K{Y(eZL&N{m>+cLYw@k6<>GQlYw1vnvSHA8B(BGdxT#q?%D~QjI#3&s{b9u;'
    '&q4gyX+`6eH8$f9N_?4>wI`{%7zQTjHZybvOK=o*n^+sVlu{<1b^2ng0#baT!{C|PT$XXD8Lb^j0cV4xz)>)@i&rJTxWeT%8a0DF'
    '7$^J$LT_gl3Ckm4+Shb}xf${MYdeH|`IKjgfMwJgS$}jOz#ZW4ILbg5To@R7X90Sa3r>Zk@M2>*hHqqJe3)RiFu<D?WbaQNm}IUw'
    'He%jFPsO={6BeoHy2Aq_m8;;WBA8Ms^Z%nFW$&5E2T-YjcLe1WB;Yzgg)pDzypF1_Ak#C-igb#RxP8gAIJq63$hh$C1rl+^+YHqT'
    'lWecNL?^2*Y?cmea8OlouD*cWNcS3hKsqk=)kn7*=)~r_G^lOLrb-seN}ttVY$WJ%GfYhci-K)i2j&Ey0%525<K%%m%r%2%egf`Z'
    'kPlj?e#|XZ9&mL@oey^N=m<-fk-pjH!I4d5WI@lDiUz=gxddAomF|F<4}Y5zVVVX)c>!xUyd049Vv6o|yuY290l<35_s(~px;AT;'
    '?njNYsxXKC3yc=IR}XO**6_Ra!+ctb;!U7-_O)X@NZM)PW0>0U3c_iCMT~%&Ci=L(2$IW?eLQIKjH0NrGZQ@<z_Q3$)a=}{cHEaE'
    'R_a~Dvx4kR)&PG{{bf}e2YD9w`dNeT0pE;8j*2=W3V^@kd$W|A8BjFw_+4KkXLdC_Ir9n4MBz%ax@6~6P4j)8m?|@w+7HWh1~iTl'
    '6b##jQzA^^#3KhAV9w+}l_rH21GbDuyC{|>;8@hQ<;DFH&%O37tP!a;saY#=G)q?akiv0isfq+ZEwwJdl$FYiaxtp7>*h{gy$r)H'
    'OJWFacU@NW1DV6z-gV1S=og}&LLGZ#sJ3hYLH)>T`Y`vyslATdtTEBWq7f1yuM0Li{leltc%(wwjn*L|8F3}vt?1nvc4cukN1ARl'
    'pU7r4fr#5cXpoV56{C%T>fv0uHj_zCalG3sFRg$2J6624yx37{4C)dnmeA1fHO1g#Kzy&V{yA)Z<Zj@sa=1Eh#ia)$c`pp1jy6=5'
    'GCGW|JStQ{(e*-_O_1sr7d``A6OyK;p#gAexM*Z%?&mzZmQ;j3@$JU-)$t(0+_j*d3eqN?jQOLE@bzxU|0=I{gXRqtyOi>4mqHy*'
    'I^TxdKBq(&2g~R#VTWdibq^cRxc$p1m(YNYO6tCp=4w4aYHE~`4!)PCeuBS11ER1bWS7u)%<<<c7G{dHv1}{sCUewtTMi|0rsW6C'
    '*r3t@K<J6K6woQzm;$9AXh6sxHjF$eWPMd@>1HiaRqnlZh2OBX(9i&;3-AdD4+V#0)Z|k>P$EEO<w+2u$4TWzcX6KnC^Y9$CrJk3'
    '7-wfgF2fg|WHu4xs2pFM++B`v&EyE2p*^tbvQn42W;+6x;fq#Naqp|5i{>wRUGgZS9b|Z<Rtl5}i4RAxQX&`fqC<-8)we(57lwGt'
    'Gr`0@$G%Eho0X`3*rI(OmX~65(OpllGC+op;{fve)9wyqy-u{KXfYl}Ki|Lf`_sM;V`yB)6Idw30S9EKK?m@vrQV|X{2{tweJ2O`'
    'xEko<RjHga99co{vRkJ}JROZ2CKqg<V0r^lsf_8~Q<kY$;rJqJ`w}d|t}2<7N9Y)jkD4eqb=h1ddNMW*#!3bcDYxrh1<)1}OHEq0'
    'GzkMZshbqs=D`m`T-?=;X5RLNTAU~msd4Ow1vs?v0yYqgSxP3Pek6o}RFl6pT^B32nF^7DIo0wHp~uENdmiT8h}l8!T~hZSQE*Qb'
    'VQDPqA+xVRNnF+`nWhF<&7i#^EHVze34U3U-)6<s+gTzd7A)*2HY=OR`VHnpw715f-azj*hTq*D)QNgL{ZXq$4L+co!v=v%7H4r('
    'DMe$U;C_mVFr+N_GK4EAtog@o4FTh*_gR+#_+W?BqF{<ME=P@8U;~8!(%O>Npd&&&UVZX7G*Tc*GP|gP*AS+gh+qL4d{peDEuSD;'
    'AftGO?fe<H()eD(4wZV`_ugfggmsJvE;pjL#%u?30;8t^+6ULgz!<T1O7W&0&yzu%gD_(aGvMXa6!`_27kh#kHr^C*nN7>Z#+IUy'
    'xS<E-!&DbX3gQW_C$gLw>wyJVd=zK_H7=(^qAv3PH+veE!v11lA3FymJhUZqNl!TwcMSj!&o{$2psAH@97-f{OOe(u{la-rRA)Ip'
    'MPg%E@1JE+a{9FQ+%&~wI;pl%UXXt(KujlFk?v|gs2tIz3}ES5){kk8k0o?YDD*zAJ9}&DEaLI1a)51X*;43*s99z?RJoqU%?OhW'
    '(%Bintt5eK7F-Vryh3C^HcpENc8{WMvU6r4o$HG|iSRmb4BAl1*G*(L7L8;VKqi;N$iWfQ2i74o0b`a@)O5P)fnf*0yjaKO3mHc@'
    '(Z?F?u%zhGVEiBlWz`f-KsnR6zKg1Ba%K4U6FNZFl{~Gf()49uT2T9CyktjY5Z<t9q5r^r01?7#LUV}8Uk>XXisFu_uO!1TprFd@'
    'MU<$7e93V;HL%0!8s@8b(LN<hDglKiJFc5Lz=WkM&|57_nI70UoquZ&LW;6rb~wG9Zv3x*@p_83X1yHlZIxdlxH^j9Uf0qjs>C8X'
    '$h70QigD1?&Q4niZI4EkSUo<Uhq{D4WuLc|oomj6>Oi@LBg;=9Q5C$+5qpNV)R{~pVdYE9$}uB49G;_|o6FHuBy*s=soy42wIAS8'
    '5>z8b5kqEYZ!IxjE<NatSJNrwiA=7}0$St+9?TPJ5Toj&l$TsK7{jt#i+*p}dL@=@RIeBIAm`PJjA2VNi-rya0#;7ivG{vJOyWI('
    '%^z<}qX0+E9o{vp-N=!j*$(q@;{^lyh?L2|f0-Vy0*aES?Q9p`HA%piJ`pe0?gFU6;6#-O$hjTb*D1@r=t<2bWm6*XOUY$P-AA{2'
    'yf}pIQS{bksSKiGY_=?@TYv4-JqsN2sgqXoUZ)`HOK&S5B~P&2ZcRyX!iB7Q+2j8y(hF?)Kx$xd)3sp}Y8-LmNTSgdmz6}_=ZkAx'
    '6l*HhWCO*Aw(e$BbdupuKh6i*oKpg@bcLhO)VT^1mzdHYoHg-stax*Cy-F?XdI^AC<W}+GRb!vb7pAM)F>v0L83PyQ4m9{Ma$|EC'
    'iG9jPy)!C+>1u*p@f;lbGuniLGOWx=rY-HUidBn{$2$tlbF2QSC9N0Pp2v+JqUFXn&XR$Qruk61$d$RO*6+D5*ikENX7Ek>0iusb'
    'sUU|$728!A84D>}RqPxuR@-k~nm}={(Zgr$NA*7uVAiDlaPv7TW!B6|C3sV#H@z%L!DI*UE6S%5$K|Z?2nJ8ziiCxt^er!L=AC=K'
    'Tz?t=<Q1dhtcy5;Kd|zZO#;J69{H@9Z9@PQ<Qz^!M~P4(M6<HW?rQWf1cRX}$6oDTg2rHC496|ciHN3#k98Z+bQ{M8Hf~u5ImAkN'
    '^qcuWzFh`C3fp4x+-0#vi8Kh5AgE}l={|-XW<IRq<B3>5U7*ZTZ^S1QHlE>rq6$Eshs;b_B^Kw7NlLI0z7Lz%ufx>LR#s^M*y)B>'
    'Cv!xJeSO-W!!V#l@VAPg@)L2X-}(Jv--ofg9Q_HKH-y?GS%G)(zD<_%&|J{=etjM};uF{b(zRxLUqQD8;U^!JCf=x>IVvu|?oE%u'
    '>b(9rBc{^XnH8X{JKK#fj@s<Ewt3iz5dfX{0MSFGQ|%(pplYiu1b#i6YRk57C~qLI*iOz5NGFtkadS>YjVC%T4T4fGCkV))?S2ux'
    'Wjxq<i3g9llYCjq$=PP@3I)@&e!?liw-J>&p8MqE1U<Gy(3l*aWesT7h<q@fnGe?c!mouPVY`vNPxwX-uZ_Td1lr$8okfJ#eysMB'
    '8OSO?rUc!ds0JE^U=}_P6WBUW$}6fLF>vhUP8s4xQ3F>b3@+juOl<-Kq=KM3S$Rrj8d6ai7x(yDb7RC7Oy1OGl($ZJ4Kr%?PE1w$'
    'nZz+7!y}Uz)XmlF7bQliM+T!2P3Z)G3(nIDyB2m+{iGnn9n9-;C8$@h7x|hS^)7SEjC}XW`DK<p^#;*AiOawO8)=aNl3d*O#X$#F'
    'u)33xo``k_+LWh8J`5C3Y%3tLMPIz=S1q3yg^Teefn`85-cJf4pdD-y!9mZHMZcufir7bdKYQ#hR(d5`(hQ8cT0eO-D@Kn)FXr4?'
    'Nw1@1P0b!zjuhwTlMxI-FT9lJ%@UYgS-S0^BE{vCqu^h(rSOr;$*zt0kwK(oHI>RW&<&zUfkTn@Lp_<%sr~OFIHhHJdHK!{7z6rr'
    '6I-=8-Z~RzR)mu|NeB0qPE>^jp_k~Qlycud=WRr{S|LY5{;6l;Y%Rj4(*ZycnfZiX7d0v?{IZB1{UY42E_McTUwKWR_gTbgBb3df'
    'dp#)#B6oDQ%A}f9yIrJ60}xucHyd!*pYINYwcJ=6<)nKjnHK6iX0yXDqLkavQt9H6*_k{Aq;01ugK$tzUQbzH18k)Vm?d4L2O0^t'
    'jv?B;rxQVGhzB{38+k6JvpOtVU0lsI2fhk~+gZKJN)n}+d@{CV8Pp3k8dTDgLQVybjoX;t`!$|Wq%J$KY9Xo48Vc$AiuZ+vrk&iJ'
    '1=SJ;%g+n$=XVq=FnZSlqn`;<0Y=O6JWAlJ&z?~={Z+N3r4wBiSXJ^~k;mN=CyBr-8rvvT68#Ln1W)H?_+0DBzy;)(6N^N^w?5A+'
    'Ym*iqj>As`4y#(x8n$jQhxZMxeyH{15eI(Fc{bj{pkeT*o+l!L%y_59H8o#<Ov6UM$?}mb;BQ?_I}2)DA{ZKsvqgebQEW*kr(E7)'
    'jlu>MplF1ydd6xNL`!lZKvYiH4{RPK)QwwSxyZRM^aXJna)6!qf3ZDHn7#l)Ea`MjB$<u^a8K-{?HZ*oRK!}Sovd0i3}vk!MC^KO'
    '2Ww=r0#V1#b_=kBD$4~H5$yzXue!6t2`Xp4c=60Mm`8}Ymy-qrkGdh*wDD+RVbu9psWaZGCnH=mo%5)0e%eWu+L6r8vWq`SC-9IO'
    'qS|eZ+Md=#Ku?R74On13ndQoTO8N&3lger0fY${|y|-($fY|OZ)~cv=oD7iI`T+kc@?yIFU_!v{A|g`qVsD-Bnz2z%Lukk5?R<|?'
    'rBV!6@GJ&gYmZWvA8}#Ze=>AMFJcU-!{GsYHhRb71Irh=S)_|h9#8~fG)rsRj?9XCCQ3G7*bp4gmIl+7g_&d%=Gjp~ItpW<aP`Dq'
    'fSk^BA=9&*>@C?Q$^Q%VX|89hg05P138HOn{=RfUTsAKYwR*O8n350%b5iqzL20Oz%`QM0SiF;^KdL=emw-ba<Ge|dq^@jR&ZOmZ'
    'HYqQEo##E_Vmw({M&fqnKElo`F#rZXDh2L+Puo6c{lXX4d5sw_S6xnnRAUiE`pHq5;--rVZ?DFJ|KD`T<Yh2H8#N#8t+QaXMlkUy'
    ')}u4GGbL1}5(6Jm-j?}lxrP5*?aZ(|-Y@2yo=<{j>+Bq*mgdVpat|NL$f;KmJX6lsC;}P{Ooyccio5j_KtavRqJY7Roh+qM%`K>D'
    'i?`V>GJ|i8RxfdLq-l1x0HXNEi$Ua@lR%1SSb~qub*S?q!pAMCIj9TvaoQyyJPEiQHUDa7*8w_mUgR3T$Y$37`T1N3`&my^wJL;2'
    'X?ro9m?zT-Zrp6lXuK}zzH4kzIto%UafwKC^3~D5X_B3!Vk+mu^<E%Xz^l4O9<*d=R1D$v`^oc(Rijs+yq)P%J;Qm}a>Nmrp|^|l'
    '5{99ldWlp=J}pUY+(e<1oiI5IX7=*%bYzcCz(rtUh6KbQHK*rY1&acbCIJoAcDgBwH;W#jdE|_<^GwL{<V9&R_p^yAvU@jc?(9zl'
    '+~fAv40<MC!Tae>tc|%k+ga-Mi@1uUMj^f<)6vSwW&ipiK{B4k@eU)Uc&#MVQ-HOj0p_hhxADW8$OZlEvI)Th)YR2w8C%<hX)J(~'
    'z<(DxyCC8K1!c+HmHD3iLqlWAjM=o5y(|I@{R6so@&Y|fVI>bgZ%QXOuwO>ajkYhBVHbWtqxfgaQDvP0_tVpwOrVt9?Jzt;sWW2a'
    ')c9ehwnRa^+tR8~AMy8YCORtVE;~3M^~=m^XfPw4R%<VQcpY){Drwasup={<WmauJcOFzRE=CfbLejgaq6ju=i?)8ikVql!rfFkb'
    ')6Ac8D2kGB9@aW=h8RyJjMm~`Q8J+ui_1Yj+z57ELlPVaQ^2~EbRG(Ka@>7P0m=7v#;nX<vP_I+<QPx%M8W%zMG*+JE?q!8hPuS('
    'XLOW^H0PrO6B5pS$+%}C9lV}~p%kV5IG!%oAUas&yIE)<ErT{gB=hcvgCVdYK?3#PlZZMKh@?d29z@H9D2_}sC~pw7@@OPtMEx)i'
    'o5P8qL_AU~+?qKaQ-;4Dbn9{`J5lIEK4}nEQ(=JzPY%RxJM4F{3LSjdOdToph*Y@7Owi{49x{OlWAO@8Yw{>N^Y}PP)CCUpePfiJ'
    'wP>v9Z(yOoqigum9+qA%<-A53bcpIku_o09j6`27TnxsPZ7BDqW6%@dZPA;4>PXaOoP`E4I_;HVJo}gg(e}FOT!vVb9S~1PUq)8w'
    'dST$@@3|D~Ao$qm^~_4LVpSHmtaVNk^zA{`01p|6mPOi}ESsP%A~7nTmC~<Uct+3#yM~aaB(<9)Eu?P8pe1hM-`^bsuM>0`vyC#B'
    '0R5z9MT)(oLxhK$(@nB)Sy&QX8oVjkXn1`wPRIe!p-0GUrekM<3sSAf-AU9T9fc^t*c%FnV_9Y}Q8cL~L?kcsfc;(N5;_|2=JWCy'
    'N2`wEPCByVq3sxyWNNC4JFe`BMqZtnrS|Wy77p5lmbk|G;L-<PbRxNme1+R#?J4^O%UQ^J7c`dj5)Rx^&>wL@8owe31#q!*ab-{e'
    '<r5VkJIq09^lu~gWuxozd8!hoHwozFe$*FfX?fTb`8H8xO-y&%;yB4B>@hGZuupoZF;3GcJaLh#z`t#d{6oA@^1LV~O1~cdHZ)&>'
    'Gsl63P?n%ZoCb-IqJe7DrH2+-bR5ljc=2Y|`kL>B_!!i0jo2Fh@CjH734MXQK?kVxxn!>M&7HZl1B?c2YLQqJw-^N-@VJ>N3rJLX'
    'Vf}qPU6gFwCh-5I{dd!nkx=DpcQClBB>7!oR3qOIsL(i6WS!vwMTe_O2Kb3T*pM(FsGrf7M0ONL7nZnlyBTH~53fw2C`{`Ty|b$z'
    '-v9qtI>~YudEerG6TxdV8uQkoOvgy5AGN<EDDINRim(;|0k@o9m+{J+R~9Nv!Y!b|fV1t0k_{SC!B5<g#XRy<2ju{(qK>!*vg*!8'
    '3dIzwpwf<hQ;Fck1SB5uh74@9VmT>Hp|<_neQ1=a9mSzeJ4E32A<5Cm@Nkt5=Z*v(($x6Fpwm=fM?ioO%_<*tp?G@eUZtYN2pO9e'
    '6!err^NYtj5u79B#9731inwUpZ*;<^`I06ahqjUXXBf|;Pa9Rq5PW6;vt=AI^I=|01(ESws*QCrF9SpgrXXA{OY^l|S%-dwm2-<^'
    'S|s|BQL5{4{(y##)GXYQZmQ`6#X{KVpjB*bPmigmGVu~_A754|MMcn$mGob&4}m~XXQto?%0i{)oJZyMig$k4S79&KWVmXSfjrAy'
    '(D<QLoQ?%@X%gg@wq;Wp5Ko&bJ)!3O#e?bS$(Tmrd4TL-nc~t6^jWJ`Ge?Cx;Z?Is)7W&+c?fidc*P#e6#PmG0U}ZEXZ3&1CYyQK'
    'oG;!$*g$)eAcs+F+!OpwJqlPs-Rhi2H{3W0mxBFdXg0*mG?_j!*F3a_2rO8-Ii9!(ezqk2C2BNYZJvwbaK*}7uIv~A#rK2NMAFWs'
    'jUP3Bv9yHVge1u0R$LSPet3w1N+)Hi^_Y2eC>7rcM%v{#ax8$HJ9OUc^qYJvK`1?uPUZ#K8f+%yl`F?`7@SDfV-I=+Gm`ZZ=utv|'
    'LE!{s67AR$_#a38rJ~soZf#6YB>#_$d10=NZ$LW~#cGa;F!Aq_Fo+XzOlthdJZgjg{e~Y{1>Wa479g5ZG*l<t6}95WOVYtrN%X&0'
    'N%%sJKnr3?YhBo$B2nZ)B3xF2Q;+;|tYC+DoYgbmh*xJtz@E~w#*E%=uHi(l7d}@4ACkPrD4Fk$N1p5rkfpTaTIu^RL-4{h<w84O'
    'Ej)noW0w5-v_FSoUL&XZ1Qe$JkiyjO{Qj`7qj+kRUH<u)hZ-LDbV79(<)T8pkWn-u1D_G*>~2oBERMPLxFWHhGGWnpbf}EuwQ1L!'
    '_JTxF&cc^NCe=~q$dK!%-b!qF1ElTG@O?ANvT<udqGy9;6R-3vw~`%~fp#Yp$OAT+$BNmE>bA5JyDY~gK(r9b@%&zM=b&Lbj-DNM'
    '6OS+sUBIkR36fW*Mv3>pD=O8FQG{$jnaNB9cYPUb2<k~f(j4>8CNoMotzgy3Gu%Ke2C*%K39ZM4LN=zqaOTu+HkT4OWDWyvYu4q1'
    'QMydWJHxkr2QYUmG&RK08!X*vksQQ#<UH~G%y)$_A2O#e6N<u2mzh?QQ7V${E#j4kkmIlQYr}pb{<}?nj$3%r&Q?l0LYUpnJ#{Wq'
    'TuE%)C@-OSM4-$JqdWQ&54hG@=8yOfaRyG=Mb#?N;^fOoZ^E~}+qf~18-vX`r5Y0r&N?CNt_Pr1c|6=pk?;UwfWp`STos~D+^Hei'
    'YeROk_#I|*mQ_X#*5Pcn!S2?~RcfnKU0aH~O>Xq7ja_bH-_#&F@F%-nmO1h7#sijyQGZpC#zWHV83L&z_W*|yh;?hLA-<z%=EC=i'
    'P|MZ}Q7_&|jlGM$pG>9i=$g@gyS(jMJXbO`jN8oW0n8vo-Fx%B#`VtZhbwccCVtLj*{v+P42u437UEXKxd<arVE$sUJ0;3Fqhf~n'
    'YWH?45|Ch~)-fP1A9A3ba+=FH`<VY-Xv3;nCUu27x-5R=_Ldo<GdjyJq@Kn1YPfw!!)$S6BZ6Q|9dVRR4O&fi_(oLV<UbevBULqC'
    '>I@U&sHpyBnUJpOi2_{;R|Wu-ngBR6PBwX{ku%QZ=kEtq15mIYNLcWo6F`6B_8TFKHn1)Gbi3O!Aqx|6W+@mpSy~l^n&o!Upi}?{'
    '&UR5hK4Vb_=nPz*Gm|vRznc1O;ftwu#%HzM4Si)CTqOxqF%Sl$fqFuPR3P+e!xHZ9-xOz#f>|7oOY2JtNe%~-g|U_8h4)H552p35'
    '!Jn^~f+@$cx~mg40b5#>cW-f>rV&2ZD$;S*u&nBRN7Tw?$=HEBw7cmMA8*T}ylN)`^oI=s3xERT48hJa@2?*Y5lmEBsFs>n|J39Y'
    '(wE)@6-=A|c#O)GkfRtwQQALQ)DA6hji~@Sa`SHE0a&6+mvaKu<=l9`fQ-;D=g9&jYkqQ9mrm(g7J(x=3{Hpz@t%z(dAZzl)QZ)+'
    'kpra_t59?bnA*!NOol>5Mruu^VY>(2%bcdlV+R>1Tjn&{*JsIV-<<XGc!1skZ-vr2%l*P~6PeEr=!3gCTuqq&t?@>PnY^K_hLLGq'
    'hB8eTK<>Q^kLtbgbx|^c#e^1#hG|9mphNB}V;M1YO8@%TmFUhJn$a@yMYCZOi-bmzJ}er6|Jb-dW53LI$0Y~L^sEvY_jv+D7X?>n'
    '@cg&Fy&k_fy@bp^6S9farUnN@B0=5n)lW5wZetnjOoq0+N@>b2kov1Fux))MOo?;OYAARJ3s#A8B_^*JtN*+-U98^$hIGy=Bowgl'
    '>xxxC$s!0FsPZB&&1=HS=`o8a*_htSnK90B;0`6t7N`A8SdirRTd5;UL(+&Y34{h%=|K!<&(@oocPyWs1W@lLX;=i|LQ|w~Y;uQ;'
    '2YSs|2b3vZ_Xri&yd(qLUz{{MU_byZo$EmH)JQ5(ZXwlkH(Ns;Q_XF|4T&)-LZ^uu)4~0)9I&dn;=J237D(<*RJ(+|unc8cX_I&i'
    'FC_>}^M!V5NV4{XU=U=8)NC<4X=~U%b|(9btWv)VwiH<z4?L-6R|;;1HZzqQgpe@~DdL_k1|i#-?Nx14sJlQhB2-DnPBdCYBK<FH'
    'HX1s%SRS*4Oyn&mmB-0w0d0l*Ouy9bAkGZ%Cb-kY;*vURN-=w`EK4=PQf?%#e-X5ub{aBpra2XUPN>{Y9e^1%mf&aRAkR`vU2Yc8'
    '>+~1qkI0VEhjhe(;9Dw--!rr00dZB}jdaHX#>Wj==~&W(N%$+`^J0)c0CWxvtK@@A0Yzcfq_{rTxc_zSNj$Aw1H<bo%K@Gd6Hh-E'
    'BMCCe!v2oeUbUOhBQe+OK&3t5rK*@>ULSFD>3QA*>EgXBI_Cm^jQT)iPn88lKDju1y^v|+a?Um;+j2x6!rdC<6&{J%BqWtq`+C2y'
    '6}Co>eJ!oY@EDYk6{g5g&0FoRmZnagmAnhcm9m(6(MU)V#h0ce@TgZ3tg)ReYC9=_2RI}FJ^>hIrcrcM2w&bSo)zJm=to@L*K*;`'
    '%vfmwMw1CO%<4Og+H$34EI!IYd8-y}zLY<+E$ifw+IV623S~G-d~hpjc2EmwE~*_}59$uc9NsT5K*-jj-DLqtThw2T>8lSKY_Oq2'
    '-OKZFM9HFh(IBzpjqFvV&CkL`#dd?Kb9Xh=tWZdEZ4>!NFxam7h_bPX?C1Yl-^?(A27Q2(M^qnE7&d^mwC;|wFFkEk+Vig#OaLG$'
    '$Ko+fEbUg8muEIjFF=UhE&2rkSS!r5f~a~DEtEj`>!nhIdN)td=hCL3bzpXD*93w9M$GkduqXeRiChsIDk>YI>;_gYWd@<w8+^U>'
    'oXm26zAoCr3M#^g(sUMm`H3J!)fy%byXI_=RkeIZ(UWf=k|-^vGiqTBUj<%8=IyR><Xr0lqC@jhgDvn^SJsFKos~6azvvE_w}3Si'
    'N^c#ALsbW&aV}H=0s&e8cK14fW8@{MyyQdkKVEzOBIpD{34F>O?xD)1oR>j}e_O^pU4@FKtjvuwsKYyvE|noGO+|2$#DAfHlYdLx'
    '8g|p5-Q#Qj+l$Mb(}A>A>At4r`dV4Q6ls6bd-*}1>(8-S5<^*WrOJb#u`cA4Yh3klr?&I`xqps@$LJbnwWS?jgigBSy#DzP-=F*E'
    '*m$UgOl4X&%ijZ8ec$iz@cp^p$4Ux(KA^?MP)jm`utF3>DBS>qEWq^N0h-lbf%i94>#{||)QypV)$8sRaIA%(P7ecd1%_Pos{EPf'
    '+YKqcVuQA~rt*-!3HXHc)nQW412%+$a2RA9lQM8!GWbLYtR-u@@ZDn~YtU_F&m^%`938SR(}tD~0+K2OnMaeD7hk1<lW_6g$FVGi'
    '2vW+;*f`_$8`n1Fx@h42)!QMy^O}3IzxE|#Y8pE-HBAT2UG8u6(S9SdQ^<f{7nm6!Y&)R)bka1}4DT{|0giuWcqfJMqiC^&rMu7r'
    'JtvXg-J-H!Dv%DP50ruu0dCY~o-vS@h&~bQo;_%A%eO78aB_xlP9G4Zfe2_!A%`8P*BQSL^rZQz@dm;F8RXE&Tjc=rC%_BZA-wD4'
    'ObTQD2);HCkb8uxS$<#{jm}<2g*^f)o!t>Z4`&h*N@H-yNbv?q1q)&IF;@mWLmIl;nKe+Cty%#GG0LG)<^_om0mfqhmPNR;#J^dy'
    'b~{YSqBj#fBC2wUZ+BqQz#r`Jv5iVOo<-VH&!X>lmOo%0o?%@j?SaToF35|02;Vp$0bv&Y_Z$hET0o3o8N@Y!S_4KRqN5*Pvgk$('
    'NPs~Th<_6Ssx#hV5b|nEbb@8v4mp7KrLfbR986m}Kso_H3INJ*m!euS6GA%leql2NniyW;sv$delTq_Z!R@&oi4VPPO8}PfQQQs4'
    '?MxJ-pJ{4~Q$Eo6n1DqMSQb1GelB%)YOIei6zJDgU!X@Z_J0gP23ohZGdRSmi5Xx%qy}b&qtn7L767p)(<T7x3{BFZNEKNhaV9Mh'
    'Q3qfU(BZ)_D;oL5%8Iq+<*Zr#CLqW9VA6!M7isOF%3fu8OZ&&{6u~8i!v&q)BFx@_prZyudZFSx((FUACh-d3k&X-roQGWG5BxQN'
    '3PKFVNrahE^xgk{s7fa#1(C99ghf~Dm98{$WA;PM2FW~WQn;XMra4d*3&gN&KAa+fn*2>eYr>2#6R4t%{hSAjWMMt3wswsS($%cB'
    '+7`xR9?6qZJRsV#u1>O7iaIGVwMB+I&j(Q|z)A@Az_;kk5sPZoFX2r)bTi97wk%l#ZZWS!uTmPv#OKH1Oo!S{C<;JFVyHfrq(ilI'
    '<b%(s>wx0|z0!<v0xc5k0hZDe<|+UnS{0s{+uq75`qkr=H6!_{V2Q@SkYtT(cAlAEF!(#JQ)sfaaN?+`SEJV(;22D$z1Z*pz4NHL'
    'mtbXKG6XVOEemUUVLkso9^}PbRHnuZULtO@t-jE%Y^;rI%tGOSg2o$6jT?w%369P7p1M6F<6RLASf4_)wFNZ5G^WFF$;Z$q79-<r'
    '6gSeEfEK9BD^gxqMX`0Zfe{^B6OlE`!6V^Q)H1SDhgRDSTrphdsO+SO;;c0QWk>z+Q|Lxyef-C@HEkrwhzcX+<;C4!i7Qw<=CTL|'
    '4pocyQ0rswE=`{w<2)#=PC+gzK2)$G1-5z{^&e-eWYL<KN1_Eq=dLW;#>%2*FIK8_`Pa^_(@p@U&|Y>y#PRVc2Pqv<&Oi-xbkbN6'
    'Bm5-YqNs63KPaVfxrVQe@~F0yZlW|^Bk$f{-I!6V#mPflWqD`NPq7gPPEyA~4wgGN>W!tD%8fEK#M0C1ux|<?@%EE`3-ypg8xo*q'
    '^w&Q5?C~_V#?+;;<%Fx6l1hGXF^6WH4FuL?+_<7U=qweuxp1E5p&u8tkSUai`%MUM5R-H9jAl(QzV$5RDA%H0<tCc;<B9PCW7}rw'
    'Si<8-rimx}DdJa^#ky8|)(&xu(#5!3U?UrgNJ#96L$Z+^{$eS&ils@S&8s-EkJZTxHkGXLYC8BH2<?gxUHJkQ79Ia(5ReWH9A_Oj'
    'zH6K3P?b=SvbB+Q-l7*j{0=n8)!Yh7%?h`(3qoM6*`es<WJOb<dXKp6OG!)o!9-JZunN^Pp9G+09xPY@pe<;U;yxX+Dw|%DVo(>Z'
    'VCVuCC|N)QqQ3a|kBb_9?hEU?<dF%$239VG01@r*!UF$D=}V-rNQ;i8C~$@dtEnAl`U6Ac#oe#+_mor~l`sC6M~OEY-n<NU2kdbD'
    '`+=(ig@1a6lqHo)rLqwO-xbN{r&N2HnvVKo#xX?3PSUMekA{u3jIg@T>rU@PHlSl2eW~G*=98GNk=|!M#=?|VfMh&TKwau?X@ox*'
    'XS55?DZxXdhgyNk!Evr>$pCBdTn<Y*&<(P?1~%6b)05<&2pYu6_CrCBK?Go_#e+w^&3HE;v>m%|Ju>*@@W>S22_Bizb!CAeKA}W_'
    'cmscc4LZ;^0+t?L@J8b}#}Z1?x_-J?(*VX)+(@LQwI3Ce7GzqrSMVOF_Xs$OFIYKhL*Sc-nKsfLkz;)kd7z3i6dGwWOfhJUB%#r7'
    'C`8K4B#Y9=ATgz`AXETgB6tpKjOPms*LpoZ1>c7;2Nqlm$U$Fl+1Aogck%nvz7Es&e%gX^^y4)3>z96i+V^2amP7sookpm$x+Ks}'
    'CHU||`sT*l<dxzNl(0el&30{rV+egk@GItw9}u^)_qQBmtWw984kWT`bfQ&{+s5<`VS;#>hW_Gkn;7ueXxH|;(7aWnQ2-cX%0Z&b'
    'HaCm=DEhG{rW#F{=dro(X=8VZ&V56_SAU`2XH7lQ1)S-a6pkcye_DwtjnL+)Z13oR=%v44BcYMqFyEXn17ll%GDJnk!1Cr@)rM$@'
    '#6NgQ2C?ersv8xwD3$oLv*`9&AD7KCqG=Rc`b%4gfz~%SC-P|WxWD{Kt3Dy+ZG>@<5V-04&xEc%!d07%9WwwZC|cP&;)f_5+HTK1'
    '*(mX#Nn-#(&+<5G0F45*xf+JJOP##B(S>W2?vX3_6H%<D`AggIx#JE+dTn%bUGRtOn^(xWj?wk<CmQ+=0=GUVJc^Zv`C$Bc(?FOz'
    '(>X(CE`H9R(2b4$VzU>}7-m`COz__L0;KZ#c{>Pu3{4P2V9unyLEG)zy-#5Sd@PJW?d96wISQ4bwJHIDb{miJ^0wL4D0~Kf5i|<D'
    'E~&4XH_hH+2H1k~x75}wAlbNfZ9${75Y7~Df;CobuG`!+%o44x`pYb~RM9-u-NhKon?v43n3Mnj&z@etF#^Qjur?K%ALa%i2MHAa'
    'JYi-NAg#wlI;*a&`AC~<Q$#9RB8Gj6W$!=NfvW&T%bOJ<bS`NMLB$+T62>dkiIhbAB}<}|DOuTwP_v6)6Ot;@`{LAkYHf4=kd7di'
    'p`!+%LawM|lZx4SYZ|7%Oj&i8h~%4YSQkdhj`0^qm_W<p^1yeBR&uHXm8PTi`Uw$`zERR5Ii@x9&JyE9@FL~ip-BJc_2-nnZM-|f'
    '#`bBL^~PzLqb=@?TnwIrb!D1GvAIN(iq`C7*A(!B0yI&75q%ym$J}`6$~(M`8=EPOHRo*2U!T2TrqwJmJv1D_vcalv9#jO)hS`!w'
    '8<68rs2$C&z4ne!hRfVse*u;TVt0Cf-zK-Ad<Mh>`{pU^>yTi_Cc2>+)%B-WMn^C^Y+k#^^*Ec^SW^$=q@*)n7iZ-s2N<{(tY@g3'
    '+G^^B-H3fLvl9f1*<HGclGc`AFQjE8Ru(jDW+m2Q(lXdEmzcIC*#X2~<Y3GuI$5`EW*aPF_{)TuEP`{nfYF3W$Pk7yt01Bdw{L4p'
    '9d%WNS(I%|Ji)HH;SVU-_dFW$NqH){mbM*E;p+jpQjhAOqJcaE@`sU|HzQl1k{L7(e<{+!Mam^_6ow$2CgPZwr!L;5#JJ`K77y4u'
    ';fI7)I?Pw(6<XVcC}b}1GU6{$Os-XPtD7Mt%QRWq@fU|`B&?lZAdgna7Qjd9VmmA??3v20HY8|!*Z4Yh{Uuu8x$#Ptf_D~?+}XO9'
    'Wh0Zy>thyBrv!AGzZ5x{AzacoyIYW<eIkD+H%(iT0;O)IZCO4d%%NSAt|=CDEva0hQA(^U_)95jlo&~D)_;Ii3rctT3+`rGk=sTq'
    '1Bz-FUbjO81yo2#9Gm9`D1}i7i{)(+pdu1JNv4V2U<KdHUmTk(_^6|7$U>C)4bL<F($|G3e@Wy0Ev(xX4_b2nDv`$qN-3p(LZ}3+'
    'Fz3DjW}g$?Nk6SBOfB@(T6XSkep=c*a;uVr7~Q{L+oPG3nao%2%s}tSoyj*DN>LaA4E=@aC{Z|dyZB@$L1x$P_r~3jzR!+(&!D3&'
    'LOO5ST!W6>HiQ2q5vWl>Mr)-AsPCI`5iPF`X<FtbIkh1APg3M!IgRA>4Uj2-oA1j;0ujq@NOH32b9prPEiX0+;m0aUzJwJwj<TJH'
    'M3`Iikl~Vb6iPmC#|oKQyFmTY5+||L2f{2g$g%!4$Vp3<!du<dSmo9%Xz4tbV0u5gxU6%8tI0YZ2D@vEZ>v4~IrACIA(g)jZIoex'
    '?=Hu`*&$-0tsNrqnZ<S%;l{Uz-j2Wpr!6beY<+j=?K8fQK&aG9eSP^IiOL4=Gx}HcWN{3Cfj>N#difCmFm#y&Il3zEJIeUdpz1>v'
    'BzpKKgI_k&;?$=Bg=}2^76GUeB?dV7IFTP`+*Zh)8o-_TrcVq^Gop~6?iaDgk*@*vjrj{xw!dwXzLJ!IkCQqca5mH@%Efvj58i1A'
    'vR1&eoN#Q>88+r(nx{N=E{Uo?SsR_EYV}0MGCCl124I(**bXn?Y2i=nC$m#v#LS#EP(A?=0lFLUeYC%LaLeDAlp@!U#>dII0`rHn'
    '9e<tzoMkZ}-#E0M#+f#L#RCdi;_}47BxQV$rx#0scJgxKO6D0xdr2ocEz1Z`n6I4DHCDTt;1?MP_?i3>NcaZ)AAC!$Z;&LQPmSfh'
    'l=53AJV0uNl`_<mUQRS4MI9*e8UnPM{QCE3kjvQmP8{ocJL#km<7hW+MLRi)IY1%KXZa0|YdoQ__>z#INN<mmcZ<=$L<oUd!7L}W'
    '3jNLOwx&d8y`DS;jGaKJG4Rh%<j0BzA~Qd6hE9~eB=0!;l7#Hz>{Rp8PnBGIKhc+wmb_%y@dvCY`2v?F+5K>3+#X!3GltW^jxp{$'
    '@f>FGH0Sb_&wg@C#Pg_fr%P)DzCRn7FK8pNvuN#kQz9JW&2KXH(FC77eoVQ*B4nA^og~4G*b8k4Z0_ERU8F!Ybjap{f!YZp1*59Y'
    'juBe7mp3q{0R=iGrN#iu+Ix3@-v5WXbIo!i*Rl0fJc3*VN$@fMiCuh~vMWnQWUt$ur40S*(`8v!L@-ETV?Rieu=OimB$}LyEfot*'
    'M#2gU9{v6tx=wJ!bP8MWQm+@~BjVb60;T<8V+2xcFtPv{i<Nsk@D3uR6e}`8MqihXAtr5AeOP(qpQ5WnU0V^3mV3}TJHEtDj#wf4'
    'V_vMxfg|j}^6I=uf{}mbdPqw9ZufYEgtV`M%!^@Vy(qH4^Xim~OFKi(ybvHGXC6^fy#ulMv<>(<F}8_ntBs3S2SwU!`{6sYE|#CP'
    '%=2+p?>27I>@kHSD1*6N-~Hm0+W>~`ko8u(ItxjJ=!&%@4P)P9W_8F*p!EKQumXDVSX?2N#C~;wO58S?kB)T{>85hziyuS!ezC47'
    'HrrHbiW!An_KZbYtx_&rz%oFMaWWmxoy*Gpzq!(=B4KdAnJ(u2vUF~6C{#L^3zoE;gV1bZIb`Dse{?KV>%&gHyzq&`=k?+zI-uIs'
    'r3&hk$?9A-?%(5Mc-xd^2p`}=DyX(bgMTsvO25CyG5$1hIZH++?P`<3{9{!GM3j^1eMO?KiWZUYU%l9i5SzOed)f7M=nibLA?Bba'
    '4i{+wJm-hVHeR!NAzINSk+dqQ5wF&-Xy0sEOmcnKi&hwM*OKJ{V}Ngd7vLThWz3_Q-hTDuqQdD1)M(}EC?%Ko5tmozy1Hf@qw3+j'
    'dwb!`M&=tsEV&(b`=X)cQ(Z}Mn05AY4aw}ik8q{daSs%ZH^=Heb~CXyFLwUle`u29&->%2MvTW72p`OYI^F~OhGSpty>a^Ad|Y?G'
    'MuoPV&f0rC6o<O^#m*`-_AlT&*}C1x@Wf8k-}N%vezNAjxS(hUuBVO&gy$EQ03(h7{|^)}@S49!>z)1Bz+e6>@>9N8wD^#ZFaAF6'
    'oc<p`eaK(0zn3JXuPg$+oN)2WU#;M~*RD<+1-}Zx>ttK>_Qfm^kC6N9lMvy$*Dq##A9KCO&K(6`Z|{EqiSzTtSi+jzA>QPnkOsfM'
    '*jwZ<cQp(>ZGSoC?QJdzD#iXn$_=PE_7^>{@Q$y1a}iovZR^F=XgMtC>sjgiI{F-NV|_88#h1qB7n>*7UF|<$)gYrO{zXf`bad|a'
    ')iRtPZu_g>9;H;7<Hc3C0Vb?3Nqgfff-70tW$ybNNDv6e`oc}el(JCATkL<E^x5}6lse&W{R<_j;Dz-j*=8Nrv-yP}91oDuzu+KY'
    '4LL{m8<lZ*b$nGd084#kcE$%vt^G!qWTeD?@eo$@7hlN@7=n#ga%}!*O(8_#FANOfkh^`+-%0?x$Bn8XC~w`qdP+vsu)jPw_Z5@u'
    'XBr8T#q8_)&9C?Y=TIdN_eQ&+oSn;8>*^o`_9ar({YI^zGTQx%uvbk2zg%}0zDJ*#V*S5PSI(w2vw!ugx-!PU`q?-+esyCSl&Y6&'
    'W4wKNRpf40e~bD-aaDbNb%*E2^7kl)nD=%6f}q6Audn*mnWV($FP<%#3a{-jnCKPXUza;=eAVq@$uhq>tH*+={Xc9#|3ROE4%^lj'
    'AU2J9pI^cLa7+A)$rBT|=`RLdVE(WE)eRjI)_nyqG@zLq%`4cfzv?485sK?Qb_#TVz1V#FZZ_WRP0UY|8AZE%c>%)+iz)5?56JNr'
    'Ai-Ud=l7n&)`8^{aZ)30f4>6NMwFy3o4u|4(fePtl~oCX$IX^pv5Rtg!p_A(1oyUmH4BhA>!GJk^a1x+qa#|MLZS;4Z!Z@tJuf4E'
    'vXWy+pt#tv$(*kk>;vD<RWCR^OTH09hJSlr6e<KeS;0=GsMGo0My8R{*s!zd@`Sw5V7dbLs_MgP7u9YR@zYNld|wwvGUCR{J(V1m'
    'D@!a85XXRX$&hn>bL(H2bHpzwbR_Gtyfm!0iQe!lIY1F(XZ<39<Er^luxURz=jDY?y7C_)JY4Z<<4V^>YDGM>p&#ehlRJ{p$QCQN'
    'k@e%)#*Z4L<K*iIYnIwv%IN+E<;uo*8*gzWM*iLvOEUtuOO*3dp#=5HGJ-5dNT^$(hi^cCTb1PTzJ56-RPCY%jT)K&h}Z`^^67O^'
    '+^j-e1oG85Qu<xMw>Y^lfUNO)q2jxmFrt|HsJIMuZ`UN<9fgkK+z`%QZ23_UVC1~TaIXD@rK_r&g2c2f`5xzc^j;X4KC;)lT}b<4'
    'f5dTqD~-HeP@~o+({V;QrDWa<SUmRS<2YVb%zy86dBQqve7QoxF3zQiKc*~_Ft%+MhsZ>yQ8Cz}!@ueqhg?s{MM5F&uM=~OQjbI*'
    'B6;zPX`t^%QYvO!mul6h1fs~1y%yB9n+!BMMIu%OAbfLOOb9zEUlnuCT}b`Czc#husZa>4bXCFcEg<$rSgBNN1(jvLs%v;U1fLL>'
    '*505!Y))ct!!xj0`zZ6GFRB|?=uSux)|eOdm{F4!TsXM_!{<#Dnu{#vU4L=<Te&c6HU%^dq72uLP5VlBR9Mix$C*e_leygmz#uj%'
    'PKCLM(Y32zkE_U@m<LDkwy)e4%D^Hv?i69heiv&Ll2oU~nKltZUv;zbgdiTYctQxD#sw{IlbTFE3PpjUx9?G*GTLZ$L~|7D?5Y=!'
    'Yn;dKyA2W&E+j(O_)q0#Lf6hBW`1Uv74?K7tcQ^deL-&DW*=L!<3<&5(Y$h-rb580eYoA^XpAD9T(DKeV;c3Mx?Yqbq(<XbeO29H'
    '+d^_06ZL5~>9XPiqchwD-N#mcH58AY5MS4=Qnl_2i68v~tQ(Ah*A?=uB2N<Sx_Hhnw6PfgN<?$Jkr;C^UsJ2xW6PA2Z@oBWMn_>l'
    'S`kpzs8=18e{9m{l(670lusKEp(~YYLm9ppS1$Ri$F9T0M&<AZKsZ9f`lmSa3pV?}Yrfct?wGCJztWCc1n8+m(KVyloL86zE)J6<'
    ';vT(xnHPg<wq_gaj0^U@fVO9J$noxDQbru*3(I>$U>0TO_!Tuy9F$j;Xo+PUtua*^LxGDoXh%0+$(3*Mw6Cl~7hgpU#2vJ9p&ptz'
    'F*hQH5q!<|-PWK)E=Eb(dW*7cUV#$@=ND7Je7%^)NBc-%PZi8D%f$*`R9L9bi69tmuVZ0N5<@F>S;3U!;>pc8R?A*PpNi#pv5#V('
    '7JF!Xs(xgNQ$I4Va>+}E;Ii%)hvJAfV=#!rEkM3hx9^gbA~R01_?u9XuGNhV8<AK!b|B!Ze(KMk_UNao@B1Hw1o7_Ju@68KuLk`?'
    '7BSMW-Y0#|V!e`PFdzZlJX<u*wlN3;T2U=FAi*fg-WYze4UUVs4BUg-<AIGRfNM)4ww-)UJf?_D?Ze~}GVc8%?t+Rxt}0H<Xoc#K'
    'uE3ICJx~nys9&7;3rJfGq!m#1_iP+7#aQaA)Ivhj>Z95XHc2D5dPfyz@#A?&@i=!@;&(=$!lN*b!C1_s?MZru7H32p7V%|Kv{xJs'
    'NoKc_6nWJ5&UuOF+M1-D*Z61_6UOKdvd4z>0@;BYX@B1#+8R7iT={3G5-~X_?U~H7kuj{Tq>#8<i^!WeZHA@4o185C*c84ZV|X&I'
    '5@R+9-KDexs}C$@p=v}4oTrA$)C~Td8%U_6=HIuX@Sof~Zpjyb=$OLpjCTOl)Tj+khA1iQam{5Vxu9FCY>BZW?=qIqte)~O3xa=6'
    'zJf<C321Aq7r_Yx>kSdbmEsbo)W}Ac#%n{(WW<@kuEFKLW|s~x4WS}trbStn1WDf^5jEDM!YB8Duk=gTp1@mF^U#^5L-ZiZ@MHFy'
    '8=hdWOr*S`v`T}%5l7hjp1-v!EJzu|><uivNls+umQK5|m-k<z5m`W75ede3H&!={WEXpyRk^cFG9PeZNX7$3<WAg%_G^<gPH0od'
    'wX!5dRxAK&&GE~zv+h{BDOdIe4#-hA5OD3|K<TAo4-B89WPwp86PC7rQH@O(EorB5xPS#dCfiaRDOibM#qXYghQNvgrkqD}zu9qd'
    'X&m)M^Yu+CCaK_aj5x_?^>@dh&U45B?&JhIBP5rS#8-k?PfpTFmD>BEZi7XLT$nZbjhi~6^xI5x8ZN{wnPm(t8?5zTeLv`_l?>kR'
    'RPPMNL8Nqmw>QKmgtx`Ru!If~7Y|vMjyF!%Dgsos1oIhc59j>laThwFECd%suV@Lt9j_C)$z7&uJ!&0j7ycsIx%b$+*tsb?h3w+q'
    'UK28fM}aVKV5fxb=-r|%A;RnL)7oLutVSp;=fP^KD!+89cv+)}T{82FCUbhm59HDBL}N)iwH9%d40V8$pnvb?_g2|cUNiEm(&kX;'
    '4^&gr-}W$W8#QzEe{pRB^yp3q*4qh+8D$@}d^AO?^I&7~^(d&Dw|hfrbzwxm;dg<(u3iOG_c61e-=X+koEV9fK%_Zw^U1jRdsF|$'
    'z5;J?kXjH;(K*3?zRDu&sT#o?*rK%9R2G&A*=pTnJ=J)wb0P$fW=*52nsBE!h<#k~beM9`8I%%xgFJ)izQN(}iJl$*;JBPc-d1w@'
    'Z47Ze9@8X|E6NYCGUC=GA8l|_;#{6CXHM?Tz)yDE0Le0z9aW}}ht=oAw>S%Q-Y34f|KjGzUayIgo3Jt$+4GdxQA>RoS_k1Jb^~Rv'
    'x!Lh%!&)hNy#}p=QYP_g-qF+#qRilAF6>;f%%45kuAogsS-H>boiB@C3%~83Kagz1o<&U%ojld_rtJoUO(sZ_WD1_sz&N|dBWx5@'
    'neZ3&&f@kV&{57NiGX)dota3_jZC2e%?xs8X9SpuN+z#oYa%o3Nkst^(2|n7i<cL>UoTRcHH}ru9S$#}gk8$XF0rr1^$b0qnK4*K'
    'RJCPBiR=@b!pd0BNTEz>WgH`jLUwSkjfFc!eoe}1_?XRCoM3Uwd%Rbo{?1WegjO{o<0rBz17gSU#bcA6Sv=oRWMjO}r?H?0-b4f>'
    'xL!0BALwDA`z~nXn1M%^nSkL;Wvv~Ke@9ea^><zr-W5xoSh>}swpyg-Rk128OT>gxvB<>rvBBT$fB<vdyLz@FSFos(7P3McQyDE%'
    '@usp+phhnb$e!%$z3kqOiAHRITeN4VG>m7|6(pOW>PB|e<}Pf`Y{v5FMs{W<O}!m&c&vQFG*)cuG6l{$=}shB*ZW+B53M-m?M3xj'
    'JmQYeL>Jp33Q5#k96-tSZ+71jXVoPC0##unN;YwNTExgQr<*tjlV{C2UUpK+PFR^#IdN~8f<bj`N1v3D29)jY#A!}p4i(`0`#Ndb'
    '0u)hWl;iNnV-uz2?}NL36s-7BO_RgZIuw~?+61;VL5@b;KOZ{c3;qKuJsT&{Bmg0@WNh(w_IQy$7<gSAuN+j3T$4DONn}bKG5O>|'
    'OUWl^oCW%5?aAd2gmLJ|sCXy2!J^ab8<=@J0h;iPJFX*SWlkp}a-_`5Lb?dRRMT7;MB(lp5}g{0Z7TNh2H=Lz(XA}9{=ct?*Q|Q<'
    'xou4Bz6L2nt598A4%9C!zBepzL>ZK&<L1MNhO9#Tv^p1VSQA-#tkG4H<Mf};a3n<~7TX3C#xj${`hULW&qq43q5THV#$<L!hwt<U'
    '4j&PUYIY1W-;d|M2$y0HA`#B-{Yay!>+seaHBsyuz!`niw|x)INneKxfSSfvoa{a@hAbJ0qn^nZRFp+EKpM(g``#g-(Xdc%P?=Hu'
    '+Bu6Y0jC@i4jovdbn_@?2wg5oC)V0=$BPC25#7>Bo$hu1d`H_U*{H)>BT7<XLrkKQ4!-H>K}6vUR0Iw|3E3y3rmKUA4^V}iI+z3|'
    '&dCT;%^-Y8+3Wp~8xmZW45V3=s2l5zf(vq<&2K~nDg=1^KnWL{P-3FS8(F?qo%<v1+&qI5f}E7;myJbgE!b!sJ8p|@%d}Z1x1zv$'
    'UHdqhEm?iUyOw1e^IemmXZFN7;;vz`^qn><Snif#rtsliAz&4f5R-l!r;xE8segu-w3IK&Fo|U6ChktWQLHe;h9epB;Sat<Jm4J1'
    'njIt77j5;JBkC}P5GCn6@GGM8zsn4=zM=~x{*iPuoJzCULxNNcqNxmLV}BS#nB5=%QY#F!j`fW5Q=F-zB|M=eMaU$v#W=_CXTM4l'
    'z7gAZL&~P7i^oKm$e#Yb_TZXxEdVcrEip@6(;!JbvppP;Sg>2`4x(6(Ud89l`M`#A)a313#5Q0_;bc;0^<Z3o92uYuz*SU+9eQl>'
    '45hNiCqPf(&rbQ?-++w6aZvF)A}yFbqU)z6I{Qt?u3V3BcZvp95OAvdoJ#MEXYExKYN#mqXC|xPv6POwMf1Tj#!-Kb7u^~LzRrNe'
    'V<8vRD)>!}L<l*Nj-WOGFautMEFG_RhPxXP$+1aApI?bVZBXH}#6e!4NehKkb&`(6R3d}xC<aHofD%<{76F^s$~V^XFn#up4Xd{n'
    '&q|bL%~uk%oFOP~ADvV?<`Gec#o3c|_#VqC)`)9-)bNz9y4J*@0d{#OSCl@~I<qy%UtdLCeQPPYl)-ywDe+KT<@Bn_thY~6DatYD'
    '+1x2+r9GkH^uZ?wv<Vcps81UM<Qo`k%0z)?kj-c|achv4CUpUCMf4YZRACBDUURFCd-FM9syKT$!Sl_Il!G05yzL>}$P$jX-U%LV'
    '59&T;s`r`!;`^h-8hs3c$9c8pym`o1<qZ^CWNb=<G<IemWhFIu)PogMrp0uW`ZmtrIGoUdx13IiLD#@cro-om-Fmefl)`Hh5?v65'
    '&<Ac97}!4c>@rBD8uAt#glW5;YtIbu7v9*+M3@4v!NE02)UG!mEWnp)JZ01%4YAN{<&1d#zKn`DN0;e@2!S;0J#idevDD$H#RW4V'
    'L8qz%=LYC=-1U0U`*>wzBXxW@Amq$MSTq-NeP?(L@&G9%7Vn%1yfd+gqeJ8b9Zc0!oXomzP*Bft*tZKejxs2y$9a^0UooVcxI2zl'
    '+yoQ?rdlm0!ft9Tps&L0T{bJF|2(v54FIq2*s|zSmu#iu9ZdRi7#eU3nU}}`xkXD6Ya}TnnDrA?+N!!pI}zUF4xE&NWvX<9=)$1-'
    'b?megS>2HFhCTrA9f}+^=SW;=AUN-en$%d*L$F|<c887$&_WKKL(|gZ{;>9(Y2Vj6TAYTE2pc~zC(Bke24+a+8e2N(p9~C>U8_UQ'
    '8F70FuG%Jk62qzbqS3u!+63+ii43PTlmT)l{(F464-Ol(Nxa$R;fUCicfZG(*HC1%SQ?#gS4rKHqQ=SVLaC@)X2U>o=J78I=!(`D'
    'LhBL{sK9NeI!+_V2ZM*`l>`j5Ng+zKn%`U)9LtbFIjZxG&-`PTP04|s6TAzXchVy1-eYmxTy<5te>jO3v+8#@38`W!CF~Q7%!rSp'
    'SC5`s5YF5Q#dS7vHaM`^svwU-xEP+?438~SLA=nwT?z<*v{w;lF*=(Rz3~1vG@*X>;gK>^;x#rgE-JJ)v4N9LcS8M*y<J`u11JSM'
    'qnZfL#@HtXmNt6EJFbDvOu++ZHN_SKccb#E17Uxshy|+9^m3$Nu4!6t%<Mg<)0KwrR>6t(n7yQSg#$|pk!BB4a*p=hTSmlU!;9P0'
    '2cr()BEwhhMH~+-{G;~hDlsg!)HsvnWz^D-u5C*n?MTpCkzhGF_Z<Hi6ixj&A7S2^a>!p!zQH1;YESFYWpN}A>DrTXhQk|M1nJE='
    'k!hjUy%B6_$O<5u;Au=@n#}m$R|p*4L)4QUY#8UM!%L~LvMh_J6G1k%-)NtC)I*!QUc$R;kHlieYC?g6A~^sh_ma)6d7lb77(f6~'
    'JXTpy&7_ppprR6OH>+TISk4q?Hwe0^*k>$G)98)+@j~CwJhM^3KOZr|6)LsYU~-sw|NQxuKcDDsQN-LJ#ueO-;Yi!?*HQHH06puh'
    'V)HC;J!54=^M}}b+&ng~F_l=@Cjw4V#>Y)xZ#?xx(o~TGQUC%td6c-88S^$#x#jP|0Y2qzQtQnxenc1LRc%biPtxU17wvY!s3JAk'
    'L|1Jz?aW6eB=U^Of&sqap8+w+7zg(opfkNP(vrRd_sl*IL8c|Da6o7bwIRlog0T<rMKhVW(bQM;zNE((8CB4?Ged8(Z#Gj-qS|9;'
    '--=ot8B}e!U$_KphDksbjpI9YHg%}II}S~_a+)*&o;epW2s6k}@lo*e=mer@N+6F~%5h~QYR2Ri(@$nX!Y7C?je;92^x?Y0@IApo'
    'm7E85xci}NQIXj<Dl73Anp1q&`mtNvasX;=rDi?RCfMPwps4|^&#FS~_v5zy!d$?HWB=1iqSG>994eV5p`GLY)26>lPlf9b@GLFi'
    '9EY`%R0i<kD!#Z=;{6V7cCL|~T5&Y1xg5t-Vc?Ove|{5$QU*xJ@jGPwGi#zK!oX}=g%9k8^-naw^X7;Rx(Q^fpu{BSz?1Cnd*lvu'
    'PHP;w{AOnuqKTG-KB6HuFfrD34_n;1+r5kZ*vVmpV89aFrNi>2#~GRD7vf~d2pql*tN}1FC2^BJG`B=CWSVoxad&QDTfPyx;2Wr)'
    'Xsl6YaxGDNMXmU;oEZUeKY+Sm=W(KBIFe&SJOl)nRvsjaQM1ZLi39T?fJAe{U#>5bZ37LOe4-)XW5(X5xGo8Fq7s4xg)Po@RUI<3'
    'SWKlaPCfzXdVGaRN6bJbfSSd7;)EzM1;(+`69UKM$s9r8CLw^)S}eHl+X6w2*IMWIma^tC>&mFBQJM0?O`up)*nrTO%<u-}8S@|{'
    '`oCulnQB|?ohYy36H$9HLj$$m<3;u?y)6zz;Z9Ky&>}J5O$h!Fu{J28yhaLtr#Eg`G8BqxYfWUFj-wUXuuXUF=g(cNdmMz36on2L'
    'NR2qM0Ivq9qqRpEDA`4Fz??TX&JinW;~z=!FdlY4nbXwUc<%v=NB2Q`X>7kcDn8JteN1jMJ29{Fn(~wSLMB|PH?Kizp|-0?2MVIT'
    'DEXsEuQ2;ElTNVy5gTUgnoKQB!=q6X{htQI$D<3InWnDg27Va>%4oDDhZjaqB3qkGNIpB!bzuFBE{mN<x<&$@O@=p+G%)M(hAEJm'
    'h<yha7&4bIp<aTE$C>z9Rw7`CvxcmhNG1ReDW?PY`+_;Ud(&9k5eqd<rS=UNM*O5Qu(H9N)s{z<#JWu$1{RTpRK5<t0qX>)t4zym'
    '_tsha6eR=xBt^*{w{Uc2)xjEK+D<}tywOmrJXMc-hqq2TXVrjm)U*U5GahUR{Jnw!jh>6rUeBj0AZc#$3NXx<7?QSZ2v;ZNA&O<%'
    '0yy}wr>~i?i{j2TZbX%?W|<ssY^<N(zc)&P*lp#o9?f1AjNg;Sf!h#;LRAwG*M4-LZpk5pX_*Erlv9-{3JF$w`MLo}v5q|vHS!p2'
    'GE<29PL&g_?jtWyUJT7m^Q2SiaEcnF;i!1k^@v$-37CitM;w3@lKg<Zq>&u2@{Ea^Z+}c=h`{VS3c}O)j%0yD0zAF{!+(!3;JGg7'
    'b4)_xP3XW00N1@|eIj6Ze2?T!rhCx+MJ#txtSri{sG{OWH~~7O3he}6cS)NWVp~6zCqLO{y}Ptonr%}H)D`!*0O2wZuGp%}KK*(x'
    '<<YPAo5!ziX6~?2B^o2usN=4fr8h5fWg7xHVDKqoQT-U87=LT&l<HEY{V^P$ps19@tk`iC!X7ui2D8O3!r7`XaskH?Vl(BzpJ-qs'
    '#>p})QIsgNY4HgLGG!i?cK3{SBgaK_NpQfIJUA-DlbK4G%qdoVL8__Z%Z>~Y0dG?R5U)^g>~pb6ey5y@xCv%oJhg#5mqs{^rX37}'
    '_|<WQX@J<U+V3kP*2t65PV7Vz2~@V434LWT&ZAN@v^ypyt@K?lq1hm`>54B0n<D@m>PAM=WTP*5nDbGIs&H@0VzV9l(Zjo1N?Bo<'
    'm(_m$*yfNf57YT*_qzPVdqPX-ydu<KGC`MnTx^PM-P{?M_jt*0;CN1$T>sYcH+D2GMlJ_7^i&vg?{<VJJF|h`N@|Ihm^$=GdGW5l'
    'gO2Y@h@*EoMQ~|z9;$0pedKxWEZhXM_f<9h#E_oGdCiYZW`m_{jYeMxt<uu4^00CPo>@haFc?8IP#$r{ZpPLbf7t|YxaOW5l(C@>'
    'DB2?_cNaM>u<S9(>{Qce+h_ATh{s5ngn6(vRQGv)ZAp6hKcB!DE$VuVEMsM4KE;2&<Ig9cNsn~{=LnZ_X8pp_t|>{HVSMu50`<ca'
    '<IgKy1v<!kPX`<BBqj>gb{XzsIP=D)O|V|{lrUrkWu<i;nG>@D!ErDm4MhwZOn$7^dtW{BTDfgbVQuX};U0OiSeuM=FdTq3A`^Yj'
    '40$FlM%;q<))t$T*f-Je|9rUvcY4ER!6mm>n7ODfGu;POr^$L2S~=MoaBaLMwGkkjktb!iJ8m+m7=4aJS!DR?O+{uE%$m1oMx!N)'
    'M}tm-FpyNL*vRA^hH1e!=7;rYyJ)hQ`)sVCNDg$*{ia-LXU@D1W*j?L<{ZGnj|v(h6?8i&vZO&~tX*hlL#}91oV22WM#J6RQ?k*F'
    'wrl%Ili?pJ-`E}+eTkkjuf#^`T14JXuz5~Vr2sM6jo-kSWVGy%mU^Es26DSSYLZiAC+;n<MCLE~p~XIfE^oZ3A~8XT@gCz3h}$;8'
    '*NjYOWI#|kF)RdA<{U1GX(F738owUY$}rnOD|CZ`Q>@lFmlR`)GKlA!IIhzH;wc{OT?e?G_s=7fIpx5*1%ABa+2nJ+uU&*;(Id$%'
    'nH~j64!KC$kpD(R8n${%YAk1g){}8~b1Ut_ScfWqV3=0saj_qn)z&h-Qas<-Li!@qU@#_YsfTTcvI9|OD*tlYp-BF?K69y@A}<TP'
    'KUQgMy0s=C<3wOyc2}l>2noz+M<13#XASs`vwgy(LqOy+Q57?R@kWuh8M!b%kzMs`YP2zgKrn8eI2!^70y2)=cB|PVUd_>A101cn'
    'G7u^@Gt^*wS=pG3LqRFYv<!CwV>&2z0tvAT+}Pxt>5WM-Mc#fW-(gbE@{P<8gwvb2M-trPQJ*Qu-idI!Ypl_RYO+@L$)_5YXy3Ym'
    'tH4jW3xvSbsqufpJd(Hb(yX+BFEBJIFH&0p3EQ40g=JyAR;B(*YUD;71}-3qwqVvr<-J;h3ZQ15vdbL^%NZ2nOS0zAA6^a4AD$8N'
    'Lx2Tp@FZK}tEaITl57)dx>Ph5yLlt2tz=3|fxjBgrLq#V_eC0Ohwp}P))?G0Tr8poQB3+sX&H#V*gL$E1t33>cJs`Rg)N>}!HZ<R'
    'H6e#-poI0&aA^ItX!tNtpQNFAvezQHjvaidZ;o!_m_F1ypdrRk9sj*WJ3=Ef<(rogm0@ULlK`Ppq6wa9tG%bQzU{Gl5|Su=0tZr?'
    'cSsTII0s{WxZ(ti9UcuxY^52sjaVsJGe>R3YF?Vi$$I+l^PJ-XF9eJgkn1+dtZx8YVTT{R`Gem$5Ms)uAyITLMAuz@4u*|{Vu~`8'
    'LzvjB%kPo|6y*LC3ES>aNJ9ya-wd23_NMb*wxnDdkV1hX7z}O9qH&zEk{qU307y7gMT2})i*ww7Y!h2m)GkZGD~x6#Q_;<46dH1r'
    '?L^ANb{xC8j^diRLpIz!Fzl)x|L;M^c<48{;)Mp9sUyuEky$V?%E*K@4Coy)rzq!E&unuh6(Bp5NBJot`;h3D^vrStuVbl6<S0g<'
    '`>rzCD4A0^o*s*43iYe){a^vLWY*F9L#eMcwk@%<RCLKwfK$NBjhT}%HpGh3Uc{T+xZ7fC&!}gk%0~JS52)sZ8>mq#FwsT;UzufO'
    'LG?6KCASeLL>=uJ+R5S?E*+BN+Tub=rFnP>9R&&JR8-wy>`+gqm}NXM0Uoe%(2-RWKHsp{y?50~D7exnm32XYDdvu~B}`-}aOG^n'
    '>gcLQ`HCC3wHH-GT1~zi*e;AF8JyIa877j`1*ItZoQ3mAm$o`#RJru&vJeBfU~(gpqh*~iIkxz+8M6Oz0_K{%h_HbUV?b$S@+vhL'
    '8vHgg$8`*bIg&~KhT$&aZ}vm+(WrnBVtyN^Hz(Cfnn*TIjey&*!<mgsKM}@%->G;)5V1kbx`a&_#_w;G{^E5~3g8Tc*1{o-oAS<_'
    '$tEKzG=Yqppb`R*D7Z1B6@dvg;7&Za*aJm(|C@1&XAoALjBof(cy{AOBC9A%7+_7wq-y7fGFz$1&dWy^8!jL1U}I!|mdqpAWga{Z'
    'Q)}Q`2{mb~8+bMK9u*d{ib}pPlztLlducsAla~aJY>7A>Ll~{?-h@~Pe})XW(j74&$I8)czp-*`#K9@vHZ(Ed;e7}IHKPVC*o#p8'
    'P;c(8ldM6%xj9H>(eu~E_?8;5FRaCj&m+P7dprxCINBq%cj0ich9C3wzwgK+*^s@aphLneb*&pPAZ#g~MJ^f0?O|gR&S^cHNkgY)'
    'd_5tMZI@nUwxslf_wM`w4tyOF6VTW28}J;A#=dXGv@U2U{jAuTd2-|54LO^W=NL}O@grIF@T42%xEn0d?OOHS)*Vo-*plL8Aq8Fq'
    '!32GsiTvcxCp_vd$8Um>S)8BA6aM`je?AcnM4V4I2%^IUNY&khYynG|aI8$y88sEby5-~_&`;XDHzq5kqsfBHqZh|U$nAY++4$Qz'
    'Ar1qT1r|dclKJ$ENi-k9qe_6lS_m`JUhZ8EfQvVJPSQq=`jRfxdooNr&ZAUB;%L%^mwvPlndekNf&w?E%6y8Vx!=GYo?cYcMtD($'
    ';~!I~K9cqZY-z#1HOSg7Hr$H8JX)sfra7n>1|r94h}h`?5)P6q6Q{UgBB$V9)HA<2L!1y*iil>5lRNgZ3^@B=@4UFjkx>XSDjIcS'
    '5jcG21*Hv-Dtt7Q!&gCXGLjW4LPfhEpY$p>s<o4!Egl+U23Tf=+yPv@cXlVrI!UQ(A1^TW#`TT%AaMFd>4&IiRK@Y;?~-|>#h$iU'
    '(Fi&jz>}fw$uY`9D+n}LJn{ymBFoIU3IP5ha8AwusHn3!&3|Q3g$aoaSO#f+s)qhnj?ta9U)b(Pxxt7!O;bSBjl&REEV2k<$)Dc<'
    '5ymhD;3OsE$w96%SrEJi)al%Y!94V%-<*ks<!C3`_~Cqp^r^J>9b8AdBCDEs#d!+wEFiy>mN;HFF-l=o#4?s*8!e5$iWT&~UX4z{'
    '2fhe=287RycGYCpmES)(7t(YBEnpxJonXQ>(e@Ih3I|-g!{fN|*bQvtXwS`{Wg2$4ChPZ_WoZ(TwCO}U$EX-#jG{Bdm_jU`{SYNe'
    'avEHk;<U^xewdc4v~iS~QNdKr5?`cCjMJ^MZN^VHF{@%b+Q_2Y5Z>=IHQ1o+EbTZwE{vLVhdO(=9~-*fl2NgYnxM`%y&mg~QC5Q3'
    '^Ifga12|VkGU)t$ecz1&{dSQmbkN3TTM<fGqM&b}IvoKZG%7kFMmb2yl5t>k?~{<r4NS=_MZqx1=FKG*)qfL3FN*u)8~hsd4HLfM'
    '^ISFhjT-O7F3DTk07(~D?GL6(Ra`*!*rNKZv2M;ZRH@rIg^nB8;b=^W7*LkBot6OcX+ox{B0v*tNH7GiD;fize)HfsLZW?ExwLM1'
    '_L8RD)_&hJc*-FiF-(;MRg}S4LdwlAn@<dEGCP7j60GqC8SUog|D@=ioG&tF3yw+?b#0nG9y@ZpR%FF#h&La+?<^_Fo02@1ES);>'
    'G^EV8T5dq5!f3dZ*#DOyQ3P75v}oCY1{?UUgZiY&4t$%E=~U6gBOdxb`R1*%@7_S8Z6!*kK~o7DT=3z72bHnz8P@Tj7Se+?9}Xv('
    '-y_-Vn+#eeTReVl%)CJ)X%sU=y+o5K+7FR;zVaa59+ydZsQAW(j<sm^mX%morueaQB5X=Cv3`S71$1|Ba*qb`D^2Y@0~&EUF&N4C'
    '!*!WM#EMn?QDy|WNDamvVd2zMF{R%(aNt@DhjFgOrP!HncLk7UVoyxci7;z|nK^hD-FnL$^A!NyMW!*loJrX2{RZOlIB^J5oeD@^'
    '7^>3V)RD;sbTHYJ!eAa}Rbs^xjBfdf%wyoWA*_IZ#kSsV&>>NGZ;hh{(`RkG8FT^*!Lc-Hx^6g89RuQ>wLP-dQ!U>eodN?-&yZcn'
    ';)s_Bz;pNt#-`X_l>0FVc8(@ZgqwKKuEVog%5z){y1IITZXy82;$6v03MC{?q$>SOlOiTII?Heb+;PvSe-M?oq+b}_U-RZuGAOc*'
    'hb|RR{G6K1H}wuwaT7={kT);#+?&sYs1UuPo{HmhuQ4R^>N7D@V$)h}Jv#N`u06P9Nj0~Ua7vHVj<J?J{css8*$~B%g`pzH%f;l8'
    '#E4`iE@RNImI`VxA3t)ZnHg*G6%$Ga(_dKu={+v5X+a!z9B`_PK&1=AK`{%;!qmeQcn7Q@D=eCYnx)t?;&o6HVhSrswd4ZMO!8L2'
    '6tDnhq>2Px#xtqRN(9n*+`koR4y5p%Q}8Z016&Hou7ja$L{SN?6Ix-J=q;Q*jC}{WR8KVY7`0Wtj4EZ|*jE=WV~GwzxrHugdQx_p'
    '>X|q~U!{)2d~P2_@9r?tWd@r$Bof;)Aq~d4RJO#w3y-;E+o$g${-=pv+{Lttf92;Z6`se2i-YT3>}kvu<ELc_MR>7|W1HAFPmtAG'
    '?A|zm^tNQL0H|&3Spjm|xVd@Q*Hcf%Tyry&j*`$2yjd|xo)QQioE23K4!x>4$YzWU>E>}CgSVg9V}m#*hscd`6<X@wU<fAa^Vf!W'
    '%sDn;(f6b!l<G((NLxm?>$Jjt1CIqm-Be_BQsUahwVupEX7Vm>oFX77%vMlV4~cD7A|Dw{=c{9i+i8>MIoq?P@_MzH!WuSn^9n*9'
    '-9HnCO!3B=1-=u-k7+ZGugx=I8Ke%*AVrJ=hj1v}0@Be)eTxxCofRiO+Ql}rz+VOdU^w1zTf>1n*0`shzVwBVd>A?n`Xd$E<pv25'
    'tE*+LYKS{YBXo!}OEdrUiFzz40Ax$%m;UEF{(OSM$?kB$XvJXMbYzfBM^<$G3B8YSpM^`!VmV-$vd6u<peW!9+O|sFsh~!DG~U?e'
    'r6?MZLA}C7No3#HttK0V@C9#YKEEA&ftGvN%^TM$``9hZhbb#k97r=<K#C87v_&riUd2IeWFD)^MwZ9gQXCW~bo6(3m=Lr$3>!X;'
    'ShY7*NJDN!$yKPS_?@tWwu^YiJ~P~{;%39-4{`HahZ(lB!JY|d7JIH^F`@*+>d9G4(&0o2ivia_LA;`otKZn=U@zC{&Wu?&Vb(f+'
    'vcsm9hn~e?7*#nEFds@DDW5=jP8jDTDgjo1miPfr(C~<caMndM!_zs@hO&GdMmJgUc+z9;Rll)7htU@&=hSl>ogdAMVOb$bq+`?!'
    '#t}0azKq!Rl7fWCCMvgSep-gSyJpfRLJk~pjWF`C(G+3zRR(K3Mt1Eb$?}Hz<|}DC0nYbOlqKnCR3$rAT2NMRcy<*JX(Vv{1~2|Z'
    'rdA2~pg{n~qNY!(f5c$>7oL#Fv=t_O@9{hvnX4*_pH4B$22<ZS8@d*uIuKNu&S42W7<{Cnae!BZcu^a=I4}I=UhaMQs{`Ecc^o}t'
    'XCH@69f?A1P;u~4jok}^N=CFLGgqrl&~Oy@knLc1G7jk0e!g>^uEYYD)^yA8A(m@hiF5#`XonN!%E;<RJwDsY^q7dDV6ifzXTS{?'
    'XY|y3YJIHA(2!fw{LFvMU*w2F>s>H-$2nX|L~Kw}vWcP<CV^Q>zcLN`x+U`g;2Mz~2+A_Z3?%RD+<B}6RW7ZwRl&G$^s;PH7CWWP'
    'fm$3}3M99!z?AQMrLYH|Lz3HvE&%_Ak}6QK{3X)_beOV9bE_;f3Ll+9gb|~ch9@I_8E!Klu?=}LQuX4d$zss`c@3Zn!qiWUh|?l~'
    'U-i;6K-Ccf_TwpswTK;!c1uTmMTEv_hbUp|7`Fd7z)JcO^hffq`fBYYxo0CUwC&b46hlmp`7XK*vzs&ISiX=mjmSEJ4o!;V^2YXR'
    '_!bR(4$L9koJx1*G3G9%h=O_6i32~BjCF5iSgfr^JHIIV*)$3;RJ0ly4%y+S3`M8OaGTaSbsv%q26hHiElze?-C~E|0Zw&%v18ZY'
    'ai}y{x~_2NEcbdU(h;RJjCH8DvGvq!IE&3EKu>4VD6a!JE%OFX-{AW!HpBhfol_P}mO79Un`L$k5*%7GUeZ}mna@HKbYk-=^L{b&'
    'SGhryAvq+4Ve-P1ypj2h>?#ZA^ghq{9i)7;eTXzxN^WD^X!|Kka8AK9qV@9gqrF932(&k>T4u#;mfpQ&vDb<{1Wm0<-mDtOT1En*'
    'q}rNslYi;pw8e5p$IFS3m6t1aplQt1c_bLJX90zfCCKhx7*(-KsXSIjq{6riDngat4PN(16Y32dA*WUzQac)5Y^JcBiIO1c1~Eln'
    '@F=iU@bOT<W>AX!r%5f<jz~!|MaGynXqD<wy)d+ilf%_2El);#8gNNpA9Zxy6aJgrnrgD1*#j8-Af&)=&LoUXR95mGqta&ffM^(;'
    '*IqTn$HgI*X;bj5*sMgam!ym(Q!_f4F1Mm0?=&OOJt>XJKVj-c)3teriGgUT%!WFfIshVY$I@H=zWyTb7z(Lja@1pD*XJVzJ-d1~'
    'C>dP>UD}b^^xE!SDRY5}r4aMl>!~XCP6l9d`KyXuWJ=2wLZ|U@bM2$ysY|>Q-ePFm>CDm*AExVbVD>?<WmlRYtnbW75%W-PTT%7m'
    '@|kP9Wt=04R8B~vM&}iW+5I9q3t5WZm~RsOwS~A4Ih#f8#^o=fc6*;E`yIxYd(t&IwTx)9aZ&L<PNgz>GN}-W0b3wm<<7VS$ymu;'
    'V+9$wvVOsN_puZ;`^K3i>Ji6MjR9L{dNwK<3CURvFil)IPJ)NjyN=Vy-?&d)dKFEFY!`q8BxpO!ppG(sy23rFcyQN*cWy6oV1)kH'
    'iMl|+8b8tmJ4N}JGX>|iBX;c*k~<&Wp5+c#q&h>SLWinTWkq>|euHCna#&zt*JKY<7?##lB|4$^k;^&ANMQe-H&1fG#x<^Sibt-g'
    'DJSXb^L@E;^#}k-y&AlOH45p)yg%au$x?kXA0n|6A^@Aa7rQyou?Tx&u4}!>e87KAX^#_B*C5Z-t!Xf3rh|w%LLDqQm*_zzh_lNv'
    '!;VMy&Z^bdqdRzckO6H<NrQb%>hbK!S;R~bt#mwlr1UT$*pWwO`Z^Hp;{D`sJ<uXkf0x#8Ab>>S=X3%_T&K}KjEW0TQWwU~i3)7V'
    'lb~@z3mEszd47V=<0yC{A=?pO)f;3!8ang%HIM@eoVHksn44t5JrhKPWe{6!osSghmKiDxvbGBj7DgB{Y9t2#Hh+FyZ(<(t&u{xA'
    'cyy)8CiTq(^Md^5c-D@St+PimX5j;!0u*k4{I%mqu)N&CP97I|Og<KY-UQtD<2RIu2p=B+#EqTlh%ib^5WL@C@#h23?bq-dln{W%'
    'T5lh=DD(%7U_xSrA2`^A4l2^I6R9^3PL6VLNOt;B1`M+30oUknTRT9lc36L?$w84CfmfE`FSv{>qK<HU*i1B;5UpMA{lE|c)S;-Y'
    'N#yU4!qT!66wKcdX@Wt;c>%MIS$}2+hIYy3@sU^yv=PhBb_0W6q^8lVR$&bdB_gv*8;Px7RpKPHz&J-8O`6H)GMOHc^2Pp(-Q579'
    'Hp3n#*<D!`b;|gbo;dzl3|})D5Tv=fqy>O;NPs7%=HD0(VM>mrmVj68FnS9!=4_5esll3#<$~8Yzp2~G!bHg}QGh8f68PO=e4d%?'
    'Ay+{lE$6GayXazDQi2jP00n4j+Qz<E_csn+S(*wK#fr;_t)Y_6JnkGmq;gaTLw(G55~v+j@8r3XCENQ6$E>Lz9sWsE6Uq(CtBx0S'
    'hYN@}o+1?_lQO5^NHVpe6pUAPFn^Kmf)u7iUIJkuzD-#j$Z7C}zk%gI^mrAkA?8PDF1w7mlDVMFX<^{c2fc3Zaql*r7N%E13$TiU'
    'Yz1faeb3ny6}{TmLH5KTO18}t0h1BDE9Mup1qDX-{<t{_KXxy?t-7E~k+g`1ux!xIUPMQW4xz!YM-Sb1G7BSZVK7Ewz*R|SHT-L`'
    '+T`Uk0MddmMdcwDl0=|viL&(fyGKSCohcSZbS}(gJk5t%GE6xPKko{yr2g=pDn2*gfrvcHn<)ZTDrWyr-p(i!s_|NqI6)c^&4*{%'
    '5M1`5O9D&-fTQN_+1QR2ua#qtxC#7Ly5`P&6xq+WMlal{^6<dR{21caM>#YOZX2b*l9jNHZ;V=!Q-V_ODWf{BjsT&VIRu_go6Ias'
    '?-iaEXYTZ5&&J;<y1^+U=c)u#sD2sY5|5awTon|dayMhkc#&?2i=NoL^9Tx5NSH`{CkVL&;aC}~z*dI+9I(}sYa-OVVfPV5)G?=?'
    'gwNdoUFH`pO<?O0^I2Cqcp|Dn<xf!wOOh?lI>s>j=E2#MSDCWBD`~XK1$&M9#)h@b=UVB~kV`8+HRY_C;{pWdCW8#LQhxTfRqh_N'
    'O6z6><Sty?(a_?5mu4VsZ8}EwM!dTe#{FcKX8tBoJ35I*SpH7vCM1ci{_}@0EFx!2Jz}?$)xC`Fvg49tdZ~=;<efu~7;rL@L!ciu'
    '>pn-@?(2AClp6$(5?Uy7;yBcbCVJG$Vy0l0_+gzRj?yB=S9@FPQET}8Lh6>A?Ua`7299c2EwR%M9JheOp-1(ZOts`irIdQqkq{mg'
    'CTbZkj&7V(LTpOR5fXHc&1yI3=NcGN7xQ3%;Q)89EOD$yg|Uogj(MgYFcCtY#9_;B7m6^o;uui7F_pUkt$rW}1&MHRA|S?XMm`DA'
    '6QALxcwDS3<P!~_84f`PfEA73akE8mFrbm#ASm1$=}4G}8Eq};#|+t%!B#urxK+x7f!hwej+%*msw@%Z`BEv$<8tl9G5>s*gf>Pg'
    'o4xqqs4UB*O;r<UKPRpD=FE6Ads@21=ned|Mg@!(If*AbIs76e6T)EF6KO0RMt@O$lt+0et6}PiXpP3vx2%$JR}Iq85O;NpIaAsR'
    'dH-8p0$$Gn7A5o~1`Mm_o`CRGi7D2_AYBzjH-TEVU(_P@d0;;YfDplP0(HxrvB80LO*}D*G&<yQb0P_q5%LRhGv|63wj-g*n)Tkq'
    'Tq?+jPJecb+~==%tKULn)H8UStA(#M139p=8z+UdA@s*hqGn0YfE?p!BJH6L>?&dsJOo~m{2EBrTy~_f_M)&Rh-U-5Lh+x{CN7UM'
    'Br^f5as(Y(>uo)dfbMOj+{S?bi9K(jZ8$W@WZAS)PogUa_z-f0U6f};Q^QBLA6&r@49Wm(e<P_!iMVw4GG4CoA)SYim+3DH`d3c3'
    'p)jgAqhha4K04O1z~@-+*u*FhV<)T(y=DzoHy#57;K(9&y-1l4L8f7#a^k<W12a~e=_l!!_~3%Fji@<8c+f3jby`*J&4Zx0-@wEP'
    'H^ho-0MB^nlwnKuB;*~Ig(jhtU5z9)=8mi0+*+UZz+6xrj#EOdoJ#9`M+{N-`7vfSoO_%|n#D*jODmV3twLOEJ`^YH@}=KB5x@<f'
    'Jc2jbjN<{Kt{FRFe3=z79C1Qnx*Odg#;zr6ZP5@$S-G{9JG%_VoNL^m3pSH{1oIp`ZpN>H<U9*HU(8l25n4n#q7LtHTrJa)p|rr('
    '6EU2T`zHOA-C+600gJYNxP&wl69`J_*nE<@V-WyXPU7Qc-qe&2k!3y*PYx|idVIsPwo+~0fcLu_v}v3W`HlP{<#lA#M9e3Y->oN0'
    'pb!qWo*_V39<Ugc<5a-DK+^ZmZ_6-{E2dk{yNs3*r>T|6E4ng~T(*F*jH}?RWh+ZslG8eqM>wzTShz;Las&CFjKJL#6c#_8hX*Ys'
    'nGcP431jbm;IZ}Ls#03d<el-{=c61Wdkg%h+`=Akud_uXkP@E?>^Squ|K|e=4!|@)>06oM#p5gfd?0Qgm;3X_6>1c6uYlhR1uXN<'
    'M8Y?9a5=e!yNK8@=gmp?1kXU^tda~IvSX?5Ylk>NLXlN>c{_k6GCG|Xv9eOogQm3(i;g{cx0{0!75+s%F>|WQ8ydEqHP^TYDsX|J'
    'Y!j_BMZs;81-82GKqtfxk3IKw2VWL(z5UGe(d-faJF_yTGS^uQGaYqdKM%BKQa#}D!g(Cui1>>O2G+>Pj@>Xx*~*7qJI-V1rrA`H'
    '?^3|UR&LS6DX^|@yobgfdT6;Iqu_n0U{+GmL-%Y(mW)g-5e^lMc9PB$;w15^n^EFWkZ#Kc*!-gto!sVz#I_6ukQJ%$Dx*H@zv5t<'
    'NS+5Ay#9?T!Gv4&su*I?JmOWDB~d-FiW^>y@EB{X9Yir1{ZDbjX!8g;z`L5;SCYv?JX)g~9UM)CEGy*Quc^4iqJm{ffkb?+WY|4v'
    '5}5cZWtWK>B9WcYmN44pbakQ-Cyd9@>(AFxz-m-^n4}p=@6IAhDo44tiap4KgKD?Oy@~41Zzcb8!?YIT(S5wH2_iN_v1DQiP1G;Y'
    'Q(2pNGAO$Lb*y`>%dl-g@Va-YffP>}!N_?vtx^zf%|fdQ!YwC9reK-d81Q5{umBug@FMQZOn3pk+r9z+V9=-Gg)9)81T4kc_T<=b'
    '+Hj%#I7rEN8Z0=~F%k1iI|!H0ss;k>Bh7`uqYQBrW@V1`lWEs`kzqua7_*(!l#_2Y$wW!GxgDh)G4Zr36}X@7EDD8$e!<C0F9@$j'
    'u90Mu6ica;?eOv;uMDVg6g9imotnecGJuo;MovjnzmY`@1G75r9yIY_767TmdY<eY@ATGlU7brU*s;|--mmU;CS^YTQJs#2AgC*Z'
    'rvqqSy`P}Cv0vn^fgzeBwB}lNK^e1XB~o&xTq>E!3^}z;D$&fYJBq7liK{00eLZHo)9+w_7#nk)E}Ly*Q-brjEF+^T?xN8+;U7TT'
    '&>`OXz1f4jAR7Xlbwb={QysZit#5k(n<TN@dmP2gXu8}Fd(C`UcRAi~V4}Wd+CyAJy*X1A4$vqlm1sBEjz%XcCh5{QG!aDc1N~NX'
    'N6t`y_=-$e+hqHKWq_<AmUvnF2H1F<qK0RV!Z+7OJj>`kix|VR)D`kYU}9<Kgc(VSxo|jv$v8sFIZKW8{P`>1h~XN7Xbgr}qCB@g'
    'dxj=5FQ?StBqB=*=4Rd5BSy}M+f0k%BQE#=CiM<3?`txK!#sMd6Szs;QL~&Ul<uX{DsY~5U9bT@&a@z@;sqey!%hcr(f&@Y<^Q^>'
    '<k)WFNv$~SpbbnK@|Kg44d+rU4e?rrFV!s5P_bGtZ)N!2=opOyd*8tvPw^e5i08?xjGswKLjobeDF;!+EfGNYJE`k14e{eL%fiPO'
    'XD+`1Y26nol9=3K_ATQL>gVLGY!jj&cr&TsBJpC7m#FJ`v5V;Zn1zVjHbe%$!E(mIuZ`0r#hV7NI{LSCq!D|0f&bAT?(o>0C9)Y!'
    '=0&})fH&ZFtevFofl1N?<}xNBk_A)9GG7!vOT_AgsMIYQl!+|P#Wc3O%|#S0H!lVvINZdetT}zIs~3n0de$xVea+z5X0fShce8P*'
    'LFD&Tdyy#yQ@e>rK4F<^*(Ec`vPi^ZKwG19VEDbbT4u&Iw;C^Ol;qe2HG(v0MRA=@dakDG84Vwf8H&hZXR7ty7sz~l@~R<EEi5s4'
    '@$W1q7wcghlh7?Tm<2!_H5Jjo04w8U|JI5a5j|4P?prK})sV<F%mVP6V`0=6(G*b7bfS+GnQF?qZHa|$Ckq|j9?y9arA5P#u?uk2'
    'E~v@yz!b?1US&^qY6)ab#it`)IY@}N6eqxw&<|A9O)WES_5O_!!MKEM=IVq&6Ud<fd#XHfPVXw;Dn(dKqJ(D(^(=f~p(Yg0HK?y+'
    'r&3^&vyK*LDr>vJ;;<KS9U&c%P+3RxS13h@2V2c7{V2Z^3k4aRRxZ9}0uDRN82iXs6_7Aa3-j}qfE+=<X(#_rh->LH+tpn5C_IfZ'
    'Kq89_+HRihKt|{|Kekd<p}K9QyfL&fvBy#+ROI3w>uQa9PSi|@5z938G^F5cFzo4L-JGFnG^yU>-QQ5ftI{h-$??odr+`pI<uP$t'
    '=?$2JRMMU$T`>3;iXUACMKotlZ8tDVH#|94RL`O^ZdhD66^@nUIIndONhp9F*tsuZ?g_V(Cb83ZEOFvpLZn{&4%|-KQw+cH;ZmFy'
    'r)0?zuA#y)(mZOOrCC$)EXsrf7PmRp`y8MgxIEeo62+{bV%t(4lf=OQ8l7g%QnWe-Of*twD5z1=$bcVcf@P5sKwm%>C{;<#rY$e;'
    '_9v8Vj5bBnV^(-#pptEh7;Ip=9&L`+K2=VpPoE68^;jEDU-KA{x1Z;dVREyWz&KSwCr?EL-9*PnMp}VwAD_bfeF+|j1(9icNst;%'
    'jiAXiLxxBF=l5?sHlrDZpbCZqW{h)Zt1D$(M+N1weBSiAjDTzQ*dV1X+|DbgHl}>kiAvl1dG0QjyI9L=hy^-{H+tVI#h>*F1)oI='
    'r1_Sun$JM>9e+N7U}*UIc{o-_BZ4l)tx)At8Czx%W=h@}c%yH;wno3X(FYX*CPuayJ2F7U%V@oARX9Nlj~hS967W3HS!eJQsKkmF'
    'RvssYffGzVAMM^XAv*)ZEi{7_x?{XojHnVom<*-1kJi60>=+jVluReX;M(Srv*9#(SpqYU8!S+2d(0T|5NyYf(=t#S!zD)qj>8U7'
    'JB)FgCwPQ<AGkVJg-4;5!fS42LCK^r5o*k|D0bgnPV!_jhoq`$7$<ft7$64KZ;U?{n9q=p-&z!D$3|=kK?=ijhvQ5n^Td|YR8QpT'
    '&K2w~E^AOV0n}%oL-os^k54}1<Rx3~zejb*A>(u=={Bkp$X@Lm`D>=RUwomWLhaWAS<XVoiMzrqd_HopF?k$G2LWZbG|@iyMsCEZ'
    'o5X>|ZDHnwFYZ9a5e<pINLIo+;?56Hu6PfI@c+aXeBjsw)sC?UMK#3zq^8E>H=dw_-XcQShePw@brg1;$hdI!rznKmlhNpmy6<FK'
    '#}kdBYmGf`4w-+D*+*7wtj4GhX=mRzYhp2)<GghAE(md;OYF>-jk>P`E6BP;Wh}4ezPILOAOOXpPt(^z_z{T<*-jHXeY~DNrAn*H'
    'd5VLxrg^<2899J8TADUb)|GU(Kkw2K6%fkF*hCg$2c%XCBg-U|9Y<d#nmSs*LoHd&Vk+`o2hdr^!eQ@(FNRS>SKcc38|D$t5%wZw'
    '6xhN(y(PT)c}yPSgk5Qq(v-Lu2TI@cGee$}Ndq^j%IIg}Fz%)X4m2)79A&pIDrc9O!P1Q1N6CGh2;-8i(SVExm!p0oLk&Q|tL-yw'
    'FxhJIZ)UhZRt1WtA`zLN%UiCW_4iN~7YLqkeH>$=ie>>TT%sa{s)pyXs@jCd?H2*%4cp9hBu@<z;DWNAx$BVB0e(TaUcB;EljfPZ'
    'duuj=gcFO_hPx5Yk)_>$(XFt>B8&Kq32&DX`WQQ_u-6faqn}#3a_C31+}+F)Ybi`RIj`Eo;oXU6d|xw5c<@z*0Q4QPyRNb^TS~Aq'
    'm(f^Pgk6_!7AyMZ!U{8`+2PTNlPpo1&`}c&*XFW_VRvHJ883UBO)238pfR@2c9f4|oNNV?{QSL$@+n@as2@De&5jq4lAL{K$#L+z'
    '>|_#loiU47H?jD@*l{1tv}D#*pn#0*f8NOwWSi{nDOO(9V{2J<T(mzA!lE`>R`4VmCu1>3Kd{+fxi`%KikbM`faADO5W@X;*f2EN'
    'iss6gd{C!MTt8wnD)#5G_A?bB1&MMr=XjXL;lVw}9dZdAj^h+LVs1OWj<seXu*%42qFB#KNHy-G8KAW($0fK>KG@KdewT6w8%xY%'
    '`qXYanf72JyF9@Xh$rjvBpi6Tu$sk?OR8<?CX;yx!d@9{gMy8|Z{YBNktpCAW*f0qyE-7BnHLtAvb;wt>Yyo(MNg_kNRa*b84f_S'
    'lV`2SgrDz_(EoCkN;0nNVTRmW+Ve)4Kcu_gas`A^pS_e!vzF}LRL@Jmw9pUdX6U1V`Yn)1@gLiZrkl`r4MUwKd7d2GdTBW~CieAp'
    'uyyiTWt2X$D5*(#ji)<yuU#=(>W_Qs_gR|gp&-0qz&ayeX7N}+m9_c6s%D#NS*AQ!U-~#f1O>ur#muM8dk+*k&WfnD#Hb&1s#r?T'
    'bOYWDFsC-)t}JprlAAXG6c#2<?%=|fn2pinjg<Qst<T}ZsFMY?QIu7RZ)iYUPO>^l$8*h{Ix{LHfPJ9zma{Hu=IHa^m{?+d0QZR|'
    'T#yZ>XqHhrAzM~xVg#JSKwSzyOO9;bfqtM8YeQ}GNO2J9&DJsFUR@{$MwPh*teJ?0@It_0=r_Nybc4{lls!SoNmwyts%#ofCiQ|!'
    'iixL&-8|hjGUr;kGs#^Y;f>Nds=@<Oy)3Tze4tYjN#*=1QCSmNdAIWb$OIq`#j|yQT(@)t=pEmxM~wBzOavjQRLoYYbh~VN<TUMm'
    '15PJSu}F<^M&0r-+K12k_a{N6U>h4X49;N$!0)@~P{vCdMW;X|8Ol-{SlMqJ(g8CuHVI;I_o*>9!nTan?I7+xT)TQJ3|3fzH@EmO'
    'UPHAk%Nghr)=`Kh&Zd)$C!O(RtYoawhOgYc+lhT6Lmt?<GJ#cDZv6(yx436*3|f%AIcm3AB%4U0Z*j+C-@$h}?$1zk)>s+C5m*SY'
    'hdO+Kuk-X<@jEaa8%ddjR!=4#bw@Xh|3NYdr?f7R6*@BzrHI8lT(a)hYoU1gUBzIqO$gz>0cm%~<8lsGly948H($!0PcC>9V1m>$'
    'R((H&+?Bz49)Ir4=FyK~3DWeBw4dMHNQ>X};GHCHM(A)<GE#`s*gMlfvSRt|JUgEWEu;f}Nmj+lgL4!$SVblJ^T?GQXL07ecy<=%'
    '0}1FdfYmYiK}>=1SJ%0kg}1@n){tl7s7EP;2UznP*h>+yh90M5JA!JBAG>GG=#6B1!LkOc22rQfpFOJ*nG=F&aA+?j&NyP28w`Ir'
    'R2CUm!j*ALL-*8b$&qKwju4wq<<SYkOvj(`_7Y>m<f}JHr{b-dcTne&tcb}q`8h@`Ajo9%<gEPj3H3qFRCMB$m`yl;f5)FsoFJ1M'
    'j24*acIh;^hIP>*f9!!<G#ajU{{j*fG#Fb{0dwBmkn?|*i}z!ngf3uYNr;D{NgR&i#^FW3fh!z%n8S=$xRzfC&=pvTU;zAYT$FE6'
    'bf=*puIkJDCX29;vSbPsx_WECvdaL}M$B^IE&^qBBMXISxof-l6sQ{mGT53|pc^)xlkI#t(0Tjj?nP(CI^5HJA>1_i(O}_MsBYKQ'
    ')i^mZ6>??Y7j@3&pouPU8Ha~oSo$K!oMJY9^C)OT`-+kWdi1z><rdO1BGo(YQTeEXYB=s2{L-Gygmf9vG$2SDjLFJ=VLAJ;usRxK'
    'M_x3HafriJ;gnbxnX+`Vco*=lZ>u)}eh*)z%QRwKd3Q^pLbTe=ssE=Wp?~-#J=_{^eeT7%4gE3t3Q)JPE``gxgV`dU>URGp<gnpH'
    'qNF&s3#x_Uvya^x;f)vXyYj>|l8OCx-XW;sci-cb@>rS1<?ajjgbAs>Dn_MUr7&>$Q_Kb2{WFC5OosP~jA!1|5n;Woh(TNKkP;SJ'
    'wu>_Wo4A3@7P9Lo7e_O#y_pPeNQ?Fh)Vdvppfb_+g+g74k7FHjr1>|aUc7Wogqtd7?u&;4Yq%b6_Zk-tl_&2T=Q*IEtM^w{`&iV&'
    '{O#?9uq9>{UX;g&UD5^QubeOZi%t1&m!Be-@sYB)@6nVn64*-ecA0Y4i%(U84DrsV^}oM1frg;7^Pphr>y2~Cj|>Ymo9W_mOrkGn'
    '!3x#G2m$BWy6ygL@0P_ENch$skzdzDPX+Pbi7u=cl3CJjq7Z133yg9T4ar0@0c#re=^huIn<U{|)Cv+|uTBxbVY$IArp`BERm_g<'
    'P;{Lbde^I^N%@cKIle}5p%;h!@RQ8JZE(Ly^v#GTH(Uk{<)e+>FUIHlm~NcI@nXkmWMv>zg*YHuz%5Eryow;c$bPC_5Z@UmjrjoO'
    'SHHmGDaA|<<hYiPb&qFAz`!z036Rc|3rYp9uuUeZ1f^eXG;|ucvBR~TZQWy+WyUl(nIT-nNpVq<ZFy}pC8OJwl_1KRp=yRC>0a7R'
    'x{7w91=uCVzIefS`^JJTjFKoz(msoHeiZcFl*jE#-U{1e!V!l|IbZ3Xg4y<-oMr4iS2FRc1QwZsGP1a67>K|vE@mhUs|%D>n>Zc^'
    '8)M`b+Q~O<m+fJW=j-xHyW)-M5`5jivY{696IUm2X<crXGH_<I^s(+=ruL}VoJ=al?(2eJ+!?>XC_G82^RlyF%a0><ywm-nKV)_e'
    'W!QT3#4oJsu*Pb<RPB&iuWGW&t2eZKN59xn;hrdCm+9Z?7l&di4m9t_`tx$l;8-q^3vd1UX5U(z;7h~P5=?jd)w3&SIL)|v&x@%A'
    ';p}zpEt8IVQQSjgkX;2^(0*l}&v{~4uHs2tuK<d*f4i-sin|DsK{DsareHVjHSe)ODR{-!NmCi+qH^Ww6j*`Gt4e=^wh_kWY!LlX'
    '$-c_+^uMG0jdCJfHUo)^-l)rvgXjYIPW25)oMtHJskK-17tJmZe4>p9CJF<PX<fXAT^@jAD$KaAo1kZV%1X$8%{b{=yI6<=<qBp='
    'hoW~8pbg@`mHWl8x7}k+FS+RZ_!XrLJe_@6t~Q=rlRr?qnxs*{OI{(t{ykRk6qTRh<(`NApI>2I{*ppqj0<m;&eL_YrA)!wO;|F%'
    'RsJjbq+PZ49ZT4zw@VY7iz;R?;(+m1NYL5sCh!b{X`Av|=~q%-rfvlfk<KAE##LQIz9($7xDV0z-UOa0xOu`OB%tIH{oT0TO2;E3'
    '!mII7N3W*RptPHi;YoQf3zjntgI$%{6Nw{g=e?MwT@BG1#P%jZk<7jB@dzQJDNPzA|002|!Jm|?Jt#)0?YeO)7^%xS$M124opj&s'
    'GKew1f)IcIA@Jw0H{hr_z~~k?@(PAQ=nFjo(O@a}uSKgIY%s1tdOP2`b|&(xAo<0LtM`lggs2+^LQw3=E^mMykua#)Ihy!=O?uc{'
    'Q9mUP?TaT#+WFYBgT%wN-s7DkwP`3q5is{EC*F_&7~VrlJiek6Qks%0ZU&-sae5do61&REvR`GToH%rn*ZoLLzjr1*ar!#s$lfk!'
    'F;FPuT&^;y>{pXAGelD^a-_`C^Q-9}6H^xgz4MP?h>r#m6>j<FC`V%4XPbu|*)L>ByQ5Wt9J#I_eTJQw20)JN^`eN!>|lr+X)mOE'
    'D~&9!3<R@Y%m+w<NABt<p)c-&%5Ork2<f|ttkjVtL5*DF;#bF?c_Xwf%I-Hf_u_nxcM5w7@)X`%6EGEiyrv=o-L3{$V3!@GydskO'
    '{m**WjG3GUY~HUR@Xm~xlo}GA_f<+wRjwQA93*phaa!-xB2BtQUabrGo}kNB&~Repb~%t2D{e|VRj*V5@)oQL_&$zluN$Dy&1^aL'
    'SG(X&H*XKCCcGB<1-wqnvxn*-IY{@*sYIg&L^BcxpIyr07dcc^g>#qhJ>;>cter_um0MN2T$-JRml1Qbef9bp&ZAqiFPZq|&lm75'
    'wrHMjoi8NKz(Acm47icS#+864Ue-y-w+wtPyhtG=G(2(OMpj?<5C||0pUh2QM=rlO2eBKPDrT`#9R2=BAx!~(fUqNH^EaQL&`dZ4'
    't%WbJU$hA=o~3JUs#vHu5HW5!!Qm&5lkB)aBj*B>mfc0U3M`X*0Qzzq#m4UUD1H_UG29$L>c$09#~9ug)P4kx>xvami&JHF8E4|M'
    'yO2K%&SWI8FKn<bP`~-W+zCqcHLmUhu1EMI(GS^8rnD1_(Ky_rtWgOvF9!$4P#j~WD1Pj+6j-EfP_cNs$uo4O<J(SD+P)euXl^1F'
    '6R+x(D1(d!S6*u&mFpgla3Gn{gs94DV}F6%%99G7>qYydq6da-l<rhuHz~QNkrBJazKx%E@fs>z8XG|d(pPK{1BfC^oKW$bob(JQ'
    'M?-HFEycEZQDI?<SZD^pOlx14)#RGRR2|o5S2Z1`;C75SUm33vt?mbX0Lg}HjB7GZ+2V|ixY8?@Ylm)7PhA?F$HkS#Ep4MW`B09s'
    'UpH)F*r)vBINH{@B6q^+v=a+I`?!BFm*!unx}hS^bp>q?DF~0=E6#*|e{hNTDLhe_;ngd9yG|At31Z3<zLEjrzsEqS`n+~8#^SmX'
    '8_%QnS2}qyLEmH`!&kJcMJJysLOT^R=5Ckr$>}nPB=Wglwf`_j_G$m)eZFd!X~Wua>1*?Q_h!-R3p!6Q<|_hcw7Q+*Q%`QJtE05U'
    'v!7=M-^)d-3$!IUNv|h7<8`)(@}lA#9QbqW%TLY9r4m;($~B`ko>^CWzixs*GB)2QOeWUS#kMbBPh49I$s}(O1;LT*aB{?nT;sjx'
    'y2b$?C*b<ENG50(HCxC3oBbd@fXl`ka3UepMM3&&>L9$>@V1QIqa-0cHjM&4SHEMQ+f#Y5QJdVO8npsnz8eF(cI`hnt(;ICc`DC-'
    'Mf*g)3kkmy!ME<QsyF;Mz)DgK?4s#~juuV337csBVh8UI{hjocjxEH#2kyC|aCR&ViB4br!mb56>qI<7{F&8@5n%5fZEIEcW$h-`'
    'FGFc-<i9#ntBY!Zz<3!H;my2@Uqe*ukq_4AJ!<f=CB5PGiZ<tpd+qpUka!f$Ty%R^rAUWt&<-$D>P<$c&P)h#5u?K0uSmxP3j%6L'
    '+u5#Ik;DQQTo>Wg+=P3LX0_lt4Y9agIR!#VmJ>bDZYx&;b9w2&8AK;K#(i|hogf=Ri+#>Zg;g}*^>~4r_sgbfh}cT&co&4blj#SV'
    'g4f-?q<uQRsrE?c>2Dwc1kc^-iHcuosek|BK|S9fGaQXxH(s}ByriW5SX;RljA;M;#t;aM>Tk$*2xp4u!O3)IVsqcpTNa;E@W(z*'
    '*ZsH*4bDq_%T{4kW|_17n(;6yrV$>mQH5DS=`r@xiOdcty<s%#1;Zh9(I}oZ<E`VqSQFu8t?i+B4duB>hlxv=PUil^vEWY&UVBYe'
    '3^3<u*0Bb+Ew*-lnqx=dt_bnijt%UWf%cZy0hopTu>T!Ms7=)&Qvp%9p7kzj0rSnLA_vT&5(Cau!l+Y;$ub1CXHfq{n<UY5Dr5;z'
    ')#ZtV-%Pa9R>2CM2qhS;S|w1vG2Y^9`91hNM{gJZ=+oCn`C9;oO!2`{lECm(;0kUh&A(^g$Iz*fF=}Mc%5AE1`2^mRK|h}w5hgWO'
    'xjC_f#=`9YDh4rXMtJcPK*F_FGKmc{TE(xznT5wWpR-uF&}a=}8C9rv{1l2&_?}CSul1>MY<Op_U$G1)JdwB(#)0g$^P5a4_$6`c'
    'k5Q~V$RS1bcvNz5eww|)8ICS!AG$5V4B7L<2D+3*^A~rx;4}s8cI{(aTG?F@2|nE#cqOyy`wW8(oZ|UeXuD_m@3)|>Kzq=Ycn<Yf'
    'ER|TU_G?H&WY6BD1=ZeSGl@&*-QzfP8jL4NF`>o|xDn?GgH9r2^?%ua_768i@l#xPINpQGiM^nM6FOl}3>X(Dod9#1fez8Ng43P='
    '%KNfqaIJAE;jn7O>;{ZBj)_f+(BJZS-;57-)RFrb1QPTLIX@P!H?H@IGT0{mh#Q?W+HmHbu__yUACjtjxp8XGj4h-QSu@xNR=ixy'
    'JIh8Hhg!#<P~GTjPrA6XLg%P*-f*n61t&eZWhe*-A8KY)t-zL<*pA4^5zOv~ZSn?rQy77gvCz@*b-lG+2updU6F3op#7=a%ORQgn'
    '9<L287}WHcT1)j*Aw!cVTF`dNo`k8AG8>=nRXcz(i49vWB@VGjMwPJvHv8}UnI2_`qZsJ&ecMf8X|*U|fxhEuG}GbZqx8{zsC=Fm'
    'zuBR@jb|87A#F4-l9j@E03#{Ja$rQVpLPrqS>l#;0WRZ&=n}CcRAC>D#=px?9IufC+HW-z122trjk(Y_jdfuc#R=u=qvO|1F^&hR'
    '&3KBqJpq$qNfhL?9U2a<qMXnQWQ_VaZlgVYp3x-vQKSx41<)cB9(*3s@)M<%>z<7qJw>(P?n9tk)ThU|qP{?bg{#V@^nwc^*%QVj'
    'jYQB|uzGsh1(kEHao!?#)0M%hVpv?IXI~}_?+FtZWvvk3SBZp!0~_BI|K88;B$X#?44<2E>_;uT@hu`o6z0hDc&tjpqA1q=sD`PG'
    'Xm}8)S--K}25pN-$6{e`E)mX((O?jP_NGb~BNJ=UQW*Z<@scB9#36YTjy6c`em)SGBrVYFSe4Ph*a=^x2f~V|*}Cu1Z#1_`W-~H='
    'H<6gT`0<0rT_mpdsar2}RnYG@Yy*-dCp-ySOpmv?uO~s8vs=R?=rD$-<~Eh^QblX7D{t-B%?$xdVU=wSonm{P{Fk~Y2>e9&{hRhy'
    'SqxnXrkwhd{Ae9x%pIQ&yM=M?VBUih<m#f3wW5!g)6x8BnTt&a*^^t^l~}tIdoO(Z;-vZT`DC|=$NIvX5JNU2nuigU;SP&Kcp)6`'
    '-`CH{7Hg!B_|rM3>=Y(cB%_RKKa0<j;<!&LNt7TnD&OHGYAfq+k=pol`5@)WQ`g|X>g?Y_!nO*w{ud5ECr_WaUM+d3Iz+2-K+aNs'
    'wyB%cLpY@kb<pu^lzE`%rKaV{SqL+i)mZvm1~NG!S;<?0<@u}>YL(>SHuR_q*NR|FF>@zEcl_|x)6mWMeG?8#trpF>lG7-C;SD_@'
    'E%CC(8X+42bNj+@52Uk;er-pJ^Oq-RJBeFU(OYn<I=CL$GCw?#9)#c!)9N5`Y|E_a4W0<AygiFM5dsvYtPRk7`j;J@v6#zFfbrAy'
    'RN-f+$cCN18Q8QnYBgoW&+)@uANs`6<W#h(G{-E;r8xQVG)<w{ZxBM83~b~l4N^_T_nahzm!TXcIsL~{x*oTnjH9UQPoSJq+$wuQ'
    'Y_6597iG~#m#*@$y}%Zgw7tp(fU`x^>zvk=z__#8|88amlT)Ttv^wK7vqf49=2y=X0R{o>AtG6jpXp!}90K^ir`5ma;zd!Sg)rJj'
    ';Ip5A_i8TyDhZQpvIgS1jG2cN7O>q5{qCH9-H$PdT7+u5+tHl3jj3MX#<Mw<XSizjr3|xJw&=fby#<8uXLbfO(tT=M=2v3~qUcu6'
    't{#v2zSlZ%rik4~dtN=kf^kN5EY`4jeztAmJcwf|_Q+V&E+6k>wl>efZ+~GujJKdp;0Z=G1p#7iwtz+J`rP6pC<hUOkRY1G6xzMC'
    'Xl%Ry?QsM9i-*@ylX8Ru%XF-)!J_JJ&joEo2IgXr4NwWrgF&^-cGsUOnp5`L)~zWAAtaRabpHOtOgP!5u-SnRdSMBY!LUw(&D>A8'
    'KY-gPAd4D>>8<lqmIA4?`AvyDN<idY2D>W8YmTX)3Ai9G?um>*@id9s;ewp~o?7VWm{w6y@bAI`At|e6GqB<&@l>XQJz+6CH5Euo'
    'w<*6DGAG4!yu`{fg7q-w3nX6jaLy=)e#VqRy5;!O#6tfYom#T^x6}VH{NeKpV|6}X>yy$1BkQqbyB<{oS|3(S_ur6cx1a4C0C72?'
    'wMWX5m=lrzQnCDd){gKl#qMIqC6DtJLVpJ;SJY3R_hmsKa4>uTDe0=j0lLJ8mqDB3JsBJZg~JDNtphNtU{(QitJHfuR);--AF36h'
    '13Q}@ZK7H{&<$V!^BW(2@#A=<h&1huYDGbqIREqmY}NV10kaAb!Km160Jz3&S&O8<*m59JKZ~bFlt#<qTS%R$1mT|jKkgG;-RVD^'
    'j$<5A8o!#<marjqXG0#$!f{o4;@U=m6HgT{D_Y+LHJ&FAlb#gyYEQf!S6n8oB*eh6HU8OMkqV6h`G>(S8~C2!!dFOKfQPhb`*<>g'
    'W@A4g^s!SDC=h>Y%<LKSY<7W7P&KK{_Iv}J8z72rfoeflSvfIDB(m9l80@1}g>|(zA}HgrbNcjnu#DcG1`fkfyD=Hnjt93Zc06nA'
    'F&XVo#W7h#1a=TfKe>23GY2GeZ0+k$!#(s`OdMyNpo@5W$CZv|Pej;%U&{cyCe6#iFynb13=hO})6vxW7t&W_eF-~pb_OCgtT7-{'
    'Cdv<ctE|m?>lW*s<PQ>Up_0H?;9YIcOiQdQn8*w+y2Y)kc1V3xvhsI&uJ{@|UoW~1QzyPDB?TO{{BXtw0N-$|>eQmd<rMl6852hh'
    'xt?^Au2F7m_k7GieTZmk(2QtsM|<Mcf)TXXQchXe{ea=VluHF?>4z<iXLH~U0^bH!_a?`$_)(ls^@*t|Tk%$10Dlt<F{xSWAwK@a'
    'U!;7mSfEh87c91_Wa_6I5y9zaAN_*K5|5rI$2iyvSoH&z{>v;XE{~misOyR2;IwS9xKD|1e>H9Fo3v=$2gSn?I~qk&7sy?I8sw%_'
    'YsU?RUqjO`6*O*d1RntW{u{I}z`{*<Z`4ccbeCX2kFeGI4rP543#-UNb}i;KAZMC8(HTEyGgAn*A;7`_*iK_Rc!<(Z&w1Y|?sQE5'
    '89%~qATq{>AzU9qa^rQkr(wtAW4sW^16>Gk2Ly^0@|+j^=ga>27&La0Fa|2gr@H6QKl$g&{(Ou<-)i>XuNp$Qn`A8?YDnVO;#tx0'
    '00L8x>(M{k0_vq-)PpDp;%Yl<-8JB9@Dtm<Cv^|=Fxr?Z*pb*G#!N)7MTM?0%VE{Low+7n(nS5TQB#>yfjP5zG-t*cO&rPo?As^)'
    'i=_XiEDTqqSH)Kc)_EZ#$(Yc#c~Aam6YK~u!t2smy<N_=2h>`+Hd?3hi2#eiXea3#M!wooFr<dt12@DSNgXQS`l%nsZ)Gfa_8NFG'
    'hNI)-%_g>Y!d<6Kc%|vV-?lI7V;%+>XcEbDfUvGy%h+HV>>9Aq^{D`Z-4%P&Pzw2&alpaNS(u!dMx5Q{yjMR<hLJ9eE$m~R?Tyy3'
    'a#COzT^qD{@^k8^jm}PtDDlQXeB}03W_ob;D@eA9YTxQ7%7i}3#wa6j7ptAQVVpz;I!dGv$ew*eWP*{`*YsrL_MLcXKRycbU<aB<'
    'Q_(*AFYsBAIaC)IN<<Y;UQCzNpi=EX+HarSKTxF<#`>|neqg?0a3*%gJbRwBU9iwkTaeK_z#B4H4+WcT+)SGMwv8Y%gB#yIl}hmq'
    't)KraKw;y>7Z(x*HmM^Tx7jxU0#tP6NDZpqfmmn8G!l50jG_34hkY9D^UKS2lW>A@xY0^Teo~J-eK9Qg#O&aMvb@0CNgwP7ccU{R'
    'V0p4KKJ}CQlRQ0`sLW&lAi<i&mxSg2AItip%yViWoLx*l=1$!>qpU5ogm7j)jW{LmDP4b#3ER~4wu$q<q78_ht$YqfLA}RzMg~2z'
    '0I&g)sGQ`#_<rZ#`N3U?ZBY#ah*caQ@PMlXZZP8>vWia)FjE*16h`A1u4C1PF+Oc|yd7N*yZ6M1Nt28x#F0bVW9)p!dyIm?op^^>'
    '0Q>z?+!ubu1bl*S#mnmAxG3--diRe7s}a=0Kf8C(woTGsS`lSk6PIEVH^&M%tkzv@-|~_NKaTprCCZCJmI&K!vMQh6>S<Kxr}r=l'
    'UqBHayL1onKjN4)^azm7V>`s#E()yj$x?}lZfs)f7#2wmNYtuvzYu_m=o1a25VN1t@RK72z=r@ISqdTI1IBdc?KR#~NE!7LzmHMy'
    'f*lVNK*|;PUOKzGxQeugaGB;OVpP}W4?CY!V6*|;F+WgA+x36|t$ivzSY9$c`27b}Imb9~GH>YQU;(%Kf6VhZqnp|UWJ?Yws>__P'
    'd?rb<OBB`KKKT}KCm3u1bXgTyYLO8#M2wj7;uO;ONjZoU3+#K~2EBS<XyOUT&dSdnXrCjML@J%@r1BYc0B>jIl&In>F-nGv{CNZe'
    '&890hXgxB}MM43-Q}VR@@8oIr4aQPUO$vRk;a#DNLDwQ-%RUMcG|$b(_}3WJK#oOzCDoKnH_2MS31i1vGF#j8)1W{u?C2WGc=#1)'
    '63oSjdH{2ZAM>eQ6&Dq)4YSL>BsuETnlQFvugBV`R;r8tc#I6ax=FDxP<f+q0effgpvt~0T-xx_{8Z4-1t>W+Q{G)p`&<vavnu8P'
    'I83Pdsf<115V)0;yo4e0%omAETxr2>g*=8bevpo?S+_WmrksV-))YJdP(FwnM!x;_Pn<V?er;lnN)Eeq9nsz#2l&kyrk<|ZJ>4GY'
    'JfkJ09YZrAb;mw1xk7{>8={teY0mWb%~a`3kozM_XSmppmY~85j?y1c%6~0pM&*-~T^3u&ygOt+EL0es6mAn_x;->FVw0Wsbip%v'
    '-!c%sl!^;Xyr&z}REVA{4v@~YbfP`;PvFZ^vm2MY*f$BDCJHnTz33}20N^dh8+m;Wl3=E0XaxXl7Xm97W{#Abb2m1{Ln52?2Lo|7'
    '_rqdmrTTQS3eXEuNzUKiY;oM4ZSbFEU{tCC+tnpdd39cC03ic|#KguwrxJGF6)8%rGPv8aca2lWb#M=;%PgI`f6)GAp?zdWS_a!0'
    'SqXtuC_nU^f%DLpXd3ko9`7_D>)ZuGC-rs>>I`^u&~PQ3Sf6SmIE-hKHy?AW^4mL{&Jzw*tk1T7a1O;nkmJ%MZ5>yFIJ)AMfp1K&'
    '9lup>{Poj7RjSgae9<nl{o;(@r+ON)7op0o|0>NI<vjJ)#-5nkObm63_Ew}*pydC-7-#0M!v|A!O7y6o7&MHryZp2ZE?{s*M796t'
    'W&p%7S=&&tkk(_vs}WY)Q8o!Jg?IgF%!~=DTRGVT!xZnymOGE#*y6pZO3(a3o6!OMOS~GgyQ%&Zl|a1gPRa|o6WvIypQuSF5yp{H'
    'K~G}B(wTL*i7q-p6sdsCKh@wmn}iF?D4G8e@7ikPkn)S!flasnU@Qre@`hs!WA`DU4VpBU$lQ<W3|2*~oBnTIvH4bUJ)KPuw6Uwh'
    'q7a9fO#{C_0g@Dn{CC(%+;v)e-;X&$p(!$vJWBh)9=3ig8Vtv23ELs1*@B{SXmq7-vcG-O8!EuE!q#Umgsnl-5SyiG@!NWM?fK_G'
    'Dz!}@uv#?p=&a@Pu2!Up@K|Aa`llJxJl-6%tOfTb`MNXf$E)99^CE39`;TcazPSbCcri!`4R<5D>7}A!Qa6jW?4N)SN`=C@c(h?t'
    'ZCI%y5KF;5^`5Nx>CC8-NBUu|HsD*j61DtvB^6J^N<X_kSqh2fz2?z}3`jQIxrR`i5r-V6kwslb6n<N?k2eE24ty7AXjo5fxe3Jk'
    'Q!)6E=fdaoO}Q5qKp-Yja6xJ9ts~?CBc0-zvz69+HXb-Q80t<&*9Mr+oWFt1Lc?|e45Fxg+VteEFdl(Lo27MPp&{YpT2tiPfsKvd'
    'lN{jvfxf|WSrTSPo4DrUWNq;ATml0<^nbvwc04OOJPB;5pazZjPq9)U8bIPd^1$&4{6hA}xicAs;MG8mh19(jBEH}n_Nvb?tGHE^'
    'sD;@(uaAPf_5(kByczZ%JGz2-Mt9a#O?QIaI<sVx`r<(fh{6KP3jdd>?#5aI$gI%40hujq0D%2AuTxdt$ERwY%-sg&6PG!3K7lX7'
    'AjRQz9UKf$#y{=JVqa77V6K6g0ENybkV*P0B@zLe%}+pq9PYKPA?roH305XH9>SVMv8XM5*Z=!wqy3|+6mz!2qf|H?nmSJcA#Qj6'
    'X^;a!Y)MQ8XlOJ!%v2^*Z(7~%S}tpR(z)<3;w?NkctxY5YA0%geFW#Ika5rG{;Bng-@_tqC#zAwE%2du$nXOOWsiJp{$u^QfK+6G'
    'a$)8$lue29(BRyYO)nI0*8Z=*v`0C?IhCVY?+ig!{3SI%{H5ui{Mj274Z0G_AQw}S7El`*XXG*PBZjQ}94>#XS=5J6-3qFOi$>YN'
    'onlE)m-hLIhOMG@>-3>{rfiq-HPKyEwPG3)!}gOChp!1cD26F6WJps^Vb$Od;EkpB7D#UnTknT|xp&@wSR_T}*8?j<vYebTKVvAc'
    '5GzXG`ZPUrq@q|N;FaC+xa$a`@c}w8kC|i>)ot7q`XgU%DA17(wH&3$j*f22bdl-<?F253<x`1aWz+M*lYmhQRI$kt=y8s$*r*D^'
    'j0s__emWCC2;y{Uvbdjs2axf-!8unXEo~BY><<z&<MNFhb+k3H+4Lq-NCnTVQ`p~#v;WtGV!S_5Bo-`e)Ce$AAplKY3<yAjVnB?}'
    '#gh88d{Cfa(}Hqyfq7|+2^*?Yk5^$Plau`C>a-{)7_C~32aKy>Tsww(jXq*QLl&)rV&ZeeozpdJfkGXQ-xLXRQJdqjGchgY6|wbc'
    '!@$)$R$$etsK4i!lRl2JugI(fiOc_^?%lFuxpl1FuBZhX1TP@j|BfAeGh}aXtw7W``<GM9hbpUTrI!^MyZ{sPGJcVZut7DZ^h96N'
    '<JD+P7|{Eb%}e%1|EFgVwse~?^l{mp#laDe&R@cL(YrJW@}#T(l>!kr-t`^qAv4J@aDxWO5r14d?_V^anJ_hpaCf>X6Y~*Qd8j1P'
    'qmeHf#dqKTJT;+qVF7?3!x(Q3;7-H}%2WU`LPlKviJx)rfYgim9Y+c$wIhQ<)jzQO!(^?EU(CT+v|UXsqmD|aiLhL`?y;&6jETc^'
    'DgVf^l!*np=fNP1Z87jt#kmm&1IP|wCG(#a5>1<{UL$5i$5<B#YOZSh=0)?j`g`An^Ln^G5#V=OqIOJr1&FcIzgoFE{!u9NCkyO>'
    'w{4C@E6$#a`%dS<tYpa8=^u4+)WB>txq1wBB;|6AXPA9CR#2D!URlD{(5TauF19hZWs*(VfRALOyT3%@_FrI!F-)ZQnq5H~>x*3r'
    'PhIsmGlG9k*&(|zBaop}yV{J!@ScIgf}RH;VD%Tfeuu)(Z!(D1yfS=YXvM7<X9P`=$(m^Y6N2ipTt>zr*JH_fI9&9$*#dy7EVzGC'
    '^yuyu;f!>5Q*M$(#x?$>Xp<N99^2$M9s1(j;`E>LD2Xf6<;h@75;@<*su0Zc&sac)tRgW=V2|Zu6Ge<BoT9VA^Fqya>M)(Tzy<DL'
    'n!x;+>H1^|;EGATiGk1b`rjm_MiJH2^EM5}8sC^&^~=2gv%hiXGrJ}@u_rn24ML(b>r&1RO+^3t8vLK7jWb#8nkKW={Yq8ZUnXCR'
    'HGrT~WBUIrDe%p?<y#kp!nn!?|C%nk++bO4W&HQhQbaO<G8s9@#{yS0VXJL?1y7EDx$}0hOE?*qVt^?lL?Cdjmc0-ya~+O<U~B*R'
    'tPX?5yn%1U<O1A4z&S<7KQ{V*{?@Ng;sDuA{6TC%_Rt2b<sakc|NH`f|E>T0tjQ!~BDVYo0R8tb``2&q_h0+>N2y!v2O>1$s+^8J'
    '$Gp5B5xH{7wDI3u=j(dLDhImqg!l$$GTIyH!L|>!DHnzWiXG=g(RAl>>#TC_eoqezOd1`TEd}$C?jBzq-VSB{_fQ!8I+TxXTikB@'
    'h<DAQ|0Nfy2fMXXn#mieebMO|b4qDVi_pKR{i=#UG^?=CHK<v@`IlwzTHjqwmHjRijJTRzZnjfnJkHT-zw9hrNsSRR791BYpFx^s'
    'd?r)w@75h#mO{|#6nQaay4RJQJ?FQWV4iu1j1~nAUy*@AP#YNbSZ#l|1*yn0nshOPY-+<b5`rN~B>#oVUYN)i8JRzAcpxJoaSl-Q'
    'GC$;l$nPv2JMYV-N$NnvB<*lVwr@o|F)FLe4W(Z3E@X_v8raf3?QaG?2xSUnWvQ2dt$CXGk1i!}jBi3&p0G1+L{8He0wg$#5D-vE'
    '{SlnJ)zK@UL}mJ7@l8o-s?GUBSPVgd%@qJs2rxjHGayMJ;rYIT3Y~FO80dgFHa7#q_*OK<`KAE}l}GP{SgFRJ;nN*=DN!VQe=Dj8'
    '_FzFSS!Qn3GYk6Q1Q7Z6kDnzzqH;y*inWWAxUy2(cVl=|hm$-}v56erk!<m>!m)2@CG@FO0)&p?p+|+Ft_^3=^2-q<7*`l>#u%`4'
    'C8%r}{NlRr-}NvCiJ<RKxxmJw%D;AA2@c0bRLyJV*g4%M5WscB#~z=(thc<{Cu+`3{SJp551R3nPVWNN`*#=770Xn9BX-Hppt1N8'
    '<pNaA&;1(c={CYZs6LG-LgXrS3ByH+@vRrd)PN_X^U$M`Khm+^g!&lG{5LCF=6@AcEhba|7>tgqjELR;ygXmNdP$zD!gFy{$6lK3'
    'm_|!Q2*h_qGXKxjwY+I#yK8jcJLXKtO6OXCCP|=tZjyzmhC+#Ql~8+ryNehM&xUvc5VmXiEDY^5zMJ;%1IEJZIGh=OHQ9&BQQyB3'
    'aqXK+m#~@DuW`Dmjgg4A`4`pcyE8S3b|N^y3l)2kFM?Wu39Z#XoUDO|zDCLm`L6736!!xS;CsjvT9zU{l*8yI591Pci+nc6_?eQY'
    'DacJsSfoc3xf^wGzxa=1d^0DLFlh+6vMVeD;(&F(LQY#P-}eomE1eSSvcefyO3z;6Q~a@P>Sd72xYv5QED??N760Q_`wQ|%M!&2f'
    '`JTPwSqP;Po|DGv^?O#Ih=h1h(l#yqP%Z;a<T<O1Z-E^&6EF%OuIS^CNDYA%2A}rJR}J*LPTht1!G_8s4kA278iKvP4SN{6SB$%H'
    '4wy@Q;iSw}6yu)t+-}em(Zp}+e!&o?gaIn;vE<GFmY$R}E$Wa=A#h4-+{i=F+oL3C-(q~4;=j!=)KiYBaHpoEiLts*<tAQngou)3'
    'z3$3rKMojVLJJ=1f>-;U4>EzA81qk9mu*Q!(4bH_lrmf2lcNZ^7uPe6Et)sJqbMFOyJzq9mtD!wf$%=3r~>3M5(s4SIFRYkH&0wV'
    'AsCA(`P~^x(7hP8^e$t0|Kl1hcr`1kysV3-xHb25WF^1sRyv^!!lvbQPB77_P!I;eC|JMw-t!c_Gp(arL*a-Txz{f@?)guZ_TM{$'
    'I;JUZa$XZ}M`p{{HP<hPkNB1v6Q2n1c<ZY0*OW{O-2A)k^AsX=&G6<HIT9r`q&^adn|=4_2jHa2Att)1{zEv69VJ?^@yn5fLv9PP'
    'Du<M7<0AGf7C4la-t3Qr#i|)CG+8Ar_REMLwv~{W|1L)wPVLYOiF4RrB8aXy1{i|&%TK@Ax;Snv(oyEtup4RpkoV6wAYC~N@I(Oo'
    'hkcFfi-s(QG-Q5{7VDRmK`bFkLt>k-E>{@b_ejUrGt;@g@yET4iH2y%l~5E+33aCZH*pe{ItLvl^A`;myDucsba(CBQORSkC8qkg'
    ';)RiA>>=Q^!u{C4-0~^6u3iC29Oa#!L3A$g%kPyMTyb0IzEL39m82mbensJZLvd)_mJ09=cWhl2>trsXez)(~CN*zqU)0kfCn<6{'
    '`3vM*^n#=s(eFu>K>4(bELpT0OcTiN-#C4;06+yrBw+9DE0G04gZhv$|19*I`Bmum0pXMqLh-CkE^FVu5{Gpyjmx@hz^_(;?}Vm%'
    'R(nYoTq?&lv`}d6upX4dNt;{t_ygrwaAlxrOmJQ;4Xi#m0&Vtl3qfF;#l&6=dGS;j%PwhHCsfh-Cqw}zTU8vxD0Y#4`el`1hcclr'
    'tP+1awA@etm5o8%y%ReHovVMlm^$vOa<z;VTU|m866@yptHAwMB@X@RWu&KYUEq#B6}X3<ZT=amxi4(2;dCN577ij-P1!Gi4MFdg'
    '=y|3u!bHC*Bn1Ni?YX|!!T2elDB<vwx43-imO37<_C1;z4>*Z)WF*oql{mhys+;V4-3SvCA1`2^rGK$18SrvA`G)*U;r5quqc$`O'
    't6fghtkTlg55>(&!0K1P^kA?{Nl;V|qf`6dE0K-T=^!Zj1_G}J6<3}{az6Q<(bUNvY1hZP434ToR+}zg`|kBG=uEaOxQmdKttbrc'
    'bd<+)yr8-Uj*Cua=|9~RUW0WNuL^QjDf7=<sZKGVU)9(e2#=tkit|{$*;HTFHbfP}P-ZI3qsg3x@UyvoSpf&szZd~eC62jZIUCX^'
    'k~{ltVY4tNnK)bwQ8~Kq%Ay~XvU2dLSsCTPYo%D33V%jXv}EMNIaS^>XWm4roJc4-nlM>0?oCbTjF{;Rzo!dfWkk3X^sP<0K$m~1'
    '(k2k4t%Kn$89*625dr62LJ0;|lE~m9I=&rbo5lqsZ0#??5!`v5Pzzb>_h>r&-r{n=)onvJXiDvDG}>K^fZv6@qDt$9v{~1oT0b_9'
    '+k0Ht*Zx+APNg&wJAw~L$1D=z8@haN*hqel=fVNCi0+Ra8>3r-IA|QPzgzW}-TM~J8hzQ~q=Kcf$jk=FU09y%Z-w5#2BeF#NX}L_'
    'H3*HqSro<cjdy{1H%dp-JC_-!xX@7krI7pGhc{4-xRU3VaNcCkxDb&lpkRG#w3H$OD`5~Q02c$IVI}aa%boqZo4W!nhI9~Jq-cyp'
    '`bA*;Z3j-snX+HT;G>E@$mzw?p%~kLgfJl0gi@)xRLUO;j_ZZ4@BN*;qwu>IMk?m>l6&;h=+NDN8R`|A#u?@lIS@sB)caKPm%3x^'
    'Z<aJ@-KKmSrOvLG2Y8XBy?&2Q1qM_ETNCwZU;cuUxzpR)FYkHaJ!y7P><o6S64PbI0I~VI>PD9s(`nh>h$?GrgeF9{%wGQ-<|;i;'
    'd4=*nA&29y5R%{B^JB>%I}R}NLqYi%-ZL>A1Zmp0X*#}78dsyzDPm5e0F_jn?*AJ<lcIew-qV-R)#Bl9vXwK}z0_YuczMJqvo8ym'
    '(Z6gKfV%)U%J(_@R4Z|55+>SoI<-|~f5oWsdwuTZD}|d)!V9N~(GTS9)&0BWIp?q_xI~J~>mY_mH7`R|_wRiz8DnxE9ZWOSxs_@c'
    'iD4D12V)v?6QR`NeBw!m+`86fX*R#d<qje$c&+iVYD^?ZAtQQQ{<>!8b|L`x6(;{!UwAzln|RMUtC#Kd&Bh4YvJGW}IT`Tm3fIL)'
    '%I7zKAn@?03a+}Xkxbj=7*?fY^7*Yk#2#)SkOd<YG&q%l;o2Kzj(;EKADFyyhD(_vnIf2b`#R?R+mwkyZWDsep8|ft6~#m!&eQmg'
    '#ekEGfZUnfk}FO3L)on4rtbw1Y4l`{v*C=)Er?N8YrQ;>O8q_{9#>7$gce9|KPjOu!cp6G{}9^G;+Cuin44TgYM%NfR^0Z@lJ!3y'
    'L)k~M-=%sTGGO`62ma6B_4kL?m9;C!>-KF8{QY12>+kyeQ~i<{{`;`ah*!yl%|BiLto<gc7Y07ZC;v2Zs~r4lAKlMm`?F=p@u@KM'
    '&#{KRB{FffUj{+QbX0XZ^MM)3_teY30!11o4+u#%t&B}g9nN;ozaU*jC__d7OqLVL&Cah)rhHpJQ~($uPwnQr(B;Q>F-Xl*1`i^V'
    '`TO5R)%+9=paMVe&MsfMEPtk{fQ_P`z>iuXAySXr*lUoTV$2u$0hf7%BWv$<M{v4q*YtzDg2`AP?MAMEhcH>K!gL|0$#K8^9MFdb'
    'qyeGuL%e!Sm~HoY<8%TKmEVQ4iqw|&WnT2>^A!X(^Xms_TR19~8`Evn7mi1zen7~JG$ihcKfeol0CcPT)8nM3@?f&_b2c6L)nL9|'
    '<V8Lt|JZk%7cUnC^89m_!9UG!yaLv~`WOB=Fa?%ay4{eVfaEV72H~Ql71kFK4`G25Act>mtZJd?H$VZ3&0UOtcAON*e`$N)2`w71'
    '3;$%64eh=jAStG##2H=&VzJr@Xs9>bH<b+nml%b>LcdU|lh1FAq~s!w#;oJo7Y$(Zx4m(rF96j<<WW&Fu$tnK8aJ=2Dw3L6*Ja2#'
    'We4WI&(^z8@YA3umj#y&^!w#zS}06!DtlJxt@|1ax0^{1qw0k~njaMEfUv{boChpWd|8#;S7p|KV@188U&&GtdS91JQt3mRxy87d'
    'dv;pMr5|eh(l0c-D_RCj31)<1%lQ}k?R%f~ua#F$xVVfcBuKdIMcPgMh}R#)B>pKUVjvvfe&eABN;34%A@le6ns&dj%fu4fD;}ru'
    'L&L`xwZ?2WP4OicE|<ooaKp>kcl=;C1i_-pEm!V+?TTqx`v#+9p1T*7CRb9Mmp;SijoFB@5tX?@`0n3Mmp79Rx%CoE$`6wP8I-2E'
    'Uo`Z^p>U%D=7%-Z91(4EW4B$E){H3!KSU!OKj27O?rZWvt|31}r+iI=g|GhltaEb}x#@?BzcixzU2hs#{=kqS=?9|ZaNl10>}$-#'
    'A;;<t7*^qr+S_lcxBRMuiMp;5TQGr<NLo%aHw_X5YfLPWpREoG&lfTG*$BlNU%ZJ05Cr3AK_t2f%w<+Jk=4S+-_Y}%JkfG@D05K#'
    '<qp$GorSJl>H*z9*&`A?GD~mZM-IPYtedC6n_R?t$C~ic70#Op5`WMEwdMJXZRG{G9$?hz3rhuv0;FdW_C(%w^y4>AL2WSi6-?G;'
    '3MHECyj$w@2IJz;P0W5|hF$&lXA%S&cC}}f7rj+EcfV1q<7nn>UU?DK`_jCqwpnkg5s(f>s!f?0mtk7`r`7@WGkFiaX+*x@1yfP`'
    'yj#y5V@(bw0xgS@=lJN2%L{=JCTn-nN2xR}BJBa;mr)Z#XYeZ?1DfZ+t>^}(Vdz;{=`a?jYD7w{yRz8irxK5A$mYl8r_%QR{j4cw'
    'gbpG}1WvTFU;b0&hKP$B_QcqPA4ZYoc)i?n|9s$n38ML7IcYHPFm^rQfpP9$^6%`NJ{o;3zv&U+JDQ8^j|COnKNMZ#w(lt<AG&^M'
    'lu`=*TQ|IYyo(jk#19iDD{&t$6Z3X)fWG+PrIfAMO~w8ES?u>qC#N(+7A-yjcfB(O^U~|)*rMG=<`D0_-7+hg*745{Geu#FuVT>4'
    ')!wjphr}W)&VH}Vq_G*JUL@Dbu=(;8KYSo0hgnjc_@T(M4bKU?xj8w5MLg#Rg*$Qv*KxL#8Iqad(E!7)(}K#H)o;cG@cSzThX_41'
    'gIESGuQyj${h}!IL&xU|p84%9$9mA6rWFF93kuZjMJOQZ(yCmt#ms|RaE<$Jm6~bQekF<YrH%noTeI6*WisF9hmK(pm-%S@9uVp5'
    'V1B4>7a}sf`7N+8`3@Dw1GJ=*Wd7iYQb7}<%46hSksF{CSU1*Ayjq2se!b&;2~^6B7as}?L4f=ah0M|wKlWYGgVpW)n&^4^N(`y<'
    'MfOENOa%)YOgT0h<|+=u-8(LJvGuQi$FGmBded;|4{TgIh51)F_ptH%*SAE5Ls~5K*<W~c6yZ0^4|Y&nSA0?hz~tYF|Hs*g0oc-H'
    'G^`((52BB3*2en+9ZWoJi-?b}V%tnCWA{zFnXkg(8bhZ0QAJ{cd<9hO)4?1_kvt<{j%$8fq%^#_n^sei9si840}a|DwV~YXMG15X'
    '8B}&D13q*GHpcHShlS;VjQ5ZyF)3LH=)UWNFULA|>n31=+ZxXWdfs*W`T=(!QPB)TMMFx^=kM>@g<BJ}w}s6jl~UC23<K<DxXBXG'
    '+=+IB@d|-<cO89~p8!L}&L{R&+SGXf0dvy7V+Mm|MkR7sF)6d#?VPQV2dqIGvH%iC@5V_qnjP~eelU_SmDx28C_Kkh3<{3#aKS$Y'
    '%~HF2S~8<hU-CmI6%3ug=r=A8%Ds?QK)e=WQc#V|Iu5|H2+zUm54T#3xg2udxS0{*kT+bK5QOaOqFdTcmMasrQWOv>4B#S1?*j(='
    'tV~It_@P{y|9%5b;oq~GeC^|*p=zK6sO#pV3UO2%w)rI!LA&W?9k7;}W-)jr%%+P~ht0y>jFAO-8THM6r8C0*Uw3J3OnzsrmzUp='
    'X&wQj^MK-nAfB=n&fx60rj&-YZq8A(>R!G?>x~#32>;(*8U9+$azwla#mv6jA(MtKDw|zo<3Vx71b^-u3>A@eVJ3J1&BB+XU7B+o'
    '-|MfXfHFikq(`!r+YQNy=ZV)n^+T)D=h*KX&-~wDGlX&I>C6athsp(5H&PPZ|DAU@aD{|8HiSRlRTQv#rY_ca%gZk|f3$D+qouLu'
    '6(`xtyfQ_{e$n^FZL*;d@k3h*+Q5|Rh8BnD!wIS5hstyT3cua?JfcQbn+HXGGfv@}F7xI*>zvygg^>9bTh4B(e$(l4f0rx834h8f'
    'inIKrs~f+iVFKCZ2LB}|E%jzt3hBP#HYWDHT<2hy?wcL0GMWJn1PC9OG}rfCRa5QC0^<j}ghTUBwcx)uiWrz>5B(tLISRvKH)Pg-'
    'd_wuT<8t-5OR@d?@A&%@M3Ti-{rl}CJ)m(IqtQ!1F?@iyk%wIlL8rI+eA}{R1$(vmwvfeNUN}oPETf`5dJ+Y|{AeoIK(>m<zoMyZ'
    'sU=vpx5O`AZmop1WAj>V-IoejMmVNV#2M^5&v<(LWUcIr*p&WOcHIs;exO5XJ9$E6k!|ZyTOx5KZlU;B)<IGW<BcoB4MIbI*w8EL'
    'g1Ehjqu^}MLeq=@Atf<Xt>=}p^E*0F^Z`*I-+wK;GZH09HO{W&t7BT~%gJ15KTTG_vXNW}Vx6-Kiqv(%HTaw-gS&;DAJG>T2FCqD'
    'i8>=LVpBvzb_)C}@QX#45)BO&b1ivWpy0$q5S%r=C$I)lrHg+KALEe9msaJ2(W56ztb?2>=88DY^J#uvGAv>cuUq4tufBm_;b^d6'
    'unI{Q4X))b!+BgK<~+YV>2d_xU*K|2Fh|h@ifCtWBa*F$H`F2+1J`$<qi2{KlEm8V3gOI%6<08)a#`hum%Lxp(m43x%|BY3>A)(Q'
    'tEJ^o{e?xJhV+Ug5Dhz#pDQ$B@ix_4X3wgGAZDIRJX=p{Cw%MrWr7xWnk7Xr*-jTt+}HfsqX3W40cp4@883;zgKck^a6{_oo=8(F'
    '<9<@Cl@zg)SUS<(nWMIPO(tI_ADveW)<vLe$%jB(;o@Vnw><eih_GZJQ$<@i#HJqga(ufkJ@fL!RnCwM_Be%*#zxt8zQP+puDnD{'
    'z7p-N@4~f^8%K?`m>ZO2{Wl>Iei>QTXqt|HuGuxjiuydNu$B#2oiqa=I*;DQ)=2IXd4FM(-Vu7YGpZlB*@pLnSNU`$_XGDADn6in'
    'gW`Pueqly+eOg1IBOZ{x6MZxZy*m?EK~&xs&;*LqTJkmE1b&4z*&)Tu7>@lOW*l)ldvcJ7l7?3e1nl0jZ-My1{zcI0R@)&3>Hahs'
    'C&o41v$cowashfagAtQe<M@_gN(~)%pW#s2Wg3eTbrIxTv*OD?sgvr$Q!w!`#;9~5+Eji}t!MCo6z^WCL8Y%uNNfIiWkL|8(Xz(_'
    'Xj)e)(yE8}xX@1cmKvIwX@W2y<B-v26wt91YgQIBB)^PIwkJReVNy=R2>>~~y>6R!`x=9HM%p7QhWq-$MdAF9mX(`u`Ux{bXT1{Q'
    'B!LHP38Vn!l>$uN2$+cBQC(WrLy!W3^Lr)XvcCZV@$x1g*L=Hn>*+z+@wk`J*W39}4aRGoq+`tJYh4dbJ)9hx`k5hpRYI;Wrm$!J'
    '&Ajq<McN?{TCaQejaC#YK32LmvTB=<fiKI^y^_KXVnn_C+6R7X_g4=VQz+2Z_`@IXpWXD}#*wW3r>S{_%eWjzIr3_@YQ?t1l0PbP'
    'PrG4O3Y5Kisl2oMHo9SAk$L(TI(R6fK&cO!w(lHc!AwJ+T8ZdWX7VIOOH;7?j7^N1sSHpgw4QU+^kDTWmDnI!?fhO*s0j+VxXx;M'
    'Q}_G7XGcqgq<w5W-d93RL}}Aq=D2DL_xV7Dl)1IZyT0#im=(_q5H_2C?rab#uS$Zv;FR@w$TVYaMyF%O50iX`gfb!WqFB86-zGNf'
    '+@y`Mw%j<|I^+rQfxu8a`kT)z5tZ3RD+Doj;;3d%9uW2%!U3+AZfEqnKBUEQIg)!xv^aXOf&Dsi(Vk4*rF;{`7*U)ZvPg$Pk;i$L'
    'vSTOJrNoI?G(tSH3@$`TznMj#%R8eX99JCVzR2>~?t;O#Cv-Zvcr^iAnk;*f<07b-{S-)Mer{$V1fE-VJK!)ScZVzbA-U+GoRhzS'
    'Qa(DzLmm(%LSHc871EKxqIkPsUfHuI)>Mjtt2i5MRH*ZeOOjLLia6`&R+1?lXyFEMp&MRace+MRw&;aCFFOQuDxbS4eHmSq&-7QX'
    '=+M|+rO!9$-=ox0<BH;<@0laHa&#t<e)RJTGwj6UG~CJ|h;^BqndBzITAtk3J*!IN4=F}TqpK^SCTOHVRFTvQ*+}ExvDcWFeKxid'
    'sYOv&LIYt*O-Y<P1>}~v&4y4ZspGI!MZ_GHUaXyDy3a-0#T4lDl*4}*Ijk4MKy>nqVL(ld;bK(OH3L){Hf}f+sCr9g*{p$RkT2=~'
    'EbDdM7>ll4?gQ~gN_J$T0wp+|Pem1%Jq2x*V$xx|pK&;0Nwrmxz)Z8`$o%^cAqw@q&p%izb;!ea-0@M{@6py*^#<<^LHxCu_ER)8'
    'TKT=~Jq!+&l;{tt3R?)-^Me<5azBVRVZ>UG2idv&-2GuvZkIJ&zEb0`SBP}_=Y{mHpC1d^MF@C!G_$L-PJc7tzz4o3QMpe<C8S&b'
    '22&8xd3~@|o6Hs-zHyYW7`3JF1(Spq6R>D7`-84zY+8W)WfB2a5Av<Tl6pU}sEfDvINj-x_ZtVU{O9&p9^`zua-r+5_b|FgsOk2w'
    'ghUg1I-Qtuj6ToVG6d5-T?CL1r$1OqKpI}Yp5E~Pabo{FRVikw53}XU9o!xW)N)eKi0OaLv28w#IH8$DKmGG1yx7w7Vb4H^sns5q'
    'Axdw5_zMb2oG&h!pefBZ`@;uf^VEk|f;!Jvtp`<ls2d(`P^6vf<dnew)&dUFDeBwbN;A%^IN0*Sr!L&-=6m>}ITVui@LC3<%<;el'
    'K*!>zNNg_+pXgH$2iQh%QyxyR-3Ycir-C9TS8h%Dnu8?xbkpv6c_*9%KKpn$ZY%m+e|l!9HuE5qtBFhe;86l|sPw0Wr?8uY?jU)#'
    '*kc8U2h~Fx2)@!*I|o3KlAw<LP$4n4e?W|;SRMzhyRn`X;4qt89sbxz)u(uPTm9b6_+Nc4V2$yBWKoIkr>O}>3(X%~4GgmBSg-Bt'
    'VK+vR@KF3?p|tCJ6n5OE)!KUO2YMOrQ4cYQv-aSlhTy8Kr^j^?vFlTVWuLFkF_#vp`gkmP_I~Q>;v6rfKYU=oZGT95tntl7jq>w*'
    'jY6P5HLOgqA5IZOT+QvVo8SIor&kE>`>Bi22i8-y!BXxA)kn3KPkRKoaBpn>9`;NC3dQ$+`-XNAidY|}Dk3M${T_hwq{dzk<(-qz'
    '@Nm${Tw{!LKxJpD)gBPLAy2FasR5dm$HCg`70~;0JgvC3KX-dTU@3WrAB3@tNxtyI=n)Q?+e0bJE}O;A!BKE|ocjTzZjK=F0cdH&'
    'fA@Y)J39)gvJa&KQ7liEITc#hNpaBHDdRya*korup30wwz2`y5P?h`E!=NnAyIIRq!H4gi9rTvABU^l^jY*l@{?NkDc;+!rf>9vW'
    'JoN(;@3;OYBXcP!pGJOSG?(`%P4S~H`{7XNlAeyIk}nR0`c$eTb($}Bensp*_Cq2A<jvh54!b52^TU<aHc12H@k+B*e*U?|ogWV~'
    'W#Y0%c{n1VfBJm5ZJW|g#(FWJB4dL6`~iD<F4U-Xol_V|nDGPvV`#I=+u(O4UQlgs-^a@B7FAj&4>GfnT?DdK;+(0Q&Ogr)su}J%'
    'bgg^5^&4vF&}%RKW41~{Nitx^;Mdk=h+%&IGaFV8N;PRnoD*H>jyrV?rPlRN>j^|L@V1{7?>*qsLul5MKx9G<h_$ONVQhK)?p~B5'
    'ygxnrR9s>dA3dHhTH=vZ#rSRX<?^>)B~we+__c6ovMRQz{cr<}k7xV}NJ?k&COTFh#;*{4lxHFv1W)+SV}#v3fod=YsVx==ob#Ox'
    'RW+WZsm8Ip&mZotueBMMtyPiWYTAudQcO&9g#=k5!FBe%buzb^j6`r5+Jv0$`Z-@XRmFO9dAzudb6)agn;gKleK7r4$yf-Z_KKW#'
    'A}T(F(rj^x4jzx;M^=AXY}n1pCVhIo0g={Xix`ID5G+f7Z;o@DdtrRIKh|vdc-^iL1M;tW{NG;BZ>cMzp6$`8d6(CvCrpY><XJ4g'
    'iU)j4Y*8BSvRzqM^$;<>Eqs~jTE>zwTyikPd$nDw9<XI{e8W(<VVB<vg16)auR&6<vdwVw>8~tr2xUh-OX_FlJrFkrhRkxRbMEfs'
    'V`YDxNbse4OJprpNRl4zlG65sp}2^x2MR=VJy-kLvv<qUBo1`tPbXKzCWvJ3rZ_Xp@^H7zA@`L#xUxaP!s~C~`5J$zkNLt-2jwHi'
    'E5q|rwzK#8s73Biy9~NKx#}i$jBLq3%dMW`5E3=%#Pp(JH=E!I!)4p;N;e+s4I0GWB}X^TB+#+9z7m}bTT*SX$QVgMVtCoCwU-n^'
    'v0ct5mG?$B$A&Sy$5&0>Q}gJYq7sN=*d9Dt@lG<_PbZn_hj@UsK1e8yXWX>;nKOa`L&kmBuejQb31wAsePZY61n2CBKp40`jNao!'
    '$Mdpg4~Z`=$7q-JUc#R@VDvVX<Td9ZT#~xt*dAExtje#oviE}#=Dy4_nx!2EW_FN%6lHjdp+e=iFCL4^hidBV{SeaR=E{1v#9(lN'
    't9xAhns*mFqm8Q9&@p!5#3H|E`^<=-7}#QLLh_PvzQd#~q;13nX@{UvapPh374`DK>3L_Hu|z>q<gt$XOHbCI6S_qZaYedp`;<;&'
    'YTCV{)4)J5!L@f(H>3L%8QhcOh7Q8W6Q;b30%s;gUTpt~SF4h~Ug)Z6dpda|`@;)4&Lw$P0mKyjSfcf$6PgPn>?&zY9m%Y7#jBoG'
    'Aw2P{FMQI`u$91S2=-}7<OcM<W8?Bg+~2D25V6u@ud!%4DfKfsPL={#x+^+hoNpy!gpmv@L)i}_JLApCp+|c^D%7VGbpyi1%CE51'
    'vVLDdfrMcR+_o3K!-5335J}%Y<0MFnmqMuZ7A#;h{TM)N-&6KlnY%{Zt?^YT=~>5c0+BDkAYmZ%tK)~Eu(%2rK+MpKnRyf?p4^02'
    'bv&<oR{E4q{1<k3_$_9=<c<<B>JO7?p3fJ;$2&?NBb!-q+coL7HkeN<*Qeql%7<Il*A>*go~R`+9a~A80(Kb1XF}dtMNxBIWBpis'
    'Z!i*$DrU>g8`yZiIR$+VN8IYh90g8U9yb=n!D3dy8F2ocG33!nYhloi*2oO_dKBs#jNC}T-7Ot6A*JNJ1*B?y91^8-xyf#Zfh#(W'
    'wbFox61QF)H%sZv2~cYBHAPDtHTs~sD~FWfaoqQKg@gQIyd)C=fh|P>#?|R@gIMOmW1|rN9#!AyadkeaQC|yg5Nt)b&89BiCF9Uk'
    'zozXD(wh)*trO@S%K`@h8>b{8;9yAr0)V&mgxwxnZK`}5lbE@h23PNR$BkMSOSD6{`*l_z{J73ntx;UpEJ@uNu#EZPWT|)twiaXn'
    '*!}gQLH!x`qwV<Z8{y{hc0T#P5*!ZPwe~2rv4n2PTHOoE7=2bWjk}!DW>rWnk^nl+*wfQ897Iy4=7o&Bp1nu3(5BFyxML4Na5lCu'
    '_>XwijB#ctVk|<MO3hI-G}TnKuXUw@**fGIoeP6)$VnJ4DjW0T-7dG@_Ihgd|Mp-Gt`=`&aJ7$&zfhtI-y~AgSB^f;uU9K3SRHY#'
    '3!u;PiVA5Fc;7fyOg$mdMo9x^N?v4zXOj$n4F{^|_vhEl;DxhCTifEOi8I8TV^^JE#gkNsQR6{2d8!FuW>G;z%NR$DuFsv6aE9e{'
    'iBj)lSiv+rJKv%!5PI8jUix8&-6%}JbK-&^%d?J_y~4O<B*GjZSEXlzIVZzywGZ&UC=Xh8E0GBuRpJ1dt}m3<8UZr{(|3rb@jJ}y'
    'H$+$FRUa!^CYBnTxLx@e#KuGHy$CM3$ZrgY%go4qqN!GW46^APSQ=aDE%v||a-pC8-mU+Zh^)qz28N2JKC#vK&!2kizOvtd0S6D9'
    'SrTJLF}tFBY&U&`k%?I1)Ugg<lY}J-(6iMT(Hxo{iu$xwY}}*ue_!kKIwb#djCV?$-!?wx0NcI+105OGq;rpdr_4C<Bx+dmY!XV8'
    'AB1wn@4e%Sien_MQ<MgNqo#|68pq-=IS;HjXq41qHxr01k2W<GFuksg3~lQ7&n1_g=OAv4-mX-+ueBTgXj~MZaS!`nY?>y9n2(!}'
    'a-D3jpSD)q+fFKSi;aCFl+@5>O1RUR;l)?e<6pk1FQ6`N+HcSoQAN-^JejWj|A**+brc^F#}MWh$e>!k!DKpUIAj}a(o_-Jq(r$H'
    'WwoekHPt{965W<58KCYP{1#QgRPHdIG*Fo>zT)aPl`vjqimJrBEB>?bz0ozSAB0EZ3PCoZm|duTRsnl%MzB?TCLRn!E*nTe$YAa_'
    'v@8yRL8>c3(NHgBRzL3=T7@t(e7DJ?#=_f+@8UH_9XwRblE@M8SnI@l-;mMx-8LachA%qYUwEN+9`(j#w72>7N@IaTSm^fqI8jvN'
    '*q+1+s-#`V5x`NfZce<o&7yimg92zvf3rBn{P0?Ee~u3w(HjO#TwHH#uNy}+RE$_Y010tW?)9<$iXHm1x?%te!0B`;L)z!o+u7wX'
    'D+_gbEXc9H68Jo};q)63S8S_|Ob&bxl}H^zSz?j?dDkpr$<P~5q^};E6vv<PWX<wIjEFuZZklK*qc@Je^26g~j4SD;8174k^6Xd+'
    'XT1LhIT3fs3(^rx600_JV2W#@gZt*HnsLgRz+&baos25z-P^{i`q4NVi=ury8NbvTBT9@LAAxu*9%6m$*bk1zF1|CIv**o6qs?6u'
    'EUU%tCr=ID!risL<!J0pl+FgqNX6!N8pG*`IO9sdD9Z+*c>Ftc{~&X7MWJQnOUHiytZax_w8Vn%TW%#3;aLjdU94=6Tju85SOf=e'
    '5IgK+ke5JQST{(`qyp}7(l~{#!3(T$e6+e=R+&cKZBd!>U$v>E_1kHi!*7heEb+pE=v}u?g?-bPL^DRV-hfrTkx$1v?{<J-YD8Js'
    '>{1Vn6$jasFMcxs9!R`!TI@YGf$GH;=lbEiHw=rEJxd-26O9j)P{7`Adm9gMiMsVB7%#!<w=Q<(%~M%)%JJR|bwlZ#wDIo|Bi8!a'
    'B~;@+FeQ7ISvWcS1;dYab2L||P_4!BTCv=75PG<%<IWJwD*VU3uff}9TFy7w?Es*lZfw2VW+#$BS!X=Rq~yJE-7;L3qp<KV9(6@`'
    '0~NX6pmk=G<2g!gL3T)5p#k__zrPov5hw37u82xM>Lp&}Cd=;5lh#0jqU!3`Y}nf+HY#_)D27!-oLW0R`xW#hV5A?A0Sn>g$_}f='
    '<0@@&OhI~Xcb%M}Fr`-8PPq^#H5Lf*V7FyTg)|V)3^oJ@0+sRsVI;<uGz^%H`+b&LKb(`xEPdp8J#y@5R7qs_+%$<0%H{kF2t7C|'
    '8dya?piQhL)*Dv8hOvrfYO~=}eZy~wUCbk|Xs<eXp|&+~?sw&nlXLH4VvSY0u#-3i+W~C_N>n<j7K_<Q_QioS9(dnO8AR~>m>fiL'
    '5`EB!>~S-6i<0EBhaAxz%;>-b-qkfvcwnOI+v16JvLam~vUlA~-IwfB<L;eA!BLn%7`y@7raWfiyz-$DeO0Yr?r2dS|J4F~87m*A'
    '@1+#p^F;@HENrJOnkWn^cx{u5Z0$EQR_qjv7>zi>K1VHA*AT7e&8CKO$~c#fe3$~&X-;lPQ}L+rm*HuE{)GHsbn&acNYZ9d-e*&h'
    'WpWaVielk>e%Cf_dY|Zxr@~s}d;<}f^>B8xRoyBr4k^N29L@+ZaJkumtX)~LI!_j5nR<F_{p{J9aGgm@E8<qi_5u$HEz>^%k){wF'
    'ic&I*;%gX~ao<@?#aAZt!=;zMXa%DbVTg4D7}2p{mF7{f42C1P@Y$<a)i_H=tuXK~iqj+(_4tTmD!z=G6k;>>W0Y=DYD49VqiD2s'
    '_gT~JhkUxi>TuoW#cm211andNM>P~3OWX!q{KkIc`whtGaTCUu<(DFvF-p9G_JdGvTxWx7bIGb3)pJ8faW^q8V!2tn5RW+8<2|DT'
    'xS=mlgk!q)v0&IV{#vjyyE(LX{P!Bw4cDZ}sRGcfjn|wQTv={FLUCZliN=U6j@!GdcQM(k@z}E)UVi%IyoH3~cwG2*%*NjZPE6YK'
    'KERQB;sO8}-npU#fsspOH;CknyGsHgk3uGH>nYu1ZHJ%b-k|$8zr1forC@+mzTVfgDNY}v<tDe%CZ3~Fmofcsw*bD5i-j40Q7}L!'
    '*EevuRmCPiyddQQBOZc&EGo*onO<X2B1MVD5{GK$wNMT2roe#7<a=zuk%AIAzLDKxaU6h&#Dg%fN5x;d#!lA|qLe!%sK>oY#9)nX'
    'iW?_R*r`ybddIl4%d;#tcL!5wq8!xTf8PtC@vcx-VFND^qk`dq=G~z?YLF<b4Vhouh|%+l19QV41Yu#$jNi#LsvPJQ^#FsKib!X0'
    '*~FE|EI!JfxK$18<F4@nB5E{VQ8F~}Yue&|C+9L;E1RTHfM(zd%&&YNKrO7y%eqV`MQ(zSK9PXjR#NDf@%>_}B0Z1dR5MG2k~h=y'
    'g*ZCOE*LC0KY{)VDg1rC4^?qI2~lr2Y?mec%kMHzE$&~;K>RwYDzFyX3#&Kh`;HZE;0g<;OlSs-AMFGdsZGrTtD0rh#nt3<GVKQA'
    'iRI6*qEL(vbX{WeQfl9v0GoRtb~Qu72BC+dxzqz5nX1qlHdq(oD^A(T+q`S~qo*K#I}U)opu5v%F7mc-?xAQsFzdZa=&?4;+vqLY'
    '0W&GXsiX3mqEAHf^?}l4d%qhMZ>}<Ii+Bu}Yq&Lex!d~%3<wp}8XRIv9=Cm$BN#or1Zpa{+np){LVh&5`{p^E+9IBL0n3*b$5PX-'
    'KAa}f5j$OFQfCUcgl$^6h2c?TKy3odwfJJ_`NtS%7unH}v&6WIC@Gv@d`7=8?CyG5(HT{o?F{|Q=DK!-{Wrb3VHP=Jdvhs8t+M0b'
    'M=xIPbp3`eaa`?N$6DES$+RR&q8Gs+6K4#12kU5DG^v0OboZi>r>y9pfQ^Iz#g`B@hOZlXlp%KP(OF&+)zaYdLZFy)-&vIv)H+Nm'
    '<A$tIgGa+^^kzVBChc~QtG<sf7dI-s^@Gf_zW`p|q10D&2{<g@NKcLXhqjEagW8ljC7P_8YmQQ{jQUlZZD!OXi$0R?*VM^c^g@uM'
    'Bg%T!98=g}Yz^42Gp;B|J|KVlPEsV)w1_JP2Dq4U@ZokxN;hcGYfms%#7gWHMCd-}@6WLe4(X&^k%b`sE-Yns^<C8&ud&5pz*t^r'
    'cx;S$xElr&VoGXhvkQWc-XR|NdRtTk$Gf<SqgRN7!LE@|y{H0r)x~rS_b4|}T@@_fWo*$8&;1<8>i1fc6K`Bib5#3|jo_aL<$pfY'
    'CGlAp_TyW6H=p>=U-S1zP+E?DK)~VJpG=HD@WtKC6dc3l+m7+m=mF)S1Ve2~*?l$#N)d+(BQE%V`-2)}y@3yY-sCT$@0b{Yz+lq('
    'A=*p18Lgns?Nw@)jk5!=)}?wrTUiH<#ayJ0+0=Kh`<eBW>k~!smJ!C~1J&0Ja?@e3&A4xIWZf+IN}SO8{SD})bGi;qc?!RKy}*?}'
    'p8<de3Jc6v>CG#{l)Gr_P8S%(VLUEE?yeI%N~{}KY$bxA>_xZIga!k7-^i1ckw@&D!OS*bFkHdP58iE!m{kgOyeWIESILBSyP3f='
    '@`rUt3Q6NzVUSqbO}Wf8(ip*@-Ra<BC`a0jR7+WzsJ-9__DP%3;6&b>7%ZPbV4(3bq_~0h?(OHMhIQa@8Ksfo#@;rYzTG+n0%W`R'
    'nX%=4Q`f=(v>W0ZSp;kI#o~gBuZm&Zd&d;#OK=H)0EH^W@|SS%HHc62IGAY|R)DQufjIGfj61dvAiZ6oHkeb&Y2j{Y!$=X9iy~`k'
    '^A8wL>j2XmUahlZ?cx^edabwouKu~;?@tk<FlmZ4ad=tGLltPK$Rfzw85%6SubF+7HU0Jh!h!wEgqP8YLtZy7wDG?I+LR()G<s_s'
    'UWaRnFNFvpU1&7bxA{W;c*lo^EXmohFPg<gB=b9u--)w$iyCuu+#3+9MD@At01Gh#3Z2jsm?m_nCKwlu*4I7DWiSm{d=}e2RMa<w'
    'iY*D{T9>#SP_qO4-B~Y3vvq0XM9Y_7DxGqD5v2j)_+@zp(bQl=L(p3LrX1N0=cK{Dt*B>BHurJA_^qw$qO`#Q63G$enBBD*i-{G('
    'X^%EzO)_b45WXYf`c#%{6-zFDnaeLawe0_&T3;$?>?j5SF$ANCU@~vs@r;sUV1+Xv_6#(7lPTQI!7SD$1W(}3%ABz|rrx)#OeV@}'
    'i;+KJ^tZN{vA<?H#Qg%;yyaN5u52RU+#8SzvMQ-{(8QbgJ;<PMBtU>Tw!X+~Mdky@YYy<k>KD(*#HBRw3KaY6&QHEC3<0tJ?WE3l'
    ')Ru#B;o-P^lirN<8+=QEjcY+G@Fuy}w=a!Z-=dDK3kdw8vxt9DZt-o**W$2-?eADLMd=hr@!?G|fD2_iFKp1&4<$IF#k%3S*SLQ%'
    '+QkI`Pc6)~HX(+0$&_)o$7;usfuZjvM?xS~Il%6+$~|9=jiXe7a3ZPex_h_9NlNNcN&9t&G8`Xw<I+ZhfQ+JwLda(kF&;o(t5EbO'
    'lq-W#6Fmxct@y~_(@MKRfJE~Fxu<KyQ*YY_bx+qEj=VkK^xFFg>K3^9Yse&t(3!YA*YwYF`;w^(V1rL^6Y>Xo+_O<swcE05^6fa;'
    '>$NNuDCLgK_X~0lI(f~&nDPw@%>_a2PJUVBAfso-y@l244m@%raarv52H_ztCZs%sN)9{WnI}NGUniGex|GT6_O4N18}pVy`Mn6u'
    '6$of|+a5Yd#ry`$Ya-r6-{}X?H*f)LeDlmw)d>w?r5k#}QsO)=ABtvj$l2U#Hw;2%nqC!Q!0C^v)sm?C1olR47jItTQ2bS3KNq`m'
    'A)t<KLoQp<3|uLYqTl=}r{|MW9PdxUr_sqoFL7h^gu9E4E<dDV8;Lv}*^3k-a<?!Y++uCuDhiWu+#Sc${fd}CJa5s;z{bT;yxyzU'
    'SkPTe#ppySt^1mHegaUE)e?bpT9t%xwHG<%Vj2f#xAc1JA{8w5vERi-%y{3_odW|1S&T8?KOM$0G9(zT&xw!;S<ASOk%YlAH?0$r'
    'r-t;d)56)|rbDFnP0DGR+01(JLT{b54WEYsy)-&@WL;I5&%A@nw=#~Ge^J;5j)~aD-wD4^s$iKv;>+IL*3AJ$UgVc+c5!tq7BRy3'
    'f?Sj#z#TiQF%gjrarSwU8QNXI#a20kK@FL>Q<kKpk=%S!Yrr6g!z;E;>|t7J-41t{1137t020VhR?-Xb$qxo*qW*69BzJseT*e(^'
    '+!T#V6BF+;U6Dfo+P2kh2sH~}5R#Y0uYYmc7KFq*X%;qiWP&WY*<9qz@}kRq@<<D6P#o4(aZWtr9cm{pyJ`C5jB4vj4r=vQWZJb&'
    'Bs{>Sr=K7caDi1{L|K^uiRyj>^fH!JEM#2AwF)gUylc+JvSkBpI4I%O?cQa|v-bN^0lIXtlIB>o3#KzCoo(H+0MYoz^J5KCtLXkn'
    '!t3b<F7e)6SxC5pmlazDf5!O#tvjd9c)3x2fRZ%7U{Zsg6G!@;m;+X5e0idWV&>s-xYT_BrG-%P&sOl>L?}HG?pg(vY<6cj34+?0'
    '4@WzXJ7vEt0><s7jng^WTrTa>sbL^P)_Q<%jUqZ(zP9vR_Hkt<mAk2mNVRTpE^o8}P?A8q(81hUfv@Xk=BY2bC7jcCzyVOZW<Hmp'
    '393WbuF*w(j=S*`{&KM$E`6d1#Zj-_`7jwCuoTwFh^0WKQ}$)!03={bjShWzAYXB*{dehiyHp->vD<fraK>8}n|s~?B{r@OP{wnZ'
    'mX7lEh5bIrRhif0eLzCX-s_qh#sPGvq>L(A@x@leh=C1G-0!@zbH>9&#}_`gC6+d&^_b%XL;(9_y80-Up=ePbMu}S2jl-Sc(mHmW'
    '*;DZegfT?daA&9mUgC8@BLFi49Kh;-AJAN3EdpCcX>Xe9RdqD$E)Y$itc)wCHnv5j3mSie=_6_@n!p>^0LEK@zb8bC=`E{JRzrk$'
    'PgIE$!iU{Xjg9k)@vRSaLgfD44B223Y{+0#H60j-?qk;jW;O9@C#s|wyAO3BWCmv&-q{mjR<mSHN{fI?;t}@A|EE4d-V?iks$xun'
    '{=3yXsCHdBwkWI{{W?{MO+TC6W5U~uq(rEd`jTnG-ggdgIJhi?&{kyX!o&sK>gF|SEH{uELJ^+~Gq`xmy73d)6NA@i*D*D9Y&&Sb'
    '+I*3M%>=1-ZkZ@N$beN0yx$-#21Tdi(lQ3*&}&mJyAy6S8M<R}YxvwZQ-<4acSk;H=FsqDU{aYqMhUn2ot4{{8}9rXc3pRSsrv^P'
    'Jy#R*cQLuZ);VnVyw&7H<uZWBFr%t<O=X(*cIRYll$ud%%I$U#V?X3IWp@UPTzLSdi;Oer+c4{^^<@w)l#jTvm&vcBi`LXeZTAYD'
    'Rd5_6Cvw##<AZg(eqvRi#E&-(@7nB3on`I!yDiPvM%=}k28-c5qyb+0UXa<bj++EPU`9=XQahdbL8&QShguAUZAk+Dz6{La${~Ob'
    'kKI*SYWrcikGp1ZD0+AAFWE54#@ONH+#GJ#GBKk<1=@(;tvgCUTX^sI4Yb-&PgD}?>eOVy<aL0=vE4;GVA<{vrzMp|C_hE^{{1?-'
    '(bPM|f5lK+evam-LGN6|{_z=+ZRvvW2*n!W&hz81zvn-nVIWuP0VpO{gq8eMoQ9_OJxJ1kSa0kH^&e7h|7`66rLXvp-E})ho+t&}'
    '_`zL_sri}@LnO=!V`rXoaAI*1<@3WM)rk~(KRi;jIsJk1W?1FcwjL&@93boG`#OX$)gI#4O!TG)N-zQ*yl);p0F}Exd>|^MR{n)*'
    'fovvg4;HIT+((~JBY8^C`~v``YrXbd$64`dwZ%nFa7upQ4EoS>&ObanAAlxI_S<}qtjcI#cj@cx@1*FFKcmOa3$@mzWqgmsDCh-G'
    '6DWx)9yUxvjepX^GF|HB;oKTbW&C0MN_=Cj4^}nsgl)9fmV(IMsy_rzA?sLW_NT)BBKzhd1>`Xg(YMTPeT|<74&WKB@-QHi@OrGL'
    'b%-eW!PcQwJYMH|aUD)7R)4Da;@2Jzl{%c-KE^p{LB*o6KPb<!*4nkcN26rCAZzb`D=7r7*h7EXl_);m19t+R{Pg)yE^GqX%7c&>'
    'Chneg&hzfLc>0rP12MVwbi8&(0q5|B7!$U;Jg2{ZUs7=&-q6%U@|<!TL;CqtJ9Gw15958j<=aV5<nqXPSW!@;S9!SdQ5N_)YYyvC'
    '<PHxY%lKULfwHW$>V42T$l$HmgN%e+nm(Sc+&DPe=m$L#piX;$WLiA4JxOO8)a&w4r6Z*6R$hB^qmAEBmH=&_0(>~aDfm`>NVbxC'
    '+g=QZiwW`oQ>nUc^F3l0`&@AdkH_|{HGhvn2>SGz4`tmTm&~8)KAa!sL$?ZEQN6tO(!x^9>*|k7(jMef<SY7<z6yqaS+zgz)nfbS'
    'apSI7-|~zwJYl-rVS7DPdLlBG7b6f2>ji&!EI#lgvqEXute*o{Hmp|G!y;@NFZp{wd3CMNQQm*(XL!Dp2c$;)l{Z0>FL&fSled4^'
    'K#&>xV4kCJn>OY_-8pgrdpKKHFObO|YF>o(`=CcEt~9j#S9X}~kIy7OxP$zLHxiy&o*oG%)k76NP&JKmQ2WQY!5$2?$HK$?2?Zn6'
    'pr5nw79pw+{Q&Ic+8$0j9KGBR;sl^Wb$?*e4h4Sr^Q5A-gNS{wbB8|^WE~TQ-JeRj2|w<jw?uxaKlFSQ*v%hUzP&=7J}4D*P_6C#'
    ')Cw*1KX20bMVGpt#(fRl%A3%*NK1Y|8O12w4~vnEh_;?uCMWUZ#WfP)&ORT&NH-{v4|tJeRO8|BX8uk-+9TRA{*CO%5jCFqL+IAH'
    'l<N<7F$CKEbXnu#uwLA<4B`mQp9li$^C_@b$4u?#fMM9*l=grI;3Vqy9)MTJN_4gU{%!2Sabi4R!6=AV>+j)&U{mgR5349jfrhSq'
    'KimUEFCR1>Gvp?9JdnPlFQlk=*j#wf_&O+#11GxDMtl1m6wl!}#Dh;~0s$fuIvyybNu4x#&2NuSL(@erdGWV?Kebx9!YUo}%bpNX'
    'hlm(^iVI%G8_-qB<@wwgyfDYy$#6wR-5$jq@!P#8R~Ey7iEbqB1Ms6Wp5A5INaTq}VQopsT<j~7DtL2PS<YmbL9BhjIs|=Tau8OO'
    '>K#XW5`lKYX|D6H&^e}B^N06Gn?x>=(LqvMOAv}SAjWSw-$qMOMgq1IE=IOr69G1{RRIY>=3AlpC1KpimJ1=8_yO1b!&9BwH@s22'
    'KjRH#yoZWO1Y<_S$iuUTHvd$Rby>K5?4<6Alenz@u?L>?>xBhZ!S<|uu+&;2Odys!j^FmNu%k|fz{oacpZG^GW;<j2!<I1YgC~NS'
    'q~T);H8=mjw|MOHk6IK@<RW*daqm65syOi@>X#@xE5J~#55(6RKK3_AVu=fW5(Bw40%;kw*BajMGDS4P-xBMexs?&xG+XUH4%u9y'
    'R^rJ$ct1O<HXGgmbTCMJW1o#=@Io=_F00_0fON=&=OV7c@LGIfVW51{Q87!z*tRAsbx+;Y_&YO4N^Y-3Oc)SXTd469LdJrVgU$pF'
    'fG5)o(keSiHqFX?D)avn2pKQd81TXSh{syVfGKK(9``x!-JZ;e<a0k`jWnc(8OH>_##$PWV^>Jt>#DnsXy<a|-Whr*GMmM=Rwi$@'
    '0ErDT<W7X{?0N`Qp5&S6+v9g{(7=<m$y7#hdB}wK=;wn<V)0{=htjwBes5ty?~^JkY8Govcfv~e1Y~3Y?|0mt+8gLG8oK3LXlM%V'
    '=bm*mL`jVJbR2<{WHtK+aKe=5FxkERkQZNp@F%6r=C-Omt2(2z+MJfkKPd_ESEg%AKJjq`w=&OwKgg>gG1_G4C$s3}w{HZMOa)7?'
    'S+kEpxi1*r4uS*^L9(NDoA@G+55jtiC3V+je@Wa`rvqO}k1S>h=`hktwnK_oW6o&S=7%#i+n%jfV(OzDxS}m~LcVZ>q{LgaC}-Cb'
    'to_=lLcE~4umK}*#FJ@v6tEbpMTJ6~$4F{%)w)#d^55F!lv<qE0w6*}Lv6`e81JEpj}iuHmVi?O)og5B1<W4f6-8Dd5dFFg>Sl7^'
    'O$eQcb6THxz-Q|;iX37eTahn#R>tj%M_y+fb}v+C8d-bY`ehk9JUJ_<&f~I)XKfq1kD5@6_(K*fuC<<&tRz5q^k8xt_ZMo@V-;5F'
    'OiaEA#!UwQyw^5K-XKuL-(U)0AY#1@$`fQaSp!ErH7?4uw6|e}=h1WHjV%^Zyfq5$<XY4LiB5YMTC7BNy2&JjIsn^iX4ELcBMM_*'
    '5TNL+_A6f_mZ)_~9NR1r^hBjY|5&>$Q=?V31iI@{DiBS<Z96Q%yT%3|Aixu#JlPSbeHoHmyTAPG!r6EVlI^5K)6BR5>&c2qH=xZ('
    '%C_jLadg|dS65H=0XotS63E1%X1M>q!nE4aKXO2lN3v`2^iyluRP<5FXxJoXM480jg=<bdWO9WmW{WneRo-gy-4fp#cZ5sqHBXw('
    'KZzQmEO$+oYJYP}g1)2eEtB+wzQOjhqlOhYNL3?}dTg6-0MWoA#X!cQg^dzC0f8H!h7pElCDv^5m3;*l-)u`juutiViRAq3M7dgA'
    'WFQyGVAn2acgwf7c%r<<jrbGmeQNr$_Zh85Rq`sw1A%@_qLQYb7-q7$ob10us$=jQ#F!*nZT$2t_P@g$q)+U4k-OpBnlK*?=w?KP'
    'AZ~!Q3<WX9x<0zHxqOaO+G5h|aQm3mKHX<Fqry1Zn2*MmoLL=f#qd(X6+G_y*a(N9afOPD@La5%<6R~A#vuBy;r<;b%WTPkvL**^'
    'z2ZCTIHXvQc5#wh=*k4zS@))hhxlLd+p``^0&pw>OM2{AhEwC1#p?2LA(2D2_ZeD>K6z5M;zQn(o?|(H9kr(T7KhBx<290$SzHeE'
    'areyxN5t@yAUa~9muFzsW{SF%LAQ^2i*-5Uep6gxsY#UkM!A5GCR0?aZ(z!6cyH9`5#k;r)ifzZff%DH=<2HKego<y&be%n`T?Us'
    '(Z%=M)5hgoioRp^omDilD2X)-!j(oHZ6=lY6cC|e<An&MktSb=%y1M7mi&Tp4wDzlPzFWP?MLelWpELy5nZ&U^}%SQm$ECORNCpp'
    '$Y36HSz5K9PArLgfiUjkQc=aZpHu^Q*)bg-@jKU&klqZBxVGW7H05Aq6Fk{yG}m!3ioX(ejGLlYg^{+JJ_aE2q&lr);LO~3&3s*_'
    'nwNRVXon0EB}-Gy&QdbuDh}B_*lm+Y=<@+c5o0FLRjKW4ymi|j>jhO}8_cO#lO-!r<%;l?)7{osZ-8hBGF4b$G>zR8Gtyp3uBlOT'
    '8i=R&j7ml;sq9_yxD+#f#A9-nG<sL-POp^AFe(|n8v&}7R<3-#<TBmSkBPRbu4JFnX2cUlj|6M6*qg&`^tDlCK(C*U^;*Ks{`*fI'
    'kCl4<X%kzh7fwff6<fyL#Md}pB}44?1`;6ZW!QO>^wIZLI@W?U0<Na+BO1|UN`S|qZ!%>nOs_YHQQ4CbCiIVmG{RFsWABmxQS~Ob'
    'cSkOvBXx1vl{0o7PANtXcOA|M)KjA6DA62Lnxbqh(KhMpix40qM6)HD85lLu5g3u!mblJeObw1QQf%CQc7H$?@nJifLE=7>Ss<2p'
    'aT7K|MIiwtav0zn%<++CL?2-ZhmLFX%G{0bIRmuf8x)^BN?5+ZpPhqL(oPMDQ7t)wQVD#H%G!+IIGKRfSj-DEDg{?1D;|HSNm%m#'
    'VVU&v9X<`+BmnVq^aPPHT#4y_lzE+SNN9srG#o^5bQ^vM<P`g+DUvv9w4`E$=+l&57{B-%%+rYVMBO{oyuP^v2uo{^hnd8Ru4D?a'
    'V57rFXn@$>-hlt(hEFsj1%^YOO~4vwf@GuH6Q$uK1|B9+BN^(HupJ``z7hY-5O!qzYvV9?l95M&>XgWT2ppFrCRV{3iN9JLQ(Qpy'
    '!4!GDuxypOc@!{*4mUj^AjPi+>OMtg+8O<0#W7PwC83M$*{2x0)hH4f_s^ZQ&aq@JC2^xw+&?^#n&5NX30&@zqy65^A=Y;vALtdl'
    'Bp3gbz|Gje1TZCiiQ@&04;s6|#tR+V0CJtg`U-_-WHtHX0&iW)o_ISW<p|ThW1DW(&nVnMjRmnvx13SW@PuMmU*V_q^#-C6#~|06'
    'Lh+(Le(%<>BNuP@t;N02FOG7eeK9u5H6mH>a3972<uHGc!h?9$#F?zZ8!3o<7wOt9>3j8;G#~-Ofi>zpqnc@o_uU`Sj=`YJ6fX^-'
    'W8C+=zbs5F*1_lA)1;o+eu$TRtzvyfJF^Dx#mMFg(dbvgr+eIrvt%Q|f~2y?TkhS#1Lh$Z4P;P+1X03aR*_9AqXg>ceWOveGP2?('
    'dXS2StDsI;ht#KF(4e@^04U=t7xJ8;5yU22Nr?xH)u23!+_&l>ogb28+fh?G=gsnr3?fQG71_F%q%esK^V6-#o@D+Tac=(`>%d`D'
    'N2eW?5>l0fx!h2%L`^jW`X*+PSS-Zmt7GCkbjS#^A?cObg+qZ%4u^^T%ruh#J$^_eE`pvAKSi&#vV^%YWdJ>dR7G*|!jW_zP|tQW'
    'gA8JKLcqR+^l{`Mg%~@b9w28QXiotUOHB@6cSajXy~xyirNy%eRil;Tx*wUtRTMzc21Ga5Sm1k{U)`C&{aL9?h_3dGY;T`^wm5YJ'
    'bdR^7IK{P3bQzqE?1fo1!~;H~+ee?0u+vPbZFad<XOa+-%%kJy<vZRMFSJD}keyJypvJE$HMW4-n5u9F9!a~pdt8t7#Zld(HXNw{'
    'CU&67?!!_WjS^=#_G0XQxB0WZC3<^_xrtpDPcM@v#o4kn!myd)7x4>L&4LEiU#6;IVies=V?VU=x&sJIZEeJpZkNr<{ih66W$9Dh'
    '(^G0dq;87n@><W>%+5fj#ZA^spha>JVct?vJsStnJip_|*Sa^!a=GK)z2kQ*0I`IITJgEF9>S0+&mu~lNP=&m`thriuMA;0GLqA('
    '?zjbZ#YwNkQ5lVrW$fYh=>wi5FkwA}KHCbKRFe^)K}$+(^lFLqx>lTVlR&$bSnDV#W=s5yXvz+WD6D#!11?tqo1Hzg`-+!TqXZwy'
    'f5_mg$tZxjzBSaIhaekG$9?=ZJ6htBd2X3v(rVlaQGUm*P+HC&?NsUrpyxa!xyL3^Pz<(JON^|x!7iX8rBLor=I{bQ0P9W8En}b3'
    'H|~w`W9#w_a2pw=5_$n<9$uqF+sQLBi7}7AgMrs-^OW%$i?!owQs{bThOa4tE=tobZF0f_o|r>$3o(=OR!*TDZoE08c=T9=rDRqE'
    'V2Fi_fu1X>^+bqX_vhjSL)ldJA)F$${f30V)e5d`w8)kB)k9c<onw?aC7dB2ZTcZF6~m;CxyT^Vr)I-b+C5*F6U6A}2MC&IS2zZZ'
    'HmLg(X9Fkah$a!-=`GYu9@}W(CgfOs?4)^9=wPM3^KMbYW(dCtNdu*{&OD<hU2>GQ@m&&T@k(Ws^$vDwCfQx9mPvl3o{U?7Xu-0x'
    'E2ZKYhMU_Mkhrt2&L9Zvs2t*Jjxs45FJN`l<HS-b{7)wwOME@!;{HHduy*UYe7Ut9od?P18N1LFD^Oh`=bTMla4$w3%vm5Q*y|ui'
    'V$Kn`44IqsO!Nb&)6MiTaVwGsDN*aAUE)>RAtZg&p_AqiCBj}A+eTV128&G3GvDu{IPK9G`Yia-ARK&itB)~ph2I$F&jq(DLsR?m'
    '&(6Whur6-U^UcXSR!!a3V89sV@<@oYZR356AqbX_EAu)5Yu*3{%ouC0XH*4kWAcQ2)im6m$uVC=&$4M(iMp~BPQx?QaUJl5`u>!{'
    '4J(I@{Tu~BRL$?LwCflkY>fQk&c~4>GlRgP(_`hvu9N=ieC0*kH46S}=mfBBsHt(zUw`Vvlhilx(L*J|n9y*4%&cZ0&qf*Q=vU7_'
    'BV|(27h)!RbM-1_?VmMHZ{R^&iO?H!AH$%M>EO2U1`K<RMaB^OHdS;wd5^o8MiHeant9;DP0|yn=O}BOQiC9NXdkspv~J0D9<6(+'
    '8`0Wl@uX60i*O)Fm~(vg%p4CJRW!ReD%LD;x+>*-kpD9fe*NrHq1f<g^#g0?xs%}WCb2>E2R7uCzjD#lu1q(gnl71ut9x7_vDqO!'
    '6i@z(hPz-?PGq}S_a9a=H6>~=pNNvHB=jt-IR;l^bT;u^J(DuwPG1RLs-Ky7@3J?G9)80fd*M*mVznB~`bDvog)K$x1{)EeGcyRb'
    '28knfYW#&M?d|n=6WOpG_4_9?v!u2A>@vpdS08x%L5sU;8wh;odh={Q*enZEBP{ZkXC`7eGC@Nl!=~9AD4B-1u(2DkqjpQjvvH$H'
    'r%cysBT;S6sO0)uM7+eMu}|@cN94M?KrVTxWmB=L$+IgTIJxJ@eL#%MB(!AdFAnX5sJ9wNev}TcYiyZ~%~9Yi6B|4-t12$Y`2hoS'
    'jYMOJDsQ4rMulP7b-F5&soE72xLeZPV!Wr2E9m7`f7siDBwi_s;Z4b>`JP=>u^HGOt;Va{se3%bVeE9{%A6x<I53c^B~Ie_3qQwt'
    ';nYy{QIKcpk1UfiDP)D9pBc}(EHfcs$46NYu08w^q>wRRZPcz9<&T7839!G3;H;4I>l;Mn!5`}wf#G`4?Y)J)y=)jGb5P1h^N_{('
    '6luz#N75IRc|M;DClW~VSl~92Pm@}?sVj^=g2}V@%muAPi%>4RJ3KBAF%7SDzSzA<ubO7{T8VaHt=*J<v+B$ySxCNMA>bM8hnSgE'
    'w2)4A80_0w6JP>glUZf^itIRJ!ZfDVQ{RKxA!f}5*)0LXFzMGyn(iHc7#AZ9m;E_PS`y$FJeoAxqaX5!hAGP;UV_-Ynh7HIkh-US'
    'bY(~;1PYHUBih@zGL#T=$hg5&aX`oRfx&oEQsU<v^&A`??z<$gMiZ5dI}J03V=|hWJdB+r)F_Nwf?q3;k}IVNz)udFA!pp`3UZEf'
    '`jkhE@^=#a#T-4ZgCyfL>R<yMMkwD_vdb9A*hHozznFx}G?Y{xtGGJj!&j0j7%vnu3ZkSbJ7b;Uo<qY-FhyLY$x}pIZ8VBhs+NFb'
    '@Yr|RSIkkdSMmx_{B#0DXsEVl2{yQFV$@({ki7%FsgPR)_|<96M&foQ!`~S@kw&)LJ0vp^7h*HI1`>()Xc<Rmk^au?X?=Mh-=4Y)'
    '$QCybX7$+&K<>#L$%rvldts4lC5rFKLqar+Iy0k$Ot|$|z+aqANs&V<PYIy3rmU&OYTxnpY7Ff&!4soyn_GLomiDt#5v1wW5Dr4q'
    '$t>O%XKlO<K{Y;;^Jm44<n8NT9a-MhoX~46&TYQ$`9eTPI%o2@NNFx(<hs$_!5s%dkE5o?R9}+P&REU$4CLD(su2y>3dzbNLtNUG'
    'h>NDOwKwSZ2UmPtf7RtH*xtL}hYm`tw2_Vxb!Zm(GR3uDar#E#Ug{fImr$_9O#qX55OR`C&^9^`G=*x4XXavD2U)m|khWsy<)?re'
    '1fOHMS(YI(ZCOi6Oj?bWlG*Uu8I2N*4$FIJlx6uLz8)HT@8~l~+ZvvUusHtain5io(UE2xStqQnnAUO|^AuQP!y;Q2#R<gp%s;dJ'
    'HG>#43Y==G*AS{!A{sY|mWMdR+tl&VNcBY|va+evAWMy8+y;Z8r|7`MnHzhz)=uHy?JK)=d_=K|PP%6-(e}9+dpoJzD-&c#?bfeU'
    'NgB(3OmZ<vX_#+DRqm-`umHKD9U$yI{VJwQ?zJ_L+SX19Xd2Q^Rt<%`W&zZT=-mkrzVrHJk?d1&j!-J`Ar;!y@Pj4|l6o^cu3d!H'
    '*=3NJEWve=Yw?1#&vl5g7pa8N-jdxD9mQN(BI;5ep=oX!XVM!OmY~jxl6on<q-7k-#gM|IEu^8U_8tbTh1}&TSILzLQZ7web1`D$'
    '=FD#T*prh(vu}7(nTCzPpINVHl2|b>Z1ahfQjCiiXc(RqpR%Wko&!gv(G$2vg|0X8hzuTp=i<_ayK-Oi(|a}=>4bd|6Wj|8wEgtX'
    '1o1;>9lvzzX|3ktYHdfn8&AiPSdLEvrNy?lWM0>?$GL{4h#;%JP~08==|%nvClRhq*()pxD~wmj$D#o89OY|KzFv%$b;yJtHQ?s>'
    '>|&=T{)hu&V{{f>e(~)MqB&~Z*?l${?8HiY?~Z@a@zd^3I)t?0=M|c=8;uI@4gA%+&QR%gM(m=UZJs8<HF65sv!pC)<I-?ct(fxc'
    'QqCyani626RK8q0nkDyD9axg!X}0Ic9jgtWiw=dDfS_^;ks&GoCALTLT^TX%_eR+Y^HCVH#u$F*CMD2nqD_cP>T{HO{=BIz(mu`b'
    'KJL)2(<j4FzQ?(P@x`(?$UkV4=tkz727>KYv#jO6{&YkiG5b5*XsAdS8(F-&m9go8F$UqipS`?+`Um15rE$e4<h@yVQ5?sT+T+r2'
    '&Mb%>%Ea7de_%kqe4oGYHVLvemZoxTwA`AwFJI!q#*vr^e%2z_JGLi@4EvDc3a(lF;tchYOH0=0jAJXkO6H@9q2$Cn$KpGvo+TMU'
    '_ghHPuNa3cnF$>xZgrIz?++`2LEYPi|LW1=j6^xqv@N%avumFwNIznyEqPWx&A?fmIvjacODUSs-F8eXHaPhYtFx+j??dXWo)7~F'
    '$Q*S}iL=JYt`A^;bQ*}WR?7leW(Vx=sIx|vt@ZB!GO+~-%R-&i5<f~z(>$I~@~objTJcJ8t3sYN4&g~#B|#)s2lA|*NaF~c=NE^2'
    'x5oKJ*(Qk|(@7cQ&9}Y^W1AB$J8^}x&n)zLZuSnAwCu;S-$9=h4bJ`|z?Q7kk0LF0)a+Y8k7Ycdi6vhB;;HSvPBDUS_+>9PPF!ZG'
    '`6MAQqhJ%mSCJ7#*6Q&E)XVbUvq6h@ruhv4&qkH;x}JS=6+?`mbKN~Lf+t|4+HTU9TShoq^4zj3Myl3?p0)~VHVnD(+0XnEJ=Vsa'
    'l4#)w^~27j6GJGE>jC+7&0r$zasjuS3@<WHVq6~4JrOnqhMRO8#4L9cbJ3aAyl$rtZ;YE&lx|D@!!Tznzg1OyMM9w+=*#Pd;>+}A'
    'opGh$hrVL}NuUM6?)F0sUIFO?jDX8LADDqUSD|F1&l*_?^Dh4r=f;=U=jE*2GC6-4STVqkXoyJOAcdjI$v)b?_}$4|PdcO8ti~DI'
    '=f}cdW>_}tA*`a|0NW1=xT)|QH^w()*zAl7urc7T-am+@ieojKrIe`fVl8mz)N#ldsM4}_aOu^!(04}P!MRZ}Oa+_%L;~a@zVbz2'
    'q;LwcAX?LC;-jUTK3Idv`)Q<Q<02|L-OOr&8t0N-HNo^l{2z=<YBN)HxrQV}vrQy4{uab!_3UbI+!YeOAWAqyNGQkDICI*E4{1QM'
    'j1LrAP1Kz@nJgg%k9c~}M<i?QkXMK*zvFr;rsK?lm~gmv*yB#jcj5+xy&8jQpsjhz8&p9V&&678EmQHp?<i1{q8}+4vC<oWn>q&)'
    'Y<d>CwXeX9O725_p-PSB?>Z(ZL~G4rhaqcmnBs=uGr2g1@+*eR1FN-k+s3yGC{lr1U+Ft%#pCyyk}<)}N@nhp=mv&Qv<%hPJ0wl9'
    '%;@K%*sdd4WgcCBFLS7iotcCBS%fskFFV!>L><dSo~BHC${7(?GE+H~MhcONg*UTw72>$8D+Ol)q>op=MRIv-SM=_(qzi)T(mPtT'
    '#;20nW$})k?6LA7Y@17|Z_(>xz>`erT^eGAZ^5pfKf7U9BgGvA29Jo$Jb7VB*02g%BxofkB!Ha*XNeG-d0t8Xj-wZ(ck3^l>dAU$'
    '^Gyy&C7CS}^Az3~pK6&dL8lZnyVf<aI-`VzVo*CtQ`{J-JWC{@SavuGjyLd<rA{%oOP)0{818L|BORS3wm>F}{mkGBf)eaUTHaaS'
    '!1}hM{N6<)Nlt<U8;-i0aHfL6A+qy~HzJF#Rg`l$Kwsx6hz4PMM*b4{%h`xN+Q}I;c=Ydc%Fr=$>&FW@%1?zd%ytMPt-{7G*cp(l'
    'yvE3QGA8>6rM1>fa~;oEKjUs`Vz#e#N|ILXhKO*)b={Fl%&Z|<xU39ufp#kyI7ie`sqrx|{6ol<QME<A(k?&qmT_p}{SYZ?m+4+x'
    'rRC7a=p=-<xH(P%H5N>I><*`IqOUUYD-mH1^s%FxO6t#r@thL08DOnLlI@I|+3~EU$*UzJ_(mhl&?q{rD5G~48#?LsJHjRGB%2>1'
    'T;lwz_#MzGjajm}J(mYfBvRn%St*LK1jYxzma(!=d5tnGWI!y_-b7eNkqrVu^RRkSfi9Y85*i!PT<eemK^a2@kBc}`M^7ZGOD;J3'
    '3a%a5MD46z7$Q1?fl;{+Gs6wCgE2TnF*o}gP!^+I=Uw!sWVhHU+3`71DG@OhSH<qhj<UGw;jCSkU>q$oZI7$T`IHcu7t&d*MKM<5'
    'u0Ww{J4=kjrfGWAV&HPCWpvs&iAk?;`RcFCpWKSjSjZ3yen_(1#iNvKFECR5;o+Xa<eQ`R%Xe&L*JPYgafR?m7w0w&U6S?nl&L^o'
    'I^n1|id=ddZiBBBXJO0*qKf3B*R1kkEZO6`(Z`Bq-B@WDg6?`f$ew2m9yji|lESL>s_8Vx@P8n?OajZ}sq6*%RH;f&EH>Y}-ybND'
    'yMm=-|9z&e-OLlCeO@0<BRJO^L`?N)hzbG_8|bQ1bruFC^0YF%^ybK-+{jAv=np?UiuE-QaYks9%rmi}dx6%LRLoGN*kC()W^5u`'
    'dX?_aK8XI|YgvBE+-=8=NEV#~x4c6(%rmE^;DoM>DKLtet+^)Wn2IRUW0B4`&}U5}m$jf4B(gR$bu=buQK@1_0&&gH&+2P#>Th{8'
    'IF2n-IO1L9I6=ie`t-Tp;L!%wnxhav6Ww3$GqdIT>rcm;mlw|Ij|Ryz>c-e}v6DPojE)r{QEp7e-XT8`Y-D^e4%+tK`tUdh>3U4k'
    'GhYccV2EfQU3oQ{*YhP2v~WxhaJMu|4SK0P#+0tAX6S9hKcf)3G%NWxXB5e!<8lXC1Eb>1@QdQVlSy0)=2V&ZOm<D1IMDq$Rua5H'
    'me%O#=y7_SiAb?_S$EefOKbhO_$>bSSop?7YnjkQL@1BzS}9UdwX{?vx0zQnf!Mud`2bMaqb|3&eacM!i@4B^=(VB&9QMN&k>Kx0'
    '{~49!)I&wTAw7erkcE6lv$07Np`G>2TLmyDn(+n0>V;uX=48IaO+l*0%rK;mD@?2@uK1k^0hyXn9k|EcbI5iz!$_C*r4cHec<>Na'
    'Z;II(Z%||74ZMlOAbG01w`D`Eb6rN-ggsZEnYX<Rgi&6pKYIgoy3@M>Pia2XJX32Lld?F}Y24%gN@z(1o0xss_pW*hJv*-F&qd_I'
    'Xz&P=y<_HaloI!Z8DlE{T%?Ric{Urpea6|tx?^mSC0l!86^K)pRrZ~!8BZR98GY|^&7Lz^sFNf@M@UGxh-Zvp`lxo<O|P;tp%#<F'
    'xG<2Z%>Kg$q<#rLVSd8TswqLR70Duv5nApxP;o45*rQw0`Fe>j1U#<B!_Qad!4B#Yc*!UYRx-qd06mDQsLntj3kHM1*KA#Zt>us='
    'p9rb3Vm(z^5hZgLBrdq)qN=a{1|E$K;sZwECQrAr-WwxZQ-WIQ&M3B0UO_`qk2RgjuJfywiaG0$okW3m?THIx#hk)+RP#&m*$L+W'
    'a64|+YgN}n+yambL?;sW;9|a*-R9dBMijq1wg_3w=WY-t;zJg_#VH2(aR(qe+aC-5SmG<qZ$6;2;$&M&CQM@$P!vRoKKzt8scnoV'
    '2ccjV_dsSx9H(O`aXL1XR+&6{5fW3OXd~J_WI$<zNJR<bA7dK~rsM$;y^|NQQTnf3CBC6|5iB%LnPVDbgI;R;a6$ul1KCJ(T%4;Y'
    'DSPs!p+Gf&tcvTVpAzh*Dg}zjAWtT<ZYM9x3FX5!V0coqjzYelBEbs8`BNTIun9t}#%9Y>Y00vw)SmmgHBxg1Vl=5LizTKwurRuc'
    'I~-l+3%&EKAz>L%%SZWaSpbc*VR7@4G`lA2TB!_Ja#Y8`a%Nvi^bLrIX~5&Z{7M=~$4&}0H5BBTb-d98#$uEtBGpZtr8%>H{id=e'
    'X~~N93Q$|szxMJLJiAJN*JS!!Ep772>v7x;p>t)tgB4A5jRskQB(I5sw5x9-j-g1$_R2&sgECHv7Yu_eb`RS1;ymqf)s#$ewhGt('
    'b%*v0F@PbxftTALYdR^ZqSn1oa{#bRl={E>RN~yQmZ-UVZ%ch}qA4NjtDsk12~x*=<jXCmr}Ei20#+yBtfRN{*R}amUc}MoH>RJ-'
    'qhd+gIGOm+%2Lc{L4sMQZdZZKM+nYJeD4$lq+hm@)MRii9m@#pin0pnD9To$rehKsS7lk_&7F#Y$_lBYhSSf;RJ8!YaU8{7yP_D@'
    '`O3hCgkiXBFi{PRY{EBz4QuPynVVB}8J`&pT@op4opuhK;g_K#emX0GJuT|#D<q4YqLNAM8Jd4i%+GOW8|4tA)yg1ySzHa56HDfT'
    'RLS=w-&Dq2dt}Vs=$1jQl$I6U@hL_+8w(?buYwS>A*j6W`nnkCY11Fy0tTFAXxg#HDOpk;SJ~Xr_WLARFV#LWnt(Bc!NAS{c<_(z'
    'a|?Z(%MNiuq`KUU;iqP~+02arPHC}W2LaL9w0$zSV=~#$>-BjGCv;2<`C<_gN1v_1xZdMLf?&=lA|WnE$-;QTReH8l!WOMBQ6W~e'
    'N;Pf>%iu%wcZ}npgUy;H&$cV&8*77Uhs-nLCE{a*FqH_64CKe3qblwkD8!Q+1{cd%`FK1m2vetIf|04)M!g2xBOx~}#I2hpxZ+f*'
    'Z{Txicto$LomG;yRx;PnSe_lXF+Aq&4W3JqekBn=8m%c+URtFdAB&3AEzN)e-e6&Oh(%Uv+9Jn!y``({_DY+lD|ag@7BQ|ewN!0D'
    '_)D}u%VN?D>WX4C(Hgcy^b>_otaSnykpiy_m?H+TlI2Je*ZyH=-Po0IE8>fWSvqz`cs2uli{FB_BQDZRMTh(9UlOd?7g^bdls#4g'
    '8>jA9f;l8N;0rhvZ@M2O+ItJQioz{>2!;wRx(^<sA3|^2h1B&X`>tA(dS^#_CF4g-F6;L5iPIB`8<UyXo7VlUTm4aKiR8CRRk~6+'
    'ZzI+Q>CxEoR(S(HK9zC7^3~9Mq@GL7V`pt%q~RgvB`wUgUkiDY9tF=KMsp4D74nloN#EJcSbKbIt|+?W+KYp(&D`~3z#~IZcTQWO'
    'fnqdruA{%bxzvoJxc{vWY-{CmEQ3QksIbfV58b@(SX-_juqd!=pSfhos_aQBaNHXNfswPM&}M_>I(yPG$OAEIF~4w_VsC(nl3Hz!'
    'zgC^|v!w+UBim7BK01k?e}lJ%-ccQPycZCOk-oLBKgHWMO0M%C4c4;gzMn0lp+%eX*<i9^7g^sRl1@H8OX+Pk-XO{B87QJTi3RxK'
    'BbL-o&I=4}X7sZk6_Z7KgNMc#Y|CF1tE41?1qj`Uj}U<)p}hiGH*qeba&bH4sO)eE*PbHUtn8d8%5p<{mqa2<+aa^tarH3{R?kFd'
    '@mEwrcnzUK%>=Q+z6XJ49-k^d`)yTReGEaF?+XXGm&d_);5Z%l@+I9UvckHjgv*~bTOx+R9I`@FMniU=qPE{xfScDJ7E~B%Qe&AC'
    '+;(OwB!{XooH1D34H@Sd`*>JS(bs?0hj3TCWfl=``;`=~_Q@<)>Z~?WqzKdAg*<!h6ptWdFltd~7VQw9v`yw97XIXZB`)D#{vfe{'
    '=lLyrbgrgcuxPsHcyEZo$dqm?Znn9=WD~aTSWK+pBa62C2DTbza`cYPOz3!N2_UeIte#4G8>;i9B|_k1$^;Xi?U2B}O?5))GB6^_'
    'oa)zZki>>={k?J>7ED->_mK>XWT{o-a6!7)*9)0-vr+a-uTe7XhflIZi$~42^9eh4ceAmMCy1G4mfBnru(P~kBA--IggZfftGK6?'
    'A10zj)j}S56OSx`R7Rt^i3QN8ibgU#V6tF17dJ?=mMmUIZ4SSJ0GK{sh%I6LS($}N&pQ$V2pL%uqj|hyde$oTs<M8`9wNojUJp+c'
    'Tr~Gb1)H=)cMYG>smHJn7{1R}qg7&^Ia&<9gV*<t^saDOW5=}%0h^g~1a7o*Der2N+jxFcSg|%A4Y0io+<me!qtog&IXX=EN(sAV'
    ';(Hq<LyvVm>LG^1C>)d_w5ZQoLf4R6GP!l)PG%J2x;}9^j}PJScZzK^DPL>XDdxjU3L0WD29tzNz(|&GJ{!_3LS7KAltXxCtfvWO'
    'AAYB04~S}=lO1v6$_B?irKbE4gg^HwcFOVMCa5z6h}M&08^?x$7j_4FNJ7awj^-PRSv;Z0Oos?QqDbs<(zrKvVN0DD#zdqMRa^X!'
    'lq7WYOf}KM&w0o&cY73Ft{{*2y%H6FoTHN(N_K=L!3{gz`ouSY`n?<y-Li=ea;&jA-Ak6T7Oi>|EF`vJ%$rnWn>_G%yCebkEOkPV'
    '9zl(`Rkig>10zDBqOGf%nRrP8NJodi+zCe>bPq{;G(<;RDo}S%u9i+n6!s1qhq4z^wI@!;Q+y9hD(X(Vj!g^}y+WJy{2FxP;{`br'
    '(N)j-Tm0*Ll`1TD?<w(?kK6G7qwZaEBT1Gf%dKz)gsIwl{~H@U;;cdgWtszg;Ego@G2Jthyjd9xcQbv693fU`L^;zbMl#UdnxX&<'
    'jW7Dh$S8wywmcmBXezTlG5$|raq`3Z2X7QKJA?kHVLQ!oW@kk~5#<kafH{)ma%?D1cnOVAh9qu*`?nVk=%FEa^5Q$>ud(#vSNr8q'
    'h7k|!UTsUF+cS5Ef?1In)TP!onQLG*R|*(okc_qig7kU2IcZz}+&A1I_}~siE*{sNr9U;X7d)c!L^Iimn^^k03L}e4qS+S;&FR8y'
    'wV5d8IQ}acAcB!)<-=7%Eg}y|JMcj)=6E-~Jql8k%5j%VOSNl8^#fYCTuzo=UqDawM{T=+|A1x8!b+S0KpWd13{adbNxFI5Iu1Cs'
    '$2~Osn%r_Gvt?@ASHDZfl(#05cD!DI&`L|Z0Sn<8m{=6eW#0nm007ij&&MezM(Mdsr#?`O8R%5sDHyLy@5Jo4-|`6oy0s`Tzcx8M'
    'S^sARWk1q>IEcF?3+)A97JDx|lQM2WkYLGY>q)BMPOe(QJXWRuXfjY_<0#na@u-U{V19{Xj_E{HVOzd^-?Qd!j=iGOVXLxN;+8<Q'
    'E4=~9wPTD+!sw#?McO*G0QF}1M7Ud=O0p^RofcmMb`lJPb&1|98mAJSc|20^j?#GzFd+2T@dP#^PiDF{t3XVuz^rB5@*PeB;SuT*'
    'yq;{1qG#Y{dHRNUd6O+8G~>$HW5Z6mqkO{&rP%ql=XVxzNDnv)J6^D3Z=eCd5i3)7SKJns^bDh+`)4)3;WLdFWVG32Z-nrZMke#c'
    'Tfs^fFD7uDl~e(%Q^R3JEAe1v`_2*1g;6WDsOnn%WC}Z0Ir6oL>1-wQ;{pf(=b$lM7GX&|*1|Eukg8=A*cFG+#5369BH1g81-JO;'
    'ILHK|-kSe%413gOJ!-P}m)dtSwqDVwdyOv)>3YlhD)w;02;ejD0?C|W8M^@dvYc?A^diKg_g$Ir5)jz3O&N}gK1CuiBqUi_Ysabl'
    '^C`ODaeYkmkMT8Vz8hWDNH>N>=qFYcz22{n8qwmqF%Kr5t<85saiVmr&wCtYm~5(9OEK*yM2YO`{sQE0B|3g|ct0a<#F_~XoF_H@'
    'kXSb9r7|5UFCzo<(Q^#z=anSzBO9t@wH4WH(#V>fw8qfIJ<E4Qs%_{USOc5gLZ>Lmxci>6$?yGcpPVl>1!T6c!K*DzQm-SfhEm0i'
    '?CS;Y9S03MWKLD6_RbJnU>keiRr`cnC~klJWNhAtf_5O!wUwv~_T%}8ij1g)^@6}G?*mnmV)Tj=N|uaviI=BW2Vvan3sUqPSjYm0'
    '5?w4+ezLvXII2gqxsw>r`};dvTO9hWS*E5e*`V_o!9>el#~W7CN~<TwBhC82RTBp$Zgq56_aBL{c7S#POd4-D|E)M6YIGP;1V$B+'
    '<xMq(r!|7uIOSiEX8|~<EkU{byUavygR9%z+m~X|Jvk|A<!u{30dv#tC|20APOM;F$!bqVtrasK6b!?cR+52vb#>sF>PDQZi3rk>'
    'fEJwPN-s!6iLf@?Bn_0*Mf+v#Hdf5lhJu;Dy(PRdkKfIT8d|QrxU-W5Ai1R*bS423S#Z1)8Z}Czl>S~bGiDFv@fj&-<(H=p7%GPl'
    'pC%Dmmf{CHgnwtL-?JnY7%7ltMbWb{&}Z54y+lV5f7R;WvwcvE2UltGD_%4$iEuyxjw>ER+z)M}>?J^G6H9sTmvC+^YU!$;hk&Pq'
    '<%LuSe?Oky7aX)Op-@VJBPj2_4BZ?MPqnW}YtVW&LhdlnOlOg|7nIFLx>Fq;+@&Fv&fr`Nrw{@{bQ8<-{bYkWTeY#?w~!JCYyD_u'
    '3{XLKb`&~wrnK?Bp{P#q>tj6|zzV&hCq-!|VreO5t;S#a!F|gDgfiK(Mn4oMNix~T&sus{N<b7h8Gjw+2W~-OUo-6;o={p3GHk78'
    's<jHFi<p-<78o7;?wr=x78}7G1EYH)$40bmWbP2iG7=s(E9kPNQDX{_eH@^RM45=C`ayp);y8zK=F}B4F&Sf2?CirliXqDi12~Bt'
    'ur@l3H7*&*!)e={<P^44S;s3F>();pR&FmqOU3il;CTW#g_`*4UBW=Af#N#4bOY0x3?>|E2p%d@o=5L*n2{mhh*xGWz%NN3AH{#P'
    '1~%&>Y{|rbi_?Vs2f}!|T>?VzA|C)QIS$O4C6WNQZh$S>sbo2NY?a2s2&ky6O9;@!0<3uDgluRvIqVl5L)6_g!<NO&R1g+EWVPzE'
    'Tw;@x(tqu*XQP#gH;I|s@s?JkStYA!;u)Z3pc+`W0GzZ{C?(EI1P`;<Y^|?wIX+=W(=b!Z&ceY!|3Bs}s;kyXxM9r*CvjSqmQx8%'
    'C&38lVqzofh_0fL_ZH`p?@R2sNaxl>cD0BDnpyf%{ce*_81LiGszZs<p<30+R{oNMM%1K7A00jXO5AENH5$3UM9ZyNGsB=DM?cK5'
    'jAR?XDNM^ze8w+iesDW)3N1#U1XLsC+c7W}E~X+@vCL;vcpzX^M2T8wQljFoloi1zjraasiMnxr_x6OWdNyz&ZO#c}qq>WqzP}6='
    '(wauU*~~pS&~myQEz!Z&H}D&*L>*R07{|vrQf&);u_Iv}km{0w_GCBEXnhCs@9b8gmyN&dnLU5dKfkaFGqu5|T5HV;vk@g{`H2ah'
    'X`mdS`qAW_YZ+rR3Jm72E&$8w$>EF50uc@`{%XIRc{yD(ZkINEWSMG{J@5p;#P@4S%{o^6AsIqksbls0S~9e!wZRbjpfQAFr4dY0'
    'J)W>MVl*CQE=ztYuKw)iDUT&BPGr21cbrfdc6W68@_5Hdl-L1kv=)#lVa>Gu0$*K=*F*W6;0oe<dH39#K)`sRr)b!5m-n7UZ?N2A'
    'VFt~hy#aDm%fe~H{u8q4>WRoll9@H5n%U&b?V0@sG-O7e%>~nXyJVQrINi;jtPbxPAe0NwFoMSk0j0`Q^~|Q?q?&J+6a~%%up#K)'
    'acY*#Vr5MiKgxJDT|4=N(XW7410B=qB_mPWlN^!;bBZl;8ZOsaBce112A-7X@#jzK7e=g<===Fzcrr#Cz^0=Y&c<%>o{i{v;xX=6'
    'kVaCOElCX>xOm(%_hbZ$_Y>dGL?V<71C3T^Qiu^=CUo3+?^0P$JamSr<1aA-I$6h^c9;_-gG!-lTXQtsmnCy!v35eY03hfZ^}SWE'
    '%zM$Z@=|6~(WuGxnCh}T%bQ;ur_`67%XMEe(QqZNa*AMs-&vtO*7wx66MaM7iOo;b$;wBZ7Qc^cEUUbK_)8nIrk<$GC-W5&K*-^V'
    'FZ2aK-xBXrpKHt6HAhl^PD3JFexk<ve!&M*o;|@v1Mp5A+1ZKKiHj01a;5=}@4VXbxB%;Omz?|J36B&yUi>CM85gCD*v>=QWyHzT'
    'ep{3~rAC|NG_gImXD@Q)dwlD2CxbRZ%sLlk(;bz$?1|__IKaK??U&d1R*<m~n%ANGW$`SV=ACmF*m9=l*l~G8<Kj;~q9^;GIKR+<'
    '4xIx#Bc1$j#&G+4Gdp4lUo8VzL><$J{h~VBnJab0^AaaoR6AonSh_p(UkrF5pmNymqy}5)hHE^O%{M>wg7lEx<qsxNwKjjZC$JZH'
    ';AoQo>rF^;rp2$F<)>-JSJw;jt{NRL!dr_E`OQN3JO}`-y_cLL7_lvJLo+mv-;a$i9EQ7GvInkc&?nEMvj0|1c&YAeQUE)MT9obB'
    'j0I_(QES6go7aGgRg)6gGYa-Vd%1cB5LH}nc;_mwlV!?9Dsc({#J_Xi?=WNJ36NqwNo?*}W~~7Up^cxy8NVRe%PT=;jf^?MiZepJ'
    'q2(S7+MXR+d%>5c+1Qlbp`!@QMrdxBCw6?YP6p5mzBJs#l@DYJYf1&qxB_hPrNwc&UXZWjKF5RL*&NiK9Mnp9A+e(O1&9CsPS3%$'
    '4336)kudeIY?J%XpTZy9ugIO72wT5YXJrw+utF@l6`UYhK+9ECmI2mtJdbPicVEl)OsnFw7=)ayOokLUT(qiRs!7HRAW^XACj0+7'
    'IOUQ;m}#HG1abAkS-H&gb8`z$1g)1K^&9Bi1|=TS)k^yKx+<7KSG->;U#f8jt~DQ1XUh&&<<D%cOl%`1a|mwi^N*9c{W2ldnQ(zn'
    'H^KHZmML9J<<mgH%EIa0;s~uQL!I@XWgiFY93L8~TT4BH1D2r-k9+>IZQIx{^O?=Te<oAFxxQDSo+<5nrFTA?q}m7Bd#z>k3MwQ1'
    '5c%Q<5w}(uy;ovP0MTV}#?aj2(4DL{vw9SlYV@>m`PenIL4UlrqkVfx)?h1M8P<EOk#1S1|M8GT7u6?r$C|n2U*mTyjwE96WFc$J'
    'M6Ft$u-u`YeFOG7PHV)RaVYgC3hxgybKAqDwQZ^2k(|_cJN5Z%yCm_@OrDOoqPq*=mna?Zc>_>xb2A;wpr-Lx^tzrbz?RI$dSy{2'
    'c|9fW2cwdpIaD@t1RYsMV3;bo=QFcyo}{OsaU3XqS;y9%_~ZwndD9*x<jBO$hF`Wk2GHrhtcTKWiyDli&D0IwpKlDwISp09_j=?_'
    '@mYC>`%G?Bth05+3(kIReeOAD?cqXM3{jn1TKz?hOQ0bh`7Bl->JnsH(W<bpzJ%6}dLyc(DmsQIYoH3NfJ=ost<3U*E`D<~Vh^1B'
    ')9u|Dw@N&yd@1If4J|nv7FIwC8r*)R*B7`Qn*y3ah!kRbYZewDmLOS21VA<)nPafxAlqLJgUD=`NOpE7cs`E8)qq1Le42%2kPWy2'
    '^GS+*MMU+6eWIdoy2LEU2kUFq2Xi}<YiSa~TajI1?VriGW<{GfvF)(rw{^*e(?(z90`9NL%qd$`&Si1)V&7*eo_QXi5wx{c53F@b'
    'wQl6yOp2|(pg3n3U5v+0hI9QMb!DkcQ6d1kQ^9MO^iKwVye$pb2Jc!D#kCYI8s*xM>LE!!tkR<Rm)0@m5)sp*$|*J!B|bC4?2sB;'
    '$<<LHwj_azXw?mTjIq|3x8zr{ELrS}I(E%Wl|mA$WBspjjAXeaI6Nf(kk2g9Nw&VDp^4JenPJrru0f-fYTrJ4muh@?@;!CZjt4YD'
    'jPfJN<cb4_6KN{LanzaRmB%v=JBR)0_+a6LifC1ZJAigb?ry$SYc`$;uQu*4>=r^*_OTFnXHQV`Jcx0kMh~$FXD#>3dX(9LCrXZ!'
    'n#jcWn2>6!)c$dVBw}922FJG=KjoPqlnp^8x`Z5s4@R#qAd{pux40HPVb)4+=9Hd@TUT;ODB8OElZ4T-3VlSYSQhm`_xzUzGOI8J'
    '=iY9k{%(^W4g0qX$^f-0dm9%^7=MJkUejnxu2DDLr&)y`|9%$5<8gTU2Up;O$EnEeIP<D>M9J#F{rD;OY$RFA1Av0u-1{Yx?T%s$'
    'xlc6sB#fmqmPX4=5U{rwfU+r1D!yl$Vq4|q@d(?M6HDyjL$b+8aUETuqf*9<^#b96qlWI}pUa;!O5$RSQj>6`;$NHSrp*3kMZw2q'
    'J<sg81oRix859-1z!FDuEty+&M6FrSF-;^SEBR$?{x+l-#0J~G#3$!QD$kDJ{YKE4t?7*&TL&$JRx4dd?P$Y#{3Vjo$JuiUHD+O6'
    '0Zd2uWUrpt9m$8~iy`fHX3>=K1C+Dt;UZtcT~mU`WOUCGt*q4U=&Wu1$);>AS>R5>vIb@c@8A6r$eIx)=d9tQzm1QWIj0@T!K_;`'
    '3S(u_bJ#-(TcAP}j_ZzA7=>;;s$Wp!*3Kr2U14TXo5q2&Jz2#Jr=F+2W{P@?pK}^$6MR<~J2M9gn^0Vz4Oz^bSsiMi*s63$ajMo^'
    'zQec<$S-DNwo)Vv6IT04A}@X9G}H+<hm7%EE-@bOMezM**E`X8c?wu51h_-stgF4j?x@rcGN|JLv@A4@xcm4js(PtvFQ8Mv(f#h;'
    '=VQZYS%#BWvQ$y`P>N=9h8?Y)1HR=39G<AVXpJjX5ozX80)LPzW?hK|U4w78nhSpS#M{}Nq~n%76TErx8$%adce2M(9w>px`0q-#'
    'pp1(q?k4Lx*W!Zx-6v;@Q;*SKJGErIyW3`|tufECgw?!_j-@_{jy*-l$H!3wV%(v<d{C{sCT>ik|B&WoW7HL|XS7~TKUH_OV;p2m'
    '_!A^tGl1FWHMS@fK%;;Q&5Z6-s2OTeYzDR6-!YGg9e6)Bf4BASMScSpTABaZ1jM*9r^`Y?8uuj3@V?(*yjpRFMxRgqNpyT}nS7f@'
    'k8~7c7FBz)mx-zzO=%n`kSoi$#FtwQ5Lk(3A+9q5T(UqKYPuGkZFI@)duKarBO#}ZvAzaC<Yw1Yr18s-n_pJbRbP<T{AW*Km#3;f'
    '6o=KaPEh$RdlN!lp$ptEqOFRl9?tV<5E5Hv>xueI3ta5eyS)G<QsHSfCWYpU=*n7iB2%0>Okn`rmlvFsB`VxPmnu1zShb0C-lD3)'
    '^v)dW4W@TzqBns9#|cg<<<8u5=ni4~^hP<gU*LGO_%=#ZkzachY#rIi4Ai+MpYajBzQ8K()0q!(tCfvFEm^)<ku|9)euRnU^&d<_'
    'qY8bH)T)tMRQ~x>R2HrF`yWR7Lf4pw5b2be0VphPGkmHquyR(EgAn9#M!eg>VzV^(Yg-V#6yJn$953UBKmUSAmi6e19x(74nU)39'
    '0jw;Kvwdd19HPH`VIKqxa*5R#eyq%wuEe$EBw#;V^>2?ptz~p_q;<k*Szl0;&qq`Rkh+>YnqCtS1%Z%Q%}IKUO3!||9GTypI7}>>'
    'HA7mu_ysJd(Gu*8^0J-~M;w|nbEkmu?-maa-oWa^Q-364R8nx(FBh!NxVHDE;6n{ClRY3B9mAnHvyK*n!DK_W0U0H-5bN#1$57Dq'
    '&lTU+(A4#gn~@_a)BS!&N!q?ZBwF@E2C>|~)Ytkg7{z#eN)#)sUwa}5!@%0KMUOXi*B2bIkG_$kv%fsV*$cicP7Ng0S*Ys}N05vS'
    ')bOLfVLejGu2@xS*_skU;@NY;8XG_E$zFWKA1;nxXG^v(L8O>L3#mZ;m_?>D6P&80#4+7OhMsxIgGu%%bWvvGBv_#re01ykldPE4'
    '^#m%zsqEx@#`jQJ|7e`Flg}^>tvH_)J)<QWE>a?3AynvhSt?tnE1BytANs7ndv+1=ur*jOaW`aZs!@g_pPasMmzjzl7yM7m9<BF0'
    '1FNk0@I8ioaMp<vqV>2Z9FTJ+k|4IFQ`Awl)}CAy;5=RQ6|7u$RP1rU5@yj=*2L|c1g<Z9*7_;KEq;Huft$e_WiKnY_Hw8}m9G!x'
    'p2@b7Y+G6M8j*nUb-e;K=p8Hzd1&nUXG@MeM=Rm*rzUqmCTfp^^<g@+I8@raCC8!=^M^lZV!y9h7$E7)RL~X0hbIo36)j(#7T+*R'
    'vUZ6zZjl*YNZ^Qzyp<FyA1C|(<%F-gBu=CCgfuNZ#J#LbJ|fuhHENG)RDP-C>p_O?R09M4e=>B}CLOd-?c<`4SUWO+>QR_t-$*<m'
    '%L|PYh73y1ncHn9PD0ADwr*kP6WK6Ie8Tky+tGBK#*;x6mw8gGk-^4t-C;jpW5YECzFPe2kd0;W8$jgs0Zm!=D{(IVn8}<#uk4Bc'
    '3%!LsR+C1GbEbI2qZ?=Llp!U0_yjWob3<s5_%d6utZXB2ctqJoVqN>7d{?2Bn{5dg=;|dC-&SblLxE(`Gw#%*v!GF_M$LAe^2seS'
    '+T-gV`-7SpDuBV1g-$9OzNBAZFzV2H;?vv$Liyvz;`T2KQ-3@?BPmK)(iX+p588}(WL8pG6GO7TJ*!uu+i6c-hvj|NWS23OTMp!?'
    'X)KpK!8^Wt$_$ZwJ{qWs<AK~9qTrV$;{`w#lZUe|X<%{+$Ht~gKIcf^X^cB$)DwMJTlTgh*#@M`n;-wd-#waynlQp}JDB;IU6*nC'
    'aCTz8j3+($Df)cgS1#3E_DTU&9mP{+k(Z^m|4+1ne0PnIP__@VWU)?|$4KmEo4r#&0%}+M9&vm(yF{7Qqc)#Bx}q=7#Ba-_r6CF;'
    '#@S1~U+e>6LGpP(6LAo|d+c2oFG?*e^@a?$D?5IXyg&5V@q&%FZza8$Ki;=E#xozM9N+V6N6$Yq?57n3)Ufji_g!YUj59DAMNbrc'
    '(F2`Jno0OR#ITV)nr%(@)~b`ZXCq4XlFw+*e8RuhthBLdp5ssUDB1F>BpQ+(f7v3aoMy6LqU16}TAf@Yzzi!ZMWYp2@uzZY6o#k?'
    '#DYz}$D2xfeT__)(5O`mZyW@oG#nSysH@xR?6EWYSX8gxs=!ryuS-7TAexZT??42aCY3Ov*g^s|LdG(mDu))WHN}wEddX*Wi!PnL'
    'CnnovBg^87qD#4{QLNQwS@p%|7k~oe<SoggRKP~tjmNPqU+LUXU2WaOa?Z?77yZSPP7WXL68HZ@Hek{O`z%h9m2MH832`(IGZRqF'
    '8t=2yZ}!T_;1bU`sg-q?xUd_~R-zJ}{b%OOo^-AkNG{^&98Bcb6T_D&&1WqSyjQJMc|k~vRZXdtn#L?G6^gRm?SKsWBEB=RajrrH'
    'W_<h6&5q=%jI$@sGWoRtK&%fB+6-z+rw{y7(oQJ_dJ0$#Pxa~15gP>V6{pR}r(N;OVk|PUMZqF3G+Juit991jj;!REXOu5*ZK`A|'
    '05?nD${a1(v%5*0*zjWo3`jHq#GD&i03;v1m(PwD4fIlL^E)F-lP?#SXx9(btgYHW0d4CuOuIyM1KU;&Q4tQUriEG^)Y}m7-XLjH'
    '+%I_gN~%zO_Zq*A>JK{9dQ}2S=8bRSq>{r16YK?ciKtmRru2%)x3T-IiHS3ghXsMp3=4V6X_6iY1ShUD+jfbGlYjn>l9cZ#u8hou'
    '>&Wugi05@p@+A_*jDaJZ@#(%siP~<pOB!I0a;5LJC98w8{c;m$)F^SW6Mq;*@f_*>zS)3BE!X46O0)Yhc9r#W+*sHF3y6t)RuPwY'
    '>#A=AEWBU9rGM#jP!K5BQs-?a=0(ViR_}jOD}c9n4lz2gF|l-lMrpWU_4$0pEBDWM*9IW*#BD~=ngu($0#r>|>Cr*lzr}U0kXby='
    '^Tm>{WdW*fJ=1%MyT=Pc5FLJQ9pWbXQ*<PgK!3hbaC<n$U*O;@<afj!Ool?WG#VD2@_bmtYrO%9$)L_aKUj&5w2gfE{`0480x5by'
    '+9QF=oe3A4<m6n`upJTC{N67Zq_lXeNEdYg(CgitS<{G^wuC6MM9b2;XvcBCMG2pUwO{qrZJ|`BEEOwx>=-^5=q*lmLsn>@ZYYd3'
    'Ok=@Iu)=KtFGRl2oY@n34TJLu+TUQzu4Ix|9aqPww6O8x8N{YC0#n!}a*xaxYKYk-YcEgm#P1ebFc)MyjTS5ecPYF9PKMPL|Au4|'
    '2k28ITEaVfPX_d4rWSo3OupIgM$ATnu$Gd`f48hN7)ge`xWG%x(l;e(#GR@FG`ScV_gT~hg_z{zioy1~olPn6s-LUAja?nK=E=7<'
    'Ge#2VXcmBsB=f26_olEJ6n5j5eF^7C7l6T95^%T}K*Sf*SY^H9oW&9iN%p&EhtO+o!YzfrFR7_}nz#rvu&ohgHo(6Z$f*oV|6VX?'
    'ti?T^nm5T~Ai@c}q-T`Rp+AgMh(wrr3t%bQG!{(#hr?KMf_3yfL?<_<EU9f1Mf#9!lVq}$39A>uIUBId_XqV#3tA&e%6R(XLTIV}'
    'cG6Jbu^r7b4JUJ$^<;NOW0hI5*>FlD$pSjd?rdUCni4JNFCPZO(Dh?o*u=r5V&aLemIA<fB6NrA?F=a5yowJ~vkcid0;JyNxc=bS'
    '@QGzA&{9Z+Pi}jBr^fZ?PcnA`CYczSqe%wLJNhLvUm4w^-i!lxLXAn#kl?jeo5Upon(Z%`0?NP_EqojYFWAHVdA_b#8#+E1V7(-j'
    ')-Gr^d-PdxD>g50h!`b0jaZ*u*xV&3Q3(d)<+86CHwH3+^HIzhvKIc_s0u#0{Qm!`WN6X)*KE;`WY*fYzq=g&@zu=?u#GpcKd^G5'
    'qpykNkQ6l<t?4D_8!v^Ei0~hGpr`gnzB)8=Q|aB8e8e%a9QamVmbC;dP1M2O$_4|Tl!yeT0%)5296m2`ky~}Gfn8$Tl9_*2VcJ&s'
    'gjz!Pr1$`3&E1dIyxbx?W^q2*%|#bVU`XN~hO5iKf=SHG`lzN@tN3+25hb@Qmg#G!H-umc3QOFZ@SwTHYXO;@;LlnUB$Q<FsJg}_'
    '#t*Z~F{;3dC@kB`Jey7&s4KIv<xN8YIgcBY-9(llKReUdC4}1a@yU$QHDMT4xp8YvIA?>9K_XV5G!@T0kflGdBb2E%Q_A!1#y$K5'
    's$wens0q+!nY!KL5luwY275yt!y>Ta{)IG&RTUjhT@puRM4@<^hloPPy6g=O)ky+obhRxmgJ#L@Md|RBG!4ffot?T|86*A>;kd7)'
    '$LsVVu5l0ciVu!E^d)jqHW@8vR=uD~q5_?g+U9%;^AX2y^yX!!`JE((4qT4lm`nB>iR))@g2k)fvT*nR^F-0LT%%TrgCz?>0rfX}'
    ')pEuyIco8UNT*djPO_bFwXl>#0WxI(t>c3&%!sqycZIn%zSC(N!+-R`W#WNYDK4PlwNe&ty#znf3FI8jL<||VWeL;s8TUJxYA@im'
    'BY5w)y|oeRZ&%XIqCL-iqVL)mnLG8NZZYdOr2O|Q=^X9%#6T7IMt{QW&Le?%Yt-Ts3>3#(roEs%Y4N968`tzpA~^XzYyzT3myjhE'
    '8heWlXKO$UAIWus%e7URde%d`y~Cl}jPa6xEo#LzcFxLCv~=gn%r<}c8sh82zsU`I%YJa%d?@b3eC;gT$gCl4=u^e@%9M5+{^lT-'
    '3Pe}AWHlH6c6fI6i!jbLOZuW|#nhCBHcK{p15`~~ENqUsE@8vj5Z5mX#=$9(bu;n<n91cyxYM7QA*~XF&I@phxiOwqDnC--Kzt@|'
    '(E11G)C8r<mP<6mQG!48(@XNx3=C?t-RPn)-z0lNMu$-aj~{qn(n1jM+@YUWCsOv3D0O{_4q%wAaLY(D>=8SXuPVjm&N;VzKzfRE'
    'Ck3p1zFkBy1D(FPa>kNXDgS1VTSKeT(bo(5U`o$Yo(SJ)S`znVbTkE)G^fX2`~H&6DouHdZ;XNI?8yg<R+h)CRE2NaPR_+m*2%=y'
    'R|$qs1{~0+t5@d#koLy<?5qtyXm?V#Q!olyhb-0`NT_DnkF^56@J9jBGM6gasKrmgM9l7;Rj_pP*3(1pU%p$7#H$^Jwn$@ZXLTCG'
    'E5$hDA@9jzhCtE+BZ&F)&gPZzSrD$e0x^qyupZ(U-i@vSEy+k4++7qCMZLy$JUOYLvX&i1LsN$CC7z!pWH=PqT}g#VD6mCXizerD'
    'nSTE%JL0!J*&4!K7;ng1`!iGvl|6@`agNk9X3p7Ju{9>wtE?9=rz=pOf@U?|v%=Oi@t-Ym?Tkh-ZkPNwfT5KgVkh6EUb1P6fByJ('
    '&_%G(n3-ipRbtDCcUhezlq>|2f%fh=lB0Mn?Upv!izfSB>xun_^tC?=X{-_<wV(GYJ4DQef?>PHp<eg>Qre0i!sx#h5Ic6s%vk6G'
    '(F0QDSbc?`Gft6$Jv9!BnOBZx1yeswJVVq`-;l_;#WU{@v#&49MrmwA(9m}Z##Uco4%|*WG4YNJgj!9*Z{u@!Bb=Uj_ZtS;vR$os'
    '!B(fEtA8-z*1>2H0^F=Guon-c(`{f|LD$=|<pVnx+Pt=EF5AB9laXX=2t9T*TBBs0Y$4^!LGfs3|ArJ*BxfvNAwZ=ZV=$1I#OU4X'
    '8|1t3h4Io@)z~14kCDz4^oNo=)fWII#s+Xq7xO){_siBd%+k(8z25KL2RH{815WiDSOb4v7{G#{sqeUKv*5xe6nFii37CF=p@HJ0'
    '!63j{;&uNin}D=C;EkCTz)>pWJdIyay(M{s-PA@GM5tleS<2R8@cKnq_^vAhC?W8L9@70gYw93oPMo~)pDS4q63n|r9GNjrvR@7p'
    'a^en3oOnP^K)59I2~BV-z4&+OiTWIYXPGW^G;Wa0eH-OY-z@MO`-N<^t*geq;8oc#Yw)q37~bgOu%R;ZEzg2;b$rKpl$6wQ5YOiO'
    'iyYaErUz}Ld;Dr~8GewEg?&!)qAB%f#%Uk~Aszt0fR@T6_ILDxhWR!H3VeQH!YJzt!(L00ILP9axFbsKH;8&Btc-;c65h^w_Z_I5'
    'J3<zrQIw28&L7H?hd^SC7wqYrXt8akE_#&&9$|$HxWR@;W}^iDGU&vK&yDsSAmk-R6^o-B9^0eV$P60MbrS}Js3Yo~K9Y_A#}XDC'
    '($dBDOwAtNUFOU<Vm>z_l%vSkvnuqN35z-C?OBHRoLe3Lr>#e9+iMbOj%v#lvI6)g>-@XS%<eK3lwT&&kmtIIPa*R?eTIAVv5v;p'
    'L0V^=u#Sj2j#*U+Ez==0D0IeeIEm~0AT^lmiTGl+l1JKXsjYSsfCNEoc~aFO5-gQmt9X=)Uy|nm0=nVuFtVOXLb}Bcu~vMCwqL?H'
    'a5kfZdNPW^F~`eAxnh!5;lL14W!aQVSy4~?W-aXn&^@@F8$AKZt}6@Fk`}U!_BAbjY9=%4HN>5NL<;e1@LOsbDza!9qSz-NEnzq~'
    '$ZFG9*Lu&qk1ej7`rJ3|7VoXihZ#ZUjj!z4EcFSSg7Vxl#Q1`}1VcY6n&_J@u{0_<tj0q+<k8FGLl&GO+%hb}GUoyXje}0E4UzdJ'
    'gV5+0(0MPAF!82k)kj>t--CGL+MT!6QwN8$Ru?Oy&s*%paXMD)NnNqFiTHm2q3AfHL@-S*4yJF-$<TDx6n2TvzaVJX{bBF#l60*D'
    'M#v;TL?PPAR6%Kuy^Mz_*UB~B1=W+H3RqX>Zq)iDh>i?D4xUE&xz3n1`I-koGdLonN3pR!IDZjBu41o2O580&+$?Y}@gbIy@sa`+'
    'ISjGf6sWWNOc6JO`J6Y3-Vd&x<ZSFVbghk=+zq2l^x##_%Uxcu8e0f2b*V=W)K}IiMOo1J^c<IS*eeoo)B<*SzjjF|h<Jw<xFPsq'
    '&9e*f>FnE&cQcDY#iQdtl(qdy5gK3OBo3f%+4yH_KAB88D;(aQ;Krs%K#VgwS@T(TcU+$UX)5eYPs+R=|DA$&fH%L-Uxqbfs2u>l'
    'xCC};M!^b!Qx;)h@-G-NFL9lABG>cK)ol>W;&hKD|8uXvuz2W@_~Vp~U#F)(wV+J$Q@eXQj_VLtKw9*EiEV(;IA^-%ahJ`K?cFfO'
    'Tbr`xzQW8&$XI+DT(=xS?cGb?s(|;d{0eQtQtdyqDpDTsW9~O-m6qabjg1*v+?o`nuFyqknkI#J>IJM&1)VT97`yG3F~-KBIKQ?+'
    '2a^LV8p$}`tio_wan<)sQbLC;NlE`6w(jVHoizC9ZWo`Sj!d^&_IE9$7Sl&%k#bx01M;mXHCw5dG{FS<G-5)h@<L1Y8J$GiirY~T'
    '%EYY_MNdcHE01fr1qIj0wX4Do*Qgt^{0<Cd@h!K<k&`jAC(}ELI^a*POC0b<oM9ZLV>m$C)Yd@U?Ot*BNeGoxCS-x&DXwt8$1PUP'
    'C=rc$d`AaSljCNQ>{Lco6kJ>|$re$&My3JrSl793k^RH6C5@$nm&i7M%nJ*tx8zBBHe4P2A|M+Awy(F$qfB)^Dmzwrt+sS|*ip+1'
    'cbD0uq8Iz_?5S{emUc;y&?e;5P1vRQy(c}9h-T&4j<&L1kegcKsaxQX2>g!bVJF{-aVBn^9b>;BPDcbr;Y&3l9jQ}@J*Q#!=Tuyo'
    'U9*+%Ns4x>#7LxQ?YL+Fc+}|daU*<o#Mbs&>XA}skrlWNwub<C!&gcYFY&iU!Gr8U8piXe3zUUFO!Zd#O6Tki!d#U@?U!o@q}UEg'
    'nw9h}$sl|W$`%~5eZ<+4D1~>F;{#4O7~SFtR(!$ORz({xyG{R~Kx1=_#7C`;10q?gi6KsQF>$#WwO6rV&qt{B5;)$Bx@%En*8QiG'
    'h}f^f9CH8QnXlKV^8&Q)3e&dQ+N-<!)}q$j$W@JL{vP8`>69P{$(CEQ+<zCcP>UzfJT4C`i-}RaKA}35be*^jR&KS$?#}m(68w|5'
    '!Jn;-#%dQGwe6kjVWUM}%*DY{kqwpUli`{!idBMd<DI;uH8jMDa}F_S<BodwOIIN(#Z@z~5aY%isUJ122FN}<nAr_>mmPH>smDe@'
    'v_NZVyr>okwW+|F>jlx;wG=kk@&Q1LtRt(o7Teoqigfg{-r%&w*Ap)pLo%xC1gC06HdN9NqivsW2=a}ZIR2#@H%fdHgcT>B3|=y)'
    'BEX603f|zfZA>4=zwCsOjXl}2C{fAcm3zbE7*E)DwRqErPHj7N+B*3&a@qnae*yi&iOkuEiZ=k$nKo>kofIPcyl2d;%PgtNfs5~K'
    'Mo(IjPZkx(121n%sPs2{Wwb9C<2IoNmK=>t6yS2BZmQ!A2m;YI3B=n~4NU^){O1o_e89M?Uw}edk6OP1(q7r2$dH;jd_iJ<|B={O'
    'Kt{dD;-=O9>gv}NkcOJ8Bd!|V(k(h-PolT_Zub|66ZiUFOpD25N=`sfL5<R%PHVbnK3i;4(Q9fDxGs_6qB$J+htV9aRkAoVF`ev7'
    '=+$h=Rq-PC9~VSR%pLBS1v-5aAaBcpeC^wiZAZTpKfdZO7srN!Z7Om&ims7ZiQ_}VE2BrqU{`WJv?rCgCSp1#bY_e#b820gLAUb-'
    '$e(Nbm%X>fNUlOCBQ7ViJ>Wh9n7j9P&*rqOk-9ge2k}udtVkPBTjR6fTK?c%3D{G7324+-)_^k!Z}BPNklV__NJ#?n@fELX-29gq'
    '1Bjvm_oR%eEKZDI9B`Ca6N|s#gvC>9aSoA{{PqZ6nBXYDr)ZPO)N#a%zGatG*rvZgf$M{_DQ-B_!*gb@R*5@gve{|`q>oIw2v%WX'
    'V2m<+-r_4VzB@PGoXyo^=`w3`>;dmDC9@He{O+}CpHOGbLgVaVoXJrB$smAbaZIkb+wjB1|D73$?OU62WW*_499=L5VVVm6gCe3g'
    'R0Z*X?3QVaqkrz*2+`xc_+>+#;(77HBdTr5l^-W(6sS?T02^r|vq6ylzT&s$3MWk-f=-GN0&|jOtoF~HZREr;@24EyI}t)Oe#!7R'
    '@n1v}m_*`rh85_(@sG`N#|#bC`*8m41>#9^ZLs2x5;A@drGDFdLt+dEBU>wqtnM#ok0O<pS<QsVnORKg#BLuQ2xMkoFCebEo~%uX'
    'Z&yHjB3yx&Z8geYsZKJC=OUXDOl~WTH|Dsdmf>rmP9{&XW>Skdvn$AlD3Z~iCifu}BfeR51KaJAk2vG|8FkM1e&T-0CVX?p$=Avw'
    '`kOsTXp`ZXkg*z{b<A55Z-y(u<!i{3mmoRgU;B*_Kt5>8n(c|3&n2CTJAbcB*2qo<Hoz~iEIUiAhX)tWsz2zIENT-Sgr-;yUCl0`'
    'D;oc1BS-`P<s2ESCAx`8WIdwHp1Gx&`EF{f_+@<iCCFd}Yc@R6R3$xA1;=N4`X#awcS*uv5WvTa=nAhb@i*VnU)VEmz^c$n(n1X#'
    'BPfvyqtZlbjdoycukufoojeeYDIEVH=!S0dlUheom3ohtlDTlpJ_hBpV-~~O=gGj3WsSvO<&q|+F^;TwDai-3w*9#;eZ{S?k=s^c'
    'U$es#iqY&mlnhYcuuNK_oe`{!?ZlHB8y9!HV^clsYWodxYV^^fodFx1N%yt~D|QA`Z2?kBt`)~EPx5I;kD7Irc*h8WB$|7DQt8B('
    '7*$SuiMI2!zKPWs%nMYGp7@X^am_Yebf%U?ZV!m6(0(gLQwC5Qn1WW-d*A9WgTg1-X_Gt{!iGIrFi7$VHKcM>rJiHG1RTe@Q!OK0'
    'bigBB32W<<^3j22ht?K~+gRzq4G4BTocz?XY%B?+(nW;BDRPN30+#$x+_Q?*q>Cq9CYt6Z{9AcJd<AJtwL{WqtRdd+CuZwgK${vQ'
    'v^X+7_xV+s8%VrwZ<xuf%E&TmEroGJ>(G;#02zM}RG!#h&qm}%^pDBivBo9Z3#ajGBbwH|;ay|_w}ahnjZ=MURFNg3c~q`#%gZU`'
    '<{f@wM!PL;13Vg*Au|ly@#5a^qKG5|ja@lN!R18RQ}-oIa6=eb;#m1wR*^ntxc#^_940^D&J=*N<L`hIjjU_#eMuX|1j$)W^DN2?'
    'oNS`vzOID#A#YttCqJWhZ=~_HLSFGDSDvw7Xt=g0Wy(mtA468qiGUy^R)nqd?xo@lm~lw=i%ioN|K_;0cXSX$XeL$oVsMQ6sX5H='
    'k$H+S5P(97TUA1wOO~dZddK%8KKsbBF5_Foc^uU!qNwCs92m-0l$g{D8ASZAe2eO4v1Zn6;X7r`#MVBAq_h-;M})fS_W4Qr*&p4D'
    '6LR*hVn_6m!5CG4k}e7Vb!>k@8?5j!<gGJ$oSBSh;-fJ0ekha8Utkeq;Sx85lkz?Zcgjvr(yADZE9!S7rRa7@xV)W+n_D7$oADYD'
    'X!up}>}FV}|JkIi^=SeZl%9RsLWU?`ayRL=GBXJMB*HB*{Id9jMKZ3ux}?I`$hHauJW>bhp891b^2XeO%ojQR%C2tMHa^sE#P{fl'
    'U2uayf2PmlyC?CT&|3OZ(C?2-?sF5yiX!ky+n%1#7Vry~LV^UD+%1tCELI8R`|3Dbmykqf)aFqO#FvS6t|g9Ws<qX?0P+l%7l_;9'
    'jl;#mwLq@P+ar8MZ=YBk!R_}fm6G><wjocgtrz5ERKBOp-EJT@X6B?tha9z9Ja~$U?Utoix-;!332||<TmrC{k+Vn1#?ln<b+R{#'
    'R(S$}fs)>dA+Vgw(Q3Sp@NY-&7tx9ew68ujjeWU@#>$WR)>n(tGMlU!qvzgMlr#8B=KZ45cybmHVFINY3#R5<(vvpplg`aM+6!_;'
    'H#Ifnd^PDc1Cd%Z_rQ4T(*=`idH+0g>#f2KPE1GLo?E78{2r9!zDIw9ueg#GC^rW=Xjw>5d}{!v;1+>%ZT5nE5>k>j_u2*%<=NxN'
    'b>;Dmt#~S5pu_8^dP>|tR`B1HmIYiHS%4FdJ68BNI5`Jas`aGYw(2RL6%Z&sa5yIA1;+;>@2Wt<+Q0LRTP_DRZLdiG;oEA=65>gN'
    'pjB5oOqkoBKC_fpJclMwaGV>+=UZY+BXg$h)*edk*l&>VqAH5&L)1XeX2$>iA?vy_UvNK@G1*Zgfm2IvCg{PyoO}PC<pm&uI4QOg'
    '3*aEWTorF<UM43yBEFUd)Izt49y)5{#dG?Cf%2#Th!_4Eq(;f&wCVXoeoAIwNw%;!Z#%XQrkie;uq>;jd*nXGCoGvkwt`m2HViW('
    'b7119bEe7GlroZB?`*1<z)M=GCqr8*1jO}le&X!yzkCCrs_yY1;#O-(8BYUe72;g<F+J1N8Ti1da&1YKepclpmc#i?rMeTjy`ei?'
    '@f@Ma-@m-LjYcf4^Ee>kt7Ue6R$tDGxv_5~C#?xn8AmM9K-tV#xyCIX8O7**&|-xH-=o#}Qlezfnu7UY^m$}!Mjw~j5BdXqhdV|t'
    '3tvDyK7^=Q@^8YI2(W;Zv^n1}P$sXWe>D~XqrUql<N}O?u!{+9Sr{XZ<FZgrW1{jmaE%KTg5_P}6JLoWe6>z&B#ZxnA3V_xv(02Q'
    'Mxct{QvGbW7mGC#qLQTEUva#QqJ52jXP^(VZWIbviMFej*-!@*Ptwy$o_YJfdE-_b5g>2bW>_){A8wBxtP&5C9i=_4HgDL%S+i!@'
    'IK#;!8H&r>C5y`=7Grl^&!hda;I%7JD835CytQQMua;;KN=P<TX8mp;K!!h2KQ+_7Kk+(PzsZ@&*zNz5r4|Ipsjxa{ygrlCPa9%f'
    'Q}JpiS2XpZZ+zFZ!@b^dJWCl)l$ojHwH)2ktJ7~R(nMb{nm|CdCp`jT1orN>&{0q$o_;cy#9BHFjiSZ}piq@Z^a7>30*QM=lu9(y'
    '%23Rms3{x_NRCQsLN$NjiMUXz+bt0=@eEd8{kX|;9o$$1qqHZKc+@Py4%P}}>l(dsbbpsf{ZQ;TG#Vq0Im<%xFw8_DJw-k#wPZYy'
    'Ce#(bhkA<}yfTj|@lPK#vze-Cz|>%oLZEUn%fk7<FHzN0jIH~Yz+M)p*6^Uh`t8|AEYG$C=-tesB@hr51Vof%ecmD~_9)kQ?Bc}v'
    'u9?)dYp51?Jpqt&Bu`+aEe4+wJ@R+EB*p+Sm`rU1Llm2t9X1}jW+E&#OJ;FUESPd2F^D$f61^?yL{OM8R9Z4)163U_C_eqDL2J#Q'
    '(1)`X?fsaEj9!L<C^^4KLX3myb3*`P>7wK!7-bTFli8nPN%aRKv^<HEiZ2-u&U)smJ(*dN@V7UxN6=264BsMlNwa${wN|s<r_@I#'
    '$9%M{7fih;(ot7EwqRc--7B>kR4L*`TDjlsw+0EhQO~r>Y%+6a6tF2}O12r7{!Trr8r2eUzp~`JHbF*$bb3;P`Uf9PC7xG7PIsaM'
    'hhR1eE}DV=D4A7JIdo>ExHNWluXKhPaH%RTx|vM4wfuHD>tV?4-@RaE*zDYZT$q{An|pFVaH>RBWmY~W77Gu&NfVhNE8@Rn)O>|s'
    '%?EkM-OyOY>Cfg(*)iEqZC3L9ROWU=Jl`I2Aw!x=q^%%bR+IyAyu`Jb*}F;1>i^`vfAj_Wzf;}>USe0M2kYGp{#x4mc64)mUkMA0'
    'y&xq4v+?}`G^i`docM8NB~Qy#oC_yWYx67aca*BhP`ooUW+T&BbzJhL+g4}~fy+^$te)xTs%*h+e~ozAR!NS+$y%57aU;zx6PtgY'
    'h6ewkHHDnU|I^dKe>|7{C!Zr(T$2FTp~AspYIaGcjA3`)Grl~?rp=NO6Z5Jc%})numjy{e+kztFcyw<;g&~+QvVJ;<%#l(1dVEAB'
    'Ky*=iQp@QeWe5CNFSm?oqRS!2*0eQ@axz&+qf%BvPQj#*eAWDtbashLJ1?1fZ!|KLAZ?BBoq2sJTb)!`QWkejs>ojFHxpQ?a`r7i'
    'a4W)n@2NpWwZ!Wi6;_<Ou0chcNfr=(2i27V#asf`+92AEOdMpBHTkCxIuGkpS;T1!P2UB!O4QfLM{l9RF5W9vgUu|r5{<&%R?d85'
    'nR)+`X4THWSb!euJL`+a_aGG{nx{5i5TFCp6K&DbTy)M{D8HToMOQj|=IrI}@BEUC8E_u-PS+(0*Z!ckCi-q{qL0|WjKq4w95tF?'
    'Q+~2#TI>~7XhpY$&Y|sWDaY?L9*1U(60^gMZO+u@N(lMi9RjU5O2!^fMtJq)xm^XHy5f_WDJ0ba#%(r>FgHAd*Uv_MA#n$T+7A+N'
    'X2Vnqv_&m@_8}T~2cZ7=DRmUbpF(-WW3KtBUt)&_qBo23u1M5O9+fc|%>mV4mgkLLrA?oxC}8fXPx5zXUe&JIFSz#Rh_q&EA&5q='
    '357ad;e*3-G#Vd_{a77I`Qp(O#N#;}YL-S&e}FM<PMZ#VxLs0&dN;7RoA6guY@UcIY{s!(<KB$UxRdCRu`?<;kWjUv_*=JlIX5x#'
    '+|`{5tiJr6L;4Y4!G^{+8l=aWuGUJ6+r5hvJ@5b2PO}#ujL>i5zej?E6p#kEH`R@|SEOwk%E{55V?tWqvtAqaO!V^#NPWYW%&BcR'
    'l-npETbvO|6TW$9QA##|*So(!TvrQOJbYzJB+eAROfu0-DsiT*7htbQF&KStm`4rLk`>P8*r>Gc?Hv|5=sEfqM@#cvYa&x-?x*Cq'
    'HW1sr;qWMN%x#!QYsK-<`XAs&!>V|kv$hwU_obt_HK9buO2(KAp-X&YKLY6&$UVKFCL%Trs@7ad?HKKHl$gULW9sb%Wr=OF7C;t*'
    'P*^QlO_8NC#S(Ld0{sO?;%@7^Mv)%Jx#h{^iynv2Gd6QkR=r^81pqJ_<nc8-+3U#%`}Yqu$sFVUN0CVqOhzP*8Vq;(n0~1J<BN(m'
    'Zj!DMeGUl6Zg<ZKDl#B&?@==0>-0+A@O#DmW^=MmTwe=)af?=@$H`Dnjzppe%8;a%OD`TtyuG!LyG!9ga^~X5YfkXccr4)U@Xvnc'
    'j%;h7YgZC%s;-N1@PHTMRJzjp8dsmii+V$_dldx^;q2l_M20Fz*I{~C88OJbSmSv{-p1oF_Tg5<-Z)l^@0`>)C?Tc{--GU0zVHc#'
    'E>3Q9?@GzGlahc3Xts(md(o<x{Jg#V;&ejlTC$RDMK^@dkHqZ5+MGPVQTex^^=mxdFazx96W^7&60(a`r-HM0KdA1E=+w?UAl|O>'
    'fMq<MlEd2tzTdYpuqW}NQBo2F+JEnXGAx=TPoYAeo8LrH!tFk~$G1HsLDmn8Jnel#6Ja5rTe=}<3p%6N{q|F63O;gaQ`mho7l%Gs'
    'Ka@llk0G<&DEr0(7>)0^ox~G-m0scx7A@BNA|#+6)M%gwJK2dHCA$-P-M9b!laS0}l#nFEX@k9<7}fvo=TW`JpA$S6&qR~f^X_tR'
    ';C00^-*B@=ThD&l-+TAg&H31&7Xhz`-~Lca$DtCGN)$(J1z)=3f8%o4mAhq!8>FYpc-EePF=k9_9g?UP>(GKEX7$JXgYmo{hBiz+'
    '+_$K;aaeA7chhC~XWcJA1^FK1(cDg&UKF+EAbFinvh@&rzat(2WwrdU`6ipKr~#m$Uh<;Hj~<lZO<7T_+$wi;L(xwk`ys3Ez%EDN'
    '1F8~A1toKz)cGVnm7$n&y~0z0TqmU$;_XQ`K?4Rt=tPY+Xr2aRB8*ElW;|l?cQPd02Ptc-`w1Aq{w8RQeEaPe=Q5-yPTcXluMPK7'
    'RAML+_tAph5aU;J$DGimi6!-LnGgwAX5W(ePkJWQ_w$qn-^h4poe%K_Rk`>0tZRJcyih3LChW6Nbm5$j&Z+w0AIc#ZH*;6;dA#>S'
    'w4QMqCgFvuz7{LM_?5LMLvydc-!ejnFpO*f0z8Th6~U+Y7t+f%kAb7gC@QEq0WOEqdDICu2(YANtgr+-1J4&jUg9ysNJ<?);+Tx5'
    '6{wTr#yITCNhmg&sV@C-Tu(62%CfuXG0}B#NzzvM+VH`?NMV}w)Ym+svLgB=-iCM=#{#Q#fUS<Lalx~~>`gjcq-hsS6s>XZbiiaq'
    '*0x3cVdN>+gNPH6KwJU6kW}H^T5zro*X!sEs<i*3;UQ<*ZrE@dWRI=hyvVXO*W=|s8U05T`V0JhBenrOK&3`FB~*;~HBJVOZG*7T'
    'M;@rQFZjq*zmx^r2vGsc5Z!xkrx%y;<Ni0(&#=bR-ye2R$Uz&DCCU;lyOr3K$q_$VNOPS~P)_F>C$?lm!9O_Th@o~uCYwxwXdGKT'
    '>L8)?bc~<1G8y3d;#sKcf-g8;z^HKIAI&PJEcB9?R>nbJtUNLb2Xsz*$I}lkM#|;?dbSt?s2YoR*`hvjV_k4T(&K97xp@Yr|HVP$'
    ';G+*`FOcFvkx_SjS|#5Ip$=UYZ)}IWOg`yks_U2p>;C&+@nUomPhK2J{!A2k?bfmGCAE`Ympf)a#N*?vda9-w@=}Sx%NP}V9uVqz'
    '_*<hIj((-<m)D09*%@Jd=vmkrLYMfyqwKHFF$-Dc@O5+H7{Q)2ygI=g*vY|i+myE}>n{Hj6E&)xAeC`a636ps#2p8KH~pQTnP5>('
    'H8^P<O)?KV#`{yK8uJ2=ids+du5rnEyzaeTUU%^U<4x;ApBp0qTq4^K^BNYGaT0S16Y~=9fh(<h-(6a-`jewND2N#y@=T3WiqPaN'
    '-)+m#RH0haqK|%9tMn)}%8om$$)f00-lupu*P$6UlC_@fw+$1FC&Sx^Mq;yIgQ5V&Wr1z4!X_(+XLv;O)Vm5Pl%_<PNBPq2aL*2o'
    'pvvGV9{1UFdB-Vr3icyzWbt<(4i}`i#>=!%KMK8Iu;f$CijJHSLmEE!e_Q&JZP|7H8p9Xn4*gJc^mrIR3*)_XL2h(wAGiy>vmfVP'
    'h+m3kDaHWU4=%F!t2#9;qQf~&V~0{d^ak;YFJ|rp)H_|1Ye#mNy3;H<^_8;hC-5H8&iiTN3tzFu--_OFQk-ofWiP0{wI;2!d<@bS'
    '^+!Bc4P@5I*1wQtRN1OK0F4IQ<?gXPQ@hT?>AbP<vY9l^KG%E49i2OuE8kY*L1JZXUBir{U&o2gqK;rl#oM~u<=I9aCt7^O#`>~+'
    '*QRBvV{15+yEZ$!S%t!b?j@2CcipK{TpBY*6+LJ-CrBA`Ez1Q%J6p0{#&>9}1qz|kyRfu948Dy`NnFG(owfl(7~0<+4i%JQ!hLhH'
    'd;F|Chw<yYV8HWGd0D)Ju~!1~ii)C^!=?k(Skxqt|DOEG#bt3gf3OWgf{jZn`Y+J|HuB*Mn&Bqy23C*~k7}EJh!J4pQ0wsJj}@IQ'
    'OrR)+;&ZI^X!8FnEgNtit2*euD>>FE9{zqaY{(>fytnbaD!k7{C*CI)lY~nk#i*SI(oU9c%}%Q{B1NDjYj}e2g@ax<*x#RMRqR^D'
    'B*$NvLk7~@(=Y}LX~h{dwF`4F0z}UphCzI@@#wc7!*G8VLUWA2{}yj)=QguS=ZT$qP}e*_J2btG11Ty!hRt+STPO121M@jKtZ1**'
    'KTsX_eZxqr$RIyu$R)jvy0xx@lZL~}c082~r8{aA@nlbUZ_uus3Qv*DKpvv<tko{~f>}7sp<%Q&st976=V59y@VCmcEt#1=L)vw+'
    'HjwSAZcm+tQ6N_aX7le4X(x)Nz2l3WL;^iYz<&C?w7GjJqMBP4`rI?7zFR@hCF}9hjX0eL(zK`LKLophW^<OBT%wzq+Jl;Hch~(<'
    '>pEXU{hG+>Ek0RqBD5T&EldsL<&A<d>Lc<$xqs@Rr=Gmd79FnUAD9ApS?8&hWsUKOrwI`6KO%~@KGb;ZN<}bC^Q*XeyI7{}1oB>N'
    'V6ZO6gFw-Zge-c5ei)xFsPxoeBrYH#j9UBH$WJQFK`=237;(1ft;^we%F5PvZCXw(R1?}i{wB<cr@gJQ6(H;!+zhDqsjr79g^G?I'
    'Yf^#}JulGFf1d?luf-JqJ4`4K9rg!}5Fu%zUBnr`hWS50i_cUmvmy76kB?!vAt7l_jMCx0f>RLRT~ns^)PPT6Xyx=3BXy(Z1d!iB'
    'Lf})s!H+x?#lu9e#J_Q9qQ^}9V5LFoL;*(piFF2Rq)n`H_fT3~Y34ROoj$<A^6tLDx0McR;w2+-fb>0P8J#4F2J>$pv__jlLxeR$'
    'Q%6AwFs}RE4hEm_nhN=4dty8xnXMLT|3i1+E_yop^&xP8Ay7wVcp_29afaog$Lo#;#vU}ej|*U9UWKXR8x{PI_zz46r&vCtf-Jl6'
    'BMU876%L`TlMua-HoW2!axr(rv>j=Dy#VOE)!)YnKk1Gn5CxI;UXimiQYdkV#+OJ`7V~@oEuvzmhx&te7+F2(LQaq51T|bfF7~Pf'
    'XKQ#YxcC{z$rN(R6WMeH(bz{{1S@d-vjr{f?hol`nfu~Hy4Z^dj-nfj!pn{TE1~bXDmTZnWZUI(POhVG^c_#2snJAQYj~meshE=^'
    '6?xept6=)m;Z|?^@;P~z?`JY7mx}07{I!#9X9+-tPJtkeEnQJ`M0Po}BpNaXhgjMSn7I3h8)2fGsYkX4IK(mOX=pokwB_sa#6{cD'
    'S8ev{R;BGE64Vs$+K%#~y5cA3Fe}cW#<qOy%fH5@Elu>!hDc2i1!xM<eV<xVdw3_bq-&~+MjFt^<=>+Z%ww!@*5cKwaS>9^9OCFE'
    '$PdzgaE~wWcTLm8>%Yg<zN_T!D18h^#5TsChaK7!f^ar9EES0syz$a5KUPhVndAqfEN?Bo#`ttZyY%B2V51f3Q$nnVw(X_X%ReHR'
    '1=EpZ8e#5u9pk+`aSoZESdR;INS4NgsEd7h`J)Px+0h3_1sAmoQu*0fmmdD%lN}A|eqACY3~u~h`sKtBL&D)o#4f#o*I1hO_=z3='
    '|NIE=BiABxo_=z6{ri{v`x5}j_Z)x!2=3^Ph;BjxjP9L7x1ScMYy_hi*a(ANIEzy67JeXrS`}d^8&JfY@&9e@wOylpI5o~jjTRTI'
    'ep+-prP{OL^Jw%R-xw2Vi)tupfx9~m<m|2Tf)s+I>y{9%wQ=&j&F?sGT?(*8c@SN7{OF;#BycynYg16F@J9HwTD#!+t&Mm|h#5>>'
    'P$)#R#RhmD5T7Je7mJ^O?GIgmu!EBkgQzeCjq($6f-qdz{qUnUsq&w4`5E+rGMpzVF{p$nK&UzaO?TRPK$Ua;Vy7vO!bXVnh%=dO'
    ')zf#1z_##@8O${?7n0!Z$B&3Fbg2aSZce#Hk#&-E#DAKMp!n~TdqrI>@Pr>A==K~dj1lPtE$)1+{q`^3xl!(pC<P5ZoL7`C?fK^&'
    'TxytBKx|@Q`q~2K9rRr+*Q*k1SvohoLUC0ec+qAs5Q-QVf{?=xFCQ|p*^`Vn`PZ4K81WG7)8{GD4ikX#-3KFVM);?h1VRC{A&3p4'
    '-(wN8|NdDRuV>*(8b3fag33+`-7x`Id?~ROe)rB(p|279Hy{V=tE1iS1SV1?y;mw~7-0&1*An0<UPrPn);#zp5p4LBV52g_d_pgK'
    '9e3Zpe6Yn`J1}gcKJQuzw3GM<=M)Mw6bw@=0uJf}R{-7-$tbnrPq(gw`e7GjvrefX@uQ$7uLJ{$1bA|57)e<Ed|qE{Xc7O;!vo#%'
    '3^lB3#)VYIk0Si!%i{h-vPrqN`gnxkADYadSJC3SNdPzy3s8(4u9uDjveT<N2b3Z~p#q=S6!D9yXf>F=>E)<E70XHYU#L<r7wc|`'
    '-zgh|cFKbRolpr>#luyX@FsC5u3x0Q@>>{$MH0qfuw>9soS<8@iA!t>WIR|7eE|YE3~h+w9Fw$|N(Mat?#0nyr;jKvK$l@f&Nlo5'
    '(N})v$|d0<{$rKUJII7nEQU1>V;Y&xFt<C^<@2Ld9fvQYMm<$0KQZ?s)dQ;L{Q?rDM~$SH(iH$YprFS4aQG5QmvqQrz|b_i0rysV'
    'y?jB=spISz<4v67r&OM|>oNWes`c?07p$SbnObd=(cp)vaQy%Lu8mU*FEoQHQ2;Nmn~ck-twk-5aEP;%p>TZIdiE~NZtKyNBHoyB'
    'LG2TC4UAUr+|&~-1J%j?F;)Nl+6podJVHVM2LGf<+}-KB8s*2@A)rOY45(P3gOndZDZ-wP7Cuh&=YFgGg3pVtow@ery5uFDj^OKb'
    'p{^fTkb=O;(BY_V{m|ubqqj~79W;~8wWA2BdVwQ{c3H?JEV7Ogdpv-$!zYFgwQ(+@@~8@MW^e9iycI?T>Jo|nxxM!d79<t`eXYk`'
    'QAikJv@s4sP&^!Ew`ZTDa)MiPv_r2A{LL{Flkws>N~ssezTim1ei5&0Q>EPTPNDaXQmr3KM4nEL<9I)ZsgR0FupE3lWO5J0<IbRy'
    '|C<Y-U7$(AC|AjqiCTuqnrV--Pp@SY(^HQzPF#!&FjY<?HrBcNJ1B1(+fU&aZXilc#GLiV3g$4?QDe__sDe~V1EOKN+3EDi-ma&C'
    '6skmnG$Q;k4p`@6KgAMN7B1+406i?a5kL-ch<@f_SVlF`yU@wDe{v$ri9L>zw6OoGWG6w=gWh<`y&*2Ul&68181a~4pNJ1wd))PV'
    'YK*w1cnAY1z(;yT#-Y%E+n^}|rV(HMah@X?Ej|zN77OjZbCch_ccC~M%(e-mh+pj#xS*c!%$TU+J>6>BtI*%^ODQ()7F4jU-7}sS'
    '!dFL`kKtgidBT{nri^zH>bZOu)z72#&Z1$+l3?;ib6`(DHO>4LEG@9S$ONM?P+jdq(?~L1nefEdst?Zqyc}N19&cJ5#JQsrMk*Um'
    '0Ym1)kHT7|zkvF&ZOWFi{Dk4exw*tGh)c05hLqFn1P7?hKxP&BWI>q&S>9{^y|NWM;Lz5WZ%vaQvVBj3A7(docAe8jk-Z#get2k*'
    'YYwXq#bG}n1SDpsHwai<p<3f+)QpH=pY@@rS(s-we#mW&$rSk1xNYn4M;j?nLPi1A<VS}c1OZJ1p6)7ENB@a$ws}DiKqL28R6FBL'
    '>yV<5aUS-c%q-(wiN~{i^TjxnnziqqTUYiWq-hJ_!Y%3X?i_UtubFgE21!*c@i(M?w%Bza>PsZ#sxK);8_fXL_|-0tdogriTU+Xi'
    '4z983ByA39&1o4*AG{g(&?oJIkj(LK9JUitiVy7EIK6Fiet~u`7wlhb;Ea3}xP!TP``T1;C)O=S-pA~=08)5uvW1(UKn=z`45Lc?'
    'PPl@w<@MqhU<)8B#+Se@Jub|`_-;w=c=#9!&%)XWK&w7(vKil{`FSO?#XuCGq-*SP$4910udZYTNOJMgPWkx{B@}%{+i_+M4rhiO'
    'tWf$%gFHYz@47|T>Z0SZ_rD>D3s}ElQU$@$NZ%esn%r>&Y)sat3<yeN0fh$Tgj*SGh&qPp!K9LV2}D$D7k~>C?a|G?#G0WEi2fnE'
    '-9r(xc<edU@aLAOIHCf>YgkVdOFUgn`xJNr-I1skzk$WcO=R0NS8vz@VSM>Gl#PQ68SFyS5l;tU@XQWwkw&#x?3v;&WHe&i)pY@?'
    'K?y+)s)5f5c>?(FX=Di`*O}5Nq;aSUE>8;Je_H|lZu$?W6f-*usCj|6a1rs1vUiA6ht?dUSvf=i@SCt64?PO)HHD(|2RM5I(G^N('
    'x{b}_QWaje{{AC96^6Z<4%7`qpRN8}hg2w(i3Qv<+4LH~Flz30`4SMjd^C9bL835zrt7y;@;qVFldL6rfZ{9~gd4e>Z?>`7L)x5$'
    'zH2<3k%LQVu}E>AZjiE|pSygs2S!R<zHAQT@8WES(E<Tu^%J;p(MSz#XBL&FvS`|eHV_MpFFPS{!Ci6fkc{eKt}t2FgMkFcfD8xR'
    'Qz<8{ONVT3Q@_cEt){I)NZf(UG3kcqLv)=>rW8-Wl`tI*3dA5^1ZgkY!@A}#j<|Ru!ALyL<*rrd5eSBpxC8bTRHI?w8>nCo#CN%F'
    '8MI<L-s1l08b7G6$E3)k6GO;?GWS6W+h^*PF%@mCQ+w;gg@<-xM@hN1#^WT8_Z|9y^dE<!Q^YiY8-~uBB%An5aY_IE8jKRt$mMHM'
    'WiMyn$wPK}q@a})YgH2+$D50!XwP-`@y(=e;(Tw>9BkvI>Admn#ol){=dh|Q3c~n*!yw+}hgstY)M+>!$(eA-4vd_~y4=0!)DcbU'
    'a5&FS+LTcet`n>;+NH{V$Owldo?szShyCq`&$$tS7f^VMn%ZcjXudAkuz1<%7)h+;u$*-gu|i1INhNC!GFN^pwl0j)`M#r<p9gLl'
    '0Kn*>Kui}_iCyXG`|rQNi}tnb2QJx3jG#JhZ+=IuDdhw&{1D%CqFIq%bR9(OT)V^7_HA}56wAHcJ%02O|Jq>~j2q#b;>XyE>S-Qc'
    'I-1`}zTP0R3q^OYNW1&xmPSZ6j$T%RImQTautfFZhqX0fgN+Eus16@?H&?z2z`H*qFgp55GE51;;q>(WJx(&)Wn_F}wea$+E|H_`'
    '+U)WVNv$K(7juJ#NyKo!G&^iL5Vo6qB@ECOClYyy{qjAIsWmVK9updgUt+PNtA~a1im%PsNDlu&sDK(YUzc+YCYZ>HAs&HsgHO8^'
    'H-3;$V~97Tpx~zKb_QywV=k{+jd-cfp3!Vebn0IlVM{Hiz7$JZn_@RigUXA#`V?bx+4rbr8)?v8iiV{;suN*3+A3P_2k)nhX`0R6'
    'F9$do-96k!;>zO##0Q+poH!gfrDCk(RwS`TV<!KuW!?Uq(#cW+Ksf5;)@{%#<CKFIPvv6XPfyku0t-{S`t9!`POZn*iwinlv`ulh'
    '#y)Zz-1!z^vlazlQ<vY)ra*MG@a|{}1-WSwwp&k|2}G${J^?4wtB56b-hTZfpzQI9q_dstp-%LASd{P1-4C9_P$9BGMafoA28l^9'
    'B153~yOxAZ5IrvVU6>68KifmvV$mR7@yTHL?@x@!E)kFK!<ntem9hSge}6(s`7tlh=nQbOEI4^SzXTpT!GI~qqj;?kE1m14UQWo{'
    '_{hhXGaomCj32AzTYKXLWddFrBX=Px6vPLr|34pH9>rRji^SJiz&*z8vew-Vm4xhsVk=M^ubzwGd{ECI+;;^ilFT)*4+5v?hd|1x'
    'Z(#^_{SIV{C74D+!N32AY6neFb@_4n6@^J2=6bBtNT|ZCS>jj2rzzD=pSM1iUPFVlD5Dx6@55>)PA^=?LrjV7s+&<!9GqFEr;n*x'
    '`6kCP?&kQ<_h0NY{fUP+o?XJ844Bh#eD}D;UH|+hs022Sc~GeF$>-EC5Lh<<n?l0q;ZBM_{U+zbOu$_vVTGu>d0;=Vt&YbB9u+&D'
    'ClmD-e-{kAuuY=$K-tQes2^@ed(O0n&#_Q)iF|%|U-(oF0SH;`7QeDZpj|uSc%#Tv|NA|*sJwASe+a(;DtU|%N8R_PCZ4Fk4^U<5'
    '3L|zWF4NhjC>3^?xxhYHyYuH3%>k4PJH<|aUJ@VgPGNb<qZU;GNumc)EYz;qg|cvg_a?2)-E#)ge$XDPM`05f=UIEN^EXZbC7dkH'
    'L?-TYg!=t7k48<pvD{&^+H1${#5cDtC%WSRtyQ5;JYQsoYW3PqfuO{1Fe;J9ZauVxJ`b{V0wB6{2?ZS)Ik~8W#|7h+LG9gs!a}3Y'
    'c(&zI+5h(gH#S^DQo&BDCHSv6i|uGWtK@gp$?Q|AccPRJGLITBvU31nzjEA~jO4m~2&vf%29vEooLX9YbnIiFyan--#KhEvu=$Hn'
    'yL$AFpqI2IuiS8Dev;a7n8P%fQfzn-`IPwGD)G6Ec`{*Qu63a+o}ARifBqNO0=QyY+~=|#G_2hIm`4xU-I)IvnqtH#xN<zMm3G=T'
    '5iVRi(llu)K|HWsJPCyPhyz*?5$Lt66zoq1(MJpnQ5HyFgOkkR^gD>PaZu0lFh?nEqJvcOkex(yJ4j4wAq0kRw!<(Z)SE)(f`Jv@'
    '#V1MB;1@;*3!`GR!-sI15#D?v+86sa=2&C2%i$Z?%Umf9x`->ARFvTJy;m>+Y^_q8KaXjY+HtA^cRHxOoX9dz1Wx>K97GyR&C8cF'
    '94ZegMUspTSR3Tg3Ds)lK@$%Bh|xBZX}wNK9Np(5nMN+FQJatk<pPJ2v=_8Z8f<LJa}sg-$r@=t(_lp|$;Akw@bs<1cIp*f8O_uB'
    'Ax5=vfsWmP$ZRShCAZ-yXL8sMu4|<;)HpD)KX#O6jGD{v(+8e3LytbV&Hemt_6?J+UAar6yuzE&w=C=(c){TVFY>4OV)x)i8Uj}-'
    'b@9J736^l|v7}v)z>TH#1{B2vVodsD=hLVO!e@GNX0*kG&K?wwGk0L3HBS)P%IF7aA$HtN+w07)ZU`uuB&*Y)B2HOopIL?PlSGpW'
    '+v(+*?Nzkh^ab-2N=C}}Dqy~dD<|?7=4zCffBy;-*$K8Jss;pCSPSY&y$VN|up~Sdtou~pwRw>3@U>1hEDQmgx`G1Y-~!DA8Eiy#'
    'aEs{elt;WC`XpN4F%|F{HxVhS(F9<aI{YYGKop9U^`+pRWuR?`Urj{;O%w>l7#v#4wpH7Q*b|e)Uu#m8Rj4T}UfVo22LmP0!_Zp@'
    '_f@I%eX67Ry7>i()d`Ql8Do`kgt8E0Ac$v$PVDuGRR2hqhy`Z}u`pRxZROL|!`6pZe~@rqkw{K~DaeVsJSYpsz+y_%h*2nV)(irT'
    '4-X=25nj<~0UBioNATEp7nK}VP69Xm{y@S-?|p#(Ao3}&Rmco2;Er_*+u@O9$Wa*g&<+py*gVgxomMnRP1X-Nrp#^6T55dAG}`H`'
    'Stwwk*(4>wyjiDI?gBTJ>`#M(MGko!H{Dx36O{l@%0p+mwmbLQ;YSH?A1cTEv9d7|=_djuTr~-b#Uf(A%4ne%|GgUkRg92HysMjE'
    'OC>d|9je5}$hL8H2`sU@&h*+(Wh#pbVBP|UP27htYEBwYf;b8mVqPWVP6v1LDf6KirxaHlsc^WdG5lmk!<|u}>sAp|sX})0)PhD@'
    'lzSDUp@~Zh55jJs`;&bg|1@T6QzR&kwnGJ_?g)dwsI!YZwnKcnTwUZ^ee<z&bdg?w?_s=m$gv~{fxd}lxnsbvvgHCiXu?2(Z*~#H'
    'RV?!S7zNbZv$YWQZ}Ns^&ahS=4Qf1uC&cJtCL1qp9Nu7qhyoxD=42})Mvc`%SKbdAImB}Ff&eE-*WU{*d)%iO*`HvvlTmalyF32z'
    'Q;0QS6q*4Y7TcH_k)LT&p#Kebdmk4pHzmETkQh*ejmhe4uE&aWDzOW^7zC<9UE!!Zs^U<TFZ>UfCJHXm=6u&>UNG2EoI+e0YpA5`'
    'UlYsuDI#qw1$B)l$D0HRq>lSEf$q{3;0=epg8h@fv|qq=T1@`A63JJMD`pAQeoV+?GO%gpnG6&xs?=kOpgjQ!7SKk%Slz3GZoB;b'
    'M|#KSo#1fdRfI8Ua;n5H;Gmp$!F7hjk2@ZDI1{a(U}WsS7B$ms4_cl%)~Xi_vIa3OwAtu!H?aFhk9(+ydt1n6{DB=i-`Y|j>7lUH'
    'w3;rf#{EWRj25R{FsPhyFuT}46CKgiHFBbcK7^b;RNWiZEA)wZCh#NqRI-`zu}zCZg=hVJ`a}gBv*q86(|0R=;*{n^L%17X)zj}Z'
    '!2#nrJ+W6g0SKKybpw$R?S8WhqUH$SD*h9dD}n}f)k)<huNu4QWt>r6DPW~vF1Z;7o7e<<9Gi>66=HVyj%YN8n!w5Cntgse&X?;`'
    'R)b_6uP3}3SZd58$BFDZi9kV?2$6h1HSb_+k2|HsRn5^xfbC?KTkvI`PHMbQs0bU@8>ZVo1A@=Yw-ArVOGp5#$3>x{&xolyFcaeW'
    'jjQ;fu~}g&gobRaalDux4Sq$~GS&C*eHi3)4I==PPJ-a##-d5?C*pMf$0N-XxS8#03e<j4&VcJ~qqP^l4AKRxiPJ)Jm_od^=3nuT'
    'sXCMAN$1kzk`_*c-|T=jMLFI2VS?F}=|Gjj1dMbyh*@@WMEp+<vpA8B*mn-Ijh|F3)w+*(_qx`zgz9}f9HerZ<8+Alt{5V)Zr$~n'
    'cD-N-&{1D#_Kw5|$cThE_3-iw?~aFo-p|m|a25H!UiFSj73;;L$$j{{<GpdA&Sqnyvo)$=b>Sv_uN@NR@F)(H9K3_kz%JPu;?BNX'
    'GwOvHWq2+IbiqxAOjunf7r&8ZPgIPt(u)5X(*V4x>oDV;gTQU0`>#yUb=|ESbjvoMqzDxEP}jgQ8#;NngL?V#HbO()3?ISJbCt&_'
    'bVw?OQpp@|u<fQf_-QQue&q$+X&^cT<+C?HA{fPU?e`8Fr!KK70<Kg>kPGk-oMN}8p?hPwPb+KBeOmo?bLPN`xgpx9%H`^?qS4y{'
    '_$hPzR@<&QYSmzEg9NzyLlTEq3I>-AJdcDb4wi93^)hW+^r|B9GiadiYqy)?Ovy|5+|6MK5}arS;YB+vG+<_K_?OHs7P}q_d>@xD'
    'qm9C@jDg6p0-B1Kb1x0V4wWrqM)WWch_|8GN0d7+XF{42Ck8}!^UuW`z=g1DIem@$NvNbYZJj~d!Zl;JlSB&P3FI7a$!;+cv!Xf$'
    'iebHl)d$h1D|ZT3W+|6Hsm$-Hz+%zrF)v$R3x}Z}Y5*G2-RDv`>11-slj>#O{xy<wwPXj3I!}Aig%#u!%Ly@@+OC%oSBF2TC3Uph'
    ';ROqF=dhpk1tf4DRX;hKu)$O%4<Y14h^tv{y<Js}+<#7`j@QRO%bK;G;>0_UJVv2Q(NRsZr`lJ${3BLTHux4hl2VTcmB512LTP9$'
    '*q@$S(wHad<??rtQ1%Moc`Y>1FenV({y`ewC!l&MnXr=M>?)kCx3ihb6x?lUH@w&ivOui+;24@%;)kGjxNN}oM}uO=@sGG&O)go6'
    'JDg-Eo#DsVX?!_UW^*wph_`ZxWOVwpQ44L@OmZF~HA3-l=&%|1Iu0w=!$>9&_0@iX?PV#gE8a`)!ilt<Wocc<y6&H!c;?tcS)=2$'
    '^8NSk`1dEO_TuyJPYIynJ<|4gf}@zJ_T(}i+}R7i0?Z$xf7*W1t#5u)anX<h$<PY$sT<zb7c)gfbJ!2T8Ns2Pu%}EI0zWIPBS*<c'
    'V4`*HW$s38wJ1>+-hW)VqKNWARy|l0B;}NCk;%D}8?y=LJB(s6Bosre#W<eTG1)+N0}_Ws<FREYh{Lr-MPoEhr@fXpMwo(%5Q7AB'
    'qodC{@YbDGJqTeVt{p=X%#F1V8v-7S-I;YdrIwf*C}0mVHt-xR{uQE%bYdQ9nr8pK$r@Ql(UMi-j~-yX@__x&f0!J&I42><McIgw'
    '3HQ$7pB)$J<45F}9u;uKIJI~%Oq`7i{EaS>9ew(t!ZzhGM-9H~Gi-}nd#qp7`b07OM8G`%rP04O<IH}V+cH$(<B_Dpcm0f1PB~HR'
    'prX+sB$R(Md4-4H0#KVhsBSR&)vsBeMa^ewE0Q9kB*c?d;w>1P#EpHxTa%?I=m@C}vPmA^1Wn9fE)xh(R3&(+cEOmMkx;y#a++_3'
    '3XdZDzuhWpG?bDoS?o)Srh4~^40kKaF}-$h3*zy4@1<*{*c0%n&@uT?;z&7}s>rLBa)>$0!YYIV=k70SGJfua7GWa6cus~uo8uI-'
    'CC$*3!lJOEMl9P;;Aa#$kGs_b35$EYlFfcWvK}JH4suixc%VTPFfxx4R%T&*T1PX7->B}Bu_A6O<H}}(EnTc*7@Y^6#-pYi2>9_i'
    '$D2)Z_gEi#LAnCr+s{M@u_jCjYX@o*8p27~sT6`(W8k0p2|H5PlVXRjq*F_f8#+&?c5~bcxNGF||8d^6e#~?q4_ls8?9LsF2tDe;'
    '-#ffXi5n1|1WLiu<M=DuVXMaC4T^6y#CTMlz)k#UuZyCHd@}U0#0~lrmo?5@87^WxrIi~S@J6rgGi523qEg%i<*S4-yRvx?-!Or&'
    'z4wQV(gp?<Fn#l&q5(Wo8ZSKAjo7m&SFK$@z$YtJ<Tuv=4GEsmjeXd?fkJ6y4n8;$m~HV$w|RMZH(%gvgr8Dhz%DQfhGp-Srx>p0'
    'x+G#?M4_=8Q{g^N{EQt9zVej(X?8i@jzJ-p@kCHQX}GVf>9}>!;@FXVSU^v2Cda)!4s@&s_25teL0zk&nB?)83Qe5l0?oRWsLZ2K'
    'if)BCOme=4Mumr)2mwdYP0a5E;yXFud#%irFjT`t4YD6I<g^=HCsko5Tk(l~TaT}8;BehXD#}EMC0;>v=ekV%#C1gaKBJD1gtKB0'
    'tvHKrzz%NmeB+-b6knOjLfQCa4@ze;dT=GkAIED;sX0#KUXEe*kyIsXalNqY+82aaldoOZ&x1T6ooJ|i1hp+nY!n@`$VC(Gk9W|I'
    'j?abRb8VOfuBO7aUoa^zLL+o>Cmx}($ekw$3^FkkW$23y;JOorR1ZT624bDk5zoGp_-gusFIZj@cPc0fDTW&?Yek8Br~!$_G*X-q'
    'hdTZX*>qbUN2qWeK-U_kXS4{!ow;A&D*~>*053MC5X)3B`{~4@g+lDX;WH-_cc~@Jjz#b2_7uGiM%8gmp0IR-ZE*bN;us;AV6!Kk'
    'pZT?nHJ=>7-l1*oM?@bF?%@SpqC(ReFn<Fh+6uh&b*#AK*_mp3;xzY%|7M4mS=bd|{B)9~I$%4PE&W5xNtD1^$T}C#c-on9vPi1c'
    'XYo#JR7!JUPinnfaM=|=I$ZMex)E~6bGZ5mkk|=T5Cow>ROvtVb-%l+9F<Mocrq-aPDEh8I8y7;P_fz2quO$TV&hC7M;|0gV#Y*u'
    '@?<>k@#qJ4R#nFY(v6E^iia}^rTEYzEbJa5Q$&|@#m6TC_=w#D=ZC%xVBP9kg2usqdfFR4q=6S}ljzGLm)ritOr{5k|2}gg){nIJ'
    '69oAQzM^D%?h8{S>%&f6?H6@dO-bC<^t}zzZ-GtJa`HmbU;xPA8{DLU#Ul|{(qSan9_H(OjFmGEOwOxb4munP)H3)QG#KtePQ`io'
    'nR~sdF(^w9LBmL<Gk6}a?gCz!9~>;0Toi<ZC#~^Zv!h@YxiL}7O3^RSZ1Bh#ThpgHo{(aRIBqAT2=Ea|A`F08D{_91)$g8LSIPGj'
    '7{a}{s_xP_v0e->Cn0kv*RCn#jQt|x(G!??IB+xnDMqIbWP09txsDMNv-}yxka%%FEn#90144tsSHO-~p}9hzJE=oXajt`^OR1{x'
    '2tbr50dceoP@>$X3stnT7^AKnYKliYIIK1sf`O)NPGHkyj|*CL+EBz5M@Bn<!>MB!&*=q`+s1sju?gAyi)e~Dh^(Ka<8hJF3(Emu'
    'L~Xf1=i?~dZL;*J!{)wZgi<fC3|iC`Q4%hM$0)o|$pNq0<rEP&D3m3C6l<b^{XOx<`-Uk(z}mv#N!AVumO1}^ZM{gWXTvp?b|>T;'
    'k^Yk?=R5ZNUJK6muCa$5b#lg3kn9X6wPF3xx_bZnBKwWh3iC}ocv0c-)`>ZOvS2CK(wk(CCNMDNoFBfn7Pl8bD7Hqm2>~{r2!p?m'
    'oxl+R><5Zo6-5L5{t3xP2|)@CF=-oEj3|13T>lNEq*Wj=AsR_4xF7&>&L3LOs8AQ+_za-chZYFbW(91Z=2#?3(#uXg5cP4?kV72r'
    '4O5vbZQc-iJq_e765Ry~hKCf5*g>w)2y_5DLC<4KpdcyXWJIIOhzh!DUplq-7a{tBs-$nB6g7J-U~E7e$9HykCz<LAkjmz{J(clj'
    'ch|#1FDFVO{4M<lxb3Hq5+D<`BhHw-uHHoxv~oEhb&9ecUtF}`Q5fwm-c3EF98YG33>g7G&5k`J&RH|}<6}{Oj3j$N+>e7G-iAY4'
    '4A%~h_XGBVZNTJme8~DfvBfe@VOTSbtsW+=a`?{^$gAY6j+{a1rGWPQfFkB@7T>ID%=kDkY@CK9Q?Xb6gb5*GVVJ+#>3}pb^jw9F'
    'ahAf>{zZ~3!O_dcljV)dnS!G315k3fj`gk?PuD7q^yP9G<g&J@U}JdOMMve_Tc@C}1y4MFzoL4enS}{E%7;v=@akCr2{Z>dh@!BU'
    '1K)<9w6%v!wH@d_Kices<KQePEOC_fsvk9wynp}L>(IMU(zJMrDmrea<71tOHgORHt9;PshL9X8y|=ps5G>AIhYwn-O%6&y_v-7l'
    '!whF{6MfqJik7a^rUEUPA`$wrVpif`eWYeh#HM|>wU$vjE++!Q3v!$K4_@PN=^ByJM!J+yco@;J8-U#@Ibcp#;Q>NRFwzGK)Nf#l'
    'pnC%kZvP*3Z<{4Yt}EG2MI+E6_yH30pV-4ThU$b|^tx;HQ)PJm=&rn{qNMb2cM=55wsk|$@FK#-DejV1YM$tS1gIRk+7n@e9V8wz'
    'V4kDn#wg{O9fL5c2T&_iOE9v|CwQX@q(OLsi2VQkCbETdE;W=vf#EVHc<jT2JC)4b#IVg$V(F0Q_2KPdo6u~qPtpcQr0fc`@BP5u'
    'nv7OW$Trw%uH=Y%iZXqiGHwb6(>k-2yf=~0XgxImJrHDXK7*@x#6Gp2ZYCBhaT~S_&rv?IS<am0w6xV|s!4^D_mC+@T@SxNO37dv'
    'X6l4Mh+A@6o@(_8-Ez9{%ah#zncH}}xDiovuQhiH7(Gb6c{l561z~@v6xIV5v|19J${uc^FlL4$!1|b~%{*{vg+1wC$bRccREP3|'
    'nqn-<6EMR6=LBPd0@DOz(*2qpHQ$Np!@&V_91T)u-pz9w<KcP<!xJ$vJ{q3FLbq;0r79=jO&&azYOZ!mlOQ0EqV4fY<CVKO##h5x'
    'ss0~xCOd(Sy)apAp;|0!z75*-`S5KKDMD$wMB@0Eu|bR`vf=3QvFgHyfmcNwQEWV&B0{1ZbRPN`<d)@ws>GPX8^#eraZsqJotsA_'
    'a(P0+;=f}=(8^K21us|-8_$ASPQ=?q^Jnc-QtlX4D0+W<yhd5_e%p_QsA!J4gZb8>Q6{Y|sYosl3-GC>z}QpIM^R7t0cOGoS>`Lz'
    '=;IT#MOJE_6j!MnDcYtwpVFw%Qde-DUKVoTr3P-Cx={>;1?JrB=$ACSuLm%z7Vu-{^F9?dAHtpZoHs37*ahsF-DR8Q*m4jT57!%Q'
    'iu<lxRgE$80SMM5<)?}2H%P`Bx_zf8k$*nH#E6i_z55=_0ss6N|9qkmWYr%qL+iA-kX<0s$(J&Z%5kEXLt$XPFHt2yqmJ|b*;Oq#'
    'h`}|{f?`Gl1>hUQQ^Y31uqhRSOCN-#;_$>FfGfm>*us2KD}54@<Q|TuhHTDbsIxLBF4F1!)YQRlqOl0{%G%4uj>L&TU_m)$Q=pCN'
    '!hejN#{PLjCkXKZ*GFULv0a*nF_26%_`wX3T*vyk4GQ(3!s8PqFLM;)aygnaMX4vU$m@Iq^K2nKWY~RTc8NE`bOk<P?VYlQh}7{H'
    '8(y(J`Eq<Rm4njK=RrAR`Lz|AK|n?t(n&)LU?<K=bH%b7L8**jkklQY5;!dd=_1lB!jvg)%F(|_kKg>qLU*6vz@&He7mc;tuA<^t'
    '5zSys#(`lC<Hd1lgf{8?TwY?g(=V)zIDcX2Fyv#QJUzHiQGhXHn{awawcw;bzh;JkmYbbNbAtR`T^;v!*rD+jO|U8wZ!y3=yf1_$'
    'bUwxdPR9vNVt+vN50(CaFGm7G2aD3zF-;+lab_?`6oA|@`ahh1IocK=bR5e-I}7%W?>l8!aJh&pVp`wI+((j9PtnO800R64=yI66'
    'Y6)WFp4}B=SXC55LMTh@@oi~`Q}|Fbx^AkYA@RlwX6=AuqnGbS;5NG(V2}#8seOJHid1IE*P5?aRlJlWa)(Clz;M0@v_YNTiP~<g'
    '#)sVqY04VdgmT@U3#$UhnSNL}(c4Jjj^@9SYs_O(#}{R<V|`*YNrNd*I!)$^IN`xC{z5><7&4J};`~`BeG1Xd7AwTfaOKK_sH!KE'
    '(2~BkDWN$Zy(%Kyic@I2DdEtS#6HpYNRJ(VF#<^qVMb&K8p(IWV6sjrLy8cVgAqvD#3qAo@&xOb>(Tgi<qa6B+tw*|Sd69o;$#mU'
    'J&)E*n^&)2^O|ty3jju0+GwsA5ua<EN^MFf-wqsn`9SW%4_CjxNt`TsLZOpvl7BzI30(0)A>6hYg4-B5QFd^7adgTRe&}~M^j1c@'
    ';^FZyCTStK6f~h|46V&`<Ko-Q_<M_=E}SC?o<3gu-eo7$6P+`-8nCg3nV#=)s+ynGBX&ha5ail7CDPe5`=QYvm0W^+VaXXWjs<K{'
    'X6poJ$a8Le(YyvB>@;8f0q5NS_=2=I*GQUXRm%Hvn)jn&)&U<xK_T}{go!vg9-St~+HJ%sS9;7`N+0mfB9P*+?nK$NS5V%XCod)*'
    'ZJCX8`^*<tyo?p6kW9tY!YUxUn8rpUtGGO%z`|3rAZ20QULtC6)7r^*%K?pAfD?`=TUg~zM>I6Jh3!Tx3Q>?JjG+ezczhr&;uu$y'
    'bO{xkroiEB-E|roxQBBWHA(~K8z)peaG;X4sw$~5#Wh2)wVvR)fP_DIj-g=GDoN{@2Q8U+6NTnJD5HY2+HKD_{@8OTphB^pgAgwI'
    'mYgNq=P#I9g)X?u`Zdtw(z`!Z!l(vl>Z_`j?ShvbXW5^9KS%!(f?MawGSsMT{Q<{C9D7swH-jo~Cu-WM{!!Y+5I_NdMmP0hkBP&S'
    '^9{%o<$S>f3*fo8gMqHjV0&9aR8+{gwiq^iQRCi?GryGaFuS<l1SQQwCBd;^<pt%hOgouv2k1yWb#JMI(dAW;7Jz6}u=Nvkq5%Bm'
    'u+h=C<#D7I*i!yAs~wkJQD!Mdy5zwHbV416RCI*tz1Ps|tKt*B<@Ek+(Q`*lxup`eWuO!mW3F{Pt{Yo03o-rXDx6t+I2gu3loHhd'
    'Zcq{F)D|aqE@Mb+NOICqKIi|i;&8GfoQ9chtEg>rnWDFoJKy0|Z>m$-q|4@_@*<b=MZ?aH4%-+Ztykor0D5*hG<e;nlO{qxor_Sn'
    '^-!F(SE{k#Pm{jHBJ=f-e$gaoq4>7C!Ihvm=Q=jZIwIz}>zSE{QO?c9*WZ(1F@yI-V~-?FFZ=JZ)=u1ze6|6XbC0{;i+n-W5e$R-'
    '8IGkehN8VN$scy&C^J~fzhCVeHVbH&<%?wTWcRYol7hOj4$=N|gi`X!N%Mt|20!Sdr`8}=sj0y-S_pcgRivK&`DLL&j(|F-^o%9Y'
    '6X%UXc?L+cb9kT@F|<3{aGy?E6_*<RC$Gv1Iv%K|pPniCe`v(%mJRfN9_SR8qNn1}Hv=&-27J&)`VPiD@wn^b!{3k<bBTr4AWRd2'
    'rB<BAnoRT~*ODd8LT<ZGrFrUQVd>le3e<yiI5-oJ2WaN4LQcSH9%bNs$_rK=)&ZfgSU#2^5R9EcOxV2HVfraSGG^bV4EPYSyw&vx'
    '9xh&H#Bt0+m|r_*{8KKWS7qOIX#kb~!W6;l;rJtTjAxzEd%pOZvIy$<fEp}tGpwv1CY3s}m4POrpUf+{*s7@vCwB5wu!Z(^=<^MH'
    'pBUyUr=Gcsw^AO^%gRI9&iUi58{C1;oU0y=Mo2h^hz$97lKqm)8|HpL<8Wha_|qbVowH&c571~Yb<pA9kdC&HHo{Ur33w5;;?-3;'
    'L16!so+gg7LMm?NYQ4>UcD5%N3na5<(%0d+7Bt8q%jRJgT+oToA}E}oF#ED(k26(uG78)jYYqeIeALe$5$VV}n=UWhzywPttDk)J'
    'Jfpr!h;dWkd`X`+4{tBu2&QL-<9WUSs24cxAHc%x)JwiQvSosmMCjCyr=6_T9eg4N8AHslpc<Xd4DpkbbaO%{OwF5U<pEtRnF9&t'
    '#DCWzzR)}lJ7FC=^6REg1Re{nqSyKKQ+`81G}KrYs0<EboZzS&-a~J6V?ImwZHgiJ>_!cdqz?HJ2#6P5Dt(MYClF(yl(7|o2N+n+'
    'GUIT{l&(Ye(MtAkX#U{il!?Qvs*o$mRB39=f{@g|W@jRzRMGRpIFM2-k?B?r)*gq!O#LD91ud5r1%0_4c^XRwVYhrH@?n3Oi;v6-'
    '!H~N={uEwnSDd&q^#8<s$Umz<j)NrK|7KQ_?37X}q1-{G(00#c9AAu3{%lv0JN=#bx%z<|x`!A|PTMX<G=ZpdoW2d}?G7=EE(z!%'
    '9M-1@kN;*lLj_Ztp4ySeL>>EzACzc(w9=T<13v56AFZ<=8QB8qb}?qOgg}&9^$|_ls_uK95DJeL@ZIBUy6}zh%rCf_sNySh!tw0o'
    'o1x}{u*dqN2qlb~PmiPhcL@qne-?NRD$uHmSRC}8oqUVnmIGlfBR?c@!Yj}A>^;fm)X=u$=jVx(K85M*39n{Uj!HYU5^X2>=0cPu'
    'Qa?d8wa(UDmDr*%9e2TI9uWOPeu9JOHAAm~#xbzK*(rmRaPh2-+mFRKh#XFbGZ`n&q>&_pw-dDoP+Y_f^Q-pQf8BqVZT_F)&WTGT'
    '1H?{{({J;J(mmACjutp8e${Q1*5Guc4W>3K1S(RFro_9aqYUzRvsgyV@a*TCx<o@f0+%t6TYrTLDlCF1-iK#>nj+X6V#bo^>q3iS'
    'F^fKlv7M1LC}QD=Bxo*zCqCR-O#5p9Kd~Z{X#V0$O|*jt0(e!f##2Yz+LWG!&bmAuNr6^q;&?*gnCCJ7v7r2ylf5>tNG5VD2))#e'
    'Yj{E~*#F@WM-hRCiC{=PNc6Z%oX&8>iy3I_icpH9dqrRHa5fBWX^!Cm^PAB*ET(zv>~g!&cye9}9Vl_?m8TPakW3d$c7VZaE*wHJ'
    'wL_`uW%B;zt()<4`g#%UXL~rR0eOcLMG!Q}7#~f95bMwjWD8|F%;B;YC1@!I@Zn2Mtqb~%<zrf8cF4zX$N2tvCKe)jimyct&x9#&'
    '|I-;wVkgvHmj=VMSsF9{)2Y9sQgY?nh*>#uuPINb!YEaZ8?wkS<2+9yK}0)|ZZSy$`U4s~C2lBlJ^L+;@Yi6NUmCzTQbfIDr^bcB'
    '07BkUC&Ua+RZo}uhbRixet@71Xq%9b1%~~TSqL#U*V=(qi^aD~ZFy>9p&5$4573oAtSJNf3370v-*6~<7A=l(W5`N%8lcgIB$=>E'
    'e!8P4>xqEFM7XguGZCIdPom$N<pDU$ncGL8H=<oR1}PATYijthXX_q&tJK+JW1Ua96dDaY<S4Yn??!?=>jOsMK${4}Q4(wrss>wo'
    'p1_lU5c6iaVEoG&Y%5UZP|uTFAYdud+W>z>Ke@IC)Phd8yyjLh!~^Fz#iXg6NY4L!LaP;x$qf`fObvhi8UK7j2A1{!{X5N%pbDUT'
    '*9>G^i~c-I|HK3e{I+t<wEfDnYX(lnq_RR_pK@$xSx~+&u$q%qMG4l0O2w<K_};@)L@2z1rA&P{1$M!F=+8ciG@ooEdmg>2%(J=l'
    '_rnIdwX1wI*PTlC2lZ2&a!ZJspRhw<<P?HHHw(l+|M<!S&}0`_J2~En*%=45W1c}7dsR5yB(?yEK><Av*IQrE7tXFIi%ieSIMGw6'
    'IYE9S-^~w%61`6J6v`Ae!z-b<{)5V<|HV*iI+l|(Lx?V=aD4y>PFsx;s8_}LnOx}*-p1z%YK}ddC@V;dlk-r>)No+V<bi3rSm2<_'
    'W)?ThqW}DQ$;N7}L1PJGpC;m=zvxvHs%I%#57BXhQ)6*hYYmzbs5Xe1u(Gk~_`z=m^4;?b_yN)^7eZX_r?~31wq<YXs|xTiQfz%m'
    'DXHqAiR*&SGym=q2dx8n7J#s`?y97DI^u}KjhD@KQMly04+RyeY5V;7!E-GF#EVLvyi#HWPD5_=b*LEfGDE}OpHGJjD;iQpI=0)U'
    '%3^(E20ege$4D{O+U5chT8@1&tRTzU4i$H$izBM=hZjh|<O6{i25VDZ7zs1sr-Ut;<Xo{?3y&^ZBY`_kUq&cpT<LRSWHg*YpMMmF'
    'vIJts{!~{aX&&TSJH;sH;unh8%Kbn&fAsKA^J^(=lo?l06v4xBvQgsf6Ku#k@^A;@wy9FH_@apvI*=(JSiYQ9nzAB}^x-aIufmax'
    'T-OTrk-YQP4ofIb!hpKN)eR047=fD|1a^|DNP>s#gIOyDUOrqW)(CTio%I`#zVPPdcJ%kYDI79YaK0U<)^t~9KOE7Kw>p(nb@*P$'
    'IY?(e{Mx7ndROJ@bT5`zGCmIS9EG*hj0Rq^06p+9#Qd6N9{b)JW83#Chj4tTV;xd%tF!6;@_%Bage9ZEoIgJ*BYV>V*tlv_?C28^'
    'F`WN2ja$%(;6*w`$DtOizsR<R_^44qW0LOn@Yy&XbCCW5VbUMArnHT3?481UJ-{fDGhHA7F5Z@t+7Ac61+i}=0Cj*8g+|lEhf}4&'
    '0SOKic=~diWU_H`AuR0o@G`wa6>wxV`0mes)~>kaTwW{ES~_eBi~o4fSnJ{f07;-M%wgg^PjnM@X~*OUQlo<kBG&U1xx~g+fQM~9'
    '7lst=#SZ|iFDW|f&`ncn-w^x#pyIA9$UBN~%G(*D=*N-XmZyU*L_t0WCj0FZ-g9O<0fv~#M`urk>OYRcP9Sl(ZrF$OS;eQHck$Gt'
    'Lhjs(d>Ot|rt7M^Py2bCMrRLMO2||Hf-e3%<%r8Wf;9%M>k`CDNw2k^c@hHAF>!z^0Znc`K)6J-sUkfUS6PWM8k8|7a}Z@iA6@(f'
    '(}{o$w;UkVb<lRUjSbPM(UczWfDLy5a5ULzD%I^2z{y&iAfdo&DGsM1bTEzs-#g6K#l1#4nY(oUH!)Odtn~n!Gk&Q>cwFCI(tj{x'
    '^H4$OD~u%+JNhI?fNH(_L8qp2o;{gQG|*i_VvkehuOuG`DSE_c397^OMGphnRSUwaHc|W#O`&VG9ghWl!F8a`M1uxGPF5-XNCMBg'
    'xT_3xXwDWq8wX=|UCenoabbaH?paAX|0+;3;%=Xc#HIhoEejoR4<|nY7GZ>Qip!0i!w!9qr)%?pm(it>HouG)y`mo)Y%IiCQ8z8b'
    'x!esAooqh6v$L&4d!%eYTQNDEUN)w+1M4FLUXng(b9g^oAJIw;;|X6@?n&@Nkh&k=i?V6lJQss)i|692wEwUIvfy%&t$M{elIzs6'
    'QyM%rPB64w^<aylZj5ypfOb)uY{`D&Lz(1KKODZPLG{hDAl6pjr{aEmf!O81&8Uw}F9MTl|BG2TWJHH7+RWHVtfi3?<pjCLUe*`}'
    '!?va$fd}i5K)}vkWo}U%9!s@@SnBiG@fnRuCWuGuQ<qxX7bR7QRlQi4dCucSW*(=>J0D$5=%4&==n_`VnGP;WP8n9I&`GTp%nLq+'
    'J-|Bv-o}i!Q%9l+|Al0UlOd&Pl&7!c)z^k^c1nkN)>NOhg(6GQlA~!OU44FI-$t)%x8UeP{><Vaenf9)<rr~(Gkj~{9i9S;RjW_e'
    'QPWy&xg&)pa;7#;9rHA;q$OUmB%TCtkmT$j>P~$Qgo45>;0IQ)QB&ak17!1gD`8S~8N%{d)1@Ap<)C4lxt!*E+awwbjtkh^=TR%h'
    '7<;}P5W(xMB)R#7@HB#Zvl`CJVNn$87+>xoBAyON0gi~<*9t!%mpk#rl&uA6Lu8F1v8%q<&z}rYKTfI&a(x?4LEw@eCtk{=aSk^G'
    'YmfbUdry2!@EnGMTks9a@0M>&d{y=Te~IJPv~8KwaZjwpvpB9Y^az>C`Y>_gx}EAOh?z}}=X@4G>nlXT$`jO&i{iN}gfmb~V7^|b'
    '$JBG%l?S`$Ls6Mhae8j2P{Xzuk<6~*kF=^Pgx&)l=-f?d*i=2GgLDJDWNxR1tE0kyEw?o0I4d6OKbXJ>F+}$!N4g5U`T*&p6Y_w#'
    'VgO3vP3zsMEl&M(cH`JIwmcrpee+HPpr(U<k3D=)895uE*9XL-0G)HpEg;voj}EQ%2@%6wm`A@U34<b$M(2r&oSU>eWNf214<v^9'
    'T8YosW6;NzqHK@`>LA!BbIUl6y8%3jWz7SxsEht^Gq9B799aOo_9m(?u%nYmLBMTI^IaqQMIxs^+fc#CWPzonpD|HZ0IeQz*l>K$'
    'x2pC}$B_v1W0ivukxvvJ0^$`pbK?){r#=Q~lqQIZ1o@%)Q*r;ASDDd5CmRt9L0|Y^QJPmjVFTOlbYNR+49alI)_%f=U^5xob_*TL'
    'Cgoi_fj2m(vvo7dQ4eX*k32!%n&Mqe-(*G7066w2WBq^rGQgQS`>GJSW$h>W)JFiW=-4lr3U1`w1VwqmE#gH5yFxS%DeAmG`TnnC'
    'pKNVgF7r#;V2lC$i<;71Cm@`ch0+TN&mdaq#Q9wl^2X9fhcsIpxt5=sf0L^f3XW+VH_+T30EmHlRZ(cUj!&8rikk7~Ckg89d+D1B'
    '@}hoxcn=zHxGiF}a)TubN7LTr4+p*eJ3EDe2^EjEYI~$>WR$v(Q^iaOX6<@jvkZ2vBJl(`(Z4e`@|`jC--a3r<OT=>)%quWJ?;d2'
    'HF-{)8~M5vw{>V&k*#jTcMem&1k~Gyz-~Fc$$DZXPkcsCIMAS+MkXQmdoE10HS)BW$705z{U9vM6C#6{(5~boa?O%9{Ry~CrAosN'
    '$t^!<0(ZPmqkm&96&pfsp=vX-sAjzz1oXKd-_AvIm11dOE?_|}?db|Qn)oqHevjPzF^PoXR5=_+kf<~oRv^ZF@k8z$@$mk)F`SXC'
    'BA1kbWT5`YedOn;hhH3{c&B62Wg5!&Y6m?$9%qoNj1!#{VK6(VAU}SxxMRijZN+6=4`PaVC3`vm2gyqiL;a6yd=cjz{lv&d6ayb~'
    '7zpmr@!aaeOPo-<Fwa*LGfw`0!g!98C~qhoiO$okyZ2A+E(z<inGJMtBj2V4rkzF#j_KtmFlk3tl4nVh(Uy3dw2e;z<j<0)PM%8A'
    '$1A7hJS<Hraqnm6zWkL)!a^BSJg4#SE5sccdGmfOx%#2QU<jC}cm_=FWA4Wj@_p0l^Y9dC!?Y>W&Khg8<|p2qQxXm*;!-t{0sXx8'
    'fXVFX+{OZmVlh|5^vYB_OVFP!4#$ZeYoj?CIz?!NN_za$H^rg>n@a{>n_fhkJ^hIGg~_Cd0mOJRZDm>X=<8b{fcKF6-2r7f;YN1i'
    'DP+Nx=cz8H#voy-k(<#Tet2HL@X3xRrn_li*z2LHab8qtvGlzL#6l{#IOWaL<<9So{e*NY7A2ab_JBocBK*aX?7-Oxcmdw*?br@r'
    '&VV&9D?XhBd3_H<KnBA3a5T*>ax0q3d7Dq5tT`D|!2Oty4rU^BZat1$We4F<wN>`@QkRv(c$DbOd?{o@1P_E}(dq*R6ONl4yBPTP'
    'VT9JK*TVYeyq|R37q6-{K`Ozl-FQ9*2{@XxDdgA)khKTg-aJ<r90|f;mqrZCCr;gpj(d{90r$15%{OAlu%C2=9w4sh$U|zCsLmch'
    'ik3_RtpfI7m2>Di*D101^#QD~Jrp#E9`H0#*?+>Xf8PjCs^jy+U_3XC9R)uuDib?nit_<3=oyCcfSrHZX$<@*$sAe?l*=1Ago*R%'
    '$Bus=1rfrdF*aBw+fy%_?Sw+<R1iQ@NcWu;Msj`&_1Qfhi8x&un^`Cfq}Hna;$W7@n;W=|;KNLU6R`2xDPz()?^>1x5LO1gGUT71'
    'uj@%t%N=vc=3!VvY(eZfWyei;HN*2Z;;;whLtaEbbTEPa4DgU#XH5QhF+UIQv=PZGMoeRJHU+sE{ZKFSiIQhOI5}&{m70g7u)o;d'
    '(-5nQ%u`6YFpB(osBVnG5rP_Va+u;-RZ`4>Q_m{PT&Ccoqj?|ai0eRUMq3ykdNd$EaTwyFamq6jFczm@n*S_+Q~az+7&M|PQzN*m'
    'Q`kv>LClAN0+<DEkwJFoP%i@6Lwp>|;uCUDTL<r8zHDW!yev|V^KvFxqF?iVOr{O|)uaz_crE956VEKq8<8~}{Tyl0p-aZZx924f'
    'QMShqjdlBcKuLk1qOpxuQfYRPcc>Gbd>bmP%~N5<TgtQdEEXIjo*FkGSp6B}8>`a24%A{p?Ppg~Kx0yS(DTAqo`h5qe?A-=3q0aB'
    'o(<bWy}C3y$O|igXz{)^JM1Qm>uX0r2JB=Zcf;{<sD(*`UZK!npgK5uFt0yA&I(Vp5_m=5D;L0>${BVtaODOObnKyDSEk)0Cq)}4'
    'G>ZM)W_9L+vE_eq5&h|T#nkNk;V@ENL67t=3Jyp5x8O!&25Yr#tzWGNXFBSI$!bg;)`uLBxiyr7zaCX<Lj!=RDldPiWV2pRA2l{t'
    ';^!)2aNAs!ayZ%Y=(MagWXNd>bfe6xe=#-%eq!(p;I=K0)UYtsQ?nj(FG&>nYq6`8Dfe<YHD_ebiPtA8EDQMgW1c?Be0J`Kp<*t^'
    'r&P)pRpxxL3N)01wX97SveRuN9)%B*T7a6DVsRd}+)CCBd?Y;V0aTHayjgs{!d5-qQL0R5Jnmv*+Q}wk=-fJ0C7-Zd6HW5v`tauX'
    ';EbEgvQas(f*bR;Z!M=PS>Ok~g&>xe_R3Q1!ii4>Qax+m&C)O$F!?mb!_z}>Ms<czlQI?^&=ih%odX~v3Nr!VrZ%=Z_HgEC1dnNR'
    'Ygmrix|o+d0hf+XNJbu4FhTxmsFsHv_DZ^+UFe&92*|mF8PZMwD&<=`{X=5$Q1{3`;}3{8r5|3%KGv9nliJ{9R}Q-Yj#@M=YW7z3'
    '-zo^MdPc}qG^?Ph!Q8T;AJgN%zCo;kzH7=MRuy3z^#q3%J<$;C8%7Hn{4gF)hj;FL@3N-zmC)f~Qq~@zL}T<c$g@X-yQ(T_0EZ`f'
    'K%>nqFw>%81+qxyg+sl;VQ<9dUm)laRHC*AoL+@XMnxq$jX3BOy9W>+jqQQ}zT%JhWxJo_#LUBFg`gA$v=0+9Z^i@WcZkJ$1NMpN'
    'l|T!UM&=P_2#^VB57lCH4CvH?>)<d^r;_$g7MppnM2{E>-W`uPB6U0?k_HT1ToAFx!JG&2z|~Gi@w2ehP#&Kmcf=0KS6C&k^sbci'
    '_0)<~Ju1*-zU&QUCaA@6GN=4E=nRMl$ggr8wx+>k{t%~=k0!abig9FJ@LQUrhj(9uN$>)Ame<&nr+Est;jG+392sVc9DGpRSpT9U'
    'Vt&<23A4o7*PTmYr&ggJZqkXlXfnUV@O-$P3Li)Tti)nGp=Jw=QV;zO*5F3O0QAx&djdz%^6+#>B67NXn)HkpggiWtCulA7CE6AE'
    'HC@J@yruba9=0wU1cjR{kUrrtt=L&`e>Q(|zmIr57FodyDLcmen!StFxf7c}QP{}_z_@nu!0Ay(Gm>|8$z}sV(GQDPV`}2w)aaS='
    '2<`UK;iW);LWUt*-N;8s02#A!UOi@CRg0#0x*#6^Vz-p@PcBBWXK6yqg`&0B5!oyB&hGJ5wX&w4xG&66*yDgh6DTICOe?3ag3}3W'
    '9y1cBg(R~57ir9@Ono$vm|-k|i^XtT$Bzc_Qhc;(b#_U_Pe^9q6Km6cDommLe^`AV0QVjCdY<xqGhvP4%$*Vhs*YCN8Yl)FnzOY0'
    'dCEh0)C}LCNRQor8sOR3M8}CU`E42;<akWtYjUCVor@HU<=`Ngg47j7rd_36j9rti$J2e~7*=i}{Dbu)OGTvn^#F>^-Gw>9gbnE!'
    's1Q2zBaVRXoh+W}FwR$*`T&vC0xq^`#<pk3a}wBmfTF-(Fr@MWe;7*;=jL|2`zS#j|2iY#tc!&mm=6#yOyr7q^|)Dq7Ho1)PB&Q8'
    'g{oXi&pHfs7*xD<LdEW!>Y)jZGF?Czl;xi<zp^MK3GRoZ>C30pX}BIPmLZ4tCf`_tvJVL+;()>EyCdLwqn+gHCVuJz;J%b|ZEgNQ'
    'VIB=Cq|a%`lcR-i;^HI?0f}Rsrw{~Z(86$y(<Ttuh9GB+Cm>NOu>LCJSW!;RkPYCV!N$OBuqQC0(`W7Elf~(RYx#u>6er+C1fJ^w'
    '2YxgkEKb|rIJXCI*<mP&2R5v`M!<zL1q6WnveuET7eqHvC>Ca0)QR?h<KdvcipUhY1EvYt=E4sxcg{n3grI-Up++57jsmA{CiqmO'
    '`$A%Bs>g+wIUmrXWNwW(nJ6X7C5HWGHjh2oSZYG!pP<{NwDtq;gVRzdNQL#F>dG-4K4Ho)%M}gf$=%|2B)j62b=?#~yP2nz<^>ng'
    'n?L=G-<h=Ucv1A{5KKIx9oHR1QV6P}3PesQ_QUhoVT^TYpA^SVcb(((IxI;ePvS!#1;2`)yR%$3KW(z}=Z-VNUeRCKtL>n%L?|(d'
    'wrNV&W5FZ*@2k>TxoW*gumP(AWjpxkgGP*YmOfDD%cEY2C_2pM^E=w4;%;QpA{nMGKk}^DnrQgGz94@Qy~2)cGbj)OzdOY42Dh=V'
    'zqgUISS_H-;swq*714FXzEDk1(m~mL2EP@x<2u==ge_;N2-#<|bbzwAGY4!|nm#jjBY_^%C|~4G5Ejy2hrL3RS&ZU&!obj#qGL_R'
    'Z_T;cboB)5<15U)sq3O*flJDJo$t`+7e1b~I$mgUWxR@La4H?JJ-L`m{8P>#iVc6Z=1#3=3QM}Gb!yUUpd!C-P@r`MaaM9=xfi)6'
    'A;oqI=o8;Fkclq27fcLPKWh%3g)2Z@&A7+%;EEg+-!vz{;%CpeN3ODjHuaJ_Z!p;DN3W`q)R)%%riAf)f1CI9fc{i{b=a(MxX+{7'
    'kAQ;M4}F%26QC6PG4uSOOP)pu*Em^-24KjU#du*_&~hJ#n7bZu;DruCL!zUZr`-mSXN~VHqap)sf%cc6D&Nz*iCAe5D~@n*q$cqE'
    'cCt*JV9$rUg|b+pPnb(dC^IqtIT&Ofcz3*Ha}mypoY(R8K{a~9LGAyrzQcbUxt_dy0IVnLFjwMntsGH>a$U-Bsnwo-dGyF;2dagb'
    'x(`YI!cPWDh|9BW&szmW*@YD^Plp#6TIVd_g4sHdkQHQu9J-{oE&ViNL@iY7-^a576bozBXu0f@0K~|3q9`t>TS!hfE-bv?GdDmX'
    'Vaj|scq%w}pxR$qxdr(u6O!FLWD1hN(uy!Cji`~HM~-h`yIgqf5nWD#kpc2gr0;E>9nTdSJx{6P{kYUpAFcoi<=A%87wSInELwN#'
    '0Mz9?*@U6z1c*{G@MJywc(H;ROYZT@`tE*Bxd#v9Cr-g4ATv>H08^PMmGN+i3j?;sU<J5F0ggNoS)8VX^g7s8U`NdM62r83cvQ%O'
    '4^J|}6hxusO9*QH1afZpziE!-0nBmZi;mkK-r$==N7K}B8cR%DJdcfbOd-dBg!iI>=9m2}avgp6PBC%9D5usjQssjO$UEi{r%=&-'
    'PnF(ta`IrpYbVGAd7-5WW;6>yTr}vOzy$#*P;);gzGc;{0rGtRvmXi;1r3k@!zegtz0<Fa7I9Rv6?MJ>c_mW`R7p`EfCU*+G*3-M'
    'YaR}jO5=xCvaYVanB7oNDCI_+lvR9VGd9czK@mfX+?yT{nn3O}s$QsC%AW_%+wD~LkjA{kE<$2N4O<5-{DTrR3yg`dx`9Sy)YO<}'
    '_!C+HjDVe!KStIvHC!le^Egvj1M3N<u}aVagsXN`5zbfp7+=!j?KO6-2Lv>8d5-dtv!WP|T|LUf%EAcPirlaO$m#~(PoJlVxVHNy'
    '4O`=cqeuR{Kj8hP-Z5;q5Yt5pVUTg8-v93td7LJ9S&{|s{WuHJAMwv85H+m%fRZ2Q$xeg+>GMV}2g0hrAP2KD*W<fdl_R96FF)fO'
    'BdLGE{3fSighWx!-@WAvXN(|?Jjo|`f&OoUH3uuC5SSZp{2M#&x?%Z&BRX%GFpzz`i3!tse@3UrI8?wJvQ>8S%GsA2KP$`?6E5M#'
    'MGT_HsQG~1#JtgTipoRC3E3uP+>v95<Nymf|MCcd-)*iA_@5c_jo;ApM#3=4{cW~(^J^kbXDmV|<z|TlysndMZkXk!`$ax$&95F8'
    '+WgD94d(E<!{u#mr%1bhz&a2~^TwP)zxQBdQ$!|V05^Zm9vA%p6aAIJV9Kn;j*EQYMoXIOj+b1Nr1_e5QxFpmE~ZU;NSkuBJ$o98'
    'I=aKvOxih;MznbTK4E)IikYrN{nmRPif8Pj<X)_DGyBGt&<R(pT5!Gru}^4oD0TGK){VRqp7boXYf7s(VOpZ+%(V?@t(RxiID%p?'
    'BxNj%z59&~3;Uo65wvPQ-#1J~4VelKp>iS#{o<6!zQY$9{&n91+Q59vhBT1#ZXAzba3Nfli)fXb2U*&oI^`_tCp-N9RNVw%yp>*V'
    'U}d}u1D8bA`LX=netQ*1BV7JRFJs(1n+!96#sI{!8`1(6z#-!qNBefepWp^ph-k!FXGLGX@CZTF0?tmX>URYdcB2`wWJu1<^5!`x'
    'w1iw1_BkE`8VCN_7F<Nx)owO#o3y0zPMaQ1d;UfU*;0X7)`A(w_udMFfUtb{f2;}ma<!Y~C^(p#%$lXu=kJBNcd?3f34)^E=(`-W'
    '1uNp%3uwO)&8-~<FD34H$ko%-7tJ15Y#;YGZYX2Hq$skRPmPzY3o&3{54k}0K!&6PH^ZGfS;!j2iNZARH@gAfLr2-61XjMbXVjqr'
    '0l?v8;lBIb*(p#B$>4o=h?B}5Y&q`uTJ0HvMk14ZzR-@hvGEi+5-i{WmjUX$SFb#Z=QanOqL+s(ybR1HXuvM9<xOOgLdKbz;X^O|'
    'rm~?8H7erHh4bYhT=Rw;!=%J_ng};lrlU21zsKP>mfu_<>Tm+$mi#~uxrfFBg(VOKcHA@qQ5OrH1UvCgE<}MG3hXw(jyF#-2?j_a'
    '1BlBP9xE6Cb{gy=jMlqT4_>VhJ;Pyjw)@hQfgHs1)$U$e22PkPH+{zI7apNRP+Urlj=Ro~NFi06*h=Qz^PBlijIa4hd_RRq{aP%$'
    '2swEJX18x}UBMKyP6-R@i4?LA_henz1^~C>H&-8dkaRge@mty5LDB#q4&D3qrO#$x9Pg_<ol$P8Qy~ReL#P1i7<UCwR`57xvDd8|'
    'tu5n_)w4>fa=}Nveeoy3PMoaQ-xfPiz8JhQx)t;Gh0MpmT$C4~`EE|U8|wz*2G?lZ+~5fd0N@wWfK&X-e|EahubHh7Guur$txO5)'
    '`&0Gq)Lvboctc?8yx?cdM%fq&Vc#q>-fvos>d!3DWQ{<LZ_wn%Hq=l>utR5M7vK79Z-u-9cKHC;QE#SUEW9FPv+(*|SM^a%i(EA;'
    '?Lj2N0zsaqb0mEZ0wsu{nKv!vTu4|ysB%YtNXEfpdYK0bg?M|rDY<iH*=cTr>2AF#HW99cNgb3+WsGOs-bv&*^}lm|8aF#2s217Y'
    '5U|IcH6APafccL8DcVDrM4pw(pi*(dt;UVJ75m77wo0UFi5mzB%3%z{4!`#q;){`WWw#wrezQ)B#bV8OX=gW^G~Kn}96`A=W?K(w'
    '%>diaqZth0)^28mxMdnG3B+r6r#&I?2AzJi1nnW43ZDESb!a|#B+}n7D>*$-l=rcd7Vf$m1%@uwSk`V2Io0DtFUVdgu<puxbaGn@'
    'TJtb|2Z^bG(FVvt3}9_N<sQ<Q22sJB(%anu0d~2_k?7oS;<EUcy{XGcO4UO)LhV@$TTpt&yzL)0sDm4TW8dKnPoL9-0U>7fMpLHo'
    'E^|GezE#@ub&(xJE7=>u8m+rMZ@0adkX2>fpdT?BWZjZ4CP-^*|1!E*ugO$9C5ZfYt!+dkxtfb;6gOovaFiAzImf+0twZQ<FPB$Y'
    'H_I&SkAfRzkTiVLVMXd4zzO+g{BJjY1dG#tko*3)JGA1P3DhH4x@_K!)q=DM#%epW=kB5fXOrf;f92M~^H-6fD}jB~`=_+<6yTVM'
    'o4yKfc@b7bV$}7F61;WRR)NK6x!jd!>@iKBa<ZtNFa2eR$rB3Iq0a9;p0_Gq$s}fyHv7C=Qt|7+6jBsK^yl9t1jih>k?I2Cm^bfE'
    '$R29(Beb5~ERBM7S=}hE84sD9S<&49T(StuygA1!cE=S=&-m$Pxm5Bx1na(Fr;q-SA{(z2Km?HD_Kis;`-YLgfE+g#cI%t7wk%nl'
    'r>!&H8vPqw7=b?d)cnY`0Y%77NfTShE~v#*tPcO$3CwpvC!i4j_r}1-q8La3IVVMYgH+JqZ9@Mah0)K=dX8r&I3RS|@b%%=d0iKB'
    '_nJ2V4tH*AH;p6MBMnR-@l35he^HrS7wTD-@F?T%IVARBZAWhSH+a8RW9kG>1$IHbdk&EmEMUX}qTBPI!v(#cmhl`&-)@Sk8oY#4'
    'jZB~4;4xf-GYj-HGRpi$lhNq|iQ$TasQY(#3~QbvZRKN#Z2#tzl?CNgvaAFI-=JLr%4gAAo(@W5_^tlRs5XS0^lu=MUPWHgrnf6M'
    'sJqdd($t4;oUd;nDs~S0(7zmF*xg7+kDp#Il~dnfSB8Y*I~Ox*sW-<D9?>>8(!1RCn&<jiI~i9RzCL8u;kkYn8fJs#!<yfd2*`Zv'
    'g1_EPL;_8Tx>hVVPf-$!EVY{EaZzq$B5Y{AvQHbd&w2Cw0>Rw|s*YIm8Qm(8O4}yYl)J~T%gMYs(|+epYD$EsrHH`1KV(=1x&x90'
    'ZRK*uuogbNYZ44S>sAq_*z!<Rn6-zn342?wN3;sw7PkY(fz(aNi*oaNH3c|!wff5H51CcbgHxE>$hT|U^bR(e1n706v32KR%r9$5'
    's9R;%e8?4ca&cJQLAl;lSpWP>1gbXt8;s+~gO>e({jc4uKfvj7`vb}1&+jT%)RRq!lCj43?t;_t1VzV~%<c^Jc~@9CLmBk-DHfl9'
    'y+s_q5T@0^$o*z^iBYr>#clL`W8N`5Ak&dvTw%RE!~!ZAwhehf3v%7_jZ>+Z_l<avo6j_Dk5lxIWjB33Bigv-z5s`$l)Hff5)}im'
    'VShPtAny`=t0@f0^qcyFOUPr7!N&gCZa)PKn`~InY<^?>4aAG1{Xn~Tb1ewJ#}DGb)?ao^A?jjyPk2SU0np?xMw<&pO}}|4mk@+M'
    'AT0KqKSfbGdGoXAsr3tEBBWFBJRbe*aWnXYh)ph}K!P`oFO=IY?;ysg@euuGmiAF<`vLlL-n=P*@-e2f>))yO9<@yXixIQ&=lTv!'
    '{(rqctNm2C4uX(dS7BbUlIklPqx&}y2*JBo#vn_V-FbR*yI8zc`PSs2ueV#bO2D6$AM_#jyBqs3N0Uh__C|}wIj)S@;=^@kor{GK'
    '&*cHJTV-NvKb_!940ax0IDkbqQVnY#)aQPO4uEi7y(x^kj~nd^fpvLwa@kyB&3lqbRFt$VTXu8&tt5Dv{E7W9OKqxKKA0pZ>c-t@'
    '(!iKBt4qF@I_}&tS@bQWR4_<l-kj38y#~^==23UdFHDMnh%!Oonquj0E`nAAcyeRlQ13iljPVRuUDgaw5kMOc6A)li^ZiCLV(QDF'
    'QpqZ{M}7Wb)<~wvdy>^Co{8DM^*$K69yHEK%K(q{Tg_tw_Qy)J<~Nv-ObKOykgIyW+2MR=WoC7m>s`OuCdeJ5S&NA3A+0_ioJaDe'
    '4bXyjPS5}KxD=UpQ#?av9>T{Kz_t&$KoW!VOCOZcKJWTquk=|gGX5`(zB!3|1Yc4fvNlqEIzUuqWZhjJoUnX26*g-gx9;SIvRMvf'
    'dk9*F{E#&iU68B#&0wGQ<)DpDeDI1pZdx0h0Cx>O9un>V>7FJ!vDk43_2sz?KmwpU`klN1@EE<fymcrTpRt=Dldq^ecTVHDD;Kk5'
    '5>}=QkblF-4Thh2yoaQ_p7MZdMl$xyzxo{s4}UqW7plZDZyKZ2!_)&+0@9m02j}y$K3`qNFKlpk%)D9w`Tom|OB||<W?DGTtKEU{'
    '5LBQfHm0<3cVvRAfVsQ!n#30xD#1PU)&)7JcN==`v2p&eWQu(Q96Hc>d2k9o&pYBKaWROEI1SAD4dfnOc_AFrW%aqi0!aq|5tk~o'
    'Tfei{z3%+PF?vU}=p+U4DcCh%IXUAB7g(yMUP`OClXGD-@K%&@(>5eyVgP=mFvz3-ygl&~?Z_IY;L+QYblCujzs@6p3siD_6#~}l'
    'oxumL+N@3Fu3G9na7Hp+dNqCBA(Lq8q5q`I(Y~@6qWLmCraC9{?wC&4=)~kYAUfURWc#+Wvraw5X#&Pc;8+Na^+w=RX=vk!V_~h|'
    'VTNnK!G~)*i)!CHX~GII-)!9h7@*&o5OUS6&LT5tWPAey7nX_&TXW@!VXm8MbWS++c?<*$G3y=i!WcFw9KBMMKjS>8yh6B)lkuzH'
    'Xr@M1<-uuc2)%YQsV32|1i5&m_}po)zh1zPVL6Ku;>7MQu;+~q#~(w`o!!e?i#EyJ3hElqn49?;z~aEgLKD3qsgNAnIY%h%t=~!@'
    'E5^|wfna+`eLCp+qpKo4+U{N)oC?$8g^HQC8ZG`Mr>}e)hd<;I9bs$&p05`X?RVZ8)(x1AIzpP?m7@Y(ijOz_A^Fn`2kS!TLqpK+'
    'V8vuGa+qiSG-B1;=7vww8<e5-kkKf7gh_TCyqNBHIE&7t44Iw5q`VyuPRx2)9P_|Gq%nhoP?0zrcd%~Nn?|QaxQ;uQ3Y|kk;>t5x'
    'pVF8iYcGcFptYGd=+=Vn?e)92IodZk>yDr9WolQpySIvAu1>_7#6AU3{O_Gmw)0`~-q{_mf_69R9VKP9I}!p>#=$6pGt|1{=(MrX'
    'h%`gI)vsH4*!YmYXyGq6YYPic7S&@@|Hhp)htx!-o(L7<j<TwEfvidvEFX95GC_(+ZKOHpe|f^e#m;C|ZJ2r|rNBq(WJ^+2w}^S`'
    'i<>+B6wpe&w_`05t(#(lzp!~2av^S8#+{f_59#3B&QHyD^M;b2({-f-W1T;wt51?wCxy16k-vGr@>IAQG%J`_Hok+poG3TA;V34f'
    'eRD(trK2ptOr~?SEO&LNGN@t`knWa{XR%h)5bSwnxd#x5cOwbO(;YDX{$mNwYj>HZbUQGyV;$%Zk>Fd&waxK>K}=Y8Lotk-wdaDF'
    'r`^27n;mgf<=hQzJ-t7KYt)JC9!;n9;kADErkSDZXyV^=n|8}OY$|hFK6Rt4Z*XB%@VD_}!2$D|{Y=N%EFV;oj9$M3Y9$N5w#eng'
    'NICA{0$U@~RhzW=4Y*n%!)umW3wnJU?e+vXvg*aF<M#aLFa|TXEIJmD^4s?d4SeNzL`2K*Z$KpB`EtDKfb-&Jtll~85x#DO9M5ln'
    '7U-lKx&N35WAA*<EQ0XnsfKIx=WhXvnQ3+VF*#x1ta7M%2I(|Bgll|*$;d)l8x9D9hWU+Pn-x$p48$(6z5#%KVLc%B0E~IG^)DFA'
    'iV@cH<YwPse8(Eti3yo`tiF@Cw}A0LhhqS}9^U|qg9(;7>NHyDcTeHm^b3fiCCWFzc9649;vEry%kPi@;}s&4<#6lY0MO^Ol^9Mi'
    's_N^;WiH~yG)XyI+%kk^U8p-PsMcrPQ^%rP%a#wfdGlNKWy&5F3~j7+J8X2L-2)K9o|1l^uN#73ig&X+cVX^k@FLmF+MT_UuAaG`'
    '-kzt0`V{kILH^WrkNUg|2LPFmK!yb~@8V6vju=lg1I?5i_Yhb|Adm*@NhhS<ksg3R2Z(McYU@S|v}3H$GvX8ThirqL;M+z1K|vLC'
    'hf<-Zj>49<{<w=@0_8Cg51{fZ{t!@{_OXK3l40XrutuKaeyaAWAc1Z)U-a;ua7Q6M;vqd18t|cCDhrZ3VKWAcydzU`zdpY~yBJx_'
    'loWI3JJ{paT%vy|%ahylYi8CtL<?O3yrfO#ec{0a!qhKfoZR_r>YpI<9ZhOfOzUP3w&2ysRgk<hwLq|`=AMQBpb=#J{9ANB>KI+r'
    'JfV`Ul<yCzQ%lx#LTax$s8Y5r<g(14B23!Fg+9RuM5+&N39(@_6oY=;@IqkEo`fZ$Ch3<G0o$4>G?v>NZ3|+7QQ@3zR}XV7ooJFS'
    ')4>E4i=(v2sL9_uKP+&NSi5ysI%6I9Merpzgua|({qo+`zD50H7j{g-Aod1V#h0)y*Un@*maN3ydp-+JiHO~V{M9QX5euuLK=~me'
    'e+5G_G8oOE3K^G@8TgH7Ew(8*j+cjGDvCxqtdoDI@HbL`Oje=%1))vb5T|+bX~so1Z!^DA@GSh$ctR|z*xH1#$tB{NRK|C~Nlo7g'
    'YGy-EIzSqM%Jhq{G|PTSKlw{Qy(7U%AbkofzGk}+kgSmDnNMYYH2?TqUJ3f?J|VEw^_LEYT-b!`Y)C04?>~QU9_4-zPr5GCPh&Q^'
    '^p7T?bHOmWkO+guxm++QDIx_KiE7?{5c~!ZbpemL9DoVW(UYl%kNz$r?;+4k`+|5Qrcg)@@}K0SAf&(`K%h~%>;?J81!gZG2(j9a'
    'ndkakcj4V+RsDMbKpYB5j%f9}^r~>l0szgd^|Mn1@@be0<UbqhSynQw$p(zuAubkE-1qvunIL90o_A&A)!KurnSsG9AHm^>OxXm%'
    'GZ1A2GW2^2^z#kLx2@;S7xM)-1!F!YEsBN8@G?Ev=+hEP?y2jwX$;s#vM}{{?lfxxXz6w354l(p_&J(5GOh@V7UTMyvQp=|F<2&x'
    'Zgr(gy`+Cy^>Si|o|@}lm;5_m%@X@(16R!;C+f4Ur@TD_im!JTeTp3<sq~;a0_qyqtH(K;d(y~TK3-_tvE!eLUC1U3+*|Ze-oliC'
    '72v(Rak=M|b$LHpmJpqnodast064J-_0E$jWTRnUkQm&lPFMbhon|WPG<t@4!VR^Y{%C#lK^&Dht@&^Hq3Lu&6T}!zUG!}T&a+V1'
    '`JNqqi%fLAm;}7zQUWkF&DUp$&>K43QiV}9^BB!?Il!WU_CnGYFdnB=&F3#1DiYuB_)FzB308~!m>)W!itq|@k#D+Z<x=_Z?B}CX'
    'cXk%C(i*s5@cTJ*><g8IGq2ESOiE^DLCFF@Ig6FAH!icTrqYK6h-_8<Eh6a7q>m@3tuL@nqJ!#1jRmSL$JNKekjI4FhY3ENm(LR;'
    'RyLKlH9iKdAEk=yo)@ki!B1ha1ir|EA+~*O!COM}SU%jz=dd#g@YPlDB1k^P%8G^D>*eLG!jy#mIXx>p4n%>n!^m=fGA11aLSvlF'
    '(XFW5CuNG~&p73F`<^-B=A<-Y_RXgkejz6`oxP~ULea06+4&%c8A;1P;odjLZF$|i=3XDHATXS~8;m5QE6Y<T0x#<w5)0P#yu@*A'
    '=ulF{z~JoQKPTH*UOI=Vartr}W)-L@M3wbz1@rz2Z2YhXCC`u$MRUPYELooC$Bxsme0+HlhFw|~Lkjj>mj(?0)@<%JI&b!MkITnF'
    '&r4*@@L(1~TLUFJ-E)w<<-_PpE8G^>3Am8W9eF?5^oc&8P`Pv!I*kw`U6-R{*h>u`+QaF-UUqtLeXI!}8$0%5zOB-o@LKq%ACA-w'
    'u(Nqo-0s_~@biE4{N;w`Z<nEX$1e`2u|w`1b*IPYLKhl#VCDTo%~s%v_QAMWQJ2g_s2Din_YaQ@W?j1?o%H5-gh##uHe+j7=eU%n'
    'S<vMvLDvs?fRK`DP>$4i_jNhIi{b<&r>PHQyDEy6)jsT*kJD2pM-XyY=cGj~UM-iOOB!p{u*)(}P=pZ01dA2mf7qRB)FoR>?){fy'
    'WsyHU_0Rl6LqM854x6Zy?Za9MHaDx*l>?H?UEXttbbwj3Xd2P}pbOcMM@G4EA?r%`FE82!b;bXt`V226ETT%xhVDI;Uj6V?UR%}b'
    'L!mXuqCr*g+i~E_Z)9Bx{k-9MY}NE|;q4(8oCr$}7TvCOxX7g2^h8;}hbCFgG={xg=Ji#@g74EnmKLFpH9$_4#E$j_<2l^-EjME+'
    '+a*pIcwv!D_fFp`)}IsLkivqdG`urT%qyRwN`I?d^iNp-I)`BVi<3DC1#-uaRjcqg*QJBKl`mXVBaTXry}FuSqSe2EQ<3jKYjxWh'
    'qsTznxnsZgtd6z)VZog{nww%C4=Vjyo{l;MYJaZE2;I=kUp{}}Zd9rCgy%yeROaRU&h_UuX?uuX^5GB%W`N~IlF}dCt90D!jZi`o'
    'k+Yi9Zr-4Yq4&*~;P=-pm%J<sb0-{l4up@B=H9oC_6lyL-IKSvz2;_Cr6mxr6611|3kr!U*sbmS@;pH`<=STzx%AYu_60g!C;<i8'
    '@PI5W0wCgn^?J=O@3NLk^evROZ7TLWpL3>oO^xcyfTFw7(di`8HD!jMWq^Br^HfcLJY2v&1{yZh*Z5C?T^0~h^oF8eOjFnrCw)hk'
    'w0>wMQW`r=mHxMoT6FK!gJ(VLl4uQR$7FzW!nsbeJDUJd<c>zhRP6>+s0I=7%C74KwW?CA*d#zP;H1WzslE8}O!lInqouRdWT?tI'
    'S)CFq7({;oy^S)UtZ1(v#K;;r;Fy`_+CBdGbZcu@54j2K&rceO`g~l<2OcTwsD$}J$znW`l|o|Wep$+A7gKJJ4SIR`gEOY8N`M2)'
    '#q`rIW+MWQCWjsRrxk5tT{ZVHbPz*SzUO*)aA*b@<J#kiQxQ6nf|dbv8{acjz;b`1McH9oD&+2m@hnaRm3fw1e_onf-_f9+C4t$L'
    'mUEE3L$+ET8lF4hF=Vx~WQk{kGcVK8+>~g#=4z8GQ@&-FF^2&)t92nX$c25;2@7V@o(t9FP;-rZ^2%GAE1?_IzkIl97l@6(;VH(4'
    'jmgGVlFJIx1i~S}^aNvaIE@&%{CXsh#_rD$5Qdszu@OZ!*Iwl(UI`pyX%rOs1IRmKa&!A@@csm`njaoU8^l^2_49-RGc+skwU9)T'
    'Bbx`n+{b?Z4T-^Qh>bm87oH*+T-Er)C)LZPbRQJSj6j$XHVxC5UrmZi{1fxTr+#@yeen$xI*^#<2|f;FuQShe9+$l(3wIzHOQ)n@'
    '3!+YPlGB-HUUf8sfj?>96pVQu=(`WZR;hja(_kTZY=)X7YnObI^A&K&f4MvV7+a3mE-y8Tr(6@A+N3kd3|wXy<QkDu7f=Fy(FMVb'
    '9|{_MxLF#yX`=OExq8sGpj8u8Ry1#1++=Ls1+x6*?oXS0fXQflP>BtQO!NjjONU=3S|mDV(Yf@45jqM+lo^1;#Gg(f^TnYeufi7x'
    'yDUgCCIr79PjhYp>vA>#Y=E{yAXnSgi^3w$q`#EwC^!!}`EpJD5DYD83s2M`8jg<n+1oz}4KT@QX=CMYGeqq{bW=E}{L$|3FF^#p'
    'n>ja!Yp`x|Vw?4{rF9w7dDpA<+og)n80j(YAO2yx4fZIFgn1Ckq`jfGJqKGBu60?>bZFCYj3CcO3unT}>3htai<$i)7u8$Bfyoyy'
    'JuF*ZIG+F6KI}Uk^eVMD=ex_>zY$0p+5xb*-p+j23kcw@u!-2VMYalfU3V!qkLYut?9aW*G2U}rqRqjyM~8b`m-n1wZKdytkS9yp'
    '7vGr*kCR%f4~|TkQ=xlp7cexgY2-VkwL!^o=|;^~H;Q>ywy*qj{$~>tV3!(mUib6d9uV%au}>^J$XzN{H8!>=7hP8uJ+N;dRB^t^'
    'eAS%u68(`?z95f;I;L@YaT18~A<nHmehvUlY+CJF>NDx<upQ}(cPK6c`!U)k>;>0Z1ztVcpN@_Pqc4J?{L`15QR_<b&LV^TKbP9U'
    '=vQhnq`SCfLlWUXW`+}^9J51?{=jb?_PZ>@?Js$2-4rG_RLDorszFSY#~jr)<!X!Z0+E9agKPvX@r#|1{}A6vukpu96SS?!SNEj`'
    '81r!-ItpBaxCg#rzPUjokkMOL0BSy6>^S}vCb&Tl4YFr$Q$H!(^5zdjC~{Q##mcrnl#diY7-u^^Uulr@`MxgmFdl=frMrI6_>?Og'
    '^H<`CeVDk#R^R9#>IbC8=*{YGE-Fa&a{Y`*xP1PZ>xVj)rpiFw@b+BU$KLeRrPlkpJH1QcdrY*yLA=7TQ^kNe4_ON}7^5>J_rqn#'
    'pPczSKD}#w-?uhDpF`Gd(+dcDpZ<2ZJ$K);2yM|AO~ybXiA#dM40ZsTbMqR01@tx7AbMRc3pGXz%t#$oLY4<qd&MrsGa4hg{7^Cr'
    '!Ah4S{|~lT8u;tO8Xn{e6F-bI(XXWQkx5ud(zKH2oN;Z4XQ2~2u^hM!#5@;q$vF_s`Ko|^cNp+@<YGLjgNuy#K|(WIYwk)P4jDtr'
    'QJ9nFfdpj*<0>R;ZEB9`-oL2|f6}9wx%b<w4cT<3FWHbTaBp*4y85beV?~2o^9MzYu|IL<qo9ZlY=5ryaG^%SvqeP~_z3kLSE41P'
    'P|Jnv(v#+~<zI84tB6U5qH^fBNn`)utEE+9Q^q&S)J|7Y5@(jQtc9qS+8^D>O6qH_dT29gUx4<OWT_!vu?M%{DRv9}^x-J~$wbHR'
    'SQpAtbaB_$)F(OTNbyiy)!dW@q7tT?U2|!B@WV`5SYd40Xq-XWf=#!4VXA4|zsvk!$Z3;@R$#(;AeRtJz5~u&Ys7;$V+7HIUsXN!'
    '4rYBZik@8u)VeN{Xh&4k^BU<d=Yo~%RDL<lD4!=Mp9eEhO-B;PjLI<IkbhbpY>@$g_#@@{f`ZOs3@P-?9~$QyBt|q~`)<ML;eM`4'
    'nv4q^LWZA3f>s`2m73t8A};L|A7v}XT#|@98?g<KMbjL)bY1PqH`lgh-=ufzg1<5<%rnm7%sa>RXv`3t1AgwX`7iV2G?g6U8+7|*'
    'ysvTenT$0IL3S{z6U=rX&0a3Y&843KiwQT4+;5eBMxN}Ou(k&CrTUW}<vbRk+SH@$UKoMW`OM2$^u(|9Xtgg$G*?Aw0DjoBLR0Tb'
    'NBhYyZq~&H$;A-T<PJRb;#TlPE04N<J-oQUqZN4=i7C9V$arb?WuY-kd+y2Du=N+P?<$P77lBZWm8;a0;rueMA7H}hXbt7<hSX~l'
    '$1Z)jc(7p6iqo{?$eJrHGrLt_5w+d=9G9Wo#7$Kt^(L#Kq2QneSBC1x^+z+#VK~iwbQoyF;Z--L9`t<k)%omKLjX7p?*Ij%&Nn%d'
    '^sXFvnqN9qfD6O&Bs3|f$ONa;XM?>LM}ED~x)rL!QHx=IDVh!C!#`+hGsLX@vyz`h^}#uRh68vrDNn@l8@YztWpi~|3iD#=ZU=`+'
    'WUet^W#u3At<edNh3e7?&a1nv-JI!9G#p+ZzI+2-i)1Aq;sLb68;sDV2K%^bQV?z|geNX4ApM(s9!w`<l4;`7yAZ+#O+4-JiWV`r'
    '`sNzt)&5~v%|z;mt%O!bVX!!`l{9WEhP8fSOKZAAGQ`LuV~5v;9j8a#dW>8v*B`E(wX#s{G~$^HVCdJj*pw6EpKbMkE}q;o$BRg~'
    '2ZAmMZSqRX)orK<FlIDE7*hs;*nS!gJgoZbG79RG7{b-`L#3AEV-4sclkCwxJoDrQ55s)lb=6yiGLRKvQnzc;pt5d4auj?P=$29Y'
    '0bxO?U$3(*jB4ZoXr|>Jd;FBOVBZk39-p34V65)>5+hm?s*flcg%ovWcb5Tn)^?I8@q<!)P}DBkrI6xk{!!0$B~)!gruE91KzEg_'
    'qix2<wNahx@^<f0pvKV0h%v`Pk7MHw<oAdfc2popS*UTtwuaKO0Oi4@rEizV`k#+c(YWOrMn5{wqoL1Rxhhv*f7m~t>PTJp$X0UD'
    '%cA+~@BH<L{qreaGB01S24d`=q{8itXlNKP7FmLev}DX`{T;L8pdRScGh*ivz-YvflKJ<7Q1Q%y7D674&N{cDIFH&Ln*%JhA$kLs'
    '{{2XKQ}_vrQfZA4qZklFW^8yz3I%p<Pqn_mK`YKYX8a1IANc@a7!0LiD~bv>0vxl4E?fw}g7V>oS)Ss+u1aoLUfO`l3^26Sx77$W'
    'a>pPh0Xp(I%q!{@!wwWTA`1`b<Lh^T99=?K5xKK1W+aqzT_}dS#onMv<8w`MY!Nmg{84HStYVnQaDwNr&bT};8uwYe$!;nqbEK2}'
    '8zooMTrDPWB34Y0ddeL1w_&IwvgO;L4?8!GqR^NWcZGQoHf(>ZUrhLo2`ml|O0OcdA&_Utl*hcLi9|lTZ^wSMuz7|>b<^Kep+V}C'
    'V$l!*RqJ1{MHps#-C+#ptlGxGbl+HtKU(T!Kg1hGfzM8!B}NOaaV-Sc!q4HeOW#Bn!T0Tp;Fz05g>f2wN&fTcg?)j}ttr`xclWJ{'
    '&1Z89p89M)&eD|s>=>N+tnW^QKyURo;GUE}Ruo`a^DK)hR@ge7tCvpbf1w|bGZm_0-kp5rmH>HCHS?|8@oaRU-~KM+Ttq*Qj~3VX'
    'IPZXS2TI>BW<G!8y;p5Y(W@IrXl$r3_Bd|<xW=&x0pdM;D0sy3&Y{HQX0SLGKzH<+Tu{9K8k$B_20bmp^EeS<RzRVTHZ1aJMa!?h'
    '9rAH;Fsxxb$rm>#^12Rh&nc-{ch(z@Xi7!a^uNQtbCj=*F&2;J51a(^gFOT(^x2nU=hx)HFt;fB1Jue`JMwLKYa!*kW(ctcJ#Yrf'
    'SPcu677maFr*AxUt~}7W?%%c1Cxg0y9D=r)&IOPg`uDSNh4=*nP^axmj~>|Aga06?leUO62_L^9welplc^zW(0Vf|ULzvrquu-vZ'
    'wE4`YcHSyPye8@8LZV1}WAW|{k~atQe{>J!Z3P+Phviz1sM9(6q3WJU@o9xr0B=YVTYK~Vpl%;Hy7{U!m864?IuZz1r7r?8$Uhl%'
    ';tDQrNOSXe2Wx{A@WptFV{&Yh(gMoYf9vf!M**-+BQQQNi$<7;p^1W<8qVHdiVj+BwJBe`BbG-In19mNgXf>Llec5Nb=ZYrYuKWJ'
    'Do)YrRJ?SVKc%MxYcSh?JPafUZvbeL>=uUugPV>=mEOiwZ$o(1x0+i1bt(-$yZsAD=Le1kMxR44z+7C$Tf(7CA7E|UlKUf+xaw<L'
    '!`(n)k<-53Okn&9inP|Rz>duy>miUG!s)V2)63s(BEXrzW-os<s*ik|^FAz73UjB?0VJ>WzW|tmaw8#wu6IfjZ(ADJjCOO~uJcjX'
    '`%(jT7}0VrOH=JBNve6HIM9&o2N~uq2!w;eL<Or?4mt^zkw!tnGKEM{Se1BR(P-v{KI+@XqCXVtIG;6AjW?2`|F>;h-3YM>*Q*X9'
    'WWF;dtPC5p#zMSmZw?qPtBu0Bhb@`YfkQ9Wy{nYBm3hXyJG_RdS-=g6Pz9bxSGvi0J+?Ec^4xqf*hYR?AqyJxU#JAqD1t%fYZ3-W'
    'E`+y3PbU@(E<VK<KZsPJQxH-E80xwDbhw$%1OhTD&w3lanG{@vPlL;~iGz?s3wxE>_bWDxy+XDwv>e3fp)plM88kWI_R8X>jNI`K'
    'KTsDrq~;&HO-rQKZm<`YMEcNkfQWu(bISDO(or*xY=e>ut=dH1AFkBx^<$ArG2Up_fp?rG48cgi5XlpMZYcTZps!_4E1&RhT)q6O'
    'rn<NrL&aGfR03V!g<2gWO)VcBK!}mIx$PWmO-G`-f8EJ>Z-rb%<-JeC9Ib-sXNKHamT+JdW3DLc0IVpwxbw|QjklzOjNXAsz73T<'
    'plf4Eqy~gOw?27}0vpgMY`FauFv1jkg-|&u@g3oTfEV*^z$CXw)!b&%eR=tNLTo@2499uEe)~G;h1#}^%n%EhGbm;qqPK_OT){#6'
    'bF1T;bE93RgvME%Bg+;LhXl)9!KX&hW4twUsX6oOdr{>$9g8O!)Cr*P(IW|!`965!p{UfFJQOnoXeP_8?}x%cBU5qpn4fz(ob-#;'
    'Al_?GDFa;~1C_-Izj+qM#ojQ<%%htON3@@C$}_EsU&G1)BBzNi`{v?kjdmLlMn+@7u8YZeQ4!>SRQZq8PQ6A^e68=8Dw-BJZ64+t'
    't`1{trO{yLmkejUZyveub?Y0yZB)5bm2(Ie#XAF0y^^Q==2hctG3c;y&5h_5%C2b2VyX1jg<>}Q;2aw^#43`!27O(5Z3cE7go6%z'
    'Rg#OhgAp%76G3Mv!=mp4mqZyAX813M`Fo3!8gbybqedL#7YSR#7R0uB7!kx~)sLQH%U;HDDs{+;oMOvA6h5p-oY&ji0-)pP;GhMb'
    'uM*~HFeyT(42eT()?xV?)!&W^-s)ybjisZ4B=g|ga)AMD$@c`C_J7rO7Nf4SK2|t)@x0e&iZ4ZnRkSDnu}jLa*dWFud7*Qv!nN*v'
    'HPA?cNE3F<@utkt!Cdnf4qz@8J@pG4XMO>lCYZhedy91C-&FLQ94DJtM;u7HaBQ^qInLxS6a8&m!I9QUil8%+<;y;sU?swj_Bxs{'
    'hWwA;xE><$`u0`!wa@MYKr4Fff^r35W5gR^U-o=dBrKzjGmK2QDTF($AKxeHdbPJcX5I!8p7*%x$cQR+8-;A=2EcWqbJ5?~>nt5('
    'u)VPM?)$x%6L&olgPlTK#$&s>L2ro#lwQdap905Q*|M$1ghKw|_p2TZBAZa$Xnc@(VJi`$#x+Y^+IUO#_tw|gIdfELEJ3tsa*=Tr'
    'X&&lu5L@#W|E^$JlqD=V+viPpKu}a%)K%F8EcETaooaJzTg;*4sBqN0iugbU#mQVW-~8fA`K{KhFKwjuO}sX=pH#B4a<<KZ%^E-6'
    '3KxK2v>Rz15^N$Yp3MK?ZtDg-j`=n@Lb#(wRYOycG?`7kR&*;}oz=A|6K@_U{*YW;woiAkvNWYP`|c{~&sPKHZ`j4T^?@6W%Z{we'
    '6m&6Qwj-K%b{)Ee?<@(b*t%2Zg?-$NXcO%beG+B@QA6vS)k(<5jg;Z?`gZzv-ohqyHr?lj1M00`LMwLs0+Qq@qbiWIGJr+n4$ofc'
    'BjrC9Z1FqiZq!7eG>TWf58PL5Pa6D#zkMfv^50DDO`6e8eI<Dy++E~Ogq1d{_Ze-IhIuEexa}OiE<_~XjA{WH(%a`MexZJR?Y3xC'
    '6(MLhVk#`KAw&l%XYVb*PT~;z4+9-K236YgwON)8okvB0IodP(MqF9xTG8b8ngX}dwi_JJMRT6^4Fci+^k>Ga4{zwDZTG$wIX8-p'
    '$Xmmt-{<bT|5n?B?HCH3fszO?qkt>r?9<RzlDPC#SC9ey>64T^<_=|h<mIT?Ss=lrjDdegN%_{!G(ujQr~_qks^~w0G?AJMS+M9K'
    'U|VdC_NKSniSAiGqTZGd2d`Ky6t~Pqz`;Ua8s&p50;Bw09@2JD5i0_OM|(*off*iHFJsww*VrryE3{vz*7FOsA9;8uC>#NIWhv`F'
    '95cr|GIh||qIZFFbYd2ykE#p-_%*roFO@gTK1iB2bT7eDoJTdldHMThWsP8o{vc0D{^AS9@w=DPH5<s9NE{i(4(Leb0tpn*=8*HU'
    'IvsD{2c$xj2CT4RFC)pPAI94jbb<z);SY8=@=GAh3s(yJ9=kY?x>FtCk*u7<C%ad9E2@1^6l59Z2yC=$fZssX&#*G2t{9xE#&4*c'
    'VD5}AxRJ&~7)kDxu2Jq(1Yy_d_k$V}CZKfj6s`fi$n(M$g(fj`zMp`yy?Fmy3K)1+*>fs_8S$&;Tzj4!Wg}<%c&~%dbDJ=e@aX7>'
    ';<*H%MBm(Ar1G&oxPLBf;Jgj2bu=Yr6Q{c6*ORxIhNusXL*Bj&l*Z76964anAVKp7nuU3PgE%x$TfT=7li$D)p&*~+;4#%OXnTmu'
    '!A6ks_~4g;fx{5!p~bFL4)Wm*4si$)hr)&Ef4-5$@*Bzrn*$oMQk~s~Iuvyy_m=K=|J~2PgsIV4;s89*LIN#cj4jy8vs8}V6`JjH'
    '3Q@QoQ?JD)47xm+l@`_pCV*JcZ~GSds(zs(W0w<-o(BgnzEuVTrBNv_z=m(<EhYwQE7+;D^BwD&t%IZTmx7~xvQi^(sxc@hq+oL3'
    '%uOX<9FFWmGeVb!2g=Xe(Ni%qT}OAB+LX|uyFO`|ht3TC-Xrc!MB303GVpaXJQ+H9;q+U+=LbI42b;vq_2+iZi=HdYpVUi;OG?#3'
    'SNeOyT=2qIQe<9<Yc5hk$52%Ch%3Mu@!dYMuLBI912)C;1WryZ;8JL@9HKd4j5yvVrySD_fP9xgi-RC&3EM~Sy6+|N<<B$;bb8Ra'
    '(m9PcoTv;&Ce9|;aEpCj)0?-U*@Q3GSt^K;!2%~)F7L?_qh5mrd$*<!%_@Cwsl5vK$H9%$2m@)*6hijC$yj;f4Ch<lxPO*d#*gbo'
    'tb_7MD=5#i|Lu0-^tQ!k%+)$Lf#J%ZkP|`4)DGu~8g#xL5wd=u@dB=kToa(=w9%y%Y_N<mwW9zYZx%pNtva47+>-UxV9$kNn`Q+V'
    'A_uLk_kye|YCPmgTPVRX59V6F8R0?c0XF!zSOfy`M0MOtIO`WqJ==9l#m2eAAO8P8x1bvv2Q|opi*y<V-nh5@8@KItj1l(Ro9IP8'
    'L+2Q~FRm}~ujl`=kxP3!08*Fa6^{*7?qjpEM~v;iQx(g(CG2f69+<eT?Y8Vt6LCtyH8l4bh2cNi*7%Rn>42>Ts-$gdeg2xM%zZy4'
    'Ze%PMZ`A_-O%Y_vB*Y8-H?@jTWLs#-e>2}L^Fxg~PCO<~Y10)qY+hWkGG0&zwDm#U0(V)WcFx|7MjIZuDegG~N6cq9qvw0%Bih?s'
    'i+MCKhs=!uUuc7<Ovh|{=@_&_o=JuTTm6zLWr@B0qNbe71wmu(6hn(|$cU_k$PWZ`w%A`8(V3tCtfa;|xs5`Mkjh)>h?HTl$b&4K'
    'VuZ>DxkLvR9i7dqJhQ0C-<Oy!xQIT5z2IWef)i+Da?6%USC1O{o%C$NDe!T?2djuHIZ(NTj5P>&WT+{|uJKmWqxB*vg{UX0E|*Fl'
    '2mzFJG{f4|t*;OEHxczAGBIB93FgBHQfh3cMznY!(lTS><!{4V!yTq>loj;1KoaQhkr)}cjEFv2gd?msvE~HnN5-J~stmCmMmiQD'
    '_zkPB9UIb`eeg79?8<A?1?0KLIxGbva3@z&Zm7FIzhOt4(LDv7DZZ9B+GGtctO*PrT!uMg&eeYlO<4Ji=Y8Uogp(pbAYrS)t;MKm'
    'bl@8Ep(}u-S2N#3Eo~LOhQf|I!EB26alPhv`mDbR?ek#yb0`BO-;AltEyP8mRhd-l102V!PrYvX$5f^!G2N8K6D6(>+Y4Any>BL}'
    '&wCwAPnNkOprD3*Pri{J+h`*zAWP`9hjQ!4VM3d$u}#OMz%3`8f)SM_71X^7x@!14|99mR-NV#TV2-(u0y>j|_ojs=r0l0TF!vWO'
    '3`Espu5LV+i|MlC2E{?!Sz)b!ff?)lz~nR>wYOKAiXcXU5I=42>asJE><^875M8t2Xv%k)(1I+>qf1-6a$jOJRluIn277<SnYA~8'
    '@EZ&K(wTC(CQ`o+LG;cZ%045WfLn?Ko5T3F>%wZGd!-$^5Xlup`@UydBfRTa^(o?v#@}d6oO#qymCYSrR_ilajmCg^K=VQ+zeS*V'
    'wLoBU!3)OHE7$pjK<Yi$OaM+<2NyUjJ?Gqb+Kn*o@IET!v-v}xbNC6xrrzteTN{?w$`gv^97jAp?4f{%-?G`V=G%<0u6Y93VZGJN'
    'uo+NmaVY(v_nRCM!3=Gml3y8Sj?_8tJnMEdjo(y~{;w;M=QYyS=}P3Lwxe6w5L5*LzyiVAAN!osL)x%*m`~y+7l;k`6|E4eh8#0?'
    'VC-9)A-|<O?eR8qly6%b=SZ&9r05*RHDzoMEnm>s&ZVJy6A}3}YhHk7Lv9H|8Ob{Ew`&+QsANzdgj~_Pf<9X{M<m~&Z(nTd<)KS@'
    '>6}UPNew`{g!PctnTJgvrG)_~@PWCrnSN_4Z3i??5ZG9-KK*4Xs?5`r?;h12df6mUb7qXJsyi8d^tWZBlFzehfNpOiz6q0n6<aBG'
    'AZV3n&oe_xPabONQE(c6Q&$XnokI^r!G(-EQvEo(R$alujX`Al+=37cz8hT6&Q%!bPRDheUja>Arh<eU*?6njV{QvS=V~3|$$^^S'
    'TsXI=9}Q!>EbqTbD>QK>290wO(u(2gjhBs!L0U0C*@{%a74whDi6if%3tLGDg(M?Y#Lo6M?V$HRVAHV2R}Bz*eRY}9>ywQ|Yr5N8'
    '6+aZT4bC`-5>R51TzpGIM$^!QDZ}{SED4lr#qc~yt<0w!moX<>PHe*kLPK8h{$=d&W!K}7`(IT=LZv&qQw>o86O{FiWMVorr$UMh'
    'c5D^;G!*m0oVDy+uM+;j=b~s35mM2}Ipp{k3N8Ggl7SwF)ULhFr<f$So)0=TeZ`7QP+8q_aA`}nddB*E*EfR@A_uR+ZehmA%V$EO'
    '7RIw^S;^7-pu5)*LNKvn7^wS=xG2cf!X?wj-u`UN`5R*}Yk-|pTm=W^JyFnmJ}gM=x!-HfcgtSeaj{(qh%YGcS}3?tEMS#jj4l@8'
    '$0o_V%<>xt6^^joM%BCr$R_DKT*da*wcdp=BO6|P&dZ`+6XPj;#Wrc8O?b=M=+xDsGqOnZt$`O*0-x)Us##XR4;t55(Gacb-sKA4'
    'G>;5fW90wB#sJ&*JsygPjkTn(K}|&6j1hp+nt-!`G#=eVyVCz3qb6XyraMB0p{|;kM`BYX8c-e}>bm!d9^=@KkdP@?C=-kmBzS+m'
    'p*CLR)Tj13Ct_0XOC)gfy4&m#N-*XqOh0GIMf^6fAftEW>q-p~V@jjPZDruDs)PT;-gyl}<m|fJVnH{I5oqozLyAS6q9y))d!jgq'
    '_@K~2zr?o>TFS@XWo)Z>cva%RT}7vcl5(TsOY3Ot3WJ)Z=yv4X92&F4@V6%TKOg0*iQPY_@OF{wBEu9*J^FhY^7>Q%_fchtF?eTv'
    'Qb2n6&v!8W|9*jAf9UIjV2(|?jrsH4*Zj|K`p;kg>yP~PNl}j0UjXu99uha(Xa}OWAu=umC+(L4(TRU%$OE5g2r|}2^B-D(=5uMD'
    'y!8!pD6PK`mDEtwqRyz_G#HQ;r;fi2aE$ejUZ*YIUM%8vS->k*0b>quEfHC9=kbU}Q++{5`*U-bnnpD6peL>)u;obiJKM1MI~Up%'
    'e({PIqwYj06S!xf0b<lEcE|72<c8uh@(1!snXRDi)od77n0<H_R5=?F08wQSfC$BLaA4u3!T<Yj#|_R3C_inIIdG>335kR%(k!}o'
    'c1>Zfx>o$?KbQ?0ZgkW8exdZj6|J1LjMy$59U8E)M!~0?S9v8!Q6UpFU`F*P4ySYqU@T+&of66<@+4J{U{fa0pm|6|WGbF(+vOfI'
    'G%5HzFbfx)F<41CN#)rItMR(VuIzrgDJ#D0>)DY)Re8dfN2-#lAnGbF7hOFhV0aioie9~ps0|9;Zf=m%*+=MW6O^czzciMD1v)92'
    '1b}UtnNj%eDd}cXT-+de4Lml6pA@KUS-_6x3nChqM<XAzDF+pRl&M~B-Nsq~qH)oZgg_Pnu~QMkdFaaL6Lu(#-!GsoQY3c%+9p=%'
    '+TBRzBiADzvdS^Yp4paSVOb_fn5?YoC9F|^(E4LS(8UeF5IxTpxYXPFaL@u48<Y@$5D8eM(LnMbCd*4=NS+3YRN5f<Ie5}7;qc8F'
    '`ME*~jBdo@h)z(jQ^4t_T})%+dq##%oi!m3(J^LRjn~95Wx0~Y*Y7W4W7LPi*c3;%@feE%68_U#9vpm6Y^KxM-R(b$WptWMx9M<6'
    'i1>SUMBzNrc45p6n2yh$eBma!yqvA_Sfg<^(MGl@xhUZp4%Vk7Ko|r%Tsic|%Vh|of6YXf^e*pzMu99>v<K}PdLxuDQyNdIz)Qay'
    'AH`6w3#8(vWKtr<wtB}!1|R|>Un=HU0|izQ&2hTR!v}}DFHmphWw-w7Zh%=pOFgz=ool#w;k#0>Z1=51<)o*H?NyhWo7d%u(4b)c'
    '^8fd(1#NIvn+7%4Z_qIp*-J9I)(#y}ZUwTdwJuv1Jjf_GzjLicY7T#9VDC|tTK>}ULm7A^zu}=Ds*Hch_;}Rwd+QfpJ3=2oB30kl'
    '*k*O~-2g;>qeuKA?Nmxo@C_9bG123bhN?gqekm%&Et;DjgXLdld0diEXsmsK(1yRLlEt~P92^E3yEZsvwGZR+apFXXs+ce;$*SOi'
    'Xd-Pfk-ywmjuxVL^0653Fdf7$m=(ww{lm)HO2Mk<O>Vft4@?&z_kcgF<$69u{l;Y@jMFX4%5~9!m!|BgbJcx<iEYT1SskE}q7)E#'
    'LF}BvzvZ@Xpe7u6iq#>r0>=q}sfC(6SDR=THlSwz1z7-uC^dS~ee<xxW+~Ui-iHMsB39)HhZMqaS&L3-JcQ%}mbk!m)=&_9*5|nP'
    'vou)P%vrl9@A$gLzzxOGEWsK~RRdW-NNng)t`9rt)>7rt^0K5y-y`zkg98Vex9jELQIg$=HJ=V7V~T?JNoK+8L0_g7v9j4pO*~R9'
    '^+UU+xHuoZepno9yp#c)p%w%vB1@*Tja<%~Tt*l8h~Czs<TDVXkRoGlu=n-*GWIaOubN+&zOTAXaLl$oTo+FghAIB$4ka8a^9L{Q'
    'WRE5E>G#$xjES|WU<DL2OtgyJ8$)nim&i~#W6R6qll!>sk3k^9Q;tOZv$em`NJrztupSvJN`_!!jXuBqaX*b#LErIr)u`~O`Kqp;'
    'iExQOs@aL$nQFG;kI`|+1tDiH);|0EXIV)rVrce_l3Th)-fMKrS-M{;QS5iBJ2C7a31NoSInd*YH@|dr#Z2$ud&aMR_|X4S1?nDN'
    'K7976Vv}uG<^X!i>^3Vfmzrn!CF@RXdAcvypBI{U3h>4Z3Vu|H%i}DUkK&pJjdChO`N3ZSPBA{5r6+_eUKIRllIK@4X8I;&^s7W0'
    'zE%^$pBWJ)0orgGXerI(n#ayTgrbk4LXb<Kdcw9dIrMUUSQdk)jhQ8zQAI7&1$xloUw{-ClX)(t1gFv&ThzP2PRqbsFQ6PE29|Js'
    'mBezLd`N09FaG0de^&fD(qhyEu2|JM*0cb$<RwM4iV!~b7Z+^GReBnG0l@nBio-B%VrHw*XPESx!q7s<zjb|sIkc(iMYuz2>Ie}^'
    'z)DF1=zl$v{r{Jh2yjug8V4CfD_P_@H$ZBoi{7!x$qXouVHP>pO21j_#SXOb;Ev_s1m5&i*DghCzH0-=Fu?=sk_n0-SB@_^-K$WL'
    ';02I^(z;lyUV1(}1$85rY5xn(ScETJ5I8RD1aA)_fF@M4egMK6uZO0M|9+d;UVAE4O=)cNUyF>&P2Gk+UJF4XkoI&cXjO#lY@B*p'
    'x!iq~y}j--dAdFZO$O-~SnBJnNI(P*Aux(_N!rlrrnXrR^1s?)C8DPUI3INeDC5}$3+1T#mn@?E)<Hc(SC7++${u^6L4=+E{Xt+s'
    'RYz>+>^5zy;+~?e_CM>3N&kc55O^V1U+8p*`*|H0(?1)q@0+4(;jRpdRwQH)UB4iyR3J%E8BMKUc7Xm1!wDWHE-q9-WNhuvcc;50'
    'N-590#+;Wn6<D3A5E}%PVVhMQqL`~vAD^wD==aw-cRW*T0v`(+)py3sxY$70A3G{w5w_tPP$HM*n^;+UJa^Ts%8w*eb%Nhjof7MX'
    's`S3%d$xR})#L<gijbfhF+G$;?oLXzH81-?ga6K`j;u$v4T-gQ>KTeGv-aQ?7j_1D(-n0$l}?H*U}`AM;(A$U^27&Q81E;D#v!LV'
    't=%qIA!X2|WiP!jVn33fbKUEr(Yxp=J8SE+UyzY4+BmLQSwOI5V(lW{$qm$ZUHkh;5S?yT1_<e$BRpPwuk?!Qp$K^Sg-jA@qcgB='
    'Gl*yx*@CNoxcP?y0;>}-JCk5oe4hB6gJXNWEUg$?r_M$b6EtphS7>6@DPgwD1b^tk08oe?gVI7l1WlHK7#~iH46VOt5CmdvKbBM4'
    '<?)NorlN&^N2L$7JrR24JJ4kd8?pfg^sviUW#Um*-SHvEkcKf_yr2M|ZS0e#Q-C}<c6LYl?@qlblSA<yms2w3DT7fCgUz>Hf5JV$'
    'yuL196+_@LUqOMLW2>5ftMATu`21|o2k*y3=tIML01<&6*6GuOk*t2v<Yc{1BP=#)5#LANORT|~lh3#?ctnDqDiGV|vX9&l<{ta@'
    'Pr9M@a||1uS2jyc*VvmA3A0_T+3$SuEz2j{LJ6<9AY_8e?XpP)&Xa#>U^lf7`vw3iYNr1bI}Y*eD4)G}AB?El6o4l{FDW%;ys#wg'
    'x+_r`I9OQ(@6vvES)|K0J>=>BQry8^A?rdML=FcGDyhb)di#gb<6=;7xa9eTu{G3%A<cKtrM(y{bcGe&Gb?2oJN{z~;%<Si#id!J'
    ';wd<@(CxzhtgxLObI@A<Y;{Q0F_}b*z0h7sFCFZ(NDtHILDRr7QV9gFV`{_OrL$T0{^0Hg?y6VKg-?LyMkix%gZeIwM5VfH-e+#t'
    '^R-}+@H6%9K390=1orY0I$Tqrf510PI~%Dj^YXiI-8Kfd3JcgIeX?QOS<w2$kVsQ}GTH&4&H+KNK}F!Ql=sVOCLAhQA4Lu_8ZvaQ'
    'b7wTR@dYKXqtjWUu8^IMgo$<*H5m2=yKGeya&npguHwa3`m;O`4JTmzqgh^wS%oiL@V4@?N@xYBKcD^2rdh9R+f;(?Iu=3lOgq)a'
    '^@SVqYKKOKYig|}aVd7y@V8RexWeJO2s+&03;>00G9}RG%S3{YbrdF04=O3(RP!BlF4w28;rF6n*)Aa)&!pGS!!vfI4!Xh1fDkpA'
    ';%@{BoT8AAZTAr^(Z3+xKnAav`-&in^PK;G+`U_t9J!9PJCz(kE&{w_{u4X+jG<Ci7Cri3-BO0fdD)7s3Q6hibP_PwrK?_pn6tjo'
    'P+-Nywp7P6oX{j;rnVJg0=@r3m^0swms2|W#p>L=!O>o<1`*yV=TIS72(C~Ug@)F-fBHB75*BZ6231b!x;aoA#+Er!P}UnG{Ge5Y'
    'euB|v)9T8<X=E%ZZ#U|DLO5_0hzqhQtQXP_1_l#@RAZL4<~LUAw!kGZKxPR{x}eSBW!@>?%8hS@(UCL>DWq<bl7g%p#Qg?rZ?%7y'
    'kl+D=!oo-cH#A5_Ja+Fwf)ISyZ?pRWo7M5F!`6kNoBRMEJ>aRV{KMA*cM=nWN`afpSk?fMJHGAYAboy=;e_jShLQmcPq`n^lEzrV'
    '0~DY2Z}lA~9Ss9v^&~grnV?>fmNs7OhCT3*L+hb_hB-bf1NSBcVOa4ueV?(rbqJ)!0A<Nivq@0Zw&UQkZ@OWYW~nOt2udpZ4m6=o'
    '@UHKwGu|)flFzek8_D6y1&_*EjIW|RMZOj7!!SUNUFqdU7Gm$mAm#v=-g?~ldQlJ}Q#NMEW!c_~f`f>3b9@E9m0`5!s?)QCKtqNV'
    'Wt9mBYMHpPyuzS{)Txz=fZo}gs_h}O8F-0(Yo+lXsrY12y*FaJA!aO`2ULVtyD_EI^}1maitp6wa{Ddp&tUA<>rP?+H?pjXa@)`c'
    '1b>-Cq9kE{FiOAi%Uc-yWtS7UpfNX(+?Csq$_||S+aApOYD0`Wx^zj@gY23qIK6*Y_}$DJlAz^*M`BhCg$NicaPS*BUJTslHzOMy'
    'UY)mDR4dh4RM&K(G+zJFZrtn9!flkeGOuwvVnXjJ`5yAL=XZG}PqN9^o1|qnC5-KrK*>7x5jQ6EqspC3vdxqdG&CF2FjIDj&LZvq'
    'X^wq~92Q8Sw&#DqiJ~`be&b=bnY>Hn(=5G^@pwV2$@|LRLPPo;FTn|Rv`3gsQ41`!t){-d{LYWpe7pI{<Fe8Tlm>@R(qiIamVJjZ'
    '2$Ha!DAAU@%#^b}6hl$!T6cbRF=?@f!hsYFY&4`x2@Ejj=>Hj5J1K(l@*+t_Mj@^l-nw`F7>=zPGkHkXq;M|=Qy5cX(`(;J*$gF%'
    '_l<?)aD_{m;$pFLEbYe6l|aO0`J*phIFzZ2z-~sZODWOc)GEXQ<)h2C0R6SwZf0kfvEjtO^*d%M?c84KL*UMKf7vdOtG(FOmD%)y'
    'nYpMsh|ipe=T~mlvr|ca<7g6k*O+!_&;@ycwn)*{3;?_RXJQ{!g9DH~Wk@ped@Pj`?X~;)1grwH9q{A~I~>x(*@)$|<nbNfhJ}KC'
    '9e{HY!0n%1E=v`@&NrkIVnXhMAq+O(gFW>u6-<AW*SoZ1+z8bGm(Iy4JLY67o02cYcd-gdB$KpcHA`)WVj4voxmNLAs?@M@1<gfM'
    'i(5uhmjZc#p1Qo4&x<c#W!qNF1^Ql~ySj6`h+J9Ur#J9eA7}KKb;HzKxK4ga`?giUUmbeXN{VCLQ)D+~VsR=q@wJnC1$1{+nKIZU'
    '`=l|O$BH_>$CNT`(FBu?tMu5u)Oq=Pa{oOMOZ0Z{4<|bUSerB<JpSE|zINy1;fZp>b=9^W;YczXMU_&&^_0A@S-%f?qQ$nOBxFM@'
    'l-OYm-)H@e4NlobatO=4Q&z6XIqk)y6^0E>Q?aH}yGmM!9XEkzO7Q%yI}Dr)7-5(akDUVk5kA+;y77==DkjOaUe{DeGp{lQDE(*q'
    'PEnyM2Z}|RAsU;e8&hVCB+2=GyAU#AIut1)j)E;#92FzIGq<^B^Sd1!{>&KcG0$a^Gqi50a7g3(?rlI>J6z`-!aZF2(W`3iiNLLW'
    '7c;N*u)0RUk4%q}P}`2)vY%k>?X)KlT-Lm7)|l0<sUpT8Fa3LxLoZeyfv<JoTASN@6|#c7>su4b_Bk7|6ho%>ZCfw0;J8V3emfWW'
    'TE0L_F%|*%0fI3!FGV!_AGRE}4FjM9h}P5#&&N9$a!>v<=Limm*>?}Q^>$s5XgQi-Ax~@Hfv11I)^I$Z`%WNAB$TF?Z)Axtf7?G_'
    'Ls~&=lzUMB&KLUUAN}WV`{!%$1lA8Ioi*}fr-!mFU&XBd`(bPX0>`VC?~;<VKksa%_^?xSlkAPD)6}oyJ^A*(EQhKouN|U^IB9cN'
    '-2b_9#+6(vbAicvck$TTy+>Q<RvKPp=z<3(C(2Lq`QL|zHwqRZv@(Dzc%MY=b{KQRp(LSZVmAnIL#|e@uRq@$YizO(`QRO$cjL@H'
    'xHvLvHmt<;0VNpI0UtGv27|-u*iM6mVLpb~e%L`twg|Xxat&lKrmTx~5T#%q=n6Z*6>&-A(YA`?&rRs5Nb^1A1FeK9{qyup6(5y*'
    '79WNI%VGO=&_!c@XvFKt1(6MF=La6#!DTksarICJuI5aOu<QTcw@`qWe}v+Vy8i#NKX@l#hG~+U9~KHY{Z2jo)Hod)wv~V|zJxy$'
    'S1}J?#SuwE2j5Hx{l>v*1&2fWVk`&qmceo<<!9cPr6P3jYlMCgcKlzbxjeM8-NK2&^~<sY0!Jn=r3ns6)Wk^t!e4)CA%lQg%EW`r'
    'V$yOV{6M^W;HxkvyMFN-b7*%ZhufI^4C$}<#p`_ALwBlZ$X#JV5`oh=p@YT27!1|%s%6WMWweS_U(|b798BGoa^`T?T#6q@c~s&+'
    '`qRv{s=aa5j6)U>^KdT$k?qunO1zb<4a)2{SfC6Q)zVde2(6aSc4(=InjV|bi5b7ZY(LNlD*u7Xj1PV2gCXVn@5Y6{TsBvs4*?g$'
    'p_k7eA}7%8kA)QtjSw9(?8f@tuj-dBqoIbDKm(w#3jDT1*(N^<-ybU0gkYwp;smQgaWnJ*SXLan(^s*t7!2C@ut>e|%?18*h+>7s'
    'OsT{N@@3G$k2na;(8U`=TRt>Eel*E$hdmqXib<u!haKWTbkI5sL3s4ExcoXXs>gG24)Y&sk8oUzZ8(@uI*Vvgiw}4`+T}9mHW5@{'
    'B_1Th4)Z<q%>#iGrFkLcgWvITgP2-(JrEfT<}+;jH_D+Cz*2B@nGC{n*8lJ0=2~re35wa7?`ef&&xg|H`?kE;qG$b1+hI%2f{Kv^'
    '{yXegh_BXAQWmZ=58D};Ti^L2i3@cm3(VLZ2h5nbq8&I*Gi0S@a$43Bo;>R~^d125awS52x3IBiGS$ZoNPi1yGd@^LU|&}G7`L3{'
    '|4Pg*V*~NQ@J-Ly4|-5c)@pquAGt!8jO6Sau>K~Kwmdu~hDRf&@pC_ze1pLc#X}YFQw`HJi9X$!9+Fg6<ha{gu&;W9f|(4zJj<w@'
    '6=i|WKm1_Cg^Z$ehVVfi+I0(U4<!Z^Rt$6a@Q>cEz5dzC{EMjx<Kn?|UF5xbaGpov*{5s`S(p@5p{TFJAxb{rjBR5*M(ZjUd%r>U'
    '$xuAzW`YDDq;wUhLzGauLMYthgJjKh9bQOWW0z_Mx<|*mtSdh2_0JayM<-@Mk&Q#FHQX-`gI$d73}F@@9H~3Hig`d{@bgo~A0Kq@'
    '9RHs?*qqT8HCUT4xjM$Q^hZ_?EIr)&i2~u*;q76$ZKvhDD4P5J6}$#D!YR;5)`zTi42<+ED!3nt@`IS!!5<-dFfWzinIF88svHkw'
    'U??%TZzsc3jqCEFl9_kVUJp4I+Ns~f=P!@*x<3G0VWZRHhnQtBIPzc$Q5d9>z;(=;ovt5JAXx+k{iFBo#lG$h^AM_+Z}8CYKL!;M'
    'jX>?^{dW#r;Bj4{08!sw>%A*U`6&%i702++C+_V4pCZMspvF+O<P;{|?8z4lZ;B?X-uzHOyQ_@nPa8fF6&N<~!O_)eS^R_t=Pxc+'
    'S>j^h8w+QJ0OP|#XO>-lF|>TQ3DXa0vWj0M=13o4<Wr&8Iv5aF#R<yOHg@1C)2uZQdpgj;!ln>VAs+sp<%GffuYnC3zol|1k6d2N'
    'N`zoJbV3+2j5W7=Uf|%v^KE0C4)n=;Gv&wVK8H^Bbpi~S+l|O;$HJswYteB+{UOydvbTJmLpFJ4EZ)|(|J~`UOIE}RXe*}QDeE9Q'
    'z(dc5{#I&kOf4E>{r39)8pJtY^tFBTm|gKZUJnJ7QZNekDp=^rX9XSWT=(AE0Fh>p5yw!l34!8w${T|lcrjMI8ccCN45*B)QxSjv'
    '-P<vZLLDQ2WEpGZj&t|S2;nwsAVwv&uWT%b(1Sx=97CxPJwM<fP)JjN1aly5;2~I*YKliHo}ySx)H{fUtYrPTs2=iDKG!iG5~y`}'
    'tP`()Y_4Ue#bH1|thWQzkO!}nb#Zf@0+qh`^)EokgEWt?Jan;^iH}g&?xDsdH>wZ#PDBTE^X-`54rea}up02ti34bI+UR&Hy}n7x'
    'bw`W8l<VQ-POYto?}zbP#~W_*j-jGzx;78P0Qux76P_VJC*vH=O}l}o0;UFa<n}ZmLpP-T(E7%qaz>vh{&U}ii>^j;&O=wYbFd*v'
    '=7+X<w}f`h*S8UK(}ZV!z)3eB!Nk1Y<FFAerB)#gS-FQRa9IuuhoVRV3ESXR13RHS^FChxgjuE)mm!vEhXKwy^?Jf&@$RNr<wK)O'
    'CrP`_L-z;Z<0dP&_9!xQCG~ZKO&TKTP|7&sOJi-Du;#;yiF6{Lu0F^v1lqsZsWHam7r_f38rm6oQO_GJY4FKj;=yyp83CZO<F|gY'
    '>z4N%9~wcqv_JD?U832rVTH4B@xj2F)nDcfkUIkN@M>g^pZk@o*<HP#u0lL-+aJ&|>T-PDyxT_<zsAF+PkzcwN!mE}gTgbk1sN9m'
    'V0Yu*H9z(w69WW^mq`8)=zK`O*~TsK|1q6|WrFv`>W2{;hw6fKDQB_Tx35<9^F#^luO>@tm<CF#{7}J-X@~kWJ_t-fduZRS(|+j{'
    'LcR}8W&&N>rgGvaAyzb;B=}I^p}@Q!#ZL4Wi0EUh8m7iHbM92v)2y*8@dEYFKGL&dn168a3N9@_Dn7*WokB4WtVBHI1_tYVSfK1!'
    '5{`M8$jFP0(_J2JBvesRmvtCGA>O7+<yZ9nsDI<91{91O820jOFS5dBOB;S(-7=M1vAsa}YAf?}MN}iPz4MwBZc9$?uRp7BjwKwU'
    'H5j>F(JIFu45V+MB~egbNUw&<)9$@<bDS;^R2j+P=E3hY?{5rd3`K~TQtStYpFw0jesPDl39(VHh+Kj*3|rkGvgPK25(8P6hS(Y|'
    'Cos_25d5&w8TnHSc9$u&48C=~WUv_K;iM~YrJAOzxqS1;gNf#)-2kP6Bv+*kfC9@@S}0Yl!{!_9kt!d!oM2^=9sK7hatiTv!P{CN'
    '2%N{5t(+trKR@E;$+fjVa2gDi0zTgvYwY+-^X#f-GSYfX27QNT1H=doQl46US5sr%n|N-%>Mr=rDg9Y=hD>P2p=6_NU&K~)jq}mN'
    'W4!I=H^0b^y#TICPzD-oF-4N(Q##Z)3Tgy7<3kYDhcNZK=>L8f2_<_@4E@Bibw0_Q`BeR8V>{c!utD@-uGVI~J21XY8pk6@hO#|N'
    'P`tSh4IB>4p{5=8N*Gd#(D1nTmJ~{|o%>i=S!+gB_W0o;N~_uT2Q_f0^k%zvA!peQKJu;$u?HkN%k?Wu(F0V-U0`6t2Z{BeoR}56'
    '`9(sfm602~7smy+U_m)fTMoo6NMBtWRr9T;k2!CCjY#)Zv=&?=<tAE)I%$X2iO$j{PCp7V<B5`2*?PPGYE^o0eu9aLun<49e3GYe'
    'Q^^I5bxJ`YKCq+*;kKKz;F-DADXTW{Bgg2uI2`E1Bn5YMWq9$SGGA`FEqjZ*KS`%(uIu5sN5`LUr|mf6|6(Uj-M*tOShTToc<+3L'
    'Yj8ReFjco@G&8quc7i+#s9fknb{n=+nAgp1*%!nZxy|LPSaBX+kU<Y~vH>@A%5tG^vxu=};wMapnvj?wG0cT-whe;4e58h59<r#p'
    'lpanZ+&D`YMzKwL=7UcT6Gg+GXT+V!6CY-^9VwW**6XjAQE3H{9+DDujZBvx9$H5VC#G#@oKHYl!wK~uHzNB1p*c!K!?R~Hg6pq8'
    'TgMK%lI6Qzy`i+rgV+S?=_$u;shMlv_HtfM5*A_>)}FV8@n$0~Xg4ru$2Mn96x>S08?KzQI!_p-K-;w;*$G2XX_x&3xwCi`9$Z95'
    'oBExyZgAw{PeZ{90RSHg5jo;8^JOYrpvf~P#&m+|@F<5uo`|d|Og)Py>uVE^#ejY}01d$r<6b`0CP$}%{n-hEhPJWvt4uSo+p2S%'
    'mNUF+o<F>=xsy_&++Y|^2vhiEezWil^Bq1o5B!d%gm$(dIsf%7wkvUS`XKopi>Wlmb_E^>mT*2!Je(qy>yQSkv0?K&MvDEdhoyOi'
    'ldN#OS_NoPH<Wo0k13j@E29LfkX=~V$)}SUTP?VE`LDSuYLI@grD}LsAV8BHuIM4WG3#gWG_pJ#;m{!Wp_x2f`-z5tkyjfsPlA2f'
    '6amruscg=|tYAOG2W(*aM)c2(ePHxJ)~CCgUY1#nBSn8<LC5r7G)`s;^-qGU5RqhxqYoiyb9`Dkjf3ahuLU0h$P7JOuZQOgG02x3'
    ';2!o#Itxw&T*{Zwjgvd7a1OSUA2Pz_=Nh`jgl?9`k6d=j;_0i8Z{&k(ql)%aPLqv){*HgXfiY~%8vw>^+MJ^m&8XXjc^DaEu$)-a'
    '_a9_1?q_O^OShTb{^xzVAZwO&5IWfv=DH|fjklHHL&LLnmHR&<%>&VOVyV@}g~u7xo!pEZhu8~P#mMj=Q~-A1U1VgKL)2gi$|?S0'
    'W@0j`3i|-WuU9`iF%cb-x~{Zd_#v4*bIZ0?%l=?pnu8UHK}M#c|6y??Q8NmY!BK6Ei{smfBO#%~*x1dke$p&2x@I>7WL{h}BE!Je'
    'A-lkO;Z8tGQyx9Y=gS{<aKtyQLKrc_`O>fOw$Gv!imYZ`jOBv|lM5zrF1>(UpbxQJmiD)w!{A6>fw;}T-!29YQ;taMgTN))?lY``'
    ';Zu`?S-lAqtSDkqy~rWCcmq3Lh=>U&Xv1&6w-UHe&`B_QY43gh0atc-25L?0;%XczDmlYU?{kA|_dnOAcb!{$q;$x$?3nPb{6(2%'
    '=kGvWs4_rUP)z+E&l|pHwgN-MKx-F{sh}A})+_)Y?V`3sDp`mUktB0nd`mu4hz@**FN`0op<%__&R@T@dP3H^BmqUfR4%kvFl+)@'
    'C;~rWLg(rU5Fd%G>)g1wBXrYbQ~ES)2mkCHky9a{C1TJO7w4MOGE)eF!mn~s=qWSXw`Iw?N>bzjt7qf^kDp{zIZnd#uXsz;i?5j2'
    'PJA_A*NdCcNDja?t+Cv15<KTw#mG3w@pHd=32}q&1wS)%`CjiIbZU6iku-~YvvrTRJ_xPAg_xZ>k{IDS`o(6HyB`reY)0`8?eAjV'
    '7<GLt6~7lAwUbOVOvtD5W-kf>@+a~h4MnK*eiKsktb_35CB<%UR}T(jRCG4MaT5K~onbQ5t}F06zX!(&U)u$CIsFlKb^O_R4@<s>'
    '6~fGwojw;fd2%>hwRMw?h6(K4@mCj>1r8S-sUQBWavMBX0cZ@cD-)VF{3g|CmLv~>Ocq&J7s#;-R*09!ylA&wSFchW^Zc6x$#yv?'
    'bwSk8F6t#Lx3a&>3v7@;FH&s0%8BfQlxR7y+@jnDHpfP{F>k2-V(~FG-KH^P$FiTntH3xu3+d{gqqqAs6jH^ip}68bQfA8hu7bL('
    'zAg&mfv(`;M0@e3I?+e@&j^K_KWIaahQE-z0iA43zq(RpsB-tCy18nMVfvEtjXPecSC@+Sa+!HM5T=cvJ5yu_)i}(^qrN!u=xZB%'
    'F3g<6-e3Sxd4jsG0GAk-8|6Um0(fzUV1fUc*ibO@V@lh1!pga&$8{3=LtUR2Cp;7-)J+v;`dnAI2J=N)D$sGapQ{7pT_@kh+$Jx<'
    'XXMa<XpN^m+QmU9yDPj(Dv2uX;wmc9DYIoiUn!)nUegdCZ|q&}Pamyv+oH(@pJ?^MbGx=jh#96_!!Ey-x5VVo-gdI}8O2ywsOM_}'
    '*|z?mL1`5eVPe|OMOVN4mZRk7TQ)C_dYssnOtBoCxZmTr6e^fC_C^J5Ty3pcqdCBf#{F^sX3?Pq82*+3E<XEvH}y8xX04GDo4f19'
    '*$WS@z^5lPqxD#C09RzAlUMf$1aIrYiFC>79~Y#o%*z!cg5u9`?2>f8-lVC(vWj*ArBYlZ6%Tq|q$~tu+m)214eSBi&I6w}k=QLR'
    '3kMhI=CLkD%BWWyg)%*4s~1aCTu6@{?`$`j&rFsa;5kP4eib-Ns-)usi5XRE7mkf2rY^$d$c?QxQDxz1(wW8%=F@R;2*{QKLO6Ad'
    'U6pm1qqIlVYaZM;L1h?hFp?b;)2)l@YNw1M1rU$q3Yg(w!I%>YSe@-Azzh>FB27sXz8Vrf+C)626j6IFG4u6gXi{Ys*!oRy8LCJ)'
    ';j?62c@emPajx)FR^b>|PY~x1ut)V5rQM@XWY@^8W#lR({^p|312ZgG)PRw8zaYgZ8VT(NYqa@IuzeWd1}he;m0E{g{_H*E<exj`'
    '>h=cxZZ5WJJNMki#i0eD+Y$fOY>(H;VCr_%13a4XMeXVlE`+COa62g(P*t;n85g*t^0k`6s(oHPdu*r%)^?$2_@8?-+4_6-n7Q@i'
    'rFk(<heplb)0NbI0WfXPuR0a$Fxo}u%R3;d2ChQcdtKZ%%+?Bo_X5aK{%k!fccv=QE!wAxg-bGB>48-!w!s&Z2ws9L{q{@H*2M<R'
    '#a28TfuC3XK?ISyEAU38?rK+NiK@=LJgPIlfX7+PxI;JM$?Hv$pHMnAoEDWwb@A#bt_aGPBsSyzSR!GE701VI96#Q>i{}LFB$b-h'
    'f4i8<qHO>%LlZJ0?vG;|%&Xbu_rr@*F90I^ObN-wp1aaS_{Etr7gj*q=5~ymB#O&JWnqnHBxKLHU@0NC_L;36y7GE8<i_^MI-}yk'
    'ZgLYR8H`(vG?-4KWX<0kHVg8@70q2xj9qYhkO|9S7OhTke*?~vHNY|qQ=t2=e&K3|SZra^DFOJX@dj50lYd}kXSIG_W$Tf!8$GWJ'
    '6HxE>U#C_AGdgJi;NM<=ELf#Z<wGHtd;@__{){kDF=fm`^|M`RCvZ<N@^1wMUX4=j<(>h+lbiek=@}PT9VWLUx;hB6xPOHL3SAnp'
    '7yUEF#SV`jkKvcF4WRA*>o9UzW!GgNLEx@kbeOEd28U37y>|bNJ~S6zQ+cg@UQ9=nwdHg-*`wtf_y&&)B9dn%7#dk!VCs$Y*l<2T'
    '`CZ=Od10<<qspAL#|251_Hs4fbGrHlQlY?~9OEFauUxG{7N1<tu>dRgzpA+kwn&yA@K0%1poPQ+Q+>a!b${fJft3S6F|cYEyqO-b'
    '!?E^{iI%@!npzZ`RRHdCo4E^**vZ6M9~1XnyV&o5CI>lJK$AW$Ml~9xPGvYSw;MOn8xoiX4QQMHTs1CvI#0te!311gTp{4560Wt9'
    'Zz6CY9fg_y$3TgEwWsughc2d_mgUNoN>M4#oF;e~)qa!nk8Tp8H>Bd)u9lx%64_le__z{sk&rn@lWe23n;;@xnU>I{sk!>q*zpn4'
    'gcVWtqkbtzS>)6qZ@7b(+<tM$1W+muzG~xY@hy1JT^>_%3A`GNR~R{b-R&km-@GrzQQc5$*}788*tP?x(Q$jZe;Eb{-XLQ)#}D^P'
    'LPF|D{gsgw^Zw|F422pxbYhe2ipT)efFn8IvU$;ziXbmbH2fvT1%f0;F&!S{mDTlg4KCnstDtg`&#+x!$^y#`dt9OA7jGXEJ0;}p'
    'c%2LE1ucL#!j2H3UR){zuHyJ#A|@|?IGzQYrv<|o@9pMSWU729eTb>TYkQ&C#YamfLM}Q$z3hm)_eV$kgFHz>HFNd3puBaZ)|MBT'
    '&7JkvuEKrclvO6x&b+|y!O2n7O{cJ=yRZv^+CTph6-R6H{+0w)e$>~`mAAIN2j5mI#P!}&#f|?U+<!usw1HNu)%SQRL9pSI)7Wj{'
    'pD>?HS|#tjPp9r8q&@N6uqn3FlP9Psj%R;E?%#7hF>kOU@8>^%Q>@uJ67ctY80WhukUzdirP`CLFZd*`^#rW{T<tFm6)Y>LS=$pF'
    'GcWx~h$HIjzOI)tW7gjL55r#BkUg<3pSxDNY}}dk&-+LVupRU#0{14F>U-oW<xBn!gpdCj`4G=&J~?>F_NebsCA2=z=Hb7=i$eHg'
    'ePRrgf5Y<o)ONBxe|ANvjO+;(S`Fa!i7f$7*Z}T+v@~s57}O{C89T|W3+#zoQC@g^c3nX~{3QGl^PaN)!j_I_S#3{>8jO$HvuJbl'
    '>;i(Hspe++;n_S!3G-P%IsO9XPeu<;AUw{qw;@HK=blf_NK|~Y+Iw^!G8CP2z5THSH6Z)#@nrv)m`T6K0X*gG^F$zlKlzL&1wRst'
    'GH#@e-T5U?l2uTXuks{U2&;g8C*I<~rF}f<0OVTO=d(+Kg46HQyE8eqD_^m#A?u%444i!m6CHjdX%9I9+0s1O7%I-oPa0H`cx(0h'
    'yh==j%QGY6${Z@5)#x(+xzVtP5!!rOb1v!|dv-}E&(^w89gcOx6I>1tOXX->muxL-pYp6YD*O{ayE0qT=;6tu8hB_uLAxC<+h@ft'
    'i3C5*Wx%gQ&UDw8W!FRe`vLit!u!e9VB)gMdl+J(ZOWF`pY*g<-fq_|?L_7{o;=+4S?!1Q!c~^lPm1r_9o71ijlPc0tU{hN`=x!l'
    'KdE}LpsIhPrNuC3wI?#b>g0kxLBz1Ci~1x1h=#uYbVnNx$>g&ySRL&-%HTI5hGO~F#*<~p%VD*)x7aHj2ZVRv-Tu$xi5j2l*<gfi'
    'aaMm1cYjFJbv}`K&Plq>6Nn3^_+>XbvKF4c{)8nmppT6worBE%<3`f0i4Ej<(k;Q(U+oECN5-bR-3WMy`5Di$Qb<Lu<=G_}#sN1<'
    'JSGp#o*4&YK`_RX5*)LU<~K@JALQ9RBj7`(vGwGdt2X{}C**<rB7ZiCDFh=t0aJ`Lr4~0r9*FOJaz&$mMm)h%a2=lWMsPkB7#>g5'
    'NA!hHH@u20$n8FtgaLlHCygpUG0K9sX#EJEukq|+<W2J)v-7+d?@wl!Xfww%uo7Qd#b`c7lz-Yy1CyE7pBy*%2DkbI=WcnTjq$`%'
    'PLbBn4(U1z0fB%1L~E^Kr~c$*$}+G%t(l;%`jh&BryBEzDH9u1sn6Ss+Mm`%_4!l%@ttoGV*F&>EHOZRlRQwOuF9Kq-A@}?EcUHG'
    'sR~+Pc{<<YvB-JE`;%oKnD&n+Bv-NaNAw$E4OzI=XLRW~2r7L7#^FLTZd84vTbmK_{{tKd_|2Y_sAyVxEpNg&DY#)iTPf@YR{DD|'
    'HFAaJ4_lt>ZA<=H??HzQvT??=r^1)q1>WS@Y+d`4qH-t^_&q2=xkJy{pDi>zJ?%Y!@70jLKVx?Hi3!wu5YrX_;WO-th~Z)Hu?H-2'
    'ke@U&mHfIDZ?eo3wPsHN_%_yiRGtfC#b@wabNcZ5CeH@)$}>a01|#E5Yzk+||9v*d0l&!S_a5Ec2%LbP=s@ep4rt7$$bgS=qajdh'
    'pJP6)q5ZibD|4-w^FGTF75REL=%%`9{TXmo>dpeZkG9A@-JWf516$_Ro}z%tyS~rFq+Xk%|3caNcs5h%7~u5#RH;qb+iN~$WA&%}'
    ')DE#R?lm&8c+v@wQn-9QgP*aj8uxjZ_<^$be8PzXllM0%<N$8-wYH~kN!0cpSCec1Gs+W80Pm0bOjC~Cmpv;dYT$qK!@CCFB_r1J'
    '`IbMMcr!EiXOj>nNnuaOYS6A{`~}bLmV1Q9lL7#SLbWHtuZAMr>v*bF80&|N#D@0p8NA3BBA(6W<|HNMKGh!6#L}M30$aDw*4_k^'
    'Ycc?Q^O(a2lKokj1GRZAPccQ2<(J6&i#NDxoB$cP)0{_rZzW|lHaU!qRu`1eFR;igc4qDfQXR(|s0i|`1Uu<nuBZJ<%@wuh;=?tr'
    'lK6d<6@(@^TM=oF`b~I&v*VVZRXGvCcIDWJGiO7EiIm+V*5yE{_FT(ew2`)JbpkPf))`P;bo(`LR#@IrBYE!+Z@Cz2Vx*SW3oHL3'
    '8HXGJLJf_utBc+cC50T{NMEpd0XyX(m&>9@t|z?wFNX8p*b<04;9&i;vl55gRfZGHqBMa^JAs(v!w_Nk#Zz{isxjUsx@EPC$$>jH'
    'iJ5$QX8FOPwHvW{#bnfDS2Rc?Xnm~y5y(j{8n9fPxkB@$#}%^OWJ+zFoWELgJ0(>8f;)j1hvGXw`*;JYzvUY;3!aVnX8N1UFzT!`'
    'yQml}L0nZX9+D9Y+hyU4+{*7iQFL1qcWrF8^9c6~e>uZ$JR3Uu=s||~Mc)+*caq%Ofl2(CjQ2;EBH<n@-?ER9i?Ni8p_b{t)@a*X'
    'bzOiA#$i9?f#361^oyC%LO>CVjMX5t&3~}OK`&=Gp)zY|7mcV<;zi3JA1Oz@g6XXAz~LzA_#9VDKkq8~!Kq2wyc!xT7TJorlV|-8'
    'N2vt%wYlX~ncZGce!|Ee%A=dSxi&8>JEO>6w+cSHO1#Vs`Ex3%HA|X0e$Z27G7=-|Dhu_`xU!|ni~G!{*2T8(;{xEHc@^etTHKjB'
    '?~+mmch0joiCml)QK}>b54aGXui}_*;C)H@q(#1xo;Q1r{@&-}SS<W(5A%S)dGg8t*IGd9uJQpc?G4x*4Z*bB>vB6a)H#<?;llEW'
    '#>$my?cz8v*{WyhoVNhM{Y?f073q;P?zBSev|T8{*lRuu`f=u1S>0TG&9V50dqj+Gy~~5Hl>I}`i1k-5h~bupAfB^rL%j9sQ93Tp'
    ';Gj5;dmHz-$h`GgqU93`8p3&TWEm~tC|TsREd1<h9jZ7WV@fj5xCub&c-QG*rylQd(Zc5{%)T45`=-PRi%U*W8+FN1u=-8jNJlz1'
    '$a<WRFUr;PDeNLyKVjKF*9F`-fJ3tPV#Tg<lh_-hG|qY>K8F0>FQ!1*Y1Q0M2?i_uS{0KW|7WkW$x*sbze~%J1xf4q5{~@f?ZQ1c'
    'vVN%@%s5}h-e<jNB^DutU1jE#`}3>?m*FO-kLvB`syK_D|03~jgSpnYsMVq~ht3+5M2;*M&Wmv_EUmIEqOhG;=L~m5SwOYst?55F'
    'yGFk98Q!}ze5GCBgQ3*1veaY@i~G~Pj3AsW(?T^`w{dmn4buKlTbs}64PtcZm8NHJz;;tE+8gUZ2&0onJrKoSAv}}HfTW{7Nq)Gv'
    'OZhqn=Jj>RZ>ay^W5nE+R2))9qT3amxVT)~7(rgtRkfhVph~rnoz}R-2y!v-PUJqvVDIXt<Uffz1-qjdWBy^h(I9@T4a>}{7c<{U'
    '@W||(8$P&pc}4VTjkHZfo%2G3VtGFA!@A{m)!GkA2&8GC6cVA-F1*|Wh^l63)3QCwr*r{{PykW2iwL-GQqZovw~jAA9%k(-pLSub'
    '0uYi{TE-jbzek?=3u}Zt(b;g{dz?Q9si71n7Fm+^xS#{t90l1*7DSAq`WuLCIzvx22$oG@R&y6q7Xp29*UOj8@cbHADH(aB!o$rY'
    '6Vu-AH_2#hm`666WBMS!O4rIu%lg3j-csg8Po?r`aPp0!Ve37vv5qX(qG`mL=X}^z&oQY22qPW~Kk|#7!#YtVJqumjYQ|kGG^&%I'
    'nKyl-@#=~$uG05WnGY)aj|<;WV_iG7c-P52Z_*!6$9H`^R4^Bvm5pWD24%Z}Idi*mmba`6sp6NiNAqdk1n3w63xVkD3O;v%%H_jp'
    'EasNoUp6E23M>GbT)C`;K6}2I_bAMx=Si5|n?y5lF$*G&FyFOoYzhA>7kL(uD|8`S&ywx4{tedW|8BU7L317>c}zldzr^ICXDQm1'
    'wd#FbKE>Qxe+8A^C8FBGvjQejdD2MKuqR##rHcAG?CM5P!w=^%GK#%P5Vehlbb}QbcD+swE?j<B+LYGcpyMB6ibea#6YR5uT@HZh'
    '4)QEfJotEnKnqvpmbK<2rM_Q03onR5XZE49<@?t!*inYpS(w9QiO)H%h6ffE8P-t9W3Gn!gJ2_kwv`WNa)|lW++s4w!doe?A*-Q_'
    '26<GxbFzpS<c{_$LI`2P9o)H3qnp2|Av~-7;>)-Gsr=#LvJpbH)&$#$&#NR=1FFM$VZcecie)xra(S3WzUKI*U09r$%=)s*$_vl^'
    '*naRu&wn~kxtiyAhrV{DZp*KUg`{tlD>}Y`Oj7da*O=Me1vrQ77qwPp$F?hNm=On5vVu??Y@P1_umUM%;c8pF_$E^GYxs$9Flf9^'
    'xJj#5`$P<;k)9Q5*5l3YZx}LGU@x36xEC@==Y^x=P`(q*)-2RC7u?!H`g+v70Ruz6dXLwvcluif;b6s|;TLUGp8J*ATjh;oq;Yln'
    'MQ%;GuJbbC6RvTS7K4*9o8fGbPy+S&-X?0Jpa6nYYjfr6*RM*p1uKl%@$?Fd<qbmb5hQ37Y6BBEzoMaLC##0Huxj6+%I@e)ib0VY'
    '$RpZCZ)(u+3e#OC+GY0ppAX&fyc$_3*?<)^FHyBlHgLAVgWAdc{`9X7W2j`MXq_@?U836AIaKeSB{e1t<NkEM3X>aL)=1cHaaA+V'
    'L9>t>=E&plE#hu^DK%S?mY3F?vFin8FVC<AU*ThDmObykIX2iNgW!#<+Vi2Uak+=&9l+*jWU*0ZyZ`psy7c-{=_s>pSyw<UQh6qm'
    '!v>Q1H@GtLu}T*FK$r|LuNrCeN<7=H$;7&Uh1_7>NJV9}_w4E~bS{=p6XRe+BXSkJ|CMo~(bO1j?>_6*lxqQcicnIk-G2kAqT>?A'
    'VFs^i)+-Fe^X(da3f#y->;8|dqPeOX?d8ixvg;Lfw1U?3k5#f=Epc@NN>*V3O<U5*evfDClc9OOK!N4Xy!zoIm*wIq)6=<?CvaG+'
    '`O*?gW925Jct?q>kTYa8<w{O%)9o{QJ{Zn8>zD9WR%Us)jBIszrQ2N?%oW;P9(T)EmQ@#?q}jmby+EZ7rE>I(WoURL3@>?hln!(K'
    '9%l!lb0{7mVBOl4CZs~x&26;H8TZ0hM5FA26)kpCt=+^d+h*+~G-N@-MlmlqQ&t@pi(Gv~^^B|H1_khBaF2Iuy^C?9K`m>ZEomPm'
    'x1TF<kMq;x715mxLA%n1m~qFd!egw>y8zBXx|%l;dc))DBJ&M%=^Fvz%e;T5C*e{Re*-s*ex+y4T_jINsE#`F-atg>-YYYwJSxhh'
    '5*=6C=G<)O;%fy?uU!q1WZj(&Ar=SaQXQ0)6CHks8v2tUhG-sn+O3c5CUT!xSEyf><vqXmF~QR8io!e^e~B>$<aNCe9b}D*@(0><'
    'BRG3rk%U}&BG=TcwLe@inc@cU(wdxb<*KEhkbik{AWXR-#q}h=<Apbrc6#-sI~LoyDd(l*f4Y8<zReu7(SZyohwFmRnqMoJBT)ol'
    'k9~E-4y9@H42iY%n_!AYl1{EW%Vu7hgLO5K&4-kaOZ<e@${T16oRKRDfqgp-ytfO~lG2gUUKRlaD_?8^zyB-y@w7MKG#fGrC?6DD'
    'GvDyWME%ef+wAxO;|DZJ@yp7ct!CaErpY(QkdxueRH}c}{I!+$%|LOTE(k5MgF>H^dm1vrQV+@u&H&m9l&;`uj<3$RIhG-1ky){i'
    '(EO(jxj?pZ;9H?w9W8hBx%q|LfTEmIc~CM>nUxr%mbvl<;zo4?0Cg7p$zTwn+-wc!yu1Z|IP+k#XgElJG^=9^ktNQ7ibsrNK>L*r'
    '3O+BS=yu8r1;s3gyd9WwB4NnaX4ONJhTWG{{<3@|o_8G(1y<^xTT|$aW}x59gFpZF0Y43QIXn13+GpnnulOf=7Oqec?1a`;=D#%}'
    '-+{V~7i2bd3V0$!JqsVaG{BmWJv1x}4s<E6(m)1df1ca~iNO2d&n+rn@O!Y<gM5i8iM4DWPb_q&1X{g*xZqfP!YWkE1EAo!%MZ-m'
    '=tY-t=UfW$Q}2)`Uv3_jiz6#woSzmBNb`qEej)7b!?ycn$TPE_QsBUV7Lb8Sk(f(g_5=rK5krDA0?W9(Q4@2=Y@_-OOwz53L)hrX'
    'H~yT5tRJ}Etg{6*Pv>{dE*vhE#ohPzY&-I~&>&^gs<!J5+U!fnz4Wz-!%oNCEU@MypVbbUrtGZcK%K=Q+f~m^LN4o9&P%s>Xwez+'
    'Ok-e70g?O3#LKH3uw?67w~mSm{%rTqq|#4`U_qL5wmph+%2qCq@7%we-yrGCMi{WPw%f@tVQwkZH~?N^Mmx3=1kHX1SS}acPhl+@'
    'JT+j1km<SbrVgFe4!&BI*pZ=sx}v+o6l3|SKh(uJ@RDnn=?^~}$LcH2=-FDjthGaVnjG~`J9Q!MvFD+JA)UGn1Ub{^>ARS?;iZ59'
    '??5<7mb<OPRyQda`3H0NhpApaDuSQ(2mL-)mS_ue<B0=HK6eR<TR-@YEPBDn$IHwr0{t!rk{#Oj)#XYWS;s=_in)`}%i)lP_*LM$'
    'jSV(W#TLZ;AsX$>^Vs1gs1K?6SZZbQ9!6}l7%I;$wa8$USXGW3=2-)ILW#}i&dZLh*O*$}fE(P=dO(70Gf!fQK%ZG39>78f97euo'
    '12xNi4fWdRT<hKldBeDuhEfK9&ILMw0rtM<a<2>*0f5_B%E#%O614j1T&8F2aD=iv;X8;l8p?s2+m==u$B7iV@`E?rNb+JR=nqXE'
    'eyt19j`NZ^J8D+)kY^9DW7aQ*yKCn=6>(6$n^^|b8;D;$#zVg2D2zJuRV#tH=wSw)y@L|u8+Oe?=$FPoc7X7VHZBPAVG4rx%fkf5'
    'Z{P&sXt?qOF}hClT8jtR@yP1uDP2VvljmonG7X6;<3!Tg14&um$qYS<w_ce<-@qr!kyn&F_P2&~;4mH>tS3X4EI|yBaDF9f)=RKg'
    'IGx*J;Fa}C?hJim6PJ&A$s{eSl63>gtl^qukn|0cjG*?O*njo|29_6HNTBt7(Rz9Xiq8RaN7XO$!oe9v{$^z!5;y349m#8@pvX=i'
    'szBD5%h|jSC1C|^vG%d1Z+oG}-I<4zb`0Kr?OqdmxNYuQ$_<SCY;^32k(AK(1mFfE$yN^yAw5`^U3=t9fVptPRpP(bo8K!WuM&or'
    'T}G7+TDwEkDvjf3so4@40WP)}?kyh1fm4l#3U9mw<8$PdXITi{sUM7Q80qDbX4czwJ#6zp<^m26CW-W@4FOtZ;{Q{I4n|1GggIe|'
    'zpLlNnFUkjgOUHr^LHXKrqhm|Z+DIHS>n)n&5HCh@(w8Mlsixx%1l4pD!P}&*tcR0Rkx*3`5{u6;Z4*6HG$E?LCk~B+z%)$3&2?6'
    'EC8eJQ;6QE_pZtEM{e8R$3W)t#iN)+IdO1`XXQ@l78AQw48aCG;c@SgS~l*Gw6Y{JLSY=!?a|d1uhZ-rzXJ*p&!!)v&cTyihtnyG'
    'xChQ2j>#I1my<H=F!%91wQ}F59Tl;KoPpd+e`+1$cH?02MQ<fFB{Qm1mWnaH7z|Ugyvp0fCqR^5*Qo-{S9if23mp!KuyYy7E$%;W'
    '2v7upC#%x9^@;&g?iGg<lR23qxnxe}IBf7TNwM??RlT<BOr(aac34ziPB>@<+n9`;2OJO2HVsX~VH*q-=vde-9PQyeVmDYw!#de&'
    '7DQ!B;V3oa2-|1<84MxxY%5si+~H(U$VWfC_Q6BQ>;8b#GqP&aSJynCU%2l((j?A~GIF8seC`_%SmuyIVCmh@V}~4pyanSBB$e$W'
    'ZVXEvu=!}PJ{7QdZ}G~&4Y22XMJ?4ii>l`p_A=iW83A+7)dnE~a(SWp3;^yZCtN#s)ke~8?(`EME!R!4o4X4((hRZWe~A1*R&7vf'
    '^#jPB%a$nBtU;<tp3{)M<dZJ#gcfzE++=(?N^h2jdJMk-sU*+jp#KFBvI;?xE6LM3f7Cd%)_IrK2qGAe@kHK%5Rd-g@0ax&RW<Y<'
    '{9PQT5C$o=#On*P3fhqiczrkyTG>Qto~<yc)AV;J$!|vyI1hQs-SsriLW@^Pj>$F~y7TrdlpE*EJ7x_}*o_<_1pR(CO!*+oVHjcP'
    'a);_LhK`Lk1e2Wn^Xp`Z!U!i18DmtwTEQ{~MDzx1Z->+{2(O!2do{B7AV77mr<YZjJgfmsMmuGRh@y5{;xODCG?B`xVtz<-GVKPB'
    'nufVkDR}PK6hiq!4p907M4h{{oZtuiUC@9m`?@%^r;&6%Xx0LD))nG`JUs7Us%8v@2YtLTNSu(mh~^r{YsPRiOzaepYly=L)_Xh9'
    '3!+J*Pq4uFq<Nrjwzz@u4?o0b=ZP8fW5gAc-S!0N%aVE_Ce21Oe}fDuQnIbjQ?CGTD~mPiXo)ry6j^W3YUW#~gbmc{`NRsu9pyEh'
    'q(nwCV}bM-&31qm3ED9a2!a)esvtsiB!vtDa2Idu^^+}=zG<ODMZ&=t7w3pEj=T}RODspgzQa+k(d{n3J5G>?QLatfSu3Ro->hfO'
    '@WymU9@lGvfru4BXxq)Fr$&eBfH(%0scm;*K6f&LGaBA}-t{51847h)X2hxVC{Qk-6dAfe5=ZfE^KqnuRf9v?1Qaibd>zlG@?+R$'
    'qq-)A756yvu{Do1j&zwI4mSF3D+VRZ;I~d-fh_Bw_-;u7+-(#gcH$FdWq>an#6I>3x4z8;ydGZpENyV1!N?geot_14$57P%U{`8L'
    'HA?|1b6qLPi~2YGBm=Ds;$si*3b=?~7eA<TvN*}pe-%nTxVp`o{KUQ4>B|PZpf4j!$X5*qrQ92TsERe%9Jz>T9@xToE(0I_I2iS+'
    '^%8y4OlTD<CD~Y>n4BPS`J&W40Ul5#hwvY}!D*fg9T#-e>iN)7bkjV>iRqQ=6{SMfoUtg+r)l19cwF6EK7zKA=*QJtu0z~Wvr%eg'
    'zAt@b!g8nQ8@#@?W^LgLFKvg5G77Y%nTDJ_BWq|Z{>!|%%A_wfy#paHShK_GFS$GA0kck?Le5M%A1^8A(M73Y&UA9XH`<BOn^^V>'
    'w>&17piJxa%Q;I5%aWRvS{BWYEo|iIe-28~JW?3IWsBrE6$o8X&2}=e&I3Mod1i4s;A<@mu{rE+Ab?Zy!~(A|LTnn^2eAp=-Zy$9'
    'Ylm$75U@f<+$$|N|J*|3V31(UTeB`f(vjbeng>U=iUi!;oTSIx-*uUV&>#|}jlpj_McJ4fL7J2+bVp4=ffo#kTU{pd+*Zhe%N>Sb'
    '-?5q4?TB75N%UI4w0I#Yc4ho*<KA0uBa*7*wo?~baEvXDDT>z<S$fb6s}HDv*(AnH=A-@Q&bf_(R<KGH{?ybYsmnd_vFL4juZVJp'
    '`J-qa92xcIj(Zhta3?t<yIg73srE5`sqJ)`twBPR85%9q-2Ctoci({wg-h1rwUc|ptlt>3ta3X4v&RK(rfXa9*63xR`YgZqo0<J8'
    '3>7lKbNP~?zIKw_DHYWed2HH$>e?ypm_&|#bKN(&ASg4$$R{?vV?_|>;LZFmIDoPfV@bY(VE~J`^_#z#upV*^BKn3KfctUw`w&J>'
    '+bk!g+-A1`xj#DU7&q@^Dh8!J&^{aL2`KKlx3CkP`b<3WOIIsC^QJ74S+hd*a`UNnm5OE{_BDDjgv=%yft_AWD%=p8)JbM1#uCAd'
    'w{q`8GpiYTG34oZD{i`^buoNT$POrY84&7xl59*87q$#HKQw(|$O1RVtJJ#evoNwtJ9LO%$YlX6CtD|CF~`9C?o<>aJ6zDYx6qDA'
    '_W>FH2B^O?Tx&~XJSMD&ZKMi4#i!a*bt{EEvjSgo&aX30*ddY~BES^u)@QDoQ9EL*8(eka4P>%m)U69Pm+XX2RbUq{O*$4*l9{&?'
    'm~<$04JE&0b!25eh}1;ZmUaVMZ7-Oe6H=Ae+!(#Jk0mdXI92#03Su#xhY;F+{OSw-4kvh&7hQud$#Pj1oilvhz~k5;wC>zJ#3n;@'
    'xXB^hPcLc%IGes@R|WS38oE<}hNL^e0BPmfoSpsp$RZ8N<PF3c8p<rWEs(}t+mtDPKbhtl#tRJ=O%d@LXBuLI)`>V;@?B_nR)jMd'
    'sBl5y_t*`F^Oa0`rmiMG2Eui^0_aZ?EUY8){nTcsE>}4CI8L=W+y^$;G0^aGgDPoT<pu%<98{J|ZswwuExB1*Jx;30OGaA&k}*Tl'
    'Y1?jJ_jc~8I)&>V$qv-#xLzJfVP)X9I_Jj%p2D=H=LVL?&%OH}k!VH-Lc^K>?&`{?l})^z2E1~}gF(m?o2U%}H|J4Fm%I4q#J@oF'
    'vnf?ZZ)e*ba~RX4>L}6B3DlF>a`PYBtqC-k0HzFNW6EuWDf~%u1x2otY6z}KSyd&0cRyJ0iz8Xx*b;f+7Z%Gw#=x;u4}#Z=VD&}n'
    'wo4FEPwr__kQB_v3>%zD?2x)N|AU*NF{ACs2o0M2V4#eVbg_f@#PE);%;b&1NXOR|XXZB71s_c(gdxkSOnZT(k9LvS!&HUI=b=Fu'
    'FOzTL&T9La9fWCQUWOWfP>lh3PCiHd^vKcnm-;|_I++uE+2T-v8r%dM90<;NK6&KQev}&&F`%qYM5pGrhA@19*Ty)u)5fEDw9|uG'
    'Azqnx$GH?oXoRolE|0*D*ac)gv%K%zb`Xp!bCj-W2ga|WHusZTl`)jhcZ`jlT^o2BcHC$;mzZ_BoJMY5LmXI47HQd8^;3k)@9*IC'
    '&!`?BA70~m?6iOVJ-@%R7&!GC@F~a#VW6sTo7`U56b-(C+T^YQJ)Xbh>e|P8M0;D;kb25DtwXIe4%<9-r=UPc@A$p38BylI9dCIP'
    ';${yW+RO(m-2#!3y(XVhHR3Pv?%1w0;IiafcVQs!<7x?~yTXgf^J472-;nMbc6Q35pA5-dHb`PNFN|`UJ>02Cm+4v`*@xu;n-v>9'
    'B3dcS)5^2K);NL=tu$PFp!~vcr@K%p*}Ygbn5<(V6A-j<{3R7_K*Y!WD5gRyM19)U09njQGW(!OW81u%$Yz6TJS!%g9ot9sxxHzy'
    '9u97$T)g2HQhb3pG~6)|XG<u6g`^C>^C09oWy_xH5bR}fY-fuPqc%Ramg47kPMnhmu)T;T1Piq7u9ka$AjQqsfJs{;v9U>^hB_W0'
    '8YZCRMGQ!$cr0u6pkiY_=MBZvm=cM%E=Q)x%oVSJfe;;Iuv%GFkn!@)e^lHG(YSi9-Jv@szplK!<R8HQ%r6(XaO(rV{k)I)b5d|b'
    'w)e8z3i0+o#Y&;<;*roy<a@t#-yNt2GGyg7!T-~<6>(*60vBQ-0JGF|)pCk0jMBRdE$ja!dKfKH8UB!sW25a0H^$&+1%EF2ujPJR'
    'Q0`ZFczFa@cx-rv)_e==kUrgalZZjmK&TEvF!OcID}@c6Ptn=tW#LuP0`ujCh!q2yQMWgXZ>=?Ze90Do;&b78I=Pihllgmwm`K|k'
    '6a5P5DtR0@s4~;^rGvqB$0QcRxJhD89CMj{b?OM-Es~{qCip8*8}`OODMsT4?>2m$jUpadRc@qLDfa(w<|qA=o|MPGIKq)$$=(#>'
    '$~@S~Qd}?#RRQe8>)pTeAV_y3%v{f(dt4B=Ios&|Dn^-$OcnA09P*(n=(TQdTsBPk$b@PJBqXE@Q6X@ZydP6dG*BhvF}U5N&S?f('
    '^$$wRfGpV{jFs`LN8oM(gXvrg<#=MrS^rX}3sFXlHP5?c3dVzmfUzBSOQc)&OeQHCp#!Zp(yt*c<n14vKcp@2o`b>GDestk#W;oJ'
    '-##UY&)-?w7dOAkvdAvm6j?{d*EE>mu?|WUr?ATN4Tl@E@hKFz!Y+G(LkW;oR2I48X~&C&u+9my{Tu`&1hQQ^BYdL(1N=eE6|q?S'
    '@1oT7_OTk)s@a}nty<w(VJMY#8Czu5=vXZ^?>9Slon9z#j*YP`&MBuP7;Kc59c`fm?oIx(Sueg%eswN3K`qYqn5UdK`RZcuJGXMc'
    '^(^sR>VWLI`t#k!z25<HBDi>N27o>p9Gy%E0PukjX(y?d9a*juO{^yDkd;>(_oBEM6XC^RFyHx#;*WO|*8(D92(!-9iiIkNUET^{'
    'Em9>B;N@N#W1GjxeMRhuLH`_R^)c!~K3zH0vQov*8+DqYROQ5DS1t(orSJkm1~#Hx>2OfEo0zRa&n5k#ZOS~T5+;#WUSLTkl8A}='
    'L!?f)T-BObr7qkf%S+EU94;;HWJ(CaKD5fjvkOYXuN<_G({?hVkBa{}ZTrfqso?n%v;)MvF*(qh<lBL{O$-c3vy|nZDQuJ2;7JqK'
    'u=Jk&F#>0b*5ugQvxgNMcN<_7x<*#!pjf6fZrHm5-({tn4W;-6Sa0x>U{gx+OZ=_O_I#AMycUejF!0Vhf+)z9H-$=q*$0~Cz`CD6'
    'v^+xjGQLif{gBl2@W~s3HO7#dd2;YJTIAZba2A{b5ZlGr0!7?lcU}Ttvp5rHcFE3xVgpYtbw;^*QIT$BYnz`|`O+?J26_gSjgnz3'
    'TO|hkSlHlD)%ii?8@LEPo+y&?jKGE{+ZY^mv;KqVmwjy8Vm?L=l!%gK)+Q4VhoRinX^?Eg0JQ!e$d3~rU%cz5G?sazkczHG&6@9G'
    'L7qZfz!IBcZgbUXrYZ+2V`0@Ymo^!+<fEzB0@DVxk=NH`bR|elaH;D;1BahUXAWAN$0@Zqi>z(HqKbOfx9N>cYdbZtyk5yo%_Xi#'
    '?EEsEorND*{^-EZFd}L`+ksavcdX#|SJ{vFNnI7l-vKm)zH=ycx)?;$%s`-vN%Z5QETdW*%xsZrM%T>KkR20fGb?r!1L_inof5z{'
    'S#5OuUZ~Q!P!yBJHH|{pzx&@s^a0jxx%n)a2#Y6!fTO^cma!TO9x-9Q_aA}??FEk8oG@$YhwS^<#Ba5$$;dYu+d*zvg%!+gVvFQ@'
    '^6<@O9im~!T9#fab~Zvl<p<dy{V@M9A80Rmamq$GX1^JX#q)5;>y<qZLpxkFa<iiKkV|;gCE-lQgt;jPqU`hcUCOM6KWEcATzZ3X'
    ';j>b=fJk{_xg2?R`B*2$o47XSHS@@!11$wfz#x&)2$ZnS9t7!0H;ws;JXs+cOpKE#)H;D^`E4d-?OxztUOsaYFpjKn7_ZP|OA*BE'
    'geHBf_!eqt8Z&1>tFFjS70YWm*XGHVwJH|H*g#2jMfsFfSKqZ&+u|E=Q4&K8zwSKGsd$4qtXzJXK4H|n{Z7%^aiHn+!<h>RSga$#'
    '$o*k_qrcu~`}rEy^XI26Sv$OUa#Kw9O9NQDR8fW0x#n%eL_Lr1g7}w4ems)J45iE<>H~Mt+>Y`rTXk!?P_C)oViF=_vW4v})+m~i'
    '3>nF`woxw0#X8#M0X~&I9VnW>6^7j*pMe5&k>yoT74{c2Nm=*+Lu5(fS!-`s8^a{B)6XE=7E3apF+2^#$TK%&>GvFX#UIrzf|O8m'
    'MENTtu7)uqW9#Wz=*Y&iVP{t0#N?^QHLH=wp0^PQN=vZvyZjTA9lcYBj>a9&4-lImKcWMVF5|=Y0uUZ|jg33v<xfSeNv+Bq{LR)<'
    '1&<mXu7^_fI{g2v)Nr8CB|FT2qpBxZj6JG>X4*c%RqQFrkoimg2l^!bbg#NivH83!JEv*V^OWr>hecerSZmt{;$gYz2eccUlJx~}'
    '$)G2ZwKR`&!-McI7o!=vw8GY`jbjVbK!ZcG4NWGr1<~}e)B+YCJI(|!Dx3uiXq#RIa!Qty`A?B+i9h0^Ufm=10a<6zg*C=tdSH}a'
    '5@+~!1;lh8PxF9#Fqjst7l{7wkAmRAbP<uaFjB|MB3f5jK0^l{C!|cApnC+?hNOeAICG6C{|DPcKsbnVh{igNU~5KI6sjgC_|1@^'
    'cVRyRxyl_1?^ktsl*bEQX2Z~F<-Q2SVLG;Tm=?;dv2{xvDC?9!_V-)vLKO|5=o7mw+onCfc$0)U<U5J>I#vi_#qC=lFT%(oYLjj2'
    'MCL?}P+$PxL9HX?ec;5{X%R+j#{p89C5sHe=T6%6uwPE-tjoOdPRpSIOM{Y6Ho{oDbv6_TUE9D5xNgQGXp_AQ%cmhS7Dr@=jHRL&'
    'OPC3U!hZs+W^iSMuGRjq+{tawG)FV?8)(FaOn4;*jt9>z;H}_`<-E42C}Q(@1{@toBX!7$Lq*z-DT6PGK^_0-+V3Rd(?J#t;1LZc'
    '1KiGGZD1M<<hyIORH&Pz?Z^#JIpC}LZffR<H~VL3K=_bUDb{rMF)tuEpfgCri=)f3m01kIRg6JG0c*I~Hj>?e7&%EE$b=v++gbGV'
    '%oXS+)aJ>&fof%{@0Q;n?Gn}rbVc}z$R&t*y(&b20NI~~>Ja_uystu;X_#do3MPc)Y5G2Sch(SQNqOk2VIC{C-Iu~)@pudSOD%_K'
    'ythP$)z<8Ca5ZWPSqR`#2LIb(TrKcO8GT@|D*iY!!$Oa+fk4KYY*>5=1U3-<(r#wq5n4|SiikoM4dP~bU>KhZ7-fIo02g|QHnfH0'
    '@8%^y^~NiDvC$80fl|A)9fr&X%`%a_U6(X=ZaA-3jr!MyAMjxgUUlBg%WUM=@y-~J7CB3b@tie8o`(5bV_SQ}d0ABmr^vZD#WP*w'
    '+Yy@D5U2(L(MT+GY`*ZCo%`I&S0fOS&<1d$L-v8<-&1j5Cq-qi=)*i+axbh)_n!C!?82Hb+MqE2g@!|m%hAH8jTdS%0Ofa7Ww(4o'
    'VMi;SnW1{PC`-KOb;Cs!<7BMJFSEw~zyhOblSvxKJX;Jl4u$rmH(bUbcNRI24dn?a!&@H3gHtcxQKhRgAEDsq-5i>*?{JB!U2qTK'
    ';m^Vqc-+Zg(ilSi_&~9YIfz4jAFA#ceWSiJUQmp(H<Nk*P%RCtOV@EUd2(Z0rNhvGtB{Xy+Z(pfO++ZJy&>=VlKYb51r2ssX<NJ6'
    'HdqRiLo8n!cGAhSWO1ln1ftDz@_+v@Ft95@AU&Pxq&lP91Yd_?0fq_BaIx<SeucO~UVAV~#_l*P7*}oAVH41hj;3>v=8C7bhxAJa'
    'ZEpB7*qF3ZOHvcI5Cpqk#y;-ZCnyT?J=!dh7D`kztmv0~No2XV2!o^>=J{4OQQkVGh=64`2A|R*2)IPJzpym{u>3o_D*dEhamdFI'
    'D9GFz;L6($(syYA79u>@5pzzs6~tSh#UT-QlI}{VEKXQPf=E-dQg<TEP!OYhvnYM?rOrF&x+|&`Cu@y)zm!=lmIg<hnY~a0UYo46'
    '*_WsM8DqY{)>-*@<8d1rgeT@xX#a#B^4e2#b$VJ{z6OT4{MbW>$6`(by&3jN)14WQ(19U((b|<Qfk<7+1$9I4QR8MjLAKti1fRN-'
    'uA6FK$%9$4l%d?Lwk;rcK!HObv-cTfH;B50;Y$NFU+kZi{UwVw+avAVX~<D7qNkCY$o7xFDPhu6#v&+BE5}|QNVIp%zzMq|8Y>S-'
    '9Db@O!Qv|E#C;sMW8TXEyDB$Obi786BA&p4S(M#K0Y4$@xx>3bKPD75;3ic#T>~v<)*he=z>L%gULER?mHo2hbpuRqvXIJ!4n_-w'
    '_8)H~WvIz1e9()!lD0YekiELPoGj^>yixGxm*ic_mH`r?eCqK6)@p^$(xA6EC&v^_Ev;9Rf`xLSi42iHG?X$rhbg=1#q^8W1>{3!'
    'oEAH!k`!X*f-wlhtk_JiE!T#1Xal4|Mvy@#G<Vg<4#S<m%(2mX!|3dV2-RH2l`e2{noa?dIQgm81Gbwlz`uOz0mHP^>D+-Gf(5Ae'
    '#RLivAqS{1b$`p&taey<I(y{AM3IQjMfAy{SPd{kD0#bjVD_<_a_$b&ahb_r{wLR;ymwiZ5}=Uxj=*G4Y6dBzw2gKUP>IJ^a7=OU'
    'mM930XWa`d_8ibj-qm^9mhE+Glx&mQX`?u*T5+tM3WQ$b{WjiWgS%7Ow&n;Du4VDqb+fs}m5P9QX{RP-P#xR-Q68(Y<(mZ(s=?aJ'
    '2H;_!DACqZ417~=r{X!}Fi{B9Ms;rrp2I1Bj28f6AbEumxRG@wlO5B%w}@yX1@3wbd9<%E5mm&IbX|zWxAxfNzUwMe^U@Ya0DwF%'
    'H-@7NTBr<()H-mBphfyp9&UwoPUsq(pkDScq<3+xQMw7*S0Lz8DQ47>(b#D0=b1ZBQZ=MkQ!4GIK*Cz44FI}XOkU}Ufr=s24tgVg'
    'PopG)#TD)BwLV_ijxnd?y0tP(a(EwH;}RVucB~*^<;LVXG}?t55N#whmY{_h<B=vnH4<oVDDY)bN;tLvOlBo77};mB+_^a4ny}o@'
    '>*h2G)*oz0%#AS9@vfWEA-ckk#6Z7Fbi!UPM}9+essowBuW=p&@@3VrUCa7ZpdA^H`ghpJ@(s&x^T2jSwgQJ~7)8EpyTt3V{a)$<'
    '(7He=;Eg5M<29jnCfY0dp|LF~zyS@R0>$aC!E1*Xn1>f+2N7loClg<|<bCPXZ$Xl-&}dLFYCACtfgqJ<(1jKOqY2IE^g9bY0k%6V'
    'qmaK%L)p-lq!Wt-G5O;2ICrFXY<J3P#b2?=Ae8@FWNm4Z9U^QEWhhr}VxRz6GJ4_Qm$#q3pI^$>nOO>FB-<1GG0H(93~Px1RNxKl'
    '2Y^K@uc4Em%AhY?jvTlMvuWjlJGs*^aU}kRlt*c!cAWtPjdC8xFIZfz4xcGoZWfE_ceorsw)ICAznJ=K;*aU>z<KOpk=$V-MHBR6'
    'N}dvBbs+wdG=!$#9`3>FfI*I28Rb+SZZ|^<p_I-T`}L}*9s_1kzFkBIEM_lD6}8TG#h}XEifTYJ#N1WM(L(v|umQ$)>VP)*sQDRt'
    'G!x@=qw|cCR=t_;AXGjkd2AurKmvnpyKPvEnmufK2sJ7lu?9np0hUPzG^C@TNm8T3K_q+eTsP1nm|HNN;pD>%py1_@r2y{q_3{1H'
    'f0YLsDaz(rgZ~AST<9AtUPab1F=?<p8$!nZ@VRc(9wfMuc1G<QU|TFggHy}qIE7nPowIO6>n0-un&pG>zcu}VqRIV}*_NarREoL2'
    '7%Ds*1VUO9r2`36aJg05r##o{svCIRJRist+ctX1%pFM)jFBNFR{P~Dp$?M04hpIq#8p8UN*t%Eh)L`cg+ik)8GCuo&DBA_5@IW0'
    'Ni)<{v7u?yK*<5Obm^Q~8lQ;AGDW4~mKvy|j-mw%jaiKf90v8oNVlw@q`OmUy@|&)#GP&6mq->H<ZW0GxmDW#Vb6omw>2iv4M08v'
    'iWf3aS%Yev$qJH}J$ZzxJ1E<@J9!hOJmW8bmp0&Lc-?$-MTWMvXP&uUbLq_o-nR#NF4_Y9v0U{Tky1)?(hQ(UQniBeW{PV(=@Fo^'
    'NzS4fsjTMU6fdQ-&?w^o!^m1q2Ioy=BgDOz&&CSQq-~3V+%xPijRtZVWGxkY!A(kBv<*uM$_ss9-VPY#C<LPP&C<;uDkMYR(MU^_'
    'JN5~>T7rn=YalNiCJRlj?k4nq?Q7CG!=EQfP1PkWTkb5}l#+)Sq~P@sy#dHB#L9GWZHSSC8tE14AgDd!7}CZRum`~JeZQb1qqH4V'
    'R+`ljFdzjhSI~3_fv(~O0zV2gCHF9zb;!dkWWd=W>^=I!iG=T>9BB7Ca$wEpImn!Xh28f6rtZX23CgsB9GULaK)>z5X1Ki%kG1PL'
    '4ahHgx8d{)80`2eKrcbmnYAc26{H5{O6;q>m%*XSi7f3O2duazTEXZKBdF?SWu`>;v1zP^60vFA-Q)#U#@&Et$XW2xh2_yPRQfQd'
    'OEGv^6FXH8lU{@Q1Js(fXHa6vWPv3qAVa`PWi^;wc<jljxKW^_C|*UA5RaW;w_H07`7nP%_(vXX9W}NAeeWnJ6plvhH|?TJjaf*8'
    '4;{bI8lYcTlX|tifCgWXpwPfoM)oGBOIXsgb0N$hCnF^XGJsdYaRWCM2%j`c#2G#nUky_eY`Y+ua7CG!_`SS(jAa#2Fc73Un#%&R'
    'fy@;_F!aSN;igoKO*xm6W+L`-#<3wz;@q{wwj4r_ySrAhZKJ(laT>I>P#_`72u;v80u0oN*GIl*T}q--ANPV&*@zGKvEFCI<69L5'
    '--!YjnoBmH^3vKqpbL6)SC?9)(uab&;An~1z?Oqtz5L=4Ql3-0(%*sxbJIjR&B9{<vg<@{6m&R4<o&zZlqVm!hC8SyU>?kY{=Q>Q'
    'Erukyh`b%jtR!}(@&}wQsDcbFCNI{uRX!#4WG`1i450zDMnm#U6(~4a+z^P9ohnBPlU{B%`HBKKh;$1j)ZoO`_Rt;ZNuD6MLpYj@'
    'K+ES4ZJk*e-J)Z*nXeMu5Vdu1ZimE+VBJDb7>hCRUC78c0~6N+oZ#eNHHO{nrmVd+a5e#mv$Hu+Qb-d45u{SW<}K=*9fVy0)`qa+'
    'C~K!0Ve#u8Nw=%ZtQHMGVJ-t3fP1$v3E2#;h)4GeFLd1@qx##7l$jeRkKEc3foe!v#b|^#U72w>!{A01nL|)Vu2z9_7u*$Xy0(F+'
    'Auk2--6@IzB)5qaH4>CVD-t|5eP5f|y|!OR|3RGZ7S=?F$sbsy?@~<cSJ_2jeNqud6qb%k6Ly=>EPz%pHh;E9s6{FpkpRN?MaN3f'
    'V$ru<XdYF-otiwwz3vHeZ5ROXZ52dFDpK3wPh2DI4ciI5ZGo2RST@G2@GGF?6#-0|fwQdK6<)$v1dsyCzA*aQ#gK9pa+)u7P2+Ok'
    'ou@>lm);W*<53nnfP7(a=9T5Hh+u|qS+`;Uf?Enh9hcG**+>;rQ(dy;u~k+Z48{~azE89;^?i*qYYnn|<&aBQq9*CUK(<=F+g+jb'
    '<Z3s+0pdhOw5O61^79m6A0k+wa0HN_Mm%>BobZdYxmhbFejX+p+%kyvpfgj7?m*d1T}LM7{x(qdx>+|_A(#ZSW)GC>+lePsU-8BR'
    '@yKrh1c8u(yeF=3)7)DWN<pEop{$HW72~)#tf56Ng9J&8`+(?%6%q`V!0%}@4saRh{a&6T6`V*eK8wSXhv%*zSu>vvuMSu>P}Z^B'
    'si^;9(K)j$6X!-;nl6J1puT|*9=Jb1Xvb$$jvNEW_$ADw!?)jgk6v!J$727}0ZE)OuwMJD&B?`C53|uJ<9oR86E3&js|qzjH!*Z7'
    'c}O(s+p|Mb7nVX@Ldna+i$Yw-CcWS2nxAcNg?nu#xsdTShuy(;I^k;?wAc|kARs8tAyQG>Y+3_nEtK!hX_F;h-`&3em3;0-Qg+?|'
    '8YD@4s9{)rV3Hj^e3S=gR9o$;H``zxZ`H-X0pMec+6>B(Y?Xy13t=%-TCNZbWV}Enz*GPD+hlz*wM@E5((_?7+^fjH?H9)bRf*~F'
    '-i`*S^x#L^c4itlWj%VOz};lY6+=j$0w4tyJ54tVwb4Ww9Q0rH1(IRg0}vo7Do?swG;H?Z{cCgJSfBDTQ}z~klrc}08?mzEfti&p'
    'fwPAhj6x2m`3urz^4yn3tRdNY?5}SXsWU)Wo0Qeq<cFvxof=?(d%?Vdh$tRd$Q=%oE+||*UZEXRPML(leUne5qhp(efo8UcAJQlG'
    '$t|hshZu3dt{x_m9|Rbf_d46>kYBtpJHtsv3kmm{Dc6v`SCo))(QmqXdDu&?!53pE0Okcp7Xyo-5%?a*W73*`6Vn5Y1_(T9FQ^`E'
    'O02-cQ?4&8P6<1Cho!yDZD?%PSZ+9D2lfTwwg$Cvf>?@GR3u2L!py~HgTYV9fTpCZN5kn79YX|My*<9~4E}=mS;OtRZJF8ZBqfEc'
    'UV!YG7UrUxvZQimER-s<y}+?qEUVCTQ3`{$8CAy6xk?9I$*L5Qq3UQvTSz0Ow=CwK3Tn4`2>gSgCu^&iwx+pt!c(kNbnmzoPEjl;'
    '#sE|cP-sK}(G*O0$HZF%g4hC)W^FG}lqs(y=6}?0;}x1~pqe$Lhg}?Tn(RAiSIAeE$o?TsxXj^=h*zKUR|vfv`B;XM95d$MVdc~b'
    '@FS&?fh@kMH?%NJDs9TKj>%?jRA`y92jU;mGvHMm1dYk4fq5+QX0mrC5EG`ulrmC~{=CGl{UST?E)`l^I?v#Bp_(XUhXaN>P3)r5'
    '2no!HXHl3~fe1;{7qqTJbSzX6fpTCn&fUfll3sHosk4|JttEqToCBh4)%xJ*<gPHJmR7w(Jz)UKV0wmams=*(y}dw^klV3_v0an!'
    'hRhn-ec4z0K#GSLSLpGGZd}xt2<GIIpjM607PfpClp>GNv2-l|i7Oovoc<!G%(kt2ql}yR<Dv4+oqb5eCdLDu1{*?q1YqD`LldtJ'
    'Vd9Waw`+pcr3Q@6pXyW~dZW4$!t)rA_7B20C257@kgH~G9(}tR!H82{2NShkaCpedVJN?W&2b2@5Oq$(bcszk_$5T1au<Q;r#LtH'
    'Rf$n6L%}7DpUi0<T!JwaouSr6X4f4n!WetcD_yLwS$1t2d;V(bd0MTZLdBr_f#nSmX{cYjJUj5`Y${|6@+C4qHg)`if5je>+QpWI'
    'A@r_|F?WMX{&yIoRUTkenrxq2(-2^foZnNm{n$nsLvc|mlT<JPGfj30Ym{WN4d0d#`*{#0RJTkTW{HdjX9zu+`wdhfVkmx0*v(c0'
    ')%`L;$69P=L%tpK+?^I)Hm3yt$eGZk15>64iRzs;4HpnaXE51rxah-+b*>2ZZ$X(R5^Qd(wx^DBXm%NxUkO+aNu!&*PzvL6=Dk6v'
    '<K&Gh97QgTgYSvNfV}sA@J--v&|C-=u}E42KQ*0&h@dD%pxVh9lyE&)Dk#!}?T^_Q`^#&H!GaXbQ14_0Iphz{2@@p3h+Bn+`z$(|'
    '!Okg$>)rr(3XpYY*C7qLiN`P>51B1R955y4#J0y^j3-BtTGZg^zSixT5CaV@^wzE!&JGZCdHJd~ci{*OOY2`kq-`7YmB$>u5qSt('
    's|qL==bbB4XDW+@h6I~+%mQ=CVrVz?X6Ii~08ZOpqi|b?a>1$1c0TXiRH2~-2D;*7%d;S))$KnT5-0<~;IshQKy{6VpJOZD=3EbN'
    'Qy;IOM<XJb!yf2fVIm(QMb}+rxqKz4`-+%Fbd%cV6yh>>(n%;i0E=h1<?=ELY`>6jPR6{*rmGmjkU1qOl7M84yBQQ?DXa1V*)qGr'
    'lF}7VjVwXP(9{k->9-i_mF;%LL0;Y`5CZFp&qXXlTQasswc1H?L2b6h`Pd*<IXPgt$z)QbkA_uO9kPZaEmQ*;M!iGK7mi#BuTWP<'
    '*FFpxTzW8$Tthf+xN=&%nTuY-Z6$!Zn@P-Fxdlv^DsS|XfC~$F%)|x)#R?NEni4~{M-V{`OdRk5;36x07KhS5G_Anj0jFDUqJJ7O'
    '3hP&fftX2)ZGAGVkf;o#M0`<dlk5eWB(rVlRMPxUr!wt;I@eB%PRsCu)N7zHtR=}OTadzWR>9b5ZZqh<!xmKb!Y*OlW%$WdP5J1C'
    'e*f7D=0%?HrWDByx^Ik#Nhw(i6_ZE74B3NMcdYc78UFLH{{Ev*6T7BhAf6sJK?bC#2@}rjgYw#i|17z${rZ7_|Es_M=#uS=ZS%+y'
    '8_0Ut#z+u65bu4t)L%dEKmX>JKPe{`ELRO%z`;=rtAy$8mwWX4N`L>jU;feGe-!WWsc{)EXaQ~fa({o{rQetO`v?B#pZ)zusl?T>'
    'FI=r~kH+qnE)Cd;gp+&)py%b;u;!Ny9dD3ak_uS+Rdk`i`JnPb$_BD>&m=>}A=$QLuOi#6vIP)?)D_*6jdi_Sf>$<4pmcl3R)B|4'
    'Q6_|PphP$~#L6ysrTq5Rl@HSVXjne7c5rYjxHYgl62``w^Rn(4ZnlVQ0X&&N4oKi|tAeeCMa{xDT3np+#b`vP`|V-QjTj4G14l<j'
    'kbQ5EzR-29FFGw~ERzv{$WVx-9HVtkS8j4qpi5Cn^tzReMFHO6a5AT2%MjXhHs0((5Zgn(s$cY2_;*#n%?3OQ249iV!+&D-Jl7`0'
    '{X3Dj4sk99HN-aJmaiq=r;1wyxj-A;>`PO{R+Hx<g7>z4ez;w0C@SICW<sRNS9KhJ9o!jT540K(TwNlQNWlv9lJgHR5`w17<n&=6'
    'P68>If{FQy6+H5<SYOsrRddWig#rW(DiOtV#}Toua5lmd`r2GlOH|Y%>ST;jbuPMLYX`YB(wBCS@A@*%TRZQ-nNkb|(rk})@^Vy{'
    'qd1}O{UY`Tie`eup038H)A!JTSLIF3oxkIMU|Pfj>iv=aQ#75&7HOyK(7a1b&ZzZE%pQqYid4ePxbUVL1QwuF!|R|ZzZ3fW*UKXL'
    'sq@`MKc^!Za{L~j?aUlm4}CqXreYeIurQa?&q7em(l~Sb9+~>@zg)Bu1T=Py4A0Fp%z-ecHZd2Msp`ONF22ghC{pz^i^V|arr#*k'
    'Q0m<By1zh7=%%}*=+I6{S>fs`tt(7sSZl!V;jd%B=)M~T7zoH7yBmT@=1(9<A2a?9VpA|unF!~ST>*PWP_fj78kGk1rV0^EU*Kmk'
    'M8Wrp?6V4}``KJxr10G4+m##+LCr-*Tbfsk54?!MDi4{b$VF%r7E#y#q)51iC=mJrKx><MV)>j$G1gn0S%X2`J?k*5l2gb%rK_#n'
    'M8qZ`91#?n>p-O*UlLKJ-q}MpwB%TYGggvK8AC4`5>&NHyZCL-#Q}mq(m6Pg{lQO|1a075ipdBEvLwnc%)})S_80ck1e}PSrVL7|'
    '1ZFp`1cUYEh7?(i`A%&tMc^VUVeNzwfw)ZC>euE&Qfh>kxCG(+KdLB!wo_*?uvG`nWc+v5Mge`5y>Wpm)^<Ksd^CBEWZlcmxc=oB'
    'M#KS&!2|@QCf&)l;+vv%6if^8bB*!Ek`{x3UNt)`gsy}_b*j=pkWX@=5RGB;%RxEiVQgENAOeX@?700}K$x;X*d1R_O1v-RCMfS$'
    '{9)uc$fCo37I)QV!iB$7(;QfeTmj(6;RV0(g=q!frKV~WIOG3L5+K5*tZg4RVbj2*X1Y&WkCFze!1FJxAq=kFWab(Kdv1TaMs}?_'
    '9_Y1@2EVKnF=k<<L$XE{vNOH#e3nHskrc26%a_Q#A$EZxX}G1pOd=8x=7YO0C{GnC|CfzL<Y$ZrX4l-orM9D}nz}dQtkKxzTKWBh'
    'ba^}MAuIS^!;+p>aM>=3DsG4UOQG1M_U<OL+<>#fYvhA_gXuLz7CBcwUl?2_9h@9};l!ANexQtI&8;$jDg>tfJ3<~NjTW&e_9&<5'
    'FqkjGH!TpZ;b2dTdp5)j<wOduxWiS#ESF0T>?{o>&=T9QX1ra&A%RV8iq%C|z>{=Jhl!Vq7sc^)rDigz?Z3BBXy%ju#0brx%0Vp{'
    'ydNAMt_U<*i*i$)AL)Xh1HUq-?ZkrAa1z)Pb;wGEsiv87;Nq)$6=px`L}BA8%Ck##D;`kYMN4R|{$+0p*-dP-_q?%Hb{Z4^33NXL'
    '0)yPZU&xPkya$?Sk4tP*nxifze3z066))VG`IwddHrmC=7Ni#kF7Rk2l)?HURLkUP!xXxI>D#di3G7R-Yow~h-g&Z%iitrz!VG?q'
    'NAa>UjHgzs;&_*W9iRq+@yhBAQfZ~@#RNkVli#9z#tv&|JvdGs=$2dtucd6m3=|k`eKDJsZ9NRUrCDbTr3nr>0ikPw28SQTeBn-G'
    'ByBd4Acny{mZ?e=!we>fYqYW*qs1@!=Z?@{V#>eZSTbe6<EN7RAZ8%W+zl$%SEn$hH$yQ9B(AV0QK)SSxxyuo`uDldm6zQTK+Y<r'
    '0aPrPPP@za31NhCG*NCg>G%G*G+e}ROTmiUa6DqI4>7m{c8mfO=a~OW7tc)tm{mR}CeLi+6>9!)JFoC2spwvIGw>mArdUo<+`ayC'
    ')@NFdn~rR)6*kbm#2`L+0+Y6aZ9<89aFRO=Y{GhR8ZYxp5CSyX#gWCjVK@Zl8hfY)WZgFA+Tw*92ukqj3X);DuWdW%4Wa<_5=29C'
    'HLLOOb{_HreuM?JlIT#KD<!4PIIkoA{eEdI1yP9Y$-#wD*{kk`Ow^zidB_Ojl_-_lBM{zTx`3R+u2ekzR6;q47sL=jC;MmRheE>U'
    '_Uwf$%Yhp}=vbxYE3<ZK-T(cznDt+Yp@a~kj4Qawk$VgI1(14;FXziX$8lz+s-edtkqbxN>`>+-+z<kq#a{+CUHQ)aVPd7`4?4({'
    'Rg{d9`nCrx{R<AHEfxD(x$I4)ANzKvV4LhkHXoS8fAN{^G7Ghx<X54vWQmV3V0VAH2jlsnD!{(<XB>h;2=IWAQ`=O)Jl-nVSEE!$'
    'gEIc^%NOG0VI#;I0qd7Ree}&i1fr#i(4zU>_0q19!@|;&k)y2ET}{8NetFvS6+WN;PuI*3*OWReH<cm3DzAjt0^Cq{(0Y#XWwXSH'
    'FQ_S)WE&8nd_!_uR;O9WldFmXD)%hd^&-Uf>6YJg-L*6z*@R{Wts4SJ`|6dkwN%WD@Ph4fZ#Mb9hTM?qjJ-uP`|=)9wLy@F93C+A'
    'hBk(|W0>W821T*VFU<|ko(<_s4~5S*Jp*>{hPj^TE(6l!U)F?%_-AYrXkDoe;KxXvhunM6utJrO`ej^!{I2s#2ntl#FwI-rj@91v'
    '8sX&e#a#k}v8HG$QZ#4T`GFHIVV|fINXp2*G5VLS9M{jqGK7PP39?SVCgVEZJ*;G1&`<jUASu&*3KVLlU^%GyJA!sX8lu)%e<DSP'
    '>a-~w`JWIhm%*K!q_W*p#cBVZWup|pc3jd0l{`vABeMo}`k<a^prK!_(vT@l2D($0GHpj{pd`)5(?PgMgM#APelIebMq$>FG#i^M'
    'Lc#*~l>(qLwQA&W*k8^DoziQPd<>bPDbj_!lt+!CfU_p;;g`u9e`tYv@rNd4X8urP^4-M5CEkR(54zazKSZ?)ryPuyQmQx1%K#YQ'
    '$o2=eG+rsTrdrKMU6tt)fP?I=m5E?;H3bH*2uFhWvLz$x;17YB$*@iFk@+hv@f2qc6<xV!8ioR2Ry?wMfKmuQbCimlpNmxiX4$5{'
    'AvCs><lSi|Odd`Ef0~GcSXL#<W3N3}jpGX#Pt}P*cWg$u!>`v23ukGf3@5%};x^Ws(qJpD*CWVAKo&F|c8TNTHQ}u;VL1{#XbF7r'
    'l5wf9aJF@*@202;_5Sc^vOxD80L0t`dF#J!8*};9x7VhO_xOW5&=AZ;9e_#5`9hGaGpfv8$noVGGBHB>P!%{r&FM`tf^F0Lg+79>'
    '(-MJ0jtj-Mh0Bublbp1p2``BCWhYbL*a1Z#$B$Na^mcYmA<se?-ec$+*uKB)t~j$<C~{eh(`}-mh5K#W{0yVLs+0>{)7qD(PK3>p'
    '45pSx0&i2~RqQY=s0@S`$Sd|=RbBJorYy#sE^B&CHwlT7Tv@fCy)dg?9yFbL$Vdxxd0uz1PRcJ$6`&zB3#z0qgcF$IrELK^dRN43'
    'OdD_r1}|O8r<4*XzRbfB0^vW8Ev(R|A{(qO(Tp=4h1)9d@^t`FtDER87fxl78ayTc<qv7#n{zMre{By4-M1$*;wXt;5nq*%|H{@-'
    'RHpv*Um){R<}U9H!YCG{59N#jMJ%E2S#-P-Ql(>+!koQDQP;{uVZ+4oPO>jf5HDX;FQz$K`z64MqOcYUP{1XGTBdL~_^9(s!183~'
    '3#974aW%0wv3P+^UKZ(iD1~3hyEK3*QzW%bg=@`Uo>fGlj7@G->O%W+Wr4g}j3<W2$ZQzD51-r07x9HStmO+eXewPjcG@5#3|bdO'
    '@PRIBPo)?ebG{T{CMmdYDO<K(vU7thKOXMcs8$ixf7w`3(nD%PU#-wtoatr43h$-JDxCE4`L)!>5F$jRl9YIDwf%-j7B1cMV;duk'
    'S5jl3=T&))GKl4U;b_#}fCFX6?gV+>*V&5PtRIP0=*!q46+}K2<sAwzwy$d6#YSNd!~F&CW^mjFBoY3V2S=p%3vGer{!yvCWj5hD'
    '9T>Qg3@Ud8_!(pDXu;p;205gU2o5JsLusBcTK+%k-ZsgNUD=YI${xWn1imr<i8b6KS*qRE?{(YJEs^?h?doiaRb(;~2t=&qYm%>n'
    '*uuqj#)o!-D#O*)5{xMW8nn=@9jlXt{^V0Gz-7O_PayP9m_{RV-u53ppdnN5)|>YzdY`ZLnQgz|c5BL+$-3K46~e%Ue~z`Ypbnnt'
    '_I;axYu-#Npl>0g8J>dOgTqC^<A5W))qm9zxIfJpfUYO)gpC5LM^1MUG2Pm<e-{-(`#6M?;?Cb>Qn9WU_9nX<gSjWg>Q9&O$q5ys'
    '$^`$AD#kN1Xf|Nkop@{@hl=6&b!&;^(DRBq9RWBiX@flq$+R%;rGWqLe-MDzdVlcw3uzM@@W3kypCU-onef@iw-bQ`Vb=A-h>M0k'
    'RS%wLSq4J1ZO)f}50)3iFf2@iHajGPp{qGCa*>XT=YIVUjCmrJ-}MQ(F5rMjtGV~dbbNKNwK(Gc-mxKv0!F}Nm!8N3w~lUNS>eH<'
    '{uj<u=fCbaih9X{#b`nX9TkKcQ^7%G7!2Jx|8<*EzG*%Nyh2aYcBDbo$Oi&e#ymNH{Moe_ralvC;DIo@$+4KU{0R1OH(j0wzMo^5'
    'p`-1>54a6<#&nNgf)^E82y0|8!LL^pk)BXYP`mT1C>)zC+~6~lY0Z^-V-aV{1O_QeZ%O5}{~FPYQ`+-3o$g<i6_)-9wl|Yb849wC'
    'K~|#!A23`1pNw`s(+dY2lOCvnO$~7mc)@77wJP53uE*`KYlwB=by9TLXb3T5U?h4On<f74r46T;55K;98!Or5Fv>_WttzZcU90#G'
    'h&^L_q3GN#K6Coi;0;M<YRq>uG)wQ^$nGoM{Ua*DfWYrl&(>@tqQK|ix?9Dvb#`j0DrDbn1OQ$CKEDlRf^g)W)LHDO^Toj+_14P&'
    'fUjG^k>Y(1L_-BynZ`G;PIqC)w+2oE9cyx$H8lXBDeIexup$Ay>lPA9%s1K`*dgwpuwc>U5r%;*xqfrZ1^E^C08CZGc(*zhWZ`F_'
    'f%mkC)I;CB`a1;=48ezlm6|9MF?xgw5vRyzzF58uX8a3HvJL6Y;t+nnL=-7J3>={;F)?x*7QYh37j>?GpYe-Gp*+R`8%6;rdWkkM'
    'Q4#d5RvDi;K4=a22V~clbxs3Sz!8EoRW&)1E#00Ul<sdIo=IuO{tg8!Req(COIK1$=QC0%_>mvJ(odi^ybmCaaFNqE;=yZGIXU9y'
    'NsIfJks{K|X{a6G9@^f=zRcW~wUr_YvroLUE%*OV4MvML+4f*UQo~mxEJSS{cof25G1`3BuS(mg*}LH6dKTf?CSx{XOq-rIDxSgY'
    '{X<7XAf^B&m{$s4yP7k_Cx_!j=8z=*@jcBta3?1f3-ja{S`Tf$+><5KWE%VohTA!x*>_MVKIxoQ)N8djV|_3&_>l<$l6*mb!bSgE'
    'Kc6K|RuyV*F?-l-&&mi>gs}i4?>_dSpPj=@2NtV8pchypOku*UmksQexwa^2OnxiR%{naM;TTUTnK^J%CyZa3(G4df9VpNmpZVeK'
    '6Ud*z<J8%^?+Qn*7!g@2Fj&s8m7o62$C7!d3Hc0gK`fO4lONgd1Ozfi!PNMnCJAc(;Cx!)BnI0P{#4RQx%V0F&z5`nBH;X63xHBj'
    '=!@}4sMRnBcn$>liT_aC?iuYvzjQN0u{3oKaXjoy8;5acOCTxtI~nHv8!gkjyo`HJLYp`tU2TYGXo~P_GK(1uu2%Wf4-TsX&tC=7'
    '(ap|dC|ed2sq7WE$lINN^1Xw@0@=H~(m}E>V!MfPXu(&lE8{3v{m`nW%E&Cr7f2yO68CV(E93<}Rew?;=KOxfg(48H@Wn9!S@9^b'
    'Inpe%U1!=jKt+G%X9p+|ZeS<7bw&{rgl;+nsa+$R?)~2pDDp7-lxm5M@r03e%ON!#0#2^#PFMTT6`poPXF3quoyciy74l}BP>_f5'
    'z|+@v(8JyK2@N*btrM|C1qj3mt&1q$!}ylb+G^i#7}_M@0lj3rer3B-tFXh*@xLpz|M=;36L3PJZV-^z5-C4nn)ZwU#@tF1a`*3T'
    '3s(qPR`3qTY&n>wJK)TIkuU8Kp^twPeha}8$H1O5jRRkZ>45xKvBgHDuebVJ7_@SukSRSh0*0p-zHYRv2oaVklHYsc4i@4?{ND1<'
    'xxe^>9-wV{H0b4T>Wl-&4v{FFN$Ij?X>P>p3yCSL#WcIT@sb4cMUZpE4b1KF2Ax}`z2i}kT7V7w+elWnZJ47S`p@B9Lqw!yo~?iA'
    'Q`o!yZJb(}ISN*80(%1zfA6%RCA}3wv{9b6Hxd)?T0pdNJOtlb>O-{fj7wxWN@{&RvwT;p1GL~!C_6!F3lUVr8<(1CT+H0=gYm1@'
    'Vl0P0`CdR^Pp=H}zMTOKxm6HSzT0=-fKEkMo%*lAU2vd03I>I5QB{xAOa6ZIBl{wnnHKRqFEZTa@n?;MbJCbtL!55k`btRvK{AL2'
    '9f}w~w<9ql2D#HtMK}CLv7Bu#@)a<APMm|x2%DIQMGUMu;`<%eZz66G?i<nbHt-sRIcHGDkke@}x`aS1^KDj$VGf-XMHneEY%L9d'
    '8Ez0q$%gHHw?6#wi;?pBP@y3u7esbhj(qT?AZ_R7iEnqt#(mC+4^d@=X$x3Kk?WVpSwl~F=x26Oj(NkI8j<mMq#fM5ADMec)rax%'
    '#QyGvEy?v?^5ROiOK#oOToew?R46PmE<aPjz#$S#N;niD^~gq8#}I$AklK<5@7n*p+dp|w>dD%of=x*#GSpYF=1mpgRI%{=_#NIS'
    'JA;!4a5!fF1xE@+xdtAdg}V#4fBCC!y)q1-SZ2|Vo<WRCE=eCxGsq7kD{{&qcKg^R5G2p3bRrtyspomQ2bI3QmzLEWqf8?Me7Cmv'
    '{X(;Di{WEGE|@|4_YwvBvgjp5qLX+ge*bzx{|;bZ?zqvv@8SVzO5iM}=fd{8QCv*6BvRPnrEzGi{}Ng-De5K+&YDJU*#G;yrv2Oz'
    'Oxnu+A1HuLZCopp4Z)DG_%06PtRd{1Sg|>(wEgc+_28U7pNhV&oGsN|lbzog-q5!*!S$Qz*W8VkSb5rnmU{R2s3Lc9Y31nj`TbQr'
    'OjnY&J`mA(|H*FKX1Uzo1)F28AJ)!r*Q^{>ZF*bei1Fnxfx{qG0~sM4!qLC0U|58VN7yAxQxblf7mFZ=P_)Tb{%z8gMu3)?K`jm5'
    'N6a>=U6W1O3I>s0?=<<FjD@C}l6-F|4H&K}<P*^rOV4p1-;Y9EiH4H_H2&b&L$Xn2A78ovKpMpP_q_bGyrw!7sKFFQ?Gz0%&V^WT'
    'Kk<@BF!uTTJU$&Nt=Oq{j9{pev^k7PCA#}<5D65-Tdw-8eii~!#Ae=FlWdtMo}3whaGZ+dYso~bA1Y~}lB_rem2nK<1eH*cd!U5U'
    'O5=u7H0b4={!OOSDkBT&?f~3}k*PT42kUsH8*~hF2L>^|yYfRX6-#>HEx-SB=70bTe`>k-MHvIr>Kmt<&^OGE1@d$JfE+c@abRjk'
    'l*e#WmDRV(^j;9UAv8}N8gwSX9DWAo4?{$&XjKTbV|_=`Li(lWR>S@{77@+`hO2lb%U1C@@wa>zks--G&A1e8`Z;5}MRcEvH)l1L'
    'f{fnwo%^aV9M&Z4P;st_aua2O3~goL4pEKY5XTj3V+9>r2^H){QNnwf4h8V@qSIIKu)edAE4seQa$d1s-=71CnEJo^HejaXLx^ss'
    '!grkrnQ;Y9Z{<SV4+P~G%aQn9WFEJc<fbBcq<zx|e*5>8HUX8Nk8@)?4LAw&$$=*9=6nY|ef|7j|DF_1^VTe2%g)S1hS+0&XXL!z'
    '&##~V>)*3YHol5uPF3ySwYtyS|Mhl${rIncj@xDYPk1T~!RA<vml%I_q-O-;8T~uz5V|tmNf$m(j9FdsaHG2)aP%ehmw2+}?Ej*?'
    'fi}S2$k6xjE6(^ZLK-DJRVp0=K0b}#0#=Yv-wL?CA%SnkT=D;Auk)|j+`^1d^S_xf)-Au7&%BMqt9GC7xdEIuGGfpn<SSwRIYbqh'
    'kGj*u-=R>!FIY`f`|I(AVWZRsyCX6-MCOcxKMQIP`bEbB{44~fz^nGORd_e6TL}ol-039e$OVPq@U$U39cWYrU<KsgNPOb1VE-Vk'
    'A8{c2ll8k0z(K+y&xDJ|8OCi6e?Ie*3P>$;(@c`yi4Co8igy8rT%OAua;Z1iFTfQjt4sMFa>oolzMqI8Xg|s6zb8q&!9CNzNms>C'
    '(ODoM?JyTU#*9o5XZ%wF0T`vuJYrQICoTr}4dMW83Se8Qrve#FMFfT+lz-cqxzvnSUH{}*{r+=Ms%*8KH~E*uYG9z}LRNPbL>YDt'
    '&{<*h0(96|qSB_l-BYMsy1!`Sb>;v^uQkZ_VgQ7`U-o-9>Pky%#xMT&$~kT;9rAoA0>~lh0ObctKk@e(taEn0p9I-kl%Y7YbxbkB'
    'MC5L3c_l$PRuC40vo3AUVs^{q9&AG7^)53qP=4C*`S5d*YjNV6D|vMg<1^co^F13V7oi;U3(6zytW~-ekgMu%#5Qg?3oYpCxCqK$'
    'Gm7{>o}>xFb5-I^5#-7*@cTeOmB=Hk4Q!h68oM!d+rPnM#vuut7(~8CbQDVnxB{zrGX?mVNUhY=Bv<&1&99?Bk-=sfiPcA@o}bcs'
    '74LFMKzt+aYP&Pi=H^BmCIxPD75IIqf#_s$?al+LuQSQ*Fal7;m8`t!Ha+*Dp&Vh_b+NMp6UAoDeJ|8$H48m*W6J!Yqq0U_i*~#k'
    '2LU4)8HP#*`=(6;4olUV;GTGSM~I~ikHn{HymHCd_x#K{ML0X*{V*4e))TZ}@$@GXD4BXBg8yWi5JKx)MbRZCLkl^y4T>aYY-Pdy'
    '4Wo{cFyKgL7RNyvWb}IZ^kxG!16JrLaY$s!JOT210_77!t0s>X2ed<+1oAI78`mVD&HeM_VmPcYH{4SDJ{X%YfgodhI>^Niaqmr%'
    'VvdnAa)0OF<AtcM|2}p)?u2~wC7W`kU8;j~An1OeanXG{fSj^Sb24Lu6;-((JJY>Au+|<%ofwTf=Vtc8#j%P_;v^p%pWcT>8U!Tl'
    'Sqzyi59(KMQsWqlrc;Etq$f!9rk$&AB|*!q(Yc=Zm7?_+<2siP?c_9o0z2D+>`)R;Ui8uouHi|G6!HW>c%Am*V3M8@a|}btiCc=x'
    '|KWJOpqDheRrL*N&Pc^p=A<)w;KYs)@*L^O{qYtOAcPs8M~M<p8e$hkd>8-bc0d?rf0ETUlo4i}5~$GLH)e~&M3$aJ`%b_yeePQ2'
    'b+IsW3+m(HRd9PK@<3bN9d}#)iccias5oVp&p{5<UB9Rhx<um!Xp%%4dS?TF%iRW>+wEPHDduLhpE)(5oDn@GiOh?)OclrTj1Xho'
    'H_O<ZBVMo@Ea6mWSq(Ss&E<Ea#9XE+Ea1iB0J$p}JMH!|z1Nzhe<%D6<d@0BFj!`F1V<JhP%<M5DT65`jE}lI;@L(cl)THAktIxF'
    '@ZE9<#ppRHk-=2%`_Ov}-{TqQ(gE;;y1Gzwb{oxS*m@;a3zELaE)E{_Y5N#r*8O(Cmh*O6Q#kS8ueK9M>?qIURs{JYPvQD3`1aFV'
    'p|Onv^hex`g~^my6qD=T!lEe;%OyvQGF|uJHUxNl|Eh|;YSf%)Y|0Ygz-)mDxp9(r`-B)ap@jB8V|6#1#y7z1vnkT8JnQ&rSxiVA'
    'DBFi2D3|g^hnTxvuSPBq;W4KXw#lS%^&eU%0J;ufE-<u;3@xURG<8_&g`M)n#km)%p{g$#8gQ5<4Y7oI`R^h2e)}{|8G*Je!1>ne'
    'L$6};LIa-@X-8i+y4>mG(O3IQ2ULg!p=pF4$@>hBf7O_vk#8c_{PXn`*1Aev*%gf8GUKpbhzd<!yTO(`_gf_YvH9vnoGxExjLFbo'
    '$^xu<<*UKSaO%dUfxDRAMZ>$txbPlmKdafJ08`YhB{y9Hy3b`A6ACps^UO@gtn`0@P;%ngF~(n0V_^p%&D`%9!!QY(BrrA)=4fZC'
    'z9Q}m@R~+_XGR*Uy%K_3Z`@@9`o&%PKn)%R$eZyt65mS59<zu8w#$Zgq3L57+J|YAt7{+YKaLXu7HMIRR~nuCbMsa`xklCr_aMRP'
    '>OfX+PXz21DCI=Jb2p#AI&mf_Z{yn%BeKXf3(`js4oJ?a$xq!u+qyUhVec~B@vcY4!y{S2lS<>*wHa+<7&<lHP^k}eny}azVXmW~'
    '7fU*W^?PCDlGX>pz%lD&a@-ImgPH#xW_+sdfo+j!MBoy@Nv8LD^5>HZGwB^6v6~Qi+}5~h4>6exQ#NE+wo4oZy-IZ$yrpWrdW+yH'
    '!h8^qE<KZ;)1Q|>BmZ`4K|QrOt_3%o-2JelLkItGUPTcOTWfd%XW0}{1_8kdcW^@IzE!d|#!40jbH+wPt5byEl1;r*LmGcqZTcAK'
    'z*ZW8B{Jc#Q3qY9>d<%6Yv6%}yw{V~@rK@dZ%;>!V;rzrfg`O6|L<H6GN~j1??kpuo;3I<ciZjic_*z*{;JYvoA%v}ZiBFKuZ)48'
    'LmF4WV4@Gty|U@vA<BSsv5S-e`ePKKG#>^!Fs?nnx@Rs{=F#4R(0;=)Q9mU7D!4V!*rG_dO~l_lkR8Vl2elrI;yuYA+w$JlBbSAo'
    '4@s3t81syFRDaTdlQRtV=6-A*StO%;enk8f^_^kcpyOBV-U~J!d2As@u+ftv0ui$|Oizy8td4JxQd1WOVi>Wpg7R5zr`Lg67`+K`'
    'U?>haJQznwQ(;kmBbZs}P2if{k`k*C9Ga2Oo{_v%M9??7vhDu#9jNgJ{R<xL?kJY-KbCqFNR7rL)@&vm_kJSS-KpU)to77)+*nMQ'
    'Hwe419qL*;XT}&ULpgw6D?I{<j$nylCGM-jzArXiiSXiH<LJjohWoJ-C<;ZAy0JRPX;_<=u?mxrTDQ+mzT7?9)M_9Y#%6l@m3@~P'
    '21mEMsS>kJMELzLw8qX@@90Z{_fOBcCGZdtUZ@2ySd=M1o(G7X>0mEl8AiHR?L)R~$bdfzIw(vC8~pWdUf_NH8j7{;DQ0A~fQ5`X'
    'TcR25P|(w2)W$>{?G3A#E_%p*%y}FenM%cxb^D-me&^1lLd%%(`~3#z8e|!qpgEyTip0B$Qhfxr{CmsypXo2k?<`F1f+jybnSWFf'
    '5d^mEP-*W(8-~|0K(s>W;qW}~0T$cB*|vN*e$`_&Fc}ecIpW?=;b(LMZ8>qQFm<_#i!LzTL)4cvY}$F{d>)Sp_=z|-5-5ni!=WS>'
    'kLhP)ZN2~Nek-PxlQ+8Z430O`*lM2h#_^iJXe<stl(t8I@Gi#CU+M1?Zn@%X1*7f7=Eh*F%Iae{z@A2U?XVL#X)`q2XUqHp>Q`g*'
    'Se-b)(jBm!Lhu(6jC2R?xAs}pKC&E@vTASnW&t-gKuWAzVMY*NADaU;LRqA66f**+w0byMkW;9;33$8zJ>2d1%wz!!W64pQ(qieH'
    'f<6qia0_%wWrIvlBv)h$jmJ55AGc*a4JDOrZ(zkjRhm?7-D%+PtLU;Y@bx?fJ8qJD80u$a&#oii8ci`~voUjUwe4&vM|v};t?VH+'
    '+b%$AOPBw6zXl$@m8?9Y9^iWL+!q6@ylICK^3{sh)S_c+Yp{LLHEr~Qph2JcL-aw7Sd(`|l}=dKJw~U@jJ&l5l)|2I0On$ZN5QQ!'
    'jLP2ZS?qf_4q&0r6d_Gptz-3!99hL;40E)PXHZYwnjR6IGGL~`ySdF2Xab^9#$#MCF|DgGALhwy8kx&h;@QGr?kRm{*Ap*Y)+w~}'
    'nocJ>t%EhW1MNeiL}&asbqM<Yk#VrMpvs`ygKpP#7K=K4$B3G0gtq4V4QUlhKTxX3J`DJR(>KNQktM=jcAw2)e%?8-Gd`w#74o|+'
    '`X2b`aj)e_&0z?PI67Qg(a`PKECv&GGI}SSLz(Tk_E|%B7jxR^E_*P)KdI)w<sd>B(gf6j$^D+1;Yp1L#*s?Cq1mBQLb33!9oe$&'
    'kRRBF7yCT7^(a5Im@b>6JWZVCsZ&OF@o41UI|&05ZveJdJKPgA+iCqHnS6GVrTDzD;4?`wGmcj)!p^tfL%mBeYLDCr*hS|3qMDe)'
    'fWmEj3M;08x6EN^%$tNj{Zp!1GDO{PaAjI&m`*IHvFB$Tq|ZPh9F#>A!bG%EI<4-m;4Qf<3<{JpgslYCa!?OMnLT|~ks>fG&CLWH'
    '$`qEt(LQT9IYdU%BLJeN%9(RYIFuUcVQJj^tia_Za4VofKNQCCe{7DIgU*DqZ@Kl%l_<VOv_wTLF`42CXz750LjrskT*A>1^IW9f'
    '$;67yx`|I{=tj@owwcIB9Rhid-F8u;vD*ea;%Bj$IG`~lt28kOecf82B2PxA(B+OmgD#^2olnbTG<C;=s;OecuE+o@n}qIByJE-4'
    '4|aL6;^?A)yUsj;=1)On<)(T66MSW<ecL%!nvEv#9Ga&Go#gcNP0gm;4Dx40b_f4_r<tSJ>?=7p1u$P=<_aS%w7xe#_I;)rwK<7+'
    'Sl9V)9hi$3iZ<s)$VG$VUOO(bDoPsj5E1xJY!*@(ZGRPt0orA1$cwR!GxA)pQ#8ZWnJ@@0Dj~JRi62;lR3Mx2s_>5bhV;vtbGNY3'
    '-cnI#c%z?zc~xrQ&>r49;3OobmVMI4SgmxcWlecB{<y{TPhmsA@!9C&sh!}AoQY&2vlg2FnP7plRZ}oAgh~d$!K`vJ?^=eiQ~kA+'
    'RWx1VXbR7KM~^<30Hc<p`Cb)*aKLXvMLs*(5;y{64NaI$^0Jb(8=c_~!6Vb#PHh?z!{gFV)&r682q5|NMq*KSwrwE+wnuRqhDc(U'
    'F?KPa*0e*VPV9(@WE{LUcq7J8dim)lj_<)-f38Ew!AK1({gXvh3|J=o3G&qLY2v88nSkdMbHkA~V3jrH-fmRqesUfc8bYjWlK~pm'
    'YhALlt<-^tQ8(v{-XMW1+^%AMg<wY}C)93^De@;yVE_SXg>yb1C|(9fjQ;1(n2`+TkYGa1x4dOg+{!^m+`^cc#E4FWo#0{6@>q2W'
    'Wn>zan8O>&sX+n<zm9vH^@~6#YRtjYH1)U`_>6m^bL^yG0!`mK2fd8zagNV)taB1U2-lUFDA?^#-1i$g6QX)CxAFJ^-sr>N8D@Df'
    'rWdUEDQ8t32pg4VR=9uh0KW%qKJc`K;)2%V;Yem9@es^|n;NbS;iL)@kZdA?B7c%268LkPrJzj)sNK#3Orw%c<Dfo4^01h|aTHnc'
    '2CrBkcMsEAu)!qZ?mjn%H+-K<%7HDAe@^4gfdzZ+M&3PMobY>?*_vXRm~F|3L}EkOHxvG3mz}v_FzcraRt!322v#NO&{DSRGZWII'
    'pYPa>I065I=bGCTNz!`9bkWgICk*i3SlVUpg3C~UazFzX#o~g(Xj3pzfA!_a6m$Q6eCvo_L~L^sChK%FRHKKmz#&lT58VsgVeFg2'
    'wyEegCt=QjoX77<R4pGC=OavSh5k@iWzFS674M;zYHhqjD@au^G@2zjX(a1nv*)pQ*>>kH=^S8&c@8ZZTWFw>sTOr+9am2ka^NQ!'
    'rfc3R)(N31H5Px*i2%O(_^?Kx;h%kF?9=M^Y`Wo(4dWZA9FZ5(ieDBrG&9||hPrWZ+=C!ao<p2<1LiVb1-p1PWsGW)Sz79{iHAev'
    '!>&U|DJop7>AU7AS8&L1=D<3sSC?K{g~!f*)adVs%u$ZcB&$=!NRyZ$Y9N@9Fotfn)W|(mh;R(74_4`=*l?toowb7&b$`_qStwNs'
    '_iX`gYaM*eSu>|SO(it?e874;Y&(t&0kt_bq-N3Jzwaj0LQMF#9gf8LA?=bY(!$RYsHP4l5Q#%58K|2JLNa@f3>YS$8Ja)@mEzW5'
    '_!!GG5Lm}9FGRt^y=36vD|OV=4^}Oo8p@<e&&9beloQY4B>WxOBupkSm9wG`2m8`xQ8;60BG+NckTmPdTm{%*-ILT2Y0`1z^GnJy'
    'pvAn{;{a^~)ggxf@y_v)8A#O8aI%Thwnb>C$(1s4m?A#1Wf~lB*nI9I62I|0-=!(=0N<*fN`Gc_-|jN55P5R7n!3{E6tbk~Qz=bg'
    '2|AuP3z`Q^X53J-It1czVB|~lm8SAwLzt3AYi1$ei``Qhd;rlg`eY1;9uRfTp2iF#yrn)v`vmo5=%iXt4$7cZ28bDK-dyy_`YgBy'
    'p*3}ItL6%UTdK-^+Ww)ol31~WJjRHEf;btvI1zUPY(71DB+1;)JeQkX3i^Ayr<R<Oz|_E+f;v*uG=B1gB9`Ul9aWgYS9F$mdC$Nb'
    'X7DrxaFl8tWQUEs)v+(_JQys@JEK2mU9O1Jr$0VkV`L4XZe!pRySxsG*WE99QLzry>n**ZN`NU57O{F%wfBUxvOPBFLLdoq@j9*o'
    '_tjWQ`2&vHRK?L~`mm^1s)<EF9Bb8v(K4kFa<^0|O+!r^>?LOn$Aif<Wcu$3Di|h&0VnLN0gctM@pam@tB6AmYK)2v@7AYUEGaL@'
    'G?DW_s!lLe-?X@OPzQXmv(3nR^5iYZ)m+n(k;b)CGBM+P)`S_OOg^A(<uL<MiVT|jnXCO#kq-Nb1haV@_Fc&GQHI|bp)4}v8m`%M'
    '&u=iITQ0YOHt_!UnwcM)wKx!7Z9zXM+)A;B5&AHV@i4D2vfwQXz1J}M^wuy8u`>;bI+su_41_!Wd|}?$3M=iRrYV=bwKBFVIBV{`'
    '!@(=t4Un3sv6$9(*;7M#F>x_P3n1ux_Vvv|YD1U+oqMw(0T3iV-Tv7ucTYcd0AJY3csQmk<*d!3z@84RvO?!?JMG9M)rBQ$IcnXa'
    'XyYBozd+AjygzCOYk9B9IgvR5Q1CZB&xLOVQy|34G9HV~r-J<c&kA<rP>84+;+ch^qEdhz{HSqMi5ZGAPlBVM4Be80W-&UcHZ>%('
    'RoPTQ%s)8!P@guE#5mRjEoFxI_G@Ty(l<9osm1=R8CFwYL~p;*0}q_T6+_1XLJ=%KM8xojIIDF$Q@DscKs2DZN9gUtahOjR6AuJb'
    'l)GAhc0g=9GwWDJDAk*VEytbxX>JZi20Mq2n)mQ?G)u^t5b9}|h%GRTN`Xp1ok#+AIk#}^Zz~ymoi;Dzv`Xo=RQY3^B1jpj7v*U|'
    'G(@v*kNR1HpJs0rViDq}9G6tSZc6o_h1#SC&<_h1gJ8m5bW>8M6lI`F9G62lrXrjUPNpI*Nhh5JhTCB<cyh=`+XRx?*dOMRX@T7i'
    'tBZ5Y&LE-!6P5yHkxd~gQ*-OYZ*%8Y9eAeh$Bc|uH*zrG8Fa=xh&~SvH-N!n>wsFa-;fNv+0iCV13I!F<%8+PdwQ^Is^nsv3&nES'
    'rblVnb-iecl-5p*6h@(*nBuh|N^Juty#Efg7svi8MfN1GGDw5I*S@D5ozoHFzw2qkebgnvHIY#bP$&n(LvyDNYSdtERj~&fL#Aq5'
    'q-71Nkz3l)8(q1W9^vltD%E-2Ysgrm>G7RjhnO-B+tX2O|645OM_dr{qYeht%As|_ht1TEW&zeH{Q2ytu(v?!t<j0ih1@M~Dgvq)'
    'QD<kTNT5DA&5zVZ?Q}GRI%ZQI8X})k7CrWdT@QBK3MYmRl3v)r<kdaaC=2tcKxy4kc_)>9OZlQomG_0X;WV2zmGQCd^aBmccn1~5'
    'JDp~<TskTnj$6BaSUZ{iE={hi$GRW~O3~xqg5)`2BhF9VQD}HE+zfC;jof)_aSkCzC@$G1j$1$-EM@f57Serv*yuR8l)W9>r(F&B'
    'w#Z^Ct=4AP36omI3Xt=5k!52*pJIjkr>tVDoOM=|VuX50#_++ly$})EQseyMrw2T4j=b@e&xC8w4@(%Y3~lROO955k7J=KtQnA=W'
    '6pnW3h*Q1PPI|39F71~iGr4gkRe_dNy<i?V7IKh9i64x$pa#gKHw963m3eR<w!hHiCjJO_U`a?|*cHva#Mo<5KNIZ|zBWWmo?@A;'
    '4|it%)X+5_O5P*DoZdM@23mv~JeD?_oue%90Ng3<!<uHbaQ{zsi(7Hs3cc6lzto`?rEm<VBqd;-Dn<Tg-W3>4*Akr+vY}XYfvZ*V'
    'N3<1|A}FyxIb9i!AfT+^X(Re$(y6>x6xJ$J_A&m05yh5oRo{b}91aw=34^Ifx8FX5(`=(RAd#4cR}col6uil`Y_2_ic62DRp78AA'
    'vt^J{Iw5&z)Gg(hdhI*}PII{usa!P?66e4=jLi~@H)#Naw3%h9tQw*tRIMguR-6VXYdh9>IC~QX&y*&V_bgU+T~Or>v2?-yWNW|{'
    'nS`=?q0n#+WnmtTO_8JT10VGeAOR`7?)~J=_NuzGGOwE!G5ee>V%8EHL;Ee#KT0M$8a^6*hm2#&f#V#yXQK5+Z6Fm{`ot}yC*=Xn'
    'ivV|?p%w9l&Jw@5@ePK`bR-mx&?mIYQz=#=vBHb)^_mVNn@xifNiwCb<^G6?jJ~D^z7#WeG|faJuQ$bwQ4IKP?w!rd#9ad6^cL~L'
    '@P*qHr5>}AfQUQzd2}Y&Z11hVq|+2?k3ZjvL*68V$!KY2YX>$av0i}9HB-_toUo6l1<tK~K=W5uy+R-q-nuhep2}!Bto>Y*-5Bg{'
    'G0_Tsnt|`61ydMoZaBtIT-r!4&&=C(G(k9_V@7p4jbn$7h7H&84yGGVBhxY={U)6Om|%sO4htV$s;6kwcxZir)+_yyA<aKF@R_-='
    '*Kw^(gj*adotQJ&R20QSG)$A6*#}qiNu7CLwUsdj#S;Uy$<(Q{P$8k|(=YZYhN<ffSoyoal9~X^uW7FuIb-VPK;Ycd!h^eA$1Pa_'
    'Or1L#GDNS8SXq48HI7Q3Z76W{hMJV#xE?Jj^n(`NxITzWJVm%IV5{trlDrPt7kyN3BZG35;t#E&<4KGfZxF#UHdPiz63m(Q9+)xq'
    '1eI(Wm;-$oh7B`qz^CdVVyd!*nap28MDwzc9a=vzyweJ~l29=O>1YM~uvP}G?d`X|9StMBU?SjHvXgg(Oz_}I8R?8{`fy&6NCPn1'
    'dc`MAGwD#I&>n`g)^+9xvwn1-1o!RsVJDE_&Q{)s=AOE3WteKGt;xN8h@BKahF42k+)k<U<T(foVj|<4G6YX(pa${q()ycK(Zj+x'
    'x3s`7A5p4$1<^DL8v$#s@hbHg`z?Bm9AK8f33v;`p?7QhA(CMxWV*4i39~>Rqy`=`fj!bpp%sY)rF8Wr_ow7n%h&Zo+#T&2;`Avw'
    '&(sr&K*yZ9Rg?I`1w^gS5SM|?vKC!AhT&Y83Dz?_$B@t%Cu2>9gUNwbeP%0}pJ&uo@p^(R8tF+z(SA$%O@<p|R7^XIQ?%_QFQkz('
    '<ol5ilGc*hQm1_@vq+&So{5g?o!MBYKVcLUx;7eg$KsBS^eCLu!-akx`vF}nTkI=coQ1}Yl!T5@XgQ}^o=llA{Klr~NXqW(0!xLD'
    'v^^`CUR;U|p7Lh);p8?cDgG@Q8nBd{p2i6x2dqbFfM+-VtY@`cRABN`Sog6=Ec~4|Z<<<pWJBGsA5K+M*$&nUYJS!X#iyy=?e5#g'
    'wnso&8Yw<xvOruNwG66e(WRj&P;NRgE?tFI`@wMEfC=534So1KbaFSX6g+xs_=I({AqDPGi<mRUw&0A(Aj*<R9~d-b8f_g9y*d%F'
    'Dj{2vTrpmcc6rdg=_lKzD{a}gUN&5A2~Zl@8tq=u$o&P63IaQrF5`QE$e$}ku)W}nIBKQS5L3B)Oo2K}nAk^3TDIE*f&k0*VN!BG'
    'B#Dhgdn1$#$||V(i{mG)qG?CGarK67INj)|)#|AqM&{M+#){|rhUXz=W8{ghpk?COuh7joe9ocsmrJq!9(r8I8-7o87@HTJBnStg'
    '2?Ui7sJKPF2WT7rz=4U5&{e8D`w_i~Vdjb6t#y)Ld2gkagiSg>1*AM-0<*Ii(FYr@N+32rP47eJX-Yhrk`ATNl$ELMH5U|)be=QN'
    'EoQJgp4s5v1^Y{*S9LVaVvNQOeou60stvXorpwqJ#Q4DK-HDiJg63quILf~p{qmH~@_A;CShTt!LuDP^?IG)vp&JstZ*4K>8~nvI'
    's_sn!4MnKAaDJ}C-ChxjF?+<v6U=~`Q)%v;Djr=f++@0cSWOUFamHZft&wRtJ*8C(iLG<oHje2kMEussIqu?AQYFxKV{GuQGk_yZ'
    '8=2?KKH|XeFY#~52o40k)-9oR!jf_d70!6tN=2BR+iJarPkJPzp$Ui>P-q}9NMISnQfL}rOvF)kdW+ypnOW!%2)$C7+-P%C)N_|4'
    'k8v~ud=`++Ojzfn&t|f1hnJeUT6Fm6O9+IRse)ROnrm~R$k{jww7HY%?!e})j4>yA+d#xV(@x0uFr4Y|L9?Sq(afS&!MN%2oz~hw'
    'L2(vV2p++Jv&(pck#G<wM~g>9aK9w{zPth4T(pp?ElLn$0Rm4+TyzEsgv2RuYbGmDN8Y2$Ovh(y$`5I;@ns!Y##(D>goIi$aHRFw'
    'kW^3^Xh}PHKfU)uq2@|<yxeGCrhk(8v(p}m8Be4m2TTSwMTCS=k_EJU0Jpjo!OjfHurrpK#E6dQ8R-$&Ps|48YZ&y%@B58-22qRT'
    'g8=dwjc>E;Cvz>RMw%Z6YQ=5_J++ZAV~<JlyuZB`cF4lg6V!_X3AfX|L9UM;x7HJx8UIlg4l=5<%zyi}zh26*8(4`c#wY8+A(=`Q'
    '=ys6IWK;<0uebZ_SAKgT4Ic%_j+&4;vQB0lwOa;me0|EVcl!E$zg|U~gW3Kv{u?aZ)_S+TU!2zm{Cb<uFZ=Z(5nlQ}P&3YN*Ynp!'
    '`TA^M@A2z*y<SEiuMbTj*`tO}1%{%t9Rn}aM2O3X^A`3a=FRP8#rTg?klGSs%VD-EYU5b=&#Gx1SyJ^sujZi73@V+69%M2$8)Q4T'
    'UVwmxHwPpLV3G27r6dMI+r}mwRJHex&J$Z;nBr7$$VXIBieO%e>3PDRXh|L`F=5QWq#Fz*9#AwBCiieVP%UuO+vxs2?FwLzDLl>G'
    '^3i~?LFR9yG|({tK7=Hua0BpcKf*eE_5&F!<I=TyauS0>H|JtKQH_EC5FvVly&k=l=?@d)8azCYzsIpdH3x`?F{5aB<$Z)$Y~E9G'
    ';tyilE%M&MT_qNCq2e13hmIjp3{$E}#Lok7z#kD<Kz7DZcTPidS{pGL&xEK<8Dt;6oAU#~X7UN?{GKf>YPlwxjt$y`1zi{6oPp4k'
    'wg#YQ5+ExlvO<h1lxATbMR(i748wLk8KWtFxYf;r;-dM_8b|dpk>L?)BsR=c8l_f7d8MMA|FVCx?hG=~Suz;6pw!{&F#|-zc#K>c'
    '8Tde+Qb*>Lnn6}<r1&TDEQ>zDAVNGsGlP$USpvxC(jIvZVh)EdGNchap)z-Fikk%|Egx)=$r!HS4;<APF9^K2#Yc9@POA4wK+1q4'
    'G-};MB@f8`nF&{tw2|;JG}K!KpzkPRzcq9^4grqH_smq%-xD*i7xKs;lZ;F+tiEOBe>R*z{(D1!x*Y8l1bNpec2nKSA{036WKdPf'
    'nRGnK984d}BfjR?*{#pbA~Ntq>9-&3;A|k+YMYbAX!3PcEqE&h2a_+#F*x2xE-25o?H86*8xt)j<v+FqFc+=*80bBtXn+_A9h+z7'
    'o*R&gM$ZUHOkNN3_ujAOf2rrjzkR<!xk4GBfm0N6+p$OZGz=|sRb3&eqpC=K{GumHs^kW**}gKK-m)b!;N3PAiLL0CI+fdck*h#W'
    'WlRR-K#axtm10L`7$ckvy^B~ke9CaJfeYSS9uH$n&%?*%K*K;A=6iM+)R{BElnEjRt)t9tH`n6BT3`QnRSzUuS3T5kg=VHF!?IN2'
    '2%bIYvl!v>Ecb`}A}>WY<V4m26s>s%xm$V_ir!?a6=+D!&hZDr#cXm7rHzy(LJ{Z*9*z*mMjPXJr~|7-ibhB>e)`e0Pmh8x0vTAW'
    'C<}}V=iIb)Kz8~aOknAh?Pm+HPt2Y#wA2JvzzgGvJ~4EaXvT_@h8r#BeA6S3;3oSkny%><uVbDeNT9S8Fl;~*h=^$K$0K=5mGUfV'
    'oc!8EFH?mbCCia1v{QTmR=}fPIfsmN1HV4-r`o{YTPAW@C_&+9LJ4m&ys_xDXNR0P{U87G$)m^0F_VQR|GxsK0kzy@ZvEVK1$1A~'
    'kXT5=7=)TCqm^@RgGjHggJfnQpqw`R{BF<a-e_Hua)aZZ1=lPWYHPY;8{xhoG4T}8%5_B7k&m<QjYxlN5;w}7KDnmdVnhBV2h%5^'
    'f<|tA*gZ7xNiV7q$z3M=S^Bq_H-cize%>mxvW8z@gEU_B#uN^NBKbQaO&Tdj<q4o^zbnII!KqXEIWA*>zFbxlXNha(dNoG60vg$o'
    '(3bd0S}j2PxJziyf<pwh<=O>1f@iH0d=2W6$NG64W^$GWn&Xtgjj_9AXmC&b05U^&$^53ep3@w2m1MmR0<6h-qMtU%rh@v+s_JBx'
    ')Hlld7C*N}s_`b7aH!vmEuwwWR;XJ5>Xer~s$R8VqV;^ormC>@L87A>N@glUlDeP4PqX>!Pr!Hbj7>&;L{Gq1Yqw_!GT>e(sPGLE'
    'C2ykA0%<8BeNP{QMjr<FJe6_uBJ_g~L{99necKf!A)p@=cFzkr&JzH~L}VC6yF^iPbfde?@xCMRtoUip_`D`g^Y(6;oYf{LD;5Dy'
    'C=B+@2To~4H(lv2^W(27bF^#^PSni^>df?wN3yBsp{dbTiKPGpL}p))KJgIZ3G>b+on|5kA=O-xun4u3*m95h-s`RR&nsjr%CloK'
    'mI9e@21UwEwXF*hFLnaRTT4(2IR(4F4udAw&>v<P0i%k6U>`4%B_Ep5q(SHu=8|}HUb0O>WA?+QqW3itVZ0BC2=a`?<q?z{`%UHW'
    'G*!~R6IRKVRDR2g%}^DtHJZl3VvfQ+)^mC2nfLbKV<)gU<yK_+bvk^6wlI<z=q$Kd>Ucac2-yaFvrWG=yeyn8*xMOowQiYkfb>Cw'
    'E3~2CQVcSoDJE{A^_{rm^e@vB+`9fvf<VT9-y$@L)7R@mjz`*<p^VEXX9gT=MpheM@nm5I%!SzJ{<3@}UZc=Tva_&&Tg9I4PY`+r'
    '^_va46EwWv1nCHP`4c5#P-zspQ9D{vw!x7VT;N0)4OFveJbQn>%1ynBse9iNx>IkHUNnqHG+B%cOhlo^Z0aO>b`nx!_vGwN^VC|g'
    'IoguU&ETa=KcHqLg1*l2rlS`Ab{YWA1Ml}eEv#-8x3mCl9v-B`tkz1IMt`b$2zVXyFEe>#GRfcF^k(4u9f?`!;7P7$O^KHhVdSj#'
    'o(BN?K=0t(x?)@5R0pc9r0A3)aKEdy!n#+c18Qc;^o=o$K%ZoyY2ZC}81u4o{QaujSEhllOsnaZB5q=BOV=9NrguT%v3+$D8Q2ex'
    'B_RJhJ#0;p)GCC?rdony06l3{{%;yV1<hZja*D5x&>_hDyJYZ}t|{o=ge~)j^+ang+Kp-b$Dud3@vKdVJDQJ9NDN44oRLlT=6Oaq'
    'Ow?ogux&Jy6^}Vt5p_xfzonhqBZv$18WHtfM|8u{IHEH&^>$yz*o9~GErOIVj@|Q&zFxyPQ-hTt()MM}X+#Wnh*cpbEb@#h&6{Qf'
    '!gFBOQ%hEzPZsVN$L(WJ&wUd~YcsVB$UdzDT-JFg;KAF<lJD^?1EO$-dHcnEL7qo2=Xy9;Kj_<vn$cL?VjZFH!~N|$qg4)^)oCUX'
    'CLso#<u&F)$1iKFmJ9)GX#GhtYzTP}ppsq`OrZhW+;dkv5x9(Nh;oRPc1h!0SIua#K#0U?NStc+EgmGg(bz=h@x2<1ncRNN?<vU&'
    'S2$6<9J_n_!HmGObZ~~^>}F(wQo+pQa2c0N$}|N_OORKWw;fv(R&szxlo6z5N2gVYA`iPPJ;VFeA5YYMe_&%32cprT#?v9P>#Qx<'
    '%({U+etxJVq>kWHA)_=O9^hCjqpyvg3h_ue_Cjk((Cq>C%ceAWfIqHv%EYia+(L#^AdaA!EVHDOsq&z}znjDp%i-J4qZ4pnCR>jX'
    'B%n>K3R}mNZ2%S)^b-)Oad93hp@g@cr1PU<wnk)gv%7|5?Kq6a3~54&;E!i0&l0yF@<V;EfFRxQtI5}k%96P=Jbhc&Xr$ZE4VO?N'
    '1+$cGtQ(O^;=%}Srr4*F#)h8Oaco5|$2fPJYOZZ^UCB7P##V^j|5gCQBz67#Jb$<FddeuKzQ}r(0h2wM(%oIo47JD%B6H*oi=>{)'
    '?jWb(o_4r7>~{dOq@TK&!ty=fRtb8Vj4byvCr}bjlaWeHO(U!1{oy$<2ue3Ewn^JlQQGgdWT9Gi8oWzL7#{@*QfFDBBU!~9KTx~w'
    '{QJ2+3u3EnSJXHN+TpGoREXSLLF_!$wN%|MSLE5#uuK%R)sVX|?D~T0Ziu$poLfRfI+s{{>O&GWr-;jfeF80Y+db8VXd#P(6{t1z'
    'q<H%aNA-^-=)ecklhI%M7IJvoz|b5ZgHTjf`gGK(@!DP{!%X)4X<&h&b3GaTX3dSPMmyd}Uf5o_+!O65ly94?kyVCzEm?5G7jp0>'
    'iBrzxn-0GaW#r1rN}(SP)|;t1^RC0eh8fP`{n3YRXUwlXd(tHsCNAX`x3Drb31<8#rAEZCcEP-#*InbRa7tuabHTkRkSru9`nZFy'
    'Vx~ihjimJK`FHgQv1}er?_{SkG+^-0tw)U-m`m?6_Gpifx>hJ!XFQ{qyC1R~qiroxrD}Qz+A_1NXHN)5OyS;VKWyx7A9o!JRXCgi'
    ')076WS*+WH5X>}-6Y!ETy<NRB>r$I9K@EsYgHVw~@H}uZ(T=2=v)458>Zchz59u{pPl+q9>J|Wx(sb`9l1pQ&a})<E1{%UUAcbwO'
    '_NI<1Mr|^dD}w`dGpRZ6ksta|F*m+-<AaP9JdgO3WS_zk7aqiF$GXcojh)1H$+mGONCpk^c&+~TSB=zPoosm>C(XzZn6Yg@#rNum'
    'BnNsb>s^-!ll=rNc8{qt&3n$vCkr321r-89Do)Qo(afu_vn`d%e3D)k$Iy(vG`*)jBmyEVx1c|?CeQNeCF-c%&gPsRp1Z;vLZu2E'
    'qfq1%9y)|ays0%-O9JYYwDn9RB3XVO-S+o#6lK=jEsv!jvLp0lL4)Y-$wD+U1SC}<lvwx>F}l+N?j(A?_2(cea5h-G>Fh2}E;5`u'
    'G94Oa#|@7eyb%lSEs^utALF)`(naXwqHkk6Or}NYnQ@dE0LZ2(q5Ctwg>MIcMlT~knQDuu(!wY7RL6&-T{`lpS8I8i`hb9m&>^aR'
    '&~P*GnnMpgXlxHzwpYF7N)+C7eXtr&y)jeY5>8o+@n{I~yiSUH(vB9{%*L*aXS!uVnH_tvnv3~2x5>5=tR_EJDXWK*EESulrW70u'
    'f}%2sr9y-1K@u5<5lLc76&RXDb_G^FZ|Wh^t!v4)qgcf?ig~G;Lk)`<0S~ahiV_or_dQ&_=6@s8@s66qjH|w7{LF737x9oe(}g-b'
    '5$9ZzIqAJET#++N=7ve1T6od&LeJ4yi8!Oov)<L3XHKtEXD8AfrlKNd4XiwDL;$C<gvak0J}?hGcX?Gbwpr1gn>xQhBLC0_n#Wqq'
    ';O9Z)<bkOq>UcX7EYGsen>kR~xg-r96hrp8-!=5AK;DY{>0=1nP$IUck#Yb#N}7!E|KeruPb{S24pYy0Nro#k9n;dOoi>SUeYA{^'
    'XFVC`X*lRIVi#DRD2%A3kHP{}T&SlLka=1j`df`Rtve)mOn9mK+!{?3I1YWXL8AphKFr2jrYUlq1)sz!S~-*@i)+VI!P$B*BaE?8'
    'b5te-#3<A5#JPT88|)tIdtLy(T{APrlT;{l6Be4YiPbYtqRrRYAK5;Av!&>Ixpp=bg)9q`WtA=iQQRE~+4{Us&-?0FCxb-YMyklv'
    'RXc3*io*aH5RwnTYrLv6`nzMq!=JhvZYIqfLM*3U8CvTg4%nHDVvX@?ACjcPM3Y#=LOV*wAft0z>XyG^f=6-}-MGBxhFJ&A^14Eu'
    'Ap5D(?q}FO8Us|u`qWvBmCyM><z-+EF}*2J%!@oBz$T<wFeiap3bk76Cx?W4@@&6mViR$I6*HkaIqj^{g2*GJtf8+*Fv>LdIbLT~'
    'U8lmVY-c<pqcP-mqG_w{iKo>Z&X#x<ZWE9aEq0K(G<UuQw1ReJU_L=<Hn*BU=K*=;Sw<m|RZ24!L$r;I(M(662xodv$G?lrtot`t'
    'CK;u@WtBS-A^_p)P;ece$ev<JFRa|O<X@_;`z(?XQ)iai<BYs4N-HsPSKu;vk)ai#v20&Ar+HF@an7A=30^^;mf9e7bVe$H0>#ww'
    'k{KGyW6Bi%5yk@0k9frHT}^woX55+C&fMPT=Vl@q^N7on)40q#O-URuIg%OBhjv$+7gp!`x<5dd!7qfLOgorQy%9$7!rCBC-51Ki'
    'r~2OYC7{_9V~abB2KR1${DD(n<`XWGVf=8U_SlfGw;1CP;dD%)jIU0LcUOM~Ks#^Yk@7L<T4&Xl2nIF6T^j{-9c91VOl*17h}EV)'
    '1ZL*hY=R%R;zOUA>mg1`L1By9DN^nk{?vIwj@1o>m{@;RBd}AqRra)w2%sYt39=T9M@N>P1E^~DEK~ox!;>oaJdV+d^D4VNEvEr+'
    '0S}O2=j?PU;0iEHr{%uNI3L&#-A_zXWAp92>sXP(G%c}b1sw}XH~by4llg4#O%)5RZJK)w35gc$JTxW>nX@{iy3!RK8U)T`tZ1CA'
    'lUUDbf=kS0DyBfsXV%5w1Z066sPZn&msu9t4ey<RmE=f17*C_iQ=JVybJT{P72MW%q}7%2{!n`p!)<ovbaWUJ1h7?;aA44@$HWV)'
    'cQ4yxFfM9IVHO8<GAgK<Zos}Yv?j|H29gA-;!TH4tqXL1?}C?-5y6qQC1uGHV;Ml_M>wx0%wID>;Crv6s5TMdH2!0lekfglajj4~'
    'n6pLNj3Iw7TWC#KNrn=b)W#rU4`EcW&VZGXb%29}#J}Fw=jj3(j+;a0q?I&`F)N@Lk3<>k#tFLBRpAZ#tvO%4x?7V2+if`&s2EnJ'
    'MCo0#U6W!i#tAf~jH*7LH!o`yIvqp43V^E=0li^%_LDhqFn<U2!44VgRKIn-s(aIbng(F1Y02piIBpgL{A_ni$KyVk?eh-D4Kr*Z'
    '01PM?5>9Nf8_}mnM)I7CIX2Tm&}jDp7Ja?i|E9EwW@O+)`;p>!ym~LNdKts{ALbJ%_JJcc(7F0$1_MPyeCRZa>VurMo=`JS7=fs@'
    'zTn7#&wZ@+M1XRx@eb(dU~>Qka84Ew3808t=KegdWo2QQjL2#@ykb~Z?7!zsV{sHpc=bM-6IAw`uY2JnN!K9NO1Xs|Z*TSw-Q|(F'
    'pXVM`ROg-jfeoW-Xd(iq$_YQJW5KZV@gyiL2E>6EYbARJ;pNLI&WA4;tQ<4nE63?`um*x|FA5UFio6OppY8K@(Ms{w4OakU%Tyit'
    '<uMZ&7=rh<jUi)f>iwR_OYfXYO&JO440r-pGKHB-K#Y?VToz4wPOo{_XI+Elhw4ns2S>h{u%uN4rqZa&_2I~jZLgK$^*qpd9ZRGF'
    'Xg~U<^e8Yh4}Fb@>EMtWGC|xzeBd#oQUi(&F^FYV#t;Fak~3BSs#v02$v)U33e6!s?KS1S)xp7>eU?Q*_tQ5bB7p2p@`>oTbS$qt'
    'BBy;|BefzTURcX3js5#4K0Kpyl<WySotaTIGkZ!YL3=qZZ9c1^fOdIzl9{>J(X)*bW-$ZpLx{BGReYZIq*4!yO|Q6D!UR0%8~4tA'
    'l0y7jPY|&Uh-6EAooNC%NHL=vbX%p;9L$7)JUaasqOFpP?L!Uw2S)nZT8NgBnLOJwzgHc`1Ti<psrea;s+!7(b-%ioc5ZKx0drVZ'
    'HKIjk6ij3gDtm9#osx(TWNzFuFYR#Fe!M?KR%Prs<J3dhNNV?K>Gmd#s+7l@V%YBL=Hk)G78D1-OTav_ifRY&B+`_W`&;(oRAl^#'
    '`4H=xlbnk9!e*QTF0R}Jb3+G`^7CC|w}nqYS|No_1^vfNf79GXtRRJt>Fi`yh=ydou)Kj@(Wp%$716X2*wGZwG>5fa5;eJk|40^m'
    'MbK&zJ)5v2+Ez3>kVB@rkrdku{UGtl+LS-M$?zN`GS6X}EVfgYl2soN7enXv#JxihTT?m6tHN5S*yTT>`bCsZfiaFfe|{=C6nFJq'
    'SzvUEt@Bs{9+@z(Lb4u6uz_MFR#z~EqvP-iJkj4o^}vk~g2vlvDNjb?T_q3(fOW1Cv2Tcjm|w5+LOW5T5cvcjYfjC3q{h&u7l*p6'
    'xen+OXIZbR1LGsZbV1ZxGwYP`81{)3rekm22vi7a5Rm$wRuuyyZNRo8&_zLPsat6@ZBBYJhia^cR6s{0&*D3i{YSjT#L_tgJ3Nvq'
    'V2%TA7g>3XCd+HUxOBGN?}6qydiq-*p2bk13B??W1&m_s0BQQ0nm-A<O-&v>Sk*Etk4A5Rl;9huV>P2ag7EqfI1ZD+`!LB}4<~VI'
    'g-7Z{j-VXFi@01BD)}Qcf^}<HB?E|!<b;N@02GX}OeDD~W8f@9q}ATkJv>OuR2>GsIR;?2M42SLqkuWFGX249FN1DYc^NxkbrYo2'
    'Ryr9I;j6J~IXrc&M91fqY?71Jl27g}h;Au;x}()KAO57HW1CCPSebjrn_RB!LzUHYQ^KnfLKwk5g7>>LpL`}_jQA)4*3%c<b`0)0'
    'U>92{)v0l*0&wV*oyZXy$PSnOHa@S)KeRfL3vC0$kQgXML*TM>j5nv~o2WbItS{9Qw(vrGWMy@;mHm{~k%+Vm@?sU81mOXvIHP|c'
    'Zp&nI_4JxaDMbN^2~wVE1#4N7p~@50`_jmVZhvu7ps^P65VA+$vRbVmu%oP#^RWhN?s^r(%<)A0mTKlu0{;!g8R}4a_EEYe4>O2Y'
    'AMxPn9IjY2_X!}nH*rubai`AG?tiJ>j7y{Je{f4jl=7Gm>H)Jc^oYx*#4mpc0Aw6yVS;w7wwL3sYN0Prkm&$sLf>?yr!{jk3IQq8'
    'j^dYe7OK;<$*_zn%vwmtlDV#&$jt#p=Os6M{t}tM|1yRJ_L-i(L(~irD@&0?ic2*AIuseN#%~dddiRGzwzVDrBFhPd;VO@<r+;QK'
    'igYDj6S4nZivw2mc<z=(=HwxgMoH`aKD<kb(zIj-(0CqB=KZ84<2_9rY@=jvvUX0H2C%fHruJUIk<Q`huTDRWjRabfL~D8oY)p^r'
    '_iF~Jo{2UhsF<6S7h;WBU_9W^dmvzwVFz|l$W=Mr`V4fn9pcq{{9s`q=1*8VW2*9GyWYYt-`2=c78;zD0jLy{uh&Y$M3F?7tDk)B'
    '3FAc&8FRNo%LX3GYb#v*rzHx>M>Z&#qjV8fahqpJejg$nzYoVKWYor+oHzF)>T|VQHM;7Vo;_@;!uwb>y$GP|tfgxGax?IdajG-U'
    't7aQ#8eGJr40AaJ_`s6KGiu`M$L)a4%plcZipZ(<>a-AKV5Uk+hh@#9cEsO>4ZmA6bRbb;2+jovlbploo_(s?$~st0$sG+eK__10'
    'e55m_xj}ZgnkW?kyd1~**v(*#o4h$Q%(IP89Di<=T{x;Wewz_IH7?V>b6H!)MHG>;K$+PQ%5zR=)m|k`g~GBtMoZ&|(Y67%uWNST'
    'jIcecqt)3jU{`RO2xcaMQMo97zyt)F7Tk~~ky{fthOlTsjm&$`q!?(xHra!T07AN3CxOgjB{d3<{VzByj_IG$8x&7~#PFyLZk#Ew'
    'jjtgkn9<SJBlBE6=CL_lEcHP#RCK!mveXcf0%Mb9=7BoXR-UxStOUq-Nu%Q#FCDLq+QZ`8c$6{X(`Om>SCBE2aLFBe4?I(cPFXL-'
    '14|jb+>A-y6ow~HbKAha4toqKB2wRNku_h`Zt6QI@D28aTyTH{q8Bsoql+4HVhRQvT>Lrvb<rT*45C;d_#2rfJaU?~g3k7s$x^vd'
    '`2(!X`jB613Ym9$9eY=O!_8zk8N;ephX8Sk`{uA0+1I0Qs@=4B-+YDLa|CPS8<C5hDGK)IP!rngw-9-znN?#Ou(W&X!8XWtLOs}@'
    'dZ`e>mO+0{ZEzAz_T(C@EaZYn(xDttm-nhdyhsj{`MWrLz(Lzzuf@AhL5>;oK27&c(k&Xqk<PnYt)<EvQqbDg;QcqNvZ@~vE<LO!'
    'WZ63fcH6iebT-oBjC>r(Q|}F^XlA`83edA4Pm(6RYl{rW(}`l$QA~Gp#v1;Kyu1Ddy;Hy03?R^yK!}VHVN0RvKwN##(}8I$GA(Vs'
    ';HY5<bGX6Lc?2)wkUp4R!6|5<!8VmGVr`<NUR;3tf#{oX8)3>B6f8(C7iIJkSZI`AhuGNdU;^`d#Sa?kYA!ol1rzcuc1$t(h74go'
    'I4E~u2O@}{Z&*)rJh9M#DX{+NT0w-@w$Yg*Z!+V*_A@}JyuW*uc`len2@_;&H=So25g!>wNzf063TelP&hxu^y{hW{ciHl}*qP;x'
    'TEvGMml_`Q`zN5{FU`!a-}Ut>I@|Z;u&w%%FqY@IZU62ToUg9t*Sq`rJ-=Rp<fNWQR6uqVL(73k_h}ctKF`-X`}JF1FImQfvIK(1'
    'j%dGHjb4FYpXKX~ef^fNmtc9E|1WwcVo8ES2x}%@U~!MVWq1R<XYOQfJgL>Gzn9VP7J;FX9>k%wz}bOQCO>q8WLvs30HWVT^|RCk'
    'gjJ)cXi`e?@{>T*Or*iXm4Qwu^fKO63+dW_O?hcyV~~69S5E&zjGp(+K_ze79o{CbSA)up?mo%NnPRbb%P}&jDosmOM%UivSsYWP'
    'gp>o}x^?W?pH*c#!bxd0oiX}_$YJ*GV%#Vk0FTTj8W6}hd?Lm;Hei9Ev64|oY=Y_79qy8wEaZvz4jP#!w3d>@DGIxISFr){1FP!o'
    'N)(#jc8Ix<ua1#q##SL)3YxiqgpD%$^G6c&GBCK<_mEY9)S3uYSuqbStjbaVs{roUY(61Oun;y+0%0(%B3j&1*$(vj`;ma8vd@5^'
    '>zJ)7Q5m4Xs`MK6ZNuNV!EjBK$lqljPb4VI4eK2KrHXKgsb+-&b}K+{##^hv9Uh1nNE87rH8O9eq*Cf>dj|qQQ|OE~=H&&ub{KH^'
    '9{~*V)?wS;6<}>gJb2|%oe+Fu=o&Mq-w6K1Fj~;7!j>zvHc5Z7{dhlUXFM3>va`#w?KB0FSoUwtonHz3PQcL%4KY`FGHFH+vebj{'
    'L5rALlyH0{L{3w55%JRf$LBFH^Gm7OGlXXP*=<BomE4;?&t^ok&3_#20PEo@GZo}7j%-2ZFYs?jJvc?FCNxT}?EvcU!G!9NF9P{9'
    '^m)@{GEe(HOT+pIvnpCfrf1Z~T48@fl0yp8uyR{QDK0OMv7Q748t%t^in$i?+$PJCFM9*pkt`X;1u3gH^X~rD%8VN^Vpu%yYT*$F'
    'h+FsPoNfLkv*mT3i;yXD&l#Tri)|c1+fmo~kXgC+B7;IH4!9oxQgoGoI8HW2CU!JzRP<Q4xj9?w1+M7%LvdlDil~_V%^6R#JWl07'
    '=*^*<41>u)?wNdZM4FRxJ6HQIC6OUP&2t$2Ov2YqzJcKTA7Bhb!b~Nv*vE&5m7OUCg!+t<89NQ|_%c(~q?uugG5`I*+`CKJ8(<5('
    '!Mbl%e5615M(oZoET@|07y^6&|NJ)eXR(QFXkmK6Q{1W(G1G7=k3IVb$!B-~h*+hOSwF1zMZh7wDMWPUe(anSaJZ*zAEvHTFZ1-%'
    '@aDi(|DNKCfs>8@U{Em_K6;2h=Wz4#6>{j|ti`;CZJmzX%vlGq1~gu0l{MojJHeL|k7Kj%bbvh~41kOVXnj4v!nHrFl(wDUheBKR'
    'lSr?s84W#^#mu@;+y<Id#!dH9Y)6%bWKeak;<OXR^X|5RGG&{yctQ{HvvRh;2Z1u+E&~{5wsOVQm2tAh-jcUlt$>K2MZASLbCRzK'
    'mz$Vt0421x+Zz@JbTMeBjwCx0qYMZ_(g2PIBJ)7JZGfc4HncU=-*`gkxI8_WnYbnbC_898btAe(>184fHK$!(38F8ov~u*i4nz1g'
    '@}JzOhh&YD7YXk+ssO@8nc(SE97&qobf0KE8syvU3Q+Yii$dL!)NUYkd@%=ZBQo7Nf9AHutme>X;mj7+Le3BQslzh-zS)X&nBy(6'
    'OBe+>0~|{AY1$YrdYXb{@i8swC6WdyjG~!ZPm~b)CKdEG@{>uFB3y`M`z`TB+TVk2p?Wn`neRfXx@gF)l<EnlG45z<`&>EYK)FM5'
    '=H!TN2_AAJXQl*k6bX6w?0uxGsEHsMl*38iX6uwB^>8LDz}H}BR@{_)x%2$Sc#r38kW-5!KM3#Z8OE3CHf63vq@%2cc2;qpakdz@'
    'Bur<vPc{<tnpsDV8-W`h4nWQRy+BLa(<l}yXCx-L86sJ=OP4Nw6oDbM$YxybFr5U{ngf<DXY~HODp{uG44IJc2FRwXM>^HDRwJIr'
    '5D#&xBvyhp*H5vfm1=RtxJM`FAikd#@dQ-vZYnybo#oEZ%vXhkBGP4Nhz3t#LNQrEmlSqjs#VcEK}WU(k5OxzlnN8RsjIyN6JRG}'
    'LVBD~$_tmhay(kwE2>(NR>EBU&x?Xk=r)dv%@OX-JlH%7vW`_Tg1BU)({tMF9=e?&aJiITaza1UprFS)x^QIjYniS}Pqie{euoi6'
    'i|zZf+8J~m1SxTD?)E@&wRO%}3__rV>x!_Yz;Z`}#-*eEJy|>r%hE*lR~g5#W!?{QkMHp$AK+Lpl$<a=Px9!GJGDcYAkra-WIFk<'
    '`(1E^O<ZJHrOaC<YZI_>kSu{L`5gWL1n{RetN&i;KTQrGa%nwdz`$<5?L=^ZTHAow_Gq@70w%|y0-EuRk&<`!0d4slbk#`Vgo>%C'
    '@K&^28r74=hwK)AxJu`aVs0^!%20ETGm#*)g5H%)V9*$2pDRF_hKX~M@(I;=gU&2?n<>#|gYHdFEhH*}h^eF+?QT?tTh*6U=H7|$'
    'I$RD$=Nm{hR`ZQa6R20A5>t&_S$(0H<Kt)ZZ%2zALuMxLz{xda`}-*w!W(tv&ILE2I|fzB$A}-R-gMOR=TCJ(8Pwrt#Yl343C`s?'
    'Q?Dd$Qm-1vB4BA2Ro=`DM3ITNB?QxCjIGNs7*GLDs<~ScUSG2aSd364VKK8q(|4eW!)h$i)0@~-D>|Gwndx!{o&skgU~>s38E*)K'
    'isbs$nL<S&Q2KC<Zr=s@HN(%L&@>FY8a*c`HMcE>VMgOk1FR8b(KfC0nh}KGT3;l1$h9eG4~4iAz?Wi3vo_Rwaer}!*a~EUgrR<H'
    'f>QZ0`+*p7QY!New-*b^Qjk$tcW4SA^<E&v5d{`_NRNyQi9539yVP;Rd1zT8OqJD{%$UlzTdjAWqNKwi9##*39z{8)DM?GDiB{tl'
    'cs_Yz<=M-kCYYWAp2qf|^C(>eNUUng^9Bn>5XgSI=C*rCn<r5pba_`&J5>3j&+I%-Zg`v(K>s&8D_zk7&es7BP}1Dyejy(<SoE5L'
    'P(w$1f)&vkANny|oNgv4;4r)XLmfl2#?m{)fPaIc1KY7JQsWo7?vF~wn0V3OC1I`58(n@z4rPp6b<@PdAm{Yk*#y``3sWpCg|c$#'
    '$lND0JYNk{H>RHBC3d}Y^^nGx2_zK`A#vya;mS~)tZHsh$%+I?EfXm6AiY(&#dJ>*d^1Mf?`T4FR_TA>6*XK4TeBApDUO#wM|f%`'
    'rKW&B%)uz4xS0&i-Td1o(@@y^09xY(Y93S;du1=oN?N#-O-&9yg8NCYC?}DXc&|LPE`XjMT3PbB-F8MVnF`W1adakXEQyxAO?uj)'
    'BE|yDkZBr>HSPKY?uR^DuJuFLa-c}0p#!CF($E<?+NL_@U1a+6;PfOF##X()6UWJ4F=D2vFcyu1DP^_!w&(Fg{5rh*XmTBrNV`4C'
    '_F@q-b3<wBMUF>>jtsEoU}t8$5@MZZ^ce^1Lr_sm-w5R}4Fp;WKn3!uno_wkXPU_vN6c716f}eA1vMWPdr!EoZVnzZ#62dqRD)xf'
    'OCxUGH7QB7vDDa)v^94lp;wM0Z-GpDOUWD_^Sg*@Wq}*7u%S=+KVk1Wm<T{2Rvdj4K`0oPPBqF})pE88QO*yeA<+UOA3^}u&Bg2E'
    ')f%YoaGjBeu?xJM9nR<xrqi{5mJ0Z|vD~kEOHE|+rw>AV6N&#|=c8cmbjie)bE%NNhN{4lkvN)M7v5g*UtO;yJN?RFu}YnWzk9IH'
    'O*`2o%ZoWIN*WA_gco)gm6R8SDLK#CSf)xona1ocM&ZCfEYplv?letkD52=XHPd}r?REy_HrNY3_f*3~J^##Ig-D!mKqQ6?_i3P4'
    '4vq99OJyy&mwZIe1d_3K_K!P#BQ&EGLk85=F(9bWszixN63|9Rf&XyO*OYOMB$+m=K-VI^vWnip{-D^mWWJ7~P}HGDXKR*eEf3@Z'
    '*KOxc9kq>1Nsn|_vZS}6ve{yt#u+KuSFXXgPQzLqIY+aCJsf}{VEasABWs$?McEeH#76(Y5!Pt5|Govl#~e-Y3=uAoWMe17e94<!'
    '3kxtTdo<aQ#-TMa3<l?D{PHHS6W26LcBc0hWP9bPpH(ibaXlFrmA099=BfLBv}vE#@IHQJ!&E?bdC^hiVod6Q?hmXPT`v3pQ4ER|'
    'Q@}X~vhW|i#OBIc6D{3!9g;@#L+1(XDa)@?&sI+dsPIf88tvH_3}rKu-Y;WPHw=CFCK%JL#>v0Uenrj3ey|)s`RZZPEK@7t*$5O!'
    'd!Wgt|FT;N*W_>#;X`l4*geJ%%$hiEvAKAE&GzRHSGYz8CvZD4==HoB-YyU~Q616dUPgWr#a7RLLY5fH7S9#o*{Ct21U}yA18Z|e'
    'CQWpsv>YzoL>0ryz$Ar_68DjW%j_Ld{J|03WIG!|#Mk8k#>|=wdUb@;R6G7f6hY4S$aL;)j&%Z=8LYDnpdj3@7NV4{EKMDiNYbVZ'
    'i$*vwocZXS=>Jt<alpoTsM%x{C^Iwz#B|ps<`DHHxqE+>ZT6l+fyfg$)r2L_hN?^Mmr9xjKoKc;14G?JZHBY>pwMc_{VFP-N`hw`'
    '=yMbOq)~0L9<I#c2jAn<N}(2mghtv-^qi*>k%H*z-btHTbEx2Uva;>#8HI>9v2og>ecJICAk}ur^XFU3_uM3w`EJB&R08#zOze@$'
    '_szA@Z|5=@%`guQI#^(X7W7E29I__T7^+lTR68VCBi$OttpH$wO~ZjvK@*aJ4dIIjmBrG%(cnf>;44#$p@@p+1(GKHqck9CI^afa'
    'eZ4{6L!v!F-YEuY)SN+Et)q90!lVuzgu`ErxRy)ICm2+=qcjR~JL@XeoYQdh1FT0uX1|6<l`Uw)D1dnC8V-!)P$p*R42lyl0pN$V'
    '^6mtQ3uAdQbW@tX2&|mibKfuv3hV*alHrz~a<;{Z=I&7RAv%|W52dgzv%tD?I3y1W;vlp)wB*=V8bY{DVU0V~ykb7GlK#Ck6`4A#'
    '4u@g*=Db7n0P*01?F&xv=8K|pFC>lrgvsGkPZB;*06y&vc$#1W*|ti#Q&Wd*-?)&xLXgiQcUcR;9^5=YA`sM2@X=a0vWHNY!H>;E'
    '&y!Lj6RkWB(Hql%z&D^%NA+*BlRudMa;mC`(^T_Jy9J#-0`uqCLVeVD@?#BkdqX4M8P3-9M?M36cmT>EVud2mHD;JI`Iha&H?!lI'
    '`vIBJk$ubWNK{-PWjV~o;{nttW~C-Sb!RROi{U0SRQ(7lft9$?O^==FS9_ZGhibB^v_&&R^OV*Xr*#*AH^i~9<#qjFcRN36?c%>M'
    'pSlZ0BqNdff_$Rkhql);Qo6OpFprbMHWuPetJO?~<%eXl`cUK?KX76+=!wOxLl(%Jki#APYS^r_A_lyEHgPKNuJn}_6`JT5>-5hK'
    'Ez~axQoB2PVL*3e#NiuSoa>l&k-y9VZ$XjLEQb-BgV@q`y+ec4LUM(nuAtiN>ZuuSVRA=D#1P~-#zC-Q#z{QGGM-`QL#<?`VOt(g'
    '5G{m5+2^sxo&^1myL!Ik&U(<z$9H$v&CQySE+g<7*eX<AN!lWv0<xBevgt0@AyV4<!+bh2=Qc7w04NIP6kC3rcqUXFry9x)DQAfy'
    'L3}=5`48=6e}3#|meIoaWR5q@d+)S2;DF)NG2J6>Igo@eRn;mP!4{GpM@j}rUw5NE6Tvp8gm0=>TxU&xU}7%f8T>7qDu&@GTRUcf'
    'R8@ta<zg~8g#HKW&)w%XD{k3FgWQDTgEFQmhZI`Ott8f*KOlvg2pa?89nvuP2uiBp+bcvoGh5a@T^-qLAxz-2gtWM0Lhl57QP8_T'
    '_^%K?O>?x@b7G)>Wk;`b&Oj@!Fh+X-?3RKwVHpua39%BKug=k+A8JL0T4<pZd~liedVnqLX+#_@%WT>oAVHc<0?1-4cgU}}Rgu(8'
    '00kO+WdoA6G&5w#Y<H@Jg$B51oqEh^QOyffr_~=CCeZYUfca0JYuMz7AVx62tod$a-8i2Oi*7MKL1=IL9Z&lKorn(!`)*;i+QCqy'
    'Wg1XNLuo({wsB}U74`R`Q$#>3K%jewY)?gt8+Qv}$`So!vP_-6nCen|H5MaCMP#EPB0oshI6Lma#G43IR)0q%su@iM8#eluj{qb%'
    'ZWhh~t$I>-&$tf<XmEpP?$2M+u&DBKg13~tcI1=){f2Q=w%<EKz9jvIr6=gv1s+)4ILL+VdXC6T7ldh6ceBKo&eW8}u(G~k8SXSl'
    '%@sw)2Muin&`I`Mz2ur4L0L`;-Qzd3YirNO(lTR8w+YwnuA<$Aq7meU#km(HPkiN)9#B%V+|m`Jre?yh%%nt2K*XoqoNR|+Li?eU'
    '6n1vt1HvaV*i1ly45_u0g0Rn`8=-N>DS^JR&hcgz`#-Qh3x^x0B2035F`hkM+mU+m+9J5pIQHXFU7iL<b*P38EbCNl!<?3eKO+DP'
    '4pbzEqdJ#DrS%_on%p~TWB?q4XYLJ}mTKUay~@-Lb<}6}5u_z;QnJ`-M1{fn;56U8wIZ8pAoVKu7*fGGV1}b>E5fpc34GMv!HjTN'
    'W8um*sd#9fsw#^+ObHGmm5CEple-vHztS=0-xI*zX`_UK=Ea7{+3Y^t%3f&NNl>(b1sH%TkIokTU0?%L=F_8IsSiEJ$T#wwPr;v('
    'I;xX!e5#O=n7IN+Y%3@DFrN}0mV07g5>eQ&bR=^aEd~a#0c<mrI8UK(s^zYB9cXE-thJ2eDy--UxwUd59mMw)^>^spW8dzGHzXJg'
    'qWgl4s^eayal8yBF_M6yMdfHgLW9NS9(az`>H<_VEe71!ti!v=#`y;Der26Js46O@N=I5)0jL-r4cY~5r@)Y@WG~4;$d0L5w?W8<'
    'qT9dSMJFbaHMB=WfdEvQh=QbZ1C4+an-N(}^{7afSyH&}`Lg0-JX+)E!4U3n@|;=t2|EmGT#`}iJHl@7<A*B8mM}{T)1Q+K0S;gS'
    '6g-rqAJbk($zACqVe<c<_s78rPz4+mCThjiga#zFwgXpcj^IRDK&1qNGXow~X-`B3_hNA8-}Q&Y!_AVk$&epCpM|rY@~mr)>yj*Q'
    '5oLY7KkI5$-c_RtsRQ^8nt@u4pk?HQ0;zZH&rO@IL2wkHDa%LR%Ac6Z1g%HlN(*JqrY~n<*lb*<@BrQRB2AKR!a@bI&xZ}2h^&=g'
    'h*pYym)TlJw4{BWm{r}%$f+h$o8SQ%oVJ2qf1@XBBJw~9juY~pD>#$Uwis?gokklq$83>ZZR0KAQN>yxc872rObs83KUz}Yd60eA'
    'oEse{X-r=`rC2CeILy@6aiX$vx)#)MwAr+Z_J_$e9kPcFq=BX!TFxy3Z+(pXnv|yx8%j=$eJmV|z7XLN#PB3};b=ML8deU1y7q@a'
    'bBF<$U`ac&ktVqK;0#8Av31ox+|P>jmglk0$Iux1%w1eQ%`1H5w?)QdRvNq^l~M>>F}$0p^rj0KqRrTI#jerlk(^RY%ZzZmaaYyh'
    '<YSk3VvLoUVm_#6<b)?F_3Y2}vz+o(87m@`3Q}Vk2Dv?+H1ll$!N-&3$h{C4mUcG}kCsEH$Qmd=RmyRAyK4~p&|taY-*hE&!$|1t'
    '&dhSscTm^)tYZv_1Cc~EkbP-H*}7s({KCmZ%6Swfh=_T_pTU=B`icRG;%V#mOc#0%Nb%gLWk6kKQ1awB53yI9u3F1k=ZD=LO+}!0'
    'N1P`H(Gz-MQ-KK+9Ah~KaVw}oD9K3=BaA5D@7GQ%M8EhzzRiKA8xjqOA@qabI58A%C8Wa~)z}|)J@iAkoED>rAjaI=fZqQ}b4isd'
    '?BoG@3p;T5P;vMl@ZdMXSbD%H-kEM8H!wvyyYE}*z_NQLavY!?LbFecgQ~WRF{oDR@RH|VppECjaPDxLiFwcU71h>0rpzKOq*b&c'
    'yUIKvn?EpC=8;Q@iro@ya<qKO-sK#OaP%-K{w#+ixZ_4C28yv&g*{MTO@<5n9@81#6ul=No+B*gQCg{^J+0n@SHp@_g@8^}1;8|R'
    '(?K&p`p_vzT{6+{@nMzNJ&)4RKr%4VN-$G&^JsL*4oA*n-xb%C07avuB%pQ#4~rw)L4iRhk)iI>No*AHX0+Q~!ZtGux9=SE5{Gz3'
    '`LN4D1ER@7&BmAk)1U`P0fRPO?oR4_K!`3&$V`i|VBNMJU@M2*04dg$Lfze&PSQ?^vwb7{DCJpS)>2vr<8d$ZI*9YLsk@vypgd>I'
    '8jfWUZwhZlxavFblt`!k4Uv4IqI6{w<bawqffCS?Z5su0C+AR;feGb@G%xxX$6#)fNPFa|54E<v0&wkU>&af=-)E*B_Zdza;x4xH'
    'FAhRdukYc(_5lShEng)k=y~yOaOoDo3{&?ljc}?1w#Yv}!GpCZ5tvlsyNN!h8C^lE%BnWg{qE=UekL@36tZ+M(MsT9>)cL92`g9P'
    'zS@K_b-Hrk>ABP-wXX4ajPt5t{xB=UEcI^qz=Y}ww`EaMlJ9cRZX%QFrZQoog~;&+cVqj{{rU|sT*(cxA3N}4F3FoLvy8wI&TUB_'
    'Bo~s7LPnM;9cw6{!?Z3Vm%4?$8{9M2Cj4w2uK^>1!>kU`N_Dn|4;j59H%#x+g=|WO2%#BLTN=3Wqp@hFGM*9ITyGGf1jAq!Yg#A`'
    'DGgxfN*~5}H{KrhbZeyh7HNUkoFy8jgd4O#>Sb`VBO|FM&10<88+`%#w;t->+7qvUC3_PVU_#<HU7-*`$*nkm&PjJc>VfoNcD=W9'
    '>yKg;!z})Mo3fzB1WnCjR#3qqyW!-Gnxa+Jm4r!=hC;+6c)ulEnS&z=8l5kdpy{c$#6SuxD*L4%sUsiB1et{b;Ui+z(qbA-L)oy`'
    '<F@G%#c><Jfno@b){`|3$(RtoKW;e#7J;3FE()p3neKwH)&MEh6x&nxBd~<<M^MNnpc@2JR1r*MGed=FDqG{>rs5SLJ_naeN5)#x'
    '&tqB{Px`>hNa{hd?Ls+EGoYwYcbn-$I~8Y@Gh1fpeZ0ZChi+o>SS^vGrFm!?c*L~98p$>uMUUxy^Ob>vtDH4wApzH7#wpB{b_8zN'
    '{FWY9<`O+KFM^$)7RmasoT>%oPfxzw9riUM*^%{Wl36D`n2?i(h1SX>)v{GDA*!dM>@CG?C`S#pBHnz$RmT{-(QkIob2Hss5g&FA'
    'bEZ=h$xl*JwUt$*#a1<cV?IhjJ1`{FSuT;fX-oz1+A8%F;%9~7!?0kLlaEwUVFC&k%-t4-Wp*UqbE=1p_*Qu{k)=3-+1?bVER&F^'
    '-mu^GokMNK<O(<uh*)}c;mxqi7nqHuNT9F*5Vkl>&1Nf=y@!Ni|4{G9IOB>9@79rs)j5{|?h$iA_!bIY1Wg2^tJj$xWhRCd8-yt2'
    '!I6{v+L+AjWhOvq6g_~g4IDx=Lp@p1@(?%K*~FOp@NFFO8sQ02;iik*Brm*ZjJZIZtNW4X+1=;TQC~ev+%?r^95zK3MOQqP6xb^#'
    '5m~%twqeJwXu2p($ZuxSf+`uJJM&vDfJopO**C+XkYiBbaNsyNCVvA}id2Fe;Ouj7D|y#$^Q34jAlYpY6lj-Jpg<@#JNiR<)Idq@'
    'x71}X9}kYpQkOFjF@)Tzh5IeU+)qP8Ent?hMy4R%BKeW<P(tWEHG+*vC;B#3IV}E559f1mTcT;UnbqgH+GZS~6hPokiq_!GEc92A'
    '6cI5$KcZ&@nn=CLOcVeUV(G~#lZC!7B8=|RIawc3G2ZtXHtRwB|HC}F63=A>BA(V=&XY;Zln5O#H46(RuU3x6vpXEa4uFl+!zTJ7'
    '<wIbVy9_r{3iLthDxI=S6!!lHzaot|Ze$voLE{G1+QIEVUXXNpP#Ys3imzTgJK;v{-I%>YADe^W26|$wO<9@Zt|XC%rm={shbYQM'
    'TaK<j3zutjUGpHf%SvV11cXfQZNn?+nXo|hg{Bb=xp7Z_5jD3FpR}|e%j#h|Q&9~DHG@gjVsm5aQ16UVSq>@tD&ULotY`xw7!!(S'
    '=d6f{0Xhw{o+4Nn%?4(=U7rp4s4L%}HSI9cN##;KB(i}z0+7XckWn%I6$q4JhWJ!@LQD<g!3wwiVC|}$RJu(4F5bw9Cd|%F06CR&'
    '$Ry`M=fZK^J<U+D!5TG%a&vN1M2z#hit}w=#^VBWpr9{7RE(OY{E@qEH#!S%II~nvY6DI(Fx|Tqk{FPjdzy8y42=3rw_-248Z|~s'
    'NPO69Bro!(d-dwk_{!h1$cSAs^`e>}GjP(;L>=Y^$<sqgg*MlKDvaoXlyn^hSH}ob>E=>?sAr&zdhgzQ5=j-EHNh}5^ESJVLU4Wq'
    'z!`JUDGiNM7~cLQ?MIa@npNbintEX+2V&YP-cKt4j0!*eJ*aG(Cil2iHTsmKHKR7l$ecYhCB7DVm1zbgRXOY*FFG*LuBE+eVwCi3'
    'x45Z=V!V%1@^Fx=BNIIh3!%JkHO-~Rd)OG9rYW-1I^>yWS~b0b%P#AH=SgMPoqjR%GH1*t(B!I;I#LL)rIiLrV`sKom035vC7Pdz'
    'hALfK)fLog;m)aMg`2%S9ba;e42Hrln1Fvwn5x~t7eW6BTL&=Ie6kwE1#XwdVC@)LnNqlu<3yWWmEAhTgb)+t9Hyew&4j6b95<N%'
    'L~&p0p4nOMzHh5wf2z*d9AYrH;qmL3SV#*ZJb(p`o}aKwO2`hYaZ~sFjz_;ft}G)aU=z(pYZ-o=YH>GnX>L?nK|4t^9yM1-YFeS<'
    'jjA9d9<OD<BDsG!X>bgd5yGib&q#5bdwswHxpy+)%ELOz!<?MvWkj$+wVL|8hY*u@VoSoLqG}LXXsKB#Bskh7@*#W$ln*L}Y-Vc}'
    '#6cmSPQ>WW@UAei!L*BJ9s%k{v|Kov^B@&=F)Xn9*3ZnfK(%4IFdbpUkdY~1GZCJ6frbw_vEa<<g;%3whtGywo#_Es&HEH~yu1%@'
    'eL`_K-2wN?SyVX*4lg{u%z=*c2vTXP3*E7h%%QSbsub){5o}<O^pY?vME4xxon;(yXevP7n8Gu`kVopYG#*K)ec61f+DNV7G&8Cx'
    '`EdYKpfC65HfC#gbZ{#|z}QQF0C0vUG5BP>gKEdS+?k*jBSCnhdpN{Rz&VYCZ`-)I=!`w)7&TwuK=?A4Fvgyb$ELarA=7N|s}B#D'
    'z2fzVGYOE@S+*8ib=gMW6?z%AWA}O@bt@@Y6}iS!Tu)BkQ;^QcZXyYcx&j+xx&Mnyre@fus;EWF!y;gL0~EI7bO-S{!#I$UJB1N~'
    'cbo3tUpno-9J5#$wG<>a_dvPHd~#!$>*4q-r67=IRO3&+e(BdM&Gcg}bX_S{S<EK9vpt>q!$OGk`PaLBe&yE-xw;E3+jPpo`1#&f'
    'mBuOg?So$T*Ju2CqhG)7>s3Q>AW?!$K(Hk(1hmZ%xUVbo>+^lR&97hg^+MFhiw@`!fuL;d>x%xmE?*z->wSLx#$PYwV7%=i?}Tiy'
    'YJ=)w!v2qL1j&KaR<waLrU;xsQ}NU?5(KrnAae=?_6Dd*sMoW>{|{m*FjnUo$bFPxpVk<h7)Nsu4|SGR>REOL30(YF9(%wD(f*zs'
    '-!ODCcl7g!ZP0F^{bmYo@CAYEcb}XI<#fX=if8EvMKL5tCO@N<YGLAbNll?h9|k*AbLeMXzCE>l`7H0^h9`8K74-oU3I$p<70jC#'
    'LpLx^M*<xw21ZS~b6DFLK2g+l_Xydm?NLjnDRdZ?H8LiE-3x#uTXie^<e33V#91D+v9-Rocy?3A_}hUpyI)UNL-jd}KYxBkY_$c+'
    ';jRRl$$<Fo8kOTeXa~+3o8=55X5F}523S8Jtu)8Ogav#W2yN-HqXMIrG<kwjr>T;{onM|52s|DV|87|oG9f89V6+6aYe3>)mDEfM'
    'Ac>NwLYlwNa4{1Rt0yUu^$R>M9Q$YtNl=0+5`%6bF0my*5#Y1c)Z~%Eb(DrP{iStB9@JqKCt<CQm45{w*c7(k8BKeU#K>p2*wiz2'
    'LsPsaVy{pE1uUN(0O4$P-Q~U_>d*MP{t{%KS3(!z5Vj*Y^Du!9J4>Fu$-v}W<V4LyIy8bUd~ngaCxLucg2dzEX%YDqN31oo+N%t0'
    'EpH7Sp{Pbz?!FS{79K;u;<&9NV&qzK+@r0^VRA8Cgv(t=GDMi2R065y)FjM|u8#a0L(1SyGY$RkC26qixIR3{3V1<jOqC!k`6V;f'
    '<f>p7vbPn&_~kW$Wr0~g&5%|i{h$9u-I*oHktE6TDs_S)Q`_zS7kkdE7KwZxvruCa5%?Jzszw4C9`0_sh_K`);2`#}2YRYfJKL7o'
    '(X?9Dw|oQB$}aVdHeW!ZXNGZEv(Y7KU2*7WIvs*Z|4ov}>&>D&l(qnC1~PP(qBuBsOV7+dtzxArDP%vSDUS5f9&eYH0~w?r(@eAM'
    '=@I3KW}@*iyqb_8E;~gN_W@c)e#0{}@no0G{{ZDda02!DX_>*60mnAehE|cHj-&#cF<h7o_5%;+f@9<xl{Y<hvOm9fd;YIe67hYM'
    'Vwqz!vB6@dky|$Zabk)bn64ZBNEgtuUPMc(Dg5fNSlHCB#IUn_o&wr)`SVfG1jIXL$>ko&((@l*n~c5Ddk#Z=N$;IHp#PphaRs=-'
    'QuMk*!h$<3LS0!)VYftMJ9VK!ighMIBzH09T6B>TXQpG2jbV>46CjhdnpcYHinkqwL;*m}{(i!W{H!Ej>xGjzgG!fft^iIMRzaJ`'
    'd`M+o>A4T~NBaO>VeS;3`L|zd-8&wW9-K>d_#sJ!kpf$4I9Nh!GO#J!xNNE4VOH|Vc-ZGK`v-aYz*i1gc`!VNv~kE|#dLkRC*Pnd'
    'V>0nPzs9Ro6BkPs)k353Bn?_H{vXZlKr4{Wm83ucjWQIOxS5u=dSz;6(6N_5k1*kEP`H+9O5@DvV}~>hqoyGh++En3SWDvof=wu9'
    'w&U$*vHp~zp+3SM0p$yDb6#B#JSm#l12JMN&>uj@KX-o>;CWOd)!V}U$a{v;K+e%w$)Fwx%%M}G&KQ!Wsr^zJjSB4f3_U5rn>6WK'
    '61Fe3e29D4N#HKsfG|3sNScKRJmhthJIuZ6F3C_2CTpmHI=heN`|dz+Wx+@I4D}r~8wkFmuwfb$)!ZIOlOZk@t(t*_|Hv+*Q86Gd'
    'Dy#@#0c~P-FjfVp8b3AAKtp^D&`J=|clS%dFmQh#mEgE}*8C0WGom1&%0iAO$|rVO$nTk&2f``Lm6b=Un*;&uN&1Cs7{-8P7*Oy~'
    '>@h=i!phrDELG+;CkIK!GrA)=GEzB!QgVOSEjR~n1V7|xZzV}K^2Z&DB;BQHo8TU$8#bs#m=`9S?v@YcF&PmhM>t&hg^B=1kHyxH'
    'bXruHhpx%mmzXz(Smq#$JL#D=z3||9Sy{<6##Bo-3Oe|jRD4AvqU4XsXu%nKY)e-&TCC@wv$@>op|0+%m+wpIxp;G!L{(*NlKESj'
    'Er1kkE(5E91FrnflNHc)v>Wt}X^MogN<3wi1H6X`Yn6_bt2dCUk;3M95LPft7e6G`BJ_Yx16U!i@@SH<c1Y^u;%)GEEr;!&<=F%S'
    'yg-s!Oz^7Z6SSr?4>vBG4oXSnKUE+iVR4)&ef&lEU|?enz-`5BJXvVg;>Jbf<~B6Fr%B@2l`pM!j2vi17duDh_QC&1U_&U&T&Xwl'
    'exd?H@7mejamW?sz)1eyk|OxuJC~V=peZ60H9r*DAq0c#IprB}Wib*e>CCU8aSbHb)#MTS?`GA-`M-DFY2s|0XbS%huB)^l-6?W4'
    'XdPu!IsYVcfEX=)9+nfEQ89vWoENhT2;)mn0Cu>e<p#C&8^UaPKO}npsWN$`8$3?0$z#H*Wbh6^>T`~pW_V=9{vCD%Q)^JU{2T=2'
    'r))^IXmvu)q|;;s*vkBuRP>BQ2?`fGs%P1IN)U`tMIELW@n%0p&uqLa!!h$rRBY!B|Hm&^VIai|UtY5FOwB}vGQ#i^h9+D$HRw^)'
    'T03bVa|gU#f@-=uriQl-b!QWYh`>WEE1QxjEQ%`$D2ups)~h&84z1(V+k>bR35c8|OoWr;&UBG+*;!q-9{!AX1o&A`Q+`FpWH9F`'
    ';=gv%GCC%WC52PY3JR*DSD3jMNU#k?n3$~PpyEdKlEy~k7t=T=TofbordLRK6C!vGB}f^hpIe}EY86YMU_yGF<ZfZ5r{}gC0mdLs'
    ';m>!A@O8Xnpp_5gh8U>3Bvu9XVkMAlqb?>!4V%U(C^nIkFzzWA<9CK7^~iJ>X!?WdmK$wCPc-l(iCP4NmE=^q@S#{gOsr{P8Uc~@'
    'AUAn_A7B_`WNgQ7Ez`8dTbe~!ku?&+0{{*3%C0%(o09xcX`*$S(z)5i;J7!7q@=dDq1b~UR7`bT(CW;6e*VjI?aa;SXf;s7KkxVL'
    'i-^+M5M$fCV_uAbRU67jl^shyEt+nlcd>}ksd63{B9PyJrzDk~H8~55UNg*U!z<yL*lalIzv`A~$lDiOIn`bxJG44AUA}QT_(PT%'
    '&)a-hb7h0T=$5w<<wHY_1ITs;ZHnAh9=q+y+8jT0iqC2QUdf$K<#47hecqkhK(e#OJLV9`QRDfSMVk+;0MMYiV~DMl4`wysCuycE'
    'pGn<P0E(G8a6B+!*hYK9eIcLFv>L8MF186-J8~9@VYx20lY&wjL*g~brd;qKsyj&bYWRLlLguD4l^jMwhtdczjed~6rM&*h1)eCw'
    '{&t2W)7p?SI~ZIvbj$sRazKK_ssnlJxCAA|sFSzOHGIDNeLYwX#SwuTd$wKv=z?V7*tQ7B(FEfjF7LK0wz#4zMs?v!F0hq$q#Mnw'
    'cwv<_!;0v{yU@>SBQv;>EA|4Z)zoPGeq<A^(o;BJsL@PyL;vKZ!oN{bHZYQ`WWtGX<KH~l*~jll`u_)G9JM`)X&QB7MUi5+oRQk_'
    'ih*DrlI**#Z{oQRm*!QLHXwN;%vjM`pEH7n-<cV~#BbM>)PcDkJ9(2IHA5HniJO48kM@pbPV`RHGy+ht+=tg0A$<i$7pKBfy8@e|'
    'W%d!<QO-FUpNi^cn=#J)*QzH$XI`qIX`T<G4hKdwkWF~4pHH1ITUMe>!$Qn0n^{*02(~!Yk;j#jn>lo@E775XoMFFptU1g#O<&eA'
    'DPJT*!$Mjx%cNc1v3AXblk-iWfU^a~uPnb$O;(E*M0ArPVdqLa&oU8@NN}ouGOn~o6Ll2R*@`Dfcj#E11QJpFQ{^lQkkFNF-HFrM'
    '*>Hw!$xQBA2+PsLc<2bxiHQKjbF>Lm#^_XQNk*{yHPW1xPZO=7k%tZB!i`w&DA2ZhWU<VqcqgbKBhQf=@INulvSeYK$n=v5sfb~0'
    'H51t$?}IX(nL0STVYoxF1^>^rI3hEZs|%i2sT?#(umLR>D;)Gh<;6+GAFo*%sw@f`g?jJGCI9n-W64S2iiHcxodI8%@X=OlB;a3r'
    '?$|H^xz~#ppQj|Qzi_ru!5!Q*ec$jl3b)1X2>ENw8;y0Ac^8QEhi>7Md^U`pcps+`HzUKzLO6fP5)My!DOS{UB2b4i|F!CP2{Q?q'
    'rV_8$a+F&=E+QiWTC!{y?ymAf)>r#Na6x!2J3@_qEnp3GC8Rb}S<QM-gFd(EaYNhivHTzY^m&<C8-EQy?%ItpJcV|5Y9fx7<lyj@'
    'Z(&GrHN>7jv2iSL;vN*94fSLZYCCg}%{yW5H(;y)wGG0Ect#qwB_k_Hb|h-B8}azG2?&Nc8;UWI+WBVg7se7a0KSz{gh?{er<9dX'
    '&(mS~Wv0r?zP_vv%P7Mp+8BQEnr5l!ih$CGG56Rx<sMXTgmA5)3<;shUlg|1Riq#lZ;?k}WA>}5RO%K0fNBgD^ZRoB4kf7WD1+)z'
    'j!up6ylVg1r>3ACh=-yr&RS?}sKe%xvQ&^Uq@8zvSkcg6PaT@$>K2L^%~;MZ8NML0>svv}>bYPIQpzhQTcFo&ZWyN4O#EHsQ9Bp&'
    '6+4yc7zd4>hcttsX3WjsME>sxBeRi)y;A4_H9ICpV+=^s=sBGm8l^cE!NU_>(nQ`c1LP|o%#t-iXb}$gb!S{{XxA(hTF8;mF&}Zo'
    '8<?}Zt|n&4tr)l#2Bes!CO){EcvUcXUCUickul>WB`tEMy&${y*MMjkB)_;g3?>A;ORTwExU7{~;><U2;VPZHkO}WIaR5O%F}rbK'
    'Oe&Ex8H&K98Ljq_0tU-j5q?xh!-7`1F}PA3<iK2**u{%W5k%iPub>+;LRkE@ffAZ>-HTIn4EFJ+(xUA?BLk=v2hcL1CIOJJ#~Zqz'
    '=OJZQe9Ste>12+Ic+<Dmb{=}}2wiQRA;p?CAd^mw1%s+{L7BwL-4d195Zn!RmB8j;n{+0t$RJ?l=*cxmkhrpuIF-dkh?}T<vMH!I'
    'mVnW9zKnE>GtX5Q{f00^);`#^cvqwLL&|*61n_Rc6e4Peij!<Uf{lpbOvY1%ji*kjW0u)$%g|&_aWULk%@o$i0T_M03=n}}r1A;V'
    '!U#he8?P|rq==*oa5>jMb`eG>_iPMI20;jS-as}DOfj2toCk)XkxV8$Nl{%1R+4eG&PZw6l0xpvLo6dza_heCjquj+PP@mH8npm}'
    'zT|_=o^yUTOIWEGg7RPbHj9E1pG<f2w-|a6@14B{Lm6kY9LWZeuts_V@`|RkfQpfo1~!f})3jzxqNb&g3k?tjqJ)eNn!4(8a4P^C'
    '8A;%yb8d9Cq#Q6Kd`5RS4wri3A*!66N)Za}bBd67?j&YZRBUr)0uU0VSkihTskuV~-5ZE2l1#%*VNrKwPK$uKa9H5$8wO(yZ8(cK'
    '!(*J(=Jj%z1JW~)-|DrZ$DYI&wwa3&de3s~v9;D+znQBgh&v9enT0SaM3W^!(#%TcU7PG8<u`PNLi}h-PJPFH#SPJh+?b99p896c'
    '!V`9qsY4oJdx8=Q8XkZ#O(rU{>J52`ws;we#@<Z<5xYcV3{ikN>nGrlTa1Vl4^iIjgr~3qF4Dh9=VDp_M)su4>o9lD{Os19EUx;7'
    'vNpo~ozbCG*t;c%zuQbMgN-T2LL;Ifo4k3<4<^O{R2FQ(X>-~$d0lUC6Wz^#rd-MdQ>nuwY@orMABIegAf%xhjEand7$7I0oK@(u'
    's<R5<LKHew3-Ce+2%(W*M}1{;-B92eGI+uIYdrmdoYOoi(S&X407hPf6wUf!2q%8Mk_09))A8LBW7SgssmOo32NYC-_kNz6p4}~-'
    'm@V#Vs4w1CR!@Lz##C;=*pjCUx)TQ)N}91_%ZRDcyb&>vx{~}eK^RbTd-LFGKSobUC!}srb_6kmf8+Hi`@S+C5Z%eh<b>2g01!{1'
    'X51L4FyEeFy!}H;90aXs*_K85;WvQ6-nQ#fA{@zIV`xXzD!YinXw4YFB+{BQMNWjN_3RQEA3MIIwir%OJtPytyI4hU7cMNo1M{Ky'
    '?`33KPM9P=0+vyo%WJ54wbx9-9J<qFj2C>INTCt?xkw7eqV*1<yQ)>EWnDNA`U<$&)l@yIk8qeS6mI9pk+ch@1(oz_&YzIp&uEVc'
    '4?>Hwe?vg}e9@#RKoVd;TOuW;%Vt=ad?wTK0K;*v;Pcmn%0?OkNQiYwi8ZT-n%O3(zdIM&?IqPFUcb_nVZ7{G7^x#e^bjP?JD|sm'
    'z=pV`#P;C3%c!kGk-yT7t6^4Q@3Mg%lqL$I=}{dqA`==MCDroKjcE5<-KGB6whZ|<beZKn*KAXofm*x)5sPp}!in%nXI48M?`7kS'
    'vHzt}F~hQwzhqSIO-+-rhz8X;zHpx-t9@7}nmi+Ru=qBCl)*Te9q}>EO|C_FMYOCK0#V)oNA#}8J&0?dh5d^`=H0--8a6OK#5335'
    'T&&uXc!OsF!6jelVZ=6^n3Gm0r^X$(d4(eva@)nx!Cd`W^D$bUI>gZWf*y9ePIK79G|I4^;=kd2FmbdRR=el=YsR7jJ6`8zY<zcv'
    '$(&psRUN^5r4w%%<TGyGkCkt-kVpY@k5E^nU1+S5Y@qvsxWgk<dbmG_8ck*1*POnMr&bGE<QX3}CpOVqZ%nh6#A3^XN=KK;V8&KJ'
    's0fbxs^P>)B@{mFP8to`M9+`$o9JzF#>`@O<^{n}uo0rGWilD<FdazMLrZgKMbgEx6j0PNkwMjvE`;rq2MrsmEjOHex+e%0yrllT'
    '>RNa>ns^>)H($Y&4(5kb0)?P~M4>{W3Z`-uxuP0WIbg7n5hkq+4cVY7L%9Wd{O6{l-Q}3c%?hB4EqJkIoL4p6ty6#<fiFzRaL#Q6'
    'MpG6tE(+dnsQg0?lSfuHt7z3XOh%&_UuqE<B(9nb+%pylhtd#<)_Yo2mu!+f%>@b2HGhp{eWqm`;mt50NSM%|pXhnRuac+|j%+iw'
    'mIb;#cWkUer#5t95Q`fg5A_dWZL&`(DSF8i(GdH>_C@vvoe|KY&+;lv-VX%RsWWQ6DxOc;OUAHK2B=jxD`<(?0}a*?@v~o{6&L8D'
    'xr=^guuC^SbM+*iR)>sgLFk0Z<=qyTob2<%t_ZIR>+U+lo`ZhCNDUwbYcvcB=asmh<POSf-P}q81rM4Q*awA%<v@6o?t-SpwD?6m'
    'hqhW%6Sb<Lx<h&Y-tE&CbtoPXoz`w}01P!UruogIJ{>>-My&y1Q#yuDKeQ6MK}1XhFoJC+Hc|By_>18)njD5e!$c4i9SycHea13A'
    'L|0?~unSUzl)<%ttjc?y%HA}1<HmYIDR@hmrOrEl$QMwZC%sRKEtqC)Po#^%m)aJ!zFv<5U3ztG4Gt87F8B?uga1B&%=d;2Fs66s'
    'm9)m{S*Dp`m*T8PGCH6v@Q{G<Dw*a7MG0ud=@Kq@N$55Vr;ii2eMYPT5GKgudF42MYZypBEA$gX3uBX7bwFci$&A9~PL`mEJ=0Y!'
    'mpe#okvW(Nb1u@5O|eLWQoAq>ID{5ppNz6DmblbNSlD1dy^IL>uor5@jMAGF#(^@$>{A9R#nn6tHkA*sF!t6p#irJ@TgFqZC$eNF'
    'Men3dD6==F5>saROxl^NGdfw`STjx8bA8wgq(X3;bRthl(KGcHGHnuj0^ml_mIRjz#2(W{oR8Z~N7PV>o5Ta=sD-3=a1EU0(D>45'
    '&nlTmg>lR@2eVgSyS0B>$|Wf9J!emR=tjj9?l}F-MfDtY29zMxgpn5EmCV4oEXt8tbHc*QK6oFdJS4~*r6;?TD^u<cxgmsNEZ@q9'
    'HG$O=5Msb{)|7OHBU>djt);@_`5>>i2~U1>W60ID80*|@+d#28(ImohIZLZ}!s%6>!aW7;6JiuS(?m+XL^{T&-5^gTQ5xcCkisE!'
    'h!3%tJ}U2VLnSS84EsFxj^`+O;mzp-6&N6y6S0EWH{U2-|C5b;n74t?0bx;)jleKsei-(-QeN)-w&_oB{sac%(&hPv+gVBr7yl6$'
    'h+soUQvjVL;h8+lba2g##uYQ)ad5@OR5TXwwl%}S8g<Kuq2nrP@l)-<5ZX+4kJWl%wjiOffVl?QT*5F^j4w#-qa|+&mQ+%$B9~ff'
    'Pc|)jg~n;Xth_`j(_D8pgxCkoWu|l!a@dkOp~K%PEl~(8wKk6yOTpGuTTA9WPHB68gN7`$Z0~dPPluy>xfw)8z;dBWHaJz;9QlyE'
    'b2*2hiLs}`YPCgm7G54u$V|Z4hGUepFXnrN7cC$?kXOw4N>XgysxWM*W;!4`=i;?A4h-H;xP4s_b_tKzn)N=tVeLCy>}5128zKW#'
    'TE2gI47UMJ#-U^&`P119(elYNWf`?CU<RZ4YBYxsN3`3=@@2UvC|YbYj%~3rJqv=#&ds~&@#f$Fo9jz2Yt7pX+u;xa3C2nT<~xjj'
    'WOj9in4#=7a#l2lu47)k!nNod&^e)Loepp5rxLE#TM~kf8JQ`w8&n8Xy*ysV7!YYhijt>2nB-zXgo4UP5Gg$fW3pQ(<}$6^yua|}'
    'Ol*BD@;bO4(n91r|M~ITQ`0R0x=+KCcMbE@Hfx5oV6egxGf&AH|9KEY(cSWeRRrSEDVh(1-C7<dI}^#;qq_MOyTU$fvLLjf0BVH)'
    'VIQ}X#@dDJqzbi#N#>?FM?fl^&_j2de|%kpZ*Y-9L4cUeyG0ONlTp(RF&`a@1mY#yEbb0EDzh}n9YBTHShXpt<i0_C*f<kA>;-H~'
    'P^~nvkgiy+xhC#FLX;g=6=RPhNiTRG<;(Yk<18yHcWUT=qQ0S^Ldg?}F&d`fU@P18CaUo<xs~Q?nI2|)KhKh(I_f@ov2jANHDskw'
    'z(97a6nOeYCeiN?nhVAnsH(brF%o82T?ivD*|Z+Y6js-pV}R9lA_ks{Yfd-$J&{g-{6bZnNcd}u=~lx4n8893%Z!#Zq;97xz3iw1'
    'mfPon0&maz;!s=7ZKCr<w2{Q=bNtwguo=`aix?(e;A4N-@SLDS&2%P>0)#ZHf^8cYK>ZcUa7ZDnqR66}u4QAJ0DYi8ab-H@$F%?9'
    ')285o(}M+VECu4o?aHN-Af5UsIS;DE1sXntrlRqJYhWs|tpVOvT#gekgh{r@=0T00?8mmYL<&*YOwTvYIoG-YX(giyXw?ql4^VM~'
    'W8D>#DbwWM7CI3ED>gF>dKGh|xrLf66G|aOhYZ^RiV=USE9CHPsX=ce(<6rY2#bP&%?D=#BYiN?!%uc{fFR5FhcJZN*jqwL_-7>K'
    'm77qU0`7S~KPtH}42|urq0*T0%&5|^)zku&y|2udGRagq=08T~qOWF_pOXXBjO$f?np2J{8!CmePW3f9M^FSDRLS_8jYg4h+vF#Q'
    'AqZ=#u%K{QSNpkTc4*Mf>`P=Ftc8v=TLRLaF0#3n_HB%(KVH&xz906IEN{_xDH}(tWX$w$<>MZ<6%3xIO6qxol6~bfMx#=HeyNtF'
    '<v?~ubU?yjF)1LUKoeVZ-C}%I0KKmnRBlC>!dW;nj<)%R21jG?Miol+XU$Q0XO=vu=R(D~84jLQ?03o$;>t!=%m_2Wa1*)!g*re2'
    'hN7!&EU}Of1DjT=Bi2@H^)o`rfblz_-jWca;EHc4%JqL7dO|M^P#D8Z*$S8mB4Jp(GKL+Z7w3p9P)hUe0gi=!JITw<NCAVxS;iVJ'
    '5}{~?+n5+1E+Drt#WP3N>XEt36`zDN6G)QW#gq2+&nM`PNMI5pImK6Wc#R@m31LF=D0LaRbQ`36XEh<j#FG?W0~?0|g)KAhzW>Pv'
    'Lm@uAo|ZYetr)l4GuG#$4%V-=7YR~Z;r&Q#xZ#+VV}~{1&M=BrGe~ol*8ZCPI-dQ1MdS_%@&Z*DXJ-mjx7;-Lim<{1{cc_!dR<E3'
    'Mw+e6uu3b?vY2cDiHwg%Nz6Q$DwQ`CCbfim)Xx$}cPEv==UahLmgz@OLc09-?yja{M8pM!@Tm|{KuKAtw%8iMn-olnya%;`nL&Dd'
    '%)bE<AtNpgv22)7HVob)wHPD`huqn*nNz$#7THw9Lz)@QVK)f_QANq`2JmVK@CLcgD(lZL;SJ<t2hZqjG>M6*Z;T4ySZn4xaUIx-'
    '9Qs}Pd4NxZ<rc08E00PD0#Z7Y!H(W6oVNv}PRLL}V@%xgziJ&Ye#~xgf%Utg10X(-87rD9%o3g&5KSS&!_xMpvPm>%J%f6>sRnB3'
    '&Rp0~K|=)7EyEUKiTKTkdWBUxS@g@zX*&k0(Qr`rg#)5Q;Vai#S7Q#-Y?E-DZr1AY9VwrFI$XjF&~zih%?LBEr(tU}@XuSCVO#fC'
    'M2)vs!qPUR&Y&&p42oNVI(VoXmx{tw@k5yzlf!%0I4=$`y??XaOxAz{yz29BR45-4Gfp)vV26(YHa7214o@L~)pR9(E9J%<x-fSs'
    'Xy6Dja5HTXjn|?f3AhW9Rxs*tH?JYJyR3A%mSeC4HDxILl8MFJhUw39U)B5^2Kn$S8>H%B-K;l>^(N~m3z{dHmSbm99)T5Y-T2_G'
    'it3hzh`iFBTg8c+*4L5n_x!=9Nj@0_C4piaVNJ!u|HfRIUSYsA;QDC~IF0c~pTDuHX43t9AQlSbKd={YW;WHZP_%`Y1i#NQmTjx%'
    'zQK>C4b_CbG`N>MU%b~H0PYOQ${ga6YWnTYuAAD%O^8;cnGxbr>oA9=BQ$$~z*$0E*@3(I=D~cbJuaSKdYH8USE%{5&O6d3tb-X9'
    'Fuyqr{~`{or+)y$D#ZeUFV7v%j6ZWOF$*x_E?_(0auQ<9EBijEjN_Q8*bO=b!Q44pwh_@@DNFK^Z0At1PpcCgV+Ds3zJqqnFNFsz'
    'lO)I0C{80LO6E;+7pz=Bz(HOaE7n0I=^`Tbjp~Vip>DBWDB8?YL68&~7;OKegwy!bB5Tx)Jxodp4~RQ$V!1N-CL=7*&^#`=n}@33'
    '+zlS8i6bEB=Y6^18Yy88C>}$67;tPh{odO4a!Hw4&RPK(!zNvs6+Y5op(UGJ@p+u#dX8K}sMS-aPVdHgV7eaQ`|OU;n`FJweGJ)<'
    'htHZ12>+v<Cu?xVzm8RhI7?@r1OF+Sw-O4OxY^O$JB^)lXESQy;n0}(HD5Y?<<?>k(2vfh{SR7{s^gC!kl5-#@C$+LN`j)C!^{ap'
    ';awFGvC1~<<j9gXw-p>eLPUOz!!YT&;{gExLk+|97@Y4LFcUH*QS)JQ&}>Bav55tW=VuZZ9YmZIy<BLMFqLpkkADhFlCF{<?q{UZ'
    '3BxF&bCwemVg&|;5!~b|q*y6dnv;vU37bm<W9{zeX7M$P&7`KwebsaY_I9#Fo<M?~Mny}`fNKFYGqrf$(E%)$B>2*&E6QABad43V'
    'v$5|hcrGS_d+7+galiv9anON^^P;{&d(;^5@qvIAD$PBfPpMDfTqZgEK!tiPiAnG$5fx7@#ksRA;FJOi5Ih-$X9OrhKNKF{e-h)Y'
    '&usNlEqmt9u|q&^&D2PmTqR|x+~zv|`X|4BP^o+*f8aUrmZFaX5`<*AnUSDN%gy)S^y?q|`Z-$Cr>q(`c)sU^6t$=a+sRnW`|-Yh'
    '&98s$>!*kgk1(K{DGVKP)2}qXU6ij6_w`G@{;{v0B7?bxuShg6!Yf|a^4lf(`fOjn<*$G4*AEhZTomXgylkk@hB@^VFTk5l0yPFF'
    'Ju*eWdh2COBPZp#WL$fOC~$i;qWrsn@54%fcqE`;#OF0?kg8#=uT0cryUXhstv5<7z=u3|?FJS9Ee+^p3b02|?LfjgD>CLryw0`9'
    'sV}s6uWV{n1GLMKrw4as+D!F2NuLLBZoJoMb{0K-1GrA+{y+#3FK%aYa>XM9?u;{sp*7Gn)0S__Ja`rH3ip{VK&B3>07hnL9yT-&'
    '7^dGi{5>&og`Reu9;8H@LFo_7nW<xan2%O+y;~jhkZZ=6dA{kk;)5UPbvVnPx9$=C8gzhE{zpD1=0ypqz5rkM1_XC(9o$yMSc(F;'
    'i<dUt7uTX9?q-)t?=|{;^o7!5ogfk5H~~?#X5B<u8yCm0+)BrbA2okac8WqsddGC-z)QuMz<W=4Jcc0;1yW)9g;G@J)t;18VKuAx'
    '^c<W1CgEn5%ewX*_?t7+X3{PiJ~{)t0Q=m`yrxtb841YCwu-;IzF}}@yBqcmUX_}KJqBFDPy`4o3&c#fRT&uvnd0Nh0N@x3C<l+r'
    'qN$!5^{K<{3!V$WyC0Y&g+fZGX40GZD%MHrJimqVj;DZK&S;Vj?$o#3U{vxAi)C;-$%zGCzyge-iVlq{-EvkilJ@tTVOGx}#@qu?'
    'Xs(nbV;f~B)QLb#+!v-P_WLZ;zESH92gTI{yk?Lx(`fqT=l9PZit6r_`^sH3forq)>ZZpP)3G<FU9}J9UbU-WJ6c~qIEBySkx11P'
    'k>4zwm{eEo)JRWtQTVc`Ke)czndWOUBp-3JGNZ3%NNgIILQ=|Dh};(x1XZRPq6nImZ4Y`cXX#5<vypb*O9GO|xO)}@D!dU6!gO{`'
    'lsnAd*ku6|qtczZO?eZf755Qc)l<o<e8hK&A&tar8Yt!;MitI_9})nCV2^1`FwZfR<Oz|>W3rdA@{FZu&p+-MWOUe(&I4wLu9AiB'
    'AP0bE`cj7JyZfnsRh9D`Jw12F!RDuP3SnqmN~@a7DPCpT@o95tYvaY?VGmabctsdB$bZ8W`eN)))g2D@Mi(AYX`Vl>(1Xrec;24U'
    'Zm?dtA{RWAX2o#NN)kiK!lXn?MLrIS#c-~&E;=HH9f*g;&k>Tb{hw-1Df)GYPon8z9SvoAAi(Lns2>ngO`~g|m!SqSe)c`o>ZqbU'
    'jtzF<ag~_ELITuWT4@+JV;$Jc!*O)w+AevjQ0frAdTGwqJl{AqGn4vY(O#4*^?KHr9JGyI81d5*pbNKoSMk5ww2-7K9fcoz-;2r?'
    'IT+ZY>z|y<nK+E?jFBU53C9-_f0^R!A6!cy>nz8)9}L}4)wP-oWR?oxZ6LQZH)O=!W`A9PE!7^?-Y)bCj0<G58mE>($%+lz#dq9j'
    'V&{|WU>Iw!SLHcDo>8$kjvUllO<107$JGTef1muSiVq+q4@G=Qk{8O{9W9{qkPLbij^k=DkaOvtfgxWYbIYRV_(CjxsB^iWUC90x'
    '7Z@NG4Ysx~Iu6S0g{Bej;1O*=1%9Oclw_d$lTe1B<Y9x^h>(zN2X~1p4)JA(bDtwyj#)d`(1Dgh?_&3(z!P#^Ka4?iq6(GjdDxF|'
    '<sc2#tD_K(znAbSBFq?~A%(6Y=ps`jy1#2x*Yk72Q3OBDyz-SfATUu^j%!i6YWPPRGbKo2uDU;GWO?4Tz0Mn)hI6D&nXB-SHSBg_'
    'cja97x$E0%lOznLi{iBgpg-wyygO|()MC~y`%R&zohOdQsNr@%eFi$VGUcRjrem#p<$}|Efpa4!da4b#C*RYPO|z$;_awpPEFjpM'
    '?VicmQ1>d1)Q@wtYf&Z}kNG>tJ2~~!kv!5%<mAMWGuJm0X3q~CNB#GneManPNQq)bLvH6XZYH_gI=8$MGz)QbD8<~9TEo(obVFQC'
    'pUpfu*MHrE@kMTDSv!sJj?W_+Aj`f%?*}Ov`=;W+w5A|V`$U+j(Tq<Q<Zx$Gy6T?Cc;#}fEGD?T;pi1nq@jm?Iw2;YEWkLm=enE4'
    'lN;Wp(x|a2z@9lci5LNiZznxiLMg?OM56c9#0VKHjWAPXPw-99b!C?qg^@y}Mf>*e=9bnQh&^&y4qbUkS7Lch<^(qH*%X9B<VV4Y'
    'rFG{IZjQ^Y3X>uMiLBKE@?RwSkjRj`L9)$(+<4QwGT78?3CM&@x!#AtfE<7wU2{3hm}EDEDF(gX9#;_#BP{mAjl!fS=kAz|u(?|('
    'K?)0FQ9D5UzKBjr9--S42GSibugo3V^;B!Y$u9<2MoGncCdXSMqxfP(0ME<9{{%pXm(O$ST&T;Wst-oObKcIEW$^)7&WhJm9{5ay'
    'wK-?&UYQ!b&dfz<d0n3XX-6lHnSi|q%tLnA<^epVa@7W<bXH%(SZ}&M)Debrv;1Ind~tJB^FVUgJ8tqP5F28&IQzy*?_p+o*N9vV'
    '3sjUtAn<l>RMpWkq2Pso#+_o3-8^hP289@sAl@~Oq_<=&2f2&?r=vTcZ<?=I)lxny_%^~xz1r;QxoK<+5HJ*c5m88{yoxngDJ?|q'
    'MjB8|?Z&X#9JR;^<iwLSdr(5QyRP7s>_sdI7OeKYTqa26SX*JZU@0P}tG<gK5mqCN-4T~nQzy0Dn*H(I0PZUx1Oi$@FDpFQtHYv('
    'wXF1Ol)a2HI{FARw?sD6H7<QiPk)ndSLE%xzp^u!fHRmL<k&NPo^0(Y8UtTzGd|}{uIFHnV650Nwu>IgSyjIYO4S|i&Nw_oA!XjY'
    'l4$}WHVQ%%m{s$2JH|FUWa@ci6`<fBfzmykO&_tZ9Au2lY;g|T%c}l`Ea`ngy@cu6LcW><ZBdz&1fUFdXuUallR}C-zL=Rj(tgt5'
    'nw0)#O5SGF&?S0y)Z)H1Urqr-Q1JsOzs!b|^PNWu9beCM5iS@_rCd4VV{z~zHXHX-s5s87c0_A<(h)3y&ZetcEWd2Rx#c(;PknXV'
    'J4dh7PY10Fg=(i;BjDuub&k>D%233V#ScatTMM8UM|ROfb`t}OAkDaO2S&#W7~W<&m-C)uy-{C8PPoO0LWnt{srJ(Nb`zK`dG_;W'
    'tDH`|nIi^=jn@#0E;6+`dx}ju)-lliSk<2MnpC;wni`^>6R&z8O{JRD;Rbe|$OowFn`occaRsZJf<yuCugfEzmgn%^GN;`vE?}g@'
    'p&+k~c4?r}0Jmr}h&)A3Ku<XrGn9@BSu!j%bp%@LP8Bw>fry!=MAV!)bf#h=_DHzGHMxkfXAt)EJok>HrKi1Ml(ijOu)d}aGKSLn'
    'fbN7Ym+xi%WOD9NV1}0m+j_{(7k}&5T@8F?UNCJYi~yOp62^*25-^IXnHi;%5hz0D$?9-CFQ{cP*zVup57Omh5+W14a26_7RHtAz'
    '91o+0A`mBxQfFRqlrx8}5gGM!gWt5y)jdO}UCuIyY-y?3{VH4l7Ip7(=wATcnz=+f7u(<>n7k|s9=`O0Jn#2-r^;|n8k4vsHl`Jo'
    '7*v4BiA-w+c057a-i^dU+WeGLX`8`<%RBTi%Wk6gG+CkFYd0ADH92)3Kp}$dw^?=h`f8bwYmvIIY|V)5)^9<=OnYfuZ4P-|H}V(Y'
    '^5EMu(~&PFW3PYiUPLywEuJeQea61j36-S0=VqKl7pb;oLd`iB`!yPU3bL*Np7~edigPSs&EY4VpY(Bs4*aAZmHf$?$#;CZf!N*M'
    '64k822{igZTHQ&(x9+pNN$Y`mmDvyIU@enqcVr)%m`WJ35I)Agh&}s#ZBU4>$rtJ8cnz9q2ubIRV=}==bz2Ut$N9LIJVb~_?TDCv'
    'cvMYeb*4GT!KQ{Sf$V^07EkMc)o`#g^lMOXpVEWn8^Tn^5RIBxXr&5bqS^n&CQ*krv0ITnr(ntDpU8-L{n^vO|1sgErNOSl-xiDV'
    'd3(?nh2&F2HOOs;W(<x+6tl`ZdzIP<CIVNimJ&qVceLn|3pocX(j}>|*fU{%0G7O5GZ%zn@*Pi(P46O=89A_$;aGKQ=qp?~F%sKE'
    'WnL5jDo7J+Z@Pw0b<VD1V;OOwGE>Q3FdsfA2&x|uLX;NFmT9@hxt&&QInwH8I52qWZbL~(Um|INYEnNz83?tiwEWwQ0#QvU3*%Pz'
    'e_-sUnHPvfnWZfCTV1_7c-i-$rP%DUEeY(aQ8Y;)B~sv0OXdd!GQdK{5&&RCaQoHDJ|6<*)-nG@S2Z829qsB3z5@X6io0tMApozs'
    '67&{HYF?s@3_u1}GsAtTR&o2rG+%3Teb=D5<22ezr^r<88N|_<La|D(RT~N&g=%e-VdKt=H8`AxE1EoBjvN!?P;@y0Gir&QErSVO'
    'OsAFCVm?qBF^xo$!$p_Jg8HxXoI?pUVOfTQu)H^|dvpL5_z)4|4*N-`Avjnx>xHfWy3f%AMS2NfyH~CZ*B4!a%0UQz-Su8FB@TN^'
    'sJX(Ca7RBySHE*=<{**;_G{hqwA9<NynW4bxtr-}>Yw9HBQc4u0;O9~P`91J7mTd57P}_w4^p9~KaO+ayi)wal#z|LF(W&pE-=-X'
    'L=0Q@+c>Z&KCF=IrDmDZ3_1x7fJwv5Az=45F;+VRDZ+_0hWqj^1<PVEx=_yH&XFdK%E35w96<1p6y0Icc|OE=tt$@aJ>tV<Ss*8?'
    '-b`t2$WYjj{vFgB(lzgggdI9_(8}_L8SKP7OY1X`(vtGDGJA~$yL#0($)7e+rtPMt7%XTo7lL;>9*-zDn$HEqIC<gY?8L-kPOax7'
    'd`8-I_QA`Gh6m49Guo=B!|K&D6}Cjfd{DX!Q*X*~<P|I&c~@uh8&=cyrgQAMchtkAL9b#EWSaGy7|UjO(6523t4RM5{LKAY=pma3'
    'aGA{nwC=8AV-b=NzfEFvf}Ej&<4RdU|DE14RIixzi%%CXnge|{037?o1z@dd>rLgh4jEI_QEbiP10O6ePx5{Eh<F|~xnx%0RsS&a'
    'fj{Ad!z5;-+Zg6p5kQ|c>CAXH_Mqbvr(eulnm{kKxj;9Rm6wd<8N5&6qp{=$D1yuCe#J$aAM5x@P#vQ_4Aj0l^+gzX$QU_Eb>s8g'
    'GeHt)D9qO=O|7Y;5&SqCy<t5MN_8c>yU;w{@}(Kk<Ps2IKpAQ0>=-<U76f$G6f)-OQUm=O9VTU#i;CAVqraELHONC30MwkzHs8AU'
    'k!*yL7(#)=%1koUSIpcBvq$L94sCe%Q6aEL{lT3B`x=Ve3HS+vE=gs$Wrsl{SilesAjZX(*vdV6BNQETg|@Pb5ZuxM+CmJ_ke~iT'
    '-cSCQuVyvwR&4vfm(1aMy1JvOr4f82U<ZD{*5!K_#W)*%lHDE3irfk8RM1IO3KzC_aIoCIaW$)(b9ZBK<0Z=I5?Qcs1&M&j<=EYK'
    'Iw#DvXZ+@_N0)IBxPh@RRGl~7DVbh3Lb45<IVRK_>ct(eiUFPm=MiHCO23*hklNlsn9Ee&j#m{~5Rne|zND|Y>;AyRI9>o#wv%P6'
    '1ubd4W~4n{gdt43lR+2wB;!#SpKB=eP1?cYJ&h01b3$K#ZY-_uTCXj2Kk^D_NerKrA+r;t@TQNWA+KOL^CN&HFmJb{Af9D2TO3NJ'
    'J~H02Ga_8>ukRk{uoQyV{#-`r+RKmrd|L^LvnrSt7Q}uR3yb%h1>DBH`gWP=M7IbWN?W<7ILQ%u@S>X7V(KM6Z#REj0xlt|Kx*ga'
    ')bEuW%miMnTNm~>Kkj{>2<w^)$VzY+fumr}*d7cvxOUnd@GfD$QO4gj282)6hYbu9=r|c-vn6!Em{{zfs}DWVt4#uQy$ZLd|HABv'
    '!FZpni$I~8_|7Ew`FnX8O+^1rKvvryI0^-1%h%j|n{BOc6UqRW!-T-18k7YA*bekL{cbhe^QTfnSci$LkYAuSZvQ`uX>uH%%PLTo'
    'r8jDd+V7RHD)6{d+0vvX@CVF!4E`l}7No>dD;}JO^G$7KYaYD~E{=!^&2B~ofV)5?{JiTxl7KD&S%bbI)_gfDksKo|4$Kd##G-H-'
    '(p?@Wfgzj-LSBYlTt!`8sYzz^dPUpw0_A8ie3>IOqh1gpVx~mfS^Iof15Uf$Wh0pQv$Z=I(t)SH^A$4h+E^lpI+|mizv|V2=k4Dy'
    '<k(=N6siwXyAW@y4)7_tVB3&0&l{og-wO1a%UCSw{dX;l(Gs;6dK!6Mj9JXZX%JDkX1jdej-`Rrf>T?mdvFeM|I+JDpMH@*Ed`_+'
    'H7XpC^HS1;ntNFwTpLA@h0jbEggu$J03w3PeZm@i#QMA=rtzFV@5PFEj>$fI0h~Ke8Cr9k+nI$uXA1*_f>U7DgcoOpzFV5g4lES}'
    '3%Y!k-m8{J4;>^g71kwYeSh9MX&ErYosQKxWc@Smkeh%xqO<=DNvL{5U_6cfS^;Q=XGSMehm2<1LWW6h-(WO8|Ag=tQ>Xg}A`}{~'
    'XW2S#)rz@94ZSjw0bwNvpP?y|E?F!2GyO$$Hglt~MBj2o7>7&?%V|S{Pz_oy*|f&U&kuZY)+RrKZ5F-34&i38jZC5D3evZf1sCZ!'
    'MwX4McTyhRm>@t>8?4hAn{dYv)6qC%m7N!v{bFD~udb>p<mvEg^c%{hM*a}s@jT-&_WH?~nI&HimeqW<dOA%Pr8OGpK?7+@uPV<f'
    '_IbFf0T{5Uly>{P7NX|&zu^(8O*USgnvwQ809;o6C8kJHuO#8>YcI%)ax*bi|7(IepgDXPoWPaMGK0o|Hzls8_7A)YFAcT0x;CU~'
    '7$GD#&JsbMA#5n{+p+NvUVE&&oGJk_?rRSbbSD&3_c1m^@m14Ib%~Da1Fr&jJJ2pb0G<+`oj9P5<wUXH@;|0y!yyU=t?_FUjvn0|'
    '*T$t@Q`pr&bs*WEyksVeA2?mGr20#1@KG0mq0aM813-xetNB_$aJ*urn&A1wEfbsFY6snCxjm16z;sAPSQVv(2^~<w0Rlr7q<I@m'
    '2$w>$8XtHOSn@MIk2Bx~3U-4)N6+B4S`w{UZYt|S%J?huRSR2h>!J(B75R7c6E>_9XA-$Atg|75ep+%}0$M=riMgTRVZKTkd>DUo'
    'c<jW$VMHs2`8kBU-sa;HufnWnL)5nWvz0@%t3#eiBxE)_QmyiNGzz5PM2RR9+IXUf5GjU+W6CfC7Mvnv(PpsLYoUP3unFK{9mgab'
    'RbvyR&j1|XeL@1u%`KZRhZx=Q;3t@QZr(pikvsEgB9mk_KHpA-xXSQy)HkJL5~hN)4f#-~$!F_WaeKg!CpEc9_9>be43cFK_qB9E'
    'gvNtLoYeD2kDBOpoa!OY2LJKgbb>AnoOt!8B4rJjMSZcNWd>qNg&P@xMl!WQSw2iUf0G^pad*Z}lhc>9RLAxG&Q=-JkonUxEQ-c`'
    '#kakRdVsBGh24_jPdh^}4`WX`qWX)T`eIOdr@xg1cQ#JIf`;3XBBPRkn}u|aIHd{Wkl*L2UDlf}5X-um^r8Xvo``@*D-`(qKAgg+'
    '=dSl8j{Q}P!-1kVg@C}Y6P3tfOz5JX$8B-q(5>>?>g^>LAm18Tu20f7#K$*NH}>c$R+dYwIYGRVvozi`1b23q9PVO18iNQe7<^pu'
    '+h{!*yRicTFnh+`9-Sk}b|@$><fMosS*|9ZyW;vfc?%=k8uux&lZ8mjwzzY8<6dm_DvMD;K1IO-6Q<gRujoB1Z|V$YS`49fJ-MC_'
    'h4nmK1w#WTj^Iok-nW${N3t@$T;{#ABgfdNfftIZ(UHb;W=745E1BH-M}SyY(IwNRVF9Gp5tNUX`3)Mxuq{kpLT_;cLAOA+PR!gY'
    '(pE}3xUiEHL?E0++QlXY)NyrK+i1;m$;=(oSa*Yhdkl#OVHLaQ)ijTm`FC1k;jZcz*3r4|Zk0m}rEX<vM1J;g>{>8-zUR%15}<E$'
    'ZI}$Dfi0I{l}s@+)L5|cqBgJg&N6t?`VgGV8Qda`gf5;762R#sVDeN_O1AxhQ$b3P4qtxdE9Lq4&+qK;hg7e|u%D;DsBEOhuP-SX'
    'F%5Jx%zt&*!M{G*0#UfO*V)8#fJMB2;5aY<>J(%LLAJu7shKb(P4{FqPmF@Bd`VouaMqpUgk#j6!MyBGSwX{{ph@P%?8P{&hKmWL'
    'T<X%lngiaK6w%exz2l@n(jrCXL|Tg%r!55~g-y=ujRoqE@LqHgO9yA?osW&2jRqM{p(J4XVcO_FC>LpqSDPO&E$$fsHWeg*CRZ;4'
    '-yc7Yk+S_#nnD7o@Ib0$Xvl2S93rH<MHn-{?3STKt6jIHIktu9J~e=0W+TG&nxU5mlNfnUM?x|T!xtXz`J&C1tQkyJRo5Wi-Z`_H'
    'v^L6Db#yZ+eYLeha!nPME1qUyhMkyEK#tMXhEQ*$S1--oZ$Qo@s)pOA=MC9Y`iScuA8;*7M0=jiysSstF9aO10!iSf7bomac^?;w'
    '`6RL!Bz-DuHk!e<zI-DiQQ_P3t|q~vpO!*pjzk3NlUkZ$vyRvH4}ZXJW}gA9=hswXi=Pa`0K=sv?O|@?1A6(vW&CrGAciILg*A=o'
    'v4}n<Q%X)|EQXB!N)@9|p3H4rSyPg`fp-Qw^GJ9ZnKAxoN34y5;@}$GPDZXrO3%piuhtu1LQ)_V1>TIOo1?VI7^m*Q&~(Ok)!e0O'
    'T3Al27NnFM*`Y*-1!t_fHY38F=;6Pppfo5F#MyiubTp&+^4XdCrgF|A5k<~ah}k~y_yZhD*~{<DAj`liLZ16{e?2Vv58)|vl55Vt'
    'i!jys9EaZ;!ZPYTygxkHw(IoQ>#7CpJu5^p-j@A!%5jLhEF>Z4(ArP;ekF6O`;x25ac680I=;R>aPx;$anjCJ;w(DQ#Xt@<-G$SB'
    'x*Sy%fHJ;X+0N_df)I#X77PS(0d#qR8Fxu?)`C~HUo|fP8<kg;eFp$NqpY771oB#EHaeeaEck`4WyAu*a5?iu-m@z{PHCUp0<l})'
    'Mk?XIGgm(Scg(|%|10XF%2611fLV`7D1hIfzm?Gj%*zD`Mz6<nCbY~KItq92d6XCUx>;xQPc8G(R!fgr16FtvWk^g0X9_2Npx3D5'
    '3eI&T84LRK8_#crkaiMoggl;{HpMxdi<MURIo6?3-1pi!CB6MI!pMcE1US?agEHd4;$hlXwXzV*faUy!ovRqitJG@pP)q*xNRBdI'
    'gA&vDEIzC-ZQL2=Mdf8X4c;iC9AhfYBsq7blt((Y-XLD5LrByz^_po_`n$bcT@Xwz23zhKUJ;W3(2bhLF-;(ZdsWYUrB~wtFtx>o'
    'l7)xB{oXm&xIF75M#V@O#JU`|c_%5gsIV-$3}pwT1B!VTs)*;Ykzjt9rLb6~gM6u5NkyFaoua8$9hrE}+siVgg;xMXQwzTUqm{<X'
    'F4QTVXTE+it$E%oIyF7SYHpFNl=Dx`KlX2i@>Zq@_$c6*q=q)|9j|V@AF(Tt3u7@~pc)Wucg}sSYS(O7+5@@RFdQ%}mUbpXc4N1Y'
    '*_eQo!4GljuaWfGZl!&hHR(%Fm*0%kZSd{`fP5y3AcfrX&Xt2{MD+=xz}zt97>*gqjyFpUvBklrd@&yOypleYcs=Ez5-*-|8bjGT'
    'k*#n67DKWAT;t*7Y0uYT0%-I3GMWi+<e^Ro8fYaf{4vj>hBx5lP=Z5g$bVI}c>Z1_mXZIUntse1Rt*LZAPb=~#J4|hhhL8c8X)da'
    '5_a-~m&}I`VCj_sMcB6ZHm7onb+@|}-(sjZbh>DQT)>s-X-|qA=LhhrW~;Y=5|tNsz4nnCMSNQugN1z&@))%vm4v7mna_0`wY{X<'
    '%(XiA7J-JFo5#H)jV1MH@2U)*0<CEC22}9c!7(jd>Dn#V8pMRgN&I9!5hhFmeZ$63cKy<evhazxNne%8F#VBCbkY?##ON4!XForC'
    'SQ_vDh1{aR7gSluNX{WHw_ig;foO#BJ$y%OXzR6GypMRLsxHDGaU*Qxu;dn@UfcGR{TjnrxMc?0%g!E|H&<Jp3{x#a&!1&+aq=70'
    'EpKQdO~Sb2sNeuwlMt*-IXyWKWb!A`j2H4Yeg0_@HguU&UWzHA2WXP^Na!LJMbo1-!4i*K7=NWv-JyLz{OSzXoJy2_7lrPCgixLv'
    'N%Y)E-hzeTt@-Qe9%yz5cY>ew{4hT=&xMd_dWPYwqBWX`<o$Z9jnr>e|9eWq8u~X=v?}%~{%VYbf>oYpV*J`%J}*Ng<)%tN*Cs>x'
    'WMplqf+^gGo@t}M!t!h*GGxKbN!Mx2-oe<@!`EXj>QigEZ{@aZ(4EsXvg38RyMTb~TM6^a%xmP2+PJ~*u}U-a4);kbFBJ<kHOAvi'
    'YgF_PoljVErzGi>43|>!)}J>SCYS(Jv%}B`7~{N*A{iNgKH<<yY2_=6f^TnM7YPBunWg+(YpMmDEtE@C>l}JvUK$^AFgNmjzRR6}'
    ')TP{Pk1N+GW@bPy0898(9n4RRsT?kkc&+7ceyfFFKShA%;jPVg*)aSO2<qC%0I|dSCIDx}Yw`B=4}JZJ)$+guILMdL=8fx$$768Y'
    '?5`og{ms7qk*}Y?TxadP46euoyYiY0fBPg~ztz{j@%jmK@lNq>1q-xq+QY98|Mfw>eyy*6<=2lW4xYXL&(jGeJ{oRE25=5QnL><s'
    'k2lh&&T+;>0m|Eh*zaY|Nf`GiFN#r02CSgMp_PJ3%E;Y2vFJ#HO!Zl65w^WNu$`xb*)Ppz2bu?MZ~enlhH#>JRO1<CMY8{z_Bn71'
    'ZBB|DbV7EwVR(wwDzWNVbAib<95aWM`HCy(OgqAH-Zb3_J3&C)ZfvBgFOTQFpaX$HT(jnlL<`mrL#xTsqkx{veBmYR-hSmdE(wW='
    '3k$>VM+4*S?5ZC%n5a&pz$q+YGT{t>P(V=jE-zpYxd9j-WmM5#;aRc7r*bqDnxgip7FlG6AxCv6Sa#GoGFggET-|`c-k`wkzMxK8'
    'IS<7ILfg%Zxw1CEW5NoSq~~TG!%~*ojw1xOF}=0_({u1ZKu0b$sbR^8Cv6LO6Sgm4Pt9kIhYvCs`JwDI9A%(RAy&#aJqmXk=4!%^'
    '@nBqMh9ZWjkw{UA?uMd-dm0G>F<t|#v*3zlaFyJvWe!^7h$D}w5xd1bJ<$P^#QL07$3I+6T{SP04L^%jI}2W(NEJf7!EYew5g;9C'
    'Xiyaj%O}z&XC_}$^0kaH+cNZ(o=s!lIx8Q3Xc3r12X~kTbH0TmB5L+dHzkWM@-92oFA~aDsRn-?iVAYP&OXWm6KIQ;LFoYH!BjYg'
    '`p+O`#5{#}C0Zg6?PIdC0|HW-m%&Dh1tLvm(DEZSZ6~$a{~8erB908@02<*sirNl@A@+Ul5JYQFkuv-OtIAD`%lkLntl?{sa-3tI'
    '8Z%5xaL|AONyIjoax;7&GH5B{qTbjxl+>XKQ}y}CK~UJNnCJ$bj@u8IU)1|eDn*0t8M@yB$(Sw<DhOK&1c%d$U<xF^1&vz<fthDV'
    'E{ww)q#<(|q|G0uZGEMJ$xAm-7+Jf*CTOAihfWz_&%&3UkLvhuKffZ<!IjiHCLgu!ydwhJ0pFzwwPENsHw!~4RJ(`$;Wpok0CAG8'
    'rUMFY&t}s*(?1>3Zk<ipUQiU#W@Bk(ZI8W&-)Ql7Jl8+Gz)y>P-mt9Tz0~q62HQ*L#%G|bI4<Sf$)S)MAd94$(&U0{@{fn9y-XFL'
    '*0a85s?P$;2J7ZABY9b$CC|pDIYCSD9aMu@H!)CTd$5K)8TyFJfcGnJvmJkr%Qc)n;kFDjZM?n1R6A`X%WTL3^V_#0L?Od11ym(E'
    '3rX@)YCsAA6)E{U#5_So+%P@il>`d_tl!VA!#Z<far~@hh1tEG?>oDHghYrbi>Ciok|{-B?CN13hSIa<gm&|yJPb^kp*Z6P7K*{U'
    'QUP>`chez;entq%Gf%obG#4XxS+p*Zz%Jls#L2kD1r)gg81il@971xCcz61l@&*{P$X;g%2xd_^(Lk|><emsdz<lUU$5~niA05+9'
    'SzL99FATIumI(*Pv@8fEp?i}LJna`pB5}!zV|FmWc`^U)oMYrhSTtD?pG??w8jH@kh4F%DA&d=o=DJ;&b{b*q5jA<YIu4SZUC7o1'
    '*2=I^ZZm@-l)r8Kkd%gX3;=<Z?b<&CjnrY748$&4b*4Rt2n?=a?1QBvR3KB}LQNtH<N1}>rJ0!~Wgb;>+=ObYLBKwt$OhLLm#{P@'
    'oQO53i;VH$LJ<HJ(kBcDyX097h)<0`oa|Iz9!`-bF*(Z1Es9|wz=`915Fo*G`69swtO@KsgUzHB@_q8ocwV8yOjje`)+NuHddK8_'
    'nAt$2-hcj<<uQAn@(j)pTmTil8QO%kpoK~*zjV5)B9s<!)=HGJ6cyiqRJCB7o(FPBj}A5tJfod_dW4LX%QFmjM$z`1dP5Fag;CcN'
    'i(ZCgeNTQ`P=~V!;<zhU-lwOEE<}$@`WgdG#@5CV;Qy!wqQWeFo~%V{;NOh&pK(k1Av`AcN=<>{xXByo#jtbg^6%*+x3MEtN=Al4'
    'P6eO6n-*7Spt`8zCBvT9M?Q39iUZGeUQ<XyMU1ern&Az02y$y|oQeTv0D~j^N)9B}28d`l4A^YO{`|>9YdHZ1@@84xP;1*ER=CeA'
    'op23svGAZFNBFvIgbEQ!kvphmUs|afX&V<fbvH?w5pZzVa;wmj<0r3We%90ql>hKE@2mGBF?oU~!u*(*1uHN<y=sQ0EiwR@{r5zv'
    '5^01IldpKdgtbFO*ybh{T(ZblkfnReg5VW&5P`5$^PT9XgsuWF*YdT#ZxXADfbL0#DFYz`4-V;L=-)>9Fc?6kQIEBS1fzq8V`|Pr'
    'by1H$6^?fBvM=2M1*s~B-81XRjrS%N+3G}LVd+JL$i;M7ddxfOH8r8+ZXgzDR9I36IQ2F!YZPMxTM$aZqM-wKpvEY3BO%R+o>&<h'
    'wQ!Cio!jRPGOFhzdw7og$gXgz!8=_AV9T*6YvrJBWsG-JDnKI)0A|eCH!D$=e^V_Ul@ZCZU8hh<;-tK5AQR8eBg<;Hr9lU>tTNdC'
    'ad{ZbuvP7{aHzcyAr%G7{5`TS<$3)1>xGK`<`+}MjGXto2U{XtYldwx`CA5cFLNCjPB2o7OTi7zb67I=4M%b|IvGiKE19a%_GqUX'
    'GA}o2l?&e#niZ<y)nzgQ{~D>kFPw}N6Dt^FgT+8XI~^sCH(W5@Aj6$7vXFu^__;b@n~)ebEP{v$^kHfxv*u^GL>SMaqX7g333HX5'
    'hM&=_ZNkX0siYlR0UR7;Z`NLBzm%01xm+;h=DdPBH&{FZ)G>65hTvor9GIk%#p{{l&zoL?G3!}z(k<2bJ+UIrc9QBO0hA5l*shR_'
    '<E=N|1QLPU!zu~VNnRS&S7xHt9MT^cVuD_nS0cAqWGbQ&eJ;xK&v?5K(53sX=GyeOEkq39l3@{#(0-mb%vFMg!C*I|<^#e3E%RZ$'
    'J`E2A$H%f7_arDFS27CByR2ol$#~05ErHI6SC>@rS(7oPDF%ap9cM)>%;=T5J+*TIiE0aNsBb2+hH+BnVl^G-D>EvdqUz|b`cgOj'
    'VMTCc#uXI#0~QAyVY9w{o}zvf`82qR?p~+5I9fnN14w1{KdX%3DKd8m@vMg+wL+H{v}_%aYz_V)rjuP+nI<!U`uq%@RvZzs#^LR-'
    'Rcln;ROp9ug{SlM-QO{G=s6z%K-qEQEkJEDVi}@htz3~yKwJck4FOd2ErO9}2ZPkmDTQN|xVG{Nqd|O`&tgd}a#OmG0R=J2NO4pW'
    'T_=SW?V3t*c3y3n+`_7PXBy$*p{FYz{29!jmgh()2RSOuo0|m={0=6*G=l2L+GOM~Z**2~KXf_IPA$6H6h_T>73*%9MYA3CqEK`y'
    '&4atSO?eg5?0UyBLkGlf>?Zcjus2nl%>TnSP#3yB84HF9{aFkqF)0HG5-4O=1eAgb)&O8NFE}YM8h2K@y(VwJ!M<hs*N{cdgPcf8'
    'z6%yEZJH-7u&?VSznjwN^Ji|{or<oeIP7_G4^O<w)i4sA7`Jq1ahGO)-!0l<dkTH@nCvu*roqkRc5qWvgBMdT9R~261YOMJ2YuA!'
    'oR^Bi9}aIe>ciM;UPOMQ%u<?==jN*Vur_AXy@6pFD&J+S&a505(SeF$;kXdda@0+7VR^$3<gZAk_e+lTF-`{-nyj|4j~&*;kRx@D'
    'GZk{1JJGE0!oaex1@6LrcyGCu6E<4eDbW~pc8Y}HO-@ycP${aYC;<S#i(q2L39>2TGrU38Et^xh(j$YP_2IEg9SJm8jsZ8oy~Jv9'
    '!+N?fa$(6=T4ou#>|ro52(l#(vQ%ePJmUNO(BoC}6-${-3o|S%sSWTIm6iI3B`^=}52X;G69K-<^J19U-E5yRq%+23$jeMG7@$rj'
    '&*O@V7<3Y&BBLIt(mbsHV2ae42o};93%f$hXvHgskwg2b0xN16MvdgC8B)hutWV!rD@!(nF8bezYoxt6!0iBEX=j~y63{qNqc=~4'
    'g-oU9JC^Q(*Q0tTlPSobwr+^uNFSAxK=_^7eaF$LY_<cbleA)VBIz@8QO?cAP8XH%b1n`rmI%R@N2YF65<sEf@ld0;EL5qZ-PE(h'
    '`-mPu#u)wfq`Wt{8r95MKnpYmV34-)9cnW(^Q3HR#$vP@dcvYW=DEy=X;O3Qd-65J&5Y6r!juI6U4DorEX$kyIuqoHxt|iXd9ry9'
    '5P=??u1ussnb-uV+d}CLoLZhY1d?`t{OP2cKi=6{n+)CERUk8dJ_tYF$qrEUPHAKFZ*B=U^yL2I@bRnkh#`ZIJ%T!0PO9~W!5bZN'
    'htMKYJb2O2U;|!3y)(FFh=!0NZmZLc3~brKGS8T1f`i>BN{+Kfi4Hq#v`cN8aI-yOo6P1pO*H;26Io?;p2b5~n1txSSS~^_0YFe0'
    '3{FvBBwQG3_1^y;yIeERm!1NVMPaE4_7nCS@)q)EOg@4|j$A9<KHIpx&Z`B04ME{;dR`u;T*LxZZWvD4ry6UXvbEh)b~0hsnKC|X'
    'VPA!=AYZRtb)VEbN}M$^39HE>nQ|~zlEMNRzDoZ}`d1C(%!{%$*Jk5pi6aO+I!Do^&>WVKGpL2BtA)bNAQY&;*L*H1N9&YSaIh@8'
    'wxfIJTrkYyIRpeW;`d$s#2Y5e;$)r&8}oR=VihS&5u3%}9&}h{Vkvll$!~()F}P^njK4IebX|ZVUt}f&#c`<fd^T^x-mgi7{w#2K'
    'a&T~|HdE^@v4PV|$TP2WRB(79rlK&q*5@TRL0vMtp63mKq&Rb>k3OIwsAUObN{cJ{@vmuQ6&NYZ#pgvV84Cm2ViH%=GF2#6@X%FM'
    '5KHPW_X#A@;j_pDi!>!FNiHzHlY>YRK|qkC#1(svzGWyxw<3MctgiKjmmlj-awv>*-N^IKTJKTJ6t@<vT+Z~F<hTiAk=R7e^fu{7'
    'Q&-Ex(m*?L<&=}`D{ll!!xgg}3y81e0b;1i!^R$+jRn$FfKr{g1w*uoAQu_GVG{yin57c>w|T>9G(+qT$*G)@F_Wjgp*IS0UsVh9'
    'h<3JJKo#`oYN<hgu~QjX6U6526rsiSIiY!7P3jVUZa1LM&r7gUdM7rbrw&01n<;=OqcJCqezF10Idv5@MFoa*=5CHlrq%Me;+)r>'
    '2A79gb;ldVTSJ_;qC^@CIAuz(FsT}faW+C4Eo5G&9(nWy+wP8+NiJ%nRS<_D8+9HT?Lt(<9nJeP+Frm1A<Q}=?DR$zl{ub@$=T~z'
    'k@?+&*m^3lumQzCNw<Wa#<mA<N{=_(V`1!Nli$XI@7&P4Z4wR}=F^QQ3Ex6^pD=0xMIp86x&IhhF?0nbyfb-c!Q4bVp^AvurQ)4-'
    'n8<kbN^S`&NQq^Jd*gj>9SXOK0UqUeBig%V99G3{KCD#VV8;-&J(fB@GcA_#v$4JRi9wI0H&1W$(dX@GaFX8%%#9r^8s3F%uOtv}'
    'ZO5qRUkr4KCX=^HHHVL%TC8zy(9~_S%EJTm*y&Wv#!V$xPtnW->ot;xo-5i@-|;Mll5;7%xSM$JbbV%XUT;h*a5^sOrf{R}U@%=#'
    'N17Xgx&d;-<SIEoo-21_n?s+e!6+k{_G%(1t122J8t@&Lbj6>Lq<}gN?Amm@b4Aw1pw{$Aj#~fxGYBEj0Iw~r8~L7b9OGue<47sk'
    'tUEiu0yilcObtaX7!Z$ofQyb3SdFrBchBHpY4zj;4#}QfJztCu%a$hDc*`l3=kk2<GvgqK76m$aa4l#yc!NBxHunue+3_Nw)iex0'
    '&%t!yl0L4jRc||R>6BSRznzLe_F12G8MNf9nZ~&bqO^M=qr4K)MeS$AstFb{%&8{Tn3s8B=o9Pg^WREhuf;$>E<k56q2SZWRis}C'
    'ZdV9P;Q&CXM|AFh48jYpw4FIs6;FsDk>G-pPbi^}_kH7-1dfEEVLcDwI4#?r#X~q~F`f&rCX=PV^IK#S&iIwEB5(SkA}eT)PtrBI'
    '$-djS@nABm36x>h!Yagm9xHvNC`6n)bFW{WDxD0<S%L{&8+U!W6~G>hin4vt)9wI%Ei-AUXRCV1fMMhdh(v~x5JJ@$x(!nfhrZ`9'
    'R1U`ovuHZQU{I{#-e4>EYgB4H*ly3+?$T8v#Nf*%8jU3gHJu60wHcM}oQ|=Sn;^%)74>>c$?&3i5tTFAf8HSw!dHhNxstY;32Ur7'
    '`Nhq3Ckn=uEY-}t%Ydjwa2T`B`7+7#8!@lvr+BqZ<ZJ*eOFwe@pHjGLLN5FZCrNbGzfU^|mzL75>n$QG(?Ni;3*eF?xn9Az0Cd*m'
    'o#Q6Jkv><22_epue$q6oqKS*G6?bY-|DmlIliMlC1uaZT?4@r*I?9EHd@xr>#Kar3*B5?d0xZM1g$Nq8z%<mH!hJBaqy;w>=XtdG'
    '94L<(6Hk`phxKl5PJofIS4Hql%jtv65Sy1ZQ{#h#mzniK=fVwK%Nv;3h>v!d)eOjk*A8+)&PMY!k8W+IqpM|HYDVr7+f0Tb*g>Zi'
    't!IX$yt;W4t1H`5Vdy^VP&Lqb)1SBVVGwlr(Y)Jfw6__k7_ph0n0)R~MINA>fdt=h7WVP?C`Eg~iFT<0c4xD)sJtSlsv+V^pPL!G'
    'D4S7il@0jIq4I8QsoK%6!rB=>&GRx}Ir`>qPcs0+7uG;P@+;Iupg?VEZYz@VKWk&!kb^fow@pzn%cP37d+AukJG6-?e@?fZu%F7R'
    'vT)edV@@=dP;{e2qSj@=C#z%p<BjuKZPH>=YJNiK>m{M6BKD{l0}g4mPeH43QyK)`la=tw;AENi)HBK#&=DzqL+c~ok(;hRJ_qHr'
    'R?UGpgC!eeG?%5g8&RfI)qd&|(6&Aw8yP22LoDG$roy+p$g!=;J`pbTxr7xm((JMt#?`<CqP>b-BV^`FieeAQyIH1V8BRsn7SmoU'
    'xM~@d&1?3s3zHy2+(|#{4R$8+n=`GjFp(&tV!!f>XKrg3njDU3h%*Iif%4_?i4hY2wLG0%K~mK#V?k#8m6Tik{?V+|OdM)0?}A!Y'
    '!1APV(+{1$;xiCVTXKg)N{Ro|t^6b_83F0@>V8W139kas9%SoRLhk3JF-=_|STF`lc&55Qqn02TeEm(N3r)dfg+j6rx#TRPjsm-$'
    'vDZznI5ad;+!9`cum_DN&4kb}pd%>jD8bLzIlB6+Fdp=zsEn#0G*;^Y;yoB$JP!h5R-P0?d-C17_4<KnB>XmqxiYHEn|%HlM%uw+'
    '&hP|z7dpi$9H2U$#Ph`duB`E4{sjgWl%y2{+yR&o3=v~#1Bb@Oh(4GN3K6OpKpKOir>P>e$x*DUAhd|6Oo=ryLU%-J0ye`hS}HbH'
    '4o19kJX^iAPg_y4{^9n}g(VD&f@93YO)&Uj8Y{xpC*nXJsM13SU6Bh>TnJQWCac%v<X02jkmydj9NGhBiIBc(Pj2kkANI7smuxcu'
    'b|a0$WT!x49zNg-7!CDIvKPl?364A9zI*%5AR`%R=#j*>x6_S-jmS}B&LZ&IUs{z3s=cLq@T0B=Db{$xYaJ?xZjN0E-O`N7+3Ta#'
    'ln=2jtr8sqA)DLY;4+p!+SrIzXK>BZUeD8ZdxY%-ZPs#4ek^S3hb3B75Mvu1)Kh82C1RlFdFrO?3y5QYQzfpad&+DB5h`n+E_RdA'
    'jjMaW8@Ni8JYy$Qx2H~uXh^54Qne68pdcR~_UdS9ga_W@!PzgdE|=|!CHn;lV)-T{l>*eP+?Vc~kTfW@+n0*#Dt7)V_)79N84iop'
    'B)__@Oy}9}%1QaJuLiM~RfH039z<6F^suH;SfSEPb5f04b}g+FTn<(!Y|zu7zFa@lqv{abHfdETfx<xnSR5XekbC%C4&i70(|m%='
    'ikp{%!onatZt%Tr4W4(UVoZCHhOgZi_NM^ngjwzqx`W>iJ(I!j)-Lq)B%O&H#+d@dhdJR~Mjf5>)iaEqG0ulw5PEhjpViz7Ji(Fp'
    'f^!k58-!#CuG09&YanDZAA}s_RUGt^a4n9ag1g;r<|woC%LHypHM0>9Q{L_9aklzd4@`(o#b~Dl<uDL<E=oo}GHTtu%3Y#<P|ZS5'
    'LeUkpa22Qg=D3(8)hJG>C)FBKYMf@H;xu~-b{$;Pl%3%)a+MEzUSfItCyeT#E4~UbS-W5sn+R_RhGjYcsLlt)7rT^RiCrC-WdvPY'
    'd2G}oi<x&<P8SJ6VO7p!mJ%28>Iq?3Ea>u23;^B^X8|K9*Q2`Krd;H3)|QH^*<YCte^y;Fj2_{Y>VhXFUL!qyOiveKOh>g$uA%5)'
    'Q{x52Mral9A*$2oWyFTNW&2VN>Ml`o%x~F+AUS=CJ4W3pZ3%HH&|Im2X*;#UK|5eIHe~Fr<oynJ1Jy&XdU>Db=ndAiDlZL-KGDn3'
    'eUad0;P;$0<2*1y?rrtTqGDfowUOR5L!sUR9Xl9X4lyB7!~2U>H**2AckqnzG(%pl<kBhnSHQ6`X76O^DA&O8VL~ts)FCdHUnnx^'
    '19)_MgTw;vksK9ax&Z~BAR}k*EP-muO@zgkYHWMRTI-PFAjl7JH3umQ%D5%bvFGA@qxi;-Rd(b}QOtqh@{@Fue|ToA5XBs}=gNWx'
    'MettOMA3aL6VwTeGHL@&`a7~<17KhAc`RFK^ER4=-Iw1o^j>Ce>MDo~Ga*c&(nf@|T>x>|qdKW7mSkoex4cu2NaOl#NydTt#L!p9'
    '<<oi-PY#x6BuoKt8+t6P<Ax^RY7B<U5M``Cri{0M*tkhBqMvoaON{g<?J$4?qEPcQvNhRed%~hZorr6gnL+`iz-p}H;$TkqT;ZxM'
    '(F7;^`g3{m%{-;JV2SFg;ABh}@*FnJTh0utL$p%nHGn5wHz13Q*%fclIX#~j1XXpyJeh|(!6mU2FXj@QOan@>IUPCkVVT91>)tzP'
    'R~TT{^5Xu0UL~EaFcM~3z7YK)b5aL4xjh-Hn#oLT*y4~>A_t0*o4UK=rBi|W(pEwfrVN3_sQnFTd`p4l$=gXR0$*Uk?usW#@x9p-'
    '$JA2Ra*Q^Mi~R0T&l$~9;-YVq_X>fG&*yKU`UPtglN~_q!%Vw{{Or?se-+DAOABmR2=dI~szREh3WR)9L~YEr@$rj>>W0cpLsPj`'
    'kSP#0pa2TtJ$C^<RoArW{}vShT9`BHr-^r15Q+tamAWRc9NvfM{-zdQ^5H5g5oQwG1e%_qC4=d9B+5uKIyvS`ULH}^h;F(orQEJG'
    'fM91yD2@d0ZESW-&7Mas-}xu4c(_GdCJir?57T^!K0WIfD6~yY$JfPxER2I!bTl(cwcu_>oykaF8Sq+&5!Hj{!#?K^l~^h-cck*N'
    'WhC2XnY1jKRr;n(+s2l;??SEMLTs|g3`m^oI7)^Smp;go=|9LPeD)BN6ci|I?zX!%`kh=$GN(LMWj~oqOYuFoYBwnTNa5(X_!a>f'
    '2tHv9U}fZ6Ue6D><uqQ}=@RNEbP4nbMvk&_UevW!*Xp4xd0xZJG+8;Dw_vy>_Ds=eCHH*@S|(ruwOF`G7UH+^8{bE;lqjiej>vn>'
    'I2X=-Brwikd4L;*0>4{OMAA7sbQBbGLKT$`{^w;!or?-_)zoG(vd^5vDxsBPjD>Kf;;4NKa~7Z5E3)!koRNb>-<cg|*(E<Nvr>{s'
    'S}VCLJC=edVXLxthu_B8LA1eLQqYDh)AG!|+tfopbPt!LpG@(MLZ+9w3A`669UuTsuQ`vCcMg#clo=)icmon8#Y+?z1y<aed4=qM'
    'APR2pQNjTbaNiiHa;*9gfQ0I*^RM*}apk3EX?$AAz}x{O1c&gkGm)Cr{(QZ5q@u&7Lf*J<gh)VdqXOUPRHmsv$#86B5pDImJfniq'
    '=&Pstu6#C8Kv+4LV@{(DZiz49sAI2q!pdh&z&E*!y=2#KIN8LK*H`Lyo@Wtxy^O+<Mvk8Nq(0Bz!_sXi=nS4{NmwaT%}En5C=FT='
    'yC2Lc+~CjunkIG(z?D(SF&~NkRn_a-2})xYQ-P==o^MndT+p&umKw8?A-b@c<Fhmf4SZoizJz-$gNQ5Bx<bnpMqbjxfj_|{hvRYI'
    '3cQ}@9(h`6NT3Yvcvvdxu4>@WkwYSYoEJ003Fd+PmkL9?C~cvwDJTd-)DLxx**wcbkamciCO!UBCj>UNSgDH|=HUcd{I@81SXc}<'
    'rph<z`-9ELG4%#LREU3HIpxOmiB)VeEEZM^T&iLyv%{?>TU4f_*fpQ?MVMSXXBU4nO>*+qF@0cj<4)zQBNdM?Xai~qs<<FLHo$Hy'
    'x(1HMPUnq4Ce^+%GHb!4K_tdf(%Xe9L@=3VXwL#?z_4?dr-hCa6GfltY&jS}<UCXxP;!EuaDi4Gq40f5%h`frk9nDb6tH{R^yvjf'
    'U$DGAp0j5dnaPw-^bUZ3UcXn%Qp{I^V;ZEKJ<l#PIIdTk%AQ}K!m#Z`_uwc_@u*30YrayHI@7Qiu3)&VK+)8hA*NYKOseQ0bVg*5'
    'fIVOt#>qO+TZFmFvh0u`qpcwI06@rH2cYQ!M#SyQy3kanWGhN_8(BRrn5fZA7o{X;s5`<a1BPuBnc8~ThXj4ubNqr<<V44We|%j#'
    '^d)I+=ieq^hkuJUTF@RYx>N)azESu+t-y$|QfNe%d(fI%l7&)pGYGh?jYTEbqC0B@j9+qxbC~8p&{*8uE2G5HI_hrZxo#u#JO1;H'
    'HB&bLV#Fcka4Z-OWs*~5@n-}h#W-thrt40io*OW7R$_RhP-_aITxkc)IuUJ#m3I`qJ~~fLy)(`WbWa?w=P;y=l@B+73i*)^Q$~tR'
    'fekF}zVSI0-2JqR1lRCfWwwjC#XIFAV9v7DQ{@f>h!Xggu`74osXSxV--xWMr&wt>3XC0mgt`nyNFq3x0?^+T2j8S%n<QRR;1R`;'
    'Tfbo-SE>S8b3b+MbWk2(|M_rD8_dM!OQExkt9a$#nRkN4dY6~nYzP5|-RQ!*14HgvDQ@u8+!52{3nI}E()Viku1H~{vlt@=M3Td3'
    'S@YyEA&fa6cQ9`(epeUH#I2HMW4;JbnOa^krXF9jm6ghkjT20J`9TU8@C}Q%wA23Tkk?z<A#XEvkQy0$zEhx@z$!E_rmJ`MamQKT'
    'aV1^=9+~b*pg5<KzWc~+)$Cdv`C3@svC-mDG~f?QQH0bK1&f9s;x0fWu2V1T#-`PxV_?TnGl#6qy6;P!zo8VrAdYhc7Y$%l%|ND!'
    '^^<<P@5tc9qa%~^WVHG}IMI)%@w~#S&T9_|&5Y|0I&P>GA{wU3D^7@bJ}}(Zfg?JlHqTk|Z&nB18Z%d))v!-ZI-G-8@IskV*yGL&'
    '!kf+Y5aWs|kPs?3lL*|KGeK#rkvR^@qdk61_+neK0MGgHC;4-VJC-k%XinUV@_s^@C_}a~Lppw}e0R8w%7?H{BXpRx@q#@2CYr1G'
    'Cw-n5%q}ZOY=;tXxe#TIN0?sJ^7=ZpRns4d&R{+)pPfmIGzcFCqgKYq*GC~?T~D##n4_u$KXtp>uVWwLn{AsY+tTA*8G=2v7pJx&'
    'AW*x;+wzZJ@u;fDdJpFGnDXs4qTj30F+3HapT-0mW<^=R@t2SS&0;M^2F@|PH_y+^-+mXOd8hE-$}|_J(7rw&)o%VwxBtVpSAoy8'
    'B9j8c*_Uxe{#|VDW|U06G|!$kp0r-u_%g8UVYJ3T!!$MM@cBD$1;f%B7QUqt#fr+mHWC%J(6Dn7Odz`SaD^?`X@=uOX0R#^^X>fH'
    'NYik~=DEm_<QTpfz^g_+u`T862*>hq0m^`5#=Ym1vFwc+kyEEF3nLZsJbY#%>BMW(d$v=rNILa}=on{aa7Y#cO#f!)L*pK2ppHP2'
    'O@&wv{#;7W*T2?A7(|~3wdZYP1}&j5ehp?RnRp5Ee1k*b$a)>6g9B-aDB<)M&QTGL@n@8)dU`91QCF2Kvc?g_2HU)ZBO26+6F&rU'
    'qTLi8WWbxwg;(q_D0cRn&I)z3E$=L_1x37T=y6S3bi>N%Kc^QTbsNA}2!(7+C@uJ%Hit5TC6k$0moD;pnTqxtcZh{^k;vcLeKKqN'
    '6=wWj3layG2etnZ_e277S?w4&UU#v0Hn3Yb3_7%Ka5A|j{z|`oR)`;tyJ(zdoP-$MsxgWcvzm4ip80i5jFNz+kn8K8{rXYwvc<?u'
    'pa+ji*2|IiWRTxxvU@pMxZL0O>)(9+9COP)FYxogCk2-NoX?DX)`chh`jB70>DRya^>Z5~1*V1r;%r*=Jty%Jb^Pn%{Q87nzv$P$'
    '`0Xb#p^Ra!)Vj0rUzhdQwfXvpzkb<Y|LCtDrPml+^*F~Ha;^-T=r>QT?S%iMr+p=b=FGYNo23KTJkocc<DgJFQ9_B02f)}ra-f3t'
    'aaVW~h&$o;vb|;skfAWRZ`z0Hwf=XEa=v1rO8{qEsDSpZv2GeS({Ps)-@clzuz_I_O(O&-vua2RpgNt??BJ1t;ZOYBOD6nD?5Tc6'
    '&Tm3dnkj68o!kEpLSiiVJWMO$Y>Pw;2@>?Z&?#Fb9ipVv{+E9YERzuv2k;Cbj5YQ76b!rwc%8NOzUw5(+5u=ODa3>0n94`geUy5n'
    '98RW)U!RNq1!J_xU<Qa2az}bqC=Z#W5%l-1)&>{ny@Gs=cnJ;{wnGdTiVQH?_x)|jQlYXE@<ov3mRJOo2&@pu^ObpqvNh*7p(42?'
    '+|vZ-C4CE$iUl|4nQ_*%uKx$_C2FWzy#wxN?m(2n$b(Yzvmr?MHA`ERT}%XV8gK$jP;Vlw{>?g+*ynGb2jZpa6P#2AD*Ss#J0umk'
    ';$+wP?H{rCQqNB1dor0(jcrH_pmxB({QbrYUh)op;dl5C0K0PP0%{9J4HDk<e_&FeaG-hP_(A??w53&AL;}W%CoJ<@)*=U4hlfMc'
    'q|hyR4ab&AbXjRM=HK&}nCrOQ<`(GP!62&R%6l&HXoeY<ZwzKG@6%84PHg!rXfGrAhaovN)-jO%XD8g@XwHh17TNao`3|2V^&i6s'
    '%#_DGJ-_p<h!@j@%h&@_7b|_Jcy!Q{G{`x=UsUMo%VFblqmZl)Rjpc$kROOBnOWKUhWTvaC(|&d69CQl2-rpm1%|0OQP6&S{5Mic'
    'tOqj}@-Rx6zKk1Df^@2Us4W`-*>)LD>;$wqw6esBH@YxdxBj6e(tkfuK?lb}J^V!Hj&Gk^G6~3v443es`1L}XilU$dbm@ds4UfV)'
    '@{SK(8pP2#KQv4{h{Bj85(imBA@~`>Zx<az5(U~X)C~KZ%UG}Crj<lGR^!5(*N|FI-?^`%857gj)vD&Ytqfr_stuXRb2f1zv%{C$'
    'zTbH3JJs>1J2KT-x`U0t$vsS!tcirE{oA(Hk;kAob4Dl<t}~Q<$_^}ljhhGf#wp*pT&SkHdY0-wrCQ@P^XYNH5edT<>YED(5l&Zd'
    '$q*G-WLp<pKt9@YMug!eKTZ{8l(~_uPp2YC{yM#$fbYYsrT%kKwBy?c78BP}zym59VADp*-KJhpVr;N1o?j<8`j#0&duU;5MiLMO'
    '(Zv=w(Z;8IgV!+alae-k!*cD1hM)|+t7lD9Cwl!}Ja;V%N!fO&NpQ*pb4WROc-hsBd#wKc?TLp=mBmQTpfu^ttWEbc$|RB;+W0oQ'
    'fETzq+Tr4zlT5S)?HW9T4J+G=^F0+<F)y1<yiFS@C@>JbE|Q@e;|l)8{XNA6Ita6QA>jH-74r+0U}Xsul3W2ZB!0B+$F0Q5f;-8G'
    '561H+^ZQz`0e8Xk4JH7>+CeW%qluG^@_$n+v7qZlu1|L(^;>^IfFOn^6#ItX2*coK5bX28xdP{nPFkEfTgr!cL<)L8JKN20?rm2w'
    '>NoRqqX<M5$3)aKjgM|>8?p0)4Jz*AIzIHoW}Ge}li#}+qyLDUZ(4yf-OrHr_7}xS|I&Miu%XYAWH&-Ob<fiR2MC4`W9n)Sa=)Zf'
    '>?x3nnblzVwN$n>lMYtq^F38@|IDjW&0`RD!XA_*0^J`r-on87DVrJpYuGEyKz>zekbB@(2LalF^5JO`S<7m8A>@;pFed}FiLcXY'
    'm(;`vwLL_`P6Q4!BV#821*Ll1=baTOOGr9m^yJl@8w=e^`VLBlL(>WUu(xk-JcJyGb{1L(mu5IXXU`0uZm1dBo%0RPg(S)y;FJ}L'
    '#JXL37bBL=+OF%V8t=P<f?DbbcItNb%^mqB()Xk~j<pv1pZKO%bb--YDl3nfOP&Dc6f;L6I0Dws`gRSQaaZYiY0N@616P=0I281v'
    '(U|`iNB@652sOcB=Za@c$%ZJB(X$pYF<n1X{hs~00UWU?H$iY2$yKq_?M!)K?B21_jc;})3>ly*tkZzZ^>b2zfgOD{(*{rO_}+Fo'
    '8run=z(TzAqYClr<{mCulA$m2+kh9>a<@81T57Pg(HS*4&5aw22Vi^~oS>DR2A@pCC!_%>bO4{VU7%;HH{*Tb!9gGOe6W?)xN;Pq'
    'FpFbrA!!7M82{6Ej54|K8!<V1_6dQVwqlT&L+ne(z}rj~`!*wGvVO2<a0_q%Q$Nm)7)YTE5FFSxZ-f>Z(l)SPY#H0MDVkQ{C2wyI'
    'x;Z9h-$+7m01i|cT$W6rOv)xJr~-keH$iODz6qx&kXMFpw)6yXS9&kjW=a8rX%Ic-o#QD$u<q0AOi|sk`pb;@+6>M?VO_sn4E$?@'
    '7OI*O<2!R2t%jKujS-i{Qsvi^GV(Dn<>;wOrBnx|0rI!UGHDLt;5URGwfFY1^U-`BMBd-XelYV->)XK|@|+*OO6x|Bo=MCV1blXB'
    'iRk7?AwrKkyM6n}q*ENR=~$truTqa`wG&7}8VauS{VH+o!FK?Dl0?C6wXmn~LH)ow?7orV`FDHUcbHJ*e?!rBn`5?R)!4ay_AWb$'
    'c33FGm^91Sd<WVscBU>_U0~1ecaq@_)1gmBM;FrvGMk7nTgrT5AwF~9iPjJQ6(W|wCV<+pW993tA5D^A3&m>r*B6E@o6kB~Rg<ZN'
    'Vv%G3vKE(*0)~&aKC?tmzLME2>naIl9W$9BH)CKn@TMQ%27Cxi8sZZ~=<*MytFMw%l|xO@4dMOvKrBQ82uG3CRiYfayp27az%-5)'
    '0@U{RBL$-&G|+##gngnbnax6aK|+k*tqJ;`^@r~cN_CYigcV>`5!uIAU!lx02kjbW$F~xdY)gF9j<j~t$w9T~qc*b`H{|?gn?@Ke'
    '`*EtGA&Nb?v`jb<sgcV;=~BzL-HAALXlm%DiqGC)|57uVcJsh8YV6-_e0bwzykJnCD0iB`+i=cQy78LGw10oc&D@jO15SlGO3p}Y'
    'cOHmZ)svM7k8ke=ik6D@+B9+QEJ77ovt$4)hj6-@=C^?$%SIUF2&U<*dIk+O=YThTLR9p(7Z>U(R79R@yun}sTUe4)Qvn~h>xKML'
    'vM&lk6x_(I78_gctd4?(ef31uj3%Ib`$qT~we6#c79al1Hj<bfH(Mrtrj~DJ+KlE{E4|va=gYidzixOX(20?rH+p|xsY_#4T(lr+'
    'L-@^`bxWh<4M0A;v32KLb7DX&@GP@N7zQHm*YD4?+;l`4xBJ83zQ2lYAhNvWXiyR1HR7<K_b{Sd2SWoR%3A-v-CNTM52LxFuJu{2'
    '=oB`#q~O7HrwBF_CI0tAs;3`(|3B*9EL*bOT8`}sTfl*d8vXB>U^Z9m6Df@$UGXqaM1H(S>N2Yxu@*23ck`S}OC+yUPTe&Ej&#HX'
    'C=9=H5Vld)f+edQDG}*tF!WjtdVeGbCX2^gfuWDaq|P{45Y()9v&ckP(d4Ml<(+Q7Z0;0{lfSa`?;bs*fj*ly|Dc?Mc65*peQrWn'
    'MOi;qkH%0wtUD0k3XGl1#|oa``8NFt{j~VigPNhFU4tn;h*d6}37;B;W|?ohID7-9=it!TXn<JYZGr+QSPp9-a6=t!&c{BL5E@0B'
    'k05@iBR0~O^$q&do;rB8B<~Uv1W1i&arvletVTJ9ue)iZUP^(-d`sid2tSlTYM<}ste7qhD<-xPO*sF?{NU7-#zX6v7(eyxYG80f'
    'nb(0;T}4LJ_fO<xlnEZeHthCDoTI0~*N#vSIo;ZCpCiNSImiug=}7^Hhl?s13t`zp1Z#$kd~Oy*b&z%-)ba}iBz76{ln4jtOuIQK'
    '+xy5$&rzqYq~PzkjHyl=PC~xrPM^ahb(Oa}(quscgw0tel8@j%^l%K;dtr-+kul<fgHv~_s-^0~U)Gwxrm>Rv8&UNq)96iP#XPIQ'
    'z=%?v0VcCsr4D_!7f#_2o$Z%(_1cu0+A1lcmWu@{Oxl{&oV>REK}s>{>7-;VQPX;Jg!$y-FHfSmJg@IctgV&s0pBF%M@{X_%D8s}'
    '(%Y|`8kD&kQliWy>J-``tFwk29eoY!Ws}`MsDeTLBcqOq2I~WWEnGwY!jp=h&Y9SH7qe|s7Rc7(w>Nq+G&F1tuq<Ma>i{g;{<RgU'
    '+Y}`HDl;>|+g6f5w19!%BsYHYR7J_ySa-uQY4AS~tBh(2d^P+XM)`hQFhZVfyzjG*2V@vroHawqY7Y7A@ZTgb^vL|AzN36jT=D^1'
    '#evDByJajpK6rkJojXDWSLP>E8^ktH?r|S6vMBHS5T+V<Hm0_Hes-E@VD62<JoB$+TE<>qIshUp8ff9x+(V9-`J>g^y5Xv0{!`vO'
    '1td<X<uq3c!)7;6lNk9ahC0taNj+crL#rubunCX?Ay?Nx+gf@J+L)Oi+xJ`3ABKx}359r&hz|n5UTDvF&AZ!|V37XyZ8K*sWXrSL'
    'Rbo74#F4n>SoL&cL}$ETFI#x`lpZud(Ne1ql#A%DmN~5w)_ya)`CcJA3DHL)4^}=HWq9E!CAnem(a8hy_P*aY1W(Nx&XP~&WV211'
    'nf2HpH*turcVA8s0C^Jb;$mO>V5o(9av?Eu2YR2scrv#&kWD5l1=c8FrxAvfPPEY^THWSh{Q@*Ag#XYI9n9PmgOl|y*l5U`9Oy0n'
    '3(JW60pSeQOa<!z&<kHXw1}bek_C;I(Az01=91AUXVEB<p?K*qM^KwAft5xWowvFbZxqUD284ssE(sO+2~dd`qr2XQ2jF-*%5}r('
    'Fm6Mf>NsZE04^zSih+M~mLk5{#Uv23^|p4E%59`B1do{mW8LWQU;g4UG8K=}0^wf^%qZTNGoBH_+HX|{m{yJHOS}O&q=jGGR+csN'
    'fL(|E2d6|lqNf=}*E3oNy~n+jk$K!l^Zo~-p!Zb}*wc;FenLINkcz!^Nq4LdGpwV2T<tr@B!n4M&(I<kBd^;YM!WLEGV=omw#uI;'
    '1M5SSJ>zxRX1BOT+DAAtB6RJKdQ+9dyl>W&h;;<Ew?x5!0nm06?bQ=1da&e5k7{Rr>F5u@9arb=;W4>jq~bayWlg9+ZTx)9`fr2w'
    'CT#>Va%e`_zWkeIO5h<LI%#yfjkH5EOnb8`-OIYec|RFDHre=^vxpeXh&TZ|<RE<TZ2(b#nwy<4QgBn);E#Cj>ddcI7R=VXUf*U!'
    '{Oo#zq})N^ZU`Edns&nA$lK+&y{n&t1@bbgCo}N?Jdsh%%90wk($gc6DAcf@Yt)cOBgO~TjD{fzHyh9wa-WS(fR*!^^AD7SvK&6_'
    'gsTZKkP?C`-=GS|7a{V6a%A@R3Extme4N@fypr5};n>o>?Q;^;|Asm4_dM~@lL=obdWKw*eN;5pd~0tZ85zoV;w^eLFeO&rm7{WB'
    '865eRXby-gH238gt@EWaXAl&}WeyCssSgys$4nnS2G%sWtM4!Gb4=rGcfyIxdCa9M0w;6P<B4AX&Yzr}V(uq6=ixL$!EkaWj|C`q'
    'mC?eh{U%WqGt;q0aO!u%z@Y0mx5c&uTSV#@5P>h<{jj&$NX1b8GCVNfs-y>&f4cw>F0&rv-4R*zP9~T^3I;U45iI)I3xpm*bEv<w'
    '&#vh$#|FwVRjGc0)$(RI1cF^a1H;Mne=IWK+L?e!@e4rY{-|L|Ym~J5i#@)x-l=FXE9Kn1rzvh&;L(_lW<>(d^XPyEsat<p+hMpO'
    '%nafyI6@|g)2VzI^v>*^q>XPC{SRSORk%&h*@c{)p5aX(VY)n?^Uzq{xWuC6mf`9OMi4MAuKHneAesespY>~h_Q|xSNFtmWCK}Qj'
    '%uhwhz$K+pGv3at<6Id)S&KwHOgRpS`<Wx3$4gmz`_>;EwPPDnOCby;b*~(C+l_-we33M#w<)=FF9PB*|H1u2iGFZ!FtITGEGHIU'
    'I!N>$-{^n?Nka$V5&Vk_>qz(4ySbk?T>AY7M=j~7rt%haM>SlKge`%#y5BtI&OTrIgJbZ=mUH;`Zln5u0rdIsvW}rZtXWRHnN;%t'
    'V1tUG<(Q1b3j!)ZT>{xSh>_()ey4@}Ajy%O!gf7lo!iK$ZH9cXOf_%aBkpa|h8ocM=__c2QFFmujnnnk`ZcZ8=Sy$%fTdP}dl<I1'
    '6K@86X-(&pleORW(0?;71<J$wG?;{-9vzM&DP^2<I9o_RNWt^lEK|JTblucZuT7I%P^g1vVuZ=_=DYW~jWa8~DfxxHNArm%q7m9Z'
    '>~$mq%cS}CzHqL-+5I&eo(bz*ZDd)qZmp+Qa?Q6<$U#utG9O}<nLdxBrA@(%(_LIHAG{ocIbF@lMrKH48YRRk08iWi!W0gg-e0x_'
    '@rvd|1d12m0IfmCv>j#MkF&omJK&-Y$%5ow@<FkM=$PobIv}Q<(knj5{|<J4(R8$REjK2l9|-1Cl*Bg7H|85l`+jx;u$q|Cv878E'
    'evqa%bQt%6J!ij};b9e*>G}?Y?Cm4CfyL>eU!N@HI=ZO%pkYUh(<(nvRj**PV;bJUv5aLNd+cA&y)_Yu6ucUS<X{um1<N7@vzzMh'
    'cPf+tO`sd5X)5ScC^rF#LLNA1@wB4UPujWY>I1^+(L6yu1GB$ngLkh<g6$Y>to`<Qn99zsm8E(IaK@Q&Z`9^WV4;XYM|!W-x2I#;'
    'GO)Er+l_k;0UVzu4f&+_?184^gP#XwKZp{rm5bOn#=${jh?ZpQD!>5U$GZ)b@V%97Qo}H-aDh=U(ZO{q%n#hPZ@)<mFpCC=56N|b'
    'YH{5E8Q0hi#_gZe?l2c3=GP<@7t)4-z=p;&utD_puFU=Y5a9R~l*ek82#~5Nq58B3%@`j`j}c#4W02N6QGqb&FwD@F3LZM<8^1*$'
    'A7{Ry3{cX}k%6Vha6jQ^Go=?DCQsL|_oW48vq+YTN)Fm)?F%0Ou{%FgM$!6>zA5q@_jjqxkksNnCz4_h+!!=WMI|7m<@;Ww@Rk?K'
    'Z_FM<z`@tzk(W2J*eTtj?9IhOI3r^kGL(RP5)ovim+4;^MqJRg-@KF)VYH^GZdI5&QQg#7tcqFVquH2szTc&I>@mX7r!?k{EV}1l'
    '=f4P^&r_zf^^4vdgB~mUBh2~u5*~;>yiW&R3kAveRr7%87KOvcVgeuxU(WsOPL6J>riNbMwIQ(Z;F%*hG?i9xFhM^q-TfO2GtvLT'
    'k}9C-bu4KBn-6O?QZhQKSfbbNcbkb>hA95^yQsxDjX}0I(&})?Ij3Q)12+DClmMp3yUcW{D>!M*CDO_yIOuj>ZP8Lc(%LV-ONL`_'
    'a)pPXrU+qpmFWqWUPK~H`)^KQ))!aw#la-tbV0xZ{W_DqShJ0cN2tFV6;zj#=ZC-AaH`(?JtnjGd*Yq($HcE1QiWuS$P6B~+5M!j'
    '*D0wqOXFh~b1td2`REHl3BAW=x&yoCe$U%SJl^V2Qg|9*efpKk;be#r1Q!JMjd3;yO{@DmkL9h02;~wz2+}afRE=DiE@P9*8tv@u'
    'dHy^#0<HLA@4yd)V|C(S2b#Wx<2|jlQ5<!=OX_hHF2~rM3}eh}=SA)2KAq-<ez@1~)S0wt^8p*Hv0R4u#SW@d5(W}~5$Q1`lfUJL'
    'wc_RxBm^Bt34Z-HWI2aG$oWL-xBpEwYh&D}ENjzg>5L&`XEY|nbs=jLB+mKZhH{djM<9?UEc7Hnm~iG$R%5@JXUcJVdvWXHZ9)ED'
    't1*G3uXAE<spY^=Z=EXKJAU2ORRo5^=qn~Ajq5UVbnqr%#mE6FMEEg&<$@$Y-rx>&epx&c8<N$00?^C*1%t|SB-)+Acsjv)=w}>M'
    '2ahj~sr_#MXMb~Bqh>Y`n+(bEbW*ObOM$<Ys&a<jZ$7082YukpsBpaQ<;%~>QEBje2=74ZQHOQfoAZlJM<oKn*qjZQE?~ozS!V7<'
    '*D{}C|8GemwK-`ddkLxAFq=d%-^eVd$QVk;E_2p*ux%)<0Q|d9Y=bVY{A;<AT`wDT3CiJpJB_1zciG*ypc!G@l;hoK4-JsmOVz^l'
    '-FU$x<{S(Q#y%-`4Az<VpM}~jHzB9~7fb^c1!ZvmRQbZ>sT%sUhj}J^oybenp6oZ1ZG{PxCM-0tD+7$s89MHF;3r5MNU_J@FB8)u'
    '&}i@z4&Mce!fG@ZZQOk+%z*vHq$3dzX{XT^2jtIj5Q^vl#ip6?_gRi#7RJE6OOwMdO&@!MIp-9XT`S9dC4d*luO5+wrH3sP8C6+~'
    ';$%TdQosN*=<em$@9t#K)BWx`(MXv!PK<^8&(?QpR@;8t-PB<$+h#vDGSM-J={bqSE@X8OAz^ygy9KXe<gM(&^+87FY2gZN8U@LX'
    '5#U{p&hq!4Pu5LTI{<uxX<=aAh`B+d7n;}m0ly0_j1F8*pN^pnUCdO^fegz@v?kHtv1N?6hS41;JxrtA$exXXEli};oPq3rw`wvE'
    'CfFcBPwJghp?sWrA5`g#FvkB7olzy)Rx#%K-=D3!2{)ucZ5}OCSJvC0Eeur(7aNSBgz>{$ubS*M%s@YOR`vb_rEZhqHzOop4_3Ja'
    'cM;2u`)V~$*U{g*tT6qR$Q=3rH7%HkxL#2;PL*h6Ue)*3jBzq=6tm)mj66d^8J$B+AT);GEx+Wg!-n?CiD=3&>7eH1+_T>RNgBk-'
    '?Bv}m>RZ@olS~@Hq<rSiptEJHP|wjrdlRu&W8^6LEBQO+7+Kaz!j(KCN_Rlc81k1^)!;o719?RDPq2+eG|BBWABURj{ccLP^n;Q|'
    'Hp|?2sFmr8hu20p!gcX0%9hzU(Lt0Ujg6J*P^j*uRG_(`E8P7T{Tc2E@Oj4&GlJ(}b#hqH5Z^np317H>-`fN3R1IgT`n6$F4`>Mh'
    '`v_l+^ddRA$J;^A6$t|%a=kLDOLX85Ts625u;`KWw9mU`nimx4U?sloeyF695o0!$Uf?_OUE}dn=l<6hz}#bMGF@n@%E+v!d?R_c'
    'QKsSLj`t<DcX1;j>NL7mo_ccOd5$}4!IR*|{a*GM0ET{VJE|qO8_Bs3KW0lQ7Am~o4|p`hbx}i9<K)UV_N=i&<1Oq!^HIgcFQ5N1'
    'vT_=fgKEa$xSDSmtm}oFgJjM2Kb?_Pa=RPRS;0{iFtMz0MPcJiE(}1-`s(l19@3XE0@I&&nd5^(pR-mzKr`jMtT!%FjC_2R{6Yg^'
    'SG2sZt(lvN^B^$#8%l+I&3g%|rRk__F`<ZXPd&++C&xIk-d{tKL>ihLWAFx870-%k+Y8lyXJFV|??N@kXtSGc19KwFK@=ajmX~5('
    'X7-1R=56>I<XL%q#Zc278Qxh4nZlAK_9<O7@AWo<;A7P1QQqtVo~O2X{DQmNMha|Wv)@nopI?vT@$l+e<4xwq)W;KIGv7sdXX?HE'
    '=|8_3Q}y69AYYn%t9%IS`^)k3Px<+ye||Bdk^81Bw_3#5{(8)}SIVaUj-NmI=NCh8ulv7%d{`wvN|J<0CW8=)tBuf8*N9~EE<9eT'
    'SIbM=#U9xW$9felB5KTo?u)}Tb*RVCy6+QlbQ88QE9X8D5=19uZe%_jGY=pm%rvpF{#jW<kPPti*d=aWW-6M^qsp?3Fd%k7g;8UG'
    '@GCF12~>L-L*#>!CIfC2cv;|zHe!dCfnhhghIrq;e2tH0xiPv|3V~p#=BWM-j0ajnvn_?Tn=H?@d!9KqCWKoT?gXiWjYe(>m7O~9'
    'zKam=Xbhw21)r~8w~3d4vqO-z;kh{qUFPJ}rm=yLQ)t^u$n9r#Yht9B0N_v^L}gXYQDh4#Pr3J3^=`RuEjQZe&*=%1r_vlZ6c{se'
    '<k0vq1Nt|$brrq@Y>P7QJdQ&Eim-mE4M7i3;1xbyV}gnOHB}A<SJHh(dwRip+!B(~^M8KgXGsnS&!3`(grUAjX#)XYGZnO>2#tZi'
    'J82M=Z<HD~KRW`b7f&$H`kOP%RyYoe%69HdZWr@+X_d(=999ZsI0<}=S5ZK2WrzAxO>-RnJjWp#I$>|PRnm{oFPSg9PNgK+CS+oa'
    '<l<#zNMjWQe(`}vPU?fX&%bF6pBh9<?KI@65l^AI{$#q&#847hChd#`Di2FFA8Esd(Vjq{rivi3n>>dhf`&k#lYjI4n+_VHrYh+H'
    'W?rZ97<Jh)!5W~D*;A*O(aPxXnZMIpltTZ<$lZ3sKWY<LikR53*S}321bQ}}>?M|E=DQQ4Ntt7uD*6N-N;`DY8_hIiz*e8??2#0N'
    'wj+akj4@A3tx-)T4({|!|2`rL+EIorwdSYf0|z6QrUhNG+F^s2&dWp0%JuXNNKZ~0r{~Q7tZ9r-<Rw4-uFp6)On{}IHUZ0UpG~4T'
    '{|P@~J$y`uDjzcA@H}^BS|l%$>*5~A2@NMPeuDeZ&_vXXZO;`v!UtN$`wFmlgJUV?fu@imNU$sCeEmM5BA8h)*xX~lSHU&#|F69L'
    '1TOKQu9BzXpOsu>dOdU7)c@aR0z!dn?-RLNJzUotCk^x7x}F>8mixCyp6Q-1)bJI@e$|4?Nry2L2PI*qEb`9m-)AJk_D|x>O!XKY'
    '7M|$SlH7aPAxy16o&08K)z82KUKG@TV;A4SzMv7TFxQlo3?Gh}Vk7Y4Bc9^{Wx?K1qK~w#W+Y0%gbo&5Q4)c-ZAj7#?3U+4YBJK6'
    'h)O;u)x<-}D4=E-G|X5~9IvpWEquxQMLt1KfjF_uj0A)qD(06S2y^L>IM2Bzs^VFCFEa?zKws|df@yCcrD!t3`Gc69Pn~&Kd!K4q'
    'jaH=$?JO@!CcI1`>r~$45K}gFiXUVqd+p!1@Fz;(RuU)4Cv-#ThVE4Yl?K-6nu5QE$Ou!29z}Z^5Z5E+R$a)z)skEPBG3ZRK<3yL'
    'a7z0px@ckZVZ=5VE0b6>n9xz8-{(1GEgN{}{{Q*;4GUi{Z$C9h6IVnOterNL(9;qO!9=v}#~?cl)$xfpJ%~IKo7ZL0d7JueWqXii'
    '@@w|^n}+~Xb^iSzaXO|B1B`Gm!C^9En~$rWmJnKbKuP^&X)%)GexCutu6pj3fPXfVvkG$X@R!tAQ8-vtVxsOQdy1MYl(}K>X<q~`'
    'b)NCs{5A240N`ZISg89wOj9vgaSlDY09@dCS#zV;Z@RmF-xCS<!2A+Wza?tI_6MI&g{IXxxgjC=uCiG;&u&!N5F@8{=wPS6{yqX`'
    '&u3nkpXGp1!~(iJ_aN!yy+Wl@AB%OZd-VOw;WuwCu~?HG<-=h9%yu78)I=f0!#bBK>Ws;{iNE;8O(@vGn?OhRg1seTm_!3y)%JO%'
    'CGedtzXXRn7ELYz{_tME&`wMLU0DFIXCE0_yHfxkiNN{kwDDNOrta&Hs&w%AA$#XsS`JWm?i&|u5a;&neFBYRV{(B$<(>}DorE!9'
    'aiuK2LS^rcdN^aZO26erU3v8t*Vg#HueoO6AImtF_yv~{KH0&G?xKQ|KA2sgQVtk(^@W{sk6-MaxoLyOe>uIdY6Ml!mWavuL_#1%'
    'cRk(eJuFz(@Yyi90fxiK2Sgrc$`)Le)zwjTUHXTR{(dq+4qkA%wSkDE@Hz1FPEF-m+ze7Gg0peeXZVhEl(~ZvaE86SGa!XIU+PMp'
    'V;{<MNOpYcL5jnMg=tbR7ee!RmMj|BCW8;ZzrMQrE~kCQofH&uMXjs@O*R#aohn077A1ovkb4%ET%C!?c^)1$ep)Ax(?CH_(F3z!'
    'T<a}ycwUXu%J=a#YD_Zr5nOXoevN*S({u|7olf5tb3$~<wf7HwJT6#~I-mz(xD>7*2Nm%|dd(gT!AmC@lzNVfZ#)`%YFC#*+#Y;D'
    'kT<G{K}^$Ew|qWKZt%0V^SoIIHkzekx8kGfg3<*L(!5k^kIitJQgO@=ncJ0l9m=%8%nNBIsh+_ilMWJI*fbeR(SlH-2OzrKxsIJ&'
    'WmK|0HB4dh=Y^EYD8cPMrSVI;L|1!oC&8#{R$xqEP%2g3b(G_*Cs@{ZTi*{&RcHzbDO3Q-1;Y%zP{p$Pc3!+vU9S?N(Mmj@4#pE<'
    'UhIR(utQCGTUs(N@2^|1+KL_dkxN|rQ}_ECLkkBhd6v^q9Jsgns^A1Euh|JVaB^%QS9~a5_x{B2K<`P4bD%c?d@B?CuuJpund@Ob'
    '>%MjT*pFQlr0Al+Rv5e;we$Xv-@vCr$<2E97CPDUulaQN^oR>>I#ERTpds)Q134W4f*0CHT1<sn{F+(M$J%tXR0Ss>Krnj1EZ}X^'
    '2Z9lg2m}R&vYek$#Ka@o#0G-60}mBMbdjsv+tW><M(KvWKA#PufeaI7Hcxu7!9#Z4(@xPkc|mDJ)Yx8pxNHeHpb$R)l*f!$@H@*V'
    'fY?%vFy6W!+Hd^ehLW>H!BnIxBM^vS<x?*QTCo}Y05YYgP9PgmP76O5L|pch$j{hUfVbo$Ya%LEhBLo7)<ZH4eOQ!~0-@8K_;YEe'
    'Hf?vsScnpZ7grule&`I?(vpXOmEBK`C*|s3=i<?>^jV!<!QQ{`ZN-g%p7&o1t~{AML<%1~6!b)jGo>W*mMnXs=U>Bvc-F`d*Z~@8'
    'MirG#&9OMQ7dAWA<RH=4H_n9D$fRfP_{GW+Gu$zT8cr$uP9&J+s3uuI7t%=)ramXAO{@hG2rxKVOh=k2SWNzS$6jlc@8bKqeyidZ'
    'wh`3aUYcfw8si^Dql~5*3D9~y`{c70&JbC&RpO+PaKKTqU6rZ6sijSy1zUfnVrE=0o@EkwJ5nngECa}c>!C)jOj(#K9`?mg10jkW'
    'B@iXF(_5rL==N9>PF8x{mje~GRrc?M;1DL@bB%H7Ug{Z@DovAuhk+2*tdVWx#H=}<I+(I$&DW@~yE!iAy|eA)cQQ_-D42@npMx4n'
    'n@oN5)C=>Jmpv7xPARSksqK_k=ugUnLZxKwli59wEFv|kiD8U%RX&~ljiYiDi~ZpWn3KdBhok<qC*xAUKZ7<(JMr!FylVGS{@$V('
    '4*v#D$3wSS__p!rwiu&(D>*YxeQu@4dS=bQatCprh~}N+jFPBjan4SgSpl1!et(S<?VOpC0sp{*?ia{-x}+ArNxx<g%?ydb55K0x'
    '6G1jX6dRk1G?I8XbiM3#i7%WPWRE$E5}8H9>lOGd0Z~|;#kwL6riyBMt|<5d=tkXEM;(zqc{sQI>Oa4es@XY-<M{x$!NniQU8=ut'
    'f-5pd=XH(GR2yvZ%XPH8v^EU^X`uBOct?5|_T=JF@H3;7p?DG6FGf2`0v21|w@a*AGZ4!_b*|)?j~IYll#LFP|LbPK!CEKewJ8MT'
    'l*cjA^jq@=#P{8BY_k?iY%2fFwq4%<Bt2%KPJMy%hmlyg6#lG<AajO|qUjb}Guw3Dgnw(NVV=Ets{_F6Vhe*TkQ21Oz~YQ@Fr#u}'
    'l*p~Yt*3L8F@0fhHPdo)2`ZKOK9V3g=_a66mvD%y5U7+?zcBey>%rBSBmdwtdf5i5l^SaDFhhuuj)Ha;Sm??_8t;93a1S92A`R{p'
    '$q<DF6>9@{v4>AnZto%ef_q!#Jwk(I%2H%}XhTqm>0QRhOBD~fJRx@e=r*`-hT55vQ<~V!$;*Qjh^Z!60B^@hzy(hei64Z4=dN1A'
    'UAw{l!5jYK@SlkW8l^9ibE`*;6H_b=Fzc8KR~yrdwam8uOw}l&&IzVcY+R5?45}J;V;Muk0*Dn9N7D)OgI0C)(YLHVbZ0W`$(*Wr'
    'Jw<=qpsp%*b_P#fIekgPs7nL4#Pk@af;k$AFwXFMmqWKjYIov8Q&grj%o+lzvM<+0a8+?%5uHMnCsDys4?hQ~?JrkAk=@IOdMmtR'
    '5up(YlwUW7zA@?{*ACX(%lL_lpz|<d-eJ(lgA+h<df#dFqli1xI3FhWAIjj4?vcY&f@B)A9?MhWV6lm?z@;x}X?w*m*IoiO0yL)f'
    '4RWMNJZ2ce6x{`;cvs*r_p`lc3AWCsmP59CkGpU>y}Ztg6Ns#E9aiSye$qP_@u4Xy7s}2PmBm0}tti^(v?A5K1kC3?0X2=W>pZr@'
    '`*S@rG^o%j5da5Fo;M&8jad5W=!Lv*{Hm|uS(F8?`2nX?fH)M`Jv(&=U`GT@u^={AdVOjVh7JoG=CRXmNj;)g8w@InT)55&sSWQ|'
    'Y5S)$Q`bT>kvai_@6?LmO;zf}j(249*2_acpN7EdA+TX$>N+Xx?`P9$QdSv_#KT_rDWE?fb%LtcQn^%YFSemXJ7cMbeJTIb5+jFt'
    '|52%Kigv+blmWN^O)KaB(q1A9CXISfUs*q|HYxCC0UDa9m%09$>E?|UGm*6JB58ex<j(cECuaf2#mv$$d#Q<Aj)j?aM%UNJ$#ZB8'
    '&U5e(e|9uf1|neR$s+^1nF{M@V%&4k{xfiWEg5*!ip&${UYmAx7|L6n90o4gZf;bBwPb4_0#Z{US2%m5z${iZ;40KIbZv|)#Y5)7'
    'm?QBTT&(7xr|G)9cT=vYTB?}_^Rb?4rea*$PFer=O}v0)BhBTm4q_fURYVuGNzy78D3w>GBjECWDI{<}-XE!H491qLD)P1m$Fq~5'
    'j_wAk)LiwHFBFIvA23sm&KQji?}R7=kA@!%_$mfP6<_Jo-;a$$qw5rJP+;{Ex&sAcv8k1KbWiKa?3|U=*-|{8s-<t!oQUt~bk|fZ'
    '4WBwQtAjzD>6A1>1XbrHq2rIACq3qabt@{3aE%2_5|$=7yMfsOBdw0L-*>W@&4fLgo^kE-5XU1UR2v{q4o#Ie@EPaJIi%;l)JhB~'
    'u79rT7fwNl@n*h^GlM@Bx=h{AXiK|zN8+*NJ2D_;;J!(ux@XBD?)Awhywx<Dtygb_KQ0)~ip8ghcNC4&AMIu<<%#p;^ak9tTD<ny'
    '$SS!fC<|EXAUM?B$USiov-kDb7^c4czd8E@oU7_@JDe)r4l;-$@5+!pI$42Bi{ROPw2RE57xE6!1l?Ke)^L)OR5`mtLI7tPv;6iv'
    'l)TWGizmtzejnc*)jF%ld%q_^Us?f-#ND4-8WHW_pU7t>vzqDm$Q>DfC|gRPzY}1gAMkI41>*l)oLdWPHdGcA2FrT+7SK~fI#?~b'
    '5618P6?b-u*sSnyABMdHWnhbud-j%oZ#Pl=ns}vN5d&IqFpxB{S(o|=keB7GY#3FKDYGtr&H7Z7r3}%E>K|!XfCl#A>De5pCKgVO'
    '9nEZ;7fD9FGhZ%pl*WAk!2mrIz(5jo*aoU}nb>se_f3UXCc3#hvA@DAUs%;On`kAiI?MCUnlb+@Tvg|6Xv=Ek-rC2YbRa0f8JPGG'
    'Hm#NA^f>DAk&@KMbsMT70qJNNC(4lVLOoHRWeAd`ty%?5lyCX?!Wk&h%j`Dlt>CyR8B3Ir$ks54R(&M00?-vqeD6a&pYE>Fqsw8d'
    'X`I$XxGedRvK+i{;FPG08){m%e7(J_VGNWvx;PV=CfW^v%bR|IM-OZEGO_O-5a+&;ySC31M~!3&k%RX-K*x{Cu_?ayXGZ>x<3C)B'
    'pl|~i;~?WnU#~zLF#px8X@rrR3`<4QRiSn+SFD0Crs|yWS4Qa3{|d{+L|D}vfL6G{jpdT0$yIW%b7z7dns0?ElMojWgwQOx5QI8Q'
    'mswarGEZOvdMH@lsjGj6lo*t@8gYy~E%^*A^B{2FLxFREHX91b<j1Y-bngfA8gpjc3=T-XDE0)?VM{Z9sNfUi_YS!gO^Ev!o*nl='
    'P2PUR^l*sZ3S8Km3N!GTt{Ud=;dw+Eugq$n*AaSOgvl=a@Vo-Y1LRc8NE+spr94j7xQDq!`9gbLCFQzIX$M^I>@e(j*a$gOP(~O0'
    'Y~`SFX&h)Tby~oaS)|zP&C-yQkK&NX`B8<cA!%e&N{uV1=nJ<<){+>kD{71>&JRoUrscZl8Xpm_iOXWhRh8G|#H&{!#DxV1GwWSW'
    '_C-a(qK>VW%Gkaoo56~TZ``G^_g#o4aR?1s{J=?01X*I5@E0)C$BBc+E>p_RYa3>MIY?9u?4fP4Z?fosPqVTBIbr&M1m*dS&jJvD'
    '?F8QiSEh1-*=8qkfARw1={^&t5A?k2W}Lr-jEBTtdeBB(9TfH3J{)&Jbt(`?8eM%@Zin%O9|2jQqu-zD)=|K1r*jimscM@eNf6Su'
    'ZySE_3K4_SURzP1F!vX*$f)$pqpT!hYJ0<5f|`4>(t!8}uJQ}>n;;-Cv8=_FCzcWjRgJoZB{7o)H=RRNA6(7{%XS;aNR-$1=>Q@f'
    'gi|6mLU=j&2-s)b+eWZ+yvI6z&qq+r?wWiAg+)?Jo@X<TvS6sFQm&vIii`wvRnJ2#Bc26-4$WveExn5yFc%YwTREMi+L72ylxZ35'
    '`(~$=6=uhL8iz%9oG0vFp91vDZm<>dS;PD|E#jq?;-MH=Xqf38Wo1w)g!d15(aXKTA-ljp!10AqN42{e07J_UbjB;Tu3>#dJ+QZ*'
    '!DPP49YI&n$J6bcE;%S8$&1StlZcrm=oTi|Vwi)LY}6WGm|$y6=a9wg>J1C(eFxZY^u!E|CGjI6dL0A~in~W}jp#EyMjMvdG_480'
    ')J$gfn2|XGL0N&Ps+9P?Pa!z@3cS()_Ibb6xMw)y)!PW*G-LXZBlL|fCzzzo_8LC2O~8XJI+TeF`}^8wO1>v22P4Ebq2yXde_@=Z'
    '0ktuOzHBE8VUTvR1G(l$<hG#cLB-tDnL6h#f(bbsu_z*V&<KRqG{y&ffEBbVGRA%K;w`1VMwbj{wlhyP!_1&5)HEqD@1yr<SnB!o'
    'M*R?qD<+c_3D);tm7qZUKU3Q!4HMQnei~)2q51u}HE!j-i1*oP?v!eG?$P?>*i`fbL?BGzqYmt!hl{`l2Ci=K)ERN;PQ}wuJ6VNJ'
    '#q3+9jWVXs?4!nhXjcM$nXpg8i{-oW@nA}v+L|6>;d+Gvh$&qPMF7dNeWAFxFJ@R=1PI}2UN$`!k53&hmB20CHcG5e?NEK|&f-L7'
    ')dZ!_whfJhJlF9dBw0MBB@sgif#{|^4`R77y8mAW;bB@A+r8&YNu!+zIS7;fM*oU=s&l3c<R;Bf5w?9b!^c{A)V&Lh0e>PU#flLa'
    '?E~!OCUBf>3Cwc3d>--rj*<L_)8He`DF-th4kq@ugf6P3izdF8I~6g>5<K=ggE2E>VxWAdB0&;_evd7G6LO1qoNAAEw!g%Zytfk)'
    '9+Jk1R01F4@KZvzBA|m_8NU`|lkqvo6G}906`b!I2j$Qee%wFy%--WVUMgqCOAE1wC}i+KJI%r!>Z2Kp@^$<Z0P*jU(FFBT|1IGg'
    'hP90p&7!z8uLyPVxMiP)MZ!Hf?Z<}{Mu64TT?r<O?g?<e;gdh9pA?J>@S>I7=R@V*Ja5J*>Dtl)Zw%9MM=B(l%kY@kGoNC}PX<?P'
    'd%_uaT8*&ECy!%<t4>lxP^U?X{*ZT`+%!Fq_}?2#S%1VTb0ji{9tz2>GUIOn^WwNKeenbhlIx3wl_3u#%?8t51r22_Vj1*+Dm%Z<'
    '5+)&}4=>k+(u`2G21TUfY{1jR@}Of5bMx||(Jvv}fPn57+jy3E<To;D+&Xg_S0`fOsJqy+exWUlaYj>}ScLoWeO6{2b7<R|;6unT'
    '?HhDL(RKW|0t4Gy2t_64sUc&SI&XNR+_tquqc@bYa+vok-{2e;{5yQfCPJz%!TO|3hogeVz@$@3rkBtGiFRSg#KLnx#b!TMQuOD!'
    '*A@L$xpw8z$Nxrw9(U`^=z#+znCBrB24P32Q_cA8C0A~F#%t~)z=UlqD}|{{Wbh0!I_fOzSt(5E&cqKG_4gV!8Rl@mm~9IB7)puF'
    '<Z=^8mWD22x=pr!d%pfs8rX`Y3uXREzm8}3a!xglDF`+BmYbxn{j9W((d)!g!8T^mrJ{ZNp3@YmCtDfF#;-A~@bB%V05l>$$Hi+1'
    'u55o4{S2G$PuJyd3=3y>{XcWr)AH>#R0AL?O~N8ToLjLI{5j1uU|<C*<|Pg_h-)v^@5tW}qagNjw#z?bn^wH|);B+9te1KR79^f2'
    '3=^A^5Wm#4<U~d?pD2yPPLf~q0*d~ga)9s$o&@4>YW2rxDZj_uRUxlpFf53^_AqDg)wOp40{bZqSo2hI59teQZ~`Y6Wn(@(YL=Nl'
    ';diImAG2oO@o0D-%y(rx!e%+R4YG$`pP2@exfg83GlyG*7@}d`1O{|9Y5hSqD$JC<e#Tof_10KB(ywqYj{%;DSzog#J-|%}{W<3g'
    'ePTP!pVM;gKJPuva7|$+?m|WqH21I<CfC4ff7HUa7d^MAUywl-#GJvPp$*0qs(|1Q)_mEqf|3YQURk*T<E@3!1xtlqzAW7UyL)0b'
    'dQq@{B>^eJoxvBbe2s8Z1;SuqnOL>UYXyGTNZ=pi0I28~D(>SLPnFXUd9>=PRF^GzJCM(NbaShy9<dIgx+~O16%+0#z5dnoGmyC*'
    'Em}XpiSEnoMR>dzN-U|u<XeqW&WCp6(R0_Q;3fBe*v`9f8{BZaba&%^K7dyce7_QWDrsKW5^>w@jH)$LtTHavKUKA&iPS`jR-sGp'
    '$AZb8fzcJ3Cf3fySOrm#=MnEQdIjyaH;cLn_U~C882D73K^N9f#(z_F@URk?wv*6VoKm7-SkG`IU|tRraXN#3(k2P=X!b%_XMJ<n'
    'dW8BJO4{*EcPo&0D#qX-uSKjh<6;UHM&^uZ0_lamr$CSykE*aT&0EZXDjt_v<>=_>?tWY%Z3%axD53_b62aRuFst0f;o_FC&~W+)'
    '@O$R)h|dXavXL<XIx^VIsQt`;yWAqSSzK}lTI;Ra0Mg&T>zL9f034O1!R_#{Zn^ddLI}h2#W6vIyOe5Sl~-j|HEpmR!#PYY%3n4K'
    '12^uO;|y?PJMJ=}R(&GgTc)6>lExB*#&yn@01L-Cd}?yKCf4ixiWy;?lTkLy__VS8KK^!UTIDY|T#Qxn_t5D@Foe4?IeScewU-<h'
    'XXxx-?D|bk*31RM?&TSjSN!VkRn346*6+s$bl*c|zg!c+u5=9k%I?q0?vUgx0An~CVj2fTNm0}E3ZMFEwXh$ezQ7g>clK74_5?^f'
    '9`n@`f!w<;$E`m8^;R*9#J6a1?;T>o-H80m4~crfmlqq&6ndT4LRbMgZ>k|o)=&T|fl-UMV1MGk1z0<al=D=}#(;5b=k}!WK~Cc<'
    'u;qJ&)ik2^CB2XJ{0sPRjpcFeKiPk~us#t~8P*E(e}JfiDM*EL9!UbJE+imu9q_8U^yjBCrjpmE9J2?=5X37TSn|dzt?<6H0Kl+1'
    's*pFd+l=Lvn=Y0|fSJwRGVkW~C3EsuIPwTZ{FX%4GD>rmLZ2c?k7Yc=%Xwa69A<+24Tm~kdxCC+*b2c1dQqqx%dDXNOzfVK`%DQG'
    '`^*YfUtu1S7<#wv$IQ%FO&9NA`5#ETD~C|SwrRfO7=~okO!Nh}T+jT`*K~pXoj~gTO66JNT8*k|z?^UhAivH@%{Kdrvw{bNF34jt'
    '_YXLJIP#YNd#GezQ&RxVHrdVm7wWxR?Hm{eudl8dF8!p!H92ow0UAspo6w041|>MD%laA1vwpyqX8I{X<4D14HK-Zc3*Ax^P&zd^'
    'w*ElyOrmSywkKWek+Izj5in3uR>&%Bi4yDmc}(@!Rs<*arJku<^-V&fpfq^cTVQ|N%6V7-JU^*AkF~&dht!t+sC+zx3I-EDV+NU%'
    'UasnQe*wfvK<;6z1-twHgfkTzA-?$LkvLweGVo9Pl})4}f8gCyi%c{ZQ@6#%iLt`7W>=(UMx^nMzFJEZ?GRA*H2<`v$cl59i?BRt'
    'i^q2euhOjJuU9Jwj@(9zpY<{b+&=-c$!_A-O!nSVc3wpVOeEpl_p^zN22mal@M!9NXy8N>FPw>X^>yxI&@q4=-xU`HH{3{l6*rs('
    'HQ8?lP2Q6ml)&pN;zv8uo{K(Er&G)nM>-)%b&%QEA3{Sg>z_XlC5{CO9H7o{PO4U!TZzu3mIJlF=URVBrnz}u-Zy1!)su)x?_O&A'
    's3a72(bP|sf+t4{Yro{>g*g_M6kgPA!;9PM2uQ%BK{y1XL@*0{%VRyIZrEyJHf5E;3Y6^co*?n*kukDG?;~SL=XlhmVOpMNar4X~'
    '0F8qh&^U$U#Qb`+tZ;`4**X6MGJOp~AbH}<j$F98aqLe<DX2e}t}#Mmn!%FfavAuOOaV>_B(TABIA{luezija<Wy$*>;Vsls)Ez~'
    '3gNR_n&8ItKM$@A3-Ocw3s}FKDpm~&flNjiJo5)|FU)=L$Qzz+Q3$e<MLl!P$ry*1a(9`iKdp-B-1dax&6Vq}`WkC4%4*acSnz!y'
    '=j);2k!Trw3nPZAjj?C%I*(zNP|mZVqlkysck1gKJk}$b=YZQV^54Euk?Klhtf%;95K0_U0$;EQxV#KMiMHYMO)WVkdWyM=-+E<Y'
    ';rB%~)n0<Fqdd#(RpvZQy0+!LdJV>zPHi)1mBWnSlm4oJVa=RkEy$*}v4F)oi-CyA+N7-D2e3+yNC)Ll(g4mO{&!9sUlB^7GF&zk'
    '@;F_l74w)GzWfZBrc96FjWhYz0Dvt0<esy+=hX1Yd=#|V_?rqCNJur)o#5np22pq-Y(2T0ug{yiZHA)cXfh9@pWozz$Ysq%LI2?j'
    'W#y28Kp#fe_y4ugmBe3(4E{wwzlVL(=*r1@4lSt^wigf`%6|UhpMTlUU-I)CI>DIgG|3ZUGz^}hmLDy`&yVu+&-(d0{`xHrn!8|v'
    'c3z_3pP&7&pX29W_4Akf{Fd9%mA@W}|10x~7ozjm-)F8ES2wXCHuUV^o@@pgT2U)8!oKAUtp?WZZsJdp+DJ5|j%MkLV~<%q&)9^='
    'DjtTI&+nS!bQ7Q9HgI==@<&QyR$m3EOWIgM%-9NmH)oXd=b)WDep;oVtCn^n<LC~v`CKVb`P?7ZXT@r8dGb&J+~WgD2ZQ+gpbEJ#'
    '`R2T`&k*U8>NX<Ja>e|d7kV4Yms!^zEPKr<Bl9}gbgkBeu7wLSJ__VTpk%bcQjbYJK%=QSl_D>C#uQF6Sb3{KXF+UR_>u5ePm1b}'
    '>_ikPI|(R@Q`vhnvu|Jmi(wke_G6t?kN73-!?z^AuKz~-=KnwJ3>5ILDl%?$>L@932BgxG|NF1HC>zPQY)+`yQgbWvHjx?*5RvH#'
    'hmj&S?r<J2j;HGBG?F2Au*-A1mt(LW)juhaminb<gkO$Ab%^p&aruA=g(L<Yoq5?|i&yF879AYw0s8@-#OJ&!PE)k7YH_6O1lDC8'
    'vFR{b&^Ke+&TtOvF;+9c&V_8o0GKd1$tpDpZnXdWTLLxaKL~qeN3KVpU+4sM`X_e{WX~{nP^=i)6w^dRcyx`fi^zg1kX{WKWjwTr'
    '>JK`hAra%jhYp(&eL=N7)0}YV0*9e%$LbQQ3n;`;NQrWame;IqJZ=8fMmv<u$FTWQzPb$QyBFj%<?lEdXA5Qq?}F*UQ0IPr#SswS'
    'Vb%JFSZa7<te#PSwP~KAr03?Vq*k22p*BTX;6IocDTbK-9R^-8jWh>?9n>+*Kr$$!(*e#iQa{-IdKQy^#ayWOQ6lo0u|UV=h27i4'
    'JJz@MQP}_N+5m*0Wx)-~Q={W4>2mjn8+riPuP*<Z$&QSF$uQND8JSV>F6fEohDKe9z#BfC!Fr8~k{QtYO-4DshlK{$(ffTDw(4TG'
    'atTgXyI#b%<L%H2$G7e^1f=<iSL8MUBg!7$aam**AQS#0t{`(Yi*~yEhe@lV+h#lfSJ2$q`<`QvAF1nuq|G{j#`gnma-OJ;+hyCy'
    'aoaGnM!kUj*B|tDAwakIB*G7ie|94zwvn8-DC^}hsvXFy1l9-4nM|Ra{J6XB`dv3Ht9VKbz5;OQ@aib2ML-4VfR0ena}oBiv^M-X'
    '`%g1qma`JL<uw-qJqKG8OA|z!L-7MaBH)_}oillS<&S>vBks8&_K)g;*esSbFykkIf-->+THa=aUwm}JGW+_nv_Og)K7+%#ia^8!'
    'wE5VQ=wX;H3`B5yDq8SVH<@m_W$E*Y0~A%qs{Ak$hB927G-C!qRl3liV2(x@5=y8z7v>RXrxJ;Bsqo5B>h|@3)7P3$1fjpn?bioh'
    'n|n1PjATH?U~&?8hpH4CV>f&jffK=y0zo(xG4mk}s_b@;^L_@-N&wXraPbSp;x(!nAyOT~95g4}1fEh~&nVJ&oR}PzaL2G<gX}eN'
    'y;lHnsBytrBgvWdi3GtxF&IFDyOGgcRUg9KA56)uZ`mxt#$ZiJ5Go4Ay~_Sg2hAjoVH=NA&`laW7MVP;jbfmLJCQ)9Lhjr|t4fP2'
    'ZUvM)J0FF4?3+a-fvs+o82+rRx172&H$>uH*!E0Bh?3R&$z=@8>IME;c(Y$=@pzsNDED@GGN6@aF3<8G#-4xylR64kzeI2pGd37v'
    'tMO&@NSlDOO=l42#$x-0Ca>@<yEMAxwkBfs4*q$uG8AZ%qtaax{f*d6@&fNi+kM*#m_R%hJ!1+Sc5jcZiKN-Mg|Tx$*wH^D_o;^@'
    '24)wFI-Tng38E|5Lp;lH{@5P?7%285Wj9A7^Xy~^Z$lE0ne2x>-S`YyjIPmDnVL`9nN>5KWEQvmbJ@h0$<cAy&|*vqxlFM;h$;BQ'
    '27w=F2}b)(^t~?6U}p+ucw|AH$9ZBDU@<=fs4Zf6jeRh>e7J~;m}Y|J_gnEPR9<JA)5=Fr>4kq#+9!{8C5Qsm?G*jb3Ce_=E?w_s'
    'wO-|8ZK6TDZ2H4^iY!M-TS7pi{PQh6_`qhi9D`GkjWyF;T(Y7nn6kJB>)z@;dlV<3!@s{9Yyi<(j6I~u9$q^-Z$6@F&<Oiio=LMt'
    'XM)UE{jTel2cixCxh!yPDww@t8V6HJ8^=4@q$=FroZLRD#diLj3af?hv58b=rfx2nISkN%K^K0QGzIIuIY_my8mpN{>=c`z%G{*N'
    'xo099gOGKM0gN;5^Ml-Bp~J%YbFVO@2COCacg?G+$v@NfA3m|6JLB1X><px)Uu7z?XiL$;OXUeXwCtf-OQ3(C;bMlry0h{6Z?kH`'
    'w4c{r!{V%oo{;*)YoZdDDFPD}7}kx<Pa;#%oajkeXkZkO5RZX)BUs5`2pJ#XV5ONd=RD`ZsZ%nYO>fnVr?07MvHFh6qI)tEpio$x'
    'uc+e361cLHK+Zt#EMoRJxv6tdw4N?CmZA&sc41QmL2mQ7(wTO`qD-&=PXXf+VsMe?-l!^)>=gWg#17BXK2Vu14^O^xZkSsE-6E4&'
    'FWW(R%G5*f0f$Rc#D)9_(M`eTV>K~Z;UPV@FB*cJX<XJ~cETW3H?G4JabP7)3WWVFrYMsnc_t&q<Blh3fhtqD=rGX)Wpo>sa;2>C'
    'fq^=GH~Vgx(}e{R+3+l7)L-TvcIOEcPK1(aFqn66)F=X7DaGIq@u3LARzxO`#Msj+3}y$@yhrEAI3Dhah0>1F^@#8s;v9|bSv_(-'
    't<p;8a~Fb65%~d-Y7mWaLb?N^RR~?w6VjGZ8B)ih8Y`OA<&vv8BS#Le8yr(vKWZ+Q>#3PgTye2-D#yr>;sK5r6bhYqfdYmOqwqYV'
    'JFRG*s_g+uQ-d-(G7S8_|I18o1%oj1Pkg{@7!)o2*@3rzr-S=3Tkt3Z@(NzfMgNENz;9J`7$I_|@HFQviz^|FJTFY-v`u%4KLZs<'
    '0#k2KQga*3Wa(iS4-5H4m=t=sV`Mxg1+<#vbS`8;%<B&eep#|F-ReKvQL_rKuY^OtIGRv1oZ<=9!~hSA6zA#I;$84vIyR&?VbfzS'
    '2ZvQW>JBQcjuHqwZh;Jo^l_sR7z&3opQ|8-0AGC1MnVu-l@JFL2#NV)aCEs{0+>yq*}OD%`^5CY^r@rCdmMc;Xhmq4AeSxN2#D%-'
    'JW31<xrb%S@iAn2gE47Kilo1w`)t{Xoq94ss~`>V(}(hd_q$=gM9Eu$4)C!x^q{n0#eEGkLHOqtWjL64R8>c^3H=KH(y=qM)m$Bu'
    'b{avui<lNt8X~juw*5Od?M~9J{qpa@`q5<K`#+%tjD$(o;mO%$e#_;5wAG_RVM~7|STIzo&bT?2#7vr|gN%s(!vCd8!=5PdrlYHX'
    'mumgS%)=~ck`;}v=eQs#aiYtj{WDJqHB`RJ73W_%_!<on60l`|OsFYMH4)}qDfAVSe;>*0%LN{ZlU6(U3)gS5+n`$7T;{9fs@1?R'
    '`~uvJgEi_g7i}mO0@Hcl=a#f12$_|PRwpIMC#o@-&upNusBP-wa1-&?PN=#F3Zs5ky^qv?m72l7gLxn(lKVttco@iBUb^n*V1m$+'
    'bE#p5ef7bU)?LR&UUif3Q?5G1hTL>Le8oB|lj*BpqI}s{x!f;jgNxoK>j8RlGG5d6Tgl-t28q!dA3$N2CrpibBSdQ3nyj>{_ZmzS'
    'eMi8$5%0mX&Nv3tnW4~>-ApIc@5#>N*WltUFjCM8$Sh@2IvV3YJd~_2u=P$BtyCZ#cxjix#B6Sp2~QxbI!b1RUFGooHRz6faB)AJ'
    '-}5YPL~I<4jD$@dAZPXLl?d`w>=bi+yT(Iu-jjh%(+Ga1A$Uc07JeREBuL0;xJP<-&^kauW6ryW#56YLwgLH;I=54ELVws8S@7L5'
    '?HipdVU%PjBZ#g%)42BMed5)ncd=gV+`-|PwaDbk+z5CIfHje>bfa3h&RL+beKF330*8wv#&a?picn^bW%B2+VnvR@!?BQYGp{v&'
    '_dm`MJczS*VrK4X8m2A=j|Uka4Z{U77R@Ri;hLpekqf?kCaVA|2BmqQnNukMo!gbJeQKUpcahEoqdJ%@cgSDJ9Sy<ep;ckX^;)@J'
    'N!F}lIvgNGS}KL;Hr--I{!Wm!nQ~Bvx|Y#QZo!bxVhZU5{RsM^@tY11KNBBWCSSO4n%q<VMFj`Ak7FTtW7`FE*1?F1E0bz|`()SQ'
    '7}@wsB$LzYku(>6L~z|08G3e6$KSplr?y*I+3!2GEZoZoSPQkB#falnj0`-$?rQ$|2A%<TpHDhU-zGBRyP}*xBFVwl8VF7vg!>gv'
    'WOy7=g>48`nw>bzo`Olf+T)_oC*OvypRm({ny>k+S)K8k#dl2>X?=%>6Izw7(1_P4D@K)GStu@AqbF~f8GN0y{y8&J-u#7jLeebQ'
    'RLpn?-4)AF3^rA@OMC<MA@c*%NDD3@#VRH9MTabcg6YGc2O=V%s1;H-sICe^9x@tGO|M5y^;L6HLKP<4;)<?J7eHuf<n)jGvZAX~'
    '-;F&&!(5fC&ZL~mmPz?N9RICq3?Ih5K2Cok3b(2jdCnZ<v`hhOz-XE{#0(z+9>rY3TeiTWz|n6GG@0lVsr`QN<c+afqeX%Mt0i%M'
    'V77eRUHX)iG}0V9uPqVdc6K~U&fAy}(=8t;1_2uXFpKAhxqpH~<CBTTxBP-`WG>c1cu(f#Qtk-N{pi=_eVYddrExUIf+$uxZl;WX'
    'wzC)2W(54boKh54wa}LK0q%Xp#Vp?=Acq~^AzVH6-i(IIZ24>08mL`@)ARuE38v1<q6q##9KC&y*4Fxh@HCniO`VV4bV3UfJBN1;'
    'XIbNB=8FM7#phKSkC`vNGEZwcj3#UpXUk5$5T91vXMbP}Usi%HA_-@`C2XcynA4jZtHz>j>wLsyShW^mkYgYmg^(|JYny^{2wuCt'
    '3SB05dZ)R;VWln;3$#10L%EN(6ln5cTFijExYR|`hZ%#sjs=}V1R%FKk1Q%m|CkTZ2ErVysRUC?>dd}kpm-ZPFi(m<b|n*Z2?~hE'
    '2rnoB(`DD^*0VU%(6D_@$l{W(fEkE6vMYLk!swE*Z!u%XVyEGziKwIbbj#yq8t(qbcIQFf!pNn?KJ628c(6T1qjy&s7gpYZgNe0G'
    ')qafLcx%nq4IaslJd6M|STQR>S<`5;5m~ivA;;8|84kM%eAlcL&l0l9UJJDanFN8Jk2rI@(boCHE`^9MgwNff7Iu)r!N0vv$Na=v'
    '45J~$;eQ>qKr7Q^8>cl=HiN8jfK5r;Oai&+L}v1s(MQhm7^_A6pgIz7_H-0OWy++hf<1|Wn)YuAu_HXvq_YHjck9vSz^>N(^9-A2'
    'x<!HFvS*k>{ejO46~x>|w=va20)?0eeb8XkzB-bs$gKmu>Y7XfwJm@K#F7)wtE6R&ESq@%kSPsjJiMEo6dd-4;28{k>v+^?<M_i8'
    'R%HVQl8JVg5zfJ|MV!h&`O9dmqFETrT77@w(SrWj%y8k`CJe0~IsKd|JP?yx4!!`&BqpCcSPVb78$sK8G8tiqfJ=j0Q{i5cN^AV='
    'u(IV%?=6{1PgeBEuHTWFa|H&<+Q3)NIlD8RFo?b;5;Z=+-2p~Wv3O`CMPOo-90j&&F1V4yoGfH!)ZSATFw6gEI>o+P-a%gmV-8H@'
    '!>LjM@QZFO_IkQeChJpE)0Y{q1OosmAZ?iZQ#4Hj%nx_8+dMX4$4Fmdh?&v>(wS!aDuC%i5XN!B%Ld5lTH&Ye_QH1d>=~O3PGe`{'
    'yZXFfp_W#O;;wW&w>>frCRl~769?(2Mn^>I6wRW)7q2c|tnW+N@uG@0{Cp{0q<R+U=kRgJ3D`**r&VV3Yu*gHc|6UDl&4BstTLLe'
    'laq`I`E?8QP1*868aBYCPF?FS(r6fs#CGzJRqVhJ6g+!}%fnD>W0)i&ZuZ}l0p?v->}aanpzT0xYO;)zj8Ly{c6?Mz8vyB%mZXaX'
    '%L@QX+`^}<N6tKr92hwc23K;Z;EfQ!Ra(-40UV##JPvzp8$TNx<JVDtQ#e5Y)%L*D&EyJ&{aBQlzQ~=9l}XFI(<XBF+LFeN&U$b|'
    '&HhLu08*SKTBylBgbmRUFf#35s@mAE^ODe)p7Ah#r#ZZgZu@_}!-eAB?EvK(-<M1hmZL)1{H#uyvmC@F4!(gPG74`120NG7Y!5Il'
    'BPWNUf<r!FjwwY4PaW*t?~O8s785Gwa%$`(aAg78%~R_sqo`&OF2}}UGG~Fw1;Tm_z2wrJ!fn9>)Zvg8r7Nt1PJ%ig_Pu(tVcju)'
    '&rw22zS&%^s^gah1{pFXCv=mDH7bM3m&=HJV~JawpMnK>d=P{zS%Bz9yBgw^8Qu)bVQ1X(YFVHx<nZiD%6uNfA<PHp0`~_W$YQ#}'
    '2Xb)T!A6<P<w5Xvj6gssV~>CSYz1S|Sv4d+TgYxaDUFBm0^JWw3^pFfhdRR^hU|KZAf(-Ts#;N4%gX+#nd1$1nKAhxdS&3AA7HSz'
    'KmB(F?I5QbY+AT82GeMyR{}#305=n0n+em^p@W2FP9ZPSKL$Q1;DP3p^I)m5qPU#Wyy)9b>Y164s^BP!?48%s9-4uGq4Y2hAW_+!'
    'k>>W9MU!A7N{m<nBW8MBZk0;3#FxcUHxr*jvoL~9m7cWwZid43Kf2KX9JPxKzC&R}96MPMCfqjUkew&u{;zbXMbjn89JmfcXoj9m'
    '7o1F<nXh&t=)l@D_cZ0MdNfnq74FYW@GhBUzX8mxM3!z_^HYR-byYhbn1LRz<wz{ypwS9^#J4n9Ph4jQvj+!vHGgOnLcnmUvQ2)D'
    'T3iqxJg~@Pw+bT*ij*ZW_X~O&CNj}Yu>MypmcETqa%KI3M~Up_fn;{5{4aSuD|cic8*J`Pgd=9cskU_Kf4<#=s0KuuTl=U|#378_'
    'j$(ub5>nW49%a_@<D`?W>km3=T3Wx^rVYdP_DZPW5Z+d0Z}7I^OsgSvGSgad`6Evl*6DUHK;A&(P!J(M=;XlCCpT7TlMMsI5NeuI'
    'c!n0l%=uk-+j@wnS49cQ{!kCA@Jvcl=u>Gemj!F1(}E{KlR^v?sFIr0EHBi}#9L-E_6vG<^<pf)0k<tAd89`VREBiUy^rr|-YR0c'
    '%~6_YRwkkbJ$8)Uf!FyS#Wjg@9jMAZAzK0^tHJML07*v&u(8!f(7;u5L%VIqdD0$lAM*gH$Slc5@XG?7DqUfxG-4ozZoKE1H4@ay'
    '|JojOTM|r<P!buq^h)f0?#YH6h1^H<TbsF0o$1#i_-|Oqb0e`x;T-L(gMc2M7u4;GHj{vP;C|HVNVO%LzL@pMI;bOA3VtwDm4V{&'
    'Hb5;GlhYTg3>*zK-9wtfO87aGMzli*o;*}Ie;?Y-fVkj!Q@-r;fE_)FwGRr`?$nUGVpwOL>BY<V6D)0X9n}lIE&DR_xQkrmnL*KA'
    'zbX#U)}A00YI(5fG`!2BW@iq7^2#qugk)et(FJx==82<+VjA*v!%X7UVnD;~1Li51&{SxuLy$#b4g|6U?%h@eA_8}(61CH=PC=gC'
    'Js3zxTg-1uQLdOzh5$i;qahNflqXN#*Bkv!AVq2MN~oWVjH_@K=JZ1Hot<x%F_146OCa?We>U-?^Nt-TeymS1j~cJ~3hnux59AfA'
    '5Jv+OEE0ac!D|A7?iPz|w7N#EnpD&dzph5o5Nq-Yp%TI>I|$ZU#|FFt;Q~yj*c=q#gX(YL4lr?E<;#*It8O!n_+Z7mo9|0#^$1%z'
    'e8ti5M|+WQ#4J1`739EV4|o7#ABUY13y#yGK{dv-s4_pIgAOd${(e4}?fCTDK8>ck0CONdEN0$d7#I!%4RGy5c3V1<f<1ZYx2BIZ'
    'KnX}_<w-DXN?ylcREXXi=oe}@2DT&TgOqyrNJS$)&X;`_2vCTjik$_LF=mYw9OLc7mQxr<Bl0958))l5{*JVQf*Vu5KOi!4(lnQ}'
    '#H#lr43QzMiR#8#A#JV0f&OJ>y12?A<CEb)94`ml3FVkbH}Ytm>k$BPfb3&?Pce@qn8@u3;b<{IXQ<5+QQ=QWNMnZB3>bWtob@R^'
    '#{4oQO(1M&QzRmBNeqfM@NaZ(j4hpLnja+3Sx%l|bpiQdbtQGI;)<}`LB=FnTSLbfxN#}d>ft}`nJBa}x6G8Cjv)!XPv?naZQKiU'
    'MQUJXPsqtD=AdDQDd9P2YT<1!D&ZO*=<!8lTaDGMFq_bf#-53ttD6sAt$CDEG=ZTja~987Po6@hrXkd&SJm+=gX+-O9wtju3A$-N'
    'WZ!s!Lw=F}d4JJhnq-2nCO+0kYM6n3Aa6BA22?E;%n4F>Q$#RV`RDaJ(Q;X4lGiRMkXeag)pK_2$+03WVaFSZCy&prZ#{p|C70P#'
    'CGTo#7x^TNu<!G=tX<Y|?PRRpz0#m=t0iQK!jcIcOoi3Q<~#RKu~H0)`De2Lp<l?*#zL9&39ln&gK7#o#XVNmsqV(K1?&RG{FH>)'
    'p6NF|Qb;H@$7HmC&n^2MpoP2Mnb4I+Ni;1KB85|j#se|kAyJZ8h=Fg7;Pio65Mkni!^Z}5VY|iJ;Q8*NnZ+9tnKnOx6%#`@<c00@'
    'w;+xzBIThMU_D&rVTS`M-%waJy8E;29CkJ>#OJ&+^txKOP_AKM-BEu6o86jGJRpfE{ujoc<%QWvLL}VRrZuOfuv*itGxQh<LH+=O'
    '4K6;Z08B~Q5)8+13J33-#Qv8?%LS}ABIgvw3US~p{wS>}04H%&-Kq6d>QvNm@v@8a3S<T>(5glO^Fd5C1xViD=EE;BI}h*67H~6E'
    '>K#Av2r))Jga3RXL8ka*>5X)E;N;*wn!tnbCuzh)C<c@ZcfSHyi)V7Hkf`1dsHSBHy}jEv<R_}?>DY0MZnuVvYIr865GsfX47}JH'
    'Pc!&Hyd0MCJXY9@4n$>HqBHqNXkVZ&ogM#t1<G)1O5g*<S0Qi}MOa-A^uTz{_5`~PaI%9i*hmzC)ZB0sWga(xOZX8d*St&#!_mNs'
    '>syzk1V|4`>@43^ZsV3iXf~0)v)M<CcH~zbe7;LO|8mg^<yv_H+_g~DV509fOhox%vuwm|${Eivko!HB1OxL=^V&P*yX>06<}1mA'
    'hN?GJCB?Y9Wl!zImwVd43brL9^Bk078qR<UPL4cVq($ALLYKC~{AEK2uqH{>G*jD}0Us107svDMK8mC-R0H$)i_u@qmS9u)Bn0gR'
    '{}5!@+!h9lpe8WTl6~EJ$Xf%oMdy6O6owfriwaTjZt%cM{3gqn2|c@!*FfI=H0xnAP>;OtIvL)0KJDT++*1xMwU=VaOv@0W%K$V<'
    '2S9YSXQHvw+{Ix+d^38)$)LlS3qubI*1{(D#?K&3H#ls`Im!w~GKATU02YdfMybc`x({<`;}k-sA<Li4{?x{fK$EX;W;#xqcQue2'
    'L?cL1J7cpfYpNMcizz7-T&DJ*5A5c#FcOPnBJzsA(S_b3inVSsg$VgV@v>E*)sng8)ICadau(9U$<lnXxKP!b2|$^vGxq5Mn91I+'
    '?s!2+H#J_o9%aaT+Ipd+Vji(g|DgQCg?l6<5^RB7iirlQhQ&`$N;=yksYhqm1W}~DfTtLRQ1m22A_GJ3Ir$Lz!22@cfYH!zipR_z'
    '2ImkB<g)reW`h!rjPLd2@Ki&|@cFZ5-k9E{Q5T{qWM+y3F?F}*e1qyG+fVUIH1wnA)}JSEX0ib8y(cXtRIf1^XQ$7z7!Wh2acqLG'
    'z*>{FgCwSYV4utss<oXXxytKJ2oVS>pM5%a?*2d{IrnqC?lN*^ctnssqujHeu?>xtP|c~3lx~VKI9^T|%-hiRFQa`C7g*K(E}<(!'
    '`Q^g=)J;fqLCS$a!Nf1-TCF5voj+)+j!JTHY&0Ns6`@ujzwvGfi_c?(6pwtD>`sTl^U$<YjG3ZYhE*BLxH&2Sk16vNuEFSnqvBA%'
    'GD%o4e^CGO)_vC(-y}9{DIi5;B_D*+f2aG<ViAg@(EzYR#}`BgSn;k5*ld`*kS*e9Bpn-NFM&63xOaw-h1N9+G8(vAjJrbp!f9=a'
    '9q4_#kPIDXy-dbXhS>Abu~^}@(e8XKQq9xqin%}F$vdW~47itA4%KM^_1pjj0@i$T5t%G(IJNm>V<vF@CKaxt<(<4v6*~KrEE&p4'
    ')h9u$PYgQ<Uzz04jEl?^DI0`TRsg?<KZ5)g!ZDQ}-|~b_HYOLMgu;Pylb%MTHIrCMg(CJJjHmM+=aIWJ6XEKhY#gKnw3?Cel0J^P'
    'V7>)><$PTs1kuPNX8C(*jBBZbq*$@=GmUd2d$a;wEKQviH-me_H3Ab*WU{wJx_{MTk3f7UX-!+A`%rOMGoTiWd>jfJMk;E;bRpbF'
    '5P2?=rx1K`{6u?Ziv-k$k<pYmjUXXZoPeg*R|KbvI5SPBg{ChG(Buq-q<qCo61HE0!*BvGrqmU~m$EoA9l~`xm&O#(ooIMGzTBDN'
    ')Yn3g!$@UT#vA}2bE{BQsMJg}Gw*53z(?dwr28^Rio`zDx$dCuZyGsB^(Pn|EjXS>jK&I;SEHn(20$A%&9UM(HhwR;&npjrlTsoY'
    'MsoM~HdN$}^X%yq+Jy;l0q1IXpD!8wu0-44@VD2IGbHmPKn$C|OFqPQBTqVksd0E%wpqFGnbE(HFfns}TxpkVK>V%~i%wKrVWi7J'
    '2K(cFbf|VYy(k$6O-``aD*CpU&;`<mghKH;kb~7gaxDh~xkTr{7r;(6uJxOzJF`yHr8ILw9TsEQX@G1@Lg(3!BXOf<_kPK|77()='
    'fKmWMtLW(MQOFi^!>+yH5Gf*^pOUMRHOrdCfC+LO4NZWcNP2?}U1lZ`m=Fn&DInnQD^KTJO;By@@(wf2?Tf@j1r-b?P(?NnKnRFq'
    'JlD7l2?BIgy4{0mYUqlqklJ|MvFo_M^xrF!xKVLAgr;vP5(^nic=P-~2k%?eGY+K!n=T$9l<hN2LB#=&L5Fzd&8|QP`DTY2d_<Ey'
    'd>o-G84t;ohW3%>l>ai4ct33YkF!T4+Ne1;(PO01(W|vRo^0@8WXN*g-R)eAHltKBl53iB^caK#H|(iS<NGsH?M*UQ<9p<bpj7mq'
    'L7)Z15F15Wn`7uJ*^VIY&7lprnTNO>jzYpscQc)J;7(n7I4V(U`$cf{@U8zZhC(f>?zv8ARvW8Hgr!Bx3QPTM*yV&gghGaX|DfAf'
    'B{IBD*g7>Pz2@KDZ-*rTNF4_MoR6N%C9_|QFsx7f4a7*`h^|UAM=g07C<%BkxBOVZ0c+X{+7~PKx*q`VXDx}>xDV4wsq6u%0$Cn0'
    'p!EwT-aZ~P?I178L^^1y4E%M9nh<<w?YG&+4kW2O_%F_5o#sN!VpuYrUapjM^msIzWjRt{XR^|cT4eyfLfG&DY@QZ$hDLQerP;94'
    '$<e%z|MD9j*_f*J#m74_DHg=OE>Yw0an<QTkr|s1)|jUpr3hNkM_y#63)25i+&JNQ*`8RUD*GP%Q|QL{HqT=?wq6d}>ut6y(?x)z'
    '%%iQ`_A-Np&a&3m>K}<R>>zIvEYH|qWYG^Dx}xO#6MI)!gb6lYKSmo;bFput%;N{^n(K5^*JrHGB%3uM$ILQAij-Y-_qjSX`cdEn'
    'JSH{;dz;E2X(%M9R;0xIKqBaTS<%J@B=T+$wIPygKK_%?ja4$alnOekslvFjJucB0s+oR{$7ica$*Zs$w8ApGmO&7cM$a{?T6Rwc'
    'ov)0qrx@qk<BxA8<1C;|_q2WWX+44?p{vN@uwcepiNCuwDgn@$tI(e4a=<GaIW0fh(kvCvLv;MhLpxm-CS~SxY~c1z_db&!gBnQ@'
    '!pl7brg8+A=aak@VJfJEIx<mbCX2YXu(4!aHZT!o^E|amPTtg|9|G_Tb){E(H;&jB3&R^zB5)H6V+m(%=MUN`;C<$DMLU7<!JOJ;'
    'HXLlcXuQ}TjK@&LMuC}yDff_rPs;F@e;w|ndj22HTP4TIsJ@hkMNReJGyr(}y+lTqYpn`K)!l)kfthgK?ga%$m8x_k4Q%;IA3(-N'
    '&=;SpPyx<YbWY}X1h$c?FiC_?IFOVbeVo^s{m8Os@^W)#h*5x7Eh*;SkLWG-V?CE<Yv?T>7jsB~E6gDYtCiSEZK3x~y*=kI-H4CP'
    'G@FZRTU3N?j^MgOO@d%U;ZTz*?tvU(0?RS6rswK3NP3VaLRGPmHfvR9xnqaZy@@)xYrLjDlN7p!<vbs9QO8VJ@ZfZnV{uw;>W&#-'
    ')Xeu^$Af+ZhCuh^@Z-OQXFu;CWzb?dd0>F@e@P21&Ma$;oh63<8k(z79$xd1{<RW!gtdo5c1>G}VOM6RGCR>W6Y*1ha=XwBrKude'
    'h(4w+a}H2nJQBGOc25f}G-lr1RX`xsE#(Ph2pXzhezI6^L&LCZWd1x2Cj=6Q%Vy|$shz~6xV9%mc3Iv~aL&j7UtGZ%8&=Z^+BifQ'
    'wUqid(kd&gZ%IZ0(`XreOUYR5K2csfjUmBs8D&UvuP+B_=gl%+u?;(%+dgwO@h=2pc^rP0C$}s!qwRqy)8JPI**_+F;Vw1a6miyI'
    ';{qgN6h(~WlCT3uAwf~;rpXCBs9wfCz@EC}Unu(gy$Xx9I#Y_OIMA585;W9FY-F6A7rLo&1N9)1V~BFl1`N(%wXoQHEjKc0-d!1`'
    'iK7s4>3tU-<es6dL0>^FTJ(5Q3gydg{$!A-GjO|yE<Fy>|ND724ArJLvf<ZF%f^92i$=ki@u~@)yAw`m1-u(*1ddtnKfg1i2jo_q'
    'org^-S2Onqnv9x(1DZ1UhUkPiT7Le@pWg?Id7)<FBleTE$uZ0LJnaM({bup`^AG*`+kSo*!?E7<=qYA_h2FCBST0m(aescipMT`f'
    'U-t9ce694Iu<WoZCX+t&XZ(C9KR?^gKl10V`|I~5poa!RzM_7hzdp^M59Q~_`|BV2^EdwW8x;vP#?$m-G!brB0lQOX`=8-PujR~U'
    'M567;CLoO=OdSnQL=fMs$}v;p%c8>P*<`hd$cn#!#~5SicllQ-cL+A5pdC&zxqSRj%CkzH0N$bD1}AFhUa92|3Xa1@tOwf&=w;yY'
    'y?z4Cnkr4cVUji|+3xd28)ftf`z}4;Zpp?@AZ?>SJw8ix2CPR;=I{<4FE0OvumURJ8DAE|3u-mjx|Y`p!I&B}t6+3%)s*ZI#FSZN'
    'Oq-JE&zR6(v+1%j@yPiY?7vOvl9bS%;HshVof#V7W7L7Hi}+YGo_#X(97fA#yelJ5{wDrh?^DUWA5gVo+i!Ha=!Pn{e+`q6&cBSj'
    '@ikEu!;i|UP;@qCT;C9M?Z`}UOcNa&NtVS;gVkm*#9@%cA1oA-hl?5*p+y7Bf=7a9O3;VlyUocmk1NbJ77}!AHMw*%gSA*LFffGU'
    'U~a7~8$oMb@dLOJuNo})J(E%Gq#%=Wxsy`ca@@z3mg-=qkU!jBi1A`3QY=$*v&!V5VFUWl&1;BQdS+lv?@BH;753fq14s3Y_p0AS'
    'CgGti;IL(?Tg^pdRt)s$wt-^R`L}jz1b>|7s-PtfzY71rL>=<l`8AyO%U#vu|KY8$%U~G3v{&9gw^<Yuv)p`UBP?2bj(S5rr7b(R'
    'SJqWzafaXHF#d`NnxOL;JXZdT(JU}vV$1;yg45U}UfdF#OqRR6FZ^q2`x_$lO+42!+}Y$Vn))#0)-jjO5Ir_O1Tr~Z!BB>oPw;(>'
    '!~dmXB4gy`K%bE@GuB0+rF-fz6%DV`;b+uxK(drEZM*}*;wa$r17D^taqHL>^pR!}Mgrlutup!o1BUlkB^+ZlY+DKU<~grUU)W*r'
    '1^gZf5^Zd%t)Ueg&e8dg`&Uzyk<L5wC!*O<kM5SEnr?UeX7B$piJdJh5SX78hi$C2Fj6oo%Xs2?a>8Vi54X=T@rgHq1uWyNTkQ0F'
    '(}||T`b#)+{Y0!06Ldo`i6Rod%0!-}2~!XAhMg+Qh==llHU}cSV=NXpv}YOzyZ>2@Yf6XvoDVR*p-uX(LWFjhdxIG?+uS~~aO6M3'
    'riEx3Bwg&;{8^nH{xQXrVh_~oLHl=4-G}{%%b0;|M_hW2fkW0%3=SF0sYOK+FtrJws9;ws@|v_z!!(VN$PnJMMT0AIO!2&G#IO1='
    '!VSyhMI(!KrPetRt}z>!dnC=oY;neT_p)_$RiO=52W<{@uu^Z7NR(d-i3TozQ+@dWX{ndJJgr0+`jfI;g|7sB4nsG%d5C7ApMk?S'
    '<Xa@1+}{26DLp=5r*o)U6lSpr=L5p2A*j<{X`~sTs_b)_`{=3$inAjE8!#`ZErSIB^U17>mK!bDwGDw1V}!%NU(ysuG)xRqp6N_G'
    'LMsz0(L*=ISn7Ph_VTITDmNhg0_YK{Hl(xxAag$5M3B&6XN7i9nKcrPq9pk!<VAJA()pW?8B#NO<@iBND#d$F@zNw%s(ECuxAQrf'
    'Ja{rj>GYVm@1;40_iJT>S+V$RPdFaU|B$~~cjDVcH}*2!6fDs6K<;dpjanN?>JP?IB$jq1?XQFl$avyN*caVft|eLk?J)%yx(3SS'
    'vD7j%q{2*s_5fHG;0_mYL(&@{HOgA?1uEjp&TP)(zv6J{bSrWuDvVY-3$gzg^H2H6PEiLhhhB9fK=LmsBOLL~+jys^L!|p*6N{V}'
    '^rTrTnh!`oTUG_rSC*k`MyetA=($cwQrtdtYZ9jEyJ!Qpzuh4oM<=!G{dAjg?<8_Fc{jk9iQkcb>jRAwsiTXY<~|)GocK$dk1x^+'
    'o`d_-xbFAs|6Of9S4UwyqT9zzD)wBoJZ&Uy&%rmtEOW$FR~P0}k%7vbhy)$pww$g3*M0E2l~y9o;$wwuOT)ylk5p&nCiJ7lBp&l1'
    'QN6%Wg@cL!+SZ*fw}w&O{!bZ;)TxmFafRCKIWeazRCP*p?glp&$IRzphVEK&s5mA^exJ!E1%L1PP@If6f7gL*$P8mrKhbSBq*YZV'
    '%d!*Qn3KPYfFov<Mc};?Y$mXXA2&sS<63?)Y@(~1b1DkFSf|IIqtPr<)@}Zqk;}BB6H%_qh$EKeyvRqk!wl^WAmfzU@SF{AGSuEw'
    'LOnfV7$avB&SlVd14U6Bf_g7)Nc+ps6s`-41Ay|}NV>)3kgWMYO6om)dyY!B2M1GOe$`UBgmQ)l#Pseiu@d!5OYc<)TTc^Q)Nb--'
    'OoMm}<e11Uai{4f-*Pz`W=oFZOplpVZrZI(zo9c>*gn$&{sH4PcaWnVZ^wJwVP#o-@eJcr&Q~R`(Q5+?lHyk)lkb-N28@JB#$EoH'
    'hx?n6*kZ861*I!;xXgnMeQ!q<Br6qSJ#D4<6R1xnYe%lhZdh8TgNOT{NnjrRgDBE}{PUXB8LYBgp)O@yD0BT+G2u}sC#GKs1J~tt'
    'I$@$8TlC?0(?##FnBm|yXWlIlrI0}|y;;+H#k0z+fdBDXZ^h+Ir5I}PaaV6JBO)#HnX5nqHd}8zhS*QAw1blPOh1ZLhGg(-!vul5'
    '6RDV{ilG%>l30%?X@arH{eOYVO!c1o$d=OKr<8#j*;O<sOf$AEOGnTE_pR*@)IN(O8Qf@5O<f&JjK*g7d4v=-rzGVG*ZSxCM64&2'
    'F3=gf+<*GKRg-Q!;Yz`Y#%V@yP@^wOE&|bsC}k#dlgTN>#0ajaH9)e*PvS!kvLXaZ2B;z*Klqr~SJ$e=tfe)VRKms#kJ`ENNYlT}'
    'p`DUKj6E%N5-;^XkLBJE@IR2ePBO=U4Gca3_B9<BFel99^*YGUcB;B?J_3(nG9j5BR=uU7?uGk)vxT_--_$qR@KRBwA0&8VJ~VB*'
    'WgkRVvxt7cL>`uj#LHS2Qh*1Py{4dno<kw_2hy1`CxKD<R<cd1EXE>wTEwnn7=5(p8%Q2el+P?eh2LZblaX|M?=1_(!h*FPwgg7M'
    '!ZF&}zL@niln#&cn6vp@JK`JKA!<kF#q5Ql&HW2`cW1IquRW2488a|G62)s%+n?BCKtqPXb8PX$Eu@1}p&qXtWmm6}5jZGL%<D5T'
    'Xc-wun&Hg+H#)kl<=oNU$?Z9R3<8|V#$jl`ZTTK{&O;ipf$oIXb+eCb;s<L6u`eJxuA<l(_+P3u{Z4N`3A{}>DiP7-*YN6O#3O!j'
    'qsd=Z-{13j;#^~1ue=Ms2>pH_o(Cgt&f=7i08Hf66g&b@?0EaR2o1*VIX@e=AcWiJ<}dOk)MCh6$%vtALO2Q2>R7a}y9ec9g|oX`'
    'wt?Ii1Z@Q-kT#6mnU2+Pr~&<(0ahT;j}<*B|8_yd$81I?b_>j?^u%kJW}&R|_cVcs+mr&MZ^bl}6;~zrj|UN+ntoPhT{x~ht84?N'
    'k7iBDD08*QvFhR`><$U45sn<EsFUU1Ao*6o)V>Vuxy=fJ`K~d9P#+|@NG$cM505K9iQxgJ#Dur(wB(SS4Eh|I@5XJ1Vvv?9m1#{&'
    '(jGoK9Vtk6eiGlctv_;_(V7x9h#smlNn*yC)eR<bz$px*mMNpKXhcQeXenL;O^ISf;?sq^pkfRZo=oMK0D)sd1-?s9!HjxFP!acc'
    '=fxPtZZ=e%ju&)4O{b=97^bR5Kku5#%u6?yiR3e}-n`L9+hN<v9D8@W2$q}xG8E<xM*RP~T%LzQoG60jS&bMIv8l}*YktlJC5uRd'
    'DrdC0dKBw~O$-zE>dY9cwjwb$yr5P14L%q$?}vwzKq<~&9#C_gYUWT)hcvLJj6|i>;;?pR%v=o5KnT32^oVE7yK_2ycoI@hF{;Xq'
    '+tQI4hSrtIEqw~3F0}p=^)=x=1&J8@lq_jzcmMz!SBky4nP!o5RgozS5cMtm$6S1zcPL;or7JUd3u4x0NQjGxk!Jp(s$x9h&x2In'
    '0c6qGr=Dpt4DAu-SgRT?%|T%%nsi&z-TD04jc7^;dA|GK{1$co)kD2PcI`5;xe@l$*OGZ<tE!Tf4q5nsm7pUM{tHs|oq!ee0Nq_A'
    'rw)GM-$iXieJ)R;JKNwsV+u|rGzN><M5>Zf&J{7722Je&NGgWAb;urDI!?<;;_$b&cmmFcUT2G<mcRFa1moLS(4DaJPg?WFG?_T8'
    '$|wp^1J~5R^U0gOZy%ChCS~$Jp2lnbgtt2866y(t6|QUyqUL4u$l^;Rk$RO^>Iemzxjs7+sVCV+I!v2BRh(-~%Eje)c0@9rx?Y9;'
    'Wof)X2k8rGt1+#pcNv6OqddY+>$G^WiZfx@9tK5i8Gp{Slp#wcOr%N^F@2UT(neq${G|A}54~RK-u9#o>vmoxoKYw?)ErMtBEviO'
    '-Vimhi|v&h$fJ(sze_P-hqd;1ediVD*zH4SI+vRH5qNP=QE`h)AXFh~{loikOnzHWdhftoTXk+zY{h0^V&Wy{V)op4OE#wp^)_rB'
    'wEdXUb!(H<*f}_jB%*U#35f}jIEMzFK)FKJs=i5Qg=ltBR|0+6z5VlRhl-@EFs?**g7*nuA-nXE_Q*l?14;_@C$T4tg#im`^Sf6N'
    'TB?@WNXA&0f=40G4NVC^D}#UgNGU{~rU)(AoGzeS7z|rxDYXZ=*oBBw>-F?8f~GP#)p2EishdABqdBD9Ypn&hWwrtlan0ZWmx+6%'
    'qb0QIcpxi6oWr6?vtG$0LG;@=W8Fy@UcjZyU?USz@l@eHR+zXMs`F<aH(IMbpIBu1<X=p6tOCpiGo+U=s6h1Ze~Aw$AoyqcA~Kd~'
    'Z0L1V^8x5msPg97MVi%6)62uJM-o_53b@lUw@o2N*pI@LyGuS253XrSW>D#?jP$N9u=E%t%vR%-OL`dkNE%AXvTdhk-{eBp^%S`x'
    'Kt}xl2o+s`R2CepX{#lkr8iyM44@flWjM78oRXeH1RUs7@hZD%fSc5DCge<lP9CqfTC$AInY!(c6jM?Rhc<FIufq^W!(o76iyLTW'
    'nISAd*V`>Td-B4kW9$!LiNPeZFVvF3P!9wgD{kHu9EmzxfD5aU0N$Aeut#}O#~Y%jCppfMxw^VD*a_2CFLdD`XxlJBZl&w$*fEZD'
    'W>rR2M*gv>qvGEip}7Jc28NN=aBXSoDn7AsGzsP2?cpiX^zS3Ma=z5ev}3ulv66uZnTXMjqE$XbbBq}kvVBm5B+rB`8H{dUOr}SO'
    'Ev8e?-7`THzJUFOW)bdJx@Xq)4Uu)b-(0q(!+{gZF5;bV<l~R~#OeZc!E_c8otYg2IoyUM(HV%yLh*j<`EWfTH?oEqIq3;klx1pS'
    '3}2$sTqyS3bOR!>;4-SkKsm~8GFQZ27Kzsd`|vY*;LJ>k!%3waQ$;3^w@pdbu_E#hJXI|L=3wFPJ2ZT4^cr!sbzSTD*q6^09SPMH'
    'l>@UJV*eU=3~A?fbtrX$iIpe+PX6*hPYeURqtFI}x`V#eJjFKbWM`|SVzU+JHH5jx@22^I^}$YPGe5&NSjq=#e&PW77LCL4aT1xV'
    '6c4T|3p;Up{9k}14pIP?ARc(A45t|JGH6-+AZ)_=cJw_H<kVdS?qn8(sZYdw{jOp7e;!PX<{pJ7>K#<m=)Cy(yQ?jJ04C5FN*G3M'
    '^u-#cks0!;be&A&5%)3rH|N}x0!7H6CEb=1oQ0EzvHfaa#?Ht5p|&nc_<iT-@wVp2vgE+%q>9?GifHPx!=b}H-ux(Rp$KQzGr&gF'
    '3yvh1d`dKDasotp%SfmRcQNs4tNPMwUGwFz@&<P9#wLW5JWDY?_ubqQ>HkjqeBp$frs`DpGy@O}k_WuNof$-B`-_lIH3^=}-?e1*'
    'II19ZxRgrdi-3^O^qWl<utx?Z5>7}g-bl%Y=#XMLpG4-a9G`sxjR|^f3AxZz_(`E#sqG}ew6oAU%EXrdtYuey|E8ijRhce(c{YxQ'
    'AB*Z5GEong+}2X>TTTPNGkkY29z?@f#qKl_UZFsx$?e3fxwvRSqJ&(!lVWQ22VJQt7N1S4fN6vVVF5X>8C>L8=s$jqo3aICbZ?`P'
    ';kz60GOKxc%1Dzuv=+Eb@Sz*0ktUjW!qllcV>_zF=Reax*SZ*Hi2eLqnfddNW}{Y)Cf=Hy0?JsMcQOu^nKNYebS1P4Atgr!-#%2H'
    'sXi&h*9)OV>oQzk%DQK2h}ztZSrg;d{J_n6TA-fdH$@p?z)x1Z<DBm8aQ~f4B4hJi1l-MR4-`p*JAqK5@gssP2s4pst8H{N9frT#'
    '50U$G&!;x7EEEBl9iWJbSfL3z5>4{sIP6vVI8wCK*a>&lE(oz#dz94wxZ`AaNoJwhSRkfWm8CqTR+V_5F~}!KsUKa6q1a1PP2*j5'
    'lSA>@;E+srB4a-2-9Z07bm(v&BRm>k#)O{YdN~@<{Gy8`nLa;}E{>#Us`?oT$6Yi_!vRjLKW66HakW+Y(&j<zK#5`6G7f|VR^w=|'
    'OnxP!G@!PnX=}$d(W56p1u`ELk!{E~Z?m?XFy}PUG6RgtZW4|pUe6fchVTSH0L7!?jF~c==*JdS!hH^ijU5k{WHcjZHQH66pIlGW'
    '4i!LFFl>NU%|u}&J!S}B0!T&C5E9Xja7EL32>imN&={gB)HS+?oQT6Z2G>Ivg3xE8quB!%T+~}DfzDctq8L^_5Tg0yyHqtfa?Pq~'
    '3cxY%6~%TAsWk&#Kb<0n8#Zju=3-=$WM^m;COa^yg4s6sWm`eNAC9Jl|3%%oCC7~<TW~8{L0d+?bN?F~xCcl=W@9~Lt){B<b;oXt'
    '5(@<&<8h83vjtIOELmr^;a*`b>;!10{w;hkM{)hkju3zgg+1PK3bm~#m_G9bfmkNXHqeB}VN=$p6`Sg?3?Q0`%^dUX%`D}xk7rp}'
    '{-E^=;2|73-s6>Xk7<!fCH;mgH66jVXEtW#j_Lj{sdu!(VGb?EKM?2oScZ7BnxV6T@Yu*tY6=*1R~tOcim7krKmg#^h>JFImYXjb'
    'SzBf!p5|&q08hc7wKa2dLtm;$98!%Ai>*VSgG_54Ed;UAWsM6`o_Ev@`W~P~P@)(Ny_?y!5+ZU`UJ>v%b(MsRR*BDqwinq%c*BvB'
    'kle;`V`?weY}AA@%1Yw;qI`7^yEv|KQB&Kq1~71s5T}X|VzVKLck`$S=b{O(U>FW<ll<q>q&>pfox*!m<tHtRn*iEi<rAqvFZ4FZ'
    'D_N-&wnGjlf#l#Z*f`~>c{imlE1;{ci_COJ0rqA)NnRY%Xz7?s>cLNLVSN@(ldBkpPhfAlDwy6`*|V~ig-k9)7!SnxLQMnHWkIx4'
    '?4nGc&L<o>vI*uEVV&LHES5NqB;mL@pT%Vkkhzl$@Q_Oh>~e{M>q%w2`F6(#s~8pN{Q<`X0+;YLBr)gKM@w^Lgt!P8alr_d_RVAp'
    'tnIE_T_!q?QA(zRS{bGzc1~n~K{C-~QY>QoD)mm%koOjf2GE1ttpa0CtqS-(6iky*Lrit^&34BkcMlQLji=_lvNS<1RdL=-Y+*ti'
    '<4LN>x3<{4BegcG5Ml@ShFO{|BdGP3ALQ_&_7FrMm@O{9fJb1!C;p@LoHdM)u*=aE@0Kh|6VKc7xP&5agwv3s;us*u`Wa<!^VCg)'
    '{K^R9J#I|iRvgM;aG%WGJ&eS<3o&Lg;CCiR?rAwAUl;=By9E;0qaO%(3k@E0R(xm7Mxl!qF)q3hWHv%hI~1LipimxR!IsQ=VNj-='
    'g&9%CR%;ZXbtaS(`6hDE;;3qIIFEO`8p3A{;N=|TSEs#e&B1WH`8!w{s*G(>$_)n+Y+_A>DAtF`%fN==XrcovE~EhRhU!>>i_1l7'
    '$J4!Fs=gvu@9!^%fHG7A0&dY<9;b%NBq%U|5J82(N&JE8l>AQ>Wx0LQ0*F_#LuKrmPFpH=dX9gX2|_$(e2M>`6&5}8n`Q8-ZvPmD'
    'gaN7~7EL3;<WY*==C}w`&B5jP=Mn3gFfhblhNOhheOC>Zm}#&F<z=TwUP><4X9jvUq%iMAY~dF^oEJ<*HtVmYGJ;_Qv(F6is*(h7'
    '^vQLhIzzP8P4l4X8xj`5p-wX~ke8qcMU}L1&}RMx+#;-#Q%Jwr6;>*gEG_ph(py<Yo+#-dDrhy`Xqv*)%?dkde48&Q^3h%s6v1ON'
    'Mxvkj&G}te`h)(=cx(R*Bf0L7sTIIu_CWx)l}GS?aqFWo5=k!m$bHY3yHIZi@To>*WE%;Pbx(zOzhHCE&2P`Gpum%PvKX3brp3qr'
    'y{#`8h)4^TbXU4D6=4qJV4^JgTkHsa4(^L%^#zQ8aZ+_~N&Qz?iB$I5)GYvm5Uh4(QU-Sovb45d=EgplF;m6E%szA=Au}!g=qb46'
    'Fp_k_9U!_ZH2G|kI6VkNY%f@YCN^0@bpg>s&^4xU*E4M7J<*V=$(A&<q1hQiNGx7B(TlAro|^{dLc3802Oz)byfQi^wR2orro>E+'
    '*%E~tvVx$sA2r~MfkT6d92mJ6{^j22q{@E6ojb44RRnQ>j8TpXSM&>=j?mDF>%~I45XJEI+FtH{8mE&H9UOhQ9cPoqDnNW^!2@YV'
    'v?t}V773P*ECP)NU8erAZi6dKB|T8x5jx>$T~H178DxbFL(~6)yI$!bdz9K=VR8hJry{e%w`O&FPG&B4i%0ztlX3TUeJwSf6)B>F'
    'T^!sU;k%kqIx7WDo@{JqPXczqVS*3MXPm|1g1C$`+n`-})FeKr(PpgEX!iH*RUZC5Y?}ep!33-O1zD?PE=+)$?o?q#alT(gP}L#Q'
    'S<)I^3F47${Xs{e*9spkvxdd=&qcmA8Y%QBb~qmqZ*a{eFmPf{V{}19rU$<BMmMZ^mvkD=I&)8b*lzRgYt@(iotbRhFaP%MN84R3'
    'Y45oJ-b;E9xo0~=yBi2pdQHQXw|k~#AK{vwSB@ZWUO9TDc+;pBgK(gYsvAV4%3|>}^}J1EyjNpN3R`!?!tw~X%GrsEFvT3C5Jr?S'
    'QVLSxBuwKEIG>o}bQCSrpQSWYysKY2F~v;#w9F~KEYzF{6j|>UH7G9KAQoe;x_TYcoa%kM647O<hMF7&8{+Rc*EoFw!>;~k-&ro_'
    'a2yq~MTdDWbf7}H&6|WG?!<WZj3tB%7o=uTTC{aCd-q5u8ETwUbzvM35oBg_mXq-~*ebhG^~^ui5Yo_~A%R~ufQk<3xjv-EV-E!h'
    '1s%*cAqABa2^-m<L6ksM%9mjjLgtJ$BJ+$L?d8sKEnDU+!!H(;2FeaGYk2+7@}rb}r`H-JNB`DPGjSacdpib6gBzCrG|B3t3{#B8'
    'gx|>`Wv_X#jO&fJA3AP8!H!fQE@zd9HI{yg<}5D78ioU*en(cV?*kNabzn<5hRCbl%UH)WG`7O=QP1;g=p?wHqx5t~KX~F79*#@^'
    'p|IKNrCHn7NHfsTUk#jaGKWQ_M_G#y8Ufmgy!u2(GE-p5yGzbvOO(VaOpZF5#3neO^wgb5`+-0H$P!2YM4|{+9PVpZwy`q!m&973'
    '+-ndU-R=-w3L?r!2+j*Rx5XdG>9z$MxRvIGJFygS!HaWdM}yyVgfQP8U|(3}mE}y@*BSOjMl{GzDog{qLHbG@*_$jUNsk<o?0&Wi'
    'RY(0eU3s*5xtMH@o0F&uN;H#a35^@rV(Fjo-|}d$Oz5apRuSsZOl>)U062W@F%g^f_5yGRPw%1_Wgcge6u_G-PYtxvMr{68)$ucg'
    '9drNSTHJ<+ktBogbwONUR6IsbIjdDNFr)Sn-iz0hZ9u%4Z%s^RNj#;<!LDwvP1&9!Lzq><=3DqAEBPoe{9~+&-0D|tjhWa?FP6tV'
    ';JX-Egx6(&)Ccjt2py~}D-Lqqt9PR(#(7o^la-Bwj)VX`{C5lLE)~U9s~*#L)oC3;BYWhDSzo~~{CoRmk%x&39a+D@JpxN;sX{iZ'
    'S%a(u9okl`5t{;|nkY(z0r?%;H2)QK9-L|OC8yN}`1>P;T^5=D8qTxLYqV0i9z_gLGEU|q5g(NiPvTi>>c;gm@|N|rTa%zEk#E0Z'
    'k#t=EyfYi6S1~b>2o*{jLG`fRkM=S5jNSa*M(wG70T<koGAKuPa8YPJOy463*^mJOdy6YuJZQ^Q4BG30Z0YJS$>Ic{sesAy!xYc0'
    '(deYu15+q|PosQ}h(4*VGWu44jd9F|6E$}{i1kqch>I~#hLe&B+;GD)^`kCkM7av3^9U2h8NqdX$scGFf;9j`#47HdG{eTu&~eo?'
    'TG2Qn4#gTKJm8-~K-`JSI?bxRYoQCnVF{_h{_d=JKz<79$wd!cK@}YIEH&BxeCp!PKT@Y&U<MgKth*|<MBb+B_WIp$=T=M`v*^yP'
    'sI!)Pa$0(eXCOk=Og^~jG}tJK(bKs4+O(AC2WUt_;|LOFKtyK2u{36vShcV2^r|ic?ikJatZh(PJ%4cCd5o+8$O|^`wdxAorZ*a`'
    'iacvO`{auA6;L(xrq}TpU42sBUzS5R4tDmCsFx}3;5nvN2wl~2(nWe~paL-6><p3m$om{9;(DbzqsuUgosSLI$_eAnFe0<I|M_N`'
    '`K>x9B8z)QF`zzilwO>m)!WV4!Bb>FE?OVbW(^NZ6^M5ncpvZdLlxt<&;^U|qT?w*8y^ITjnYgUH!zu<H%HzL8)T=i>9MEdijjLW'
    '5*o7A#yU%=95KZF`Rm&)82kgnVqKw1X$#tAx-$xsAohb<LlII~KjJ6r$h6zW6~F#h%XZ{su6M@mF64h&F$;UeRE8}rx@Q}u2$C#g'
    'brh~}CP^7(e?pVrgq=(thaBO>G6|C%i#p1InvU0*!m{m~cPqdqhKz}eQDuy$I71}0Y?)rpORsli;N+;e<jLd|+@R-9vEB?<D8ZY7'
    'k91uIdO@V&-;1N=$VZCVJgF72gnU=CYr}oCDZRSC?cZdF^1kCDuuNL~Uf2D@FG0t|ih8Amjv#03DN`~HZytz0CR6pCcnasR%Q)mA'
    'Y-3E_BSh+Phk+%!80Znvy>Ej&eh^$g4f$q<YNjd{k0Th7S6OwK%IbSK4ll!7VdvULULn#f6|h$~))eMxZ^v7Gy|xo?y^H=ebW6KA'
    'lMWWk2b}c(a0tZFG=l5}4NvHe0}-SWpR5^Pw;brbXeH<4Gv5u9V8MN0=>S^Ce`VCE94gpp^S(hF;Yx<6_CkRf_)?eo<ydi4P)n(k'
    '0D8T#Ee`NCl3VI<vsV*pKXPdWESRX-4Cx`*l8D>EZWt)Qaev%CTx!>Qph7R6%TTCnP$rTKZby{1KX`Ws3nm;l)%RT5M4aq1NoE>0'
    '%@wu#+sB5rGZoA=Rnsi3|7|*(ZEfcwhKtEQp){2>NM<zoAV$I;;TvLv!OTs9c(dibUM?ZQt+L?c&KqW!@cwg3O7MgB%ZjzP7Xw<)'
    '&2_rYUlcf&tA;&z`a@PSCv$z|JygS_OUu)za;XC?q>+&n21sm|5>&@cTk|4P7zzU3$5;g`d)g2Y)_rG~7<=6)E-rxnOkH!F6HP60'
    'OYz~L@HRhr!COA~z%Y%WC$Ar?l^I$gRx-n3K#3B=ez9aOLm8|0ET%E(z`~NCdnN@Y31E6Q>9Fs_ZgFNf^5l)sZq<TN_L=FS4FnSX'
    '^4!i7NtPl@8krB%DCE*L>nq$?>b6~U5i;dnVTndY;v|S}^4ivMR25F#VavmRm>9id2?7X4VUcmooasd@V&qz-F)cBrTuq8;)XY1>'
    'AYLR_EX{dKkv)mW1o4G8;AnZ#s<&-%)LA2wPF;cK);=!)-X$)#uh1}u^-lSwut4lMq3>wFTd5Wz@>q!==WFNPPvGgHeq@*~U1I9P'
    '()=HMK|JZkbL|AAWm<DLMi@!+aR7*)%R~>Irq}1*hEaI>LO$tUarWqxIU(nZQz8gO3!gx4yU+{nCIO!uX&4Aoxcy3ozY`wwzo6x~'
    'qER^*%ycrD6Kx9;W8MlPt#~~nmmMFiCTCYOIzlCp^%u-+%FFO*;hoMA9zL!;aSAAqgG9GwP~%kCTdYCxi)B2k%fafceH|n%9YyWl'
    '{a1KhR_lsgE%!UJBJbb8g9ebR>?B$Ettjp6V1HqqF>q|=X!%>#qo##3$WSQhHaNccQWZzCfV%Fw>7Bvzc-W*Le-9r5jy2%GqnZtP'
    'y}e0cPsv9>I>3^EG3;>i8c!7&(QPzzC0>Ct8M3trlbyuB=8O>6JGv8Gk4EhBhTE1FGQ`9!3@<W-MqZ-9D?ZGIuZ#t;t6;AO8`F<G'
    'BW%FM<^w&8g%PivVP>{Pp-JN|R&M&vo+(-i#*)_&#ZyBa$URwyDQ<OSw-dO{%P_(V%$&4`>TscYnkanY5eXZMLf=E-XQp}QO?$+S'
    'Caq;uZJ(?a)nb|=DE5bo%L@su_;shv1nr^wtUS2T>xIJ6CmD^`6GOJVr^W}(d=8036iUCqz^;^*ZCTc#7sdHQi4-2rbZ?e{0_I8k'
    'i{YQ*J7sV^F}VZ}_D;QSqnu8nx8yMX(}6hbnR&fexNQ|3HdB!RLXTmqdHt;;1z5>Mh5x^D7znkQ<w~LFTE#yrnPMNM-<}3L4UmqG'
    '2~J9(<Bi%IhQhM!pq-vNRVN2FN8f1auI8o-W(q@0)69sM&8d<pyQx+Vt~@PP99=)$y+*<{TDoaYPZ08S{X+c%M8#;rY~h;WioBG_'
    'a@hZ4y;rT8HFF_66Gp+bjD;zce)DBUrxh~`hml+F;5(?`UxQPxngGzG-tij(#bm>?3{&<C;!sVGOH18Ruz>K9yAIuSsL&NFFQ6$N'
    '2JNuLrbuM~SNUL5Ms1`)x3V<ZGt!pFWgIf*xKRc|l`49)n)9;J?wO;-+rtoQ$60P{kb)q`M0o&H(O|71>F_?p_AxJ=kOL!cLqD2X'
    'eVndiXD~K#u54zz&QH`Oh(_5Mb=Pa>X2v-uIkvu}ChstUlE52Z$1qsAsEk@>^TkR?;55q{nk~Qj$TOU_n8F6UMbLBhO&`Rzib|Bg'
    '=nRz%R2$(;lt3LL<B?t!wCi6#877-h7Sj9k?tuX7vwBR_^mqt4e*Dq}VkxhikALy==ioGNJWH5-w{8oz39y$9dq~+Q-SQo5?eF>d'
    'uYLX$)?m8}y1I(Z0~8ap>Sa4CVCUn*eg2N0|JLVEVTUj5Ah!UgbNe4gj*dRB$>(SL{3SpBxyR3$Bv|hDV9k_Ce_YGYYx4Q&zJAfK'
    'fAQ-lspBThP$oXGzm(v2LXtPbQGp}|Pi4l&*G|*b+4T(34Dy**vOd}3l1;M={Cm$?zWgx(Ebs!z8p^txbleGb^@foBDSL=7Toln|'
    'V*4uXf~e5LfSmh)7Ojq4r!nV!xR2rZm1Oq@h)~A5MZ%_LK1cIBqwPJvPgR^ua%HM8=Y%L!i;)}6Tq#+n=K;lb2@G`FbxOewNrMIc'
    'eJNe3YB%PSg#ebfvr$JSAq6*Njx}4Ow1{$MIjMWu%?;lKurn@Yz)RBmfirOeChxS!{Qo+lTgL2l;p#cWCTj*u4&6M1GLEKr1k_e5'
    '1vU`0WL;`tOU=C&<8HJ<&6e}4F8gS%5EoTrd5xntuqNkBL`V-d!e6xlvn{MKJ9EXbn%xClNf`y><X!%S<u~8`V!3AcwM|f$D{V?}'
    'GkRHE-TE@jqpj|FnY-pdkP~*{*Z^^9gj?aUA0LT)ier(<G*`Lt^$n~{lAW+Z0&p)C($`Fl;TIZi3Vxc_Kz5}>nSGdrES2Hadp||b'
    '2{Y#Uj;0<0>lpPg<A3`CzLWSDH2U-^JEMW8T757d)gJ2xmjKB$oBT`3UK1E&Ch*97TOTAoTsJd?XL&zvJwn3vGBlZ{ni*cnViL`k'
    'Q}S?1^Y~}1n04Xt2qTxj%6u4uF8?Bfdu0HBFaD4*+|2y$P{w<8aokKBpJ&Uk<_Wd*OOeI6vDUnpeElqy=mMOTyJ<n>x}VY)@Q29F'
    'S>T~GVHgQ|KR2K2BuxKh#}s(JCSCNpPcRAKZ)ADI4Mgd}o25Op1R(m5+0iH<p-JvHelg%Nl;O;*-|}c0cK``QHc{T-&=~)+dga8r'
    'zV|<`&~V3loXvOecq|I&1z969A`KVEynLcPlqtloYOGg6#r1$$uhoxy&vwYugsxZSdsbi9M*>W8wdZuz+a4U}C5<wIPP3EIE)0g@'
    'Ul^&)#_Xi}-;5pD|7?K~LNVNNWmnpRRRv|FDqhBr&(O7M_9U8sNuxL)UgEE;vh69l>m_Q(c{1MBw=?Q{scuPodJy_h*=Yw^j##TT'
    'OJfPb1o*mtG900fhtC^aU^oJtLT+j@0YSl%Wds*olW?@htut`$%$pAd@cn9F8TnTQyDceh^vmD-3P1v-MaA%vE+unnhmor6uJEKT'
    '|Lfy=6{quu%_M>OeSwmIm}VrXcI5!%^M@EkuW`qtna%`CMNb0*g_=<U<LM?6giC<NsP}QvLLqL~2(jvIgjmn+PXz!o!7g%%*dq3h'
    'P6TitX83K*poL!!0HDUk*1<%Ar8a?cn+!F@L`WUlxkWIaKf^whvW<>tLiDdfa>)u{SYt#2X}`}G4P&tZVbI_he~2kFI~oHyP%bp5'
    'mOotz9%+Q`1x+upx4hGdUB;X#qLNFo&-z}*QkRg|GeVa<VUTXrm{ndjejxhW{=gVRBR5Hjfi9FoQezFtWN<#HN6hysk4lOWS7Vlq'
    '1;k_sL?9b{pC8=c;Cd;P{=#MvFOT`nRoEcAgyWJ=WCb#dYIsD6M}6(Lgw*4xVF;sVs}Te2z_CDB>hpSfOR|qSVKt+JhtZ!?Z(i7)'
    '&TX)4WU$PiR&i9$P~i@hv~YWh6k9uTKG+lK%~-hAM;zp4x;&4Q(Zv9+30_A%?7cqH0oKQxfm0tHGb7#NE(i@kG^vyY{rPoZuqfpU'
    'aPN+1R|U~4?*FnBkb+4dD;AM@uXaf;(H_V@2EvCdjLD-c#hUr?+&K_O?x(_}0;hdl@yY2%az*eR4u@Na{CnSSO(Js&aHyU~LB&M0'
    '6alRqd9vSeHsK7ZEF({jdFOg02_Pw-2L?P-6jC}D0g^oh&DV7}>U_p4*%C=;lap^yA<_eE>!#VVWj8&L-51l6ucP19E4C7Lv7N32'
    'x@~t8PYuA(cx15f<A-!Ck0g$CB$!Qolfh2}kD_o#SJcLS;UIL^+(ZVsf#UnXku^>~R>w6hPQ+q8Z{=lTf^ZqCO?_-w3Sh4)O~$eV'
    '(EG{U0jYRZm6;b%d657)DLk_n2MzS<%1>Sx!yS^E3_QMl7Q%LU5$aUbunuhHr^cJZpU#^7`$yzr4I>vb_hPsJd>o=Qc6;BOld+9k'
    ';toR6jFHa55T)uX?+y}jb(o<gW=Wcv!%C&JATbQz6T#v9n#3p(!P_V#0<RfU=7r9|JZA<`u2c5ohm130*Vxn)kq-(F2+6e2Jys`V'
    'cbORdx=cdKc~*CRX&{+_q~Hr3ZWh_=tetT>pVcy6G`p_TaDin88L*PfghMrpd~B4*Xoif!%7uayry`ko`SSKKN(I0kR;gEZwD-7d'
    'OFL}!ZYwi!Z)4p8KzyAC&tAQ!7z~5`#BolC+0i@Lqrd^|q>{`>=VgX$>-VEVtRe+;09>dJjMr_ien3RK%F%i!k(&<7VTd@PI+-xo'
    '4sO&p*LYD}2;f;_>sk99-^s>l>ItYxB$$s6%svGdjLh_wVP&YYjL0}_6B%f<c(nZrCT#H-7Gf}^x~(iW>1z&2h{U(qrG34G&@RyJ'
    '!kq7BgzB$SYy;Gk)sA+&@69|#9^PaPjAH04k3-881H(d$pNz*9BFg}CKf(@XoMh)3_(NB|ktnpoq4lPlRg_MKry#5=7*v}Hg=BKG'
    '77>U!GST<zIY2$JGNG>{$>PYk9v%C9FoVbfX6*O;$cIO?i}jp{Pr5Yx!5E5>iahao%i6zeg1R1#;oTA+!BZ_1#~30Hu~E}3jR&NQ'
    '+o++292!hGN!iXXUK87idZGnPnEt%D^%g}-_QjZiPN`79HN){r7I3wG@A0#dF$x?3a6?Q5??WDn?!gF3XWFXuN2!U<XeaXj)C^C)'
    '4w~+rqErr~PuY(r$S4S2z!Fzb&hQHD{T>ok>>bo7mBgseD~noaTic=Eq3q+IEF;5OmPXaCWb`pC!lgcL>KsW)m8Y91Q@k)Fsw1w9'
    '`7h*$5;iyJ;|YJYh%f{OHMYe!_Z))7XU-ng%0X}hp{N;>$?^$t0OOvG+K=~ec~)XhC=opw0s$R_$bfF{#r5V4foep`J{ReTuUG_g'
    'o)u1nX3_`cFMqb~Oo7p!@vT9jJRj891bE5r;tnr$WMox8J%ex$8DY;K+7i(ahLSSN4+XOl{RG8)3LGBK(nnh%=#^!<K$dV~)EG(^'
    'c3Myec6(u0$YAn%0H(i5BX`e?W+*rn^8gea>iG%o?CXcvQ`MV@<`C1cr)f0o(ds&_>E2Y?69k)hz(-;bwWu)^%jk{kn{wmY|F{WQ'
    '!vj`i-7l4o!Xcr{GTt;Of+`RRSv!wZu_N|wD;!1`#_6<7E~zTzkEDzMLh+UZFph=8x}!!n&oCz$0|&ivf{m{=7}SjND6bRdzY$Nf'
    'kXz`q4Ms586Gqua4XpL3J;4d5Yv{^7UQcfU2YS!r=CF(4M4$L#UN;v+Dv4GSQ#gPQWcSp{cq~HZEUL^d=W0GigJ5jTIF4M0MH7>%'
    'S8;1o5eZ0=%46)0{;)9QLHv9D8{xh|33~=KW39$NQ(yRI2z(;5YnMSAOKhz)xuOSoQ(mkz<CLES6U@JeZju{eDl2wrwBOu)gVK|Y'
    '_h^rAd}sRT_KHj^8iue;hK4*+)SDU25igvTVsfr&pzcg;mL6`|(V(i2ZW#H5<4GH1ue{Sg+VCR_D`#Vhr>>QiPP+QY4T1J?kZB+^'
    'j7)4jh#qQE|A0%07TfiOV=(jv2#!<MEO!|H)v-)xtBDj4pL^|(76p$WnaF}-Tubv@ZFaeg(XOL5ZUE2YU?F<QjU%JuN?r)PP&dlF'
    'y0FLyL@^<+o0J;*t9yr`hHlkJNbpEW_>oBce1#zA>O@o>t<7a51{35X7+M&kBJBHm)|T{t@=r*CU~u02r)l6ZS+mKZAU*AcOf2sT'
    'M>}};D)D0U1o>3smBP6vp=LbldYMz~5hN_LbM1QSCnYw5JS6?>{2xdO%I1MEkF?)uc0H4#=9G2j%7-z*>@SRDihvf2E9*>bS}mK;'
    '>kS)<4R*%NX-(=s2_CM`&@~#>?$V7>jaf}>BthRb;WhoI)}xs8$rizNwZvvnY-Fs^gs{x7ZM||h2%G?S7q5uTgpMA6lW_wcuW2y`'
    'abBLS82(8%&B&7rfn|=s@y&E*fQTe;qaKdcubvNmQD!$sCPR@izz=U>mfEm(yl{>Vt>~(#ngpv`)-?Z;J<d5isgw8)_oK9Ah|w9%'
    'fr4Y~>#hWisB#P_Wzw$3dsd@RHbV0VHeQ`vvg%nKl4n7?h+n_82TKn6=;ATB0jIqA5JuCDz#TRN(NN6!o^44W1FzIN##-S87eh7N'
    'tD*MQXUcI;yx;@zY9k)9frrVo{oHalj0H=d2<pv8HAcY(uEu3t$AV=Y;k<%nv|-D)`S<gUgOO2j7`uq0#0t|I`hNQGDk+bR=r=>}'
    'eL+;>#Cbt<KD*K`k{)$Yi`1_9Ui(_VMB=mW0N7lqShPYiAUp)9bSsZSk}&TEZU=|xa*oD+Q}hPuEQ2Mcg5QVk4o2K^;d`)$^V)&x'
    'O@+n}qYJ^{98byqQmKd}Ac$wE!%6&UBTa6T*&=O`%Z^`uym*Gt{x4~&EuV{JhCRA?4p5(_@z_$5_trsY06rfKI#?Q9eGX@LP2B5r'
    'um2FayWO)M#zTPG4bfMXs|y%{f)4z2JVt)FWINeMGuOI1(~XVrA$oK`|B#K-zqjHiQ$}n1%znqUkQ5-^oofhM0@re$zx+ww1<juM'
    'GVYQlid&ZD;I;Xo5|4G$=HejW#SL08hWLrw9X#TKQ0@JCU5!HfRRF&5R!|6!VRgYb=)%@cJ`&~mN@iLR(-H-FJyY9*Tm-al6Uv)f'
    'kcc;pc0(>aS-OtyGGM3P((-^l*B1gl)juej?#rIX0NEumkF9xP6cRdXrBNVm1AT<+c0IF^;l8X)-Zdx*9cabHI)$hK94Q$n|CS>@'
    'ZEJMm2&_;EM4Hd{$Y4btH5OxnVd|qLtN81$ViGkbH?mLy#D0C~6eeKXpdp|}<2(dTfhDIvg#e{+WMfagoRM!${O*!TIBj01ZDb&W'
    'q3=ei)jPnVEgjw1oeiHnId*SP*A#{-OyqJx@k;~AsHcqNx}1@LAxt{P^TLd~Pld4-IMCQB9S#j@o?>bxf8~7PGbZ+eP&NJY6H6Te'
    'y-cg5#@Z^LTxsk-?oCjM;|OR7RntE8lu*Yd2`6Rf3_L*`xBl2lC7JKEbX*&Pf~mf-nh_DhIG4U<oi}|$;{8a=|Ml&zM42H(-W>R1'
    '<Md83rn{cs@toeZmfSx`giGjD-KkiDD>^ifwUJ8CxBTey;0wt&9db1g1tG;cq=_=Gth!WN2CIJ-kP>8vjIjcqNVOimN_KXl9l`Xh'
    'UEg%u0TtTcuA)94a$fNqZOKhd>zha!#K8M88j3;=#%>w>UaN>IW3i4)&MM6hphlVdT{k^5Z&x=xB?MWi^%YDNilab`QRmO+iVHT`'
    ')$q{YWCK7APxXs;Bm~Y2P3rIYtp}i(GBU;sq6-`<a@lX-2Sj-8d(bp^f*#*F(5tHDuAxpCn2;;&($Xv`c%tp|rmr|ekzp7$5=5yk'
    'XmTeBK@Oys9z$${tv@D=%;Ou(-gR0Uxn$WgP{+kjqikkSEpPgm<FDD$SSVFjxe7g|06kf~FtfcX28=&PeE1t=pR3dy=!JMqfJ7V#'
    '3*3$ye_Hvy*C%KH1Oziq4sC87b_1YjVt5gXjTEjvbDWqqR;=SF(62In*yb;M=7Ve4;e<i<=%>HwM`uX_buQp7NxkLGC8ood{Ba@e'
    '0Q@{XEURQ&BT#$woK^Hqpwrp8bfVU)`^@+J<Z(&{00e!?Ae-rkD$<ePV2lVkJ1P>^V@ku@5av>K#?!Q{&86yikWhDS{$qJlPnOUM'
    'x2zk$UZYSaaY~5nb-VLI7w>jmmh&h|D16J1`#v&J2f$%>N`spfT+!43zu)sa7hTsR<f23xM0RosCURzYvup6*DmVLjF#o{>gYtsU'
    'T7=9AwpQD?@vr10b-(FD9ut315p}_!4LS*|%3^>8tvQ1Y$%^AV9a|MHscdWyb>Pz$>tqY6nFNMUBt{T&_jR=(o$?d_4^-_%NsOf!'
    'L0eOWO4bJI#@Fqb4fdvl!IWy|xyZC>!ZSjXHZySOx+eX}m~u_hObO!k`2!fc^jaL5bay}r<LSuUW(@%&MDxe=gJiPk3RQ4Hvg1y;'
    '%M16a90g~>Y%G~}l;Bbsbkjnh%1a*4?JpHVusmhzu#YJ$9Gs(i_*b%NrqiG2{653Ot2a^eMRK){2o$FXhiP|ky>4D8Qelv@n#l{z'
    'jG@#KbiQ%3wSX4xCC4nD)o)Lr<xa?6C?&uCNEM7YVYDxkA|Bcn8I>+Aga)e0i0c3ui3z|OZxP*J7>|72_~a0Pa6!mr@*W9>E9cb{'
    'SnzU}d>(Itle@O1&x9`+n<)CXo<8(X8`nelth;R`d2rx~V>2D3))^~Vayih?3ne=sDX*S=`>xv)$xTrRzuh6j9$ZCs)FaHT(H?Y{'
    '=(y9c9l=S}*VJLM5^#GS&la~?19?59#UtrOj()B?hs?l*tuLV^FPt1~)yw3XZ&v?1pXFk(v$JrM*6Q<fp`qXV#!WZ8!fr?QEiM5g'
    'EU^Bmhg(&IErT)`vop;%2VI?9%h{58a^i)V*(6|eloPml`?x>YAX|hRUvdh~pt!N;!{1PH^Y8~@guoI?QLnuc*1*(cwl|%OF%vv+'
    'P^ThQfI^;;Eqj8A^=9DseZCqfNe+S(-ij<t5c9iUMt%}FY@}F}Hyw)c8v@qkw5-0Cn*94(S4yr9)B~Z(=h5(A-9PsBR+Cd(^s7>e'
    '-je6Z2RB$2G|yMzO%K9Lg)W?BD9_en%|PM3;X_VzNc(YOu8jSnB)@Hc=q(F_7!7P&*AB9vZ)QgXvfvH~_?Wg(M`mioji)z^nM$)f'
    ')DzCiE4Dai<y7jZ8Y`J4LtPTde#m{}p?^L202ngkUQMp^;>Aj+2gJZ>hQLFtbyP|=AxIm5DV*eC$Q1$)RFxLG?54j&r8or1rlG}W'
    'Y=G=|aud_~$}FT7yn3hkFYHp-k#p#AFA?cwOstu$1Ws;#R|34+ey|e*1nx1H1N9TJLnHCm+tR8yeuQ6Kf4X`IiR!feH3nelo<&H;'
    'jV!4a3MT=0Usn*L<4!KVGk*|3!xx%`HS7J!`Ss0nucI5t#UQFiipUZf$bq{L{PxXDK2IaUA(m^oP3)P4fg6}wOuI3rw+*;2p32*r'
    'd`R4N<|gpH9|AtHlP(gnKto+M!Rs0vHRk5J*PHh%(wAcw)zRTtVX%K5R7-%)9jD4N(E@EG1qTw6>m({<BTY~{t-mSQZmA=`-grFs'
    'lu5#vMz^tGh~*n^j6aa0+stD|0NzAw5{7-<(hM*9y>)6$D>BtXeNN;ddrr3$S5<L65qJN6NIg*@zOY#g#+36t2RF=Y(9nTPlY|sv'
    'Mafp@!=rAjEL$o(()JYdU+=7yRYa}KKHSj~H?_AbOE7TMT;?0J`gk=s7dVDUkaSZ}bKz*3Ot|L}^5PRS$_uyHm5|;s2~)vFusB2E'
    '&H*E1BP&dv;V$vA`$2o{GV8f6g?Bqt43k|>RmNb77MB*9-+E|M0`FF#;}0g=2Z-;%EiDAMw@h#k%<N~ar$Ap4xmiyD7iTx^(67xg'
    'K>awa#&g`lqijx@s$_D+5Drvf7^>A1TuP$R^IJC}3C|FI|I<i_Vm*`LAT)1gJ&9DWPk&bTcY_d8mJTM3OGVaOwCh*sMs5*4f0b?Z'
    'xk8}rJ7!2SY;%l4%y>-}KB!kj0Yu8pQJ>gd{z=n9FlQXXo5*CY?jxASE`#8VFsP3;X$a3=Vrua}v2xxvR8tz$QI7%{xQ&NxB<bqJ'
    'r5I&Us728m4&Lrt?yf03Z0FZwH)&MIpnBvlSM+W02_!$r{&kypebpB6FEr-Y`wK^)E188;#qb-vZ_5cSdP}2Cri1zi`>v7^5`htM'
    '%Y%1}2jjCl*a!a!KRX=OKa*X9vV9Mu9m$Tq9*m40zzD%&C~D?$t^E*S1{(N*oSc|GhUnM%Lww)$U}mH^5kWXV*K@SWu(e&U@2rU#'
    'A5zuA5(D;f?sTpb>p5FmD3&Jy3ecQq8Xqbd9(K~*Konw)%TqE$dxE?Id9IPBNd6lJA0u>-Zt%Uyy7;=EvC3?dmhi-%Oe<}ftAQKY'
    'tftHie^i5&hoz|u<2Sj!yvY6FH2I3k6<j24oSz{q*O^vf++=XVKv_+vyd5=j-jRc}%>V9BbsZ~owH4<cvY|O!GzkU}qc4#6GWySx'
    'EYrEDU4slIuA|0uZl`0;`$3aeehL==;DPjt6z-vXgS(2%8k5j~upk^?|0gvk+^4u|VX|0^jHlUYIdP}A;mODEK7YDa<m?n#K_KKy'
    'z!<}-g1dn<s+E~V6)D>DR)7z%%!S$aU}34+jp~YH4-?m}yDXkTvkpL4z5X<($o2Y#(?l?gb;Q+zm~!0`QuSFQQ-yTXxhY$?S^#3*'
    'j8cMy(H1qM+8<5z%mGS<5aR7jqZ(?->uB#`#0gFH^G(PjxY@LOf-UYaS`L-xOVc%OAz74<maH}lVqTz!_#`prWs=>LkS;>eiW)sU'
    '>Q6p6fM8+gh~gXOD$<3n^Ywb3yOaHyr*PlTHw7wQ7wjIieDjV>Fut12P2?uYM)7l3##9B-Sh>2W;?98ar`asDrc`lc+3u%;l0eGq'
    '3_d;?R~tREgol2DOTruiN~1R8$@MmO)HfzL9B?o|ZzJPtBC{HK`>eX2N6H3u&*h_L3S;8UR9@NHw`DL7@1Mu4+_ifphWvWR7-YC|'
    ';i9kVI=Nstm>Fw0%;W78$ZR)UCar><F}n#kj9LiLsr>1LF|PX4>w=W2rX1<C+knxf(;`epa?eE>Pfpz7F`{{;@bKNh_<1m~VM<w8'
    'f*XzUq;~}Fve;&u$}?y=6&mM`L1~cj4cX~WAG8`;ehT1JFSoHTc%hny+^n*;p<MI-e9J&mF&BanQ6Ynj7f-9^bPY(^rU7!BJ$3d$'
    'F}48_``#ERmC5V8I!N+1ecpH8GOIjqA}~;$@ns;HTEpQKgHe-)Hg@nPs``3q%%7WT1`A85!%D+e0VIGyXOq@B734g|H`jyW9Na9~'
    'PEh1jRV`y;3bbLJ$9sA|-)MSSa0*1taNK34=tj!7#S9uHi;VqQa8=&dCfanzJN)55^pL@<$BKlC%PslWnaGl$nD$CxHzeMngZGc)'
    ')f+|u&X$=NU;iLCcTrV346U2;n~K<gR?@(0d#4rY>l={o#V<-MMPn!rj*)MCRq58@Z9VIwB=M#&-^dmjkCSqxeLI4eh2G9Q;IG#_'
    'rhWocgK(orkd5kNUru}E`fr|pVR(H#gge0~CTDMyI0W}jSfDOu9oR9OV`2BWO|&W*S?O{Dd9Go7$6|^qSY6*Lxsmz0Bt+18Nk&?@'
    '<er;`rB@8o7%=DkTzVOw!P1M#7!YQpl1SK#33BVs^n5(OL7YT^wswepqM%}AwSa-K7p?}q9UsSM<OF_;Fru4;HL6mwOE>oR4e~m8'
    '!~^x&7~}uE#PuT$LWHE@7Y6a|krCXPeR|BbKhNLgWN{hIyc<5hHvp4MY5{<^S6NOa@pypoNx5@gZVJGU8+(Rjj5{6s7mEL%t<A*5'
    '8Z2kl$W9Jyld?lvwuzf?cb>LCFRVeB9Q96u3{T0_R+5(G@}6gJ=IJtUNg8;6-BQbU&_VLzvQv<8alTT2dT{xbZ7~haf^b{d7m%#h'
    'tQsT#&bc05xZou?%|vH3rv?Rl#NO?W0-wViZ%)$loK;vNJK_Oi*qsTVOv`3m-SYu~Q1X-|pjA6Yn-SH?PLdbS%+S-#`kgs;|7rM*'
    'R%)IMRqu1&elx=psfwU~V0gTm-DkIh78Lk4LDq{0i5%UrPky^Zi92*Y<@x%fJ(mz2kg};5aq3of!waPrERaL^^@-<GVRG4>Ti?KI'
    'Lk%dT{^Fk}@H}I_&vT)6l^`RDM8R0S1H9YV<Iyen4KJhnfUfVppJ6s1e>^aUCys<7UJXE_+&DRNkU#PDmbm}!&p+HL1zDD>pJS9a'
    'O>B;z>X-ZX`23qc|1isALH^Wf^U03+`klXijnBXO^G~b!{1?dhQ=LCq-`9<UoA`Qj(in7KOw8B$eXkcNJ|J>d436a~S{ZzXcf=lp'
    '2cQ`((-|ocF^2+)wXYU`Mw$J+2Dz~Bgv~?h#rMR)cYcUuy`E)C=>ouv{-^SP&#Zrr112Qwnto1kejrw{yrf(Q%2Em=)xgU@Q$^Xc'
    '2%(QK?Gr+;9Y7Zn*huANUOQ09R_+UP2_QG1mBL_hT!&YM_wO~Co;KcG0DMf;JN%jx4=_X5?v|7H0m@&aGnGVfN-{U5wjHc(;jGSC'
    '5+T|p;f0!O`4pm9(d=Ds&;&DdGKzo&1?)y+x7m4dg_lJu#|4!sl`-Q?%y%F>4t6Ep^1uH&vW1aq(+uc_tIH-@g4=Ryy##CQ+HY7L'
    'jT1bAf-<?WUA+osac@iP{0<fx{fF}cID&P-MrzCKo<gb@99LUPy3B@&y3$ECN+L9>MwwGCu*j=qbfR;s%zp-7N5*)OCQ9hF1Fg+>'
    'r!4v}lQEmAdX!2~dw7Sj=X4`+eXL<}{6WX2QIcUyHB_F?i64hmWF#A{KUS(?B=a^*Z2khsi!{|q>&)jmR7jLRwn?x!aja(y9!VVP'
    'cgUY%;KcpwaFv3Q|1`SwCDEgtjeLrh8LPLo6SgHkgdhnu1OH`d>l8b@9b%IbPigX(Ua3KYhH}EpI2Q?wAv2TC*m~2{Ds@+J2*a?N'
    '>01i2;Wn*Ik#j)CQDd|=6Uv@;yz32{31+5@kYF|jV>JNzhTMrrG0>ML@1O58jZ+t02J1LaG6r43%xc;_=!2OY8{CIsE`QZ!90oI?'
    '^uu1a!zRS(szdZ0Aa}^iwvZd}Elf_r_=s-esXv6__-E)2@zZ<R=Etw_5$Y!wp}X3-`aY5tG}1D}GM5<t9r{o6%deLT9VtON_hN_<'
    '0)Mpqb-sxh&vDV4bDK_fIJ4|rshmicoSzD#KhKc54@%l|Ea*1+^DvRFH2S8Kx>`5_%o#MG(DyQerrz0EtcDT?1gwW)7{P6$Q+807'
    'EC!=YirA0RH==-LVE6m?-;DfEmO)2$69+Ivh)t`{*o32%GMZ6n=W`d7smneR?-9YyM4W#pI*{T(p}@l)<lHR=&vd$bzt?_$u>%BU'
    'g;+<v4PHBFk-U-V_O9Ae)y)Uuj(GL@=M+{*wCa}p$Q+q$3-e<=LJCS!C<By6eYz5$Y3e*H4BIC8Q0zNIAfA>tel-Gm5xPobEN~5`'
    'i?mT7%{F`t(eUHFY$}-v!r+8q$aOgO<XI86NLs@@ZU7FgyHgIPFf6GW87aqi77~_qPj<U_Botj!9Kn&yGyF4%=U6B%#)y@PZXbC|'
    'b&0aZZ7~qB4A}{Wr!B1?t_I1^?!jL`KR<ij&f>=TyhY|9=S%<5J<FR%Bv0h|%owTDW^?VTa`e8li0p#@U<w$4cQRi-IJ1{L&9ZY$'
    'ibCB?ih-I45&`ILnJ1{T0W|_>;cFXA-M|eMs5Kz@2LD2rkFza*+xW=oT36Li`rNn~?VSRJ^0!+m7yi@tG<<a7hi`ehc3S|cId%9|'
    '=gu6!VImnq#4Rwc3)U$HjI-;Ep1ZBidYRa2!-}BGsX@y%tSZKWIAe4X^IOvQi95Bi!xO<)e2`c)+N^23C{+P2J8&Y`jj2xUI=`x@'
    '$jMT4nBTg0b`WTdnSd0BvSw|m&FiTw;O|*;-BTOZnxwQ>_Pc~O5L4fDfs&!I?f@c8?$^CD%#hjppLedqfKDm4tgERt4S*ut4JI<K'
    'f{p@VM}marb~?qnWag506Jm_Mc!f!NB^5eN#5lyZaNw7D9~0m<f8o+|!$z2F6Eqb9m*CHLhVacyUSWxG^Sz`uf3$U^O*0m!qUgB>'
    '@8%4WTP0*XxK+4fWtfXt(`v^R`$;$X@5XE6Qd$O!2i&@9ic+~_2{Zl&*%Z)p_v|B^v0Z)aL5aX9nyxjEz2oF{$ewt6M{={8w$s_E'
    '@zdN$CofC=h^@3qpoiKgGTEY#V@yvs8etR67a3C$9C?*;zyFBlI&`iTiF$*snij3+r)FDKJvaxY5%By@iU{JK&0)aZ1)oJ)`9v@f'
    'z}?L8w0s2#joA*Jo<WV!_-RGUo0TJeNNl^tD{jNYv2inARwT=}FRYq}lM|0JyXy9TQ<30*XY_FEEg<Cjlj$1+nyjJdKpV^lt;Y5>'
    '`hSo+M*o08q<+A-a}<v=yGD<~L)}phk?B%!Q-b-@OAIme1mp3>efDO<y;`u<<uGINCDR9q0<4jn+@@@h>F}ho3GBSviTMO(pg2l*'
    '1A<}<jDHPR<Afa&ldV)zvrjM&J80o0gwPigm0`{+$#_LIIm<J)6#&9&&a<sZm%gx0(h-b5kkL7_ZhgI=Ja>U5x*R<oE(><}mGj)S'
    '%%O;m#t1tT<yrl;OoKOr_WA)1FYtI(Q9}2OXmprho3kR|j%)UIw6&U@a8B5b0+=!uk28=%p1u3lgFW)qOnrORuc5*}d8Xt_KIN3Z'
    '?TQVv<yuQ^&di0BR*Rwja{?d_jwzjlQ7yMw9}-Yo@RO<W0A~-NJC^xUFl&cJ!@0hROm$k?MHvyZW2L@9;Ee9GvMi&znruXyI7v?Z'
    '#qyFmZsAJQ)NB>9%n#cy=!b$6ydx3`Mx)zsP*R}T{{`#YTie8RodHQ?Xq7DFiHa9RDr`zjWCeR6F&g|q3Gk!of-yh$-h47|;{qtw'
    'Lx8)`8dASbfFEX|v6dqhd{Sppbge4OlQxoni!7YgUU0dX6-{KiN8E;rG**~UI)SpG&Du0asEe~VM5AG28EHoyDocn*LXSb!S@)eq'
    '-MqKZjhV2<Fjg}RaW;5*kUK5r@j;>7GJ<!o6Q+A9glJ<<taMqH-~ywmH|7hr7<oq#$bP+?xk&83IR_YpFG#^SJ`U6y^=rIEt_Sfj'
    'XhKC(-M%NUG8noM*<1{eWE&Z3KBT)g<94ooX=Mf@Q#HB@)#4OhlK~rA=<WaDdR7y`8)qf^F+?h4z#us4zyDetLfKqr0mi@~(E0yJ'
    'YU5UMzuM6#;{L&wJV!30N4+nr%64=JMUphCb@JATXceEXdF&((mh+eP@_Nv2?Dnxprb1tYi!4H$GA;YM+mDSQ!m%66e(o#w0-ro='
    'q|Iyw?(1MlFRquw4V>!OzpftgBeMi~2D8W`7dY2ob*8<Dg$I2`K`lsALKV>NtB#as(-vC25yEG{`~)+DCl;Y4OR%#I$Xe;EZ@DY}'
    'flOsux05r2s-;uXGA1Lw&5;Qz+1|Y!PwEDnSpeLQfI5NR@G?A46~w<wOC;@{7fcd(_>9I_{+K;dLklmVQVVTKl4nU33pS}tRLmJf'
    '*1u?%X@wUPUiM~YpxdHhB5=S%{un%2BWehm*=1&<YAM{*{bwFiK$WM{n}fe^!bURsj>fm85OOgH=}gqzM?yp;VUoPZ;uwejlC=HK'
    '=F@Fv`zAIo1iC#nZLSQ^X<FKDFDML7U}lYV3M#BMRk-fdqH~_r&gg**vg2Bjv>kHF1`dhaY`#OQI@>J1tgZz8y?UZ$v9lCj7MJMT'
    'g>&9phmSPb*kP7~?uJ8#%G9F@<8`IW>;5Nl>9G(aa^ryOs4)0G4P!$FL{&hNT{&--;%*j*>~1`%+@Z|CH%l&LJB=+D{z#;rDyC|='
    'T?Y8X;BhNXK*^^qhHt+ii5-BssXI_41(yhGaUJOl+c@3fJX{q=O8EjgB4dzwgp>n|JUOyks4*dlpFkeNFoudYoMRnlI}^zzQCkZI'
    'O%DHC#y=nGIGk{(d(Hn=to=O?7Z^LH!el+NGh#fc3qV~5OE=z6<_%OB@%Sm|Bu|3#O>vs65NE>hC)^=pMG3stjD9G`*<3L7^gAkE'
    'TvohHT(|dM_*NP?$9eoZ_9?{J9Ug=;d<Mp-sdZxrS7l_EIoWzBk(o2c;{NTt!Z@IjsL3cogWW|Pp;S6pUc2hJP0+1J1sEwc3Oees'
    '8o{^akxAweki+^5qmpS_y04{%q+=QWH_k}WNFZn<)TuED1u9>f!S97Ts^!znoy9lMD?>QW4u3`I^2b-=;UZ`#*zMfT;FD<T!lD}2'
    'z>4~z!r?IFL1)O=4fy0QBu*8*la?yQXjTU;^LH0QA7;`gj0J6X0238Z_kU0#G7M&Nw!bN5L%|_F5XZMH+Q~=kX4_DR4mxpMENq(T'
    '0xHG|&2qFCs7CR;@zb00mY*I-{`u+S7O3AJ1Mo8lG-5c4JajazLHU8h48s5lo?1;VG=Vl;A~k?q=m1YxdV(Q)0Yna{*>!=KaI;({'
    '@t4j!>rMNu;kkaG5pWPR95<p)VB%<x;Pz5AK)N`Gv19S-uDyE`7;lwf^xU1?C}qAE?1SbdXXSW@)GKik(DJ%5a5>_{j~g4c7c4#M'
    '$5F<g!?<rKa?W=2UD2g<U1uegP(f>0@G8_{LN!vVsiCrYXAC9~KaQ!adD3pOQxI~OJ`x#sRR(wV_A%$6Iuc{iRNsshSnclX)y!9f'
    'OZ%ul4mYDKL6|30cG<)Pm{=sp*L$#1$#?!(C5b%6;V^*)^BZzRMt8)uw5l*(Z8GeUhY?{#l^rHCp@chH6$ta=(BpN_sBfP8k`5es'
    'Y_JQ4IS?S@u+#1|5$C*MxJ-pTrXn-U2Te$Qn+mef`BQC%{C*s@VFEP*CrYLxim&VDK$TN9tIl-sxb(xH;?y}1w9^(F64;X0nP6&k'
    's5Fok94jNqg|x4pJh8=e0rrq}K4rzyT8QXk{{9Wk)nK8B^u{7v3(NLZy8M-{c8nXLuC3{bz}hlN8=E>m=?McnJu!D^!p-f*73|Pq'
    'ZmX{Wzi~PD$K6H{2PNBfC^okjtSp~SOvR>hgIkNlrkI(`1zYeNSpqPj^@{EDf)u!zo~}GHR-fJZ0%m~XfGg0@>z~toj6w#!7K#P^'
    'mkQgA>%h*&I?t5oo)N4Vc@G&1l0=FyXSn>>H_#uRs@9^g@Jqm(&9oWMN5itx2)@c?#`;R*%cPGyFCbng#A2=6>L<E-qT)WjS9oWD'
    'KBX){PkhjO_+~<MG{{Xou5s|?8MkxP3?0fxk*CYlO)2Qx4r-ZbqJ)&U5tS+{9~{ohstaE>{vM)W23Q^Ie!f1W)C;rPpr+{Yg1d|2'
    'wod`~#sebH(Bh^Y1R*ucvw{%0Y3_u<Kti{XB|fZW8-m*$H#eFwO7P8f=3RD(9NUCEFm2r$N0BL#x5Y^?Q?8H`?Me4(<~NFH0vsdv'
    'j!gae3Q9>MkWr}P1=+vJMq#FmZ@*9x(PT|V4Y&obE~clYk~tAfgAVu$3m0BJZo!QVWhQ+Z8Opim?yP-?R-Se{H?Bw@#WXsd95c{@'
    'yQOEQ0sNqB(YM^Rf>BSSa5O|~`C6RTxAg_}*>#VwT=;PY-rkWf*nP}@Ah#0LCR<iqEeHQLK3PUE?6a7@BXDO6TEiiyhMk!W%@PvH'
    '-hxpo4Jg)pO>qzt7kCyL*{h0Y&b|`70{+I*j+%XA>tgm(ww;@3w*HDc3h)IoKfn=@=(5?w#R6qu{24fF;3Ytm;5m}j1DFTOj9oj3'
    'Ilg6+OIJ<#ka1|xocZf$tQ(Ur6YT&I%<Ba<2{>Se5O{OUgH>uoJ&d{vVsBFh#$8Q|jDozppnAY*YLFDWWzAI1%?<V&Gl8p<aXDiR'
    'iY&#m%QW1&oed_$f%J8p2u(f2&MaV{DLMN9OE82<Fe{)I*)I&#{;zXBcz=ix7c@zCQ_P2Go`+8d%0(Unj%nPEwW#{USVJ`fb`&o)'
    '#Ja|{!JxG9Krm`4Oeu`SWqD|3BkyO?JnOx>7nt>|Jg9LZ`9&;wPxIp*SsgYam^g5vF8u5;Lk_Q!>txfxhD)_FuS>f1B<kK9h?n_Q'
    'g0AU#IMO=LirN7c_0-_qEm;CcE45MA6G}d@=4S-UNSB?QlKaBs^x$ilMri^Trb<Qd+fcV-0bp%*q~z$JI08i*Zvf<{i#hx^y^F9{'
    ';$fx-!*`F4v%{6)gE(g7bQ31P>DG`laJu1x{sAj%F$dM$iV{PI^0=L~tCxgWPKtshyppC4HtIGqZX}-yqIeC*Pe_mWR*&F;ZG6te'
    'PVA7HwG}KDXbF_8Th}Zy5Po+SL5<N8A42kpcRp_iT*KAMmC#Q~=q%_Em!`UpfL|UF!Y{9aColY_7f?mOX@ZPm^SAx;@ZMlrI0Wh5'
    'QJVv{I{yd@WpIY*UMa+vYIsWY<nc6R;Zx&?-x{)625(i+DM7g0#k72UMB(&d3J)I<r*+rUlp1FPzL*CJK^mwYv7k+J64;t7=f?T~'
    'Pw{GBS6zP7cAI_g3Q7tm2I~~4z`^ir@(rOsfNnN1|9;EyP^e|0!=P^C&4(7b9*4q_MotdAVz6kJQnIDR&S>d|Uy@~k1*wE?A4ece'
    '0^}kz^BYOZR91<88bN^}KkO`rv-`pn^u8V<YezO|z?MALJMdMCz`-tg{r<B>lv=kp!Ya)I6FU$u9EQ?Eny6!HV~mg?Q9oSw(q@Ru'
    '6h@p9IzU}D?cDZI<kdx)Jnaru4J(Y(=ZAH+yjNxf@#wnQhl}9be!1NoV&m)x&ETpM_%mn{0SH*3t<wLTktc^&!GX{+*?!%er&KVf'
    'vW3?kpn$7EW&aUcAZRhx&b(<UNexRa1xRB}LUFldn{P1$OQzwHgv3&TR79G-HtKs_t?M=E>rVkQDDL{fkAxi;@VuZ>RqDTvMy|5t'
    '8O!YeFE&hd)iEUQ{mO0A%?H#vg02|Z)kbY4rW5PQbva1J(1D9lO_v{@PsgK&c@14MM@=DL73a=%=!JCKZ3?Kt^Z4`VS6cb~JOBx!'
    'V`;fAZkZ7_BFM+7I2Ds`J~J<0UbQtON@Ka@o*ML~5bgic0?ZI~0Tgqw?7UMQIky|`jYL2VyJTIb=(mxCMAUSE3FxHg;5;r<pA&<X'
    '=mwDtG?g&CoLL=-k1vo~UC$2N)WWl)B0<I7=u*1zf|%Zp=9ucM@8ot}H$p#e8ld*y_2(RRl)DwXh-RFt1i&IqoZM9a;*sV~kTQl8'
    'ed^`Fw;wu_87VRNkkYsGTBVC<qgrBaauAqp6?X(0RjT@tNWZh&dnqz11z1DHTd`Mc0|u+K9`0}|fTlg2=_;z4if7tv@Cwh(%OD5E'
    'mSIysVRS=4u`n^KSjP#x6uJl{MgGL7JA+&Mn_UnQV!SeO;5Xg+!G_{Uh9n=zcSXtLqe6^D*~&IDsywltJTWd80};TI*V&Auu4R=q'
    'S==xE3tEDaVqRB<OH_KinKQ-N3sd9@AfRhWfnTt7(37unaMi@|9hM9NG-ZcN!$^dIsDQe4AHN0Ulzlrj3PtPT0kO|QufLQZ!o4p)'
    'L+RdTAqlAZUPw|?ON|?Dgz#1_IBxD~o3rSnwUf7r8%ce!eE9(2wgXaZb)YNdvnj|;J!^AlzXA{B%8;;F@3qN}ZM+ujK@Uscc?;;K'
    'Y!2XcOJMZE_H2;d5rHZA!@LXk9Np9^cG1&mGrC;-0$9|rtqA!%qF`}aYONnBmw&GWM}<qgJg=^TE%i>$`*7>SIQ{AY8=MN(7e5dk'
    'Q<zRJBm$>M!rKByyi{pon#L~d*pOW!1sY49g%yx9E1(&&<T+0u-P^=E4pFYK*(>9{li<y9qbZqGT{kt$b$0tWqqnqZr#DRz_~5Mw'
    'Yq23VLo?YC<Dr#2qEiQ)({3QkVF)6XeAYKRYzgqw+ZfczeThHErDW-djyGSgkjI;WcNoSSolh{(C=h|S>vE(WGQ4uCvW>{l8tn*g'
    'LndG^eA0pHl=+#5{5Hv#bNX#N3`J$(aLj#j20$XqIA;5-8~JI>HPg9+cA8+fCFV`kb;~i4#4?y>c!t^gE+CWaeM^ps7)`xd{%w_g'
    'a|9t>kNNmGCd97O%S^&s1nRW~s^&2N!FI&Rt+4*z`hqd2nGub_p{tPE&dhqf2c?b11R23#zN=RD>P+73YLMtmr4TAznanu9bsvs^'
    'Wsz}r^|>UNL-mmt4}1RVY*F@}BhM0NIlh{Xj+gDDR?1;gC`ftCN7WH)hjm1<**<S(4>mVp8-KvSx3iT;GD3uV6lemTASUzi|5DW%'
    '_(VBymZv3JU??rzHU<lKq;iZ%hp^m{*~nE}<Iui!O#iEICZ;D^g6E_qXmv8~UDCj$e=2c+#)P4%D{ZQPA;4l?O+WHTLBIR1x3YVI'
    'KQ<gj_f*0a#?!(<(Mw-%8ptNs|KP~d@_}p@2L<nv8e_So>jwHTk6k_YmyM$f9BTnB%F0w`@$a`ZvQ#RfyHMC8j9D^Jqs!>dAo*zb'
    'XeW`JXvI)<scgD#G?V23R&$!L<VGcnl7OMKoEJ4bjGz%RE!KJhc(L{n>C#~oK=)`c!_wV16a#lL@jR8oy#NOnU&Zw%JoN=ImjI=<'
    'j^Z0MARIC5@MZ9_*DL^c!OL$tL!6Hds|)!U{unPS$+<wO4RoPysRk6_$)ZW1W~)e_)?^n^vl;B5YPq8&56vmlzGnO&cP#4i+-(Ff'
    'a=L7`VC7yQ?U*m{fI_gl4%(GjB^#6j&2pz|)se^a{y0;*qvy)?gz9VT_zupV`Sa-J7O;*?@w#0L3p0nv_o>7Tvf7<2*Qv3mTwZ7*'
    '$Id~g(Htf1$%{!O^}t#q7wj9_n<V?aUm??hNJ2@%qu0iLK1UVu!?+iXTSsQdtA^i~rG-;+hP@WYL}noTQvrz7pu@`@j+wY>8Y5Hw'
    'cKy*06V_q7$iL}`kC5ZrZ+j3evXZ*~Ov?t$R3PTi=j17|;}tH@MFPpmIGu&}hK&btIhmM-+MDNx+@~cPpyVBL-B!z%U}XxKkulvu'
    'icct-PFFvQq@xw)denDndM4&HphIAOhRrLXUrgP5idc0FHK9CPslwRaWr9P0|IX|ve`!i?s_EVc3k;aSrt10_z1M`tE#;Z}r?1SO'
    'qwKkAh?53=ED=0n+DUkjx|s|e(8N&YH5T`p5ndz>^Q428!P=A2xcf$DW2h~fS*EE*!>4WoK}?C%`Td6>{Z%En09Q`aU(X?~sR-p2'
    'fIdEC*K$LKS*v&Wwqjcxm@FLeSwlC#Jhc#IU@_@OlNrYr0treoN>))+<GpHZ+U5}>>%OIxsP!8Mrk#$7PTxEUrDd=h56J|2hg?a='
    'tEh<UzN7dohdK!7SBC$<;8CPRIk|I-xoY$`6X~3q>r!Oanbw1%UuGdwxJswU_Nx=0g!H8}4ti!);~o!>ph`3|Y9#4nQOBqzL1}z6'
    ')w2%L-a0ZWK?^QqgqzSWH6Bq%Pq`CMNW%cOYzQ;-#fGLzE4y9YejU7sU<jhaIJ(G;Extt~r{7-N?R!84-}<J@S$7Wj4<(=yKjA2I'
    '`)hB2Q(K!7HUd5vvg0FZ5!OEI72zYB^aS9a4pq_7(YxXWga<H~46|LSysp@;7`|{!?m<qCg=y{#E}l$>ro%Ua3K%&fRe^&K!Fh?k'
    'y}l*7rD=L9YEEoq8m2L*iimC*)3cB?I)Y;XkAnuP&h<-_bbwM>qd8$r-saYyN^(~kfXB!^5%r!UiSY|7v<y>2s*8#lPms@@sfa+l'
    'vCv~PI6w`yq-sK&Vknk85d7O!9N=I-zV~nja$6fM3ir_f63R`DFG%c{-8Xv<<_#b>^e^m`sa_Ap^j_Qg7$`PK-CODRnECgyC^(!N'
    'lxR%2*&21<xzrlQ2yiSRxzh7%_y7lcvyb_ndzUP|#`<ni=@Y!b_r^06gu&pJ0^hBNgR2eg5%ka@$&m2k#cW}zWkQ@qIA)r<S+$!h'
    '#l(B3%&ZwGBt3^GK}s?=*!HgMnU{<J0?F{<wS??=Rj(@F05MQhcW3?n73}8TX6gtFspzI0<|W?x)q<oD`vOr*5GRmhASrAFf^2->'
    'Rgw~zYrz{7u$&QSMk!5rGU5&nDs$yDvzPap=>qQ365Tx~x}I-+D5!DDwkGO{n3`H&Gy}6Yjz4E_jy#6Q0DZPV)tRMmzzhg9SIz9g'
    '0?IomBq*BIpcF<kl@oJ~luqz&;p~KZI*Gw%G9w3-s1+6f9F(k*&CI06r;i?cycfEy5x;r8lUh11Yu62iwC2b#tOP+eFn9d+98i^4'
    '<AeZJISmd_sY;qq;g+sY-X(7a1oJl1j)<ph0dYTy6`;b?Z~zXk0o^7;vk1zOooHMT=OPn^K{?Mn;h=mw>RMZKg0--R3;sGU@c34`'
    'jzvvfD;OeAee_Ntvrf(mMNTbZGgzEP+-~`b-H;llf^w?dRu!8WGUo^_04j56>_^Y-WOo&rk)b+hS<LBJ+@3Qk9XPKVQYJd;WSAcj'
    'dn`7K&@b1!%wVM4OBwYGV?o|BK5j7j;`FKlcCbO9gQv>53X`FPxKKxf!q})mg4aW2j*vnCW%DPOdm^GjqozbwZlCL)B7()lg9b66'
    'OZg9!D5|;!1z~tL&td*BFJ`2kf!J|V=yIYIQT*aoP)l_)i*OSYExI_83=iS9&ZX@#ZWOH&GnnurV7`unWsJ2$O-7Oz9<@_ssNGJb'
    'X~?N%mf(iVvM(rLSsuIcDE{*-)6Fwwi&2_GwUXYg%(uf?YS2&7T<GD9x71}oS$>~*a}7+*u^Dt~qA!bsOin0+Q*a{|Q2ky@{loMz'
    'JlL|ofmy4E>7My8j==I9zk<9%++W5jpsjDlt4N<zzF#wU3aX>{`UGYOy@p|=`mkm)55t{7F-k0wYLkDyXT%#2A7=5CMgpW~3Ja=)'
    '7dR=@tUi1;a8b_f5}K79o*wrv5xJ5b>9Rj0NrhL7$0eQ80Bq$@QYpi;3#8!9Fb8sQCuky^PuFd>axhb9gaesKS40BP_3(5jvvC@;'
    '0LaSliKZgFXzwHJi`sNrOJ=0gRvTF$h5E?AP%`AnbPWDh(ke!%#oHOXVC$||hrR*0$!1_gMN9wzv?>hF0~Yr6H-bB1{>J?)N*gO~'
    '5%-V3iJOb)6LgQ1r+3U(QN~*ALztT5E@$zU9vJnL6u>B5(A6_QKm!mM)p;{PP`sWfUKGB`kP~oyi-CHsjPXEHYEfn(rQq9Q6b76n'
    'C}Jnu((4AsA&Dx{U^hZ8d0X`=XcR;Mz+uqh$$<zZ@m!x^OL%a{0oqN_D1BKuTDNv`yZ{z*zaeTEJQwt3hYnWN+w|YVuS)V7Uo7(S'
    'p;?}J8v<7>{>zkybRi-uHG6UyBY|#n8~hAK%Q>5dxGkAKEGP$)itS97y+f)qPcFz77TM#xyxrn)H$D8Yzh_T`_EtfF6or>D6P&*n'
    'zbMZ~?!)Vd<01F++k`jXT3*ENfQZ*z`uJG2;6soEV{$me@BnnxfNk#hjKkR4rDvi`gT1!jdL_p*cxOlEm*9Sid+Sx2HsI@hu9v78'
    'N*Lye)H}*`V*c^aoyTy^xFMeSiMmh&c3;}o(MJMn!b8xe>5_Ng4|2+mx{vpaZyeTQWO*xi8?=uz|IEq~4_2xj6E_^w+QqZx2u2^s'
    '^@|uEbAx7w;c_K2zE8KGXpT+_yAftEGQ@pJT*pTw9O5wfoyIC)g+*Kdg3Z;$%upB)Z9P`3gpZG!Gvlsw2f3PmjyDD)9iu93-C|ia'
    'u$A|zH+)DNmH=9=d&pDmIX<p;N@!-Lw~yqy>3Ycc7@j{<fv_%3(~h~+S<{rAdD+@BaK&#GgnW^(ZlxPtABP0!m>CMGo`kD4c@NO^'
    'R}@L!d+-$^-&Hh`o6ApY9#n!H#s?>}8XbQabgniFmQJR6FbGqw$6H>x9sxCs+F@#D3N25)NjS(PH+yzwPN#oR2N6T@Z#0+yXWxAN'
    'C`<&-ydzz^c^j-N%hD-{EU*gVXp*CKvsI5=p7pkJ7I%!GQobo@0Gah<e3YxHmhfX_H&d|<gKodMd2$~bXjF?*0*%lHP<NAIB~UYA'
    'GWW)mOtXMcBDJM#<z&0!H|zrkoRL4P_%Wc)(>hf?%T&HUGZe9F7lYn}?ToL@Sa^V`D;&gN2EjN4zhJrSx>Fy`fneOe%r)R4qkq~N'
    'PdUkuX<4b(96#+GOz{39fSvoMWHO-d+6^eke8Nus9lE~(6NdD3Y9NGX1+|}JJ`gi$(l?=z{Hd9!5iU7n$iGiT#ssR-AIHn`A?G;;'
    'fRUC3Wm9Ih4u$!gBE06py#m~*TPL#ob(zW3=}hd%zb3|HhcxK(fjI^WHaK`f!jxZMr401#-Gs*;ZBPfd*|u-6v|yjl=#OiXgJNT#'
    'H@heXIYfu%mU8moS@!}t1oDSYltIq*g6a7TE+r~6nSOp;li59Si>z&k7!X*Qlx?aa-|L7Qj>zHbTcEU`mg*VF#iN7$gXQ^xN5C7D'
    '_q|VoG!sL<0zHwRCD_^crLjXgmhp;2nwCky6TG2Cqi~6?gT>w<6HvR)m4nTV_SdF>UjsQsH$i67%5$-(_}VdWE2zc!z3B?~w0Ktu'
    'Sxs;g+HgZx{OVcxn$w13VVVfv;Nbg25~t9B8~K1sdAR<qey6xnid6}8lwv5#wg4@|bOXhp_kX+hfY&FD^i)!fm9lbL*UZI5`d>s`'
    'lYX>v2evpAGqRkkE}ow|qC>&i$+7J^{m-~VMfHm+-;7)CO9jFk*Ig$2?|+T}!dg$5NH9KaOzXIB2<OK3K*E`^W$-J+DWN^?z78ns'
    'uO^W%qYFj4pBtANG`iJ&2`wQ1uP_V{$(1)nSp7Evo?%1N)bI{b0yBx8d^8svXn~Po9do~12K_B#eVT>Y5%i15O)@V%4f2{{jQP_1'
    '8C3PtBmmu{taS4WZ=A)<X3-2v-7e+@^M;~H-#9$_N#q9`ZyMZyfEYlANfVD|(DqK@$AC?zZZnv++g?u|&cKPLKbjOn`lJ8#)<aAb'
    '^uru8*;x7vc<dp6<9`je6_$*cwee#pwZ2~Bimp#4#{|Jr1wWTcU_e3FFX_kT%Vw1_KAr|x-PiA^?5k_ky{h=|mEJ%y2`N{lPc<=b'
    'jo{$%LyTFQD=B)~krMU-?f^^rB5#prwMWT)pBoqatcL>{L#p9|v<y3QzRf-)YL=O}TXn7|(pDNIV|%s&XapZj!f2OqN}1r|eQ69O'
    ';U6lo=DPFkuZo_R0RKo?A9XNqw4_vZpV*Os<drVax?|FBdHmYW7FTTiB@l>9ShI1htHt=<NycK6ote7EKOYKCS9^CafH!lSidgUQ'
    'jT2&`edtLjj&_cfX7d)}ksa;GML3LwkQN!N=AoK>Usb8+0ByBf=_jJ}<g>&h%tCfCTEogQhse;#>I#L!sw{Z70ZG&M{3MwUoB9#e'
    'TiqWX6(E^`&aHpC5|fR9Ygz)()JO7sQ?_k_5S*_gfLa*>ySnWs%jF*_>tQ?(a44|2>9XTEqeqa1Pf1E^+&%Gms;{D`=fLlJ(^A?n'
    '3Y!k{D&BmX<C=DH7Ea=Qt{3(|^?7oRR9R{%4m?$jZSX~F|Lcs$_=@IC`x0-7C4Vo5tJp9gnnkX4ezo`RZqW5Ym_2NduKdg6fB{bx'
    '=`KV&Nc601%7ciXP<IF0cXCZBU6O1Ct5rxjT#?OV#Y+s}kpdO$iPLN!%;k<~xc&}4M-s9$7#MQP3T^5;2S3K*8m{KDaBE%5%`wV='
    '3)H>E3K8pQE0Km2RvUQHb?aU)81|%|pG@tlUr#f-F@8bq%0iwAFiVo_2&E%7k27}OVV;kea5Bf&XmW#I1yeXb`us_?!b6cLUG+A6'
    '(#9t;$=nVCX+GOL30)NNU;pCgkAdl4Oc&KsBclOhz?V$D)VaghRfUiA=dbzskA3`<?LKb|o`cC+?D(!hK)`Mph{@x#ef)}_|Jdh`'
    '`Q|K<tq|6+)NNJ*vd-h8e0;Xg-}3X{`|0P<EzzR}$7wfE=JRTPUX{-e`1PxP{fl2eN$zoD%mk>FmJrUqtqXM@oKkY@2oa+q03gD7'
    'S<L&a)54lFOcECxYcQ3`B4D^$g|IV9W+s<dOM*jbZr^H4#0pDOJ5^Hrs0MnGLq6!2p<x>_4b}q6&BkK<Ck2on$tX_L?^awY+a%tE'
    'mH;1HWYDt^H=S&8xA$eA%*5*)Ds<2jnJaZ^0a_gz+M&{-bT_yR_#OHWk_3!Pek)c3i!~yuhRMLZFeost*LzN|BgGY4A(Suo(^U6v'
    'Fxi;NWk+*Glste1bADA*d0EXUVr|UJA?*w@z}q&j?{M#sDt>e>7XvbQ5z@nN<HHJZ54DdQ8Nj>?lHKnI`A$fOO`KQd*dk_V;ks~_'
    'iO{~;2wX)AS$E3_K`QP8Eu8BM`;c0eG<yT~`f-NC5{8VpSyHA&r#Tr@$!VWNDJ(4!#_DyexkHO#Z3b~7H`;s$KL7{pa_5*&R$}C7'
    '?~#*qHgjQiv})1ZQS$a1E(z&;TgE$91Fw%Jzq?H$u*$BRY(N~v#3^Tm3e3Lu4CJmZAld!>LaGQIilJhLz1Z%8baD(0P>n6Ul3Et`'
    ')w96_P&CJt_K)dgEm}F3q2=NKXnuw;qOLGjtL080tGy+sTy34RW~yj$w7p6wX0WO6mpjz)zrpS3R4L?}Kg3RH8mv9-$k|g(O)q45'
    '`zDDDkmk>nTu=fXb=J;P!nst(M=4KoCLJjr&B(9p7-Md1qJJNeGrxca<qTY)>iAgE?X(XoLf(?dU`htKcV@aAx`oTnozgRa+`#MV'
    '2Qw!()Yk<1B2GSm#p&dUu6^QUa{IDZ1ZAKVh+MKTA-tuQjw07dJAPWLT@KdB7ep8g$0m}Wndog$Y2;nA?o`jfa9fi3o3X0eMdF*('
    '&V0`6!E%3>++5{U%mzRRnm5i6Fu|W!iRZ2s9=v5@W7#V@mGQn;Kn#ZrGm5~HB`^K68+J@E+>g}oTGau{B5<0icIJ&jkC@!i;4d7s'
    'PP{$J3#_Ils0i=mM!Aak>${j>zu2I9kqUY$nVeya{AM_+%l?=^KDzp5P^Anh$xg_Mjaz7%s>O2PI@#NMStY3>kP_;eL4ALF9;hxE'
    'IYRp95S%Run@mmnn+0qC{Y!*}#^Oa?nF}&YArysbD@FUoCXG2wVdfYH0BqmR<xW;Wb*rrpReED3iYOlj!4JkkWiyEDF26DgK~os9'
    'Y86TRTV0QG?!`U3=bo}dItc<KkV}+31UfCoKc{qV51I&WdKy)5L=AxhGm+qScY<bo9J&_~<=5L}ZdI1N_*e!r8eks_BQ59!Z2>0%'
    '!jl&bL8i{M$ydNel8F%<a6(Xwb^HAV{h4-aJGt2^4#9M@4eH7@N(NxAZ2~!^Bi3tx6HF2}VO+plydwJiws!h$SV>XRIpr?SWJ`if'
    '-#NHw+KZ++8>}C~AP<C1)VB2wwN*|;rUh+6-Fi<Of|$G!k;PNw217tLY6bCB6@5lYI2C&*Fc(}z7tk`lDU6)u%UNj(7(`Fg_Hd39'
    'k5(MW@E92t>T<K905_xfI3sHU8VRQ{|BDuumez<0F_H*d)%$m+XZeW7zNy{~U8EW`N)-aMPXVdS%KIv%Gr9#aZRXGL{yQ@R#fGqw'
    'ojFqhbr@>3+!a>L*Nw7evtIR5ro$MCFw9&4k4cgu0uKRtIH8EGsoh3{)43<lH|)v$F2Y}D*Ixke5(r+8PJX~C<TrBSQyWu<@9y2v'
    'bBfwU1FuxQ&uRP-9|QR{De@c>+%I)n#z)#zO-Jds;cfSuBfK?vQsFZ$V(wXZkrJg{Sj|;eS`p^s9#&6Xg0}hZ4B%F(pqU@@FRc8Q'
    'S>K{lWNC;*NhJrEN&bv3SFW?FkZ<X>+|rpCS78Xyj{;{S(y$4~6r#thC}YD5O}Bt$@0XG3%IrWkJ<N53@i`*XS5V;5mtUt%^MUBn'
    '<mF`Z!r-#v9&UJzYz;`{g5XGuwPJ9C0=e0Vb#tde(oN8G{Pu|r=prf!0}B|USAEkoAt&;!Sj}r)HJk2j#7s|)8&9;N$7<kirG1*2'
    'Lmupa7nCU7r?k2}o13&{OqZND?*+oTpKEg1xsT}(h1=8K>R70vvgondW`jC{mNx$+T|0d+L)uD@sEsX_n=aSCi18?ytLDme>}%bM'
    'kR+<(C0FvDrjq4pGsu|#;owT46PCfk)Dl~aVLLr{5&;q5y#mS6lTaxJ=g##GcDc?njnx~geb@ac!kQXRZPy4q;fJvGHpZQ%dfegB'
    'JGTSr+}6S|w_^Eb-UN#-9;`^v%Z=Md6~Rt+>b1f%8WbtlS;taG^AOlq5iSUeWK}F@iuL#3eSy)XC>*K>?XSW;g!k$K5DjPWI<mQ='
    '%EVWQ7jVGEXZK2GC;9^*1NS*b0pvTG_cpdDXak#d4|NZz1CnwnXOo|-^|0rghZJBf<}De(xs4s%jsU&48v*Ks^8&Xrj}?ijYQ?bA'
    'G8j^w$0B$t7;r6<$jO-rGK~XP<XqwY%6;?BtcA_avYSxIQRnalv~rhacno3?06VpIX<Ejf$|Yl(yl=nO<i6&L?{;U9h8@761EAh5'
    'ErIF3f7)*0=WK%liMr?{cXBFdiI#gO;@;fLOtlqE+`>8JcFWuFsvv-Efapl8K#<fv9v)?%2*rKCAVcy)#V`k`(uNoURJTZ=0IoaB'
    '!in->6qg%S%cE)3nmx`dkQBVrH*XK6!2Al>??y)Yo|JXEOk)RaR@fS;=w}0^m}F^OSG6?kGI--OnM$>Jt)@KlMe_o2db{-)+TM3A'
    'I8NdM$&KOuqoF@YNRUdANNLhEm1p-g@`@K2ir@x%QCKY&8l>0t(8S{3>`PxQ!d1nZ`cNG0;%=Ef5;HAt*9~_)NNg;SqcX*xj9y8V'
    'Tr_b;W0G{GrY(%BWs1i>tSMB?Z9r#vK~{RDdb(cyy#FKrjG-wRa(-Sgb_-Np$Y(vmrwZqlM5Ub!6N&d6bJ(xiba;!dgLf4%lE@0V'
    'Lk2!9vKBkjsR}EP(-*Qkt98Sj3NLE2Qwg8LN70RRruYUP54UyDlt5sq_1>2iQ$E@5ZfOks>o0g|VYp~2y;s)y_$0UCo(RmewhKT+'
    'P^x?wfy9`pCWIT%sJ0j&H10k&U<)o}Ro-yY?=Bohm=Nh+{1gf@9;Y614OX)6oAA-=l50C-C(jtC5RA=3SSZiE<fzMX-Ncn<GRgv~'
    '8U@G#mytYuzX9+ZvEuq+^uWm_`1O0HV36MR3Z<H9W_Df%*kf%dNj}^LKzSKZh#oy9-Sn7cRBYS1QfP>4)^#%~he5bPUtLv;bWhMJ'
    '0H)zRq+$zH_gu7A-Ps1E*WC#Ae1f-4FOr*8GddnOiN9cFYr*60?vI<3^oy$fqgpK&(^x5@`1N|)G`{DyMf_4wzfbk#GH>u?i?Y}m'
    'Fm2sVWJiyW<UZ~Wsx&M`o%<M*CIu<Mvjv_!P0+yy2z#xC*AlQ8r=$5uZ^j)&F4!4aphYeInxP3S<8WI}glPT*Oz}`LEkz=TQH;lI'
    'lU9oTwE&A_U>78ca@37tJvCBytVPzr2~v@|-m6svNN|;Y-3f3U<txx|7zQ=!1BI$BE3IXqDyA|3`M?kGe$8V<2i7qTPaj4GX$~|T'
    '2QtHi_*HIrFmk<PHWbC&zF&L$m~f5$w59MtNjXLRVaAQdppp>APpQEG+H#<|SSJ!<$DGOdMDl{QvcUyxgI#(xhj2ObYvY^1pl&hP'
    ';7#8}T##mpcKDkNbjW5?o3w|(+sXdGUSoE<?Dc#6+#Hk&<ypdJqKmsa>2IcpAKmoO>sw%!hSby!lfyITq&%2}9vTP)p#})Ap<!;j'
    'OEYg}&A64xC!uXiC`)hzU$02!u7y7WQg1p$Jo0~zu{g`GiDu(YS(&UN?D$wl^QN5$^sxO@<(U}4&;zO-WD@_66)3ct3`N1OZ*2AM'
    'A`qQ5pezPkb4!}bZu(FpMHh1<O{IjuVc-svYKRti1X~p~+mq%r$r=gZq0K6;^Yzhp4qOL*3Zq)U|8Tf8EM1h}PMRgM+#7Me*h1qb'
    'CdrGj?xgeX$b=L$9ykoS{x+BkwZCYine2AU_lUlbUC##32!a@v8YkS-;h@h`I9hc=)TJP`<^7l;8K}8-axDso6n^@^zH>&v5SpK!'
    '@VxoyLv_SYZ>Rmn#IsCdGMtYtNA;iz;Thl+%J&8fn*ClF_1FZB|Nb(M-vFV!8L)0>cr1B6G3`{&L#8!tWfT3dGH_m#hu0UEH;2P^'
    'I!`p9;uF;nqz2NEkDF07O@7BeciwD(hAypRaPbWT^Mh}z$bM&=z^lgLJN~z(@vP_OB+8p{|NkBbV)CQ8GPnTUK*RI^N`YX7ihzm^'
    '6qFmgN-mndP<Dl25)|+^*?km^Dh{Z>uw?M0AN*}B7o{YC;hI$ChXVT*6vDsPSD~%be+WS;fc3|4h@}S-8;*>H5_jw#fXBa(5*sJ5'
    'mE{F1Rc#(LOLS||OtK;-fUZ$CI125lwUSF(www$a6UuH9+rZUQzf8<v6OXkInVJD!*o=6wOMqP0_<kZ!u@_96uQvB+^Fr8J(!j-8'
    '0Z`SL5Zz#Cgu$i;SgpiepnKoCGBr|Zlm#17)8oJKNwf1FMH2A!^=K^d){Lb&7s~?rVFv%?{V@w2$~X5)jt6acx4}nfHQeg583&{#'
    'L-z$*-K%^#Hz1u=#L)B2_{LHc$aB@-Y*E3;@47-0;S;U~wUzbuPh&&Vi;Jm;cCD~n-UVM;qz0m4q0y3yHEHNpk;2&~!+9tZ&7+^b'
    '^4vu-{Wato_pB>TtTlsB5DvV67LA$%`I`NEiQCdJ#IuxXC>BK#f=@kblg}MfLj&9fxUn3;nj?%XL8HKhM6|o0(FmqTfVPb_?Fj@v'
    'D0nc`Y|JDHv|`>2*N*x{OEE+Rz?|3HF9)>MBwd*!<rUmxG#X5Tu`Mq|4Vy5ZF!d4IZb^1wvJW99ZpHtYS$vUg7rbt9u4j|1S1_9c'
    '>NSRfO%f-w_rf#)OW<U%XJD8Y_c%=G-d58jDrPen-W%85JB})gy)`e{lPd#?v=@{E(ijF0Q-mPd>lbF2d>!mN0pPdT6`De3=g1nY'
    'X;7*wt*kp`HArtJZ33pBU^+ZR65``Q*{vCs-7+avdbamg@4?=kqCWhws^{x^WrcIXx|L<n{aD^3jTC2%Tj>?a+xW0lVevBA%RJ<o'
    '<4i~vO{T2o0upW+{%!r}uq|nj=)JE>;SLWn9?^L5W_cOfat(Uy!bR4ygutVQP-M!5KjH=#N?~S**U=FaEM)Nxr#j=@N!-`NeLb`^'
    'aQzi2<*Whmf$byuBpc)nKYgIs<uU+U*O`7@K_FKe9<39aEXv7%Kv58%8<|iCOWBxa9ot{DJ1!RE4qZjbJtP)QU>D(2$oJCC3q3jf'
    'Ea(8}QS_U#v1~{LObqTXczG2<Rv{3`vy_OASup7rWEQhCB`5TRcK#v(ryxZ~<4OQ*6|4Z59fLSe4TY3h+(wbJ#AR%+Q&ToB^bFuS'
    'G}46}MOy0}ba+xmMoK)EUC-t~tT{lO8e!Qty`Z4o<6g4J5I1)nD?8pU_JvcW>8gnzW6wOWQ&5Nos%Y?_k&hmYIzr5W(%2H8>|Nr5'
    '`I`&td&w=rLzv{emk<}QlDlG!mYC7yZ8DtdnMA%MSY8MIvRPShfOksLV0#8{8dV=#Z$D@SBGEq8K+IeeQ>an%xMFLu>$`!F(H9_9'
    'I2_pU;FqHo1H*xytZt;rY&8wOD@}xCbn!MfA!0#a;p$BGRr0mSes+xj@nEy>hnl@pq{!ju==?MhZB@XRoE$f+ScO=qGAuz7Hr*0B'
    '9>h0t)9{t3<G_-S?|NGY7Q8jGJs9Z+d@r(bSGoVp_3;uWcE&fArdsf)JSrW^Fa(O8>szjA;N9BkbmNs&r$8UP7Suev(GSi@A-U%E'
    'fEvBm#hYLQ4W-EMko#6(#qcLj(ZkUGGUbM)rK4#&ttF~_<UfwjyTBD<pQ+)Pb$OS*OQbyK?aVb^_|mJas1X7$<Aji+C=&n^g`&ee'
    '!-I$w)BxU`;gWsjTvX>cCy$o$hR|Z9P}M20fAkjuQp_Epfs}-ULYZf6py6;sSW>&3>|L;YR11mm0^`<Zu52Wx3{>=tv>f5H0}9q8'
    '=r~awL*rg_n%i*Zof+MXK=>336o9-x94dsvOu7k5{l<~tk^qG;(6mH=U{jDP2XqQ>2p|L_>>)!9X8P1@Tko-$LST`^9?fquHyB5P'
    'M-HFjA(W9isF+?JlChQ};KiDO$FEl@HI&57`^Vg7QCz@dpv9Z)%}=+3kX*9-?Sxc%$-lW7okF9!ijiT21(KVg$7?%^FDC5>qp0_v'
    'V4nda36uSAH-Bb^7QYhfc+P_&;)g2h+^c`q&*W?kc|eA-qY26G<tiLf{N)ZGe46%q79}B5UJEv(cc;&!G@(f%^W&{tun+;5AXo(D'
    '>VDDAHR%O2(K+%2JTb}U2A2U0=}(z|_8cBz>|`Ek^0TDh4azI%E8Kb>08lsw&?Q`bN<&V~E_lLxfuyWHVPk$58y^BTrY+jRLArVF'
    '8KCY3cqnIyr@h&=f>zGVji`1duNZK#gbVGLsinTfmwR(X&IQ!-j=p*@=c|WcZ7nrL;y2?xsaSGq1&gq=t0-s|dp*GQ=`yX`&BxN>'
    'Srgj@bx1i0gHAYpK0ys-2!DF0`ZD9`qi{JdaJf~OJ5>>NXe~DNk=^>ylZX-lab(2}Uxq2=*os7KrOAXb4CtIva;~`z&rNO~s;sR*'
    'R_d?z8|fIZv-I$iM5!-OHsMV2nj+<`NSP`kJ7zQ`y`+NVc&F-Vh2-^$SnDWqQH&wKx)hJX|5SHwNs{C?5WJC$;2Zb9n9?2DJ+nU='
    'vt8Sn(%;&sOWjq)Btak|Tyn&$d$AdX_CDJSsHZYCB3roX{|&R8%;>?8R9+^b-a(+`(Q<xpv9bNwOd?661I`T8IGq@qPIZ;$?yo|)'
    'HxU?gCAWG!1a&45+S?vNcpVI<A-U(nW-S(NSdpINb{2Igj?!53A4wc0IkjW8M>6p&ct+JY*rC)-vgNl1zlwyOxLjyDir&UNM04&K'
    'T2CXE*#Uw(vPPfX?ZcR%GCs7`(E1pp`oyN}N>2;dKf7_ruw(nc;T{Czh&oS(7-BDih$7jLmU86nW*S|X)!$~Ed2FwBDK<G<e%Kfe'
    '`Na{rjh;X>xA3^oEWWcGkJM?;iD0lkf^lvO9Sh&!97-ILgp^~91ZZHp|8~4x#%+34>LtZyIug{KDK7ETR)?2e<}d+gZaLQ|Z=j(q'
    'c$183Wa$`l!C3MVMAm71E~2#*{tj=06>VKWIaDrW@)X2FxywskLy{nchfd&Up~`tr&E$mr>SCT_QfbTbL2pbsoGIoteriZg8<p=H'
    'h))owR<8o|AU*=p<_^*3(|UYXux@aF6IoqpXk1i2Y;{b|c|$mB62?P<Dx=Yua!+6<tf4i52$AI2;KAraSPh>7j|7wY^}2Ocv8g28'
    't71L2f)xuf^jy?mISStVQ6C2o#BHoL`J;THQC*IsFmMma9w5HLX(}Zb8HrClhHw?+C$nTa9~IN;y)>8<)}?173<iNU$?OJnz7WDY'
    'Es76A!*w*hyBVuXNQ%S<)NGghfSeMhMr9^H-ziTBCX;P}+Fmlrt#FoQr^NJs`1O~j;Ve+hwB;V^5y<AdCi?I7VTiaM=9F@>n1_Fj'
    'LzS<#ZMHdKGi7z+6bTl<X<}NI!Gv@vH^uiZcM~TKh{2NichPz2!$TmUj6p)wtWXR{`!G^@(v%bwfMxfk*I!J$2sfH>&|LS%Ii}l;'
    '3b*P+@LbGjNEId)iMkXDyQ3;YQB3qJ;Q$5|GEmt%BXWBm7OU`KsB$~9CN}k&=$k~`)#0=KwNWm?5ynDEHEVDEK@^;k3z{I9uA!-?'
    'I9teJr+O)&7tGt7K{T|S-HT5ry63a9OQukYTgl^r2TUtgS9C;q9_tM_B+j>j)g_sqFF_<p-ktu(mWZ%QAUX^xVOHDF`3e{iXwPhl'
    'Cif+-aD3L{u-@v+C>O|1s+=5n2ooH7GqFL^7phN9E(W1cs2ds=+Nbg$dIB-ZG-Qm}4tgKAKp1q7IYy4U4G}qnAq76&TP*36sb(M!'
    '4nE2ma};0!aGueG6ID)%T{FGB6|Cp?8yy3_tSko-fr(&94YPdNx;AuYBR_IWc5^oQJoM#6L0}FW4iDBOD~{pb!d-&7VZ?oJb`|85'
    '|EYK|vm_h?C{I!E)rV4vUiqv}1etcgHP~6P-v*fYn)m~&F9^L`_l1zpAr^aUYbVSnZ>37p{Yb|_x*Un7Gpv4>7~}|bfl`NQBX1?K'
    'm^{QsHwjzgvAl@ld6l_+%#2dJgPA${1qlx4P-B9H@L6T}lJnB4%S}fv#~;us8!Oy#4N8<U_8SK^iTcZ|Il2RpB?6#*dt6vAJEhY0'
    '+wyWq%$sx+?mn$l>9=;c=Ld>Wdbx3@8L0JxT(aRPAC$mIT`qPmy2D0zF&|F!C;Utk@lNBo{|QfMlYv49gkAu1J>-y7YDgE(dO0vj'
    'UiiK3D-akTF`uGWL56G`TUVPkC=Dh(l2jPH;_k=PkIY~>f@H%33x@CJQgEX#s7!H9f#yLnrew5B3N(ar^0ArAb^S+wz!8d_&bV^y'
    '3?#5S%OZKkQGi3yo~6D*)-){pOT*@bytxC@*EL>Pw+x~a#H|cNSq{7TGtY%IbTT6!f%G>Wes_(f;gD7A4CY1}<%kmT<<fxH2*cOp'
    '2=n(3$r$KuU}=OR9g<O(+Ag*KSZLZi;7sj1T+tCQv4G^+npA~1hZY7LGC3-Q#tw7}q(+gTl#JAv+XlC)qi%^oH&aR7F=)ETL)ZK?'
    'vH;!)2sFPlthY3`Sm%~D=kG7ExOJK0zQuuH$R~Si8v{$Joo`G&5+Xu&D2}8j5EA9K!Tp01z~FzRs<-t@lTg3)s+3f9#*(98D*R{F'
    'Pl|Hj4Q!(>&}IRtz(}KmN?foNAUaz7xM4>Mk{Q8gmG3kv$$@6a^x0%Q$6SbXGxEGHqA_}a6Sp5t!+c7k50mE@Ey^)I>2tQqZd);G'
    'DzD7O%P6)O^4}|RmgLnhq0Z&MhC9{<DIsf^9f1)KsIps8WOZtZ{s9_v*|t$ujFjSyWBb+}rK-q@p^Nx=gygi?X&B3bx|FBBYMk1-'
    '_(~$p1m|lz`^!0c%$$H_=Fco@V07ps+rMR~hHsBM+0NE|*<{Ng3o09%7*9R)99vf<&fX?5lBB-~QinJp5lwxc<nN%uuBUaaU9=Qi'
    '0SwPJW2%HUf%hOyAUVY!pN82?M;dQ$6;IGH;;ztKnv<DXVm2k0Q~L~ymaCF%t8U+ImWGO!B^ql~MI4O6P*wNiPXtJ#-kb-9%eTQq'
    '7`DY`ah#eC&Ci5zLaO0^qDfEww4u&xmIaQ9%r!WVc@&{-)Q0F}po)Y{c7lV=I-y$P4JR`!P7z~0=YIF`m$-8nxEpGitC(?j%XK_n'
    '!;Nc_Z_bd_TxqPR1ROC=-B*`44S${!c!nsU?%@!jeMeP9)gh;ftNYxNpL<b9sJ;mzDk%MtB=qkrUPAn*+AcmIf?K&Pq3jYj7PS_Y'
    'qR^xpC>&WkIv7#gFPy3e@f(rq5LzbND=&Zc*%!A>B!W5|46pJ!Fth115-f$wA>abYYnux0I>`!ju-YKlXwwT_uX6BD94}XrVEKvD'
    's6dRI;DOQSifyCu)|d_PGYzM*OO&S_l;x(_@TNz(x(u<RQAKh!WeoWMQz`*rq7GWo@2~Y?PJc$H%xTzO&w5>|9T^r`1$PN2MHzq|'
    't7FNf{^AH7A66FpZx^OSBBT1`DoG}0f#T|tLad8mpsG7a&Ia@nNv4Cr_!5XQ4NRT}zDy2)gO^k8DdBA)*-E}o9hSqE3*z4~KTF{}'
    '5Rb%>LT=7G#BtFrg4}VWVv|BzgC>#LI&ZNV(*XxBaBuemWV9Ov*s&?m@@A9h!j}x4nPy6E)sQfgXeAv6?_8;wLq^?aK{_Kqpg`yb'
    'kx#Qz;pP-wISKN;-WOhy>N)cd%-7lAmZBsHvm&;`P}Qv3D5q`yFk>Fw0p%7a4uQEGw}>+YGNZQHeCYH#<=5&C<t~;rUVE-JY(haJ'
    'r<uKYsFhe)tiq#3jf<~1JET*SPeq4}P6qBRDbJcK<Jct8p|iFZRt6<<;^9%x{dyr9iA7ujlz~v?b?)M6(vsgS#x90~iefA0&Msk}'
    ')yiDh<Y^CVJ}I#=iob(2(K#eRlN-y#yu*xZf<c5UwPMCGREaXMDiREwJ(W7oglJ(-0n1L6ODN~$V1}V|3q2f_3-@ep`2w5i@&r|I'
    '_is;P9U)~*dt=IJ2!S0EtHBirK1M#3ViXB@?5ETvC<3G=NVjDuYDY*!1Ojh+lp&3?pkQd+yXAP(u^Qev)v9c7<@8Fj=wvT#Pkn3*'
    '#)JNwY=W!s8%e7q?0^yIkExiKHAbY_tbtT}e?<(**SC1rc*YQUKc)672?^IlAfL%z)UD^qT}5k+p(z+<;w{&dV1iXGjdUU<3Ep)@'
    'eek@jmk#Y85X;f&6KRgZ#59c1P1GdF2g<;G>I)d~=;EF?x4(Bchb^+4>^nadn5UxBge4OlT>X~g2oEnFlTE}z`1A<mLd@K2-G$uo'
    'B8Lb7CkjqLp6stF`Vz-1v9LIU0#A$Rv1arhnOvK0I>s&Pfc>KAUT<tlR&3EK_5m#lkJv0HYCAqy*Vah&BLki25rVCO6eNnnQoj5>'
    'oILj~KiOzv-EwxO_4%YYxNg*BR^~PZ^fA_Z6bAO(UQVcKx?6(_Kr4Zzu-L^|3Y8;e_)6-rY8WuA3uQs^s%+9`w5^G@(26C}g6)zE'
    '`}3@Wnm9$@O&m6H+iL(t(?>z>o@*qOSPZqB4wc?iB{Yavcw)yO?tjsML!SV{b3s-|`PP@z<XJ8c%DCHfX^C<v&L&UQ<?L1;h9w>+'
    '6L*eD)MU{BTQ!j;>Akd`a+wxSf2OMkVQ#`kT}iT_m^ZBp?g7WfH)2n|CZjkFrq^7I(ii5*tf%Qk7zb*cO%V2TRU(?ct{=`Uyq+@p'
    '6Dc}gtH)El#><EQ)g&VL(n4~ruG|W2rPifsfGa+(gc5J0+y5fUOFS|yB&@z-=a*by+Ol?$0w_or<nyfRCRm;Xw#>rK_$ZGu^G9LN'
    'si#Lxa<UjWT&%&dY$F1OMBy(@Frn2kMevy_yX|$zec;b-RByBjpVgorb|bkzhKu#OsSmWZ$c-<|pcIFl6_$_FRyrcPrnx@SNrAbM'
    'M0BJdk6L`vL6qno&NxZD-AO;q!`B#Z=uZlSq_jCQ5Tjl;x*s~y?jx=f)&qY0%XU4=E;p!vkYbMFqqFl{hOqBlS_JgKo_DEom{FGd'
    'JXM0W_bdhi>>*BBE{3#;Ir(@@x0gINLr!hdH-dzY^l$|dleTk!#VY20o?6O5(ZH05oXhN^LqPK&wLzdBPhgUkk8<4#ibr^3JGJEL'
    ';)2Nq?nUXNs@Y!C7eB<E_4($H^Qc`NOq9x_zck}f)|X9@%00b9&4fe0FuU4jN4AN0PZ|Se#DSw4(rzMv6KRY?C{5cZ9s*t}_hbc1'
    'x5Bv&HY~fa2*eq{A}=gAa(bY!*B!r;sXk$Uarm&(ZCY(yF`qYRQ6!2~1P`d9L<^+Q==o%U&|?QRK!p7|#QVHrnXnf!*+Biau9W$!'
    '<r@*aq`;x*@t&eU))!KN#I{Yc#EU4~&?tbq*)I-#hbjg9J($U|Y^~Ria*`v<x%sS@ZOrZ_tC;cM0<p=!F`0T1AbPY^9j&zQh<jE_'
    'W_E>PI7G5sQlL!T)mTWpCWIep8R){u*_&h*5(!O)E&qT1PZBIFP`#<up}g{bt^UGzlSbeU#d&zB4!ET2LbEkGwiHnF)QSO?<%0di'
    '?m4L#Ddah3YxNk54OM-6W0dbi1_>}Ix<t6g!V*^`xm;4uG;U>SNur>ef*dbdR0s?Lf^Kb!y>I&MbMH7N#J~a9dns@=EHqlpidg5q'
    'a|-R*-|BA(RUgqxt0dbxcF~#VSK8sg(nk|=rrYqVL8)er1>);NainSi>i&l#SZR+Ye0EeU{3$1z2Me6za{oJqY2G4x&puxU<x@di'
    '5viK)kTQ25^8qs}L0pl;Z}-%UlfDWZc8fg%R)NCURdGv{!`A<urZcwSGR(V@)vU&Ev(}K-_?B@M(~Urui*cn6ohU-K=#h|w$tK-d'
    'h|hV%zj<8o4N32nQc4{k^v~&)4T*7<jK|-=mqiDuic*AL&rV2VGG(gL;KX&UTq87BNxjs@u!t-WgT9*KHv1Mx((`#vuVk)x<{~z;'
    'MFBx0F~Wodonw9=uR%Nmq}{*#yjhnFy*Dch0X>g~atQGCh5&OO^01cUbpQR<aOVw8tkHoDU5AFdK%ci%rfbbNmp(l#w_|mc@7&6q'
    ')AMgfy)J0LIkZ_F9sFtct1ZhB)XDDBqBvZSRrj;yL*_eXx)@Q(RH;=IAbBRL^H`OpB#fvD6fdG45A`Z#&6_N%W#@y&sQ5ufb7bR?'
    'B}(h0>4@%L^3A0&9IYfNGj?s(cf3R4%`Yd0{}er$cJ+Ku(J=PM&Sn-z7zUQH>I>^O_;u1p?f4i7DHz<u3*5w}PTls4^ubW)9^T%$'
    '(fgPPR5X5wu0U@u(7Ig(g51gEbW==s%MS+s%dMM)&{$r92*zxqg76Q~anD!E_gc4N-ux(3eO}X0oQhUg_9f-;?&!*rC&pxbm)^J{'
    '_!}P?75?3p2d(h-0<bl>qp;$npN2(^k|*5evob*wZABlWt#@;|4BK&;UFs(U(0yrXrj11>#l-n%S-znh@Af;;U)6!%jA?JANArEM'
    'Ne<yZqo4}UrpGQTB;9sVd)t?O0j$sAX@?s7DKCU0GbClxTUH939uBl#T(ZhljPnK>2looslBZrP-`H61g<{YOw>mky5)kV)K!j$n'
    '3|Ad(R#@W&4flGZz1N|iV8vor$0*&^sPvx2z>-@=iVSt~0;`x=bvm;6{=30ty;lqj9i)@*%t%D&)u+}gW&1SKJ4xgu5QnKFBqR#5'
    'K`O&0m<@R{1x_N;Koz&Aoe+~M-DkM}Ah9INVnoH^AE|&g=je-d(~oC;bIq$ZNA$izJWXNsOIOu+tl?+xFL}P`$eb&F`NN%L`b-d8'
    'YJA=%TZ}KIF^uF}G+rv)<KcV1C!J(uK1LExmk4@BtJ@|aIvhcxf?$o{!n3FIl)dQcs-$?CDm2-;WR-}Zzq9UK^PM5H#Y%goidD1h'
    'qpOJ3i^y}4;-&C}C>7Dx?KE73O9~U}i%lJ16TZZwa%B`I)gUokN(@aFr&B6lfR&BA5b2nAD<amZkB+x?Hh4|<gs!($>bJKLqlB~G'
    'ab}~b4ao=>XP!zwXX^Lmt~+mqIzR7eR~Ps1i(iSzvKHWR8(!nV9xtURxvTMHtFF8z20nRp+qNv|6|zrz<qoCdG$gPhSr1V%;&bhU'
    'd3!tPsOfzE%-?SguMmXztT2yazR&YbUz6=^nmP6?f~D8X=&Yz54Tv^D%YR-`<Mg1A8=)LvHLFnTYj|ruuy@G{={I~Hl1lIt+{Lv2'
    'iXlRnLx(DHw~__iE3uEv^65p0M26#~lC;;ViM=4}`~?(C@rVQdIKGlhivRI?^S2Cz2gv@y0#h9kWGgBji>N%OWk-}Z?>(_)B?8*i'
    'fIRG7h-8fJ1Doa}IU-R_;YQ4#N^hB|=<O00wVP6*BdK{V=(MHLEv)#<D=7pSB&O6<wxZQ5h)LAYv=!}0fuD+s)~un)`#Zm1dAvHI'
    'SfcfM?EtcWkgJKzuCA9099-@nf9m%y{C*kd&>rIr17i$BoY6=DqC{43)B5oSzyHqfU-$jG(zhdtbO$?j5edeeeXXBAn(ue`{g;0K'
    '%AZ~d9gL{~r_ZWE#802;_b2n?Eq?v6KYr!+3z>2}2i*KG9o>F~'
)
_D3_C6_CACHE = None


def _d3_reference_c6():
    global _D3_C6_CACHE
    if _D3_C6_CACHE is None:
        import base64
        import zlib
        text = zlib.decompress(base64.b85decode(_D3_C6_ENCODED)).decode('ascii')
        _D3_C6_CACHE = np.fromstring(text, sep=' ').reshape(5356, 7, 7)
    return _D3_C6_CACHE


def _d3_surrogate_remainder(atoms, cartesian_map, scale):
    """Coordination-dependent pair D3(BJ) beyond the origin quadratic jet."""
    dimension = cartesian_map.shape[1]
    numbers = atoms.numbers
    count = len(atoms)
    if (atoms.pbc.any() or count < 2
            or np.any((numbers < 1) | (numbers > 103))):
        return lambda y: (0.0, np.zeros(dimension))
    first, second = np.triu_indices(count, 1)
    high = np.maximum(numbers[first], numbers[second])
    low = np.minimum(numbers[first], numbers[second])
    pair_index = low + high * (high - 1) // 2 - 1
    reference = _d3_reference_c6()[pair_index].copy()
    # Stored matrices have axes [second-reference, first-reference].
    transpose = numbers[first] > numbers[second]
    reference[transpose] = reference[transpose].transpose(0, 2, 1)
    reference_cn = _D3_CN_REFERENCES[numbers - 1]
    valid_reference = reference_cn >= 0
    radii = (4.0 / 3.0) * _D3_COVALENT_RADII[numbers - 1] / units.Bohr
    pair_radii = radii[first] + radii[second]
    q = np.sqrt(0.5 * np.sqrt(numbers) * _D3_R4_OVER_R2[numbers - 1])
    ratio = 3.0 * q[first] * q[second]
    damping = 0.4289 * np.sqrt(ratio) + 4.4407
    cartesian = atoms.positions / units.Bohr
    mapping = cartesian_map[:3 * count].reshape(count, 3, dimension) / units.Bohr
    pair_origin = cartesian[first] - cartesian[second]
    pair_map = mapping[first] - mapping[second]
    conversion = units.Hartree / scale

    def evaluate(y):
        pair = pair_origin + np.einsum('pij,j->pi', pair_map, y)
        distance = np.sqrt(np.sum(pair * pair, axis=1))
        if np.any(distance.real <= 0) or not np.all(np.isfinite(distance)):
            raise ValueError('Invalid D3 pair geometry')
        occupation = 1.0 / (1.0 + np.exp(-16.0 * (pair_radii / distance - 1.0)))
        cn = np.zeros(count, dtype=distance.dtype)
        np.add.at(cn, first, occupation)
        np.add.at(cn, second, occupation)
        difference = cn[:, None] - reference_cn
        logits = np.where(valid_reference, -4.0 * difference**2, -np.inf)
        # A shared real shift cancels algebraically even for complex queries.
        shift = np.max(logits.real, axis=1, keepdims=True)
        weights = np.exp(logits - shift)
        weights /= np.sum(weights, axis=1, keepdims=True)
        log_derivative = -8.0 * difference
        derivatives = weights * (log_derivative
                                 - np.sum(weights * log_derivative, axis=1,
                                          keepdims=True))
        c6 = np.einsum('pi,pij,pj->p', weights[first], reference, weights[second])
        dc6_first = np.einsum('pi,pij,pj->p', derivatives[first], reference,
                             weights[second])
        dc6_second = np.einsum('pi,pij,pj->p', weights[first], reference,
                              derivatives[second])
        inverse6 = 1.0 / (distance**6 + damping**6)
        inverse8 = 1.0 / (distance**8 + damping**8)
        potential = inverse6 + 0.7875 * ratio * inverse8
        energy = -np.sum(c6 * potential)
        cn_gradient = np.zeros(count, dtype=distance.dtype)
        np.add.at(cn_gradient, first, -potential * dc6_first)
        np.add.at(cn_gradient, second, -potential * dc6_second)
        dcn_dr = -16.0 * pair_radii / distance**2 * occupation * (1.0 - occupation)
        radial = (c6 * (6.0 * distance**5 * inverse6**2
                        + 0.7875 * ratio * 8.0 * distance**7 * inverse8**2)
                  + (cn_gradient[first] + cn_gradient[second]) * dcn_dr)
        gradient_y = np.einsum('p,pi,pij->j', radial / distance, pair, pair_map)
        if not np.isfinite(energy) or not np.all(np.isfinite(gradient_y)):
            raise ValueError('Nonfinite D3 reference interpolation')
        return energy * conversion, gradient_y * conversion

    origin = np.zeros(dimension)
    value0, gradient0 = evaluate(origin)
    hessian0 = np.empty((dimension, dimension))
    for column in range(dimension):
        shifted = np.zeros(dimension, dtype=complex)
        shifted[column] = 1e-20j
        _, derivative = evaluate(shifted)
        hessian0[:, column] = derivative.imag / 1e-20
    hessian0 = 0.5 * (hessian0 + hessian0.T)
    if not np.all(np.isfinite(hessian0)):
        raise ValueError('Nonfinite D3 origin Hessian')

    def remainder(y):
        value, gradient_y = evaluate(y)
        return (float(value - value0 - gradient0 @ y - 0.5 * y @ hessian0 @ y),
                gradient_y - gradient0 - hessian0 @ y)
    return remainder


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
        if rs in ['mis', 'ras']:
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]
        self.delta_cell = delta0

        self.sigma_inc = sigma_inc if sigma_inc is not None else default['sigma_inc']
        self.sigma_dec = sigma_dec if sigma_dec is not None else default['sigma_dec']
        self.rho_inc = rho_inc if rho_inc is not None else default['rho_inc']
        self.rho_dec = rho_dec if rho_dec is not None else default['rho_dec']
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

        return self._surrogate_step(s, smag)

    def _surrogate_step(self, ordinary, magnitude):
        """Small gradient-enhanced GP with a physical quadratic prior.

        The surrogate is fit and minimized only during remote optimization;
        its microiterations never request an additional physical force call.
        """
        if not self.internal or self.pes.cons.has_inequalities():
            return ordinary, magnitude
        if getattr(self, '_gp_owner', None) is not self.pes:
            self._gp_owner = self.pes
            self._gp_history = []
        q, g, energy = self.pes.get_x(), self.pes.get_g(), self.pes.get_f()
        history = self._gp_history
        history.append((q.copy(), g.copy(), energy, self.pes.get_Unred().copy()))
        history[:] = [point for point in history[-5:]
                      if np.max(np.abs(self.pes.wrap_dx(point[0] - q)))
                      <= 2.0 * self.delta]
        if len(history) < 3:
            return ordinary, magnitude
        free = self.pes.get_Ufree()
        offsets = np.array([self.pes.wrap_dx(point[0] - q) for point in history])
        directions = free @ (free.T @ np.column_stack((ordinary, offsets.T)))
        basis, singular, _ = np.linalg.svd(directions, full_matrices=False)
        if not singular.size or singular[0] < 1e-10:
            return ordinary, magnitude
        basis = basis[:, singular > 1e-6 * singular[0]]
        d = basis.shape[1]
        local_h = self.pes.get_HL_projected(basis).asarray()
        eig, vec = eigh(local_h)
        eig = np.maximum(np.abs(eig), 1e-4)
        basis = basis @ vec
        ordinary_z = basis.T @ ordinary
        scale = max(float(np.sum(eig * ordinary_z**2)), 1e-8)
        transform = basis * np.sqrt(scale / eig)
        points = (offsets @ basis) * np.sqrt(eig / scale)
        g0 = g @ transform / scale
        embedding = basis * np.sqrt(eig / scale)
        derivative_maps = np.array([
            (embedding.T @ old_basis) @ (old_basis.T @ transform)
            for _, _, _, old_basis in history])
        # The current transform lies in the current tangent range exactly.
        derivative_maps[-1] = np.eye(d)
        angle_rows = []
        diagonal = getattr(self.pes, '_curvature_metric_diagonal', None)
        if diagonal is not None:
            internals = self.pes.int
            row = internals.ntrans + internals.nbonds
            for angle, active in zip(internals.internals['angles'],
                                     internals._active['angles']):
                if not active:
                    continue
                if (all(0 <= i < len(self.pes.atoms) for i in angle.indices)
                        and 0.0 < q[row] < np.pi
                        and np.sin(q[row])**2 > 64.0 * np.finfo(float).eps
                        and np.isfinite(diagonal[row]) and diagonal[row] > 0.0):
                    angle_rows.append(row)
                row += 1
        angle_rows = np.asarray(angle_rows, dtype=int)
        reference_angles = q[angle_rows]
        angle_transform = transform[angle_rows]
        angle_stiffness = (diagonal[angle_rows] / scale if angle_rows.size
                           else np.empty(0))
        reference_sine_squared = np.sin(reference_angles)**2
        reference_h = np.tanh(2.0 * np.sin(reference_angles / 2.0))

        try:
            micro_dispersion = _d3_surrogate_remainder(
                self.pes.atoms, self.pes._get_Binv() @ transform, scale)
        except ValueError:
            return ordinary, magnitude

        def bending_correction(y):
            delta = angle_transform @ y
            angles = reference_angles + delta
            if (not np.all(np.isfinite(angles)) or np.any(angles <= 0.0)
                    or np.any(angles >= 2.0 * np.pi)):
                raise ValueError('GP bending query leaves the angular domain')
            sine = np.sin(angles)
            difference = -2.0 * np.sin(reference_angles + delta / 2.0) * np.sin(delta / 2.0)
            denominator_base = sine**2 + 3.0 * reference_sine_squared
            h = np.tanh(2.0 * np.sin(angles / 2.0))
            dh = np.cos(angles / 2.0) * (1.0 - h**2)
            denominator = denominator_base * h
            ddenominator = 2.0 * sine * np.cos(angles) * h + denominator_base * dh
            prefactor = 2.0 * angle_stiffness * reference_h
            value = prefactor * difference**2 / denominator
            gradient = prefactor * (-2.0 * difference * sine / denominator
                                    - difference**2 * ddenominator / denominator**2)
            if not np.all(np.isfinite(value)) or not np.all(np.isfinite(gradient)):
                raise ValueError('Nonfinite GP bending correction')
            micro_value, micro_gradient = micro_dispersion(y)
            return (float(np.sum(value - 0.5 * angle_stiffness * delta**2)) + micro_value,
                    angle_transform.T @ (gradient - angle_stiffness * delta) + micro_gradient)

        def bending_features(y):
            delta = angle_transform @ y
            angles = reference_angles + delta
            if (not np.all(np.isfinite(angles)) or np.any(angles <= 0.0)
                    or np.any(angles >= 2.0 * np.pi)):
                raise ValueError('GP bending feature leaves the angular domain')
            sine = np.sin(angles)
            difference = -2.0 * np.sin(reference_angles + delta / 2.0) * np.sin(delta / 2.0)
            denominator_base = sine**2 + 3.0 * reference_sine_squared
            h = np.tanh(2.0 * np.sin(angles / 2.0))
            dh = np.cos(angles / 2.0) * (1.0 - h**2)
            denominator = denominator_base * h
            ddenominator = 2.0 * sine * np.cos(angles) * h + denominator_base * dh
            root = np.sqrt(reference_h / denominator)
            coordinate = -2.0 * difference * root
            slope = 2.0 * root * (sine + 0.5 * difference * ddenominator / denominator)
            feature = y + embedding[angle_rows].T @ (coordinate - delta)
            mapping = (np.eye(d) + embedding[angle_rows].T
                       @ ((slope - 1.0)[:, None] * angle_transform))
            if not np.all(np.isfinite(feature)) or not np.all(np.isfinite(mapping)):
                raise ValueError('Nonfinite GP bending feature')
            return feature, mapping

        try:
            warped_history = [bending_features(point) for point in points]
        except ValueError:
            return ordinary, magnitude
        kernel_points = np.array([point for point, _ in warped_history])
        kernel_maps = np.array([mapping @ old_mapping
                                for (_, mapping), old_mapping
                                in zip(warped_history, derivative_maps)])
        count = len(history)
        size = count * (d + 1)
        # Squared-exponential covariance and its analytic mixed derivatives.
        inv_l2 = 0.25
        kernel = np.empty((size, size))
        target = np.empty(size)
        identity = np.eye(d)
        for i, (_, old_g, old_e, _) in enumerate(history):
            a = i * (d + 1)
            try:
                extra_mean, extra_gradient = bending_correction(points[i])
            except ValueError:
                return ordinary, magnitude
            target[a] = ((old_e - energy) / scale - g0 @ points[i]
                         - 0.5 * points[i] @ points[i] - extra_mean)
            target[a+1:a+d+1] = (old_g @ transform / scale
                                   - derivative_maps[i].T @ (g0 + points[i] + extra_gradient))
            for j in range(count):
                b = j * (d + 1)
                diff = kernel_points[i] - kernel_points[j]
                k = np.exp(-0.5 * inv_l2 * (diff @ diff))
                kernel[a, b] = k
                kernel[a, b+1:b+d+1] = (inv_l2 * diff * k) @ kernel_maps[j]
                kernel[a+1:a+d+1, b] = kernel_maps[i].T @ (-inv_l2 * diff * k)
                mixed = (inv_l2 * identity - inv_l2**2 * np.outer(diff, diff)) * k
                kernel[a+1:a+d+1, b+1:b+d+1] = (
                    kernel_maps[i].T @ mixed @ kernel_maps[j])
        kernel.flat[::size+1] += 1e-7
        try:
            from scipy.linalg import cho_factor, cho_solve
            factor = cho_factor(kernel, lower=True, check_finite=False)
            weights = cho_solve(factor, target, check_finite=False)
        except np.linalg.LinAlgError:
            return ordinary, magnitude

        def features(y):
            feature, query_map = bending_features(y)
            diff = feature - kernel_points
            kval = np.exp(-0.5 * inv_l2 * np.sum(diff * diff, axis=1))
            value = np.column_stack((kval, inv_l2 * diff * kval[:, None]))
            deriv = np.empty((count, d + 1, d))
            deriv[:, 0, :] = -inv_l2 * diff * kval[:, None]
            deriv[:, 1:, :] = (inv_l2 * identity[None, :, :]
                               - inv_l2**2 * diff[:, :, None] * diff[:, None, :]
                               ) * kval[:, None, None]
            for i, mapping in enumerate(kernel_maps):
                value[i, 1:] = value[i, 1:] @ mapping
                deriv[i, 1:, :] = mapping.T @ deriv[i, 1:, :]
            return value.ravel(), deriv.reshape(size, d) @ query_map

        origin_value, origin_deriv = features(np.zeros(d))
        correction0 = origin_value @ weights
        correction_g0 = origin_deriv.T @ weights

        def objective(y):
            value, deriv = features(y)
            correction = value @ weights - correction0 - correction_g0 @ y
            extra_mean, extra_gradient = bending_correction(y)
            return (g0 @ y + 0.5 * y @ y + correction + extra_mean,
                    g0 + y + deriv.T @ weights - correction_g0 + extra_gradient)

        from scipy.optimize import minimize as minimize_surrogate
        initial = ordinary_z * np.sqrt(eig / scale)
        try:
            fit = minimize_surrogate(objective, initial, jac=True, method='BFGS',
                                     options={'maxiter': 12, 'gtol': 1e-5})
        except ValueError:
            return ordinary, magnitude
        if not np.all(np.isfinite(fit.x)):
            return ordinary, magnitude
        candidate = transform @ fit.x + self.pes.get_scons()
        candidate_norm = np.linalg.norm(candidate)
        candidate_magnitude = np.max(np.abs(candidate))
        if (g @ candidate >= 0 or candidate_magnitude > self.delta
                or candidate_norm > 1.5 * np.linalg.norm(ordinary)):
            return ordinary, magnitude
        try:
            predicted, _ = objective(fit.x)
            baseline, _ = objective(initial)
        except ValueError:
            return ordinary, magnitude
        value, _ = features(fit.x)
        variance = max(0.0, 1.0 - value @ cho_solve(factor, value, check_finite=False))
        if (predicted >= baseline or predicted >= 0
                or np.sqrt(variance) > 0.5 * abs(predicted)):
            return ordinary, magnitude
        return candidate, candidate_magnitude

    def _new_covalent_contact(self):
        """Detect unseen contacts using the existing automatic bond rule."""
        if (not self.internal or not isinstance(self.user_internal, bool)
                or self.pes.atoms.pbc.any()
                or self.pes.cons.residual().size
                or self.pes.cons.has_inequalities()):
            return False
        atoms = self.pes.atoms
        numbers = atoms.numbers
        count = len(atoms)
        seen = getattr(self, '_seen_covalent_contacts', None)
        if seen is None:
            seen = set()
            self._seen_covalent_contacts = seen
        seen.update(tuple(sorted(bond.indices))
                    for bond in self.pes.int.internals['bonds']
                    if all(0 <= i < count and numbers[i] > 0
                           for i in bond.indices))
        first, second = np.triu_indices(count, 1)
        real = (numbers[first] > 0) & (numbers[second] > 0)
        first, second = first[real], second[real]
        radii = covalent_radii[numbers]
        distance = np.linalg.norm(atoms.positions[first] - atoms.positions[second],
                                  axis=1)
        close = distance <= 1.25 * (radii[first] + radii[second])
        contacts = set(zip(first[close].tolist(), second[close].tolist()))
        new = bool(contacts - seen)
        seen.update(contacts)
        return new

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
        smag *= getattr(self.pes, '_collision_step_fraction', 1.0)

        # Check for bad internals, and if found, reset PES object.
        # This skips the trust radius update.
        if self.internal and (self.pes.int.check_for_bad_internals()
                              or self._new_covalent_contact()):
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
                self.delta = max(smag_int * self.sigma_dec, self.delta_min)
                if smag_cell > 0:
                    self.delta_cell = max(self.delta_cell * self.sigma_dec,
                                          self.delta_min)
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

def minimize_func(positions, atomic_numbers, calc, max_force_calls, converged):
    pos_ang = np.array(positions) / _ANGSTROM_TO_NM
    atoms = Atoms(numbers=atomic_numbers, positions=pos_ang)
    wrapper = _WrappedCalc(calc)
    atoms.calc = wrapper
    opt = Sella(atoms, internal=True, order=0, allow_fragments=True, logfile=None)
    # Collective coordinates amplify internal motion differently from bond,
    # angle and torsion coordinates. Keep their conservative initial radius.
    if opt.pes.int.ntrans == 0 and opt.pes.int.nrotations == 0:
        opt.delta = 0.2
    for _ in opt.irun(fmax=0, steps=max_force_calls - 1):
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
