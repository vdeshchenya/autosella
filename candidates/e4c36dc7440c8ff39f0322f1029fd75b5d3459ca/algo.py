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

# D4 reference polarizabilities/data adapted from dftd4, LGPL-3.0-or-later.
# https://github.com/dftd4/dftd4/tree/82fbaf41724ab9a3c0a38ddc978ad0c38c4659b4
# Distributed without warranty under GNU LGPL v3 or later:
# https://www.gnu.org/licenses/lgpl-3.0.html
# Encoded JSON holds generic elemental reference data, not molecular results.
_D4_DATA_ENCODED = (
    'c$|#AOOkC#mSZ=UYFj+f|A$iv$rdHqWTV*R<cS5SdHOZ4Vp>v)@xsFW*}dne?hiJA{r~=t|M{Q)*MI)c|NOuI&;R^?{>T6E?|=Wh'
    '{3%~Q|6hIo@89!JnWNP8HU8YSPOJLEZe#TO)&JaG_F4KvU2~TH)&7ig)>!&i@7l$Gn0uBp^oMrJXqUfruGzbO%{l$HsXuJ}Yvt$<'
    'Q-4ia`on0e-KIbEzOH^(`O|h=g@3rUaVGz89jBkoAIiA?hs!Fx@}K<{|HY;5Ir+Cs(LXW&<3GOiZO<M0JNlYqm7)Lb*1B)2r?pnc'
    'k0&v|pTZb*>M6`W{Z@Sx{h^Ju_Th1i)#lb8YHdv)vFj<ceXkyee&=e7=Q7VKO@AnRbsoZA`mFrzv-We3{?N-Q+n?v2z4>1I=-cZ*'
    'S3B$Qy=EQL&tSBpZ_mRxYcKxUR>p3A81?MQzg6~b{z>WM_B@Td&-o9Y!vFtEull@WbzYx-VXK`#^j&$4dgPNY&}*NAFW_gX|E&5f'
    '{wGUso6mMfEq+mFPJI@>(AI<Ib?Dq|Kl{mf;R{_IGX9aFk6-vl`aFGdN_x3Gd0u?uzt^><|A|_3Nc_Uqt$xY+#kc#5>XH8Ddye~O'
    'ouL!<Rf~Rd``+_w)<1f)?dXSxzpCnJo<CRr>+k{42hv&Zf9ATo@|NfBW%JGN?0fQ%=4f^M(9gjc)!#kz68ut^z8D{J-liwUSKg}}'
    'pY_|1Phi*Y@K<bIFT!tC=jBYlUj1+W(x)^Z7u_nIXa3OEsOz0^z3a7gKiky*$IH-j)~~Lc*T`SgcUk&xdZ=gU2mkuVPwL;FRNH^_'
    'XvfyqojTR6l>OD}AH{>gsnFTqb?b~?&c1H**ZQ+-(=#J(oVgp<LQh(!Nq^9>xZBs_>eo$Qi@sLnUQC^u>g%xk@NugCxAdLrsy<Ik'
    'kKNZ{jqb3a8`OQ&n+{KX&r7F{KQ!ISmDr%~<x`^TdU#Aj@n-9jCcx;IbRB73k$z_7f&BQ%@bS^dt(!-3#-BCj)rUQuQ&d~mInk3Q'
    '7V1H5eJlOOy6)G~^*K67y>YmB@q4weLENnREa$5){k?jfI_KTrubidlv-CZ55cIyAPMPB0811WG!ldp~RcyYBs(MH|di&0)TYmrO'
    '4KIS|A=Wfq5q(4bY8|`b*uD1L1aDorGYLsm!DS3itHS7Bwfd%3J}{^A)Ts;ji`z8XpR+Gfzh9k^JN1&fzGJUfA4eZsVSA7h8Y!UZ'
    '1a18bz3L86J?qkmp=CM>>t8FnU57KWx-JH1q|~}N5lesbamdlLc3`=jTz~G$x#w*v2>Md#)OQ~MeGJ_%Un)g3pN&zgZ&%@+`42j}'
    '37);QcaxNy`szC7OP9(wNpH97FVVBQ&O^J;sj#IeRP-+Y^^c!?+$22_MJweczv<TQ@|~)u-0GuMiPCvWydoVhC4$!P(tBJwkP4vF'
    'xo?tk%G9^g<5FyDl`FM*b-dO-^t*Yh$_u*qTq%y7FZR^IT#n3~5I;9%^6ST?_w8Ub=5jLCi|USCRi9xwyz0OlKP|<?&ST=<E*)!y'
    'xJf{D0icJ<kylWf4r;oHy4rQlqp0Y|G^bK}QF0)s@`nEwaEM2%^SO1Hj_&><9AAAJiir7V>8uhT^hK3ubbK#A8v5*8;agF1=y~bq'
    'M<02T|Bs%^;K6ZC$k9Wui5#u-bqM1+cbpV`7rpDvb5cywTjrqYfAZtGZ7qjSy$P~Faf0--Y<%(R^Kesh9R1sVUQ_@6;5i-D5&HD}'
    '!+FNrfp_fx3NF|61`%9WL^p$f{kBa7rW<|rcj@8js_EArUWY=*)^$=+{OXD=Ejdu}x)wbKeJ_1g)dW+O8_#qU9vA0q6DzrxdX{=='
    's)D|%V)Ry7!tv8vIF)0;CmLP9jwh?%zD6NvDmWkAC_P)%YKw|Y|FT|Y={(e<W2CIcIZ&{#<Wd)Iir_rFqSWB<-1SG@zF$NaML>Pp'
    'qwA!+(-jDtW6J1y+?5Qpmg>67!&P1;Yi|ASM#xn~)m*shbMpK<QScIWb<b4|4sv)SJ`fGt-mcITS@a6{+$unp3z}7uX83ZB7i*;Q'
    'Pz5w*f{dyOUaf9_-!4}O>@WY66sJ1%HNoX}DpZ~FJ>IV%2kRlJ#MV9bB~d^fQ_ouAq`YeP??0$NZq3x2=Eu4<x@`(Bx;2VMJODkU'
    'sj5N0sFGjjsuVVSm&#>P3ZbmmanKj9RJQuadQiOBwv+<(2W9IzdBIyX*>*w8g&?}<Dl1~U&{rC+u6g!#4mrE&?z`Muz7Z7Qj@M0b'
    'f6r8PwL3klL`@m4h}LVOi!QMOq)Nuqed$$OD&MM$*gS{JbJD|A^`$dAd<XT{>QL&amidpsqMW&?y{~?&4)*3Y97PDq*rBdOVeL5n'
    '-qUsOE`|8gP0@i;lv7Y#fB04UgARXh3MhJ3Q_Vo57gASK$b72e9DF!iilD?3olB)f&K9*Tw?pAyCy!62?#Shw3eW1gcearxh631i'
    'u9}PD;G-L}9CP#zM<R>fi1LTh)=^EQ$VOmOMLtvk>Z0oI@`z}07A5LUAyrg$>%S>2yOQl6weeP2#YIpg(f!f8smebp<n(XylvGgi'
    '+Vy++JDTsE{w|#o9VrEWUH+}7qhH0@QpD$8Odq>VH%V7{Nc7}##Z7(QLl;NldM8S`__(gtzpT>4kE=N761sfx)LoTicjem3|D>;b'
    '9b6r={j?Mt=S;yjl#tY~>G5C68!iNoP~oC->;IJqsZ^nZQ<jTM{k1xY%IJz7)N+6C3p;Q5a3u66>Jwe{-nio2-loj0Zcitt(Iqw~'
    'Xp(N>r0MA>HR??fyYvC8pQg^$U5;Y9vi&+AG}Rx}O1jRJO0?FO!defx`g$s2(VKJ*M&A@odZlH!&m8*0b|<30&`p?r;NUg%nugC+'
    'yRN5|H1GaR+ti3ClOv&c+4cGc5k<$UEm{EezE{6o*X34<u%&ORx8tUHE9q0X=<}+mP~qHeatBS^<k+gGTP|Q4tu2>Cm)w`WEA=~*'
    ')T-XkwOG}GbVIovTU8@p*RMLd(<epW?IaE>%sLpUoM}1z${qJrc{&YuQ2WlFqE5$K1(Avuy{ofR)TzQ(IMF@pHF^Hy2Ooi$P^(Vp'
    'IOw*`0VccZ<t#!Y8BBM|{gQIC{;(;zbz^y0UD0=PR}>324@>154KxXMQrL95?{SwCEeYS%O9Kg@605^Sr~c;QR-H7S$EC_^r-Lff'
    '`BxEY_H<77y05G$HV6){eJ}c*-(}tf@KIV+(SH1$?>+RD3YvdCCuQNHYdY1*wcs=#KWKlyK&not>J<W&-{}RG`$~oP%@gDCEJtI7'
    'Tzz)Fpd;*b4%5NY`}v}?;0!9!-8o#RsOwW+Bk<_x=~zt=4Ze=v#@MPmd4>v2t^hY38)-tpd0wX%9Z#O6dhO|SuDo20#pn!GM-{d8'
    'iv7;?Ix>E8dNwt&Sb2lntL9ek$+1nRP$yMQcn8z@m(h$_l%e!0c1@g7MjK8QicYT}{C-Rd7J5(mj{1|DKI*y!UP1TGGgM|M&S>g`'
    'NT@t7g|y-GSNLSUqN=(!Efk-65^WWsbbNhT)d_40JC%~Ts(NRQ%%*<eF83HZPnGAS{HfDTs3t|bB~y=gbJoB&xRFYzj>BsDn`@(1'
    '&vDYH#G1}p9q3R0WTy(GCRRGoNxBqeHBEQ}E!6c?@LYj8bO~x#aNe!C68-yg(h=xNw~N@T$D?=66YDzLdg^MkR3ubzIJ!x-XVX1}'
    'ew}IvJu=WVof;jnJN;QH0t%oBkF2Y%AJqZ*5WJ`_dok9bT1Z7FScjfR<CK-++CdLZ&+!n~2ouGfQPDFu;-^X>>5LxBU8_Qp0kXzj'
    '@!ip*+BsBu2)gUanhIPlURCNSXeh|2qObE6i^_0*Ud0#v#}D$b^d$f5|58EzdmV}rg6X2l*E`qkQHoKBQx=@d@Sb{%K~mQTsc!NW'
    'U#J^874e{n(s@#d;t7+~7VUK-ZC{`+J?lw&?S9sytl=_iS3uqz$_{{f=+%jss!8>~ZEDifuJbi5%QCX6vZy)1UH%#%=;4E>JlBmO'
    '>+QF{x_DYmor1$xb|}>UO~2dHU6LwB^+EswD9*|I{K<E0Cw~2VlSWmY(Q6KpGmR#PQ6YN~yA{;?a^pFi@04<;^PNrw$6VD(BW$bj'
    '8pI${FW;-AI8ZgcmMb3jCkD-(bdIV3r#dC}1SFMm4ztQF<E6X<`ka(ElkwIoh1AH6-*j6#1Nt`#aY6T&=%JVjaMdY(Rb~%DgyKsJ'
    '#HSLfj09JmMZTxnf#WZ_)!sG(6FTe$bUM`*t(L-FQkEi>(Z9~^=Pp+rti2p~XVSKg+&PKBGdW+OG>~@#%Ae`H+XF}Swo@Q0x_;yS'
    '`ancGo#}08NL7kzSN&zGk%tip#SWDtdfgODL7_i>@R8${cEDJ!kq2E$APD{Q)dyH<!dYFPdibGnX`I#lJBQ0Rg*MKH{yRCY`{;7B'
    '0P*>^XtM_RV56UW&q~kJm7Oj~N<78wnMP2he^9V?jC4(g9KH#2)E9zVFbcg2SGUB`eO{NP$FHle&v2IyWziv}b-UyZKc7;mAaohm'
    '${qXs!AEZAx~G|;1y$rCwd&$5I$mH0J%1h_;dbc&T!&kp#UdE!0f1x>ZmIfI|0>xiIxPBv%_G$n0neI7irnBgg@suSZt5O!>U4dM'
    '8+@IXVT>ugD3c8udp@f2?{G7rys~q$HaDJ*>&;p8uPE3h4y%VIp{V^lbU7q<pm#F&p_Bd>0bRFylM6<$c{LqMjH(!v@5=tsX{n-7'
    'e59mNAni07{Hf`a^Z8tjwC$ciF((1iI03g)-{!c(T54t0*-rpKx;2HKzG<mxdJZX!U!!Ni<55OZ2W|j<E(10y^3?z5E@p72f{Dth'
    'f}Ksbo$8RL0YKfBWb_+6r!vP&Zl)%ACG{aw(<qEomV{h=BY@%lzZ<ks)KLcYG!(V?0jOqbBTW&10gx1fqyEJS17kNb<)&T_g1VmX'
    'uL5wiW3aU3qUqB!Md8LNc~D35*el=P*dc!au9->G;NFB{C8cH*aK74x=(5wJtI^qx6N)VfL2aH?gQVPFgxU#=&CE-GP(5q3QZZ{?'
    'j}}#O(Jj$``VAgpCI?)}v*cBFMricmifY%h#z_>>S1ELy`X(ojk3r%a-l?&!{z_^?41r%}shoWs;pSR|VT%D^#n#dCx@j_eElPC+'
    ')nhopI@JZW{ylqAuu=0gi8a$NzNt{4SfLkom{kTCDrFXz@#`NysGq4w51H0bf6x(Uvce@M=`BzMWl&zWPPG4^^FX&q>2i-ztg_YG'
    'k&=pA!48g+_y)dU$U(`qc-Y{oZ5lM!<@MZiEbgOzck%Fo;Z<5o6=nf3^`V;4JAgbVE7kt>74`EQ{!{h15@#1NR<Un+K+F3%OI?>r'
    'hz2gh<O#%^fzK+zmI;*#aumw_6VFo4(P=p(pGM9!UaMemnE-WadY+tu>2&PWHK%HxG}Ws)O6a13jDwp@12P~a!$DN7M0Yx(syC9='
    '+KZxVREG=oDjpUNKncUVTwu@ZbkY5Mn3MMmcl8Pq4mSfXLk{%~t8dZ}DW5Q<sI!5|e0~m(SkYO5=KgO22jf%y3)_`?9i-2iDXH7+'
    'UR9s~cPfPLVkn!rGW|9Lg9?^+!F;YqysLUzpu)%Cp7PFc`F4&9UQ#tJ?Mwl;8MElNZ7>5*f%U!J6kR}iXXs1>t_st{jWbS~BQ*!t'
    'aYO&a8}KfbBfp3n)RIQ%I!Aq7qJmPf4i>MDDL_7_Itz}tp6YdY(2MWGV{1bfd~$s&W8dYH@AwKes5)Rm^UJ-{qY?FJ+k<WOJ?mu3'
    'uS$5OKpU<)^z5#oO!Y1q!m5Q>hGiQi>Fbqm%`8cw;pWAIBq|Oo#*FJJu3mgoIo!eht1kwyxZV~BW=g&vKd7IPRWYYb!;WBy!5>%4'
    'M5ua*ItYq0z<{!UH-f9K$#ZHjjcZ)gN6<~&ee;4oJp-o#+STAL<1Cl%RbOctuiDK|5vYFp!>M5Od*9U5u4Z{-jvcxr<FD55O%A4i'
    'QmBsGZ)PNJ;xC_XF@E9Y>Z9l*=I&N^?`jD1(Hk{2KjFH#98kC7icA?!uXzP~GoHX-sK5sp3%kbPh|16E)=MYqy|3HXWA-%9>^T4~'
    '6;`PMYKLBzZ?$gZbW5XOohhz9Ubr7h&#nHZP0{Q9L+RAP)b1a;MvCo5?gwa7pAXE^T}2dng6STuQ7Vxl);6*+m_rfZHg$2!Fwjg6'
    'o$|Kw{b{xhq^iSnj)Icp)@1dJHPrhY--V@1!An*8zAbp9ff5h?ivOU`y7PysAeA;v#lyP^2lft=(iULz0$HnYUi5)3=}EA<uZz5W'
    'Lf`oK$;V9?<@<^B?Ypp<J`<GyB94)rJS62#a>$_$Qtb@3Q7?0y!I^DT9H;j{8_EhMH?6Cu#vjzScRw1P!Ly7(4j>KA2vObDBoi{T'
    'g7!%A*V79IseBV=54eqwkFplf0C(XA!qlO@!HyPyyFO?=4>>7J-PL}N<{JFRPkyjBu0A&Cy8_kK$87rAD(%jJs6(CpwYR5w5PR0-'
    'q$t>`s)Pnd*|5I`0NlO`A7A}2Y+0Z>COmM4CSb_eye{Ma3uwL?4k4-cWl;0Bbs14rkZ`>~qOB*TgP!M-w+#&)@^oz;IjK?t3nMM5'
    'E#YA@#8xS4-pu8qrcfFBs6L=z9&`St7~PY97`E9u<S)j?lNrsV^=hgyRH+=#f(~c8XWj}tO0%n^W0IUc;PFtqa+5O*qttYnTKISb'
    'PBRTCwT`bE$H0fC>WT!f9%DObs79ZzrzWfthmi$C(78p2&1Xd|T40ZQLJ(BkRM2}xDQ{TefkA<98!xUZ=qhB*k|z1;#W6nXyvTq3'
    '<0l_CY8l!AIh~u>4XQPG%MFyW>G2HSC#Ows6R4HTrRcZ;T~%)Kp%+XZg=0#8>M;XF$`NO~`qFzGdA9^g#U!RY9*8K)881=IRt*<5'
    'sE>Z>y#}aOEV|Afljg!b%6cqIMXl|@kN(M2Xhm=9j#r)PM7Blts3~Lg*HDtE8r|KXqFOCwX!XV*tur94ZZLF8=EOe=)Xpem&>iV+'
    '%&rb+5PbA)p)a&<oREOE4T2SOI`q)hUo%VEFdD$wbQHU>Rth=pfhUPnXH73Dpc80iY8=%f-w=~?@t)ubflaF0{x(+FKL=oeNX%RG'
    '15(=<s!#`Pb84D{uEHbJYaGBJ@iLO>ZUC~{zw#fST#h|F<bpc88@2GRV8;6oT`(-jxnMt4h;WK20BNLW#T1%zJyIr$9Kp(*XZz%P'
    'U`hcvd817IHy_looQAM@0-%tg`(ljs7Kbh0R=o`}0U$zky{Ct39GR!PR1=zrr4Xn4?U@tPcVj6Xf`BPS71YQ@{tkaKV3Xd}FM`aX'
    'H)uSm2`;AZKKSOFlClwGU<jQ+^IU1pgYL<uwW;I^Q`!^}Dvb<r&)7TljdAe-mciRUkZ!p-`X83t72Q8iE2%#C0u-{hB*k)1-L9x&'
    'F^s0S()<$zg4WX4!C-YhBCGi3sAl~J7#An9(WRzCylT=p(@NvTz(HZWtEks_%Dy>2e)3trF4_!c6H`?ibtWN3bp-vD9@7AZ_0$J6'
    's(3D6!o&2qLe~|9TJKZsribvCK14T157D@-a_0`0RX5nPFv=VU8_2cB3Km>3*L!-dF8mZW!qQe$^x&8od<advLXVWyAEuq*mIW$M'
    '?=v4+tP>?0$$2Je=@fwEka^zBQDGgVP;ut^Vu82m7Io^<u|7l0U;?&PP<)NzvQ)8V!h2%?ODtCXe$9zFgtZVD7$IGwOUv*OIebG5'
    '+&W|&Z%_qwcV>Q`>)@<--$xn(WqyO%a1yV-f~P}dcNNSY?+j0!OoR?x1cAjVYK(9RUsqa8L~-RzTRB>QMkQ_d3OcFvI;yJ2fhfQX'
    'ubLsQhjyXmDdsaE<}s-#BU|ZLnSc2aiBt0H`**HzW!Bl#XB1Dz)GWe8Zu~XP`9$TwHha8!*ta@3gX%)+=L2Y1%rDj3XZ|(mHu$>+'
    'Gu2_nkgnM_y$;f(H3BN`uBl;QS$<LqbNtO*#t~%zXC<WE-AVwq?mhyZWPmoBv|shfhYVKW$AJ>QM08bRQH_y#msuD@PSp#p#lSoC'
    '5%pDR5daHxHp()U3~ZMZ!BGG$(Va48-<i;u)T~o%;^oml>!>qLA#u)ux!~HY7V}hh_?HZ)Lm*QZJHunynSiLAOq;>SD}QMi=%hO|'
    'n6RZ(`m2sq{DVSnM5RJbqZ?$to}=;clltQ!0l4sN^fn<^ZJrXppqQTuV|(}tmHQ~)P;LU$Dok#Z<P`vKDAjtnhq^}Z@i;oGci7xZ'
    '(5{x675wbD51nQ)t4Mpu$zbf`Ud!k`p;ec4xU-^e<O+d}xnCwhd8VO9@Inn~>diFBGjy7(=ZNWln+;O%pm_FkDGL~eTN8CfDGY7Z'
    '%Q5)-S<P%>TW_XpQ^|Iolin3H<_39#lBk)71RqQ{NY{6TsagSTscL8yHi`Fq&kf-LOcToLD8`=jD1zi<FsPItz)7vpG%IS;WSX|U'
    '9xJXuk$c0z8ePSj88NG+>OmwGVw1uODw4-nAk#sZQV_c4j;Zm{QB~F6DIT4ubp9Bqs=^FKn%W9RZy}X_@)guZkP(!PH*urFqNR)K'
    '4oa|F73T;K=qfzn?PmV%or(}Ht!MK`fw!CLuIkm(0DAN@Lt>x~GUvU7zO@*#H=jFr@ol*{Q~)*K9X3%*6y5%k`Uqy|{DBi}s-1e9'
    '8(!y0fb9x;<4uA_(igK$o$3Z+cb#gCmM}ii2OKX91D5CzZ}OGT6#qPln<p?oSqn?=Qb$;cfXYahQ~$<<zW`nEYp5RLxKxsxq8B8z'
    '?q<N#b&{0wyP0JdxXv9eY}S@rIv_rC_&C+B0BPUNl7YZ(eionG9p!;F9~y?!8N_nX%aCK5p}*Jehu<0Y74^UK|E_s%%H@m${(Hn~'
    '0w5ZWf=H~tdC)8a|AgOrHf6$hA__0QP009Mqlp-ZFH+5;U%)HSF&U+(7W3q!cLTP1Y1fl3^#&oDLCQ9$39Hz{U07o#vCP0X3Y9nL'
    'g3181Qz1F65La*HH_E(crxHZz_6=d{U(_es)^~h))XW7x2n<dOos4&?->1A7Uhn4*e#&q71UGCJwF)h7p2K~C6m%hmr>pv0#BT5e'
    'W_+#$uLT%p`Z7I)9d)8*QuEtQ%cOJj0buG_9&p@TMEjeNE-ujQaS?sS3L#JV$F;Mru*Vz=Vb5!!Y5-ckPY8HHd33Cm93Gua<%|My'
    'VNeu#oTABA*L8?Kj?UmJ1lI~T;Opmq{}?#W3Erq?9GGS@1gPk3+VbRmz%1o$Kv#i&NVq0~c7rmil|dnxHXXtPnx?y$h|stbRhmxR'
    'ogT8$n{oBkzMSI5jV|4l)gQ8G^AlW_YH{K`($*_9g~7mZP&|}zk*i$OVyOt#WjmYcMf6t)qB>C9+5pT<p@1{FC!MV4%RWvzU>wPY'
    'GR38#E;7j1BV_zNhZXA)kqKN62d9}k3(e0pR;Au55rt=0=_SW>9S?``xP+m*QDNwD)1kSfjjW7E$Y#v`I9Cb_mM-Y?k%|3O3Q%J+'
    'OzsPb_{gC120Y2Uq(yFV7=61uS}XSts)N(mjnq@|rv0i6G&ioBWqSb{9{FpcNp!bdA;x<04so;8K*kI7_j=wn2$?|M)4W1h(Igsu'
    'HZzkT8oJ_Dn;c!>ManZX8d|1)=~;vI0-~Ep?t#j@sc>dSVk%S|e3%f1w6S|m3zVUeR;r!vNTUFtQdD*K8S9Z+&Oq+f9Je_d`VLS6'
    '<~gYje*EAw(l9n2)OE=F^g4^<ivsd_G>%LE%#5t>?!*E}nn0tI2CQ{R;EX_sG=)NXMGQlSpew+E9OebCZ62n@B`$Fw8Z-;DJbhX)'
    'usnYn5fLk8hknTv%d)*eR&}a|qmc@yj2?k##t5zhhcy$Z47Z~zlt2jLG?+K^c+{V20d*Qr&KQJ6cw5msf6533EIA6##^K(Ou~fM*'
    'Y`78H=<Qi&Q}_kf@x&mij+HV2LjbgTDySf?GHZ7xhEiyS7eUwL78NPO1&rePH6l}7t`eHTk6g)s)j2dDu&XZ*_5=2D8Z&dOaG~8w'
    'mEap5$mbHN-}@Ew8&;nJbF?JYnpD0RNU6`lf!}ZS5MWH<>MT07WuU%-<}O0zv4olu<3iP_3=TYlCj7ags@IPkc-B)ZrCHNq{=n)q'
    'qg{=5F*i=Z{4q?BAo~s-73n@5t%Oer9l=>>yx}B7mHA{ihxbdmR=^MNjP~$)?gl-^TD3y=rDCq0Oy@p-@X7w(1pktJf{LUV?AHxI'
    '<AH8-Lg03%2}AzofdSM&Z9yQ!Ez;eq@SHBZU_UOHpC+^@$r4qehg450mC-WYRpd2TrwF>uE;#yt5uG5zHH^K3OKcsMz!40K19Y8o'
    'a);J<gWBHDZ`7aq1`M*mSV5uFB=Ts>n-b8sEZ~D`J>lV1Y|FGv-Q@7Ay4@bmf;byS(An2nH?i;zOC0uOw&WZL9mx=kicPn6sN8%q'
    'W`ZlcAGeQsv-_!Q-XH-6lj{j!pcaVtRec2zdj1AkSnH1!)AYp{0u;{o;q*=v)XJ&40+p|iXd2?7^QOMtsHP&oV`o&5smb=}#kB+?'
    '6slK+kufa5>f5~0!ek$Oj?vPzwn<OMJgtowyFZBASw>&1J*rAw(VFXnId7^m$#7X&I9L=BdVNpgclYt9V0jO4=|{TdV}u<vXmlGh'
    '4wbrGkk21{;+PCr4<r3S$7`;h`e?OGH}rMgUyqa*gcY<-R+#K%00t_IBcu^GB4k04KM4eNhc?xw!v_)%Bfv9iD(HXREd^D}46W+|'
    'sX0{YdRs-z&^OgsC=CiYHA7Z~1;aDIb1(JkPk|O-$Z>;TV^BgME&rG_R>C5@21;gFZNT%i8D&HKt^h!H`909q>iF>BFAW+{E4o}a'
    'hNv5%+JFT!9RZibj)EHH!p4TGsubI)-<ZQ6nrcRBU^8SXelf_wwaT~D3<-eRj_V$sXXKE^M0KFiPQtpZP%t>2YE6r^I(kJWoLGBX'
    'x*;2t?3tM;TPnf)%6?%n@?o~1j|P(n;Nt=qES<<<T#0VCS?vloV_D5KdIh%O?gyPaw}WY#3JM>M!9c|mewD4)uVff$wdDS|pPhUQ'
    '_kkW0yWF^;zfhe;!8o7$2m^U%23@=20KyY;r&C>5_-xeiE$lqBF>dN->(LV-B^l5xSJcpm?kC;RiXi02Pd*dM<N_!(=LjwGl1E4='
    '*^NM53;o>4=rnHwPsKxPzfNWPFvF;Ar}^qV!aQ{hA;###MkIUAG?|E&BVdr>VOqKwjyhD@kmOeQgen<FSlTBERL8EHfUSG8;?|)$'
    'eTI4rG?=PP)i+GU%8h6l1dPjoR`;Ts0RYQJcUouMfy=W-vdiGywE9EbNqix753UBC8Lb(8y#DD+x1%P@CKlEdmKxPZbNN6ed#4|m'
    '2bvTj^m&^)b6cBGtF<1tn&OY8!m&U`Z`+;)uoe7-KmwMtFrnOEj3Y@G4R6S$33A7bnXZdvDt_i0JR-zc;f6pH@v|c98>BW%vxTqD'
    'e|%0GFSD{`n&5S41A_QEkY1uK+2fD9NyV4dY#>3<n-)Gp81{&XS(DypKNfE>%I`)5$ZwDUpz*Ch5QL@3%oXUadqnPXv0INmwQlJ!'
    'eU?Es&7g}N3Rq6cB_qv%P^La#m%NQ>b_9(aK9BH_?)Xl^jW^E>9b>(}5-b8A!D83DJ*EfpPU-b&N|cK#g}6K3ZJfq!puU%f0BF8E'
    '7ebm<Qk?hdvIOPO^<-$1S}ka7*HqHeA;$X*m_=ye&v<I7k|E@F1w?v1BbGnSgc8!}4Q!&Y(RLE`88I+95Rl%mzMx)Zs9{Djd?lW-'
    '{;NJs7!#$!qA;6ieP18|Lp}d}f1ul=Ta<L2T1p%|NGw)2HA|)Ti!xY!IDzQ*?WY93!zh&byu~^Q*wsdyCOUWGO*4RLoSTx2v2Cj@'
    'dc23rViM$tgYk!@jJ_B~PVmlfd7-Gao&q?2M}X0OV!X-pHws@5L7AB4DBChwj&2enpXwPtpc{jQtrRV-S+EwAD^%))LEWqU6G0i%'
    'h%G8yAVyuM>o`-exbbw2f~}KvnLk8@Ps>W?^6c@X%0d2bi8<A+sUe0FyQ^ZP5Ii#U-yT9D)(8XZ#()bZkRRE?G$b<66HpTiA`^3K'
    '+&fhNP4|I{d(+zK1}E#0t46uOwof?pE_QYVoz3Xqn@xYCwmRC=wk#8ohw2eaz7C}KW!~L@QUY7$n#^IbmeLO91<iq>Gw3Lme?Y$m'
    'cDCleTkc$4j#)&*)E3HNw+oZXT?=alvjh{=5D(UC+L3;Dzy*vRig7uHLED-TeKtp$W<`DJ5z*Z3-a)ry#<_vkJd-zcFRCzq)%@5M'
    't<&GYS%J^}wL9e?*^j}ayD{qyRg<`q0Uj)tx+qs(f4$ZWCT9faNb{OO{`%4!a#Vs2p8`&^fwWRQ5JyJ4-SIKC;SYM@HN{!m!5ha3'
    'p<HpS$6swB?sa}XvE;JL#(m+14|KPks%j>5NcmLEj2>bh1^b2WPbdvw%eY6)HiX=9e#93zbKFYMSasrtW09U#-xln)(2H3mKcL3C'
    'bJ7)$IDQpVZVc7HBR-H(p_SoLu)7Co)m(ZPG*BC$^_Fc!n|9G*MaQZ7vh?{zLUa^H!iFAd_iR+9PN0OT9?J1$wAixgUftkv3cGQr'
    '==OoANQ>3Kw2XvkwFpVW;{gc6OgEL~`V&fb6ut1v`nLnje?6$ffF3>HAWOl2#xDw1F4(f!`ozBmqrWm%$HT`Xcci}v;R0nc$TF0P'
    'c%?Yu@>M;(A55o=$82&jj|>I0rvRP{^gm83(OW9lHus&F<+=Eq@gE@pxPI9nsCIR0HC_YQd&F8m?bu%)3>~8Lf{WvsTRJBgF!R42'
    'LK<xH*`;*D+6EQUwQbU=mdEP6)Z4AeQS7<ocP;`{Y#XN$dm5&LAuL|TA2AO?aZDY^6Dnq&#BWJyrW?t`-hvRVno4=|ju6mp@GD4r'
    'G|v<jDl%OIUbIG|yKD~o^Un#A$gNQkn?%;IN6Wdwu`eTkEFXRqWeUiY;kU!A{i5gvf~Z9HKL2>i?I+NnusGMCK<iIDwnFVKocis4'
    '392EI#1;ztvKI(%p;K45U0T=m7lR6)gi1;7Z`BxMILxf>z6qgKL_{k^{{*#h6{AeEUuj#PJqPYej}HWOt01>!v`4<L7F(YcvZrdG'
    '&TI@C-FVhfjp+l4cNjbV``sCc&#6-ef}SX+Hb)R=vF0Q+QtavcGu~#p*g`bxrnXr+;@R?n=5ReV!yb@x%aqrsC9%&+eN+Z%0AT8T'
    '#f!yffGA}h<qlnQWKIv2+Qo|vmVjmABZR%q4e0~X-lG9g&;akj-a45pU~F{s`r=^OuWr}Hk@19&sYf;N8rZaBKjenPls;3sdC=y9'
    'Z)n4I;sYDNNQqRn){J7a+BOUTd&6G>!~m+Nq16Af4DLoqwydTv%ZNYmW&rc-^<)**<SEEX9xbjOPlG=|24-hYv>4_9pB4;8DJ*>P'
    'fqR8VuZU88mXsu4_~mw~Lz)>z<{9{Ce}t3E_(qqwQPrv)+s4zOKdM~q?xUF!V~x=m64mZO4tN4r%r<-tJC^0yH0nQuwlG)QLaO41'
    '=MJcZaS>3k{T`BZW^vXqB2=>XLJT6RRpb|9v<aB%fd(HPhFZ>7M^84J-2@o9`Wh;1L|vqy4{7}?)-E2p(ku603B8YVo@ZOd7(hG<'
    'i_MEjFW8h)yDfyvt|&p0jk7C9DWxvbFI+5pN2xLxj^io;Fk{O^`NpMYhC7E(ks(Z#h$Ffq^0_B~n2H6fE+&U7q+EX`I;9;xa~~nk'
    'LAlGQ^kyhj^-cOWjd>zN>=eN%T$qOsDzhEa*lp=7BEB8-1o|ozP|T3$4?8-LHY})8xImHgt<k><ow<=*&Iz`5^fhx58r*}F`52^7'
    '$TAhVo`p1$#Iljq@+kiJr8f^2?d+%YY}XCt6^FRZWH4<Jow^Au%V&y|T3z+)(p5ht3@PLAq9g2To<<~wMuVb>r=%l%@grI2*IkRF'
    '-cs;Ixkabv2W1u}*HZuD2$2knCE&L{%*grVtm31?DYDcZ_<G{;u*nDm6hnVoZk>s8q;~aYrhsHjh|qw@bWO^$W0@I1$MF}9v|dC8'
    'X%`nTl#k19k{wbqEdh7L1Coo=qG>?551K=X#MAeu#L%{#o%t)2;4cp`qg2A7qWd|~3+q=c7$Z01%njE@k0xe`h&(pga2-An(FbYt'
    '#Gr*n+<YIN&q=p%Y25iTgpOrKs_|r<5#|U`S2oi@5|1+cMOO-0+Dw@ND#egzKq;`9=i(>FydYKY)AlEw;AC4JILeuI*Kn!Ec8;?S'
    'JOqcvRe>vf;5%9h3I1bi!HCD=9O<;<sNCkS!b`K7qq-Zy@PpEv3TiVPq`n(*q`B5aPQTwy74GXoRZlOBv8}b5_1ZtfdI@+v!)<>^'
    'SOb?jd-}B!BIaSlYgKQ1faR&jYA%NMi7zDl$dTrLdWjRNzs_WLSY;so6nVNUkHZXuqAG-NZ#<e_FcLAJL2}@K4PFuBSWx76`7zFE'
    'MtRTp0bSA~D({9_h)E6XKVE?Opt>3y?p;I9y%Zq4NJY*Q8!nWJ?v)o6_c0;F4l3nbS1US;$1+#CHFJFm{&Zw!q^pLi=waPcD)roq'
    ')cUaH7Tm0x2s1+d)-Q}{>VUZKFKiDJ0?ead;#>-oSOHUUW^hyWL>O`N*R^k2Di+Dmj#OP6-zwbA3*8I|tbQuln5kVTF1EwOixura'
    '1>G?QGpvB2MIz*^FkA4wFh+?Ky|}-$^XiGc9qkt#{3x#q+~|%RVW_SBJk@l=m-Yi)9x_m0Sr|v|c7+4<Y{JLRRf9#3W2Z09i1jsE'
    'd{-|W$AO3nP{;p9H!)LzH8e+qvK+i_xa5Q$uwE^z&+Km9U?#-ILytx&3RC+4%uZXaTdRUfz}CQkEhnCAH3SwwGBb4bVk4|6xVT$_'
    'G@SWy9Bn}!D+3OH!@awCwA^eTkaFWhPWGGMh*N#uj&@ktrr`N?|8bN6xyD-{j^Davy#r*%Fcrb%z!rKVBKTj(jeu<|H(Coya0dZt'
    'wn>1Q?o<Sm`M`k>Ah&JPO$D;-FJZS%t0xZhB;e`gnH#<md)u>^!U3MN$IH&um=D+zL{B-!)(yuuv**-dSms6m*ioLi#wyl{pE&ZZ'
    'kel1E%WCKfu?;{>2+s!xJ^1$w(mO|z+UFtSG$tG`^UY7V&42q*+M17|hS+pS%T1R`-<JcW<PKA7d*H{{GJ6JHu*<>LULj0hR1Iv>'
    'FnHpG2Q?i|H{5uhqI5n`xZCRp*??z4(8L-JY}3w^ZlhHLZ=18}+#XZi6jP7K+ZTY=U8dfC=CwBY{Icw1)aPV4e=9+PuG=j^YWfY!'
    ';wx1&crTMZiymDZ;P-I|WMYWgfyMwXEcFHv-=pYZ6^QB$^>rhDPloNCLG3L^#`Yi@Bqf}kULTOMkLi@qNjxm7I05D=)lBK}lQdI9'
    'WnN+}ULkR1X14V`4QP60o5zd~lAVn8%(`8PW-#B0#HK>Hk5x5akA4&t)sLG5-ed-3H-*h;BT2au(wg6~yEdMd+9S(qbeWNJQO#6O'
    'ZZ6bwB}+RT&TYhYG5#@ra>zSgEzxZ`fi3WkwzzKt^;n_|kJ8|>J0~GGH+WmFizuqG<`FE#eRB7$nY%XZaKFv4?wbZ+%?rh_pKpmJ'
    'Vp2@|>~-RlKMgf+?m!6LdPf~0r@wRm@u0B8)|cg~q)euN+@7ysJVKFH$sx;=<`E^&N#c%9)UTr2(uTIi<t(Op_FA6YPJ*^wuYFI7'
    ')d7?6VxAP;)pDiQ!`lUva1NhV=Wwj>^Ds~sb``Ct?dRbbmUw~cSd}~`gbt_*VRXB!<^>+4oYZkJ)1AC^GVxfd(O?w&8FM%n#H9NZ'
    'm#I*=2N9F(KfEUdjHhETS6srHiGtz`Ob@4Y*E}>M&ej+|9~<@ZYOXIPw3+MZc0FTfW(N$nV_#;?+eKXFmO$4&l+@Cra2zyPcFq}3'
    'M17bB_$ETN&LdkRO8$)*tYeKb_-|mlWx|&0>Wf|7P6>}uoU0Qcx8mstBdv5)Dru)3E&14nQy5oyaEADx2Ux*o)^*$h$nCa_^#JV{'
    'X+c@j!khFW5|WOhvO!w}=0*<+F@{hxF_3ep>9PH(=3MecO9&2t^!qWl*XdXibQ_=5wC>y1i@4N;=Y1yETu-z`8>{<-xrIl%mSuk4'
    'AthWd4`5JwG0xx)MZC;(`Pz>MLdr4j+?lC&w@xS$`8JoXnf94lnw2ijF}t{>;qF)wf?uscF-$6~4XBilHKUYZx2ja?h<*7@dEakO'
    'aK<~zi2jP-jF*>05(5?%=IO+nJ1zx^yoE*qpWJTEHy|tFf>D>B6c7pVXCmuL+Qz5~zgu8Dca5L;odfBB9@*F?!t4=>j3)#7mfhyV'
    '1m5FFty-kV6kK_M?h@jU!r?WeTb;~EKaCK&n70S|FqsliE<KhSt6)OTn?`o90QY2VpI%HdS1a~3en_cJ#k-z@SMU6YNrq}9cT)_Y'
    'v*n~hnNOolzYYd6>2Yk}s-M_h)^Ea33#TF`@c0+Mr-j!6!kJ@X^XJ3~ngx|E1Us9p=(*W3(2omFRO~`$404){=FW-y=vd-L(HB8Y'
    'H%r56el}wiivplqnf82~>6S0U@UvPl<_Oq#WbXjalaiGAayKHOPNRp=s`@v6oYRwHKdTiT623cmRKH?lPS{L#6hcz2XSj?KsHhz`'
    'f^qw(VKZ0X-Vk<~Sd*s4sMrCPB=xYo+dYJA)1Mgw{Suv!Dj~beKtgL7K<_N)z(DmxbB$N7QL1ijvdzv<_D_^+eaytf$3bU3r8V}s'
    'X(lnb{#HFAVHpU3uUk+3|MlPT0^Q^JJwiTfZoJ=B`uxd9Gf7>;(RbF-YcWX(nujIE=&CtoE87Li4EPDt<JdUg$=BotvkeV-Gpuzr'
    'jSoG}&9`_egs8;)kV>uT!iECm#hOsk&yBIMPG_$^2(3Mz#XDd72>5AXpRV9DK&n}f#9Ze5hQvJ(_uCq8SXPb`J(xo%SE#S1r^I$>'
    'T#vBR;pTG2QJ7tyaY*bw<FHNqWn+c{Q;|>2Q1p7e3DndT%OTZ2ra>8N^`W;6oYPU=)*!l=;o5Oo{Z<va!^j5)3jc`?F%42b-y28}'
    '>)}F$Yr>bo!9(=X8yqQqp^yU9hO=mK5tfMHE~W0D8%S<E-+~}egf_b`v##s70XhcBl#(Xy_MW#MX1rlIb*r@C%xt)r8=O9#)V<ql'
    'o3TPW_MlSC>n?luTk7FuPB)7w>KpHmZsI+J?&$#>u6Z%0g$hxF$z)}in!e|X!u?2Lhfm;dLdP;5ggXN$>;f)V&S@~7Oag**VwM+*'
    'lhfi=d53i}2QosWK)2rrKQLvo6D}ee>MnfwJX}6_hg%>uw(D?QP_naz?L!`!;n8Y}p^|OQt{;0j@MU{s^`k#Z9fn5XFpL?YeW5(w'
    'FM43>&uR(+UhEkPHs}jROm;-^O35YU&GgToKlw>(hTj3-1P2Bj>YQ3G7n9S|9M@a1z0jRmE^$0;#uK@|{Tto~1%&}WVxSqgE}kiK'
    'a7=#3dd2HER7bi#nYg@LDx*pvYT#)xCQodR_){s-hrD|6OU$VVGD9HFgIY7hGoZBl+Q{vF9UeQ3>~MibGPe?J5FWkaYU)vA1d!*1'
    'fdup?Zs?EMi4hJSMfcPii>!`SHr<7<RKrZhS&911k!Q!phKa1fD2`U6xMF;5vM7{BxTKWC{(z?A)nLG+leFE@0>EC?(=~gw;_Be$'
    '*h$*#)eNVP{hSCm6pe9tFBzlM2n*@bKyrQ)X5sl6nCVv8*)xki1dgAf6urB+Qh-eXF_=aO3}fi7GZ8vWo2)dX*fh&&V(@S0I;c<%'
    'xbT|H)IDDQ*KC)NS7*~iy1;99JEBGpSC`qHZQH*6Z%2&YODsm5QdWTPg0I*2YB@$BYA+Bx#^nTn0Z(z7RR7bU(il7z?9Wo6ef*^U'
    'I38H>k{s{bVP`HwaCOr0@*tYyp&%SJ#f)pc4TD##vJk$^bg3ou2Bjemw|QYJNUdEQy!)(2EmK2rX0EXlT&J063`-(=)tAiR5eupM'
    'EVsuqrBJRhnSti}iRyQI6M}=iiT<4dWzou|L063U#*~!6bJxLEFEpzR3$K8EMWfcAyawjF^<;XjMrwBV&CU3GU~c};!!st^9@=7I'
    'Fdw_WS<TA(?oi<IYDcR0<xZain1xx^RNxhp!sVJu-@^p<I-7xAKm^oYod_AQ?)Y1P7;`4t&5p3?2>{P3rG%X>pT?*nDSe0xk6<HD'
    'Fdf}y_p6&MxEM^@fWZ6zuw1a_7@9Z(>w{1GPeg{9e$93)?81*Djw5r@Ii`M2;tZXN!5J(sr{@5=Rx8fSY%?=G6B(>!D8(LPv_p87'
    'oH)-?z76Iev5myy7=#S%M0YgoES$;?yHO%F#||#rGUr>0HOhUy=1O!eU|s2PpWaUk?BS<ku73F`+2eO$Fp_L1wzk2l20k$d0?p3i'
    'PneX)4s`2&GR?Z(+r3a=Fw}wo2B1ZAnny*y7r3v^ESW&6R<ebyU|Z{v$-%nQ6Xar@B4kYywzl1=EYBxx*ozN?@wkChYNrSI*|-v%'
    '-?Zs!y_lJ-+rvktoS79i$`Y9@oRU=DfF82LMp^SW+ATZ4vB|Z2QrGyM1lzhZDbg?V+QQ@NYP$(2t*7-X-Oe9}!NKryo>uJn#N>2a'
    'TF)rZ$D;Gjt$e0Oy-}Xjw>ql@rR_IX!#A$0;01x{TB^CXM+@LNa-^MeVb`N41jZi&pJ#mgqc}NW&QN`5sW=yx^(g53S2DnlAN=f%'
    'ytuOm{Y6d4qw3yy;Z+)XDRG3+L7UH`L6voLSmyOU3tp^9=p5knmY%B<4^}W`bA&El8O`cG^qJiMaV;zOgQvPTi||jlfn%oQMgdh;'
    'j{)b3MGjM;n(d}NKrqXtW~3Ahm7nbZAl?PN9$7GTZB9g}E6kbyW$o{U@TN*TvK7G8W0>iL+Vzi_6MiGzOYp3{LMKBuw*i#}S(csb'
    'j@?+cYOTyl4)tYy8c>StOHl6~KT#}<VbsXj5>}MzV6L0-<!!?d%#+RRqwu;cSF6zFZnml3w_i>zkLlpv#}k<E+(6!|JQhvYoqo;('
    'N~H=944(2jE_P)g7m7&Z?Hr-dJ=hi1lHHlZg&>D#{n#o-Vvko8acA494xb_XLxwM4vvCI(kk!UU&9C=GFxX`OS7>?*@^GlYDr()B'
    'o|$OFvZTK4ceV_Z{qfp04`X`BClrCryV<~Ho(in2>{Xbvzr6q}MIa|A?Wcn34kS89sPi2O?(d9Fkldg6Y+Y*>(B!pcA)ObNj(CEe'
    '((H%G_;R|zg7!HfFTz?It7k2TwQe+P@NrB>q&4zRGw6>2wy$I99!xeO4~P!|`$ZgEPv7~nJHprlitv~6Dk?~>*H=6y0yUI5zrzk;'
    '8g0d0&jPr3DP8N{pzIzS4+v6#H-h}*2Om)fI2@9V8%+qZaaDp@yv<3(4(_J=Bp4p^tWKmoSl^eY-_miK9;TG#LxHuh1t>+$kVQst'
    '>UIpNZnFTTX5|Uy&J4axOv@G)+|06$MfInVD|3xq;H;Ky#4uD7X>`}qDpX$4U;YDNf^=E4MbZLYyc)wUVTZ660|j-1X5!{`FXnQG'
    'EC17n$^tS~0Ty|dZfYyq>dLy5781ZU(?KuxG59XbLq{Ae4&8>bfu&I4%;q7;!PYn?9%t0-I6TKbf5^>`pP1n(Yn=G-J52f-<}Q@F'
    'xVfGi86#Q7;6&@$ooXU|fZz5!UA9eq|NC2HA7dOqd}9FH$UipDA*Lz^_7J=ky01lN%O2O&dx1zTG~6sNGa$p1)y$d^+geqFFJO3Y'
    '33O0cccG1Cwh}`qtZPod7i=tSQh}}1+p7Bv7GR7=ZM{}Di&Vz3p+D==e4Dx9=-b%_K+oe#w7s***k#TJ#IudG-x`m9{NVEvc^?xE'
    'euKepYYH&99?~-qa)`aylWgM>bQ?8PMP>NvIu8G-o`EC;bt0>Z*b?^>=sQ+%ot8z!+pGS%ok~r5R)Lz4GJC&lNDQ6Z-Cv6y)3D}t'
    '7MeQBQCSo_kcD)WHG9+zzX`E~&TV|8E>mISy3p}PuoV`Z{u$64^O2nYV_A)b=BS~dS?um<r@l#cCg7d6v!wcxaxLI$X8de8$swIp'
    'Q}3zOd^PacY+=f<fcdLtWbekiltfsRGw!$1-hOWZcuOa*L&q;r22YzvRL!1z4czbgsozxc3@nXiW5zt=HLa+VMZ}i8Hk(7$X%lC}'
    '_>PoxWbSYl4ao=~6?Bq9)*4})sz>bUAVhjh-(o(%{uH}0^ykzJ-zpoNQ2pk6zg3jL9$CQ6s?qapLSWxMymLa7qF&IWy34DO?4mg_'
    'D8fy1@0bvL{N&?Stmf6Z5XIRT)FB29axCj=Eem+?`$e{ZeF$-6t(afx(%9I9o5#8vn>gcg<mDRn)?K!#G1qR{agtPA*a%D@Ve^)Y'
    'p~A`nlFYX|5YD@1xS+?0xN(ms+C!ia;QM9=3s}Od{qLiFz$vjZzCVcEPr@VaPa^^_C+v1zp~EGc*HYK!-`G5-Lmta85~&d7i9Rc%'
    'Ei=Nc_*~4e5gb%Z3D8dKpNHa$cM~qFgRXF(rC}_omdVR6JN3zCdl`=!x^c@MKJv!QZaBUBlD2Z97kLIL7~>w(o!CftyrPiv7%?1M'
    ';3By|TJLo?C}vNV*ee;0-(Yp>)GT)egj)zi7F0wW8cfD8#qeZ|&=6!cZ)o3eT&lASy+kSxI{xlml;{??P=(<zZp@w_0j<o46f29B'
    '5;KP*C7!H??{MfgpPQj~iC-~tNQcZ5h=t<TH7Js@lKQM<&k6(T{%z%Pah(Az@GfNUWnz-r*<Kn;gT_`$#%$a*AD+*jd?uIDz3d%q'
    '4D?A|F4GNIQG<#P5YLvz)`LPEw+qgh8Ji*lz)d{d%4&~q)x``Mwk_rDLcOO)_PB7HO~o=f?={iyHE|eVzMk-1paT0EUB5K!Pj<;5'
    '2|=4Hk7w35(A<5?klI^BvcC870BnIk4bSTR^jTnx73@l}x5#^kt!VWvT~L`>@ycX&|MimzzfG;jKp$>V_4Y(v7&=|1Wb4h_7>5iq'
    '>0IT<yCJ<JZig(<bwY02M$U%|*u9%ir}Ns5Ods3aM7(@M<^+FO@g$hxq_XnK3^Wuf9?T+ZH?XpZ{$JtrCJoQ__NyQoi>p2Z(r+t*'
    '=8v1C^>&`|AD><X)xL4XlCl~<J?pm!GQgaG2G<1f&W|^Ueu>5o*reOB6)QhuDxFJsajV%T`eN@uD}>YYfu3b4!P23uw%)1O0#%nj'
    'iYzw!rXAQmTj-A&9tXS9J?$fiA5=L0*|62Rm&a_Z(rUJD!EZ4m7B!k%0-JD~ipNTVQGZpcz=6jgk^aA%$cB?1x@Cp}oD=IwJdQ@g'
    ')plB*aYEd`S(j?1Vu`)h!F0hEb=}5l(8iAENjg;3>_nh7+iOAIwGy_Q69Yg=)9&d=ZEqFh$dpk<To8~|ytwZzEplnbw#M9|`Jh}c'
    'irI<)>N!rcF&0+EBU|0(3Yn1w$4t(cj(9P-3f~567)i#*EvmNfwinG|S*@Xfg%dX<09&G7I|S*CgFaJxaq|b&80W8IJm;EE0jh>1'
    'HoUljz|%wI)JMdRivp4b;a&k{8WXcOqboGiGYq?1Tt+ol8#e}`t2*@jHLq967m+=Y&8&9Jdjfmod42=0ctDPsX|<6ljB6&2*gV<`'
    'p=1vkW76JUo2Vm(bFUsJI-XB>W!h+j`#!^^0A2I8dP-WXoyp-h%Gu-1e}G?X)b$)2Yhgp+$iuU<f^D^HW|tOwAMElxvhRao#*I|1'
    '8pmUgIrk70VR*qF>Dx43a9uY67%Uh8Zyuy$Gq7G=Sa&4B;6|}@W`F#z76H-wu3=25S|LgFJM7+Oldu`idZ^Qd9#V<k!T_J^@m6sd'
    'h!0z_?ibgLy)4b+|1A4u&yTCSmd$clJe&74chDrnsHVffBTXB?f<UYE!la-xjkB7XfoyP9HhI7s7TW<1+J8y#J@%xVSV2qBC&CkI'
    '_Dg>FjIIfL*BC??NuVOY1ptKfM_R&QuF;xLGBsW(%&$SIn+I=fl~0`m>t`84(8mIwtv1-L%it6Y=Q4tZyKiOt>F62J4mj0jE$)~{'
    '8A9|bwFL#Vd&h$(*oHBzF@ke#v}H0k<vQ)RynW-pWuMf-u+c}Pg@(zfa`mSIg$9AugqJ0Gx`0bK_;j{aOpm)kKXC0!W6cs2Ff}ZM'
    'KSv`I_I76{&7|+YBg{>j%<ic!lecT~&;>@f3>{wQ_2FVq$F_NDSq=g=Pc^N<r@xvY%M^OjP7D?tct+wqa;$}PCPKnp@F-!h)Mr3g'
    'YDnJeSmAeq;kxvLQsKg8(xdXAR#p;nA$EU@le||oCYd+f%+~)G@_X8r#R$_L=X{7`U>j8!vzeaI%|NVq1X3t0U=Urlf_N0kikRhU'
    'Avgb~JIA@w7w@u0y;_M+ua)UR^=_yZ6e8)(j$_B3qQmlV1H8?Fg5!m3&#(M=5j`C_E?!ZSH=_cW>)udVAy#C&r5A$yYT+J%Wgh!u'
    '-(yE-R=PO7f{rzBoPyD{EOib?M)7up`r+jgi7@QRGRhC3!Xs<&IE#NvwlZQ2bj%;Kv9g!sWVl%Lm_6&)X(=bZAEP+5OhT6x5{*iP'
    'uc=>uLl?8q*h{T*P^y`r-_)rJ!Eh;^pHQbJ6KNLKwk*}Av@hmlV)&Nr*i7RlPS?BqAoEV|NdO{T7VhHOF&_)WA!a#FV3$9;i&CbZ'
    '(|93_&y%gdrWNe)d-pcK7s0zn*SH~h@K){APn9z$e2B^bp8@ZB%d*`CP8EJ{O%}rR%}Y8|UT3CwT`d~4%i3&2>EV8$CmP>7IWS*3'
    '27RBdqj__Q?dw8TV~ZJ59v;1g=#ZrcK+78Y9|@bGo+(7i{CqdINbz{Q!$fg@hXt{?&n|8*0EFAm-CDN3;N)A4Q($(V9hh8JZ~38K'
    '@FAmGvlNvrIM+<*dQITHm;;^Y%d`H4?kc0Pxw1iN7_~RNd7_k^o{Brh(A7A|!2gfUM=`cqJ$_O%k<uk*<A0AaUMWUd?_~f$64?vD'
    'eq*m=oThM6Uv<bUYN&=W1ap*zWdeAsd)p!D($+JcQlD*yxAa^9`qku_Sc9-C$D8b8$ji<L6WC}^Z7qXqmb+3t_ff3jMJsx(Qz8o7'
    '9R&t~dxXbJWBg@(d`$s^XzZ?ZRJM)yjb_%91Eg~s!T_ohFG^Cy%gZv8gWT7=teDNtpqzNR`2>Ke^_dOY428dJt;m8}iy@XROq989'
    '4^m|+uHB5MdmP(Ay2YL=G<`4fHIU0i`ii3DO<+fMXfgWBL>>d$Ff>~Xh*is!_Wb_TUgE~XNY&=i<jrE&f5dS7>$Vxiw|OEsb7Er8'
    '2%evN{?yER^%`S*TI|IZ)NIhQM;PMSO?lRU`;<Ly768%A$B=b0u?J#m0a+^Ca{7XSDhh6%NfWjg!Uu&qW*!Egd0x(jwq5n}!e80X'
    'OXncG7=;M~dti-OW75MW0|81`Fuj`FLO*44KP3ZK81LZw%@%V$7&OHK%k=EGMYA9Fl`+>eIR=Iz?(3K<|Ip;G9U7>$<Tz^cCyP{J'
    'FHP!-WmU?+Umdi{odTBUq<G-Koh2|#`i8GrTQLO3Q0j&&+`X!)+WW`}3DpiSpAiC83Go0{!1flyRqD(l9pF62o|+l6>1DAsC~r=n'
    'I~H65uOyc>16D&c1G-^snZmxeSlhZ8KH<XSWiLE${RJC^{yFohj-lY<RJDe!LqAVdCA0*4n6?QRrnWMx7!;uhK3oWO9HUL<V~gO9'
    'ae-;6l@*%TfO3q<Vk%f^Wn-~APKav^)f}Ml8uj)Z^UFdaX)w`;rCVOU317pP5jBQ;PMC1@dyl=xM{Hsa=SmqRG*3(|exr-L4S5}w'
    '{l~4Oa^TnMF4|+&1?mxFJB4D~O>J93_4_OnOUCl*7p7rQT=sSpIVWw@2Jp!Vm}baK)>-)`we5CFe+d*;IP#}@>IR9;Llig)CN?T='
    'H17yNwY{G~OvoQ>4Na>}QTzHR(|l>2n4Iw!!R%79i9KRYlpTI-DF0SZfL$N2Lu@~s%vARQxFG03C{eO%;sSg(xD7zDD8Q#_VcoF@'
    '*e-VkYn<bTz<V3b%EM9<K#V~$l9Xdp%0bvBve*F}i`M80IPHFBc%u24AujYYyVkMrVdB9_IE#qwOjmCv8$1s{L{ZQEx_yiGSt_v@'
    'cdBgJ7Sj#X&<7N5W*y>z#0|-o-9~S@)bxf;pn>^Av_Fpt%Q~F8TUgfXlrw};QMu@-*cQ5cv)GU9ZsV|oJSwJQMj!?ci`K?lDO}-U'
    'o^fIcXTbDfV<>f+YaqL=te8iaC{Oh{eE1M4HVz*&3XA8Ia#{&KI_yQ%vj_x^!|e49uGf0@tMmM3zs(wyA$@tF53J8>7!#bZ8iw)4'
    '?QFWCVl?=IclveR3uE0Gi6+aY>7C-T+8&ZUJm!CN`XZt;tPXEhK+x<|ep{>$59z_|y4G4I3~0`{2OSXSVOOu&MowouhN*rk;fxBZ'
    '()D*LZ%lES<e!;M$&+(l@m$zDX7ZSyOmEff_idtJuQ96|=TCC0;z98=MOlcBX4<(I56)%c84Q@?9Sd+id$vR2J!l39xcPoQOI}Lt'
    '9RSOkgc`*a9#^oJn8+}!anHa#yw2u_f~30>3vi{+Nui1RRd%0)`k1EI>3`j8TgvV95<a4`T~?6ko9gU64-85eo<#S~D=)Qjnbd7r'
    '_hl1CP-!%oGq6PY_|XsDKE=+k|HA~0Y_=n|f|Ffe!>S|jd^;-+d&7Yc951|Epv?>fF!}Ip0|zU`SPt5hR@0dCvZ8zM$JLtMllyYJ'
    'iVKwyX3MN$G~tD466{(Q)xxJTSkX3#){GEL_rA=X={H@a&T<YIa)RI7r{k5b!v)N>fP;sC!;NL?7i(Xa#^{!d`^k##$0l6VxZ@3R'
    'i#Hg8;FM^%+Voy7WG?phn|Az2KM5s)>r`Fx<0)6eRb4eU&)ccT-W3rSw6UVkQf`cgykbZ-C9wrQP0d%^H$SK$S88mC8!-zm#7Dbo'
    'WNV8uzAgb?3m|A55*V1a4rVXqkNae25Wf@N{ji<$Hn?=`gsrzjrN?xnZku0jz@rmBh@O~tx?uO$!{bz(AmAMDZap~$*Qmu|qi(aF'
    'X@m`eFsMW)IAiWxapupAvuA4^*1i#}7-6!yZ^lgt&wE$L`c>z7P+4^0U3CiiNb_;g=#hA^Nw6B<cW&w|+l4Z<(n7%LB=KRiVAjGX'
    'Q(Tvf+O&H*4!N~^0h!CWO-kR@RB?L#*jvwTCI^LG17UF@hx8S&0zIcm<M0k`BEF~yx5*~eUOd1#CX#ILo44kU4W4M`y2pZEp=f<5'
    'n^Rlb%Db~}3)m&|NNSTWZ?SlJoO%672M(hJ9+1)wIvGmsL43B9C%Y3d<HGzWuxd2goU*IBA}HkF;|{)0^pC2B!LB`909M~w=g`AO'
    'D{Vb;BMY=#_#{rF$l!g_BJi@`cwy#v+0+k*`sJy47F$j<3O4Xk-gcLzkZDpW?o5PdbsZKI;!PfHY3O(u3<Mod>jj$n%iY9Gdu<GI'
    'h7TgH@+yIr)6f~&xMrb;j{t+A9{*4)4K7#`!zuiH%doFj@p6Wg_Ol1Ee1NlxCI@R`^>!n~V%n1OcjQa5PB-xdj)KeI#kykSR_t;j'
    'zKj`GIP(HGm$;;szxH4P^6}T)7WYn~F$?NAAlaXOfyD=)e+_d}SBw-lSQq?r<*utMFrMzIYP9DEM?=F^=-o>I;O-g``XRPG8DBQL'
    'RG~QEmbS53%>|-qHT0I-4*i`9XJO67Om4bA!2JT?!E^xJ7-NJz&C}aBM`dd{v`!CR&eJK?87U<2Nfx1pIOu`NiEIZa&`M`hV4@#)'
    'lO-F_Ra7-Y8>)ApsIrcWGQtstr`DKfuKh!=PmcCKPJ;k245okB5yP}&aGR}fz^<s;1+X!Rn<ng9pn~0$rTgB|=OjHW>LF~DI1DP2'
    '#@|OtuS|~<cC%mADF~I_JOIUWm0}b;V5U_29-J2KFNI>A8NGy=h1Ech8=9q=24r-;ej_Bc)RP(C@Zt%A0T>^Qa(i*<w??EyBN*Tx'
    'QOr(;hK05a9q==|?=M8e%R84n*j@lUMijk;oIZKl7P;F>$)hjjv0@DlYRH}a!!<h@7HzhZ^*R|Z(M_1c3xelX66JFGqdV4koM*Tk'
    '=&$#R5vi`AMa7t7?a08WdiB6J%gDl#|M<zL)WVK9wxWPqQQ(;OZ28z3r?5`bA|*B$_AoZD2N*V3T8Mt}A;7iKq-R*TB%I=jKpdeh'
    'Io>WII0P6p*tASFwz;mDuGwh^8B;KZ^;|eH(zBodR*DA#$YR}oqug*VYF?M?WUUkZ8x{n+8G(4X4t?5tZJ?0wZbtBGcg?KpIW23('
    '!YHVI-?qRaZ4xUM%_<csYgdCVKwkA24Wma=tlcWi;$YLS=V&a-Fo_7TK3lW41uQg*E`?0GvFHYQVI*xUY8<`o6*UgV)Nhi4Hw!zu'
    'OM44VX2-YEcAqWAO@x(UOgqC`!ixeDLE!upme_=WD9Fh`uj%!bShUm5!Qf@6Vqr;jVu96*gJ;y}*ZVEJbZFA5TwD&0ZH-3=?7mjV'
    'z45Ds89a7#Z2p*!?9`;A8`>336^{<*;+{CW%oNa#dj^^p7wGn?6L&*w48EaV&Hft*d{fE8wl!k82?VIEnc|-Kx-swbngNyD-PwhD'
    'Fl&~*Oy0|*=zeUs3RYo$2Vuc>t9Z(BA@D|vDxH14J{|jb!`3Ckk4)N5TumyPbC>xXDo2hRx0{12e8{aW{N>}1SWK=u?OTGr8T~Q0'
    'X)!-%2>_l2jO;A{WfIM5)8p+1|N6&IJ~^#~NA`poq#GCU-t@CO;at+hjf_b82`Wj@Zzox}jO@9rFsN*#%6KG;sj*>ziWfs{@ZEX8'
    'HxXR>Je(+QyPtyLg;Tt|7-Gv`LVgdhh#W0@Ot4h({aK74UAy9d;VjI)^luG1Q$x$R(phnd5DqYTi2JjeK6Xi9<@K6oi?Avd3U&M('
    'NqQDT^^I?&_dde@wMKr^IUd-lJqCf_=JE7M&3a=9@Hfkc&#dEn3BmN_enav##HsgcO%8N+cSH;az5bj&;f$CbL{Ez;i5JE2OIBp|'
    'J=mn_K0Y3bx+X5_p9#maR|5iEB@|3gJzMCcVe{^3xD>heVV~=_Dw1Aev8%=l$X(}ecC20;)PJ&whPQb!P9LUBl0TKWn<ZD+GvRtV'
    'jcsBn+GD5L=0gy?Od(D=&oZUK{PSwV35T1)n2DW-`G6pvcsH+EQP;xQ&6E7%ZQ%?Cydegb?RG{a{`;<bGj{fO7@_1z-Dif~0Z^jH'
    'QS$tiY0fqz(!4)-un~Jv`y&f3-hhOu@-PNPmBMIxBjk0C?4A?GX@C0GL6mB6AJA8mkpb(dVcaVBn_Y2CyI05^EB4NBp6=G(IjON='
    '*HOm`F^ZCtm~&#Mi8DsFl;eMpdbYigbIrs;@JKh~=CKY(gK2FbSkt7b(i=f1AN1+tC!bg@O+8D4DQDH(+!(MH8=R>HfP+vxEz-eh'
    '@x8;eH#64_E)r-H^{X=yTm0gQz3hcfwPVXOTKLE<*v;rUHLOq@ZC;DNm{YTo))D7EF?#GXh&U^vN9JN9JASZybI3U-zyeCf`VGM-'
    'mrPupu`6)ywlEV1xNU+KQW`yS=)p5f+t`cC_5}UvwtkEV$XL*krk~WL4B$zT&DgDBL7+wcC}P>IXi?Mg_aUFL+w){bU`2(EJf_|4'
    '*vesJWLBB@VA;*md)(L+;Jy-{E!s(FzP?9=&DzGT7=zeTvyUhaU>gQn@Va;c&TZL0#RSSV?PYQL3pjxIx(X~V^I=tFxGbvj!er+$'
    'Hf&@10sHq^1=9Q0YVY_TC=4?I@TTGAuozq7a>n6v<uk4sF4Tnl&t^d!7CY`Z0(LWThsD0t+h;c|c+JA@nVNWdX)c@ZSVjVA5n6!z'
    'Au^LA4*0D{kvQ(V6y7z=K5ApvgE(NhLM-^jJ2or67&vF=Zx#o^J(^FiUr=hY8!d!L82t@e56w%&`8IW^ZQHRqer8rwcPj~hEQT4L'
    'dFc1C4rbXf6sHlod-o3vCSqvDe7D;gxG;~6mKL9I&r2M4jEW-0Y0hXkD~A|+ZQOqBpEFuCP|aKKkZXAB9Kp_}gD^@hfjZWA6R<o#'
    'S^oP?e#>G=N`UMfs%ypI05SA0Evx#p7Jx39CBvQ$fR{QYv4MH}yZVU5O%bl}Orhhso0s4KrCqyj+NS83VaQa{!PVXV@$OYuf06E#'
    '5#A0t(|arMJiO6qSRdrI5!Se2W#icF#q60pryf+EOFgKo`W>6md+J;jELoiIHx@mH*W0+??%4Q-DDcz~+l!AMe8gr9eXpSryf%lM'
    '>S10OluM|V&3F>ax9l`rosvO10kTwaZ-WVZ<bb}&yB;b71%h^AA*ka-s2cS6_97aHpSaB}GA`aRz}T6bf^<9L;32WYFOPX5{%9py'
    'r`~fs^w!?g8avn*oLM8nKvHKdG+0<y`WBeIKYN3Uwz43AjSG!?ZCzZ8<364f3$_p=zzd!rgW>PdhQ%oaK82f5>!Pz#r<hqqu<j%('
    'WORFEceAehhBQ2@z2m($al5+$Z7rV8l@`|?+~0%uT^mV49J$Xmn%xv0-6DDO=$4AjeZupe0ql$S41BWyn|PQ#Y#b*EosNl0+{U)u'
    'n+bGY?s&PKmJS5?dV7zf4Q})lV)mT>@t~-2w+YmW5_SRXHpr_sjq`J|a|SC0SA?cs`~m-gwN+00dXeO-YVpZRGBf^ZgES)I{q{O&'
    'S>MoO)d|{AjZ`6s;b<|kk3o$02O{d5Ua*JnNZf!M)h+=C&xiQ*-~{Bre0=%-dwUSV=L7-o+x0^+9dGVs^JahtGoSdnwDaeUPZ+M6'
    'O+KwPKvTpi^bGwhPjJ!4#!!=$k@i_!wycNux3`znviRN^sbY%mejDFI`(d!tz||nEsOa6+e!Am1{D<Xjx2|(`{C!nStnmz-&?c5Z'
    '2+vXNV0May1CHxq#}GvIPr(nc<Jad8KBEphGzx`c?DtK-J>K?EGI~UH(&Fc<^zE>-(Y~37MY}yCqVq}-&u&?zM@<O5=43(6d0;TL'
    'Yxf$1l?lEA-gaxngmukVjTq{iEi_^b54PuWY{oVb7Btvd8*9I}%5J7iM3nzcjYVvSmr2JA!hnuTXD>KAudo6BGf!`pxMk^K@`!PS'
    't3UcsZqHsO68Ym<WMaEZ&?-Y4-mk_=nSxI@p$B8GF}|6MGfrXUo@ZHK?P3CV5&uzB7eS{9)-4|AFaehJS;mlfSZtD;F{C{8x$q&7'
    '<||^UWDhK?tYaH8Q#eBliRJo;1{dS5Ur8)`%h)W=C<xEGtqV3PL%uGXIF89>$|4AZXs%|li_Gv&i58FAh06?$gM}{MZO|;(M&4Vz'
    'qIbvolx$<|&Y^RRT)D<71%ZTDK~6WixUd&f1SxonB?go}KKzR%MA`YwKqZ+;Wz=72ux&0`^y}AuX?AfzEsRvM_XGb5<-%h>kAM8&'
    '6AgyY=1m$<Utu$N5ePcE>z#Mhjqyb0QABH?gI2<W*%)TEsM@k>GrMg(WNBD1daD7<ve;N5pGEI9Z(L>IMVe?9Pb4H9&7xF$t=d(F'
    '-<xfTHzn}(bjfW4A}*(=`j)Vw>iv~N$~OBLQZ<@cgqc1Tth{i~jjq=AQ##wgw#*X|Os=R(+1Bcai#w31yw%<xY{D`1jN3DZ>8}X-'
    'o}!)p%?W1R7O(-<raMa*Yt$}Edrz0*&0MgMtr9bq75`e+A-SCC9HX6f>Ga<y_$Pa$&b4U*h<I^RsB>luvtt?zwm{vZ02X9blV3V('
    '8{e>g^GTVeE{yn07eT`74ZO{=H$U7^vUTLiAY|odPJ4(2+015MTKpv~=Vw=dH3c|P#INi`I07b|SB;b7_v|@{OJNJq(;pwNQ*Azs'
    '&NeW-L81`8h4VWbCcG#!i-Ei!Kly|_O%!4rGI9Puf@u8`AY7PZ&ua}<P`Q~Qu(&+vRQ-#I*-hGTS8y#tpnPL3GUXdqamFGTfS%wr'
    'u1HvM9Xa4C*H9<>J6j}_FKVJu_71y|5iQFNvX%`TB)#;EVU>>gnz4vM5uD*(3|#XhZu^IfAKeyPN-5e|YseyP);gi&!<UH(D-Wq}'
    '3+)J_b$Yc63nZI`9qsF2PK__}^lBy`NZr>3g9<@I1p_PW5S#Xprn?^@>9MR~XV-xWE*lKdPI~)iOguqXzYS-MIs*%-gWP6CEc;PE'
    'OZ#bn2Y<TbK6~>NJ)L9IRQPBVkH1=wsc(Z=$~Z1YC;XX0ojTeS$jPA~7O+p0Vf6`_8SBzKd$4w6E@M8b5M765>a}<xv8l4V4W^)J'
    'Tdw0Jh{hji!lC<4YtoskdW4_{RwKFo%exF~erMDb6C4a(`g$q#dBo_f*6(Y?BJYW7@>y9KuJ@pR7CK!Vp&LM_9+cYaOr?MP<TJ~Z'
    '3lrB)`g_{R5;8Ka59EGBdislZ7#+-3T-#GMhAY&LN8QYzS00*I!uGe#8lcap3kh?3ebh0`3vb?}-`y=CgI|X<$?(*S<uQw{&FRp4'
    'v4ogWYxEQ>A@u!VokM5~AWC1>)!s*@QauO<6pYGS<ftswF2jrVW(hZ&0AAQC62MXNgO9W8m2&cjl%ExBS>mxU@UY<p&hgmYqXVS('
    'p($86_S-$M#1rUb4sA~h)Dtz^rKV{>Bxb_O^-3AcVTe2hpG?y_ZP{<v8iYT?c375xVQ*rew^L#uXc}`f<T<;61V$q}t9W+{y@P#I'
    'SuMJ<v=~?O9R$tcdq#{s^UJ>ersJQsz8N7e<47>Q=d`lOgT<|G?ST1Vza58}0wYsj$TgQmZc&0TMOmK3c$hyaG8v_NlRET5FA}&;'
    'pUBy<)>AlYDGS)222Brs9}tNp2=v^~HJB-D<PV2wP6+5sEUy<Yuyx9#ZtnoeSwqHWblqDnJUM_*)_#+^n->ENKBrQgJ-!TXWBxG_'
    'YIxP1F4SA)<`GE?)i~?WJ9Al&YAa?8PZ<hl;#v-EEP=(1rmey}&G6uOlEuCvccx@6_#e<g_w`EEfkxac2l_A*eDOH<N>${W&PUJ}'
    'J-cVm%kv%T-~P2o?bu(a@ORVt;55PwDbLo5)hZlUU5yh%b--)yl4Mx)!v|WS?~mPYNklHsxM&;GO$cgc0l;{-#tB^RuFRNh`GFh1'
    '%9PfFjklH=6UI2>+o3+s4b|sN-3Nbx$@V(k+hA#5s}m06a62LB(T4P}ePcYax2ek?Ea!~aF-CTXMOKi_kTZP|z<PaadWXH&HbdIw'
    'IX?Ds9W5b4HvXDkPG~9gWJsG>oaJAa?HQjD2dykC+hB+tQtxCZIAP3dH0SY~N2-r1z#0?=ob`!rtpUhNOvL>b$@Q#J$HOjr)W^2X'
    'tA*ecQgF0J=5hLz$NV%naK?8LKs#bXlJQ|HHY#(Yh>qF?vHgSX`}<66@raMYiA`m>Rn5CeSaZN~NbhLeaG^I#I(J;+RLGEyJI*O_'
    'cS6L(@&^b(-cJ7a2Sy`XATh$=)4|khp7y8R-jlMsg&EEManlD&J&fT3K5X5wi(pbz9@)ezDxvjpJFG?Sn5&qih%m!Dq1l&`Wq|DE'
    'F@~eVou2dLhym`zI*!#zGkML<BaoD92}>Ao+{Lt42-*W-{stOvu)&KBwe)y#J9F{rs;}2f=`P;9Bq$WTCnj*i@&V^1K!PkCYBSt<'
    '=+Rkvs55LL(XH;R=>eU`TeUqlV}NC17ut=Pez9d=Eem%Xco^aP>(Yq%TOPZ=e8rh3hOIUcb!n{k^L?ctC}Jpv0Y+S55e8A#lMH=G'
    'up!XlBNfL>#0nu)_35EM4vE=_)>AT+#I68d2@>?-WmJl^!-ls?RW{mZbyFlr`kO;l!-m>J1-<9;2w25+lqr#|#QHc=TixIqr2#?@'
    'D-CWsdeFLV__wi#$V)d#U$%_Hh|+Uc<Z@<t>KD|wOj6bPeAM$RmX-3aUmI_BqqemLPqnm-wsG2Nt%el-Jg^X)9QXM6^iSLUm{Qq2'
    'e!SV}avW1U6d0k`AiZqf#NF5!U|+!NG>+L~e9v(9O^*rS@AFZ;gUS=z6X?Z-S)iAUigcJyusMjC?|9(7$xR&R;eLZDU+q>Z`?0~D'
    'WbH8)^6kYC9GnUZVa`wY6dKC%_G_|-V_O|bv8LAo%$b<jjWv7Ef!CXpL%sO9o143`=`conmxmnKih*nv%i|6LZ{*mR6A)%^JBvWq'
    'V6?}(X>KJ5S0>Omeqx#No#)v`qrK##Gvcu8%-EqXdfe~g7})ezl?;{2@IvuD-V$L954?=U3ofR~J{Yv#WXkL0Ypn>-`)J|wdJ{DO'
    'FWajO&yibz=MD;eODuryplvoAM~@KNsAJ&sgtn<BD1M69exKGk?pUWoTywGshYKj87=`KS&<uB<7-RdC&@2Nu4i5|%>2L5QqvLKc'
    'z4))Q<u}T|rgx8EhDFIHz<shrFvc!k$XxJTFtZonJDR1`0Va#bdy{#<_qdIHUUr$Y5cjDj$5_vea(XDtIUHfXB0Vu`1K%XXadt6#'
    'KzVzy<tEd3dkbQ7;Yh;d<$9CQ4LAVXs6`y=o#YH*_j#*y6yJX3k;kWJQM5k&H7}s0l4JK0NW7(1Xuq%*Y6j*Ql$37>Am|wwJO#~{'
    'ABK&cz`yLn#DOr%Uyqk?6Me8N`?8$?U=G7Y1VB`5aByAf&RjAxyfPB$_R(;IRh%fMm&P%MWlxyFDn;-9eMFe-7Wi;A6dZop9uDzY'
    '|Az{ceU`;JEk~rO>wpK_Ep_=65h!zUv}Z~Ryiy74IA{Ba;N0$|Q)G;Ad~23jUg$kuZ|;d(3B0KWI1b^E<?c25Z6ZGaUyDttUT8m@'
    'kbXRQ88%^0kL_L5R<*r#D9WncMLn~lm`;uv7*D@GUXwU-+_yD!-(TgL(*H!oT^FqE?cF^EY4f1HVejFAn`SOY^%{FjQo7m6U<nHQ'
    'vDl{(8OL0N4$)(C6qu<lqppT=SyW}k<H<62Z$fF|jFqVOl*f)m`D$cRtD;j0&=04>aRi}w78bYI^uj=MSyD0LeF9<>Dhms<?v5V%'
    ')x@Q;)W<3!&M>aRQhhvoK!tt$0g(m%a5hs30=bcDiuQUF76uj=E~m%^Rx;u@sd@^&a#qp@S;NvTbqMwdaY<?3qh#Sx0TY4bOQA%9'
    '#<UF1i^u(6Znfh_%6Q9ll4?{#?))>=`_t{QxLw{etNillu=c-jeut4ZLV~_nv;4+IXISipu@wG*hxv=rM6(r4^#&zSv}5oe=snJe'
    'lewGg5r=Bu1)APwU1En)L#~;o(UWVsHw^dg`-O){MTJORzn%r&^Mtvqu=7qOGbDMj<#{Ui$x0GV2=JKMz0B+*9dSiv((0K1)dI46'
    '8obTkeUt<hw4H4ceq8iY!|V-PBqWY>_W58!G1or|64EZMD^ku&#f=pe^T6t9>r|Zg9PO~b_c)M@>}9@Q_FFPz#TwAL2F$%b&M(IC'
    'etPb7d|GDctH&k<(+SlpM$yihl(|63tnb8c%zHXu3tCA$YTVNGN$UA#Mm-DpnJv6FFZzvmq_O-HBQoac<}wl4PdM$&Dv%#GM-sr='
    'TH~ERN|vNC;j<rzi3W7u2p5qld6?`D_9=^;RB`d45W)S$V5A~LmX!7SkX%os;Tg_XkLXpG_#2P-8lr3VScs(&R(i6QDV!g)oZrcY'
    'X%3qz?=tJd3(f45Q8GTkMx2dd_yN6gME^Jxy=BYD&ojaV-6p-$mzjHxK5r4NEY51K>FNG}y1WmxqMtN1bJ|(JZD1d<5JGgE9aM}>'
    'x>q=F1w9%<edZ+Sk+n5IZpGf5z7fZCLt3zC1-b>A)}DB?J4Lhm8WLdGx^3TZ=a7a6{cgWnW2hr)KO2e*nC^H99D|HZ{jBT6)7yC4'
    '-~e(yCY6CXPB_9WlMUF=<ZcE;-O-YL>fZUzmc+Gefe$I%OLy(*VF%7?4VwDl5urG+sLAiF2<u5T#s<M?A`Ck1tt_V~txrXgO%h)C'
    '`rY>;dW4UarQ7t&SR&(;=Cq_=Rc&DfwfFTk6DtYUEuDXWNzgL<3TF0?P3y(`sZWxHgBd#waq#or5p-v;M;&?@+?=rB^^s4M)Q8<o'
    '9*;CrduS{RK8dVIg0`PQBU;R98y#=7W3Ro9`MmwX5RP8qeuGA@bMRd8WR7kZKR8cs;~x#<3P=El={MPCV^jjURbj;F2s5P%(s)dW'
    'nBMLYGcqr^v#kVjmFyVy-fxq-0S+=zy|y>rp7nN*lUWTuvzw(F1!4&wbyum7XRX+YZ|w5&U&1Dv9!wqVult1sWi1JTIW`@N9$C*l'
    'W9R~d_OI@D;|vPOv|_k_UDGMf>>j^#=$r-3@nD*oQ=%AhC?@x?Pv@sjacsM9&Eig)O}X1dr|x)LpNrb@%&ObH8$>YcvlMtsINpaX'
    'ItJl8WR;!e3nvNytoTgm@m3;J;;rZ)MPt`?JVqNmwwkyy_%O4<ibFyV>52q%qKv_wslbABzk(#i?6E@e`3;OuJ2NeJc5z}d)?FCB'
    'I<lGJ=<)kBhjrO|4QWx?FOn3y8gtIO>n^;Rs#VOxLU=sGZKCCEsmm5(>~~1Kgm2@kP={ykXc!Hp(Kgn&-sA`)^UVGc?+S@&>S)hO'
    's=mEy3g6#$b#@b2B&u^CHHw-JIuy3j91?^jIN3}u!Z9q(N*{;J*?+*S91YY2PBCX$5X1~*pDb}#+swXW8o=yFYu2xi_{(nG!b4ez'
    'k#QqzkNWPHO|{{CVKZYpaJ0A<O+y6qEUX5Ut{@NQe0}eC#Nq8k!g{>>4nLHb7hdDaU$*bgmivyqSj+-gv-qnsf>?hrv8Yx&n^keC'
    '7_Qfv+6|WB6C(>SYY%u}qrN>PRMXxT%^j@XhqaON@DG_vfG=ll=C<{nFha8OZo8R{z$W$V*gw800}8;FEAjRf(vmuiD84+?)Ed~e'
    'aTc;4Re$<UHV4URarQU~BM2CYa?{U<373Kj3f}wI3F+)6)8kw}sr)`6c<`B(!P@E<WwGt#h*ZY<u~uM@Ta9N{oD7M)_Ti`&d7tTI'
    '<JvuIeWv?yFHNQTB*{Z${mQgDTT_vz0T^JXwT&rR)6=r6$b7ujyg1%YxM3mVOyGh<X?8#A<PG8oJ??j5q^FZ5acqm!zb;nDRt^j='
    '3pr8au>!HWAi)+ndrlR#;t0DKgt*~F6CZKMUG})V<!!WVR)9EGwqwdDtb8xv#<1Djf`+yGU)~(NJki2|+l>vp>Hd2Cc-f3>4)xrj'
    'Jnv;3z`q+Cc+Jx=UkZ}JYK22p6LWAZM7X71T$5B&onq&<QRn9w;d8Sc$=wXqmL>5uE8Z#H4gd^c<N#n=>4^dF^h}GK({e@>A4kqe'
    'PxIm00J#f1Ie(>1vwEGLuZ7w%!kAIO7kng)DxH4PvW;%X>S6^QPmN_lUk-e+jD_rFcp1au$LXi{|IHckLP{d88D8z$f^XYg8WJN^'
    '(33S`!`v%FDNeKfK5Z+5tV`S}rpOkMDozQT`Lg0lU0XJS6%Uh!U7KwqJCV9LTq@?f4q;JjP6<lZY(Bl7hIvVXZ9OS^Q;=ZHm<gx_'
    'GR0e||0-LLM~uh<nrC`kI7}uo>vQ?r*vGW(^`1z=x)^3WqyNBJ!9qsVliMy;&AGUlH)d~Cx)U#M(Kndp=u=ZK$B{OvWIV@f*uXHR'
    '(ZkP}Mim1{_o15|ru49JJiC`^oB&7=*?1-4rMuduU3ifq0Lgks1Y&cxRI@LEGvdwDoRNyhnt_71grc(7>sN84s9a#ty*e*;QbC%='
    'XSjP^q5ZMCa~{-y>57#FrNe`DFS~1j(3ZUfz*)D|Rr`<kKR2s-uWjzR)-5~ZsM<Z{_4E=_{mv9`b}QRhK;i`B{lWD46)(oDjhJzt'
    '8SG1q+dbKVZ_R=ECKL>?Ee3Y+NWxc*4M?mx%|0I_ka%Z$4D_#H)4drc8}*00E|3uv#BI(Q_2Bg4CzP@spCUBF|6h4$vn$z=Ue|Zw'
    '_%?Ck??i;(MFYW-;DI0m5<IXCy}M(r9a+t~Ey4u@p4c+l>@F6IH#1L0d_Q}yEkbU(fJO7Ae&1Z&S33t<``7MNZ8%b@ZQ#+XwD|o{'
    'hDE`+$15|G;pKuGQz%fi^t;my&3p6N7p7SEp12sC5F`;5C4tEU;aKL4I%Rgh7<0_L53T?U7*@#*Y(s-o7_ZFgPp)?w`5@mRi*)lG'
    '_@aEflB-=-M*;k>BI-0PCS<i0oJfM?2qE)^YJ6#Do~-Fe_N7Ig$&X^X-Bq08h;-(-2&~x7qk)srJTso7nyDlWI7O;Xfm`-P*HTr~'
    '{h@nmNhGW*IW_f;23~Z$X_W=L(kq`Y=Ld3BH6yj$;(yLEuUZa9FkzrlA+4cd#NN`Q-eUHkA9KEPl%&m710i{OSTcJBnVgKY1e7@>'
    '#7Od)c4=1k4JuA1wC^f$FsMt{7H<-kfw65OtDGt<cKmGvJI*OG^fp9>=ofOePo#I3c$Mc$4lDCmtV?%6%&zNKIKyb8NcfiZ*@}41'
    'h_I{cPDaCJWK^FyEJy|O+ZhI&A1JHM(;iKmI}G~XNz^i)s|1G|CJ5qjd-cjLW{x+itF`m~Oj~1w_@1xqdW*XW`Q3IsTRMV)tR6HQ'
    'W4-0dQ-cElTzA<4NYB~xoNE0IqvD)T55wXgVZrC*VxRehq(-<+aVI#R{*9!@t_ykHR2&PKXIr*37cp%*09tM|;AlO;P&*nte-C10'
    '4Q+aotBFbyqkz`bRmC%7`S6q>*4b(G>0n|ATFt4esex(*aCj@1>FJ;-cL|5&^g>SKuzC#l8h}bv&?5M42>~d^BXe61^%?kKz{fdc'
    'S}=1@t(ZH$DYH2B5x|HW^~z<6`qk9XpKH3NP>mEK8)0VjOkB<vo(|?!zTamS$XNN}ow9mGCpqM!sNs7`G<@^C*d!k<n;Hg@=QAA+'
    'A_+Lkx27g@?U{bN5fkIk3j?Sq&1Q<0yo6U75UT@Vxt|L=&^Z%fIb|qaJ!No*=Sb%Bp^2BXyDtD=>gG>}40MfGIH2=s1Ham9@F3j)'
    'YAwJ*(5Z_Vv^0Rg-PsTf$M;Zch?T?Q%rUU^cXaxkAWun7*qO(jn`j}pWh*joH1#6SU?q9Ebpvq8#-#%`-S3=subQ~UKv5D9+*>Q0'
    'b3wm-=PTH|;~a(&a3d`7AE4BH`<O?RU#Vp#Igkw$&Wm`+hzqF7)zgNtEl2LijngR#Bj_9vVO21h;e>I9M%~w4R!~L+6xVMm9O7dP'
    'uj&+EPtWuBM*Q3i&^2|azow(9!IsV}{fmOzyXz(D-tk+16ZhMx5pqed{c)s{HAuGnqaufI;ue@%SXFowmvif%_lZE<h?A&=;LL!u'
    '{hoA$u3&I7@dll{02eR%Jq9VfQ_RPw=@RVVmdf>0UNiF%`coZj_sqz2!k~|M{QhvhnSF|3bp)8iiqoEmDWW^n5wjE@_mk+bI}%_r'
    'EnSVy!YlxWWQoBo(MZ0@M)#7wSpf3JF46o{1e>WIWa_kOaK^~q!~}T-%KR@tc`OcM=sj;<gEkVb{`X@U-{QGbUUZQSbkU2Ci75+9'
    '5#W?HghI-|24a9O5nsJYK$OYC%p3X$NoY$nLVFw@JfYjhI1z}n>x|bIYNvD$zf)$vpPzG@{SLOh$?Zlz@JE-(D2XM&(VIobM(#gS'
    'H18D6i2HxV=P-HWU#7-yj2@5@N6=EnJhA<J^uU~xa9qUedLrmL{iUFGT3q&FoD}Dr9qRR~iC%f`_|_sxxyJuT<2P+-PqW5p@n(ez'
    'N}X4U$jv3YF`bquqr3TAc<(Mfue2ybOA-m$bJTfz1RB$GHF1VX@c`a8A<7pvHvDcL3%(UZl83&RJ;`~R$dCATdp9;pTsLD$7MY>%'
    'ZABAUy{mbp=dD`ID%3_2b{}-?mcVTm$-a0`x?7{TNe-PBJ3GN^QlW*ck)p=+O?oM#U1T#okm)sZeede~!%oK<g}LdK(6-RaIe7zg'
    'yh<I&aoY=NtAk^7L_(qNlyl0kl6$!uE;xz_?r9od$f$9|H^PQE3KPlq(;Gz%xP^%n0m(&X^|_a;8bQ*0tLu4)=q+~W9WW$VvEi<d'
    'DZWo0B!vA0TF0iih)bR4>>D=)a$TpI^bMaq(!&%@Zp1Z|L6n`UfOy*N!PnJ?5zgVUeRI+foqK(ERwN_erZRqIkvKM&zfpg$P`Bp-'
    '8oN0d6cjW~QnZNhHj0n+aQEg#gcPqgSZC$I7}u<W!LGR~>P+e1q&q!wCW-(PW-|>XyaL+QBWr$)Q(?=df7QFDyEtXB;70vC?kiz>'
    '<VN^)Vzs3dgV=+6M`0S`7CX<jy~A>)3mUVJ>qJ{3vnOx<4Mw1{9jvbBf51(kx~UFg&P}#eGi|iM+dPDG0y3uN%jCv}z0BMZ?IVZY'
    'Gt|n~1LWA06y679aM(Yh%UoNw4X{G(yt^u*eX6bvK9TJ-nYXiqriGB&Mv{maHR({$Eb&C*+9a(+N)uY%FN@=RESGxE;Fkc_^Nq6B'
    'Br)q86O;7w3uZz1NM7^+t>riB55Ka%W4cXFuO7|k{RMeVk$j#K+AHUE#LzYRrYQ-6`=YFy&sR(+t)`gJ^vJzG-|rNGB_a=wWjk?>'
    'xp5;QO=q7OUpyn?K?`7H>afo&`a}RDlzK&gvO9c&W0tpPUUSSuo%;W#m8|kB+Y)R&=G{ATpxKh>LrP3|E$2(Gg`kOKh*e?o!G+`V'
    '8mXdcc0_W_2*^`{QBd=S;_D{51)H@)@nX_EW7P+_I&7MEm^R16Ur-TmfGaiBe}XvA0jvrX__op>=HKOmpAPy6r|tTHbhq{ol5J;F'
    '%{v0p*D#^nM+HOpBA+oS3lG-vWj5ga&e5&T;2#@q#(Fy=85|5Mjk&j9$HA`|fEr=4O-5<F*gtuob_Mi?8DVxz*;j_~E95j;{-1~W'
    '*=c2ZcLI1m{;d_7``NG3d08`==FIh6yD|dGt{Pl#1RQtrcQ^7)k=mD0BYln!s#~wnDRx(Ne20FYh7+H`W%l(6Oe^|Z!BOY<!ZnD6'
    'eti=%=!6#RM9Of{t-#Xpy!-qb|MH6HpG@7l+>g0>mLy6uJ`%+ywa2B)U~=1Tj;dYxIdG|%0O-Ct{tN&x!tk8aPsG_Ps^B$BX2fJc'
    'so?Jy9R_iwCX}Tn|MNWY+j0-3%ZOwS+SsK;K?Q7YQ?IRi1ccy2479jhY3~#4Wjkk0LWB^SZD$sxYTf@CGjZ-vr%|l5CjdH8&3dZ0'
    'PMD#P`okZaJ9*rwBl*1yxfaS=)j{Yj67EAm3`MLYYIKA;j7tBpDuVuIIOypf)VlBwpZ|hj4RIrkY<&T%2IM^moLVb2@<=Ony8TPq'
    'WF;Op()lCqX6{V_B;$#>?-Y;{qv7DYN|SR>lh22c@Xg_kQ{Z&T4cV0q2cSAu{4XuW`Wbc3?sBmHkxa&u!7Kyam}rxsilvbKo7#~o'
    'S#|}wrkl;SlQ_oPVYSW0n&cn6(3A-k-{=HHmQjWT`+*Qd=V!xq$2U7MT$m$+rRyGCC}f@YJn;~<Id8<@{3Yj5k3)TV1vT2#B66bW'
    '7vv5{2%xnA%0gaEmuSd7dptAO%{%O9wqx0q#br%w7DhvJKsidwwu?YsrWB}FUmjSmaEECAe4(#PNr#fE(}?sA{c7`~Id|#|=Dj^8'
    'P;ixx&-@RmH_it%u3l*y#tHMMC5Mm$xptB6*$9h8I=5{FBJk|LIiop5`2PAv4R78C<P$fl)h}pNeZM@UBed>mWI5j*O{-3YTjg6;'
    'To@(Po+dTov$m`hXT#rQa^6T3F(I{Xw7D|HG@R>^7QdMwuf&eIpgTKJK!~|N3*J+?&S<CJa6>n^(d2+4As}_JLj7}O|DJaZ2FF~{'
    '&*MxiNWq}Rr9K3RhbNg;@A`&1?<$0Oh|waFOo8ql+G2op=)t!mvC4u;RpR;g53c6(&yrSh;rd;l0UN75;#2Dgl`~xLpS(8mhPJ29'
    '`56bj_#lBvcqD8HVaC<2wc{wxd&Yl@k!DD@W@(bA3$UZx@J!p1)$vQp_vSaAe>J1dCmt`B`c>CX7m!&4?l@J|dUFqT-UK>va6F;P'
    'b{~fYJuxxlt4X%r6fP^?h-u6joc=IGTfHbgmyS3zyXYXa1mcm;d%N5ZN)#KH^jMx?n)ke_n3h*fZ&Kz5Z{z>irhU1WS+Xpl7$+X{'
    'IY$CUH6DzfFVoMz@>8D)?EG{!O#&u(+mXwN8Hk<3CfO<=iMc#~=sN0!tS++jf<3B!{fZAM_x}>nt#Sem`eM&%q8ek`^Nbtj@_rmW'
    'aQPk!;HehYB+tkF9UeY6;qT+|MG>~Mz@u2QZ^E!2>;M|OUpa17(7k-gp?dRa&M^5f9-FR*zrk4zT>3x*`uUDU9`Zs2R0K2JS!32?'
    '&62Hdk3qbYL{&3Ma*<GQNQFQAOa+&3iyu91d|vLn`c)A0#gT8EI*%GXUNi&skJx_%n+gQvubGa>@jw>q5?^A~i4EUbd@}^aZwd-1'
    '^BR;bs1%_Ad<UN6m!?+XEE%pDsAV1`c4WMtuNVsu2_5oQ3>NmydLN;{&2ZQZIIxo$c05_L+LFWXR25R>4FaC(E4idEx)bFR?~j4F'
    'cxa`ePPotm&ub$I?uBZ5g-v1jxxSfezvFoR6<NqMCJsm%s0s8JW=@mf(6qxN4u9N7GDwP=4r0@jVG0PUke0dBxQhqricxH<51cB6'
    '0zW$Sz2lYQ%21Wr);(j6xUcqo1(&&C;1g<@Gjk6n&UB0F;|qc6{oXIhKA;x9NeK&f|D&{h?0}-ow5nc9-o)^`r@A*>jeq%@U-{|q'
    'm<<{XBP22{A7%D**KhK2&q3iDS~2f=m|JjBRX{z@2bu&HOOtFu^C?R_MrcKw!R5dhKP)kBU$&B}QYT?7BlGd$?F)@y-i(Q=V}^Dh'
    ';euc>_?g+Sx;!~ILB2!DNaEp`{q~eIlREy5RcjGyd-pEny##j7;PAY0S;XF<XXCbqH2xiSUd#c$vBDWC7z?&wa)8u0ub`@VenC+<'
    'TsRfL+^)1&W6q_=nW&|CmzHl#@Xwk-l@OIUKRHm$;hD6L2)@s!(CSy<1a@&`d>X#M>yWR~7~{Hud0Asq&<0yiss9#NPHpNkPfgoz'
    'S9$*JLv`Z$#mJ0KM^)U%mc$ztQEZjS)c3aJ-Iza8u(Ra&V|hntd!x$&F8eZ~)a=W|!E=bu`ArBjB(UH|&S{2)31Ul4pzMPtE{fnb'
    '-jLVeJG{X^7NC@xY_OlvR%%&b+t0v&yNMRcCycyWC+x1a*iEp$cGxha{`sWMD=En%VpOdT3QEskxsjNSLs%`Kv;fBA5CFFGKwRT`'
    'eiBJB6a&UE0teY44RG+SZiYVOSr)UxZk7TDofM}%;+@cV(Ys=h!HK}5F<73dI6|`Lqwy?tRp*0xv}8Hsc3-<{6GH~(eWb9x*?FgW'
    'Nq*<)mw$kD&u4N<2BL?oY+$kR2&i%O`$P)p;3zM*z5ezqKOH+`N&{vMVqPsqO(w*{a-0DrRh5mjFWBOxXWGDC(qW}8&c|Yw#&I`k'
    'SREdtl8GWOw))LCzhv12qWrUCP7%gK(O+}H0F7iklL4i~Wm&%D;XN&P61HiZ=EV3^$19&c34bEP9_@X6kf&}#?TR`lJ3RQ^14u{-'
    '>yoW8lZ=-?=i8|h3451!{+KDxnDad3-sf)+AKu~&VBZytPj`Mf&1?Kb+q^SnG4O>q;&4W_h{ItYcL|2|(ZdBUrwDg>;cxR~ouAPs'
    '3lF;f0o;<?L#L-8ofs!nOT+f`dGf2}_oxR#fOzYAEf_?f?{)(_pMZr&ec&XfG0B)T@l55Y6z;J2T$!kMPG4QVdo7x?IZz2j1yS3X'
    'G@M(sM!UvCll0}MUSr?+QHvs+R?S7mA&i{PM{B`Oxcs2&A$M7e++wh>BngW9#Aw6!HT2{KdoB6=ef6pE(E0*94>t$Sxd0+7-^dZZ'
    'q~T0$IsK7;6}?q94m7~0Do1J4vLig_^g18Zc`W<7j@kt+t*X$uXLHx=^zwZ4zmcvzq_D1N*E+khZYYT>>P<*1MPJj8S4(CwvfF8F'
    '5`c@ZT*k8|bmKRqooob1no}%ywnzS($am5;7+sx{bd`W2gVjH;tf*%W^(-01+8$YLN3Jv=dVhtx>FwW@BYjB@dVeVaXYh^}kun>e'
    'oU0Z~VnGRPVINdJ^bO}Lw3O-%v(yR>a1(1ld2ih%+91ng8sS2uFlFj$TzQ!}aDd}GcyOa}<xze(eBxC0D<qZMeEINQ9%9+jCi&PF'
    'p3o=ko;tE}WMuUu_6Vl<2YXIf)+SGQ`aW&rgddn6ud)Irb##y*hsw<*{OMQ9xA*ELs*W`!0;=J;!@RG=ad4``$M^s+gS&(=J8g`0'
    'U$6-&&Ro@N7O`~-+KPm4UF(bpW=1&NV{2>ZVV%C!uQM)BBE0Gdi;Qg38PAldVvjDYr)yX%`F*dNNXK_Ka||oO5A6BkJwbjJAo%$P'
    '5BxfKfhEH{q)#C(P~+;_x0<&I^YnY36sJum6qvWFGyiIOyqK@4W!r=n71fX|=62kZz#}w^t|cs&sY=mgLl9IQ@lhhA9e#}<9xxgz'
    '4AkVpRGep~Es@LUY*RG>513WFCe0y<i>oSF7{#qn`&R^Qj00Tqa%ev}dWh}$z|S|_AAGE>Btes78~((M<l>nUFYzGm%Wg&mM7zsG'
    '<GV{inb?Y1vTxjZ#=)GE3N*{65jUI|i+~L$>+hnqMEaz0ds1qbTx(W&n8CB8X(iK(C;9CQ;B3jzqGPvaXhHc?Oza~!$|8VnRbh$O'
    'A4Bu1&=ggWss%I#5m;c!@V@{&nK^z?H2=zM7^@ptbJ~VNn7jL_^JXpiNaxvTp=hDqc%69SKg^}X#?>=+vsFm61UXw2JVYGI`T$d3'
    'yNslav&K2@^w?s1bRx_h8;$5Y1HW0lIx;AxW^ndY{}el|Y%BS01Sv(lN9cCw2Yy5@6lSI1Gp}&q+`|Bu7HjN#AF&Ev4DniGUw-<9'
    'pU$2R4W?0&_Czg~Y&wy1n7~D`#=SH}QCz#z!`l(N6A?2Cj9%3ql+$?ql7Tphe5BQST?TVx3rTp{n`NWY)Dy@ni+TVd#&!t!lBO=}'
    'BO=t_VTEc0Dn+-6aMS63IVIq;f`d6-&*$(tS4QXhUM`zaK@t!`rYV({-=9;DqfI|=rpyqX-TQoOm{`7N5~GY5I&ip!+BM*URG3jr'
    'PKAFAh4)rZ(v-rc>M0nYqkAmjV>MsMjrOR%McaLXBh-epOYS3~d|q!jRK5&Ddqr<S3{<BDL{)zsFK~qzUzR6zk+j0ssRV-2ANyVo'
    'pVKq{3d{hWCEGZ;_%Lsa!wLjHS9ec$Xp#4Fd?hrO2twUIpm!58OhRlJI%srb?Eu^I74wuW7sKJ49$E2s0S~_11yxBsBpUB87qnRV'
    'P)!s8MMMx3@dnI|o$yRe8eUK<fl<|0aKBFo2^a}z9dIzjrjQi5JrX0!Ec4XXryYBOBK`IYKYyP{I0)bfr-_BxdHZmVETjKWMy45m'
    'nk%bnmek6McbYonY<)aC{@hdLv9~QM?Oq$zhkjp;iN$E7y<0i&s3sYhSr)V==_aysfHm`GA&ha$a(bM{1A*y7fMdJnSZSmQdiXT1'
    'j$e658Yyq=6G?bngA<~u*l`Buw}U-U_@O;#?&%ob%L?L_%R@quxH1hiEnt&$+UcH1G0=3tcDh7iHQoNtJ|mhIvYDg}zvS|=lt1wd'
    'M-$8}WoPu6jKiSpbBv_TY^T@w6@hR^HB=<-0=XYpJ^%(Gj@Ik+u-8EkgE+HtfSWGza`HEo4;Y9-0Hp$#Fg^WBRPAegBesltgZmgf'
    'JYFh8lnY;iOBy54$t$I09QB6K@al(fDO3}<vLkRkpZbT+%qPcG6&94=0z}%HNItoa^Ul8K_a}a;maM8*2avcggL6rcYq&8+lCKuO'
    '$;+dL9XlC!`=q-osOt4=9`kB2!ucT{SBh`+F`H~7s|j<B))NvAe0Fz5;$fM`<*)S(X?p6mq!X>5+tjy#!1`a|bi7?U5C9{Tx|-A6'
    'opS~chDF%Fm`eWGl?~q|R%PPM88(6OGLBx54V$@`Cx{QJGF^gTK;R~Z^Kqn|`-Y9!?(x>YPrkbDX*EXmyhvAor@l!tdbt_e^%I-I'
    '^pKW^v2t`}FW_o|q()}q^`=4l*Yfjw;0R5|D8>&yhfRG<?ZUNaFV|wrylQxhM}P@CRTbh$H}9^0ikU#F4bPF`B9{0g=ONQlajzR5'
    '#>7Sqyj!mGKYRn$)=`R@LP6oYTydoN;g9A#r|rD}EY{e}SWj~&-UJOT4$AZIBZkaay&9SO_xR28ke^bA#EGeQxH58lWr#3z?S=<d'
    'K_qX`{U&&tHa*)XBib68aX?1W=j6KkzoDTXSPf@6!Dq<}AFh63eJ6UeI2L-IlD$(47@34{jqc|e3Y#UdwWl|j0V<60YxJ|Y){%~z'
    'fKJQDkDh%F(bu{_M#438dx%RwPkJi5+*{gPb-j<9ggt@+oQ6zMGfqRV8TvUP?9~L%ojhl@Jn7S;b6YF2^8lV{a};j1eIre3O+Zk4'
    '?}Lu7k(_F5^PXL-nZV}b&r&`i<m4~+uRlHL=Z_&Vl!lOAbLqx?Uj9tnGTc0Pi`mpnHcFwJJv9x}dsWx2p(DDEb+_u<25g2l0%9~m'
    'Yt;LWly&NRu$$M0g46qYm8AAYlSPT*t)(N$335iGLMW5kcNa2n{!G6F*>2knw2O?9Ey&FcXbRxnzc<<Cb-ch%(pnWl&nhY|-;5~c'
    '%zkso5PiK-vhlVW$5mM06u4V6Qryd<-=<(hA%OAimA2qeZ!W+0uj2CtRMfJiX_1#c?2veB`Q!4wG;qK>wo7ln6XJF^E$2pL7BS@('
    'i<o)HzCQH@n*4;puY0Stzy9Ah`VeLU*7BTzurc}SLlzG@1fdph#$u;NRY%f<7CKGrTy)wwi<-VzD&I#-_hS-Gs^GhBD32=@*#2Qb'
    '47nkCgzSyF%(8J+5zF_C8b2Y<31Da!a(8Z;ov9{*RQ})F(bDtgvU-95hlwHT>vE^QbB&*8n4m+pEbpy(6Moi`f#-1V?P_Ad6mXFP'
    'VX6$Y$ujKNw{)a5zZnmiCDuHTYz&6SY+Z}otcU^LxJe!NX};A+3i*RZXA)P!We?;ojvnXWb+|cOBgm9ah3UYctvlIW>{dA~-X;=`'
    'mfW4wK<dly-Ej#%Llf&C-+{?GZJDg01fz;h*}BFDlMh7-)fb_V9Hqp}Ft_Kc!Pv!FE^6Wk@3V<Q#Q8)S!Bk!>F{Y+3Z7?I^kL1YC'
    'qW$A{@itBqx>~QIi<Z%OLai_^B(G(L_kMbWb%lmcRUMBP<4dsuDoCOR?i&2LYR$nC{r;bPl0V!NXqcRgo#XL$%BUfbxe_z<qMh%F'
    'ihZ8@<sc*XO#lKI1okFqlf5{~)foyOlLu*Lwa_4)*+M&FPjR!fW!`n;SC+CeE-GyYQr+2eYu(vTjMzQs>26i_{Qj#<;iB|wI9sjv'
    'JtrEXp^g1Ax`vsmM2x8n6&$>r%SS4>t?vruvJSaROvSTMJ$Hpgz74y~!mv*j3JT+K?>r^WHR3N6%AHTiFaDEtA|&FV`<nZ*e#0`@'
    'Xy4lqK!r2Yo$hxdML)mN))NHP14#NW_pd+Sh8;$Pw8|P9+~5od!e`K!_vsse0JZY@5}z)qpxkH{_WaGq9X&_Zt)IOGn;zz^WH-nW'
    'vey__dI||*v^?&D>X6&Sq@}A?+XmnK^Cb>*yZ5v?Gu12!-S;{Kdc{)$%qZcT{8^NJ=$5~9w%?23{bs-liJq<b#tJ~oMO4N_|2@>o'
    'j0{7toZ9FRZz4BpDJa6W<v3A@=S385`90j_El&ekxzt93AHdYGzN2nmD&)XSQ5(z*KF;Nm67>p9$?oQ}#kBaPoH-DFBvRKd{kZLO'
    'SsyL?`Db8o)#<-GCvdKSRsZnS|LHP3r(VM=2*{EJ%gfS<1xrUFph9Ar8qGaW*hz8^kdYoy=>wIG@5SsFFdiujQpl0S)?Iaa0`Ast'
    '^ace(RBpiH;w|0tSR`%_z%|kggb%eb)7cH(N!N}XFwvjPc}X>PwMvFOOX3%H&&ZY0KKsx5?J9hm%b$&vo`R~`|26Ce3{t&>+D#pH'
    '`H2d~vDl>Rui0a3czqP&>A&Lg<{B7&0r~0Qgb&Tkb{N|hr97|w04iXsf@)GSj8(^^o_goC<!hcMpXFLmKfefXA-O|VYB3j+)lhk)'
    'kkjoldjt{CB;)H;ASK<XAjG%9EGCYMi)|&3f%TvHeUnHD;$?SC+_MM5xrGi^1@%)K4M%3lk@reiOW;b1o#0AqgcH5qgklos@1b|f'
    'Y`w)blFt`l4#j^7W6wd@U+!OjKJo_VkY^WTT$5NgQyeJilpRo-nxNI;_Oj;~($sQ;&u{gQHzOIUGcM)_(WxBQRg+ah`G94#=eR31'
    '4R4}iiA%EOFC%g8HNj#S?lk<&$C?K@aqhMA<+=wROXY_U8(Uc>_Q><-rLOtH1iB=p_V4w7l}&U$FM3@N*VDX{WxhFQp@Ym9`+<U2'
    'B~mD`n%Sz8exS{(gs3}-%<d?aDZJH+7oP#a97{YrJ+RK>Rcpv`F!Yc>G~o`YJks;lk<i#g+7M@#D4-hDH}jNx;vr8i2s^u}jK2$r'
    'Iu#$0J8H>s7Ut$Nzw|#W{}(TCUC0&R3=(ZBIfZ)QpdQl^FN+POAvMYnBP#h`?D}Tu$Uw5C`fcmRyw{)(njz3k3A;2nyrg<|jNt3Z'
    'EQsni?D#71R2;Hlb2IzI{XX~L>o>LgeDFE>NU?>AxVJS!(;fj5^Chy&_io>Lcl(=jZ>h?nxO>4yygbY<jbuLrdNyQJ@U2<8YFUzY'
    'EldKnva2>S07}Hg91qAZ9(s!^ZsBK6kk7kC2@=U@6zc20&#9)8#pOj<M+%{?au20*3Fs?$FA>bCd{I0bm{y1i)|5-uG%)~#!SI{w'
    'opHXgLI3d$;0VVD`l9YaISu|CoFluSXBlLGr5+(63)L>Z-62Qp#A8q&NQdar+YR0P+A8$GQDJRYK<vH)yLj}l7oB~CCh6@Ji*DRy'
    '4+E{~=w^-kY6u`wB(t#B9Xr$!1$F=O-~RcZ|Mz9D_bH!~tty2j`d<23+_KZGcWQY=^6FF&=47Pp!H*K5<`og9kz|?vC==VR`^2Zd'
    'Zq~_8$qkUYcht;wukeYWTGv|kN@h^Guvo$Not8TNUd?Q~-rk?Omk<P=?I>6FQj~bE4g%f2vP{1v-}~(jXx*HhE#r=EdXX_XnU%<Q'
    '@A%U2%~Y!ptW5$QL^Rw|k8;!?w=!}IJgA&0^^8a3c7G(RLWW5K(+<KLT2JPlo<&e#?D<c%EIx^FEU$r?l;2lsPHzdrJ|FqhKDR(4'
    'P^0LfP~p>2+P&x<cR0kY-6#Hizn8sqZG0oy<Z&Rp4g#x5^P3q8&v_w5Le`(#HUmP8(|a{i_BcC*NaN=2VP~-O5vI#w#5xC_qu{?f'
    '602g}fBHUsoUdDHd{te!@)KDU&y4CM&UGvQjKOBT<DWrDrH7pHQvBFrH>oe`^-qt#w*23j)A(!}ecqJcHKfsQBSZtf(FZ(kRx9+C'
    '_{Hld>Y8~r6VWx3n|O-%_U2LGjZoAd&lDnI%&^svgVwU=7&WQSd}an4JS+B&ZKHOcIn6%jHj_mTH!GO2_KaKYqAi_Tp1Q-d-NdZI'
    '3ur;ZK<D=z=<ob?>I3Zu4K0oYbRR;cW*vZnfhw$-{8#P$dY&SKmIK0ZqHDE3qQ`atYbwvYg&}4j&8jQ~Fmm!#t6KNoTf7$zz_y{r'
    'pl1~%4VOpfpIhM;?ojBuN^au~?y-OFX8?|BAbeMu0V0VnLE?aJjA~-_ZzjsQe^0d49MYc_-+^cP9iWz_wqfg!8^n8W^ji`q2HGBU'
    '>2<(iVzp0XNXfa-^Q{0j%hkVGurTHrMOn$iqL?3O_vnqWwtd~;0w)0%Ah8p2k?Qg>kJ7R5`PTa$@JGN(htRjCtf|dp24Ukl(`pD?'
    '_b^2L_}5D4r?Jm=z=%G2n60?KX1A>@+4ylnLsMYhs&@?ja+1-Pt~j*GUV-4`o%y^v5vYhj12`PeH-3&0DisAI)1v_)9`A-}uogs1'
    'kqAt^B~Rl?)W^v3e<sMlj8-nd^-ExyGOJF3C2busGEXThMYeH*!FsX?&cD7n*8@f!n<U+$nxPgTG{GDk_?QuBdMYa`(KJ<O#<}DZ'
    '5oEc9GP5S95Qn}lmT_@%-5YAo?kpepvGxB`{A+jU=##w5j}^#Z!hQaYal~z;2<MC7yy!x<vjWMV`P0xxODtcnXS)*HGyv$`CQ=5N'
    'Y;JTP`~9j`U<m_mGeiGZ`d{$|8d-5^V-iyF+TCNSq0WU>K%M(25asC5RJ)ZmB#BtM!&-^(e2P2nYA-~iSCSnZ<|5|aEB-lW<LQ2g'
    '>wFx%qDot=`5Uo``R30ZV|Dv31>aN3-KOY&&4?Q}JWAc%!w*;2?Z~Pkw$X!%v4<b#894ZRboA?*=1E0nhH2e252J-2FV&%X)ALEr'
    'Hq+D@dqE2x9}0SGZZ1?AjA!nJS|xL@1W=GQeD@z4kr-etRO-%r&^33~K)2+jbxR}s{8<MyoJu|hbU?(Ej7<u%l+rCK91T#biFofg'
    'gq(sx(A1EV9CK>$9*2pcLy72M`-2t{fvGC`z$72%4eZ8D=ji+K-8j{f4G{W0H611UL6Q-&AB+MQGC=kkd)y^RLvgXtdnGj<t*Akm'
    'KF|tSZ%f?!_p9(_I$=^KN(L8Ju%3tk>(1SW_>-r*nPr8`evKjE@LV!TIW9jQEXM=u#uBb^=b^p7=Dbj~na;f=?!V#%I#oiJLDhvL'
    'y`JYoZHa5|S$tdQdKKQ1_&gtP?n2fulRy}_Iy`D2g}uPnB);|Mc0g=2e!dKsbEoscPKE4KZdnm|$7M*sD=f)ZEe7QDarvX81p2W|'
    '8f7;K#oFmW*QT0+!M6gu$~IY$YLaoXX^fllg&(KyYmn$k1H*}VT@k(mcW2D3O90-p!^=6%>bh59WlKP!GtxT=!VA_9#`YDlX}ho-'
    'D9?X#719^1hgM|2%5pl<E#m<vJUq?#IVnpJAv9&riQ|MpsOBq6L59P?vGj-U6_-7+I~j>08eyNd%@Y`l#0I__uwVx1T+Yp)jY;u%'
    'ivx0a1QLV<g7K~<YWzHBEBK)E?*4K+2=s9)V<|d?*alOpzG?r@1=7+CO<58gOn#am8coa5Xjo$JG<ccD>6K4RGN+ONh7&^YtGP{&'
    'VMVNV_y_T^x@h2eaN>EM;RMmn1m_h@FJI=$gv5NQ@Tcx8pDU4Uv2$aM=aT;jBa-gj_KL#k^SQs@*C6@~s|OaJ24BxrOj~=EpL>`!'
    '&}#gO;WO8`!F2f|TM$I4=9`g4jFCwgVz}x|G}vJxkcJcQ*ALgPSTmW}U^x6#e#j~*IS}&}TpJnY$7rUn3kc@<hhM$Oqji7Z(@Nrg'
    'Z5W#`<GbB8$oqW_!t%oa^~t^lDA@bl2eW%#Wb+HCR4}RrC?l-9n5V{b?tNI)Oj7s4<=zZ-Rsa%R+CAMLmEOGTz2%Hds8XUN=wX$6'
    '_27ihi?SPhmwcMF?sfRYjeu3>uZc77>AjfOOHhE^l+b2-YRx^@;Qgm^fT3UwsTSUMml7y<YP7Mh)1IxDCdZ%L5D_F)HYJ$k6?5v0'
    'ChRc+hDV}ZN~{g=jAHE-i1x<se8)clx-Y=<x}4wS-R&g7qYaH&IlF!AHQofXeR!}H@x}S@Fkj~JVz&TOt$|mxF)Y{Y?x*ncO)xY+'
    'Og<zLXVtKdq=*>b1gU>0kHJc+9)g&l2>;XuN|u@Nf3st>UZhA3;oh)WaI{?ONT8=VR>P!X!NTGYNMo0MP%;;-p^1r}$>e`%Z^NCQ'
    '-ni#!P^34gX2`9a(IK7B19`~#Jgz?aJ`QBwd{YFJkl{H0CvDC|w1N10c-=f222HQ5(6)^zXbi?5oleZ4oRZY!n>M?EdJ|<57J?kp'
    'K~bkZ0V|kepK4BV$hc^P_YGk?AT-pFd>`NpWForUhmxWlp#S?^F~FiVZd{U~Lc885P@t(9GEdvrzT^n#=Zhh6mBWhxJIMKA349$@'
    '65&HI1BU}OpTOYIya1&r6RN7Xp38%iE{Z`+qkc_#mZ+J^T|mS;a$d&_-JvLHk)aLgK0b|%E?h*I9*{N6FLaj5rbdW6qhvUG`{Q4b'
    '0dT%4YOgc_;24_uTb;un>!4KVnHcizXfu&?fJYK~vHA^QZ?=NleXZ-2EgNMOr=ahO+J#gNS3!Jev_bnu(hak`IXmrk@mt0Zk$!h-'
    '9Hk?BZU}oDA00r&%yPP^X`>p6FAWjq?dSkVhxp6=>$eM#<%J2Z4<r~4<Y}l}HyEZid5=EmHDF(LJ}(ZS8Ikie{<1jmmmY3Z*D@D4'
    'WVgOk_&4YNjyKUk1Sb~Bb2DJ28nevZH_RPYOfs_(jM7=lO1Q_DsdP#pO<d=)=;<U!!HdMSE`w9-;CcDa_l?*F_k_4*P6@NgU)Rb5'
    '$n913yI>J?F29;kmwB5JGsxFSs^g<Bt_Uj`D%*3=;c0B4<hb}@oZ;G|6rTvHRa5FOEdlwj{M8v1a1dIk(X9PjqL1p`2(tdKI+&Oa'
    'J;^p9Y`A05NyekbZvjbb{^{kX!dH_$gRMRd@~JNU%~q&kQMmLEgogJAFSj5mPYbT_ebryn8{pB<QDjy=s=%57ZR!N<fsjDD7StYr'
    '5iOgK^Hc@!vD~H5iLG$9#{AG!uNdSr5`773-J+6=SK5s|40xcjuhhNddr$;cfA0f%F-}GcqIBt!OsQD_vu$Q5+&KV%7eCVYOK?`2'
    '#H(pZ=Pr3>7GRq}g?C?z5RQi@BNA>>uD;@YV2Chr52ZGcs9~3y+o;z1w6LI3{+*V|X*hS<n~N`^r(df1yAu`<FBT6B#1?hmlQ1ky'
    '*X`yiq`G0P>c;oY##eav%O?uSsd<Ynu{ErwZZu`e+|_FGNa~7bZlsyHN|%N1TE|?<y;rjYR^1jZN%!e(@1179cmH#axz2}NODX_-'
    '7YhtbUHO<xX;%dN-4j<3G8J#?hZP5Iws&VPV3pT^?UJUa$;+hLC!9gn3PqKEETvBq`hLpGqV8cpa-J%)HQ}yE+18HgvW(CZ$D2{_'
    '!(0Kwj;jRB%nH|wj|j*84Z2=$6g=Gh_q;<qw>FD7yxjP35Fnn1m9M?Kk9`%Ns1&}tY)?7BJZ_6y(-DX%Ux`1wH*aw=RYTV!NXsuc'
    'Ig)Dlw&Bjujf?swqD5LFN=J}^umPzKA07R5Ci{ROdDAsOx@q+4<6h=X{_jvu4?|3$&nlq^pVGiP3={6B+CS9|`92h1Nl`d=BKXeJ'
    '@W1ow<1_5|+vKKMdaT7}-SYZhHxEq>@S9K(LU2V+7GL)Adh**%z=UrE!eTpLXfeDGC~<XtP^>URZ7%<25DG}^-Z07^Nh)LY;Me^a'
    'm?Jot{rwZ=wcNj{`-`9W9O9pG*R$bC!qMs3sh$gez;-%ySj_x85zd5rf8Y2oagpb<n@(4IqTn?cboEto=mWQ-qN5OZ;}N(f7a8pv'
    'Ge4xFRzdGw|FKBt?mGudW&OoZ$kku{kcYI3<=ro@60x$`kCTM!9QNVfm3|@qZ<aq6XPJP`JMkid;yhx=S&Ok)8my7<j+r$Ha*Ztg'
    '+us!VWOK2&*>)PscZ02J9*v4mzzm2tL0%lu&VwP6?O?X~yjk|IGUEwYB{IlxQCy9uC@khNb29QwqYrqa`eo2=G1zxxESUhDCrX}X'
    'gRtiwaE;IC(~lSI{4Mn1Rz0Mz+N=zjX%DwkUy1or$r*ap`d;Xd$GCNVYCbINIg9RXhX{SD1@;x|_c)`$*f!>fY=swkD{A5_$Qf$h'
    '7<A(BoCMDdR`~Ia&*ZbA1bCQKa7NA<hC1FQx57nb5W;Vz0Z$1B`D$OcgkU5vGw3C7@4pmnhgCd?Qwk8iBa2K+{A#_28=>x@y4~17'
    'aFQ51%w49Q@qCB&JQOh!(tcUbt~Uu}eHFE*<!s=KC}&C{REtuuN>qtxyQh1^{(h6oo+mu!d1L#UvsHWb@ln1N1QpQFTa++B1elp)'
    'SN1V^b~1zf>$iIkl@Jx>UYuMvPDu+ZTW}vgG)Y?5jbzv(jr_hcuel<18^W`(4R3o`$*9w$1QouE$z<2#<8*;Q7PSth8nEFvv#rV)'
    'fUp#4@`}mrR=G{F1wMyMY&MLzn2lB571%6^0;n>!>Tr-i?}r5Rz07xxy%HN-iW5j?%e46|Qrqk|jJg41V@ZQ|wbH(!@+MO<gN!0g'
    'I0gAKbATpC#pZjDZzejR8(A1P=bZeU!b<vf*F&;a<lg?dW_R3#^=s12tfrmk<D+>%-_#*1#_S8H=^W&z46DgdsyCcu_CK9*VQ;Sn'
    ')iT{+jt1u2#8{*;Lu?a7zLB&jDk{gFo1o7;5$M2KbI+MDmvdh)-jN9i^2<4aQ+4hF<uAK=^ReahiI|!@0hxFKn#s+`viZ4!n^FWS'
    'FOZfm6FeZ3<s<clFvtG*xuzEkORGNT7izRJT*BevkMEWKVp9i%b2^Ek>lxbm6LMM5x;E|}5#yWzEn>Ha`B_|Z!wV2hVUpAyfq<-6'
    'AEEFK%x7w1IX0!b^D1Q0^Hg*`1D=XRZ;P{MTo?~PT^IcRS@^U$ZZ1jO?UoUI9WX&9ni^<uC&dPzOc?P!r-6*#Z`U8bH~gHNfgu~j'
    '`N-`P7IBKnsStP)2FLSo+ThpeFF75Kf7#*&ep~HsT0(=J0}wP6$U9mj1GsfRitLt1t={j=YKf_e%Z<}M#Pial1I5VTtTTvDQmxbQ'
    ')piGcn?(+Jz;(tlZq9s%4XJL%jE)cq=bu?#PLIF{zeW>>1WO6L07}-P6BJ;hZoFOaR<w&3SbDa3rz<R26DzUHv>oAh>;5fW98c2q'
    'Oq{|CmD_qN5LkjmyA1iHia#;D3^nvvrXu}*SM)${ZU<)kzjOgcsi~I-yw|zrNpeyR%46Iz(Pg4-=GlBrZo{3hLs=YTSjDeDst)$G'
    'sPLkNdf8DohdZpQbUdcqe|M6)@uTq}6B*%??n+Esxzv7~uV2W?Blv2cs1%&wpEb{e!-nB*u<Mw{MNO?0?Kcj$kJIh@{g*Ufhq|;6'
    '=zW&9Fa{hP5!GrhwvIPA4}5(Jv)RP5!Y|LQi+6$(P&?KlQyxTSl>Arg_a^iCF&Ysnj)UBD_vH2y(8^HI+h6l2w_nS?nov=1|8tZ7'
    'FuE{dMjP|Q=K?8YqS@}rR>47>GVj}hXFLws2kQ(*s`9RB_?Lb*Qe>4j4Rw**cr2BUuX`v09%FAhe>BH@v*8^~2w3Z@9_Gdnqm*P#'
    'dxN<8NI}%)l%^lmnN92Y#B=Y(qB2*|ZGcN|!~5gnJFQ*EOjUcN(V&yKKilfj{Ey3*I#=YMU3~cbP)sWKc*Kq_rN%39LAL@d!eh<r'
    '6{v%!nWI{<AEyi>fO7<f93;gC)yN~(1y4vKY?&(<Lga=8FOp@xNgv4n6ZSobd$f}2&Sk5Z2majnUOOPaw7%~Tr31W-r;$F_qx#c-'
    'hJLbT&-H^8^xf=<!Hx7ozxym3(YrH}Mf2wnv?;D_vbA<(S8V<em8<o;OW#*ZT2VaJq5($hw*z41P;jU|ob>B;l9ky-0XpMzQ;Te}'
    '$LeXQBtdu0whx`zNbg(Xa_sY0Xf}m-4cA(1aUL4&rI3UU9FHwS99O<OGDfh-e$kQMrJIK+Z`cOX&gIv``gW#!I=`%WwHV=sSwUp='
    '>n;r4HiRdnW(jpUUwBs(!;mE<9F}>f^2`m@M*Ipult1U(X;mh?*jDO?b=~<Mjif4k%v}BZAx9Gi<dAUYDVyk}m(^o88Ov2&9mNz!'
    'Z9Vst2PHQ{wS0fL7M`heaSGBUf?NO?X4fcJc6_T>aRY1o%Nzk4c_Q_s_kd*Qq<i1!H#872+*={{j7T$}2Y(8E@InAnV0(q_bnb5w'
    'a;g1eYkdE(y83?B74twaZILb&RW}4E3jOCz%Ds%-G&-VMhWp*iu*g{Fd6z(*w+6H4+rVXGHTP)gJkNU$yT`NcTwYdn^$DYzO_Fdc'
    'K_Eq9huoqWeOH4Roi68xmyf9z`Ki#ubVHo$_B2PxDgtLN)FZ<5@PhxxhuwDH%4%`n_1$PPC!6UIADKVVvdTILZ@JOrNe*ioDT+$Z'
    '@Tgw{H%Q|sI@qe4>sL{95NYKJ-DF5NIh}dU0}**|-!U?q@AVagq*K0Wy-tdByZz4lIqIH7YvP_V?z|#7cmTcTAu9cj^P5<i{{1N6'
    'yniWiUqy1SX)?=}hZ1$_<J;!1vbg}wSJXf>ufNmr2IIpf;thP!_-sCeC(KvJ<r2(2@9-Zv(C8&eB2B=_D}WrKADubRVOXz~fZMO&'
    'J=|qOqw&B$pLpJdmriDokxeL%!Wn?WB*gT&l2JG>U9pUB$}ZQMolhCV*TcAjMDiwoyNhY@d%<lY+@xC@R^Aq@uK0R*zm;_H`Re!T'
    'vGpv~&D#OyP~Ydm^L!<Zt8WjkZv$0l7$5;g;~P=xo;)2Wi7k`9l}X>TZsKIUdmU1Pp}`V^+>>%t3NoUGq|gFjUS6wt-hU$OYL;_)'
    'Rydb18BG)y50%k$oG?w9aeQdgW9nigULKgIhR)2dgXzmyK(v*aZjVyGQ(@T!pObHKR!r|U;x6O8n!D{#@dCftmmdh4C;EmNw8^NE'
    'c(Arl+f#pqk9b}pyIHhco=8+@prK7^W|!T<W-W#YAyeC)e6GL3>Fp??g;hD>X>dD?O`=?R-5|uw;3)Sk;nm4StlBtEk>Li5L89C$'
    'vE8Wm`kN_N=q58yJB{{d#^vgP0FS6Tz^ayZ-s88s27Q<!{pJ4kr`vBct#=m_R?(7IP`F-uCa4y8+YCq=jTN5B`O1|5`%qoR!vP_M'
    'vTiA5YtJKK6{)OQs>FBOv<*K|@F!06nOU(IK-|fsctzap3Cb?w;+)<@F<@Kw=KECmEO8m*7bMdTL~zkm0<!P=a(!yMZ?_)09i69>'
    'nE{y@XqgNz2W~AahxQJAYn1I2>y!&qekubLAKysP!}bxnvK)gj;9?Q)7XN*p$ajkkPDf|wD8G6QHR5wP@So{q4+*@R{9(%qk;xXv'
    'ObMZmpE@5c&hWclz4{gbDe{pa%6s+gd*1JF=5^E>#o@g|MyAU8FajiIiKc-+j7#i(B|%sv;$?W$`{G-!!v7)XbBl&Da*fy~Tr9O4'
    'qYD*UINfL|?ICXQ445@$gn3GP==7ZQHL9C)cg4l|-1z3FT!;xdR%K8J_8J1@OVkiqb5FA!C?6REI-GCU-8qzhH^onE_tMDnVxtPh'
    'fk#pqAErugyJGYuzp&$N4j@!Lhr#D^eRhh$7Hub8SkD*1yC3L*lcg1Achgf1*{sTAo}vy=N_Q0v)gU__(d5Z?QXeeuOX{Ddx1EXR'
    'i*{KK_<E8GOy{$5-QD`9e0TK}mU&GG$16FwqGW#A=Jz4cRDvN?0NpD&vq3J$y)=C~EeD^rBJeqH(-pXmEhUy{H2VHB{W=ilM&ApK'
    '#vOi55El4OF$OckXo4C&-pBCD+V%Pyy7P{T=paL_iHgqHVvxADkpd6NaH&2uh4LrYUH{Xu){LRMxaen`>|E=SnJt#u$>|hLh{Oeb'
    '(oT#EojY>?-eMn(t+R;@eoGPX1fBwrtgp1bkE#x`u4s>@4pPp@^AA(ClqZLy?hp3fOREF2-I>8w5FhWgIsldaJFdIm{_P+C^MC!9'
    'fB5@<{qMj1r~mm|`RzZQYhs}HKHrXEmkE?2jsoBqHrzlYqphjcwL|CCB}z#F7Ao8IK7ob5;^r8ps}~r!nuP99czYM`l^=_J9G~7x'
    'hGQgWeLUmyEc;2a==CL>0_`4KD@qN#Oa2PZ_jZF41Bw2v`of<7_g>iR3oG{-#cxi_#{`Zun<QK66Uab3nDkb-IH|}8GHx<fPZPv8'
    'r2&a5SfUi93EGChF&tDS*qMTm#=b?D^U4VC>JGWEvZgLE|KZ2jU9&R1)a1d%e(!-^%r<o^Ve<wJ%zNy;ASx8K4y2o$I9}^Kuk{w2'
    '+?O~>i(6D4`tVOz(vX;D&A{)UnxADea^7FLmP)s`YQqx%!Ay&K-fDts+^sYJ8t?mNFRD<MtPnw3=OXrOXLM~}C$dU;9qn|#Na~Sr'
    '>VD-^S6<5;&z%E?baDZ{ep^3YT@~)!6}6IV7T+Uwn-#I|xo&0<;;a&shn!RFG2H4I!ztNY-ef+UqX1|E_egoe_&?{Jk|E>+PK+vi'
    'Hfa$J1%^BF*X1zj>W+H{Zb`{WO?vzz;=-XAd2hnmX?<oIhS!dZ6`1+CzwOV-;*fW$!MlcvmT)1B>vX__(F=JwmCGD)0Lg{CBx6s*'
    'Gkx6jpC=A}Qa7<QqXJg-s{VJ$Mz9b;#Y+P{WGfKjWne~1eBO|LsUx<~7v4-|aAWo4A@=NSjZqBl22fM=lPk}U7stt8)f_RDx<y}W'
    'PR2OTDQZpWTFb)cd{)cqcbqRoV*+;GcJ2k&EnO2&!OIuBN_)eY!+mN$a6X=7Q9vV;c}TCGm}5FeUflUI)_F_Bm2`?{=hdBWOrXW5'
    'Lrw0>DGqHA$(^!E{;mAD6i`}(`37%;{BgN+$iCI&fZbuTZ?uBLnwQVIO42tXygAo_+#Wgy@ZlOZZ+_gS&_^gJHWBM9HbJX9@);xH'
    'tUjMJa2KY!(mWyVE(wQ-nIdRCAC_ii;2R?!r1L+&i>$V!K7kP5MhN{@=I7Po%-h8F2_%9C!c38tzpm_7U&%v=&a0}5(sQ!|&0z^%'
    'BGzMl?qcYwob=%<3QJ_S`xa(_c>8u@wH{9>iq&dQPBL>~pKCsM7)ml!(vn`N+A}gQwl!Zd1VpaPTNldVK;(R_ZEX9K_(<sIdFlWD'
    '_1k~>`+xn1fBDCM`#0YKa*s@3f0B>bc6|RG-K`fE9EL!j&)@9(vGD3Z^15~C`5xn*&esSff+ryqg%ZC#07_|R&eh)@DK3_$c|Rdc'
    '^FguF_w2V{#Z43PYxEVt5<&KbjIN#kbZgg^VoJJGKwx6EN9@QCUT?fulMoXzM0q^W$l+xJ?0-GA@xB}K^PavfubudrP!rA<e&wZA'
    'VzYaGax4;Q$?!zf9o+?h^ythd_WFdZgMk{+VYPEj%o!~bSuL5vou_O(wsq${?7LkUH3kuX#B&(75UWkb8~2s<d8w`E9bMM@_Jdm%'
    ';-kNR-9G`V)O$Dk#G{w{0f$FKaWYpbX%zV@XJgYS={XOF<sDHEsW*KtH~TzW3E&t)A-wTUol{2!&BUqunHIh!^KRNmjeK%T&URT<'
    'g&jq`w~P4Az?imge_r}?-+LC{ccb6=9a(h}W);?Dg<AgXd=bv?h`fX6XD1}gwB9rWWL4H0V(#lIG*M9u)K{3H#@YCpp;655?QDQX'
    ';EB8;X-25RcfKae^kdJSQS*M5E_gdy8&)_sHfeIo_~~<6hfx5v<(YmW46j*JeQChu+vBfp+ML{253HhuRS`k>`ke(2*Xs__cF-bT'
    'jmtEY&bBd1_`GLpg$MAJ2-IBn&X(hmY&(TcUh3aY(0=hh=Lr5E|M*Y;bZa5TFiaw7#_2`r!Orbs`0%C|b2evJ=ETOSg`K;dt1TP9'
    'znE6cKY`9fVYzY(4{pT<M!0aKKRWa&fgcY0hr|Cd^4k@MW@V-xkHbv5|No0W%FloBFZ?8pe+3{t{S`V4@mW6%Y#x?${~!H$=TZ8{'
    'e>o4*DT;iESPxRjD%#EziC=e(Vl^x}hTo<*)Y{1fndJt*U+c7H67JOx&Y$JQ?}<mSnDl=dp%(eLD`@{5;0pXwR+sMY@?$Vly+5O0'
    'CeadV7u@DM(SPrck}z8TOUaKa)hLJ~*B=f<C<|NybI~gJ;&+*<l=wT<0hWdQe-?<A!2G9t{`IOyBAkv@)HO0c7cWx5!5@FD&yOZE'
    'f%isHO4=oR<{!WD{!pKvDSsTT&F2lZ9&f+!pYebCpYweDYyVd!1IR4bj*P?-?|~c7<kF!XUg3m>2(h;6i`VQDw}zI3CcvN=EpDD;'
    'zf(}VV-yhVy#>~n*HJ&AC0}v7I6Y*3R})3!Lw=c`>qbHYZVv9!xQ{+#_@jk(geKHpe{|aL?n+VXKfhfA)f3+KGV+|Cq#2aKNSI+g'
    '^)};$*Vf;OeD$OkNh{Tbu#TDurs?)rN-nE}?$5;29tjUG_3d(s_72MXS)rCLbt-y=wt~K%Iahk2M1wm$M;%Tbl+jrx%ujwSd+BPr'
    'OE~h6aY05{3e>96*<2x6HLH$SQG7KJ?fl<0udXOco&n_{PBx5bh^@#Fe?_Z%UqbK6*Zc5@huodjg%PoMixJs(tf-EL@6r?fHN;po'
    'dG5m*--A28Zh7xvB9F`o*{SkljL0LYJR%>8Vzkr?&o?!Id5u4P(tPNrLY5nElE{*Ag7G)6Msc%N%PwbW-M`;v1}J0*@d3Rqt0ab^'
    '+HoL?`^z4$XFcn@0eqZ`xuqykc%OjXOE8mRS?M9)^KAUxzyB|29<t>'
)
_D4_DATA_CACHE = None

_EEQ2019_ATOMIC = np.array([
    (1.23695041, -0.35015861, 0.04916110, 0.55159092),
    (1.26590957, 1.04121227, 0.10937243, 0.66205886),
    (0.54341808, 0.09281243, -0.12349591, 0.90529132),
    (0.99666991, 0.09412380, -0.02665108, 1.51710827),
    (1.26691604, 0.26629137, -0.02631658, 2.86070364),
    (1.40028282, 0.19408787, 0.06005196, 1.88862966),
    (1.55819364, 0.05317918, 0.09279548, 1.32250290),
    (1.56866440, 0.03151644, 0.11689703, 1.23166285),
    (1.57540015, 0.32275132, 0.15704746, 1.77503721),
    (1.15056627, 1.30996037, 0.07987901, 1.11955204),
    (0.55936220, 0.24206510, -0.10002962, 1.28263182),
    (0.72373742, 0.04147733, -0.07712863, 1.22344336),
    (1.12910844, 0.11634126, -0.02170561, 1.70936266),
    (1.12306840, 0.13155266, -0.04964052, 1.54075036),
    (1.52672442, 0.15350650, 0.14250599, 1.38200579),
    (1.40768172, 0.15250997, 0.07126660, 2.18849322),
    (1.48154584, 0.17523529, 0.13682750, 1.36779065),
    (1.31062963, 0.28774450, 0.14877121, 1.27039703),
    (0.40374140, 0.42937314, -0.10219289, 1.64466502),
    (0.75442607, 0.01896455, -0.08979338, 1.58859404),
    (0.76482096, 0.07179178, -0.08273597, 1.65357953),
    (0.98457281, -0.01121381, -0.01754829, 1.50021521),
    (0.96702598, -0.03093370, -0.02765460, 1.30104175),
    (1.05266584, 0.02716319, -0.02558926, 1.46301827),
    (0.93274875, -0.01843812, -0.08010286, 1.32928147),
    (1.04025281, -0.15270393, -0.04163215, 1.02766713),
    (0.92738624, -0.09192645, -0.09369631, 1.02291377),
    (1.07419210, -0.13418723, -0.03774117, 0.94343886),
    (1.07900668, -0.09861139, -0.05759708, 1.14881311),
    (1.04712861, 0.18338109, 0.02431998, 1.47080755),
    (1.15018618, 0.08299615, -0.01056270, 1.76901636),
    (1.15388455, 0.11370033, -0.02692862, 1.98724061),
    (1.36313743, 0.19005278, 0.07657769, 2.41244711),
    (1.36485106, 0.10980677, 0.06561608, 2.26739524),
    (1.39801837, 0.12327841, 0.08006749, 2.95378999),
    (1.18695346, 0.25345554, 0.14139200, 1.20807752),
    (0.36273870, 0.58615231, -0.05351029, 1.65941046),
    (0.58797255, 0.16093861, -0.06701705, 1.62733880),
    (0.71961946, 0.04548530, -0.07377246, 1.61344972),
    (0.96158233, -0.02478645, -0.02927768, 1.63220728),
    (0.89585296, 0.01909943, -0.03867291, 1.60899928),
    (0.81360499, 0.01402541, -0.06929825, 1.43501286),
    (1.00794665, -0.03595279, -0.04485293, 1.54559205),
    (0.92613682, 0.01137752, -0.04800824, 1.32663678),
    (1.09152285, -0.03697213, -0.01484022, 1.37644152),
    (1.14907070, 0.08009416, 0.07917502, 1.36051851),
    (1.13508911, 0.02274892, 0.06619243, 1.23395526),
    (1.08853785, 0.12801822, 0.02434095, 1.65734544),
    (1.11005982, -0.02078702, -0.01505548, 1.53895240),
    (1.12452195, 0.05284319, -0.03030768, 1.97542736),
    (1.21642129, 0.07581190, 0.01418235, 1.97636542),
    (1.36507125, 0.09663758, 0.08953411, 2.05432381),
    (1.40340000, 0.09547417, 0.08967527, 3.80138135),
    (1.16653482, 0.07803344, 0.07277771, 1.43893803),
    (0.34125098, 0.64913257, -0.02129476, 1.75505957),
    (0.58884173, 0.15348654, -0.06188828, 1.59815118),
    (0.68441115, 0.05054344, -0.06568203, 1.76401732),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.56999999, 0.11000000, -0.11000000, 1.63999999),
    (0.87936784, -0.02786741, -0.03585873, 1.47055223),
    (1.02761808, 0.01057858, -0.03132400, 1.81127084),
    (0.93297476, -0.03892226, -0.05902379, 1.40189963),
    (1.10172128, -0.04574364, -0.02827592, 1.54015481),
    (0.97350071, -0.03874080, -0.07606260, 1.33721475),
    (1.16695666, -0.03782372, -0.02123839, 1.57165422),
    (1.23997927, -0.07046855, 0.03814822, 1.04815857),
    (1.18464453, 0.09546597, 0.02146834, 1.78342098),
    (1.14191734, 0.21953269, 0.01580538, 2.79106396),
    (1.12334192, 0.02522348, -0.00894298, 1.78160840),
    (1.01485321, 0.15263050, -0.05864876, 2.47588882),
    (1.12950808, 0.08042611, -0.01817842, 2.37670734),
    (1.30804834, 0.01878626, 0.07721851, 1.76613217),
    (1.33689961, 0.08715453, 0.07936083, 2.66172302),
    (1.27465977, 0.10500484, 0.05849285, 2.82773085),
    (1.06598299, 0.10034731, 0.00013506, 1.04059593),
    (0.68184178, 0.15801991, -0.00020631, 0.60550051),
    (1.04581665, -0.00071039, 0.00473118, 1.22262145),
    (1.09888688, -0.00170887, 0.01590519, 1.28736399),
    (1.07206461, -0.00133327, 0.00369763, 1.44431317),
    (1.09821942, -0.00104386, 0.00417543, 1.29032833),
    (1.10900303, -0.00094936, 0.00706682, 1.41009404),
    (1.01039812, -0.00111390, 0.00488679, 1.25501213),
    (1.00095966, -0.00125257, 0.00505103, 1.15181468),
    (1.11003303, -0.00095936, 0.00710682, 1.42010424),
    (1.16831853, -0.00102814, 0.00463050, 1.43955530),
    (1.00887482, -0.00104450, 0.00387799, 1.28565237),
    (1.05928842, -0.00112666, 0.00296795, 1.35017463),
    (1.07672363, -0.00101529, 0.00400648, 1.33011749),
    (1.11308426, -0.00059592, 0.00548481, 1.30745135),
    (1.14340090, -0.00012585, 0.01350400, 1.26526071),
    (1.13714110, -0.00140896, 0.00675380, 1.34071499),
])

_EEQ2019_COVALENT = np.array([
    0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63,
    0.64, 0.67, 1.40, 1.25, 1.13, 1.04, 1.10, 1.02,
    0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21, 1.10,
    1.07, 1.04, 1.00, 0.99, 1.01, 1.09, 1.12, 1.09,
    1.15, 1.10, 1.14, 1.17, 1.89, 1.67, 1.47, 1.39,
    1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23,
    1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76,
    1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51, 1.52,
    1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46, 1.37,
    1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32,
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42, 2.01, 1.81,
    1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49, 1.49,
    1.51, 1.51, 1.48, 1.50, 1.56, 1.58, 1.45, 1.41,
    1.34, 1.29, 1.27, 1.21, 1.16, 1.15, 1.09, 1.22,
    1.36, 1.43, 1.46, 1.58, 1.48, 1.57,
])

# EEQ2019 and covalent-radii tables: grimme-lab/multicharge
# 65d73577ed398a9cbafa6ee964effab3900f66d2 and mctc-lib
# 5901f69c61834c8f034bf32383da0e0621071000, Apache-2.0.
# Licensed under the Apache License, Version 2.0;
# http://www.apache.org/licenses/LICENSE-2.0
# Distributed AS IS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.

def _d4_reference_data():
    global _D4_DATA_CACHE
    if _D4_DATA_CACHE is None:
        import base64
        import json
        import zlib
        data = json.loads(zlib.decompress(base64.b85decode(_D4_DATA_ENCODED)))
        hardness = np.asarray(data['hardness'])
        zeff = np.asarray(data['zeff'])
        cn = np.zeros((103, 7))
        charge = np.zeros_like(cn)
        multiplicity = np.zeros_like(cn, dtype=int)
        alpha = np.zeros((103, 7, 23))
        for element, rows in enumerate(data['references']):
            if rows is None:
                continue
            rounded = [min(int(np.floor(row[1] + 0.5)), 19) for row in rows]
            for ref, row in enumerate(rows):
                covcn, _, qref, qsecondary, hydrogens, scaling, system, raw = row
                cn[element, ref] = covcn
                charge[element, ref] = qref
                repeats = rounded.count(rounded[ref]) + int(rounded[ref] == 0)
                multiplicity[element, ref] = repeats * (repeats + 1) // 2
                if not system:
                    continue
                system = int(system)
                secondary_scale, secondary_alpha = data['secondary'][str(system)]
                reference_z = zeff[system - 1]
                effective = qsecondary + reference_z
                zeta = (np.exp(3.0) if effective <= 0.0 else
                        np.exp(3.0 * (1.0 - np.exp(2.0 * hardness[system - 1]
                               * (1.0 - reference_z / effective)))))
                secondary = secondary_scale * np.asarray(secondary_alpha) * zeta
                alpha[element, ref] = np.maximum(
                    scaling * (np.asarray(raw) - hydrogens * secondary), 0.0)
        _D4_DATA_CACHE = (cn, charge, multiplicity, alpha, hardness, zeff,
                          np.asarray(data['en']), np.asarray(data['r4r2']))
    return _D4_DATA_CACHE


def _d4_surrogate_remainder(atoms, cartesian_map, scale):
    """D4 EEQ pair-dispersion component beyond its local quadratic jet."""
    from scipy.special import erf, erfc
    dimension = cartesian_map.shape[1]
    numbers = atoms.numbers
    count = len(atoms)
    if (atoms.pbc.any() or count < 2
            or np.any((numbers < 1) | (numbers > 103))):
        return lambda y: (0.0, np.zeros(dimension))
    cn_table, q_table, ngw_table, alpha_table, eta_table, z_table, en_table, r_table = (
        _d4_reference_data())
    index = numbers - 1
    reference_cn = cn_table[index]
    reference_q = q_table[index]
    multiplicity = ngw_table[index]
    reference_alpha = alpha_table[index]
    if np.any(np.sum(multiplicity, axis=1) == 0):
        return lambda y: (0.0, np.zeros(dimension))
    effective_z = z_table[index]
    charge_steepness = 2.0 * eta_table[index]
    gaussian_order = np.arange(1, int(np.max(multiplicity)) + 1)
    gaussian_mask = gaussian_order[None, None, :] <= multiplicity[:, :, None]
    chi, hardness, cn_shift, width = _EEQ2019_ATOMIC[index].T
    radii = (4.0 / 3.0) * _EEQ2019_COVALENT[index] / units.Bohr
    first, second = np.triu_indices(count, 1)
    pair_origin = (atoms.positions[first] - atoms.positions[second]) / units.Bohr
    mapping = cartesian_map[:3 * count].reshape(count, 3, dimension) / units.Bohr
    pair_map = mapping[first] - mapping[second]
    radius_sum = radii[first] + radii[second]
    beta = 1.0 / np.sqrt(width[first]**2 + width[second]**2)
    diagonal = hardness + np.sqrt(2.0 / np.pi) / width
    en = en_table[index]
    en_factor = 4.10451 * np.exp(-(np.abs(en[first] - en[second]) + 19.08857)**2
                               / (2.0 * 11.28174**2))
    r4r2 = np.sqrt(0.5 * np.sqrt(numbers) * r_table[index])
    ratio = 3.0 * r4r2[first] * r4r2[second]
    damping = 0.38574991 * np.sqrt(ratio) + 4.80688534
    frequency = np.array([0.000001, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9,
                          1., 1.2, 1.4, 1.6, 1.8, 2., 2.5, 3., 4., 5., 7.5, 10.])
    quadrature = np.zeros(23)
    quadrature[:-1] += 0.5 * np.diff(frequency)
    quadrature[1:] += 0.5 * np.diff(frequency)
    quadrature *= 3.0 / np.pi
    conversion = units.Hartree / scale

    def evaluate(y):
        pairs = pair_origin + np.einsum('pij,j->pi', pair_map, y)
        radius = np.sqrt(np.sum(pairs * pairs, axis=1))
        if np.any(radius.real <= 0.0) or not np.all(np.isfinite(radius)):
            raise ValueError('Invalid D4 pair geometry')
        argument = 7.5 * (radius / radius_sum - 1.0)
        support = radius.real <= 25.0
        occupation = np.where(support, 0.5 * erfc(argument), 0.0)
        occupation_slope = np.where(support, -7.5 * np.exp(-argument**2)
                                   / (np.sqrt(np.pi) * radius_sum), 0.0)
        raw_cn = np.zeros(count, dtype=pairs.dtype)
        cov_cn = np.zeros_like(raw_cn)
        for endpoint in (first, second):
            np.add.at(raw_cn, endpoint, occupation)
            np.add.at(cov_cn, endpoint, en_factor * occupation)
        exponential = np.exp(8.0 - raw_cn)
        soft_slope = exponential / (1.0 + exponential)
        capped_cn = np.log1p(np.exp(8.0)) - np.log1p(exponential)
        regularized = capped_cn + 1e-14
        linear = -chi + cn_shift * capped_cn / np.sqrt(regularized)
        linear_slope = (cn_shift * (0.5 * capped_cn + 1e-14)
                        / regularized**1.5 * soft_slope)
        kernel = erf(beta * radius) / radius
        kernel_slope = (2.0 * beta * np.exp(-(beta * radius)**2)
                        / (np.sqrt(np.pi) * radius) - kernel / radius)
        matrix = np.zeros((count + 1, count + 1), dtype=pairs.dtype)
        matrix[np.arange(count), np.arange(count)] = diagonal
        matrix[first, second] = matrix[second, first] = kernel
        matrix[:count, count] = matrix[count, :count] = 1.0
        rhs = np.concatenate((linear, np.zeros(1, dtype=pairs.dtype)))
        try:
            charges = np.linalg.solve(matrix, rhs)[:count]
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular D4 EEQ response') from exc
        difference = cov_cn[:, None] - reference_cn
        logits = -6.0 * gaussian_order * difference[:, :, None]**2
        logits = np.where(gaussian_mask, logits, -np.inf)
        shift = np.max(logits.real, axis=(1, 2), keepdims=True)
        gaussian = np.exp(logits - shift)
        gaussian /= np.sum(gaussian, axis=(1, 2), keepdims=True)
        log_slope = -12.0 * gaussian_order * difference[:, :, None]
        slope_mean = np.sum(gaussian * log_slope, axis=(1, 2), keepdims=True)
        weights_cn = np.sum(gaussian, axis=2)
        derivatives_cn = np.sum(gaussian * (log_slope - slope_mean), axis=2)
        qmod = charges + effective_z
        positive_charge = qmod.real > 0.0
        safe_charge = np.where(positive_charge, qmod, 1.0)
        qref = reference_q + effective_z[:, None]
        inner = np.exp(charge_steepness[:, None] * (1.0 - qref / safe_charge[:, None]))
        scaling = np.where(positive_charge[:, None], np.exp(3.0 * (1.0 - inner)),
                           np.exp(3.0))
        scaling_slope = np.where(positive_charge[:, None],
                                -3.0 * charge_steepness[:, None] * inner * scaling
                                * qref / safe_charge[:, None]**2, 0.0)
        polar = np.einsum('nr,nrf->nf', weights_cn * scaling, reference_alpha)
        polar_cn = np.einsum('nr,nrf->nf', derivatives_cn * scaling, reference_alpha)
        polar_q = np.einsum('nr,nrf->nf', weights_cn * scaling_slope, reference_alpha)
        c6 = np.einsum('pf,f,pf->p', polar[first], quadrature, polar[second])
        inverse6 = 1.0 / (radius**6 + damping**6)
        inverse8 = 1.0 / (radius**8 + damping**8)
        potential = inverse6 + 0.95948085 * ratio * inverse8
        energy = -np.sum(c6 * potential)
        derivative_cn = np.zeros(count, dtype=pairs.dtype)
        derivative_q = np.zeros_like(derivative_cn)
        for endpoint, other in ((first, second), (second, first)):
            np.add.at(derivative_cn, endpoint, -potential * np.einsum(
                'pf,f,pf->p', polar_cn[endpoint], quadrature, polar[other]))
            np.add.at(derivative_q, endpoint, -potential * np.einsum(
                'pf,f,pf->p', polar_q[endpoint], quadrature, polar[other]))
        try:
            adjoint = np.linalg.solve(matrix, np.concatenate(
                (derivative_q, np.zeros(1, dtype=pairs.dtype))))[:count]
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular D4 charge derivative') from exc
        radial = (c6 * (6.0 * radius**5 * inverse6**2
                        + 0.95948085 * ratio * 8.0 * radius**7 * inverse8**2)
                  + (derivative_cn[first] + derivative_cn[second])
                  * en_factor * occupation_slope
                  + (adjoint[first] * linear_slope[first]
                     + adjoint[second] * linear_slope[second]) * occupation_slope
                  - (adjoint[first] * charges[second]
                     + adjoint[second] * charges[first]) * kernel_slope)
        gradient = np.einsum('p,pi,pij->j', radial / radius, pairs, pair_map)
        if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
            raise ValueError('Nonfinite D4 response')
        return energy * conversion, gradient * conversion

    value0, gradient0 = evaluate(np.zeros(dimension))
    hessian0 = np.column_stack([
        evaluate(1e-20j * axis)[1].imag / 1e-20 for axis in np.eye(dimension)])
    hessian0 = 0.5 * (hessian0 + hessian0.T)
    if not np.all(np.isfinite(hessian0)):
        raise ValueError('Nonfinite D4 origin curvature')

    def remainder(y):
        value, gradient = evaluate(y)
        return (float(value - value0 - gradient0 @ y - 0.5 * y @ hessian0 @ y),
                gradient - gradient0 - hessian0 @ y)
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
            micro_dispersion = _d4_surrogate_remainder(
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
