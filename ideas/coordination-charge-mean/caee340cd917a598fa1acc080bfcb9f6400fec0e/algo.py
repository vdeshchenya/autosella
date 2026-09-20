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

# EEQBC2025 parameters from grimme-lab/multicharge, commit
# 65d73577ed398a9cbafa6ee964effab3900f66d2, param/eeqbc2025.f90.
# Pauling EN and pair vdW radii from grimme-lab/mctc-lib, commit
# 5901f69c61834c8f034bf32383da0e0621071000, data/paulingen.f90 and data/vdwrad.f90.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these tables except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Tables reformatted from Fortran to Python.
_EEQBC2025_ATOMIC = np.array([
    (1.7500687479, 0.3572813340, 0.4537866920, 1.3415783494, 0.7122604774, 1.8222099473, 3.4358731613, 1.1980006149, 0.3921100000),
    (0.7992983109, 14.1713349136, 0.8971879958, 2.4226307746, -1.7351284097, -0.2575679643, 0.2563012350, 2.2610217725, 0.0810600000),
    (0.8817302909, -0.0335574715, 0.3987756594, 0.0910702713, 3.0089829052, 0.4393826724, 1.7336935111, 2.3787175190, 0.9910100000),
    (1.2122559922, -2.2617753890, 0.2435934990, -0.2802662922, 2.1166762050, 1.1102162003, 1.4252599447, 2.4632164676, 0.7499500000),
    (1.4042606312, -2.9993990603, 0.2119711703, -0.0464303067, 1.5179774898, 1.2310070946, 1.9821377790, 2.4613895807, 1.1543700000),
    (1.7373300176, -2.8456422314, 0.2064066867, 0.3049790613, 1.2180269092, 0.9818102022, 7.9575330990, 2.6763007964, 1.6691400000),
    (1.9224220861, -2.2316836385, 0.2398313485, 0.5014914830, 1.0873609014, 0.1502230497, 5.2650283958, 2.7655085211, 1.4250300000),
    (2.0295674708, -0.9048085573, 0.3482853216, 0.7131712513, 0.8994075937, 0.4134119032, 5.3394223720, 2.6466398902, 0.8718100000),
    (2.0914017724, -3.3402942035, 0.1479057386, 1.5978006993, 0.1658248786, 2.5030512016, 4.7702507597, 2.0647114131, 0.6334000000),
    (0.2783743672, 11.6677100883, 1.4433940527, 4.6934800245, -2.5747028940, -0.4998596384, 0.5095753028, 2.2964278893, 0.0876700000),
    (0.7909141712, 0.0461110187, 0.6317031456, -0.2311835622, 3.1762170214, 2.1023399046, 5.7961811482, 3.0473595746, 0.8740600000),
    (0.9333749946, -0.1623149426, 0.7152255265, -0.5722047540, 2.3987338612, 1.1266337899, 2.8738819069, 3.3597126173, 0.8754800000),
    (1.1280735350, -0.1976009198, 0.6920759433, -0.1872404228, 2.2469063726, 1.3785272689, 1.5730116016, 2.9413551863, 1.2147200000),
    (1.3504642320, -3.6156182254, 0.1952261525, 0.1355861183, 1.5639940746, 0.9471745876, 0.7813507196, 3.3593150082, 1.1335000000),
    (1.7084529806, -4.8040123811, 0.1478738486, 0.5037598487, 1.2412557993, 1.6601128471, 1.0337776163, 3.7124038217, 1.6890600000),
    (1.9657999323, -5.8989254120, 0.1173410276, 0.8257488249, 1.6283237163, -0.0156796346, 1.4123845734, 3.7950496861, 1.0221600000),
    (1.8796814465, -1.7918672558, 0.2188836429, 1.5828922925, 1.5628790844, 0.6525286877, 3.0340296817, 3.6218412465, 0.5386400000),
    (0.8120477849, 3.2077831067, 0.7265491450, 5.6324196990, -0.9249536928, -2.8148211211, 0.5326667425, 2.3368507550, 0.0827800000),
    (0.6229777212, 0.4598658365, 1.0062576628, -1.3574661808, 3.0733040004, 1.8730352397, 6.4794438076, 3.2678005729, 1.4096300000),
    (0.8955669337, -0.3196730368, 0.6529550574, -0.7114730764, 2.7596745507, 0.4148795713, 4.1572236543, 2.6934639460, 1.1954700000),
    (0.8887941055, -0.0066012997, 1.0787300626, -0.8412840531, 2.9366708989, 1.9811917137, 2.6197028418, 3.0942813806, 1.5142100000),
    (0.9249293933, -0.0650415781, 1.0194369772, -0.8100781799, 2.7004746183, 1.3666346630, 1.9926557922, 3.1994190934, 1.7892000000),
    (0.8910306356, 0.0116105065, 0.7673907688, -0.7321477749, 2.2295030415, 0.4773540249, 1.4258893003, 3.1865351525, 2.0646100000),
    (0.8730274586, -0.2020240365, 0.8234907812, -0.5690936866, 2.0304690076, 0.6660383739, 3.4184301443, 3.0245247746, 1.6905600000),
    (1.0692963783, -0.0451985500, 0.7956000862, -0.7978421025, 1.9683561829, 0.4949831426, 3.1337436912, 2.9516405455, 1.6563700000),
    (1.1430792497, -0.8983846024, 0.4194926962, -0.7081664947, 2.2302711526, 0.9260098769, 4.5345735628, 2.7967091405, 1.5128400000),
    (1.2352658732, -0.5087624261, 0.6577871621, -0.5311094926, 1.8504904266, 1.4071496248, 6.3426635435, 2.8624752847, 1.3179000000),
    (1.2511161359, -0.9360254729, 0.4350022430, -0.5561735098, 2.0575510119, 0.7430722161, 4.8622181062, 2.9325871383, 0.9749800000),
    (1.0995052580, -0.3137611925, 0.5436327263, 0.1043470768, 2.2756603413, 1.4792830405, 3.9658581319, 2.8750457420, 0.5334600000),
    (1.0059572004, 0.3714666864, 1.2387687941, -0.2459258932, 2.2094576537, 1.4211880229, 2.4205042838, 3.2880254556, 0.6585000000),
    (1.0390725738, -0.5637510788, 0.5125789654, -0.2244771250, 2.1544064368, 0.6613271421, 2.0153453160, 3.4129389757, 0.9696500000),
    (1.2885924052, -1.5811888792, 0.3834386963, 0.0378446029, 1.9327504630, 1.3109487181, 1.3655709456, 3.5547538315, 1.0083100000),
    (1.4638654613, -2.5680164043, 0.2781070074, 0.2939641775, 1.4451438826, 0.9539967321, 1.0879161652, 3.8824044195, 1.0871000000),
    (1.7797597799, -3.3791525742, 0.2053677667, 0.7336233202, 1.4813741556, 0.0441858334, 0.8125045161, 4.1349986852, 0.8222200000),
    (1.6400765990, -0.9039263250, 0.3191301456, 1.1960377617, 2.0308095325, 0.8506553360, 3.4331186365, 3.9489265588, 0.5449300000),
    (0.8859889377, 2.6191171553, 3.4957602962, 1.5974038323, 0.4032186085, -0.7778128954, 1.1410555369, 3.6264077864, 0.1647100000),
    (0.5142094052, 0.4517188832, 0.8847073217, -0.5630850954, 3.6036894994, 2.4456255294, 5.3302096260, 3.8936921777, 1.2490800000),
    (0.8785352464, -0.4737572247, 0.6739335178, -1.1059510466, 2.6513413398, 0.6279760783, 8.9866820455, 3.5213571939, 1.2198700000),
    (0.9716967887, -0.3291918172, 0.8092111775, -0.7830773028, 2.6634586616, 0.8504097502, 8.0879982654, 3.2417303558, 1.5657400000),
    (0.8109573582, -0.0641706161, 0.8229663676, -0.9114834757, 2.3940154835, 0.1275277215, 1.3505819625, 3.5464510864, 1.8697600000),
    (0.9361297862, -0.4365721167, 0.7341667740, -0.4093603622, 2.3527262731, 1.0244946467, 1.9761405818, 3.6460575764, 1.8947900000),
    (0.9872048394, -0.1388382729, 0.8802988629, -0.2717170095, 2.0735381213, 0.3991961865, 4.8306789723, 3.4102131328, 1.7085000000),
    (1.1290914832, 0.0445179428, 1.1234870897, -0.4691579275, 1.7234564437, 0.3007399180, 2.6167089975, 3.4469009914, 1.5521300000),
    (1.0416755409, -0.3077776724, 0.5654595735, -0.2257381361, 2.2302635382, 0.8892405348, 4.9413659163, 3.3043609242, 1.4903300000),
    (1.1579060755, -0.1421769591, 0.7749739189, -0.1375984198, 2.1871313764, 1.0358999274, 5.5889636514, 3.3574938698, 1.3177400000),
    (1.1371382461, -0.3718332953, 0.6091511140, 0.3330053570, 1.8061408427, 0.5910349581, 3.7289038580, 3.4236287446, 0.6991700000),
    (1.1490154759, -0.9003899901, 0.4788100227, 0.0221109296, 1.9051691947, 1.3306044793, 2.2978010245, 3.6341385880, 0.5528200000),
    (1.0811257447, -0.5034953355, 0.6104947355, -0.0920402467, 2.0424482278, 1.0116510919, 2.9915912946, 3.8313216784, 0.6642200000),
    (1.0201038561, -0.3154724874, 0.6518973596, -0.3096506887, 2.8036578365, 1.2017335753, 3.2084006372, 3.8271624151, 0.9069800000),
    (1.2317318949, -1.2061278491, 0.4348284778, 0.0088013637, 2.0783981020, 1.0749481071, 2.4592286766, 4.1254263093, 1.0976200000),
    (1.2546053590, -1.0351395610, 0.4885595700, 0.0730363100, 2.0481231960, 1.5278450966, 1.0482227697, 3.9895425056, 1.2183000000),
    (1.6136334955, -2.4727516433, 0.2660054523, 0.4356094483, 1.8544101088, 0.3830852785, 1.4124670516, 4.5341418141, 0.7321900000),
    (1.5949826440, -0.5377076044, 0.4274914591, 1.0199146044, 2.1888387015, 0.8039617911, 2.0699368746, 4.6758001321, 0.5498700000),
    (0.8548714270, 2.1647776210, 2.3114324559, 1.0092039203, 0.5779869189, -1.6689377641, 2.3426022325, 4.3188262237, 0.2467100000),
    (0.5116591821, 0.3592585022, 0.9734795056, -0.7528024837, 3.2064625646, 1.3153512507, 4.9766316345, 2.8314213049, 1.5680600000),
    (0.8221154800, -0.6373016543, 0.6329900422, -1.1365506475, 2.7406551784, 0.6850807472, 4.7445931148, 4.8914010315, 1.1677300000),
    (0.8992637384, -0.1481956999, 1.0109847900, -0.9661197708, 2.5529621630, 0.4068053082, 7.6556126582, 3.6004533315, 1.6642500000),
    (0.7835477700, -0.4595916155, 0.6287499845, -1.1514088354, 2.5391757608, 0.2805275842, 2.2792827162, 2.8758092017, 1.6032600000),
    (0.6865502434, -0.6048435529, 0.5401093486, -1.1092964223, 2.4348800350, 0.2612355874, 2.2265798615, 2.9967129499, 1.6032600000),
    (0.6063027416, -0.7208619618, 0.4679527826, -1.0718762355, 2.3484586230, 0.2457254002, 2.2270872929, 3.1042023757, 1.6032600000),
    (0.5428052646, -0.8076468424, 0.4122802864, -1.0391482749, 2.2799115250, 0.2339970224, 2.2808050104, 3.1982774792, 1.6032600000),
    (0.4960578124, -0.8651981945, 0.3730918601, -1.0111125406, 2.2292387408, 0.2260504541, 2.3877330140, 3.2789382603, 1.6032600000),
    (0.4660603851, -0.8935160183, 0.3503875036, -0.9877690325, 2.1964402705, 0.2218856952, 2.5478713036, 3.3461847191, 1.6032600000),
    (0.4528129826, -0.8926003136, 0.3441672169, -0.9691177507, 2.1815161141, 0.2215027459, 2.7612198793, 3.4000168555, 1.6032600000),
    (0.4563156049, -0.8624510805, 0.3544310001, -0.9551586951, 2.1844662716, 0.2249016061, 3.0277787411, 3.4404346695, 1.6032600000),
    (0.4765682521, -0.8030683191, 0.3811788531, -0.9458918658, 2.2052907430, 0.2320822757, 3.3475478889, 3.4674381612, 1.6032600000),
    (0.5135709241, -0.7144520292, 0.4244107759, -0.9413172627, 2.2439895282, 0.2430447548, 3.7205273228, 3.4810273305, 1.6032600000),
    (0.5673236209, -0.5966022109, 0.4841267686, -0.9414348859, 2.3005626274, 0.2577890434, 4.1467170428, 3.4812021775, 1.6032600000),
    (0.6378263425, -0.4495188642, 0.5603268311, -0.9462447354, 2.3750100404, 0.2763151415, 4.6261170489, 3.4679627021, 1.6032600000),
    (0.7250790890, -0.2732019891, 0.6530109634, -0.9557468111, 2.4673317674, 0.2986230491, 5.1587273410, 3.4413089043, 1.6032600000),
    (0.8290818603, -0.0676515856, 0.7621791656, -0.9699411130, 2.5775278082, 0.3247127662, 5.7445479192, 3.4012407842, 1.6032600000),
    (0.8697550816, -0.1339322663, 1.0577606985, -0.9467711075, 2.6463737671, 0.9329386915, 1.9450532464, 3.5004027339, 1.8191000000),
    (1.0442533196, -0.7103642117, 0.6844888492, -0.5854657957, 2.3987259080, 1.1124975975, 1.2082681633, 3.6576246465, 1.8175100000),
    (1.1429836348, -0.1700796179, 0.9102124518, -0.1956906192, 2.0862161326, 0.3105056463, 5.4761913827, 3.2722427492, 1.6802300000),
    (1.1622493128, -0.1362891699, 0.8550543040, -0.3841246137, 1.8045334538, 0.2119489274, 2.8688258387, 3.4847840299, 1.5224100000),
    (1.2650483683, -1.0705189016, 0.4138761210, -0.2184058724, 2.0382923920, 0.3490965682, 3.4269533511, 3.3869572767, 1.4602600000),
    (1.2650500943, -0.8229572159, 0.5593056202, -0.2071244723, 1.6579982531, 0.9303004996, 1.2827929585, 3.4600493844, 1.1110400000),
    (1.3607929134, -1.3207540081, 0.3751752813, 0.1769757167, 1.8353080915, 0.6578893166, 4.2446334525, 3.5857257632, 0.9102600000),
    (1.3186071563, -2.0554362750, 0.2949155601, 0.5363613694, 1.8450710788, 0.7625190003, 8.5466705292, 3.5138481825, 0.5218000000),
    (1.0545750683, -0.2654477885, 0.6769971683, 0.0342662426, 1.5696036105, 0.6067448860, 2.7030553995, 4.0752898970, 1.4895900000),
    (0.9074468503, -0.0736143849, 0.7124606732, -0.5074824777, 2.8136219641, 1.1098111282, 1.7482905639, 4.2705802544, 0.8441800000),
    (1.0892548243, -1.1221956034, 0.4519163133, -0.0048092213, 2.3784572290, 0.9571986961, 4.5652515937, 4.3281934906, 0.9426900000),
    (1.1983441731, -0.1821999108, 1.0405678353, -0.0546120433, 1.9914691678, 1.4674965889, 2.0750200204, 4.0616856521, 1.5171900000),
    (1.3955974910, -0.7727065022, 0.6688421527, 0.0560290491, 1.8625351100, 0.7713149335, 2.1042278455, 4.3269140322, 0.7287100000),
    (1.6266506350, -0.4699768943, 0.4838599292, 0.8822097689, 2.1579257719, 0.5513455799, 2.9249818593, 4.7950102995, 0.5137000000),
    (0.9802627692, 0.6377347433, 0.9792188430, 0.9546406691, 0.6206683275, -0.7227615433, 1.1606670882, 4.0621301306, 0.2678200000),
    (0.4952498716, 0.4140010159, 0.8793273061, -1.8612818673, 3.5103871382, 1.2895674764, 5.1339954989, 4.7045604278, 1.2122500000),
    (0.7903508991, -0.2353223377, 0.8333325045, -1.2559850201, 2.7327597379, 0.5960416182, 5.4015367551, 4.3693314868, 1.5797100000),
    (0.7482689572, -0.1309097826, 0.8202868436, -0.8232940275, 2.7369312006, 0.1671277145, 1.5278253705, 3.2349557337, 1.7549800000),
    (0.8666000614, 0.1881855179, 1.7807640816, -0.7432092987, 2.6004448612, 0.1575313114, 0.7201439348, 2.0334056417, 1.7549800000),
    (0.8153381406, 0.2007222471, 1.5641357264, -0.9259469469, 2.7011486104, 0.2863965715, 0.8778110607, 2.5551666814, 1.7549800000),
    (0.7700731721, 0.1912792246, 1.3644976007, -1.0588247895, 2.7879694953, 0.4002506248, 1.0152634518, 3.0015806363, 1.7549800000),
    (0.7308051560, 0.1598564505, 1.1818497047, -1.1418428264, 2.8609075157, 0.4990934714, 1.1325011080, 3.3726475065, 1.7549800000),
    (0.6975340922, 0.1064539248, 1.0161920382, -1.1750010577, 2.9199626718, 0.5829251111, 1.2295240293, 3.6683672921, 1.7549800000),
    (0.6702599807, 0.0310716475, 0.8675246014, -1.1582994833, 2.9651349636, 0.6517455441, 1.3063322158, 3.8887399930, 1.7549800000),
    (0.6489828216, -0.0662903814, 0.7358473941, -1.0917381033, 2.9964243909, 0.7055547703, 1.3629256675, 4.0337656091, 1.7549800000),
    (0.6337026148, -0.1856321619, 0.6211604164, -0.9753169176, 3.0138309539, 0.7443527897, 1.3993043843, 4.1034441406, 1.7549800000),
    (0.6244193604, -0.3269536941, 0.5234636683, -0.8090359263, 3.0173546524, 0.7681396024, 1.4154683662, 4.0977755873, 1.7549800000),
    (0.6211330583, -0.4902549779, 0.4427571498, -0.5928951293, 3.0069954867, 0.7769152082, 1.4114176133, 4.0167599494, 1.7549800000),
    (0.6238437086, -0.6755360133, 0.3790408609, -0.3268945267, 2.9827534565, 0.7706796073, 1.3871521256, 3.8603972267, 1.7549800000),
    (0.6325513112, -0.8827968003, 0.3323148016, -0.0110341184, 2.9446285619, 0.7494327996, 1.3426719030, 3.6286874194, 1.7549800000),
    (0.6472558662, -1.1120373389, 0.3025789719, 0.3546860955, 2.8926208030, 0.7131747852, 1.2779769455, 3.3216305273, 1.7549800000),
    (0.6679573735, -1.3632576292, 0.2898333718, 0.7702661151, 2.8267301797, 0.6619055639, 1.1930672532, 2.9392265506, 1.7549800000),
])
_EEQBC2025_PAULING = np.array([
    2.20, 3.00, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44,
    3.98, 4.50, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58,
    3.16, 3.50, 0.82, 1.00, 1.36, 1.54, 1.63, 1.66,
    1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01,
    2.18, 2.55, 2.96, 3.00, 0.82, 0.95, 1.22, 1.33,
    1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69,
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60, 0.79, 0.89,
    1.10, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18, 1.20,
    1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.30,
    1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20, 0.79, 0.90,
    1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30, 1.30,
    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
    1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
])
_EEQBC2025_VDW_PAIR = np.array([
    2.1823, 1.8547, 1.7347, 2.9086, 2.5732, 3.4956, 2.3550, 2.5095,
    2.9802, 3.0982, 2.5141, 2.3917, 2.9977, 2.9484, 3.2160, 2.4492,
    2.2527, 3.1933, 3.0214, 2.9531, 2.9103, 2.3667, 2.1328, 2.8784,
    2.7660, 2.7776, 2.7063, 2.6225, 2.1768, 2.0625, 2.6395, 2.6648,
    2.6482, 2.5697, 2.4846, 2.4817, 2.0646, 1.9891, 2.5086, 2.6908,
    2.6233, 2.4770, 2.3885, 2.3511, 2.2996, 1.9892, 1.9251, 2.4190,
    2.5473, 2.4994, 2.4091, 2.3176, 2.2571, 2.1946, 2.1374, 2.9898,
    2.6397, 3.6031, 3.1219, 3.7620, 3.2485, 2.9357, 2.7093, 2.5781,
    2.4839, 3.7082, 2.5129, 2.7321, 3.1052, 3.2962, 3.1331, 3.2000,
    2.9586, 3.0822, 2.8582, 2.7120, 3.2570, 3.4839, 2.8766, 2.7427,
    3.2776, 3.2363, 3.5929, 3.2826, 3.0911, 2.9369, 2.9030, 2.7789,
    3.3921, 3.3970, 4.0106, 2.8884, 2.6605, 3.7513, 3.1613, 3.3605,
    3.3325, 3.0991, 2.9297, 2.8674, 2.7571, 3.8129, 3.3266, 3.7105,
    3.7917, 2.8304, 2.5538, 3.3932, 3.1193, 3.1866, 3.1245, 3.0465,
    2.8727, 2.7664, 2.6926, 3.4608, 3.2984, 3.5142, 3.5418, 3.5017,
    2.6190, 2.4797, 3.1331, 3.0540, 3.0651, 2.9879, 2.9054, 2.8805,
    2.7330, 2.6331, 3.2096, 3.5668, 3.3684, 3.3686, 3.3180, 3.3107,
    2.4757, 2.4019, 2.9789, 3.1468, 2.9768, 2.8848, 2.7952, 2.7457,
    2.6881, 2.5728, 3.0574, 3.3264, 3.3562, 3.2529, 3.1916, 3.1523,
    3.1046, 2.3725, 2.3289, 2.8760, 2.9804, 2.9093, 2.8040, 2.7071,
    2.6386, 2.5720, 2.5139, 2.9517, 3.1606, 3.2085, 3.1692, 3.0982,
    3.0352, 2.9730, 2.9148, 3.2147, 2.8315, 3.8724, 3.4621, 3.8823,
    3.3760, 3.0746, 2.8817, 2.7552, 2.6605, 3.9740, 3.6192, 3.6569,
    3.9586, 3.6188, 3.3917, 3.2479, 3.1434, 4.2411, 2.7597, 3.0588,
    3.3474, 3.6214, 3.4353, 3.4729, 3.2487, 3.3200, 3.0914, 2.9403,
    3.4972, 3.7993, 3.6773, 3.8678, 3.5808, 3.8243, 3.5826, 3.4156,
    3.8765, 4.1035, 2.7361, 2.9765, 3.2475, 3.5004, 3.4185, 3.4378,
    3.2084, 3.2787, 3.0604, 2.9187, 3.4037, 3.6759, 3.6586, 3.8327,
    3.5372, 3.7665, 3.5310, 3.3700, 3.7788, 3.9804, 3.8903, 2.6832,
    2.9060, 3.2613, 3.4359, 3.3538, 3.3860, 3.1550, 3.2300, 3.0133,
    2.8736, 3.4024, 3.6142, 3.5979, 3.5295, 3.4834, 3.7140, 3.4782,
    3.3170, 3.7434, 3.9623, 3.8181, 3.7642, 2.6379, 2.8494, 3.1840,
    3.4225, 3.2771, 3.3401, 3.1072, 3.1885, 2.9714, 2.8319, 3.3315,
    3.5979, 3.5256, 3.4980, 3.4376, 3.6714, 3.4346, 3.2723, 3.6859,
    3.8985, 3.7918, 3.7372, 3.7211, 2.9230, 2.6223, 3.4161, 2.8999,
    3.0557, 3.3308, 3.0555, 2.8508, 2.7385, 2.6640, 3.5263, 3.0277,
    3.2990, 3.7721, 3.5017, 3.2751, 3.1368, 3.0435, 3.7873, 3.2858,
    3.2140, 3.1727, 3.2178, 3.4414, 2.5490, 2.7623, 3.0991, 3.3252,
    3.1836, 3.2428, 3.0259, 3.1225, 2.9032, 2.7621, 3.2490, 3.5110,
    3.4429, 3.3845, 3.3574, 3.6045, 3.3658, 3.2013, 3.6110, 3.8241,
    3.7090, 3.6496, 3.6333, 3.0896, 3.5462, 2.4926, 2.7136, 3.0693,
    3.2699, 3.1272, 3.1893, 2.9658, 3.0972, 2.8778, 2.7358, 3.2206,
    3.4566, 3.3896, 3.3257, 3.2946, 3.5693, 3.3312, 3.1670, 3.5805,
    3.7711, 3.6536, 3.5927, 3.5775, 3.0411, 3.4885, 3.4421, 2.4667,
    2.6709, 3.0575, 3.2357, 3.0908, 3.1537, 2.9235, 3.0669, 2.8476,
    2.7054, 3.2064, 3.4519, 3.3593, 3.2921, 3.2577, 3.2161, 3.2982,
    3.1339, 3.5606, 3.7582, 3.6432, 3.5833, 3.5691, 3.0161, 3.4812,
    3.4339, 3.4327, 2.4515, 2.6338, 3.0511, 3.2229, 3.0630, 3.1265,
    2.8909, 3.0253, 2.8184, 2.6764, 3.1968, 3.4114, 3.3492, 3.2691,
    3.2320, 3.1786, 3.2680, 3.1036, 3.5453, 3.7259, 3.6090, 3.5473,
    3.5327, 3.0018, 3.4413, 3.3907, 3.3593, 3.3462, 2.4413, 2.6006,
    3.0540, 3.1987, 3.0490, 3.1058, 2.8643, 2.9948, 2.7908, 2.6491,
    3.1950, 3.3922, 3.3316, 3.2585, 3.2136, 3.1516, 3.2364, 3.0752,
    3.5368, 3.7117, 3.5941, 3.5313, 3.5164, 2.9962, 3.4225, 3.3699,
    3.3370, 3.3234, 3.3008, 2.4318, 2.5729, 3.0416, 3.1639, 3.0196,
    3.0843, 2.8413, 2.7436, 2.7608, 2.6271, 3.1811, 3.3591, 3.3045,
    3.2349, 3.1942, 3.1291, 3.2111, 3.0534, 3.5189, 3.6809, 3.5635,
    3.5001, 3.4854, 2.9857, 3.3897, 3.3363, 3.3027, 3.2890, 3.2655,
    3.2309, 2.8502, 2.6934, 3.2467, 3.1921, 3.5663, 3.2541, 3.0571,
    2.9048, 2.8657, 2.7438, 3.3547, 3.3510, 3.9837, 3.6871, 3.4862,
    3.3389, 3.2413, 3.1708, 3.6096, 3.6280, 3.6860, 3.5568, 3.4836,
    3.2868, 3.3994, 3.3476, 3.3170, 3.2950, 3.2874, 3.2606, 3.9579,
    2.9226, 2.6838, 3.7867, 3.1732, 3.3872, 3.3643, 3.1267, 2.9541,
    2.8505, 2.7781, 3.8475, 3.3336, 3.7359, 3.8266, 3.5733, 3.3959,
    3.2775, 3.1915, 3.9878, 3.8816, 3.5810, 3.5364, 3.5060, 3.8097,
    3.3925, 3.3348, 3.3019, 3.2796, 3.2662, 3.2464, 3.7136, 3.8619,
    2.9140, 2.6271, 3.4771, 3.1774, 3.2560, 3.1970, 3.1207, 2.9406,
    2.8322, 2.7571, 3.5455, 3.3514, 3.5837, 3.6177, 3.5816, 3.3902,
    3.2604, 3.1652, 3.7037, 3.6283, 3.5858, 3.5330, 3.4884, 3.5789,
    3.4094, 3.3473, 3.3118, 3.2876, 3.2707, 3.2521, 3.5570, 3.6496,
    3.6625, 2.7300, 2.5870, 3.2471, 3.1487, 3.1667, 3.0914, 3.0107,
    2.9812, 2.8300, 2.7284, 3.3259, 3.3182, 3.4707, 3.4748, 3.4279,
    3.4182, 3.2547, 3.1353, 3.5116, 3.9432, 3.8828, 3.8303, 3.7880,
    3.3760, 3.7218, 3.3408, 3.3059, 3.2698, 3.2446, 3.2229, 3.4422,
    3.5023, 3.5009, 3.5268, 2.6026, 2.5355, 3.1129, 3.2863, 3.1029,
    3.0108, 2.9227, 2.8694, 2.8109, 2.6929, 3.1958, 3.4670, 3.4018,
    3.3805, 3.3218, 3.2815, 3.2346, 3.0994, 3.3937, 3.7266, 3.6697,
    3.6164, 3.5730, 3.2522, 3.5051, 3.4686, 3.4355, 3.4084, 3.3748,
    3.3496, 3.3692, 3.4052, 3.3910, 3.3849, 3.3662, 2.5087, 2.4814,
    3.0239, 3.1312, 3.0535, 2.9457, 2.8496, 2.7780, 2.7828, 2.6532,
    3.1063, 3.3143, 3.3549, 3.3120, 3.2421, 3.1787, 3.1176, 3.0613,
    3.3082, 3.5755, 3.5222, 3.4678, 3.4231, 3.1684, 3.3528, 3.3162,
    3.2827, 3.2527, 3.2308, 3.2029, 3.3173, 3.3343, 3.3092, 3.2795,
    3.2452, 3.2096, 3.2893, 2.8991, 4.0388, 3.6100, 3.9388, 3.4475,
    3.1590, 2.9812, 2.8586, 2.7683, 4.1428, 3.7911, 3.8225, 4.0372,
    3.7059, 3.4935, 3.3529, 3.2492, 4.4352, 4.0826, 3.9733, 3.9254,
    3.8646, 3.9315, 3.7837, 3.7465, 3.7211, 3.7012, 3.6893, 3.6676,
    3.7736, 4.0660, 3.7926, 3.6158, 3.5017, 3.4166, 4.6176, 2.8786,
    3.1658, 3.5823, 3.7689, 3.5762, 3.5789, 3.3552, 3.4004, 3.1722,
    3.0212, 3.7241, 3.9604, 3.8500, 3.9844, 3.7035, 3.9161, 3.6751,
    3.5075, 4.1151, 4.2877, 4.1579, 4.1247, 4.0617, 3.4874, 3.9848,
    3.9280, 3.9079, 3.8751, 3.8604, 3.8277, 3.8002, 3.9981, 3.7544,
    4.0371, 3.8225, 3.6718, 4.3092, 4.4764, 2.8997, 3.0953, 3.4524,
    3.6107, 3.6062, 3.5783, 3.3463, 3.3855, 3.1746, 3.0381, 3.6019,
    3.7938, 3.8697, 3.9781, 3.6877, 3.8736, 3.6451, 3.4890, 3.9858,
    4.1179, 4.0430, 3.9563, 3.9182, 3.4002, 3.8310, 3.7716, 3.7543,
    3.7203, 3.7053, 3.6742, 3.8318, 3.7631, 3.7392, 3.9892, 3.7832,
    3.6406, 4.1701, 4.3016, 4.2196, 2.8535, 3.0167, 3.3978, 3.5363,
    3.5393, 3.5301, 3.2960, 3.3352, 3.1287, 2.9967, 3.6659, 3.7239,
    3.8070, 3.7165, 3.6368, 3.8162, 3.5885, 3.4336, 3.9829, 4.0529,
    3.9584, 3.9025, 3.8607, 3.3673, 3.7658, 3.7035, 3.6866, 3.6504,
    3.6339, 3.6024, 3.7708, 3.7283, 3.6896, 3.9315, 3.7250, 3.5819,
    4.1457, 4.2280, 4.1130, 4.0597, 3.0905, 2.7998, 3.6448, 3.0739,
    3.2996, 3.5262, 3.2559, 3.0518, 2.9394, 2.8658, 3.7514, 3.2295,
    3.5643, 3.7808, 3.6931, 3.4723, 3.3357, 3.2429, 4.0280, 3.5589,
    3.4636, 3.4994, 3.4309, 3.6177, 3.2946, 3.2376, 3.2050, 3.1847,
    3.1715, 3.1599, 3.5555, 3.8111, 3.7693, 3.5718, 3.4498, 3.3662,
    4.1608, 3.7417, 3.6536, 3.6154, 3.8596, 3.0301, 2.7312, 3.5821,
    3.0473, 3.2137, 3.4679, 3.1975, 2.9969, 2.8847, 2.8110, 3.6931,
    3.2076, 3.4943, 3.5956, 3.6379, 3.4190, 3.2808, 3.1860, 3.9850,
    3.5105, 3.4330, 3.3797, 3.4155, 3.6033, 3.2737, 3.2145, 3.1807,
    3.1596, 3.1461, 3.1337, 3.4812, 3.6251, 3.7152, 3.5201, 3.3966,
    3.3107, 4.1128, 3.6899, 3.6082, 3.5604, 3.7834, 3.7543, 2.9189,
    2.6777, 3.4925, 2.9648, 3.1216, 3.2940, 3.0975, 2.9757, 2.8493,
    2.7638, 3.6085, 3.1214, 3.4006, 3.4793, 3.5147, 3.3806, 3.2356,
    3.1335, 3.9144, 3.4183, 3.3369, 3.2803, 3.2679, 3.4871, 3.1714,
    3.1521, 3.1101, 3.0843, 3.0670, 3.0539, 3.3890, 3.5086, 3.5895,
    3.4783, 3.3484, 3.2559, 4.0422, 3.5967, 3.5113, 3.4576, 3.6594,
    3.6313, 3.5690, 2.8578, 2.6334, 3.4673, 2.9245, 3.0732, 3.2435,
    3.0338, 2.9462, 2.8143, 2.7240, 3.5832, 3.0789, 3.3617, 3.4246,
    3.4505, 3.3443, 3.1964, 3.0913, 3.8921, 3.3713, 3.2873, 3.2281,
    3.2165, 3.4386, 3.1164, 3.1220, 3.0761, 3.0480, 3.0295, 3.0155,
    3.3495, 3.4543, 3.5260, 3.4413, 3.3085, 3.2134, 4.0170, 3.5464,
    3.4587, 3.4006, 3.6027, 3.5730, 3.4945, 3.4623, 2.8240, 2.5960,
    3.4635, 2.9032, 3.0431, 3.2115, 2.9892, 2.9148, 2.7801, 2.6873,
    3.5776, 3.0568, 3.3433, 3.3949, 3.4132, 3.3116, 3.1616, 3.0548,
    3.8859, 3.3719, 3.2917, 3.2345, 3.2274, 3.4171, 3.1293, 3.0567,
    3.0565, 3.0274, 3.0087, 2.9939, 3.3293, 3.4249, 3.4902, 3.4091,
    3.2744, 3.1776, 4.0078, 3.5374, 3.4537, 3.3956, 3.5747, 3.5430,
    3.4522, 3.4160, 3.3975, 2.8004, 2.5621, 3.4617, 2.9154, 3.0203,
    3.1875, 2.9548, 2.8038, 2.7472, 2.6530, 3.5736, 3.0584, 3.3304,
    3.3748, 3.3871, 3.2028, 3.1296, 3.0214, 3.8796, 3.3337, 3.2492,
    3.1883, 3.1802, 3.4050, 3.0756, 3.0478, 3.0322, 3.0323, 3.0163,
    3.0019, 3.3145, 3.4050, 3.4656, 3.3021, 3.2433, 3.1453, 3.9991,
    3.5017, 3.4141, 3.3520, 3.5583, 3.5251, 3.4243, 3.3851, 3.3662,
    3.3525, 2.7846, 2.5324, 3.4652, 2.8759, 3.0051, 3.1692, 2.9273,
    2.7615, 2.7164, 2.6212, 3.5744, 3.0275, 3.3249, 3.3627, 3.3686,
    3.1669, 3.0584, 2.9915, 3.8773, 3.3099, 3.2231, 3.1600, 3.1520,
    3.4023, 3.0426, 3.0099, 2.9920, 2.9809, 2.9800, 2.9646, 3.3068,
    3.3930, 3.4486, 3.2682, 3.1729, 3.1168, 3.9952, 3.4796, 3.3901,
    3.3255, 3.5530, 3.5183, 3.4097, 3.3683, 3.3492, 3.3360, 3.3308,
    2.5424, 2.6601, 3.2555, 3.2807, 3.1384, 3.1737, 2.9397, 2.8429,
    2.8492, 2.7225, 3.3875, 3.4910, 3.4520, 3.3608, 3.3036, 3.2345,
    3.2999, 3.1487, 3.7409, 3.8392, 3.7148, 3.6439, 3.6182, 3.1753,
    3.5210, 3.4639, 3.4265, 3.4075, 3.3828, 3.3474, 3.4071, 3.3754,
    3.3646, 3.3308, 3.4393, 3.2993, 3.8768, 3.9891, 3.8310, 3.7483,
    3.3417, 3.3019, 3.2250, 3.1832, 3.1578, 3.1564, 3.1224, 3.4620,
    2.9743, 2.8058, 3.4830, 3.3474, 3.6863, 3.3617, 3.1608, 3.0069,
    2.9640, 2.8427, 3.5885, 3.5219, 4.1314, 3.8120, 3.6015, 3.4502,
    3.3498, 3.2777, 3.8635, 3.8232, 3.8486, 3.7215, 3.6487, 3.4724,
    3.5627, 3.5087, 3.4757, 3.4517, 3.4423, 3.4139, 4.1028, 3.8388,
    3.6745, 3.5562, 3.4806, 3.4272, 4.0182, 3.9991, 4.0007, 3.9282,
    3.7238, 3.6498, 3.5605, 3.5211, 3.5009, 3.4859, 3.4785, 3.5621,
    4.2623, 3.0775, 2.8275, 4.0181, 3.3385, 3.5379, 3.5036, 3.2589,
    3.0804, 3.0094, 2.9003, 4.0869, 3.5088, 3.9105, 3.9833, 3.7176,
    3.5323, 3.4102, 3.3227, 4.2702, 4.0888, 3.7560, 3.7687, 3.6681,
    3.6405, 3.5569, 3.4990, 3.4659, 3.4433, 3.4330, 3.4092, 3.8867,
    4.0190, 3.7961, 3.6412, 3.5405, 3.4681, 4.3538, 4.2136, 3.9381,
    3.8912, 3.9681, 3.7909, 3.6774, 3.6262, 3.5999, 3.5823, 3.5727,
    3.5419, 4.0245, 4.1874, 3.0893, 2.7917, 3.7262, 3.3518, 3.4241,
    3.5433, 3.2773, 3.0890, 2.9775, 2.9010, 3.8048, 3.5362, 3.7746,
    3.7911, 3.7511, 3.5495, 3.4149, 3.3177, 4.0129, 3.8370, 3.7739,
    3.7125, 3.7152, 3.7701, 3.5813, 3.5187, 3.4835, 3.4595, 3.4439,
    3.4242, 3.7476, 3.8239, 3.8346, 3.6627, 3.5479, 3.4639, 4.1026,
    3.9733, 3.9292, 3.8667, 3.9513, 3.8959, 3.7698, 3.7089, 3.6765,
    3.6548, 3.6409, 3.5398, 3.8759, 3.9804, 4.0150, 2.9091, 2.7638,
    3.5066, 3.3377, 3.3481, 3.2633, 3.1810, 3.1428, 2.9872, 2.8837,
    3.5929, 3.5183, 3.6729, 3.6596, 3.6082, 3.5927, 3.4224, 3.2997,
    3.8190, 4.1865, 4.1114, 4.0540, 3.6325, 3.5697, 3.5561, 3.5259,
    3.4901, 3.4552, 3.4315, 3.4091, 3.6438, 3.6879, 3.6832, 3.7043,
    3.5557, 3.4466, 3.9203, 4.2919, 4.2196, 4.1542, 3.7573, 3.7039,
    3.6546, 3.6151, 3.5293, 3.4849, 3.4552, 3.5192, 3.7673, 3.8359,
    3.8525, 3.8901, 2.7806, 2.7209, 3.3812, 3.4958, 3.2913, 3.1888,
    3.0990, 3.0394, 2.9789, 2.8582, 3.4716, 3.6883, 3.6105, 3.5704,
    3.5059, 3.4619, 3.4138, 3.2742, 3.7080, 3.9773, 3.9010, 3.8409,
    3.7944, 3.4465, 3.7235, 3.6808, 3.6453, 3.6168, 3.5844, 3.5576,
    3.5772, 3.5959, 3.5768, 3.5678, 3.5486, 3.4228, 3.8107, 4.0866,
    4.0169, 3.9476, 3.6358, 3.5800, 3.5260, 3.4838, 3.4501, 3.4204,
    3.3553, 3.6487, 3.6973, 3.7398, 3.7405, 3.7459, 3.7380, 2.6848,
    2.6740, 3.2925, 3.3386, 3.2473, 3.1284, 3.0301, 2.9531, 2.9602,
    2.8272, 3.3830, 3.5358, 3.5672, 3.5049, 3.4284, 3.3621, 3.3001,
    3.2451, 3.6209, 3.8299, 3.7543, 3.6920, 3.6436, 3.3598, 3.5701,
    3.5266, 3.4904, 3.4590, 3.4364, 3.4077, 3.5287, 3.5280, 3.4969,
    3.4650, 3.4304, 3.3963, 3.7229, 3.9402, 3.8753, 3.8035, 3.5499,
    3.4913, 3.4319, 3.3873, 3.3520, 3.3209, 3.2948, 3.5052, 3.6465,
    3.6696, 3.6577, 3.6388, 3.6142, 3.5889, 3.3968, 3.0122, 4.2241,
    3.7887, 4.0049, 3.5384, 3.2698, 3.1083, 2.9917, 2.9057, 4.3340,
    3.9900, 4.6588, 4.1278, 3.8125, 3.6189, 3.4851, 3.3859, 4.6531,
    4.3134, 4.2258, 4.1309, 4.0692, 4.0944, 3.9850, 3.9416, 3.9112,
    3.8873, 3.8736, 3.8473, 4.6027, 4.1538, 3.8994, 3.7419, 3.6356,
    3.5548, 4.8353, 4.5413, 4.3891, 4.3416, 4.3243, 4.2753, 4.2053,
    4.1790, 4.1685, 4.1585, 4.1536, 4.0579, 4.1980, 4.4564, 4.2192,
    4.0528, 3.9489, 3.8642, 5.0567, 3.0630, 3.3271, 4.0432, 4.0046,
    4.1555, 3.7426, 3.5130, 3.5174, 3.2884, 3.1378, 4.1894, 4.2321,
    4.1725, 4.1833, 3.8929, 4.0544, 3.8118, 3.6414, 4.6373, 4.6268,
    4.4750, 4.4134, 4.3458, 3.8582, 4.2583, 4.1898, 4.1562, 4.1191,
    4.1069, 4.0639, 4.1257, 4.1974, 3.9532, 4.1794, 3.9660, 3.8130,
    4.8160, 4.8272, 4.6294, 4.5840, 4.0770, 4.0088, 3.9103, 3.8536,
    3.8324, 3.7995, 3.7826, 4.2294, 4.3380, 4.4352, 4.1933, 4.4580,
    4.2554, 4.1072, 5.0454, 5.1814, 3.0632, 3.2662, 3.6432, 3.8088,
    3.7910, 3.7381, 3.5093, 3.5155, 3.3047, 3.1681, 3.7871, 3.9924,
    4.0637, 4.1382, 3.8591, 4.0164, 3.7878, 3.6316, 4.1741, 4.3166,
    4.2395, 4.1831, 4.1107, 3.5857, 4.0270, 3.9676, 3.9463, 3.9150,
    3.9021, 3.8708, 4.0240, 4.1551, 3.9108, 4.1337, 3.9289, 3.7873,
    4.3666, 4.5080, 4.4232, 4.3155, 3.8461, 3.8007, 3.6991, 3.6447,
    3.6308, 3.5959, 3.5749, 4.0359, 4.3124, 4.3539, 4.1122, 4.3772,
    4.1785, 4.0386, 4.7004, 4.8604, 4.6261, 2.9455, 3.2470, 3.6108,
    3.8522, 3.6625, 3.6598, 3.4411, 3.4660, 3.2415, 3.0944, 3.7514,
    4.0397, 3.9231, 4.0561, 3.7860, 3.9845, 3.7454, 3.5802, 4.1366,
    4.3581, 4.2351, 4.2011, 4.1402, 3.5381, 4.0653, 4.0093, 3.9883,
    3.9570, 3.9429, 3.9112, 3.8728, 4.0682, 3.8351, 4.1054, 3.8928,
    3.7445, 4.3415, 4.5497, 4.3833, 4.3122, 3.8051, 3.7583, 3.6622,
    3.6108, 3.5971, 3.5628, 3.5408, 4.0780, 4.0727, 4.2836, 4.0553,
    4.3647, 4.1622, 4.0178, 4.5802, 4.9125, 4.5861, 4.6201, 2.9244,
    3.2241, 3.5848, 3.8293, 3.6395, 3.6400, 3.4204, 3.4499, 3.2253,
    3.0779, 3.7257, 4.0170, 3.9003, 4.0372, 3.7653, 3.9672, 3.7283,
    3.5630, 4.1092, 4.3347, 4.2117, 4.1793, 4.1179, 3.5139, 4.0426,
    3.9867, 3.9661, 3.9345, 3.9200, 3.8883, 3.8498, 4.0496, 3.8145,
    4.0881, 3.8756, 3.7271, 4.3128, 4.5242, 4.3578, 4.2870, 3.7796,
    3.7318, 3.6364, 3.5854, 3.5726, 3.5378, 3.5155, 4.0527, 4.0478,
    4.2630, 4.0322, 4.3449, 4.1421, 3.9975, 4.5499, 4.8825, 4.5601,
    4.5950, 4.5702, 2.9046, 3.2044, 3.5621, 3.8078, 3.6185, 3.6220,
    3.4019, 3.4359, 3.2110, 3.0635, 3.7037, 3.9958, 3.8792, 4.0194,
    3.7460, 3.9517, 3.7128, 3.5474, 4.0872, 4.3138, 4.1906, 4.1593,
    4.0973, 3.4919, 4.0216, 3.9657, 3.9454, 3.9134, 3.8986, 3.8669,
    3.8289, 4.0323, 3.7954, 4.0725, 3.8598, 3.7113, 4.2896, 4.5021,
    4.3325, 4.2645, 3.7571, 3.7083, 3.6136, 3.5628, 3.5507, 3.5155,
    3.4929, 4.0297, 4.0234, 4.2442, 4.0112, 4.3274, 4.1240, 3.9793,
    4.5257, 4.8568, 4.5353, 4.5733, 4.5485, 4.5271, 2.8878, 3.1890,
    3.5412, 3.7908, 3.5974, 3.6078, 3.3871, 3.4243, 3.1992, 3.0513,
    3.6831, 3.9784, 3.8579, 4.0049, 3.7304, 3.9392, 3.7002, 3.5347,
    4.0657, 4.2955, 4.1705, 4.1424, 4.0800, 3.4717, 4.0043, 3.9485,
    3.9286, 3.8965, 3.8815, 3.8500, 3.8073, 4.0180, 3.7796, 4.0598,
    3.8470, 3.6983, 4.2678, 4.4830, 4.3132, 4.2444, 3.7370, 3.6876,
    3.5935, 3.5428, 3.5314, 3.4958, 3.4730, 4.0117, 4.0043, 4.2287,
    3.9939, 4.3134, 4.1096, 3.9646, 4.5032, 4.8356, 4.5156, 4.5544,
    4.5297, 4.5083, 4.4896, 2.8709, 3.1737, 3.5199, 3.7734, 3.5802,
    3.5934, 3.3724, 3.4128, 3.1877, 3.0396, 3.6624, 3.9608, 3.8397,
    3.9893, 3.7145, 3.9266, 3.6877, 3.5222, 4.0448, 4.2771, 4.1523,
    4.1247, 4.0626, 3.4530, 3.9866, 3.9310, 3.9115, 3.8792, 3.8641,
    3.8326, 3.7892, 4.0025, 3.7636, 4.0471, 3.8343, 3.6854, 4.2464,
    4.4635, 4.2939, 4.2252, 3.7169, 3.6675, 3.5739, 3.5235, 3.5126,
    3.4768, 3.4537, 3.9932, 3.9854, 4.2123, 3.9765, 4.2992, 4.0951,
    3.9500, 4.4811, 4.8135, 4.4959, 4.5351, 4.5105, 4.4891, 4.4705,
    4.4515, 2.8568, 3.1608, 3.5050, 3.7598, 3.5665, 3.5803, 3.3601,
    3.4031, 3.1779, 3.0296, 3.6479, 3.9471, 3.8262, 3.9773, 3.7015,
    3.9162, 3.6771, 3.5115, 4.0306, 4.2634, 4.1385, 4.1116, 4.0489,
    3.4366, 3.9732, 3.9176, 3.8983, 3.8659, 3.8507, 3.8191, 3.7757,
    3.9907, 3.7506, 4.0365, 3.8235, 3.6745, 4.2314, 4.4490, 4.2792,
    4.2105, 3.7003, 3.6510, 3.5578, 3.5075, 3.4971, 3.4609, 3.4377,
    3.9788, 3.9712, 4.1997, 3.9624, 4.2877, 4.0831, 3.9378, 4.4655,
    4.7974, 4.4813, 4.5209, 4.4964, 4.4750, 4.4565, 4.4375, 4.4234,
    2.6798, 3.0151, 3.2586, 3.5292, 3.5391, 3.4902, 3.2887, 3.3322,
    3.1228, 2.9888, 3.4012, 3.7145, 3.7830, 3.6665, 3.5898, 3.8077,
    3.5810, 3.4265, 3.7726, 4.0307, 3.9763, 3.8890, 3.8489, 3.2706,
    3.7595, 3.6984, 3.6772, 3.6428, 3.6243, 3.5951, 3.7497, 3.6775,
    3.6364, 3.9203, 3.7157, 3.5746, 3.9494, 4.2076, 4.1563, 4.0508,
    3.5329, 3.4780, 3.3731, 3.3126, 3.2846, 3.2426, 3.2135, 3.7491,
    3.9006, 3.8332, 3.8029, 4.1436, 3.9407, 3.7998, 4.1663, 4.5309,
    4.3481, 4.2911, 4.2671, 4.2415, 4.2230, 4.2047, 4.1908, 4.1243,
    2.5189, 2.9703, 3.3063, 3.6235, 3.4517, 3.3989, 3.2107, 3.2434,
    3.0094, 2.8580, 3.4253, 3.8157, 3.7258, 3.6132, 3.5297, 3.7566,
    3.5095, 3.3368, 3.7890, 4.1298, 4.0190, 3.9573, 3.9237, 3.2677,
    3.8480, 3.8157, 3.7656, 3.7317, 3.7126, 3.6814, 3.6793, 3.6218,
    3.5788, 3.8763, 3.6572, 3.5022, 3.9737, 4.3255, 4.1828, 4.1158,
    3.5078, 3.4595, 3.3600, 3.3088, 3.2575, 3.2164, 3.1856, 3.8522,
    3.8665, 3.8075, 3.7772, 4.1391, 3.9296, 3.7772, 4.2134, 4.7308,
    4.3787, 4.3894, 4.3649, 4.3441, 4.3257, 4.3073, 4.2941, 4.1252,
    4.2427, 3.0481, 2.9584, 3.6919, 3.5990, 3.8881, 3.4209, 3.1606,
    3.1938, 2.9975, 2.8646, 3.8138, 3.7935, 3.7081, 3.9155, 3.5910,
    3.4808, 3.4886, 3.3397, 4.1336, 4.1122, 3.9888, 3.9543, 3.8917,
    3.5894, 3.8131, 3.7635, 3.7419, 3.7071, 3.6880, 3.6574, 3.6546,
    3.9375, 3.6579, 3.5870, 3.6361, 3.5039, 4.3149, 4.2978, 4.1321,
    4.1298, 3.8164, 3.7680, 3.7154, 3.6858, 3.6709, 3.6666, 3.6517,
    3.8174, 3.8608, 4.1805, 3.9102, 3.8394, 3.8968, 3.7673, 4.5274,
    4.6682, 4.3344, 4.3639, 4.3384, 4.3162, 4.2972, 4.2779, 4.2636,
    4.0253, 4.1168, 4.1541, 2.8136, 3.0951, 3.4635, 3.6875, 3.4987,
    3.5183, 3.2937, 3.3580, 3.1325, 2.9832, 3.6078, 3.8757, 3.7616,
    3.9222, 3.6370, 3.8647, 3.6256, 3.4595, 3.9874, 4.1938, 4.0679,
    4.0430, 3.9781, 3.3886, 3.9008, 3.8463, 3.8288, 3.7950, 3.7790,
    3.7472, 3.7117, 3.9371, 3.6873, 3.9846, 3.7709, 3.6210, 4.1812,
    4.3750, 4.2044, 4.1340, 3.6459, 3.5929, 3.5036, 3.4577, 3.4528,
    3.4146, 3.3904, 3.9014, 3.9031, 4.1443, 3.8961, 4.2295, 4.0227,
    3.8763, 4.4086, 4.7097, 4.4064, 4.4488, 4.4243, 4.4029, 4.3842,
    4.3655, 4.3514, 4.1162, 4.2205, 4.1953, 4.2794, 2.8032, 3.0805,
    3.4519, 3.6700, 3.4827, 3.5050, 3.2799, 3.3482, 3.1233, 2.9747,
    3.5971, 3.8586, 3.7461, 3.9100, 3.6228, 3.8535, 3.6147, 3.4490,
    3.9764, 4.1773, 4.0511, 4.0270, 3.9614, 3.3754, 3.8836, 3.8291,
    3.8121, 3.7780, 3.7619, 3.7300, 3.6965, 3.9253, 3.6734, 3.9733,
    3.7597, 3.6099, 4.1683, 4.3572, 4.1862, 4.1153, 3.6312, 3.5772,
    3.4881, 3.4429, 3.4395, 3.4009, 3.3766, 3.8827, 3.8868, 4.1316,
    3.8807, 4.2164, 4.0092, 3.8627, 4.3936, 4.6871, 4.3882, 4.4316,
    4.4073, 4.3858, 4.3672, 4.3485, 4.3344, 4.0984, 4.2036, 4.1791,
    4.2622, 4.2450, 2.7967, 3.0689, 3.4445, 3.6581, 3.4717, 3.4951,
    3.2694, 3.3397, 3.1147, 2.9661, 3.5898, 3.8468, 3.7358, 3.9014,
    3.6129, 3.8443, 3.6054, 3.4396, 3.9683, 4.1656, 4.0394, 4.0158,
    3.9498, 3.3677, 3.8718, 3.8164, 3.8005, 3.7662, 3.7500, 3.7181,
    3.6863, 3.9170, 3.6637, 3.9641, 3.7503, 3.6004, 4.1590, 4.3448,
    4.1739, 4.1029, 3.6224, 3.5677, 3.4785, 3.4314, 3.4313, 3.3923,
    3.3680, 3.8698, 3.8758, 4.1229, 3.8704, 4.2063, 3.9987, 3.8519,
    4.3832, 4.6728, 4.3759, 4.4195, 4.3952, 4.3737, 4.3551, 4.3364,
    4.3223, 4.0861, 4.1911, 4.1676, 4.2501, 4.2329, 4.2208, 2.7897,
    3.0636, 3.4344, 3.6480, 3.4626, 3.4892, 3.2626, 3.3344, 3.1088,
    2.9597, 3.5804, 3.8359, 3.7251, 3.8940, 3.6047, 3.8375, 3.5990,
    3.4329, 3.9597, 4.1542, 4.0278, 4.0048, 3.9390, 3.3571, 3.8608,
    3.8056, 3.7899, 3.7560, 3.7400, 3.7081, 3.6758, 3.9095, 3.6552,
    3.9572, 3.7436, 3.5933, 4.1508, 4.3337, 4.1624, 4.0916, 3.6126,
    3.5582, 3.4684, 3.4212, 3.4207, 3.3829, 3.3586, 3.8604, 3.8658,
    4.1156, 3.8620, 4.1994, 3.9917, 3.8446, 4.3750, 4.6617, 4.3644,
    4.4083, 4.3840, 4.3625, 4.3439, 4.3253, 4.3112, 4.0745, 4.1807,
    4.1578, 4.2390, 4.2218, 4.2097, 4.1986, 2.8395, 3.0081, 3.3171,
    3.4878, 3.5360, 3.5145, 3.2809, 3.3307, 3.1260, 2.9940, 3.4741,
    3.6675, 3.7832, 3.6787, 3.6156, 3.8041, 3.5813, 3.4301, 3.8480,
    3.9849, 3.9314, 3.8405, 3.8029, 3.2962, 3.7104, 3.6515, 3.6378,
    3.6020, 3.5849, 3.5550, 3.7494, 3.6893, 3.6666, 3.9170, 3.7150,
    3.5760, 4.0268, 4.1596, 4.1107, 3.9995, 3.5574, 3.5103, 3.4163,
    3.3655, 3.3677, 3.3243, 3.2975, 3.7071, 3.9047, 3.8514, 3.8422,
    3.8022, 3.9323, 3.7932, 4.2343, 4.4583, 4.3115, 4.2457, 4.2213,
    4.1945, 4.1756, 4.1569, 4.1424, 4.0620, 4.0494, 3.9953, 4.0694,
    4.0516, 4.0396, 4.0280, 4.0130, 2.9007, 2.9674, 3.8174, 3.5856,
    3.6486, 3.5339, 3.2832, 3.3154, 3.1144, 2.9866, 3.9618, 3.8430,
    3.9980, 3.8134, 3.6652, 3.7985, 3.5756, 3.4207, 4.4061, 4.2817,
    4.1477, 4.0616, 3.9979, 3.6492, 3.8833, 3.8027, 3.7660, 3.7183,
    3.6954, 3.6525, 3.9669, 3.8371, 3.7325, 3.9160, 3.7156, 3.5714,
    4.6036, 4.4620, 4.3092, 4.2122, 3.8478, 3.7572, 3.6597, 3.5969,
    3.5575, 3.5386, 3.5153, 3.7818, 4.1335, 4.0153, 3.9177, 3.8603,
    3.9365, 3.7906, 4.7936, 4.7410, 4.5461, 4.5662, 4.5340, 4.5059,
    4.4832, 4.4604, 4.4429, 4.2346, 4.4204, 4.3119, 4.3450, 4.3193,
    4.3035, 4.2933, 4.1582, 4.2450, 2.8559, 2.9050, 3.8325, 3.5442,
    3.5077, 3.4905, 3.2396, 3.2720, 3.0726, 2.9467, 3.9644, 3.8050,
    3.8981, 3.7762, 3.6216, 3.7531, 3.5297, 3.3742, 4.3814, 4.2818,
    4.1026, 4.0294, 3.9640, 3.6208, 3.8464, 3.7648, 3.7281, 3.6790,
    3.6542, 3.6117, 3.8650, 3.8010, 3.6894, 3.8713, 3.6699, 3.5244,
    4.5151, 4.4517, 4.2538, 4.1483, 3.8641, 3.7244, 3.6243, 3.5589,
    3.5172, 3.4973, 3.4715, 3.7340, 4.0316, 3.9958, 3.8687, 3.8115,
    3.8862, 3.7379, 4.7091, 4.7156, 4.5199, 4.5542, 4.5230, 4.4959,
    4.4750, 4.4529, 4.4361, 4.1774, 4.3774, 4.2963, 4.3406, 4.3159,
    4.3006, 4.2910, 4.1008, 4.1568, 4.0980, 2.8110, 2.8520, 3.7480,
    3.5105, 3.4346, 3.3461, 3.1971, 3.2326, 3.0329, 2.9070, 3.8823,
    3.7928, 3.8264, 3.7006, 3.5797, 3.7141, 3.4894, 3.3326, 4.3048,
    4.2217, 4.0786, 3.9900, 3.9357, 3.6331, 3.8333, 3.7317, 3.6957,
    3.6460, 3.6197, 3.5779, 3.7909, 3.7257, 3.6476, 3.5729, 3.6304,
    3.4834, 4.4368, 4.3921, 4.2207, 4.1133, 3.8067, 3.7421, 3.6140,
    3.5491, 3.5077, 3.4887, 3.4623, 3.6956, 3.9568, 3.8976, 3.8240,
    3.7684, 3.8451, 3.6949, 4.6318, 4.6559, 4.4533, 4.4956, 4.4641,
    4.4366, 4.4155, 4.3936, 4.3764, 4.1302, 4.3398, 4.2283, 4.2796,
    4.2547, 4.2391, 4.2296, 4.0699, 4.1083, 4.0319, 3.9855, 2.7676,
    2.8078, 3.6725, 3.4804, 3.3775, 3.2411, 3.1581, 3.1983, 2.9973,
    2.8705, 3.8070, 3.7392, 3.7668, 3.6263, 3.5402, 3.6807, 3.4545,
    3.2962, 4.2283, 4.1698, 4.0240, 3.9341, 3.8711, 3.5489, 3.7798,
    3.7000, 3.6654, 3.6154, 3.5882, 3.5472, 3.7289, 3.6510, 3.6078,
    3.5355, 3.5963, 3.4480, 4.3587, 4.3390, 4.1635, 4.0536, 3.7193,
    3.6529, 3.5512, 3.4837, 3.4400, 3.4191, 3.3891, 3.6622, 3.8934,
    3.8235, 3.7823, 3.7292, 3.8106, 3.6589, 4.5535, 4.6013, 4.3961,
    4.4423, 4.4109, 4.3835, 4.3625, 4.3407, 4.3237, 4.0863, 4.2835,
    4.1675, 4.2272, 4.2025, 4.1869, 4.1774, 4.0126, 4.0460, 3.9815,
    3.9340, 3.8955, 2.6912, 2.7604, 3.6037, 3.4194, 3.3094, 3.1710,
    3.0862, 3.1789, 2.9738, 2.8427, 3.7378, 3.6742, 3.6928, 3.5512,
    3.4614, 3.4087, 3.4201, 3.2607, 4.1527, 4.0977, 3.9523, 3.8628,
    3.8002, 3.4759, 3.7102, 3.6466, 3.6106, 3.5580, 3.5282, 3.4878,
    3.6547, 3.5763, 3.5289, 3.5086, 3.5593, 3.4099, 4.2788, 4.2624,
    4.0873, 3.9770, 3.6407, 3.5743, 3.5178, 3.4753, 3.3931, 3.3694,
    3.3339, 3.6002, 3.8164, 3.7478, 3.7028, 3.6952, 3.7669, 3.6137,
    4.4698, 4.5488, 4.3168, 4.3646, 4.3338, 4.3067, 4.2860, 4.2645,
    4.2478, 4.0067, 4.2349, 4.0958, 4.1543, 4.1302, 4.1141, 4.1048,
    3.9410, 3.9595, 3.8941, 3.8465, 3.8089, 3.7490, 2.7895, 2.5849,
    3.6484, 3.0162, 3.1267, 3.2125, 3.0043, 2.9572, 2.8197, 2.7261,
    3.7701, 3.2446, 3.5239, 3.4696, 3.4261, 3.3508, 3.1968, 3.0848,
    4.1496, 3.6598, 3.5111, 3.4199, 3.3809, 3.5382, 3.2572, 3.2100,
    3.1917, 3.1519, 3.1198, 3.1005, 3.5071, 3.5086, 3.5073, 3.4509,
    3.3120, 3.2082, 4.2611, 3.8117, 3.6988, 3.5646, 3.6925, 3.6295,
    3.5383, 3.4910, 3.4625, 3.4233, 3.4007, 3.2329, 3.6723, 3.6845,
    3.6876, 3.6197, 3.4799, 3.3737, 4.4341, 4.0525, 3.9011, 3.8945,
    3.8635, 3.8368, 3.8153, 3.7936, 3.7758, 3.4944, 3.4873, 3.9040,
    3.7110, 3.6922, 3.6799, 3.6724, 3.5622, 3.6081, 3.5426, 3.4922,
    3.4498, 3.3984, 3.4456, 2.7522, 2.5524, 3.5742, 2.9508, 3.0751,
    3.0158, 2.9644, 2.8338, 2.7891, 2.6933, 3.6926, 3.1814, 3.4528,
    3.4186, 3.3836, 3.2213, 3.1626, 3.0507, 4.0548, 3.5312, 3.4244,
    3.3409, 3.2810, 3.4782, 3.1905, 3.1494, 3.1221, 3.1128, 3.0853,
    3.0384, 3.4366, 3.4562, 3.4638, 3.3211, 3.2762, 3.1730, 4.1632,
    3.6825, 3.5822, 3.4870, 3.6325, 3.5740, 3.4733, 3.4247, 3.3969,
    3.3764, 3.3525, 3.1984, 3.5989, 3.6299, 3.6433, 3.4937, 3.4417,
    3.3365, 4.3304, 3.9242, 3.7793, 3.7623, 3.7327, 3.7071, 3.6860,
    3.6650, 3.6476, 3.3849, 3.3534, 3.8216, 3.5870, 3.5695, 3.5584,
    3.5508, 3.4856, 3.5523, 3.4934, 3.4464, 3.4055, 3.3551, 3.3888,
    3.3525, 2.7202, 2.5183, 3.4947, 2.8731, 3.0198, 3.1457, 2.9276,
    2.7826, 2.7574, 2.6606, 3.6090, 3.0581, 3.3747, 3.3677, 3.3450,
    3.1651, 3.1259, 3.0147, 3.9498, 3.3857, 3.2917, 3.2154, 3.1604,
    3.4174, 3.0735, 3.0342, 3.0096, 3.0136, 2.9855, 2.9680, 3.3604,
    3.4037, 3.4243, 3.2633, 3.1810, 3.1351, 4.0557, 3.5368, 3.4526,
    3.3699, 3.5707, 3.5184, 3.4085, 3.3595, 3.3333, 3.3143, 3.3041,
    3.1094, 3.5193, 3.5745, 3.6025, 3.4338, 3.3448, 3.2952, 4.2158,
    3.7802, 3.6431, 3.6129, 3.5853, 3.5610, 3.5406, 3.5204, 3.5036,
    3.2679, 3.2162, 3.7068, 3.4483, 3.4323, 3.4221, 3.4138, 3.3652,
    3.4576, 3.4053, 3.3618, 3.3224, 3.2711, 3.3326, 3.2950, 3.2564,
    2.5315, 2.6104, 3.2734, 3.2299, 3.1090, 2.9942, 2.9159, 2.8324,
    2.8350, 2.7216, 3.3994, 3.4475, 3.4354, 3.3438, 3.2807, 3.2169,
    3.2677, 3.1296, 3.7493, 3.8075, 3.6846, 3.6104, 3.5577, 3.2052,
    3.4803, 3.4236, 3.3845, 3.3640, 3.3365, 3.3010, 3.3938, 3.3624,
    3.3440, 3.3132, 3.4035, 3.2754, 3.8701, 3.9523, 3.8018, 3.7149,
    3.3673, 3.3199, 3.2483, 3.2069, 3.1793, 3.1558, 3.1395, 3.4097,
    3.5410, 3.5228, 3.5116, 3.4921, 3.4781, 3.4690, 4.0420, 4.1759,
    4.0078, 4.0450, 4.0189, 3.9952, 3.9770, 3.9583, 3.9434, 3.7217,
    3.8228, 3.7826, 3.8640, 3.8446, 3.8314, 3.8225, 3.6817, 3.7068,
    3.6555, 3.6159, 3.5831, 3.5257, 3.2133, 3.1689, 3.1196, 3.3599,
    2.9852, 2.7881, 3.5284, 3.3493, 3.6958, 3.3642, 3.1568, 3.0055,
    2.9558, 2.8393, 3.6287, 3.5283, 4.1511, 3.8259, 3.6066, 3.4527,
    3.3480, 3.2713, 3.9037, 3.8361, 3.8579, 3.7311, 3.6575, 3.5176,
    3.5693, 3.5157, 3.4814, 3.4559, 3.4445, 3.4160, 4.1231, 3.8543,
    3.6816, 3.5602, 3.4798, 3.4208, 4.0542, 4.0139, 4.0165, 3.9412,
    3.7698, 3.6915, 3.6043, 3.5639, 3.5416, 3.5247, 3.5153, 3.5654,
    4.2862, 4.0437, 3.8871, 3.7741, 3.6985, 3.6413, 4.2345, 4.3663,
    4.3257, 4.0869, 4.0612, 4.0364, 4.0170, 3.9978, 3.9834, 3.9137,
    3.8825, 3.8758, 3.9143, 3.8976, 3.8864, 3.8768, 3.9190, 4.1613,
    4.0566, 3.9784, 3.9116, 3.8326, 3.7122, 3.6378, 3.5576, 3.5457,
    4.3127, 3.1160, 2.8482, 4.0739, 3.3599, 3.5698, 3.5366, 3.2854,
    3.1039, 2.9953, 2.9192, 4.1432, 3.5320, 3.9478, 4.0231, 3.7509,
    3.5604, 3.4340, 3.3426, 4.3328, 3.8288, 3.7822, 3.7909, 3.6907,
    3.6864, 3.5793, 3.5221, 3.4883, 3.4649, 3.4514, 3.4301, 3.9256,
    4.0596, 3.8307, 3.6702, 3.5651, 3.4884, 4.4182, 4.2516, 3.9687,
    3.9186, 3.9485, 3.8370, 3.7255, 3.6744, 3.6476, 3.6295, 3.6193,
    3.5659, 4.0663, 4.2309, 4.0183, 3.8680, 3.7672, 3.6923, 4.5240,
    4.4834, 4.1570, 4.3204, 4.2993, 4.2804, 4.2647, 4.2481, 4.2354,
    3.8626, 3.8448, 4.2267, 4.1799, 4.1670, 3.8738, 3.8643, 3.8796,
    4.0575, 4.0354, 3.9365, 3.8611, 3.7847, 3.7388, 3.6826, 3.6251,
    3.5492, 4.0889, 4.2764, 3.1416, 2.8325, 3.7735, 3.3787, 3.4632,
    3.5923, 3.3214, 3.1285, 3.0147, 2.9366, 3.8527, 3.5602, 3.8131,
    3.8349, 3.7995, 3.5919, 3.4539, 3.3540, 4.0654, 3.8603, 3.7972,
    3.7358, 3.7392, 3.8157, 3.6055, 3.5438, 3.5089, 3.4853, 3.4698,
    3.4508, 3.7882, 3.8682, 3.8837, 3.7055, 3.5870, 3.5000, 4.1573,
    4.0005, 3.9568, 3.8936, 3.9990, 3.9433, 3.8172, 3.7566, 3.7246,
    3.7033, 3.6900, 3.5697, 3.9183, 4.0262, 4.0659, 3.8969, 3.7809,
    3.6949, 4.2765, 4.2312, 4.1401, 4.0815, 4.0580, 4.0369, 4.0194,
    4.0017, 3.9874, 3.8312, 3.8120, 3.9454, 3.9210, 3.9055, 3.8951,
    3.8866, 3.8689, 3.9603, 3.9109, 3.9122, 3.8233, 3.7438, 3.7436,
    3.6981, 3.6555, 3.5452, 3.9327, 4.0658, 4.1175, 2.9664, 2.8209,
    3.5547, 3.3796, 3.3985, 3.3164, 3.2364, 3.1956, 3.0370, 2.9313,
    3.6425, 3.5565, 3.7209, 3.7108, 3.6639, 3.6484, 3.4745, 3.3492,
    3.8755, 4.2457, 3.7758, 3.7161, 3.6693, 3.6155, 3.5941, 3.5643,
    3.5292, 3.4950, 3.4720, 3.4503, 3.6936, 3.7392, 3.7388, 3.7602,
    3.6078, 3.4960, 3.9800, 4.3518, 4.2802, 3.8580, 3.8056, 3.7527,
    3.7019, 3.6615, 3.5768, 3.5330, 3.5038, 3.5639, 3.8192, 3.8883,
    3.9092, 3.9478, 3.7995, 3.6896, 4.1165, 4.5232, 4.4357, 4.4226,
    4.4031, 4.3860, 4.3721, 4.3580, 4.3466, 4.2036, 4.2037, 3.8867,
    4.2895, 4.2766, 4.2662, 4.2598, 3.8408, 3.9169, 3.8681, 3.8250,
    3.7855, 3.7501, 3.6753, 3.5499, 3.4872, 3.5401, 3.8288, 3.9217,
    3.9538, 4.0054, 2.8388, 2.7890, 3.4329, 3.5593, 3.3488, 3.2486,
    3.1615, 3.1000, 3.0394, 2.9165, 3.5267, 3.7479, 3.6650, 3.6263,
    3.5658, 3.5224, 3.4762, 3.3342, 3.7738, 4.0333, 3.9568, 3.8975,
    3.8521, 3.4929, 3.7830, 3.7409, 3.7062, 3.6786, 3.6471, 3.6208,
    3.6337, 3.6519, 3.6363, 3.6278, 3.6110, 3.4825, 3.8795, 4.1448,
    4.0736, 4.0045, 3.6843, 3.6291, 3.5741, 3.5312, 3.4974, 3.4472,
    3.4034, 3.7131, 3.7557, 3.7966, 3.8005, 3.8068, 3.8015, 3.6747,
    4.0222, 4.3207, 4.2347, 4.2191, 4.1990, 4.1811, 4.1666, 4.1521,
    4.1401, 3.9970, 3.9943, 3.9592, 4.0800, 4.0664, 4.0559, 4.0488,
    3.9882, 4.0035, 3.9539, 3.9138, 3.8798, 3.8355, 3.5359, 3.4954,
    3.3962, 3.5339, 3.7595, 3.8250, 3.8408, 3.8600, 3.8644, 2.7412,
    2.7489, 3.3374, 3.3950, 3.3076, 3.1910, 3.0961, 3.0175, 3.0280,
    2.8929, 3.4328, 3.5883, 3.6227, 3.5616, 3.4894, 3.4241, 3.3641,
    3.3120, 3.6815, 3.8789, 3.8031, 3.7413, 3.6939, 3.4010, 3.6225,
    3.5797, 3.5443, 3.5139, 3.4923, 3.4642, 3.5860, 3.5849, 3.5570,
    3.5257, 3.4936, 3.4628, 3.7874, 3.9916, 3.9249, 3.8530, 3.5932,
    3.5355, 3.4757, 3.4306, 3.3953, 3.3646, 3.3390, 3.5637, 3.7053,
    3.7266, 3.7177, 3.6996, 3.6775, 3.6558, 3.9331, 4.1655, 4.0879,
    4.0681, 4.0479, 4.0299, 4.0152, 4.0006, 3.9883, 3.8500, 3.8359,
    3.8249, 3.9269, 3.9133, 3.9025, 3.8948, 3.8422, 3.8509, 3.7990,
    3.7570, 3.7219, 3.6762, 3.4260, 3.3866, 3.3425, 3.5294, 3.7022,
    3.7497, 3.7542, 3.7494, 3.7370, 3.7216, 3.3286, 3.5286, 4.5857,
    4.2143, 3.9714, 3.9428, 3.8857, 3.7714, 3.7857, 3.8286, 4.8714,
    4.6571, 4.4857, 4.3571, 4.4429, 4.3286, 4.2857, 4.2428, 5.3857,
    5.0714, 4.7714, 4.6143, 4.6000, 4.4429, 4.4000, 4.3571, 4.3000,
    4.2857, 4.3143, 4.4286, 4.4714, 4.4286, 4.5143, 4.4429, 4.5000,
    4.5429, 5.5714, 5.2572, 4.9714, 4.8572, 4.7571, 4.6428, 4.5143,
    4.4857, 4.4857, 4.4143, 4.5143, 4.6286, 4.7000, 4.6714, 4.6714,
    4.6286, 4.7571, 4.7429, 5.8571, 5.3857, 5.1857, 4.9714, 5.1286,
    5.1143, 5.1000, 5.0857, 5.0286, 5.0428, 5.0286, 5.0143, 5.0000,
    5.0000, 4.9857, 5.0571, 4.9571, 4.8286, 4.7429, 4.6286, 4.5572,
    4.5286, 4.4572, 4.4714, 4.4857, 4.7571, 4.7286, 4.7286, 4.8143,
    4.7429, 4.8429, 4.9000, 5.7428, 3.0429, 3.2429, 4.3000, 3.9286,
    3.6857, 3.6571, 3.6000, 3.4857, 3.5000, 3.5428, 4.5857, 4.3714,
    4.2000, 4.0714, 4.1572, 4.0429, 4.0000, 3.9571, 5.1000, 4.7857,
    4.4857, 4.3286, 4.3143, 4.1572, 4.1143, 4.0714, 4.0143, 4.0000,
    4.0286, 4.1429, 4.1857, 4.1429, 4.2285, 4.1572, 4.2143, 4.2571,
    5.2857, 4.9714, 4.6857, 4.5714, 4.4714, 4.3571, 4.2285, 4.2000,
    4.2000, 4.1286, 4.2285, 4.3429, 4.4143, 4.3857, 4.3857, 4.3429,
    4.4714, 4.4572, 5.5714, 5.1000, 4.9000, 4.6857, 4.8429, 4.8286,
    4.8143, 4.8000, 4.7429, 4.7571, 4.7429, 4.7286, 4.7143, 4.7143,
    4.7000, 4.7714, 4.6714, 4.5429, 4.4572, 4.3429, 4.2714, 4.2428,
    4.1715, 4.1857, 4.2000, 4.4714, 4.4429, 4.4429, 4.5286, 4.4572,
    4.5572, 4.6143, 5.4571, 5.1714, 2.8428, 3.0429, 4.1000, 3.7286,
    3.4857, 3.4572, 3.4000, 3.2857, 3.3000, 3.3429, 4.3857, 4.1715,
    4.0000, 3.8714, 3.9571, 3.8428, 3.8000, 3.7572, 4.9000, 4.5857,
    4.2857, 4.1286, 4.1143, 3.9571, 3.9143, 3.8714, 3.8143, 3.8000,
    3.8286, 3.9428, 3.9857, 3.9428, 4.0286, 3.9571, 4.0143, 4.0571,
    5.0857, 4.7714, 4.4857, 4.3714, 4.2714, 4.1572, 4.0286, 4.0000,
    4.0000, 3.9286, 4.0286, 4.1429, 4.2143, 4.1857, 4.1857, 4.1429,
    4.2714, 4.2571, 5.3714, 4.9000, 4.7000, 4.4857, 4.6428, 4.6286,
    4.6143, 4.6000, 4.5429, 4.5572, 4.5429, 4.5286, 4.5143, 4.5143,
    4.5000, 4.5714, 4.4714, 4.3429, 4.2571, 4.1429, 4.0714, 4.0429,
    3.9714, 3.9857, 4.0000, 4.2714, 4.2428, 4.2428, 4.3286, 4.2571,
    4.3571, 4.4143, 5.2572, 4.9714, 4.7714, 2.7143, 2.9143, 3.9714,
    3.6000, 3.3572, 3.3286, 3.2714, 3.1571, 3.1714, 3.2143, 4.2571,
    4.0429, 3.8714, 3.7429, 3.8286, 3.7143, 3.6714, 3.6286, 4.7714,
    4.4572, 4.1572, 4.0000, 3.9857, 3.8286, 3.7857, 3.7429, 3.6857,
    3.6714, 3.7000, 3.8143, 3.8571, 3.8143, 3.9000, 3.8286, 3.8857,
    3.9286, 4.9571, 4.6428, 4.3571, 4.2428, 4.1429, 4.0286, 3.9000,
    3.8714, 3.8714, 3.8000, 3.9000, 4.0143, 4.0857, 4.0571, 4.0571,
    4.0143, 4.1429, 4.1286, 5.2429, 4.7714, 4.5714, 4.3571, 4.5143,
    4.5000, 4.4857, 4.4714, 4.4143, 4.4286, 4.4143, 4.4000, 4.3857,
    4.3857, 4.3714, 4.4429, 4.3429, 4.2143, 4.1286, 4.0143, 3.9428,
    3.9143, 3.8428, 3.8571, 3.8714, 4.1429, 4.1143, 4.1143, 4.2000,
    4.1286, 4.2285, 4.2857, 5.1286, 4.8429, 4.6428, 4.5143, 2.6286,
    2.8286, 3.8857, 3.5143, 3.2714, 3.2429, 3.1857, 3.0715, 3.0857,
    3.1285, 4.1715, 3.9571, 3.7857, 3.6571, 3.7429, 3.6286, 3.5857,
    3.5428, 4.6857, 4.3714, 4.0714, 3.9143, 3.9000, 3.7429, 3.7000,
    3.6571, 3.6000, 3.5857, 3.6143, 3.7286, 3.7714, 3.7286, 3.8143,
    3.7429, 3.8000, 3.8428, 4.8714, 4.5572, 4.2714, 4.1572, 4.0571,
    3.9428, 3.8143, 3.7857, 3.7857, 3.7143, 3.8143, 3.9286, 4.0000,
    3.9714, 3.9714, 3.9286, 4.0571, 4.0429, 5.1571, 4.6857, 4.4857,
    4.2714, 4.4286, 4.4143, 4.4000, 4.3857, 4.3286, 4.3429, 4.3286,
    4.3143, 4.3000, 4.3000, 4.2857, 4.3571, 4.2571, 4.1286, 4.0429,
    3.9286, 3.8571, 3.8286, 3.7572, 3.7714, 3.7857, 4.0571, 4.0286,
    4.0286, 4.1143, 4.0429, 4.1429, 4.2000, 5.0428, 4.7571, 4.5572,
    4.4286, 4.3429, 2.6429, 2.8428, 3.9000, 3.5286, 3.2857, 3.2571,
    3.2000, 3.0857, 3.1000, 3.1428, 4.1857, 3.9714, 3.8000, 3.6714,
    3.7572, 3.6429, 3.6000, 3.5571, 4.7000, 4.3857, 4.0857, 3.9286,
    3.9143, 3.7572, 3.7143, 3.6714, 3.6143, 3.6000, 3.6286, 3.7429,
    3.7857, 3.7429, 3.8286, 3.7572, 3.8143, 3.8571, 4.8857, 4.5714,
    4.2857, 4.1715, 4.0714, 3.9571, 3.8286, 3.8000, 3.8000, 3.7286,
    3.8286, 3.9428, 4.0143, 3.9857, 3.9857, 3.9428, 4.0714, 4.0571,
    5.1714, 4.7000, 4.5000, 4.2857, 4.4429, 4.4286, 4.4143, 4.4000,
    4.3429, 4.3571, 4.3429, 4.3286, 4.3143, 4.3143, 4.3000, 4.3714,
    4.2714, 4.1429, 4.0571, 3.9428, 3.8714, 3.8428, 3.7714, 3.7857,
    3.8000, 4.0714, 4.0429, 4.0429, 4.1286, 4.0571, 4.1572, 4.2143,
    5.0571, 4.7714, 4.5714, 4.4429, 4.3571, 4.3714, 2.6572, 2.8571,
    3.9143, 3.5428, 3.3000, 3.2714, 3.2143, 3.1000, 3.1143, 3.1571,
    4.2000, 3.9857, 3.8143, 3.6857, 3.7714, 3.6571, 3.6143, 3.5714,
    4.7143, 4.4000, 4.1000, 3.9428, 3.9286, 3.7714, 3.7286, 3.6857,
    3.6286, 3.6143, 3.6429, 3.7572, 3.8000, 3.7572, 3.8428, 3.7714,
    3.8286, 3.8714, 4.9000, 4.5857, 4.3000, 4.1857, 4.0857, 3.9714,
    3.8428, 3.8143, 3.8143, 3.7429, 3.8428, 3.9571, 4.0286, 4.0000,
    4.0000, 3.9571, 4.0857, 4.0714, 5.1857, 4.7143, 4.5143, 4.3000,
    4.4572, 4.4429, 4.4286, 4.4143, 4.3571, 4.3714, 4.3571, 4.3429,
    4.3286, 4.3286, 4.3143, 4.3857, 4.2857, 4.1572, 4.0714, 3.9571,
    3.8857, 3.8571, 3.7857, 3.8000, 3.8143, 4.0857, 4.0571, 4.0571,
    4.1429, 4.0714, 4.1715, 4.2285, 5.0714, 4.7857, 4.5857, 4.4572,
    4.3714, 4.3857, 4.4000, 2.6714, 2.8714, 3.9286, 3.5571, 3.3143,
    3.2857, 3.2286, 3.1143, 3.1285, 3.1714, 4.2143, 4.0000, 3.8286,
    3.7000, 3.7857, 3.6714, 3.6286, 3.5857, 4.7286, 4.4143, 4.1143,
    3.9571, 3.9428, 3.7857, 3.7429, 3.7000, 3.6429, 3.6286, 3.6571,
    3.7714, 3.8143, 3.7714, 3.8571, 3.7857, 3.8428, 3.8857, 4.9143,
    4.6000, 4.3143, 4.2000, 4.1000, 3.9857, 3.8571, 3.8286, 3.8286,
    3.7572, 3.8571, 3.9714, 4.0429, 4.0143, 4.0143, 3.9714, 4.1000,
    4.0857, 5.2000, 4.7286, 4.5286, 4.3143, 4.4714, 4.4572, 4.4429,
    4.4286, 4.3714, 4.3857, 4.3714, 4.3571, 4.3429, 4.3429, 4.3286,
    4.4000, 4.3000, 4.1715, 4.0857, 3.9714, 3.9000, 3.8714, 3.8000,
    3.8143, 3.8286, 4.1000, 4.0714, 4.0714, 4.1572, 4.0857, 4.1857,
    4.2428, 5.0857, 4.8000, 4.6000, 4.4714, 4.3857, 4.4000, 4.4143,
    4.4286, 2.5857, 2.7857, 3.8428, 3.4714, 3.2286, 3.2000, 3.1428,
    3.0286, 3.0429, 3.0857, 4.1286, 3.9143, 3.7429, 3.6143, 3.7000,
    3.5857, 3.5428, 3.5000, 4.6428, 4.3286, 4.0286, 3.8714, 3.8571,
    3.7000, 3.6571, 3.6143, 3.5571, 3.5428, 3.5714, 3.6857, 3.7286,
    3.6857, 3.7714, 3.7000, 3.7572, 3.8000, 4.8286, 4.5143, 4.2285,
    4.1143, 4.0143, 3.9000, 3.7714, 3.7429, 3.7429, 3.6714, 3.7714,
    3.8857, 3.9571, 3.9286, 3.9286, 3.8857, 4.0143, 4.0000, 5.1143,
    4.6428, 4.4429, 4.2285, 4.3857, 4.3714, 4.3571, 4.3429, 4.2857,
    4.3000, 4.2857, 4.2714, 4.2571, 4.2571, 4.2428, 4.3143, 4.2143,
    4.0857, 4.0000, 3.8857, 3.8143, 3.7857, 3.7143, 3.7286, 3.7429,
    4.0143, 3.9857, 3.9857, 4.0714, 4.0000, 4.1000, 4.1572, 5.0000,
    4.7143, 4.5143, 4.3857, 4.3000, 4.3143, 4.3286, 4.3429, 4.2571,
    2.5857, 2.7857, 3.8428, 3.4714, 3.2286, 3.2000, 3.1428, 3.0286,
    3.0429, 3.0857, 4.1286, 3.9143, 3.7429, 3.6143, 3.7000, 3.5857,
    3.5428, 3.5000, 4.6428, 4.3286, 4.0286, 3.8714, 3.8571, 3.7000,
    3.6571, 3.6143, 3.5571, 3.5428, 3.5714, 3.6857, 3.7286, 3.6857,
    3.7714, 3.7000, 3.7572, 3.8000, 4.8286, 4.5143, 4.2285, 4.1143,
    4.0143, 3.9000, 3.7714, 3.7429, 3.7429, 3.6714, 3.7714, 3.8857,
    3.9571, 3.9286, 3.9286, 3.8857, 4.0143, 4.0000, 5.1143, 4.6428,
    4.4429, 4.2285, 4.3857, 4.3714, 4.3571, 4.3429, 4.2857, 4.3000,
    4.2857, 4.2714, 4.2571, 4.2571, 4.2428, 4.3143, 4.2143, 4.0857,
    4.0000, 3.8857, 3.8143, 3.7857, 3.7143, 3.7286, 3.7429, 4.0143,
    3.9857, 3.9857, 4.0714, 4.0000, 4.1000, 4.1572, 5.0000, 4.7143,
    4.5143, 4.3857, 4.3000, 4.3143, 4.3286, 4.3429, 4.2571, 4.2571,
    2.6143, 2.8143, 3.8714, 3.5000, 3.2571, 3.2286, 3.1714, 3.0572,
    3.0715, 3.1143, 4.1572, 3.9428, 3.7714, 3.6429, 3.7286, 3.6143,
    3.5714, 3.5286, 4.6714, 4.3571, 4.0571, 3.9000, 3.8857, 3.7286,
    3.6857, 3.6429, 3.5857, 3.5714, 3.6000, 3.7143, 3.7572, 3.7143,
    3.8000, 3.7286, 3.7857, 3.8286, 4.8572, 4.5429, 4.2571, 4.1429,
    4.0429, 3.9286, 3.8000, 3.7714, 3.7714, 3.7000, 3.8000, 3.9143,
    3.9857, 3.9571, 3.9571, 3.9143, 4.0429, 4.0286, 5.1429, 4.6714,
    4.4714, 4.2571, 4.4143, 4.4000, 4.3857, 4.3714, 4.3143, 4.3286,
    4.3143, 4.3000, 4.2857, 4.2857, 4.2714, 4.3429, 4.2428, 4.1143,
    4.0286, 3.9143, 3.8428, 3.8143, 3.7429, 3.7572, 3.7714, 4.0429,
    4.0143, 4.0143, 4.1000, 4.0286, 4.1286, 4.1857, 5.0286, 4.7429,
    4.5429, 4.4143, 4.3286, 4.3429, 4.3571, 4.3714, 4.2857, 4.2857,
    4.3143, 2.6143, 2.8143, 3.8714, 3.5000, 3.2571, 3.2286, 3.1714,
    3.0572, 3.0715, 3.1143, 4.1572, 3.9428, 3.7714, 3.6429, 3.7286,
    3.6143, 3.5714, 3.5286, 4.6714, 4.3571, 4.0571, 3.9000, 3.8857,
    3.7286, 3.6857, 3.6429, 3.5857, 3.5714, 3.6000, 3.7143, 3.7572,
    3.7143, 3.8000, 3.7286, 3.7857, 3.8286, 4.8572, 4.5429, 4.2571,
    4.1429, 4.0429, 3.9286, 3.8000, 3.7714, 3.7714, 3.7000, 3.8000,
    3.9143, 3.9857, 3.9571, 3.9571, 3.9143, 4.0429, 4.0286, 5.1429,
    4.6714, 4.4714, 4.2571, 4.4143, 4.4000, 4.3857, 4.3714, 4.3143,
    4.3286, 4.3143, 4.3000, 4.2857, 4.2857, 4.2714, 4.3429, 4.2428,
    4.1143, 4.0286, 3.9143, 3.8428, 3.8143, 3.7429, 3.7572, 3.7714,
    4.0429, 4.0143, 4.0143, 4.1000, 4.0286, 4.1286, 4.1857, 5.0286,
    4.7429, 4.5429, 4.4143, 4.3286, 4.3429, 4.3571, 4.3714, 4.2857,
    4.2857, 4.3143, 4.3143, 2.5714, 2.7714, 3.8286, 3.4572, 3.2143,
    3.1857, 3.1285, 3.0143, 3.0286, 3.0715, 4.1143, 3.9000, 3.7286,
    3.6000, 3.6857, 3.5714, 3.5286, 3.4857, 4.6286, 4.3143, 4.0143,
    3.8571, 3.8428, 3.6857, 3.6429, 3.6000, 3.5428, 3.5286, 3.5571,
    3.6714, 3.7143, 3.6714, 3.7572, 3.6857, 3.7429, 3.7857, 4.8143,
    4.5000, 4.2143, 4.1000, 4.0000, 3.8857, 3.7572, 3.7286, 3.7286,
    3.6571, 3.7572, 3.8714, 3.9428, 3.9143, 3.9143, 3.8714, 4.0000,
    3.9857, 5.1000, 4.6286, 4.4286, 4.2143, 4.3714, 4.3571, 4.3429,
    4.3286, 4.2714, 4.2857, 4.2714, 4.2571, 4.2428, 4.2428, 4.2285,
    4.3000, 4.2000, 4.0714, 3.9857, 3.8714, 3.8000, 3.7714, 3.7000,
    3.7143, 3.7286, 4.0000, 3.9714, 3.9714, 4.0571, 3.9857, 4.0857,
    4.1429, 4.9857, 4.7000, 4.5000, 4.3714, 4.2857, 4.3000, 4.3143,
    4.3286, 4.2428, 4.2428, 4.2714, 4.2714, 4.2285, 2.6000, 2.8000,
    3.8571, 3.4857, 3.2429, 3.2143, 3.1571, 3.0429, 3.0572, 3.1000,
    4.1429, 3.9286, 3.7572, 3.6286, 3.7143, 3.6000, 3.5571, 3.5143,
    4.6571, 4.3429, 4.0429, 3.8857, 3.8714, 3.7143, 3.6714, 3.6286,
    3.5714, 3.5571, 3.5857, 3.7000, 3.7429, 3.7000, 3.7857, 3.7143,
    3.7714, 3.8143, 4.8429, 4.5286, 4.2428, 4.1286, 4.0286, 3.9143,
    3.7857, 3.7572, 3.7572, 3.6857, 3.7857, 3.9000, 3.9714, 3.9428,
    3.9428, 3.9000, 4.0286, 4.0143, 5.1286, 4.6571, 4.4572, 4.2428,
    4.4000, 4.3857, 4.3714, 4.3571, 4.3000, 4.3143, 4.3000, 4.2857,
    4.2714, 4.2714, 4.2571, 4.3286, 4.2285, 4.1000, 4.0143, 3.9000,
    3.8286, 3.8000, 3.7286, 3.7429, 3.7572, 4.0286, 4.0000, 4.0000,
    4.0857, 4.0143, 4.1143, 4.1715, 5.0143, 4.7286, 4.5286, 4.4000,
    4.3143, 4.3286, 4.3429, 4.3571, 4.2714, 4.2714, 4.3000, 4.3000,
    4.2571, 4.2857, 2.6857, 2.8857, 3.9428, 3.5714, 3.3286, 3.3000,
    3.2429, 3.1285, 3.1428, 3.1857, 4.2285, 4.0143, 3.8428, 3.7143,
    3.8000, 3.6857, 3.6429, 3.6000, 4.7429, 4.4286, 4.1286, 3.9714,
    3.9571, 3.8000, 3.7572, 3.7143, 3.6571, 3.6429, 3.6714, 3.7857,
    3.8286, 3.7857, 3.8714, 3.8000, 3.8571, 3.9000, 4.9285, 4.6143,
    4.3286, 4.2143, 4.1143, 4.0000, 3.8714, 3.8428, 3.8428, 3.7714,
    3.8714, 3.9857, 4.0571, 4.0286, 4.0286, 3.9857, 4.1143, 4.1000,
    5.2143, 4.7429, 4.5429, 4.3286, 4.4857, 4.4714, 4.4572, 4.4429,
    4.3857, 4.4000, 4.3857, 4.3714, 4.3571, 4.3571, 4.3429, 4.4143,
    4.3143, 4.1857, 4.1000, 3.9857, 3.9143, 3.8857, 3.8143, 3.8286,
    3.8428, 4.1143, 4.0857, 4.0857, 4.1715, 4.1000, 4.2000, 4.2571,
    5.1000, 4.8143, 4.6143, 4.4857, 4.4000, 4.4143, 4.4286, 4.4429,
    4.3571, 4.3571, 4.3857, 4.3857, 4.3429, 4.3714, 4.4572, 2.7143,
    2.9143, 3.9714, 3.6000, 3.3572, 3.3286, 3.2714, 3.1571, 3.1714,
    3.2143, 4.2571, 4.0429, 3.8714, 3.7429, 3.8286, 3.7143, 3.6714,
    3.6286, 4.7714, 4.4572, 4.1572, 4.0000, 3.9857, 3.8286, 3.7857,
    3.7429, 3.6857, 3.6714, 3.7000, 3.8143, 3.8571, 3.8143, 3.9000,
    3.8286, 3.8857, 3.9286, 4.9571, 4.6428, 4.3571, 4.2428, 4.1429,
    4.0286, 3.9000, 3.8714, 3.8714, 3.8000, 3.9000, 4.0143, 4.0857,
    4.0571, 4.0571, 4.0143, 4.1429, 4.1286, 5.2429, 4.7714, 4.5714,
    4.3571, 4.5143, 4.5000, 4.4857, 4.4714, 4.4143, 4.4286, 4.4143,
    4.4000, 4.3857, 4.3857, 4.3714, 4.4429, 4.3429, 4.2143, 4.1286,
    4.0143, 3.9428, 3.9143, 3.8428, 3.8571, 3.8714, 4.1429, 4.1143,
    4.1143, 4.2000, 4.1286, 4.2285, 4.2857, 5.1286, 4.8429, 4.6428,
    4.5143, 4.4286, 4.4429, 4.4572, 4.4714, 4.3857, 4.3857, 4.4143,
    4.4143, 4.3714, 4.4000, 4.4857, 4.5143, 2.5286, 2.7286, 3.7857,
    3.4143, 3.1714, 3.1428, 3.0857, 2.9714, 2.9857, 3.0286, 4.0714,
    3.8571, 3.6857, 3.5571, 3.6429, 3.5286, 3.4857, 3.4429, 4.5857,
    4.2714, 3.9714, 3.8143, 3.8000, 3.6429, 3.6000, 3.5571, 3.5000,
    3.4857, 3.5143, 3.6286, 3.6714, 3.6286, 3.7143, 3.6429, 3.7000,
    3.7429, 4.7714, 4.4572, 4.1715, 4.0571, 3.9571, 3.8428, 3.7143,
    3.6857, 3.6857, 3.6143, 3.7143, 3.8286, 3.9000, 3.8714, 3.8714,
    3.8286, 3.9571, 3.9428, 5.0571, 4.5857, 4.3857, 4.1715, 4.3286,
    4.3143, 4.3000, 4.2857, 4.2285, 4.2428, 4.2285, 4.2143, 4.2000,
    4.2000, 4.1857, 4.2571, 4.1572, 4.0286, 3.9428, 3.8286, 3.7572,
    3.7286, 3.6571, 3.6714, 3.6857, 3.9571, 3.9286, 3.9286, 4.0143,
    3.9428, 4.0429, 4.1000, 4.9428, 4.6571, 4.4572, 4.3286, 4.2428,
    4.2571, 4.2714, 4.2857, 4.2000, 4.2000, 4.2285, 4.2285, 4.1857,
    4.2143, 4.3000, 4.3286, 4.1429,
])


def _eeqbc_surrogate_remainder(atoms, cartesian_map, scale):
    """Variational EEQBC2025 energy beyond its local quadratic jet."""
    from scipy.special import erf, erfc
    dimension = cartesian_map.shape[1]
    numbers = atoms.numbers
    count = len(atoms)
    if (atoms.pbc.any() or count < 2
            or np.any((numbers < 1) | (numbers > len(_EEQBC2025_ATOMIC)))):
        return lambda y: (0.0, np.zeros(dimension))
    (chi, eta, atomic_width, kcnchi, kqchi, kqeta, capacity,
     diameter, average_cn) = _EEQBC2025_ATOMIC[numbers - 1].T
    radii = 0.5 * diameter
    electronegativity = _EEQBC2025_PAULING[numbers - 1].copy()
    electronegativity[numbers >= 90] = 1.30
    electronegativity[numbers == 87] = 0.80
    electronegativity[numbers == 89] = 1.00
    electronegativity[np.isin(numbers, [90, 91, 92, 95])] = 1.10
    electronegativity[np.isin(numbers, [93, 94, 97, 103])] = 1.20
    electronegativity /= 3.98
    first, second = np.triu_indices(count, 1)
    pair_origin = (atoms.positions[first] - atoms.positions[second]) / units.Bohr
    mapping = cartesian_map[:3 * count].reshape(count, 3, dimension) / units.Bohr
    pair_map = mapping[first] - mapping[second]
    radius_sum = radii[first] + radii[second]
    maximum_z = np.maximum(numbers[first], numbers[second])
    minimum_z = np.minimum(numbers[first], numbers[second])
    vdw_pair = _EEQBC2025_VDW_PAIR[minimum_z - 1 + maximum_z * (maximum_z - 1) // 2]
    # The author code uses get_vdw_rad()*autoaa, i.e. the unconverted table.
    capacity_pair = np.sqrt(capacity[first] * capacity[second])
    en_difference = electronegativity[second] - electronegativity[first]
    width_slope = -0.14 * atomic_width / average_cn**0.75
    conversion = units.Hartree / scale
    self_constant = np.sqrt(2.0 / np.pi)

    def evaluate(y):
        pairs = pair_origin + np.einsum('pij,j->pi', pair_map, y)
        radius = np.sqrt(np.sum(pairs * pairs, axis=1))
        if np.any(radius.real <= 0.0) or not np.all(np.isfinite(radius)):
            raise ValueError('Invalid EEQBC pair distance')
        argument = 2.0 * (radius - radius_sum) / radius_sum**0.75
        support = radius.real <= 25.0
        cn_pair = np.where(support, 0.5 * erfc(argument), 0.0)
        cn_slope = np.where(support, -2.0 * np.exp(-argument**2)
                            / (np.sqrt(np.pi) * radius_sum**0.75), 0.0)
        coordination = np.zeros(count, dtype=pairs.dtype)
        local_charge = np.zeros(count, dtype=pairs.dtype)
        np.add.at(coordination, first, cn_pair)
        np.add.at(coordination, second, cn_pair)
        np.add.at(local_charge, first, en_difference * cn_pair)
        np.add.at(local_charge, second, -en_difference * cn_pair)
        width = atomic_width + width_slope * coordination
        if np.any(width.real <= 0.0) or not np.all(np.isfinite(width)):
            raise ValueError('Invalid EEQBC effective width')
        hardness = eta + kqeta * local_charge + self_constant / width
        linear = -chi + kcnchi * coordination + kqchi * local_charge
        beta = 1.0 / np.sqrt(width[first]**2 + width[second]**2)
        kernel = erf(beta * radius) / radius
        gaussian = 2.0 / np.sqrt(np.pi) * np.exp(-(beta * radius)**2)
        kernel_slope = (beta * gaussian - kernel) / radius
        capacity_argument = 0.60 * (radius / vdw_pair - 1.0)
        capacitance = 0.5 * capacity_pair * erfc(capacity_argument)
        capacitance_slope = (-0.60 * capacity_pair * np.exp(-capacity_argument**2)
                             / (np.sqrt(np.pi) * vdw_pair))
        laplacian = np.zeros((count, count), dtype=pairs.dtype)
        laplacian[first, second] = -capacitance
        laplacian[second, first] = -capacitance
        diagonal = -np.sum(laplacian, axis=1)
        laplacian[np.arange(count), np.arange(count)] = diagonal
        rhs = np.concatenate((laplacian @ linear, np.zeros(1, dtype=pairs.dtype)))
        matrix = np.zeros((count + 1, count + 1), dtype=pairs.dtype)
        matrix[np.arange(count), np.arange(count)] = 1.0 + hardness * diagonal
        matrix[first, second] = -capacitance * kernel
        matrix[second, first] = -capacitance * kernel
        matrix[:count, count] = 1.0
        matrix[count, :count] = 1.0
        try:
            charges = np.linalg.solve(matrix, rhs)[:count]
        except np.linalg.LinAlgError as exc:
            raise ValueError('Singular EEQBC response') from exc
        # Derivatives of the explicit functional with q held stationary.
        charge_product = charges[first] * charges[second]
        capacity_coefficient = (
            0.5 * (charges[first]**2 * hardness[first]
                   + charges[second]**2 * hardness[second]) - charge_product * kernel
            - (charges[first] - charges[second]) * (linear[first] - linear[second]))
        width_gradient = -0.5 * charges**2 * diagonal * self_constant / width**2
        width_pair = capacitance * charge_product * gaussian * beta**3
        np.add.at(width_gradient, first, width_pair * width[first])
        np.add.at(width_gradient, second, width_pair * width[second])
        charge_flow = laplacian @ charges
        cn_gradient = width_gradient * width_slope - charge_flow * kcnchi
        local_gradient = 0.5 * charges**2 * diagonal * kqeta - charge_flow * kqchi
        pair_slope = (capacitance_slope * capacity_coefficient
                      - capacitance * charge_product * kernel_slope
                      + cn_slope * (cn_gradient[first] + cn_gradient[second]
                                    + en_difference * (local_gradient[first]
                                                       - local_gradient[second])))
        pair_gradient = pair_slope[:, None] * pairs / radius[:, None]
        value = -0.5 * rhs[:count] @ charges * conversion
        gradient = np.einsum('pi,pij->j', pair_gradient, pair_map) * conversion
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ValueError('Nonfinite EEQBC energy or gradient')
        return value, gradient

    value0, gradient0 = evaluate(np.zeros(dimension))
    spacing = 1e-20
    hessian0 = np.column_stack([
        evaluate(1j * spacing * axis)[1].imag / spacing
        for axis in np.eye(dimension)])
    hessian0 = 0.5 * (hessian0 + hessian0.T)
    if not np.all(np.isfinite(hessian0)):
        raise ValueError('Nonfinite EEQBC curvature')

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
            micro_eeqbc = _eeqbc_surrogate_remainder(
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
            micro_value, micro_gradient = micro_eeqbc(y)
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
