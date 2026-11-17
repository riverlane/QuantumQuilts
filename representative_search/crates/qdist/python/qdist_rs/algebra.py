from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def binary_matrix(matrix: Any, *, name: str = "matrix") -> np.ndarray:
    array = np.asarray(matrix, dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    return np.ascontiguousarray(array & 1)


def rref(matrix: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = binary_matrix(matrix).copy()
    rank = 0
    pivots: list[int] = []
    for column in range(array.shape[1]):
        candidates = np.flatnonzero(array[rank:, column])
        if not len(candidates):
            continue
        pivot = rank + int(candidates[0])
        array[[rank, pivot]] = array[[pivot, rank]]
        for row in np.flatnonzero(array[:, column]):
            if row != rank:
                array[row] ^= array[rank]
        pivots.append(column)
        rank += 1
        if rank == array.shape[0]:
            break
    return np.ascontiguousarray(array[:rank]), tuple(pivots)


def row_basis(matrix: Any) -> np.ndarray:
    return rref(matrix)[0]


def nullspace(checks: Any) -> np.ndarray:
    matrix = binary_matrix(checks, name="checks")
    reduced, pivots = rref(matrix)
    pivot_set = set(pivots)
    free = [column for column in range(matrix.shape[1]) if column not in pivot_set]
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    for index, column in enumerate(free):
        basis[index, column] = 1
        for row, pivot in enumerate(pivots):
            basis[index, pivot] = reduced[row, column]
    return np.ascontiguousarray(basis)


def quotient_complement(ambient: Any, subspace: Any) -> np.ndarray:
    ambient_basis = row_basis(ambient)
    current = row_basis(subspace)
    selected: list[np.ndarray] = []
    rank = current.shape[0]
    for row in ambient_basis:
        trial = row_basis(np.vstack([current, row]))
        if trial.shape[0] > rank:
            selected.append(row.copy())
            current = trial
            rank += 1
    width = ambient_basis.shape[1]
    return np.ascontiguousarray(np.asarray(selected, dtype=np.uint8).reshape((-1, width)))


def symplectic_dual(matrix: Any) -> np.ndarray:
    array = binary_matrix(matrix)
    if array.shape[1] % 2:
        raise ValueError("a symplectic matrix must have width 2n")
    n = array.shape[1] // 2
    return np.ascontiguousarray(np.concatenate([array[:, n:], array[:, :n]], axis=1))


def symplectic_commutator(left: Any, right: Any) -> np.ndarray:
    left_array = binary_matrix(left, name="left")
    right_array = binary_matrix(right, name="right")
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("symplectic matrices must have equal widths")
    return np.ascontiguousarray((left_array @ symplectic_dual(right_array).T) & 1)


@dataclass(frozen=True)
class SubsystemProblem:
    gauge: np.ndarray
    center: np.ndarray
    bare_logicals: np.ndarray
    n: int
    gauge_rank: int
    center_rank: int
    gauge_qubits: int
    logical_qubits: int


@dataclass(frozen=True)
class OperationProblem:
    """A dressed subsystem quotient with a declared benign logical subspace.

    The operation distance searches ``S_aux^perp / (G_aux + C)``, where
    ``S_aux = Z(G_aux)`` and ``C`` is the span of benign measured products.
    Unlike the dressed distance of ``<G_aux, C>``, the detector space remains
    ``S_aux``; errors that flip a measured product therefore remain visible.
    """

    auxiliary_gauge: np.ndarray
    center: np.ndarray
    benign: np.ndarray
    trivial: np.ndarray
    operation_probes: np.ndarray
    n: int
    auxiliary_logical_qubits: int
    benign_rank: int
    operation_dimension: int


@dataclass(frozen=True)
class CenterDecomposition:
    """Pure and mixed parts of a symplectic center.

    ``mixed`` spans ``S / (S_X + S_Z)``. The corresponding pure-Z
    ``cancellation_probes`` detect the common nonzero X/Z cancellation
    syndrome of an error in ``S^perp``.
    """

    pure_x: np.ndarray
    pure_z: np.ndarray
    mixed: np.ndarray
    cancellation_probes: np.ndarray

    @property
    def cancellation_rank(self) -> int:
        return self.mixed.shape[0]


def decompose_center(center: Any) -> CenterDecomposition:
    """Split a center into pure-X, pure-Z, and mixed quotient bases."""

    center_basis = row_basis(binary_matrix(center, name="center"))
    if center_basis.shape[1] % 2:
        raise ValueError("center rows must have width 2n")
    n = center_basis.shape[1] // 2
    x = center_basis[:, :n]
    z = center_basis[:, n:]
    pure_x_coefficients = nullspace(z.T)
    pure_z_coefficients = nullspace(x.T)
    pure_x = row_basis((pure_x_coefficients @ center_basis) & 1)
    pure_z = row_basis((pure_z_coefficients @ center_basis) & 1)
    split = row_basis(np.vstack([pure_x, pure_z]))
    mixed = quotient_complement(center_basis, split)
    probes = np.zeros_like(mixed)
    probes[:, n:] = mixed[:, n:]
    return CenterDecomposition(
        pure_x=pure_x,
        pure_z=pure_z,
        mixed=mixed,
        cancellation_probes=np.ascontiguousarray(probes),
    )


def build_subsystem(gauge: Any) -> SubsystemProblem:
    """Derive ``S=Z(G)`` and a bare-logical basis from arbitrary gauge rows."""

    gauge_array = binary_matrix(gauge, name="gauge")
    if gauge_array.shape[1] % 2:
        raise ValueError("gauge rows must have width 2n")
    gauge_basis = row_basis(gauge_array)
    commutator = symplectic_commutator(gauge_basis, gauge_basis)
    center_coefficients = nullspace(commutator)
    center = row_basis((center_coefficients @ gauge_basis) & 1)
    gauge_centralizer = nullspace(symplectic_dual(gauge_basis))
    bare_logicals = quotient_complement(gauge_centralizer, center)
    gauge_minus_center = gauge_basis.shape[0] - center.shape[0]
    if gauge_minus_center % 2:
        raise ValueError("gauge rank minus center rank is odd")
    n = gauge_basis.shape[1] // 2
    gauge_qubits = gauge_minus_center // 2
    logical_qubits = n - center.shape[0] - gauge_qubits
    if bare_logicals.shape[0] != 2 * logical_qubits:
        raise RuntimeError("bare-logical quotient has inconsistent dimension")
    if np.any(symplectic_commutator(center, gauge_basis)):
        raise RuntimeError("computed center does not centralize the gauge group")
    return SubsystemProblem(
        gauge=gauge_basis,
        center=center,
        bare_logicals=bare_logicals,
        n=n,
        gauge_rank=gauge_basis.shape[0],
        center_rank=center.shape[0],
        gauge_qubits=gauge_qubits,
        logical_qubits=logical_qubits,
    )


def build_operation_problem(auxiliary_gauge: Any, benign: Any) -> OperationProblem:
    """Build the quotient used by a measurement operation distance.

    Args:
        auxiliary_gauge: Symplectic rows spanning the auxiliary gauge space
            ``G_aux``.
        benign: Commuting symplectic rows spanning the deliberately measured
            logical products ``C``. They must centralize ``G_aux``.

    Returns:
        The detector centre ``S_aux`` and probes detecting membership outside
        ``G_aux + C``.
    """

    auxiliary = build_subsystem(auxiliary_gauge)
    benign_basis = row_basis(binary_matrix(benign, name="benign"))
    if benign_basis.shape[1] != auxiliary.gauge.shape[1]:
        raise ValueError("auxiliary_gauge and benign must have equal widths")
    if np.any(symplectic_commutator(benign_basis, auxiliary.gauge)):
        raise ValueError("benign products must centralize the auxiliary gauge group")
    if np.any(symplectic_commutator(benign_basis, benign_basis)):
        raise ValueError("benign products must commute with one another")

    trivial = row_basis(np.vstack([auxiliary.gauge, benign_basis]))
    benign_rank = trivial.shape[0] - auxiliary.gauge_rank
    trivial_centralizer = nullspace(symplectic_dual(trivial))
    operation_probes = quotient_complement(trivial_centralizer, auxiliary.center)
    operation_dimension = 2 * auxiliary.logical_qubits - benign_rank
    if operation_probes.shape[0] != operation_dimension:
        raise RuntimeError("operation quotient has inconsistent dimension")

    return OperationProblem(
        auxiliary_gauge=auxiliary.gauge,
        center=auxiliary.center,
        benign=benign_basis,
        trivial=trivial,
        operation_probes=operation_probes,
        n=auxiliary.n,
        auxiliary_logical_qubits=auxiliary.logical_qubits,
        benign_rank=benign_rank,
        operation_dimension=operation_dimension,
    )
