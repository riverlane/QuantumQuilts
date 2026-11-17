"""Binary symplectic algebra for general Pauli quilt subsystems.

Paulis on ``n`` qubits are integer bitmasks in two-block ``X | Z`` form:
the low ``n`` bits are the X support and the next ``n`` bits are the Z
support.  A Y has both bits set.  Global phases are intentionally discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ._gf2 import all_ones, nullspace, parity, rank, reduce_with_basis, row_reduce


def pauli(x: int, z: int, n: int) -> int:
    """Pack X and Z support masks into a two-block symplectic row."""

    if n < 0 or x < 0 or z < 0:
        raise ValueError("n and Pauli support masks must be non-negative")
    mask = all_ones(n)
    if x & ~mask or z & ~mask:
        raise ValueError("Pauli support exceeds the declared qubit count")
    return x | (z << n)


def x_part(row: int, n: int) -> int:
    return int(row) & all_ones(n)


def z_part(row: int, n: int) -> int:
    return (int(row) >> n) & all_ones(n)


def pauli_weight(row: int, n: int) -> int:
    """Return physical Pauli weight, counting Y once."""

    return (x_part(row, n) | z_part(row, n)).bit_count()


def symplectic_product(left: int, right: int, n: int) -> int:
    """Return one iff two phase-free Paulis anticommute."""

    return parity(
        (x_part(left, n) & z_part(right, n))
        ^ (z_part(left, n) & x_part(right, n))
    )


def symplectic_dual(row: int, n: int) -> int:
    """Swap X and Z blocks so ordinary dot product gives commutation."""

    return pauli(z_part(row, n), x_part(row, n), n)


def xor_rows(coefficients: int, rows: Sequence[int]) -> int:
    out = 0
    for index, row in enumerate(rows):
        if (coefficients >> index) & 1:
            out ^= int(row)
    return out


def symplectic_centralizer(rows: Iterable[int], n: int) -> list[int]:
    """Return a basis for all Paulis commuting with every supplied row."""

    basis, _ = row_reduce(rows, 2 * n)
    centralizer, _ = row_reduce(
        nullspace([symplectic_dual(row, n) for row in basis], 2 * n),
        2 * n,
    )
    return centralizer


def gauge_center(gauge_rows: Iterable[int], n: int) -> list[int]:
    """Return ``span(G) intersect C(G)`` for a possibly nonabelian gauge group."""

    gauge, _ = row_reduce(gauge_rows, 2 * n)
    coefficient_checks = []
    for right in gauge:
        check = 0
        for index, left in enumerate(gauge):
            check |= symplectic_product(left, right, n) << index
        coefficient_checks.append(check)
    coefficient_kernel = nullspace(coefficient_checks, len(gauge))
    center, _ = row_reduce((xor_rows(coefficients, gauge) for coefficients in coefficient_kernel), 2 * n)
    return center


def quotient_complement(ambient_rows: Iterable[int], subspace_rows: Iterable[int], width: int) -> list[int]:
    """Choose rows whose cosets form a basis of ``ambient/subspace``."""

    space, pivots = row_reduce(subspace_rows, width)
    echelon = {pivot: row for pivot, row in zip(pivots, space, strict=True)}
    ambient, _ = row_reduce(ambient_rows, width)
    complement: list[int] = []
    for row in ambient:
        remainder = int(row)
        while remainder:
            pivot = remainder.bit_length() - 1
            basis_row = echelon.get(pivot)
            if basis_row is None:
                echelon[pivot] = remainder
                complement.append(row)
                break
            remainder ^= basis_row
    return complement


def rows_commute(rows: Sequence[int], n: int) -> bool:
    return all(
        symplectic_product(rows[i], rows[j], n) == 0
        for i in range(len(rows))
        for j in range(i + 1, len(rows))
    )


@dataclass(frozen=True)
class GeneralSubsystem:
    n: int
    base_stabilizers: tuple[int, ...]
    added_gauges: tuple[int, ...]
    gauge_basis: tuple[int, ...]
    center_basis: tuple[int, ...]
    bare_logical_basis: tuple[int, ...]
    dressed_logical_basis: tuple[int, ...]
    gauge_qubits: int
    logical_qubits: int


@dataclass(frozen=True)
class MeasurementAnalysis:
    measured_rank: int
    desired_rank: int
    contains_desired: bool
    no_extra_measurements: bool
    exact: bool
    demoted_stabilizers: int
    logical_gauge_qubits: int
    preserves_unmeasured_logicals: bool
    valid: bool


def build_general_subsystem(
    *,
    n: int,
    base_stabilizers: Sequence[int],
    added_gauges: Sequence[int],
) -> GeneralSubsystem:
    """Build the center and dressed-logical quotient of a Pauli gauge group."""

    if n < 0:
        raise ValueError("n must be non-negative")
    width = 2 * n
    limit = 1 << width
    base = tuple(int(row) for row in base_stabilizers)
    added = tuple(int(row) for row in added_gauges)
    if any(row < 0 or row >= limit for row in (*base, *added)):
        raise ValueError("a Pauli row exceeds the declared symplectic width")
    if not rows_commute(base, n):
        raise ValueError("base stabilizers must commute")

    gauge, _ = row_reduce([*base, *added], width)
    center = gauge_center(gauge, n)
    center_centralizer = symplectic_centralizer(center, n)
    gauge_centralizer = symplectic_centralizer(gauge, n)
    if any(reduce_with_basis(center_centralizer, row, width) for row in gauge):
        raise RuntimeError("internal error: gauge group does not centralize its center")
    dressed_logicals = quotient_complement(center_centralizer, gauge, width)
    bare_logicals = quotient_complement(gauge_centralizer, center, width)

    gauge_minus_center = len(gauge) - len(center)
    if gauge_minus_center & 1:
        raise RuntimeError("gauge rank minus center rank must be even")
    gauge_qubits = gauge_minus_center // 2
    logical_qubits = n - len(center) - gauge_qubits
    if len(dressed_logicals) != 2 * logical_qubits or len(bare_logicals) != 2 * logical_qubits:
        raise RuntimeError("dressed logical quotient has inconsistent dimension")
    return GeneralSubsystem(
        n=n,
        base_stabilizers=tuple(row_reduce(base, width)[0]),
        added_gauges=added,
        gauge_basis=tuple(gauge),
        center_basis=tuple(center),
        bare_logical_basis=tuple(bare_logicals),
        dressed_logical_basis=tuple(dressed_logicals),
        gauge_qubits=gauge_qubits,
        logical_qubits=logical_qubits,
    )


def analyze_measurement(
    subsystem: GeneralSubsystem,
    desired_logicals: Sequence[int],
) -> MeasurementAnalysis:
    """Check the center modulo the original stabilizers against a target space.

    Exactness means

    ``span(S0, center(G)) == span(S0, desired_logicals)``.

    Consequently the newly fixed logical subspace is precisely the requested
    incidence subspace, including for incidence rank greater than one.
    """

    width = 2 * subsystem.n
    base = list(subsystem.base_stabilizers)
    desired_rows = [int(row) for row in desired_logicals]
    if any(symplectic_product(row, stabilizer, subsystem.n) for row in desired_rows for stabilizer in base):
        raise ValueError("desired rows must represent logical operators of the original code")
    if not rows_commute(desired_rows, subsystem.n):
        raise ValueError("the desired measurement subspace must commute")
    measured, _ = row_reduce([*base, *subsystem.center_basis], width)
    desired, _ = row_reduce([*base, *desired_rows], width)
    base_rank = rank(base, width)
    measured_rank = len(measured) - base_rank
    desired_rank = len(desired) - base_rank
    contains_desired = all(reduce_with_basis(measured, row, width) == 0 for row in desired)
    no_extra = all(reduce_with_basis(desired, row, width) == 0 for row in measured)
    exact = contains_desired and no_extra
    center_rank = len(subsystem.center_basis)
    surviving_original_stabilizers = center_rank + base_rank - len(measured)
    demoted_stabilizers = base_rank - surviving_original_stabilizers
    logical_gauge_qubits = subsystem.gauge_qubits - demoted_stabilizers
    preserves = logical_gauge_qubits == 0
    return MeasurementAnalysis(
        measured_rank=measured_rank,
        desired_rank=desired_rank,
        contains_desired=contains_desired,
        no_extra_measurements=no_extra,
        exact=exact,
        demoted_stabilizers=demoted_stabilizers,
        logical_gauge_qubits=logical_gauge_qubits,
        preserves_unmeasured_logicals=preserves,
        valid=exact and preserves,
    )
