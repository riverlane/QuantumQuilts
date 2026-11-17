"""Stim-flow verification of claimed parities.

Nothing here trusts the compiler's bookkeeping: every detector and
observable is checked to be a deterministic parity of the ideal circuit, and
`complete_detector_set` refuses a set that is not an independent basis of the
deterministic parity space. This is the check that catches an observable
referencing gauge randomness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import stim

from ._gf2 import nullspace, reduce_with_basis, row_reduce
from ._symplectic import pauli, x_part, z_part


def stim_pauli_from_row(row: int, n: int) -> stim.PauliString:
    x = x_part(row, n)
    z = z_part(row, n)
    labels = [
        ("_", "X", "Z", "Y")[((x >> q) & 1) | (((z >> q) & 1) << 1)]
        for q in range(n)
    ]
    return stim.PauliString("".join(labels))


def row_from_stim_pauli(value: stim.PauliString, n: int | None = None) -> int:
    if n is None:
        n = len(value)
    if len(value) > n:
        raise ValueError("Stim Pauli exceeds the declared symplectic width")
    # Bulk conversion rather than a Python loop over qubits.  The loop cost
    # one iteration per qubit per call -- 576 at bb288, several thousand calls
    # per build -- and was ~10% of a whole circuit build for no reason: stim
    # already holds the X and Z bits as arrays.  Verified identical to the
    # per-qubit form on random Paulis; ~85x faster at 576 qubits.
    xs, zs = value.to_numpy()
    x = int.from_bytes(np.packbits(xs, bitorder="little").tobytes(), "little")
    z = int.from_bytes(np.packbits(zs, bitorder="little").tobytes(), "little")
    return pauli(x, z, n)


def measurement_mask(indices: Sequence[int]) -> int:
    out = 0
    for index in indices:
        index = int(index)
        if index < 0:
            raise ValueError("flow measurement indices must be absolute and non-negative")
        out ^= 1 << index
    return out


def measurement_indices(mask: int) -> list[int]:
    return [index for index in range(int(mask).bit_length()) if (int(mask) >> index) & 1]


def flow_boundary_mask(flow: stim.Flow, n: int) -> int:
    input_row = row_from_stim_pauli(flow.input_copy(), n)
    output_row = row_from_stim_pauli(flow.output_copy(), n)
    return input_row | (output_row << (2 * n))


def _reduce_augmented(
    basis: dict[int, tuple[int, int]],
    boundary: int,
    measurements: int,
) -> tuple[int, int]:
    boundary = int(boundary)
    measurements = int(measurements)
    while boundary:
        pivot = boundary.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            break
        boundary ^= existing[0]
        measurements ^= existing[1]
    return boundary, measurements


@dataclass(frozen=True)
class FlowMeasurementSolver:
    """A cached Pauli-flow basis for repeated queries on one circuit."""

    circuit: stim.Circuit
    n: int
    basis: dict[int, tuple[int, int]]

    @classmethod
    def build(cls, circuit: stim.Circuit) -> "FlowMeasurementSolver":
        n = circuit.num_qubits
        basis: dict[int, tuple[int, int]] = {}
        for generator in circuit.flow_generators():
            boundary = flow_boundary_mask(generator, n)
            measurements = measurement_mask(generator.measurements_copy())
            boundary, measurements = _reduce_augmented(
                basis, boundary, measurements
            )
            if boundary:
                basis[boundary.bit_length() - 1] = (boundary, measurements)
        return cls(circuit=circuit, n=n, basis=basis)

    def solve(self, *, input_row: int, output_row: int) -> int:
        target = int(input_row) | (int(output_row) << (2 * self.n))
        remainder, solution = _reduce_augmented(self.basis, target, 0)
        if remainder:
            raise ValueError(
                "requested Pauli flow is not supported by the physical circuit"
            )
        flow = stim.Flow(
            input=stim_pauli_from_row(input_row, self.n),
            output=stim_pauli_from_row(output_row, self.n),
            measurements=measurement_indices(solution),
        )
        if not self.circuit.has_flow(flow, unsigned=True):
            raise AssertionError("internal flow solution failed Stim validation")
        return solution


def solve_flow_measurements(
    circuit: stim.Circuit,
    *,
    input_row: int,
    output_row: int,
) -> int:
    return FlowMeasurementSolver.build(circuit).solve(
        input_row=input_row,
        output_row=output_row,
    )


def deterministic_measurement_basis(circuit: stim.Circuit, *, seed: int = 0) -> tuple[int, ...]:
    """Return all deterministic parities using a rank-certified affine sample.

    Stim supplies the exact number of determined measurements. Sampling stops
    only when the observed difference space reaches the complementary exact
    rank, at which point its orthogonal complement is the full deterministic
    parity space.
    """

    num_measurements = circuit.num_measurements
    # Purify the circuit input with unmeasured Bell partners. The system's
    # reduced input is then maximally mixed, excluding relations that hold only
    # because Stim otherwise initializes unreset qubits in |0>.
    n = circuit.num_qubits
    purification = stim.Circuit()
    purification.append("H", range(n, 2 * n))
    purification.append(
        "CX",
        [target for q in range(n) for target in (n + q, q)],
    )
    analysis_circuit = purification + circuit
    deterministic_rank = int(analysis_circuit.count_determined_measurements())
    random_rank = num_measurements - deterministic_rank
    sampler = analysis_circuit.compile_sampler(seed=int(seed))
    baseline = np.asarray(sampler.sample(1)[0], dtype=np.uint8)
    pivots: dict[int, int] = {}

    def bool_row_to_int(row: np.ndarray) -> int:
        packed = np.packbits(np.asarray(row, dtype=np.uint8), bitorder="little")
        return int.from_bytes(packed.tobytes(), byteorder="little")

    stagnant_batches = 0
    while len(pivots) < random_rank:
        before = len(pivots)
        for sample in sampler.sample(256):
            value = bool_row_to_int(np.asarray(sample, dtype=np.uint8) ^ baseline)
            while value:
                pivot = value.bit_length() - 1
                existing = pivots.get(pivot)
                if existing is None:
                    pivots[pivot] = value
                    break
                value ^= existing
        if len(pivots) == before:
            stagnant_batches += 1
            if stagnant_batches >= 8:
                raise RuntimeError(
                    f"noiseless samples stalled at random rank {len(pivots)} of {random_rank}"
                )
        else:
            stagnant_batches = 0

    deterministic, _ = row_reduce(nullspace(pivots.values(), num_measurements), num_measurements)
    if len(deterministic) != deterministic_rank:
        raise AssertionError(
            f"deterministic rank mismatch: built {len(deterministic)}, expected {deterministic_rank}"
        )
    return tuple(deterministic)


def sparse_deterministic_candidates(circuit: stim.Circuit) -> tuple[int, ...]:
    """Extract a sparse basis of zero-boundary measurement flows.

    Stim's flow generators are local for these syndrome-extraction circuits.
    Eliminating only their Pauli boundaries exposes deterministic measurement
    relations without row-reducing the measurement part into a dense basis.
    """

    boundary_basis: dict[int, tuple[int, int]] = {}
    measurement_basis: dict[int, int] = {}
    candidates: list[int] = []

    def add_measurement(row: int) -> bool:
        original = int(row)
        while row:
            pivot = row.bit_length() - 1
            existing = measurement_basis.get(pivot)
            if existing is None:
                measurement_basis[pivot] = int(row)
                candidates.append(original)
                return True
            row ^= existing
        return False

    for generator in circuit.flow_generators():
        boundary = flow_boundary_mask(generator, circuit.num_qubits)
        measurements = measurement_mask(generator.measurements_copy())
        while boundary:
            pivot = boundary.bit_length() - 1
            existing = boundary_basis.get(pivot)
            if existing is None:
                boundary_basis[pivot] = (boundary, measurements)
                break
            boundary ^= existing[0]
            measurements ^= existing[1]
        if not boundary and measurements:
            add_measurement(measurements)
    return tuple(candidates)


def complete_detector_set(
    deterministic_basis: Sequence[int],
    observables: Sequence[int],
    preferred_detectors: Sequence[int],
    sparse_candidates: Sequence[int],
    num_measurements: int,
) -> tuple[int, ...]:
    """Complete preferred detector flows without changing their sparse basis.

    The returned detectors and observables are an independent basis of the
    full deterministic parity space. Preferred relations are considered
    first; sparse zero-boundary Stim flows fill any remaining rank.
    """

    observable_basis, _ = row_reduce(observables, num_measurements)
    if len(observable_basis) != len(observables):
        raise ValueError("observable measurement parities are linearly dependent")
    if any(reduce_with_basis(deterministic_basis, row, num_measurements) for row in observables):
        raise ValueError("an observable is not a deterministic parity of the ideal circuit")

    pivots: dict[int, int] = {}

    def add(row: int) -> bool:
        row = int(row)
        while row:
            pivot = row.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = row
                return True
            row ^= existing
        return False

    for row in observable_basis:
        if not add(row):
            raise AssertionError("observable basis unexpectedly lost rank")

    detectors: list[int] = []
    for relation in preferred_detectors:
        relation = int(relation)
        if reduce_with_basis(deterministic_basis, relation, num_measurements):
            raise ValueError("a preferred detector is not deterministic")
        if add(relation):
            detectors.append(relation)

    fill_candidates = [*sparse_candidates, *deterministic_basis]
    for relation in sorted(
        dict.fromkeys(map(int, fill_candidates)),
        key=lambda row: (row.bit_count(), row),
    ):
        if reduce_with_basis(deterministic_basis, relation, num_measurements):
            continue
        if add(relation):
            detectors.append(relation)
        if len(pivots) == len(deterministic_basis):
            break
    if len(pivots) != len(deterministic_basis):
        raise AssertionError("detectors and observables do not span all deterministic parities")
    return tuple(detectors)


def independent_detector_subset(
    deterministic_basis: Sequence[int],
    observables: Sequence[int],
    detector_candidates: Sequence[int],
    num_measurements: int,
) -> tuple[int, ...]:
    """Select independent deterministic detectors without completing the space.

    This is useful when a decoder should intentionally receive only a chosen
    subspace of the available deterministic relations. Candidate order is
    preserved so callers can supply a physically meaningful detector basis.
    """

    observable_basis, _ = row_reduce(observables, num_measurements)
    if len(observable_basis) != len(observables):
        raise ValueError("observable measurement parities are linearly dependent")
    if any(reduce_with_basis(deterministic_basis, row, num_measurements) for row in observables):
        raise ValueError("an observable is not a deterministic parity of the ideal circuit")

    pivots: dict[int, int] = {}

    def add(row: int) -> bool:
        row = int(row)
        while row:
            pivot = row.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = row
                return True
            row ^= existing
        return False

    for row in observable_basis:
        if not add(row):
            raise AssertionError("observable basis unexpectedly lost rank")

    detectors: list[int] = []
    for relation in detector_candidates:
        relation = int(relation)
        if reduce_with_basis(deterministic_basis, relation, num_measurements):
            raise ValueError("a detector candidate is not deterministic")
        if add(relation):
            detectors.append(relation)
    return tuple(detectors)


def detector_complement(
    deterministic_basis: Sequence[int],
    observables: Sequence[int],
    num_measurements: int,
) -> tuple[int, ...]:
    """Choose sparse detector generators complementary to chosen observables."""

    observable_basis, _ = row_reduce(observables, num_measurements)
    if len(observable_basis) != len(observables):
        raise ValueError("observable measurement parities are linearly dependent")
    if any(not reduce_with_basis(deterministic_basis, row, num_measurements) == 0 for row in observables):
        raise ValueError("an observable is not a deterministic parity of the ideal circuit")

    pivots: dict[int, int] = {}

    def add(row: int) -> bool:
        row = int(row)
        while row:
            pivot = row.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = row
                return True
            row ^= existing
        return False

    for row in observable_basis:
        if not add(row):
            raise AssertionError("observable basis unexpectedly lost rank")
    detectors: list[int] = []
    for relation in sorted(deterministic_basis, key=lambda row: (int(row).bit_count(), int(row))):
        if add(relation):
            detectors.append(int(relation))
    if len(pivots) != len(deterministic_basis):
        raise AssertionError("detectors and observables do not span all deterministic parities")
    return tuple(detectors)
