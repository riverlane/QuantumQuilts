import numpy as np
import pytest

from qdist_rs import (
    build_operation_problem,
    build_subsystem,
    decompose_center,
    search_gauge,
    search_operation,
)
from qdist_rs.algebra import row_basis, symplectic_commutator


SYMPLECTIC_METHODS = (
    "grouped",
    "local-clifford",
    "location-linear",
    "three-block",
    "three-block-global",
)


def symplectic_weight(witness: bytes) -> int:
    row = np.frombuffer(witness, dtype=np.uint8)
    n = len(row) // 2
    return sum(bool(row[q] or row[q + n]) for q in range(n))


def five_qubit_stabilizers() -> np.ndarray:
    """Cyclic generators XZZXI of the [[5,1,3]] perfect code."""

    paulis = ("XZZXI", "IXZZX", "XIXZZ", "ZXIXZ")
    rows = np.zeros((len(paulis), 10), dtype=np.uint8)
    for row, pauli in zip(rows, paulis, strict=True):
        for q, letter in enumerate(pauli):
            row[q] = letter in "XY"
            row[q + 5] = letter in "YZ"
    return rows


def bacon_shor_3_gauges() -> np.ndarray:
    """Nearest-neighbour gauge generators for the [[9,1,4,3]] code."""

    rows: list[np.ndarray] = []
    for r in range(3):
        for c in range(2):
            row = np.zeros(18, dtype=np.uint8)
            row[3 * r + c] = 1
            row[3 * r + c + 1] = 1
            rows.append(row)
    for c in range(3):
        for r in range(2):
            row = np.zeros(18, dtype=np.uint8)
            row[9 + 3 * r + c] = 1
            row[9 + 3 * (r + 1) + c] = 1
            rows.append(row)
    return np.asarray(rows, dtype=np.uint8)


def test_operation_distance_quotients_benign_logical_without_losing_its_flip():
    # Auxiliary stabilizer <Z0 Z1> has a benign weight-one logical Z0 and its
    # harmful conjugate X0 X1 at weight two.
    auxiliary_gauge = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
    benign = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)

    dressed = search_gauge(
        auxiliary_gauge,
        trials=20,
        seed=100,
        min_distance=1,
        threads=1,
        batch_size=1,
    )
    operation = search_operation(
        auxiliary_gauge,
        benign,
        trials=20,
        seed=100,
        min_distance=2,
        threads=1,
        batch_size=1,
    )
    witness = np.frombuffer(operation.witness, dtype=np.uint8)

    assert dressed.distance == 1
    assert operation.distance == 2
    assert np.array_equal(witness, np.asarray([1, 1, 0, 0], dtype=np.uint8))


def test_operation_problem_has_expected_quotient_dimensions():
    auxiliary_gauge = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
    benign = np.asarray([[0, 0, 1, 0]], dtype=np.uint8)
    problem = build_operation_problem(auxiliary_gauge, benign)

    assert problem.auxiliary_logical_qubits == 1
    assert problem.benign_rank == 1
    assert problem.operation_dimension == 1
    assert problem.operation_probes.shape == (1, 4)
    assert not np.any(symplectic_commutator(problem.operation_probes, problem.trivial))


def test_operation_problem_rejects_noncentral_benign_product():
    auxiliary_gauge = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
    noncentral = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)

    with pytest.raises(ValueError, match="must centralize"):
        build_operation_problem(auxiliary_gauge, noncentral)


@pytest.mark.parametrize(
    ("gauge", "expected"),
    [
        (five_qubit_stabilizers(), (4, 4, 0, 1)),
        (bacon_shor_3_gauges(), (12, 4, 4, 1)),
    ],
)
def test_small_codes_build_expected_subsystem_parameters(gauge, expected):
    problem = build_subsystem(gauge)
    assert (
        problem.gauge_rank,
        problem.center_rank,
        problem.gauge_qubits,
        problem.logical_qubits,
    ) == expected


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
@pytest.mark.parametrize("gauge", [five_qubit_stabilizers(), bacon_shor_3_gauges()])
def test_small_codes_search_known_distance(method, gauge):
    result = search_gauge(
        gauge,
        trials=100,
        seed=1234,
        min_distance=3,
        threads=1,
        batch_size=1,
        pair_window=8,
        method=method,
    )
    assert result.distance == 3
    assert result.target_hit
    assert symplectic_weight(result.witness) == 3


def test_bare_logical_pairing_matches_gauge_membership_on_small_random_codes():
    rng = np.random.default_rng(20260716)
    for n in range(1, 5):
        width = 2 * n
        for _ in range(8):
            gauge = rng.integers(0, 2, size=(2 * n, width), dtype=np.uint8)
            problem = build_subsystem(gauge)
            for value in range(1 << width):
                error = np.asarray([value >> bit & 1 for bit in range(width)], dtype=np.uint8)
                commutes_center = not np.any(symplectic_commutator(error[None, :], problem.center))
                logical_effect = np.any(symplectic_commutator(error[None, :], problem.bare_logicals))
                in_gauge = row_basis(np.vstack([problem.gauge, error])).shape[0] == problem.gauge_rank
                if commutes_center:
                    assert logical_effect == (not in_gauge)


def test_center_decomposition_identifies_tiny_cancellation_sector():
    center = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    decomposition = decompose_center(center)

    assert decomposition.pure_x.shape == (0, 4)
    assert decomposition.pure_z.shape == (0, 4)
    assert decomposition.cancellation_rank == 1
    assert np.array_equal(decomposition.mixed, center)
    assert np.array_equal(
        decomposition.cancellation_probes,
        np.asarray([[0, 0, 1, 1]], dtype=np.uint8),
    )
