import numpy as np
import pytest

from qdist_rs import search_css, search_subsystem, search_symplectic
from qdist_rs.algebra import row_basis


SYMPLECTIC_METHODS = (
    "grouped",
    "local-clifford",
    "location-linear",
    "three-block",
    "three-block-global",
)


def steane_checks() -> np.ndarray:
    """A self-orthogonal parity check for the [[7,1,3]] Steane code."""

    return np.asarray(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )


def validate_css_witness(checks: np.ndarray, gauges: np.ndarray, witness: bytes) -> np.ndarray:
    vector = np.frombuffer(witness, dtype=np.uint8)
    assert vector.shape == (checks.shape[1],)
    assert not np.any((checks @ vector) & 1)
    assert row_basis(np.vstack([gauges, vector])).shape[0] > row_basis(gauges).shape[0]
    return vector


def test_css_search_finds_and_validates_steane_distance():
    checks = steane_checks()
    result = search_css(
        checks,
        checks,
        trials=20,
        seed=123,
        min_distance=3,
        threads=1,
        batch_size=1,
    )

    witness = validate_css_witness(checks, checks, result.witness)
    assert result.distance == 3
    assert result.target_hit
    assert int(witness.sum()) == 3


def test_css_search_is_reproducible_across_thread_counts_for_fixed_budget():
    checks = steane_checks()
    one = search_css(checks, checks, trials=100, seed=991, threads=1)
    four = search_css(checks, checks, trials=100, seed=991, threads=4, batch_size=17)

    assert one.distance == four.distance
    assert one.witness == four.witness
    assert one.hits == four.hits


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_methods_are_reproducible_across_thread_counts(method):
    detectors = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    observables = np.asarray(
        [[0, 0, 1, 1], [1, 1, 0, 0]],
        dtype=np.uint8,
    )
    one = search_symplectic(
        detectors,
        observables,
        trials=100,
        seed=992,
        threads=1,
        method=method,
    )
    four = search_symplectic(
        detectors,
        observables,
        trials=100,
        seed=992,
        threads=4,
        batch_size=17,
        method=method,
    )

    assert one.distance == four.distance
    assert one.witness == four.witness
    assert one.hits == four.hits


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_symplectic_search_finds_noncss_cancellation_witness(method):
    # Center <Y0 Y1>. Its pure-X and pure-Z distances are two, but Y0 is a
    # weight-one logical in a nonzero cancellation sector.
    detectors = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    observables = np.asarray(
        [
            [0, 0, 1, 1],  # symplectic dual of X0 X1
            [1, 1, 0, 0],  # symplectic dual of Z0 Z1
        ],
        dtype=np.uint8,
    )

    result = search_symplectic(
        detectors,
        observables,
        trials=20,
        seed=456,
        min_distance=1,
        threads=1,
        batch_size=1,
        method=method,
    )
    witness = np.frombuffer(result.witness, dtype=np.uint8)

    assert result.distance == 1
    assert result.target_hit
    assert np.all((detectors @ witness) % 2 == 0)
    assert np.any((observables @ witness) % 2 == 1)
    assert sum(bool(witness[q] or witness[q + 2]) for q in range(2)) == 1


@pytest.mark.parametrize("method", SYMPLECTIC_METHODS)
def test_constrained_subsystem_search_requires_selected_effect(method):
    center = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    bare_logicals = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    required = np.asarray([[1, 1, 0, 0]], dtype=np.uint8)

    result = search_subsystem(
        center,
        bare_logicals,
        required_effects=required,
        trials=20,
        seed=457,
        min_distance=1,
        threads=1,
        batch_size=1,
        method=method,
    )
    witness = np.frombuffer(result.witness, dtype=np.uint8)

    assert result.distance == 1
    assert np.any((required[:, [2, 3, 0, 1]] @ witness) & 1)


def test_symplectic_search_rejects_unknown_method():
    matrix = np.asarray([[1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown symplectic search method"):
        search_symplectic(matrix, matrix, trials=1, method="not-a-method")


@pytest.mark.parametrize(
    "method", ["grouped", "local-clifford", "location-linear", "three-block"]
)
def test_location_group_counts_a_two_qubit_pauli_as_one_fault(method):
    # The detector kernel is exactly <X0 X1>.
    detectors = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    observables = np.asarray([[1, 0, 0, 0]], dtype=np.uint8)

    ordinary = search_symplectic(
        detectors, observables, trials=1, threads=1, method=method
    )
    grouped = search_symplectic(
        detectors,
        observables,
        trials=1,
        threads=1,
        method=method,
        location_groups=[[0, 1]],
    )

    assert ordinary.distance == 2
    assert grouped.distance == 1
    assert grouped.witness == ordinary.witness


def test_explicit_singleton_location_groups_preserve_distance():
    detectors = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    observables = np.asarray(
        [[0, 0, 1, 1], [1, 1, 0, 0]],
        dtype=np.uint8,
    )
    ordinary = search_symplectic(detectors, observables, trials=100, seed=993, threads=1)
    explicit = search_symplectic(
        detectors,
        observables,
        trials=100,
        seed=993,
        threads=1,
        location_groups=[[0], [1]],
    )
    assert explicit.distance == ordinary.distance


@pytest.mark.parametrize(
    ("method", "expected_hits"),
    [
        ("grouped", 173),
        ("local-clifford", 144),
        ("location-linear", 147),
        ("three-block", 157),
        ("three-block-global", 157),
    ],
)
def test_grouped_fixed_seed_preserves_witness_ties_and_hits(method, expected_hits):
    rng = np.random.default_rng(20260911)
    rng.integers(0, 2, size=(7, 24), dtype=np.uint8)
    rng.integers(0, 2, size=(4, 24), dtype=np.uint8)
    detectors = rng.integers(0, 2, size=(8, 24), dtype=np.uint8)
    observables = rng.integers(0, 2, size=(4, 24), dtype=np.uint8)
    groups = [[0, 1], [2], [3, 4], [5], [6, 7], [8], [9, 10], [11]]
    expected_witness = bytes(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    )

    result = search_symplectic(
        detectors,
        observables,
        trials=37,
        seed=0x5EEE,
        threads=1,
        batch_size=7,
        pair_window=8,
        method=method,
        location_groups=groups,
    )
    threaded = search_symplectic(
        detectors,
        observables,
        trials=37,
        seed=0x5EEE,
        threads=4,
        batch_size=7,
        pair_window=8,
        method=method,
        location_groups=groups,
    )

    assert result.distance == threaded.distance == 2
    assert result.witness == threaded.witness == expected_witness
    assert result.hits == threaded.hits == expected_hits
    assert result.trials_completed == threaded.trials_completed == 37


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([[0, 1], [1]], "more than one location group"),
        ([[0, 0]], "distinct"),
        ([[0, 2]], "outside"),
        ([[0, 1, 2]], "one or two"),
    ],
)
def test_location_groups_are_validated(groups, message):
    matrix = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        search_symplectic(matrix, matrix, trials=1, location_groups=groups)


def test_order_weight_requires_groups_and_must_be_finite_nonnegative():
    matrix = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="requires location_groups"):
        search_symplectic(
            matrix,
            matrix,
            trials=1,
            multi_qubit_group_order_weight=2.0,
        )
    for invalid in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            search_symplectic(
                matrix,
                matrix,
                trials=1,
                location_groups=[[0, 1]],
                multi_qubit_group_order_weight=invalid,
            )


def test_subsystem_wrapper_accepts_center_and_bare_logicals_in_xz_form():
    center = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)  # Y0 Y1
    bare_logicals = np.asarray(
        [
            [1, 1, 0, 0],  # X0 X1
            [0, 0, 1, 1],  # Z0 Z1
        ],
        dtype=np.uint8,
    )

    result = search_subsystem(
        center,
        bare_logicals,
        trials=20,
        seed=789,
        min_distance=1,
        threads=1,
        batch_size=1,
    )

    assert result.distance == 1
    assert result.target_hit
