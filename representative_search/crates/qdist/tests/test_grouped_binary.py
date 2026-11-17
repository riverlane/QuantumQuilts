import itertools

import numpy as np
import pytest

from qdist_rs import search_grouped_binary, search_symplectic


def normalized_groups(ncols: int, groups: list[list[int]]) -> list[list[int]]:
    present = {column for group in groups for column in group}
    return [*groups, *([column] for column in range(ncols) if column not in present)]


def grouped_weight(vector: np.ndarray, groups: list[list[int]]) -> int:
    return sum(bool(np.any(vector[group])) for group in groups)


def brute_force_distance(
    detectors: np.ndarray,
    observables: np.ndarray,
    groups: list[list[int]],
) -> int | None:
    best = None
    for bits in itertools.product((0, 1), repeat=detectors.shape[1]):
        vector = np.asarray(bits, dtype=np.uint8)
        if np.any((detectors @ vector) & 1):
            continue
        if not np.any((observables @ vector) & 1):
            continue
        weight = grouped_weight(vector, groups)
        best = weight if best is None else min(best, weight)
    return best


@pytest.mark.parametrize(
    ("detectors", "observables", "explicit_groups"),
    [
        (
            [[1, 0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1],
             [1, 1, 0, 0, 0, 1, 0]],
            [[1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0]],
            [[0, 2, 4], [1, 5]],
        ),
        (
            [[1, 0, 0, 1, 0, 0, 0, 1, 0],
             [0, 1, 0, 1, 0, 0, 0, 0, 1],
             [0, 0, 1, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 1, 0, 0],
             [1, 1, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 1, 0, 1, 0],
             [0, 0, 0, 1, 1, 0, 0, 0, 1]],
            [[1, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 3, 6, 8], [1, 5]],
        ),
    ],
)
def test_grouped_binary_matches_brute_force_on_odd_widths(
    detectors, observables, explicit_groups
):
    detectors = np.asarray(detectors, dtype=np.uint8)
    observables = np.asarray(observables, dtype=np.uint8)
    groups = normalized_groups(detectors.shape[1], explicit_groups)
    expected = brute_force_distance(detectors, observables, groups)

    result = search_grouped_binary(
        detectors,
        observables,
        trials=64,
        seed=20260911,
        threads=1,
        batch_size=7,
        pair_window=16,
        location_groups=explicit_groups,
    )
    witness = np.frombuffer(result.witness, dtype=np.uint8)

    assert result.distance == expected
    assert witness.shape == (detectors.shape[1],)
    assert not np.any((detectors @ witness) & 1)
    assert np.any((observables @ witness) & 1)
    assert grouped_weight(witness, groups) == result.distance

    randomized_basis = search_grouped_binary(
        detectors,
        observables,
        trials=64,
        seed=20260912,
        threads=1,
        batch_size=7,
        pair_window=16,
        location_groups=explicit_groups,
        randomize_within_location_basis=True,
    )
    randomized_witness = np.frombuffer(randomized_basis.witness, dtype=np.uint8)
    assert randomized_basis.distance == expected
    assert not np.any((detectors @ randomized_witness) & 1)
    assert np.any((observables @ randomized_witness) & 1)
    assert grouped_weight(randomized_witness, groups) == randomized_basis.distance


def test_omitted_binary_columns_are_singleton_locations():
    detectors = np.asarray(
        [[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]],
        dtype=np.uint8,
    )
    observables = np.asarray([[1, 0, 1, 0, 1]], dtype=np.uint8)
    explicit_groups = [[0, 2, 4]]
    groups = normalized_groups(5, explicit_groups)
    result = search_grouped_binary(
        detectors,
        observables,
        trials=32,
        threads=1,
        pair_window=8,
        location_groups=explicit_groups,
    )
    witness = np.frombuffer(result.witness, dtype=np.uint8)

    assert result.distance == brute_force_distance(detectors, observables, groups)
    assert grouped_weight(witness, groups) == result.distance


def test_grouped_binary_min_distance_stops_after_a_complete_batch():
    detectors = np.zeros((0, 3), dtype=np.uint8)
    observables = np.asarray([[1, 0, 0]], dtype=np.uint8)
    result = search_grouped_binary(
        detectors,
        observables,
        trials=100,
        seed=27,
        min_distance=1,
        threads=1,
        batch_size=7,
    )

    assert result.distance == 1
    assert result.target_hit
    assert result.trials_requested == 100
    assert result.trials_completed == 7


def test_grouped_binary_is_reproducible_across_thread_counts():
    rng = np.random.default_rng(918273)
    detectors = rng.integers(0, 2, size=(5, 13), dtype=np.uint8)
    observables = rng.integers(0, 2, size=(3, 13), dtype=np.uint8)
    groups = [[0, 2, 7], [1, 4, 8, 12], [3, 9]]
    kwargs = dict(
        trials=41,
        seed=0xB1A4,
        batch_size=7,
        pair_window=8,
        location_groups=groups,
        multi_qubit_group_order_weight=0.25,
    )

    one = search_grouped_binary(detectors, observables, threads=1, **kwargs)
    two = search_grouped_binary(detectors, observables, threads=2, **kwargs)

    assert one.distance == two.distance
    assert one.witness == two.witness
    assert one.hits == two.hits
    assert one.trials_completed == two.trials_completed == 41


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        ([[0, 1], [1, 2]], "more than one binary location group"),
        ([[0, 0]], "distinct column indices"),
        ([[0, 5]], "outside"),
        ([[]], "one to four"),
        ([[0, 1, 2, 3, 4]], "one to four"),
    ],
)
def test_grouped_binary_validates_location_groups(groups, message):
    matrix = np.asarray([[1, 0, 1, 0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        search_grouped_binary(matrix, matrix, trials=1, location_groups=groups)


def test_grouped_binary_validates_width_and_order_weight():
    detectors = np.asarray([[1, 0, 1]], dtype=np.uint8)
    observables = np.asarray([[1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="equal widths"):
        search_grouped_binary(detectors, observables, trials=1)

    for invalid in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            search_grouped_binary(
                detectors,
                detectors,
                trials=1,
                location_groups=[[0, 1, 2]],
                multi_qubit_group_order_weight=invalid,
            )

    with pytest.raises(ValueError, match="requires location_groups"):
        search_grouped_binary(
            detectors,
            detectors,
            trials=1,
            multi_qubit_group_order_weight=0.25,
        )

    for invalid in (0, -0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            search_grouped_binary(
                detectors,
                detectors,
                trials=1,
                location_groups=[[0, 1, 2]],
                preferred_location_groups=[[0, 1, 2]],
                preferred_group_order_weight=invalid,
            )

    with pytest.raises(ValueError, match="requires preferred_location_groups"):
        search_grouped_binary(
            detectors,
            detectors,
            trials=1,
            location_groups=[[0, 1, 2]],
            preferred_group_order_weight=0.05,
        )

    with pytest.raises(ValueError, match="requires location_groups"):
        search_grouped_binary(
            detectors,
            detectors,
            trials=1,
            randomize_within_location_basis=True,
        )


def test_grouped_binary_randomized_basis_is_reproducible_across_threads():
    rng = np.random.default_rng(772901)
    detectors = rng.integers(0, 2, size=(6, 15), dtype=np.uint8)
    observables = rng.integers(0, 2, size=(3, 15), dtype=np.uint8)
    kwargs = dict(
        trials=43,
        seed=0xB4515,
        batch_size=7,
        pair_window=8,
        location_groups=[[0, 2, 7], [1, 4, 8, 12], [3, 9]],
        multi_qubit_group_order_weight=0.02,
        randomize_within_location_basis=True,
    )

    one = search_grouped_binary(detectors, observables, threads=1, **kwargs)
    two = search_grouped_binary(detectors, observables, threads=2, **kwargs)

    assert one.distance == two.distance
    assert one.witness == two.witness
    assert one.hits == two.hits
    assert one.trials_completed == two.trials_completed == 43

    with pytest.raises(ValueError, match="requires location_groups"):
        search_grouped_binary(
            detectors,
            detectors,
            trials=1,
            preferred_location_groups=[[0]],
            preferred_group_order_weight=0.05,
        )

    with pytest.raises(ValueError, match="not a complete location group"):
        search_grouped_binary(
            detectors,
            detectors,
            trials=1,
            location_groups=[[0, 1, 2]],
            preferred_location_groups=[[0, 1]],
            preferred_group_order_weight=0.05,
        )


def test_grouped_binary_matches_equivalent_symplectic_grouping():
    detectors = np.asarray(
        [[1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0],
         [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]],
        dtype=np.uint8,
    )
    observables = np.asarray(
        [[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0]], dtype=np.uint8
    )
    symplectic_groups = [[0, 1], [2]]
    binary_groups = [[0, 1, 3, 4], [2, 5]]
    kwargs = dict(
        trials=128,
        seed=7761,
        threads=1,
        batch_size=11,
        pair_window=16,
        multi_qubit_group_order_weight=0.25,
    )

    symplectic = search_symplectic(
        detectors,
        observables,
        location_groups=symplectic_groups,
        **kwargs,
    )
    binary = search_grouped_binary(
        detectors,
        observables,
        location_groups=binary_groups,
        **kwargs,
    )

    assert binary.distance == symplectic.distance
    binary_witness = np.frombuffer(binary.witness, dtype=np.uint8)
    assert not np.any((detectors @ binary_witness) & 1)
    assert np.any((observables @ binary_witness) & 1)
    assert grouped_weight(binary_witness, binary_groups) == binary.distance
