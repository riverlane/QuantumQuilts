"""Small GF(2) linear algebra helpers using Python integer bitmasks.

Coordinate ``i`` is represented by bit ``1 << i``.  Matrices are sequences of
integer row masks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence


def parity(x: int) -> int:
    """Return the parity of the Hamming weight of ``x``."""

    return x.bit_count() & 1


def hamming_weight(x: int) -> int:
    """Return the number of one bits in ``x``."""

    return x.bit_count()


def all_ones(n: int) -> int:
    """Return the length-``n`` all-ones vector."""

    if n < 0:
        raise ValueError("n must be non-negative")
    return (1 << n) - 1


def mask_vector(bits: Iterable[int]) -> int:
    """Build a bitmask vector from coordinate indices."""

    out = 0
    for bit in bits:
        if bit < 0:
            raise ValueError("bit indices must be non-negative")
        out |= 1 << bit
    return out


def row_reduce(rows: Iterable[int], n: int | None = None) -> tuple[list[int], list[int]]:
    """Return a reduced row basis and its pivot columns over GF(2).

    Pivots are reported from highest coordinate to lowest.  If ``n`` is given,
    input rows are truncated to the first ``n`` coordinates.
    """

    if n is not None and n < 0:
        raise ValueError("n must be non-negative")

    mask = None if n is None else all_ones(n)
    pivots_to_rows: dict[int, int] = {}

    for row in rows:
        if row < 0:
            raise ValueError("rows must be non-negative bitmasks")
        x = row if mask is None else row & mask
        while x:
            pivot = x.bit_length() - 1
            basis_row = pivots_to_rows.get(pivot)
            if basis_row is None:
                pivots_to_rows[pivot] = x
                break
            x ^= basis_row

    for pivot in sorted(pivots_to_rows):
        row = pivots_to_rows[pivot]
        for other_pivot in list(pivots_to_rows):
            if other_pivot != pivot and ((pivots_to_rows[other_pivot] >> pivot) & 1):
                pivots_to_rows[other_pivot] ^= row

    pivots = sorted(pivots_to_rows, reverse=True)
    return [pivots_to_rows[pivot] for pivot in pivots], pivots


def rank(rows: Iterable[int], n: int | None = None) -> int:
    """Return the GF(2) row rank."""

    basis, _ = row_reduce(rows, n)
    return len(basis)


def reduce_with_basis(basis: Sequence[int], vector: int, n: int | None = None) -> int:
    """Reduce a vector against a basis returned by :func:`row_reduce`."""

    if vector < 0:
        raise ValueError("vector must be a non-negative bitmask")
    x = vector if n is None else vector & all_ones(n)
    for row in basis:
        x = min(x, x ^ int(row))
    return x


def rowspace_contains(rows: Iterable[int], vector: int, n: int | None = None) -> bool:
    """Return whether ``vector`` is in the GF(2) span of ``rows``."""

    if vector < 0:
        raise ValueError("vector must be a non-negative bitmask")
    basis, _ = row_reduce(rows, n)
    return reduce_with_basis(basis, vector, n) == 0


def nullspace(rows: Iterable[int], n: int) -> list[int]:
    """Return a basis for ``{v : rows * v = 0}`` over GF(2)."""

    basis, pivots = row_reduce(rows, n)
    pivot_set = set(pivots)
    free_columns = [col for col in range(n) if col not in pivot_set]
    null_basis: list[int] = []

    for free_col in free_columns:
        vector = 1 << free_col
        for row, pivot in zip(basis, pivots, strict=True):
            if (row >> free_col) & 1:
                vector |= 1 << pivot
        null_basis.append(vector)

    return null_basis


def dot(row: int, vector: int) -> int:
    """Return the GF(2) dot product of two bitmasks."""

    return parity(row & vector)


def mat_vec(rows: Sequence[int], vector: int) -> int:
    """Return the syndrome ``rows * vector`` as a bitmask of row outcomes."""

    if vector < 0:
        raise ValueError("vector must be a non-negative bitmask")
    out = 0
    for row_index, row in enumerate(rows):
        if row < 0:
            raise ValueError("rows must be non-negative bitmasks")
        out |= dot(row, vector) << row_index
    return out


def iter_span(rows: Sequence[int]) -> Iterable[int]:
    """Yield every vector in the span of ``rows`` exactly once."""

    basis, _ = row_reduce(rows)
    span = [0]
    for row in basis:
        span += [vector ^ row for vector in span]
    return iter(span)


def min_weight_in_span(
    rows: Sequence[int],
    n: int,
    predicate: Callable[[int], bool] | None = None,
    *,
    include_zero: bool = False,
) -> tuple[int, int] | None:
    """Return ``(weight, vector)`` for the lightest span vector satisfying a predicate.

    This is intentionally exhaustive and suitable only for small spans.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    predicate = (lambda _vector: True) if predicate is None else predicate
    best: tuple[int, int] | None = None

    for vector in iter_span(rows):
        vector &= all_ones(n)
        if vector == 0 and not include_zero:
            continue
        if not predicate(vector):
            continue
        candidate = (hamming_weight(vector), vector)
        if best is None or candidate < best:
            best = candidate

    return best


def min_distance(
    rows: Sequence[int],
    n: int,
    predicate: Callable[[int], bool] | None = None,
) -> int | None:
    """Return the minimum nonzero weight in a small GF(2) span.

    ``predicate`` can restrict which span vectors count.  ``None`` is returned
    if no nonzero span vector satisfies the predicate.
    """

    result = min_weight_in_span(rows, n, predicate)
    return None if result is None else result[0]
