"""The correctness gate: does an interface measure only what we intend?

A quilt puts one weight-two stitching operator on every qubit of the chosen
representative.  Measuring those operators reveals every closed operator that
fits inside the representative's support -- not just the one we meant to
measure.  If anything else fits, the measurement is destroying logical
information we did not agree to destroy.

So, writing O = supp(r) and taking K to be the closed sector-s operators
supported inside O, we require

    K = <1_O>,

the one-dimensional space spanned by the representative itself.  Two distinct
things are ruled out.  If K contains a second *logical* class the interface
destroys information we never agreed to measure.  If K contains a *stabiliser*
the representative is reducible -- multiplying by that stabiliser gives a
lighter operator of the same class -- and the local kernel is no longer
span(all-ones).

An earlier version of this gate tested only the image of K in label space.  That
is strictly weaker: stabilisers carry label zero, so a representative containing
a whole stabiliser slipped through.  Both conditions are checked here.

This is cheap -- a nullspace over |O| columns -- and it is never skipped.  A
small support does not make it unnecessary: two weight-D logicals overlapping
in one qubit sit inside |O| = 2D-1, and among the bb72 seeds 171
representatives fail the test.  Here it is a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .codes_and_symmetries import Code, parity


def _restricted_nullspace(rows: Sequence[int], sites: Sequence[int]) -> list[int]:
    """Basis of the vectors on `sites` orthogonal to every row of `rows`."""

    width = len(sites)
    reduced: dict[int, int] = {}
    for row in rows:
        packed = 0
        for column, site in enumerate(sites):
            if (row >> site) & 1:
                packed |= 1 << column
        current = packed
        while current:
            pivot = current.bit_length() - 1
            if pivot in reduced:
                current ^= reduced[pivot]
            else:
                reduced[pivot] = current
                break
    free = [column for column in range(width) if column not in reduced]
    basis = []
    for column in free:
        vector = 1 << column
        for pivot in sorted(reduced):
            if parity(reduced[pivot] & vector):
                vector ^= 1 << pivot
        basis.append(vector)
    return basis


def _span(values: Sequence[int]) -> dict[int, int]:
    reduced: dict[int, int] = {}
    for value in values:
        current = value
        while current:
            pivot = current.bit_length() - 1
            if pivot in reduced:
                current ^= reduced[pivot]
            else:
                reduced[pivot] = current
                break
    return reduced


def _contains(reduced: dict[int, int], target: int) -> bool:
    current = target
    while current:
        pivot = current.bit_length() - 1
        if pivot not in reduced:
            return False
        current ^= reduced[pivot]
    return True


@dataclass(frozen=True)
class Cleanliness:
    clean: bool
    kernel_dimension: int          # must be 1
    measured_rank: int             # rank of the label image; must be 1
    contains_target: bool

    def __bool__(self) -> bool:
        return self.clean


def inspect(mask: int, *, code: Code, sector: str) -> Cleanliness:
    """Does the interface on supp(mask) measure exactly mask's logical class?"""

    sites = [q for q in range(code.n) if (mask >> q) & 1]
    if not sites:
        return Cleanliness(clean=False, kernel_dimension=0, measured_rank=0, contains_target=False)

    kernel = _restricted_nullspace(code.detecting_checks(sector), sites)
    all_ones = (1 << len(sites)) - 1
    kernel_is_span_of_mask = len(kernel) == 1 and kernel[0] == all_ones

    labels = []
    for vector in kernel:
        local = 0
        for column, site in enumerate(sites):
            if (vector >> column) & 1:
                local |= 1 << site
        label = code.label(local, sector)
        if label:                      # stabilisers carry label zero; ignore them
            labels.append(label)

    reduced = _span(labels)
    rank = len(reduced)
    target = code.label(mask, sector)
    contains = _contains(reduced, target) if rank else False
    return Cleanliness(
        clean=kernel_is_span_of_mask and rank == 1 and contains,
        kernel_dimension=len(kernel),
        measured_rank=rank,
        contains_target=contains,
    )
