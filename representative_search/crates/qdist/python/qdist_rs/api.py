from __future__ import annotations

from typing import Any

import numpy as np

from ._core import SearchResult, search_symplectic, search_symplectic_constrained
from .algebra import binary_matrix as _binary_matrix
from .algebra import (
    build_operation_problem,
    build_subsystem,
    decompose_center,
    symplectic_dual,
)


def search_subsystem(
    center: Any,
    bare_logicals: Any,
    *,
    required_effects: Any | None = None,
    **kwargs: Any,
) -> SearchResult:
    """Search the dressed subsystem quotient ``S^perp \\ G``.

    Args:
        center: Binary symplectic rows spanning ``S = Z(G)`` in ``X | Z`` form.
        bare_logicals: Symplectic rows spanning ``G^perp / S``. Their pairings
            detect whether an undetected error lies outside the gauge group.
        required_effects: Optional symplectic Pauli rows. If supplied, the
            witness must anticommute with at least one of these rows as well as
            with at least one bare logical. This selects a diagnostic sector
            without changing the subsystem quotient.
        **kwargs: Search controls accepted by :func:`search_symplectic`.
    """

    center_array = _binary_matrix(center, name="center")
    logical_array = _binary_matrix(bare_logicals, name="bare_logicals")
    if center_array.shape[1] != logical_array.shape[1]:
        raise ValueError("center and bare_logicals must have equal widths")
    detectors = symplectic_dual(center_array)
    observables = symplectic_dual(logical_array)
    if required_effects is None:
        return search_symplectic(detectors, observables, **kwargs)
    required_array = _binary_matrix(required_effects, name="required_effects")
    if center_array.shape[1] != required_array.shape[1]:
        raise ValueError("center and required_effects must have equal widths")
    return search_symplectic_constrained(
        detectors,
        observables,
        symplectic_dual(required_array),
        **kwargs,
    )


def search_gauge(gauge: Any, **kwargs: Any) -> SearchResult:
    """Build and search a subsystem code directly from gauge generators."""

    problem = build_subsystem(gauge)
    return search_subsystem(problem.center, problem.bare_logicals, **kwargs)


def search_operation(
    auxiliary_gauge: Any,
    benign: Any,
    **kwargs: Any,
) -> SearchResult:
    """Search the operation quotient ``S_aux^perp / (G_aux + C)``.

    ``auxiliary_gauge`` spans ``G_aux`` and ``benign`` spans the deliberately
    measured logical products ``C``. The search keeps ``S_aux = Z(G_aux)`` as
    its detector space, so it includes both surviving memory errors and errors
    that flip a measured product, while excluding the measured products
    themselves.
    """

    problem = build_operation_problem(auxiliary_gauge, benign)
    if problem.operation_dimension == 0:
        raise ValueError("the operation quotient has no harmful logical sectors")
    return search_subsystem(problem.center, problem.operation_probes, **kwargs)


def search_cancellation_subsystem(
    center: Any,
    bare_logicals: Any,
    **kwargs: Any,
) -> SearchResult:
    """Search only nonzero X/Z cancellation sectors of a mixed center."""

    center_array = _binary_matrix(center, name="center")
    decomposition = decompose_center(center_array)
    if decomposition.cancellation_rank == 0:
        raise ValueError("the center has no mixed cancellation sector")
    return search_subsystem(
        center_array,
        bare_logicals,
        required_effects=decomposition.cancellation_probes,
        **kwargs,
    )
