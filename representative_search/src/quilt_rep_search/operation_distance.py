from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from quilt_construction.generation import QuiltInput, load_quilt
from quilt_construction.generation._gf2 import rank, row_reduce
from quilt_construction.generation._symplectic import gauge_center
from quilt_construction.hashes import sha256_file, stable_json_sha256


def operation_spec_from_quilt(
    quilt: QuiltInput | str | Path,
    *,
    trials: int = 10_000,
    seed: int = 0,
    threads: int = 0,
    method: str = "three-block",
    target_distance: int | None = None,
    declared_lower_bound: int | None = None,
    lower_bound_source: str | None = None,
) -> dict[str, Any]:
    """Encode the manuscript's C-perp outside G problem for the legacy solver.

    The open-stitch encoding is retained for file compatibility, but is usable
    only after proving exactly that its centre equals C = G0 intersect G-perp.
    Thus neither a choice of omitted gauge nor an accidental extra detector
    can silently change the distance problem.
    """

    if not isinstance(quilt, QuiltInput):
        quilt = load_quilt(quilt)
    width = 2 * quilt.spec.total_n
    canonical = quilt.ledger.canonical_stitch_gauges
    measured = quilt.ledger.measured_products
    if len(quilt.gauge_groups) != len(measured):
        raise ValueError("one ordered gauge group is required per measured product")

    auxiliary_rows = list(quilt.ledger.base_stabilizer_basis)
    benign_rows: list[int] = []
    for group_index, (group, product) in enumerate(
        zip(quilt.gauge_groups, measured, strict=True)
    ):
        if not group:
            raise ValueError(f"gauge group {group_index} is empty")
        gauges = [canonical[index] for index in group]
        closed_product = 0
        for gauge in gauges:
            closed_product ^= gauge
        if closed_product != product.row:
            raise ValueError(
                f"gauge group {group_index} does not close to its measured product"
            )
        auxiliary_rows.extend(gauges[:-1])
        benign_rows.append(closed_product)

    auxiliary = tuple(row_reduce(auxiliary_rows, width)[0])
    benign = tuple(row_reduce(benign_rows, width)[0])
    merged = tuple(row_reduce(quilt.ledger.merged.gauge_basis, width)[0])
    if tuple(row_reduce([*auxiliary, *benign], width)[0]) != merged:
        raise ValueError("open gauges plus benign products do not generate the quilt gauge group")
    inherited_center = tuple(row_reduce(quilt.ledger.reliable_center_basis, width)[0])
    auxiliary_center = tuple(row_reduce(gauge_center(auxiliary, quilt.spec.total_n), width)[0])
    if auxiliary_center != inherited_center:
        raise ValueError(
            "legacy open-stitch centre differs from the manuscript's inherited "
            "centre C=G0 intersect Gperp; use a direct Cperp outside G search"
        )
    search: dict[str, Any] = {
        "trials": int(trials),
        "seed": int(seed),
        "threads": int(threads),
        "method": str(method),
    }
    if target_distance is not None:
        search["min_distance"] = int(target_distance)
    return {
        "schema": "quilt-construction/operation-distance-input/v1",
        "case_id": quilt.case_id,
        "n": quilt.spec.total_n,
        "auxiliary_gauge_hex": [hex(row) for row in auxiliary],
        "benign_hex": [hex(row) for row in benign],
        "inherited_center_hex": [hex(row) for row in inherited_center],
        "merged_gauge_hex": [hex(row) for row in merged],
        "construction": {
            "method": "ordered open-stitch subgroup with exact manuscript-quotient equivalence",
            "operational_problem": "Cperp outside G; C=G0 intersect Gperp",
            "manuscript_quotient_verified": True,
            "inherited_center_rank": len(inherited_center),
            "merged_gauge_rank": len(quilt.ledger.merged.gauge_basis),
            "benign_rank": rank(benign, width),
            "auxiliary_rank": rank(auxiliary, width),
            "gauge_group_sizes": [len(group) for group in quilt.gauge_groups],
        },
        "search": search,
        "declared_lower_bound": declared_lower_bound,
        "lower_bound_source": lower_bound_source,
    }


def write_operation_spec(
    quilt: QuiltInput | str | Path,
    output: str | Path,
    **search: Any,
) -> dict[str, Any]:
    payload = operation_spec_from_quilt(quilt, **search)
    Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _matrix_from_hex(rows: Sequence[str | int], n: int) -> np.ndarray:
    width = 2 * int(n)
    result = np.zeros((len(rows), width), dtype=np.uint8)
    for row_index, encoded in enumerate(rows):
        value = int(encoded, 0) if isinstance(encoded, str) else int(encoded)
        if value < 0 or value >= (1 << width):
            raise ValueError(f"row {row_index} exceeds symplectic width {width}")
        for column in range(width):
            result[row_index, column] = (value >> column) & 1
    return result


def _result_payload(result: Any, n: int) -> dict[str, Any]:
    witness = getattr(result, "witness", None)
    payload = {
        "distance": getattr(result, "distance", None),
        "trials_requested": int(getattr(result, "trials_requested", 0)),
        "trials_completed": int(getattr(result, "trials_completed", 0)),
        "hits": int(getattr(result, "hits", 0)),
        "elapsed_seconds": float(getattr(result, "elapsed_seconds", 0.0)),
        "target_hit": bool(getattr(result, "target_hit", False)),
    }
    if witness:
        bits = (
            np.frombuffer(witness, dtype=np.uint8)
            if isinstance(witness, bytes)
            else np.asarray(witness)
        )
        bits = bits.reshape(-1).astype(np.uint8) & 1
        value = sum(int(bit) << index for index, bit in enumerate(bits))
        payload["witness_hex"] = hex(value)
        payload["witness_support"] = [
            q for q in range(n) if int(bits[q]) or int(bits[n + q])
        ]
    return payload


def certify_operation_file(
    spec_path: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Run the randomized operation-distance search from JSON."""

    try:
        from qdist_rs import build_operation_problem, search_operation
        from qdist_rs.algebra import symplectic_commutator
    except ImportError as error:
        raise RuntimeError(
            "qdist-rs is not installed; run `pip install ./rust/operational-distance`"
        ) from error

    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    schema = spec.get("schema")
    if schema not in {None, "quilt-construction/operation-distance-input/v1"}:
        raise ValueError(f"unsupported operation-distance input schema {schema!r}")
    n = int(spec["n"])
    auxiliary = _matrix_from_hex(spec["auxiliary_gauge_hex"], n)
    benign = _matrix_from_hex(spec["benign_hex"], n)
    problem = build_operation_problem(auxiliary, benign)
    search = dict(spec.get("search", {}))
    result = search_operation(auxiliary, benign, **search)
    result_payload = _result_payload(result, n)
    valid = None
    if "witness_hex" in result_payload:
        witness = _matrix_from_hex([result_payload["witness_hex"]], n)
        center_syndrome = symplectic_commutator(witness, problem.center)
        operation_effect = symplectic_commutator(witness, problem.operation_probes)
        valid = bool(not np.any(center_syndrome) and np.any(operation_effect))
        result_payload["witness_validation"] = {
            "valid": valid,
            "center_syndrome_weight": int(np.count_nonzero(center_syndrome)),
            "operation_effect_weight": int(np.count_nonzero(operation_effect)),
        }
    payload = {
        "schema": "quilt-construction/operation-distance-certificate/v1",
        "input": str(spec_path),
        "input_sha256": sha256_file(spec_path),
        "problem": {
            "n": n,
            "auxiliary_gauge_rank": int(problem.auxiliary_gauge.shape[0]),
            "center_rank": int(problem.center.shape[0]),
            "benign_rank": int(problem.benign_rank),
            "operation_dimension": int(problem.operation_dimension),
        },
        "search": result_payload,
        "claim_scope": "randomized upper bound unless a separate lower-bound theorem is cited",
        "declared_lower_bound": spec.get("declared_lower_bound"),
        "lower_bound_source": spec.get("lower_bound_source"),
    }
    payload["certificate_content_sha256"] = stable_json_sha256(payload)
    if valid is False:
        raise RuntimeError("distance finder returned an invalid witness")
    if output is not None:
        Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


__all__ = [
    "certify_operation_file",
    "operation_spec_from_quilt",
    "write_operation_spec",
]
