"""Build a quilt to disk: circuit, DEM, routing and summary.

The outermost entry point. `build_quilt_files(payload, output)` is what the
rest of the pipeline calls; everything below it is the construction itself.
Emits `routing.json` under the `modgari/blockwise-routing/v1` schema, which
is read only by the GARI DEM transform and plays no part in the distance
argument.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import (
    BB_SEC_SCHEDULE_LEGACY,
    MemorySpec,
    QuiltInput,
    build_control,
    build_memory,
    build_quilt,
    load_memory_spec,
    load_quilt,
)


def build_quilt_files(
    quilt: QuiltInput | str | Path,
    output: str | Path,
    *,
    physical_p: float | None = None,
    stitch_rounds: int | None = None,
    section_variant: int | None = None,
    preserve_raw_checks: bool | None = None,
) -> dict[str, Any]:
    """Build a validated quilt circuit and flattened DEM into ``output``."""

    if not isinstance(quilt, QuiltInput):
        quilt = load_quilt(quilt)
    generated = build_quilt(
        quilt,
        p=physical_p,
        rounds=stitch_rounds,
        section_variant=section_variant,
        preserve_raw_checks=preserve_raw_checks,
    )
    physical_p = (
        quilt.build.physical_error_rate if physical_p is None else float(physical_p)
    )
    stitch_rounds = (
        quilt.build.surgery_rounds if stitch_rounds is None else int(stitch_rounds)
    )
    section_variant = (
        quilt.build.canonical_section_variant
        if section_variant is None
        else int(section_variant)
    )
    preserve_raw_checks = (
        quilt.build.preserve_raw_checks
        if preserve_raw_checks is None
        else bool(preserve_raw_checks)
    )
    circuit_bytes = generated.text.encode("utf-8")
    dem_bytes = generated.dem_text.encode("utf-8")
    circuit = generated.circuit
    dem = generated.detector_error_model()
    builder_summary = generated.summary

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "circuit.stim").write_bytes(circuit_bytes)
    (output / "dem.stim").write_bytes(dem_bytes)
    routing = {
        "schema": "modgari/blockwise-routing/v1",
        "case_id": quilt.case_id,
        "detector_block_types": builder_summary["detector_block_types"],
        "source": "raw-sparse-centre-compiler",
    }
    routing_text = json.dumps(routing, indent=2, sort_keys=True) + "\n"
    (output / "routing.json").write_text(routing_text, encoding="utf-8")

    summary = {
        "schema": "quilt-construction/case-build/v1",
        "case_id": quilt.case_id,
        "case": quilt.name,
        "validation": quilt.validation_summary(),
        "protocol": builder_summary["protocol"],
        "sec_schedule_policy": builder_summary["sec_schedule_policy"],
        "physical_round_kinds": builder_summary["physical_round_kinds"],
        "physical_p": physical_p,
        "stitch_rounds": stitch_rounds,
        "section_variant": section_variant,
        "preserve_raw_checks": preserve_raw_checks,
        "circuit_sha256": generated.circuit_sha256,
        "dem_sha256": generated.dem_sha256,
        "routing_sha256": hashlib.sha256(routing_text.encode("utf-8")).hexdigest(),
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        # Which observable is which. The decoding stage must separate the
        # spacelike memory directions from the measurement-parity readout to
        # attribute a failure, and the labels cannot be recovered downstream:
        # Stim circuits carry no observable names.
        "observable_basis": builder_summary["observable_basis"],
        "dem_errors": dem.num_errors,
        "preferred_blockwise_center_history_candidate_count": builder_summary[
            "preferred_blockwise_center_history_candidate_count"
        ],
        "preferred_raw_check_relation_candidate_count": builder_summary[
            "preferred_raw_check_relation_candidate_count"
        ],
        "selected_blockwise_center_history_detector_count": builder_summary[
            "selected_blockwise_center_history_detector_count"
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_control_files(
    quilt: QuiltInput | str | Path,
    output: str | Path,
    *,
    case_id: str | None = None,
    physical_p: float | None = None,
    rounds: int | None = None,
) -> dict[str, Any]:
    """Build the matched noninteracting memory control for a quilt."""

    if not isinstance(quilt, QuiltInput):
        quilt = load_quilt(quilt)
    generated = build_control(
        quilt, case_id=case_id, p=physical_p, rounds=rounds
    )
    circuit_bytes = generated.text.encode("utf-8")
    dem_bytes = generated.dem_text.encode("utf-8")
    circuit = generated.circuit
    dem = generated.detector_error_model()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "circuit.stim").write_bytes(circuit_bytes)
    (output / "dem.stim").write_bytes(dem_bytes)
    routing = {
        "schema": "modgari/blockwise-routing/v1",
        "case_id": generated.case_id,
        "detector_block_types": generated.summary["detector_block_types"],
        "source": "quilt-construction-matched-control",
    }
    routing_text = json.dumps(routing, indent=2, sort_keys=True) + "\n"
    (output / "routing.json").write_text(routing_text, encoding="utf-8")
    summary = {
        "schema": "quilt-construction/control-build/v1",
        "case_id": generated.case_id,
        "control_for": quilt.case_id,
        "physical_p": generated.summary["noise"]["p_sec_1q"],
        "rounds": generated.summary["rounds"],
        "blocks": generated.summary["blocks"],
        "circuit_sha256": generated.circuit_sha256,
        "dem_sha256": generated.dem_sha256,
        "routing_sha256": hashlib.sha256(routing_text.encode("utf-8")).hexdigest(),
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "dem_errors": dem.num_errors,
        "measurement_rank": generated.summary["measurement"]["rank"],
        "physical_round_kinds": generated.summary["physical_round_kinds"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_memory_files(
    block: str,
    output: str | Path,
    *,
    case_id: str | None = None,
    physical_p: float = 0.003,
    rounds: int = 14,
    distance: int = 12,
    sec_schedule: str = BB_SEC_SCHEDULE_LEGACY,
) -> dict[str, Any]:
    """Build one ordinary single-block idle-memory circuit and its artifacts."""

    generated = build_memory(
        block,
        case_id=case_id,
        p=physical_p,
        rounds=rounds,
        distance=distance,
        sec_schedule=sec_schedule,
    )
    circuit_bytes = generated.text.encode("utf-8")
    dem_bytes = generated.dem_text.encode("utf-8")
    circuit = generated.circuit
    dem = generated.detector_error_model()
    builder_summary = generated.summary

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "circuit.stim").write_bytes(circuit_bytes)
    (output / "dem.stim").write_bytes(dem_bytes)
    routing = {
        "schema": "modgari/blockwise-routing/v1",
        "case_id": generated.case_id,
        "detector_block_types": builder_summary["detector_block_types"],
        "source": "quilt-construction-single-block-memory",
    }
    routing_text = json.dumps(routing, indent=2, sort_keys=True) + "\n"
    (output / "routing.json").write_text(routing_text, encoding="utf-8")

    memory_block = dict(builder_summary["memory_block"])
    summary = {
        "schema": "quilt-construction/memory-build/v1",
        "case_id": generated.case_id,
        "block": memory_block,
        "physical_p": float(physical_p),
        "rounds": int(rounds),
        "circuit_sha256": generated.circuit_sha256,
        "dem_sha256": generated.dem_sha256,
        "routing_sha256": hashlib.sha256(routing_text.encode("utf-8")).hexdigest(),
        "total_qubits": builder_summary["num_total_qubits"],
        "reference_qubits": builder_summary["num_reference_qubits"],
        "measurements": circuit.num_measurements,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "dem_errors": dem.num_errors,
        "measurement_rank": builder_summary["measurement"]["rank"],
        "observable_basis": builder_summary["observable_basis"],
        "physical_round_kinds": builder_summary["physical_round_kinds"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_memory_spec_files(
    memory: MemorySpec | str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build artifacts from a public memory spec and bind its exact input."""

    source_bytes: bytes | None = None
    if isinstance(memory, MemorySpec):
        spec = memory
    else:
        source_path = Path(memory)
        source_bytes = source_path.read_bytes()
        spec = load_memory_spec(source_path)
    payload = spec.to_payload()
    canonical_bytes = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    summary = build_memory_files(
        spec.adapter_kind,
        output,
        case_id=spec.case_id,
        physical_p=spec.physical_error_rate,
        rounds=spec.rounds,
        distance=spec.distance,
        sec_schedule=spec.effective_sec_schedule,
    )
    summary["memory_spec"] = payload
    summary["memory_spec_sha256"] = hashlib.sha256(
        source_bytes if source_bytes is not None else canonical_bytes
    ).hexdigest()
    summary["memory_spec_canonical_sha256"] = hashlib.sha256(
        canonical_bytes
    ).hexdigest()
    output = Path(output)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "build_control_files",
    "build_memory_files",
    "build_memory_spec_files",
    "build_quilt_files",
]
