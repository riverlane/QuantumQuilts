"""Public surface of the generation package.

`build_quilt` validates the build parameters and hands off to the circuit
compiler; `load_quilt` turns a `quilt-spec/v1` payload into the validated
case object everything else consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import stim

from ._algebra import QuiltSpec, StabilizerBlock, StitchingFamily
from ._circuit import GenericQuiltCircuitCase, build_circuit as _build_circuit
from ._core import (
    BB144_SEC_SCHEDULE_BRAVYI_2024,
    BB_SEC_SCHEDULE_LEGACY,
    NoiseConfig,
)
from ._sec_blocks import BBSecBlock, BB144SecBlock, RotatedSurfaceSecBlock
from ._symplectic import analyze_measurement, build_general_subsystem
from .spec import (
    BuildConfig,
    MemorySpec,
    QuiltInput,
    load_memory_spec,
    load_quilt,
    quilt_from_parts,
)


@dataclass(frozen=True)
class GeneratedCircuit:
    case_id: str
    text: str
    summary: dict[str, object]

    @property
    def circuit(self) -> stim.Circuit:
        return stim.Circuit(self.text)

    @property
    def circuit_sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def detector_error_model(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(
            decompose_errors=False,
            allow_gauge_detectors=False,
            flatten_loops=True,
        )

    @property
    def dem_text(self) -> str:
        return str(self.detector_error_model())

    @property
    def dem_sha256(self) -> str:
        return hashlib.sha256(self.dem_text.encode()).hexdigest()



# Idle-noise rate as a multiple of the physical error rate. 0.0 means qubits
# idling in a tick are noiseless. Set it via QUILT_IDLE_NOISE_FACTOR: 1.0 is
# the pessimistic convention in which an idling qubit depolarises at the
# two-qubit gate rate.
import os as _os

def _idle_rate(p: float) -> float:
    return float(p) * float(_os.environ.get("QUILT_IDLE_NOISE_FACTOR", "0.0"))

def build_quilt(
    quilt: QuiltInput | str | Path,
    *,
    p: float | None = None,
    rounds: int | None = None,
    section_variant: int | None = None,
    preserve_raw_checks: bool | None = None,
) -> GeneratedCircuit:
    """Generate a circuit from explicit, validated quilt inputs."""

    if not isinstance(quilt, QuiltInput):
        quilt = load_quilt(quilt)
    p = quilt.build.physical_error_rate if p is None else float(p)
    rounds = quilt.build.surgery_rounds if rounds is None else int(rounds)
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
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if rounds <= 0:
        raise ValueError("rounds must be positive")
    if section_variant < 0:
        raise ValueError("section_variant must be nonnegative")
    text, summary = _build_circuit(
        quilt.circuit_case,
        stitch_rounds=rounds,
        noise=NoiseConfig(*([p] * 6), p_idle=_idle_rate(p)),
        prefer_blockwise_center_history=True,
        preserve_raw_center_checks=preserve_raw_checks,
        center_section_variant=section_variant,
    )
    summary["protocol"] = quilt.build.protocol
    summary["sec_schedule_policy"] = [
        {
            "block": block_id,
            "schedule": getattr(getattr(block, "base", block), "schedule_provenance", "rotated_surface_hook_order"),
            "alternation": "reverse_cnot_epochs_on_odd_physical_rounds",
        }
        for block_id, block in zip(quilt.block_ids, quilt.blocks, strict=True)
    ]
    return GeneratedCircuit(case_id=quilt.case_id, text=text, summary=summary)


def build_control(
    quilt: QuiltInput | str | Path,
    *,
    case_id: str | None = None,
    p: float | None = None,
    rounds: int | None = None,
) -> GeneratedCircuit:
    """Generate the noninteracting memory control matched to a quilt's blocks.

    ``rounds`` is the total number of ordinary SEC rounds.  By default it is
    the quilt's surgery-round count plus its opening and closing rounds.
    """

    if not isinstance(quilt, QuiltInput):
        quilt = load_quilt(quilt)
    p = quilt.build.physical_error_rate if p is None else float(p)
    rounds = quilt.build.surgery_rounds + 2 if rounds is None else int(rounds)
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if rounds < 3:
        raise ValueError("a matched control requires at least three SEC rounds")
    before = quilt.ledger.before
    control_case = GenericQuiltCircuitCase(
        name=case_id or f"CONTROL_{quilt.case_id}",
        blocks=quilt.blocks,
        subsystem=before,
        added_gauges=(),
        gauge_groups=(),
        measured_logicals=(),
        measurement_analysis=analyze_measurement(before, ()),
        target_labels=(),
    )
    text, summary = _build_circuit(
        control_case,
        stitch_rounds=rounds - 2,
        noise=NoiseConfig(*([p] * 6), p_idle=_idle_rate(p)),
        prefer_blockwise_center_history=True,
        preserve_raw_center_checks=True,
        center_section_variant=0,
    )
    summary["control_for"] = quilt.case_id
    summary["control"] = True
    return GeneratedCircuit(
        case_id=case_id or f"CONTROL_{quilt.case_id}", text=text, summary=summary
    )


def build_memory(
    block: str,
    *,
    case_id: str | None = None,
    p: float = 0.003,
    rounds: int = 14,
    distance: int = 12,
    sec_schedule: str = BB_SEC_SCHEDULE_LEGACY,
) -> GeneratedCircuit:
    """Generate ordinary single-block idle memory using a bundled SEC adapter.

    The block executes only its native syndrome-extraction schedule. In
    particular, this builder never inserts a local-Clifford wrapper. ``rounds``
    is the total number of ordinary SEC rounds, including the opening and
    closing rounds used by the flow construction.
    """

    key = str(block).lower().replace("-", "_")
    if key == "bb144":
        adapter = BB144SecBlock(
            block_index=0,
            schedule_provenance=str(sec_schedule),
        )
        default_case_id = "IDLE_BB144"
        block_kind = "bb144"
        block_distance: int | None = 12
    elif key in {"rotated_surface", "surface"}:
        if sec_schedule != BB_SEC_SCHEDULE_LEGACY:
            raise ValueError("rotated-surface memory does not use a BB SEC schedule")
        adapter = RotatedSurfaceSecBlock(distance=int(distance), block_index=0)
        default_case_id = f"IDLE_SC_D{int(distance)}"
        block_kind = "rotated_surface"
        block_distance = int(distance)
    else:
        raise ValueError(
            f"unknown memory block {block!r}; expected 'bb144' or 'rotated_surface'"
        )
    if not 0 <= float(p) <= 1:
        raise ValueError("p must be in [0, 1]")
    if int(rounds) < 3:
        raise ValueError("idle memory requires at least three SEC rounds")
    if case_id is not None and not str(case_id):
        raise ValueError("case_id must be nonempty")

    stabilizer_block = StabilizerBlock(
        name=str(adapter.name),
        n=int(adapter.n_data),
        stabilizers=tuple(int(check.pauli_row) for check in adapter.checks),
    )
    subsystem = build_general_subsystem(
        n=stabilizer_block.n,
        base_stabilizers=stabilizer_block.stabilizers,
        added_gauges=(),
    )
    analysis = analyze_measurement(subsystem, ())
    memory_case = GenericQuiltCircuitCase(
        name=str(case_id or default_case_id),
        blocks=(adapter,),
        subsystem=subsystem,
        added_gauges=(),
        gauge_groups=(),
        measured_logicals=(),
        measurement_analysis=analysis,
        target_labels=(),
    )
    text, summary = _build_circuit(
        memory_case,
        stitch_rounds=int(rounds) - 2,
        noise=NoiseConfig(*([float(p)] * 6), p_idle=_idle_rate(float(p))),
        prefer_blockwise_center_history=True,
        preserve_raw_center_checks=True,
        center_section_variant=0,
    )
    summary["memory"] = True
    memory_block: dict[str, object] = {
        "kind": block_kind,
        "distance": block_distance,
        "data_qubits": int(adapter.n_data),
        "physical_qubits": int(adapter.n_total),
        "checks": len(adapter.checks),
    }
    if hasattr(adapter, "schedule_provenance"):
        memory_block["sec_schedule"] = adapter.schedule_provenance
    summary["memory_block"] = memory_block
    return GeneratedCircuit(
        case_id=str(case_id or default_case_id),
        text=text,
        summary=summary,
    )


def build_memory_from_spec(
    memory: MemorySpec | str | Path,
) -> GeneratedCircuit:
    """Generate idle memory from a validated public ``memory.json`` input."""

    if not isinstance(memory, MemorySpec):
        memory = load_memory_spec(memory)
    return build_memory(
        memory.adapter_kind,
        case_id=memory.case_id,
        p=memory.physical_error_rate,
        rounds=memory.rounds,
        distance=memory.distance,
        sec_schedule=memory.effective_sec_schedule,
    )


__all__ = [
    "BB144_SEC_SCHEDULE_BRAVYI_2024",
    "BB_SEC_SCHEDULE_LEGACY",
    "BuildConfig",
    "GeneratedCircuit",
    "MemorySpec",
    "QuiltInput",
    "QuiltSpec",
    "StabilizerBlock",
    "StitchingFamily",
    "build_quilt",
    "build_control",
    "build_memory",
    "build_memory_from_spec",
    "load_memory_spec",
    "load_quilt",
    "quilt_from_parts",
]
