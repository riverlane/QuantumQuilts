"""Load and validate explicit quilt construction inputs.

The source package intentionally contains no representative catalogue.  A
caller supplies syndrome-extraction blocks and ordered local stitching pieces;
all subsystem, centre, logical, and measurement data are derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._algebra import (
    QuiltAlgebraLedger,
    QuiltSpec,
    StabilizerBlock,
    StitchingFamily,
)
from ._circuit import GenericQuiltCircuitCase
from ._core import (
    BB144_SEC_SCHEDULE_BRAVYI_2024,
    BB_SEC_SCHEDULE_LEGACY,
)
from ._sec_blocks import (
    BBSecBlock,
    BB144SecBlock,
    RotatedSurfaceSecBlock,
    SecBlock,
    UniformLocalCliffordSecBlock,
)
from ._symplectic import pauli


SCHEMA = "quilt-construction/quilt-spec/v1"
MEMORY_SCHEMA = "quilt-construction/memory-spec/v1"


def _as_mapping(value: object, *, subject: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{subject} must be an object")
    return value


def _as_sequence(value: object, *, subject: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{subject} must be an array")
    return value


def _indices(value: object, *, n: int, subject: str) -> tuple[int, ...]:
    values = tuple(int(q) for q in _as_sequence(value, subject=subject))
    if len(values) != len(set(values)):
        raise ValueError(f"{subject} contains duplicate qubits")
    if any(q < 0 or q >= n for q in values):
        raise ValueError(f"{subject} contains a qubit outside 0..{n - 1}")
    return values


def _pauli_piece(value: object, *, n: int, subject: str) -> int:
    payload = _as_mapping(value, subject=subject)
    unknown = set(payload) - {"x", "z"}
    if unknown:
        raise ValueError(f"{subject} has unknown fields: {sorted(unknown)}")
    xs = _indices(payload.get("x", ()), n=n, subject=f"{subject}.x")
    zs = _indices(payload.get("z", ()), n=n, subject=f"{subject}.z")
    x = sum(1 << q for q in xs)
    z = sum(1 << q for q in zs)
    if not x and not z:
        raise ValueError(f"{subject} is the identity")
    return pauli(x, z, n)


def _load_sec_adapter(payload: Mapping[str, Any], *, index: int) -> SecBlock:
    kind = str(payload.get("kind", ""))
    if kind in {"bb72", "bb144", "bb288"}:
        block: SecBlock = BBSecBlock(
            code_name=kind,
            block_index=index,
            schedule_provenance=str(
                payload.get("sec_schedule", BB_SEC_SCHEDULE_LEGACY)
            ),
        )
    elif kind == "rotated_surface":
        block = RotatedSurfaceSecBlock(
            distance=int(payload["distance"]),
            block_index=index,
        )
    elif kind == "uniform_local_clifford":
        block = UniformLocalCliffordSecBlock(
            base=_load_sec_adapter(
                _as_mapping(payload.get("base"), subject="circuit_adapter.base"),
                index=index,
            ),
            frame=str(payload["frame"]),
        )
    else:
        raise ValueError(
            f"unknown SEC adapter {kind!r}"
        )
    return block


def _load_sec_block(payload: Mapping[str, Any], *, index: int) -> SecBlock:
    adapter = payload.get("circuit_adapter")
    if adapter is None:
        # Concise form accepted for hand-written inputs.
        adapter = {"kind": payload.get("sec")}
        if "distance" in payload:
            adapter["distance"] = payload["distance"]
        if str(payload.get("frame", "I")).upper() != "I":
            adapter = {
                "kind": "uniform_local_clifford",
                "base": adapter,
                "frame": payload["frame"],
            }
    return _load_sec_adapter(
        _as_mapping(adapter, subject=f"block {payload.get('id', index)!r}.circuit_adapter"),
        index=index,
    )


def _stabilizer_block(
    block_id: str,
    payload: Mapping[str, Any],
    sec: SecBlock,
) -> StabilizerBlock:
    raw_stabilizers = payload.get("stabilizer_generators")
    if raw_stabilizers is None:
        stabilizers = tuple(int(check.pauli_row) for check in sec.checks)
    else:
        values = _as_sequence(
            raw_stabilizers, subject=f"block {block_id!r}.stabilizer_generators"
        )
        stabilizers = tuple(
            _pauli_piece(
                value,
                n=int(sec.n_data),
                subject=f"block {block_id!r}.stabilizer_generators[{index}]",
            )
            for index, value in enumerate(values)
        )
    block = StabilizerBlock(
        name=block_id,
        n=int(sec.n_data),
        stabilizers=stabilizers,
    )
    measured = StabilizerBlock(
        name=block_id,
        n=int(sec.n_data),
        stabilizers=tuple(int(check.pauli_row) for check in sec.checks),
    )
    if block.stabilizers != measured.stabilizers:
        raise ValueError(
            f"block {block_id!r}: circuit adapter does not measure the supplied code"
        )
    if "n" in payload and int(payload["n"]) != block.n:
        raise ValueError(f"block {block_id!r}: declared n does not match the adapter")
    if (
        "logical_qubits" in payload
        and int(payload["logical_qubits"]) != block.logical_qubits
    ):
        raise ValueError(
            f"block {block_id!r}: declared logical-qubit count is inconsistent"
        )
    return block


@dataclass(frozen=True)
class BuildConfig:
    physical_error_rate: float = 0.003
    surgery_rounds: int = 12
    preserve_raw_checks: bool = True
    canonical_section_variant: int = 0
    protocol: str = "subsystem"

    def __post_init__(self) -> None:
        if self.protocol != "subsystem":
            raise ValueError("only the subsystem circuit protocol is implemented")
        if not 0 <= float(self.physical_error_rate) <= 1:
            raise ValueError("physical_error_rate must be in [0, 1]")
        if int(self.surgery_rounds) <= 0:
            raise ValueError("surgery_rounds must be positive")
        if int(self.canonical_section_variant) < 0:
            raise ValueError("canonical_section_variant must be nonnegative")


@dataclass(frozen=True)
class MemorySpec:
    """Primary input for a reproducible ordinary single-block memory build."""

    case_id: str
    adapter_kind: str
    distance: int
    physical_error_rate: float
    rounds: int
    sec_schedule: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        if self.adapter_kind not in {"bb144", "rotated_surface"}:
            raise ValueError(f"unsupported memory adapter {self.adapter_kind!r}")
        if self.adapter_kind == "rotated_surface" and self.distance < 2:
            raise ValueError("rotated-surface distance must be at least 2")
        if self.adapter_kind == "rotated_surface" and self.sec_schedule is not None:
            raise ValueError("rotated-surface memory does not use a BB SEC schedule")
        if self.adapter_kind == "bb144" and self.effective_sec_schedule not in {
            BB_SEC_SCHEDULE_LEGACY,
            BB144_SEC_SCHEDULE_BRAVYI_2024,
        }:
            raise ValueError(f"unknown BB SEC schedule {self.effective_sec_schedule!r}")
        if not 0 <= self.physical_error_rate <= 1:
            raise ValueError("physical_error_rate must be in [0, 1]")
        if self.rounds < 3:
            raise ValueError("idle memory requires at least three SEC rounds")

    @property
    def effective_sec_schedule(self) -> str:
        return self.sec_schedule or BB_SEC_SCHEDULE_LEGACY

    def to_payload(self) -> dict[str, Any]:
        adapter: dict[str, Any] = {"kind": self.adapter_kind}
        if self.adapter_kind == "rotated_surface":
            adapter["distance"] = self.distance
        elif self.sec_schedule is not None:
            adapter["sec_schedule"] = self.sec_schedule
        return {
            "schema": MEMORY_SCHEMA,
            "case_id": self.case_id,
            "circuit_adapter": adapter,
            "physical_error_rate": self.physical_error_rate,
            "rounds": self.rounds,
        }


def load_memory_spec(source: str | Path | Mapping[str, Any]) -> MemorySpec:
    """Load the concise, catalogue-free input for one idle-memory build."""

    if isinstance(source, Mapping):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    payload = _as_mapping(payload, subject="memory spec")
    unknown = set(payload) - {
        "schema",
        "case_id",
        "circuit_adapter",
        "physical_error_rate",
        "rounds",
    }
    if unknown:
        raise ValueError(f"memory spec has unknown fields: {sorted(unknown)}")
    if payload.get("schema") != MEMORY_SCHEMA:
        raise ValueError(f"unsupported memory schema {payload.get('schema')!r}")
    adapter = _as_mapping(payload.get("circuit_adapter"), subject="circuit_adapter")
    kind = str(adapter.get("kind", ""))
    if kind == "bb144":
        adapter_unknown = set(adapter) - {"kind", "sec_schedule"}
        distance = 12
    elif kind == "rotated_surface":
        adapter_unknown = set(adapter) - {"kind", "distance"}
        if "distance" not in adapter:
            raise ValueError("rotated_surface circuit_adapter requires distance")
        distance = int(adapter["distance"])
    else:
        raise ValueError(
            f"unsupported memory circuit_adapter kind {kind!r}; "
            "expected 'bb144' or 'rotated_surface'"
        )
    if adapter_unknown:
        raise ValueError(
            f"memory circuit_adapter has unknown fields: {sorted(adapter_unknown)}"
        )
    try:
        case_id = str(payload["case_id"])
        physical_error_rate = float(payload["physical_error_rate"])
        rounds = int(payload["rounds"])
    except KeyError as exc:
        raise ValueError(f"memory spec is missing required field {exc.args[0]!r}") from exc
    return MemorySpec(
        case_id=case_id,
        adapter_kind=kind,
        distance=distance,
        physical_error_rate=physical_error_rate,
        rounds=rounds,
        sec_schedule=(
            str(adapter["sec_schedule"])
            if "sec_schedule" in adapter
            else None
        ),
    )


@dataclass(frozen=True)
class QuiltInput:
    """A fully validated quilt plus its physical SEC implementations."""

    case_id: str
    name: str
    block_ids: tuple[str, ...]
    blocks: tuple[SecBlock, ...]
    spec: QuiltSpec
    ledger: QuiltAlgebraLedger
    target_labels: tuple[str, ...]
    gauge_groups: tuple[tuple[int, ...], ...]
    build: BuildConfig
    expected: Mapping[str, Any]

    @property
    def circuit_case(self) -> GenericQuiltCircuitCase:
        measured = tuple(product.row for product in self.ledger.measured_products)
        return GenericQuiltCircuitCase(
            name=self.name,
            blocks=self.blocks,
            subsystem=self.ledger.merged,
            added_gauges=self.ledger.canonical_stitch_gauges,
            gauge_groups=self.gauge_groups,
            measured_logicals=measured,
            measurement_analysis=self.ledger.analysis,
            target_labels=self.target_labels,
        )

    def validation_summary(self) -> dict[str, Any]:
        return {
            "schema": "quilt-construction/quilt-validation/v1",
            "case_id": self.case_id,
            "blocks": list(self.block_ids),
            "data_qubits": self.spec.total_n,
            "stitch_gauges": len(self.ledger.canonical_stitch_gauges),
            "measured_rank": self.ledger.measured_rank,
            "surviving_logical_qubits": self.ledger.surviving_logical_count,
            "contains_desired": self.ledger.measures_all_intended,
            "no_overmeasurement": self.ledger.no_overmeasurement,
            "preserves_unmeasured_logicals": self.ledger.preserves_unmeasured_logicals,
            "valid": self.ledger.valid,
        }


def load_quilt(source: str | Path | Mapping[str, Any]) -> QuiltInput:
    """Parse primary inputs and derive the complete validated quilt ledger."""

    if isinstance(source, Mapping):
        payload = source
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported quilt schema {payload.get('schema')!r}")

    raw_blocks = _as_sequence(payload.get("blocks"), subject="blocks")
    if len(raw_blocks) < 2:
        raise ValueError("a quilt requires at least two blocks")
    block_payloads = tuple(
        _as_mapping(block, subject=f"blocks[{index}]")
        for index, block in enumerate(raw_blocks)
    )
    block_ids = tuple(str(block["id"]) for block in block_payloads)
    if any(not block_id for block_id in block_ids) or len(set(block_ids)) != len(block_ids):
        raise ValueError("block ids must be nonempty and unique")
    block_index = {block_id: index for index, block_id in enumerate(block_ids)}
    sec_blocks = tuple(
        _load_sec_block(block, index=index)
        for index, block in enumerate(block_payloads)
    )
    algebra_blocks = tuple(
        _stabilizer_block(block_id, block_payload, sec)
        for block_id, block_payload, sec in zip(
            block_ids, block_payloads, sec_blocks, strict=True
        )
    )

    raw_families = _as_sequence(
        payload.get("stitching_families", payload.get("families")),
        subject="families",
    )
    families: list[StitchingFamily] = []
    labels: list[str] = []
    gauge_groups: list[tuple[int, ...]] = []
    gauge_cursor = 0
    for family_index, raw_family in enumerate(raw_families):
        family = _as_mapping(
            raw_family, subject=f"stitching_families[{family_index}]"
        )
        family_name = str(family["name"])
        raw_incidence = _as_sequence(
            family.get("incidences", family.get("incidence_patterns")),
            subject=f"{family_name}.incidence_patterns",
        )
        incidences: list[int] = []
        for incidence_index, raw_ids in enumerate(raw_incidence):
            ids = tuple(
                str(value)
                for value in _as_sequence(
                    raw_ids,
                    subject=f"{family_name}.incidences[{incidence_index}]",
                )
            )
            if len(ids) != len(set(ids)):
                raise ValueError(f"{family_name}: an incidence repeats a block")
            try:
                incidence = sum(1 << block_index[block_id] for block_id in ids)
            except KeyError as exc:
                raise ValueError(
                    f"{family_name}: incidence names unknown block {exc.args[0]!r}"
                ) from exc
            incidences.append(incidence)

        raw_pieces = _as_mapping(
            family.get("pieces_by_block", family.get("local_pieces_by_block")),
            subject=f"{family_name}.local_pieces_by_block",
        )
        unknown_blocks = set(map(str, raw_pieces)) - set(block_ids)
        if unknown_blocks:
            raise ValueError(
                f"{family_name}: pieces name unknown blocks {sorted(unknown_blocks)}"
            )
        active = {
            block_id
            for raw_ids in raw_incidence
            for block_id in map(str, raw_ids)
        }
        piece_count: int | None = None
        parsed_by_block: list[tuple[int, ...]] = []
        for block_id, block in zip(block_ids, algebra_blocks, strict=True):
            raw_local = raw_pieces.get(block_id)
            if block_id in active and raw_local is None:
                raise ValueError(f"{family_name}: active block {block_id!r} has no pieces")
            values = () if raw_local is None else _as_sequence(
                raw_local, subject=f"{family_name}.pieces_by_block.{block_id}"
            )
            if block_id in active:
                if piece_count is None:
                    piece_count = len(values)
                elif len(values) != piece_count:
                    raise ValueError(
                        f"{family_name}: active blocks have different piece counts"
                    )
                parsed = tuple(
                    _pauli_piece(
                        value,
                        n=block.n,
                        subject=f"{family_name}.{block_id}[{piece_index}]",
                    )
                    for piece_index, value in enumerate(values)
                )
            else:
                if values:
                    raise ValueError(
                        f"{family_name}: inactive block {block_id!r} has pieces"
                    )
                parsed = ()
            parsed_by_block.append(parsed)
        if not piece_count:
            raise ValueError(f"{family_name}: no stitching pieces were supplied")
        parsed_by_block = [
            values if values else (0,) * piece_count for values in parsed_by_block
        ]
        families.append(
            StitchingFamily(
                name=family_name,
                local_pieces_by_block=tuple(parsed_by_block),
                incidence_patterns=tuple(incidences),
            )
        )
        raw_labels = family.get("labels")
        if raw_labels is None:
            raw_labels = [
                family.get("label", family.get("target_label", family_name))
            ] * len(incidences)
        family_labels = tuple(
            str(label)
            for label in _as_sequence(raw_labels, subject=f"{family_name}.labels")
        )
        if len(family_labels) != len(incidences):
            raise ValueError(f"{family_name}: one label is required per incidence")
        labels.extend(family_labels)
        for _ in incidences:
            gauge_groups.append(tuple(range(gauge_cursor, gauge_cursor + piece_count)))
            gauge_cursor += piece_count

    spec = QuiltSpec(blocks=algebra_blocks, families=tuple(families))
    ledger = QuiltAlgebraLedger.build(spec)
    if ledger.measured_rank != len(gauge_groups):
        raise ValueError(
            "measured incidence products must be independent for circuit generation"
        )
    build_payload = _as_mapping(payload.get("build", {}), subject="build")
    build = BuildConfig(
        protocol=str(build_payload.get("protocol", "subsystem")),
        physical_error_rate=float(build_payload.get("physical_error_rate", 0.003)),
        surgery_rounds=int(
            build_payload.get("surgery_rounds", build_payload.get("stitch_rounds", 12))
        ),
        preserve_raw_checks=bool(build_payload.get("preserve_raw_checks", True)),
        canonical_section_variant=int(
            build_payload.get(
                "canonical_section_variant", build_payload.get("section_variant", 0)
            )
        ),
    )
    result = QuiltInput(
        case_id=str(payload.get("case_id", payload.get("name", "quilt"))),
        name=str(payload.get("name", payload.get("case_id", "quilt"))),
        block_ids=block_ids,
        blocks=sec_blocks,
        spec=spec,
        ledger=ledger,
        target_labels=tuple(labels),
        gauge_groups=tuple(gauge_groups),
        build=build,
        expected=_as_mapping(payload.get("expected", {}), subject="expected"),
    )
    summary = result.validation_summary()
    for key in (
        "data_qubits",
        "stitch_gauges",
        "measured_rank",
        "surviving_logical_qubits",
    ):
        expected_key = (
            "merged_logical_qubits"
            if key == "surviving_logical_qubits"
            else key
        )
        if expected_key in result.expected and int(result.expected[expected_key]) != int(summary[key]):
            raise ValueError(
                f"expected {expected_key}={result.expected[expected_key]}, derived {summary[key]}"
            )
    return result


def quilt_from_parts(
    *,
    case_id: str,
    name: str,
    blocks: Sequence[SecBlock],
    spec: QuiltSpec,
    target_labels: Sequence[str],
    build: BuildConfig = BuildConfig(),
) -> QuiltInput:
    """Validate a programmatically supplied quilt without JSON adapters."""

    blocks = tuple(blocks)
    if tuple(int(block.n_data) for block in blocks) != tuple(
        block.n for block in spec.blocks
    ):
        raise ValueError("SEC block widths do not match the algebraic code blocks")
    for index, (sec, block) in enumerate(zip(blocks, spec.blocks, strict=True)):
        sec_rows = StabilizerBlock(
            name=block.name,
            n=block.n,
            stabilizers=tuple(check.pauli_row for check in sec.checks),
        ).stabilizers
        if sec_rows != block.stabilizers:
            raise ValueError(f"SEC block {index} does not measure the supplied code")
    ledger = QuiltAlgebraLedger.build(spec)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for family in spec.families:
        for _ in family.incidence_patterns:
            groups.append(tuple(range(cursor, cursor + family.piece_count)))
            cursor += family.piece_count
    labels = tuple(map(str, target_labels))
    if len(labels) != len(groups):
        raise ValueError("one target label is required per measured incidence")
    return QuiltInput(
        case_id=str(case_id),
        name=str(name),
        block_ids=tuple(block.name for block in spec.blocks),
        blocks=blocks,
        spec=spec,
        ledger=ledger,
        target_labels=labels,
        gauge_groups=tuple(groups),
        build=build,
        expected={},
    )


__all__ = [
    "BuildConfig",
    "MEMORY_SCHEMA",
    "MemorySpec",
    "QuiltInput",
    "SCHEMA",
    "load_memory_spec",
    "load_quilt",
    "quilt_from_parts",
]
