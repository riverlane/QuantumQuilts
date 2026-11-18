"""Generic N-block quilt circuits with algebraically derived observables."""

from __future__ import annotations

import os as _os

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import stim

from ._core import StimTextBuilder
from ._algebra import (
    StabilizerBlock,
    _SparseSyndromeMap,
    _subspace_intersection,
    blockwise_css_basis,
    blockwise_pauli_types,
    glued_center_candidates,
    sparse_n2_blockwise_css_center_basis,
)
from ._flows import (
    measurement_indices,
    FlowMeasurementSolver,
    complete_detector_set,
    deterministic_measurement_basis,
    sparse_deterministic_candidates,
)
from ._core import (
    PAULI_LABELS,
    NoiseConfig,
    PauliTerm,
    add_single_qubit_pauli,
    append_basis_into_z,
    append_basis_out_of_z,
    append_boundary_terms_n,
    append_detectors_and_observables,
    append_idle_noise,
    css_sector_code,
    detector_sector_from_basis,
    measurement_bases,
    append_noise_after_cx,
    append_noise_after_reset,
    append_noise_before_measure,
    assert_pairwise_commuting,
    audit_observable_gauge_record_support,
    dem_logical_support_audit,
    shifted_mask,
    symplectic_pairs,
)
from ._gf2 import nullspace, reduce_with_basis, row_reduce
from ._symplectic import (
    GeneralSubsystem,
    MeasurementAnalysis,
    analyze_measurement,
    pauli,
    pauli_weight,
    symplectic_product,
    x_part,
    z_part,
)


@dataclass(frozen=True)
class GenericQuiltCircuitCase:
    """The algebraic and physical data needed for one quilt circuit."""

    name: str
    blocks: tuple[Any, ...]
    subsystem: GeneralSubsystem
    added_gauges: tuple[int, ...]
    gauge_groups: tuple[tuple[int, ...], ...]
    measured_logicals: tuple[int, ...]
    measurement_analysis: MeasurementAnalysis
    target_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a circuit requires at least one block")
        if len(self.gauge_groups) != len(self.measured_logicals):
            raise ValueError("one gauge group is required per measured logical")
        if len(self.target_labels) != len(self.measured_logicals):
            raise ValueError("one label is required per measured logical")
        if tuple(range(len(self.added_gauges))) != tuple(
            index for group in self.gauge_groups for index in group
        ):
            raise ValueError("gauge groups must partition the gauge ledger in order")
        for group, logical in zip(
            self.gauge_groups, self.measured_logicals, strict=True
        ):
            product = 0
            for index in group:
                product ^= self.added_gauges[index]
            if product != logical:
                raise ValueError("a gauge-group product is not its measured logical")
        if not self.measurement_analysis.valid or not self.measurement_analysis.exact:
            raise ValueError("the quilt must measure exactly the declared logical space")
        if self.measurement_analysis.measured_rank != len(self.measured_logicals):
            raise ValueError("declared measured rank does not match the exact analysis")
        n = self.subsystem.n
        if any(
            symplectic_product(left, right, n)
            for index, left in enumerate(self.added_gauges)
            for right in self.added_gauges[index + 1 :]
        ):
            raise ValueError("the physical stitching pieces must commute")
        if any(pauli_weight(row, n) == 0 for row in self.added_gauges):
            raise ValueError("the physical stitching pieces must be nonzero")
        support = 0
        for row in self.added_gauges:
            current = x_part(row, n) | z_part(row, n)
            if support & current:
                raise ValueError("the physical stitching pieces must be disjoint")
            support |= current


@dataclass(frozen=True)
class GenericReferenceLayout:
    start: int
    protected: tuple[int, ...]
    measured: tuple[int, ...]
    num_qubits: int


@dataclass(frozen=True)
class GenericPhysicalWindow:
    text: str
    round_kinds: tuple[str, ...]
    gauge_measurements: tuple[tuple[int, ...], ...]
    check_measurements: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GenericBoundary:
    stabilizers: tuple[PauliTerm, ...]
    protected: tuple[PauliTerm, ...]
    measured_bell: tuple[PauliTerm, ...]
    measured_ref: tuple[PauliTerm, ...]


def _block_data_offsets(case: GenericQuiltCircuitCase) -> tuple[int, ...]:
    offsets = []
    cursor = 0
    for block in case.blocks:
        offsets.append(cursor)
        cursor += int(block.n_data)
    if cursor != case.subsystem.n:
        raise ValueError("block data widths do not match the subsystem width")
    return tuple(offsets)


def _block_physical_offsets(case: GenericQuiltCircuitCase) -> tuple[int, ...]:
    offsets = []
    cursor = 0
    for block in case.blocks:
        offsets.append(cursor)
        cursor += int(block.n_total)
    return tuple(offsets)


def references(case: GenericQuiltCircuitCase) -> GenericReferenceLayout:
    physical_n = sum(int(block.n_total) for block in case.blocks)
    protected_count = case.subsystem.logical_qubits
    protected = tuple(range(physical_n, physical_n + protected_count))
    measured = tuple(
        range(
            physical_n + protected_count,
            physical_n + protected_count + len(case.measured_logicals),
        )
    )
    return GenericReferenceLayout(
        start=physical_n,
        protected=protected,
        measured=measured,
        num_qubits=physical_n + protected_count + len(measured),
    )


def data_row_to_circuit(
    row: int,
    *,
    case: GenericQuiltCircuitCase,
    num_qubits: int,
) -> int:
    data_offsets = _block_data_offsets(case)
    physical_offsets = _block_physical_offsets(case)
    total_data_n = case.subsystem.n
    out_x = 0
    out_z = 0
    source_x = x_part(row, total_data_n)
    source_z = z_part(row, total_data_n)
    for block, (data_offset, physical_offset) in enumerate(
        zip(data_offsets, physical_offsets, strict=True)
    ):
        n = int(case.blocks[block].n_data)
        mask = (1 << n) - 1
        out_x |= ((source_x >> data_offset) & mask) << physical_offset
        out_z |= ((source_z >> data_offset) & mask) << physical_offset
    return pauli(out_x, out_z, num_qubits)


def data_row_to_sec_frame(
    row: int,
    *,
    case: GenericQuiltCircuitCase,
) -> int:
    """Express a physical-frame data Pauli at an in-round SEC marker."""

    data_offsets = _block_data_offsets(case)
    total_data_n = case.subsystem.n
    source_x = x_part(row, total_data_n)
    source_z = z_part(row, total_data_n)
    out_x = 0
    out_z = 0
    for block, data_offset in zip(case.blocks, data_offsets, strict=True):
        n = int(block.n_data)
        mask = (1 << n) - 1
        local = pauli(
            (source_x >> data_offset) & mask,
            (source_z >> data_offset) & mask,
            n,
        )
        local = int(block.to_sec_frame_pauli_row(local))
        out_x |= x_part(local, n) << data_offset
        out_z |= z_part(local, n) << data_offset
    return pauli(out_x, out_z, total_data_n)


def boundary_terms(
    case: GenericQuiltCircuitCase,
    refs: GenericReferenceLayout,
) -> GenericBoundary:
    pairs = symplectic_pairs(case.subsystem.bare_logical_basis, case.subsystem.n)
    if len(pairs) != case.subsystem.logical_qubits:
        raise AssertionError("merged bare-logical basis has the wrong dimension")
    protected: list[PauliTerm] = []
    for index, (logical_x, logical_z) in enumerate(pairs):
        x_row = data_row_to_circuit(
            logical_x, case=case, num_qubits=refs.num_qubits
        )
        z_row = data_row_to_circuit(
            logical_z, case=case, num_qubits=refs.num_qubits
        )
        protected.extend(
            (
                PauliTerm(
                    f"protected_X_bell_{index}",
                    add_single_qubit_pauli(
                        x_row,
                        q=refs.protected[index],
                        label="X",
                        n=refs.num_qubits,
                    ),
                ),
                PauliTerm(
                    f"protected_Z_bell_{index}",
                    add_single_qubit_pauli(
                        z_row,
                        q=refs.protected[index],
                        label="Z",
                        n=refs.num_qubits,
                    ),
                ),
            )
        )

    stabilizers = tuple(
        PauliTerm(
            f"base_stabilizer_{index}",
            data_row_to_circuit(row, case=case, num_qubits=refs.num_qubits),
        )
        for index, row in enumerate(case.subsystem.base_stabilizers)
    )
    measured_bell = []
    measured_ref = []
    for index, (logical, label) in enumerate(
        zip(case.measured_logicals, case.target_labels, strict=True)
    ):
        data = data_row_to_circuit(
            logical, case=case, num_qubits=refs.num_qubits
        )
        measured_bell.append(
            PauliTerm(
                f"measured_{label}_bell",
                add_single_qubit_pauli(
                    data,
                    q=refs.measured[index],
                    label="Z",
                    n=refs.num_qubits,
                ),
            )
        )
        measured_ref.append(
            PauliTerm(
                f"measured_{label}_reference",
                pauli(0, 1 << refs.measured[index], refs.num_qubits),
            )
        )
    assert_pairwise_commuting(
        [*stabilizers, *protected, *measured_bell], refs.num_qubits
    )
    assert_pairwise_commuting(
        [*stabilizers, *protected, *measured_bell, *measured_ref],
        refs.num_qubits,
    )
    return GenericBoundary(
        stabilizers=stabilizers,
        protected=tuple(protected),
        measured_bell=tuple(measured_bell),
        measured_ref=tuple(measured_ref),
    )


def raw_stabilizer_terms(
    case: GenericQuiltCircuitCase,
    refs: GenericReferenceLayout,
) -> tuple[PauliTerm, ...]:
    """Physical SEC checks in the exact common ancilla-readout order."""

    rows: list[tuple[str, int]] = []
    data_offsets = _block_data_offsets(case)
    total_n = case.subsystem.n
    for readout_name in ("readout_z", "readout_x"):
        for block_index, block in enumerate(case.blocks):
            schedule = block.sec_schedule(reverse=False)
            checks_by_label = {check.label: check for check in block.checks}
            for readout in getattr(schedule, readout_name):
                check = checks_by_label[readout.check_label]
                local_x = x_part(int(check.row), int(block.n_data))
                local_z = z_part(int(check.row), int(block.n_data))
                embedded = pauli(
                    local_x << data_offsets[block_index],
                    local_z << data_offsets[block_index],
                    total_n,
                )
                rows.append(
                    (
                        f"raw_{check.sector}_b{block_index}_{check.label}",
                        data_row_to_circuit(
                            embedded, case=case, num_qubits=refs.num_qubits
                        ),
                    )
                )
    return tuple(PauliTerm(label, row) for label, row in rows)


def _local_label(row: int, q: int, n: int) -> str:
    key = ((x_part(row, n) >> q) & 1, (z_part(row, n) >> q) & 1)
    try:
        return PAULI_LABELS[key]
    except KeyError as exc:
        raise ValueError("stitching support contains an identity factor") from exc


def _gauge_sites(
    case: GenericQuiltCircuitCase,
    refs: GenericReferenceLayout,
) -> tuple[tuple[tuple[int, str], ...], ...]:
    data_offsets = _block_data_offsets(case)
    physical_offsets = _block_physical_offsets(case)
    result: list[tuple[tuple[int, str], ...]] = []
    for gauge in case.added_gauges:
        sites: list[tuple[int, str]] = []
        for block_index, block in enumerate(case.blocks):
            n = int(block.n_data)
            local_x = (x_part(gauge, case.subsystem.n) >> data_offsets[block_index]) & (
                (1 << n) - 1
            )
            local_z = (z_part(gauge, case.subsystem.n) >> data_offsets[block_index]) & (
                (1 << n) - 1
            )
            local = pauli(local_x, local_z, n)
            original_support = local_x | local_z
            local = int(block.to_sec_frame_pauli_row(local))
            local_x = x_part(local, n)
            local_z = z_part(local, n)
            support = local_x | local_z
            if support != original_support:
                raise AssertionError(
                    "an SEC-frame conversion changed stitching coordinates"
                )
            if support.bit_count() > 1:
                raise ValueError(
                    "the point-stitch circuit requires at most one factor per block"
                )
            if support:
                q = (support & -support).bit_length() - 1
                sites.append(
                    (physical_offsets[block_index] + q, _local_label(local, q, n))
                )
        if len(sites) < 2:
            raise ValueError("every stitching piece must touch at least two blocks")
        result.append(tuple(sites))
    return tuple(result)


def _uniform_gauge_basis(
    gauge_sites: Sequence[Sequence[tuple[int, str]]],
) -> str | None:
    """"X", "Z", or None when the gauges mix Pauli types."""

    labels = {label for gauge in gauge_sites for _, label in gauge}
    return labels.pop() if len(labels) == 1 and labels <= {"X", "Z"} else None


def _skip_rotation_for_uniform_x() -> bool:
    """Opt in to measuring an all-X gauge without rotating into Z.

    `append_basis_into_z` rotates every site of a gauge into the Z basis so a
    single generic routine can measure any Pauli product. For a gauge that is
    X on every site that is pure overhead: conjugating the CX ladder by H on
    both ends only reverses it, so running the ladder the other way and
    measuring the pivot in the X basis measures the same operator with no
    rotation at all. An all-Z gauge already needs no rotation, and a mixed
    gauge genuinely does, so this helps the X case alone.

    Off by default: turning it on changes the circuit, and with it the DEM and
    the measured error rate.
    """

    return _os.environ.get("QUILT_UNIFORM_X_GAUGE_NO_ROTATION", "0") not in ("", "0")


def _append_fused_morph_in(
    builder: StimTextBuilder,
    *,
    gauge_sites: Sequence[Sequence[tuple[int, str]]],
    noise: NoiseConfig,
    skip_rotation: bool = False,
) -> None:
    all_sites = [
        gauge[block_position]
        for block_position in range(max(len(gauge) for gauge in gauge_sites))
        for gauge in gauge_sites
        if block_position < len(gauge)
    ]
    if not skip_rotation:
        append_basis_into_z(builder, all_sites, noise.p_morph_1q)
    max_controls = max(len(gauge) - 1 for gauge in gauge_sites)
    for depth in range(max_controls):
        pairs = [
            (gauge[0][0], gauge[depth + 1][0]) if skip_rotation
            else (gauge[depth + 1][0], gauge[0][0])
            for gauge in gauge_sites
            if depth + 1 < len(gauge)
        ]
        emitted = builder.pairs("CX", pairs)
        append_noise_after_cx(builder, emitted, noise.p_morph_2q)
        builder.tick()


def _append_fused_morph_out(
    builder: StimTextBuilder,
    *,
    gauge_sites: Sequence[Sequence[tuple[int, str]]],
    noise: NoiseConfig,
    skip_rotation: bool = False,
) -> None:
    max_controls = max(len(gauge) - 1 for gauge in gauge_sites)
    for depth in range(max_controls - 1, -1, -1):
        pairs = [
            (gauge[0][0], gauge[depth + 1][0]) if skip_rotation
            else (gauge[depth + 1][0], gauge[0][0])
            for gauge in gauge_sites
            if depth + 1 < len(gauge)
        ]
        emitted = builder.pairs("CX", pairs)
        append_noise_after_cx(builder, emitted, noise.p_morph_2q)
        builder.tick()
    if skip_rotation:
        return
    append_basis_out_of_z(
        builder,
        [
            gauge[block_position]
            for block_position in range(max(len(gauge) for gauge in gauge_sites))
            for gauge in gauge_sites
            if block_position < len(gauge)
        ],
        noise.p_morph_1q,
    )


def append_sec_round(
    builder: StimTextBuilder,
    *,
    case: GenericQuiltCircuitCase,
    round_index: int,
    noise: NoiseConfig,
    reverse: bool,
    before_readout: Callable[[StimTextBuilder], None] | None = None,
    after_readout: Callable[[StimTextBuilder], None] | None = None,
) -> tuple[int, ...]:
    """Batch heterogeneous block SEC schedules around one shared callback."""

    physical_offsets = _block_physical_offsets(case)
    schedules = tuple(block.sec_schedule(reverse=reverse) for block in case.blocks)
    for block_index, schedule in enumerate(schedules):
        offset = physical_offsets[block_index]
        for layer in schedule.pre_data_gates:
            targets = [offset + int(q) for q in layer.targets]
            builder.gate(layer.gate, targets)
            if noise.p_sec_1q > 0:
                builder.gate(
                    f"DEPOLARIZE1({float(noise.p_sec_1q):.12g})", targets
                )
            builder.tick()
    reset_x = [
        physical_offsets[index] + q
        for index, schedule in enumerate(schedules)
        for q in schedule.reset_x
    ]
    reset_z = [
        physical_offsets[index] + q
        for index, schedule in enumerate(schedules)
        for q in schedule.reset_z
    ]
    builder.line(
        f"# === round {round_index}: {'reverse' if reverse else 'forward'} generic SEC ==="
    )
    builder.gate("RX", reset_x)
    append_noise_after_reset(builder, reset_x, "X", noise.p_sec_1q)
    builder.gate("R", reset_z)
    append_noise_after_reset(builder, reset_z, "Z", noise.p_sec_1q)
    builder.tick()
    max_epochs = max(len(schedule.epochs) for schedule in schedules)
    for epoch in range(max_epochs):
        pairs = [
            (
                physical_offsets[block_index] + control,
                physical_offsets[block_index] + target,
            )
            for block_index, schedule in enumerate(schedules)
            if epoch < len(schedule.epochs)
            for control, target in schedule.epochs[epoch]
        ]
        if len({q for pair in pairs for q in pair}) != 2 * len(pairs):
            raise AssertionError("a generic SEC epoch contains a qubit collision")
        emitted = builder.pairs("CX", pairs)
        append_noise_after_cx(builder, emitted, noise.p_sec_2q)
        builder.tick()
    if before_readout is not None:
        before_readout(builder)
    z_targets = [
        physical_offsets[index] + readout.target
        for index, schedule in enumerate(schedules)
        for readout in schedule.readout_z
    ]
    x_targets = [
        physical_offsets[index] + readout.target
        for index, schedule in enumerate(schedules)
        for readout in schedule.readout_x
    ]
    append_noise_before_measure(builder, z_targets, "Z", noise.p_sec_1q)
    z_records = builder.measure("MZ", z_targets)
    append_noise_before_measure(builder, x_targets, "X", noise.p_sec_1q)
    x_records = builder.measure("MX", x_targets)
    if after_readout is not None:
        after_readout(builder)
    for block_index, schedule in enumerate(schedules):
        offset = physical_offsets[block_index]
        for layer in schedule.post_data_gates:
            targets = [offset + int(q) for q in layer.targets]
            builder.gate(layer.gate, targets)
            if noise.p_sec_1q > 0:
                builder.gate(
                    f"DEPOLARIZE1({float(noise.p_sec_1q):.12g})", targets
                )
            builder.tick()
    if after_readout is None and not any(
        schedule.post_data_gates for schedule in schedules
    ):
        builder.tick()
    return tuple([*z_records, *x_records])


def physical_window(
    case: GenericQuiltCircuitCase,
    *,
    stitch_rounds: int,
    noise: NoiseConfig,
) -> GenericPhysicalWindow:
    if stitch_rounds <= 0:
        raise ValueError("stitch_rounds must be positive")
    refs = references(case)
    builder = StimTextBuilder()
    builder.gate("I", [*refs.protected, *refs.measured])
    builder.tick()
    gauge_sites = _gauge_sites(case, refs)
    round_kinds = (
        ("normal", *("stitch" for _ in range(stitch_rounds)), "normal")
        if gauge_sites
        else tuple("normal" for _ in range(stitch_rounds + 2))
    )
    check_rounds: list[tuple[int, ...]] = []
    gauge_rounds: list[tuple[int, ...]] = []
    for round_index, kind in enumerate(round_kinds):
        reverse = bool(round_index & 1)
        if kind == "normal":
            check_rounds.append(
                append_sec_round(
                    builder,
                    case=case,
                    round_index=round_index,
                    noise=noise,
                    reverse=reverse,
                )
            )
            continue
        records: list[int] = []

        skip_rotation = (
            _uniform_gauge_basis(gauge_sites) == "X" and _skip_rotation_for_uniform_x()
        )
        gauge_basis = "X" if skip_rotation else "Z"

        def before(local_builder: StimTextBuilder) -> None:
            _append_fused_morph_in(
                local_builder, gauge_sites=gauge_sites, noise=noise,
                skip_rotation=skip_rotation,
            )

        def after(local_builder: StimTextBuilder) -> None:
            anchors = [gauge[0][0] for gauge in gauge_sites]
            append_noise_before_measure(
                local_builder, anchors, gauge_basis, noise.p_morph_meas
            )
            records.extend(
                local_builder.measure("MX" if skip_rotation else "MZ", anchors)
            )
            local_builder.tick()
            append_noise_after_reset(
                local_builder, anchors, gauge_basis, noise.p_morph_post_measure
            )
            _append_fused_morph_out(
                local_builder, gauge_sites=gauge_sites, noise=noise,
                skip_rotation=skip_rotation,
            )

        check_rounds.append(
            append_sec_round(
                builder,
                case=case,
                round_index=round_index,
                noise=noise,
                reverse=reverse,
                before_readout=before,
                after_readout=after,
            )
        )
        gauge_rounds.append(tuple(records))
    return GenericPhysicalWindow(
        text=builder.text(),
        round_kinds=tuple(round_kinds),
        gauge_measurements=tuple(gauge_rounds),
        check_measurements=tuple(check_rounds),
    )


def append_raw_experiment(
    *,
    physical: GenericPhysicalWindow,
    refs: GenericReferenceLayout,
    boundary: GenericBoundary,
    stabilizers: Sequence[PauliTerm] | None = None,
) -> tuple[str, dict[str, dict[str, int]], int]:
    builder = StimTextBuilder()
    boundary_stabilizers = (
        boundary.stabilizers if stabilizers is None else tuple(stabilizers)
    )
    initial_stabs = append_boundary_terms_n(
        builder,
        boundary_stabilizers,
        n=refs.num_qubits,
        heading="noiseless initial stabilizer boundary MPPs",
    )
    initial_protected = append_boundary_terms_n(
        builder,
        boundary.protected,
        n=refs.num_qubits,
        heading="noiseless initial protected logical Bell MPPs",
    )
    initial_targets = append_boundary_terms_n(
        builder,
        boundary.measured_bell,
        n=refs.num_qubits,
        heading="noiseless initial measured-logical Bell MPPs",
    )
    physical_offset = builder.rec_count
    builder.lines.extend(physical.text.splitlines())
    builder.rec_count += stim.Circuit(physical.text).num_measurements
    final_protected = append_boundary_terms_n(
        builder,
        boundary.protected,
        n=refs.num_qubits,
        heading="noiseless final protected logical Bell MPPs",
    )
    final_targets = append_boundary_terms_n(
        builder,
        boundary.measured_bell,
        n=refs.num_qubits,
        heading="noiseless final measured-logical Bell MPPs",
    )
    final_refs = append_boundary_terms_n(
        builder,
        boundary.measured_ref,
        n=refs.num_qubits,
        heading="noiseless final measured-logical reference MPPs",
    )
    final_stabs = append_boundary_terms_n(
        builder,
        boundary_stabilizers,
        n=refs.num_qubits,
        heading="noiseless final stabilizer boundary MPPs",
    )
    return (
        builder.text(),
        {
            "initial_stabs": initial_stabs,
            "initial_protected": initial_protected,
            "initial_targets": initial_targets,
            "final_protected": final_protected,
            "final_targets": final_targets,
            "final_refs": final_refs,
            "final_stabs": final_stabs,
        },
        physical_offset,
    )


def observable_masks(
    *,
    case: GenericQuiltCircuitCase,
    ideal_physical: stim.Circuit,
    records: dict[str, dict[str, int]],
    physical_offset: int,
    boundary: GenericBoundary,
    physical: GenericPhysicalWindow | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    masks: list[int] = []
    metadata: list[dict[str, Any]] = []
    flow_solver = FlowMeasurementSolver.build(ideal_physical)
    for term in boundary.protected:
        internal = flow_solver.solve(
            input_row=term.row, output_row=term.row
        )
        mask = (
            (1 << records["initial_protected"][term.label])
            ^ shifted_mask(internal, physical_offset)
            ^ (1 << records["final_protected"][term.label])
        )
        masks.append(mask)
        metadata.append(
            {
                "label": term.label,
                "kind": "protected_memory",
                "record_weight": mask.bit_count(),
            }
        )
    measured = tuple(
        enumerate(
            zip(
                boundary.measured_bell,
                boundary.measured_ref,
                case.target_labels,
                strict=True,
            )
        )
    )
    memory_by_group: dict[int, int] = {}
    memory_index: dict[int, int] = {}
    for group_index, (bell, _ref, label) in measured:
        memory_internal = flow_solver.solve(
            input_row=bell.row, output_row=bell.row
        )
        memory = (
            (1 << records["initial_targets"][bell.label])
            ^ shifted_mask(memory_internal, physical_offset)
            ^ (1 << records["final_targets"][bell.label])
        )
        memory_by_group[group_index] = memory
        memory_index[group_index] = len(masks)
        masks.append(memory)
        metadata.append(
            {
                "label": f"measured_{label}_memory",
                "kind": "measured_memory",
                "record_weight": memory.bit_count(),
            }
        )
    # The parity is read every stitch round, and a group's gauge records
    # multiply to that round's reading.  Swapping the final reading for the
    # first moves the comparison to the opening rounds; the difference between
    # two readings is the telescoped gauge-history detectors, which are
    # deterministic, so the swapped row is still a deterministic parity.
    cap_swap: dict[int, int] = {}
    if physical is not None and len(physical.gauge_measurements) > 1:
        for group_index in range(len(case.gauge_groups)):
            swap = 0
            for gauge_index in case.gauge_groups[group_index]:
                swap ^= 1 << (
                    physical_offset + physical.gauge_measurements[0][gauge_index]
                )
                swap ^= 1 << (
                    physical_offset + physical.gauge_measurements[-1][gauge_index]
                )
            cap_swap[group_index] = swap

    measurement_by_group: dict[int, int] = {}
    for group_index, (bell, ref, label) in measured:
        measurement_internal = flow_solver.solve(
            input_row=bell.row, output_row=ref.row
        )
        measurement = (
            (1 << records["initial_targets"][bell.label])
            ^ shifted_mask(measurement_internal, physical_offset)
            ^ (1 << records["final_refs"][ref.label])
        )
        measurement_by_group[group_index] = measurement
        # Emit the comparison in its cap form when we can: the same
        # operational class, read at the first parity round instead of the
        # last, and therefore confined to the opening rounds.  It replaces the
        # late form rather than joining it -- emitting both would inflate the
        # operational count, because `complete_detector_set` seats observables
        # before detectors and would drop the very consistency detector that
        # certifies the two are the same class.  The late form stays available
        # as the XOR with the memory observable, modulo those detectors.
        # `{memory, measurement}` and `{cap, cup}` span the same two
        # operational directions per measured logical -- the second pair is a
        # change of basis, not an enlargement -- so emitting the confined pair
        # costs no rank and gives the temporal argument both of its boundary
        # cochains directly from the circuit.
        #
        #   cap = measurement + (first parity reading + last parity reading)
        #   cup = memory + measurement
        #
        # The bracketed term is the telescoped gauge-history detectors, so the
        # cap is the same class read at the opening instead of the closing
        # rounds; the cup drops the shared `initial_targets` anchor and so
        # lives at the closing end.  Their sum is the memory direction modulo
        # those same consistency detectors.
        swap = cap_swap.get(group_index, 0)
        cap = measurement ^ swap
        masks.append(cap)
        metadata.append(
            {
                "label": f"measured_{label}_cap",
                "kind": "measured_outcome",
                "record_weight": cap.bit_count(),
                "gauge_group": group_index,
                "stitch_round": 0 if swap else -1,
                "read_at": "first_parity_round" if swap else "final_reference",
            }
        )
        # Emit the cup in place of the memory observable, so both temporal
        # boundaries are visible in the circuit.  Neither cap nor cup descends
        # through the spatial fold on its own -- each lives at one end, and the
        # fold maps onto a merged interval in the middle -- but their sum is
        # the memory direction and does descend, so exactly one direction is
        # still killed.  The extractor presents that sum to the fold; see the
        # cap/cup re-basing in `quilt_verify.circuit.extract`.
        if swap:
            cup = memory_by_group[group_index] ^ measurement
            masks[memory_index[group_index]] = cup
            metadata[memory_index[group_index]].update(
                {
                    "label": f"measured_{label}_cup",
                    "kind": "measured_cup",
                    "record_weight": cup.bit_count(),
                    "gauge_group": group_index,
                    "stitch_round": -1,
                    "read_at": "closing_rounds",
                }
            )
    expected = 2 * (
        case.subsystem.logical_qubits + len(case.measured_logicals)
    )
    if len(masks) != expected:
        raise AssertionError(f"expected {expected} observables, got {len(masks)}")
    return masks, metadata


def _audit_observable_records(
    *,
    case: GenericQuiltCircuitCase,
    observables: Sequence[int],
    metadata: Sequence[dict[str, Any]],
    physical: GenericPhysicalWindow,
    physical_offset: int,
) -> dict[str, Any]:
    all_gauge_records = {
        physical_offset + record
        for round_records in physical.gauge_measurements
        for record in round_records
    }
    failures = []
    for index, (mask, row) in enumerate(zip(observables, metadata, strict=True)):
        used = {
            record
            for record in all_gauge_records
            if (int(mask) >> record) & 1
        }
        # An individual stitch outcome is random; the product over a whole
        # gauge group is the measured logical and is not.  So the rule is
        # completeness of the group, at whichever round the observable
        # declares -- the outcome observable reads the last round, the cap
        # reads the first.  Anything else, or a partial group, is random.
        if row["kind"] not in ("measured_outcome", "measured_cap",
                               "measured_cup") and used:
            failures.append(
                f"observable {index} ({row['label']}) uses random gauge records"
            )
        if row["kind"] in ("measured_outcome", "measured_cap", "measured_cup"):
            group = case.gauge_groups[int(row["gauge_group"])]
            round_index = int(row.get("stitch_round", -1))
            expected = {
                physical_offset
                + physical.gauge_measurements[round_index][gauge_index]
                for gauge_index in group
            }
            if used != expected:
                where = "final" if round_index == -1 else f"round {round_index}"
                failures.append(
                    f"observable {index} ({row['label']}) does not use exactly "
                    f"its complete gauge-group product at {where}"
                )
    if failures:
        raise AssertionError("; ".join(failures))
    return {
        "gauge_record_count": len(all_gauge_records),
        "protected_observables_with_gauge_records": 0,
        "measured_outcome_observables": len(case.measured_logicals),
    }


def preferred_gauge_history_detectors(
    *,
    case: GenericQuiltCircuitCase,
    physical: GenericPhysicalWindow,
    physical_offset: int,
) -> tuple[int, ...]:
    """Compare every measured gauge-group product between adjacent rounds."""

    detectors: list[int] = []
    for previous, current in zip(
        physical.gauge_measurements,
        physical.gauge_measurements[1:],
        strict=False,
    ):
        for group in case.gauge_groups:
            mask = 0
            for gauge_index in group:
                mask ^= 1 << (physical_offset + previous[gauge_index])
                mask ^= 1 << (physical_offset + current[gauge_index])
            detectors.append(mask)
    return tuple(detectors)


def _row_coefficients(rows: Sequence[int], target: int) -> int:
    """Solve ``xor(rows[j] for j in support(c)) == target`` over GF(2)."""

    pivots: dict[int, tuple[int, int]] = {}
    for index, source in enumerate(rows):
        row = int(source)
        coefficients = 1 << index
        while row:
            pivot = row.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = (row, coefficients)
                break
            row ^= existing[0]
            coefficients ^= existing[1]
    row = int(target)
    coefficients = 0
    while row:
        pivot = row.bit_length() - 1
        existing = pivots.get(pivot)
        if existing is None:
            raise ValueError("target is not in the supplied row span")
        row ^= existing[0]
        coefficients ^= existing[1]
    return coefficients


def _minimum_row_coefficients(rows: Sequence[int], target: int) -> int:
    """Return a minimum-cardinality representation of ``target`` by ``rows``."""

    pivots: dict[int, tuple[int, int]] = {}
    relations: list[int] = []
    for index, source in enumerate(rows):
        row = int(source)
        coefficients = 1 << index
        while row:
            pivot = row.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = (row, coefficients)
                break
            row ^= existing[0]
            coefficients ^= existing[1]
        if not row:
            relations.append(coefficients)

    row = int(target)
    coefficients = 0
    while row:
        pivot = row.bit_length() - 1
        existing = pivots.get(pivot)
        if existing is None:
            raise ValueError("target is not in the supplied row span")
        row ^= existing[0]
        coefficients ^= existing[1]

    if len(relations) <= 18:
        coefficients = min(
            (
                coefficients ^ _xor_selected(mask, relations)
                for mask in range(1 << len(relations))
            ),
            key=lambda value: (value.bit_count(), value),
        )
    return coefficients


def _xor_selected(mask: int, rows: Sequence[int]) -> int:
    result = 0
    for index, row in enumerate(rows):
        if (int(mask) >> index) & 1:
            result ^= int(row)
    return result


def _blockwise_center_coefficients(
    *,
    center_row: int,
    types: Sequence[str],
    case: GenericQuiltCircuitCase,
    refs: GenericReferenceLayout,
    stabilizers: Sequence[PauliTerm],
) -> int:
    """Realise a blockwise-CSS row using the fewest raw checks per sector."""

    total_n = case.subsystem.n
    data_offsets = _block_data_offsets(case)
    result = 0
    for block_index, (block, sector, data_offset) in enumerate(
        zip(case.blocks, types, data_offsets, strict=True)
    ):
        if sector == "I":
            continue
        if sector not in {"X", "Z"}:
            raise ValueError("a blockwise centre row has a mixed restriction")
        mask = (1 << int(block.n_data)) - 1
        local_x = (x_part(int(center_row), total_n) >> data_offset) & mask
        local_z = (z_part(int(center_row), total_n) >> data_offset) & mask
        component = pauli(
            local_x << data_offset,
            local_z << data_offset,
            total_n,
        )
        circuit_component = data_row_to_circuit(
            component,
            case=case,
            num_qubits=refs.num_qubits,
        )
        selected_indices = [
            index
            for index, term in enumerate(stabilizers)
            if term.label.startswith(f"raw_{sector}_b{block_index}_")
        ]
        selected_rows = [stabilizers[index].row for index in selected_indices]
        local_coefficients = _minimum_row_coefficients(
            selected_rows,
            circuit_component,
        )
        for local_index, global_index in enumerate(selected_indices):
            if (local_coefficients >> local_index) & 1:
                result ^= 1 << global_index
    return result


@dataclass(frozen=True)
class SparseRawCenterGenerator:
    row: int
    coefficients: int
    block_types: tuple[str, ...]


def _sec_frame_block_types(
    row: int,
    *,
    case: GenericQuiltCircuitCase,
) -> tuple[str, ...]:
    offsets = _block_data_offsets(case)
    total_n = case.subsystem.n
    result: list[str] = []
    for block, offset in zip(case.blocks, offsets, strict=True):
        n = int(block.n_data)
        mask = (1 << n) - 1
        local = pauli(
            (x_part(int(row), total_n) >> offset) & mask,
            (z_part(int(row), total_n) >> offset) & mask,
            n,
        )
        local = int(block.to_sec_frame_pauli_row(local))
        local_x = x_part(local, n)
        local_z = z_part(local, n)
        result.append(
            "X"
            if local_x and not local_z
            else "Z"
            if local_z and not local_x
            else "M"
            if local_x or local_z
            else "I"
        )
    return tuple(result)


def research_glued_center_generators(
    *,
    case: GenericQuiltCircuitCase,
    stabilizers: Sequence[PauliTerm],
    presentation_seed: int = 0,
) -> tuple[SparseRawCenterGenerator, ...]:
    """The published basis, as circuit generators.

    This is the selection behind the sparsity and congestion figures, run on
    the builder's own rows so that the circuit is built from the basis that was
    measured.
    It differs from `sparse_raw_center_generators` in three ways, all of which
    matter for reproducing those figures:

      * the pool is the glued interface graph alone -- free checks, Horton
        cycles, minimum T-joins -- with no pairwise-section or syndrome-map
        kernel candidates, which would make it a strict superset;
      * selection is weight-first with no congestion tie-break, matching
        `select(pool, required, False)`;
      * a mixed restriction is not penalised in the ordering.  Demoting them
        changes which generator wins a weight tie, and the published basis
        does not demote them.
    """

    total_n = case.subsystem.n
    offsets = _block_data_offsets(case)
    check_lookup = [
        {check.label: check for check in block.checks} for block in case.blocks
    ]
    algebra_rows: list[int] = []
    syndromes: list[int] = []
    for term in stabilizers:
        _raw, _declared, block_label, check_label = term.label.split("_", 3)
        block_index = int(block_label.removeprefix("b"))
        block = case.blocks[block_index]
        local = int(check_lookup[block_index][check_label].pauli_row)
        row = pauli(
            x_part(local, int(block.n_data)) << offsets[block_index],
            z_part(local, int(block.n_data)) << offsets[block_index],
            total_n,
        )
        algebra_rows.append(row)
        syndromes.append(
            sum(
                symplectic_product(row, gauge, total_n) << gauge_index
                for gauge_index, gauge in enumerate(case.added_gauges)
            )
        )

    reliable = _subspace_intersection(
        case.subsystem.base_stabilizers,
        case.subsystem.center_basis,
        2 * total_n,
    )
    required = len(row_reduce(tuple(reliable), 2 * total_n)[0])

    pool = glued_center_candidates(
        rows=algebra_rows, syndromes=syndromes, pauli_width=2 * total_n
    )
    # weight, then number of source checks, then the row itself: the same
    # ordering the published selection uses, and deterministic
    ordered = sorted(
        pool,
        key=lambda item: (
            pauli_weight(item[1], total_n),
            item[0].bit_count(),
            item[1],
        ),
    )
    # Alternate presentations come from the ties in that ordering, and only
    # from the ties.  A weight class routinely holds more independent
    # candidates than the greedy has slots for -- 96 candidates for 13 slots is
    # typical at the heavy end -- so permuting within a class selects a
    # different basis for the same span.  Because the permutation never crosses
    # a class boundary, the selection is still a minimum-weight matroid greedy
    # and the weight profile is identical for every seed; only which rows win
    # their ties changes, and with them the detectors, and with those the DEM.
    # Seed 0 is the published order, untouched.
    if presentation_seed:
        import random

        rng = random.Random(presentation_seed)
        classes: dict[tuple[int, int], list] = {}
        for item in ordered:
            classes.setdefault(
                (pauli_weight(item[1], total_n), item[0].bit_count()), []
            ).append(item)
        ordered = []
        for key in sorted(classes):
            members = classes[key]
            rng.shuffle(members)
            ordered.extend(members)
    span: list[int] = []
    chosen: list[SparseRawCenterGenerator] = []
    seen: set[int] = set()
    for coefficients, row in ordered:
        if not row or row in seen:
            continue
        seen.add(row)
        if not reduce_with_basis(span, row):
            continue
        span = list(row_reduce((*span, row), 2 * total_n)[0])
        chosen.append(
            SparseRawCenterGenerator(
                row=row,
                coefficients=coefficients,
                block_types=_sec_frame_block_types(row, case=case),
            )
        )
        if len(chosen) == required:
            break
    if len(chosen) != required:
        raise ValueError(
            "the glued pool does not span the inherited centre "
            f"({len(chosen)} of {required}); this record needs the wider pool"
        )
    return tuple(chosen)


def sparse_raw_center_generators(
    *,
    case: GenericQuiltCircuitCase,
    stabilizers: Sequence[PauliTerm],
    section_variant: int = 0,
    allow_mixed_restrictions: bool = True,
) -> tuple[SparseRawCenterGenerator, ...]:
    """Construct a sparse inherited-centre basis from the supplied SEC checks.

    ``allow_mixed_restrictions`` admits generators carrying both sectors on a
    single block.  They are markedly sparser, and they are admitted by
    default: excluding them is not merely a sparsity cost but a feasibility
    one, since for a mixed-logical target the inherited centre frequently has
    no blockwise-CSS generating set at all.  Such a generator's detector is
    built exactly like any other; only its routing label differs, and the
    blockwise transform skips unlabelled detectors by design.
    """

    total_n = case.subsystem.n
    offsets = _block_data_offsets(case)
    check_lookup = [
        {check.label: check for check in block.checks} for block in case.blocks
    ]
    algebra_rows: list[int] = []
    syndromes: list[int] = []
    component_indices: dict[tuple[int, str], list[int]] = {}
    for index, term in enumerate(stabilizers):
        _raw, _declared_sector, block_label, check_label = term.label.split("_", 3)
        block_index = int(block_label.removeprefix("b"))
        block = case.blocks[block_index]
        check = check_lookup[block_index][check_label]
        local = int(check.pauli_row)
        row = pauli(
            x_part(local, int(block.n_data)) << offsets[block_index],
            z_part(local, int(block.n_data)) << offsets[block_index],
            total_n,
        )
        sec_local = int(block.to_sec_frame_pauli_row(local))
        sec_x = x_part(sec_local, int(block.n_data))
        sec_z = z_part(sec_local, int(block.n_data))
        sector = (
            "X"
            if sec_x and not sec_z
            else "Z"
            if sec_z and not sec_x
            else "M"
        )
        if sector == "M":
            raise ValueError("a supplied check is mixed even in its SEC frame")
        algebra_rows.append(row)
        syndromes.append(
            sum(
                symplectic_product(row, gauge, total_n) << gauge_index
                for gauge_index, gauge in enumerate(case.added_gauges)
            )
        )
        component_indices.setdefault((block_index, sector), []).append(index)

    @dataclass(frozen=True)
    class Component:
        block_index: int
        sector: str
        indices: tuple[int, ...]
        syndrome_map: _SparseSyndromeMap

    components: list[Component] = []
    candidates: list[SparseRawCenterGenerator] = []
    for (block_index, sector), indices_list in sorted(component_indices.items()):
        indices = tuple(indices_list)
        syndrome_map = _SparseSyndromeMap(
            checks=tuple(algebra_rows[index] for index in indices),
            syndromes=tuple(syndromes[index] for index in indices),
            pauli_width=2 * total_n,
            syndrome_width=len(case.added_gauges),
        )
        components.append(Component(block_index, sector, indices, syndrome_map))
        for local_coefficients in syndrome_map.kernel_coefficients:
            coefficients = sum(
                1 << indices[local_index]
                for local_index in range(len(indices))
                if (local_coefficients >> local_index) & 1
            )
            row = _xor_selected(coefficients, algebra_rows)
            if row:
                candidates.append(
                    SparseRawCenterGenerator(
                        row=row,
                        coefficients=coefficients,
                        block_types=_sec_frame_block_types(row, case=case),
                    )
                )

    syndrome_width = len(case.added_gauges)
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            if left.block_index == right.block_index:
                continue
            intersection = _subspace_intersection(
                left.syndrome_map.image_basis,
                right.syndrome_map.image_basis,
                syndrome_width,
            )
            if not intersection:
                continue
            if len(intersection) <= 16:
                syndrome_candidates = [0]
                for basis_row in intersection:
                    syndrome_candidates += [
                        value ^ basis_row for value in syndrome_candidates
                    ]
                syndrome_candidates = syndrome_candidates[1:]
            else:
                intersection_basis = row_reduce(intersection, syndrome_width)[0]
                syndrome_candidates = list(intersection_basis)
                for component in (left, right):
                    syndrome_candidates.extend(
                        syndrome
                        for syndrome in component.syndrome_map.syndromes
                        if syndrome
                        and not reduce_with_basis(intersection_basis, syndrome)
                    )

            pair_candidates: list[tuple[int, SparseRawCenterGenerator]] = []
            for syndrome in set(syndrome_candidates):
                left_row, left_coefficients = left.syndrome_map.section(syndrome)
                right_row, right_coefficients = right.syndrome_map.section(syndrome)
                coefficients = sum(
                    1 << left.indices[local_index]
                    for local_index in range(len(left.indices))
                    if (left_coefficients >> local_index) & 1
                ) ^ sum(
                    1 << right.indices[local_index]
                    for local_index in range(len(right.indices))
                    if (right_coefficients >> local_index) & 1
                )
                row = left_row ^ right_row
                pair_candidates.append(
                    (
                        syndrome,
                        SparseRawCenterGenerator(
                            row=row,
                            coefficients=coefficients,
                            block_types=_sec_frame_block_types(row, case=case),
                        ),
                    )
                )
            pair_span: list[int] = []
            for syndrome, generator in sorted(
                pair_candidates,
                key=lambda item: (
                    pauli_weight(item[1].row, total_n),
                    item[1].coefficients.bit_count(),
                    item[0],
                ),
            ):
                if reduce_with_basis(pair_span, syndrome):
                    candidates.append(generator)
                    pair_span = list(
                        row_reduce((*pair_span, syndrome), syndrome_width)[0]
                    )
                    if len(pair_span) == len(intersection):
                        break

    global_map = _SparseSyndromeMap(
        checks=algebra_rows,
        syndromes=syndromes,
        pauli_width=2 * total_n,
        syndrome_width=syndrome_width,
    )
    for coefficients in global_map.kernel_coefficients:
        row = _xor_selected(coefficients, algebra_rows)
        if row:
            candidates.append(
                SparseRawCenterGenerator(
                    row=row,
                    coefficients=coefficients,
                    block_types=_sec_frame_block_types(row, case=case),
                )
            )

    # Additional candidates from the glued interface graph.  The syndrome
    # coordinates are shared by every block, so a check of syndrome weight two
    # is an edge of one graph spanning the whole quilt and any cycle of that
    # graph is a centre element.  This reaches cross-block generators directly,
    # where the pairwise construction above can only combine two separately
    # chosen sections.  Generators that mix both sectors on a single block are
    # dropped: the blockwise-CSS detector presentation cannot express them.
    for coefficients, row in glued_center_candidates(
        rows=algebra_rows,
        syndromes=syndromes,
        pauli_width=2 * total_n,
    ):
        block_types = _sec_frame_block_types(row, case=case)
        if "M" in block_types and not allow_mixed_restrictions:
            continue
        candidates.append(
            SparseRawCenterGenerator(
                row=row, coefficients=coefficients, block_types=block_types
            )
        )

    reliable = _subspace_intersection(
        case.subsystem.base_stabilizers,
        case.subsystem.center_basis,
        2 * total_n,
    )
    chosen: list[SparseRawCenterGenerator] = []
    span: list[int] = []
    unique: dict[int, SparseRawCenterGenerator] = {}
    for generator in candidates:
        incumbent = unique.get(generator.row)
        if incumbent is None or (
            "M" in generator.block_types,
            generator.coefficients.bit_count(),
            generator.coefficients,
        ) < (
            "M" in incumbent.block_types,
            incumbent.coefficients.bit_count(),
            incumbent.coefficients,
        ):
            unique[generator.row] = generator
    # Sorting by weight and taking the first independent generator is the
    # matroid greedy, so the chosen basis already minimises its own maximum
    # weight over the pool.  Ties are then broken by current qubit load: a
    # sparse row basis can still make a poor detector graph if one qubit is
    # reused by many rows, and breaking ties this way costs no weight.
    load: dict[int, int] = {}

    def qubits(row: int) -> tuple[int, ...]:
        mask = (row & ((1 << total_n) - 1)) | (row >> total_n)
        out: list[int] = []
        while mask:
            out.append((mask & -mask).bit_length() - 1)
            mask &= mask - 1
        return tuple(out)

    buckets: dict[tuple[bool, int, int], list[SparseRawCenterGenerator]] = {}
    for generator in unique.values():
        buckets.setdefault(
            (
                "M" in generator.block_types,
                pauli_weight(generator.row, total_n),
                generator.coefficients.bit_count(),
            ),
            [],
        ).append(generator)
    for key in sorted(buckets):
        ordered = sorted(
            buckets[key],
            key=lambda item: (
                sum(load.get(qubit, 0) for qubit in qubits(item.row)),
                item.row,
            ),
        )
        for generator in ordered:
            if reduce_with_basis(span, generator.row):
                chosen.append(generator)
                span = list(row_reduce((*span, generator.row), 2 * total_n)[0])
                for qubit in qubits(generator.row):
                    load[qubit] = load.get(qubit, 0) + 1
                if len(chosen) == len(reliable):
                    break
        if len(chosen) == len(reliable):
            break
    if row_reduce((item.row for item in chosen), 2 * total_n)[0] != row_reduce(
        reliable, 2 * total_n
    )[0]:
        raise AssertionError("the raw sparse generators do not span the reliable centre")
    baseline = tuple(chosen)
    if section_variant == 0:
        return baseline
    if section_variant < 0:
        raise ValueError("section_variant must be nonnegative")

    # Prototype alternate sections. Multiplying a cross-block centre generator
    # by a block-local kernel generator changes one local section without
    # changing its stitch syndrome. Keep both row weight and basis-column
    # congestion bounded: a sparse row basis can still make a poor detector
    # graph if one source generator is reused by many rows.
    max_pauli_weight = max(pauli_weight(item.row, total_n) for item in baseline)
    max_check_count = max(item.coefficients.bit_count() for item in baseline)
    local_indices = [
        index
        for index, item in enumerate(baseline)
        if sum(kind != "I" for kind in item.block_types) == 1
    ]
    cross_indices = [
        index
        for index, item in enumerate(baseline)
        if sum(kind != "I" for kind in item.block_types) >= 2
    ]

    def variant_key(target: int, source: int) -> int:
        value = (
            int(section_variant) * 0x9E3779B97F4A7C15
            + int(target) * 0xBF58476D1CE4E5B9
            + int(source) * 0x94D049BB133111EB
        ) & ((1 << 64) - 1)
        value ^= value >> 30
        value = (value * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
        value ^= value >> 27
        return value

    result = list(baseline)
    coordinates = [1 << index for index in range(len(baseline))]

    # Keep the canonical local-kernel basis. The section freedom lives in the
    # cross-block rows, and changing local rows as well needlessly perturbs the
    # sparse detector bulk.

    options: dict[int, list[tuple[int, SparseRawCenterGenerator, int]]] = {}
    for target_index in cross_indices:
        target = baseline[target_index]
        for source_index in local_indices:
            source = result[source_index]
            source_blocks = [
                index
                for index, kind in enumerate(source.block_types)
                if kind != "I"
            ]
            block_index = source_blocks[0]
            if target.block_types[block_index] != source.block_types[block_index]:
                continue
            row = target.row ^ source.row
            coefficients = target.coefficients ^ source.coefficients
            block_types = _sec_frame_block_types(row, case=case)
            if "M" in block_types:
                continue
            if pauli_weight(row, total_n) > max_pauli_weight:
                continue
            if coefficients.bit_count() > max_check_count:
                continue
            options.setdefault(target_index, []).append(
                (
                    source_index,
                    SparseRawCenterGenerator(
                        row=row,
                        coefficients=coefficients,
                        block_types=block_types,
                    ),
                    coordinates[target_index] ^ coordinates[source_index],
                )
            )

    eligible = sorted(options, key=lambda target: variant_key(target, target))
    if not eligible:
        raise ValueError("no bounded alternate sections exist for this case")
    column_loads = [0] * len(baseline)
    for coordinate_row in coordinates:
        for index in range(len(baseline)):
            if (coordinate_row >> index) & 1:
                column_loads[index] += 1
    used_sources: set[int] = set()
    for target_index in eligible:
        choices = [
            item
            for item in options[target_index]
            if item[0] not in used_sources
            and all(
                column_loads[index] < 3
                for index in range(len(baseline))
                if (coordinates[item[0]] >> index) & 1
            )
        ]
        if not choices:
            continue
        choices.sort(
            key=lambda item: (
                pauli_weight(item[1].row, total_n),
                item[1].coefficients.bit_count(),
                variant_key(target_index, item[0]),
            )
        )
        best_weight = pauli_weight(choices[0][1].row, total_n)
        best_check_count = min(
            item[1].coefficients.bit_count()
            for item in choices
            if pauli_weight(item[1].row, total_n) == best_weight
        )
        near_optimal = [
            item
            for item in choices
            if pauli_weight(item[1].row, total_n) == best_weight
            and item[1].coefficients.bit_count() == best_check_count
        ]
        source_index, replacement, replacement_coordinates = min(
            near_optimal,
            key=lambda item: variant_key(target_index, item[0]),
        )
        result[target_index] = replacement
        coordinates[target_index] = replacement_coordinates
        used_sources.add(source_index)
        for index in range(len(baseline)):
            if (coordinates[source_index] >> index) & 1:
                column_loads[index] += 1

    if max(column_loads) > 3:
        raise AssertionError("alternate section basis exceeded column congestion 3")

    if row_reduce((item.row for item in result), 2 * total_n)[0] != row_reduce(
        reliable, 2 * total_n
    )[0]:
        raise AssertionError("an alternate section basis changed the reliable centre")
    return tuple(result)


def sparse_raw_check_relations(
    stabilizers: Sequence[PauliTerm],
    *,
    num_qubits: int,
) -> tuple[int, ...]:
    """Return a sparse basis of products of supplied checks equal to identity."""

    equations = [
        sum(
            ((int(term.row) >> coordinate) & 1) << index
            for index, term in enumerate(stabilizers)
        )
        for coordinate in range(2 * num_qubits)
    ]
    relations = tuple(nullspace(equations, len(stabilizers)))
    candidates = list(relations)
    if len(relations) <= 18:
        candidates = [
            _xor_selected(mask, relations)
            for mask in range(1, 1 << len(relations))
        ]
    chosen: list[int] = []
    span: list[int] = []
    for relation in sorted(set(candidates), key=lambda value: (value.bit_count(), value)):
        if reduce_with_basis(span, relation):
            chosen.append(relation)
            span = list(row_reduce((*span, relation), len(stabilizers))[0])
            if len(chosen) == len(relations):
                break
    if len(chosen) != len(relations):
        raise AssertionError("failed to span the raw check relations")
    return tuple(chosen)


def preferred_raw_check_relation_detectors(
    *,
    stabilizers: Sequence[PauliTerm],
    physical: GenericPhysicalWindow,
    physical_offset: int,
    boundary_records: dict[str, dict[str, int]],
    num_qubits: int,
) -> tuple[int, ...]:
    """Measure every raw-check identity relation independently in each round."""

    relations = sparse_raw_check_relations(
        stabilizers,
        num_qubits=num_qubits,
    )
    record_sets = [
        tuple(boundary_records["initial_stabs"][term.label] for term in stabilizers),
        *(
            tuple(physical_offset + record for record in records)
            for records in physical.check_measurements
        ),
        tuple(boundary_records["final_stabs"][term.label] for term in stabilizers),
    ]
    return tuple(
        sum(
            1 << records[index]
            for index in range(len(stabilizers))
            if (relation >> index) & 1
        )
        for records in record_sets
        for relation in relations
    )


def preferred_blockwise_center_history_detectors(
    *,
    case: GenericQuiltCircuitCase,
    refs: GenericReferenceLayout,
    stabilizers: Sequence[PauliTerm],
    physical: GenericPhysicalWindow,
    physical_offset: int,
    boundary_records: dict[str, dict[str, int]] | None = None,
    preserve_raw_checks: bool = False,
    section_variant: int = 0,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Temporal detectors from a blockwise-CSS basis of the reliable centre."""

    checks = physical.check_measurements
    if any(len(round_records) != len(stabilizers) for round_records in checks):
        raise ValueError("raw check ledger does not match the SEC rounds")
    algebra_blocks = tuple(
        StabilizerBlock(
            name=str(block.name),
            n=int(block.n_data),
            stabilizers=tuple(int(check.pauli_row) for check in block.checks),
        )
        for block in case.blocks
    )
    reliable = _subspace_intersection(
        case.subsystem.base_stabilizers,
        case.subsystem.center_basis,
        2 * case.subsystem.n,
    )
    if preserve_raw_checks:
        # Build from the glued basis, so the circuit's generators are the ones
        # whose sparsity and congestion were measured and plotted.
        selected_generators = research_glued_center_generators(
            case=case,
            stabilizers=stabilizers,
            presentation_seed=section_variant,
        )
    else:
        try:
            sparse_generators = sparse_n2_blockwise_css_center_basis(
                blocks=algebra_blocks,
                stitching_rows=case.added_gauges,
            )
            center = tuple(generator.row for generator in sparse_generators)
            if row_reduce(center, 2 * case.subsystem.n)[0] != row_reduce(
                reliable,
                2 * case.subsystem.n,
            )[0]:
                raise AssertionError(
                    "the sparse transport basis does not span the reliable centre"
                )
        except ValueError:
            center = blockwise_css_basis(reliable, algebra_blocks)
        selected_generators = tuple(
            SparseRawCenterGenerator(
                row=center_row,
                coefficients=_blockwise_center_coefficients(
                    center_row=center_row,
                    types=blockwise_pauli_types(center_row, algebra_blocks),
                    case=case,
                    refs=refs,
                    stabilizers=stabilizers,
                ),
                block_types=blockwise_pauli_types(center_row, algebra_blocks),
            )
            for center_row in center
        )
    result: list[tuple[int, tuple[str, ...]]] = []
    for generator in selected_generators:
        center_row = generator.row
        types = generator.block_types
        # A mixed restriction is emitted, not refused.  The detector mask
        # below is built from `coefficients` alone -- the XOR of the
        # constituent source checks' measurement records across two rounds --
        # and `types` is carried only as routing metadata.  So nothing about
        # the circuit requires a row to be purely X or purely Z on each block,
        # and requiring it would make the generic cohorts unbuildable,
        # because their inherited centre often admits no blockwise-CSS
        # generating set.  A mixed row is labelled unroutable downstream and
        # the transform leaves it alone, which is the correct behaviour.
        coefficients = generator.coefficients
        history_pairs: list[tuple[Sequence[int], Sequence[int]]] = [
            (
                tuple(physical_offset + record for record in previous),
                tuple(physical_offset + record for record in current),
            )
            for previous, current in zip(checks, checks[1:], strict=False)
        ]
        if boundary_records is not None:
            initial = tuple(
                boundary_records["initial_stabs"][term.label]
                for term in stabilizers
            )
            final = tuple(
                boundary_records["final_stabs"][term.label]
                for term in stabilizers
            )
            history_pairs.insert(0, (initial, tuple(physical_offset + r for r in checks[0])))
            history_pairs.append((tuple(physical_offset + r for r in checks[-1]), final))
        for previous, current in history_pairs:
            mask = 0
            for index, (left, right) in enumerate(
                zip(previous, current, strict=True)
            ):
                if (coefficients >> index) & 1:
                    mask ^= 1 << left
                    mask ^= 1 << right
            result.append((mask, types))
    return tuple(result)


def label_blockwise_css_detectors(
    *,
    detectors: Sequence[int],
    case: GenericQuiltCircuitCase,
    stabilizers: Sequence[PauliTerm],
    physical: GenericPhysicalWindow,
    physical_offset: int,
    boundary_records: dict[str, dict[str, int]] | None = None,
) -> tuple[tuple[str, ...] | None, ...]:
    """Label sparse emitted detectors by their block-local CSS restrictions."""

    data_offsets = _block_data_offsets(case)
    algebra_rows_by_label: dict[str, int] = {}
    for block_index, (block, data_offset) in enumerate(
        zip(case.blocks, data_offsets, strict=True)
    ):
        for check in block.checks:
            local = int(check.pauli_row)
            algebra_rows_by_label[
                f"raw_{check.sector}_b{block_index}_{check.label}"
            ] = pauli(
                x_part(local, int(block.n_data)) << data_offset,
                z_part(local, int(block.n_data)) << data_offset,
                case.subsystem.n,
            )
    algebra_rows = tuple(
        algebra_rows_by_label[term.label] for term in stabilizers
    )
    check_sec_types = tuple(
        next(
            sector
            for sector in _sec_frame_block_types(row, case=case)
            if sector != "I"
        )
        for row in algebra_rows
    )
    reliable = _subspace_intersection(
        case.subsystem.base_stabilizers,
        case.subsystem.center_basis,
        2 * case.subsystem.n,
    )

    check_records: dict[int, tuple[int, int]] = {}
    for round_index, records in enumerate(physical.check_measurements):
        for check_index, record in enumerate(records):
            check_records[physical_offset + record] = (round_index, check_index)
    if boundary_records is not None:
        for check_index, term in enumerate(stabilizers):
            check_records[boundary_records["initial_stabs"][term.label]] = (
                -1,
                check_index,
            )
            check_records[boundary_records["final_stabs"][term.label]] = (
                len(physical.check_measurements),
                check_index,
            )
    gauge_records = {
        physical_offset + record
        for records in physical.gauge_measurements
        for record in records
    }

    result: list[tuple[str, ...] | None] = []
    for detector in map(int, detectors):
        selected_measurements: list[int] = []
        remaining = detector
        while remaining:
            least_bit = remaining & -remaining
            selected_measurements.append(least_bit.bit_length() - 1)
            remaining ^= least_bit
        if any(measurement in gauge_records for measurement in selected_measurements):
            result.append(None)
            continue
        checks_by_round: dict[int, int] = {}
        check_indices: list[int] = []
        for measurement in selected_measurements:
            check_record = check_records.get(measurement)
            if check_record is None:
                continue
            round_index, check_index = check_record
            check_indices.append(check_index)
            checks_by_round[round_index] = (
                checks_by_round.get(round_index, 0) ^ algebra_rows[check_index]
            )
        rows = {row for row in checks_by_round.values() if row}
        if not rows and check_indices:
            identities = {
                (
                    check_sec_types[index],
                    stabilizers[index].label.split("_", 3)[2],
                )
                for index in check_indices
            }
            if len(identities) == 1:
                sector, block_label = identities.pop()
                block_index = int(block_label.removeprefix("b"))
                types = ["I"] * len(case.blocks)
                types[block_index] = sector
                result.append(tuple(types))
                continue
        if len(rows) != 1:
            result.append(None)
            continue
        row = rows.pop()
        if reduce_with_basis(reliable, row, 2 * case.subsystem.n):
            result.append(None)
            continue
        types = _sec_frame_block_types(row, case=case)
        result.append(types if "M" not in types else None)
    return tuple(result)


def build_circuit(
    case: GenericQuiltCircuitCase,
    *,
    stitch_rounds: int,
    noise: NoiseConfig,
    prefer_blockwise_center_history: bool = False,
    preserve_raw_center_checks: bool = False,
    center_section_variant: int = 0,
    compress: bool = False,
) -> tuple[str, dict[str, Any]]:
    refs = references(case)
    boundary = boundary_terms(case, refs)
    raw_stabilizers = raw_stabilizer_terms(case, refs)
    if row_reduce(term.row for term in raw_stabilizers) != row_reduce(
        term.row for term in boundary.stabilizers
    ):
        raise AssertionError("raw SEC checks do not span the declared base stabilizers")
    zero = NoiseConfig(0, 0, 0, 0, 0, 0)
    ideal_window = physical_window(case, stitch_rounds=stitch_rounds, noise=zero)
    ideal_physical = stim.Circuit(ideal_window.text)
    raw_ideal, records, physical_offset = append_raw_experiment(
        physical=ideal_window,
        refs=refs,
        boundary=boundary,
        stabilizers=raw_stabilizers,
    )
    observables, observable_metadata = observable_masks(
        case=case,
        ideal_physical=ideal_physical,
        records=records,
        physical_offset=physical_offset,
        boundary=boundary,
        physical=ideal_window,
    )
    gauge_audit = _audit_observable_records(
        case=case,
        observables=observables,
        metadata=observable_metadata,
        physical=ideal_window,
        physical_offset=physical_offset,
    )
    raw_ideal_circuit = stim.Circuit(raw_ideal)
    deterministic = deterministic_measurement_basis(raw_ideal_circuit)
    sparse = sparse_deterministic_candidates(raw_ideal_circuit)
    preferred_gauge_history = preferred_gauge_history_detectors(
        case=case,
        physical=ideal_window,
        physical_offset=physical_offset,
    )
    preferred_blockwise_center = (
        preferred_blockwise_center_history_detectors(
            case=case,
            refs=refs,
            stabilizers=raw_stabilizers,
            physical=ideal_window,
            physical_offset=physical_offset,
            boundary_records=records if preserve_raw_center_checks else None,
            preserve_raw_checks=preserve_raw_center_checks,
            section_variant=center_section_variant,
        )
        if prefer_blockwise_center_history
        else ()
    )
    preferred_raw_relations = (
        preferred_raw_check_relation_detectors(
            stabilizers=raw_stabilizers,
            physical=ideal_window,
            physical_offset=physical_offset,
            boundary_records=records,
            num_qubits=refs.num_qubits,
        )
        if preserve_raw_center_checks
        else ()
    )
    preferred_detectors = (
        *preferred_gauge_history,
        *preferred_raw_relations,
        *(mask for mask, _types in preferred_blockwise_center),
    )
    detectors = complete_detector_set(
        deterministic,
        observables,
        preferred_detectors,
        sparse,
        raw_ideal_circuit.num_measurements,
    )
    detector_block_types = label_blockwise_css_detectors(
        detectors=detectors,
        case=case,
        stabilizers=raw_stabilizers,
        physical=ideal_window,
        physical_offset=physical_offset,
        boundary_records=records,
    )

    actual_window = physical_window(case, stitch_rounds=stitch_rounds, noise=noise)
    raw_actual, actual_records, actual_offset = append_raw_experiment(
        physical=actual_window,
        refs=refs,
        boundary=boundary,
        stabilizers=raw_stabilizers,
    )
    if actual_records != records or actual_offset != physical_offset:
        raise AssertionError("noise changed the measurement ledger")
    labels = [str(row["label"]) for row in observable_metadata]
    # One coordinate per detector naming its CSS sector, so a decoder can be
    # restricted to the X or Z half without re-deriving the classification.
    _bases = measurement_bases(stim.Circuit(raw_actual))
    detector_sectors = tuple(
        detector_sector_from_basis(measurement_indices(d), _bases) for d in detectors
    )
    noisy_text = append_detectors_and_observables(
        raw_actual,
        detectors=detectors,
        observables=observables,
        observable_labels=labels,
        detector_sectors=detector_sectors,
    )
    ideal_text = append_detectors_and_observables(
        raw_ideal,
        detectors=detectors,
        observables=observables,
        observable_labels=labels,
        detector_sectors=detector_sectors,
    )
    if compress:
        from ._pack import pack

        noisy_text = str(pack(stim.Circuit(noisy_text))) + "\n"
    circuit = append_idle_noise(stim.Circuit(noisy_text), noise.p_idle)
    # The idle pass rewrites the circuit, so the emitted text has to come from
    # it rather than from noisy_text; otherwise every downstream check runs on
    # the idle-noise circuit while the file on disk has none.
    if noise.p_idle > 0:
        noisy_text = str(circuit) + "\n"
    ideal = stim.Circuit(ideal_text)
    samples, observable_samples = ideal.compile_detector_sampler().sample(
        256, separate_observables=True
    )
    if samples.any() or observable_samples.any():
        raise AssertionError("noiseless generic circuit has an operational flip")
    if compress:
        samples, observable_samples = circuit.without_noise().compile_detector_sampler().sample(
            256, separate_observables=True
        )
        if samples.any() or observable_samples.any():
            raise AssertionError("noiseless packed circuit has an operational flip")
    dem = circuit.detector_error_model(
        decompose_errors=False, allow_gauge_detectors=False
    )
    dem_audit = dem_logical_support_audit(dem)
    if dem_audit["logical_fault_mechanisms_without_detector"]:
        raise AssertionError("a single DEM fault flips an observable undetected")
    if len(detectors) != len(deterministic) - len(observables):
        raise AssertionError("detector/observable rank does not exhaust determinism")
    summary = {
        "name": case.name,
        "schema": "quilt8_generic/circuit_summary/v1",
        "blocks": [str(block.name) for block in case.blocks],
        "N": len(case.blocks),
        "stitch_rounds": stitch_rounds,
        "rounds": stitch_rounds + 2,
        "noise": asdict(noise),
        "subsystem": {
            "n": case.subsystem.n,
            "gauge_rank": len(case.subsystem.gauge_basis),
            "center_rank": len(case.subsystem.center_basis),
            "gauge_qubits": case.subsystem.gauge_qubits,
            "logical_qubits": case.subsystem.logical_qubits,
        },
        "measurement": {
            "rank": case.measurement_analysis.measured_rank,
            "exact": case.measurement_analysis.exact,
            "valid": case.measurement_analysis.valid,
            "targets": list(case.target_labels),
        },
        "num_total_qubits": refs.num_qubits,
        "num_reference_qubits": len(refs.protected) + len(refs.measured),
        "num_measurements": circuit.num_measurements,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_deterministic_parities": len(deterministic),
        "detector_plus_observable_rank": len(detectors) + len(observables),
        "observable_basis": observable_metadata,
        "observable_gauge_record_audit": gauge_audit,
        "preferred_blockwise_center_history_candidate_count": len(
            preferred_blockwise_center
        ),
        "center_section_variant": center_section_variant,
        "preferred_raw_check_relation_candidate_count": len(
            preferred_raw_relations
        ),
        "selected_blockwise_center_history_detector_count": sum(
            types is not None for types in detector_block_types
        ),
        "detector_block_types": detector_block_types,
        "preferred_gauge_history_detector_count": len(preferred_gauge_history),
        "record_ledger": records,
        "physical_round_kinds": list(actual_window.round_kinds),
        "physical_check_record_counts": [
            len(records) for records in actual_window.check_measurements
        ],
        "physical_gauge_record_counts": [
            len(records) for records in actual_window.gauge_measurements
        ],
        **dem_audit,
    }
    return noisy_text, summary
