"""Per-block syndrome-extraction schedules.

Each block type -- bivariate-bicycle, rotated surface -- supplies its own gate
layers, reset and measure sets, and a frame map. Alternate rounds run the
schedule reversed, so hook errors do not accumulate in one direction.
`to_sec_frame_pauli_row` converts a physical-frame Pauli into the callback
frame, which is where a block's local Clifford (if any) is accounted for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol, Sequence

from ._core import (
    BB_SEC_SCHEDULE_LEGACY,
    BB_SPECS,
    BBLayout,
    StimTextBuilder,
    append_noise_after_cx,
    append_noise_after_reset,
    append_noise_before_measure,
)
from ._surface import (
    build_rotated_surface,
    rotated_surface_local_checks,
)
from ._core import NoiseConfig


ReadoutCallback = Callable[[StimTextBuilder], None]


class CheckSpec(NamedTuple):
    sector: str
    row: int
    label: str

    @property
    def pauli_row(self) -> int:
        return self.row


class ReadoutSpec(NamedTuple):
    target: int
    check_label: str


class GateLayer(NamedTuple):
    gate: str
    targets: tuple[int, ...]


@dataclass(frozen=True)
class SecRoundSchedule:
    """Local-qubit schedule for one syndrome-extraction round.

    A multi-block orchestrator can batch the corresponding phases from several
    schedules. CNOT schedules with fewer epochs are padded with empty epochs by
    the orchestrator. Readout order is always Z checks followed by X checks.
    """

    pre_data_gates: tuple[GateLayer, ...]
    x_reset_targets: tuple[int, ...]
    z_reset_targets: tuple[int, ...]
    cnot_epochs: tuple[tuple[tuple[int, int], ...], ...]
    z_readout: tuple[ReadoutSpec, ...]
    x_readout: tuple[ReadoutSpec, ...]
    post_data_gates: tuple[GateLayer, ...]

    @property
    def readout(self) -> tuple[ReadoutSpec, ...]:
        return self.z_readout + self.x_readout

    @property
    def reset_x(self) -> tuple[int, ...]:
        return self.x_reset_targets

    @property
    def reset_z(self) -> tuple[int, ...]:
        return self.z_reset_targets

    @property
    def epochs(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return self.cnot_epochs

    @property
    def readout_z(self) -> tuple[ReadoutSpec, ...]:
        return self.z_readout

    @property
    def readout_x(self) -> tuple[ReadoutSpec, ...]:
        return self.x_readout


class SecBlock(Protocol):
    name: str
    n_data: int
    n_total: int
    checks: tuple[CheckSpec, ...]
    data_coords: tuple[tuple[float, float], ...]
    ancilla_coords: tuple[tuple[float, float], ...]

    def round_schedule(self, *, reverse: bool = False) -> SecRoundSchedule:
        ...

    def sec_schedule(self, reverse: bool = False) -> SecRoundSchedule:
        ...

    def to_sec_frame_pauli_row(self, row: int) -> int:
        """Map a physical-frame data Pauli into the SEC callback frame."""

        ...

    def append_sec_round(
        self,
        builder: StimTextBuilder,
        *,
        offset: int,
        round_index: int,
        noise: NoiseConfig,
        reverse: bool,
        before_readout: ReadoutCallback | None = None,
        after_readout: ReadoutCallback | None = None,
    ) -> tuple[int, ...]:
        ...


def _mask(row: Sequence[int]) -> int:
    out = 0
    for q, bit in enumerate(row):
        if int(bit) & 1:
            out |= 1 << q
    return out


def _pauli_weight(row: int, n: int) -> int:
    return ((int(row) & ((1 << n) - 1)) | (int(row) >> n)).bit_count()


def _symplectic_product(left: int, right: int, n: int) -> int:
    mask = (1 << n) - 1
    left_x = int(left) & mask
    left_z = int(left) >> n
    right_x = int(right) & mask
    right_z = int(right) >> n
    return ((left_x & right_z).bit_count() ^ (left_z & right_x).bit_count()) & 1


def _offset_targets(offset: int, targets: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(offset) + int(q) for q in targets)


def _append_depolarize1(
    builder: StimTextBuilder,
    targets: Sequence[int],
    probability: float,
) -> None:
    if probability > 0 and targets:
        builder.gate(f"DEPOLARIZE1({float(probability):.12g})", targets)


def _append_gate_layers(
    builder: StimTextBuilder,
    layers: Sequence[GateLayer],
    *,
    offset: int,
    noise: NoiseConfig,
) -> None:
    for layer in layers:
        targets = _offset_targets(offset, layer.targets)
        builder.gate(layer.gate, targets)
        _append_depolarize1(builder, targets, noise.p_sec_1q)
        builder.tick()


def _validate_schedule(block: SecBlock, schedule: SecRoundSchedule) -> None:
    local_targets = [
        *schedule.x_reset_targets,
        *schedule.z_reset_targets,
        *(target for layer in schedule.pre_data_gates for target in layer.targets),
        *(target for layer in schedule.post_data_gates for target in layer.targets),
        *(readout.target for readout in schedule.readout),
        *(
            target
            for epoch in schedule.cnot_epochs
            for pair in epoch
            for target in pair
        ),
    ]
    if any(q < 0 or q >= block.n_total for q in local_targets):
        raise ValueError("SEC schedule contains a target outside the local block")
    for epoch_index, epoch in enumerate(schedule.cnot_epochs):
        flat = [q for pair in epoch for q in pair]
        if len(flat) != len(set(flat)):
            raise ValueError(f"qubit collision in CNOT epoch {epoch_index}")
    labels = tuple(readout.check_label for readout in schedule.readout)
    if labels != tuple(check.label for check in block.checks):
        raise ValueError("SEC readout order is not aligned to checks")


def _append_sec_round(
    block: SecBlock,
    builder: StimTextBuilder,
    *,
    offset: int,
    round_index: int,
    noise: NoiseConfig,
    reverse: bool,
    before_readout: ReadoutCallback | None,
    after_readout: ReadoutCallback | None,
) -> tuple[int, ...]:
    offset = int(offset)
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    schedule = block.round_schedule(reverse=reverse)
    _validate_schedule(block, schedule)

    direction = "reverse" if reverse else "forward"
    block_index = getattr(block, "block_index", 0)
    builder.line(f"# === block {block_index} round {round_index}: {direction} SEC ===")
    _append_gate_layers(builder, schedule.pre_data_gates, offset=offset, noise=noise)

    x_reset = _offset_targets(offset, schedule.x_reset_targets)
    z_reset = _offset_targets(offset, schedule.z_reset_targets)
    builder.gate("RX", x_reset)
    append_noise_after_reset(builder, x_reset, "X", noise.p_sec_1q)
    builder.gate("R", z_reset)
    append_noise_after_reset(builder, z_reset, "Z", noise.p_sec_1q)
    builder.tick()

    for epoch_index, local_pairs in enumerate(schedule.cnot_epochs, start=1):
        builder.line(f"# SEC CNOT epoch {epoch_index} ({direction})")
        pairs = [
            (offset + int(control), offset + int(target))
            for control, target in local_pairs
        ]
        emitted = builder.pairs("CX", pairs)
        append_noise_after_cx(builder, emitted, noise.p_sec_2q)
        builder.tick()

    if before_readout is not None:
        before_readout(builder)

    builder.line("# batched SEC ancilla readout")
    z_targets = _offset_targets(
        offset, tuple(readout.target for readout in schedule.z_readout)
    )
    x_targets = _offset_targets(
        offset, tuple(readout.target for readout in schedule.x_readout)
    )
    append_noise_before_measure(builder, z_targets, "Z", noise.p_sec_1q)
    z_records = builder.measure("MZ", z_targets)
    append_noise_before_measure(builder, x_targets, "X", noise.p_sec_1q)
    x_records = builder.measure("MX", x_targets)

    if after_readout is not None:
        after_readout(builder)

    _append_gate_layers(builder, schedule.post_data_gates, offset=offset, noise=noise)
    if not schedule.post_data_gates and after_readout is None:
        builder.tick()

    records_by_label = {
        readout.check_label: record
        for readout, record in zip(schedule.readout, [*z_records, *x_records])
    }
    return tuple(records_by_label[check.label] for check in block.checks)


@dataclass(frozen=True)
class BBSecBlock:
    """Bravyi seven-epoch SEC adapter for a bundled BB code."""

    code_name: str
    block_index: int = 0
    schedule_provenance: str = BB_SEC_SCHEDULE_LEGACY

    def __post_init__(self) -> None:
        try:
            spec = BB_SPECS[str(self.code_name)]
        except KeyError as exc:
            raise ValueError(f"unknown bundled BB code {self.code_name!r}") from exc
        layout = BBLayout(spec)
        checks = tuple(
            [
                CheckSpec("Z", int(row) << layout.n_data, f"Z{i}")
                for i, row in enumerate(layout.h_z_rows())
            ]
            + [
                CheckSpec("X", int(row), f"X{i}")
                for i, row in enumerate(layout.h_x_rows())
            ]
        )
        coords = {q: (float(x), float(y)) for q, x, y, _, _ in layout.qubit_coords()}
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "name", f"{spec.key}_block_{self.block_index}")
        object.__setattr__(self, "n_data", layout.n_data)
        object.__setattr__(self, "n_total", layout.n_total)
        object.__setattr__(self, "checks", checks)
        object.__setattr__(
            self,
            "data_coords",
            tuple(coords[q] for q in range(layout.n_data)),
        )
        object.__setattr__(
            self,
            "ancilla_coords",
            tuple(coords[q] for q in range(layout.n_data, layout.n_total)),
        )
        _validate_schedule(self, self.round_schedule())

    def round_schedule(self, *, reverse: bool = False) -> SecRoundSchedule:
        layout = self._layout
        epoch_numbers = range(7, 0, -1) if reverse else range(1, 8)
        return SecRoundSchedule(
            pre_data_gates=(),
            x_reset_targets=tuple(layout.qX(i) for i in range(layout.N)),
            z_reset_targets=tuple(layout.qZ(i) for i in range(layout.N)),
            cnot_epochs=tuple(
                tuple(
                    layout.cnot_epoch_pairs(
                        epoch,
                        schedule_provenance=self.schedule_provenance,
                    )
                )
                for epoch in epoch_numbers
            ),
            z_readout=tuple(
                ReadoutSpec(layout.qZ(i), f"Z{i}") for i in range(layout.N)
            ),
            x_readout=tuple(
                ReadoutSpec(layout.qX(i), f"X{i}") for i in range(layout.N)
            ),
            post_data_gates=(),
        )

    def sec_schedule(self, reverse: bool = False) -> SecRoundSchedule:
        return self.round_schedule(reverse=reverse)

    def to_sec_frame_pauli_row(self, row: int) -> int:
        return int(row)

    def append_sec_round(
        self,
        builder: StimTextBuilder,
        *,
        offset: int,
        round_index: int,
        noise: NoiseConfig,
        reverse: bool,
        before_readout: ReadoutCallback | None = None,
        after_readout: ReadoutCallback | None = None,
    ) -> tuple[int, ...]:
        return _append_sec_round(
            self,
            builder,
            offset=offset,
            round_index=round_index,
            noise=noise,
            reverse=reverse,
            before_readout=before_readout,
            after_readout=after_readout,
        )


class BB144SecBlock(BBSecBlock):
    """Backward-compatible BB144 adapter."""

    def __init__(
        self,
        block_index: int = 0,
        *,
        schedule_provenance: str = BB_SEC_SCHEDULE_LEGACY,
    ) -> None:
        super().__init__(
            code_name="bb144",
            block_index=int(block_index),
            schedule_provenance=str(schedule_provenance),
        )


def _surface_check_center(
    support: Sequence[int],
    *,
    distance: int,
) -> tuple[float, float]:
    rows = [int(q) // distance for q in support]
    columns = [int(q) % distance for q in support]
    if len(support) == 4:
        return (
            (min(rows) + max(rows)) / 2,
            (min(columns) + max(columns)) / 2,
        )
    if len(support) != 2:
        raise ValueError("a rotated-surface check must have weight two or four")
    if len(set(rows)) == 1:
        row = rows[0]
        center_row = -0.5 if row == 0 else distance - 0.5
        return center_row, (min(columns) + max(columns)) / 2
    if len(set(columns)) == 1:
        column = columns[0]
        center_column = -0.5 if column == 0 else distance - 0.5
        return (min(rows) + max(rows)) / 2, center_column
    raise ValueError("a weight-two surface check is not on a boundary")


def _surface_direction(
    q: int,
    *,
    center: tuple[float, float],
    distance: int,
) -> str:
    row, column = divmod(int(q), int(distance))
    vertical = "N" if row < center[0] else "S"
    horizontal = "W" if column < center[1] else "E"
    return vertical + horizontal


@dataclass(frozen=True)
class RotatedSurfaceSecBlock:
    """One-ancilla-per-check SEC adapter for a rotated surface code."""

    distance: int = 12
    block_index: int = 0

    def __post_init__(self) -> None:
        code = build_rotated_surface(int(self.distance))
        n_data = int(code.n)
        raw_x, raw_z = rotated_surface_local_checks(int(self.distance))
        z_rows = tuple(_mask(row) for row in raw_z)
        x_rows = tuple(_mask(row) for row in raw_x)
        checks = tuple(
            [
                CheckSpec("Z", row << n_data, f"Z{i}")
                for i, row in enumerate(z_rows)
            ]
            + [
                CheckSpec("X", row, f"X{i}")
                for i, row in enumerate(x_rows)
            ]
        )
        n_checks = len(checks)
        n_total = n_data + n_checks
        data_coords = tuple(
            (float(q % self.distance), float(q // self.distance))
            for q in range(n_data)
        )
        ancilla_coords: list[tuple[float, float]] = []
        support_by_check: list[tuple[int, ...]] = []
        for check_index, check in enumerate(checks):
            ancilla = n_data + check_index
            support_bits = (
                check.pauli_row >> n_data
                if check.label.startswith("Z")
                else check.pauli_row
            )
            support = tuple(q for q in range(n_data) if (support_bits >> q) & 1)
            if not support:
                raise ValueError(f"surface check {check.label} has empty support")
            support_by_check.append(support)
            ancilla_coords.append(
                (
                    sum(data_coords[q][0] for q in support) / len(support),
                    sum(data_coords[q][1] for q in support) / len(support),
                )
            )

        object.__setattr__(self, "n_data", n_data)
        object.__setattr__(self, "n_total", n_total)
        object.__setattr__(
            self,
            "name",
            f"rotated_surface_d{self.distance}_block_{self.block_index}",
        )
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "data_coords", data_coords)
        object.__setattr__(self, "ancilla_coords", tuple(ancilla_coords))
        # This is Stim's hook-aware rotated-memory ordering, expressed as the
        # direction of the data qubit relative to its check center.
        # Our check geometry is Stim's rotated-memory geometry with X and Z
        # exchanged, so the corresponding directional orders exchange too.
        x_directions = ("SE", "NE", "SW", "NW")
        z_directions = ("SE", "SW", "NE", "NW")
        epochs: list[list[tuple[int, int]]] = [[] for _ in range(4)]
        for check_index, (check, support) in enumerate(
            zip(checks, support_by_check, strict=True)
        ):
            ancilla = n_data + check_index
            center = _surface_check_center(
                support,
                distance=int(self.distance),
            )
            by_direction = {
                _surface_direction(
                    q,
                    center=center,
                    distance=int(self.distance),
                ): q
                for q in support
            }
            order = z_directions if check.sector == "Z" else x_directions
            for epoch, direction in enumerate(order):
                q = by_direction.get(direction)
                if q is None:
                    continue
                epochs[epoch].append(
                    (q, ancilla) if check.sector == "Z" else (ancilla, q)
                )
        for epoch, pairs in enumerate(epochs):
            flat = [q for pair in pairs for q in pair]
            if len(flat) != len(set(flat)):
                raise AssertionError(
                    f"hook-aware surface epoch {epoch} has a collision"
                )
        object.__setattr__(
            self,
            "_epochs",
            tuple(tuple(epoch) for epoch in epochs),
        )
        _validate_schedule(self, self.round_schedule())

    def round_schedule(self, *, reverse: bool = False) -> SecRoundSchedule:
        z_count = sum(check.label.startswith("Z") for check in self.checks)
        z_readout = tuple(
            ReadoutSpec(self.n_data + i, check.label)
            for i, check in enumerate(self.checks[:z_count])
        )
        x_readout = tuple(
            ReadoutSpec(self.n_data + i, check.label)
            for i, check in enumerate(self.checks[z_count:], start=z_count)
        )
        return SecRoundSchedule(
            pre_data_gates=(),
            x_reset_targets=tuple(readout.target for readout in x_readout),
            z_reset_targets=tuple(readout.target for readout in z_readout),
            # Reverse the unitary SEC body on alternating physical rounds.
            cnot_epochs=tuple(reversed(self._epochs)) if reverse else self._epochs,
            z_readout=z_readout,
            x_readout=x_readout,
            post_data_gates=(),
        )

    def sec_schedule(self, reverse: bool = False) -> SecRoundSchedule:
        return self.round_schedule(reverse=reverse)

    def to_sec_frame_pauli_row(self, row: int) -> int:
        return int(row)

    def append_sec_round(
        self,
        builder: StimTextBuilder,
        *,
        offset: int,
        round_index: int,
        noise: NoiseConfig,
        reverse: bool,
        before_readout: ReadoutCallback | None = None,
        after_readout: ReadoutCallback | None = None,
    ) -> tuple[int, ...]:
        return _append_sec_round(
            self,
            builder,
            offset=offset,
            round_index=round_index,
            noise=noise,
            reverse=reverse,
            before_readout=before_readout,
            after_readout=after_readout,
        )


_FRAME_DATA: dict[str, tuple[str | None, str | None, tuple[int, int], tuple[int, int]]] = {
    "I": (None, None, (1, 0), (0, 1)),
    "H": ("H", "H", (0, 1), (1, 0)),
    "S": ("S", "S_DAG", (1, 1), (0, 1)),
    "S_DAG": ("S_DAG", "S", (1, 1), (0, 1)),
    "H_XY": ("H_XY", "H_XY", (1, 1), (0, 1)),
    "H_YZ": ("H_YZ", "H_YZ", (1, 0), (1, 1)),
}


def _map_pauli_row(
    row: int,
    *,
    n: int,
    x_image: tuple[int, int],
    z_image: tuple[int, int],
) -> int:
    mask = (1 << n) - 1
    old_x = int(row) & mask
    old_z = int(row) >> n
    new_x = 0
    new_z = 0
    for q in range(n):
        x = (old_x >> q) & 1
        z = (old_z >> q) & 1
        mapped_x = (x * x_image[0]) ^ (z * z_image[0])
        mapped_z = (x * x_image[1]) ^ (z * z_image[1])
        new_x |= mapped_x << q
        new_z |= mapped_z << q
    return new_x | (new_z << n)


def _image_sector(image: tuple[int, int]) -> str:
    if image == (1, 0):
        return "X"
    if image == (0, 1):
        return "Z"
    if image == (1, 1):
        return "Y"
    raise ValueError(f"invalid nonidentity Pauli image {image}")


@dataclass(frozen=True)
class UniformLocalCliffordSecBlock:
    """Uniform physical Clifford frame around another block's SEC."""

    base: SecBlock
    frame: str

    def __post_init__(self) -> None:
        frame = self.frame.upper()
        if frame not in _FRAME_DATA:
            raise ValueError(
                f"unsupported local Clifford frame {self.frame!r}; "
                f"choose from {sorted(_FRAME_DATA)}"
            )
        forward, inverse, x_image, z_image = _FRAME_DATA[frame]
        mapped = tuple(
            CheckSpec(
                _image_sector(x_image if check.sector == "X" else z_image),
                _map_pauli_row(
                    check.pauli_row,
                    n=self.base.n_data,
                    x_image=x_image,
                    z_image=z_image,
                ),
                check.label,
            )
            for check in self.base.checks
        )
        for old, new in zip(self.base.checks, mapped):
            if _pauli_weight(old.pauli_row, self.base.n_data) != _pauli_weight(
                new.pauli_row, self.base.n_data
            ):
                raise AssertionError("local Clifford changed a check's weight")
        for left in range(len(mapped)):
            for right in range(left + 1, len(mapped)):
                if _symplectic_product(
                    self.base.checks[left].pauli_row,
                    self.base.checks[right].pauli_row,
                    self.base.n_data,
                ) != _symplectic_product(
                    mapped[left].pauli_row,
                    mapped[right].pauli_row,
                    self.base.n_data,
                ):
                    raise AssertionError("local Clifford did not preserve the symplectic form")

        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_forward_gate", forward)
        object.__setattr__(self, "_inverse_gate", inverse)
        object.__setattr__(self, "_x_image", x_image)
        object.__setattr__(self, "_z_image", z_image)
        a, b = x_image
        c, d = z_image
        object.__setattr__(self, "_inverse_x_image", (d, b))
        object.__setattr__(self, "_inverse_z_image", (c, a))
        object.__setattr__(self, "block_index", getattr(self.base, "block_index", 0))
        object.__setattr__(self, "name", f"{self.base.name}_frame_{frame}")
        object.__setattr__(self, "n_data", self.base.n_data)
        object.__setattr__(self, "n_total", self.base.n_total)
        object.__setattr__(self, "checks", mapped)
        object.__setattr__(self, "data_coords", self.base.data_coords)
        object.__setattr__(self, "ancilla_coords", self.base.ancilla_coords)
        _validate_schedule(self, self.round_schedule())

    def map_pauli_row(self, row: int) -> int:
        """Map a base-frame Pauli into the physical Clifford frame."""

        return _map_pauli_row(
            row,
            n=self.n_data,
            x_image=self._x_image,
            z_image=self._z_image,
        )

    def to_sec_frame_pauli_row(self, row: int) -> int:
        """Map a physical-frame Pauli into the base SEC callback frame."""

        return _map_pauli_row(
            row,
            n=self.n_data,
            x_image=self._inverse_x_image,
            z_image=self._inverse_z_image,
        )

    def round_schedule(self, *, reverse: bool = False) -> SecRoundSchedule:
        schedule = self.base.round_schedule(reverse=reverse)
        data = tuple(range(self.n_data))
        pre = (
            (GateLayer(self._inverse_gate, data),)
            if self._inverse_gate is not None
            else ()
        )
        post = (
            (GateLayer(self._forward_gate, data),)
            if self._forward_gate is not None
            else ()
        )
        return SecRoundSchedule(
            pre_data_gates=pre + schedule.pre_data_gates,
            x_reset_targets=schedule.x_reset_targets,
            z_reset_targets=schedule.z_reset_targets,
            cnot_epochs=schedule.cnot_epochs,
            z_readout=schedule.z_readout,
            x_readout=schedule.x_readout,
            post_data_gates=schedule.post_data_gates + post,
        )

    def sec_schedule(self, reverse: bool = False) -> SecRoundSchedule:
        return self.round_schedule(reverse=reverse)

    def append_sec_round(
        self,
        builder: StimTextBuilder,
        *,
        offset: int,
        round_index: int,
        noise: NoiseConfig,
        reverse: bool,
        before_readout: ReadoutCallback | None = None,
        after_readout: ReadoutCallback | None = None,
    ) -> tuple[int, ...]:
        return _append_sec_round(
            self,
            builder,
            offset=offset,
            round_index=round_index,
            noise=noise,
            reverse=reverse,
            before_readout=before_readout,
            after_readout=after_readout,
        )
