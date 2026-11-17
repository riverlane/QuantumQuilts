"""Stim text emission and noise insertion.

`StimTextBuilder` accumulates one instruction per line -- the flat form the
downstream extractor requires -- and the noise helpers place depolarising
channels after each CX layer, after resets and before measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import stim

from ._flows import measurement_indices
from ._symplectic import pauli, symplectic_product, x_part, z_part


PAULI_LABELS = {(1, 0): "X", (1, 1): "Y", (0, 1): "Z"}

BB_SEC_SCHEDULE_LEGACY = "legacy_identity_first_v1"
BB144_SEC_SCHEDULE_BRAVYI_2024 = "bravyi_2024_table5"

# Directions index [A1, A2, A3, B1, B2, B3] for X checks and
# [B1, B2, B3, A1, A2, A3] for Z checks in the normalized, identity-first
# representation below. The legacy schedule applied Table 5 after normalization.
_BB_LEGACY_EPOCH_DIRECTIONS = (
    (None, 3),
    (1, 5),
    (4, 0),
    (3, 1),
    (5, 2),
    (0, 4),
    (2, None),
)

# Bravyi et al., arXiv:2308.07915, Table 5, transported into BB144's
# normalized coordinates while preserving the paper's original monomial roles.
_BB144_BRAVYI_2024_EPOCH_DIRECTIONS = (
    (None, 5),
    (1, 3),
    (4, 2),
    (5, 1),
    (3, 0),
    (2, 4),
    (0, None),
)


@dataclass(frozen=True)
class BBSpec:
    key: str
    ell: int
    m: int
    expected_k: int
    expected_d: int
    a2: tuple[int, int]
    a3: tuple[int, int]
    b2: tuple[int, int]
    b3: tuple[int, int]


BB72 = BBSpec(
    key="bb72",
    ell=6,
    m=6,
    expected_k=12,
    expected_d=6,
    a2=(0, 5),
    a3=(3, 4),
    b2=(5, 0),
    b3=(4, 3),
)
BB144 = BBSpec(
    key="bb144",
    ell=12,
    m=6,
    expected_k=12,
    expected_d=12,
    a2=(0, 5),
    a3=(3, 4),
    b2=(11, 0),
    b3=(10, 3),
)
BB288 = BBSpec(
    key="bb288",
    ell=12,
    m=12,
    expected_k=12,
    expected_d=18,
    a2=(0, 5),
    a3=(3, 10),
    b2=(11, 0),
    b3=(10, 3),
)
BB_SPECS = {spec.key: spec for spec in (BB72, BB144, BB288)}


class StimTextBuilder:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines: list[str] = list(lines or [])
        self.rec_count = 0

    def line(self, text: str) -> None:
        self.lines.append(text)

    def tick(self) -> None:
        self.line("TICK")

    def gate(self, name: str, targets: Iterable[int]) -> None:
        ts = [str(int(target)) for target in targets]
        if ts:
            self.line(f"{name} " + " ".join(ts))

    def pairs(
        self, name: str, pairs: Iterable[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        values = [(int(control), int(target)) for control, target in pairs]
        if values:
            flat = [str(target) for pair in values for target in pair]
            self.line(f"{name} " + " ".join(flat))
        return values

    def measure(self, name: str, qubits: Iterable[int]) -> list[int]:
        values = [int(qubit) for qubit in qubits]
        if not values:
            return []
        self.gate(name, values)
        records = list(range(self.rec_count, self.rec_count + len(values)))
        self.rec_count += len(values)
        return records

    def detector(self, records: Iterable[int], sector: int | None = None) -> None:
        """Emit a detector, optionally tagged with its CSS sector.

        `sector` becomes the detector's single coordinate: 0 for X-type, 1 for
        Z-type, 2 for anything that is mixed or could not be classified. The
        circuits carry no other detector coordinates, so there is nothing to
        collide with; omitting it leaves the detectors untagged.
        """
        parts = [f"rec[-{self.rec_count - int(record)}]" for record in records]
        if not parts:
            return
        head = "DETECTOR" if sector is None else f"DETECTOR({int(sector)})"
        self.line(head + " " + " ".join(parts))

    def observable(self, index: int, records: Iterable[int]) -> None:
        parts = [f"rec[-{self.rec_count - int(record)}]" for record in records]
        if parts:
            self.line(f"OBSERVABLE_INCLUDE({int(index)}) " + " ".join(parts))

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


class BBLayout:
    def __init__(self, spec: BBSpec = BB144) -> None:
        self.spec = spec
        self.N = spec.ell * spec.m
        self.n_data = 2 * self.N
        self.n_total = 4 * self.N

    def idx(self, a: int, b: int) -> int:
        return (a % self.spec.ell) * self.spec.m + (b % self.spec.m)

    def add(self, index: int, shift: tuple[int, int]) -> int:
        a = index // self.spec.m
        b = index % self.spec.m
        return self.idx(a + shift[0], b + shift[1])

    def sub(self, index: int, shift: tuple[int, int]) -> int:
        return self.add(index, (-shift[0], -shift[1]))

    def qL(self, index: int) -> int:
        return int(index)

    def qR(self, index: int) -> int:
        return self.N + int(index)

    def qX(self, index: int) -> int:
        return 2 * self.N + int(index)

    def qZ(self, index: int) -> int:
        return 3 * self.N + int(index)

    def x_check_support(self, index: int) -> list[int]:
        return [
            self.qL(index),
            self.qL(self.add(index, self.spec.a2)),
            self.qL(self.add(index, self.spec.a3)),
            self.qR(index),
            self.qR(self.add(index, self.spec.b2)),
            self.qR(self.add(index, self.spec.b3)),
        ]

    def z_check_support(self, index: int) -> list[int]:
        return [
            self.qL(index),
            self.qL(self.sub(index, self.spec.b2)),
            self.qL(self.sub(index, self.spec.b3)),
            self.qR(index),
            self.qR(self.sub(index, self.spec.a2)),
            self.qR(self.sub(index, self.spec.a3)),
        ]

    def h_x_rows(self) -> list[int]:
        return [_mask_from_support(self.x_check_support(i)) for i in range(self.N)]

    def h_z_rows(self) -> list[int]:
        return [_mask_from_support(self.z_check_support(i)) for i in range(self.N)]

    def cnot_epoch_pairs(
        self,
        epoch: int,
        *,
        schedule_provenance: str = BB_SEC_SCHEDULE_LEGACY,
    ) -> list[tuple[int, int]]:
        if schedule_provenance == BB_SEC_SCHEDULE_LEGACY:
            directions = _BB_LEGACY_EPOCH_DIRECTIONS
        elif schedule_provenance == BB144_SEC_SCHEDULE_BRAVYI_2024:
            if self.spec.key != "bb144":
                raise ValueError(
                    f"{schedule_provenance!r} is audited only for bb144, "
                    f"not {self.spec.key}"
                )
            directions = _BB144_BRAVYI_2024_EPOCH_DIRECTIONS
        else:
            raise ValueError(f"unknown BB SEC schedule {schedule_provenance!r}")
        if epoch < 1 or epoch > len(directions):
            raise ValueError(f"unknown epoch {epoch}")

        x_direction, z_direction = directions[epoch - 1]
        pairs: list[tuple[int, int]] = []
        for index in range(self.N):
            if x_direction is not None:
                pairs.append(self._x_cnot_pair(index, x_direction))
            if z_direction is not None:
                pairs.append(self._z_cnot_pair(index, z_direction))
        return pairs

    def _x_cnot_pair(self, index: int, direction: int) -> tuple[int, int]:
        spec = self.spec
        targets = (
            self.qL(index),
            self.qL(self.add(index, spec.a2)),
            self.qL(self.add(index, spec.a3)),
            self.qR(index),
            self.qR(self.add(index, spec.b2)),
            self.qR(self.add(index, spec.b3)),
        )
        return self.qX(index), targets[int(direction)]

    def _z_cnot_pair(self, index: int, direction: int) -> tuple[int, int]:
        spec = self.spec
        controls = (
            self.qL(index),
            self.qL(self.sub(index, spec.b2)),
            self.qL(self.sub(index, spec.b3)),
            self.qR(index),
            self.qR(self.sub(index, spec.a2)),
            self.qR(self.sub(index, spec.a3)),
        )
        return controls[int(direction)], self.qZ(index)

    def qubit_coords(self) -> list[tuple[int, float, float, str, str]]:
        result: list[tuple[int, float, float, str, str]] = []
        for a in range(self.spec.ell):
            for b in range(self.spec.m):
                index = self.idx(a, b)
                result.append((self.qL(index), float(a), float(b) + 0.18, "DATA", "#d4a017"))
                result.append((self.qR(index), float(a) + 0.48, float(b) - 0.18, "DATA", "#2e8b57"))
                result.append((self.qX(index), float(a) + 0.24, float(b) + 0.72, "XANC", "#c0392b"))
                result.append((self.qZ(index), float(a) + 0.24, float(b) - 0.72, "ZANC", "#2f5fd0"))
        return sorted(result)


def _mask_from_support(support: Iterable[int]) -> int:
    mask = 0
    for qubit in support:
        mask ^= 1 << int(qubit)
    return mask


@dataclass(frozen=True)
class NoiseConfig:
    p_sec_1q: float = 0.001
    p_sec_2q: float = 0.001
    p_morph_1q: float = 0.001
    p_morph_2q: float = 0.001
    p_morph_meas: float = 0.001
    p_morph_post_measure: float = 0.001
    # Depolarising noise on qubits that no gate touches in a tick. Zero makes
    # idling noiseless.
    p_idle: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PauliTerm:
    label: str
    row: int


def append_noise_after_reset(
    builder: StimTextBuilder, qubits: Iterable[int], basis: str, probability: float
) -> None:
    if probability <= 0:
        return
    gate = "Z_ERROR" if basis.upper() == "X" else "X_ERROR"
    builder.gate(f"{gate}({probability})", qubits)


def append_noise_before_measure(
    builder: StimTextBuilder, qubits: Iterable[int], basis: str, probability: float
) -> None:
    if probability <= 0:
        return
    gate = "Z_ERROR" if basis.upper() == "X" else "X_ERROR"
    builder.gate(f"{gate}({probability})", qubits)


def append_noise_after_cx(
    builder: StimTextBuilder, pairs: list[tuple[int, int]], probability: float
) -> None:
    if probability <= 0 or not pairs:
        return
    builder.gate(
        f"DEPOLARIZE2({probability})",
        [target for pair in pairs for target in pair],
    )


def symplectic_pairs(rows: Sequence[int], n: int) -> tuple[tuple[int, int], ...]:
    remaining = [int(row) for row in rows]
    pairs: list[tuple[int, int]] = []
    while remaining:
        left = remaining.pop(0)
        partner_index = next(
            (
                index
                for index, right in enumerate(remaining)
                if symplectic_product(left, right, n)
            ),
            None,
        )
        if partner_index is None:
            raise ValueError("logical quotient has a degenerate symplectic form")
        right = remaining.pop(partner_index)
        cleaned: list[int] = []
        for row in remaining:
            if symplectic_product(row, right, n):
                row ^= left
            if symplectic_product(row, left, n):
                row ^= right
            cleaned.append(row)
        remaining = cleaned
        pairs.append((left, right))
    return tuple(pairs)


def add_single_qubit_pauli(row: int, *, q: int, label: str, n: int) -> int:
    label = label.upper()
    x = x_part(row, n)
    z = z_part(row, n)
    if label in {"X", "Y"}:
        x ^= 1 << int(q)
    if label in {"Z", "Y"}:
        z ^= 1 << int(q)
    return pauli(x, z, n)


def _mpp_text(row: int, n: int) -> str:
    x = x_part(row, n)
    z = z_part(row, n)
    pieces: list[str] = []
    for qubit in range(n):
        xb = (x >> qubit) & 1
        zb = (z >> qubit) & 1
        if xb or zb:
            pieces.append(
                f"{('X' if xb and not zb else 'Z' if zb and not xb else 'Y')}{qubit}"
            )
    if not pieces:
        raise ValueError("cannot measure an identity MPP term")
    return "*".join(pieces)


def append_boundary_terms_n(
    builder: StimTextBuilder,
    terms: Sequence[PauliTerm],
    *,
    n: int,
    heading: str,
) -> dict[str, int]:
    builder.line(f"# {heading}")
    records: dict[str, int] = {}
    for term in terms:
        builder.line(f"# {term.label}")
        builder.line(f"MPP {_mpp_text(term.row, n)}")
        records[term.label] = builder.rec_count
        builder.rec_count += 1
    return records


def _append_depolarize1(
    builder: StimTextBuilder, qubits: Sequence[int], probability: float
) -> None:
    if probability > 0 and qubits:
        builder.gate(f"DEPOLARIZE1({float(probability):.12g})", qubits)


def append_basis_into_z(
    builder: StimTextBuilder, sites: Sequence[tuple[int, str]], probability: float
) -> None:
    x_sites = [q for q, label in sites if label == "X"]
    y_sites = [q for q, label in sites if label == "Y"]
    if x_sites:
        builder.gate("H", x_sites)
    if y_sites:
        builder.gate("H_YZ", y_sites)
    if x_sites or y_sites:
        _append_depolarize1(builder, [*x_sites, *y_sites], probability)
        builder.tick()


def append_basis_out_of_z(
    builder: StimTextBuilder, sites: Sequence[tuple[int, str]], probability: float
) -> None:
    append_basis_into_z(builder, sites, probability)


def shifted_mask(mask: int, offset: int) -> int:
    return int(mask) << int(offset)


def assert_pairwise_commuting(terms: Sequence[PauliTerm], n: int) -> None:
    for index, left in enumerate(terms):
        for right in terms[index + 1 :]:
            if symplectic_product(left.row, right.row, n):
                raise ValueError(
                    f"noncommuting boundary terms {left.label} and {right.label}"
                )


def append_detectors_and_observables(
    raw_text: str,
    *,
    detectors: Sequence[int],
    observables: Sequence[int],
    observable_labels: Sequence[str] | None = None,
    detector_sectors: Sequence[int | None] | None = None,
) -> str:
    if observable_labels is not None and len(observable_labels) != len(observables):
        raise ValueError("one label is required for each observable")
    if detector_sectors is not None and len(detector_sectors) != len(detectors):
        raise ValueError("one sector is required for each detector")
    circuit = stim.Circuit(raw_text)
    builder = StimTextBuilder(raw_text.splitlines())
    builder.rec_count = circuit.num_measurements
    builder.line("# automatic deterministic detector complement")
    for position, detector in enumerate(detectors):
        sector = None if detector_sectors is None else detector_sectors[position]
        builder.detector(measurement_indices(detector), sector)
    builder.line("# protected and measured-channel observables")
    for index, observable in enumerate(observables):
        if observable_labels is not None:
            builder.line(f"# observable {index}: {observable_labels[index]}")
        builder.observable(index, measurement_indices(observable))
    return builder.text()


def dem_logical_support_audit(dem: stim.DetectorErrorModel) -> dict[str, int]:
    logical_mechanisms = 0
    logical_without_detector = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        targets = instruction.targets_copy()
        has_logical = any(target.is_logical_observable_id() for target in targets)
        has_detector = any(target.is_relative_detector_id() for target in targets)
        logical_mechanisms += int(has_logical)
        logical_without_detector += int(has_logical and not has_detector)
    return {
        "logical_fault_mechanisms": logical_mechanisms,
        "logical_fault_mechanisms_without_detector": logical_without_detector,
    }


def audit_observable_gauge_record_support(*args: object, **kwargs: object) -> None:
    raise NotImplementedError("generic circuits use their case-aware record audit")


CSS_SECTOR_CODE = {"X": 0, "Z": 1}
CSS_SECTOR_MIXED = 2


def measurement_bases(circuit: "stim.Circuit") -> list[str]:
    """Basis of every measurement record, in order.

    MPP contributes one record per term, not one per instruction -- miscounting
    that silently misaligns every rec[] offset after the first MPP.
    """
    out: list[str] = []
    for instruction in circuit.flattened():
        name = instruction.name
        if name == "MPP":
            previous_was_combiner = False
            current: set[str] = set()
            for target in instruction.targets_copy():
                if target.is_combiner:
                    previous_was_combiner = True
                    continue
                if not previous_was_combiner and current:
                    out.append("X" if current == {"X"} else "Z")
                    current = set()
                previous_was_combiner = False
                current.add("X" if target.is_x_target else "Z")
            if current:
                out.append("X" if current == {"X"} else "Z")
        elif name in ("M", "MZ", "MR"):
            out.extend("Z" for _ in instruction.targets_copy())
        elif name in ("MX", "MRX"):
            out.extend("X" for _ in instruction.targets_copy())
        elif name in ("MY", "MRY"):
            out.extend("Y" for _ in instruction.targets_copy())
    return out


def detector_sector_from_basis(records: Iterable[int], bases: Sequence[str]) -> int:
    """Classify a detector by the basis of the records it compares.

    Every detector of a CSS surgery circuit compares checks of a single type,
    so this is complete where the algebraic labelling in
    `label_blockwise_css_detectors` declines -- it returns None for anything
    touching a gauge record. Checked against that labelling on a full
    bb144 BB2_ZZ circuit: agrees on all 4089 detectors it classifies, and
    resolves the remaining 62.
    """
    kinds = {bases[int(record)] for record in records}
    if kinds == {"X"}:
        return CSS_SECTOR_CODE["X"]
    if kinds == {"Z"}:
        return CSS_SECTOR_CODE["Z"]
    return CSS_SECTOR_MIXED


def css_sector_code(types: Sequence[str] | None) -> int:
    """Collapse a per-block type tuple to one coordinate.

    `label_blockwise_css_detectors` returns e.g. ("X", "I") or ("Z", "Z"), or
    None when the detector could not be classified. A detector counts as
    X- or Z-type when every block it touches agrees; anything else is mixed.
    """
    if types is None:
        return CSS_SECTOR_MIXED
    present = {t for t in types if t != "I"}
    if len(present) == 1:
        return CSS_SECTOR_CODE.get(present.pop(), CSS_SECTOR_MIXED)
    return CSS_SECTOR_MIXED


_NOISE_OPS = frozenset({
    "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
    "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1",
})
_ANNOTATIONS = frozenset({
    "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK",
})


def physical_qubits(circuit: "stim.Circuit") -> set[int]:
    """Qubits the circuit itself ever applies noise to.

    Everything else is a noiseless bookkeeping qubit -- the reference qubits
    that define the observables -- and must stay noiseless: noising one creates
    an undetectable logical error by construction, since no detector covers it.
    """
    found: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name in _NOISE_OPS:
            found.update(
                target.qubit_value
                for target in instruction.targets_copy()
                if target.qubit_value is not None
            )
    return found


def append_idle_noise(circuit: "stim.Circuit", probability: float) -> "stim.Circuit":
    """Depolarise every physical qubit that no gate touches, in every tick.

    Applied as a pass over the finished circuit rather than inside the builder,
    because the condition is "untouched in this tick", which the builder does
    not track. Verified to leave the detector error model's fault-mechanism
    support exactly unchanged: only the probabilities move.
    """
    if probability <= 0:
        return circuit
    physical = physical_qubits(circuit)
    if not physical:
        return circuit

    def rebuild(block: "stim.Circuit") -> "stim.Circuit":
        out = stim.Circuit()
        touched: set[int] = set()

        def flush() -> None:
            idle = sorted(physical - touched)
            if idle:
                out.append("DEPOLARIZE1", idle, probability)

        for instruction in block:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                flush()
                touched.clear()
                out.append(
                    stim.CircuitRepeatBlock(
                        instruction.repeat_count, rebuild(instruction.body_copy())
                    )
                )
                continue
            if instruction.name == "TICK":
                flush()
                touched.clear()
                out.append(instruction)
                continue
            out.append(instruction)
            if instruction.name not in _ANNOTATIONS and instruction.name not in _NOISE_OPS:
                touched.update(
                    target.qubit_value
                    for target in instruction.targets_copy()
                    if target.qubit_value is not None
                )
        flush()
        return out

    return rebuild(circuit)
