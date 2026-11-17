"""Convert a public representative record into an explicit quilt input.

The compact data archive embeds the target, selected representatives, and
validation summary in one record. Shared piece indices define the pairing;
every algebraic object is reconstructed by the normal quilt loader.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .generation import load_quilt


SELECTED_SCHEMA = "quilt-selected-representative-v1"
RESULT_SCHEMA = "quilt-representative-result-v2"
RECORD_SCHEMA = "quilt-representative-v2"
# Records with schema v3 have the same record and block fields as v2.
# Records with schema v6 or v8 carry the same quilt algebra -- same blocks,
# same pieces, same masks -- and differ only in surrounding metadata, so they
# are read directly.
RECORD_SCHEMAS = frozenset(
    {RECORD_SCHEMA, "quilt-representative-v3", "quilt-representative-v6",
     "quilt-representative-v8"}
)
_HEX_MASK = re.compile(r"0x[0-9a-fA-F]+\Z")


def _mapping(value: object, subject: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{subject} must be an object")
    return value


def _sequence(value: object, subject: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{subject} must be an array")
    return value


def _integer(value: object, subject: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{subject} must be an integer")
    if value < minimum:
        raise ValueError(f"{subject} must be at least {minimum}")
    return value


def _mask(value: object, subject: str, *, n: int) -> int:
    if not isinstance(value, str) or _HEX_MASK.fullmatch(value) is None:
        raise ValueError(f"{subject} must be a 0x-prefixed hexadecimal string")
    result = int(value, 16)
    if result >= 1 << n:
        raise ValueError(f"{subject} exceeds the declared {n}-qubit width")
    return result


def _selected_candidate(payload: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    schema = payload.get("schema")
    if schema not in {None, SELECTED_SCHEMA, RESULT_SCHEMA, *RECORD_SCHEMAS}:
        raise ValueError(f"unsupported selected-representative schema {schema!r}")
    if "candidates" not in payload:
        candidate = payload
        target_id = str(payload.get("target_id", payload.get("candidate_id", "")))
        if not target_id:
            raise ValueError("selected representative has no target_id")
        return target_id, candidate
    selected = payload.get("selected_candidate_id")
    if not selected:
        raise ValueError("representative result has no selected candidate")
    matches = [
        _mapping(candidate, "candidate")
        for candidate in _sequence(payload["candidates"], "candidates")
        if _mapping(candidate, "candidate").get("candidate_id") == selected
    ]
    if len(matches) != 1:
        raise ValueError(f"selected candidate {selected!r} is not unique")
    target_id = str(payload.get("target_id", selected))
    if not target_id:
        raise ValueError("representative result has no target_id")
    return target_id, matches[0]


def _adapter(code: str) -> tuple[dict[str, Any], int]:
    bb_widths = {"bb72": 72, "bb144": 144, "bb288": 288}
    if code in bb_widths:
        return {"kind": code}, bb_widths[code]
    prefix = "rotated_surface_d"
    if code.startswith(prefix):
        suffix = code[len(prefix):]
        if not suffix.isdecimal() or int(suffix) < 2:
            raise ValueError(f"invalid rotated-surface code identifier {code!r}")
        distance = int(suffix)
        return {"kind": "rotated_surface", "distance": distance}, distance**2
    raise ValueError(f"no bundled SEC adapter for code {code!r}")


def _piece(
    pauli: Mapping[str, Any],
    *,
    n: int,
    subject: str,
) -> dict[str, list[int]]:
    if "pauli" in pauli or "qubit" in pauli:
        if "pauli" not in pauli or "qubit" not in pauli:
            raise ValueError(f"{subject} requires both pauli and qubit")
        label = str(pauli["pauli"])
        if label not in {"X", "Y", "Z"}:
            raise ValueError(f"{subject}.pauli is unsupported: {label!r}")
        qubit = _integer(pauli["qubit"], f"{subject}.qubit")
        result = {
            "x": [qubit] if label in {"X", "Y"} else [],
            "z": [qubit] if label in {"Y", "Z"} else [],
        }
    elif "x_support" in pauli or "z_support" in pauli:
        result = {
            "x": [
                _integer(q, f"{subject}.x_support[{index}]")
                for index, q in enumerate(
                    _sequence(pauli.get("x_support", ()), f"{subject}.x_support")
                )
            ],
            "z": [
                _integer(q, f"{subject}.z_support[{index}]")
                for index, q in enumerate(
                    _sequence(pauli.get("z_support", ()), f"{subject}.z_support")
                )
            ],
        }
    else:
        result = {
            "x": [
                _integer(q, f"{subject}.x[{index}]")
                for index, q in enumerate(
                    _sequence(pauli.get("x", ()), f"{subject}.x")
                )
            ],
            "z": [
                _integer(q, f"{subject}.z[{index}]")
                for index, q in enumerate(
                    _sequence(pauli.get("z", ()), f"{subject}.z")
                )
            ],
        }
    if not result["x"] and not result["z"]:
        raise ValueError(f"{subject} is the identity")
    for axis in ("x", "z"):
        support = result[axis]
        if len(support) != len(set(support)):
            raise ValueError(f"{subject}.{axis} contains duplicate qubits")
        if any(qubit >= n for qubit in support):
            raise ValueError(f"{subject}.{axis} contains a qubit outside 0..{n - 1}")
    support = set(result["x"]) | set(result["z"])
    if "num_qubits" in pauli and _integer(
        pauli["num_qubits"], f"{subject}.num_qubits", minimum=1
    ) != n:
        raise ValueError(f"{subject}.num_qubits does not match its block")
    if "weight" in pauli and _integer(pauli["weight"], f"{subject}.weight") != len(support):
        raise ValueError(f"{subject}.weight does not match its Pauli support")
    return result


def _piece_masks(pieces: Sequence[dict[str, list[int]]]) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for piece in pieces:
        for qubit in piece["x"]:
            x_mask ^= 1 << qubit
        for qubit in piece["z"]:
            z_mask ^= 1 << qubit
    return x_mask, z_mask


def _v6_record_width(payload: Mapping[str, Any]) -> int:
    """Validate a v6 or v8 record and return its declared interface width.

    These records split the checks differently from v2/v3.  The rank condition
    and SkipTree connectivity are terminal gates *inside* the representative
    search -- a state failing either is never emitted -- so they are recorded
    as search telemetry, not as validation.  What remains in `validation` is
    the one check made after the fact, the distance screen.
    """

    required = {"schema", "scale", "target_id", "cohort",
                "operation_threshold", "blocks", "pairing", "validation"}
    missing = required - set(payload)
    unexpected = set(payload) - required - {"telemetry"}
    if missing or unexpected:
        raise ValueError(
            f"v6 record fields wrong; missing {sorted(missing)}, "
            f"unexpected {sorted(unexpected)}"
        )
    _integer(payload.get("operation_threshold"), "operation_threshold", minimum=1)
    pairing = _mapping(payload.get("pairing"), "pairing")
    unexpected = set(pairing) - {"width", "strategy", "coordinates", "resample_seed"}
    if unexpected:
        raise ValueError(f"v6 pairing has unsupported fields: {sorted(unexpected)}")
    width = _integer(pairing.get("width"), "pairing.width", minimum=1)
    validation = _mapping(payload.get("validation"), "validation")
    screen = _mapping(validation.get("distance_screen"), "validation.distance_screen")
    if screen.get("outcome") != "no_subthreshold_witness":
        raise ValueError(
            "v6 record did not pass its distance screen: "
            f"{screen.get('outcome')!r}"
        )
    return width


def _record_width(payload: Mapping[str, Any]) -> int:
    if payload.get("schema") in ("quilt-representative-v6", "quilt-representative-v8"):
        return _v6_record_width(payload)
    expected_fields = {
        "schema",
        "scale",
        "target_id",
        "cohort",
        "operation_threshold",
        "blocks",
        "pairing",
        "validation",
    }
    # These records may carry a `telemetry` block recording how the
    # representative was found.  It is not algebra, so it is permitted but not
    # required; the fields the quilt is built from are still checked exactly.
    optional_fields = {"telemetry"}
    missing = expected_fields - set(payload)
    unexpected = set(payload) - expected_fields - optional_fields
    if missing or unexpected:
        raise ValueError(
            f"representative fields must be exactly {sorted(expected_fields)}"
            + (f"; missing {sorted(missing)}" if missing else "")
            + (f"; unexpected {sorted(unexpected)}" if unexpected else "")
        )
    operation_threshold = _integer(
        payload.get("operation_threshold"), "operation_threshold", minimum=1
    )
    pairing = _mapping(payload.get("pairing"), "pairing")
    # v3 and later records also describe how the pairing was produced (the
    # explicit piece coordinates, and a seed when the pairing was resampled).
    # The width claim below is the one that matters.
    unexpected_pairing = set(pairing) - {
        "width",
        "charge_rank",
        "strategy",
        "source_coordinate_convention",
        "coordinates",
        "resample_seed",
    }
    if unexpected_pairing:
        raise ValueError(
            f"pairing contains unsupported fields: {sorted(unexpected_pairing)}"
        )
    width = _integer(pairing.get("width"), "pairing.width", minimum=1)
    charge_rank = pairing.get("charge_rank")
    if charge_rank is not None:
        rank = _mapping(charge_rank, "pairing.charge_rank")
        if set(rank) != {"rank", "maximum"}:
            raise ValueError("pairing.charge_rank requires exactly rank and maximum")
        value = _integer(rank["rank"], "pairing.charge_rank.rank")
        maximum = _integer(rank["maximum"], "pairing.charge_rank.maximum")
        if value > maximum:
            raise ValueError("pairing.charge_rank.rank exceeds maximum")

    validation = _mapping(payload.get("validation"), "validation")
    if set(validation) != {"rank_condition", "skiptree", "distance_screen"}:
        raise ValueError("validation has unsupported or missing fields")
    if validation.get("rank_condition") != "passed":
        raise ValueError("validation.rank_condition must be 'passed'")
    skiptree = _mapping(validation.get("skiptree"), "validation.skiptree")
    # v2 reports one combined product bound; v3 splits it into the per-edge and
    # per-basis-element source-check counts the strict criterion actually uses.
    fields = set(skiptree)
    if fields == {"connected", "maximum_check_product"}:
        bounds = {"maximum_check_product": 3}
    elif fields == {
        "connected",
        "maximum_edge_source_checks",
        "maximum_basis_source_checks",
    }:
        bounds = {"maximum_edge_source_checks": 1, "maximum_basis_source_checks": 3}
    else:
        raise ValueError("validation.skiptree has unsupported or missing fields")
    if skiptree.get("connected") is not True:
        raise ValueError("validation.skiptree.connected must be true")
    for field, expected in bounds.items():
        if _integer(
            skiptree.get(field), f"validation.skiptree.{field}", minimum=1
        ) != expected:
            raise ValueError(f"validation.skiptree.{field} must be {expected}")
    screen = _mapping(
        validation.get("distance_screen"), "validation.distance_screen"
    )
    if set(screen) != {"threshold", "outcome", "trials_by_method"}:
        raise ValueError("validation.distance_screen has unsupported or missing fields")
    threshold = _integer(
        screen.get("threshold"), "validation.distance_screen.threshold", minimum=1
    )
    if threshold != operation_threshold:
        raise ValueError("distance-screen threshold does not match operation_threshold")
    if screen.get("outcome") != "no_subthreshold_witness":
        raise ValueError(
            "validation.distance_screen.outcome must be 'no_subthreshold_witness'"
        )
    trials = _mapping(
        screen.get("trials_by_method"),
        "validation.distance_screen.trials_by_method",
    )
    if not trials:
        raise ValueError("validation.distance_screen.trials_by_method must not be empty")
    for method, count in trials.items():
        if not isinstance(method, str) or not method:
            raise ValueError("distance-screen method names must be nonempty strings")
        _integer(count, f"distance-screen trials for {method!r}", minimum=1)
    return width


def quilt_payload_from_selected_record(
    source: str | Path | Mapping[str, Any],
    *,
    target_id: str | None = None,
    physical_error_rate: float = 0.003,
    surgery_rounds: int = 12,
) -> dict[str, Any]:
    """Return a ``quilt-spec/v1`` payload from one public result or candidate."""
    if isinstance(source, Mapping):
        payload = source
    else:
        path = Path(source)
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            if target_id is None:
                raise ValueError(
                    f"{path} is JSON Lines; select a record with target_id"
                ) from None
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            matches = [row for row in rows if str(row.get("target_id")) == target_id]
            if len(matches) != 1:
                raise ValueError(
                    f"target_id {target_id!r} occurs {len(matches)} times in {path}"
                )
            payload = matches[0]
    payload = _mapping(payload, "selected record")
    target_id, candidate = _selected_candidate(payload)
    record_width = _record_width(payload) if payload.get("schema") in RECORD_SCHEMAS else None
    raw_blocks = _sequence(candidate.get("blocks"), "candidate.blocks")
    if len(raw_blocks) < 2:
        raise ValueError("a selected representative requires at least two blocks")
    blocks: list[dict[str, Any]] = []
    block_ids: dict[int, str] = {}
    block_widths: dict[int, int] = {}
    block_payloads: dict[int, Mapping[str, Any]] = {}
    for offset, raw in enumerate(raw_blocks):
        block = _mapping(raw, f"blocks[{offset}]")
        if candidate.get("schema") in RECORD_SCHEMAS and set(block) != {
            "block_index",
            "code",
            "num_qubits",
            "logical_x",
            "logical_z",
            "x_mask",
            "z_mask",
            "pieces",
        }:
            raise ValueError(f"blocks[{offset}] has unsupported or missing fields")
        index = _integer(block.get("block_index", offset), f"blocks[{offset}].block_index")
        if index in block_ids:
            raise ValueError(f"duplicate block_index {index}")
        block_id = f"block_{index}"
        block_ids[index] = block_id
        code = str(block["code"])
        adapter, n = _adapter(code)
        if "num_qubits" in block and _integer(
            block["num_qubits"], f"blocks[{offset}].num_qubits", minimum=1
        ) != n:
            raise ValueError(
                f"blocks[{offset}].num_qubits does not match code {code!r}"
            )
        if (
            candidate.get("schema") in {SELECTED_SCHEMA, *RECORD_SCHEMAS}
            and "num_qubits" not in block
        ):
            raise ValueError(f"blocks[{offset}] has no num_qubits")
        if candidate.get("schema") in RECORD_SCHEMAS:
            for field in ("logical_x", "logical_z"):
                if (
                    not isinstance(block.get(field), str)
                    or _HEX_MASK.fullmatch(block[field]) is None
                ):
                    raise ValueError(
                        f"blocks[{offset}].{field} must be a 0x-prefixed hexadecimal string"
                    )
        block_widths[index] = n
        block_payloads[index] = block
        blocks.append({"id": block_id, "circuit_adapter": adapter})
    expected_indices = list(range(len(raw_blocks)))
    if sorted(block_ids) != expected_indices:
        raise ValueError(f"block_index values must be exactly {expected_indices}")
    blocks.sort(key=lambda row: int(row["id"].split("_")[-1]))

    by_block: dict[int, list[tuple[int, dict[str, list[int]]]]] = {
        index: [] for index in block_ids
    }
    if "stitch_pieces" in candidate:
        for raw_stitch in _sequence(candidate["stitch_pieces"], "candidate.stitch_pieces"):
            stitch = _mapping(raw_stitch, "stitch piece")
            coordinate = _integer(stitch["piece_index"], "stitch piece.piece_index")
            seen: set[int] = set()
            raw_locals = _sequence(
                stitch.get("blocks"), f"stitch_pieces[{coordinate}].blocks"
            )
            for raw_local in raw_locals:
                local = _mapping(raw_local, "local stitch piece")
                index = _integer(local["block_index"], "local stitch piece.block_index")
                if index not in by_block or index in seen:
                    raise ValueError(f"invalid block {index} at stitch coordinate {coordinate}")
                seen.add(index)
                by_block[index].append(
                    (
                        coordinate,
                        _piece(
                            _mapping(local["pauli"], "local Pauli"),
                            n=block_widths[index],
                            subject=f"stitch_pieces[{coordinate}].blocks[{index}].pauli",
                        ),
                    )
                )
            if seen != set(block_ids):
                raise ValueError(f"stitch coordinate {coordinate} is not present on every block")
    else:
        # Public release records keep each local piece beside its block. Shared
        # piece_index values are the explicit cross-block matching.
        for offset, raw in enumerate(raw_blocks):
            block = _mapping(raw, f"blocks[{offset}]")
            index = _integer(block.get("block_index", offset), f"blocks[{offset}].block_index")
            raw_pieces = _sequence(block.get("pieces"), f"blocks[{offset}].pieces")
            for piece_offset, raw_piece in enumerate(raw_pieces):
                local_piece = _mapping(raw_piece, "local stitch piece")
                if candidate.get("schema") in RECORD_SCHEMAS and set(local_piece) != {
                    "piece_index",
                    "qubit",
                    "pauli",
                }:
                    raise ValueError(
                        f"blocks[{offset}].pieces[{piece_offset}] has unsupported "
                        "or missing fields"
                    )
                by_block[index].append(
                    (
                        _integer(
                            local_piece["piece_index"],
                            f"blocks[{offset}].pieces[{piece_offset}].piece_index",
                        ),
                        _piece(
                            local_piece,
                            n=block_widths[index],
                            subject=f"blocks[{offset}].pieces[{piece_offset}]",
                        ),
                    )
                )

    coordinates = None
    pieces_by_block: dict[str, list[dict[str, list[int]]]] = {}
    for index in sorted(by_block):
        ordered = sorted(by_block[index], key=lambda item: item[0])
        local_coordinates = tuple(coordinate for coordinate, _ in ordered)
        if len(local_coordinates) != len(set(local_coordinates)):
            raise ValueError(f"block {index} contains duplicate piece_index values")
        if record_width is not None and local_coordinates != tuple(range(record_width)):
            raise ValueError(
                f"block {index} piece_index values must be exactly "
                f"{list(range(record_width))}"
            )
        if coordinates is None:
            coordinates = local_coordinates
        elif local_coordinates != coordinates:
            raise ValueError("blocks do not share the same ordered stitch coordinates")
        pieces = [piece for _, piece in ordered]
        if not pieces:
            raise ValueError(f"block {index} has no stitching pieces")
        used_qubits: set[int] = set()
        for piece in pieces:
            support = set(piece["x"]) | set(piece["z"])
            overlap = used_qubits & support
            if overlap:
                raise ValueError(
                    f"block {index} repeats qubits across pieces: {sorted(overlap)}"
                )
            used_qubits.update(support)
        block = block_payloads[index]
        has_x_mask = "x_mask" in block
        has_z_mask = "z_mask" in block
        if has_x_mask != has_z_mask:
            raise ValueError(f"block {index} must supply both x_mask and z_mask")
        if (
            candidate.get("schema") in {SELECTED_SCHEMA, *RECORD_SCHEMAS}
            and not has_x_mask
        ):
            raise ValueError(f"block {index} has no x_mask/z_mask")
        if has_x_mask:
            expected_x = _mask(block["x_mask"], f"block {index}.x_mask", n=block_widths[index])
            expected_z = _mask(block["z_mask"], f"block {index}.z_mask", n=block_widths[index])
            actual_x, actual_z = _piece_masks(pieces)
            if (actual_x, actual_z) != (expected_x, expected_z):
                raise ValueError(
                    f"block {index} piece product does not match x_mask/z_mask"
                )
        pieces_by_block[block_ids[index]] = pieces

    result = {
        "schema": "quilt-construction/quilt-spec/v1",
        "case_id": target_id,
        "blocks": blocks,
        "stitching_families": [
            {
                "name": "measurement",
                "incidences": [[block_ids[index] for index in sorted(block_ids)]],
                "pieces_by_block": pieces_by_block,
                "labels": [target_id],
            }
        ],
        "build": {
            "physical_error_rate": float(physical_error_rate),
            "surgery_rounds": int(surgery_rounds),
            "preserve_raw_checks": True,
            "canonical_section_variant": 0,
        },
    }
    # The archive-boundary checks above establish record integrity. The normal
    # loader then verifies closure, logical nontriviality, measurement rank,
    # no overmeasurement, and the bundled SEC/circuit restrictions.
    load_quilt(result)
    return result


def write_quilt_payload_from_selected_record(
    source: str | Path | Mapping[str, Any],
    output: str | Path,
    *,
    target_id: str | None = None,
    physical_error_rate: float = 0.003,
    surgery_rounds: int = 12,
) -> dict[str, Any]:
    payload = quilt_payload_from_selected_record(
        source,
        target_id=target_id,
        physical_error_rate=physical_error_rate,
        surgery_rounds=surgery_rounds,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
