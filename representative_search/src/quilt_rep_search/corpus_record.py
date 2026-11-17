"""Turning a chosen interface into a corpus record, pairing included.

The pairing is the bijection between blocks' stitching pieces.  It is derived
deterministically from the masks, by the SkipTree rule: order each block's
pieces by qubit index, build the graph whose edges are the weight-two
restrictions of individual source checks, take a minimum spanning tree, relabel
by a depth-first walk of that tree, and pair equal new coordinates across
blocks.

It is nonetheless written into the record explicitly, and it is not always the
pairing we ship.  The SkipTree rule is built to bound the weight of an H_ac
basis; it turns out to be actively poor for *distance* when a surface block
meets a single bivariate bicycle block.  Measured on bb72 SC_BB_ZZ, nine of nine
candidates that the deterministic pairing failed were rescued by the first
random re-pairing of the same representative.  So the rule is: try the
deterministic pairing, and resample it if the distance screen objects.

A resampled pairing cannot be recomputed from the masks, which is precisely why
the coordinates are written out.  The validator recomputes the deterministic
pairing and reports whether this record happens to match it, rather than
requiring that it does.
"""

from __future__ import annotations

import random
from typing import Sequence

from .codes_and_symmetries import Code
from .quilt_targets import Block, State, Target, interface_width


def pieces_in_qubit_order(x_mask: int, z_mask: int, n: int) -> list[tuple[int, str]]:
    """The stitching pieces, ordered by ascending physical qubit index."""

    pieces = []
    for qubit in range(n):
        x_bit, z_bit = (x_mask >> qubit) & 1, (z_mask >> qubit) & 1
        if x_bit and z_bit:
            pieces.append((qubit, "Y"))
        elif x_bit:
            pieces.append((qubit, "X"))
        elif z_bit:
            pieces.append((qubit, "Z"))
    return pieces


def skiptree_order(code: Code, x_mask: int, z_mask: int) -> list[tuple[int, str]]:
    """Reorder a block's pieces by a depth-first walk of the source-check tree."""

    pieces = pieces_in_qubit_order(x_mask, z_mask, code.n)
    width = len(pieces)
    if width == 0:
        raise ValueError("empty representative")

    # cheapest individual source check supplying each weight-two edge
    best: dict[tuple[int, int], tuple[int, str, int]] = {}
    for axis, checks, hits_z in (("x", code.source_x_checks, True),
                                 ("z", code.source_z_checks, False)):
        for index, check in enumerate(checks):
            touched = [
                coordinate
                for coordinate, (qubit, pauli) in enumerate(pieces)
                if (check >> qubit) & 1 and (pauli in "ZY" if hits_z else pauli in "XY")
            ]
            if len(touched) != 2:
                continue
            edge = (min(touched), max(touched))
            key = (int(check).bit_count(), axis.upper(), index)
            if edge not in best or key < best[edge]:
                best[edge] = key

    parent = list(range(width))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    selected = []
    for (left, right), (weight, axis, index) in sorted(
        best.items(), key=lambda row: (row[1][0], row[0][0], row[0][1], row[1][1], row[1][2])
    ):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
            selected.append((left, right))
    if len(selected) != width - 1:
        raise ValueError("individual-source-check restriction graph is disconnected")

    adjacency: list[list[int]] = [[] for _ in range(width)]
    for left, right in selected:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbours in adjacency:
        neighbours.sort()

    order: list[int] = []
    stack = [(0, -1)]
    while stack:
        vertex, previous = stack.pop()
        order.append(vertex)
        for neighbour in reversed(adjacency[vertex]):
            if neighbour != previous:
                stack.append((neighbour, vertex))
    return [pieces[coordinate] for coordinate in order]


def block_orders(target: Target, state: State, *, pairing_seed: int | None) -> list[list[tuple[int, str]]]:
    """Per-block piece orderings, which together define the pairing.

    Block zero always takes the deterministic SkipTree order: it fixes the
    coordinate labels, and relabelling every block alike changes nothing.  The
    remaining blocks take the same rule, or a seeded shuffle when we are
    resampling.
    """

    orders = []
    for index, (block, (x_mask, z_mask)) in enumerate(zip(target.blocks, state, strict=True)):
        if pairing_seed is None or index == 0:
            orders.append(skiptree_order(block.code, x_mask, z_mask))
        else:
            pieces = pieces_in_qubit_order(x_mask, z_mask, block.code.n)
            random.Random(f"{pairing_seed}:{index}").shuffle(pieces)
            orders.append(pieces)
    return orders


def build_record(
    target: Target,
    state: State,
    *,
    scale: str,
    telemetry: dict,
    pairing_seed: int | None = None,
) -> dict:
    """Assemble one record from a chosen interface and pairing."""

    widths = {interface_width(block_state) for block_state in state}
    if len(widths) != 1:
        raise ValueError(f"interface widths differ: {sorted(widths)}")
    width = widths.pop()

    orders = block_orders(target, state, pairing_seed=pairing_seed)
    blocks = []
    coordinates = []
    for index, (block, (x_mask, z_mask)) in enumerate(zip(target.blocks, state, strict=True)):
        ordered = orders[index]
        hex_width = (block.code.n + 3) // 4
        key_width = (block.code.k + 3) // 4
        blocks.append({
            "block_index": index,
            "code": block.code.code_id,
            "logical_x": f"0x{block.logical_x:0{key_width}x}",
            "logical_z": f"0x{block.logical_z:0{key_width}x}",
            "num_qubits": block.code.n,
            "pieces": [{"pauli": pauli, "piece_index": coordinate, "qubit": qubit}
                       for coordinate, (qubit, pauli) in enumerate(ordered)],
            "x_mask": f"0x{x_mask:0{hex_width}x}",
            "z_mask": f"0x{z_mask:0{hex_width}x}",
        })
        coordinates.append([qubit for qubit, _pauli in ordered])

    return {
        "schema": "quilt-representative-v6",
        "scale": scale,
        "cohort": target.cohort,
        "target_id": target.target_id,
        "operation_threshold": target.distance,
        "blocks": blocks,
        "pairing": {
            "strategy": "skiptree-deterministic" if pairing_seed is None else "resampled",
            "resample_seed": pairing_seed,
            "width": width,
            "coordinates": coordinates,
        },
        "telemetry": telemetry,
    }
