"""What we are searching for: a target, and a state that might satisfy it.

A *target* says which logical class is to be measured on each block.  A *state*
is one concrete choice of physical representative per block.  The search moves
between states inside the stabiliser cosets, which never changes the target.

The state is stored as an (x_mask, z_mask) pair per block rather than a single
Pauli, because the two sectors move independently: multiplying block i by one of
its Z stabilisers changes z_mask alone.  For a pure-Z target every x_mask stays
zero, which is what the sector lock enforces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .codes_and_symmetries import Code, bit_count


@dataclass(frozen=True)
class Block:
    code: Code
    logical_x: int
    logical_z: int

    @property
    def is_pure_z(self) -> bool:
        return self.logical_x == 0

    @property
    def is_pure_x(self) -> bool:
        return self.logical_z == 0

    @property
    def is_mixed(self) -> bool:
        return not self.is_pure_z and not self.is_pure_x


@dataclass(frozen=True)
class Target:
    target_id: str
    cohort: str
    blocks: tuple[Block, ...]

    @property
    def distance(self) -> int:
        """The quilt can protect no better than its weakest constituent code."""

        return min(block.code.distance for block in self.blocks)


# A state is one (x_mask, z_mask) per block.
State = tuple[tuple[int, int], ...]


def interface_width(block_state: tuple[int, int]) -> int:
    """Number of stitching pieces on this block: one per qubit it touches."""

    x_mask, z_mask = block_state
    return bit_count(x_mask | z_mask)


def overlap(block_state: tuple[int, int]) -> int:
    """Qubits where X and Z both act, i.e. the Y sites."""

    x_mask, z_mask = block_state
    return bit_count(x_mask & z_mask)


def interface_components(block_state: tuple[int, int], code: Code) -> int:
    """Connected components of the block's interface-coordinate graph.

    Vertices are the W stitching coordinates.  An edge joins two coordinates
    whenever some *individual* source check anticommutes with exactly those two
    pieces.  Three details matter and are easy to get wrong:

      - individual checks only, never a product of them;
      - the physical `source_checks` presentation, not the row-reduced basis;
      - restrictions of weight four or six contribute nothing, so connectivity
        must be carried by the graph-like part of what is really a hypergraph.

    The deterministic SkipTree pairing spans this graph, so it exists only when
    the graph is connected.  A state with more than one component cannot be
    paired at all, which is why the search treats this as hard infeasibility
    rather than as a preference.
    """

    x_mask, z_mask = block_state
    pieces = []
    for qubit in range(code.n):
        x_bit, z_bit = (x_mask >> qubit) & 1, (z_mask >> qubit) & 1
        if x_bit or z_bit:
            pieces.append((qubit, bool(x_bit), bool(z_bit)))
    width = len(pieces)
    if width == 0:
        return 0

    parent = list(range(width))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    # an X-type check anticommutes with the Z and Y pieces, and vice versa
    for checks, hits_z_part in ((code.source_x_checks, True), (code.source_z_checks, False)):
        for check in checks:
            touched = [
                coordinate
                for coordinate, (qubit, has_x, has_z) in enumerate(pieces)
                if (check >> qubit) & 1 and (has_z if hits_z_part else has_x)
            ]
            if len(touched) != 2:
                continue
            left, right = find(touched[0]), find(touched[1])
            if left != right:
                parent[right] = left

    return len({find(vertex) for vertex in range(width)})


def labels_match(block: Block, block_state: tuple[int, int]) -> bool:
    x_mask, z_mask = block_state
    return (
        block.code.label(x_mask, "X") == block.logical_x
        and block.code.label(z_mask, "Z") == block.logical_z
    )
