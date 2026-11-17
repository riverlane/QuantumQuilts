"""Code definitions and GF(2) helpers for the glued basis.

The sparsity and congestion figures and the circuits are computed from this one
implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def insert(basis: dict[int, int], row: int) -> bool:
    """Add `row` to a GF(2) echelon basis; True if it was independent."""

    while row:
        pivot = row.bit_length() - 1
        if pivot not in basis:
            basis[pivot] = row
            return True
        row ^= basis[pivot]
    return False


def rank(rows) -> int:
    basis: dict[int, int] = {}
    for row in rows:
        insert(basis, row)
    return len(basis)


class Code:
    """Per-code precomputation: incidence lists and the physical check rank."""

    def __init__(self, name: str, data_root: Path | None = None):
        root = Path(data_root) if data_root is not None else DEFAULT_DATA
        definition = json.loads(
            (root / "code_definitions" / f"{name}.json").read_text()
        )
        self.name = name
        self.n = int(definition["num_qubits"])
        # 'checks' is a canonical row basis for algebra, while 'source_checks'
        # is the physical local presentation that the SkipTree condition and
        # every source-check budget refer to.  Use the latter: the detector a
        # generator becomes is the XOR of *these* checks' measurement records.
        x_rows = [int(s, 16) for s in definition["source_checks"]["x"]]
        z_rows = [int(s, 16) for s in definition["source_checks"]["z"]]
        self.num_x = len(x_rows)
        self.num_checks = len(x_rows) + len(z_rows)
        # Physical symplectic rows, in the same index order as the syndromes.
        self.rows = [r for r in x_rows] + [r << self.n for r in z_rows]
        # Checks touching qubit q, split by sector.  X checks carry only an X
        # part, Z checks only a Z part, so incidence determines commutation.
        self.x_at: list[list[int]] = [[] for _ in range(self.n)]
        self.z_at: list[list[int]] = [[] for _ in range(self.n)]
        for index, row in enumerate(x_rows):
            for q in range(self.n):
                if (row >> q) & 1:
                    self.x_at[q].append(index)
        for index, row in enumerate(z_rows):
            for q in range(self.n):
                if (row >> q) & 1:
                    self.z_at[q].append(self.num_x + index)
        self.pauli_rank = rank(self.rows)

    def check_label(self, index: int) -> tuple[str, int]:
        """The sector and within-sector index of source check `index`.

        The circuit builder names checks by sector and position, so a
        coefficient recorded here has to be translatable into that naming.
        """

        return ("X", index) if index < self.num_x else ("Z", index - self.num_x)

    def syndromes(self, pieces) -> list[int]:
        """Induced anticommutation pattern of every check, as a bitmask."""

        syndrome = [0] * self.num_checks
        for j, piece in enumerate(pieces):
            q, p = piece["qubit"], piece["pauli"]
            bit = 1 << j
            if p in ("X", "Y"):
                for index in self.z_at[q]:
                    syndrome[index] ^= bit
            if p in ("Z", "Y"):
                for index in self.x_at[q]:
                    syndrome[index] ^= bit
            if p not in ("X", "Y", "Z"):
                raise ValueError(f"unexpected stitching Pauli {p!r}")
        return syndrome


def bfs(w: int, adj, source: int):
    dist = [-1] * w
    parent: list[tuple[int, int] | None] = [None] * w
    dist[source] = 0
    queue = [source]
    while queue:
        nxt = []
        for u in queue:
            for v, slot in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, slot)
                    nxt.append(v)
        queue = nxt
    return dist, parent


def to_root(parent, target: int) -> int:
    mask = 0
    while parent[target] is not None:
        u, slot = parent[target]
        mask ^= 1 << slot
        target = u
    return mask
