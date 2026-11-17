"""Exact GF(2) structure and observable ledger for Pauli code quilts.

The physical Pauli convention is the one used by :mod:`quilts.symplectic`:
an operator on ``n`` qubits is an integer whose low ``n`` bits are its X
support and whose next ``n`` bits are its Z support.

A :class:`StitchingFamily` consists of disjoint local Pauli pieces on every
block and a list of binary incidence patterns.  For every local-piece index
``a`` and incidence pattern ``b``, the corresponding global gauge generator
is the product of piece ``a`` over the blocks selected by ``b``.  Multiplying
these generators over ``a`` gives the intended measured logical product.

The resulting :class:`QuiltAlgebraLedger` records the exact gauge group and
center, the reliable and measured central sectors, symplectically paired bare
logical bases before and after stitching, and explicit classical and memory
observable representatives.  Every reported object depends only on row
spaces, so changing the gauge-generator basis cannot change the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ._gf2 import (
    all_ones,
    nullspace,
    reduce_with_basis,
    row_reduce,
)
from ._symplectic import (
    GeneralSubsystem,
    MeasurementAnalysis,
    analyze_measurement,
    build_general_subsystem,
    gauge_center,
    pauli,
    pauli_weight,
    quotient_complement,
    rows_commute,
    symplectic_centralizer,
    symplectic_product,
    xor_rows,
    x_part,
    z_part,
)


def _basis(rows: Iterable[int], width: int) -> tuple[int, ...]:
    return tuple(row_reduce((int(row) for row in rows), width)[0])


def _same_span(left: Iterable[int], right: Iterable[int], width: int) -> bool:
    return _basis(left, width) == _basis(right, width)


def _in_span(rows: Sequence[int], row: int, width: int) -> bool:
    return reduce_with_basis(rows, int(row), width) == 0


def _subspace_intersection(
    left_rows: Sequence[int],
    right_rows: Sequence[int],
    width: int,
) -> tuple[int, ...]:
    """Return a canonical basis for the intersection of two row spaces."""

    left = _basis(left_rows, width)
    right = _basis(right_rows, width)
    combined = (*left, *right)
    equations = [
        sum(((row >> column) & 1) << index for index, row in enumerate(combined))
        for column in range(width)
    ]
    relations = nullspace(equations, len(combined))
    left_mask = all_ones(len(left))
    return _basis(
        (xor_rows(relation & left_mask, left) for relation in relations),
        width,
    )


def blockwise_css_basis(
    rows: Sequence[int],
    blocks: Sequence["StabilizerBlock"],
) -> tuple[int, ...]:
    """Choose a basis whose restriction to every block is pure X or pure Z.

    The requested row space need not be CSS globally: a selected row may be
    X-type on one block and Z-type on another.  We enumerate the two local
    type choices per block, intersect the row space with each resulting
    coordinate subspace, and retain independent rows without row-operating
    across different local-type choices.

    Raises:
        ValueError: if blockwise-CSS rows do not span the requested space.
    """

    total_n = sum(block.n for block in blocks)
    width = 2 * total_n
    source = _basis(rows, width)
    if not source:
        return ()
    offsets = _block_offsets(blocks)
    candidates: list[int] = []
    for coloring in range(1 << len(blocks)):
        forbidden_columns: list[int] = []
        for block_index, (block, offset) in enumerate(
            zip(blocks, offsets, strict=True)
        ):
            use_z = (coloring >> block_index) & 1
            start = offset if use_z else total_n + offset
            forbidden_columns.extend(range(start, start + block.n))
        equations = [
            sum(
                ((row >> column) & 1) << row_index
                for row_index, row in enumerate(source)
            )
            for column in forbidden_columns
        ]
        coefficients = nullspace(equations, len(source))
        candidates.extend(
            _basis(
                (xor_rows(coefficient, source) for coefficient in coefficients),
                width,
            )
        )

    chosen: list[int] = []
    span: list[int] = []
    for candidate in candidates:
        if reduce_with_basis(span, candidate, width):
            chosen.append(candidate)
            span = list(_basis((*span, candidate), width))
    if not _same_span(chosen, source, width):
        raise ValueError(
            "the requested space does not admit a blockwise-CSS generating set"
        )
    return tuple(chosen)


def blockwise_pauli_types(
    row: int,
    blocks: Sequence["StabilizerBlock"],
) -> tuple[str, ...]:
    """Return ``I``, ``X``, ``Z``, or ``M`` for each block restriction."""

    total_n = sum(block.n for block in blocks)
    result: list[str] = []
    for block, offset in zip(blocks, _block_offsets(blocks), strict=True):
        mask = all_ones(block.n)
        local_x = (x_part(int(row), total_n) >> offset) & mask
        local_z = (z_part(int(row), total_n) >> offset) & mask
        if not local_x and not local_z:
            result.append("I")
        elif local_x and not local_z:
            result.append("X")
        elif local_z and not local_x:
            result.append("Z")
        else:
            result.append("M")
    return tuple(result)


def _quotient_basis(
    candidates: Sequence[int],
    denominator: Sequence[int],
    width: int,
) -> tuple[int, ...]:
    """Choose candidate rows whose cosets basis the requested quotient."""

    span = list(_basis(denominator, width))
    chosen: list[int] = []
    for candidate in candidates:
        candidate = int(candidate)
        if reduce_with_basis(span, candidate, width):
            chosen.append(candidate)
            span = list(_basis([*span, candidate], width))
    return tuple(chosen)


def _commutator_rows(
    left_rows: Sequence[int],
    right_rows: Sequence[int],
    n: int,
) -> tuple[int, ...]:
    """Return the binary commutator matrix, packed one row per left operand."""

    return tuple(
        sum(
            symplectic_product(left, right, n) << right_index
            for right_index, right in enumerate(right_rows)
        )
        for left in left_rows
    )


def _block_offsets(blocks: Sequence["StabilizerBlock"]) -> tuple[int, ...]:
    offsets: list[int] = []
    offset = 0
    for block in blocks:
        offsets.append(offset)
        offset += block.n
    return tuple(offsets)


def embed_block_pauli(
    row: int,
    *,
    block_n: int,
    offset: int,
    total_n: int,
) -> int:
    """Embed a local packed Pauli into a heterogeneous global register."""

    if offset < 0 or offset + block_n > total_n:
        raise ValueError("block does not fit inside the global register")
    return pauli(
        x_part(int(row), block_n) << offset,
        z_part(int(row), block_n) << offset,
        total_n,
    )


def _extract_block_pauli(
    row: int,
    *,
    block_n: int,
    offset: int,
    total_n: int,
) -> int:
    mask = all_ones(block_n)
    return pauli(
        (x_part(int(row), total_n) >> offset) & mask,
        (z_part(int(row), total_n) >> offset) & mask,
        block_n,
    )


def block_permutation_pauli(
    row: int,
    blocks: Sequence["StabilizerBlock"],
    order: Sequence[int],
) -> int:
    """Move a global Pauli under a permutation of heterogeneous blocks.

    ``order[new_index]`` is the old block placed at ``new_index``.
    """

    order = tuple(int(index) for index in order)
    if sorted(order) != list(range(len(blocks))):
        raise ValueError("order must be a permutation of all block indices")
    old_offsets = _block_offsets(blocks)
    new_blocks = tuple(blocks[index] for index in order)
    new_offsets = _block_offsets(new_blocks)
    total_n = sum(block.n for block in blocks)
    out = 0
    for new_index, old_index in enumerate(order):
        block = blocks[old_index]
        local = _extract_block_pauli(
            row,
            block_n=block.n,
            offset=old_offsets[old_index],
            total_n=total_n,
        )
        out ^= embed_block_pauli(
            local,
            block_n=block.n,
            offset=new_offsets[new_index],
            total_n=total_n,
        )
    return out


@dataclass(frozen=True)
class LocalClifford:
    """A phase-free single-qubit Clifford specified by images of X and Z."""

    image_x: int
    image_z: int
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.image_x not in (1, 2, 3) or self.image_z not in (1, 2, 3):
            raise ValueError("single-qubit Clifford images must be nonidentity Paulis")
        if symplectic_product(self.image_x, self.image_z, 1) != 1:
            raise ValueError("images of X and Z must anticommute")

    @classmethod
    def named(cls, name: str) -> "LocalClifford":
        """Return one of the common phase-free single-qubit Clifford maps."""

        key = str(name).upper()
        maps = {
            "I": (1, 2),
            "H": (2, 1),
            "S": (3, 2),
            "S_DAG": (3, 2),
            "HS": (2, 3),
            "SH": (3, 1),
            "HSH": (1, 3),
            "H_XY": (3, 2),
            "H_YZ": (1, 3),
        }
        try:
            image_x, image_z = maps[key]
        except KeyError as exc:
            raise ValueError(f"unknown local Clifford {name!r}") from exc
        return cls(image_x=image_x, image_z=image_z, name=key)

    def apply(self, local_pauli: int) -> int:
        """Apply the binary linear map to I, X, Z, or Y."""

        local_pauli = int(local_pauli)
        if local_pauli < 0 or local_pauli > 3:
            raise ValueError("a one-qubit Pauli must be in the range 0..3")
        out = 0
        if local_pauli & 1:
            out ^= self.image_x
        if local_pauli & 2:
            out ^= self.image_z
        return out


def _coerce_clifford(value: LocalClifford | str) -> LocalClifford:
    return value if isinstance(value, LocalClifford) else LocalClifford.named(value)


def apply_local_cliffords(
    row: int,
    *,
    n: int,
    maps: Sequence[LocalClifford | str],
) -> int:
    """Apply a tensor product of local Cliffords to a packed Pauli.

    Valid local Clifford maps preserve both physical Pauli weight and the
    symplectic product.  Weight preservation is asserted on every call; the
    latter follows from the checked one-qubit images and is also exercised by
    :meth:`QuiltSpec.transformed`.
    """

    if len(maps) != n:
        raise ValueError("one local Clifford is required per qubit")
    x = x_part(int(row), n)
    z = z_part(int(row), n)
    out_x = 0
    out_z = 0
    for qubit, raw_map in enumerate(maps):
        clifford = _coerce_clifford(raw_map)
        local = ((x >> qubit) & 1) | (((z >> qubit) & 1) << 1)
        image = clifford.apply(local)
        out_x |= (image & 1) << qubit
        out_z |= ((image >> 1) & 1) << qubit
    out = pauli(out_x, out_z, n)
    if pauli_weight(out, n) != pauli_weight(int(row), n):
        raise AssertionError("a local Clifford changed physical Pauli weight")
    return out


@dataclass(frozen=True)
class SymplecticBasis:
    """A full paired basis for a nondegenerate binary symplectic space."""

    pairs: tuple[tuple[int, int], ...]
    n: int
    modulus_basis: tuple[int, ...] = ()

    @property
    def rows(self) -> tuple[int, ...]:
        return tuple(row for pair in self.pairs for row in pair)

    @property
    def qubits(self) -> int:
        return len(self.pairs)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[int],
        *,
        n: int,
        modulus_basis: Sequence[int] = (),
    ) -> "SymplecticBasis":
        """Pair a quotient basis using exact symplectic Gram-Schmidt."""

        width = 2 * n
        modulus = _basis(modulus_basis, width)
        work = list(_quotient_basis(tuple(int(row) for row in rows), modulus, width))
        if len(_basis([*modulus, *work], width)) != len(modulus) + len(work):
            raise ValueError("logical representatives are dependent modulo the modulus")
        original_span = _basis([*modulus, *work], width)
        pairs: list[tuple[int, int]] = []

        while work:
            left = work.pop(0)
            partner_index = next(
                (
                    index
                    for index, right in enumerate(work)
                    if symplectic_product(left, right, n)
                ),
                None,
            )
            if partner_index is None:
                raise ValueError("logical quotient has a degenerate symplectic form")
            right = work.pop(partner_index)
            orthogonalized: list[int] = []
            for row in work:
                adjusted = row
                if symplectic_product(adjusted, right, n):
                    adjusted ^= left
                if symplectic_product(adjusted, left, n):
                    adjusted ^= right
                orthogonalized.append(adjusted)
            work = orthogonalized
            pairs.append((left, right))

        result = cls(pairs=tuple(pairs), n=n, modulus_basis=modulus)
        if _basis([*modulus, *result.rows], width) != original_span:
            raise AssertionError("symplectic pairing changed the represented quotient")
        for pair_index, (left, right) in enumerate(result.pairs):
            if symplectic_product(left, right, n) != 1:
                raise AssertionError("a logical pair does not anticommute")
            for other_index, (other_left, other_right) in enumerate(result.pairs):
                if pair_index == other_index:
                    continue
                if symplectic_product(left, other_left, n):
                    raise AssertionError("distinct logical pairs do not commute")
                if symplectic_product(left, other_right, n):
                    raise AssertionError("distinct logical pairs do not commute")
                if symplectic_product(right, other_left, n):
                    raise AssertionError("distinct logical pairs do not commute")
                if symplectic_product(right, other_right, n):
                    raise AssertionError("distinct logical pairs do not commute")
        return result


@dataclass(frozen=True)
class StabilizerBlock:
    """One general stabilizer-code block in packed binary Pauli form."""

    name: str
    n: int
    stabilizers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a block name is required")
        if self.n <= 0:
            raise ValueError("a block must contain at least one qubit")
        limit = 1 << (2 * self.n)
        if any(int(row) <= 0 or int(row) >= limit for row in self.stabilizers):
            raise ValueError("stabilizers must be nonzero Paulis of the declared width")
        basis = _basis(self.stabilizers, 2 * self.n)
        if not rows_commute(basis, self.n):
            raise ValueError("stabilizer generators must commute")
        object.__setattr__(self, "stabilizers", basis)

    @classmethod
    def css(
        cls,
        *,
        name: str,
        n: int,
        x_checks: Sequence[int] = (),
        z_checks: Sequence[int] = (),
    ) -> "StabilizerBlock":
        """Construct a CSS block from n-bit X- and Z-check masks."""

        mask = all_ones(n)
        if any(int(row) <= 0 or int(row) & ~mask for row in (*x_checks, *z_checks)):
            raise ValueError("CSS check support exceeds the declared block")
        return cls(
            name=name,
            n=n,
            stabilizers=tuple(
                [pauli(int(row), 0, n) for row in x_checks]
                + [pauli(0, int(row), n) for row in z_checks]
            ),
        )

    @property
    def logical_qubits(self) -> int:
        return self.n - len(self.stabilizers)

    @property
    def bare_logicals(self) -> SymplecticBasis:
        subsystem = build_general_subsystem(
            n=self.n,
            base_stabilizers=self.stabilizers,
            added_gauges=(),
        )
        return SymplecticBasis.from_rows(
            subsystem.bare_logical_basis,
            n=self.n,
            modulus_basis=subsystem.center_basis,
        )

    def transformed(
        self,
        maps: Sequence[LocalClifford | str],
        *,
        name: str | None = None,
    ) -> "StabilizerBlock":
        transformed = tuple(
            apply_local_cliffords(row, n=self.n, maps=maps)
            for row in self.stabilizers
        )
        for left in self.stabilizers:
            for right in self.stabilizers:
                before = symplectic_product(left, right, self.n)
                after = symplectic_product(
                    apply_local_cliffords(left, n=self.n, maps=maps),
                    apply_local_cliffords(right, n=self.n, maps=maps),
                    self.n,
                )
                if before != after:
                    raise AssertionError("local Clifford failed to preserve commutation")
        return StabilizerBlock(
            name=name or f"{self.name}[LC]",
            n=self.n,
            stabilizers=transformed,
        )


@dataclass(frozen=True)
class StitchingFamily:
    """Disjoint local pieces and the incidence patterns measured from them.

    ``local_pieces_by_block[i][a]`` is local piece ``a`` on block ``i``.
    Bit ``i`` of an incidence pattern selects block ``i``.  A block absent
    from every incidence pattern of the family is represented by zero local
    pieces.  The same nonzero local piece may occur in multiple global gauge
    rows when incidence codewords overlap, as in ``B=<110,101>``.
    """

    name: str
    local_pieces_by_block: tuple[tuple[int, ...], ...]
    incidence_patterns: tuple[int, ...]

    @property
    def piece_count(self) -> int:
        return len(self.local_pieces_by_block[0]) if self.local_pieces_by_block else 0


@dataclass(frozen=True)
class BlockwiseCssGenerator:
    """A reliable-centre generator with CSS restrictions on every block."""

    row: int
    block_types: tuple[str, ...]
    constituent_check_count: int
    kind: str


class _SparseSyndromeMap:
    """A sparse section and kernel for one pure CSS check family."""

    def __init__(
        self,
        *,
        checks: Sequence[int],
        syndromes: Sequence[int],
        pauli_width: int,
        syndrome_width: int,
    ) -> None:
        if len(checks) != len(syndromes):
            raise ValueError("checks and syndromes must have equal length")
        self.checks = tuple(map(int, checks))
        self.syndromes = tuple(map(int, syndromes))
        self.pauli_width = int(pauli_width)
        self.syndrome_width = int(syndrome_width)
        self._pivots: dict[int, tuple[int, int]] = {}
        fundamental_relations: list[int] = []
        zero_relations: list[int] = []

        for index, syndrome in enumerate(self.syndromes):
            reduced = syndrome
            coefficients = 1 << index
            while reduced:
                pivot = reduced.bit_length() - 1
                existing = self._pivots.get(pivot)
                if existing is None:
                    self._pivots[pivot] = (reduced, coefficients)
                    break
                reduced ^= existing[0]
                coefficients ^= existing[1]
            if not reduced:
                if syndrome:
                    fundamental_relations.append(coefficients)
                else:
                    zero_relations.append(coefficients)

        self.image_basis = tuple(row for row, _coefficients in self._pivots.values())
        self._nonzero_kernel_basis = self._minimum_check_basis(
            fundamental_relations
        )
        self._section_kernel_span: tuple[int, ...] | None = None
        self.kernel_coefficients = (
            *zero_relations,
            *self._nonzero_kernel_basis,
        )
        expected_kernel_rank = len(self.checks) - len(self.image_basis)
        if len(self.kernel_coefficients) != expected_kernel_rank:
            raise AssertionError("syndrome-map kernel rank mismatch")

    def _physical_row(self, coefficients: int) -> int:
        return xor_rows(coefficients, self.checks)

    def _coefficient_cost(self, coefficients: int) -> tuple[int, int, int]:
        row = self._physical_row(coefficients)
        return (
            int(coefficients).bit_count(),
            pauli_weight(row, self.pauli_width // 2),
            int(coefficients),
        )

    def _graph_candidates(self) -> tuple[int, ...]:
        """Shortest-path cycle pool over the weight-two syndrome rows.

        Rows whose syndrome has weight two are edges of a graph on the
        stitching coordinates, and the part of the kernel they generate is
        that graph's cycle space. Horton's construction supplies a pool
        containing a minimum cycle basis, which the exhaustive span cannot
        reach once the relation count leaves brute force behind.
        """

        edges: list[tuple[int, int, int]] = []
        for index, syndrome in enumerate(self.syndromes):
            if int(syndrome).bit_count() != 2:
                continue
            low = (syndrome & -syndrome).bit_length() - 1
            edges.append((low, int(syndrome).bit_length() - 1, index))
        if not edges:
            return ()
        vertices = sorted({v for edge in edges for v in edge[:2]})
        position = {v: i for i, v in enumerate(vertices)}
        adjacency: list[list[tuple[int, int]]] = [[] for _ in vertices]
        for slot, (low, high, _index) in enumerate(edges):
            adjacency[position[low]].append((position[high], slot))
            adjacency[position[high]].append((position[low], slot))

        pool: set[int] = set()
        for source in range(len(vertices)):
            parent: list[tuple[int, int] | None] = [None] * len(vertices)
            seen = [False] * len(vertices)
            seen[source] = True
            frontier = [source]
            while frontier:
                following: list[int] = []
                for u in frontier:
                    for v, slot in adjacency[u]:
                        if not seen[v]:
                            seen[v] = True
                            parent[v] = (u, slot)
                            following.append(v)
                frontier = following

            def to_source(node: int) -> int:
                mask = 0
                step = parent[node]
                while step is not None:
                    mask ^= 1 << step[1]
                    step = parent[step[0]]
                return mask

            for slot, (low, high, _index) in enumerate(edges):
                if not (seen[position[low]] and seen[position[high]]):
                    continue
                mask = to_source(position[low]) ^ to_source(position[high])
                mask ^= 1 << slot
                if mask:
                    pool.add(mask)

        # Rows of syndrome weight four or more are not edges, so no cycle
        # reaches them. Close each one against the graph instead: the row plus
        # a least-cost pairing of its syndrome support by shortest paths.
        parents: list[list[tuple[int, int] | None]] = []
        for source in range(len(vertices)):
            parent = [None] * len(vertices)
            seen = [False] * len(vertices)
            seen[source] = True
            frontier = [source]
            while frontier:
                following = []
                for u in frontier:
                    for v, slot in adjacency[u]:
                        if not seen[v]:
                            seen[v] = True
                            parent[v] = (u, slot)
                            following.append(v)
                frontier = following
            parents.append(parent)

        def route(parent: list[tuple[int, int] | None], node: int) -> int | None:
            mask = 0
            step = parent[node]
            while step is not None:
                mask ^= 1 << step[1]
                step = parent[step[0]]
            return mask

        def pair_up(terminals: tuple[int, ...]) -> int | None:
            if not terminals:
                return 0
            head, rest = terminals[0], terminals[1:]
            best: int | None = None
            best_size = None
            for i, other in enumerate(rest):
                if parents[head][other] is None and head != other:
                    continue
                joined = route(parents[head], other)
                tail = pair_up(rest[:i] + rest[i + 1:])
                if joined is None or tail is None:
                    continue
                combined = joined ^ tail
                size = combined.bit_count()
                if best_size is None or size < best_size:
                    best, best_size = combined, size
            return best

        def to_coefficients(mask: int) -> int:
            coefficients = 0
            rest = mask
            while rest:
                slot = (rest & -rest).bit_length() - 1
                coefficients ^= 1 << edges[slot][2]
                rest &= rest - 1
            return coefficients

        translated = [to_coefficients(mask) for mask in pool]

        for index, syndrome in enumerate(self.syndromes):
            if int(syndrome).bit_count() < 4:
                continue
            terminals: list[int] = []
            rest = int(syndrome)
            outside = False
            while rest:
                bit = (rest & -rest).bit_length() - 1
                if bit not in position:
                    outside = True
                    break
                terminals.append(position[bit])
                rest &= rest - 1
            if outside:
                continue
            joined = pair_up(tuple(terminals))
            if joined is not None:
                translated.append(to_coefficients(joined) ^ (1 << index))

        return tuple(translated)

    def _reduce_pairwise(self, chosen: Sequence[int]) -> tuple[int, ...]:
        """Replace b_i by b_i + b_j whenever that is cheaper. Span preserved."""

        rows = list(map(int, chosen))
        for _sweep in range(4):
            improved = False
            for i, current in enumerate(rows):
                cost = self._coefficient_cost(current)
                for j, other in enumerate(rows):
                    if i == j:
                        continue
                    candidate = current ^ other
                    if not candidate:
                        continue
                    candidate_cost = self._coefficient_cost(candidate)
                    if candidate_cost < cost:
                        rows[i] = current = candidate
                        cost = candidate_cost
                        improved = True
            if not improved:
                break
        return tuple(rows)

    def _minimum_check_basis(self, relations: Sequence[int]) -> tuple[int, ...]:
        relations = tuple(map(int, relations))
        if not relations:
            return ()
        if len(relations) > 18:
            candidates = relations + self._graph_candidates()
        else:
            candidates = tuple(
                xor_rows(mask, relations)
                for mask in range(1, 1 << len(relations))
            )
        span: list[int] = []
        chosen: list[int] = []
        for candidate in sorted(
            set(candidates),
            key=self._coefficient_cost,
        ):
            if reduce_with_basis(span, candidate, len(self.checks)):
                chosen.append(candidate)
                span = list(_basis((*span, candidate), len(self.checks)))
                if len(chosen) == len(relations):
                    break
        if len(chosen) != len(relations):
            raise AssertionError("failed to span the nonzero syndrome kernel")
        return self._reduce_pairwise(chosen)

    def section(self, syndrome: int) -> tuple[int, int]:
        """Return a minimum-check representative and its check coefficients."""

        reduced = int(syndrome)
        coefficients = 0
        while reduced:
            pivot = reduced.bit_length() - 1
            existing = self._pivots.get(pivot)
            if existing is None:
                raise ValueError("syndrome is outside the image")
            reduced ^= existing[0]
            coefficients ^= existing[1]

        kernel = self._nonzero_kernel_basis
        if len(kernel) <= 18:
            if self._section_kernel_span is None:
                span = [0]
                for row in kernel:
                    span += [value ^ row for value in span]
                self._section_kernel_span = tuple(span)
            # Check count is the primary cost. Expand physical Paulis only for
            # its minimizers, retaining the exact physical-weight/integer ties.
            minimum = len(self.checks) + 1
            finalists: list[int] = []
            for relation in self._section_kernel_span:
                candidate = coefficients ^ relation
                weight = candidate.bit_count()
                if weight < minimum:
                    minimum = weight
                    finalists = [candidate]
                elif weight == minimum:
                    finalists.append(candidate)
            coefficients = min(finalists, key=self._coefficient_cost)
        else:
            # Too many kernel elements to expand the span, so descend greedily
            # instead of leaving the elimination representative untouched.
            cost = self._coefficient_cost(coefficients)
            for _sweep in range(4):
                improved = False
                for relation in kernel:
                    candidate = coefficients ^ relation
                    if not candidate:
                        continue
                    candidate_cost = self._coefficient_cost(candidate)
                    if candidate_cost < cost:
                        coefficients = candidate
                        cost = candidate_cost
                        improved = True
                if not improved:
                    break
        return self._physical_row(coefficients), coefficients


def _independent_preserving_order(
    rows: Sequence[int],
    width: int,
) -> tuple[int, ...]:
    chosen: list[int] = []
    span: list[int] = []
    for row in map(int, rows):
        if reduce_with_basis(span, row, width):
            chosen.append(row)
            span = list(_basis((*span, row), width))
    return tuple(chosen)


def _embed_local_pauli(
    row: int,
    *,
    block_index: int,
    blocks: Sequence["StabilizerBlock"],
) -> int:
    total_n = sum(block.n for block in blocks)
    block = blocks[block_index]
    offset = _block_offsets(blocks)[block_index]
    return pauli(
        x_part(int(row), block.n) << offset,
        z_part(int(row), block.n) << offset,
        total_n,
    )


def _local_stitch_syndrome(
    row: int,
    *,
    block_index: int,
    blocks: Sequence["StabilizerBlock"],
    stitching_rows: Sequence[int],
) -> int:
    total_n = sum(block.n for block in blocks)
    block = blocks[block_index]
    offset = _block_offsets(blocks)[block_index]
    mask = all_ones(block.n)
    syndrome = 0
    for stitch_index, stitch in enumerate(stitching_rows):
        local = pauli(
            (x_part(int(stitch), total_n) >> offset) & mask,
            (z_part(int(stitch), total_n) >> offset) & mask,
            block.n,
        )
        syndrome |= symplectic_product(int(row), local, block.n) << stitch_index
    return syndrome


def _interface_bfs(
    width: int, adjacency: Sequence[Sequence[tuple[int, int]]], source: int
) -> tuple[list[int], list[tuple[int, int] | None]]:
    """Breadth-first tree of the glued interface graph rooted at one coordinate."""

    distance = [-1] * width
    parent: list[tuple[int, int] | None] = [None] * width
    distance[source] = 0
    frontier = [source]
    while frontier:
        following: list[int] = []
        for node in frontier:
            for neighbour, slot in adjacency[node]:
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[node] + 1
                    parent[neighbour] = (node, slot)
                    following.append(neighbour)
        frontier = following
    return distance, parent


def _path_to_root(parent: Sequence[tuple[int, int] | None], target: int) -> int:
    mask = 0
    while parent[target] is not None:
        node, slot = parent[target]
        mask ^= 1 << slot
        target = node
    return mask


def glued_center_candidates(
    *,
    rows: Sequence[int],
    syndromes: Sequence[int],
    pauli_width: int,
) -> tuple[tuple[int, int], ...]:
    """Candidate inherited-centre generators from the glued interface graph.

    The stitch syndrome of a source check records which interface coordinates it
    anticommutes with.  Because those coordinates are shared by every block, a
    check of syndrome weight two is an edge of a single graph spanning the whole
    quilt, and any cycle of that graph is a centre element.  Working on the glued
    graph rather than one block at a time is what lets a cross-block generator be
    found directly instead of by pairing two separately chosen sections.

    Returns ``(coefficients, row)`` pairs: the set of source checks used, and
    their product.  Three families are produced.

    * Checks of syndrome weight zero, which are already central.
    * Fundamental cycles of the graph, taken from a breadth-first tree rooted at
      every coordinate in turn, plus the two-edge cycles of parallel edges.  This
      is Horton's minimum cycle basis pool.
    * For a check of syndrome weight three or more, the check together with a
      least-cost pairing of its syndrome support by tree paths, that is, a
      minimum T-join closing it into the centre.
    """

    width = max((syndrome.bit_length() for syndrome in syndromes), default=0)
    free: list[int] = []
    edges: list[tuple[int, int, int]] = []
    hyper: list[tuple[int, tuple[int, ...]]] = []
    for index, syndrome in enumerate(syndromes):
        count = syndrome.bit_count()
        if count == 0:
            free.append(index)
            continue
        terminals: list[int] = []
        rest = syndrome
        while rest:
            terminals.append((rest & -rest).bit_length() - 1)
            rest &= rest - 1
        if count == 2:
            edges.append((terminals[0], terminals[1], index))
        elif count > 2:
            hyper.append((index, tuple(terminals)))

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(width)]
    for slot, (left, right, _index) in enumerate(edges):
        adjacency[left].append((right, slot))
        adjacency[right].append((left, slot))
    trees = [_interface_bfs(width, adjacency, node) for node in range(width)]

    def product(slots: Iterable[int], extra: int | None = None) -> tuple[int, int]:
        coefficients = 0
        for slot in slots:
            coefficients ^= 1 << edges[slot][2]
        if extra is not None:
            coefficients ^= 1 << extra
        row = 0
        rest = coefficients
        while rest:
            index = (rest & -rest).bit_length() - 1
            row ^= int(rows[index])
            rest &= rest - 1
        return coefficients, row

    candidates: list[tuple[int, int]] = []
    for index in free:
        if int(rows[index]):
            candidates.append((1 << index, int(rows[index])))

    cycle_masks: set[int] = set()
    for _distance, parent in trees:
        for slot, (left, right, _index) in enumerate(edges):
            mask = _path_to_root(parent, left) ^ _path_to_root(parent, right)
            mask ^= 1 << slot
            if mask:
                cycle_masks.add(mask)
    first_seen: dict[tuple[int, int], int] = {}
    for slot, (left, right, _index) in enumerate(edges):
        key = (min(left, right), max(left, right))
        if key in first_seen:
            cycle_masks.add((1 << slot) | (1 << first_seen[key]))
        else:
            first_seen[key] = slot
    for mask in cycle_masks:
        slots: list[int] = []
        rest = mask
        while rest:
            slots.append((rest & -rest).bit_length() - 1)
            rest &= rest - 1
        coefficients, row = product(slots)
        if row:
            candidates.append((coefficients, row))

    distances = [tree[0] for tree in trees]
    parents = [tree[1] for tree in trees]

    def route(source: int, target: int) -> list[int]:
        slots: list[int] = []
        parent = parents[source]
        node = target
        while parent[node] is not None:
            previous, slot = parent[node]
            slots.append(slot)
            node = previous
        return slots

    def cheapest_join(terminals: tuple[int, ...]) -> list[int] | None:
        """Least-cost pairing of the syndrome support by tree paths."""

        if not terminals:
            return []
        head, rest = terminals[0], terminals[1:]
        best: list[int] | None = None
        for position, other in enumerate(rest):
            if distances[head][other] < 0:
                continue
            tail = cheapest_join(rest[:position] + rest[position + 1 :])
            if tail is None:
                continue
            candidate = route(head, other) + tail
            if best is None or len(candidate) < len(best):
                best = candidate
        return best

    for index, terminals in hyper:
        slots = cheapest_join(terminals)
        if slots is None:
            continue
        coefficients, row = product(slots, extra=index)
        if row:
            candidates.append((coefficients, row))

    return tuple(candidates)


def select_sparse_basis(
    candidates: Sequence[tuple[int, int]],
    *,
    required: int,
    pauli_width: int,
    congestion_aware: bool = True,
) -> tuple[tuple[int, int], ...]:
    """Greedy independent selection, cheapest first.

    Sorting by physical weight and taking the first independent generator is the
    matroid greedy, so the chosen basis minimises its own maximum weight over the
    candidate pool.  Ties are then broken by current qubit load: a sparse row
    basis can still make a poor detector graph if one qubit is reused by many
    rows, and breaking ties this way costs nothing in weight.
    """

    n = pauli_width // 2

    def support(row: int) -> int:
        return (row & ((1 << n) - 1)) | (row >> n)

    basis: dict[int, int] = {}
    chosen: list[tuple[int, int]] = []

    def admit(row: int) -> bool:
        reduced = row
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = reduced
                return True
            reduced ^= basis[pivot]
        return False

    if not congestion_aware:
        for coefficients, row in sorted(
            candidates,
            key=lambda item: (pauli_weight(item[1], n), item[0].bit_count(), item[1]),
        ):
            if admit(row):
                chosen.append((coefficients, row))
                if len(chosen) == required:
                    break
        return tuple(chosen)

    load: dict[int, int] = {}
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coefficients, row in candidates:
        buckets.setdefault(
            (pauli_weight(row, n), coefficients.bit_count()), []
        ).append((coefficients, row))
    for key in sorted(buckets):
        bucket = buckets[key]

        def loaded(item: tuple[int, int]) -> tuple[int, int]:
            mask, total = support(item[1]), 0
            while mask:
                qubit = (mask & -mask).bit_length() - 1
                total += load.get(qubit, 0)
                mask &= mask - 1
            return total, item[1]

        for coefficients, row in sorted(bucket, key=loaded):
            if admit(row):
                chosen.append((coefficients, row))
                mask = support(row)
                while mask:
                    qubit = (mask & -mask).bit_length() - 1
                    load[qubit] = load.get(qubit, 0) + 1
                    mask &= mask - 1
                if len(chosen) == required:
                    break
        if len(chosen) == required:
            break
    return tuple(chosen)


def sparse_n2_blockwise_css_center_basis(
    *,
    blocks: Sequence["StabilizerBlock"],
    stitching_rows: Sequence[int],
) -> tuple[BlockwiseCssGenerator, ...]:
    """Construct the inherited centre of a two-block CSS stitch sparsely.

    The construction keeps the pure-X and pure-Z kernels local.  It then
    matches sparse section representatives across the two blocks.  Syndrome
    directions unique to one CSS sector contribute one matched pair; a
    direction shared by the X and Z images contributes the cheapest three
    edges of the cross-block K_2,2 graph.
    """

    blocks = tuple(blocks)
    if len(blocks) != 2:
        raise ValueError("the sparse transport construction currently requires N=2")
    if not stitching_rows:
        raise ValueError("at least one stitching row is required")
    syndrome_width = len(stitching_rows)
    maps: dict[tuple[int, str], _SparseSyndromeMap] = {}

    for block_index, block in enumerate(blocks):
        pure: dict[str, list[int]] = {"X": [], "Z": []}
        for stabilizer in block.stabilizers:
            local_x = x_part(int(stabilizer), block.n)
            local_z = z_part(int(stabilizer), block.n)
            if local_x and not local_z:
                pure["X"].append(int(stabilizer))
            elif local_z and not local_x:
                pure["Z"].append(int(stabilizer))
            else:
                raise ValueError("every source stabilizer must be pure X or pure Z")
        for sector in ("X", "Z"):
            ordered_checks = sorted(
                pure[sector],
                key=lambda check: (
                    bool(
                        _local_stitch_syndrome(
                            check,
                            block_index=block_index,
                            blocks=blocks,
                            stitching_rows=stitching_rows,
                        )
                    ),
                    _local_stitch_syndrome(
                        check,
                        block_index=block_index,
                        blocks=blocks,
                        stitching_rows=stitching_rows,
                    ).bit_count(),
                    pauli_weight(check, block.n),
                    check,
                ),
            )
            checks = _independent_preserving_order(
                ordered_checks,
                2 * block.n,
            )
            syndromes = tuple(
                _local_stitch_syndrome(
                    check,
                    block_index=block_index,
                    blocks=blocks,
                    stitching_rows=stitching_rows,
                )
                for check in checks
            )
            maps[(block_index, sector)] = _SparseSyndromeMap(
                checks=checks,
                syndromes=syndromes,
                pauli_width=2 * block.n,
                syndrome_width=syndrome_width,
            )

    for sector in ("X", "Z"):
        left = maps[(0, sector)]
        right = maps[(1, sector)]
        if not _same_span(left.image_basis, right.image_basis, syndrome_width):
            raise ValueError(
                f"the two blocks have different pure-{sector} syndrome images"
            )

    x_map = maps[(0, "X")]
    z_map = maps[(0, "Z")]
    intersection = _subspace_intersection(
        x_map.image_basis,
        z_map.image_basis,
        syndrome_width,
    )

    def section_cost(syndrome: int, sector: str) -> tuple[int, int]:
        cost = 0
        physical_weight = 0
        for block_index in range(2):
            row, coefficients = maps[(block_index, sector)].section(syndrome)
            cost += coefficients.bit_count()
            physical_weight += pauli_weight(row, blocks[block_index].n)
        return cost, physical_weight

    intersection_candidates = (
        tuple(
            xor_rows(mask, intersection)
            for mask in range(1, 1 << len(intersection))
        )
        if len(intersection) <= 16
        else intersection
    )
    intersection_basis: list[int] = []
    intersection_span: list[int] = []
    for syndrome in sorted(
        set(intersection_candidates),
        key=lambda value: (
            section_cost(value, "X")[0] + section_cost(value, "Z")[0],
            section_cost(value, "X")[1] + section_cost(value, "Z")[1],
            value,
        ),
    ):
        if reduce_with_basis(intersection_span, syndrome, syndrome_width):
            intersection_basis.append(syndrome)
            intersection_span = list(
                _basis((*intersection_span, syndrome), syndrome_width)
            )
            if len(intersection_basis) == len(intersection):
                break
    if not _same_span(intersection_basis, intersection, syndrome_width):
        raise AssertionError("failed to choose the shared syndrome basis")

    def complement_basis(sector: str) -> tuple[int, ...]:
        syndrome_map = maps[(0, sector)]
        candidates = sorted(
            set(syndrome_map.syndromes) - {0},
            key=lambda value: (*section_cost(value, sector), value),
        )
        span = list(_basis(intersection_basis, syndrome_width))
        chosen: list[int] = []
        target_rank = len(syndrome_map.image_basis)
        for syndrome in candidates:
            if reduce_with_basis(span, syndrome, syndrome_width):
                chosen.append(syndrome)
                span = list(_basis((*span, syndrome), syndrome_width))
                if len(span) == target_rank:
                    break
        if not _same_span(
            (*intersection_basis, *chosen),
            syndrome_map.image_basis,
            syndrome_width,
        ):
            raise AssertionError(f"failed to span the pure-{sector} syndrome image")
        return tuple(chosen)

    x_only = complement_basis("X")
    z_only = complement_basis("Z")
    result: list[BlockwiseCssGenerator] = []

    for block_index in range(2):
        for sector in ("X", "Z"):
            syndrome_map = maps[(block_index, sector)]
            for coefficients in syndrome_map.kernel_coefficients:
                local = xor_rows(coefficients, syndrome_map.checks)
                row = _embed_local_pauli(
                    local,
                    block_index=block_index,
                    blocks=blocks,
                )
                result.append(
                    BlockwiseCssGenerator(
                        row=row,
                        block_types=blockwise_pauli_types(row, blocks),
                        constituent_check_count=coefficients.bit_count(),
                        kind=f"local_{sector}",
                    )
                )

    def cross_pair(
        syndrome: int,
        left_sector: str,
        right_sector: str,
        *,
        kind: str,
    ) -> BlockwiseCssGenerator:
        left_row, left_coefficients = maps[(0, left_sector)].section(syndrome)
        right_row, right_coefficients = maps[(1, right_sector)].section(syndrome)
        row = _embed_local_pauli(
            left_row,
            block_index=0,
            blocks=blocks,
        ) ^ _embed_local_pauli(
            right_row,
            block_index=1,
            blocks=blocks,
        )
        return BlockwiseCssGenerator(
            row=row,
            block_types=blockwise_pauli_types(row, blocks),
            constituent_check_count=(
                left_coefficients.bit_count() + right_coefficients.bit_count()
            ),
            kind=kind,
        )

    for syndrome in x_only:
        result.append(cross_pair(syndrome, "X", "X", kind="cross_X_only"))
    for syndrome in z_only:
        result.append(cross_pair(syndrome, "Z", "Z", kind="cross_Z_only"))
    for syndrome in intersection_basis:
        candidates = [
            cross_pair(syndrome, left, right, kind=f"cross_shared_{left}{right}")
            for left in ("X", "Z")
            for right in ("X", "Z")
        ]
        candidates.sort(
            key=lambda generator: (
                generator.constituent_check_count,
                pauli_weight(generator.row, sum(block.n for block in blocks)),
                generator.kind,
            )
        )
        result.extend(candidates[:3])

    width = 2 * sum(block.n for block in blocks)
    if len(_basis((generator.row for generator in result), width)) != len(result):
        raise AssertionError("the sparse blockwise-CSS generators are dependent")
    if any("M" in generator.block_types for generator in result):
        raise AssertionError("a sparse generator has a mixed local restriction")
    return tuple(result)


@dataclass(frozen=True)
class MeasuredProduct:
    """One intended logical product generated by an incidence codeword."""

    family: str
    pattern_index: int
    incidence: int
    row: int

    @property
    def label(self) -> str:
        return f"{self.family}:b{self.pattern_index}"


@dataclass(frozen=True)
class QuiltSpec:
    """Physical code blocks and incidence-generated stitching families."""

    blocks: tuple[StabilizerBlock, ...]
    families: tuple[StitchingFamily, ...]

    def __post_init__(self) -> None:
        if len(self.blocks) < 2:
            raise ValueError("a quilt requires at least two blocks")
        if not self.families:
            raise ValueError("a quilt requires at least one stitching family")
        if len({family.name for family in self.families}) != len(self.families):
            raise ValueError("stitching-family names must be unique")

        occupied = [0] * len(self.blocks)
        for family in self.families:
            if not family.name:
                raise ValueError("a stitching-family name is required")
            if len(family.local_pieces_by_block) != len(self.blocks):
                raise ValueError("each family needs local pieces on every block")
            if family.piece_count <= 0:
                raise ValueError("each stitching family needs at least one local piece")
            if any(
                len(pieces) != family.piece_count
                for pieces in family.local_pieces_by_block
            ):
                raise ValueError("a family must have the same piece count on every block")
            if not family.incidence_patterns:
                raise ValueError("each family needs at least one incidence pattern")
            if len(set(family.incidence_patterns)) != len(family.incidence_patterns):
                raise ValueError("incidence patterns within a family must be distinct")

            for pattern in family.incidence_patterns:
                if pattern <= 0 or pattern & ~all_ones(len(self.blocks)):
                    raise ValueError("an incidence pattern exceeds the block count")
                if int(pattern).bit_count() < 2:
                    raise ValueError("every stitching gauge must touch at least two blocks")
            incidence_basis = _basis(
                family.incidence_patterns,
                len(self.blocks),
            )
            if any(
                _in_span(incidence_basis, 1 << block_index, len(self.blocks))
                for block_index in range(len(self.blocks))
            ):
                raise ValueError(
                    f"stitching family {family.name!r} has incidence-code distance one"
                )

            common_kernel: tuple[int, ...] | None = None
            common_kernel_block: str | None = None
            for block_index, (block, pieces) in enumerate(
                zip(self.blocks, family.local_pieces_by_block, strict=True)
            ):
                active = any(
                    (pattern >> block_index) & 1
                    for pattern in family.incidence_patterns
                )
                if not active:
                    if any(int(piece) for piece in pieces):
                        raise ValueError(
                            "an incidence-inactive block must use zero local pieces"
                        )
                    continue
                local_product = 0
                block_limit = 1 << (2 * block.n)
                for piece in pieces:
                    piece = int(piece)
                    if piece <= 0 or piece >= block_limit:
                        raise ValueError("local stitching pieces must be nonzero and in range")
                    support = x_part(piece, block.n) | z_part(piece, block.n)
                    if support & occupied[block_index]:
                        raise ValueError(
                            "local stitching pieces must be physically disjoint on each block"
                        )
                    occupied[block_index] |= support
                    local_product ^= piece
                if any(
                    symplectic_product(local_product, stabilizer, block.n)
                    for stabilizer in block.stabilizers
                ):
                    raise ValueError("the product of local pieces must be a block logical")
                if _in_span(block.stabilizers, local_product, 2 * block.n):
                    raise ValueError(
                        "the product of local pieces must be a nontrivial block logical"
                    )
                anticommutation_rows = _commutator_rows(
                    block.stabilizers,
                    pieces,
                    block.n,
                )
                kernel = _basis(
                    nullspace(anticommutation_rows, family.piece_count),
                    family.piece_count,
                )
                if common_kernel is None:
                    common_kernel = kernel
                    common_kernel_block = block.name
                elif kernel != common_kernel:
                    raise ValueError(
                        f"stitching family {family.name!r} has different local "
                        f"anticommutation kernels on active blocks "
                        f"{common_kernel_block!r} and {block.name!r}"
                    )

    @property
    def total_n(self) -> int:
        return sum(block.n for block in self.blocks)

    @property
    def offsets(self) -> tuple[int, ...]:
        return _block_offsets(self.blocks)

    def permuted(self, order: Sequence[int]) -> "QuiltSpec":
        """Return the same quilt with its block order changed."""

        order = tuple(int(index) for index in order)
        if sorted(order) != list(range(len(self.blocks))):
            raise ValueError("order must be a permutation of all block indices")
        inverse = {old: new for new, old in enumerate(order)}
        families = tuple(
            StitchingFamily(
                name=family.name,
                local_pieces_by_block=tuple(
                    family.local_pieces_by_block[old] for old in order
                ),
                incidence_patterns=tuple(
                    sum(
                        ((pattern >> old) & 1) << inverse[old]
                        for old in range(len(self.blocks))
                    )
                    for pattern in family.incidence_patterns
                ),
            )
            for family in self.families
        )
        return QuiltSpec(
            blocks=tuple(self.blocks[old] for old in order),
            families=families,
        )

    def transformed(
        self,
        maps_by_block: Sequence[Sequence[LocalClifford | str]],
    ) -> "QuiltSpec":
        """Apply a local Clifford to checks and stitching pieces together."""

        if len(maps_by_block) != len(self.blocks):
            raise ValueError("one local-Clifford map list is required per block")
        transformed_blocks = tuple(
            block.transformed(maps)
            for block, maps in zip(self.blocks, maps_by_block, strict=True)
        )
        transformed_families = tuple(
            StitchingFamily(
                name=family.name,
                local_pieces_by_block=tuple(
                    tuple(
                        apply_local_cliffords(piece, n=block.n, maps=maps)
                        for piece in pieces
                    )
                    for block, maps, pieces in zip(
                        self.blocks,
                        maps_by_block,
                        family.local_pieces_by_block,
                        strict=True,
                    )
                ),
                incidence_patterns=family.incidence_patterns,
            )
            for family in self.families
        )
        transformed = QuiltSpec(
            blocks=transformed_blocks,
            families=transformed_families,
        )

        for block_index, (block, maps) in enumerate(
            zip(self.blocks, maps_by_block, strict=True)
        ):
            rows = [
                *block.stabilizers,
                *(
                    piece
                    for family in self.families
                    for piece in family.local_pieces_by_block[block_index]
                ),
            ]
            images = [
                apply_local_cliffords(row, n=block.n, maps=maps) for row in rows
            ]
            for left_index, left in enumerate(rows):
                for right_index, right in enumerate(rows):
                    if symplectic_product(left, right, block.n) != symplectic_product(
                        images[left_index],
                        images[right_index],
                        block.n,
                    ):
                        raise AssertionError(
                            "local Clifford failed to preserve the symplectic form"
                        )
        return transformed


@dataclass(frozen=True)
class QuiltAlgebraLedger:
    """Exact structural and observable ledger for a quilt subsystem."""

    spec: QuiltSpec
    base_stabilizer_basis: tuple[int, ...]
    canonical_stitch_gauges: tuple[int, ...]
    supplied_stitch_gauges: tuple[int, ...]
    measured_products: tuple[MeasuredProduct, ...]
    desired_measurement_basis: tuple[int, ...]
    before: GeneralSubsystem
    merged: GeneralSubsystem
    before_bare_logicals: SymplecticBasis
    merged_bare_logicals: SymplecticBasis
    reliable_center_basis: tuple[int, ...]
    measured_center_basis: tuple[int, ...]
    classical_observables: tuple[int, ...]
    memory_observables: SymplecticBasis
    base_stitch_commutator: tuple[int, ...]
    gauge_commutator: tuple[int, ...]
    analysis: MeasurementAnalysis

    @classmethod
    def build(
        cls,
        spec: QuiltSpec,
        *,
        gauge_generators: Sequence[int] | None = None,
        require_valid: bool = True,
    ) -> "QuiltAlgebraLedger":
        """Construct and exactly validate the merged quilt algebra.

        ``gauge_generators`` may replace the canonical stitch generators by
        any row-operated generating set for the same total gauge group,
        including generators multiplied by original stabilizers.
        """

        total_n = spec.total_n
        width = 2 * total_n
        offsets = spec.offsets
        base = _basis(
            (
                embed_block_pauli(
                    stabilizer,
                    block_n=block.n,
                    offset=offset,
                    total_n=total_n,
                )
                for block, offset in zip(spec.blocks, offsets, strict=True)
                for stabilizer in block.stabilizers
            ),
            width,
        )

        canonical_gauges: list[int] = []
        measured_products: list[MeasuredProduct] = []
        for family in spec.families:
            for pattern_index, incidence in enumerate(family.incidence_patterns):
                pattern_gauges: list[int] = []
                for piece_index in range(family.piece_count):
                    gauge = 0
                    for block_index, (block, offset) in enumerate(
                        zip(spec.blocks, offsets, strict=True)
                    ):
                        if (incidence >> block_index) & 1:
                            gauge ^= embed_block_pauli(
                                family.local_pieces_by_block[block_index][piece_index],
                                block_n=block.n,
                                offset=offset,
                                total_n=total_n,
                            )
                    pattern_gauges.append(gauge)
                    canonical_gauges.append(gauge)
                product = 0
                for gauge in pattern_gauges:
                    product ^= gauge
                measured_products.append(
                    MeasuredProduct(
                        family=family.name,
                        pattern_index=pattern_index,
                        incidence=incidence,
                        row=product,
                    )
                )

        canonical = tuple(canonical_gauges)
        if not rows_commute(canonical, total_n):
            raise ValueError("stitching gauge generators must commute with one another")
        supplied = tuple(
            int(row) for row in (canonical if gauge_generators is None else gauge_generators)
        )
        limit = 1 << width
        if any(row <= 0 or row >= limit for row in supplied):
            raise ValueError("supplied stitch gauges must be nonzero and in range")
        if not _same_span([*base, *canonical], [*base, *supplied], width):
            raise ValueError(
                "supplied gauge generators do not generate the canonical total gauge group"
            )

        desired = _quotient_basis(
            tuple(product.row for product in measured_products),
            base,
            width,
        )
        if not rows_commute(desired, total_n):
            raise ValueError("intended measured logical products must commute")
        if any(
            symplectic_product(row, stabilizer, total_n)
            for row in desired
            for stabilizer in base
        ):
            raise ValueError("an intended measured product is not an original logical")

        before = build_general_subsystem(
            n=total_n,
            base_stabilizers=base,
            added_gauges=(),
        )
        merged = build_general_subsystem(
            n=total_n,
            base_stabilizers=base,
            added_gauges=supplied,
        )
        exact_center = tuple(gauge_center(merged.gauge_basis, total_n))
        if exact_center != merged.center_basis:
            raise AssertionError("GeneralSubsystem returned an incorrect gauge center")
        if any(
            symplectic_product(center, gauge, total_n)
            for center in merged.center_basis
            for gauge in merged.gauge_basis
        ):
            raise AssertionError("a purported central generator anticommutes with the gauge")

        reliable = _subspace_intersection(
            base,
            merged.center_basis,
            width,
        )
        measured_center = tuple(
            quotient_complement(merged.center_basis, reliable, width)
        )
        analysis = analyze_measurement(merged, desired)

        before_logicals = SymplecticBasis.from_rows(
            before.bare_logical_basis,
            n=total_n,
            modulus_basis=before.center_basis,
        )
        merged_logicals = SymplecticBasis.from_rows(
            merged.bare_logical_basis,
            n=total_n,
            modulus_basis=merged.center_basis,
        )

        ledger = cls(
            spec=spec,
            base_stabilizer_basis=base,
            canonical_stitch_gauges=canonical,
            supplied_stitch_gauges=supplied,
            measured_products=tuple(measured_products),
            desired_measurement_basis=desired,
            before=before,
            merged=merged,
            before_bare_logicals=before_logicals,
            merged_bare_logicals=merged_logicals,
            reliable_center_basis=reliable,
            measured_center_basis=measured_center,
            classical_observables=desired,
            memory_observables=merged_logicals,
            base_stitch_commutator=_commutator_rows(base, supplied, total_n),
            gauge_commutator=_commutator_rows(
                merged.gauge_basis,
                merged.gauge_basis,
                total_n,
            ),
            analysis=analysis,
        )
        ledger._assert_internal_consistency(require_valid=require_valid)
        return ledger

    @property
    def n(self) -> int:
        return self.spec.total_n

    @property
    def measured_rank(self) -> int:
        return self.analysis.measured_rank

    @property
    def surviving_logical_count(self) -> int:
        return self.merged.logical_qubits

    @property
    def measures_all_intended(self) -> bool:
        return self.analysis.contains_desired

    @property
    def no_overmeasurement(self) -> bool:
        return self.analysis.no_extra_measurements

    @property
    def preserves_unmeasured_logicals(self) -> bool:
        return self.analysis.preserves_unmeasured_logicals

    @property
    def valid(self) -> bool:
        return self.analysis.valid

    def equivalent_classical(self, left: int, right: int) -> bool:
        """Whether central representatives agree modulo reliable stabilizers."""

        return _in_span(
            self.reliable_center_basis,
            int(left) ^ int(right),
            2 * self.n,
        )

    def equivalent_bare_observable(self, left: int, right: int) -> bool:
        """Whether bare observables agree modulo the merged center."""

        return _in_span(
            self.merged.center_basis,
            int(left) ^ int(right),
            2 * self.n,
        )

    def equivalent_dressed_error(self, left: int, right: int) -> bool:
        """Whether dressed errors agree modulo the merged gauge group."""

        return _in_span(
            self.merged.gauge_basis,
            int(left) ^ int(right),
            2 * self.n,
        )

    def assert_generator_invariant(self, other: "QuiltAlgebraLedger") -> None:
        """Assert equality of every physical sector after gauge row operations."""

        if self.n != other.n:
            raise AssertionError("ledgers act on different physical spaces")
        width = 2 * self.n
        spaces = (
            ("base stabilizers", self.base_stabilizer_basis, other.base_stabilizer_basis),
            ("gauge", self.merged.gauge_basis, other.merged.gauge_basis),
            ("center", self.merged.center_basis, other.merged.center_basis),
            (
                "reliable center",
                self.reliable_center_basis,
                other.reliable_center_basis,
            ),
            (
                "measured center",
                [*self.reliable_center_basis, *self.measured_center_basis],
                [*other.reliable_center_basis, *other.measured_center_basis],
            ),
            (
                "classical observables",
                [*self.reliable_center_basis, *self.classical_observables],
                [*other.reliable_center_basis, *other.classical_observables],
            ),
            (
                "memory observables",
                [*self.merged.center_basis, *self.memory_observables.rows],
                [*other.merged.center_basis, *other.memory_observables.rows],
            ),
        )
        for label, left, right in spaces:
            if not _same_span(left, right, width):
                raise AssertionError(f"{label} changed under gauge-generator row operations")
        if (
            self.measured_rank,
            self.surviving_logical_count,
            self.no_overmeasurement,
            self.preserves_unmeasured_logicals,
        ) != (
            other.measured_rank,
            other.surviving_logical_count,
            other.no_overmeasurement,
            other.preserves_unmeasured_logicals,
        ):
            raise AssertionError("ledger dimensions changed under generator row operations")

    def _assert_internal_consistency(self, *, require_valid: bool) -> None:
        width = 2 * self.n
        if len(self.before_bare_logicals.rows) != 2 * self.before.logical_qubits:
            raise AssertionError("the pre-stitch bare logical basis is incomplete")
        if len(self.memory_observables.rows) != 2 * self.merged.logical_qubits:
            raise AssertionError("the merged bare logical basis is incomplete")
        if len(self.measured_center_basis) != (
            len(self.merged.center_basis) - len(self.reliable_center_basis)
        ):
            raise AssertionError("reliable and measured center dimensions do not add up")
        if not _same_span(
            [*self.reliable_center_basis, *self.measured_center_basis],
            self.merged.center_basis,
            width,
        ):
            raise AssertionError("reliable and measured sectors do not span the center")
        if self.measured_rank != len(self.measured_center_basis):
            raise AssertionError("measured-sector rank disagrees with measurement analysis")
        if len(self.desired_measurement_basis) != self.analysis.desired_rank:
            raise AssertionError("intended measured-product rank is inconsistent")

        for product in self.measured_products:
            if not _in_span(self.merged.center_basis, product.row, width):
                raise AssertionError(
                    f"intended measured product {product.label} is not central"
                )
        for observable in self.classical_observables:
            if not _in_span(self.merged.center_basis, observable, width):
                raise AssertionError("an intended classical observable is not central")
        if self.analysis.exact and not _same_span(
            [*self.reliable_center_basis, *self.classical_observables],
            self.merged.center_basis,
            width,
        ):
            raise AssertionError("classical representatives do not span the measured sector")

        gauge_centralizer = tuple(
            symplectic_centralizer(self.merged.gauge_basis, self.n)
        )
        for observable in self.memory_observables.rows:
            if not _in_span(gauge_centralizer, observable, width):
                raise AssertionError("a memory observable does not commute with the gauge")
            if _in_span(self.merged.center_basis, observable, width):
                raise AssertionError("a memory observable is a merged stabilizer")

        expected_survivors = self.before.logical_qubits - self.measured_rank
        if self.preserves_unmeasured_logicals and (
            self.surviving_logical_count != expected_survivors
        ):
            raise AssertionError("surviving logical count is inconsistent")
        if require_valid and not self.valid:
            raise ValueError(
                "the quilt does not measure exactly the intended sector while "
                "preserving every unmeasured logical"
            )
