"""Choosing the interface: the search that actually sets the corpus numbers.

Every block's representative can be multiplied by that block's stabilisers
without changing what is measured, so the search is free to roam a product of
cosets.  What it is looking for is a state where the blocks can actually be
stitched together, and where the resulting interface is as narrow as possible.

Three things must hold before a state is usable at all:

  equal width      every block must offer the same number of stitching pieces,
                   since the pieces are paired one-to-one across blocks;
  connected        each block's interface-coordinate graph must be connected,
                   or no deterministic pairing exists to build at all (checked
                   at the end, not during the walk -- see `_acceptable`);
  clean interface  on each block the local kernel must be span(all-ones), so the
                   interface measures the intended class and nothing else;
  enough overlap   for mixed targets, sum_i o_i >= D, without which the
                   relative-sector certificate of the appendix cannot reach D.

Among states that satisfy those, we want the smallest total interface width.
Note that width already prices overlap correctly: a qubit carrying both X and Z
is one stitching piece, not two, so W_i = |x_i| + |z_i| - o_i.  Overlap beyond
the threshold therefore needs no separate reward -- the earlier pipeline paid
+8 per unit of overlap on top of this, which bought overlap at the price of
width and is why the generic cohorts came out wide.

The objective is a strict priority order, encoded as weights far apart in scale:
feasibility first, then width.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Sequence

from .representative_bank import Entry, improvise
from .interface_cleanliness import inspect
from .codes_and_symmetries import Code, bit_count
from .quilt_targets import (
    Block, State, Target, interface_components, interface_width, labels_match, overlap,
)


@dataclass(frozen=True)
class Score:
    width_gap: int            # spread of interface widths; must reach zero
    overlap_deficit: int      # shortfall against sum_i o_i >= D; must reach zero
    total_width: int          # what we actually minimise
    feasible: bool

    @property
    def value(self) -> float:
        """Lower is better.  Feasibility dominates width by construction."""

        return (
            1000.0 * self.width_gap
            + 1500.0 * self.overlap_deficit
            + float(self.total_width)
        )


def score(state: State, target: Target) -> Score:
    widths = [interface_width(b) for b in state]
    mixed = any(block.is_mixed for block in target.blocks)
    overlap_sum = sum(overlap(b) for b in state)
    deficit = max(0, target.distance - overlap_sum) if mixed else 0
    gap = max(widths) - min(widths)
    return Score(
        width_gap=gap,
        overlap_deficit=deficit,
        total_width=sum(widths),
        feasible=gap == 0 and deficit == 0,
    )


def _acceptable(state: State, target: Target) -> bool:
    """Terminal conditions: the gate, and a pairable interface, on every block.

    Both are expensive relative to a single annealing move and neither usefully
    steers the walk, so they are checked only on a candidate we would otherwise
    keep.  Intermediate states are free to be disconnected or unclean -- an
    endpoint is not.
    """

    for block, block_state in zip(target.blocks, state, strict=True):
        if interface_components(block_state, block.code) != 1:
            return False
    return _clean_everywhere(state, target)


def _clean_everywhere(state: State, target: Target) -> bool:
    """The cleanliness gate, applied blockwise.  Never skipped, whatever the weight."""

    for block, (x_mask, z_mask) in zip(target.blocks, state, strict=True):
        if block.is_pure_z:
            if not inspect(z_mask, code=block.code, sector="Z").clean:
                return False
        elif block.is_pure_x:
            if not inspect(x_mask, code=block.code, sector="X").clean:
                return False
        else:
            # a mixed block must be clean in each sector separately
            if not inspect(x_mask, code=block.code, sector="X").clean:
                return False
            if not inspect(z_mask, code=block.code, sector="Z").clean:
                return False
    return True


def _moves(target: Target) -> list[tuple[int, str, int]]:
    """Legal moves: multiply one block, in one sector, by one stabiliser.

    A sector the target does not use is locked.  Without this the search will
    happily multiply a pure-Z block by an X stabiliser -- in-coset, so legal,
    but it leaves the pure sector and nothing in the objective pulls it back.
    """

    moves = []
    for index, block in enumerate(target.blocks):
        if not block.is_pure_z:
            moves += [(index, "X", check) for check in block.code.x_checks]
        if not block.is_pure_x:
            moves += [(index, "Z", check) for check in block.code.z_checks]
    return moves


def _apply(state: State, move: tuple[int, str, int]) -> State:
    index, sector, check = move
    x_mask, z_mask = state[index]
    updated = (x_mask ^ check, z_mask) if sector == "X" else (x_mask, z_mask ^ check)
    return state[:index] + (updated,) + state[index + 1 :]


def bank_starts(
    target: Target,
    banks: dict[str, dict[tuple[str, int], list[Entry]]],
    *,
    limit: int,
) -> list[State]:
    """Starting states built from bank candidates, narrowest combinations first.

    The bank gives several low-weight representatives per class.  Their widths
    rarely agree across blocks -- that is exactly what the annealer is for --
    but starting from them lands far closer than starting from the canonical
    basis product.
    """

    per_block: list[list[tuple[int, int]]] = []
    for block in target.blocks:
        # A code with only one logical class -- the surface blocks -- has no
        # bank: there is nothing for the orbit expansion to spread over.  Fall
        # back to the canonical representative and let the annealer work.
        bank = banks.get(block.code.code_id, {})

        def choices(sector: str, label: int) -> list[int]:
            entries = bank.get((sector, label), [])[:limit]
            if entries:
                return [entry.mask for entry in entries]
            # no bank for this code: improvise several distinct starts rather
            # than handing every restart the same canonical representative
            return [entry.mask for entry in improvise(block.code, sector, label, count=limit)]

        if block.is_pure_z:
            options = [(0, z) for z in choices("Z", block.logical_z)]
        elif block.is_pure_x:
            options = [(x, 0) for x in choices("X", block.logical_x)]
        else:
            options = [(x, z)
                       for x in choices("X", block.logical_x)
                       for z in choices("Z", block.logical_z)]
        per_block.append(options)

    states: list[State] = [()]
    for options in per_block:
        states = [prefix + (choice,) for prefix in states for choice in options]
    states.sort(key=lambda s: score(s, target).value)
    return states[:limit]


def describe_starts(target: Target, banks, *, limit: int) -> list[list[dict]]:
    """Where each block's starting representatives came from.

    A bare index into the bank only means anything if you already know how the
    bank was ordered.  Record the seed's own identity.
    """

    described = []
    for block in target.blocks:
        bank = banks.get(block.code.code_id, {})
        rows = []
        for sector, label in (("X", block.logical_x), ("Z", block.logical_z)):
            if not label:
                continue
            entries = bank.get((sector, label), [])[:limit]
            if not entries:
                entries = improvise(block.code, sector, label, count=limit)
            rows += [{"sector": sector, "label": f"0x{label:03x}",
                      "weight": entry.weight, "origin": entry.source_class}
                     for entry in entries]
        described.append(rows)
    return described


@dataclass
class Result:
    """Several acceptable states, narrowest first.

    The annealer visits many acceptable states and it costs nothing to keep the
    best few rather than one.  That matters downstream: the distance screen is
    ~19 s per record at bb288 while a whole re-search is ~1 s, so a rejected
    candidate should be replaced from a ranked list rather than by searching
    again.  Each entry carries its own deterministic pairing.
    """

    candidates: list[State]             # distinct, narrowest first
    widths: list[int]                   # per-block width of each candidate
    seconds: float
    gate_calls: int
    accepted_moves: int
    start_rank: int | None = None       # which bank start led to the best one
    started_feasible: bool = False      # did the bank alone already work?
    seed: int = 0                       # the seed this search ran under
    starts_used: list[list[dict]] = field(default_factory=list)

    @property
    def state(self) -> State | None:
        return self.candidates[0] if self.candidates else None

    @property
    def width(self) -> int:
        return self.widths[0] if self.widths else 0

    def __bool__(self) -> bool:
        return bool(self.candidates)


def search(
    target: Target,
    banks: dict[str, dict[tuple[str, int], list[Entry]]],
    *,
    seed: int = 0,
    restarts: int = 8,
    sweeps: int = 20_000,
    start_limit: int = 8,
    keep: int = 8,
) -> Result:
    """Anneal within the cosets for the narrowest clean, stitchable interface."""

    began = time.monotonic()
    starts = bank_starts(target, banks, limit=start_limit)
    moves = _moves(target)
    gate_calls = accepted = 0

    # width -> state, keeping the narrowest `keep` distinct acceptable states
    found: dict[State, int] = {}
    best_rank = None

    def offer(candidate: State, width: int, rank: int | None) -> None:
        nonlocal best_rank
        if candidate in found:
            return
        found[candidate] = width
        if not found or width <= min(found.values()):
            best_rank = rank

    def worst_kept() -> int | None:
        if len(found) < keep:
            return None
        return max(found.values())

    # A bank pair that is already acceptable costs nothing to check, and for
    # easy targets it ends the search before any annealing happens.
    for rank, state in enumerate(starts):
        if score(state, target).feasible:
            gate_calls += 1
            if _acceptable(state, target):
                offer(state, score(state, target).total_width, rank)
    started_feasible = bool(found)

    for restart in range(restarts):
        rng = random.Random((seed << 8) ^ (restart * 0x9E3779B1))
        # Vary the destination and the start independently.  Taking only the
        # closest start to each width would trade one collapse for another:
        # where the blocks already agree the menu holds a single width, and
        # every restart would then begin from the same state.
        rank = restart % len(starts)
        state = starts[rank]
        current = score(state, target).value

        hot, cold = 60.0, 0.2
        for step in range(sweeps):
            temperature = hot * (cold / hot) ** (step / sweeps)
            candidate = _apply(state, moves[rng.randrange(len(moves))])
            candidate_score = score(candidate, target)
            delta = candidate_score.value - current
            if delta > 0 and rng.random() >= math.exp(max(-50.0, -delta / temperature)):
                continue
            state, current = candidate, candidate_score.value
            accepted += 1

            if not candidate_score.feasible or candidate in found:
                continue
            ceiling = worst_kept()
            if ceiling is not None and candidate_score.total_width >= ceiling:
                continue
            gate_calls += 1
            if _acceptable(state, target):
                offer(state, candidate_score.total_width, rank)
                if len(found) > keep:                       # drop the widest
                    del found[max(found, key=lambda s: (found[s], s))]

    ranked = sorted(found, key=lambda s: (found[s], s))
    blocks = len(target.blocks)
    return Result(
        seed=seed,
        starts_used=describe_starts(target, banks, limit=start_limit),
        candidates=ranked,
        widths=[found[s] // blocks for s in ranked],
        seconds=time.monotonic() - began,
        gate_calls=gate_calls,
        accepted_moves=accepted,
        start_rank=best_rank,
        started_feasible=started_feasible,
    )
