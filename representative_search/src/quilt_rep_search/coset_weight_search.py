"""Finding a clean, low-weight representative of one logical class.

Multiplying an operator by a stabiliser of its own code changes neither its
logical class nor what the interface measures, so every member of the coset

    representative(label) + <stabilisers>

is an equally valid choice.  They are not equally *useful*: the interface width
of a quilt is the weight of the representative, so we want the lightest one we
can find, and it has to pass the cleanliness gate.

Minimum-weight coset search is hard in general, so this is simulated annealing:
propose a stabiliser, accept downhill always and uphill with the usual
Metropolis probability, and keep the best clean state seen.  The gate is the
expensive part, so it is only evaluated when a state is both light enough to be
interesting and new -- an annealing run touches far more states than it needs to
test.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from .interface_cleanliness import inspect
from .codes_and_symmetries import Code, bit_count


@dataclass(frozen=True)
class Found:
    mask: int
    weight: int
    gate_calls: int
    accepted_moves: int
    restarts_used: int

    def __bool__(self) -> bool:
        return self.mask != 0


def find_representative(
    code: Code,
    sector: str,
    label: int,
    *,
    seed: int = 0,
    restarts: int = 4,
    sweeps: int = 20_000,
    gate_budget: int = 300,
    start_masks: Sequence[int] = (),
) -> Found:
    """Search the coset of `label` for the lightest clean representative found.

    `start_masks` lets a caller hand in bank candidates as starting points; with
    none supplied the search begins from the canonical basis product, which is
    typically much heavier.
    """

    if label == 0:
        raise ValueError("the identity class has no representative to find")

    stabilisers = code.z_checks if sector == "Z" else code.x_checks
    if not stabilisers:
        raise ValueError(f"code {code.code_id} has no {sector} stabilisers")

    canonical = code.representative(label, sector)
    starts = list(start_masks) or [canonical]

    best_mask, best_weight = 0, None
    gate_calls = accepted = 0
    tested: set[int] = set()

    for restart in range(restarts):
        rng = random.Random((seed << 8) ^ (restart * 0x9E3779B1))
        current = starts[restart % len(starts)]
        if restart >= len(starts):                 # diversify beyond the seeds
            for stabiliser in stabilisers:
                if rng.getrandbits(1):
                    current ^= stabiliser
        weight = bit_count(current)

        hot, cold = 40.0, 0.15
        for step in range(sweeps):
            temperature = hot * (cold / hot) ** (step / sweeps)
            candidate = current ^ stabilisers[rng.randrange(len(stabilisers))]
            delta = bit_count(candidate) - weight
            if delta > 0 and rng.random() >= math.exp(max(-50.0, -delta / temperature)):
                continue
            current, weight = candidate, weight + delta
            accepted += 1

            interesting = weight and (best_weight is None or weight < best_weight)
            if interesting and gate_calls < gate_budget and current not in tested:
                tested.add(current)
                gate_calls += 1
                if inspect(current, code=code, sector=sector).clean:
                    best_mask, best_weight = current, weight

        if best_mask and best_weight is not None and best_weight <= code.distance:
            break                                   # cannot do better than d

    return Found(
        mask=best_mask,
        weight=best_weight or 0,
        gate_calls=gate_calls,
        accepted_moves=accepted,
        restarts_used=min(restarts, restart + 1),
    )
