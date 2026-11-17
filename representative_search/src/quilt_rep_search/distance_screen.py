"""Screening a candidate for sub-threshold operational errors.

Two searches are run against each candidate, `grouped` and `three-block`. They
are near-duplicates in outcome -- of 2,779 witnesses found over the population,
2,776 were found by both and only 3 by one alone -- but they are not redundant
in cost: the deepest single-engine search to a witness ran 199,088 trials while
the shallower of the pair found the same witness by 94,992. Running both is a
hedge on depth, and two engines at N trials has the same detection power as one
at 2N.

Budget.  Constructing the problem costs about 19 s per bb288 record.  The
engines run to their budget (see the comment on min_distance below), so the
trials are a real cost.  The problem is built **once** and both engines run
against it on the same budget.

Depth.  Measured over 20,970 records screened at 200,000 trials: 2,780
witnesses, of which 2,524 were found within 1,000 trials but 39 needed more
than 10,000 and the deepest ran to 94,992.  A budget of 20,000 would have
missed the deepest witnesses outright, so 200,000 is the default here.  What a
record reports is `trials_completed` per engine -- the depth actually searched
-- not the budget requested.

A witness found by any engine rejects the candidate.  No engine finding one is
evidence, not proof: this is a randomised non-find and is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
import time
import zlib
from typing import Any, Sequence


@dataclass
class EngineResult:
    engine: str
    trials: int                   # budget requested
    trials_completed: int         # how far it got; less than `trials` means it
                                  # stopped early, which for a rejection says
                                  # how quickly the witness was reachable
    hits: int                     # how many sub-threshold operators it saw
    seconds: float
    distance: int | None          # lowest weight seen, None if nothing found
    witness: bool                 # did it find something below threshold?
    target_hit: bool              # the engine's own report that it STOPPED EARLY
                                  # on reaching min_distance. With min_distance
                                  # = threshold - 1 this should agree with
                                  # `witness`; they are computed independently,
                                  # so a disagreement means the screen and the
                                  # engine disagree about what was found.


@dataclass
class ScreenResult:
    passed: bool
    threshold: int
    engines: list[EngineResult] = field(default_factory=list)
    build_seconds: float = 0.0

    @property
    def rejected_by(self) -> list[str]:
        return [e.engine for e in self.engines if e.witness]

    def to_json(self) -> dict:
        return {
            "outcome": "no_subthreshold_witness" if self.passed else "subthreshold_witness",
            "threshold": self.threshold,
            "problem_build_seconds": round(self.build_seconds, 2),
            "rejected_by": self.rejected_by,
            "engines": [
                {"engine": e.engine, "trials": e.trials,
                 "trials_completed": e.trials_completed, "hits": e.hits,
                 "seconds": round(e.seconds, 2), "lowest_weight_seen": e.distance,
                 "stopped_early": e.target_hit}
                for e in self.engines
            ],
        }


def engine_seed(seed: int, method: str) -> int:
    """A per-engine seed that is a function of (seed, method) and nothing else.

    The seed is derived with a stable hash, not Python's `hash()`, which is
    randomised per process for strings: the same recorded seed must give the
    same engine seed in every process, or the recorded `search_seed` could not
    reproduce the screen it labels.

    crc32 is stable across processes, versions and platforms. It is not a
    cryptographic hash and does not need to be: all that is wanted is two
    decorrelated streams from one recorded number.
    """

    return (int(seed) ^ (zlib.crc32(method.encode()) & 0xFFFF)) & 0x7FFFFFFF


def record_as_v3(record: dict) -> dict:
    """A v3-shaped view of a record, for the construction package's reader.

    The quilt algebra is identical across record schemas -- same blocks, same
    pieces, same masks -- so the construction package is handed a view in the
    schema it reads.  Only the surrounding metadata differs.
    """

    return {
        "schema": "quilt-representative-v3",
        "scale": record["scale"],
        "target_id": record["target_id"],
        "cohort": record["cohort"],
        "operation_threshold": record["operation_threshold"],
        "blocks": record["blocks"],
        "pairing": {
            "width": record["pairing"]["width"],
            "strategy": "minimum-source-weight-skiptree-v1",
            "source_coordinate_convention": "ascending-physical-qubit-index",
        },
        "validation": {
            "rank_condition": "passed",
            "skiptree": {"connected": True,
                         "maximum_edge_source_checks": 1,
                         "maximum_basis_source_checks": 3},
            "distance_screen": {"outcome": "no_subthreshold_witness",
                                "threshold": record["operation_threshold"],
                                "trials_by_method": {"pending": 1}},
        },
    }


def proof_covered(record: dict) -> bool:
    """Does this record satisfy Theorem `weightDquilt`, making the screen moot?

    The theorem (distance_results.tex): if corresponding local stitching
    operators have the same weights |S^i_a| = w_a on every block and
    sum_a w_a = D, the quilt has operational distance at least D. Note the
    condition is sum = D EXACTLY, not <= D; above D the theorem simply stops
    applying, which is why the empirical clean rate falls off a cliff rather
    than degrading smoothly (bb288/SC_BB_ZZ: 100% clean at W=18, 46% at 22,
    28% at 24, 17% at 26, 0 of 18 at 28+).

    Here every stitching operator is weight-2, carrying weight 1 on each of the
    two blocks it joins, so w_a = 1 for all a and sum_a w_a = W. The hypotheses
    therefore reduce to: the width is D, every block carries exactly D pieces,
    and every pairing list has D entries. Checked rather than assumed, because
    a record failing the shape test while matching on width would otherwise be
    certified on a theorem that does not apply to it.

    A record passing this needs no screen at all: its distance is proved, not
    estimated, and no false-negative rate attaches to it.
    """

    threshold = int(record["operation_threshold"])
    pairing = record.get("pairing") or {}
    if int(pairing.get("width", -1)) != threshold:
        return False
    blocks = record.get("blocks") or ()
    if not blocks or any(len(b.get("pieces", ())) != threshold for b in blocks):
        return False
    coordinates = pairing.get("coordinates") or ()
    if len(coordinates) != len(blocks):
        return False
    return all(len(c) == threshold for c in coordinates)


def screen(
    record: dict,
    *,
    trials: int = 5000,
    seed: int = 0,
) -> ScreenResult:
    """Build the operation problem once, then run every engine against it."""

    import numpy as np
    from qdist_rs import search_operation
    from .operation_distance import operation_spec_from_quilt
    from quilt_construction.selected_record import quilt_payload_from_selected_record

    threshold = int(record["operation_threshold"])

    began = time.monotonic()
    quilt = quilt_payload_from_selected_record(record_as_v3(record))
    spec = operation_spec_from_quilt(
        quilt, trials=trials, seed=seed, threads=0,
        method="grouped", target_distance=threshold,
    )
    n = int(spec["n"])

    def matrix(rows: Sequence[str]) -> "np.ndarray":
        out = np.zeros((len(rows), 2 * n), dtype=np.uint8)
        for index, text in enumerate(rows):
            value = int(text, 16)
            for column in range(2 * n):
                out[index, column] = (value >> column) & 1
        return out

    auxiliary = matrix(spec["auxiliary_gauge_hex"])
    benign = matrix(spec["benign_hex"])
    build_seconds = time.monotonic() - began

    engines: list[EngineResult] = []
    for method in ("grouped", "three-block"):
        started = time.monotonic()
        result = search_operation(
            # threshold - 1, NOT threshold. The engine breaks out of its trial
            # loop as soon as the running best weight is <= min_distance. With
            # min_distance = threshold, a quilt whose distance IS the threshold
            # would stop at the first batch that found a threshold-weight
            # operator, long before the budget, and "no sub-threshold witness"
            # would mean only "none turned up before we stopped looking".
            #
            # With threshold - 1 the engine stops only on a genuine
            # sub-threshold witness and otherwise spends the whole budget, so
            # trials_completed is a search depth and a pass means what it says.
            auxiliary, benign, trials=int(trials), seed=engine_seed(seed, method),
            threads=0, method=method, min_distance=threshold - 1,
        )
        distance = getattr(result, "distance", None)
        # No default. `trials_completed` IS the measurement -- the depth at
        # which the screen would have rejected, and the only quantity the
        # reliability model consumes. Defaulting a missing attribute to the
        # full budget would report a complete spend for a search whose depth
        # is unknown. If the engine stops reporting it, fail here rather than
        # fabricate a survivor.
        if not hasattr(result, "trials_completed"):
            raise RuntimeError(
                f"qdist engine {method!r} reported no trials_completed; "
                "the screen cannot record a depth and must not assume one")
        engines.append(EngineResult(
            engine=f"qdist:{method}", trials=int(trials),
            trials_completed=int(result.trials_completed),
            hits=int(getattr(result, "hits", 0)),
            seconds=time.monotonic() - started,
            distance=None if distance is None else int(distance),
            witness=distance is not None and int(distance) < threshold,
            target_hit=bool(getattr(result, "target_hit", False)),
        ))

    return ScreenResult(
        passed=not any(e.witness for e in engines),
        threshold=threshold,
        engines=engines,
        build_seconds=build_seconds,
    )
