"""The two things that make a slow target recoverable.

Measured on bb144, SC_BB_ZZ needs a median of 4 screens per redraw and reaches
438; at bb288 rejection times that tail runs to nineteen hours. Two properties
keep a long redraw from costing the run:

  * the held representative's verdict is handed over before any redraw
    starts, so losing the redraw does not lose the verdict;
  * the candidate loop has a wall clock, so a doomed target ends as one
    recorded failure instead of an unbounded loop.

Both are control-flow properties of `one_target`, invisible in a finished
record, so they are exercised against stubbed search and screen machinery
rather than against real codes.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_corpus as gc


FAILED_VERDICT = {"cohort": "bb288/SC_BB_ZZ", "target_id": "SC_BB_ZZ-0001",
                  "survived": False, "witness_depth": 1234,
                  "witness_weight": 17, "trials": 200000, "threshold": 18}

RAW = {"target_id": "SC_BB_ZZ-0001",
       "blocks": [{"code": "x", "logical_x": "0x1", "logical_z": "0x2"}],
       "incumbent": {"pairing": {"width": 20}}}


@pytest.fixture
def stubbed(monkeypatch):
    """A target whose held representative fails and whose every candidate is rejected."""

    monkeypatch.setattr(gc, "screen_incumbent",
                        lambda record, *, trials, seed: dict(FAILED_VERDICT))
    monkeypatch.setattr(gc, "STATE", {"codes": {"x": object()}, "banks": {}})
    monkeypatch.setattr(gc, "Block", lambda code, lx, lz: (code, lx, lz))
    monkeypatch.setattr(gc, "Target", lambda tid, cohort, blocks: (tid, cohort))

    found = types.SimpleNamespace(candidates=[object()] * 50,
                                  widths=[20] * 50, seed=1, starts_used=1,
                                  gate_calls=0, accepted_moves=0, start_rank=0,
                                  started_feasible=False)
    monkeypatch.setattr(gc, "search", lambda *a, **k: found)
    monkeypatch.setattr(gc, "build_record",
                        lambda *a, **k: {"pairing": {"strategy": "s", "width": 20,
                                                     "resample_seed": None},
                                         "operation_threshold": 18})
    monkeypatch.setattr(gc, "proof_covered", lambda record: False)

    rejected = types.SimpleNamespace(
        passed=False, rejected_by="qdist:grouped",
        engines=[types.SimpleNamespace(engine="qdist:grouped", trials=200000,
                                       trials_completed=10, witness=True,
                                       distance=17)])
    monkeypatch.setattr(gc, "screen", lambda record, *, trials, seed: rejected)
    return gc


def test_verdict_is_handed_over_before_the_repair(stubbed):
    """The callback must fire while the redraw is still ahead, not after it.

    Written as an ordering assertion rather than a "was it called" assertion:
    calling it on the way out would satisfy the latter and lose exactly the
    case it exists for, a process stopped deep in a long redraw.
    """

    events = []
    monkey_screen = stubbed.screen
    stubbed_screen = lambda record, *, trials, seed: (events.append("screen"),
                                                      monkey_screen(record, trials=trials,
                                                                    seed=seed))[1]
    stubbed.screen = stubbed_screen
    try:
        stubbed.one_target(("bb288/SC_BB_ZZ", RAW, 200000, 50, 1),
                           on_incumbent=lambda row: events.append(row["witness_depth"]),
                           time_budget=0.0)
    finally:
        stubbed.screen = monkey_screen

    assert events and events[0] == 1234, events
    assert "screen" not in events[:1]


def test_exhausted_budget_is_a_recorded_outcome(stubbed):
    """Zero budget: one named failure carrying the verdict, not an exception."""

    out = stubbed.one_target(("bb288/SC_BB_ZZ", RAW, 200000, 50, 1),
                             time_budget=0.0)
    assert out["outcome"] == "repair_budget_exhausted"
    assert out["incumbent_screen"]["witness_depth"] == 1234
    assert out["candidates_tried"] == 0
    assert out["candidates_offered"] == 50


def test_no_budget_means_the_old_behaviour(stubbed):
    """Omitting the budget must not cut anything short.

    The CLI path builds its jobs without one, and a default that silently
    truncated the candidate list would weaken every corpus built locally.
    """

    out = stubbed.one_target(("bb288/SC_BB_ZZ", RAW, 200000, 50, 1))
    assert out["outcome"] == "all_candidates_rejected"
    assert len(out["rejected"]) == 50 * 5
