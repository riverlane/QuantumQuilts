"""What `min_distance` means to the engine, pinned.

Suppose candidates are screened with `min_distance = threshold`. The engine's loop is

    if min_distance > 0 && global.distance <= min_distance { break; }

so it stops on *reaching* `min_distance`, not on going below it. A
weight-`threshold` operator always exists in these problems, so the search
would stop almost immediately -- after 40 to 80 trials -- and report "no
sub-threshold witness" on the strength of having found a threshold-weight
one. Witnesses below the threshold have been found at depths 296, 1,448 and
11,456, far beyond that.

These tests pin the behaviour:

  1. the engine really does stop early at `min_distance`, so the parameter is
     not advisory;
  2. `screen()` passes `threshold - 1`, so a pass means "nothing below the
     threshold was found", not "something at the threshold was found";
  3. the engine's own `target_hit` flag, which reports the early stop, is
     carried into the record, so an early stop cannot be silent.

If a future change makes any of these fail, the screen no longer runs to its budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

qdist_rs = pytest.importorskip("qdist_rs")

from quilt_rep_search import distance_screen


from quilt_rep_search.codes_and_symmetries import default_definitions

DEFS = default_definitions()


def _code_problem(name: str = "bb72"):
    """A real CSS code, whose distance we already know independently.

    A contrived problem is a poor test here: the behaviour only matters when
    the search would otherwise take many trials to reach its floor, which is
    exactly the regime a toy problem does not reproduce.
    """
    from quilt_rep_search.codes_and_symmetries import Code

    code = Code.load(DEFS / f"{name}.json")
    n = max(max(r.bit_length() for r in code.x_checks),
            max(r.bit_length() for r in code.z_checks))

    def rows(masks):
        out = np.zeros((len(masks), n), dtype=np.uint8)
        for i, m in enumerate(masks):
            for c in range(n):
                out[i, c] = (m >> c) & 1
        return out

    return rows(code.x_checks), rows(code.z_checks), code.distance


@pytest.mark.skipif(not DEFS.exists(), reason="code definitions not available")
def test_engine_stops_early_at_min_distance() -> None:
    """min_distance is a stopping rule, not a filter. This is the whole bug."""
    checks, gauges, d = _code_problem()

    unbounded = qdist_rs.search_css(
        checks, gauges, trials=4000, seed=1, min_distance=0, threads=0)
    assert unbounded.trials_completed == 4000, (
        "with min_distance=0 the engine must spend its whole budget")
    assert not unbounded.target_hit
    assert int(unbounded.distance) == d, "sanity: the engine finds the code distance"

    # Asking it to stop at the weight it is going to find means it stops
    # almost at once.
    stopped = qdist_rs.search_css(
        checks, gauges, trials=4000, seed=1, min_distance=d, threads=0)
    assert stopped.trials_completed < 4000, (
        "min_distance = the attainable weight must cut the search short; "
        "if this fails the engine's early exit has changed and the screen's "
        "choice of threshold - 1 needs revisiting")
    assert stopped.target_hit, "the engine must report that it stopped early"
    assert int(stopped.distance) <= d

    # And the point of the fix: asking for strictly less than the floor makes
    # the engine spend the whole budget and find nothing below it.
    below = qdist_rs.search_css(
        checks, gauges, trials=4000, seed=1, min_distance=d - 1, threads=0)
    assert below.trials_completed == 4000, (
        "min_distance = d - 1 must not stop early on a weight-d operator")
    assert not below.target_hit
    assert int(below.distance) >= d


def test_screen_asks_for_strictly_below_threshold() -> None:
    """`screen()` must pass threshold - 1, so a pass is a real non-find.

    Checked by inspecting the call rather than by running a full screen, which
    needs the construction package and a real record.
    """
    source = (ROOT / "src" / "quilt_rep_search" / "distance_screen.py").read_text()
    assert "min_distance=threshold - 1" in source, (
        "screen() must search for operators strictly below the threshold; "
        "passing `threshold` stops the search at the first threshold-weight operator")
    assert "min_distance=threshold," not in source


def test_engine_result_carries_the_early_stop_flag() -> None:
    """A record must say whether its search stopped early.

    A record must say whether its search ran to budget. Every engine row
    carries `stopped_early`.
    """
    row = distance_screen.EngineResult(
        engine="qdist:grouped", trials=200_000, trials_completed=40, hits=1,
        seconds=0.1, distance=18, witness=False, target_hit=True)
    result = distance_screen.ScreenResult(passed=True, threshold=18, engines=[row])
    emitted = result.to_json()["engines"][0]
    assert emitted["stopped_early"] is True
    assert emitted["trials_completed"] == 40
    assert emitted["trials"] == 200_000, (
        "the requested budget and the depth actually searched must both be "
        "recorded")
