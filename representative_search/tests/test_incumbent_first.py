"""The held-representative-first path, and the seed that has to be reproducible.

These pin two properties of the screen-then-redraw pass that cannot be checked
by looking at a finished record.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quilt_rep_search.distance_screen import engine_seed, proof_covered


def test_engine_seed_is_stable_across_processes():
    """The recorded seed must reproduce the screen it labels.

    The engine seed must not depend on Python's `hash()`, which is randomised
    per process for strings: otherwise the same recorded seed would give a
    different engine seed on every run, and a recorded non-find could not be
    reproduced.

    Checked by running it in SEPARATE interpreters: calling it twice in one
    process cannot detect per-process randomisation.
    """

    program = (
        "import sys; sys.path.insert(0, %r);"
        "from quilt_rep_search.distance_screen import engine_seed;"
        "print(engine_seed(7, 'grouped'), engine_seed(7, 'three-block'))"
        % str(ROOT / "src")
    )
    runs = {subprocess.run([sys.executable, "-c", program],
                           capture_output=True, text=True, check=True).stdout.strip()
            for _ in range(3)}
    assert len(runs) == 1, f"engine seed varies across processes: {runs}"


def test_engine_seed_separates_the_engines():
    """Two engines on one problem must not search the same stream."""

    assert engine_seed(7, "grouped") != engine_seed(7, "three-block")
    assert engine_seed(7, "grouped") != engine_seed(8, "grouped")


def _corpus_record(scale: str, cohort: str) -> dict:
    path = (ROOT / "tests" / "fixtures" / "representatives"
            / f"{scale}__{cohort}.jsonl")
    if not path.exists():
        pytest.skip(f"{path} not present")
    with open(path) as stream:
        return json.loads(stream.readline())


def test_proof_covered_accepts_a_released_record_shape():
    """`proof_covered` must run on a finished record unchanged.

    The held representative is a finished record, not something the search
    just built, so the proof test and the screen both have to accept that
    shape directly. If this regresses, every held representative silently
    falls through to being screened -- or worse, to being replaced.
    """

    record = _corpus_record("bb288", "SC_BB_ZZ")
    assert isinstance(proof_covered(record), bool)


def test_proof_covered_requires_width_exactly_d():
    """The theorem's hypothesis is sum of stitch weights == D, not <= D.

    Above D it stops applying rather than degrading, which is why the clean
    rate falls off a cliff at W > d instead of sloping. A record certified on
    width <= d would be certified on a theorem that does not apply to it.
    """

    record = dict(_corpus_record("bb288", "SC_BB_ZZ"))
    threshold = int(record["operation_threshold"])
    record["pairing"] = dict(record["pairing"])
    record["pairing"]["width"] = threshold + 2
    assert proof_covered(record) is False


def test_incumbent_survivor_emits_no_new_geometry():
    """A target whose representative survives must keep it.

    The survivor path returns `record: None` and a verdict row. If it ever
    starts returning a record, every survivor silently acquires a fresh
    representative, and the sparsity and congestion data computed over the old
    one no longer describes it.
    """

    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_corpus as gc

    captured = {}

    def fake_screen_incumbent(record, *, trials, seed):
        captured["called"] = True
        return {"target_id": record["target_id"],
                "cohort": f'{record["scale"]}/{record["cohort"]}',
                "threshold": int(record["operation_threshold"]),
                "survived": True, "witness_weight": None,
                "witness_depth": None, "trials": trials,
                "budget_anomaly": False, "engines": [],
                "source": "incumbent", "seconds": 0.0}

    original = gc.screen_incumbent
    gc.screen_incumbent = fake_screen_incumbent
    try:
        incumbent = _corpus_record("bb288", "SC_BB_ZZ")
        raw = {"target_id": incumbent["target_id"],
               "blocks": incumbent["blocks"], "incumbent": incumbent}
        out = gc.one_target(("bb288/SC_BB_ZZ", raw, 1000, 10, 1))
    finally:
        gc.screen_incumbent = original

    assert captured.get("called"), "the incumbent was never screened"
    assert out["outcome"] == "ok"
    assert out["source"] == "incumbent"
    assert out["record"] is None, "a survivor must not emit new geometry"
    assert out["incumbent_screen"]["survived"] is True
