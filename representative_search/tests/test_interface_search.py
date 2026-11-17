#!/usr/bin/env python3
"""Invariants of the interface search.

The one that matters here is width agreement.  Every block must offer the same
number of stitching pieces, and reaching that is the hard part of the search,
for a reason that is easy to miss: a block's bank seeds all have the *same*
width, because the orbit expansion is built from translations and translation
preserves weight.  So a block cannot be nudged a little wider or narrower by
picking a different seed -- agreement has to be annealed for, and the blocks
frequently start far apart.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quilt_rep_search.codes_and_symmetries import Code
from quilt_rep_search.quilt_interface_search import search
from quilt_rep_search.quilt_targets import Block, Target, interface_width
from quilt_rep_search.representative_bank import expand, read

ROOT = Path(__file__).resolve().parents[1]
from quilt_rep_search.codes_and_symmetries import default_definitions

DEFS = default_definitions()

_CACHE: dict = {}


def fixtures():
    if not _CACHE:
        codes = {p.stem: Code.load(p) for p in sorted(DEFS.glob("*.json"))}
        _CACHE["codes"] = codes
        _CACHE["banks"] = {
            n: expand(read(ROOT / "banks" / f"{n}.json")[0], code=codes[n])
            for n in ("bb72", "bb288")
        }
        _CACHE["targets"] = json.loads((ROOT / "targets" / "population_v8.json").read_text())["cohorts"]
    return _CACHE["codes"], _CACHE["banks"], _CACHE["targets"]


def as_target(raw, codes, cohort):
    blocks = tuple(
        Block(codes[b["code"]], int(b["logical_x"], 16), int(b["logical_z"], 16))
        for b in raw["blocks"]
    )
    return Target(raw["target_id"], cohort, blocks)


def test_bank_seeds_within_a_class_share_one_width():
    """The premise of everything below: seeds give masks, not widths."""

    _, banks, _ = fixtures()
    for name, bank in banks.items():
        single = sum(1 for entries in bank.values() if len({e.weight for e in entries}) == 1)
        assert single > 0.85 * len(bank), (name, single, len(bank))


def test_targets_with_disjoint_block_widths_still_resolve():
    """Targets whose blocks start at disjoint widths must still yield an interface.

    Nothing about the search changed -- they were always reachable, just not
    from the three seeds the run happened to draw.  The generator now draws
    eight, so this asserts what the extra draws buy.
    """

    codes, banks, targets = fixtures()
    hard = {"SC_BB2_ZZZ-0071", "SC_BB2_ZZZ-0330", "SC_BB2_ZZZ-0533", "SC_BB2_ZZZ-0725"}
    for cohort in ("bb288/SC_BB2_ZZZ", "bb72/SC_BB2_ZZZ"):
        for raw in targets[cohort]:
            if raw["target_id"] not in hard:
                continue
            target = as_target(raw, codes, cohort.split("/")[1])
            # mirrors the generator's retry schedule
            found = None
            for retry in range(8):
                found = search(target, banks, seed=12345 + retry * 104729, keep=4)
                if found:
                    break
            assert found, (cohort, raw["target_id"])
            widths = [interface_width(b) for b in found.state]
            assert len(set(widths)) == 1, (cohort, raw["target_id"], widths)


def test_the_narrowest_candidate_is_not_always_the_usable_one():
    """Why the generator keeps a deep candidate list rather than a few.

    For two targets, candidates 0, 1 and 2 failed all sixteen pairings tried
    while candidate 14 passed -- and every candidate had the same interface
    width.  So neither width nor pairing separates them: some representatives
    simply admit a sub-threshold witness and others do not, and the search
    ranks by width, which knows nothing about that.

    This asserts only the cheap half -- that the search really does offer a
    deep list of distinct same-width candidates to choose from.  Running the
    distance screen here would cost minutes.
    """

    codes, banks, targets = fixtures()
    for raw in targets["bb72/SC_BB_ZZ"]:
        if raw["target_id"] != "SC_BB_ZZ-0688":
            continue
        target = as_target(raw, codes, "SC_BB_ZZ")
        found = search(target, banks, seed=11, keep=20)
        assert found and len(found.candidates) >= 15, len(found.candidates)
        assert len(set(found.candidates)) == len(found.candidates)
        assert len(set(found.widths)) == 1, found.widths


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all invariants hold")
