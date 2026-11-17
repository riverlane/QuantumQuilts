"""Invariants of the bank and its gate.

Run with:  python -m pytest tests -q      (or just: python tests/test_bank.py)
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quilt_rep_search.representative_bank import Entry, expand, read
from quilt_rep_search.interface_cleanliness import inspect
from quilt_rep_search.codes_and_symmetries import Code, bit_count, dualise, translate

ROOT = Path(__file__).resolve().parents[2]
from quilt_rep_search.codes_and_symmetries import default_definitions

DEFS = default_definitions()
BANKS = Path(__file__).resolve().parents[1] / "banks"


def code(name):
    return Code.load(DEFS / f"{name}.json")


def test_translation_is_a_code_symmetry():
    """Translating a closed operator keeps it closed and keeps its weight."""
    for name in ("bb72", "bb288"):
        c = code(name)
        rng = random.Random(0)
        for _ in range(20):
            mask = c.representative(rng.randrange(1, 1 << c.k), "Z")
            moved = translate(mask, c, rng.randrange(c.ell), rng.randrange(c.m))
            assert c.is_closed(moved, "Z")
            assert bit_count(moved) == bit_count(mask)


def test_duality_swaps_sectors():
    """phi carries a closed Z operator to a closed X operator of equal weight."""
    for name in ("bb72", "bb288"):
        c = code(name)
        mask = c.representative(0x5a5, "Z")
        dual = dualise(mask, c)
        assert c.is_closed(dual, "X")
        assert bit_count(dual) == bit_count(mask)


def test_gate_rejects_an_overmeasuring_support():
    """Two independent logical classes inside one support must be caught."""
    c = code("bb72")
    dirty = c.representative(0x001, "Z") | c.representative(0x002, "Z")
    verdict = inspect(dirty, code=c, sector="Z")
    assert not verdict.clean and verdict.measured_rank >= 2


def test_expanded_candidates_stay_clean_and_keep_their_label():
    """The whole bank rests on this: symmetry preserves the gate."""
    for name in ("bb72", "bb288"):
        c = code(name)
        seeds, _provenance = read(BANKS / f"{name}.json")
        bank = expand(seeds, code=c)
        rng = random.Random(1)
        sample = rng.sample(sorted(bank), 60)
        for sector, label in sample:
            for candidate in bank[(sector, label)][:2]:
                assert candidate.label == label
                assert c.label(candidate.mask, sector) == label, "claimed label is wrong"
                assert c.is_closed(candidate.mask, sector), "candidate is not closed"
                assert inspect(candidate.mask, code=c, sector=sector).clean, "expansion broke the gate"


def test_gate_rejects_a_contained_stabiliser():
    """A representative that swallows a whole stabiliser is reducible."""
    c = code("bb72")
    logical = c.representative(0x001, "Z")
    stabiliser = next(s for s in c.z_checks if s & ~logical)   # not already inside
    verdict = inspect(logical | stabiliser, code=c, sector="Z")
    assert not verdict.clean, "kernel must be span(all-ones), not merely rank one in labels"
    assert verdict.kernel_dimension > 1


def test_expansion_covers_an_x_sector_seed():
    """Adding an X seed must close X slots -- expansion follows the seed's sector.

    Uses a deliberately truncated bank so that gaps exist: the bank in this
    folder is complete, and would not exercise the gap-finding.
    """
    c = code("bb72")
    seeds, _ = read(BANKS / "bb72.json")
    truncated = seeds[:1]
    before = expand(truncated, code=c)
    uncovered_x = [lab for lab in range(1, 1 << c.k) if ("X", lab) not in before]
    assert uncovered_x, "truncated bank should still have gaps"

    from quilt_rep_search.coset_weight_search import find_representative
    target = uncovered_x[0]
    found = find_representative(c, "X", target, seed=target, restarts=2, sweeps=5000)
    assert found, "could not find an X representative to test with"

    grown = expand(list(truncated) + [Entry("X", target, found.mask, found.weight, "test")], code=c)
    assert ("X", target) in grown, "an X seed failed to cover its own slot"
    assert len(grown) > len(before)


def test_bank_round_trips():
    for name in ("bb72", "bb288"):
        seeds, provenance = read(BANKS / f"{name}.json")
        assert seeds and provenance["code"] == name
        assert provenance["seeds_dropped_unclean"] >= 0
        assert all(isinstance(e, Entry) for e in seeds)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print(f"ok  {fn.__name__}")
    print("all invariants hold")
