"""The representative bank: a few seeds per class, expanded by symmetry.

A bank entry is one low-weight, clean representative of one logical class.
Expanding it over the ell*m torus translations and the duality phi turns each
entry into one candidate for each of many other classes, in both sectors.  That
is why a bank of a few hundred entries covers all 4,095 nonzero labels twice
over: it is storing orbit seeds, not representatives.

The bank is a seed source for the annealer, not a substitute for it.  A quilt
needs two blocks of *equal* interface width, and two independently chosen
low-weight representatives rarely happen to match, so the annealer still moves
within the cosets to equalise.  What the bank buys is a good starting point --
cold annealing lands roughly 20% wider.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from .codes_and_symmetries import Code, all_shifts, bit_count, dualise, translate
from .interface_cleanliness import inspect


@dataclass(frozen=True)
class Entry:
    """One seed: a clean representative of `label` in `sector`."""

    sector: str
    label: int
    mask: int
    weight: int
    source_class: str

    def to_json(self) -> dict:
        return {
            "sector": self.sector,
            "label": f"0x{self.label:03x}",
            "mask": f"0x{self.mask:x}",
            "weight": self.weight,
            "source_class": self.source_class,
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Entry":
        return cls(
            sector=str(raw["sector"]),
            label=int(raw["label"], 16),
            mask=int(raw["mask"], 16),
            weight=int(raw["weight"]),
            source_class=str(raw["source_class"]),
        )


def keep_best(entries: Iterable[Entry], *, code: Code, per_class: int) -> list[Entry]:
    """Drop unclean seeds, then keep the `per_class` lightest of each class."""

    clean = [e for e in entries if inspect(e.mask, code=code, sector=e.sector).clean]
    grouped: dict[str, list[Entry]] = {}
    for entry in clean:
        grouped.setdefault(entry.source_class, []).append(entry)
    kept = []
    for _class_id, group in sorted(grouped.items()):
        group.sort(key=lambda e: (e.weight, e.mask))
        kept.extend(group[:per_class])
    return kept


def expand(entries: Sequence[Entry], *, code: Code) -> dict[tuple[str, int], list[Entry]]:
    """Push every seed around the torus, and through phi, into both sectors.

    Returns a map from (sector, label) to candidates, lightest first.  A seed of
    weight w produces only weight-w candidates: translation and duality are
    permutations of the qubits.
    """

    found: dict[tuple[str, int], dict[int, Entry]] = {}

    def offer(entry: Entry) -> None:
        slot = found.setdefault((entry.sector, entry.label), {})
        slot.setdefault(entry.mask, entry)

    for seed in entries:
        # phi carries the seed into the opposite sector; translations then move
        # both copies around their torus.  The seed's own sector is whatever it
        # says it is -- assuming Z here silently mislabels every X-sector seed.
        partner_sector = "X" if seed.sector == "Z" else "Z"
        partner = dualise(seed.mask, code)
        for dx, dy in all_shifts(code):
            own = translate(seed.mask, code, dx, dy)
            offer(Entry(seed.sector, code.label(own, seed.sector), own,
                        seed.weight, seed.source_class))
            other = translate(partner, code, dx, dy)
            offer(Entry(partner_sector, code.label(other, partner_sector), other,
                        seed.weight, seed.source_class))

    return {
        key: sorted(slot.values(), key=lambda e: (e.weight, e.mask))
        for key, slot in found.items()
        if key[1] != 0            # the identity class needs no representative
    }


_IMPROVISED: dict[tuple[str, str, int], list[Entry]] = {}
def improvise(code: Code, sector: str, label: int, *, count: int) -> list[Entry]:
    """Low-weight representatives for a code that has no bank.

    The rotated surface blocks carry a single logical class, so there is no
    orbit for a bank to spread over and no bank file to read.  Falling back to
    one canonical representative is worse than it looks: every annealing restart
    then begins from the *same* interface on that block, so the candidates all
    inherit its weaknesses and a rejection cannot be recovered by trying the
    next one.  Here we search for several distinct representatives instead.

    There is exactly one class per surface code, so this is computed once and
    reused for every target.
    """

    from .coset_weight_search import find_representative

    key = (code.code_id, sector, label)

    # A rotated surface code gets constructed paths rather than annealed ones:
    # exact weights, so the starts span a band of widths, and simple paths by
    # construction rather than whatever the coset walk happens to land on.
    if code.family_is_rotated_surface and sector == "Z":
        cached = _IMPROVISED.get(key)
        if cached is None or len(cached) < count:
            from .surface_paths import paths_over_widths
            widths = range(code.distance, code.distance + 2 * count, 2)
            cached = [
                Entry("Z", label, path.mask, path.weight, f"surface-path:w{path.weight}")
                for path in paths_over_widths(code, widths)
            ]
            _IMPROVISED[key] = cached
        return cached[:count]

    cached = _IMPROVISED.get(key)
    if cached is not None and len(cached) >= count:
        return cached[:count]

    seen: dict[int, Entry] = {e.mask: e for e in (cached or [])}
    attempt = 0
    while len(seen) < count and attempt < 12 * count:
        found = find_representative(
            code, sector, label, seed=0x51 + attempt, restarts=2, sweeps=8000,
        )
        attempt += 1
        if found and found.mask not in seen:
            seen[found.mask] = Entry(sector, label, found.mask, found.weight,
                                     f"improvised:{code.code_id}")
    # a restart that wandered can return something far heavier than the rest;
    # such a start is worse than useless, so keep only the competitive ones
    if seen:
        floor = min(e.weight for e in seen.values())
        seen = {m: e for m, e in seen.items() if e.weight <= max(floor + 4, floor * 3 // 2)}
    entries = sorted(seen.values(), key=lambda e: (e.weight, e.mask))
    _IMPROVISED[key] = entries
    return entries[:count]


def write(path: Path, entries: Sequence[Entry], *, provenance: dict) -> None:
    payload = {
        "schema": "quilt-representative-bank-v6",
        "provenance": provenance,
        "entries": [entry.to_json() for entry in entries],
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def read(path: Path) -> tuple[list[Entry], dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Entry.from_json(raw) for raw in payload["entries"]], payload["provenance"]
