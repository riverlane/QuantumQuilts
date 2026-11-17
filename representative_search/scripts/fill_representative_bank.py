#!/usr/bin/env python3
"""Grow a representative bank until it covers every logical class.

    python scripts/fill_representative_bank.py --code bb72 \
        --definition ../construction/src/quilt_construction/data/code_definitions/bb72.json \
        --bank banks/bb72.json

Expands the bank, finds the labels it does not reach, searches for a clean
low-weight representative of one of them, adds it, and repeats.  Because a seed
covers its whole symmetry orbit, one find typically closes dozens of labels at
once, so the loop converges in a few tens of searches rather than thousands.

With `--from-scratch` the bank is rebuilt from nothing, so a bank can always
be regenerated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quilt_rep_search.representative_bank import Entry, expand, read, write
from quilt_rep_search.codes_and_symmetries import Code
from quilt_rep_search.coset_weight_search import find_representative

SECTORS = ("Z", "X")


def uncovered(bank, code: Code) -> list[tuple[str, int]]:
    """Every (sector, label) the bank cannot currently supply, lowest first."""

    return [
        (sector, label)
        for label in range(1, 1 << code.k)
        for sector in SECTORS
        if (sector, label) not in bank
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--max-searches", type=int, default=10_000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=20_000)
    args = parser.parse_args()

    code = Code.load(args.definition)
    if args.from_scratch or not args.bank.exists():
        seeds, provenance = [], {"code": code.code_id, "origin": "fill_bank --from-scratch"}
    else:
        seeds, provenance = read(args.bank)

    started = time.monotonic()
    searches = failures = 0
    added: list[Entry] = []
    log: list[dict] = []
    blocked: set[tuple[str, int]] = set()      # slots no search could close

    bank = expand(seeds, code=code) if seeds else {}
    while searches < args.max_searches:
        missing = [slot for slot in uncovered(bank, code) if slot not in blocked]
        if not missing:
            break
        sector, label = missing[0]

        began = time.monotonic()
        found = find_representative(
            code, sector, label,
            seed=label, restarts=args.restarts, sweeps=args.sweeps,
        )
        searches += 1
        elapsed = time.monotonic() - began

        if not found:
            failures += 1
            blocked.add((sector, label))
            log.append({"sector": sector, "label": f"0x{label:03x}",
                        "outcome": "no_clean_representative", "seconds": round(elapsed, 2)})
            print(f"  {sector} 0x{label:03x}: no clean representative ({elapsed:.1f}s)", flush=True)
            continue

        entry = Entry(sector=sector, label=label, mask=found.mask,
                      weight=found.weight, source_class=f"fill:{sector}:0x{label:03x}")
        seeds.append(entry)
        added.append(entry)
        bank = expand(seeds, code=code)        # one seed closes a whole orbit
        log.append({"sector": sector, "label": f"0x{label:03x}", "outcome": "found",
                    "weight": found.weight, "gate_calls": found.gate_calls,
                    "seconds": round(elapsed, 2), "slots_covered_after": len(bank)})
        print(f"  {sector} 0x{label:03x}: weight {found.weight:3}  {elapsed:5.1f}s"
              f"  -> {len(bank)} slots covered", flush=True)

    bank = expand(seeds, code=code)
    still = uncovered(bank, code)
    provenance = {
        **provenance,
        "code": code.code_id,
        "seeds_total": len(seeds),
        "seeds_added_by_fill": len(added),
        "searches_run": searches,
        "searches_failed": failures,
        "labels_uncovered": len(still),
        "fill_seconds": round(time.monotonic() - started, 1),
        "fill_log": log,
        "gate": "cleanliness.inspect, run unconditionally on every seed",
    }
    write(args.bank, seeds, provenance=provenance)
    print(f"{code.code_id}: {len(seeds)} seeds ({len(added)} new), "
          f"{len(bank)} of {2 * ((1 << code.k) - 1)} slots covered, "
          f"{len(still)} missing, {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
