#!/usr/bin/env python3
"""Freeze the target population, before any search touches it.

    python scripts/freeze_targets.py --out targets/population_v8.json --per-cohort 2500

A target names the logical class to be measured on each block.  Freezing the
population first, and hashing it, is what makes the yield figures meaningful: a
population chosen after seeing which targets were easy would not be a sample of
anything.

Eligibility.  Under the strict weight-two stitching primitive the blocks'
support parities must agree.  Support parity is invariant under dressing by the
even-weight BB stabilisers and equals the quadratic form

    q(a, b) = popcount(a & b) mod 2

on the (logical_x, logical_z) label pair.  For a pure X or pure Z target one of
a, b vanishes, so q = 0 automatically and every class is eligible.  For generic
targets we require only that q *agrees* across blocks, which is the true
condition; demanding q = 0 on every block independently would sample only the
all-even sector, a quarter of the eligible space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quilt_rep_search.codes_and_symmetries import Code, default_definitions, bit_count

# Each cohort: the block codes, and whether the targets are pure Z or generic.
COHORTS = {
    "BB2_ZZ":              {"blocks": ("bb", "bb"),             "kind": "pure_z"},
    "BB2_GENERIC_MATCHED": {"blocks": ("bb", "bb"),             "kind": "generic"},
    "SC_BB_ZZ":            {"blocks": ("surface", "bb"),        "kind": "pure_z"},
    "SC_BB2_ZZZ":          {"blocks": ("surface", "bb", "bb"),  "kind": "pure_z"},
}
# The 144-qubit gross code sits between the other two, and its surface partner
# is chosen to match its distance, as at the other scales.  Each cohort draws
# from its own generator, seeded by (seed, scale, cohort), so adding a scale
# leaves every existing cohort's targets bit-identical.
SCALES = {
    "bb72":  {"bb": "bb72",  "surface": "rotated_surface_d6"},
    "bb144": {"bb": "bb144", "surface": "rotated_surface_d12"},
    "bb288": {"bb": "bb288", "surface": "rotated_surface_d18"},
}


def q_form(logical_x: int, logical_z: int) -> int:
    return bit_count(logical_x & logical_z) & 1


def draw(cohort: str, scale: str, codes: dict, count: int, rng: random.Random) -> list[dict]:
    spec = COHORTS[cohort]
    roles = spec["blocks"]
    seen: set[tuple] = set()
    targets: list[dict] = []
    attempts = 0
    limit = 500 * count

    while len(targets) < count and attempts < limit:
        attempts += 1
        blocks = []
        parities = set()
        for role in roles:
            code = codes[SCALES[scale][role]]
            if role == "surface":
                logical_x, logical_z = 0, 1        # k = 1, the pure Z logical
            elif spec["kind"] == "pure_z":
                logical_x, logical_z = 0, rng.randrange(1, 1 << code.k)
            else:
                logical_x = rng.randrange(0, 1 << code.k)
                logical_z = rng.randrange(0, 1 << code.k)
                if logical_x == 0 and logical_z == 0:
                    break
            parities.add(q_form(logical_x, logical_z))
            blocks.append({"code": code.code_id,
                           "logical_x": f"0x{logical_x:03x}",
                           "logical_z": f"0x{logical_z:03x}"})
        if len(blocks) != len(roles) or len(parities) != 1:
            continue                                # parities must agree
        # blocks of the same code are interchangeable: measuring L1 on block 0
        # and L2 on block 1 is the same operation as the swap, so the dedup key
        # is order-insensitive within each group of identical codes.
        key = tuple(sorted((b["code"], b["logical_x"], b["logical_z"]) for b in blocks))
        if key in seen:
            continue
        seen.add(key)
        targets.append({"target_id": f"{cohort}-{len(targets):04d}", "blocks": blocks})

    if len(targets) < count:
        raise RuntimeError(
            f"{scale}/{cohort}: only {len(targets)} distinct targets in {attempts} draws; "
            "the eligible space may be smaller than the requested cohort"
        )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definitions", type=Path,
                        default=None,
                        help="defaults to the quilt_construction copy")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cohort", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--scales", help="comma-separated subset of scales to draw")
    args = parser.parse_args()
    if args.definitions is None:
        args.definitions = default_definitions()

    codes = {p.stem: Code.load(p) for p in sorted(args.definitions.glob("*.json"))}
    population = {}
    wanted = args.scales.split(",") if args.scales else list(SCALES)
    for scale in wanted:
        for cohort in COHORTS:
            rng = random.Random(f"{args.seed}:{scale}:{cohort}")
            rows = draw(cohort, scale, codes, args.per_cohort, rng)
            population[f"{scale}/{cohort}"] = rows
            space = len({tuple(sorted((b["logical_x"], b["logical_z"]) for b in r["blocks"]))
                         for r in rows})
            print(f"  {scale:6}/{cohort:22} {len(rows):5} targets, {space} distinct")

    payload = {
        "schema": "quilt-rep-target-population-v6",
        "seed": args.seed,
        "per_cohort": args.per_cohort,
        "eligibility": "q(a,b) = popcount(a & b) mod 2 must agree across blocks",
        "cohorts": population,
    }
    body = json.dumps(payload["cohorts"], sort_keys=True, separators=(",", ":"))
    payload["population_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    total = sum(len(v) for v in population.values())
    print(f"{total} targets over {len(population)} cohorts -> {args.out}")
    print(f"population_sha256 = {payload['population_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
