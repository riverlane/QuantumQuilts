#!/usr/bin/env python3
"""Draw a fresh representative for a target whose circuit conceded.

The target never changes.  The redraw continues the strategy of
scripts/generate_corpus.py past the representative already drawn, and it uses
that script's own pieces (the interface search, the record builder, the
code-capacity screen):

  positions 0..3     four re-pairings of the representative the target had
  positions 4..      the annealer's candidates in its order, each with its
                     deterministic pairing and then four re-pairings

At each position the record is built, skipped if it is one already drawn for
this target, and screened at code capacity (200,000 trials per engine).  The
first that survives is the redraw; the caller builds and folds it.  A later
redraw of the same target starts at the position after the last one taken, so
the walk is one fixed sequence per target and no representative is used twice.

Everything is a function of the target, the starting position and the list of
representatives already drawn: no clock, no unseeded randomness.

    python scripts/redraw.py RECORD.json --out NEW.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

SEARCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEARCH / "src"))
sys.path.insert(0, str(SEARCH / "scripts"))

TRIALS = 200_000        # code-capacity trials per engine, as for the data
KEEP = 100              # annealer candidates kept, as for the data
REPAIRINGS = 4          # re-pairings tried per representative, as for the data
SEARCH_RETRIES = 12
SEED = 20261116


def pairing_sha(record: dict) -> str:
    """The name of a representative-with-pairing ."""
    return hashlib.sha256(json.dumps(record["pairing"]["coordinates"], sort_keys=True).encode()).hexdigest()[:12]


def _pipeline():
    import generate_corpus as gc                       # the per-target code itself; not a copy
    if not gc.STATE:
        from quilt_rep_search.codes_and_symmetries import default_definitions
        gc.setup(str(default_definitions()), str(SEARCH / "banks"))
    return gc


def draw(base: dict, *, start: int = 0, exclude=(), trials: int = TRIALS, log=print) -> dict:
    """The next representative of `base`'s target from position `start`.

    `base` is the target's first representative record; `exclude` holds the
    pairing hashes of every representative already drawn for the target.
    Returns {"outcome": "ok", "record", "pairing", "position", "next", ...}
    or an outcome naming why there is none.
    """
    gc = _pipeline()
    from quilt_rep_search.corpus_record import build_record
    from quilt_rep_search.distance_screen import proof_covered, screen
    from quilt_rep_search.quilt_interface_search import search
    from quilt_rep_search.quilt_targets import Block, Target

    scale, cohort, tid = base["scale"], base["cohort"], base["target_id"]
    codes, banks = gc.STATE["codes"], gc.STATE["banks"]
    target = Target(tid, cohort, tuple(
        Block(codes[b["code"]], int(b["logical_x"], 16), int(b["logical_z"], 16)) for b in base["blocks"]))
    base_state = tuple((int(b["x_mask"], 16), int(b["z_mask"], 16)) for b in base["blocks"])
    salt = zlib.crc32(f"{scale}/{tid}".encode())
    seen = set(exclude)
    rejected, screens, found, search_seconds = [], 0, None, 0.0
    began = time.monotonic()
    summary = lambda: {"start": start, "code_capacity_screens": screens, "rejected": rejected,   # noqa: E731
                       "search_seconds": round(search_seconds, 1), "seconds": round(time.monotonic() - began, 1)}

    p = start
    while True:
        if p < REPAIRINGS:
            state, where = base_state, {"representative": "incumbent", "pairing_index": p + 1}
            pairing_seed = salt * 1000 + p
        else:
            if found is None:
                t0 = time.monotonic()
                for retry in range(SEARCH_RETRIES):
                    found = search(target, banks, seed=SEED + retry * 104729, keep=KEEP)
                    if found:
                        break
                search_seconds = time.monotonic() - t0
                if not found:
                    return {"outcome": "no_interface_found", **summary()}
            c, j = divmod(p - REPAIRINGS, REPAIRINGS + 1)
            if c >= len(found.candidates):
                return {"outcome": "all_candidates_rejected", "candidates": len(found.candidates), **summary()}
            state, where = found.candidates[c], {"representative": f"annealer candidate {c}", "pairing_index": j}
            pairing_seed = None if j == 0 else salt * 1000 + p
        record = build_record(target, state, scale=scale, telemetry={}, pairing_seed=pairing_seed)
        sha = pairing_sha(record)
        if sha in seen:
            p += 1
            continue
        seen.add(sha)
        proved = proof_covered(record)
        verdict = None
        if not proved:
            verdict = screen(record, trials=trials, seed=SEED + p)
            screens += 1
        if verdict is not None and not verdict.passed:
            weight = min((e.distance for e in verdict.engines if e.witness and e.distance is not None), default=None)
            depth = min((e.trials_completed for e in verdict.engines if e.witness), default=None)
            rejected.append({"position": p, **where, "pairing": sha, "witness_weight": weight, "witness_depth": depth})
            log(f"{scale} {tid}: position {p} ({where['representative']}, pairing {where['pairing_index']}) "
                f"rejected at code capacity, weight {weight} at trial {depth}")
            p += 1
            continue
        record["telemetry"] = {"redraw": {"position": p, **where, "pairing_strategy": record["pairing"]["strategy"],
                                          "pairing_resample_seed": pairing_seed, "search_seed": getattr(found, "seed", None),
                                          "excluded": sorted(exclude), "proof_covered": proved, **summary()}}
        record["validation"] = (
            {"weight_d_theorem": {"outcome": "proved", "theorem": "weightDquilt",
                                  "threshold": record["operation_threshold"], "width": record["pairing"]["width"]}}
            if proved else {"distance_screen": verdict.to_json()})
        log(f"{scale} {tid}: position {p} ({where['representative']}, pairing {where['pairing_index']}) taken, "
            f"{'proved' if proved else f'{trials:,} code-capacity trials survived'}, {screens} screen(s), "
            f"{time.monotonic() - began:.0f}s")
        return {"outcome": "ok", "record": record, "pairing": sha, "position": p, "next": p + 1,
                "representative": where["representative"], "pairing_index": where["pairing_index"],
                "proof_covered": proved, **summary()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path, help="the target's first representative record (one JSON object)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=None, help="pairing hashes already drawn (default: the record's own)")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base = json.loads(a.record.read_text())
    out = draw(base, start=a.start, exclude=a.exclude if a.exclude is not None else [pairing_sha(base)], trials=a.trials)
    if out["outcome"] != "ok":
        sys.exit(f"no redraw: {out['outcome']}")
    a.out.write_text(json.dumps(out["record"]))
    print(f"pairing {out['pairing']}, position {out['position']}, next start {out['next']}")


if __name__ == "__main__":
    main()
