#!/usr/bin/env python3
"""Produce representative records for a population of targets.

    python scripts/generate_corpus.py --targets targets/population_v8.json \
        --out OUT --workers 9

One worker per cohort, so a surprise in any cohort shows up in the first
minutes rather than after another cohort has finished.  Each cohort streams to
its own JSONL file and its own checkpoint, so the run resumes where it stopped.

Per target: search the interface, then screen candidates in width order until
one survives.  Every record carries how it was made -- which bank seed, how
long, how many candidates the distance search rejected -- because the absence
of exactly that information is what made the previous corpus impossible to
reason about.
"""

from __future__ import annotations

import argparse
import json
import collections
import fcntl
from itertools import zip_longest
import os
from multiprocessing import Pool
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quilt_rep_search.codes_and_symmetries import Code, default_definitions
from quilt_rep_search.corpus_record import build_record
from quilt_rep_search.distance_screen import proof_covered, screen
from quilt_rep_search.quilt_interface_search import search
from quilt_rep_search.quilt_targets import Block, Target
from quilt_rep_search.representative_bank import expand, read

STATE: dict = {}


def setup(definitions: str, banks_dir: str) -> None:
    """Load codes and banks once per worker."""

    definitions = Path(definitions)
    STATE["codes"] = {p.stem: Code.load(p) for p in sorted(definitions.glob("*.json"))}
    STATE["banks"] = {
        path.stem: expand(read(path)[0], code=STATE["codes"][path.stem])
        for path in sorted(Path(banks_dir).glob("*.json"))
        if path.stem in STATE["codes"]
    }


def screen_incumbent(record: dict, *, trials: int, seed: int) -> dict:
    """Screen the representative a target already has, before drawing a new one.

    Screening a held representative and drawing a new one are the same
    operation: the held representative is simply the first candidate. The
    expected cost per target is T_clean + ((1-p)/p) * T_fail either way,
    because exactly one candidate ever spends the full budget and the
    rejections exit early, so trying the held one first is free. What it buys
    is that a target whose existing representative is sound keeps it.

    Returns a verdict row -- cohort, target_id, survived, witness_depth,
    witness_weight, trials, threshold -- where `witness_depth` is a minimum
    over engines.

    `None` means the record is proof-covered and was NOT screened. Such a
    record must not enter the reliability fit at all: it is not a survivor of
    a randomised search, it is proved, and counting it as a survivor would
    bias the fit towards the screen looking more reliable than it is.
    """

    from quilt_rep_search.distance_screen import proof_covered, screen

    if proof_covered(record):
        return None

    began = time.monotonic()
    verdict = screen(record, trials=trials, seed=seed)
    depths = [e.trials_completed for e in verdict.engines if e.witness]
    weights = [e.distance for e in verdict.engines
               if e.witness and e.distance is not None]
    return {
        "target_id": record["target_id"],
        "cohort": f'{record["scale"]}/{record["cohort"]}',
        "threshold": int(record["operation_threshold"]),
        "survived": verdict.passed,
        "witness_weight": min(weights) if weights else None,
        "witness_depth": min(depths) if depths else None,
        "trials": trials,
        # A clean record must have spent its whole budget: with
        # min_distance = threshold - 1 the engines exit early only on a real
        # witness. trials_completed < trials on a survivor means the budget
        # was not honoured.
        "budget_anomaly": bool(verdict.passed and any(
            e.trials_completed < e.trials for e in verdict.engines)),
        "engines": [{"engine": e.engine, "trials": e.trials,
                     "trials_completed": e.trials_completed,
                     "lowest_weight_seen": e.distance, "witness": e.witness,
                     "stopped_early": e.target_hit} for e in verdict.engines],
        "source": "incumbent",
        "seconds": round(time.monotonic() - began, 2),
    }


def one_target(job: tuple, *, on_incumbent=None,
               time_budget: float | None = None) -> dict:
    """Screen the representative a target already holds, and draw another if it fails.

    `on_incumbent`, when given, is called with the held representative's
    verdict row the moment it exists, before any redrawing starts. The verdict
    is finished within the first screen, while the redraw that follows can run
    much longer, so it is handed over immediately.

    `time_budget`, when given, caps the wall clock of the CANDIDATE LOOP in
    seconds. It cannot cap the first screen, which is a single uninterruptible
    call into qdist. Measured on bb144, SC_BB_ZZ needs a median of 4 screens
    per redraw but reaches 438. The cap turns such a tail into one recorded
    failure: the target keeps its verdict row, and only its replacement is
    missing.
    """
    key, raw, trials, keep, seed = job
    scale, cohort = key.split("/")

    # The held representative, when the targets file carries one. Screening it
    # first makes this function screen and redraw in a single pass.
    incumbent = raw.get("incumbent")
    incumbent_row = None
    if incumbent is not None:
        incumbent_row = screen_incumbent(incumbent, trials=trials, seed=seed)
        if incumbent_row is not None and on_incumbent is not None:
            # Before the redraw, not after. See the docstring.
            on_incumbent(incumbent_row)
        if incumbent_row is None:
            # Proof-covered by Theorem `weightDquilt`. Nothing to screen and
            # nothing to redraw; the record stands on the theorem.
            return {"target_id": raw["target_id"], "cohort": key,
                    "outcome": "ok", "source": "incumbent",
                    "width": int(incumbent["pairing"]["width"]),
                    "proof_covered": True, "incumbent_screen": None,
                    "record": None}
        if incumbent_row["survived"]:
            # A survivor keeps its existing representative. No new record is
            # emitted; the only new thing on disk is the verdict row.
            return {"target_id": raw["target_id"], "cohort": key,
                    "outcome": "ok", "source": "incumbent",
                    "width": int(incumbent["pairing"]["width"]),
                    "proof_covered": False,
                    "incumbent_screen": incumbent_row, "record": None}

    # Only a target that still needs a NEW representative gets past here, so
    # the codes and banks are loaded below rather than above: a survivor needs
    # neither, and survivors are the overwhelming majority.
    codes, banks = STATE["codes"], STATE["banks"]

    blocks = tuple(
        Block(codes[b["code"]], int(b["logical_x"], 16), int(b["logical_z"], 16))
        for b in raw["blocks"]
    )
    target = Target(raw["target_id"], cohort, blocks)

    # Whether a target yields an interface is seed-sensitive, and a few targets
    # per thousand are marginal enough to lose several draws in a row.  The
    # blocks of such a target start far apart in width -- a class's bank seeds
    # all share one width, since the orbit expansion translates and translation
    # preserves weight -- and the spread the annealer descends on is flat along
    # most of the route to agreement, so a marginal target stays marginal.
    #
    # Measured per-seed success on marginal targets is 0.67 to 1.0, so three
    # draws leave a few in ten thousand unreached while eight leave none.
    # Extra draws are the same algorithm on another seed: a target that already
    # succeeds is unaffected, so no cohort's width statistics move.  A search
    # is ~1 s against a ~19 s screen, and only failures pay at all.
    began = time.monotonic()
    for retry in range(12):
        found = search(target, banks, seed=seed + retry * 104729, keep=keep)
        if found:
            break
    search_seconds = time.monotonic() - began
    if not found:
        return {"target_id": raw["target_id"], "cohort": key, "outcome": "no_interface_found",
                "incumbent_screen": incumbent_row,
                "search_seconds": round(search_seconds, 2), "search_retries": 12}

    # Try a representative, then four re-pairings of it, then the next.
    # Re-pairing rescues many rejections, the deterministic pairing being tuned
    # for H_ac weight rather than for distance -- but it does not rescue all of
    # them, and the depth of the candidate list matters independently.  On the
    # two targets that exhausted the old budget, candidates 0, 1 and 2 each
    # failed all sixteen pairings tried, while candidate 14 passed; every
    # candidate had the same width, so what separates them is the
    # representative itself, not how wide it is or how it was paired.
    # Only a rejection pays for the extra attempts -- the overwhelming majority
    # of targets pass on the first candidate with its deterministic pairing --
    # so the budget is set by what the stubborn few need, not by the average.
    rejected = []
    repair_began = time.monotonic()
    attempt = 0
    candidates_tried = 0
    screens_run = 0
    proved = False
    for candidate_index, (state, width) in enumerate(zip(found.candidates, found.widths)):
        candidates_tried = candidate_index + 1
        # Checked per candidate rather than per pairing: a pairing of an
        # already-built record is cheap next to the screen it triggers, and
        # cutting between the pairings of one candidate would leave that
        # candidate half-tried for no saving worth having.
        if (time_budget is not None
                and time.monotonic() - repair_began > time_budget):
            return {"target_id": raw["target_id"], "cohort": key,
                    "outcome": "repair_budget_exhausted",
                    "incumbent_screen": incumbent_row,
                    "repair_seconds": round(time.monotonic() - repair_began, 1),
                    "time_budget": time_budget,
                    "candidates_tried": candidates_tried - 1,
                    "candidates_offered": len(found.candidates),
                    "search_seconds": round(search_seconds, 2),
                    "rejected": rejected}
        pairings = [None] + [seed * 7919 + attempt + k for k in range(4)]
        for pairing_index, pairing_seed in enumerate(pairings):
            record = build_record(target, state, scale=scale, telemetry={},
                                  pairing_seed=pairing_seed)

            # Theorem `weightDquilt` settles this record outright when the
            # stitch weights sum to exactly D, so screening it would spend the
            # whole budget to re-derive, as evidence, something already proved.
            # Cheap to check, and it is the only route by which a record
            # carries no false-negative rate at all.
            if proof_covered(record):
                proved = True
                verdict = None
            else:
                verdict = screen(record, trials=trials, seed=seed + attempt)
                screens_run += 1
            attempt += 1
            if verdict is not None and not verdict.passed:
                # Keep the depth and the weight of a rejection's witness:
                # these are the numbers a budget has to beat.
                rejected.append({"width": width,
                                 # Which candidate, and which pairing of it.
                                 # Without these a target's rejection list is a
                                 # flat tally and you cannot tell a
                                 # representative that fails under every pairing
                                 # from one unlucky pairing of a good
                                 # representative -- the distinction that
                                 # decides whether a target is merely hard or
                                 # has no distance-preserving quilt at all.
                                 "candidate_index": candidate_index,
                                 "pairing_index": pairing_index,
                                 "pairing": record["pairing"]["strategy"],
                                 "by": verdict.rejected_by,
                                 "witness_weight": min(
                                     (e.distance for e in verdict.engines
                                      if e.witness and e.distance is not None),
                                     default=None),
                                 "witness_depth": min(
                                     (e.trials_completed for e in verdict.engines
                                      if e.witness), default=None),
                                 "engines": [
                                     {"engine": e.engine, "trials": e.trials,
                                      "trials_completed": e.trials_completed,
                                      "witness": e.witness,
                                      "lowest_weight_seen": e.distance}
                                     for e in verdict.engines]})
                continue
            record["telemetry"] = {
                "pairing_strategy": record["pairing"]["strategy"],
                "screen_attempts": attempt,
                "pairing_resample_seed": record["pairing"]["resample_seed"],
                "search_seed": found.seed,
                "search_retries": retry,
                "bank_starts": found.starts_used,
                "search_seconds": round(search_seconds, 2),
                "search_gate_calls": found.gate_calls,
                "search_accepted_moves": found.accepted_moves,
                "bank_start_rank": found.start_rank,
                "solved_by_bank_alone": found.started_feasible,
                "candidates_offered": len(found.candidates),
                "candidates_tried": candidates_tried,
                # Screens actually RUN, not attempts made. A proved record
                # consumes an attempt slot but spends no budget, and reporting
                # otherwise would overstate the evidence behind it.
                "screens_spent": screens_run,
                "candidates_rejected_by_screen": len(rejected),
                "accepted_candidate_index": candidate_index,
                "accepted_pairing_index": pairing_index,
                "proof_covered": proved,
                "rejected": rejected,
            }
            # A proved record has no screen to report. Say so explicitly rather
            # than emitting an empty screen result that a reader would mistake
            # for a non-find: the two are not the same kind of claim.
            record["validation"] = (
                {"weight_d_theorem": {"outcome": "proved",
                                      "theorem": "weightDquilt",
                                      "threshold": record["operation_threshold"],
                                      "width": record["pairing"]["width"]}}
                if proved else
                {"distance_screen": verdict.to_json()})
            return {"target_id": raw["target_id"], "cohort": key,
                    "outcome": "ok", "width": width, "source": "repair",
                    "incumbent_screen": incumbent_row,
                    "proof_covered": proved, "record": record}

    return {"target_id": raw["target_id"], "cohort": key, "outcome": "all_candidates_rejected",
            "incumbent_screen": incumbent_row,
            "search_seconds": round(search_seconds, 2), "rejected": rejected}


def already_done(out: Path, key: str) -> set[str]:
    stream = out / f"{key.replace('/', '__')}.jsonl"
    if not stream.exists():
        return set()
    return {json.loads(line)["target_id"]
            for line in stream.read_text().splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, default=None,
                        help="code definitions dir; defaults to the copy "
                             "packaged with quilt_construction")
    parser.add_argument("--banks", type=Path, default=Path("banks"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=9)
    # 200,000. Witnesses have been found as deep as trial 94,992
    # (bb288/BB2_GENERIC_MATCHED) and 58,440 (bb288/SC_BB_ZZ), so a budget of
    # 20,000 would screen to a markedly weaker standard.
    parser.add_argument("--trials", type=int, default=200000)
    # 100, not 20. Failures cost 10-36 s and only failures pay, so a deeper
    # list is nearly free -- and at 20 candidates with a ~33% pass rate you
    # cannot distinguish a target with no valid quilt from one that merely ran
    # out of budget. Whether such targets exist is an open scientific question,
    # and 20 cannot answer it.
    parser.add_argument("--keep", type=int, default=100)
    parser.add_argument("--limit", type=int, help="targets per cohort, for a pilot")
    parser.add_argument("--cohorts",
                        help="comma-separated substrings; keep cohorts matching any")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if args.definitions is None:
        args.definitions = default_definitions()

    import hashlib

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    population = json.loads(args.targets.read_text())
    args.out.mkdir(parents=True, exist_ok=True)

    # The duplicate guard below is per-process: each worker pool holds its own
    # `written` set, seeded once from what was on disk at startup.  Two
    # generators running against the same directory therefore cannot see each
    # other's rows, and both would certify the same target under different
    # seeds.  An exclusive flock on the output directory makes the guard's
    # single-writer assumption true.  The handle is deliberately left open for
    # the lifetime of the process; the lock is released when it exits, however
    # it exits.
    lock = (args.out / ".corpus.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another generator already holds {args.out}/.corpus.lock; "
              f"refusing to write concurrently", file=sys.stderr)
        return 1
    lock.write(f"pid {os.getpid()}\n")
    lock.flush()
    (args.out / "run.json").write_text(json.dumps({
        "schema": "quilt-representative-run-v6",
        "population": {k: v for k, v in population.items() if k != "cohorts"},
        "population_file_sha256": digest(args.targets),
        "banks": {p.stem: {"path": str(p), "sha256": digest(p)}
                  for p in sorted(args.banks.glob("*.json"))},
        "code_definitions": {p.stem: digest(p)
                             for p in sorted(args.definitions.glob("*.json"))},
        "screen": {"trials_per_engine": args.trials,
                   "engines": ["qdist:grouped", "qdist:three-block"]
},
        "search": {"base_seed": args.seed, "candidates_kept": args.keep,
                   "pairings_per_candidate": 5},
        "rule": "one representative, then four re-pairings of it, then the next",
    }, indent=1, sort_keys=True) + "\n")

    options = {"out": str(args.out), "trials": args.trials, "keep": args.keep,
               "seed": args.seed}
    # Interleave the cohorts.  Taking them in order would drain the first one
    # entirely before starting the next, so a partial run would hold thousands
    # of records from one cohort and none from the rest -- useless for any
    # comparison across cohorts, which is most of what we want to look at.
    per_cohort = []
    for key, rows in sorted(population["cohorts"].items()):
        if args.cohorts and not any(w in key for w in args.cohorts.split(",")):
            continue
        done = already_done(args.out, key)
        selected = rows[: args.limit] if args.limit else rows
        per_cohort.append([
            (key, raw, args.trials, args.keep, args.seed + index)
            for index, raw in enumerate(selected)
            if raw["target_id"] not in done
        ])
    jobs = [job for group in zip_longest(*per_cohort) for job in group if job is not None]
    print(f"{len(jobs)} targets outstanding over "
          f"{len(per_cohort)} cohorts, {args.workers} workers", flush=True)

    streams = {}
    logs = {}
    # Target ids already on disk, per cohort.  `already_done` filters the job
    # list at startup, but that is not enough on its own: a target a worker was
    # part-way through when a previous run was stopped is not in the stream, so
    # it is dispatched again -- and if the earlier run had in fact written it,
    # the cohort would end up with two records for one target.  A record
    # already present is therefore never overwritten or appended to; the first
    # certified record for a target is the one that stands.
    written = {}
    counts = collections.Counter()
    started = time.monotonic()
    try:
        with Pool(args.workers, initializer=setup,
                  initargs=(str(args.definitions), str(args.banks))) as pool:
            for result in pool.imap_unordered(one_target, jobs, chunksize=1):
                key = result["cohort"]
                stem = key.replace("/", "__")
                if key not in streams:
                    streams[key] = (args.out / f"{stem}.jsonl").open("a")
                    logs[key] = (args.out / f"{stem}.log.jsonl").open("a")
                if key not in written:
                    written[key] = already_done(args.out, key)
                target_id = str(result["target_id"])
                if result["outcome"] == "ok" and target_id in written[key]:
                    counts[f"{key}:duplicate"] += 1
                elif result["outcome"] == "ok":
                    streams[key].write(json.dumps(result["record"], separators=(",", ":"),
                                                  sort_keys=True) + "\n")
                    streams[key].flush()
                    written[key].add(target_id)
                    counts[key] += 1
                else:
                    counts[f"{key}:failed"] += 1
                logs[key].write(json.dumps({k: v for k, v in result.items()
                                            if k != "record"}) + "\n")
                logs[key].flush()
                # Only plain cohort keys count towards progress: the ":failed"
                # and ":duplicate" keys are annotations on the same work.
                total = sum(v for k, v in counts.items() if ":" not in k)
                if total and total % 100 == 0:
                    rate = total / (time.monotonic() - started)
                    print(f"  {total} records, {rate * 3600:.0f}/h", flush=True)
    finally:
        for handle in list(streams.values()) + list(logs.values()):
            handle.close()

    duplicates = 0
    for key in sorted(k for k in counts if ":" not in k):
        dup = counts.get(key + ":duplicate", 0)
        duplicates += dup
        print(f"  {key:26} {counts[key]:5} ok, "
              f"{counts.get(key + ':failed', 0):4} failed, {dup:4} duplicate")
    if duplicates:
        print(f"  {duplicates} duplicate results discarded (a target was "
              f"dispatched twice within this run)")
    print(f"total {time.monotonic() - started:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
