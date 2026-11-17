# A worked example

This walks through every entry point in order: draw a target, find a representative for it, check the quilt's
distance at code capacity, build its circuit, and check the circuit's distance. It uses the 72-qubit bivariate
bicycle code, where every check finishes in a couple of minutes and the exhaustive checks are feasible.

All commands are run from the top of the repository, after the install in `README.md`. Outputs go to a scratch
folder:

    mkdir -p scratch

Parts 1 and 2 use only this repository. Part 3 needs the circuit distance data release, unpacked at
`data/circuit_distance`, and its two crates built:

    (cd data/circuit_distance/finder/pivotbeam && maturin develop --release)
    (cd data/circuit_distance/exact/exact5 && maturin develop --release)

## 1. A target and a representative

A *target* is one logical class per block: the joint logical measurement the quilt is to perform. There are four
kinds, named by their blocks and operator: `SC_BB_ZZ`, `BB2_ZZ`, `BB2_GENERIC_MATCHED` and `SC_BB2_ZZZ`.

**Draw targets.** Three random targets of each kind on the 72-qubit code, from a seed:

    python representative_search/scripts/freeze_targets.py --out scratch/targets.json --per-cohort 3 --scales bb72 --seed 1

The population used in the paper is `representative_search/targets/population_v8.json`, drawn the same way.

**Search.** For each target: choose a low-weight representative on every block, pair the blocks' supports, and
test the resulting quilt for operators below the code distance with the randomised distance engine.

    python representative_search/scripts/generate_corpus.py --targets scratch/targets.json --out scratch/corpus \
        --cohorts SC_BB_ZZ --trials 2000 --workers 2

This writes `scratch/corpus/bb72__SC_BB_ZZ.jsonl`, one record per line. A record holds the target, the
representative, the pairing, and the result of the test: how many trials ran and the lowest weight seen. The
paper's data used 200,000 trials; 2,000 keeps the example quick.

The search draws block representatives from a *bank* of low-weight seeds per logical class, expanded by the
code's symmetries. The banks for the three codes are in `representative_search/banks/`. To extend one, or build
one for a code from its definition:

    cp representative_search/banks/bb72.json scratch/bb72_bank.json
    python representative_search/scripts/fill_representative_bank.py --code bb72 \
        --definition construction/src/quilt_construction/data/code_definitions/bb72.json --bank scratch/bb72_bank.json

**Draw again.** If a representative turns out to be unsuitable later (for instance, its circuit has a low-weight
fault), a different one can be drawn for the same target. The script takes one record as a JSON file:

    head -1 scratch/corpus/bb72__SC_BB_ZZ.jsonl > scratch/record.json
    python representative_search/scripts/redraw.py scratch/record.json --out scratch/redrawn.json --trials 2000

It walks a fixed sequence of alternatives (re-pairings first, then other representatives), tests each, and
writes the first that passes.

## 2. The quilt and its distance at code capacity

**Exhaustive check.** The randomised test can only fail to find an operator. At distance 6 the question can be
settled: enumerate every operator of weight 5 or less.

    python representative_search/scripts/certify_exact.py --corpus scratch/corpus --cohort bb72__SC_BB_ZZ --limit 3

Each line of output reports `certified: true` with `ruled_out_to: 5`, or lists the operators found.

**Build the quilt and its circuit.** Pick a record by its target id:

    quilt-import-representative scratch/corpus/bb72__SC_BB_ZZ.jsonl --target-id SC_BB_ZZ-0001 \
        --out scratch/quilt.json --physical-p 0.003 --rounds 6
    quilt-build scratch/quilt.json --out scratch/build

`scratch/quilt.json` is the explicit quilt: blocks, stitching operators, what is measured and what is preserved.
`scratch/build/` holds `circuit.stim` (the noisy syndrome-extraction circuit with detectors and observables),
`dem.stim`, `routing.json` and `summary.json`. The build checks the quilt algebraically and checks every
detector and observable against the noiseless circuit before writing anything.

Two companions for comparison experiments:

    quilt-build-control scratch/quilt.json --out scratch/control          # the same blocks, idle, with no stitch
    quilt-build-memory bb144 --out scratch/memory --rounds 12 --physical-p 0.003   # one block held in memory

## 3. The circuit and its distance

These steps use the tools in `data/circuit_distance`.

**Build and fold.** From a record, build the circuit and fold its fault model: every fault is translated in time
onto one round, which shrinks the search by a factor of the order of the number of rounds.

    python data/circuit_distance/population/build.py --record bb72 SC_BB_ZZ my-case scratch/record.json \
        --out scratch --name quilts

This writes `scratch/quilts/folds/my-case/` (the circuit, the source and folded models, and `verification.json`)
and `scratch/quilts/manifest.json`. The build stops unless the fold is verified exact.

**Search for a low-weight fault.** Six trials of the randomised level search, each looking for an undetectable
logical fault of weight 5 or less:

    python data/circuit_distance/trials/screen_population.py scratch/quilts --runs 6 --K 20000

Each circuit is reported as `survived` or as rejected with the weight of the fault found; details go to
`scratch/quilts/trials.jsonl`.

**Exhaustive check.** At target 6 the circuit can be settled too:

    python data/circuit_distance/exact/certify_population.py scratch/quilts --all

`exact min None` means no undetectable logical fault of weight 5 or less exists in the folded model.

**Complete the validation of the fold.** Two further checks tie the folded model to the physical circuit:

    python data/circuit_distance/fold/audit_noise.py --root scratch/quilts/folds
    python data/circuit_distance/fold/build_temporal.py --evidence scratch/quilts/folds

The first confirms that every fault of the full noise model is represented in the folded model
(`first_unmatched` empty); the second certifies the one output the fold does not retain. Their reports are
written beside the fold.

## Where the paper's results come from

| result | made by |
|---|---|
| the representatives | part 1, over `targets/population_v8.json` with 200,000 trials |
| the code-capacity calibration | part 1 and the exhaustive check of part 2 on fresh draws; fitted in the representative data release |
| the twelve circuits and their distances | part 3, with a list of 80,000 and 300 trials at 144 and 288 qubits, exhaustively at 72 |
