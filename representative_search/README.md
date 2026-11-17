# quilt-rep-search

Chooses a representative and a pairing for each target of a code quilt, and tests the result for operators of
weight below the code distance.

A target is one logical class per block. The search picks, for every block, a low-weight representative of the
class whose support measures only the target, pairs the blocks' supports qubit by qubit, and screens the
resulting quilt at code capacity.

## Install

Install `quilt-construction` first. Then, from this folder:

    (cd crates/qdist && maturin develop --release)
    pip install -e .

`crates/qdist` is the randomised distance engine (Rust, with Python bindings as `qdist_rs`). Python needs `numpy`
and `ortools`.

## Scripts

| script | what it does |
|---|---|
| `scripts/fill_representative_bank.py` | finds a few low-weight clean representatives per logical class of a code (`banks/`) |
| `scripts/freeze_targets.py` | draws the population of targets (`targets/population_v8.json`) |
| `scripts/generate_corpus.py` | for each target: search, pair, screen, write a record |
| `scripts/redraw.py` | draws a fresh representative for one target, given a record of that target |
| `scripts/certify_exact.py` | exhaustive check that a quilt has no operator below its threshold (meet in the middle) |

    python scripts/generate_corpus.py --targets targets/population_v8.json --out OUT --cohorts SC_BB_ZZ --limit 10
    python scripts/redraw.py RECORD.json --out NEW.json --trials 200000
    python scripts/certify_exact.py --corpus FOLDER --cohort bb72__SC_BB_ZZ --limit 1

## The two distance checks

**Randomised.** `distance_screen.py` builds the quilt's distance problem once (`operation_distance.py`) and runs
two variants of the engine against it for a set number of trials each, looking for an operator of weight at most
threshold - 1. A record stores the number of trials completed and the lowest weight seen. Finding nothing is
evidence, not proof.

**Exhaustive.** `certify_exact.py` enumerates every operator of weight below the threshold by meeting in the
middle on the syndrome. It is practical at distance 6 and is how the randomised check was calibrated.

## Data

    banks/                      representative banks for bb72, bb144, bb288
    targets/population_v8.json  the frozen population of targets

## Tests

    python -m pytest tests -q
    (cd crates/qdist && python -m pytest tests -q)
