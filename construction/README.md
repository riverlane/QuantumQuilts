# quilt-construction

Builds a code quilt and its syndrome-extraction circuit from a representative record.

A record names a target (one logical class per block), the representative chosen for it and the pairing between
blocks. This package turns that into the explicit quilt (blocks, stitching operators, measured and preserved
logicals) and then into a noisy `stim` circuit with detectors and observables.

## Install

    pip install -e .

Needs Python 3.11 or later, `numpy` and `stim`. Nothing else is needed.

## Use

    quilt-import-representative RECORDS.jsonl --target-id SC_BB_ZZ-0249 --out quilt.json --physical-p 0.003 --rounds 6
    quilt-build quilt.json --out build/

`quilt-import-representative` reads one record (a JSON object, or one line of a JSON Lines file picked with
`--target-id`) and writes the explicit quilt. `quilt-build` writes

| file | contents |
|---|---|
| `circuit.stim` | the circuit, with noise, detectors and observables |
| `dem.stim` | its detector error model |
| `routing.json` | which measurement feeds which detector and observable |
| `summary.json` | sizes, hashes, the schedule used, and the algebraic validation of the quilt |

Before anything is written the quilt is checked algebraically (the target is measured, nothing else is, the
unmeasured logicals survive) and every detector and observable of the circuit is checked to be a deterministic
parity of the noiseless circuit.

`quilt-build-control` builds the matched control for a quilt and `quilt-build-memory` an idle memory circuit.

All three build commands take two options, both off by default:

| option | effect |
|---|---|
| `--idle-noise` | every qubit that no gate touches in a tick is depolarised at the physical error rate |
| `--compress` | the circuit is packed into fewer ticks: every operation moves as early as it can, resets as late as they can, and no qubit's own sequence of operations changes. Detector and observable numbering is unchanged |

With both, idle noise is added to the packed circuit. Packing is checked on every build: per-qubit operation
order, the number of detectors and observables, and the absence of detector or observable flips without noise.
`summary.json` records both settings.

## Codes

`src/quilt_construction/data/code_definitions/` holds the six codes used: the bivariate bicycle codes `bb72`,
`bb144`, `bb288` and the rotated surface codes of distance 6, 12 and 18. Each file gives the checks, the local
check presentation and a logical basis. Every later stage reads its code definitions from here.

The syndrome-extraction gate order of the bivariate bicycle blocks is the one named `legacy_identity_first_v1` in
`generation/`, with the CNOT layers taken in reverse order on alternate rounds. `summary.json` records the
schedule of every block.

## Layout

    src/quilt_construction/
        cli.py                the commands
        selected_record.py    record -> explicit quilt
        generation/           quilt -> circuit (algebra, schedules, emission, parity checks, tick packing)
        codes.py              code definitions
        glued_basis.py        the glued inherited-centre basis of a quilt
        hashes.py             content hashes
