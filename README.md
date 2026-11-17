# QuantumQuilts

Code for constructing code quilts and for choosing the logical representatives they are built on.

| folder | package | what it does |
|---|---|---|
| `construction/` | `quilt_construction` | builds a quilt, and its syndrome-extraction circuit, from a representative record |
| `representative_search/` | `quilt_rep_search` | searches for representatives and pairings, and tests the resulting quilts for operators below the code distance; contains the randomised distance engine `crates/qdist` |

`representative_search` depends on `construction`; `construction` depends on nothing else here.

## Install

Python 3.11 or later, and a Rust toolchain with `maturin` for the distance engine.

    pip install -e construction
    (cd representative_search/crates/qdist && maturin develop --release)
    pip install -e representative_search

Each folder has its own README with the commands it provides.

## Data

The data sets that accompany the paper (the representatives, the circuits and their distance evidence, the
decoding results) are published separately. They read their code definitions from `construction`.

## Licence

See `LICENSE.md`.
