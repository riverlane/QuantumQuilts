# QuantumQuilts
<img width="805" height="679" alt="Screenshot 2026-11-17 at 23 33 38" src="https://github.com/user-attachments/assets/ef17ae44-b056-468f-ac5d-3b143887b748" />

This repository hosts most of the code required to construct codes and circuits for quantum quilts, a method of performing logical Pauli-product measurements in general codes detailed in [this](todo) paper.

The machinery for, and results of, the empirical distance studies and decoding experiments can be found [here](todo). The some of the data in that repository follows data formats specified in `/construction`.

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

## Licence

See `LICENSE.md`.
