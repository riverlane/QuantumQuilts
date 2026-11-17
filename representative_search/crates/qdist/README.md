# qdist-rs

`qdist-rs` is a randomized Rust distance-search library with Python bindings
for quantum stabilizer and subsystem codes. The target problem is

```text
min PauliWeight(e) such that detector(e) = 0 and observable(e) != 0.
```

For a subsystem code with gauge space `G` and center `S = Z(G)`, use the
symplectic checks from `S` as detectors and a bare-logical basis of `G^perp/S`
as observables. The resulting search is exactly over `S^perp \ G`, including
non-CSS cancellation sectors.

The implementation follows the randomized information-set methodology used by
QDistRnd: compute a kernel basis once, repeatedly randomize coordinates and
perform Gaussian elimination, then test reduced rows for nontrivial logical
effect. The Rust backend adds deterministic multicore batches, full local
Clifford randomization, exact group-local linear randomization, and the
redundant `(x,z,x+z)` representation.

The returned witness is always checked exactly. Its weight is an **upper
bound** on distance. A matching lower bound requires a separate exhaustive or
theorem-based certificate; a failed search is never presented as proof.

```python
from qdist_rs import (
    search_css,
    search_gauge,
    search_grouped_binary,
    search_operation,
    search_subsystem,
)

# CSS quotient ker(H_X) / row(H_Z).
z_result = search_css(H_X, H_Z, trials=10_000, threads=8)

# General subsystem quotient S^perp / G. Matrices use X | Z symplectic form.
result = search_subsystem(center, bare_logicals, trials=10_000, threads=8)

# Or derive the center and bare-logical quotient from arbitrary gauge rows.
result = search_gauge(gauge_generators, trials=10_000, threads=8)

# Measurement-operation quotient S_aux^perp / (G_aux + benign).
result = search_operation(
    auxiliary_gauge,
    benign_measured_products,
    trials=10_000,
    threads=8,
)

# Try a different exact information-set geometry for a difficult mixed code.
result = search_gauge(
    gauge_generators,
    trials=10_000,
    threads=8,
    method="three-block",
)

# Count each declared pair as one fault location. Qubits omitted from the
# list remain independent singleton locations.
result = search_operation(
    auxiliary_gauge,
    benign_measured_products,
    trials=10_000,
    location_groups=[[left_q0, right_q0], [left_q1, right_q1]],
    method="location-linear",
)

# Search arbitrary binary fault coordinates without imposing an X | Z layout.
# Omitted columns become singleton locations; each explicit group costs one if
# any of its columns is occupied.
result = search_grouped_binary(
    detector_matrix,
    observable_matrix,
    trials=10_000,
    location_groups=[[0, 1, 2], [3, 4, 5, 6]],
)
```

`location_groups` is optional on every general symplectic search. Groups must
be disjoint and contain one or two physical-qubit indices. The cost of a Pauli
is the number of groups on which it is nonidentity, so an arbitrary correlated
Pauli on a declared pair costs one location. This is the native cost model for
closing-CNOT faults; it is not a post-processing rescore.

`multi_qubit_group_order_weight` optionally biases the randomized pivot order
of two-qubit groups relative to singleton groups. `1.0` is unbiased, values
above one move paired groups earlier on average, and values below one move them
later. `0.0` is the limiting stratified order: all shuffled singleton groups
precede all shuffled two-qubit groups. It is an empirical search heuristic only and defaults to `1.0`; the
returned witness and grouped cost are checked exactly for every setting.

`search_grouped_binary` applies the same randomized RREF search directly to
arbitrary binary columns. Detector and observable matrices may have any equal
width, including an odd width, and do not use a symplectic `X | Z` layout.
Explicit `location_groups` are disjoint lists of one to four column indices;
every omitted column is filled in as a singleton group. The objective is the
number of occupied groups subject to zero detector syndrome and at least one
observable flip. Candidate validation and the returned witness always use the
original binary columns.

For this binary API only, `multi_qubit_group_order_weight` classifies groups of
size three or four as the biased class, while groups of size one or two remain
the baseline class. This exact convention makes rank-2 boundary signatures the
baseline and rank-3 morph-CNOT signatures the biased class in F31. Ordering is
the same deterministic exponential race used by grouped symplectic search;
the weight changes search order only, never the exact grouped objective.

The calibration benchmark uses actual BB72, BB144, and BB288 single-code,
rank-one ZZ-stitch, and rank-one non-CSS-stitch quotients. It first runs the
ordinary finder, plants disjoint helper groups inside verified minimum
representatives, and measures whether the grouped finder reaches the planted
upper bound or improves it:

```bash
PYTHONPATH=src rust/operational-distance/.venv/bin/python \
  rust/operational-distance/benchmarks/calibrate_location_groups.py \
  --workers 8 --quick
```

## Development

```bash
uv venv
uv pip install -e ".[test]"
pytest -q
cargo test --locked
```

The tests use small explicit CSS, stabilizer, subsystem, mixed-center, and
operation-quotient fixtures. They validate every returned witness, grouped
location costs, input validation, and determinism across thread counts. The
precise search formulation is documented in [`docs/algorithm.md`](docs/algorithm.md).
