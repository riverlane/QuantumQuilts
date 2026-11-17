#!/usr/bin/env python3
"""Exact certification that a quilt carries no sub-threshold operator.

The screen is a randomised non-find: it says no witness turned up within the
budget. This says there is none, by exhaustion.

The problem, as `quilt_construction.distance` poses it, is to find an
operator in C-perp outside G of Pauli weight below the threshold d, where
C = G0 intersect G-perp is the inherited centre. So certification means ruling
out every operator of weight <= d-1. At bb72, d = 6, which is what makes this
cheap: it is a BOUNDED search, not a minimum-distance computation.

MEET IN THE MIDDLE. Write any weight-5 operator as a weight-2 piece and a
weight-3 piece on disjoint qubits. Syndromes are linear, so

    syn(v) = 0   <=>   syn(two-qubit part) = syn(three-qubit part)

and the pair can be found by hashing one side and streaming the other. Every
weight-<=5 operator decomposes this way -- pick any 2 of its qubits -- so the
search is exhaustive, not probabilistic. Hashing weight <= 2 and streaming
weight <= 3 covers every weight from 0 to 5: 5 = 2+3, 4 = 2+2 or 1+3, and so
on down.

The saving is the whole point. Enumerating weight <= 5 directly over n = 180
qubits is sum_w C(180,w) 3^w = 3.6e11 operators. Hashing 145,531 and
streaming 25,736,940 is four orders of magnitude less, and the lists are a
few MB.

A collision is a candidate, not a witness. It must still have disjoint
support (overlapping qubits give a different, lower weight, which the smaller
weight classes already cover) and must lie OUTSIDE G -- the gauge group is
full of genuine low-weight elements, the weight-2 stitches above all, and
those are not logical operators. Both are checked exactly.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def build(record: dict):
    """(n, constraint rows, G basis row-reduced, threshold) for one record."""
    from quilt_rep_search.distance_screen import record_as_v3
    from quilt_rep_search.operation_distance import operation_spec_from_quilt
    from quilt_construction.selected_record import quilt_payload_from_selected_record
    from quilt_construction.generation._gf2 import row_reduce

    quilt = quilt_payload_from_selected_record(record_as_v3(record))
    spec = operation_spec_from_quilt(
        quilt, trials=1, seed=0, threads=0, method="grouped",
        target_distance=int(record["operation_threshold"]))
    n = int(spec["n"])
    centre = [int(h, 16) for h in spec["inherited_center_hex"]]
    gauge = [int(h, 16) for h in spec["merged_gauge_hex"]]
    # v is in C-perp iff it commutes with every centre element. Commutation is
    # the symplectic product, which becomes an ordinary dot product once the
    # X and Z halves of the centre row are swapped -- so the swapped rows are
    # exactly the parity checks defining C-perp.
    mask = (1 << n) - 1
    checks = [((c >> n) & mask) | ((c & mask) << n) for c in centre]
    basis, _ = row_reduce(gauge, 2 * n)
    return n, checks, list(basis), int(record["operation_threshold"])


def syndrome_table(checks: list[int]) -> list[int]:
    """Column syndromes: bit i of out[j] is check i's coefficient on column j."""
    out = []
    for column in range(max(c.bit_length() for c in checks) if checks else 0):
        value = 0
        for index, check in enumerate(checks):
            if (check >> column) & 1:
                value |= 1 << index
        out.append(value)
    return out


def qubit_syndromes(n: int, columns: list[int]):
    """[(syn_X, syn_Z, syn_Y)] per qubit, padded for short column lists."""
    def col(j):
        return columns[j] if j < len(columns) else 0
    return [(col(q), col(q + n), col(q) ^ col(q + n)) for q in range(n)]


def in_gauge(row: int, basis: list[int], n: int) -> bool:
    """Exact membership of the gauge group, by reduction against its basis."""
    value = row
    for b in basis:
        if value == 0:
            return True
        high = b.bit_length() - 1
        if (value >> high) & 1:
            value ^= b
    return value == 0


def certify(record: dict, *, verbose=False) -> dict:
    n, checks, basis, threshold = build(record)
    columns = syndrome_table(checks)
    qs = qubit_syndromes(n, columns)
    limit = threshold - 1                  # the weight we must rule out
    half = limit // 2                      # hashed side
    rest = limit - half                    # streamed side

    began = time.monotonic()
    # ---- hash every operator of weight <= half, by syndrome ---------------
    table: dict[int, list] = collections.defaultdict(list)
    table[0].append(((), ()))
    small = 0
    for w in range(1, half + 1):
        for qubits in itertools.combinations(range(n), w):
            for paulis in itertools.product((0, 1, 2), repeat=w):
                syn = 0
                for q, p in zip(qubits, paulis):
                    syn ^= qs[q][p]
                table[syn].append((qubits, paulis))
                small += 1
    hashed = time.monotonic() - began

    # ---- stream every operator of weight <= rest, look for collisions -----
    def row_of(qubits, paulis):
        value = 0
        for q, p in zip(qubits, paulis):
            if p in (0, 2):
                value |= 1 << q
            if p in (1, 2):
                value |= 1 << (q + n)
        return value

    witnesses, streamed, collisions = [], 0, 0
    for w in range(0, rest + 1):
        for qubits in itertools.combinations(range(n), w):
            seen = set(qubits)
            for paulis in itertools.product((0, 1, 2), repeat=w):
                syn = 0
                for q, p in zip(qubits, paulis):
                    syn ^= qs[q][p]
                streamed += 1
                hits = table.get(syn)
                if not hits:
                    continue
                for other_q, other_p in hits:
                    if seen.intersection(other_q):
                        continue        # overlap: a lower weight class covers it
                    collisions += 1
                    row = row_of(qubits, paulis) ^ row_of(other_q, other_p)
                    if row == 0 or in_gauge(row, basis, n):
                        continue
                    weight = ((row & ((1 << n) - 1))
                              | (row >> n)).bit_count()
                    if weight <= limit:
                        witnesses.append((weight, hex(row)))
    seconds = time.monotonic() - began
    return {"target_id": record["target_id"], "cohort": record["cohort"],
            "scale": record["scale"], "n": n, "threshold": threshold,
            "ruled_out_to": limit, "hashed": small, "streamed": streamed,
            "collisions": collisions,
            "certified": not witnesses,
            "witnesses": sorted(witnesses)[:5],
            "hash_seconds": round(hashed, 2), "seconds": round(seconds, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="folder of <scale>__<class>.jsonl record files")
    ap.add_argument("--cohort", default="bb72__SC_BB_ZZ")
    ap.add_argument("--limit", type=int, default=1, help="records to certify")
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    path = args.corpus / f"{args.cohort}.jsonl"
    done = 0
    for index, line in enumerate(open(path)):
        if not line.strip() or index < args.skip:
            continue
        out = certify(json.loads(line))
        print(json.dumps(out))
        done += 1
        if done >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
