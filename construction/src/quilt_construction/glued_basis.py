"""The glued inherited-centre basis: candidate pool and selection.

This is the selection the sparsity and congestion figures were computed with.
Two things are added to it, neither of which the selection can see:

  * every candidate now records *which* source checks it is built from, as a
    bitmask over the per-block check list.  A detector is the XOR of its
    constituent checks' measurement records, so a basis without this cannot be
    turned into a circuit.  Recovering it afterwards by solving in the row span
    is possible but may return a different set of checks for the same operator,
    which would make the built circuit and the measured basis different objects.
  * `select` returns the pool entries unchanged, so callers can read either.

Pool entry layout, positionally stable because `select` indexes it:

    (num_source_checks, pauli_weight, packed_row, qubit_support, kind, coefficients)
     0                  1             2           3               4     5
"""

from __future__ import annotations

import collections

from .codes import Code, bfs, insert, rank, to_root

CODES: dict[str, Code] = {}


def init(data_root=None) -> None:
    """Load every code definition present."""

    from pathlib import Path
    from .codes import DEFAULT_DATA

    root = Path(data_root) if data_root is not None else DEFAULT_DATA
    for path in sorted((root / "code_definitions").glob("*.json")):
        CODES[path.stem] = Code(path.stem, data_root=root)


def generators(rec):
    """Return (required_rank, [(checks, weight, phys_key, qubit_support)])."""
    w = len(rec['blocks'][0]['pieces'])
    if any(len(b['pieces']) != w for b in rec['blocks']):
        return None
    nblk = len(rec['blocks'])
    offs, off = [], 0
    for b in rec['blocks']:
        offs.append(off); off += CODES[b['code']].n
    items = []
    for bi, b in enumerate(rec['blocks']):
        code = CODES[b['code']]
        for ci, syn in enumerate(code.syndromes(b['pieces'])):
            items.append((bi, ci, syn, code.rows[ci]))
    rank_pi = sum(CODES[b['code']].pauli_rank for b in rec['blocks'])
    required = rank_pi - rank(s for _b, _c, s, _r in items)

    edges, hyper, frees = [], [], []
    for idx, (_bi, _ci, syn, _row) in enumerate(items):
        pc = syn.bit_count()
        if pc == 0:
            frees.append(idx)
        elif pc == 2:
            edges.append(((syn & -syn).bit_length() - 1, syn.bit_length() - 1, idx))
        else:
            t, rest = [], syn
            while rest:
                t.append((rest & -rest).bit_length() - 1); rest &= rest - 1
            hyper.append((idx, t))

    adj = [[] for _ in range(w)]
    for s, (u, v, _i) in enumerate(edges):
        adj[u].append((v, s)); adj[v].append((u, s))
    trees = [bfs(w, adj, x) for x in range(w)]
    connected = all(d >= 0 for d in trees[0][0])

    def pack(acc):
        key, support, shift = 0, 0, 0
        for bi, row in enumerate(acc):
            n = CODES[rec['blocks'][bi]['code']].n
            key |= row << shift
            support |= (((row & ((1 << n) - 1)) | (row >> n)) << offs[bi])
            shift += 2 * n
        return key, support

    def coeff_of(slots, extra=None):
        '''Which source checks this candidate is built from, as a bitmask over
        `items`.  Purely additive bookkeeping: the pool entry gains a sixth
        element and nothing reads it here.  The circuit builder needs the
        decomposition -- a detector is the XOR of its constituent checks'
        measurement records -- and recovering it afterwards by solving in the
        row span can return a different set of checks for the same operator.
        Recording it at the point of construction keeps the built circuit and
        the measured basis the same object.'''
        mask = 0
        for s in slots:
            mask ^= 1 << edges[s][2]
        if extra is not None:
            mask ^= 1 << extra
        return mask

    def rows_of(slots, extra=None):
        acc = [0] * nblk
        for s in slots:
            bi, _ci, _s, row = items[edges[s][2]]
            acc[bi] ^= row
        if extra is not None:
            bi, _ci, _s, row = items[extra]
            acc[bi] ^= row
        return acc

    pool = []
    for idx in frees:
        bi, _ci, _s, row = items[idx]
        acc = [0] * nblk; acc[bi] = row
        key, sup = pack(acc)
        pool.append((1, sup.bit_count(), key, sup, 'free', 1 << idx))

    if connected and len(edges) - w + 1 > 0:
        cands = set()
        for _d, par in trees:
            for s, (x, y, _i) in enumerate(edges):
                m = to_root(par, x) ^ to_root(par, y) ^ (1 << s)
                if m: cands.add(m)
        seen = {}
        for s, (x, y, _i) in enumerate(edges):
            k = (min(x, y), max(x, y))
            if k in seen: cands.add((1 << s) | (1 << seen[k]))
            else: seen[k] = s
        for m in cands:
            slots, rest = [], m
            while rest:
                slots.append((rest & -rest).bit_length() - 1); rest &= rest - 1
            key, sup = pack(rows_of(slots))
            if key: pool.append((len(slots), sup.bit_count(), key, sup, 'cycle',
                                 coeff_of(slots)))

    if connected:
        dist = [t[0] for t in trees]; par = [t[1] for t in trees]
        def route(src, dst):
            slots, node, p = [], dst, par[src]
            while p[node] is not None:
                u, s = p[node]; slots.append(s); node = u
            return slots
        def best(ts):
            if not ts: return []
            head, restl = ts[0], ts[1:]
            bestv, bl = None, None
            for i, other in enumerate(restl):
                if dist[head][other] < 0: continue
                tail = best(restl[:i] + restl[i+1:])
                if tail is None: continue
                cand = route(head, other) + tail
                if bl is None or len(cand) < bl: bestv, bl = cand, len(cand)
            return bestv
        for idx, ts in hyper:
            slots = best(ts)
            if slots is None: continue
            key, sup = pack(rows_of(slots, extra=idx))
            if key: pool.append((len(slots) + 1, sup.bit_count(), key, sup, 'hyper',
                                 coeff_of(slots, extra=idx)))
    return required, pool, off

def select(pool, required, congestion_aware):
    basis, chosen = {}, []
    if not congestion_aware:
        order = sorted(pool, key=lambda t: (t[1], t[0], t[2]))
        for g in order:
            if insert(basis, g[2]):
                chosen.append(g)
                if len(chosen) == required: break
        return chosen
    load = collections.Counter()
    buckets = collections.defaultdict(list)
    for g in pool: buckets[(g[1], g[0])].append(g)
    for key in sorted(buckets):
        bucket = buckets[key]
        bucket.sort(key=lambda g: (sum(load[q] for q in bits(g[3])), g[2]))
        for g in bucket:
            if insert(basis, g[2]):
                chosen.append(g)
                for q in bits(g[3]): load[q] += 1
                if len(chosen) == required: break
        if len(chosen) == required: break
    return chosen

def bits(mask):
    out = []
    while mask:
        out.append((mask & -mask).bit_length() - 1); mask &= mask - 1
    return out

def split_stats(chosen):
    out = {}
    nonfree = [g for g in chosen if g[4] != 'free']
    for name, sel in (('free', [g for g in chosen if g[4] == 'free']),
                      ('nonfree', nonfree)):
        if not sel:
            out[name + '_n'] = 0
            continue
        ws = sorted(g[1] for g in sel)
        out[name + '_n'] = len(sel)
        out[name + '_mean_w'] = round(sum(ws) / len(ws), 2)
        out[name + '_med_w'] = ws[len(ws) // 2]
        out[name + '_p90_w'] = ws[int(0.9 * (len(ws) - 1))]
        out[name + '_max_w'] = ws[-1]
    return out


def stats(chosen, nqubits):
    deg = collections.Counter()
    for g in chosen:
        for q in bits(g[3]): deg[q] += 1
    degs = [deg[q] for q in range(nqubits) if deg[q]]
    weights = [g[1] for g in chosen]
    checks = [g[0] for g in chosen]
    return {'max_checks': max(checks), 'max_weight': max(weights),
            'mean_weight': round(sum(weights)/len(weights), 2),
            'max_qubit_degree': max(degs) if degs else 0,
            'mean_qubit_degree': round(sum(degs)/len(degs), 2) if degs else 0,
            'touched_qubits': len(degs)}

