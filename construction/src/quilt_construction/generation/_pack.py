"""Pack a finished circuit into fewer ticks, without reordering anything on any qubit.

A circuit is a partial order and its ticks are one scheduling of it; operations on disjoint qubits commute exactly.
Here every operation moves as early as it can go, except resets, which move as late as they can go.  Each qubit's
own sequence of operations is unchanged, so the unitary, the error propagation and the fault mechanisms are
unchanged; what changes is the depth and where qubits wait.  An ancilla ends up measured in the first tick after
its last gate and reset in the last tick before its next one, so its waiting falls between its own measurement and
its own reset, where noise has no effect.

Every gate is scheduled on its own (one pair, one measured qubit, one MPP term) with its noise carried alongside,
and gates are regrouped into instructions only when the circuit is written out.

Measurements move relative to each other, so every rec[-k] shifts.  Offsets are resolved to absolute measurement
indices first and rebuilt afterwards, and annotations are written in their original order so detector and
observable indices are preserved.  Idle noise is dropped on the way in: it belongs to a schedule, and is added
again for the new one by the caller.

`pack` raises if any qubit's sequence of operations changed, or if the number of detectors or observables did.
"""
from __future__ import annotations

from collections import defaultdict

import stim

ANNOT = {"DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS"}
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Y_ERROR", "Z_ERROR",
         "PAULI_CHANNEL_1", "PAULI_CHANNEL_2"}
MEAS = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRZ", "MPP"}
RESET = {"R", "RX", "RY", "RZ"}
TWO_Q = {"CX", "CY", "CZ", "XCX", "XCZ", "ZCX", "SWAP", "ISWAP"}


class Atom:
    """One gate on one or two qubits, with the noise bound to it."""
    __slots__ = ("name", "args", "targets", "qubits", "pre", "post",
                 "nmeas", "first_meas")

    def __init__(self, name, args, targets):
        self.name, self.args, self.targets = name, args, targets
        self.qubits = {t.value for t in targets if t.is_qubit_target}
        self.pre, self.post = [], []
        self.nmeas = 0
        self.first_meas = None


def split_targets(ins):
    """The instruction's targets grouped one group per gate."""
    tg = ins.targets_copy()
    if ins.name == "MPP":
        out, k = [], 0
        while k < len(tg):
            grp = [tg[k]]; k += 1
            while k < len(tg) and tg[k].is_combiner:
                grp.append(tg[k]); k += 1
                grp.append(tg[k]); k += 1
            out.append(grp)
        return out
    # Two-qubit noise is written over the same flat target list as the gate it
    # follows, so it has to be grouped in pairs too or it will not line up.
    if ins.name in TWO_Q or ins.name in ("DEPOLARIZE2", "PAULI_CHANNEL_2"):
        return [[tg[i], tg[i + 1]] for i in range(0, len(tg), 2)]
    return [[t] for t in tg]


def parse(circuit):
    """(atoms, annotations, coords, n_measurements) with rec made absolute."""
    atoms, annots, coords = [], [], []
    meas_total = 0
    held = []
    last_group = []          # atoms produced by the most recent gate
    for ins in circuit:
        if isinstance(ins, stim.CircuitRepeatBlock):
            raise SystemExit("flatten the circuit first")
        if ins.name == "QUBIT_COORDS":
            coords.append(ins); continue
        if ins.name == "TICK":
            continue
        if ins.name == "DEPOLARIZE1":
            continue                       # idle; re-derived after scheduling
        if ins.name in ANNOT:
            # `barrier` is how many atoms preceded this annotation in the
            # original stream. A detector has to keep its place in that stream,
            # not merely follow its own records: emitted any earlier it can sit
            # before a measurement that was originally behind it, and stim then
            # reports it as non-deterministic because the collapse it now
            # straddles anticommutes with what the detector is sensitive to.
            annots.append((ins, [meas_total + t.value for t in ins.targets_copy()
                                 if t.is_measurement_record_target], len(atoms)))
            continue
        if ins.name in NOISE:
            groups = split_targets(ins)
            if last_group and len(groups) == len(last_group) and all(
                    {t.value for t in g if t.is_qubit_target} == a.qubits
                    for g, a in zip(groups, last_group)):
                for g, a in zip(groups, last_group):
                    a.post.append((ins.name, ins.gate_args_copy(), g))
            else:
                held.append((ins.name, ins.gate_args_copy(), groups))
            continue

        groups = split_targets(ins)
        made = []
        for g in groups:
            a = Atom(ins.name, ins.gate_args_copy(), g)
            if ins.name in MEAS:
                a.nmeas = 1
                a.first_meas = meas_total
                meas_total += 1
            made.append(a)
        # noise seen before this gate binds to it when the targets line up
        keep = []
        for name, args, groups_n in held:
            if len(groups_n) == len(made) and all(
                    {t.value for t in g if t.is_qubit_target} == a.qubits
                    for g, a in zip(groups_n, made)):
                for g, a in zip(groups_n, made):
                    a.pre.append((name, args, g))
            else:
                keep.append((name, args, groups_n))
        held = keep
        atoms.extend(made)
        last_group = made
    if held:
        raise SystemExit(f"{len(held)} noise instruction(s) could not be bound")
    return atoms, annots, coords, meas_total


def schedule(atoms):
    """ASAP under per-qubit precedence, with every collapse held in order.

    Per-qubit order is enough for unitary gates, because gates on disjoint
    qubits commute exactly. It is NOT enough for measurements and resets. A
    Z-basis measurement of a data qubit and an X-type check that covers it
    anticommute: once the check's CX has entangled them, collapsing either one
    changes the other, even though the two collapses name different qubits.
    Reordering them silently produces non-deterministic detectors.

    So every M and R keeps its position in the sequence of collapses, which
    also keeps the measurement record order identical -- and with it every
    rec[-k] and every detector index, so the v6 presentation maps still apply.
    Unitary gates are free to move, which is where the compression comes from.
    """
    free = {}
    tick = [0] * len(atoms)
    collapse = 0
    for i, a in enumerate(atoms):
        t = max((free.get(q, 0) for q in a.qubits), default=0)
        if a.name in MEAS or a.name in RESET:
            t = max(t, collapse)
            collapse = t
        tick[i] = t
        for q in a.qubits:
            free[q] = t + 1
    depth = (max(tick) + 1) if tick else 0

    # Resets go as late as their own qubit allows, without overtaking the next
    # collapse in the chain.
    nxt, nxt_collapse = {}, depth
    for i in range(len(atoms) - 1, -1, -1):
        a = atoms[i]
        if a.name in RESET:
            limit = min((nxt.get(q, depth) for q in a.qubits), default=depth)
            tick[i] = max(tick[i], min(limit, nxt_collapse) - 1)
        if a.name in MEAS or a.name in RESET:
            nxt_collapse = tick[i]
        for q in a.qubits:
            nxt[q] = tick[i]
    return tick, ((max(tick) + 1) if tick else 0)


def emit(atoms, tick, annots, coords, depth):
    out = stim.Circuit()
    for c in coords:
        out.append(c)
    by_tick = defaultdict(list)
    for i in sorted(range(len(atoms)), key=lambda i: (tick[i], i)):
        by_tick[tick[i]].append(i)

    # The tick by which every atom that originally preceded an annotation has
    # been emitted. Running maximum, so a later annotation never comes due
    # before an earlier one.
    ready, run = [], -1
    for _ins, _refs, barrier in annots:
        run = max(run, max((tick[i] for i in range(barrier)), default=-1))
        ready.append(run)

    newpos, emitted = {}, 0
    done = [False] * len(annots)
    now = [-1]

    def flush():
        nonlocal emitted
        for n, (ins, refs, _barrier) in enumerate(annots):
            if done[n]:
                continue
            if any(not done[m] for m in range(n)):
                return
            if ready[n] > now[0]:
                return
            if any(r not in newpos for r in refs):
                return
            if ins.name == "SHIFT_COORDS":
                out.append(ins)
            else:
                out.append(ins.name,
                           [stim.target_rec(newpos[r] - emitted) for r in refs],
                           ins.gate_args_copy())
            done[n] = True

    def emit_phase(bucket, phase):
        """Emit one phase (pre-noise / gate / post-noise), regrouping atoms."""
        nonlocal emitted
        group, key = [], None
        for i in bucket:
            a = atoms[i]
            items = ([(a.name, a.args, a.targets)] if phase == "gate"
                     else (a.pre if phase == "pre" else a.post))
            for name, args, targets in items:
                k = (name, tuple(args))
                if k != key and group:
                    out.append(key[0], group, list(key[1]))
                    group = []
                key = k
                group.extend(targets)
                if phase == "gate" and a.nmeas:
                    newpos[a.first_meas] = emitted
                    emitted += 1
        if group:
            out.append(key[0], group, list(key[1]))

    for t in range(depth):
        bucket = by_tick.get(t, [])
        emit_phase(bucket, "pre")
        emit_phase(bucket, "gate")
        emit_phase(bucket, "post")
        now[0] = t
        flush()
        if t + 1 < depth:
            out.append("TICK")
    now[0] = depth
    flush()
    if not all(done):
        raise SystemExit(f"{done.count(False)} annotation(s) unresolved")
    return out


def check_order(before, after):
    """Every qubit must see exactly the same gate sequence, in the same order."""
    def seq(c):
        per = defaultdict(list)
        for ins in c:
            if ins.name in ANNOT or ins.name in NOISE:
                continue
            if ins.name in ("TICK", "QUBIT_COORDS"):
                continue
            for g in split_targets(ins):
                qs = tuple(sorted(t.value for t in g if t.is_qubit_target))
                for q in qs:
                    per[q].append((ins.name, qs))
        return per
    a, b = seq(before), seq(after)
    return [q for q in a if a[q] != b.get(q)]


def pack(circuit):
    """The packed circuit.  Checked: per-qubit operation order, detector count and observable count are unchanged."""
    atoms, annots, coords, _nmeas = parse(circuit)
    tick, depth = schedule(atoms)
    out = emit(atoms, tick, annots, coords, depth)
    changed = check_order(circuit, out)
    if changed:
        raise AssertionError(f"packing changed the operation order on {len(changed)} qubit(s): {changed[:8]}")
    if out.num_detectors != circuit.num_detectors or out.num_observables != circuit.num_observables:
        raise AssertionError("packing changed the number of detectors or observables")
    return out
