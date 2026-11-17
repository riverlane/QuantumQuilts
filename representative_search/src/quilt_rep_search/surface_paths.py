"""Surface-code logical representatives, constructed rather than searched.

A rotated surface code has one logical qubit, so there is no orbit for a bank to
spread over and nothing for the translation trick to exploit.  Annealing the
coset is a poor substitute: it converges on whatever width it likes, giving no
diversity, and a closed operator that passes the rank-one test need not be a
*simple path* -- it can be a path carrying extra structure, which is exactly the
sort of interface that tends to fail a distance screen.

So we construct instead.  Take the X checks as vertices and the data qubits as
arcs between the (at most two) checks they touch; boundary qubits in the top and
bottom rows become arcs to virtual terminals.  Closing the loop with a fixed
bottom-to-top arc makes a simple top-to-bottom path the unique cycle, which
`AddCircuit` then enforces.  Self-loops let unused checks drop out, so a
disconnected stabiliser loop cannot be selected.  Fixing the number of chosen
arcs pins the weight exactly.

The result is a simple path of exactly the requested weight, so asking for a
range of weights gives interface diversity by construction -- which is what a
screen rejection needs in order to be recoverable.

The distinctive property of these paths is structural, not statistical:
`AddCircuit` forbids the decorated paths an annealer will happily return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .codes_and_symmetries import Code, bit_count
from .interface_cleanliness import inspect


@dataclass(frozen=True)
class SurfacePath:
    mask: int
    weight: int


def _incidence(code: Code) -> list[list[int]]:
    """Which X checks touch each data qubit, in the local presentation."""

    incident: list[list[int]] = [[] for _ in range(code.n)]
    for index, check in enumerate(code.source_x_checks):
        rest = check
        while rest:
            qubit = (rest & -rest).bit_length() - 1
            rest &= rest - 1
            incident[qubit].append(index)
    return incident


def simple_path(code: Code, target_weight: int, *, timeout: float = 30.0,
                seed: int = 0) -> SurfacePath | None:
    """A simple top-to-bottom Z path of exactly `target_weight` qubits."""

    from ortools.sat.python import cp_model

    distance = code.distance
    incident = _incidence(code)
    check_count = len(code.source_x_checks)
    top, bottom = check_count, check_count + 1

    model = cp_model.CpModel()
    arcs: list[tuple[int, int, object]] = [
        (vertex, vertex, model.new_bool_var(f"unused_{vertex}"))
        for vertex in range(check_count)
    ]
    chosen = []
    for qubit in range(code.n):
        touching = incident[qubit]
        if len(touching) == 2:
            left, right = touching
            forward = model.new_bool_var(f"q{qubit}_fwd")
            reverse = model.new_bool_var(f"q{qubit}_rev")
            selected = model.new_bool_var(f"q{qubit}")
            model.add(selected == forward + reverse)
            arcs += [(left, right, forward), (right, left, reverse)]
        elif len(touching) == 1 and qubit < distance:
            selected = model.new_bool_var(f"q{qubit}_top")
            arcs.append((top, touching[0], selected))
        elif len(touching) == 1 and qubit >= code.n - distance:
            selected = model.new_bool_var(f"q{qubit}_bottom")
            arcs.append((touching[0], bottom, selected))
        else:
            raise ValueError(f"qubit {qubit} has unexpected X incidence {touching}")
        chosen.append(selected)

    arcs.append((bottom, top, model.new_constant(1)))
    model.add_circuit(arcs)
    model.add(sum(chosen) == int(target_weight))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(timeout)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    mask = sum(1 << qubit for qubit, var in enumerate(chosen) if solver.value(var))
    if not code.is_closed(mask, "Z"):
        raise AssertionError("constructed surface path is not Z-closed")
    if code.label(mask, "Z") == 0:
        raise AssertionError("constructed surface path is Z-trivial")
    if not inspect(mask, code=code, sector="Z").clean:
        raise AssertionError("constructed surface path lacks a rank-one kernel")
    return SurfacePath(mask=mask, weight=bit_count(mask))


def paths_over_widths(code: Code, widths: Iterator[int], *, per_width: int = 1,
                      timeout: float = 30.0) -> list[SurfacePath]:
    """One or more simple paths at each requested weight, lightest first."""

    found: dict[int, SurfacePath] = {}
    for width in widths:
        for attempt in range(per_width):
            path = simple_path(code, width, timeout=timeout, seed=width * 1009 + attempt)
            if path is not None and path.mask not in found:
                found[path.mask] = path
    return sorted(found.values(), key=lambda p: (p.weight, p.mask))
