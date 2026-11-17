"""Rotated surface code blocks: layout, local checks, and logicals."""

from __future__ import annotations

from dataclasses import dataclass


Matrix = list[list[int]]


@dataclass(frozen=True)
class SurfaceCode:
    n: int


def _check_row(distance: int, qubits: list[tuple[int, int]]) -> list[int]:
    row = [0] * (distance * distance)
    for r, c in qubits:
        if r < 0 or r >= distance or c < 0 or c >= distance:
            raise ValueError("surface qubit coordinate is outside the code")
        row[r * distance + c] ^= 1
    return row


def rotated_surface_local_checks(distance: int) -> tuple[Matrix, Matrix]:
    distance = int(distance)
    if distance < 2:
        raise ValueError("rotated surface distance must be at least 2")
    if distance % 2 == 0:
        top_parity, bottom_parity, left_parity, right_parity = 0, 0, 1, 1
    else:
        top_parity, bottom_parity, left_parity, right_parity = 0, 1, 1, 0
    x_parity = 0
    h_x: Matrix = []
    h_z: Matrix = []

    def add_check(parity: int, qubits: list[tuple[int, int]]) -> None:
        (h_x if parity == x_parity else h_z).append(_check_row(distance, qubits))

    for r in range(distance - 1):
        for c in range(distance - 1):
            add_check(
                (r + c) & 1,
                [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)],
            )
    for c in range(distance - 1):
        if c % 2 == top_parity:
            add_check((-1 + c) & 1, [(0, c), (0, c + 1)])
        if c % 2 == bottom_parity:
            add_check(
                (distance - 1 + c) & 1,
                [(distance - 1, c), (distance - 1, c + 1)],
            )
    for r in range(distance - 1):
        if r % 2 == left_parity:
            add_check((r - 1) & 1, [(r, 0), (r + 1, 0)])
        if r % 2 == right_parity:
            add_check(
                (r + distance - 1) & 1,
                [(r, distance - 1), (r + 1, distance - 1)],
            )
    return h_x, h_z


def build_rotated_surface(distance: int) -> SurfaceCode:
    distance = int(distance)
    rotated_surface_local_checks(distance)
    return SurfaceCode(n=distance * distance)
