"""Codes, and the symmetries that let a handful of representatives cover them all.

A bivariate bicycle code lays its n = 2*ell*m data qubits out as two copies of
an ell-by-m torus.  Translating the torus is a symmetry of the code: it maps
stabilisers to stabilisers and logical operators to logical operators.  It does
*not* fix a logical operator's class, though -- it permutes the classes.  That
is the whole trick behind the representative bank: one good low-weight operator,
pushed around the torus, becomes ell*m good low-weight operators spread across
ell*m different logical classes.

The other symmetry is the duality phi, which swaps the two torus halves and
inverts the coordinates.  It exchanges the X and Z sectors, so a Z-sector
representative also yields an X-sector one for free.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence


def default_definitions() -> Path:
    """Where the code definitions live: packaged with quilt_construction."""

    from quilt_construction.codes import DEFAULT_DATA

    return DEFAULT_DATA / "code_definitions"


def bit_count(value: int) -> int:
    return int(value).bit_count()


def parity(value: int) -> int:
    return bit_count(value) & 1


@dataclass(frozen=True)
class Code:
    """One code definition, read straight from `code_definitions/`."""

    code_id: str
    n: int
    k: int
    distance: int
    x_checks: tuple[int, ...]
    z_checks: tuple[int, ...]
    x_logicals: tuple[int, ...]
    z_logicals: tuple[int, ...]
    source_x_checks: tuple[int, ...]
    source_z_checks: tuple[int, ...]
    ell: int
    m: int

    @classmethod
    def load(cls, path: Path) -> "Code":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        parameters = raw.get("parameters", {})
        return cls(
            code_id=str(raw["code_id"]),
            n=int(raw["num_qubits"]),
            k=int(raw["num_logicals"]),
            distance=int(raw["distance"]),
            x_checks=tuple(int(row, 16) for row in raw["checks"]["x"]),
            z_checks=tuple(int(row, 16) for row in raw["checks"]["z"]),
            x_logicals=tuple(int(row, 16) for row in raw["logical_basis"]["x"]),
            z_logicals=tuple(int(row, 16) for row in raw["logical_basis"]["z"]),
            # the physical presentation, not the row-reduced basis: the
            # connectivity condition is stated about individual source checks
            source_x_checks=tuple(int(row, 16) for row in raw["source_checks"]["x"]),
            source_z_checks=tuple(int(row, 16) for row in raw["source_checks"]["z"]),
            ell=int(parameters.get("ell", 0)),
            m=int(parameters.get("m", 0)),
        )

    @property
    def family_is_rotated_surface(self) -> bool:
        return self.code_id.startswith("rotated_surface")

    @property
    def half(self) -> int:
        """Qubits per torus half.  Zero for codes that are not bivariate bicycle."""

        return self.ell * self.m

    def detecting_checks(self, sector: str) -> tuple[int, ...]:
        """The checks a sector-`sector` operator can anticommute with."""

        return self.x_checks if sector == "Z" else self.z_checks

    def label(self, mask: int, sector: str) -> int:
        """Logical class of a pure-`sector` operator, as a k-bit label."""

        basis = self.x_logicals if sector == "Z" else self.z_logicals
        return sum(parity(mask & row) << index for index, row in enumerate(basis))

    def representative(self, label: int, sector: str) -> int:
        """Any member of the class, namely the product of the selected basis rows."""

        basis = self.z_logicals if sector == "Z" else self.x_logicals
        mask = 0
        for index, row in enumerate(basis):
            if (label >> index) & 1:
                mask ^= row
        return mask

    def is_closed(self, mask: int, sector: str) -> bool:
        return all(parity(mask & row) == 0 for row in self.detecting_checks(sector))


def _shift_site(site: int, *, ell: int, m: int, dx: int, dy: int) -> int:
    x, y = divmod(site, m)
    return ((x + dx) % ell) * m + ((y + dy) % m)


def _invert_site(site: int, *, ell: int, m: int) -> int:
    x, y = divmod(site, m)
    return ((-x) % ell) * m + ((-y) % m)


def translate(mask: int, code: Code, dx: int, dy: int) -> int:
    """Translate an operator by (dx, dy), each torus half moving together."""

    half = code.half
    out = 0
    rest = mask
    while rest:
        q = (rest & -rest).bit_length() - 1
        rest &= rest - 1
        if q < half:
            out |= 1 << _shift_site(q, ell=code.ell, m=code.m, dx=dx, dy=dy)
        else:
            out |= 1 << (half + _shift_site(q - half, ell=code.ell, m=code.m, dx=dx, dy=dy))
    return out


def dualise(mask: int, code: Code) -> int:
    """Apply phi: swap the torus halves and invert coordinates, Z <-> X."""

    half = code.half
    out = 0
    rest = mask
    while rest:
        q = (rest & -rest).bit_length() - 1
        rest &= rest - 1
        if q < half:
            out |= 1 << (half + _invert_site(q, ell=code.ell, m=code.m))
        else:
            out |= 1 << _invert_site(q - half, ell=code.ell, m=code.m)
    return out


def all_shifts(code: Code) -> tuple[tuple[int, int], ...]:
    return tuple((dx, dy) for dx in range(code.ell) for dy in range(code.m))
