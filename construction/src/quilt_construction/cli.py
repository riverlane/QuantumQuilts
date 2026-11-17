from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generation import (
    BB144_SEC_SCHEDULE_BRAVYI_2024,
    BB_SEC_SCHEDULE_LEGACY,
)
from .generation.core import (
    build_control_files,
    build_memory_files,
    build_memory_spec_files,
    build_quilt_files,
)
from .selected_record import write_quilt_payload_from_selected_record


def import_representative_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a public selected representative into quilt.json."
    )
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-id", help="record to select when the input is JSON Lines")
    parser.add_argument("--physical-p", type=float, default=0.003)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args(argv)
    payload = write_quilt_payload_from_selected_record(
        args.record,
        args.out,
        target_id=args.target_id,
        physical_error_rate=args.physical_p,
        surgery_rounds=args.rounds,
    )
    print(json.dumps({"output": str(args.out), "case_id": payload["case_id"]}, indent=2))
    return 0


def build_quilt_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a validated quilt circuit and DEM.")
    parser.add_argument("quilt", type=Path, help="explicit quilt.json input")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-p", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--section-variant", type=int)
    parser.add_argument(
        "--no-raw-checks",
        action="store_false",
        dest="preserve_raw_checks",
        default=None,
    )
    args = parser.parse_args(argv)
    summary = build_quilt_files(
        args.quilt,
        args.out,
        physical_p=args.physical_p,
        stitch_rounds=args.rounds,
        section_variant=args.section_variant,
        preserve_raw_checks=args.preserve_raw_checks,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_control_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the noninteracting memory control matched to a quilt."
    )
    parser.add_argument("quilt", type=Path, help="explicit quilt.json input")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--physical-p", type=float)
    parser.add_argument("--rounds", type=int, help="total ordinary SEC rounds")
    args = parser.parse_args(argv)
    summary = build_control_files(
        args.quilt,
        args.out,
        case_id=args.case_id,
        physical_p=args.physical_p,
        rounds=args.rounds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_memory_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an ordinary single-block idle-memory circuit and DEM."
    )
    parser.add_argument(
        "source",
        help="memory.json, or legacy direct adapter: bb144/rotated-surface",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--physical-p", type=float)
    parser.add_argument("--rounds", type=int)
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="rotated-surface distance (ignored for BB144)",
    )
    parser.add_argument(
        "--sec-schedule",
        choices=(BB_SEC_SCHEDULE_LEGACY, BB144_SEC_SCHEDULE_BRAVYI_2024),
        default=None,
        help="BB144 SEC schedule provenance (legacy direct adapter only)",
    )
    args = parser.parse_args(argv)
    if args.source in {"bb144", "rotated-surface"}:
        summary = build_memory_files(
            args.source,
            args.out,
            case_id=args.case_id,
            physical_p=0.003 if args.physical_p is None else args.physical_p,
            rounds=14 if args.rounds is None else args.rounds,
            distance=12 if args.distance is None else args.distance,
            sec_schedule=(
                BB_SEC_SCHEDULE_LEGACY
                if args.sec_schedule is None
                else args.sec_schedule
            ),
        )
    else:
        if any(
            value is not None
            for value in (
                args.case_id,
                args.physical_p,
                args.rounds,
                args.distance,
                args.sec_schedule,
            )
        ):
            parser.error(
                "memory.json supplies case_id, physical_error_rate, rounds, and "
                "circuit_adapter (including sec_schedule); do not also pass "
                "build overrides"
            )
        summary = build_memory_spec_files(Path(args.source), args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


