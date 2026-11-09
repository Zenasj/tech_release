"""M7.4: persist validated (program, input) pairs and feed induction.

A ``ValidatedProgramInput`` from M7.2 is a loose fact until it is durably
catalogued and offered to the inducer. This module persists the catalog
deterministically and exposes only the program IDs that actually validated, so
``InductionContext.program_ids`` is fed reproducible, executable programs and
never a pair that failed or ran non-deterministically (open questions doc S3
item 3).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from mrminer.s3_oracle.program_input_validation import (
    ProgramInputStatus,
    ValidatedProgramInput,
)


def persist_validated_pairs(
    pairs: Iterable[ValidatedProgramInput],
    path: Path,
) -> bytes:
    """Write validated pairs as deterministic JSONL and return the bytes.

    Rows are ordered by ``(program_id, input_id, device)`` and each line is
    canonical JSON, so re-persisting the same catalog (dry-run then apply) is
    byte-identical.
    """

    ordered = sorted(
        pairs,
        key=lambda pair: (pair.program_id, pair.input_id, pair.device),
    )
    payload = "".join(
        json.dumps(pair.to_record(), sort_keys=True, separators=(",", ":")) + "\n"
        for pair in ordered
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def load_validated_pairs(path: Path) -> list[ValidatedProgramInput]:
    return [
        ValidatedProgramInput.from_record(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def eligible_program_ids(
    pairs: Iterable[ValidatedProgramInput],
) -> tuple[str, ...]:
    """Sorted, unique program IDs whose pair validated.

    Only ``VALIDATED`` pairs qualify; every other typed status is excluded so
    the inducer is never handed a program that did not run reproducibly.
    """

    return tuple(
        sorted(
            {
                pair.program_id
                for pair in pairs
                if pair.status is ProgramInputStatus.VALIDATED
            }
        )
    )
