#!/usr/bin/env python3
"""Index the official StableHLO operations and audit a pinned TableGen source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


OPS_PATTERN = re.compile(
    r'^def\s+StableHLO_\w+Op\s*:\s*StableHLO_\w+<\s*"([a-z][a-z0-9_]*)"',
    re.MULTILINE,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to stablehlo/dialect/StablehloOps.td")
    parser.add_argument("--spec-ops", type=Path,
                        default=Path("configs/stablehlo_spec_ops.json"),
                        help="Official specification operation list")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int,
                        help="Optional guard for a known StableHLO revision")
    args = parser.parse_args()
    content = args.source.read_bytes()
    runtime_names = OPS_PATTERN.findall(content.decode("utf-8"))
    spec = json.loads(args.spec_ops.read_text())
    names = spec["ops"]
    if (not names or len(names) != len(set(names))
            or not runtime_names or len(runtime_names) != len(set(runtime_names))):
        raise ValueError("StableHLO specification or TableGen definitions are invalid")
    if args.expected_count is not None and len(names) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} specification operations, found {len(names)}")
    commit = subprocess.run(
        ["git", "-C", str(args.source.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    payload = {
        "dialect": "stablehlo", "operator_count": len(names),
        "ops": [f"op:stablehlo::{name}" for name in sorted(names)],
        "spec_source": spec.get("source"),
        "spec_only": sorted(set(names) - set(runtime_names)),
        "tablegen_only": sorted(set(runtime_names) - set(names)),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_commit": commit.stdout.strip() if commit.returncode == 0 else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"operator_count": len(names), "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
