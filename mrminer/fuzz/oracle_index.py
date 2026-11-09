"""Read-only access to the mined oracle library (the agent's knowledge base).

Reuses the existing index verbatim; does not modify it.
"""
from __future__ import annotations

import json
from pathlib import Path

from mrminer.fuzz.types import OracleRef

DEFAULT_INDEX = Path("reports/oracle_validation_v2/operator_oracle_index_full.json")


def load_index(path: Path = DEFAULT_INDEX) -> dict[str, list[dict]]:
    return json.loads(Path(path).read_text())["index"]


def attach_core_contextual_ops(
    observed_ops: list[str], index: dict[str, list[dict]],
) -> list[str]:
    """Expose Core retrieval hooks when their original ATen source is observed."""
    present = set(observed_ops)
    added = []
    for key, items in index.items():
        if any(item.get("home_tier") in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"}
               and (item.get("guard") or {}).get("static", {}).get("source_entry", {}).get("operator") in present
               for item in items):
            added.append(key)
    return list(dict.fromkeys(added + observed_ops))


def oracles_for(ops, index: dict[str, list[dict]]) -> list[OracleRef]:
    """All oracles bound to any of `ops` (the ones an IR seed makes applicable)."""
    refs: list[OracleRef] = []
    present = set(ops)
    for op in ops:
        entries = index.get(op, [])
        if any(item.get("home_tier") in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"}
               and (item.get("guard") or {}).get("static", {}).get("source_entry", {}).get("operator") in present
               for item in entries):
            entries = sorted(entries, key=lambda item: (
                0 if item.get("home_tier") in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"}
                and (item.get("guard") or {}).get("static", {}).get("source_entry", {}).get("operator") in present
                else 1))
        for o in entries:
            motif = (o.get("guard") or {}).get("static", {}).get("lowering_motif") or []
            if motif and not set(motif).issubset(present):
                continue
            source = (o.get("guard") or {}).get("static", {}).get("source_entry") or {}
            if op.startswith("op:stablehlo::") and source.get("operator") not in present:
                continue
            tier = o.get("home_tier")
            if tier in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"} and source.get("operator") not in present:
                continue
            claim = o.get("claim", {}).get("summary", "")
            if op.startswith("op:stablehlo::") and source.get("operator"):
                claim = f"TF source operation: {source['operator']}. {claim}"
            if tier in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"}:
                claim = f"ATen source operation: {source['operator']}. {claim}"
            target_op = (source.get("operator", op) if op.startswith("op:stablehlo::")
                         or tier in {"CORE_SOURCE_CONTEXTUAL", "CORE_COMPOSITE_OUTPUT"} else op)
            refs.append(
                OracleRef(
                    oracle_id=o["oracle_id"],
                    op=target_op,
                    aspect=o["aspect"],
                    variation=(o.get("variation") or {}).get("param_grid") or {},
                    claim=claim,
                    params=(o.get("relation") or {}).get("params") or {},
                )
            )
    return refs
