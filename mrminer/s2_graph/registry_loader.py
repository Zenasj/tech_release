"""Build a version-bound ATen operator registry from source and runtime facts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

import yaml

from mrminer.common.schema import (
    EntityKind,
    EntitySource,
    EntitySpec,
    OperatorIntrinsic,
)

_ATEN_NAME = re.compile(
    r"^(?:aten(?:::|\.))(?P<base>[A-Za-z_][A-Za-z0-9_]*)(?:\.(?P<overload>[A-Za-z0-9_]+))?$"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class RegistrySnapshot:
    artifact_id: str
    source_root: Path
    source_commit: str
    torch_version: str | None
    native_functions_sha256: str
    entities: tuple[EntitySpec, ...]
    dispatch_symbols: dict[str, tuple[str, ...]]
    source_counts: dict[str, int]

    @property
    def operator_ids(self) -> frozenset[str]:
        return frozenset(entity.entity_id for entity in self.entities)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "operator_registry_manifest_v1",
            "artifact_id": self.artifact_id,
            "source_root": str(self.source_root),
            "source_commit": self.source_commit,
            "torch_version": self.torch_version,
            "native_functions_path": str(
                self.source_root
                / "aten"
                / "src"
                / "ATen"
                / "native"
                / "native_functions.yaml"
            ),
            "native_functions_sha256": self.native_functions_sha256,
            "operator_count": len(self.entities),
            "dispatch_symbol_count": len(self.dispatch_symbols),
            "source_counts": dict(sorted(self.source_counts.items())),
        }


def operator_entity_id(base_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", base_name):
        raise ValueError(f"invalid ATen base name: {base_name!r}")
    return f"op:aten::{base_name}"


def parse_aten_name(value: str) -> tuple[str, str] | None:
    """Return (base, overload), accepting dispatcher and Python spellings."""
    match = _ATEN_NAME.fullmatch(value.strip())
    if match is None:
        return None
    return match.group("base"), match.group("overload") or "default"


def _schema_name(func: str) -> tuple[str, str] | None:
    head = func.partition("(")[0].strip()
    if "::" in head:
        namespace, name = head.split("::", 1)
        if namespace != "aten":
            return None
    else:
        name = head
    base, separator, overload = name.partition(".")
    if not _IDENTIFIER.fullmatch(base):
        return None
    return base, overload if separator else "default"


def _dispatch_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _dispatch_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _dispatch_values(item)


def _dispatch_identifier(value: str) -> str | None:
    candidate = value.strip().split("::")[-1]
    if _IDENTIFIER.fullmatch(candidate):
        return candidate
    return None


def _table_names(table: Any) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for raw_key in table:
        keys = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        for key in keys:
            try:
                value = key.name()
            except (AttributeError, TypeError):
                value = str(key)
            parsed = parse_aten_name(value)
            if parsed is not None:
                result.add(parsed)
    return result


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
        raise ValueError("PyTorch source root did not return a full commit SHA")
    return commit


def load_operator_registry(
    source_root: Path,
    *,
    include_runtime: bool = True,
) -> RegistrySnapshot:
    """Load ATen schemas and optional runtime registration tables.

    Runtime introspection is accepted only when the imported torch package lives
    inside ``source_root``. This prevents silently mixing unrelated revisions.
    """
    source_root = source_root.resolve()
    native_path = (
        source_root
        / "aten"
        / "src"
        / "ATen"
        / "native"
        / "native_functions.yaml"
    )
    payload = native_path.read_bytes()
    documents = yaml.safe_load(payload)
    if not isinstance(documents, list):
        raise ValueError("native_functions.yaml must contain a top-level array")

    overloads: dict[str, set[str]] = defaultdict(set)
    dispatch_symbols: dict[str, set[str]] = defaultdict(set)
    native_schema_count = 0
    for item in documents:
        if not isinstance(item, dict) or not isinstance(item.get("func"), str):
            continue
        parsed = _schema_name(item["func"])
        if parsed is None:
            continue
        base, overload = parsed
        entity_id = operator_entity_id(base)
        overloads[base].add(overload)
        native_schema_count += 1
        for raw_symbol in _dispatch_values(item.get("dispatch")):
            symbol = _dispatch_identifier(raw_symbol)
            if symbol is not None:
                dispatch_symbols[symbol].add(entity_id)

    runtime_names: set[tuple[str, str]] = set()
    decomp_names: set[tuple[str, str]] = set()
    lowering_names: set[tuple[str, str]] = set()
    meta_names: set[tuple[str, str]] = set()
    torch_version: str | None = None
    if include_runtime:
        import torch

        torch_path = Path(torch.__file__).resolve()
        if not torch_path.is_relative_to(source_root):
            raise ValueError(
                f"torch runtime {torch_path} is not built from {source_root}"
            )
        torch_version = torch.__version__
        runtime_names = {
            parsed
            for value in torch._C._dispatch_get_all_op_names()
            if (parsed := parse_aten_name(value)) is not None
        }
        from torch._decomp import decomposition_table, meta_table
        from torch._inductor.lowering import lowerings

        decomp_names = _table_names(decomposition_table)
        meta_names = _table_names(meta_table)
        lowering_names = _table_names(lowerings)
        for base, overload in runtime_names:
            overloads[base].add(overload)

    def registered(
        table: set[tuple[str, str]],
        base: str,
    ) -> bool:
        return any(item_base == base for item_base, _ in table)

    source_commit = _git_commit(source_root)
    artifact_id = f"art:pytorch-{source_commit[:12]}"
    entities = tuple(
        EntitySpec(
            entity_kind=EntityKind.OPERATOR,
            entity_id=operator_entity_id(base),
            display_name=f"aten.{base}",
            intrinsic=OperatorIntrinsic(
                overloads=tuple(sorted(values)),
                has_lowering=registered(lowering_names, base),
                has_decomp=registered(decomp_names, base),
                has_meta=registered(meta_names, base),
                decomp_targets=(),
            ),
            first_seen=artifact_id,
            source=EntitySource.REGISTRY,
        )
        for base, values in sorted(overloads.items())
    )
    return RegistrySnapshot(
        artifact_id=artifact_id,
        source_root=source_root,
        source_commit=source_commit,
        torch_version=torch_version,
        native_functions_sha256=hashlib.sha256(payload).hexdigest(),
        entities=entities,
        dispatch_symbols={
            symbol: tuple(sorted(entity_ids))
            for symbol, entity_ids in sorted(dispatch_symbols.items())
        },
        source_counts={
            "native_schemas": native_schema_count,
            "runtime_overloads": len(runtime_names),
            "decomposition_overloads": len(decomp_names),
            "lowering_overloads": len(lowering_names),
            "meta_overloads": len(meta_names),
        },
    )
