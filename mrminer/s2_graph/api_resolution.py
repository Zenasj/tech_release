"""R2.3a: conservative aten -> public Python API resolution (graph_model_v1 §5).

A PR modifies aten/C++ implementation, so its diff evidence is at the aten level.
To let the graph node be a public Python API, we resolve an aten operator to a
Python API by a principled **namesake** rule: aten op ``X`` resolves to the
public functional ``torch.nn.functional.X`` iff that API exists in the API
registry (R2.2). When there is no namesake API, resolution returns ``None`` and
the linker keeps the aten operator node as fallback. This is conservative and
non-arbitrary (no hand-tuned table); diff evidence still points at the aten
symbol. Non-namesake cases (e.g. aten::addmm) can be resolved later via tracing.
"""

from __future__ import annotations

from collections.abc import Iterable

from mrminer.common.schema.entity import api_operator_id
from mrminer.s2_graph.registry_loader import parse_aten_name


def _aten_base(aten_op_id: str) -> str | None:
    """Return the base op name of an ``op:aten::X`` id, or None if not aten."""

    if not aten_op_id.startswith("op:aten::"):
        return None
    parsed = parse_aten_name(aten_op_id.removeprefix("op:"))
    if parsed is not None:
        return parsed[0]
    base = aten_op_id.removeprefix("op:aten::")
    return base.split(".", 1)[0] or None


def resolve_aten_to_api(
    aten_op_id: str,
    api_registry_ids: Iterable[str] | frozenset[str],
) -> str | None:
    """Resolve an aten operator ID to a public functional API ID, or None.

    Namesake rule: ``op:aten::relu`` -> ``api:torch.nn.functional.relu`` iff that
    API is present in ``api_registry_ids``. Returns None (aten fallback) when the
    input is not aten, has no base name, or has no namesake API in the registry.
    """

    base = _aten_base(aten_op_id)
    if base is None:
        return None
    ids = api_registry_ids if isinstance(api_registry_ids, frozenset) else frozenset(
        api_registry_ids
    )
    candidate = api_operator_id(f"torch.nn.functional.{base}")
    return candidate if candidate in ids else None
