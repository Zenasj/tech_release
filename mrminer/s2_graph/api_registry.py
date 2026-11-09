"""R2.2: enumerate the public Python API surface as ``api:*`` operators.

graph_model_v1 §3 redefines an operator as an application-layer public Python
API. This module mints the API universe from the two cleanest, test-generation
relevant namespaces:

- ``torch.nn.functional`` functions (defined in that module, not imported types);
- ``torch.nn`` ``nn.Module`` subclasses.

Each becomes an ``api:<qualname>`` :class:`EntitySpec` with an ``ApiIntrinsic``.
``resolved_aten`` is left empty here; the aten->API resolution (R2.3 linker)
fills it, and an empty binding is a legal fallback (graph_model_v1 §5). Top-level
``torch.*`` ops can be added later without changing this contract.
"""

from __future__ import annotations

import inspect

from mrminer.common.schema import (
    ApiIntrinsic,
    EntityKind,
    EntitySource,
    EntitySpec,
    api_operator_id,
)


def _functional_qualnames() -> dict[str, str]:
    import torch

    module = torch.nn.functional
    found: dict[str, str] = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        # Keep torch callables that are functions (not classes/modules), so
        # C-backed ops like linear/conv2d (module torch._C._nn / torch) are
        # included while imported types (Tensor -> class, Optional -> typing)
        # are dropped.
        if (
            callable(obj)
            and not inspect.isclass(obj)
            and not inspect.ismodule(obj)
            and getattr(obj, "__module__", "").split(".", 1)[0] == "torch"
        ):
            found[f"torch.nn.functional.{name}"] = "function"
    return found


def _module_qualnames() -> dict[str, str]:
    import torch

    found: dict[str, str] = {}
    for name in dir(torch.nn):
        if name.startswith("_"):
            continue
        obj = getattr(torch.nn, name)
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            found[f"torch.nn.{name}"] = "module"
    return found


def load_api_registry(*, first_seen: str) -> tuple[EntitySpec, ...]:
    """Deterministically mint the public Python API operator universe.

    ``first_seen`` is the artifact ID the enumeration was taken against (the
    universe is torch-version bound, like the aten registry).
    """

    surface: dict[str, str] = {}
    surface.update(_functional_qualnames())
    surface.update(_module_qualnames())

    entities = tuple(
        EntitySpec(
            entity_kind=EntityKind.API,
            entity_id=api_operator_id(qualname),
            display_name=qualname,
            intrinsic=ApiIntrinsic(qualname=qualname, api_kind=api_kind),
            first_seen=first_seen,
            source=EntitySource.REGISTRY,
        )
        for qualname, api_kind in sorted(surface.items())
    )
    return entities
