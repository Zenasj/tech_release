"""M7.1: build reconstructible v1 InputManifests for the supported generator.

Generalizes the hand-authored anchor input (``anchors/pr168157._input``) into a
reusable builder so program/input catalog entries no longer hand-write pytrees.
Only the ``torch.randn.v1`` REGENERATE generator that
``s6_runner.input_materializer.TorchInputMaterializer`` can replay is produced,
so every minted manifest is materializable and deterministic by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from mrminer.common.schema import (
    GeneratorSpec,
    InputManifest,
    InputNodeKind,
    InputTree,
    MaterializationMode,
    MaterializationSpec,
    TensorSpec,
)

GENERATOR_ID = "torch.randn.v1"
ALGO_VERSION = "1"
DISTRIBUTION = "normal_0_1"


@dataclass(frozen=True, kw_only=True)
class TensorInputSpec:
    """One positional tensor argument of a reconstructible program."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    device_class: str
    layout: str = "contiguous"
    requires_grad: bool = False


def build_regenerate_input(
    tensors: Sequence[TensorInputSpec],
    *,
    seed: int,
) -> InputManifest:
    """Mint a ``torch.randn.v1`` REGENERATE InputManifest of positional tensors.

    The input tree is a tuple of tensor leaves in ``tensors`` order; the
    generator records that order as ``tensor_order`` so a manifest reviewer can
    map leaves to argument names without re-parsing the tree.
    """

    if not tensors:
        raise ValueError("a reconstructible input requires at least one tensor")
    names = [spec.name for spec in tensors]
    if any(not name for name in names):
        raise ValueError("tensor names must be non-empty")
    if len(names) != len(set(names)):
        raise ValueError("tensor names must be unique")

    children = tuple(
        InputTree(
            kind=InputNodeKind.TENSOR,
            tensor_spec=TensorSpec(
                shape=tuple(spec.shape),
                dtype=spec.dtype,
                device_class=spec.device_class,
                layout=spec.layout,
                requires_grad=spec.requires_grad,
            ),
        )
        for spec in tensors
    )
    input_tree = InputTree(kind=InputNodeKind.TUPLE, children=children)
    generator = GeneratorSpec(
        generator_id=GENERATOR_ID,
        algo_version=ALGO_VERSION,
        params={"distribution": DISTRIBUTION, "tensor_order": list(names)},
    )
    return InputManifest.create(
        input_tree=input_tree,
        generator=generator,
        seed=seed,
        materialization=MaterializationSpec(mode=MaterializationMode.REGENERATE),
    )
