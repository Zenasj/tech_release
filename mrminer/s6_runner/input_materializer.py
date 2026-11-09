"""Strict PyTorch materialization for supported InputManifest generators."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from mrminer.common.schema import (
    InputManifest,
    InputNodeKind,
    InputTree,
    MaterializationMode,
)


class UnsupportedInputMaterialization(ValueError):
    pass


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(seed % (2**32))


def _torch_dtype(name: str):
    import torch

    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return supported[name]
    except KeyError as error:
        raise UnsupportedInputMaterialization(
            f"torch.randn.v1 does not support dtype {name!r}"
        ) from error


def _tensor_count(tree: InputTree) -> int:
    if tree.kind is InputNodeKind.TENSOR:
        return 1
    if tree.kind in {InputNodeKind.TUPLE, InputNodeKind.LIST}:
        return sum(_tensor_count(child) for child in tree.children)
    if tree.kind is InputNodeKind.DICT:
        return sum(_tensor_count(child) for _, child in tree.entries)
    return 0


def _materialize_tree(tree: InputTree, *, device: str | None) -> Any:
    import torch

    if tree.kind is InputNodeKind.TENSOR:
        spec = tree.tensor_spec
        if spec is None:
            raise AssertionError("validated tensor node has no TensorSpec")
        if spec.layout != "contiguous":
            raise UnsupportedInputMaterialization(
                f"torch.randn.v1 does not support layout {spec.layout!r}"
            )
        requested_device = device or spec.device_class
        if requested_device.split(":", 1)[0] != spec.device_class:
            raise UnsupportedInputMaterialization(
                f"device override {requested_device!r} conflicts with "
                f"TensorSpec device_class={spec.device_class!r}"
            )
        return torch.randn(
            spec.shape,
            dtype=_torch_dtype(spec.dtype),
            device=requested_device,
            requires_grad=spec.requires_grad,
        )
    if tree.kind is InputNodeKind.SCALAR:
        return tree.scalar
    if tree.kind is InputNodeKind.NONE:
        return None
    if tree.kind is InputNodeKind.TUPLE:
        return tuple(
            _materialize_tree(child, device=device)
            for child in tree.children
        )
    if tree.kind is InputNodeKind.LIST:
        return [
            _materialize_tree(child, device=device)
            for child in tree.children
        ]
    if tree.kind is InputNodeKind.DICT:
        return {
            key: _materialize_tree(child, device=device)
            for key, child in tree.entries
        }
    raise UnsupportedInputMaterialization(f"unknown input node kind: {tree.kind}")


@dataclass(frozen=True)
class TorchInputMaterializer:
    manifest: InputManifest
    device: str | None = None

    def __call__(self) -> Any:
        generator = self.manifest.generator
        if self.manifest.materialization.mode is not MaterializationMode.REGENERATE:
            raise UnsupportedInputMaterialization(
                "TorchInputMaterializer only supports regenerate manifests"
            )
        if generator.generator_id != "torch.randn.v1":
            raise UnsupportedInputMaterialization(
                f"unsupported input generator: {generator.generator_id}"
            )
        if generator.algo_version != "1":
            raise UnsupportedInputMaterialization(
                f"unsupported torch.randn algorithm version: "
                f"{generator.algo_version}"
            )
        if generator.params.get("distribution") != "normal_0_1":
            raise UnsupportedInputMaterialization(
                "torch.randn.v1 requires distribution='normal_0_1'"
            )
        tensor_order = generator.params.get("tensor_order")
        if tensor_order is not None and (
            not isinstance(tensor_order, list)
            or any(not isinstance(name, str) or not name for name in tensor_order)
            or len(tensor_order) != len(set(tensor_order))
            or len(tensor_order) != _tensor_count(self.manifest.input_tree)
        ):
            raise UnsupportedInputMaterialization(
                "tensor_order must contain one unique name per tensor leaf"
            )
        unknown_params = set(generator.params) - {
            "distribution",
            "tensor_order",
        }
        if unknown_params:
            raise UnsupportedInputMaterialization(
                "unsupported torch.randn.v1 params: "
                + ", ".join(sorted(unknown_params))
            )
        _seed_all(self.manifest.seed)
        return _materialize_tree(
            self.manifest.input_tree,
            device=self.device,
        )
