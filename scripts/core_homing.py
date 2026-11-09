#!/usr/bin/env python3
"""Core-ATen homing transform for MRMiner property subjects.

Given a property's current subject operator, decide the Core ATen operator the
property should be homed onto, and the entry operator to record in the
applicability predicate P. Homing tiers:

  A  subject is already Core ATen            -> keep subject; entry = itself
  B  in-place `X_` whose `X` is Core ATen    -> subject = X; entry = `X_`
  E  otherwise: trace the Core ATen decomposition (make_fx + core_aten_decompositions,
     fake tensors, no GPU, no real data) and take the output node's producing op.
       - single Core ATen output op           -> subject = that op; entry = original
       - tuple / getitem (multi-output)        -> NOT homed (keep subject, flag)
       - non-decomposing primitive / trace fail-> NOT homed (keep subject, flag)

Residual policy (ops that cannot be cleanly homed to a single Core op): keep the
original subject, still record the entry in P, and flag `core_homed = False`. No
property is dropped and nothing is falsely labelled Core.

This module is pure analysis: it reads the framework only, writes nothing.

Self-test: python scripts/core_homing.py --selftest

Used by scripts/migrate_library_to_core.py to transform the existing library.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch._ops import OpOverload, OpOverloadPacket
from torch._decomp import core_aten_decompositions
from torch.fx.experimental.proxy_tensor import make_fx

_CORE_PACKETS: set[str] | None = None


def core_packets() -> set[str]:
    """Names of ATen OpOverloadPackets that carry at least one `core`-tagged overload."""
    global _CORE_PACKETS
    if _CORE_PACKETS is None:
        s: set[str] = set()
        for n in dir(torch.ops.aten):
            pk = getattr(torch.ops.aten, n, None)
            if isinstance(pk, OpOverloadPacket) and any(
                torch.Tag.core in getattr(pk, ov).tags for ov in pk.overloads()
            ):
                s.add(n)
        _CORE_PACKETS = s
    return _CORE_PACKETS


def packet_name(entity: str) -> str:
    """Normalise `op:aten::add`, `aten::add`, `aten.add.Tensor`, `add` -> `add`."""
    s = entity
    for pre in ("op:aten::", "aten::", "op:aten.", "aten."):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    # strip overload / dotted tail, e.g. add.Tensor -> add
    return s.split(".", 1)[0]


def is_core(name: str) -> bool:
    return packet_name(name) in core_packets()


# Factory/allocator ops with no `core` tag whose Core equivalent is an exact
# value-identity (ones == full fill 1, zeros == full fill 0) or the same allocation
# under a layout argument (empty_permuted / new_empty_strided are empty / empty_strided
# with a stride spec). Homing to the Core equivalent is consistent with the accepted
# ones_like -> full_like ruling; the entry stays in P so the fill/layout is recoverable.
_SEMANTIC_CORE: dict[str, str] = {
    "ones": "full",
    "zeros": "full",
    "empty_permuted": "empty",
    "new_empty_strided": "empty_strided",
}


def peel_to_core(name: str) -> tuple[str | None, str | None]:
    """Peel a traced terminal op to its Core subject.

    Three cases, each principled (never invents a decomposition):
      - in-place `X_`  -> Core functional `X`                 (Tier-B rule on the terminal)
      - functionalized view `X_copy` -> Core view `X`
      - factory/allocator semantic equivalent (ones/zeros -> full, ...)
    Return `(core_name, relation)` when a Core subject exists, else `(None, None)`.
    """
    base = packet_name(name)
    cp = core_packets()
    if base.endswith("_") and base[:-1] in cp:           # in-place -> functional
        return base[:-1], "inplace_functional"
    if base.endswith("_copy") and base[:-5] in cp:        # functionalized view -> view
        return base[:-5], "view_functional"
    tgt = _SEMANTIC_CORE.get(base)                        # factory/allocator equivalent
    if tgt and tgt in cp:
        return tgt, "semantic_equivalent"
    return None, None


@dataclass
class HomeResult:
    new_subject: str          # packet name to home the property onto
    entry: str                # entry/source operator to record in P
    core_homed: bool          # True iff new_subject is a Core ATen op
    tier: str                 # A | B | E-traced | E-multiout | E-noncore-out | E-tracefail | E-noinput
    note: str = ""
    device: str = "cpu"       # device the homing was established on; material context for P
                              # ("cuda" iff only a real GPU trace could decompose the op)


def _output_producer(gm) -> str | None:
    """Return the str(target) of the node that produces the graph's single output,
    or None when the output is a tuple / getitem (multi-output)."""
    out = next(n for n in gm.graph.nodes if n.op == "output")
    val = out.args[0]
    if isinstance(val, (tuple, list)):
        if len(val) != 1:
            return None
        val = val[0]
    tgt = getattr(val, "target", None)
    if tgt is None:
        return None
    s = str(tgt)
    if "getitem" in s:
        return None
    return s


def home_subject(subject: str, fn=None, example_inputs=None) -> HomeResult:
    """Decide the Core homing for one subject operator.

    `fn` + `example_inputs` enable Tier-E tracing; omit them to skip tracing
    (non-core subjects then return a non-homed result with tier E-noinput).
    """
    name = packet_name(subject)

    if is_core(name):
        return HomeResult(name, name, True, "A", "already core")

    if name.endswith("_") and is_core(name[:-1]):
        return HomeResult(name[:-1], name, True, "B", "in-place -> functional core")

    if fn is None or example_inputs is None:
        return HomeResult(name, name, False, "E-noinput", "no example to trace")

    try:
        gm = make_fx(fn, decomposition_table=core_aten_decompositions(),
                     tracing_mode="fake")(*example_inputs)
    except Exception as e:  # noqa: BLE001 - record and continue
        return HomeResult(name, name, False, "E-tracefail", f"{type(e).__name__}: {str(e)[:70]}")

    producer = _output_producer(gm)
    if producer is None:
        return HomeResult(name, name, False, "E-multiout", "tuple/getitem output")
    out_pkt = packet_name(producer)
    if is_core(out_pkt):
        return HomeResult(out_pkt, name, True, "E-traced", f"output={producer}")
    return HomeResult(name, name, False, "E-noncore-out", f"output={producer}")


def _selftest() -> int:
    x = torch.randn(4, 3)
    cases = [
        ("div", (lambda a: torch.div(a, a + 1), (x,))),                 # A: core
        ("div_", (lambda a: a.clone().div_(a + 1), (x,))),              # B: in-place -> div
        ("empty_like", (lambda a: torch.empty_like(a), (x,))),          # E: single core out
        ("floor_divide", (lambda a: torch.floor_divide(a, a + 1), (x,))),  # E: -> div
        ("baddbmm", (lambda a: torch.baddbmm(torch.randn(2, 3, 3),
                                             torch.randn(2, 3, 4),
                                             torch.randn(2, 4, 3)), (x,))),  # E: -> add
        ("native_batch_norm", (lambda a: torch.ops.aten.native_batch_norm(
            torch.randn(2, 3, 4, 4), torch.ones(3), torch.zeros(3),
            torch.zeros(3), torch.ones(3), False, 0.1, 1e-5), (x,))),   # E: multi-output
        ("set_", (None, None)),                                         # E: no input -> not homed
    ]
    print(f"torch {torch.__version__}  | core packets: {len(core_packets())}")
    print(f"{'subject':20s} {'->new_subject':22s} {'entry':14s} {'homed':6s} tier")
    for subj, (fn, ex) in cases:
        r = home_subject(subj, fn=fn, example_inputs=ex)
        print(f"{subj:20s} {r.new_subject:22s} {r.entry:14s} {str(r.core_homed):6s} {r.tier}  {r.note}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true", help="run the built-in demo and exit")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
