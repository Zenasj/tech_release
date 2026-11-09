"""Additional seed sources for the campaign (diversifies beyond seeds_from_index).

Each source emits `Seed`s the same way `seeds.seeds_from_index` does, so they drop
into `fuzz_campaign` unchanged. Sources here:

  seeds_from_op_db      -- PyTorch's OpInfo op_db: every op that (a) has a mined
                           oracle and (b) appears in op_db becomes a seed carrying
                           a real sample-input domain hint (shape/dtype), so the
                           generator produces in-domain tests. Much wider, more
                           realistic op/input coverage than one bare op per index.
  compositional_seeds   -- combine 2-3 oracle-carrying ops into one seed, so the
                           generator writes a multi-op program (closer to real IR);
                           the loop attaches every combined op's oracles.

The private model zoo (whole nn.Modules) is consumed by the no-LLM fast-scan lane
(scripts/sample_and_test_model.py) rather than here: those seeds are not op-keyed
until traced, which the fast lane already does at run time.
"""
from __future__ import annotations

import random

from mrminer.fuzz.types import Seed


def _op_id(name: str) -> str:
    return f"op:aten::{name}"


def _sample_hint(op_name: str, device: str) -> str:
    """A short natural-language domain hint from one real op_db sample input."""
    try:
        import torch
        from torch.testing._internal.common_methods_invocations import op_db
    except Exception:
        return ""
    oi = next((o for o in op_db if o.name == op_name), None)
    if oi is None:
        return ""
    try:
        for dtype in (torch.float32, torch.int64, torch.bool):
            if dtype not in (oi.dtypes or {dtype}):
                continue
            sis = list(oi.sample_inputs(device, dtype, requires_grad=False))
            if not sis:
                continue
            s = sis[0]
            inp = s.input
            shape = tuple(getattr(inp, "shape", ()) or ())
            dt = getattr(inp, "dtype", None)
            nargs = len(s.args)
            kw = list(s.kwargs)
            return (f"op_db sample: input shape={shape} dtype={dt}; "
                    f"{nargs} positional arg(s); kwargs={kw}; device='{device}'")
    except Exception:
        return ""
    return ""


def seeds_from_op_db(
    index: dict[str, list[dict]], *,
    limit: int | None = None,
    device: str = "cuda",
    with_hints: bool = True,
) -> list[Seed]:
    """One seed per op that is BOTH in op_db and carries a mined oracle."""
    try:
        from torch.testing._internal.common_methods_invocations import op_db
    except Exception:
        return []
    names = sorted({o.name for o in op_db})
    seeds: list[Seed] = []
    for name in names:
        op = _op_id(name)
        if op not in index:            # only ops we actually have oracles for
            continue
        note = f"device='{device}'"
        if with_hints:
            hint = _sample_hint(name, device)
            if hint:
                note = hint
        seeds.append(Seed(seed_id=f"seed:opdb:{name}", ops=(op,), note=note, origin="op_db"))
        if limit is not None and len(seeds) >= limit:
            break
    return seeds


def seeds_from_index_non_opdb(
    index: dict[str, list[dict]], *, device: str = "cuda",
) -> list[Seed]:
    """Oracle-carrying ops that are NOT in PyTorch's OpInfo op_db.

    op_db is the community's own, heavily-tested surface -- fuzzing there rediscovers
    what their tests (and other fuzzers) already find. Ops with a mined oracle but
    absent from op_db are OFF that trodden path (less community coverage), so they are
    the higher-yield place to look for *unreported* bugs."""
    try:
        from torch.testing._internal.common_methods_invocations import op_db
        db = {o.name for o in op_db}
    except Exception:
        db = set()
    seeds: list[Seed] = []
    for op in sorted(index.keys()):
        name = op.split("::")[-1]
        if name in db:
            continue
        seeds.append(Seed(seed_id=f"seed:cold:{name}", ops=(op,),
                          note=f"device='{device}'; off-op_db (under-tested) op",
                          origin="non-opdb"))
    return seeds


# Optimization regimes the generic eager-vs-compiled path under-covers; steering the
# generator into them is where bugs others miss tend to live.
_AXES = {
    "dynamic_shape": ("stress DYNAMIC shapes: vary the batch/seq dim across calls, use "
                      "sizes that are not compile-time constants; also try 0-sized and "
                      "size-1 dims"),
    "layout": ("stress non-contiguous / channels_last LAYOUT: build inputs via "
               ".transpose(...)/.permute(...)/[::2] slices/channels_last so strides are "
               "non-default; the op must still match eager"),
    "edge_shape": ("stress EDGE shapes: 0-dim scalars, empty tensors, size-1 broadcast "
                   "dims, very large single dims, high rank"),
    "mixed_dtype": ("stress low/mixed PRECISION: fp16/bf16 inputs, integer/bool where "
                    "valid, and dtype-promotion boundaries"),
}


def axis_directed_seeds(
    index: dict[str, list[dict]], *,
    k: int = 60, device: str = "cuda", seed: int | None = 0,
) -> list[Seed]:
    """`k` seeds that pin an op to a specific optimization regime via the note, so the
    generator explores off-trodden-path input domains instead of the default one."""
    ops = sorted(index.keys())
    if not ops:
        return []
    rng = random.Random(seed)
    axes = list(_AXES.items())
    seeds: list[Seed] = []
    for i in range(k):
        op = rng.choice(ops)
        axis, hint = axes[i % len(axes)]
        seeds.append(Seed(
            seed_id=f"seed:axis:{axis}:{op.split('::')[-1]}:{i}"[:80],
            ops=(op,),
            note=f"device='{device}'; AXIS={axis} -- {hint}",
            origin=f"axis:{axis}",
        ))
    return seeds


# --- fusion_seeds: pattern-templated compositions that stress inductor/XLA fusion ---
# aten op stems grouped by the role they play in a fusion pattern. Curated to common,
# fusible ops; used to fill the non-target slots of a pattern.
_ROLE_OPS: dict[str, tuple[str, ...]] = {
    "unary":   ("relu", "gelu", "sigmoid", "tanh", "exp", "log", "log1p", "neg", "abs",
                "sqrt", "rsqrt", "sin", "cos", "erf", "reciprocal", "sign", "floor",
                "ceil", "round", "clamp", "softplus", "silu", "square"),
    "binary":  ("add", "sub", "mul", "div", "maximum", "minimum", "pow", "atan2",
                "remainder", "fmod", "hypot", "logaddexp", "copysign"),
    "inplace": ("add_", "sub_", "mul_", "div_", "relu_", "clamp_", "neg_", "abs_",
                "sqrt_", "sigmoid_"),
    "reduce":  ("sum", "mean", "amax", "amin", "prod", "logsumexp", "logcumsumexp",
                "cumsum", "cumprod", "var", "std", "norm", "count_nonzero"),
    "matmul":  ("mm", "bmm", "addmm", "matmul", "mv", "dot", "addbmm", "baddbmm"),
    "layout":  ("t", "transpose", "permute", "reshape", "view", "flatten", "squeeze",
                "unsqueeze", "expand", "as_strided", "contiguous", "narrow", "select"),
}
_ROLE_OF: dict[str, str] = {op: role for role, ops in _ROLE_OPS.items() for op in ops}

# Fusion patterns as ordered role sequences. A target op (drawn from the mined index so
# it carries an oracle) is placed in a slot matching its role; the other slots are filled
# from _ROLE_OPS. Each is a shape known to exercise a distinct inductor/XLA fusion path.
_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pointwise_chain",      ("unary", "binary", "unary")),   # elementwise fusion + dtype promotion
    ("pointwise_then_reduce", ("unary", "binary", "reduce")),  # pointwise -> reduction fusion
    ("reduce_then_pointwise", ("reduce", "unary")),            # reduction -> broadcast-back pointwise
    ("matmul_epilogue",      ("matmul", "binary")),            # gemm epilogue fusion
    ("layout_break",         ("layout", "unary")),             # non-contiguous input into a fused op
    ("inplace_in_chain",     ("unary", "inplace", "unary")),   # in-place op inside a fusible chain
    ("mixed_dtype_chain",    ("binary", "unary")),             # dtype promotion across fusion
)
# dtypes to cycle: weighted toward the risky ones that produced real bugs (complex/bool/
# low-precision), with float32 as the baseline.
_FUSION_DTYPES: tuple[str, ...] = (
    "float32", "complex64", "bool", "uint8", "float16", "bfloat16", "int32", "float64",
)


def fusion_seeds(
    index: dict[str, list[dict]], *,
    k: int = 150,
    device: str = "cuda",
    seed: int | None = 0,
) -> list[Seed]:
    """`k` fusion-directed compositional seeds.

    Each seed anchors on an oracle-carrying op from `index` (so a violated MR-oracle is a
    real finding), places it in a fusion pattern matching its role, fills the other slots
    from `_ROLE_OPS`, and targets a (often risky) dtype. Unlike `compositional_seeds`'
    random op bag, these describe a concrete fusible graph shape -- the region where
    inductor/XLA bugs concentrate (fusion, layout, in-place, cross-op dtype promotion)
    and where single-op seeding has saturated.
    """
    rng = random.Random(seed)
    # oracle-carrying ops, grouped by role (only roles we know how to place)
    idx_ops = [o for o in index if o.startswith("op:aten::")]
    by_role: dict[str, list[str]] = {}
    for o in idx_ops:
        role = _ROLE_OF.get(o.split("::")[-1])
        if role:
            by_role.setdefault(role, []).append(o)
    if not by_role:
        return []
    placeable = sorted(by_role)                      # roles for which we have a target op

    def _fill(role: str) -> str:
        """An op stem for a slot, preferring one that also carries an oracle in `index`."""
        in_idx = [f"op:aten::{s}" for s in _ROLE_OPS[role]
                  if f"op:aten::{s}" in index]
        pick = rng.choice(in_idx) if in_idx else f"op:aten::{rng.choice(_ROLE_OPS[role])}"
        return pick

    seeds: list[Seed] = []
    for i in range(k):
        target_role = rng.choice(placeable)
        target = rng.choice(by_role[target_role])
        # patterns whose role sequence contains the target's role
        cands = [p for p in _PATTERNS if target_role in p[1]]
        if not cands:
            continue
        pname, roles = rng.choice(cands)
        tgt_slot = roles.index(target_role)          # place target at the first matching slot
        ops_by_slot = [target if s == tgt_slot else _fill(role)
                       for s, role in enumerate(roles)]
        # ordered, de-duplicated ops tuple with the target first (oracle priority)
        ops: list[str] = [target]
        for o in ops_by_slot:
            if o not in ops:
                ops.append(o)
        stems = [o.split("::")[-1] for o in ops_by_slot]
        # a readable single-graph chain to steer the generator
        chain = _chain_desc(roles, stems)
        dtype = _FUSION_DTYPES[i % len(_FUSION_DTYPES)]
        note = (
            f"[FUSION seed / pattern={pname}] Build ONE graph inside fn so the compiler "
            f"FUSES these ops into as few kernels as possible (do NOT split into separate "
            f"compiled calls). Chain: {chain}. Target op = {target.split('::')[-1]} "
            f"(the oracle under test). Use dtype {dtype} for the main tensor(s) where legal. "
            f"Vary shapes / strides / broadcasting / edge values to stress the fused kernel: "
            f"prefer non-contiguous inputs (transpose/as_strided), mixed dtypes, in-place "
            f"mutation, and 0-d / empty / size-1 edge shapes. device='{device}'."
        )
        seeds.append(Seed(
            seed_id=f"seed:fusion:{i}:{pname}:{target.split('::')[-1]}"[:80],
            ops=tuple(ops),
            note=note,
            origin="fusion",
        ))
    return seeds


def model_seeds(
    index: dict[str, list[dict]], *,
    path: str = "reports/model_seeds.json",
    k: int | None = None,
    device: str = "cuda",
    seed: int | None = 0,
) -> list[Seed]:
    """Seeds derived from REAL models (precomputed by scripts/build_model_seeds.py).

    Each record is a real model fragment reduced to its oracle-carrying ops + real input
    shapes/dtypes + op chain. The model library is used as SEED MATERIAL only: a seed
    anchors on the model's oracle-carrying ops (so the loop attaches those mined oracles)
    and hands the LLM the real op chain + shapes as realistic context to build
    oracle-targeted tests -- NOT to reproduce the model's own (possibly planted) crash.
    Demonstrates the mined oracles' power on realistic, model-scale op compositions.
    """
    import json
    import os

    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return []
    # keep only records whose target ops still exist in the current index
    usable = []
    for r in records:
        ops = tuple(o for o in r.get("target_ops", []) if o in index)
        if ops:
            usable.append((r, ops))
    if k is not None and len(usable) > k:
        usable = random.Random(seed).sample(usable, k)

    seeds: list[Seed] = []
    for i, (r, ops) in enumerate(usable):
        chain = " -> ".join(o.split("::")[-1] for o in r.get("op_seq", [])[:12])
        shapes = r.get("input_shapes", [])
        dtypes = r.get("input_dtypes", [])
        note = (
            f"[MODEL seed / family={r.get('family','?')}] Derived from a REAL model that chains: "
            f"{chain}. Real input shapes={shapes} dtypes={dtypes}. "
            f"Write focused MR-oracle-targeted `def build(): return fn, args` tests that exercise "
            f"the TARGET op inside a realistic multi-op context like this model (compose it with a "
            f"few of the neighbor ops above; reuse similar shapes/dtypes), and vary edges "
            f"(non-contiguous, mixed dtype, 0-d/empty) to stress the oracle. Do NOT just paste the "
            f"whole model. device='{device}'."
        )
        # cap ops so the loop's oracle retrieval stays focused
        seeds.append(Seed(
            seed_id=f"seed:model:{i}:{ops[0].split('::')[-1]}"[:80],
            ops=ops[:4],
            note=note,
            origin="model",
        ))
    return seeds


def issue_seeds(
    index: dict[str, list[dict]], *,
    path: str = "reports/issue_seeds.json",
    k: int | None = None,
    device: str = "cuda",
    seed: int | None = 0,
) -> list[Seed]:
    """Seeds from REAL upstream PR/issue repros (precomputed by build_issue_seeds.py).

    Each record is a real repro snippet from a collected PR plus the oracle-carrying ops
    it statically references. The repro is handed to the LLM as `mutate_from` (seed
    material) and the seed anchors on its oracle ops, so the LLM rewrites a real upstream
    program into MR-oracle-targeted tests. A finding is a mined-oracle violation on the
    generated test, not the repro's own (often already-fixed) outcome.
    """
    import json
    import os

    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    except Exception:
        return []
    usable = []
    for r in records:
        ops = tuple(o for o in r.get("target_ops", []) if o in index)
        code = r.get("code", "")
        if ops and code:
            usable.append((r, ops, code))
    if k is not None and len(usable) > k:
        usable = random.Random(seed).sample(usable, k)

    seeds: list[Seed] = []
    for i, (r, ops, code) in enumerate(usable):
        note = (
            f"[ISSUE seed / pr={r.get('pr_number')}] `mutate_from` is a REAL upstream repro. "
            f"Take its idea but REWRITE into a focused `def build(): return fn, args` test that "
            f"exercises the TARGET op ({ops[0].split('::')[-1]}) to stress its oracle; drop any "
            f"triton/numpy/timing/class scaffolding, construct tensors directly, vary shapes / "
            f"dtypes / strides / edges. device='{device}'."
        )
        seeds.append(Seed(
            seed_id=f"seed:issue:{r.get('pr_number')}:{ops[0].split('::')[-1]}"[:80],
            ops=ops[:4],
            note=note,
            parent_code=code[:1200],   # cap tokens; the LLM only needs the gist
            origin="issue",
        ))
    return seeds


def _chain_desc(roles: tuple[str, ...], stems: list[str]) -> str:
    """A compact 'v0 -> v1=op(...) -> ...' description of the fused graph."""
    parts, prev = [], "x0"
    for j, (role, op) in enumerate(zip(roles, stems)):
        v = f"v{j}"
        if role == "binary":
            parts.append(f"{v}={op}({prev}, w{j})")
        elif role == "matmul":
            parts.append(f"{v}={op}({prev}, m{j})")
        elif role == "reduce":
            parts.append(f"{v}={op}({prev}, dim=...)")
        elif role == "layout":
            parts.append(f"{v}={op}({prev})  # changes shape/stride")
        elif role == "inplace":
            parts.append(f"{v}={op}({prev}, ...)  # in-place")
        else:  # unary
            parts.append(f"{v}={op}({prev})")
        prev = v
    return " -> ".join(parts)


def compositional_seeds(
    index: dict[str, list[dict]], *,
    k: int = 40,
    max_ops: int = 3,
    device: str = "cuda",
    seed: int | None = 0,
) -> list[Seed]:
    """`k` seeds, each combining 2..max_ops oracle-carrying ops into one program."""
    ops = sorted(index.keys())
    if len(ops) < 2:
        return []
    rng = random.Random(seed)
    seeds: list[Seed] = []
    for i in range(k):
        n = rng.randint(2, max(2, max_ops))
        combo = tuple(rng.sample(ops, min(n, len(ops))))
        stems = "+".join(o.split("::")[-1] for o in combo)
        seeds.append(Seed(
            seed_id=f"seed:combo:{i}:{stems}"[:80],
            ops=combo,
            note=(f"compose these ops into ONE program (chain their outputs where "
                  f"types allow); device='{device}'"),
            origin="compositional",
        ))
    return seeds
