"""Link a TF/XLA PR diff to the TF ops it touches (S2 for the TF line).

Analog of mrminer/s2_graph/linker.py's aten registration-form parsers, for TF's
registration forms. From a PR's touched-file patches we detect these forms and
extract the op name, then validate it against the TF op universe
(reports/tf/op_universe.json). Output: op -> {signals} per PR, i.e. the
TF-op<->PR edges.

Registration forms:
  schema   REGISTER_OP("X")                         tensorflow/core/ops/*.cc
  kernel   REGISTER_KERNEL_BUILDER(Name("X")...)    tensorflow/core/kernels/*
  xla      REGISTER_XLA_OP(Name("X"), ...)          compiler/tf2xla/kernels/*
  grad     @ops.RegisterGradient("X")               python/ops/*_grad.py
  mlir     TF::XOp / TF_XOp (legalize patterns)     compiler/mlir/.../legalize_tf*

    PYTHONPATH=. python -m mrminer.tf.linker_tf --corpus data/pr_corpus_tf_smoke/tensorflow__tensorflow
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

_UNIVERSE: set[str] | None = None


def universe(path: str | None = None) -> set[str]:
    global _UNIVERSE
    if _UNIVERSE is None:
        p = Path(path or os.environ.get("MRMINER_TF_OP_UNIVERSE", "runs/tensorflow/op_universe.json"))
        _UNIVERSE = set(json.loads(p.read_text())["ops"].keys()) if p.exists() else set()
    return _UNIVERSE


def op_id(name: str) -> str:
    return f"op:tf::{name}"


_NAME = r'([A-Za-z][A-Za-z0-9_]*)'
_PATTERNS = [
    ("schema", re.compile(r'REGISTER_OP\(\s*"' + _NAME + r'"')),
    ("kernel", re.compile(r'REGISTER_KERNEL_BUILDER\([^)]*Name\(\s*"' + _NAME + r'"')),
    ("xla", re.compile(r'REGISTER_XLA(?:_[A-Z0-9]+)?_OP\([^)]*Name\(\s*"' + _NAME + r'"')),
    ("grad", re.compile(r'RegisterGradient\(\s*"' + _NAME + r'"')),
    ("mlir", re.compile(r'\bTF(?:::|_)' + _NAME + r'Op\b')),   # TF::MatMulOp / TF_MatMulOp -> MatMul
]


def link_patch(patch: str) -> dict[str, set[str]]:
    """op name -> {signals} found in one file's patch."""
    found: dict[str, set[str]] = {}
    for raw in patch.splitlines():
        if not raw.startswith(("+", "-")) or raw.startswith(("+++", "---")):
            continue
        for sig, pat in _PATTERNS:
            for m in pat.finditer(raw):
                found.setdefault(m.group(1), set()).add(sig)
    return found


def link_pr(pr: dict, *, validate: bool = True) -> dict[str, set[str]]:
    """op name -> {signals} across a PR's touched files (universe-validated)."""
    uni = universe() if validate else None
    ops: dict[str, set[str]] = {}
    for tf_file in pr.get("touched_files_patch") or []:
        patch = tf_file.get("patch")
        if not patch:
            continue
        for name, sigs in link_patch(patch).items():
            if uni is not None and name not in uni:
                continue
            ops.setdefault(name, set()).update(sigs)
        # A change to a uniquely named implementation file also supplies a
        # location signal, even when the changed hunk contains no REGISTER_OP.
        # For example, xla/hlo/builder/lib/svd.cc can be attributed to Svd.
        if uni:
            stem = Path(tf_file.get("path") or tf_file.get("filename") or "").stem.lower()
            if stem:
                matches = [name for name in uni if name.lower() == stem]
                if len(matches) == 1:
                    ops.setdefault(matches[0], set()).add("implementation_path")
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of <pr>.json records")
    ap.add_argument("--out", default="reports/tf/linker_manifest.json")
    ns = ap.parse_args()

    uni = universe()
    print(f"op universe: {len(uni)} ops", flush=True)
    op_prs: dict[str, set[int]] = {}
    sig_counts: Counter = Counter()
    linked_prs = 0
    total = 0
    for f in sorted(Path(ns.corpus).glob("*.json")):
        if not f.stem.isdigit():
            continue
        try:
            pr = json.loads(f.read_bytes())
        except Exception:
            continue
        total += 1
        ops = link_pr(pr)
        if ops:
            linked_prs += 1
        for name, sigs in ops.items():
            op_prs.setdefault(name, set()).add(pr["pr_number"])
            for s in sigs:
                sig_counts[s] += 1

    manifest = {
        "corpus": ns.corpus,
        "prs_scanned": total,
        "prs_linked": linked_prs,
        "ops_linked": len(op_prs),
        "signal_counts": dict(sig_counts),
        "op_prs": {op_id(k): sorted(v) for k, v in sorted(op_prs.items())},
    }
    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(json.dumps(manifest, indent=2))
    print(f"scanned {total} PRs; {linked_prs} linked to >=1 op; {len(op_prs)} distinct ops")
    print(f"signals: {dict(sig_counts)}")
    top = sorted(op_prs.items(), key=lambda kv: -len(kv[1]))[:10]
    print("top linked ops:", [(k, len(v)) for k, v in top])
    print(f"-> {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
