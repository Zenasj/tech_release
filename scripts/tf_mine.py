"""Mine TensorFlow properties from an attributed operator/PR plan.

The JSONL record is append-only and each completed pair is skipped on resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import os

from mrminer.fuzz.llm import LocalOpenAIProvider
from mrminer.llm.provider import invoke_structured

PLAN = Path("runs/tensorflow/mining_plan.json")
CORPUS = Path("runs/tensorflow/pr_corpus/tensorflow__tensorflow")
RECORDS = Path("runs/tensorflow/mine_records.jsonl")
INDEX = Path("runs/tensorflow/operator_oracle_index_tf.json")

TASK = "tf.mine_oracle"
SYSTEM_PROMPT = (
    "You extract a metamorphic-relation oracle (MR-oracle) for ONE TensorFlow "
    "operator from a pull request. The compiler under test is XLA "
    "(tf.function(jit_compile=True)); an oracle states a property the operator must "
    "satisfy that a semantics-preserving compilation should NOT change. "
    "Given the op and the PR evidence, decide whether the PR actually establishes "
    "such a property for THIS op (not just a refactor / build / unrelated change). "
    "If yes, state it as a checkable predicate and the input domain where it holds. "
    "aspect='semantic' = an output-value/relation invariant (eager and XLA agree); "
    "aspect='effect' = a validation invariant (an input the op must reject/raise on). "
    "Be conservative: if the PR does not clearly establish a property for this op, "
    "set is_mr=false. Reason briefly, then output ONLY the JSON object."
)
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_mr": {"type": "boolean"},
        "aspect": {"type": "string", "enum": ["semantic", "effect", "none"]},
        "claim_summary": {"type": "string"},
        "predicate": {"type": "string"},
        "input_domain": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["is_mr", "aspect", "claim_summary"],
}


def _body_text(d: dict) -> str:
    b = d.get("body")
    return str(b.get("text") or "") if isinstance(b, dict) else str(b or "")


def evidence(op_stem: str, d: dict, max_patch: int = 1800) -> dict:
    title = str(d.get("title") or "")
    body = _body_text(d)[:1500]
    hunks: list[str] = []
    budget = max_patch
    for tf in d.get("touched_files_patch") or []:
        patch = tf.get("patch") or ""
        path = tf.get("path") or ""
        if op_stem not in patch and op_stem not in path:
            continue
        keep = [ln for ln in patch.splitlines()
                if op_stem in ln or ln[:1] in "+-" and len(ln) < 200]
        for ln in keep:
            if budget <= 0:
                break
            hunks.append(ln[:200])
            budget -= len(ln)
        if budget <= 0:
            break
    return {"title": title, "body": body, "patch_excerpt": "\n".join(hunks)[:max_patch],
            "tests_added": bool(d.get("tests_added"))}


def mine_one(provider, op: str, pr: int) -> dict:
    stem = op.split("::")[-1]
    f = CORPUS / f"{pr}.json"
    if not f.exists():
        return {"op": op, "pr": pr, "status": "no_pr_json"}
    try:
        d = json.loads(f.read_bytes())
    except Exception as e:
        return {"op": op, "pr": pr, "status": "bad_pr_json", "err": str(e)[:100]}
    ev = evidence(stem, d)
    payload = {"op": op, "op_name": stem, "pr_number": pr,
               "title": ev["title"], "body": ev["body"],
               "patch_excerpt": ev["patch_excerpt"], "tests_added": ev["tests_added"]}
    try:
        res = invoke_structured(provider, task=TASK, system_prompt=SYSTEM_PROMPT,
                                input_payload=payload, output_schema=OUTPUT_SCHEMA)
        p = res.payload
    except Exception as e:
        return {"op": op, "pr": pr, "status": "llm_error", "err": f"{type(e).__name__}: {e}"[:160]}
    if (not p.get("is_mr") or p.get("aspect") not in ("semantic", "effect")
            or not p.get("predicate") or not p.get("input_domain")):
        return {"op": op, "pr": pr, "status": "not_mr",
                "aspect": p.get("aspect"), "summary": p.get("claim_summary", "")[:160]}
    return {"op": op, "pr": pr, "status": "ok", "aspect": p["aspect"],
            "summary": p.get("claim_summary", ""), "predicate": p.get("predicate", ""),
            "domain": p.get("input_domain", ""), "confidence": p.get("confidence", "")}


def load_done() -> set:
    done = set()
    if RECORDS.exists():
        for line in RECORDS.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["op"], r["pr"]))
            except Exception:
                continue
    return done


def build_index() -> dict:
    """Rebuild the oracle index (fuzz-consumable shape) from the ok records."""
    index: dict[str, list] = {}
    n = 0
    for line in (RECORDS.read_text().splitlines() if RECORDS.exists() else []):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok":
            continue
        grid = {"$exec_mode": ["eager", "xla"]}
        oid = "tf-mined::" + hashlib.sha256(
            f"{r['op']}|{r['pr']}|{r['predicate']}".encode()).hexdigest()[:16]
        index.setdefault(r["op"], []).append({
            "oracle_id": oid, "aspect": r["aspect"],
            "variation": {"param_grid": grid},
            "claim": {"summary": f"{r['summary']} Predicate: {r['predicate']}. "
                                 f"Applicability: {r.get('domain', '')}."},
            "provenance": {"pr": r["pr"], "predicate": r["predicate"],
                           "domain": r.get("domain", ""), "confidence": r.get("confidence", "")},
        })
        n += 1
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps({"index": index, "meta": {"oracles": n, "ops": len(index)}}, indent=2))
    return {"oracles": n, "ops": len(index)}


def main() -> int:
    global PLAN, CORPUS, RECORDS, INDEX
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="mine at most N pairs this run")
    ap.add_argument("--rebuild-index-only", action="store_true")
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--records", type=Path, default=RECORDS)
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--api-base", default=os.environ.get("MRMINER_API_BASE", ""))
    ap.add_argument("--model", default=os.environ.get("MRMINER_MODEL", ""))
    ns = ap.parse_args()
    PLAN, CORPUS, RECORDS, INDEX = ns.plan, ns.corpus, ns.records, ns.index

    if ns.rebuild_index_only:
        print("index:", build_index(), "->", INDEX)
        return 0

    plan = json.loads(PLAN.read_text())["plan"]
    pairs = [(o["op"], pr) for o in plan for pr in o["plan_prs"]]
    done = load_done()
    todo = [(op, pr) for (op, pr) in pairs if (op, pr) not in done]
    if ns.limit:
        todo = todo[: ns.limit]
    if not ns.api_base or not ns.model:
        ap.error("set MRMINER_API_BASE and MRMINER_MODEL or pass --api-base and --model")
    provider = LocalOpenAIProvider(base_url=ns.api_base, model=ns.model)
    print(f"[tf_mine] plan pairs={len(pairs)} done={len(done)} todo={len(todo)}", flush=True)

    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "not_mr": 0, "err": 0}
    t0 = time.time()
    with RECORDS.open("a") as fh:
        for i, (op, pr) in enumerate(todo, 1):
            r = mine_one(provider, op, pr)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            st = r["status"]
            stats["ok" if st == "ok" else "not_mr" if st == "not_mr" else "err"] += 1
            if st == "ok":
                print(f"  [{i}/{len(todo)}] OK  {op.split('::')[-1]:24} #{pr}  {r['aspect']:8} {r['summary'][:70]}", flush=True)
            elif i % 20 == 0:
                el = time.time() - t0
                print(f"  [{i}/{len(todo)}] .. ok={stats['ok']} not_mr={stats['not_mr']} err={stats['err']} "
                      f"({el/i:.1f}s/call)", flush=True)
    idx = build_index()
    print(f"[tf_mine] done. this run: {stats}; index now {idx} -> {INDEX}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
