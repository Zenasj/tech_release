"""Append-only findings ledger with dedup + known-bug filtering.

A continuous campaign re-derives the same handful of triggers many times. The
ledger collapses them to a signature and marks each as one of:
  * known   -- matches an already-reported bug (see KNOWN_BUGS); count, don't alert
  * dup     -- signature already in the ledger this campaign; count, don't alert
  * NEW     -- a signature we have not seen and that is not known -> surface it

Only NEW triggers are worth a human look (and are auto-minimized upstream). The
ledger is JSONL so it appends safely from a long-running detached process and can
be re-read/aggregated later.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from mrminer.fuzz.types import Finding

# Users may supply their own known-bug patterns to Ledger.
KNOWN_BUGS: list[tuple[str, str, str]] = []


def _error_class(detail: str) -> str:
    """A stable, low-cardinality key from a finding's detail string."""
    if not detail:
        return ""
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*Error|Exception|Assert[A-Za-z]*)", detail)
    exc = m.group(1) if m else ""
    # first quoted token or first identifier-ish word after the exception
    key = ""
    q = re.search(r"'([^']{1,40})'", detail)
    if q:
        key = q.group(1)
    return f"{exc}:{key}".strip(":")


def signature(f: Finding) -> str:
    return f"{f.op}|{f.verdict}|{_error_class(f.detail)}"


def is_known(f: Finding, known: list[tuple[str, str, str]] | None = None) -> bool:
    d = (f.detail or "")
    for op_s, v_s, e_s in (KNOWN_BUGS if known is None else known):
        if op_s in f.op and v_s in f.verdict and e_s in d:
            return True
    return False


@dataclass
class Verdict:
    status: str          # "known" | "dup" | "NEW"
    sig: str


class Ledger:
    def __init__(self, path: str | Path, known_bugs: list | None = None):
        self.path = Path(path)
        self._known = known_bugs            # None -> use module KNOWN_BUGS (PyTorch)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    self._seen.add(json.loads(line)["sig"])
                except Exception:
                    continue

    def classify(self, f: Finding) -> Verdict:
        sig = signature(f)
        if is_known(f, self._known):
            return Verdict("known", sig)
        if sig in self._seen:
            return Verdict("dup", sig)
        return Verdict("NEW", sig)

    def record(self, f: Finding, *, code: str = "", minimized: str = "",
               extra: dict | None = None) -> Verdict:
        v = self.classify(f)
        self._seen.add(v.sig)
        if v.status == "dup":
            return v            # already recorded once; don't bloat the ledger with repeats
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": v.status, "sig": v.sig,
            "op": f.op, "aspect": f.aspect, "verdict": f.verdict,
            "detail": f.detail, "test_id": f.test_id, "oracle_id": f.oracle_id,
            "contexts": list(f.contexts),
            "code": code, "minimized": minimized,
        }
        if extra:
            row.update(extra)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return v
