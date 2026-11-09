"""S1 triage: pure functions that turn raw PR data into pr_corpus_v1 fields.

No network access here — everything operates on already-fetched JSON,
so this module is easy to unit-test and to re-run offline.
"""

from __future__ import annotations

import re

from mrminer.s1_miner.pr_schema import (
    AddedTest,
    DefSpan,
    IssueLinkEvidence,
    IssueLinkKind,
    RevertEvidence,
    RevertEvidenceKind,
    TouchedFilePatch,
)

LINK_RE = re.compile(r"(?:fixes|closes|resolves|fix|close|resolve)\s+#(\d+)", re.I)
COMMENT_REF_RE = re.compile(r"(?<![\w/])#(\d+)\b")
INLINE_CODE_RE = re.compile(r"(`+)([^\r\n]*?)\1")
REVERT_TITLE_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)*(?:revert|back\s*out)\b", re.I)
PR_REF_RE = re.compile(r"#(\d+)")
BOT_REVERT_PR_RE = re.compile(
    r"\bthis\s+(?:pull\s+request|pr)\s+"
    r"(?:reverts?|backs?\s+out)\s+(?:pr\s*)?#(\d+)\b",
    re.I,
)
BODY_REVERT_PR_RE = re.compile(
    r"(?im)^\s*(?:this\s+)?(?:reverts?|backs?\s+out)\s+"
    r"(?:[\w.-]+/[\w.-]+)?#(\d+)\b"
)
# A revert-intent phrase and the rest of its line, e.g. "This reverts PRs
# .../pull/120338 and .../pull/119926". Line-bounded so unrelated PR links on
# other lines are never captured.
REVERT_INTENT_SPAN_RE = re.compile(
    r"(?im)(?:this\s+)?(?:reverts?|backs?\s+out)\b[^\n]*"
)
# Any single PR reference form inside such a span: #123, /pull/123, PR123,
# PR #123, PR 123. (Only used within an already revert-anchored span.)
PR_REF_ANY_RE = re.compile(r"(?:#|/pull/|\bPR\s*#?)(\d+)\b", re.I)
TEST_DEF_RE = re.compile(r"^\+\s*def\s+(test_\w+)", re.M)
TEST_DEF_LINE_RE = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)")
HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@")
BENCHMARK_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:ms|us|µs|ns|GB/s|it/s|%|x|×)\b")

DOC_FILE_RE = re.compile(r"(^docs?/|\.md$|\.rst$|^CODEOWNERS$)", re.I)
CI_FILE_RE = re.compile(r"(^\.github/|^\.ci/|^\.circleci/|^tools/ci|\.yml$|\.yaml$)", re.I)
TEST_FILE_RE = re.compile(r"(^|/)tests?/.*\.py$|(^|/)test_[^/]*\.py$")

OPT_KEYWORDS = re.compile(
    r"\b(optimi[sz]e|speed\s*up|speedup|perf|faster|fus(?:e|ion)|lowering|"
    r"vectori[sz]e|autotun|memory\s+planning|inline|fold|pattern)\b", re.I)
BUGFIX_KEYWORDS = re.compile(r"\b(fix(?:es|ed)?|bug|incorrect|wrong|crash|error|regression)\b", re.I)
FEATURE_KEYWORDS = re.compile(r"\b(add|support|enable|implement|introduce|new)\b", re.I)
REFACTOR_KEYWORDS = re.compile(r"\b(refactor|cleanup|clean\s+up|rename|move|reland|typing|lint)\b", re.I)


def subsystem_labels(files: list[dict], rules: dict[str, str]) -> list[str]:
    """Map changed file paths to subsystem labels via longest-prefix rules."""
    labels = set()
    for f in files:
        path = f.get("filename", "")
        best = ""
        for prefix, label in rules.items():
            if path.startswith(prefix) and len(prefix) > len(best):
                best, best_label = prefix, label
        if best:
            labels.add(best_label)
    return sorted(labels)


def classify_change_kind(title: str, body: str, files: list[dict]) -> str:
    if REVERT_TITLE_RE.search(title or ""):
        return "revert"
    paths = [f.get("filename", "") for f in files]
    if paths and all(DOC_FILE_RE.search(p) for p in paths):
        return "docs"
    if paths and all(CI_FILE_RE.search(p) or DOC_FILE_RE.search(p) for p in paths):
        return "infra"
    if paths and all(TEST_FILE_RE.search(p) for p in paths):
        return "test_only"
    text = f"{title}\n{body or ''}"
    has_linked_fix = bool(LINK_RE.search(text))
    if OPT_KEYWORDS.search(title or "") or BENCHMARK_RE.search(body or ""):
        return "optimization"
    if has_linked_fix or BUGFIX_KEYWORDS.search(title or ""):
        return "bugfix"
    if FEATURE_KEYWORDS.search(title or ""):
        return "feature"
    if REFACTOR_KEYWORDS.search(title or ""):
        return "refactor"
    return "unknown"


def linked_issue_numbers(title: str, body: str) -> list[int]:
    return sorted(
        {
            number
            for raw_number in LINK_RE.findall(f"{title}\n{body or ''}")
            if (number := int(raw_number)) > 0
        }
    )


def _mask_markdown_code(text: str) -> str:
    """Replace Markdown code with spaces while preserving source offsets."""
    masked = list(text)
    in_fence = False
    fence_char = ""
    fence_length = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip(" \t")
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        mask_line = in_fence or marker is not None
        if marker is not None:
            token = marker.group(1)
            if not in_fence:
                in_fence = True
                fence_char = token[0]
                fence_length = len(token)
            elif token[0] == fence_char and len(token) >= fence_length:
                in_fence = False
        if mask_line:
            for index in range(offset, offset + len(line)):
                if masked[index] not in "\r\n":
                    masked[index] = " "
        offset += len(line)

    prose = "".join(masked)
    for match in INLINE_CODE_RE.finditer(prose):
        for index in range(match.start(), match.end()):
            masked[index] = " "
    return "".join(masked)


def issue_link_evidence(
    text: str,
    *,
    source_id: str,
    locator_prefix: str,
    kind: IssueLinkKind,
) -> dict[int, tuple[IssueLinkEvidence, ...]]:
    pattern = LINK_RE if kind is IssueLinkKind.FIXES_KEYWORD else COMMENT_REF_RE
    search_text = (
        _mask_markdown_code(text or "")
        if kind is IssueLinkKind.COMMENT_MENTION
        else text or ""
    )
    discovered: dict[int, list[IssueLinkEvidence]] = {}
    for match in pattern.finditer(search_text):
        number = int(match.group(1))
        if number <= 0:
            continue
        discovered.setdefault(number, []).append(
            IssueLinkEvidence(
                kind=kind,
                source_id=source_id,
                locator=(
                    f"{locator_prefix}@chars {match.start()}-{match.end()}"
                ),
            )
        )
    return {
        number: tuple(
            sorted(
                set(evidence),
                key=lambda item: (
                    item.kind.value,
                    item.source_id,
                    item.locator,
                ),
            )
        )
        for number, evidence in discovered.items()
    }


def merge_issue_link_evidence(
    *collections: dict[int, tuple[IssueLinkEvidence, ...]],
) -> dict[int, tuple[IssueLinkEvidence, ...]]:
    merged: dict[int, set[IssueLinkEvidence]] = {}
    for collection in collections:
        for number, evidence in collection.items():
            merged.setdefault(number, set()).update(evidence)
    return {
        number: tuple(
            sorted(
                evidence,
                key=lambda item: (
                    item.kind.value,
                    item.source_id,
                    item.locator,
                ),
            )
        )
        for number, evidence in sorted(merged.items())
    }


def reverts_of(
    title: str,
    body: str,
    *,
    current_pr: int | None = None,
) -> list[int]:
    """All PR numbers this revert targets, best-effort from text, ordered so the
    first element is the canonical primary target. The primary is chosen by the
    original priority (quoted title #, title #, then a line-anchored body
    ``reverts #N``) so existing single-target results are unchanged; extra
    targets are appended from revert-intent spans that also carry ``/pull/N`` or
    ``PR N`` forms (e.g. "This reverts PRs .../pull/A and .../pull/B")."""
    if not REVERT_TITLE_RE.search(title or ""):
        return []
    ordered: list[int] = []

    def add(number: int) -> None:
        if number > 0 and number != current_pr and number not in ordered:
            ordered.append(number)

    # --- primary priority (unchanged from the original revert_of) ---
    quoted = re.search(r'"[^"]*#(\d+)[^"]*"', title or "")
    if quoted:
        add(int(quoted.group(1)))
    for match in PR_REF_RE.finditer(title or ""):
        add(int(match.group(1)))
    explicit_body = BODY_REVERT_PR_RE.search(body or "")
    if explicit_body:
        add(int(explicit_body.group(1)))

    # --- additional targets from revert-intent spans (URL / PR-word / multi) ---
    text = f"{title or ''}\n{body or ''}"
    for span in REVERT_INTENT_SPAN_RE.finditer(text):
        for match in PR_REF_ANY_RE.finditer(span.group(0)):
            add(int(match.group(1)))
    return ordered


def revert_of(
    title: str,
    body: str,
    *,
    current_pr: int | None = None,
) -> int | None:
    """The primary PR this revert targets, or None. Thin wrapper over
    :func:`reverts_of` preserving the original single-target contract."""
    targets = reverts_of(title, body, current_pr=current_pr)
    return targets[0] if targets else None


def extract_tests_added(files: list[dict]) -> list[dict]:
    """Added test functions, with the added-lines patch hunk as MVP code capture."""
    tests = []
    for f in files:
        path = f.get("filename", "")
        patch = f.get("patch") or ""
        if not TEST_FILE_RE.search(path) or not patch:
            continue
        names = TEST_DEF_RE.findall(patch)
        if not names:
            continue
        added_lines = "\n".join(
            line[1:] for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        for name in names:
            tests.append({"path": path, "test_name": name, "code": added_lines})
    return tests


def _patch_is_truncated(file_record: dict) -> bool:
    patch = file_record.get("patch")
    if patch is None:
        return True
    expected = file_record.get("changes")
    if not isinstance(expected, int):
        additions = file_record.get("additions")
        deletions = file_record.get("deletions")
        if isinstance(additions, int) and isinstance(deletions, int):
            expected = additions + deletions
    if not isinstance(expected, int):
        return True
    changed_lines = sum(
        1
        for line in patch.splitlines()
        if (
            (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
        )
    )
    return changed_lines < expected


def touched_file_patches(
    files: list[dict],
    *,
    subsystem_rules: dict[str, str] | None = None,
    compile_stack: set[str] | None = None,
) -> tuple[TouchedFilePatch, ...]:
    records = []
    for file_record in files:
        path = file_record.get("filename") or "<unknown>"
        if subsystem_rules is not None and compile_stack is not None:
            labels = subsystem_labels(
                [{"filename": path}],
                subsystem_rules,
            )
            if not set(labels) & compile_stack:
                continue
        patch = file_record.get("patch")
        records.append(
            TouchedFilePatch(
                path=path,
                patch=patch,
                truncated=_patch_is_truncated(file_record),
            )
        )
    return tuple(sorted(records, key=lambda record: record.path))


def extract_tests_added_v2(files: list[dict]) -> tuple[AddedTest, ...]:
    """Extract added test definitions using unified-diff new-file line numbers."""

    tests: list[AddedTest] = []
    seen: set[str] = set()
    for file_record in files:
        path = file_record.get("filename", "")
        patch = file_record.get("patch")
        if not TEST_FILE_RE.search(path) or not patch:
            continue

        added_code = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        new_line: int | None = None
        for line in patch.splitlines():
            hunk = HUNK_RE.match(line)
            if hunk:
                new_line = int(hunk.group(1))
                continue
            if new_line is None or line.startswith("\\"):
                continue
            if line.startswith("-") and not line.startswith("---"):
                continue

            content = line[1:] if line.startswith("+") else line[1:] if line.startswith(" ") else line
            if line.startswith("+") and not line.startswith("+++"):
                match = TEST_DEF_LINE_RE.match(content)
                if match:
                    name = match.group(1)
                    node_id = f"{path}::{name}@L{new_line}"
                    if node_id not in seen:
                        seen.add(node_id)
                        tests.append(
                            AddedTest(
                                path=path,
                                test_node_id=node_id,
                                test_name=name,
                                def_span=DefSpan(
                                    start_line=new_line,
                                    end_line=new_line,
                                    complete=False,
                                ),
                                added_code=added_code,
                            )
                        )
            if not line.startswith("-"):
                new_line += 1
    return tuple(tests)


def structured_revert_evidence(
    title: str,
    body: str,
    comments: list[dict],
    *,
    current_pr: int | None = None,
) -> RevertEvidence | None:
    targets = reverts_of(title, body, current_pr=current_pr)
    if targets:
        return RevertEvidence(
            kind=RevertEvidenceKind.TITLE_PATTERN,
            target_pr=targets[0],
            additional_target_prs=tuple(targets[1:]),
            source_id="title",
            quote=title,
        )

    for index, comment in enumerate(comments):
        text = comment.get("body") or ""
        author = (comment.get("user") or {}).get("login") or ""
        if not author.casefold().endswith(("bot", "[bot]")):
            continue
        target = explicit_bot_revert_target(text)
        if target is not None:
            return RevertEvidence(
                kind=RevertEvidenceKind.BOT_COMMENT,
                target_pr=target,
                source_id=f"comment:{index}",
                quote=text,
            )
    return None


def explicit_bot_revert_target(text: str) -> int | None:
    """Return a target only when the comment states this PR's direction."""

    match = BOT_REVERT_PR_RE.search(text)
    if match is None:
        return None
    target = int(match.group(1))
    return target if target > 0 else None


def info_richness(body: str, tests_added: list[dict]) -> dict:
    body = body or ""
    has_claim = len(body.strip()) >= 200 and bool(
        OPT_KEYWORDS.search(body) or BUGFIX_KEYWORDS.search(body) or FEATURE_KEYWORDS.search(body))
    has_benchmark = bool(BENCHMARK_RE.search(body))
    has_tests = bool(tests_added)
    score = round(0.4 * has_claim + 0.3 * has_benchmark + 0.3 * has_tests, 2)
    return {"has_claim": has_claim, "has_benchmark": has_benchmark,
            "has_tests": has_tests, "score": score}


def is_merged(detail: dict, labels: list[str], comments: list[str]) -> bool:
    """PyTorch merges via pytorchmergebot (GitHub 'merged' stays False), so also
    accept the 'Merged' label or a mergebot merge confirmation comment."""
    if detail.get("merged"):
        return True
    if any(l.lower() == "merged" for l in labels):
        return True
    return any("has been merged" in c or "merged successfully" in c for c in comments[:50])
