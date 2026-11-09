"""Parse a full GitHub pull-request diff using Git's native patch parser."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from pathlib import Path


DEFAULT_PARSE_TIMEOUT_SECONDS = 60.0
DEFAULT_GIT_TIMEOUT_SECONDS = 300.0
DEFAULT_GIT_CACHE_ROOT = Path("/tmp/mrminer_git_cache")
PATCH_BATCH_SIZE = 100
_BLOCK_START = re.compile(br"(?m)^diff --git ")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40,64}")


def _decode_path(raw: bytes) -> str:
    path = raw.decode("utf-8", errors="strict")
    if not path:
        raise ValueError("git numstat returned an empty path")
    return path


def _parse_count(raw: bytes) -> int:
    if raw == b"-":
        return 0
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"git numstat returned invalid count: {raw!r}") from error
    if value < 0:
        raise ValueError("git numstat returned a negative count")
    return value


def _parse_numstat(raw: bytes) -> list[tuple[int, int, str]]:
    records: list[tuple[int, int, str]] = []
    position = 0
    while position < len(raw):
        first_tab = raw.find(b"\t", position)
        second_tab = raw.find(b"\t", first_tab + 1)
        if first_tab < 0 or second_tab < 0:
            raise ValueError("git numstat output is malformed")
        additions = _parse_count(raw[position:first_tab])
        deletions = _parse_count(raw[first_tab + 1:second_tab])
        position = second_tab + 1

        if position < len(raw) and raw[position] == 0:
            position += 1
            old_end = raw.find(b"\0", position)
            new_end = raw.find(b"\0", old_end + 1)
            if old_end < 0 or new_end < 0:
                raise ValueError("git rename numstat output is malformed")
            path = _decode_path(raw[old_end + 1:new_end])
            position = new_end + 1
        else:
            path_end = raw.find(b"\0", position)
            if path_end < 0:
                raise ValueError("git numstat path is not NUL-terminated")
            path = _decode_path(raw[position:path_end])
            position = path_end + 1
        records.append((additions, deletions, path))
    return records


def _diff_blocks(raw: bytes) -> list[bytes]:
    starts = [match.start() for match in _BLOCK_START.finditer(raw)]
    if not starts:
        return []
    if raw[:starts[0]].strip():
        raise ValueError("full pull diff has data before the first file block")
    return [
        raw[start:end]
        for start, end in zip(starts, (*starts[1:], len(raw)))
    ]


def _status(block: bytes) -> str:
    if b"\nnew file mode " in block:
        return "added"
    if b"\ndeleted file mode " in block:
        return "removed"
    if b"\nrename from " in block and b"\nrename to " in block:
        return "renamed"
    if b"\ncopy from " in block and b"\ncopy to " in block:
        return "copied"
    return "modified"


def _merged_status(statuses: list[str]) -> str:
    if len(set(statuses)) == 1:
        return statuses[0]
    if statuses[0] == "added" and statuses[-1] != "removed":
        return "added"
    if statuses[-1] == "removed" and statuses[0] != "added":
        return "removed"
    return "modified"


def parse_full_pull_diff(
    raw: bytes,
    *,
    expected_count: int,
    timeout_seconds: float = DEFAULT_PARSE_TIMEOUT_SECONDS,
) -> list[dict]:
    if expected_count < 0:
        raise ValueError("expected_count cannot be negative")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        result = subprocess.run(
            ["git", "apply", "--numstat", "-z"],
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"git diff parser exceeded {timeout_seconds:g}s"
        ) from error
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"git diff parser failed with exit {result.returncode}: "
            f"{error[:1000]}"
        )

    stats = _parse_numstat(result.stdout)
    blocks = _diff_blocks(raw)
    if len(stats) != len(blocks):
        raise ValueError(
            "full pull diff block count mismatch: "
            f"numstat={len(stats)} "
            f"blocks={len(blocks)}"
        )

    grouped: dict[str, list[tuple[int, int, str, bytes]]] = {}
    for (additions, deletions, path), block in zip(stats, blocks):
        grouped.setdefault(path, []).append(
            (additions, deletions, _status(block), block)
        )
    if len(grouped) != expected_count:
        raise ValueError(
            "full pull diff file count mismatch: "
            f"expected={expected_count} unique_paths={len(grouped)} "
            f"blocks={len(blocks)}"
        )

    files = []
    for path, changes in grouped.items():
        additions = sum(change[0] for change in changes)
        deletions = sum(change[1] for change in changes)
        files.append(
            {
                "filename": path,
                "status": _merged_status(
                    [change[2] for change in changes]
                ),
                "additions": additions,
                "deletions": deletions,
                "changes": additions + deletions,
                "patch": b"".join(
                    change[3] for change in changes
                ).decode("utf-8", errors="replace"),
            }
        )
    return files


def _run_git(
    arguments: list[str],
    *,
    timeout_seconds: float,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            arguments,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"git command exceeded {timeout_seconds:g}s: "
            f"{' '.join(arguments[:4])}"
        ) from error
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(
            f"git command failed with exit {result.returncode}: "
            f"{message[:1000]}"
        )
    return result


def _prepare_cache(
    *,
    repository: str,
    cache_path: Path,
    remote_url: str,
    base_sha: str,
    head_sha: str,
    timeout_seconds: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        _run_git(
            ["git", "init", "--bare", str(cache_path)],
            timeout_seconds=timeout_seconds,
        )
    try:
        current_remote = subprocess.run(
            ["git", "-C", str(cache_path), "remote", "get-url", "origin"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"git remote inspection exceeded {timeout_seconds:g}s"
        ) from error
    if current_remote.returncode != 0:
        _run_git(
            [
                "git",
                "-C",
                str(cache_path),
                "remote",
                "add",
                "origin",
                remote_url,
            ],
            timeout_seconds=timeout_seconds,
        )
    elif (
        current_remote.stdout.decode("utf-8", errors="replace").strip()
        != remote_url
    ):
        raise ValueError(
            f"git cache remote mismatch for {repository}: {cache_path}"
        )
    _run_git(
        [
            "git",
            "-C",
            str(cache_path),
            "fetch",
            "--no-tags",
            "--filter=blob:none",
            "--depth=1",
            "origin",
            base_sha,
            head_sha,
        ],
        timeout_seconds=timeout_seconds,
    )


def _parse_name_status(raw: bytes) -> dict[str, str]:
    tokens = raw.split(b"\0")
    if tokens and not tokens[-1]:
        tokens.pop()
    if len(tokens) % 2:
        raise ValueError("git name-status output is malformed")
    status_names = {
        "A": "added",
        "D": "removed",
        "M": "modified",
        "T": "modified",
        "U": "modified",
        "X": "modified",
        "B": "modified",
    }
    records: dict[str, str] = {}
    for index in range(0, len(tokens), 2):
        raw_status = tokens[index].decode("ascii", errors="strict")
        path = _decode_path(tokens[index + 1])
        status = status_names.get(raw_status[:1])
        if status is None:
            raise ValueError(f"unsupported git file status: {raw_status}")
        if path in records:
            raise ValueError("git base/head diff returned a duplicate path")
        records[path] = status
    return records


def recover_git_commit_diff(
    *,
    repository: str,
    base_sha: str,
    head_sha: str,
    retain_patch: Callable[[str], bool],
    cache_path: Path | None = None,
    remote_url: str | None = None,
    timeout_seconds: float = DEFAULT_GIT_TIMEOUT_SECONDS,
) -> list[dict]:
    parts = repository.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"invalid GitHub repository: {repository!r}")
    if not _COMMIT_SHA.fullmatch(base_sha) or not _COMMIT_SHA.fullmatch(
        head_sha
    ):
        raise ValueError("git fallback requires full base/head commit SHAs")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    cache_path = cache_path or (
        DEFAULT_GIT_CACHE_ROOT / f"{parts[0]}__{parts[1]}.git"
    )
    remote_url = remote_url or f"https://github.com/{repository}.git"
    _prepare_cache(
        repository=repository,
        cache_path=cache_path,
        remote_url=remote_url,
        base_sha=base_sha,
        head_sha=head_sha,
        timeout_seconds=timeout_seconds,
    )
    prefix = ["git", "-C", str(cache_path)]
    revision = [base_sha, head_sha]
    statuses = _parse_name_status(
        _run_git(
            [
                *prefix,
                "diff-tree",
                "-r",
                "--no-commit-id",
                "--no-renames",
                "--name-status",
                "-z",
                *revision,
            ],
            timeout_seconds=timeout_seconds,
        ).stdout
    )
    stats = _parse_numstat(
        _run_git(
            [
                *prefix,
                "diff",
                "--no-renames",
                "--numstat",
                "-z",
                *revision,
            ],
            timeout_seconds=timeout_seconds,
        ).stdout
    )
    stat_paths = [path for _additions, _deletions, path in stats]
    if len(stat_paths) != len(set(stat_paths)):
        raise ValueError("git base/head numstat returned a duplicate path")
    if set(stat_paths) != set(statuses):
        raise ValueError("git base/head status and numstat paths disagree")

    retained_paths = [path for path in stat_paths if retain_patch(path)]
    patches: dict[str, str] = {}
    for offset in range(0, len(retained_paths), PATCH_BATCH_SIZE):
        batch = retained_paths[offset:offset + PATCH_BATCH_SIZE]
        raw_patch = _run_git(
            [
                *prefix,
                "diff",
                "--no-renames",
                "--binary",
                "--full-index",
                *revision,
                "--",
                *batch,
            ],
            timeout_seconds=timeout_seconds,
        ).stdout
        parsed = parse_full_pull_diff(
            raw_patch,
            expected_count=len(batch),
            timeout_seconds=timeout_seconds,
        )
        patches.update(
            (item["filename"], item["patch"])
            for item in parsed
        )
    if set(patches) != set(retained_paths):
        raise ValueError("git base/head patch paths are incomplete")

    return [
        {
            "filename": path,
            "status": statuses[path],
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
            "patch": patches.get(path),
        }
        for additions, deletions, path in stats
    ]
