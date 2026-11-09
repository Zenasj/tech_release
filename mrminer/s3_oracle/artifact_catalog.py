"""Provision validated VersionArtifact records from immutable wheel files."""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from mrminer.common.canonical import content_id
from mrminer.common.schema import (
    BuildKind,
    CommitInterval,
    EnvironmentConstraints,
    VersionArtifact,
)
from mrminer.common.storage import ContentAddressedFileStore
from mrminer.s2_graph import EntityGraph


_COMMIT_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
_CPYTHON_TAG = re.compile(r"cp(?P<major>[0-9])(?P<minor>[0-9]{1,2})\Z")


class ArtifactImportStatus(str, Enum):
    READY = "ready"
    INVALID_MANIFEST = "invalid_manifest"
    WHEEL_MISSING = "wheel_missing"
    INVALID_WHEEL = "invalid_wheel"
    PACKAGE_MISMATCH = "package_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    COMMIT_ABSENT = "commit_absent"
    ANCESTRY_UNAVAILABLE = "ancestry_unavailable"
    INVALID_INTERVAL = "invalid_interval"
    BRANCH_MISMATCH = "branch_mismatch"


class CatalogGit(Protocol):
    def commit_exists(self, commit: str) -> bool | None:
        """Return exact local object presence."""

    def is_ancestor(
        self,
        branch: str,
        older: str,
        newer: str,
    ) -> bool | None:
        """Return Git commit ancestry or None when unavailable."""

    def resolve_branch_tip(self, branch: str) -> str | None:
        """Resolve the canonical local or origin branch tip."""


@dataclass(frozen=True)
class ArtifactImportCheck:
    name: str
    status: str
    detail: str

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WheelMetadata:
    package_version: str
    git_version: str
    cuda_version: str | None
    python_versions: tuple[str, ...]
    package_hash: str
    version_path: str

    def to_record(self) -> dict[str, Any]:
        return {
            "package_version": self.package_version,
            "git_version": self.git_version,
            "cuda_version": self.cuda_version,
            "python_versions": list(self.python_versions),
            "package_hash": self.package_hash,
            "version_path": self.version_path,
        }


@dataclass(frozen=True)
class ArtifactImportRequest:
    wheel_path: Path
    build_kind: BuildKind
    base_branch: str
    built_at: str
    from_exclusive: str
    to_inclusive: str
    python_constraint: str
    cuda_constraint: str | None

    @classmethod
    def from_record(
        cls,
        record: dict[str, Any],
        *,
        manifest_directory: Path,
    ) -> ArtifactImportRequest:
        expected = {
            "wheel",
            "build_kind",
            "base_branch",
            "built_at",
            "commit_interval",
            "environment_constraints",
        }
        if set(record) != expected:
            raise ValueError(
                "entry fields must be exactly "
                + ", ".join(sorted(expected))
            )
        wheel = record["wheel"]
        branch = record["base_branch"]
        built_at = record["built_at"]
        interval = record["commit_interval"]
        environment = record["environment_constraints"]
        if not isinstance(wheel, str) or not wheel:
            raise ValueError("wheel must be a non-empty path")
        if not isinstance(branch, str) or not branch:
            raise ValueError("base_branch must be a non-empty string")
        if not isinstance(built_at, str) or not built_at:
            raise ValueError("built_at must be a non-empty ISO-8601 timestamp")
        parsed_at = dt.datetime.fromisoformat(built_at.replace("Z", "+00:00"))
        if parsed_at.tzinfo is None:
            raise ValueError("built_at must include a timezone")
        if not isinstance(interval, dict) or set(interval) != {
            "from_exclusive",
            "to_inclusive",
        }:
            raise ValueError(
                "commit_interval requires from_exclusive and to_inclusive"
            )
        low = interval["from_exclusive"]
        high = interval["to_inclusive"]
        if (
            not isinstance(low, str)
            or _COMMIT_SHA.fullmatch(low) is None
            or not isinstance(high, str)
            or _COMMIT_SHA.fullmatch(high) is None
        ):
            raise ValueError("commit interval bounds must be full Git SHAs")
        if low == high:
            raise ValueError("commit interval bounds must differ")
        if not isinstance(environment, dict) or set(environment) != {
            "python",
            "cuda",
        }:
            raise ValueError(
                "environment_constraints requires python and cuda"
            )
        python_constraint = environment["python"]
        cuda_constraint = environment["cuda"]
        if not isinstance(python_constraint, str) or not python_constraint:
            raise ValueError("python constraint must be non-empty")
        if cuda_constraint is not None and (
            not isinstance(cuda_constraint, str) or not cuda_constraint
        ):
            raise ValueError("cuda constraint must be null or non-empty")
        try:
            SpecifierSet(python_constraint)
            if cuda_constraint is not None:
                SpecifierSet(cuda_constraint)
        except InvalidSpecifier as error:
            raise ValueError(f"invalid environment constraint: {error}") from error
        wheel_path = Path(wheel)
        if not wheel_path.is_absolute():
            wheel_path = manifest_directory / wheel_path
        return cls(
            wheel_path=wheel_path.resolve(),
            build_kind=BuildKind(record["build_kind"]),
            base_branch=branch,
            built_at=parsed_at.isoformat(timespec="seconds"),
            from_exclusive=low,
            to_inclusive=high,
            python_constraint=python_constraint,
            cuda_constraint=cuda_constraint,
        )


@dataclass(frozen=True)
class ArtifactImportResult:
    index: int
    wheel_path: str | None
    status: ArtifactImportStatus
    checks: tuple[ArtifactImportCheck, ...]
    metadata: WheelMetadata | None
    artifact: VersionArtifact | None
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "wheel_path": self.wheel_path,
            "status": self.status.value,
            "checks": [item.to_record() for item in self.checks],
            "metadata": (
                self.metadata.to_record()
                if self.metadata is not None
                else None
            ),
            "artifact": (
                self.artifact.to_record()
                if self.artifact is not None
                else None
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArtifactApplyReport:
    catalog_paths: tuple[str, ...]
    graph_artifacts_appended: int
    graph_index_records: int | None
    graph_index_checksum_ok: bool | None

    def to_record(self) -> dict[str, Any]:
        return {
            "catalog_paths": list(self.catalog_paths),
            "graph_artifacts_appended": self.graph_artifacts_appended,
            "graph_index_records": self.graph_index_records,
            "graph_index_checksum_ok": self.graph_index_checksum_ok,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _literal_assignments(source: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"torch/version.py is invalid Python: {error}") from error
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"__version__", "git_version", "cuda"}:
            continue
        try:
            values[target.id] = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
    return values


def inspect_torch_wheel(path: Path) -> WheelMetadata:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        distribution, filename_version, _build, tags = parse_wheel_filename(
            path.name
        )
    except InvalidWheelFilename as error:
        raise ValueError(f"invalid wheel filename: {error}") from error
    if distribution != "torch":
        raise ValueError(
            f"expected torch distribution, found {distribution!r}"
        )
    python_versions: set[str] = set()
    for tag in tags:
        match = _CPYTHON_TAG.fullmatch(tag.interpreter)
        if match is None:
            raise ValueError(
                f"unsupported non-CPython wheel tag: {tag.interpreter}"
            )
        python_versions.add(
            f"{int(match.group('major'))}.{int(match.group('minor'))}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name == "torch/version.py"
                or name.endswith("/torch/version.py")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    "wheel must contain exactly one torch/version.py"
                )
            version_path = candidates[0]
            source = archive.read(version_path).decode("utf-8")
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid wheel ZIP: {error}") from error
    values = _literal_assignments(source)
    package_version = values.get("__version__")
    git_version = values.get("git_version")
    cuda_version = values.get("cuda")
    if not isinstance(package_version, str) or not package_version:
        raise ValueError("torch/version.py lacks literal __version__")
    if (
        not isinstance(git_version, str)
        or _COMMIT_SHA.fullmatch(git_version) is None
    ):
        raise ValueError("torch/version.py lacks a full literal git_version")
    if cuda_version is not None and (
        not isinstance(cuda_version, str) or not cuda_version
    ):
        raise ValueError("torch/version.py cuda must be a string or null")
    try:
        if Version(package_version) != filename_version:
            raise ValueError(
                "wheel filename version disagrees with torch/version.py: "
                f"{filename_version} != {package_version}"
            )
    except InvalidVersion as error:
        raise ValueError(f"invalid torch package version: {error}") from error
    return WheelMetadata(
        package_version=package_version,
        git_version=git_version,
        cuda_version=cuda_version,
        python_versions=tuple(sorted(python_versions)),
        package_hash=_sha256_file(path),
        version_path=version_path,
    )


def _artifact_id_payload(
    *,
    request: ArtifactImportRequest,
    metadata: WheelMetadata,
) -> dict[str, Any]:
    return {
        "compiler": "pytorch",
        "build_kind": request.build_kind.value,
        "source_commit": metadata.git_version,
        "base_branch": request.base_branch,
        "package_version": metadata.package_version,
        "package_hash": metadata.package_hash,
        "built_at": request.built_at,
        "commit_interval": {
            "branch": request.base_branch,
            "from_exclusive": request.from_exclusive,
            "to_inclusive": request.to_inclusive,
        },
        "environment_constraints": {
            "python": request.python_constraint,
            "cuda": request.cuda_constraint,
        },
    }


def _failure(
    index: int,
    status: ArtifactImportStatus,
    reason: str,
    *,
    path: Path | None,
    checks: list[ArtifactImportCheck],
    metadata: WheelMetadata | None = None,
) -> ArtifactImportResult:
    checks.append(ArtifactImportCheck("result", "failed", reason))
    return ArtifactImportResult(
        index=index,
        wheel_path=str(path) if path is not None else None,
        status=status,
        checks=tuple(checks),
        metadata=metadata,
        artifact=None,
        reason=reason,
    )


def inspect_artifact_entry(
    index: int,
    raw: Any,
    *,
    manifest_directory: Path,
    git: CatalogGit,
) -> ArtifactImportResult:
    checks: list[ArtifactImportCheck] = []
    if not isinstance(raw, dict):
        return _failure(
            index,
            ArtifactImportStatus.INVALID_MANIFEST,
            "artifact entry must be a JSON object",
            path=None,
            checks=checks,
        )
    try:
        request = ArtifactImportRequest.from_record(
            raw,
            manifest_directory=manifest_directory,
        )
    except (ValueError, TypeError) as error:
        return _failure(
            index,
            ArtifactImportStatus.INVALID_MANIFEST,
            str(error),
            path=None,
            checks=checks,
        )
    checks.append(
        ArtifactImportCheck("manifest", "passed", "entry contract is valid")
    )
    if not request.wheel_path.is_file():
        return _failure(
            index,
            ArtifactImportStatus.WHEEL_MISSING,
            "wheel path does not exist as a regular file",
            path=request.wheel_path,
            checks=checks,
        )
    try:
        metadata = inspect_torch_wheel(request.wheel_path)
    except ValueError as error:
        return _failure(
            index,
            ArtifactImportStatus.INVALID_WHEEL,
            str(error),
            path=request.wheel_path,
            checks=checks,
        )
    checks.append(
        ArtifactImportCheck(
            "wheel",
            "passed",
            "package hash and literal torch version metadata computed",
        )
    )
    if metadata.git_version != request.to_inclusive:
        return _failure(
            index,
            ArtifactImportStatus.PACKAGE_MISMATCH,
            "wheel git_version must equal commit_interval.to_inclusive",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    python_spec = SpecifierSet(request.python_constraint)
    unsupported_python = [
        version
        for version in metadata.python_versions
        if Version(version) not in python_spec
    ]
    if unsupported_python:
        return _failure(
            index,
            ArtifactImportStatus.ENVIRONMENT_MISMATCH,
            (
                "wheel Python tags violate the declared constraint: "
                + ", ".join(unsupported_python)
            ),
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    if metadata.cuda_version is None:
        cuda_matches = request.cuda_constraint is None
    elif request.cuda_constraint is None:
        cuda_matches = False
    else:
        cuda_matches = (
            Version(metadata.cuda_version)
            in SpecifierSet(request.cuda_constraint)
        )
    if not cuda_matches:
        return _failure(
            index,
            ArtifactImportStatus.ENVIRONMENT_MISMATCH,
            "wheel CUDA build version violates the declared constraint",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    checks.append(
        ArtifactImportCheck(
            "environment",
            "passed",
            "wheel Python/CUDA metadata satisfies declared constraints",
        )
    )
    commits = (
        request.from_exclusive,
        request.to_inclusive,
        metadata.git_version,
    )
    presence = {commit: git.commit_exists(commit) for commit in commits}
    if any(value is None for value in presence.values()):
        return _failure(
            index,
            ArtifactImportStatus.ANCESTRY_UNAVAILABLE,
            "one or more Git commit presence queries were unavailable",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    absent = [commit for commit, value in presence.items() if value is False]
    if absent:
        return _failure(
            index,
            ArtifactImportStatus.COMMIT_ABSENT,
            "Git object database lacks: " + ", ".join(sorted(absent)),
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    ordered = git.is_ancestor(
        request.base_branch,
        request.from_exclusive,
        request.to_inclusive,
    )
    if ordered is None:
        return _failure(
            index,
            ArtifactImportStatus.ANCESTRY_UNAVAILABLE,
            "commit interval ancestry query was unavailable",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    if not ordered:
        return _failure(
            index,
            ArtifactImportStatus.INVALID_INTERVAL,
            "from_exclusive is not an ancestor of to_inclusive",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    branch_tip = git.resolve_branch_tip(request.base_branch)
    if branch_tip is None:
        return _failure(
            index,
            ArtifactImportStatus.ANCESTRY_UNAVAILABLE,
            f"base branch {request.base_branch!r} could not be resolved",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    on_branch = git.is_ancestor(
        request.base_branch,
        request.to_inclusive,
        branch_tip,
    )
    if on_branch is None:
        return _failure(
            index,
            ArtifactImportStatus.ANCESTRY_UNAVAILABLE,
            "base branch membership query was unavailable",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    if not on_branch:
        return _failure(
            index,
            ArtifactImportStatus.BRANCH_MISMATCH,
            "to_inclusive is not an ancestor of the declared base branch",
            path=request.wheel_path,
            checks=checks,
            metadata=metadata,
        )
    checks.append(
        ArtifactImportCheck(
            "git_ancestry",
            "passed",
            "interval bounds exist, are ordered, and belong to base_branch",
        )
    )
    payload = _artifact_id_payload(request=request, metadata=metadata)
    artifact = VersionArtifact(
        artifact_id=content_id("art", payload),
        compiler=payload["compiler"],
        build_kind=request.build_kind,
        source_commit=metadata.git_version,
        base_branch=request.base_branch,
        package_version=metadata.package_version,
        package_hash=metadata.package_hash,
        built_at=request.built_at,
        commit_interval=CommitInterval(
            branch=request.base_branch,
            from_exclusive=request.from_exclusive,
            to_inclusive=request.to_inclusive,
        ),
        environment_constraints=EnvironmentConstraints(
            python=request.python_constraint,
            cuda=request.cuda_constraint,
        ),
    )
    checks.append(
        ArtifactImportCheck(
            "result",
            "passed",
            "VersionArtifact is ready for explicit apply",
        )
    )
    return ArtifactImportResult(
        index=index,
        wheel_path=str(request.wheel_path),
        status=ArtifactImportStatus.READY,
        checks=tuple(checks),
        metadata=metadata,
        artifact=artifact,
        reason="wheel and Git facts satisfy the VersionArtifact contract",
    )


def inspect_artifact_manifest(
    manifest_path: Path,
    *,
    git: CatalogGit,
) -> tuple[ArtifactImportResult, ...]:
    payload = json.loads(manifest_path.read_bytes())
    if not isinstance(payload, dict) or set(payload) != {
        "manifest_version",
        "artifacts",
    }:
        raise ValueError(
            "manifest requires exactly manifest_version and artifacts"
        )
    if payload["manifest_version"] != "version_artifact_import_v1":
        raise ValueError("unsupported version artifact import manifest")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest artifacts must be a non-empty list")
    return tuple(
        inspect_artifact_entry(
            index,
            raw,
            manifest_directory=manifest_path.resolve().parent,
            git=git,
        )
        for index, raw in enumerate(artifacts)
    )


def apply_artifact_results(
    results: tuple[ArtifactImportResult, ...],
    *,
    catalog_root: Path,
    graph: EntityGraph | None,
) -> ArtifactApplyReport:
    rejected = [
        result
        for result in results
        if result.status is not ArtifactImportStatus.READY
        or result.artifact is None
    ]
    if rejected:
        raise ValueError(
            "refusing partial artifact apply; rejected entry indexes: "
            + ", ".join(str(item.index) for item in rejected)
        )
    store = ContentAddressedFileStore(
        catalog_root,
        id_field="artifact_id",
    )
    paths = []
    graph_appended = 0
    for result in results:
        assert result.artifact is not None
        paths.append(str(store.put(result.artifact.to_record())))
        if graph is not None:
            graph_appended += int(graph.add_artifact(result.artifact))
    index_records = None
    index_checksum_ok = None
    if graph is not None:
        index_report = graph.rebuild_index()
        if not index_report.checksum_ok:
            raise RuntimeError(
                f"graph rebuild failed: {graph.last_rebuild_errors}"
            )
        index_records = index_report.records
        index_checksum_ok = index_report.checksum_ok
    return ArtifactApplyReport(
        catalog_paths=tuple(sorted(paths)),
        graph_artifacts_appended=graph_appended,
        graph_index_records=index_records,
        graph_index_checksum_ok=index_checksum_ok,
    )
