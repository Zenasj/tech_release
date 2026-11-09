"""Crash-tolerant local stores with JSON facts and rebuildable SQLite indexes."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from mrminer.common.canonical import (
    HashProjectable,
    canonical_json,
    fs_safe,
    sha256_digest,
)
from mrminer.common.schema.interfaces import IndexReport


class StoreError(RuntimeError):
    pass


class RecordConflict(StoreError):
    pass


class StoreCorruption(StoreError):
    pass


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class ContentAddressedFileStore:
    def __init__(self, root: Path, *, id_field: str):
        self.root = root
        self.id_field = id_field
        self.lock_path = root / ".store.lock"

    def path_for(self, record_id: str) -> Path:
        return self.root / f"{fs_safe(record_id)}.json"

    def put(self, record: dict[str, Any]) -> Path:
        record_id = record.get(self.id_field)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record requires string field {self.id_field!r}")
        payload = (canonical_json(record) + "\n").encode("utf-8")
        path = self.path_for(record_id)
        self.root.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.lock_path):
            if path.exists():
                if path.read_bytes() != payload:
                    raise RecordConflict(
                        f"record {record_id} already exists with different bytes"
                    )
                return path
            with tempfile.NamedTemporaryFile(
                dir=self.root,
                prefix=".tmp-",
                delete=False,
            ) as stream:
                temp_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
        return path

    def put_projected(self, record: HashProjectable) -> Path:
        """Persist only the ID and semantic hash projection of a schema object."""
        record_id = getattr(record, self.id_field, None)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(
                f"projectable record requires string attribute {self.id_field!r}"
            )
        semantic_payload = dict(record.to_hash_payload())
        if self.id_field in semantic_payload:
            raise ValueError(
                f"hash projection must not contain ID field {self.id_field!r}"
            )
        return self.put({self.id_field: record_id, **semantic_payload})

    def get(self, record_id: str) -> dict[str, Any]:
        return json.loads(self.path_for(record_id).read_text())


class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    @staticmethod
    def _envelope(
        record: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "record_sha": sha256_digest(record),
            "record": record,
        }
        if idempotency_key is not None:
            envelope["idempotency_key"] = idempotency_key
        return envelope

    @staticmethod
    def _validate_envelope(envelope: dict[str, Any], *, line_no: int) -> None:
        record = envelope.get("record")
        if not isinstance(record, dict):
            raise StoreCorruption(f"line {line_no} has no object record")
        expected = sha256_digest(record)
        if envelope.get("record_sha") != expected:
            raise StoreCorruption(f"line {line_no} record checksum mismatch")

    def _recover_tail_locked(self) -> int:
        if not self.path.exists():
            return 0
        data = self.path.read_bytes()
        if not data:
            return 0
        lines = data.splitlines(keepends=True)
        truncate_at: int | None = None
        offset = 0
        for index, raw_line in enumerate(lines, 1):
            is_last = index == len(lines)
            if not raw_line.endswith(b"\n"):
                if is_last:
                    truncate_at = offset
                    break
                raise StoreCorruption(f"non-tail line {index} lacks newline")
            try:
                envelope = json.loads(raw_line)
                self._validate_envelope(envelope, line_no=index)
            except (json.JSONDecodeError, UnicodeDecodeError, StoreCorruption) as error:
                if is_last:
                    truncate_at = offset
                    break
                raise StoreCorruption(f"corrupt non-tail line {index}: {error}") from error
            offset += len(raw_line)
        if truncate_at is not None:
            with self.path.open("r+b") as stream:
                stream.truncate(truncate_at)
                stream.flush()
                os.fsync(stream.fileno())
            return len(data) - truncate_at
        return 0

    def recover_tail(self) -> int:
        with _exclusive_lock(self.lock_path):
            return self._recover_tail_locked()

    def iter_envelopes(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            for line_no, raw_line in enumerate(stream, 1):
                try:
                    envelope = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise StoreCorruption(f"invalid JSON at line {line_no}") from error
                self._validate_envelope(envelope, line_no=line_no)
                yield envelope

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for envelope in self.iter_envelopes():
            yield envelope["record"]

    def append(
        self,
        record: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> bool:
        envelope = self._envelope(record, idempotency_key)
        encoded = (canonical_json(envelope) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.lock_path):
            self._recover_tail_locked()
            if idempotency_key is not None and self.path.exists():
                for existing in self.iter_envelopes():
                    if existing.get("idempotency_key") != idempotency_key:
                        continue
                    if existing == envelope:
                        return False
                    raise RecordConflict(
                        f"idempotency key {idempotency_key!r} has different payload"
                    )
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        return True

    def initialize(
        self,
        records: Iterable[tuple[dict[str, Any], str | None]],
    ) -> int:
        """Stream a new store with one durability barrier.

        This is intentionally restricted to empty stores. Production builders
        can therefore avoid one fsync per fact without introducing replacement
        semantics for existing append-only data.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.lock_path):
            self._recover_tail_locked()
            if self.path.exists() and self.path.stat().st_size:
                raise RecordConflict(
                    f"cannot initialize non-empty store {self.path}"
                )
            seen: dict[str, str] = {}
            written = 0
            with self.path.open("ab") as stream:
                for record, idempotency_key in records:
                    envelope = self._envelope(record, idempotency_key)
                    if idempotency_key is not None:
                        record_sha = envelope["record_sha"]
                        previous = seen.get(idempotency_key)
                        if previous == record_sha:
                            continue
                        if previous is not None:
                            raise RecordConflict(
                                f"idempotency key {idempotency_key!r} "
                                "has different payload"
                            )
                        seen[idempotency_key] = record_sha
                    stream.write(
                        (canonical_json(envelope) + "\n").encode("utf-8")
                    )
                    written += 1
                stream.flush()
                os.fsync(stream.fileno())
            return written


def rebuild_index(
    stores: Sequence[Path],
    sqlite_path: Path,
    *,
    record_id_field: str,
) -> IndexReport:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path)
    total = 0
    try:
        connection.execute("DROP TABLE IF EXISTS records")
        connection.execute(
            """
            CREATE TABLE records (
                store_path TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                record_id TEXT NOT NULL,
                record_sha TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY (store_path, line_no),
                UNIQUE (store_path, record_id)
            )
            """
        )
        for store_path in stores:
            store = JsonlStore(store_path)
            for line_no, envelope in enumerate(store.iter_envelopes(), 1):
                record = envelope["record"]
                record_id = record.get(record_id_field)
                if not isinstance(record_id, str) or not record_id:
                    raise StoreCorruption(
                        f"{store_path}:{line_no} lacks {record_id_field}"
                    )
                connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                    (
                        str(store_path),
                        line_no,
                        record_id,
                        envelope["record_sha"],
                        canonical_json(record),
                    ),
                )
                total += 1
        connection.commit()

        checksum_ok = True
        for record_sha, record_json in connection.execute(
            "SELECT record_sha, record_json FROM records"
        ):
            if sha256_digest(json.loads(record_json)) != record_sha:
                checksum_ok = False
                break
    finally:
        connection.close()
    return IndexReport(
        store=",".join(str(path) for path in stores),
        records=total,
        rebuilt_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        checksum_ok=checksum_ok,
    )
