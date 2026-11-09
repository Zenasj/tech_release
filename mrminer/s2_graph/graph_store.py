"""JSONL-backed EntityGraph with a rebuildable SQLite query index."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from collections.abc import Iterable
from typing import Protocol

from mrminer.common.canonical import canonical_json, sha256_digest
from mrminer.common.schema import (
    ChangeLanding,
    EntitySpec,
    EvidenceSpec,
    IndexReport,
    OracleSpec,
    OracleState,
    OracleStateProjection,
    VersionArtifact,
)
from mrminer.common.storage import JsonlStore
from mrminer.s1_miner.pr_schema import PrCorpusV2
from mrminer.s2_graph.schema import (
    ClaimCandidate,
    GraphEdge,
    GraphEdgeKind,
    GraphNodeKind,
    IssueNode,
    NodeRef,
    OracleProvenanceAnnotation,
)


class IncomparableArtifacts(ValueError):
    pass


class AncestryUnavailable(RuntimeError):
    pass


class CommitAncestry(Protocol):
    def is_between(
        self,
        branch: str,
        low_exclusive: str,
        candidate: str,
        high_inclusive: str,
    ) -> bool | None:
        """Return None when ancestry for any referenced commit is unavailable."""


class StaticCommitAncestry:
    """Finite ancestry fixture/provider; commits are ordered oldest to newest."""

    def __init__(self, branches: dict[str, tuple[str, ...]]):
        self._positions = {
            branch: {commit: index for index, commit in enumerate(commits)}
            for branch, commits in branches.items()
        }
        if any(len(index) != len(branches[branch]) for branch, index in self._positions.items()):
            raise ValueError("commit ancestry sequences must not contain duplicates")

    def is_between(
        self,
        branch: str,
        low_exclusive: str,
        candidate: str,
        high_inclusive: str,
    ) -> bool | None:
        positions = self._positions.get(branch)
        if positions is None or any(
            commit not in positions
            for commit in (low_exclusive, candidate, high_inclusive)
        ):
            return None
        return (
            positions[low_exclusive]
            < positions[candidate]
            <= positions[high_inclusive]
        )


def _parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("graph timestamps must include a timezone")
    return parsed


class EntityGraph:
    def __init__(
        self,
        root: Path,
        *,
        ancestry: CommitAncestry | None = None,
    ):
        self.root = root
        self.sqlite_path = root / "index.sqlite"
        self.ancestry = ancestry
        self.last_rebuild_errors: tuple[str, ...] = ()
        self._stores = {
            "entities": JsonlStore(root / "entities.jsonl"),
            "prs": JsonlStore(root / "prs.jsonl"),
            "issues": JsonlStore(root / "issues.jsonl"),
            "claims": JsonlStore(root / "claims.jsonl"),
            "oracles": JsonlStore(root / "oracles.jsonl"),
            "oracle_annotations": JsonlStore(
                root / "oracle_annotations.jsonl"
            ),
            "artifacts": JsonlStore(root / "artifacts.jsonl"),
            "landings": JsonlStore(root / "landings.jsonl"),
            "projections": JsonlStore(root / "projections.jsonl"),
            "edges": JsonlStore(root / "edges.jsonl"),
        }

    def add_entity(self, entity: EntitySpec) -> bool:
        return self._stores["entities"].append(
            entity.to_record(),
            idempotency_key=sha256_digest(entity.to_record()),
        )

    def add_pr(self, pr: PrCorpusV2) -> bool:
        return self._stores["prs"].append(
            pr.to_record(),
            idempotency_key=sha256_digest(pr.to_record()),
        )

    def add_issue(self, issue: IssueNode) -> bool:
        return self._stores["issues"].append(
            issue.to_record(),
            idempotency_key=sha256_digest(issue.to_record()),
        )

    def add_claim(self, claim: ClaimCandidate, *, at: str) -> bool:
        appended = self._stores["claims"].append(
            claim.to_record(),
            idempotency_key=claim.claim_id,
        )
        pr_ref = NodeRef(
            kind=GraphNodeKind.PR,
            key=f"pr:{claim.source_pr}",
        )
        claim_ref = NodeRef(kind=GraphNodeKind.CLAIM, key=claim.claim_id)
        self.add_edge(
            GraphEdge.create(
                kind=GraphEdgeKind.HAS_CLAIM,
                source=pr_ref,
                target=claim_ref,
                evidence=claim.evidence,
                confidence=claim.confidence,
                context={},
                at=at,
            )
        )
        for entity_id in claim.target_entities:
            self.add_edge(
                GraphEdge.create(
                    kind=GraphEdgeKind.CLAIM_TARGETS,
                    source=claim_ref,
                    target=NodeRef(
                        kind=GraphNodeKind.ENTITY,
                        key=entity_id,
                    ),
                    evidence=claim.evidence,
                    confidence=claim.confidence,
                    context={},
                    at=at,
                )
            )
        return appended

    def add_oracle(self, oracle: OracleSpec) -> bool:
        semantic_record = {
            "schema_version": oracle.schema_version,
            "oracle_id": oracle.oracle_id,
            **oracle.to_hash_payload(),
        }
        appended = self._stores["oracles"].append(
            semantic_record,
            idempotency_key=oracle.oracle_id,
        )
        annotation = OracleProvenanceAnnotation.create(
            oracle_id=oracle.oracle_id,
            provenance=oracle.provenance,
        )
        self._stores["oracle_annotations"].append(
            annotation.to_record(),
            idempotency_key=annotation.annotation_id,
        )
        return appended

    def add_artifact(self, artifact: VersionArtifact) -> bool:
        return self._stores["artifacts"].append(
            artifact.to_record(),
            idempotency_key=artifact.artifact_id,
        )

    def add_landing(self, landing: ChangeLanding) -> bool:
        record = landing.to_record()
        return self._stores["landings"].append(
            record,
            idempotency_key=sha256_digest(record),
        )

    def add_projection(self, projection: OracleStateProjection) -> bool:
        return self._stores["projections"].append(
            projection.to_record(),
            idempotency_key=projection.head_event_id,
        )

    def add_edge(self, edge: GraphEdge) -> bool:
        return self._stores["edges"].append(
            edge.to_record(),
            idempotency_key=edge.edge_id,
        )

    def bulk_add_entities(self, entities: Iterable[EntitySpec]) -> int:
        return self._stores["entities"].initialize(
            (
                (entity.to_record(), sha256_digest(entity.to_record()))
                for entity in entities
            )
        )

    def bulk_add_prs(self, prs: Iterable[PrCorpusV2]) -> int:
        return self._stores["prs"].initialize(
            (
                (pr.to_record(), sha256_digest(pr.to_record()))
                for pr in prs
            )
        )

    def bulk_add_issues(self, issues: Iterable[IssueNode]) -> int:
        return self._stores["issues"].initialize(
            (
                (issue.to_record(), sha256_digest(issue.to_record()))
                for issue in issues
            )
        )

    def bulk_add_landings(self, landings: Iterable[ChangeLanding]) -> int:
        return self._stores["landings"].initialize(
            (
                (landing.to_record(), sha256_digest(landing.to_record()))
                for landing in landings
            )
        )

    def bulk_add_edges(self, edges: Iterable[GraphEdge]) -> int:
        return self._stores["edges"].initialize(
            ((edge.to_record(), edge.edge_id) for edge in edges)
        )

    def _fact_paths(self) -> tuple[Path, ...]:
        return tuple(store.path for store in self._stores.values())

    def _ensure_index(self) -> None:
        existing_facts = [path for path in self._fact_paths() if path.exists()]
        if not self.sqlite_path.exists() or any(
            path.stat().st_mtime_ns > self.sqlite_path.stat().st_mtime_ns
            for path in existing_facts
        ):
            self.rebuild_index()

    def rebuild_index(self) -> IndexReport:
        self.root.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        total = 0
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS nodes;
                DROP TABLE IF EXISTS edges;
                DROP TABLE IF EXISTS oracle_annotations;
                DROP TABLE IF EXISTS landings;
                DROP TABLE IF EXISTS projections;
                CREATE TABLE nodes (
                    kind TEXT NOT NULL,
                    node_key TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (kind, node_key)
                );
                CREATE TABLE edges (
                    edge_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX edge_source ON edges(source_kind, source_key);
                CREATE INDEX edge_target ON edges(target_kind, target_key);
                CREATE TABLE oracle_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    oracle_id TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX annotation_oracle
                    ON oracle_annotations(oracle_id);
                CREATE TABLE landings (
                    landing_key TEXT PRIMARY KEY,
                    pr INTEGER,
                    branch TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX landing_pr ON landings(pr);
                CREATE TABLE projections (
                    oracle_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                """
            )
            node_specs = (
                ("entities", GraphNodeKind.ENTITY, "entity_id"),
                ("prs", GraphNodeKind.PR, "pr_number"),
                ("issues", GraphNodeKind.ISSUE, "issue_number"),
                ("claims", GraphNodeKind.CLAIM, "claim_id"),
                ("oracles", GraphNodeKind.ORACLE, "oracle_id"),
                ("artifacts", GraphNodeKind.ARTIFACT, "artifact_id"),
            )
            for store_name, kind, key_field in node_specs:
                try:
                    for record in self._stores[store_name].iter_records():
                        raw_key = record[key_field]
                        key = (
                            f"pr:{raw_key}"
                            if kind is GraphNodeKind.PR
                            else f"issue:{raw_key}"
                            if kind is GraphNodeKind.ISSUE
                            else str(raw_key)
                        )
                        connection.execute(
                            "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?)",
                            (kind.value, key, canonical_json(record)),
                        )
                        total += 1
                except Exception as error:
                    errors.append(f"{store_name}: {error}")
                    continue

            for store_name in (
                "edges",
                "oracle_annotations",
                "landings",
                "projections",
            ):
                try:
                    for record in self._stores[store_name].iter_records():
                        if store_name == "edges":
                            edge = GraphEdge.from_record(record)
                            connection.execute(
                                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (
                                    edge.edge_id,
                                    edge.kind.value,
                                    edge.source.kind.value,
                                    edge.source.key,
                                    edge.target.kind.value,
                                    edge.target.key,
                                    edge.at,
                                    canonical_json(record),
                                ),
                            )
                        elif store_name == "oracle_annotations":
                            annotation = OracleProvenanceAnnotation.from_record(
                                record
                            )
                            connection.execute(
                                "INSERT OR IGNORE INTO oracle_annotations VALUES (?, ?, ?)",
                                (
                                    annotation.annotation_id,
                                    annotation.oracle_id,
                                    canonical_json(record),
                                ),
                            )
                        elif store_name == "landings":
                            landing = ChangeLanding.from_record(record)
                            connection.execute(
                                "INSERT OR IGNORE INTO landings VALUES (?, ?, ?, ?)",
                                (
                                    sha256_digest(record),
                                    landing.pr,
                                    landing.branch,
                                    canonical_json(record),
                                ),
                            )
                        else:
                            projection = OracleStateProjection.from_record(record)
                            connection.execute(
                                "INSERT OR REPLACE INTO projections VALUES (?, ?, ?)",
                                (
                                    projection.oracle_id,
                                    projection.state.value,
                                    canonical_json(record),
                                ),
                            )
                        total += 1
                except Exception as error:
                    errors.append(f"{store_name}: {error}")
                    continue

            dangling_checks = (
                (
                    "edges.source",
                    """
                    SELECT COUNT(*) FROM edges
                    LEFT JOIN nodes
                      ON nodes.kind = edges.source_kind
                     AND nodes.node_key = edges.source_key
                    WHERE nodes.node_key IS NULL
                    """,
                ),
                (
                    "edges.target",
                    """
                    SELECT COUNT(*) FROM edges
                    LEFT JOIN nodes
                      ON nodes.kind = edges.target_kind
                     AND nodes.node_key = edges.target_key
                    WHERE nodes.node_key IS NULL
                    """,
                ),
                (
                    "oracle_annotations",
                    """
                    SELECT COUNT(*) FROM oracle_annotations
                    LEFT JOIN nodes
                      ON nodes.kind = 'oracle'
                     AND nodes.node_key = oracle_annotations.oracle_id
                    WHERE nodes.node_key IS NULL
                    """,
                ),
                (
                    "projections",
                    """
                    SELECT COUNT(*) FROM projections
                    LEFT JOIN nodes
                      ON nodes.kind = 'oracle'
                     AND nodes.node_key = projections.oracle_id
                    WHERE nodes.node_key IS NULL
                    """,
                ),
                (
                    "landings",
                    """
                    SELECT COUNT(*) FROM landings
                    LEFT JOIN nodes
                      ON nodes.kind = 'pr'
                     AND nodes.node_key = 'pr:' || landings.pr
                    WHERE landings.pr IS NOT NULL
                      AND nodes.node_key IS NULL
                    """,
                ),
            )
            for label, query in dangling_checks:
                missing = connection.execute(query).fetchone()[0]
                if missing:
                    errors.append(f"{label}: {missing} dangling reference(s)")
            connection.commit()

        self.last_rebuild_errors = tuple(errors)
        return IndexReport(
            store=str(self.root),
            records=total,
            rebuilt_at=dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            checksum_ok=not errors,
        )

    def timeline(
        self,
        entity_id: str,
        until: str | None = None,
    ) -> list[GraphEdge]:
        self._ensure_index()
        until_time = _parse_time(until) if until is not None else None
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM edges
                WHERE (source_kind = 'entity' AND source_key = ?)
                   OR (target_kind = 'entity' AND target_key = ?)
                """,
                (entity_id, entity_id),
            ).fetchall()
        edges = [GraphEdge.from_record(json.loads(row[0])) for row in rows]
        if until_time is not None:
            edges = [edge for edge in edges if _parse_time(edge.at) <= until_time]
        return sorted(edges, key=lambda edge: (_parse_time(edge.at), edge.edge_id))

    def claims(
        self,
        entity_id: str,
        aspect: str | None = None,
    ) -> list[ClaimCandidate]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT nodes.record_json
                FROM edges
                JOIN nodes
                  ON nodes.kind = 'claim'
                 AND nodes.node_key = edges.source_key
                WHERE edges.kind = 'claim_targets'
                  AND edges.target_kind = 'entity'
                  AND edges.target_key = ?
                """,
                (entity_id,),
            ).fetchall()
        claims = [ClaimCandidate.from_record(json.loads(row[0])) for row in rows]
        if aspect is not None:
            claims = [claim for claim in claims if claim.aspect.value == aspect]
        return sorted(claims, key=lambda claim: claim.claim_id)

    def oracles(
        self,
        entity_id: str,
        state: str | OracleState | None = OracleState.ACTIVE,
    ) -> list[OracleSpec]:
        self._ensure_index()
        state_value = state.value if isinstance(state, OracleState) else state
        query = """
            SELECT DISTINCT nodes.record_json, annotations.record_json
            FROM edges
            JOIN nodes
              ON nodes.kind = 'oracle'
             AND nodes.node_key = edges.source_key
            JOIN oracle_annotations AS annotations
              ON annotations.annotation_id = (
                  SELECT MIN(a2.annotation_id)
                  FROM oracle_annotations AS a2
                  WHERE a2.oracle_id = nodes.node_key
              )
            LEFT JOIN projections
              ON projections.oracle_id = nodes.node_key
            WHERE edges.kind = 'bound_to'
              AND edges.target_kind = 'entity'
              AND edges.target_key = ?
        """
        params: list[str] = [entity_id]
        if state_value is not None:
            query += " AND projections.state = ?"
            params.append(state_value)
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(query, params).fetchall()
        records = []
        for semantic_json, annotation_json in rows:
            record = json.loads(semantic_json)
            annotation = OracleProvenanceAnnotation.from_record(
                json.loads(annotation_json)
            )
            record["provenance"] = annotation.provenance.to_record()
            records.append(OracleSpec.from_record(record))
        return sorted(records, key=lambda oracle: oracle.oracle_id)

    def _artifact(self, artifact_id: str) -> VersionArtifact:
        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                "SELECT record_json FROM nodes WHERE kind = 'artifact' AND node_key = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown artifact: {artifact_id}")
        return VersionArtifact.from_record(json.loads(row[0]))

    def prs_between(
        self,
        entity_id: str,
        art_lo: str,
        art_hi: str,
    ) -> list[PrCorpusV2]:
        self._ensure_index()
        low = self._artifact(art_lo)
        high = self._artifact(art_hi)
        branch = low.commit_interval.branch
        if branch != high.commit_interval.branch:
            raise IncomparableArtifacts("artifacts are on different branches")
        if self.ancestry is None:
            raise AncestryUnavailable("no commit ancestry provider configured")
        low_commit = low.commit_interval.to_inclusive
        high_commit = high.commit_interval.to_inclusive
        ordered = self.ancestry.is_between(
            branch,
            low_commit,
            high_commit,
            high_commit,
        )
        if ordered is None:
            raise AncestryUnavailable("artifact boundary ancestry is unavailable")
        if not ordered:
            raise IncomparableArtifacts("artifact bounds are not ordered")

        with sqlite3.connect(self.sqlite_path) as connection:
            direct = {
                int(row[0].removeprefix("pr:"))
                for row in connection.execute(
                    """
                    SELECT source_key FROM edges
                    WHERE kind = 'implementation_touches'
                      AND target_kind = 'entity'
                      AND target_key = ?
                    """,
                    (entity_id,),
                )
            }
            claimed = {
                int(row[0].removeprefix("pr:"))
                for row in connection.execute(
                    """
                    SELECT ownership.source_key
                    FROM edges AS targets
                    JOIN edges AS ownership
                      ON ownership.kind = 'has_claim'
                     AND ownership.target_key = targets.source_key
                    WHERE targets.kind = 'claim_targets'
                      AND targets.target_key = ?
                    """,
                    (entity_id,),
                )
            }
            candidate_prs = direct | claimed
            rows = connection.execute(
                "SELECT pr, record_json FROM landings WHERE branch = ?",
                (branch,),
            ).fetchall()

        selected: set[int] = set()
        unknown_ancestry = False
        for pr_number, record_json in rows:
            if pr_number not in candidate_prs:
                continue
            landing = ChangeLanding.from_record(json.loads(record_json))
            for commit in landing.landed_commits:
                in_range = self.ancestry.is_between(
                    branch,
                    low_commit,
                    commit,
                    high_commit,
                )
                if in_range is True:
                    selected.add(pr_number)
                    break
                if in_range is None:
                    unknown_ancestry = True
        if unknown_ancestry:
            raise AncestryUnavailable("landing commit ancestry is unavailable")

        if not selected:
            return []
        placeholders = ",".join("?" for _ in selected)
        with sqlite3.connect(self.sqlite_path) as connection:
            pr_rows = connection.execute(
                f"""
                SELECT node_key, record_json FROM nodes
                WHERE kind = 'pr' AND node_key IN ({placeholders})
                """,
                tuple(f"pr:{number}" for number in sorted(selected)),
            ).fetchall()
        records = [PrCorpusV2.from_record(json.loads(row[1])) for row in pr_rows]
        return sorted(records, key=lambda pr: pr.pr_number)

    def touch_evidence(
        self,
        pr: int,
        entity_id: str,
    ) -> list[EvidenceSpec]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM edges
                WHERE kind = 'implementation_touches'
                  AND source_kind = 'pr'
                  AND source_key = ?
                  AND target_kind = 'entity'
                  AND target_key = ?
                """,
                (f"pr:{pr}", entity_id),
            ).fetchall()
        evidence = [
            item
            for row in rows
            for item in GraphEdge.from_record(json.loads(row[0])).evidence
        ]
        return sorted(
            evidence,
            key=lambda item: (
                item.artifact_ref,
                item.locator,
                item.quote_hash,
            ),
        )

    def _node_record(
        self,
        kind: GraphNodeKind,
        key: str,
    ) -> dict:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            row = connection.execute(
                """
                SELECT record_json FROM nodes
                WHERE kind = ? AND node_key = ?
                """,
                (kind.value, key),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown {kind.value}: {key}")
        return json.loads(row[0])

    def entity(self, entity_id: str) -> EntitySpec:
        return EntitySpec.from_record(
            self._node_record(GraphNodeKind.ENTITY, entity_id)
        )

    def pr(self, pr_number: int) -> PrCorpusV2:
        return PrCorpusV2.from_record(
            self._node_record(GraphNodeKind.PR, f"pr:{pr_number}")
        )

    def issue(self, issue_number: int) -> IssueNode:
        return IssueNode.from_record(
            self._node_record(
                GraphNodeKind.ISSUE,
                f"issue:{issue_number}",
            )
        )

    def pr_numbers(self) -> list[int]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT node_key FROM nodes
                WHERE kind = 'pr'
                ORDER BY node_key
                """
            ).fetchall()
        return sorted(int(row[0].removeprefix("pr:")) for row in rows)

    def artifacts(self) -> list[VersionArtifact]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM nodes
                WHERE kind = 'artifact'
                ORDER BY node_key
                """
            ).fetchall()
        return [
            VersionArtifact.from_record(json.loads(row[0]))
            for row in rows
        ]

    def landings(
        self,
        *,
        pr_number: int | None = None,
        branch: str | None = None,
    ) -> list[ChangeLanding]:
        self._ensure_index()
        clauses: list[str] = []
        params: list[int | str] = []
        if pr_number is not None:
            clauses.append("pr = ?")
            params.append(pr_number)
        if branch is not None:
            clauses.append("branch = ?")
            params.append(branch)
        query = "SELECT record_json FROM landings"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY branch, pr, landing_key"
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            ChangeLanding.from_record(json.loads(row[0]))
            for row in rows
        ]

    def static_test_mentions(
        self,
        entity_id: str,
    ) -> list[tuple[PrCorpusV2, GraphEdge]]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            edge_rows = connection.execute(
                """
                SELECT edges.source_key, edges.record_json
                FROM edges
                WHERE edges.kind = 'statically_mentions'
                  AND edges.source_kind = 'pr'
                  AND edges.target_kind = 'entity'
                  AND edges.target_key = ?
                ORDER BY edges.source_key, edges.edge_id
                """,
                (entity_id,),
            ).fetchall()
            source_keys = sorted({row[0] for row in edge_rows})
            if not source_keys:
                return []
            placeholders = ",".join("?" for _ in source_keys)
            pr_rows = connection.execute(
                f"""
                SELECT node_key, record_json FROM nodes
                WHERE kind = 'pr' AND node_key IN ({placeholders})
                """,
                source_keys,
            ).fetchall()
        prs = {
            node_key: PrCorpusV2.from_record(json.loads(record))
            for node_key, record in pr_rows
        }
        return [
            (
                prs[source_key],
                GraphEdge.from_record(json.loads(edge_record)),
            )
            for source_key, edge_record in edge_rows
        ]

    def search_entities(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> list[EntitySpec]:
        if not query.strip():
            raise ValueError("entity search query cannot be empty")
        if not 1 <= limit <= 1000:
            raise ValueError("entity search limit must be between 1 and 1000")
        self._ensure_index()
        escaped = (
            query.strip()
            .lower()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        pattern = f"%{escaped}%"
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM nodes
                WHERE kind = 'entity'
                  AND (
                    lower(node_key) LIKE ? ESCAPE '\\'
                    OR lower(
                        json_extract(record_json, '$.display_name')
                    ) LIKE ? ESCAPE '\\'
                  )
                ORDER BY node_key
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [
            EntitySpec.from_record(json.loads(row[0])) for row in rows
        ]

    def related_prs(
        self,
        entity_id: str,
        *,
        edge_kinds: tuple[GraphEdgeKind, ...] = (
            GraphEdgeKind.IMPLEMENTATION_TOUCHES,
            GraphEdgeKind.STATICALLY_MENTIONS,
        ),
        limit: int = 100,
    ) -> list[PrCorpusV2]:
        if not edge_kinds:
            return []
        if not 1 <= limit <= 10000:
            raise ValueError("related PR limit must be between 1 and 10000")
        self._ensure_index()
        placeholders = ",".join("?" for _ in edge_kinds)
        params: list[str | int] = [
            *(kind.value for kind in edge_kinds),
            entity_id,
            limit,
        ]
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT nodes.record_json
                FROM edges
                JOIN nodes
                  ON nodes.kind = 'pr'
                 AND nodes.node_key = edges.source_key
                WHERE edges.kind IN ({placeholders})
                  AND edges.target_kind = 'entity'
                  AND edges.target_key = ?
                ORDER BY CAST(substr(nodes.node_key, 4) AS INTEGER)
                LIMIT ?
                """,
                params,
            ).fetchall()
        records = [
            PrCorpusV2.from_record(json.loads(row[0])) for row in rows
        ]
        return sorted(records, key=lambda pr: pr.pr_number)

    def related_pr_summaries(
        self,
        entity_id: str,
        *,
        edge_kinds: tuple[GraphEdgeKind, ...] = (
            GraphEdgeKind.IMPLEMENTATION_TOUCHES,
            GraphEdgeKind.STATICALLY_MENTIONS,
        ),
        limit: int = 100,
    ) -> list[dict]:
        if not edge_kinds:
            return []
        if not 1 <= limit <= 10000:
            raise ValueError("related PR limit must be between 1 and 10000")
        self._ensure_index()
        placeholders = ",".join("?" for _ in edge_kinds)
        params: list[str | int] = [
            *(kind.value for kind in edge_kinds),
            entity_id,
            limit,
        ]
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT
                    CAST(json_extract(
                        nodes.record_json, '$.pr_number'
                    ) AS INTEGER),
                    json_extract(nodes.record_json, '$.title'),
                    json_extract(nodes.record_json, '$.html_url'),
                    json_extract(nodes.record_json, '$.merged'),
                    json_extract(nodes.record_json, '$.merged_at'),
                    json_extract(nodes.record_json, '$.change_kind'),
                    json_extract(
                        nodes.record_json, '$.subsystem_labels'
                    ),
                    json_extract(
                        nodes.record_json, '$.linked_issues'
                    )
                FROM edges
                JOIN nodes
                  ON nodes.kind = 'pr'
                 AND nodes.node_key = edges.source_key
                WHERE edges.kind IN ({placeholders})
                  AND edges.target_kind = 'entity'
                  AND edges.target_key = ?
                ORDER BY CAST(substr(nodes.node_key, 4) AS INTEGER)
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "pr_number": row[0],
                "title": row[1],
                "html_url": row[2],
                "merged": bool(row[3]),
                "merged_at": row[4],
                "change_kind": row[5],
                "subsystem_labels": json.loads(row[6]),
                "linked_issues": [
                    {
                        "number": issue["number"],
                        "fetch_status": issue["fetch_status"],
                        "link_kinds": [
                            evidence["kind"]
                            for evidence in issue["link_evidence"]
                        ],
                    }
                    for issue in json.loads(row[7])
                ],
            }
            for row in rows
        ]

    def edges_for_node(
        self,
        node: NodeRef,
    ) -> list[GraphEdge]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            rows = connection.execute(
                """
                SELECT record_json FROM edges
                WHERE (source_kind = ? AND source_key = ?)
                   OR (target_kind = ? AND target_key = ?)
                ORDER BY at, edge_id
                """,
                (
                    node.kind.value,
                    node.key,
                    node.kind.value,
                    node.key,
                ),
            ).fetchall()
        return [
            GraphEdge.from_record(json.loads(row[0])) for row in rows
        ]

    def stats(self) -> dict[str, dict[str, int] | int]:
        self._ensure_index()
        with sqlite3.connect(self.sqlite_path) as connection:
            nodes = {
                kind: count
                for kind, count in connection.execute(
                    """
                    SELECT kind, COUNT(*) FROM nodes
                    GROUP BY kind ORDER BY kind
                    """
                )
            }
            edges = {
                kind: count
                for kind, count in connection.execute(
                    """
                    SELECT kind, COUNT(*) FROM edges
                    GROUP BY kind ORDER BY kind
                    """
                )
            }
            landings = connection.execute(
                "SELECT COUNT(*) FROM landings"
            ).fetchone()[0]
        return {"nodes": nodes, "edges": edges, "landings": landings}
