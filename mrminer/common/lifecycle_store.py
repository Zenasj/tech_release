"""Linear lifecycle event persistence with predecessor CAS."""

from __future__ import annotations

from pathlib import Path

from mrminer.common.schema import (
    LifecycleEvent,
    LifecycleEventType,
    OracleState,
    OracleStateProjection,
)
from mrminer.common.storage import JsonlStore, RecordConflict, _exclusive_lock


class LifecycleTransitionError(ValueError):
    pass


_ALLOWED_FROM: dict[LifecycleEventType, frozenset[OracleState]] = {
    LifecycleEventType.VALIDATED: frozenset({OracleState.DRAFT}),
    LifecycleEventType.ACTIVATED: frozenset({OracleState.VALIDATED}),
    LifecycleEventType.VALIDATION_FAILED: frozenset(
        {OracleState.DRAFT, OracleState.VALIDATED}
    ),
    LifecycleEventType.SUSPENDED_PROBE_DRIFT: frozenset({OracleState.ACTIVE}),
    LifecycleEventType.SUSPENDED_ENVIRONMENT: frozenset({OracleState.ACTIVE}),
    LifecycleEventType.SUSPENDED_PENDING_ARBITRATION: frozenset(
        {OracleState.ACTIVE}
    ),
    LifecycleEventType.SUSPENDED_EVIDENCE_WEAK: frozenset(
        {OracleState.DRAFT, OracleState.VALIDATED, OracleState.ACTIVE}
    ),
    LifecycleEventType.SUPERSEDED: frozenset(
        {
            OracleState.DRAFT,
            OracleState.VALIDATED,
            OracleState.ACTIVE,
            OracleState.SUSPENDED,
        }
    ),
    LifecycleEventType.REBASELINED: frozenset(
        {OracleState.VALIDATED, OracleState.ACTIVE, OracleState.SUSPENDED}
    ),
    LifecycleEventType.RETIRED_ENTITY_REMOVED: frozenset(
        {OracleState.DRAFT, OracleState.VALIDATED, OracleState.ACTIVE, OracleState.SUSPENDED}
    ),
    LifecycleEventType.RETIRED_EXPLICIT_CHANGE: frozenset(
        {OracleState.DRAFT, OracleState.VALIDATED, OracleState.ACTIVE, OracleState.SUSPENDED}
    ),
    LifecycleEventType.RESUMED: frozenset({OracleState.SUSPENDED}),
}


def fold_events(
    oracle_id: str,
    events: list[LifecycleEvent],
) -> OracleStateProjection:
    if not events:
        raise LifecycleTransitionError("cannot project an empty event stream")
    first = events[0]
    if first.oracle_id != oracle_id or first.event is not LifecycleEventType.CREATED:
        raise LifecycleTransitionError("event stream must start with created")
    if first.predecessor_event_id is not None:
        raise LifecycleTransitionError("created event cannot have a predecessor")

    state = OracleState.DRAFT
    reason: str | None = None
    head = first.event_id
    active_cells: tuple[str, ...] = ()
    last_pass_artifact: str | None = None
    successors: set[str] = set()

    for event in events[1:]:
        if event.oracle_id != oracle_id:
            raise LifecycleTransitionError("event stream contains another oracle")
        if event.predecessor_event_id != head:
            raise RecordConflict(
                f"predecessor CAS failed: expected {head}, "
                f"got {event.predecessor_event_id}"
            )
        allowed = _ALLOWED_FROM.get(event.event, frozenset())
        if state not in allowed:
            raise LifecycleTransitionError(
                f"event {event.event.value} is invalid from state {state.value}"
            )

        if event.event is LifecycleEventType.VALIDATED:
            state = OracleState.VALIDATED
        elif event.event is LifecycleEventType.ACTIVATED:
            state = OracleState.ACTIVE
            reason = None
            active_cells = tuple(
                sorted(set(event.payload.get("active_support_cells", [])))
            )
        elif event.event in {
            LifecycleEventType.SUSPENDED_PROBE_DRIFT,
            LifecycleEventType.SUSPENDED_ENVIRONMENT,
            LifecycleEventType.SUSPENDED_PENDING_ARBITRATION,
            LifecycleEventType.SUSPENDED_EVIDENCE_WEAK,
        }:
            state = OracleState.SUSPENDED
            reason = event.event.value.removeprefix("suspended_")
        elif event.event is LifecycleEventType.RESUMED:
            state = OracleState.ACTIVE
            reason = None
        else:
            state = OracleState.RETIRED
            reason = None
            active_cells = ()
        if event.successor_spec is not None:
            successors.add(event.successor_spec)
        if isinstance(event.payload.get("last_pass_artifact"), str):
            last_pass_artifact = event.payload["last_pass_artifact"]
        head = event.event_id

    return OracleStateProjection(
        oracle_id=oracle_id,
        state=state,
        reason=reason,
        head_event_id=head,
        active_support_cells=active_cells,
        last_pass_artifact=last_pass_artifact,
        successor_specs=tuple(sorted(successors)),
    )


class LifecycleEventStore:
    def __init__(self, path: Path):
        self.store = JsonlStore(path)
        self.cas_lock = path.with_suffix(path.suffix + ".cas.lock")

    def events(self) -> list[LifecycleEvent]:
        return [
            LifecycleEvent.from_record(record)
            for record in self.store.iter_records()
        ]

    def projection(self, oracle_id: str) -> OracleStateProjection | None:
        events = self.events()
        return fold_events(oracle_id, events) if events else None

    def append(self, event: LifecycleEvent) -> bool:
        with _exclusive_lock(self.cas_lock):
            events = self.events()
            for existing in events:
                if existing.command_id == event.command_id:
                    if existing.event_id == event.event_id:
                        return False
                    raise RecordConflict(
                        f"command {event.command_id!r} already has another event"
                    )
            current_head = events[-1].event_id if events else None
            if event.predecessor_event_id != current_head:
                raise RecordConflict(
                    f"predecessor CAS failed: expected {current_head}, "
                    f"got {event.predecessor_event_id}"
                )
            fold_events(event.oracle_id, [*events, event])
            return self.store.append(
                event.to_record(),
                idempotency_key=event.command_id,
            )
