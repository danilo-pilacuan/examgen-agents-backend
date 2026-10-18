"""Event emitter for typed agent events (parallel channel to markdown).

Tools emit structured events while the conversational graph generates the
markdown response. The markdown is preserved for history and for OpenWebUI
(legacy); these events feed the frontend widgets (stage 4) via SSE/WebSocket
from exam-app.

Mechanism:
  - `EventBus` in memory, partitioned by `thread_id` (one conversation
    = one LangGraph thread = one event channel).
  - Tools obtain the bus with `get_event_bus()` (contextvar propagated by
    exam-app during `astream_events`) and publish with `emit_event()`.
  - If nobody is subscribed, the event is discarded (publishing is
    best-effort: the markdown remains the source of truth).

Event contract (versioned schema, see docs/arquitectura-objetivo.md §5):
  { "v": 1, "type": "question_generated", "thread_id": "...", ...payload }
"""
from __future__ import annotations

import asyncio
import threading
from collections import defaultdict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Event types that tools emit to the frontend."""

    QUESTION_GENERATED = "question_generated"   # payload: serialized GeneratedItem
    QUESTIONS_LISTED = "questions_listed"       # payload: {"questions": [...], "total": n}
    ARTIFACT_READY = "artifact_ready"           # payload: {"kind": "pdf|csv|xlsx", "url", "filename"}
    MATERIAL_INDEXED = "material_indexed"       # payload: {"chunks": n, "filename", "subject_name"}
    QUESTION_SAVED = "question_saved"           # payload: {"question_id", "verified_by_human"}
    SUBJECT_REGISTERED = "subject_registered"   # payload: {"subject_id", "subject_name"}
    CANCELLED = "cancelled"                     # payload: {"message"} — cancellation via WS


SCHEMA_VERSION = 1


@dataclass
class AgentEvent:
    """Typed event emitted by an agent tool."""

    type: EventType
    thread_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serializes the event to the versioned frontend contract."""
        return {
            "v": self.v,
            "type": self.type.value,
            "thread_id": self.thread_id,
            **self.payload,
        }


class EventBus:
    """In-memory event bus, partitioned by `thread_id`.

    - `publish()` is thread-safe and best-effort (never blocks nor raises if
      there are no subscribers): tools run on the event loop or in graph
      threads and must not fail because of the event channel.
    - `subscribe()` returns an async iterator (for SSE/WS in exam-app).
    - Events accumulate in a bounded per-subscriber buffer so a slow consumer
      does not lose the first events of the turn.
    """

    def __init__(self, max_buffer: int = 256) -> None:
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._queues: dict[str, deque[AgentEvent]] = defaultdict(deque)
        self._waiters: dict[str, list[asyncio.Event]] = {}

    # ------------------------------------------------------------------
    # Publish (tools side)
    # ------------------------------------------------------------------
    def publish(self, event: AgentEvent) -> None:
        """Publishes an event for the thread_id (thread-safe, best-effort)."""
        with self._lock:
            q = self._queues[event.thread_id]
            q.append(event)
            if len(q) > self._max_buffer:
                q.popleft()
            waiters = list(self._waiters.get(event.thread_id, ()))
        for ev in waiters:
            try:
                ev.set()
            except RuntimeError:
                pass  # loop cerrado: nadie consume, se descarta

    # ------------------------------------------------------------------
    # Subscription (exam-app side: SSE / WS)
    # ------------------------------------------------------------------
    def subscribe(self, thread_id: str) -> "EventBusSubscription":
        """Creates an async-iterable subscription for the given thread."""
        return EventBusSubscription(self, thread_id)

    def _register_waiter(self, thread_id: str, ev: asyncio.Event) -> None:
        with self._lock:
            self._waiters.setdefault(thread_id, []).append(ev)

    def _unregister_waiter(self, thread_id: str, ev: asyncio.Event) -> None:
        with self._lock:
            waiters = self._waiters.get(thread_id)
            if waiters and ev in waiters:
                waiters.remove(ev)
            if waiters is not None and not waiters:
                self._waiters.pop(thread_id, None)

    def _drain(self, thread_id: str) -> list[AgentEvent]:
        with self._lock:
            q = self._queues.get(thread_id)
            if not q:
                return []
            items = list(q)
            q.clear()
            return items

    def cancel_thread(self, thread_id: str) -> None:
        """Marks the thread as cancelled: publishes the event and wakes consumers."""
        self.publish(
            AgentEvent(
                type=EventType.CANCELLED,
                thread_id=thread_id,
                payload={"message": "cancelled by the user"},
            )
        )


class EventBusSubscription:
    """Async subscription to the events of a thread (async iterator)."""

    def __init__(self, bus: EventBus, thread_id: str) -> None:
        self._bus = bus
        self._thread_id = thread_id
        self._closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> AgentEvent:
        if self._closed:
            raise StopAsyncIteration
        ev = asyncio.Event()
        self._bus._register_waiter(self._thread_id, ev)
        try:
            while True:
                items = self._bus._drain(self._thread_id)
                if items:
                    return items[0]
                try:
                    await asyncio.wait_for(ev.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
        finally:
            self._bus._unregister_waiter(self._thread_id, ev)

    def close(self) -> None:
        self._closed = True


# ----------------------------------------------------------------------
# Context: the active bus during graph execution
# ----------------------------------------------------------------------
_event_bus: ContextVar[EventBus | None] = ContextVar("event_bus", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)


def set_event_context(bus: EventBus | None, thread_id: str | None) -> None:
    """Installs the bus and thread_id for the tools of the current turn."""
    _event_bus.set(bus)
    _thread_id.set(thread_id)


def get_event_bus() -> EventBus | None:
    """Active bus (None if the tool runs outside a turn with events)."""
    return _event_bus.get()


def get_current_thread_id() -> str | None:
    """thread_id of the current turn (None outside a turn)."""
    return _thread_id.get()


def emit_event(event_type: EventType, payload: dict[str, Any] | None = None) -> None:
    """Helper for tools: publishes if there is an active bus (best-effort)."""
    bus = get_event_bus()
    tid = get_current_thread_id()
    if bus is None or tid is None:
        return
    bus.publish(
        AgentEvent(
            type=event_type,
            thread_id=tid,
            payload=payload or {},
        )
    )
