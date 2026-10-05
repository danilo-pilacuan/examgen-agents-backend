"""Emisor de eventos tipados del agente (canal paralelo al markdown).

Las tools emiten eventos estructurados mientras el grafo conversacional
genera el markdown de respuesta. El markdown se conserva para el historial
y para OpenWebUI (legado); estos eventos alimentan los widgets del frontend
(etapa 4) vía SSE/WebSocket desde exam-app.

Mecanismo:
  - `EventBus` en memoria, particionado por `thread_id` (una conversación
    = un thread de LangGraph = un canal de eventos).
  - Las tools obtienen el bus con `get_event_bus()` (contextvar propagada
    por exam-app durante `astream_events`) y publican con `emit_event()`.
  - Si nadie está suscrito, el evento se descarta (publicar es best-effort:
    el markdown sigue siendo la fuente de verdad).

Contrato de eventos (schema versionado, ver docs/arquitectura-objetivo.md §5):
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
    """Tipos de evento que las tools emiten al frontend."""

    QUESTION_GENERATED = "question_generated"   # payload: GeneratedItem serializado
    QUESTIONS_LISTED = "questions_listed"       # payload: {"questions": [...], "total": n}
    ARTIFACT_READY = "artifact_ready"           # payload: {"kind": "pdf|csv|xlsx", "url", "filename"}
    MATERIAL_INDEXED = "material_indexed"       # payload: {"chunks": n, "filename", "subject_name"}
    QUESTION_SAVED = "question_saved"           # payload: {"question_id", "verified_by_human"}
    SUBJECT_REGISTERED = "subject_registered"   # payload: {"subject_id", "subject_name"}
    CANCELLED = "cancelled"                     # payload: {"message"} — cancelación vía WS


SCHEMA_VERSION = 1


@dataclass
class AgentEvent:
    """Evento tipado emitido por una tool del agente."""

    type: EventType
    thread_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    v: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serializa el evento al contrato versionado del frontend."""
        return {
            "v": self.v,
            "type": self.type.value,
            "thread_id": self.thread_id,
            **self.payload,
        }


class EventBus:
    """Bus de eventos en memoria, particionado por `thread_id`.

    - `publish()` es thread-safe y best-effort (nunca bloquea ni lanza si
      no hay suscriptores): las tools corren en el event loop o en hilos
      del grafo y no deben fallar por el canal de eventos.
    - `subscribe()` devuelve un iterador asíncrono (para SSE/WS en exam-app).
    - Los eventos se acumulan en un buffer acotado por suscriptor para que
      un consumidor lento no pierda los primeros eventos del turno.
    """

    def __init__(self, max_buffer: int = 256) -> None:
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._queues: dict[str, deque[AgentEvent]] = defaultdict(deque)
        self._waiters: dict[str, list[asyncio.Event]] = {}

    # ------------------------------------------------------------------
    # Publicación (lado de las tools)
    # ------------------------------------------------------------------
    def publish(self, event: AgentEvent) -> None:
        """Publica un evento para el thread_id (thread-safe, best-effort)."""
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
    # Suscripción (lado de exam-app: SSE / WS)
    # ------------------------------------------------------------------
    def subscribe(self, thread_id: str) -> "EventBusSubscription":
        """Crea una suscripción iterables para el thread dado."""
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
        """Marca el thread como cancelado: publica el evento y despierta."""
        self.publish(
            AgentEvent(
                type=EventType.CANCELLED,
                thread_id=thread_id,
                payload={"message": "cancelado por el usuario"},
            )
        )


class EventBusSubscription:
    """Suscripción asíncrona a los eventos de un thread (async iterator)."""

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
# Contexto: el bus activo durante la ejecución del grafo
# ----------------------------------------------------------------------
_event_bus: ContextVar[EventBus | None] = ContextVar("event_bus", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)


def set_event_context(bus: EventBus | None, thread_id: str | None) -> None:
    """Instala el bus y el thread_id para las tools del turno actual."""
    _event_bus.set(bus)
    _thread_id.set(thread_id)


def get_event_bus() -> EventBus | None:
    """Bus activo (None si la tool corre fuera de un turno con eventos)."""
    return _event_bus.get()


def get_current_thread_id() -> str | None:
    """thread_id del turno actual (None fuera de un turno)."""
    return _thread_id.get()


def emit_event(event_type: EventType, payload: dict[str, Any] | None = None) -> None:
    """Helper para las tools: publica si hay bus activo (best-effort)."""
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
