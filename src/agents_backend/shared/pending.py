"""Almacén de la pregunta pendiente de guardar (sustituye al global de módulo).

Antes: `_PENDING_ITEM` era una variable global en `conversational/tools.py`,
lo que mezclaba conversaciones concurrentes (todas compartían el mismo
pendiente) y no sobrevivía a reinicios.

Ahora: `PendingItemStore` en memoria, particionado por `thread_id` (igual
que el EventBus). El estado del grafo (`ConversationalState.pending_item`)
sigue siendo la fuente de verdad persistida por el checkpointer; este store
es el puente de lectura/escritura que `_call_tools_node` sincroniza con el
estado al final de cada turno de tools.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingItem:
    """Pregunta generada a la espera de confirmación humana."""

    item: dict[str, Any]                 # GeneratedItem serializado
    subject_id: str
    subject_name: str | None = None


class PendingItemStore:
    """Store thread-safe de pendientes por thread_id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, PendingItem] = {}

    def set(self, thread_id: str, pending: PendingItem) -> None:
        with self._lock:
            self._items[thread_id] = pending

    def get(self, thread_id: str) -> PendingItem | None:
        with self._lock:
            return self._items.get(thread_id)

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._items.pop(thread_id, None)


# Singleton compartido por exam-app y agents-backend (misma instancia en
# el mismo proceso; en despliegue multi-proceso se movería a Redis).
pending_store = PendingItemStore()
