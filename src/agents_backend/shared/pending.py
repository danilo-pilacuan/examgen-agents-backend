"""Store of the pending question to save (replaces the module-level global).

Before: `_PENDING_ITEM` was a global variable in `conversational/tools.py`,
which mixed concurrent conversations (all shared the same pending item) and
did not survive restarts.

Now: `PendingItemStore` in memory, partitioned by `thread_id` (like the
EventBus). The graph state (`ConversationalState.pending_item`) remains the
source of truth persisted by the checkpointer; this store is the read/write
bridge that `_call_tools_node` syncs with the state at the end of each tools
turn.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PendingItem:
    """Generated question awaiting human confirmation."""

    item: dict[str, Any]                 # GeneratedItem serializado
    subject_id: str
    subject_name: str | None = None


class PendingItemStore:
    """Thread-safe store of pending items per thread_id."""

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


# Singleton shared by exam-app and agents-backend (same instance in the same
# process; in multi-process deployment it would move to Redis).
pending_store = PendingItemStore()
