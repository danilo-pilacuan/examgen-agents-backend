"""Token and metrics instrumentation for the agentic world.

Elements of actions 5 and 6 of the thesis: measure tokens consumed (prompt +
completion) per LLM call and in aggregate, to report the real cost per exam /
per question (local vs. cloud comparison).

The `TokenTracker` is a LangChain callback attached to the LLMs
(`generator_llm` / `evaluator_llm` in `agents_backend/shared/llm.py`). Each
model call triggers `on_llm_end`, where `usage_metadata` (the LangChain
standard for ChatOpenAI) is extracted and accumulated.

Usage:
    from agents_backend.shared.metrics import token_tracker, get_usage_summary
    token_tracker.reset()
    ...  # run the pipeline / graph
    print(get_usage_summary())
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TokenTracker(BaseCallbackHandler):
    """LangChain callback that accumulates token usage per LLM call."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Resets the accumulators (call at the start of each run)."""
        with self._lock:
            self.llm_calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.calls: list[dict] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Fires when each LLM call finishes."""
        usage = self._extract_usage(response)
        if usage is None:
            return
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]
            self.total_tokens += usage["total_tokens"]
            self.calls.append(usage)

    @staticmethod
    def _extract_usage(response: Any) -> dict | None:
        """Extracts (prompt, completion, total) from a LangChain response.

        Supports:
          - `usage_metadata` from the message (LangChain v0.2+ standard).
          - `llm_output["token_usage"]` (OpenAI-style responses).
        """
        try:
            generations = getattr(response, "generations", None)
            if generations:
                first = generations[0][0]
                message = getattr(first, "message", None)
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    return {
                        "prompt_tokens": int(meta.get("input_tokens") or 0),
                        "completion_tokens": int(meta.get("output_tokens") or 0),
                        "total_tokens": int(meta.get("total_tokens") or 0),
                    }

            llm_output = getattr(response, "llm_output", None) or {}
            tu = llm_output.get("token_usage") or {}
            if tu:
                prompt = int(tu.get("prompt_tokens") or 0)
                completion = int(tu.get("completion_tokens") or 0)
                return {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": int(tu.get("total_tokens") or (prompt + completion)),
                }
        except Exception:  # noqa: BLE001
            return None
        return None

    def usage(self) -> dict:
        """Returns the current totals as a JSON-serializable dict."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "calls": list(self.calls),
            }

    def summary(self) -> str:
        """One-line readable summary (for logs)."""
        with self._lock:
            return (
                f"llm_calls={self.llm_calls} prompt={self.prompt_tokens} "
                f"completion={self.completion_tokens} total={self.total_tokens}"
            )


# Global singleton (thread-safe) reusable across the app.
token_tracker = TokenTracker()


def reset_token_usage() -> None:
    """Resets the global token counter."""
    token_tracker.reset()


def get_token_usage() -> dict:
    """Returns the accumulated token usage (JSON-serializable dict)."""
    return token_tracker.usage()


def get_usage_summary() -> str:
    """Returns a one-line readable summary."""
    return token_tracker.summary()
