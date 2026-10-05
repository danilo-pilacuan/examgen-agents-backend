"""Instrumentación de tokens y métricas del mundo agéntico.

Elemento de acción 5 y 6 de la tesis: medir tokens consumidos (prompt +
completion) por llamada LLM y en agregado, para poder reportar el costo
real por examen / por pregunta (comparativa local vs. nube).

El `TokenTracker` es un callback de LangChain que se adjunta a los LLMs
(`generator_llm` / `evaluator_llm` en `agents_backend/shared/llm.py`). Cada llamada
del modelo dispara `on_llm_end`, donde se extrae `usage_metadata`
(estándar de LangChain para ChatOpenAI) y se acumula.

Uso:
    from agents_backend.shared.metrics import token_tracker, get_usage_summary
    token_tracker.reset()
    ...  # ejecutar el pipeline / grafo
    print(get_usage_summary())
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class TokenTracker(BaseCallbackHandler):
    """Callback de LangChain que acumula el uso de tokens por llamada LLM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Reinicia los acumuladores (llamar al inicio de cada corrida)."""
        with self._lock:
            self.llm_calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.calls: list[dict] = []

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Se dispara al terminar cada llamada al LLM."""
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
        """Extrae (prompt, completion, total) de una respuesta LangChain.

        Soporta:
          - `usage_metadata` del mensaje (estándar LangChain v0.2+).
          - `llm_output["token_usage"]` (respuestas tipo OpenAI).
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
        """Devuelve los acumulados actuales como dict serializable a JSON."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "calls": list(self.calls),
            }

    def summary(self) -> str:
        """Resumen legible de una línea (para logs)."""
        with self._lock:
            return (
                f"llm_calls={self.llm_calls} prompt={self.prompt_tokens} "
                f"completion={self.completion_tokens} total={self.total_tokens}"
            )


# Singleton global (thread-safe) reutilizable en toda la app.
token_tracker = TokenTracker()


def reset_token_usage() -> None:
    """Reinicia el contador global de tokens."""
    token_tracker.reset()


def get_token_usage() -> dict:
    """Devuelve el uso de tokens acumulado (dict serializable)."""
    return token_tracker.usage()


def get_usage_summary() -> str:
    """Devuelve un resumen legible de una línea."""
    return token_tracker.summary()
