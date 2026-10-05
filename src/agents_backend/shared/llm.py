"""Factory de LLM: apunta a la infraestructura local (OpenAI-compatible).

El pipeline es agnóstico al proveedor: basta con cambiar `llm_base_url`
en `.env` para usar OpenAI real u otro endpoint compatible.
"""
from langchain_openai import ChatOpenAI

from core.config import settings


def get_llm(
    reasoning_effort: str = "low",
    temperature: float | None = None,
) -> ChatOpenAI:
    """Devuelve un ChatOpenAI apuntando a la infra local.

    `reasoning_effort` controla la deliberación del modelo. El servidor vLLM
    soporta 7 niveles: none | minimal | low | medium | high | xhigh | max.
    El Generador usa `none` (JSON en content); el Evaluador usa `none`
    con temperatura baja (juicio determinista y JSON válido).
    """
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=reasoning_effort,
    )


# Instancias con el nivel de razonamiento apropiado para cada nodo.
# NOTA: ambos usan `none` porque usan with_structured_output. Con
# reasoning_effort > none, vLLM pone el razonamiento en `reasoning` y deja
# `content` vacío, rompiendo el JSON estructurado.
generator_llm = get_llm(settings.llm_reasoning_generator)
evaluator_llm = get_llm(
    settings.llm_reasoning_evaluator,
    temperature=settings.llm_temperature_evaluator,
)
