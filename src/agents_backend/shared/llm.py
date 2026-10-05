"""LLM factory: points to the local infrastructure (OpenAI-compatible).

The pipeline is provider-agnostic: just change `llm_base_url` in `.env` to
use real OpenAI or another compatible endpoint.
"""
from langchain_openai import ChatOpenAI

from core.config import settings


def get_llm(
    reasoning_effort: str = "low",
    temperature: float | None = None,
) -> ChatOpenAI:
    """Returns a ChatOpenAI pointing to the local infrastructure.

    `reasoning_effort` controls the model's deliberation. The vLLM server
    supports 7 levels: none | minimal | low | medium | high | xhigh | max.
    The Generator uses `none` (JSON in content); the Evaluator uses `none`
    with low temperature (deterministic judgment and valid JSON).
    """
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=reasoning_effort,
    )


# Instances with the reasoning level appropriate for each node.
# NOTE: both use `none` because they use with_structured_output. With
# reasoning_effort > none, vLLM puts the reasoning in `reasoning` and leaves
# `content` empty, breaking the structured JSON.
generator_llm = get_llm(settings.llm_reasoning_generator)
evaluator_llm = get_llm(
    settings.llm_reasoning_evaluator,
    temperature=settings.llm_temperature_evaluator,
)
