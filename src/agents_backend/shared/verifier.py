"""Verificación de preguntas por IA.

Un LLM evalúa cada pregunta generada en busca de AMBIGÜEDAD, corrección de la
respuesta y relevancia. El resultado alimenta los campos:
  - `verified_by_ai` (bool)
  - `ai_review_notes` (texto)
  - `ai_review_priority` (alta | media | baja)

Las preguntas con prioridad alta (ambiguas / respuesta dudosa) se marcan para
que un humano las revise con prioridad (requisito de Felipe).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from agents_backend.shared.llm import evaluator_llm


class AIReviewResult(BaseModel):
    """Resultado de la verificación por IA de una pregunta."""
    verified_by_ai: bool = Field(..., description="¿La pregunta es correcta, clara y relevante?")
    is_ambiguous: bool = Field(default=False, description="¿El enunciado es ambiguo?")
    answer_correct: bool = Field(default=True, description="¿La respuesta esperada es correcta?")
    priority: str = Field(
        default="baja",
        description="Prioridad de revisión humana: 'alta' | 'media' | 'baja'",
    )
    notes: str = Field(default="", description="Notas de revisión para el profesor")


def _validate_priority(v: str) -> str:
    if v not in ("alta", "media", "baja"):
        return "media"
    return v


def review_question(item: dict | "GeneratedItem") -> AIReviewResult:
    """Verifica una pregunta (dict de `GeneratedItem`) con el LLM evaluador.

    Args:
        item: representación dict de la pregunta (statement, options,
            expected_answer, solution_explanation, topic, etc.). También
            acepta una instancia `GeneratedItem`.

    El input se re-valida contra `GeneratedItem` antes de enviarlo al LLM:
    si el dict viene con valores fuera del dominio (enums inválidos,
    opciones mal formadas), se rechaza sin gastar una llamada al LLM.
    """
    # Validación del input (acepta dict o GeneratedItem ya validado).
    if isinstance(item, dict):
        try:
            from core.domain.schemas.exam import GeneratedItem

            item = GeneratedItem.model_validate(item)
        except Exception as e:  # noqa: BLE001
            # Input inválido: no se puede revisar con garantías.
            return AIReviewResult(
                verified_by_ai=False,
                is_ambiguous=True,
                answer_correct=False,
                priority="alta",
                notes=f"Pregunta inválida (no cumple el esquema): {str(e)[:200]}",
            )

    prompt = (
        "Eres un revisor riguroso de preguntas de examen. Evalúa la pregunta "
        "siguiente y responde con la estructura solicitada.\n\n"
        "Criterios:\n"
        "- ¿El enunciado es AMBIGUO o confuso?\n"
        "- ¿La respuesta esperada es CORRECTA y está bien justificada?\n"
        "- ¿La pregunta es RELEVANTE al tema?\n\n"
        "Pregunta a revisar:\n"
        f"{item}"
    )
    try:
        structured = evaluator_llm.with_structured_output(AIReviewResult)
        result: AIReviewResult = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        # Si falla el LLM, no bloqueamos: marcamos para revisión humana.
        return AIReviewResult(
            verified_by_ai=False,
            is_ambiguous=True,
            answer_correct=False,
            priority="alta",
            notes=f"Error en la verificación por IA: {str(e)[:200]}",
        )

    # Consistencia: si es ambigua o la respuesta es incorrecta, no verificada.
    if result.is_ambiguous or not result.answer_correct:
        result.verified_by_ai = False
        if result.priority == "baja":
            result.priority = "media"
    result.priority = _validate_priority(result.priority)
    return result


def apply_review_to_question(
    question_id: str,
    review: AIReviewResult,
    db=None,
) -> None:
    """Aplica el resultado de la verificación IA a una pregunta en el banco.

    Usa el service layer para la escritura segura. Si la pregunta ya está
    verificada por humano, no se tocan los campos de verificación IA (para no
    alterar información de una pregunta protegida).
    """
    from core.domain.bank import service as bank

    q = bank.get_question(question_id, db=db)
    if q is None:
        return
    if q.verified_by_human:
        # No modificar preguntas verificadas por humano.
        return

    bank.update_question(
        question_id,
        fields={
            "verified_by_ai": review.verified_by_ai,
            "ai_review_notes": review.notes,
            "ai_review_priority": review.priority,
        },
        db=db,
    )
