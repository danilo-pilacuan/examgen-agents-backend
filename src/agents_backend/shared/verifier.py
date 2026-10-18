"""AI verification of questions.

An LLM evaluates each generated question for AMBIGUITY, answer correctness
and relevance. The result feeds the fields:
  - `verified_by_ai` (bool)
  - `ai_review_notes` (text)
  - `ai_review_priority` (high | medium | low)

Questions with high priority (ambiguous / doubtful answer) are flagged for
priority human review (Felipe's requirement).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from agents_backend.shared.llm import evaluator_llm


class AIReviewResult(BaseModel):
    """Result of the AI verification of a question."""
    verified_by_ai: bool = Field(..., description="Is the question correct, clear and relevant?")
    is_ambiguous: bool = Field(default=False, description="Is the statement ambiguous?")
    answer_correct: bool = Field(default=True, description="Is the expected answer correct?")
    priority: str = Field(
        default="low",
        description="Priority of human review: 'high' | 'medium' | 'low'",
    )
    notes: str = Field(default="", description="Review notes for the professor")


def _validate_priority(v: str) -> str:
    if v not in ("high", "medium", "low"):
        return "medium"
    return v


def review_question(item: dict | "GeneratedItem") -> AIReviewResult:
    """Reviews a question (dict of `GeneratedItem`) with the evaluator LLM.

    Args:
        item: dict representation of the question (statement, options,
            expected_answer, solution_explanation, topic, etc.). Also
            accepts a `GeneratedItem` instance.

    The input is re-validated against `GeneratedItem` before sending it to the
    LLM: if the dict comes with out-of-domain values (invalid enums, malformed
    options), it is rejected without spending an LLM call.
    """
    # Input validation (accepts dict or an already-validated GeneratedItem).
    if isinstance(item, dict):
        try:
            from core.domain.schemas.exam import GeneratedItem

            item = GeneratedItem.model_validate(item)
        except Exception as e:  # noqa: BLE001
            # Invalid input: cannot be reviewed reliably.
            return AIReviewResult(
                verified_by_ai=False,
                is_ambiguous=True,
                answer_correct=False,
                priority="high",
                notes=f"Invalid question (does not comply with the schema): {str(e)[:200]}",
            )

    prompt = (
        "You are a rigorous reviewer of exam questions. Evaluate the following "
        "question and respond with the requested structure.\n\n"
        "Criteria:\n"
        "- Is the statement AMBIGUOUS or confusing?\n"
        "- Is the expected answer CORRECT and well justified?\n"
        "- Is the question RELEVANT to the topic?\n\n"
        "Question to review:\n"
        f"{item}"
    )
    try:
        structured = evaluator_llm.with_structured_output(AIReviewResult)
        result: AIReviewResult = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        # If the LLM fails, do not block: flag for human review.
        return AIReviewResult(
            verified_by_ai=False,
            is_ambiguous=True,
            answer_correct=False,
            priority="high",
            notes=f"Error in AI verification: {str(e)[:200]}",
        )

    # Consistency: if ambiguous or the answer is incorrect, not verified.
    if result.is_ambiguous or not result.answer_correct:
        result.verified_by_ai = False
        if result.priority == "low":
            result.priority = "medium"
    result.priority = _validate_priority(result.priority)
    return result


def apply_review_to_question(
    question_id: str,
    review: AIReviewResult,
    db=None,
) -> None:
    """Applies the AI verification result to a bank question.

    Uses the service layer for safe writes. If the question is already
    verified by a human, the AI verification fields are not touched (to avoid
    altering information of a protected question).
    """
    from core.domain.bank import service as bank

    q = bank.get_question(question_id, db=db)
    if q is None:
        return
    if q.verified_by_human:
        # Do not modify questions verified by a human.
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
