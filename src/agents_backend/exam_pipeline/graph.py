"""LangGraph graph that orchestrates the exam generation pipeline.

Flow:
    ingestor → planner → generator → evaluator → (conditional)
                approved → specs remaining → generator (loop)
                approved → no specs → assembler → compiler → END
                rejected → retry or abandon spec → generator

Conditionals:
    - `after_evaluation`: decides to approve/reject the current item.
    - `should_continue`: decides whether specs remain or we assemble.
    - `should_retry_compilation`: decides whether to retry compilation.
"""
from typing import Any

from langgraph.graph import StateGraph, END

from agents_backend.exam_pipeline.nodes import (
    assembler_node,
    compiler_node,
    evaluator_node,
    generator_node,
    ingestor_node,
    planner_node,
)
from agents_backend.exam_pipeline.state import ExamGenerationState
from core.config import settings
from core.domain.schemas.exam import ExamBlueprint, ItemSpec


# ---------------------------------------------------------------------------
# Conditionals
# ---------------------------------------------------------------------------
def after_evaluation(state: dict) -> str:
    """Route after evaluating an item: 'accepted' or 'rejected'."""
    evaluation = state.get("current_evaluation")
    if evaluation is not None and evaluation.approved:
        return "accepted"
    return "rejected"


def should_continue(state: dict) -> str:
    """Are there specs left to generate, or do we move to assembly?"""
    if state.get("pending_specs"):
        return "continue"
    return "assemble"


def should_retry_compilation(state: dict) -> str:
    """Do we retry compilation or finish?"""
    if state.get("status") == "completed":
        return "end"
    if state.get("compilation_attempts", 0) >= settings.max_compilation_attempts:
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Loop helper nodes (accept / reject)
# ---------------------------------------------------------------------------
def _accept_item(state: dict) -> dict:
    """Approves the current item: appends it to approved_items and resets retries."""
    item = state.get("current_item")
    eval_result = state.get("current_evaluation")
    spec = state.get("current_spec")
    if item is None:
        return {}

    update: dict[str, Any] = {
        "approved_items": [item],
        "item_retry_count": 0,
        "total_llm_calls": state.get("total_llm_calls", 0),
    }
    # Record previous rejections of the current spec (for metrics).
    return update


def _reject_item(state: dict) -> dict:
    """Records the rejection of the current item in the audit log."""
    spec = state.get("current_spec")
    eval_result = state.get("current_evaluation")
    retry_count = state.get("item_retry_count", 0) + 1

    reason = "low_quality"
    if eval_result is not None:
        if eval_result.is_duplicate:
            reason = "duplicate"
        elif not eval_result.difficulty_match:
            reason = "difficulty_mismatch"
        elif eval_result.rejection_reason and "error" in eval_result.rejection_reason:
            reason = "validation_error"

    entry = {
        "spec_id": spec["spec_id"] if isinstance(spec, dict) else spec.spec_id,
        "attempt_number": retry_count,
        "reason": reason,
        "quality_score": eval_result.quality_score if eval_result else None,
        "similarity_score": eval_result.similarity_score if eval_result else None,
    }

    # If the retry limit was exceeded, abandon this spec.
    abandon = retry_count >= settings.max_item_retries
    update: dict[str, Any] = {
        "item_retry_count": retry_count,
        "rejection_log": [entry],
    }
    if abandon:
        # Discard the current spec without approving it and move to the next.
        # We remove it from pending_specs so the generator won't pick it again.
        remaining = list(state.get("pending_specs", []))
        current = state.get("current_spec")
        if current is not None:
            cid = current["spec_id"] if isinstance(current, dict) else current.spec_id
            remaining = [s for s in remaining if (s["spec_id"] if isinstance(s, dict) else s.spec_id) != cid]
        update["pending_specs"] = remaining
        update["current_spec"] = None
        update["current_item"] = None
        update["current_evaluation"] = None
    return update


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ExamGenerationState)

    graph.add_node("ingestor", ingestor_node)
    graph.add_node("planner", planner_node)
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("accept", _accept_item)
    graph.add_node("reject", _reject_item)
    graph.add_node("assembler", assembler_node)
    graph.add_node("compiler", compiler_node)

    graph.set_entry_point("ingestor")
    graph.add_edge("ingestor", "planner")

    # Planner → generator if specs remain, otherwise straight to assembly.
    graph.add_conditional_edges(
        "planner",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # Generator → evaluator.
    graph.add_edge("generator", "evaluator")

    # Evaluator → approve or reject.
    graph.add_conditional_edges(
        "evaluator",
        after_evaluation,
        {"accepted": "accept", "rejected": "reject"},
    )

    # After approval: specs remaining? → generator or assembly.
    graph.add_conditional_edges(
        "accept",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # After rejection: if the spec was abandoned, move to the next;
    # if no specs remain, go to assembly.
    graph.add_conditional_edges(
        "reject",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # Assembly → compilation.
    graph.add_edge("assembler", "compiler")

    # Compilation: retry or finish.
    graph.add_conditional_edges(
        "compiler",
        should_retry_compilation,
        {"retry": "compiler", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Invocation helper
# ---------------------------------------------------------------------------
def run_exam_pipeline(
    subject_id: str,
    subject_name: str,
    template_id,
    raw_materials_paths: list[str],
    blueprint: ExamBlueprint,
    persist: bool = True,
) -> dict:
    """Entry point: builds the initial state and runs the graph.

    If `persist=True`, saves the approved items to the DB (with embeddings).
    """
    initial_state: dict[str, Any] = {
        "subject_id": subject_id,
        "subject_name": subject_name,
        "template_id": template_id,
        "raw_materials_paths": raw_materials_paths,
        "syllabus_summary": None,
        "indexed_chunks_count": 0,
        "blueprint": blueprint,
        "pending_specs": [],
        "current_spec": None,
        "current_item": None,
        "current_evaluation": None,
        "item_retry_count": 0,
        "max_item_retries": settings.max_item_retries,
        "approved_items": [],
        "rejection_log": [],
        "latex_source": None,
        "latex_sources": None,
        "compiled_exam": None,
        "pdf_paths": None,
        "zip_path": None,
        "compilation_attempts": 0,
        "max_compilation_attempts": settings.max_compilation_attempts,
        "compilation_log": [],
        "total_llm_calls": 0,
        "max_total_llm_calls": settings.max_total_llm_calls,
        "status": "in_progress",
        "error_message": None,
    }
    graph = build_graph()
    result = graph.invoke(initial_state)

    # Persist approved items to the DB (Phase 4)
    if persist and result.get("approved_items"):
        from agents_backend.shared.persistence import persist_generated_items

        persist_result = persist_generated_items(
            subject_id=subject_id,
            items=result["approved_items"],
            rejection_log=result.get("rejection_log", []),
            total_llm_calls=result.get("total_llm_calls", 0),
        )
        result["persisted"] = persist_result

    return result
