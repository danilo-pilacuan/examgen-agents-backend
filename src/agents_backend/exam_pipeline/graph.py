"""Armado del grafo LangGraph que orquesta el pipeline de generación de exámenes.

Flujo:
    ingestor → planner → generator → evaluator → (condicional)
                aprobado → si quedan specs → generator (loop)
                aprobado → sin specs → assembler → compiler → END
                rechazado → reintenta o abandona spec → generator

Condicionales:
    - `after_evaluation`: decide aprobar/rechazar el ítem actual.
    - `should_continue`: decide si quedan specs por generar o se ensambla.
    - `should_retry_compilation`: decide si se reintenta la compilación.
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
# Condicionales
# ---------------------------------------------------------------------------
def after_evaluation(state: dict) -> str:
    """Ruta tras evaluar un ítem: 'accepted' o 'rejected'."""
    evaluation = state.get("current_evaluation")
    if evaluation is not None and evaluation.approved:
        return "accepted"
    return "rejected"


def should_continue(state: dict) -> str:
    """¿Quedan specs por generar o pasamos al ensamblado?"""
    if state.get("pending_specs"):
        return "continue"
    return "assemble"


def should_retry_compilation(state: dict) -> str:
    """¿Reintentamos la compilación o terminamos?"""
    if state.get("status") == "completed":
        return "end"
    if state.get("compilation_attempts", 0) >= settings.max_compilation_attempts:
        return "end"
    return "retry"


# ---------------------------------------------------------------------------
# Nodos de ayuda para el bucle (aprobación / rechazo)
# ---------------------------------------------------------------------------
def _accept_item(state: dict) -> dict:
    """Aprueba el ítem actual: lo agrega a approved_items y resetea reintentos."""
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
    # Registrar rechazos previos del spec actual si los hubo (para métricas).
    return update


def _reject_item(state: dict) -> dict:
    """Registra el rechazo del ítem actual en el log de auditoría."""
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

    # Si se superó el límite de reintentos, abandonamos este spec.
    abandon = retry_count >= settings.max_item_retries
    update: dict[str, Any] = {
        "item_retry_count": retry_count,
        "rejection_log": [entry],
    }
    if abandon:
        # Descartar el spec actual sin aprobarlo y seguir con el siguiente.
        # Lo removemos de pending_specs para que generator no lo retome.
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
# Construcción del grafo
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

    # Planificador → generador si hay specs, si no directamente al ensamblado.
    graph.add_conditional_edges(
        "planner",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # Generador → evaluador.
    graph.add_edge("generator", "evaluator")

    # Evaluador → aprobar o rechazar.
    graph.add_conditional_edges(
        "evaluator",
        after_evaluation,
        {"accepted": "accept", "rejected": "reject"},
    )

    # Tras aprobar: ¿quedan specs? → generador o ensamblado.
    graph.add_conditional_edges(
        "accept",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # Tras rechazar: si abandonamos el spec, seguimos con el siguiente;
    # si no quedan specs, pasamos al ensamblado.
    graph.add_conditional_edges(
        "reject",
        should_continue,
        {"continue": "generator", "assemble": "assembler"},
    )

    # Ensamblado → compilación.
    graph.add_edge("assembler", "compiler")

    # Compilación: reintentar o terminar.
    graph.add_conditional_edges(
        "compiler",
        should_retry_compilation,
        {"retry": "compiler", "end": END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Helper de invocación
# ---------------------------------------------------------------------------
def run_exam_pipeline(
    subject_id: str,
    subject_name: str,
    template_id,
    raw_materials_paths: list[str],
    blueprint: ExamBlueprint,
    persist: bool = True,
) -> dict:
    """Punto de entrada: construye el estado inicial y ejecuta el grafo.

    Si `persist=True`, guarda los ítems aprobados en la BD (con embeddings).
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

    # Persistir ítems aprobados en la BD (Fase 4)
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
