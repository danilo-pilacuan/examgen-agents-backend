"""Persistencia de resultados del pipeline en la BD.

Guarda los ítems generados (aprobados y rechazados) junto con sus embeddings
en `generated_questions`, y registra las métricas del examen.
"""
import uuid

from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from core.infrastructure.db.connection import SessionLocal
from core.domain.models import GeneratedQuestion
from core.infrastructure.embeddings.embedder import embedder


def persist_generated_items(
    subject_id: str,
    items: list,
    rejection_log: list,
    total_llm_calls: int,
) -> dict:
    """Guarda los ítems aprobados en `generated_questions` con sus embeddings.

    Devuelve un resumen de lo persistido.
    """
    if not items:
        return {"saved": 0, "subject_id": subject_id}

    # Generar embeddings para todos los ítems aprobados
    texts = [i.statement for i in items]
    vectors = embedder.embed(texts)

    db = SessionLocal()
    saved = 0
    try:
        sid = uuid.UUID(subject_id)
        for item, vec in zip(items, vectors):
            db.add(GeneratedQuestion(
                subject_id=sid,
                topic=item.topic,
                bloom_level=item.bloom_level.value,
                difficulty=item.difficulty.value,
                question_text=item.statement,
                embedding=vec,
                # --- Estructura completa del ítem ---
                question_type=item.question_type.value if item.question_type else None,
                options=(
                    [o.model_dump() for o in item.options]
                    if item.options else None
                ),
                expected_answer=item.expected_answer,
                solution_explanation=item.solution_explanation,
                subtopic=item.subtopic,
                points=item.points,
                # --- Procedencia ---
                source="generated",
                created_by="pipeline",
            ))
            saved += 1
        db.commit()
    finally:
        db.close()

    return {
        "saved": saved,
        "subject_id": subject_id,
        "rejections": len(rejection_log),
        "total_llm_calls": total_llm_calls,
    }
