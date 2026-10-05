"""Persistence of pipeline results into the database.

Saves the generated items (approved and rejected) along with their embeddings
in `generated_questions`, and records the exam metrics.
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
    """Saves the approved items in `generated_questions` with their embeddings.

    Returns a summary of what was persisted.
    """
    if not items:
        return {"saved": 0, "subject_id": subject_id}

    # Generate embeddings for all approved items
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
                # --- Full item structure ---
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
