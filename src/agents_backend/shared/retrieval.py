"""Semantic similarity search (RAG) over material indexed in pgvector.

Given a query text (e.g. a slot topic), retrieves the most relevant chunks
of a subject's material using pgvector.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.infrastructure.db.connection import SessionLocal
from core.domain.models import MaterialChunk
from core.infrastructure.embeddings.embedder import embedder


def retrieve_chunks(
    query: str,
    subject_id: str,
    top_k: int = 4,
    min_score: float = 0.15,
    db: Session | None = None,
) -> list[dict]:
    """Retrieves the most relevant chunks of a subject's material.

    Returns a list of dicts: [{"content": str, "score": float}, ...]
    """
    if not query.strip():
        return []

    own_session = db is None
    if own_session:
        db = SessionLocal()

    try:
        # Embedding of the query
        vec = embedder.embed([query])[0]

        # Cosine similarity search in pgvector
        # (the <=> operator is cosine distance; 1 - distance = similarity)
        stmt = (
            select(
                MaterialChunk.content,
                (1 - MaterialChunk.embedding.cosine_distance(vec)).label("score"),
            )
            .where(MaterialChunk.subject_id == uuid.UUID(subject_id))
            .order_by(MaterialChunk.embedding.cosine_distance(vec))
            .limit(top_k)
        )
        rows = db.execute(stmt).all()

        results = [
            {"content": r.content, "score": float(r.score)}
            for r in rows
            if float(r.score) >= min_score
        ]
        return results
    finally:
        if own_session:
            db.close()


def format_context(chunks: list[dict], max_chars_per_chunk: int = 1200) -> str:
    """Formats the retrieved chunks as context for the LLM."""
    if not chunks:
        return ""
    parts = []
    for i, c in enumerate(chunks, start=1):
        content = c["content"][:max_chars_per_chunk]
        parts.append(f"[Chunk {i} (score {c['score']:.2f})]\n{content}")
    return "\n\n".join(parts)
