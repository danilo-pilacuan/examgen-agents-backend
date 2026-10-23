"""Tools of the conversational agent.

Each tool encapsulates ONE scoped action on the question bank. The LLM
invokes them via tool-calling. All writes go through the service layer
(`src/bank/service.py`), never direct SQL.

Guardrails applied here:
  - `_validate_exam_topic()`: rejects topics outside the exam scope
    (hyper-specialization).
  - Generation of ONE question per call.
  - Explicit human confirmation to mark `verified_by_human`.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agents_backend.shared.llm import generator_llm
from agents_backend.shared.retrieval import retrieve_chunks, format_context
from agents_backend.shared.events import emit_event, EventType
from agents_backend.shared.pending import pending_store, PendingItem
from core.domain.bank import service as bank
from core.domain.schemas.exam import GeneratedItem, RetrievalConfig
from core.domain.schemas.tools import (
    AddMaterialFileInput,
    ConfirmVerificationInput,
    COLUMN_LABELS,
    DEFAULT_TABLE_COLUMNS,
    DeleteMaterialFileInput,
    DeleteQuestionInput,
    ExportQuestionsInput,
    GenerateExamPdfInput,
    GenerateQuestionInput,
    ListMaterialInput,
    ListSubjectsInput,
    RegisterSubjectInput,
    SavePendingQuestionInput,
    SaveQuestionInput,
    SearchQuestionsInput,
    SelectSubjectInput,
    UpdateQuestionInput,
    TableColumn,
)

# Topics that are OUTSIDE the agent's scope (hyper-specialization).
# English keywords first; Spanish synonyms kept so Spanish-speaking
# professors also get a graceful rejection instead of a generated item.
_OFF_TOPIC_KEYWORDS = [
    "recipe", "cooking", "movie", "cinema", "film", "football", "soccer",
    "sport", "news", "gossip", "general politics", "weather", "travel",
    "music", "video game",
    "receta", "cocina", "pelicula", "cine", "futbol", "deporte", "noticia",
    "chisme", "politica general", "clima", "viaje", "musica", "videojuego",
]

# Pending item to save: store partitioned by thread_id (see
# agents_backend/shared/pending.py). Filled by `generate_question` and
# consumed by `save_pending_question`; `_call_tools_node` syncs it with
# ConversationalState.pending_item so the checkpointer persists it.
def get_pending_item(thread_id: str | None = None) -> tuple[dict | None, str | None]:
    """Returns (pending_item, subject_id) of the current thread (or the given one)."""
    tid = thread_id or _current_thread()
    pending = pending_store.get(tid) if tid else None
    if pending is None:
        return None, None
    return pending.item, pending.subject_id


def set_pending_item(item: dict | None, subject_id: str | None, thread_id: str | None = None) -> None:
    """Saves (or clears) the pending item of the current thread (or the given one)."""
    tid = thread_id or _current_thread()
    if tid is None:
        return
    if item is None:
        pending_store.clear(tid)
        return
    pending_store.set(tid, PendingItem(item=item, subject_id=subject_id or ""))


def _current_thread() -> str | None:
    """thread_id of the current turn according to the event context."""
    from agents_backend.shared.events import get_current_thread_id
    return get_current_thread_id()


def _validate_exam_topic(topic: str) -> str | None:
    """Returns a rejection message if the topic is out of scope, or None."""
    t = topic.lower()
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in t:
            return (
                f"Sorry, I can't help you with '{topic}': it is outside my function. "
                "I am an assistant specialized in generating and managing exam questions "
                "from your subjects' material. Would you like me to help you with a topic "
                "from your notes?"
            )
    return None


def _render_item_for_chat(item: GeneratedItem) -> str:
    """Formats a generated item to show it in the chat (no LaTeX)."""
    lines = [f"📝 **{item.statement}**"]
    if item.options:
        for i, o in enumerate(item.options, start=1):
            lines.append(f"   {i}. {o.text}")
    lines.append(f"✅ **Expected answer:** {item.expected_answer}")
    if item.solution_explanation:
        lines.append(f"💡 **Explanation:** {item.solution_explanation}")
    lines.append(
        f"🏷 Topic: {item.topic} · Level: {item.bloom_level.value} · "
        f"Difficulty: {item.difficulty.value} · Type: {item.question_type.value}"
    )
    return "\n".join(lines)


def _cell_value(q, column: TableColumn) -> str:
    """Extracts the value of a column for a question (table cell)."""
    if column == TableColumn.N:
        return ""  # filled with the row number
    if column == TableColumn.ID:
        return f"`{q.id}`"
    if column == TableColumn.VERIFIED_BY_HUMAN:
        return "✅" if q.verified_by_human else "⏳"
    if column == TableColumn.VERIFIED_BY_AI:
        return "✅" if q.verified_by_ai else "—"
    if column == TableColumn.AI_REVIEW_PRIORITY:
        return (q.ai_review_priority or "—").capitalize()
    if column == TableColumn.QUESTION_TEXT:
        # The statement can be long: it is truncated so the table stays readable.
        text = (q.question_text or "").replace("\n", " ").replace("|", "\\|")
        return text[:80] + ("…" if len(text) > 80 else "")
    if column == TableColumn.EXPECTED_ANSWER:
        text = (q.expected_answer or "—").replace("\n", " ").replace("|", "\\|")
        return text[:60] + ("…" if len(text) > 60 else "")
    value = getattr(q, column.value, None)
    if value is None:
        return "—"
    return str(value).replace("|", "\\|")


def _render_questions_table(questions: list, columns: list[TableColumn]) -> str:
    """Renders the questions as a markdown table with the requested columns."""
    header = [COLUMN_LABELS[c] for c in columns]
    rows = []
    for i, q in enumerate(questions, start=1):
        cells = []
        for c in columns:
            if c == TableColumn.N:
                cells.append(str(i))
            else:
                cells.append(_cell_value(q, c))
        rows.append(cells)

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@tool(args_schema=SearchQuestionsInput)
def search_questions(
    subject_id: str,
    topic: str | None = None,
    limit: int = 20,
    columns: list[str] | None = None,
) -> str:
    """Retrieves questions ALREADY EXISTING in the bank (does not generate new ones).

    Use it when the professor asks 'give me N questions on these topics' without
    wanting to generate new ones. Returns the stored questions in markdown TABLE
    format. With `columns` the professor can request which columns to show
    (e.g. 'give me id, topic and difficulty in a table'); if omitted the default
    set is used.
    """
    rejection = _validate_exam_topic(topic or "")
    if rejection:
        return rejection

    questions = bank.list_questions(
        subject_id=subject_id,
        topic=topic,
        limit=limit,
    )
    if not questions:
        return (
            f"There are no questions in the bank for the topic '{topic or 'all'}' "
            "of this subject. I can generate a new one if you ask me to."
        )

    # Resolve requested columns (validated by args_schema as TableColumn);
    # if none were requested, use the default set.
    if columns:
        table_columns = [TableColumn(c) for c in columns]
    else:
        table_columns = list(DEFAULT_TABLE_COLUMNS)

    table = _render_questions_table(questions, table_columns)

    # Typed event for the frontend's interactive table.
    emit_event(
        EventType.QUESTIONS_LISTED,
        {
            "questions": [
                {
                    "id": str(q.id),
                    "topic": q.topic,
                    "subtopic": q.subtopic,
                    "bloom_level": q.bloom_level,
                    "difficulty": q.difficulty,
                    "question_type": q.question_type,
                    "question_text": q.question_text,
                    "options": q.options,
                    "expected_answer": q.expected_answer,
                    "solution_explanation": q.solution_explanation,
                    "points": q.points,
                    "verified_by_human": q.verified_by_human,
                    "verified_by_ai": q.verified_by_ai,
                    "ai_review_priority": q.ai_review_priority,
                }
                for q in questions
            ],
            "total": len(questions),
        },
    )

    return f"📚 I found {len(questions)} question(s) in the bank:\n\n{table}"


@tool(args_schema=GenerateQuestionInput)
def generate_question(
    subject_id: str,
    subject_name: str,
    topic: str,
    difficulty: str = "medium",
) -> str:
    """Generates ONE new question on a topic, using RAG over the material.

    Generates ONE question at a time (never several). Does NOT save yet: it
    returns the question for the professor to review and confirm before saving.
    """
    rejection = _validate_exam_topic(topic)
    if rejection:
        return rejection

    # RAG: retrieve chunks relevant to the topic from the indexed material.
    retrieval_cfg = RetrievalConfig(top_k=3, min_score=0.15, use_rag=True)
    context = ""
    try:
        chunks = retrieve_chunks(
            query=topic,
            subject_id=subject_id,
            top_k=retrieval_cfg.top_k,
            min_score=retrieval_cfg.min_score,
        )
        if chunks:
            context = format_context(chunks)
    except Exception:  # noqa: BLE001
        context = ""

    prompt = (
        "You are an exam item generator. Generate EXACTLY ONE question "
        f"on the topic '{topic}' of the subject '{subject_name}', with difficulty "
        f"'{difficulty}'. Use the material context if available.\n\n"
        f"{context or 'No additional context from the material.'}\n\n"
        "Return only the requested structure."
    )

    try:
        structured = generator_llm.with_structured_output(GeneratedItem)
        item: GeneratedItem = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        return f"❌ I could not generate the question: {str(e)[:200]}"

    # Store the pending item so that, upon confirmation, `save_pending_question`
    # persists it without the LLM having to re-specify all the data.
    set_pending_item(item.model_dump(), subject_id)

    # Typed event for the frontend's question widget (HITL).
    emit_event(EventType.QUESTION_GENERATED, {"question": item.model_dump()})

    # Not saved yet: returned for human review (HITL).
    return (
        "I generated this question (NOT saved yet):\n\n"
        f"{_render_item_for_chat(item)}\n\n"
        "Shall I save it to the bank? Answer **yes** to save it as verified, "
        "or tell me what to adjust."
    )


@tool(args_schema=SaveQuestionInput)
def save_question(
    subject_id: str,
    topic: str,
    bloom_level: str,
    difficulty: str,
    question_text: str,
    question_type: str,
    options: list[dict] | None = None,
    expected_answer: str | None = None,
    solution_explanation: str | None = None,
    subtopic: str | None = None,
    points: float = 1.0,
) -> str:
    """Saves a question to the bank (with AI verification).

    ⚠️ This tool does NOT mark `verified_by_human`. Human verification is done
    separately after the professor's explicit confirmation.
    """
    # Embedding for saving (question RAG).
    from core.infrastructure.embeddings.embedder import embedder

    vec = embedder.embed([question_text])[0]

    # AI verification BEFORE persisting (to set the review priority).
    from agents_backend.shared.verifier import review_question

    item_for_review = {
        "topic": topic,
        "bloom_level": bloom_level,
        "difficulty": difficulty,
        "question_type": question_type,
        "statement": question_text,
        "options": options,
        "expected_answer": expected_answer,
        "solution_explanation": solution_explanation,
    }
    review = review_question(item_for_review)

    q = bank.create_question(
        subject_id=subject_id,
        topic=topic,
        bloom_level=bloom_level,
        difficulty=difficulty,
        question_text=question_text,
        embedding=vec,
        question_type=question_type,
        options=options,
        expected_answer=expected_answer,
        solution_explanation=solution_explanation,
        subtopic=subtopic,
        points=points,
        source="chat",
        created_by="agent",
        verified_by_ai=review.verified_by_ai,
        ai_review_notes=review.notes,
        ai_review_priority=review.priority,
    )
    review_note = (
        f"\n🔍 **AI verification:** {'approved' if review.verified_by_ai else 'requires review'} "
        f"(priority {review.priority}). {review.notes}"
        if review.notes
        else ""
    )

    # Typed event: question saved (the frontend refreshes the bank).
    emit_event(
        EventType.QUESTION_SAVED,
        {"question_id": str(q.id), "verified_by_human": False},
    )

    return (
        f"✅ Question saved to the bank (id: {q.id}).{review_note}\n"
        "It is NOT marked as verified by human yet. "
        "Do you confirm it as verified?"
    )


@tool(args_schema=ConfirmVerificationInput)
def confirm_human_verification(question_id: str) -> str:
    """Marks a question as VERIFIED BY HUMAN.

    ⚠️ It must ONLY be invoked after the professor's explicit confirmation in
    the chat (human-in-the-loop). Once verified, the question is protected
    against modifications by the agent.
    """
    try:
        q = bank.set_human_verification(question_id, verified=True)
    except Exception as e:  # noqa: BLE001
        return f"❌ Could not verify: {str(e)[:200]}"
    return (
        f"✅ Question {q.id} marked as **verified by human**. "
        "From now on it cannot be modified by the agent."
    )


@tool(args_schema=DeleteQuestionInput)
def delete_question_tool(question_id: str) -> str:
    """DELETES a question from the bank.

    Use it when the professor explicitly asks to delete/remove a question
    ('delete question X', 'remove that question'). NEVER delete without the
    professor's explicit request. Questions verified by a human CANNOT be
    deleted (guardrail): the professor must first un-verify them from the
    Bank view.

    Args:
        question_id: id (UUID) of the question to delete. Get it from
            `search_questions` if you don't know it.
    """
    try:
        bank.delete_question(question_id)
    except bank.SecurityError as e:
        return f"🔒 {str(e)[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"❌ Could not delete: {str(e)[:200]}"
    return f"🗑 Question {question_id} deleted from the bank."


@tool(args_schema=UpdateQuestionInput)
def update_question_tool(
    question_id: str,
    topic: str | None = None,
    subtopic: str | None = None,
    bloom_level: str | None = None,
    difficulty: str | None = None,
    question_type: str | None = None,
    question_text: str | None = None,
    options: list[dict] | None = None,
    expected_answer: str | None = None,
    solution_explanation: str | None = None,
    points: float | None = None,
) -> str:
    """UPDATES one or more fields of an existing bank question.

    Use it when the professor asks to modify/fix a question ('change the
    difficulty to hard', 'fix the statement', 'update the answer'). Only the
    provided fields are updated. Questions verified by a human CANNOT be
    modified (guardrail): the professor must first un-verify them from the
    Bank view.

    Args:
        question_id: id (UUID) of the question to update.
        topic / subtopic / bloom_level / difficulty / question_type /
        question_text / options / expected_answer / solution_explanation /
        points: new values (only send the ones to change).
    """
    fields: dict[str, Any] = {}
    if topic is not None:
        fields["topic"] = topic
    if subtopic is not None:
        fields["subtopic"] = subtopic
    if bloom_level is not None:
        fields["bloom_level"] = bloom_level
    if difficulty is not None:
        fields["difficulty"] = difficulty
    if question_type is not None:
        fields["question_type"] = question_type
    if question_text is not None:
        fields["question_text"] = question_text
    if options is not None:
        fields["options"] = options
    if expected_answer is not None:
        fields["expected_answer"] = expected_answer
    if solution_explanation is not None:
        fields["solution_explanation"] = solution_explanation
    if points is not None:
        fields["points"] = points

    if not fields:
        return "⚠️ No fields to update were provided."

    try:
        q = bank.update_question(question_id, fields=fields)
    except bank.SecurityError as e:
        return f"🔒 {str(e)[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"❌ Could not update: {str(e)[:200]}"

    updated = ", ".join(fields.keys())
    return (
        f"✅ Question {q.id} updated ({updated}).\n\n"
        f"{_render_item_for_chat_from_question(q)}"
    )


def _render_item_for_chat_from_question(q) -> str:
    """Formats a bank question for the chat (update confirmation)."""
    lines = [f"📝 **{q.question_text}**"]
    if q.options:
        for i, o in enumerate(q.options, start=1):
            mark = "✔" if o.get("is_correct") else ""
            lines.append(f"   {i}. {o.get('text', '')} {mark}")
    if q.expected_answer:
        lines.append(f"✅ **Expected answer:** {q.expected_answer}")
    lines.append(
        f"🏷 Topic: {q.topic} · Level: {q.bloom_level} · "
        f"Difficulty: {q.difficulty} · Type: {q.question_type}"
    )
    return "\n".join(lines)


@tool(args_schema=SavePendingQuestionInput)
def save_pending_question(verify_human: bool = False) -> str:
    """Saves the LAST generated question to the bank (pending confirmation).

    Uses the data of the question that `generate_question` left pending, so the
    LLM does not have to re-specify all the fields. If `verify_human=True`, it
    also marks it as verified by human (after the professor's confirmation).
    """
    item, subject_id = get_pending_item()
    if not item or not subject_id:
        # Tolerate re-queries after a dropped stream (BUG-10): the save may
        # have completed even though the pending item was already cleared.
        # The pending store keeps the last saved question id, so report the
        # real bank state instead of a dead end.
        tid = _current_thread()
        saved_id = pending_store.get_saved_question_id(tid) if tid else None
        if saved_id:
            q = bank.get_question(saved_id)
            if q is not None:
                state_txt = (
                    "verified by human" if q.verified_by_human
                    else "saved but NOT verified by human yet"
                )
                return (
                    f"✅ Your question was already saved (id: {q.id}) and is "
                    f"currently {state_txt}. No action needed."
                )
        return (
            "There is no pending question to save. First generate a question "
            "with the `generate_question` tool."
        )

    # Re-validate the pending item against the GeneratedItem contract: the dict
    # lived in module memory and must comply with the same strict schema.
    try:
        item_model = GeneratedItem.model_validate(item)
    except Exception as e:  # noqa: BLE001
        return (
            "⚠️ The pending question does not comply with the expected schema and was not saved. "
            "Generate it again with `generate_question`. "
            f"Detail: {str(e)[:200]}"
        )
    item = item_model.model_dump()

    from core.infrastructure.embeddings.embedder import embedder
    from agents_backend.shared.verifier import review_question

    # Embedding for saving (question RAG).
    vec = embedder.embed([item["statement"]])[0]

    # AI verification before persisting.
    review = review_question(item)

    q = bank.create_question(
        subject_id=subject_id,
        topic=item["topic"],
        bloom_level=item["bloom_level"],
        difficulty=item["difficulty"],
        question_text=item["statement"],
        embedding=vec,
        question_type=item.get("question_type"),
        options=item.get("options"),
        expected_answer=item.get("expected_answer"),
        solution_explanation=item.get("solution_explanation"),
        subtopic=item.get("subtopic"),
        points=item.get("points", 1.0),
        source="chat",
        created_by="agent",
        verified_by_ai=review.verified_by_ai,
        ai_review_notes=review.notes,
        ai_review_priority=review.priority,
    )

    review_note = (
        f"\n🔍 **AI verification:** {'approved' if review.verified_by_ai else 'requires review'} "
        f"(priority {review.priority}). {review.notes}"
        if review.notes
        else ""
    )

    # Record the saved id BEFORE clearing the pending item (BUG-10): if the
    # stream dies right after, a re-query reports the real bank state.
    from agents_backend.shared.events import get_current_thread_id

    _tid = get_current_thread_id()
    if _tid:
        pending_store.set_saved_question_id(_tid, str(q.id))

    if verify_human:
        try:
            bank.set_human_verification(q.id, verified=True)
        except Exception as e:  # noqa: BLE001
            emit_event(
                EventType.QUESTION_SAVED,
                {"question_id": str(q.id), "verified_by_human": False},
            )
            set_pending_item(None, None)
            return (
                f"✅ Question saved (id: {q.id}).{review_note}\n"
                f"⚠️ Could not mark it as verified: {str(e)[:120]}"
            )
        emit_event(
            EventType.QUESTION_SAVED,
            {"question_id": str(q.id), "verified_by_human": True},
        )
        set_pending_item(None, None)
        return (
            f"✅ Question saved and **verified by human** (id: {q.id}).{review_note}\n"
            "From now on it cannot be modified by the agent."
        )

    emit_event(
        EventType.QUESTION_SAVED,
        {"question_id": str(q.id), "verified_by_human": False},
    )
    set_pending_item(None, None)
    return (
        f"✅ Question saved to the bank (id: {q.id}).{review_note}\n"
        "It is NOT marked as verified by human yet. "
        "Do you confirm it as verified?"
    )


@tool(args_schema=ListSubjectsInput)
def list_subjects() -> str:
    """Lists the subjects registered in the system.

    Use it when the professor asks which subjects exist or which ones they can
    use. Returns the id and name of each registered subject.
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject

    db = SessionLocal()
    try:
        subjects = db.query(Subject).all()
    finally:
        db.close()

    if not subjects:
        return "There are no subjects registered yet. You can register one by telling me its name."

    lines = ["📚 Registered subjects:"]
    for s in subjects:
        lines.append(f"- **{s.name}** (id: `{s.id}`)")
    lines.append("\nTell me the id of the subject you want to work with.")
    return "\n".join(lines)


@tool(args_schema=GenerateExamPdfInput)
def generate_exam_pdf(
    subject_id: str,
    subject_name: str,
    total_questions: int = 5,
    use_bank: bool = True,
    num_versions: int = 1,
    exam_name: str = "",
) -> str:
    """Generates an exam from the question bank and returns a ZIP package.

    Builds an exam using the subject's questions ALREADY EXISTING in the bank
    (if `use_bank=True`) or generating new ones. Compiles the LaTeX to PDF
    and packages everything into a ZIP.

    Each version is compiled TWICE: `<Name>_A.pdf` (student copy, without
    answers) and `<Name>_A_respondido.pdf` (instructor copy, with answers),
    where `<Name>` is the exam name and A/B/C... is the version letter.
    The ZIP `<Name>.zip` contains all the PDFs and the exam is persisted
    in the database (visible in the Exams view).

    IMPORTANT (human-in-the-loop): when the professor asks for an exam,
    do NOT call this tool yet. First show the blueprint summary (number of
    questions, Bloom/difficulty distribution) and ASK for TWO things:
    1. a NAME for the exam (e.g. 'Midterm 1'); if the professor does not
       provide one, propose one based on the subject and date;
    2. how many VERSIONS they want (1-10, shuffled anti-copy style).
    Call this tool only after the professor confirms both.
    """
    from agents_backend.exam_pipeline.graph import run_exam_pipeline
    from core.domain.schemas.exam import (
        BloomLevel,
        Difficulty,
        ExamBlueprint,
    )
    from core.infrastructure.latex.templates.registry import TemplateId

    # Retrieve bank questions to build the exam (reuse, don't generate).
    questions = bank.list_questions(
        subject_id=subject_id,
        limit=total_questions,
    )
    if not questions:
        return (
            f"There are no questions in the bank for the subject '{subject_name}'. "
            "First generate and save some questions, or ask to generate an exam "
            "with new questions."
        )

    # Build a blueprint that requests exactly `total_questions` questions,
    # uniformly distributed (the pipeline will generate them with material RAG).
    blueprint = ExamBlueprint(
        subject_id=subject_id,
        total_questions=total_questions,
        bloom_distribution={
            BloomLevel.REMEMBER: 0.25,
            BloomLevel.UNDERSTAND: 0.25,
            BloomLevel.APPLY: 0.25,
            BloomLevel.ANALYZE: 0.25,
        },
        difficulty_distribution={
            Difficulty.EASY: 0.4,
            Difficulty.MEDIUM: 0.4,
            Difficulty.HARD: 0.2,
        },
        topics_weight={"General": 1.0},
        estimated_duration_minutes=60,
    )
    blueprint.metadata.num_versions = num_versions
    blueprint.metadata.name = exam_name.strip() or f"{subject_name} exam"

    try:
        result = run_exam_pipeline(
            subject_id=subject_id,
            subject_name=subject_name,
            template_id=TemplateId.BASE_EXAM,
            raw_materials_paths=[],
            blueprint=blueprint,
            persist=False,  # no duplicates: the questions are already in the bank
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Error generating the exam: {str(e)[:300]}"

    # Fail fast: an exam with zero approved items is an error, not a
    # deliverable (it used to compile and publish empty PDFs).
    approved = result.get("approved_items") or []
    if result.get("status") != "completed" or not approved:
        rejection_lines = []
        for entry in (result.get("rejection_log") or [])[:10]:
            reason = entry.get("reason", "unknown")
            spec_id = entry.get("spec_id", "?")
            rejection_lines.append(f"  - spec `{spec_id}`: {reason}")
        rejection_summary = (
            "\n".join(rejection_lines)
            if rejection_lines
            else "  - (no rejection details recorded)"
        )
        return (
            f"❌ The exam could not be generated: no question was approved by the "
            f"evaluator (status: {result.get('status')}).\n"
            f"Rejections:\n{rejection_summary}\n"
            f"Total LLM calls: {result.get('total_llm_calls', 0)}. "
            "Nothing was compiled or published. Please try again."
        )

    # The compiler node packages all the PDFs (with/without answers per
    # version) into a single ZIP artifact and persists the exam in the DB.
    zip_path = result.get("zip_path")
    if not zip_path:
        return (
            "Could not compile the exam package. "
            f"Status: {result.get('status')} · Error: {result.get('error_message', 'unknown')}"
        )

    zip_name = Path(zip_path).name
    num_pdfs = len(result.get("pdf_paths") or [])

    # Typed event: the ZIP artifact card in the frontend.
    emit_event(
        EventType.ARTIFACT_READY,
        {
            "kind": "zip",
            "url": f"/output/{zip_name}",
            "filename": zip_name,
        },
    )

    # Upload the ZIP to Open WebUI as a native chat attachment.
    try:
        from exam_app.infrastructure.openwebui_bridge import upload_file_to_openwebui, build_attachment_markdown

        upload_result = upload_file_to_openwebui(zip_path)
        attachment_md = build_attachment_markdown(upload_result)
    except Exception as e:  # noqa: BLE001
        # If the upload fails, fall back to the direct HTTP link.
        attachment_md = (
            f"📦 **Exam package ready.**\n"
            f"- 📥 [Download {zip_name}](http://localhost:8000/output/{zip_name})\n\n"
            f"*(Could not attach to Open WebUI: {str(e)[:120]})*"
        )

    version_note = f" · Versions: {num_versions}" if num_versions > 1 else ""
    return (
        f"✅ **Exam '{blueprint.metadata.name}' generated successfully.**\n"
        f"- Subject: **{subject_name}**\n"
        f"- Questions: {len(result.get('approved_items', []))}{version_note}\n"
        f"- Package: {num_pdfs} PDFs (each version without answers + answered copy)\n"
        f"- Saved to the Exams view\n\n"
        f"{attachment_md}"
    )


@tool(args_schema=ExportQuestionsInput)
def export_questions(
    subject_id: str,
    format: str = "csv",
    topic: str | None = None,
    limit: int = 100,
) -> str:
    """EXPORTS the bank questions to a CSV, Excel (xlsx) or JSON file.

    Use it when the professor asks 'export the questions', 'download them in
    Excel/CSV/JSON' or wants to take the bank to a spreadsheet (or download
    the raw JSON data). Generates the file with ALL the bank columns
    (statement, options, answer, solution, verification, etc.) and uploads
    it as a downloadable chat attachment.

    Args:
        subject_id: subject id (UUID).
        format: 'csv', 'xlsx' (Excel) or 'json' (raw data).
        topic: optional filter by exact topic.
        limit: max number of questions to export (1-500).
    """
    import csv
    import io
    import json as jsonlib
    from datetime import datetime

    from exam_app.infrastructure.openwebui_bridge import upload_file_to_openwebui, build_attachment_markdown

    fmt = (format or "csv").strip().lower()
    if fmt not in ("csv", "xlsx", "json"):
        fmt = "csv"

    questions = bank.list_questions(
        subject_id=subject_id,
        topic=topic,
        limit=limit,
    )
    if not questions:
        return (
            f"There are no questions in the bank to export "
            f"(topic: '{topic or 'all'}')."
        )

    # Full bank columns (in a spreadsheet-friendly readable order).
    export_columns = [
        TableColumn.N,
        TableColumn.ID,
        TableColumn.TOPIC,
        TableColumn.SUBTOPIC,
        TableColumn.BLOOM_LEVEL,
        TableColumn.DIFFICULTY,
        TableColumn.QUESTION_TYPE,
        TableColumn.QUESTION_TEXT,
        TableColumn.OPTIONS,
        TableColumn.EXPECTED_ANSWER,
        TableColumn.SOLUTION_EXPLANATION,
        TableColumn.POINTS,
        TableColumn.VERIFIED_BY_HUMAN,
        TableColumn.VERIFIED_BY_AI,
        TableColumn.AI_REVIEW_PRIORITY,
        TableColumn.SOURCE,
        TableColumn.CREATED_AT,
    ]

    def _plain(q, column: TableColumn) -> str:
        """Plain value of a cell (no markdown, for CSV/Excel)."""
        if column == TableColumn.N:
            return ""
        if column == TableColumn.VERIFIED_BY_HUMAN:
            return "yes" if q.verified_by_human else "no"
        if column == TableColumn.VERIFIED_BY_AI:
            return "yes" if q.verified_by_ai else "no"
        if column == TableColumn.OPTIONS:
            opts = q.options or []
            if not opts:
                return ""
            parts = []
            for i, o in enumerate(opts, start=1):
                mark = "*" if o.get("is_correct") else ""
                parts.append(f"{i}) {mark}{o.get('text', '')}")
            return " | ".join(parts)
        value = getattr(q, column.value, None)
        if value is None:
            return ""
        return str(value).replace("\n", " ")

    rows = []
    for i, q in enumerate(questions, start=1):
        row = []
        for c in export_columns:
            if c == TableColumn.N:
                row.append(i)
            else:
                row.append(_plain(q, c))
        rows.append(row)

    headers = [COLUMN_LABELS[c] for c in export_columns]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory (shared with latex-compiler and served by /output).
    from core.infrastructure.files.output_paths import get_output_dir

    OUTPUT_DIR = get_output_dir()

    try:
        if fmt == "xlsx":
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "Questions"
            ws.append(headers)
            for row in rows:
                ws.append(row)
            # Adjust column widths (capped at 60 chars to avoid distortion).
            for col_idx, h in enumerate(headers, start=1):
                width = max((len(str(row[col_idx - 1])) for row in rows), default=10)
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max(width + 2, 10), 60)
            out_path = OUTPUT_DIR / f"questions_{stamp}.xlsx"
            wb.save(out_path)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif fmt == "json":
            # Raw data: full records (all fields), not just the table columns.
            payload = {
                "subject_id": subject_id,
                "topic_filter": topic,
                "exported_at": datetime.now().isoformat(),
                "count": len(questions),
                "questions": [
                    {
                        "id": str(q.id),
                        "subject_id": str(q.subject_id),
                        "topic": q.topic,
                        "subtopic": q.subtopic,
                        "bloom_level": q.bloom_level,
                        "difficulty": q.difficulty,
                        "question_type": q.question_type,
                        "question_text": q.question_text,
                        "options": q.options,
                        "expected_answer": q.expected_answer,
                        "solution_explanation": q.solution_explanation,
                        "points": q.points,
                        "verified_by_human": q.verified_by_human,
                        "verified_by_ai": q.verified_by_ai,
                        "ai_review_notes": q.ai_review_notes,
                        "ai_review_priority": q.ai_review_priority,
                        "source": q.source,
                        "created_by": q.created_by,
                        "created_at": str(q.created_at),
                    }
                    for q in questions
                ],
            }
            out_path = OUTPUT_DIR / f"questions_{stamp}.json"
            out_path.write_text(
                jsonlib.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mime = "application/json"
        else:
            buf = io.StringIO()
            writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
            writer.writerows(rows)
            out_path = OUTPUT_DIR / f"questions_{stamp}.csv"
            out_path.write_text("\ufeff" + buf.getvalue(), encoding="utf-8")  # BOM for Excel
            mime = "text/csv"
    except Exception as e:  # noqa: BLE001
        return f"❌ I could not generate the export file: {str(e)[:200]}"

    # Typed event: downloadable artifact card in the frontend.
    emit_event(
        EventType.ARTIFACT_READY,
        {
            "kind": fmt,
            "url": f"/output/{out_path.name}",
            "filename": out_path.name,
        },
    )

    # Upload the file to Open WebUI as a native chat attachment.
    try:
        upload = upload_file_to_openwebui(out_path, mime_type=mime)
        title_by_fmt = {
            "xlsx": f"📊 **Export of {len(questions)} question(s) in Excel attached to the chat.**",
            "csv": f"📊 **Export of {len(questions)} question(s) in CSV attached to the chat.**",
            "json": f"🧾 **Export of {len(questions)} question(s) in JSON attached to the chat.**",
        }
        label_by_fmt = {
            "xlsx": "Download the Excel",
            "csv": "Download the CSV",
            "json": "Download the JSON",
        }
        attachment_md = build_attachment_markdown(
            upload, title=title_by_fmt[fmt], label=label_by_fmt[fmt]
        )
    except Exception as e:  # noqa: BLE001
        # Fallback: direct HTTP link served by the API.
        attachment_md = (
            f"📥 [Download {out_path.name}](http://localhost:8000/output/{out_path.name})\n\n"
            f"*(Could not attach to Open WebUI: {str(e)[:120]})*"
        )

    return (
        f"✅ Exported **{len(questions)} question(s)** from the bank "
        f"(format: {fmt.upper()}).\n\n{attachment_md}"
    )


@tool(args_schema=AddMaterialFileInput)
def add_material_file(
    file_id: str,
    subject_id: str,
    subject_name: str,
    clear_previous: bool = False,
) -> str:
    """Adds a chat-attached file as study material of a subject.

    Downloads the file uploaded to Open WebUI (identified by its `file_id`),
    extracts its text, splits it into chunks and indexes those chunks into the
    given subject (so the agent can use them as RAG context when generating
    questions or exams).

    Args:
        file_id: id of the file attached in Open WebUI (comes in the
            `<file id="..."/>` tag of the user message).
        subject_id: id of the subject to add the material to.
        subject_name: name of the subject (for confirmation).
        clear_previous: if True, replaces the previous material of that file.
    """
    from core.domain.bank.material import index_text_for_subject
    from exam_app.infrastructure.openwebui_bridge import get_file_metadata, download_file_content

    # 1) Get file metadata (name, type).
    try:
        meta = get_file_metadata(file_id)
    except Exception as e:  # noqa: BLE001
        return (
            f"❌ I could not access the attached file (id: {file_id}). "
            f"Detail: {str(e)[:200]}"
        )

    filename = meta.get("filename", "file")
    data = meta.get("data", {}) or {}
    # Open WebUI already extracts the textual content of the file into `data.content`.
    content = data.get("content") or ""

    # 2) Download the file binary (to physically store it in the subject's
    #    material repository).
    raw_bytes: bytes | None = None
    try:
        raw_bytes = download_file_content(file_id)
    except Exception:  # noqa: BLE001
        raw_bytes = None

    # 3) If Open WebUI did not extract the content, extract it from the binary.
    if not content.strip() and raw_bytes:
        try:
            from core.domain.bank.material import extract_file_text

            tmp_path = Path(f"/tmp/owui_{file_id}")
            tmp_path.write_bytes(raw_bytes)
            content = extract_file_text(tmp_path)
            tmp_path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            return (
                f"❌ I could not extract the content of the file '{filename}'. "
                f"Detail: {str(e)[:200]}"
            )

    if not content.strip():
        return (
            f"⚠️ The file '{filename}' contains no extractable text "
            "(it may be a scanned image or an unsupported format)."
        )

    # 4) Save the physical file in the subject's material directory.
    saved_path = None
    if raw_bytes:
        try:
            from core.domain.bank.material import save_file_to_subject_folder

            saved_path = save_file_to_subject_folder(raw_bytes, filename, subject_name)
        except Exception as e:  # noqa: BLE001
            saved_path = None  # do not block indexing if the physical save fails

    # 5) Index the content into the subject (RAG).
    try:
        res = index_text_for_subject(
            text=content,
            subject_id=subject_id,
            source_label=f"chat:{filename}",
            clear_previous=clear_previous,
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Error indexing the material: {str(e)[:200]}"

    if res.get("chunks_saved", 0) == 0:
        return f"⚠️ Could not index the file '{filename}': {res.get('message', '')}"

    # Typed event: progress/confirmation toast in the frontend.
    emit_event(
        EventType.MATERIAL_INDEXED,
        {
            "chunks": res["chunks_saved"],
            "filename": filename,
            "subject_name": subject_name,
        },
    )

    # Build the confirmation (includes whether the physical file was saved).
    lines = [
        f"✅ Material added to the subject **{subject_name}**.",
        f"- File: **{filename}**",
        f"- Indexed chunks: **{res['chunks_saved']}**",
    ]
    if saved_path:
        lines.append(f"- 📁 File saved in the material repository: `{saved_path}`")
    else:
        lines.append("- ⚠️ The physical file could not be saved in the repository (only the text was indexed in the DB).")
    lines.append("\nThe agent can now use this material as context when generating questions or exams for the subject.")
    return "\n".join(lines)


@tool(args_schema=ListMaterialInput)
def list_subject_material(subject_id: str) -> str:
    """Lists the indexed study material of a subject.

    Use it when the professor asks what material is available for a subject,
    or to verify whether a file was indexed. Returns the source files and the
    number of indexed chunks of each one.
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject, MaterialChunk

    db = SessionLocal()
    try:
        subject = db.get(Subject, subject_id)
        if subject is None:
            return f"The subject with id `{subject_id}` does not exist."

        rows = (
            db.query(MaterialChunk.source_file, MaterialChunk.subject_id)
            .filter(MaterialChunk.subject_id == subject.id)
            .all()
        )
    finally:
        db.close()

    if not rows:
        return (
            f"The subject **{subject.name}** has no indexed material yet. "
            "You can upload a file attached to the chat and ask me to add it."
        )

    # Group by source file.
    from collections import Counter

    counts = Counter(source for source, _ in rows)

    lines = [f"📚 Indexed material of **{subject.name}**:\n"]
    for source, count in sorted(counts.items()):
        lines.append(f"- **{source}** · {count} chunk(s)")
    lines.append("\nThis material is used as RAG context to generate questions and exams.")
    return "\n".join(lines)


@tool(args_schema=DeleteMaterialFileInput)
def delete_material_file_tool(
    subject_id: str,
    source_file: str,
    delete_physical_file: bool = True,
) -> str:
    """Deletes an indexed material file from a subject.

    Use it ONLY when the professor explicitly asks to delete/remove a file
    from the subject's material. Get the exact `source_file` from
    `list_subject_material` first (never invent it). This removes the
    indexed chunks (RAG context) and, by default, the physical file from
    the subject's material folder.
    """
    from core.domain.material.service import delete_source_file

    try:
        result = delete_source_file(
            subject_id,
            source_file,
            delete_physical_file=delete_physical_file,
        )
    except ValueError as e:
        return f"⚠️ {str(e)}"

    if result["deleted_chunks"] == 0:
        return (
            f"⚠️ No indexed chunks found for source **{source_file}** in "
            f"**{result['subject']}**. Check the exact name with "
            "`list_subject_material`."
        )

    lines = [
        f"🗑️ Deleted material of **{result['subject']}**:",
        f"- Source: **{source_file}**",
        f"- Indexed chunks removed: **{result['deleted_chunks']}**",
    ]
    physical = result["physical_file"]
    if physical == "deleted":
        lines.append("- Physical file: deleted from the material folder")
    elif physical == "not_found":
        lines.append("- Physical file: not found in the material folder (only chunks removed)")
    elif physical == "kept":
        lines.append("- Physical file: kept (not requested for deletion)")
    lines.append("\nThe remaining material is still available as RAG context.")
    return "\n".join(lines)


@tool(args_schema=SelectSubjectInput)
def select_subject(subject_id: str) -> str:
    """Makes a subject the ACTIVE one for the conversation.

    Use it when the professor asks to work on a subject ('work on Physics 1',
    'select Applied Economics', 'switch to Calculus'). The frontend reacts to
    this event: the subject selector in the header updates automatically.

    Args:
        subject_id: id (UUID) of the subject to activate. Get it from
            `list_subjects` if you don't know it.
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject

    db = SessionLocal()
    try:
        try:
            sid = uuid.UUID(subject_id)
        except ValueError:
            return f"⚠️ '{subject_id}' is not a valid subject id (UUID)."
        subject = db.get(Subject, sid)
        if subject is None:
            return f"The subject with id `{subject_id}` does not exist."
        emit_event(
            EventType.SUBJECT_SELECTED,
            {"subject_id": str(subject.id), "subject_name": subject.name},
        )
        return (
            f"✅ Active subject set to **{subject.name}** (id: `{subject.id}`). "
            "The selector in the header has been updated."
        )
    finally:
        db.close()


@tool(args_schema=RegisterSubjectInput)
def register_subject(name: str) -> str:
    """Registers a NEW subject in the system.

    Use it when the professor asks to register a subject that does not exist
    yet (for example 'register the subject Applied Economics'). Creates the
    subject and returns its id to work with it.

    Args:
        name: name of the subject to register (e.g. 'Applied Economics').
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject, MaterialChunk

    name = name.strip()
    if not name:
        return "⚠️ You must provide the name of the subject to register."

    db = SessionLocal()
    try:
        # Avoid duplicates by name (case-insensitive).
        existing = db.query(Subject).filter(Subject.name.ilike(name)).first()
        if existing is not None:
            return (
                f"ℹ️ The subject **{existing.name}** is already registered "
                f"(id: `{existing.id}`). No duplicate was created."
            )
        subject = Subject(name=name)
        db.add(subject)
        db.commit()
        db.refresh(subject)
        emit_event(
            EventType.SUBJECT_REGISTERED,
            {"subject_id": str(subject.id), "subject_name": subject.name},
        )
        return (
            f"✅ Subject **{subject.name}** registered successfully.\n"
            f"- Id: `{subject.id}`\n\n"
            "You can now add study material, generate questions or build "
            "exams for this subject."
        )
    finally:
        db.close()
