"""Herramientas (tools) del agente conversacional.

Cada tool encapsula UNA acción acotada sobre el banco de preguntas. El LLM
las invoca mediante tool-calling. Toda escritura pasa por el service layer
(`src/bank/service.py`), nunca por SQL directo.

Guardrails aplicados aquí:
  - `_validate_exam_topic()`: rechaza temas fuera del ámbito de exámenes
    (hiperespecialización).
  - Generación de a UNA pregunta por llamada.
  - Confirmación humana explícita para marcar `verified_by_human`.
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
    AgregarMaterialArchivoInput,
    BuscarPreguntasInput,
    COLUMN_LABELS,
    ConfirmarVerificacionInput,
    DEFAULT_TABLE_COLUMNS,
    ExportarPreguntasInput,
    GenerarExamenPdfInput,
    GenerarPreguntaInput,
    GuardarPreguntaInput,
    GuardarPreguntaPendienteInput,
    ListarMaterialInput,
    ListarMateriasInput,
    RegistrarMateriaInput,
    TableColumn,
)

# Temas que quedan FUERA del ámbito del agente (hiperespecialización).
_OFF_TOPIC_KEYWORDS = [
    "receta", "cocina", "película", "cine", "fútbol", "deporte", "noticia",
    "chisme", "política general", "clima", "viaje", "música", "videojuego",
]

# Ítem pendiente de guardar: store particionado por thread_id (ver
# agents_backend/shared/pending.py). Lo rellena `generar_pregunta` y lo
# consume `guardar_pregunta_pendiente`; `_call_tools_node` lo sincroniza con
# ConversationalState.pending_item para que el checkpointer lo persista.
def get_pending_item(thread_id: str | None = None) -> tuple[dict | None, str | None]:
    """Devuelve (item_pendiente, subject_id) del thread actual (o el dado)."""
    tid = thread_id or _current_thread()
    pending = pending_store.get(tid) if tid else None
    if pending is None:
        return None, None
    return pending.item, pending.subject_id


def set_pending_item(item: dict | None, subject_id: str | None, thread_id: str | None = None) -> None:
    """Guarda (o limpia) el ítem pendiente del thread actual (o el dado)."""
    tid = thread_id or _current_thread()
    if tid is None:
        return
    if item is None:
        pending_store.clear(tid)
        return
    pending_store.set(tid, PendingItem(item=item, subject_id=subject_id or ""))


def _current_thread() -> str | None:
    """thread_id del turno actual según el contexto de eventos."""
    from agents_backend.shared.events import get_current_thread_id
    return get_current_thread_id()


def _validate_exam_topic(topic: str) -> str | None:
    """Devuelve un mensaje de rechazo si el tema está fuera del ámbito, o None."""
    t = topic.lower()
    for kw in _OFF_TOPIC_KEYWORDS:
        if kw in t:
            return (
                f"Lo siento, no puedo ayudarte con '{topic}': está fuera de mi función. "
                "Soy un asistente especializado en generar y gestionar preguntas de examen "
                "a partir del material de tus materias. ¿Quieres que te ayude con un tema "
                "de tus apuntes?"
            )
    return None


def _render_item_for_chat(item: GeneratedItem) -> str:
    """Formatea un ítem generado para mostrarlo en el chat (sin LaTeX)."""
    lines = [f"📝 **{item.statement}**"]
    if item.options:
        for i, o in enumerate(item.options, start=1):
            lines.append(f"   {i}. {o.text}")
    lines.append(f"✅ **Respuesta esperada:** {item.expected_answer}")
    if item.solution_explanation:
        lines.append(f"💡 **Explicación:** {item.solution_explanation}")
    lines.append(
        f"🏷 Tema: {item.topic} · Nivel: {item.bloom_level.value} · "
        f"Dificultad: {item.difficulty.value} · Tipo: {item.question_type.value}"
    )
    return "\n".join(lines)


def _cell_value(q, column: TableColumn) -> str:
    """Extrae el valor de una columna para una pregunta (celda de tabla)."""
    if column == TableColumn.N:
        return ""  # se rellena con el número de fila
    if column == TableColumn.ID:
        return f"`{q.id}`"
    if column == TableColumn.VERIFIED_BY_HUMAN:
        return "✅" if q.verified_by_human else "⏳"
    if column == TableColumn.VERIFIED_BY_AI:
        return "✅" if q.verified_by_ai else "—"
    if column == TableColumn.AI_REVIEW_PRIORITY:
        return (q.ai_review_priority or "—").capitalize()
    if column == TableColumn.QUESTION_TEXT:
        # El enunciado puede ser largo: se trunca para que la tabla sea legible.
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
    """Renderiza las preguntas como tabla markdown con las columnas pedidas."""
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


@tool(args_schema=BuscarPreguntasInput)
def buscar_preguntas(
    subject_id: str,
    topic: str | None = None,
    limit: int = 20,
    columns: list[str] | None = None,
) -> str:
    """Recupera preguntas YA EXISTENTES del banco (no genera nuevas).

    Úsala cuando el profesor pida 'dame N preguntas de estos temas' sin querer
    generar nuevas. Devuelve las preguntas almacenadas en formato TABLA
    markdown. Con `columns` el profesor puede pedir qué columnas mostrar
    (ej. 'dame id, tema y dificultad en tabla'); si se omite se usa el set
    por defecto.
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
            f"No hay preguntas en el banco para el tema '{topic or 'todas'}' "
            "de esta materia. Puedo generar una nueva si me lo pides."
        )

    # Resolver columnas pedidas (validadas por args_schema como TableColumn);
    # si no se pidieron, usar el set por defecto.
    if columns:
        table_columns = [TableColumn(c) for c in columns]
    else:
        table_columns = list(DEFAULT_TABLE_COLUMNS)

    table = _render_questions_table(questions, table_columns)

    # Evento tipado para la tabla interactiva del frontend.
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

    return f"📚 Encontré {len(questions)} pregunta(s) del banco:\n\n{table}"


@tool(args_schema=GenerarPreguntaInput)
def generar_pregunta(
    subject_id: str,
    subject_name: str,
    topic: str,
    difficulty: str = "medio",
) -> str:
    """Genera UNA pregunta nueva sobre un tema, usando RAG sobre el material.

    Genera de a UNA pregunta (nunca varias). NO guarda todavía: devuelve la
    pregunta para que el profesor la revise y confirme antes de guardar.
    """
    rejection = _validate_exam_topic(topic)
    if rejection:
        return rejection

    # RAG: recuperar chunks relevantes al tema desde el material indexado.
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
        "Eres un generador de ítems de examen. Genera EXACTAMENTE UNA pregunta "
        f"sobre el tema '{topic}' de la materia '{subject_name}', con dificultad "
        f"'{difficulty}'. Usa el contexto del material si está disponible.\n\n"
        f"{context or 'Sin contexto adicional del material.'}\n\n"
        "Devuelve solo la estructura solicitada."
    )

    try:
        structured = generator_llm.with_structured_output(GeneratedItem)
        item: GeneratedItem = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        return f"❌ No pude generar la pregunta: {str(e)[:200]}"

    # Guardar el ítem pendiente para que, al confirmar, `guardar_pregunta_pendiente`
    # lo persista sin que el LLM tenga que re-especificar todos los datos.
    set_pending_item(item.model_dump(), subject_id)

    # Evento tipado para el widget de pregunta del frontend (HITL).
    emit_event(EventType.QUESTION_GENERATED, {"question": item.model_dump()})

    # No se guarda aún: se devuelve para revisión humana (HITL).
    return (
        "Generé esta pregunta (aún NO está guardada):\n\n"
        f"{_render_item_for_chat(item)}\n\n"
        "¿La guardo en el banco? Responde **sí** para guardarla como verificada, "
        "o dime qué ajustar."
    )


@tool(args_schema=GuardarPreguntaInput)
def guardar_pregunta(
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
    """Guarda una pregunta en el banco (con verificación por IA).

    ⚠️ Esta tool NO marca `verified_by_human`. La verificación humana se hace
    por separado tras la confirmación explícita del profesor.
    """
    # Embedding para guardar (RAG de preguntas).
    from core.infrastructure.embeddings.embedder import embedder

    vec = embedder.embed([question_text])[0]

    # Verificación por IA ANTES de persistir (para marcar prioridad de revisión).
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
        f"\n🔍 **Verificación por IA:** {'aprobada' if review.verified_by_ai else 'requiere revisión'} "
        f"(prioridad {review.priority}). {review.notes}"
        if review.notes
        else ""
    )

    # Evento tipado: pregunta guardada (el frontend actualiza el banco).
    emit_event(
        EventType.QUESTION_SAVED,
        {"question_id": str(q.id), "verified_by_human": False},
    )

    return (
        f"✅ Pregunta guardada en el banco (id: {q.id}).{review_note}\n"
        "Aún NO está marcada como verificada por humano. "
        "¿La confirmas como verificada?"
    )


@tool(args_schema=ConfirmarVerificacionInput)
def confirmar_verificacion_humana(question_id: str) -> str:
    """Marca una pregunta como VERIFICADA POR HUMANO.

    ⚠️ SOLO debe invocarse tras la confirmación explícita del profesor en el
    chat (human-in-the-loop). Una vez verificada, la pregunta queda protegida
    contra modificaciones del agente.
    """
    try:
        q = bank.set_human_verification(question_id, verified=True)
    except Exception as e:  # noqa: BLE001
        return f"❌ No se pudo verificar: {str(e)[:200]}"
    return (
        f"✅ Pregunta {q.id} marcada como **verificada por humano**. "
        "A partir de ahora no podrá ser modificada por el agente."
    )


@tool(args_schema=GuardarPreguntaPendienteInput)
def guardar_pregunta_pendiente(verificar_humano: bool = False) -> str:
    """Guarda en el banco la ÚLTIMA pregunta generada (pendiente de confirmación).

    Usa los datos de la pregunta que `generar_pregunta` dejó pendiente, así el
    LLM no tiene que re-especificar todos los campos. Si `verificar_humano=True`,
    además la marca como verificada por humano (tras confirmación del profesor).
    """
    item, subject_id = get_pending_item()
    if not item or not subject_id:
        return (
            "No hay una pregunta pendiente de guardar. Primero genera una pregunta "
            "con la tool `generar_pregunta`."
        )

    # Re-validar el pendiente contra el contrato GeneratedItem: el dict quedó
    # en memoria de módulo y debe cumplir el mismo esquema estricto.
    try:
        item_model = GeneratedItem.model_validate(item)
    except Exception as e:  # noqa: BLE001
        return (
            "⚠️ La pregunta pendiente no cumple el esquema esperado y no se guardó. "
            "Vuelve a generarla con `generar_pregunta`. "
            f"Detalle: {str(e)[:200]}"
        )
    item = item_model.model_dump()

    from core.infrastructure.embeddings.embedder import embedder
    from agents_backend.shared.verifier import review_question

    # Embedding para guardar (RAG de preguntas).
    vec = embedder.embed([item["statement"]])[0]

    # Verificación por IA antes de persistir.
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

    # Limpiar el pendiente.
    set_pending_item(None, None)

    review_note = (
        f"\n🔍 **Verificación por IA:** {'aprobada' if review.verified_by_ai else 'requiere revisión'} "
        f"(prioridad {review.priority}). {review.notes}"
        if review.notes
        else ""
    )

    # Si el profesor confirmó la verificación humana, marcarla.
    if verificar_humano:
        try:
            bank.set_human_verification(q.id, verified=True)
        except Exception as e:  # noqa: BLE001
            emit_event(
                EventType.QUESTION_SAVED,
                {"question_id": str(q.id), "verified_by_human": False},
            )
            return (
                f"✅ Pregunta guardada (id: {q.id}).{review_note}\n"
                f"⚠️ No se pudo marcar como verificada: {str(e)[:120]}"
            )
        emit_event(
            EventType.QUESTION_SAVED,
            {"question_id": str(q.id), "verified_by_human": True},
        )
        return (
            f"✅ Pregunta guardada y **verificada por humano** (id: {q.id}).{review_note}\n"
            "A partir de ahora no podrá ser modificada por el agente."
        )

    emit_event(
        EventType.QUESTION_SAVED,
        {"question_id": str(q.id), "verified_by_human": False},
    )
    return (
        f"✅ Pregunta guardada en el banco (id: {q.id}).{review_note}\n"
        "Aún NO está marcada como verificada por humano. "
        "¿La confirmas como verificada?"
    )


@tool(args_schema=ListarMateriasInput)
def listar_materias() -> str:
    """Lista las materias registradas en el sistema.

    Úsala cuando el profesor pregunte qué materias existen o cuáles puede
    usar. Devuelve el id y nombre de cada materia registrada.
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject

    db = SessionLocal()
    try:
        subjects = db.query(Subject).all()
    finally:
        db.close()

    if not subjects:
        return "No hay materias registradas todavía. Puedes registrar una indicándome su nombre."

    lines = ["📚 Materias registradas:"]
    for s in subjects:
        lines.append(f"- **{s.name}** (id: `{s.id}`)")
    lines.append("\nIndícame el id de la materia sobre la que quieras trabajar.")
    return "\n".join(lines)


@tool(args_schema=GenerarExamenPdfInput)
def generar_examen_pdf(
    subject_id: str,
    subject_name: str,
    total_questions: int = 5,
    use_bank: bool = True,
    num_versions: int = 1,
) -> str:
    """Genera un examen a partir del banco de preguntas y devuelve un PDF.

    Construye un examen usando las preguntas YA EXISTENTES del banco de la
    materia (si `use_bank=True`) o generando nuevas. Compila el LaTeX a PDF
    y lo sube como adjunto a Open WebUI.

    Si `num_versions > 1`, genera varias versiones del examen (preguntas y
    opciones barajadas, estilo AMC anti-copia) y adjunta cada PDF.
    """
    from agents_backend.exam_pipeline.graph import run_exam_pipeline
    from core.domain.schemas.exam import (
        BloomLevel,
        Difficulty,
        ExamBlueprint,
    )
    from core.infrastructure.latex.templates.registry import TemplateId

    # Recuperar preguntas del banco para armar el examen (reutilizar, no generar).
    questions = bank.list_questions(
        subject_id=subject_id,
        limit=total_questions,
    )
    if not questions:
        return (
            f"No hay preguntas en el banco para la materia '{subject_name}'. "
            "Primero genera y guarda algunas preguntas, o pide generar un examen "
            "con preguntas nuevas."
        )

    # Construir un blueprint que pida exactamente `total_questions` preguntas,
    # distribuidas uniformemente (el pipeline las generará con RAG del material).
    blueprint = ExamBlueprint(
        subject_id=subject_id,
        total_questions=total_questions,
        bloom_distribution={
            BloomLevel.RECORDAR: 0.25,
            BloomLevel.COMPRENDER: 0.25,
            BloomLevel.APLICAR: 0.25,
            BloomLevel.ANALIZAR: 0.25,
        },
        difficulty_distribution={
            Difficulty.FACIL: 0.4,
            Difficulty.MEDIO: 0.4,
            Difficulty.DIFICIL: 0.2,
        },
        topics_weight={"General": 1.0},
        estimated_duration_minutes=60,
    )
    blueprint.metadata.num_versions = num_versions

    try:
        result = run_exam_pipeline(
            subject_id=subject_id,
            subject_name=subject_name,
            template_id=TemplateId.BASE_EXAM,
            raw_materials_paths=[],
            blueprint=blueprint,
            persist=False,  # no duplicar: las preguntas ya están en el banco
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Error al generar el examen: {str(e)[:300]}"

    # Rutas de los PDFs (varias si num_versions>1).
    pdf_paths = None
    if result.get("compiled_exam"):
        pdf_paths = result["compiled_exam"].pdf_paths or (
            [result["compiled_exam"].pdf_path] if result["compiled_exam"].pdf_path else None
        )

    if not pdf_paths:
        return (
            "No se pudo compilar el PDF del examen. "
            f"Status: {result.get('status')} · Error: {result.get('error_message', 'desconocido')}"
        )

    # Eventos tipados: un artifact_ready por cada versión del PDF.
    for i, p in enumerate(pdf_paths):
        emit_event(
            EventType.ARTIFACT_READY,
            {
                "kind": "pdf",
                "url": f"/output/{Path(p).name}",
                "filename": Path(p).name,
                "version": i + 1,
            },
        )

    # Subir cada PDF a Open WebUI como adjunto nativo del chat.
    try:
        from exam_app.infrastructure.openwebui_bridge import upload_file_to_openwebui, build_attachment_markdown

        attachments = []
        for pdf_path in pdf_paths:
            upload_result = upload_file_to_openwebui(pdf_path)
            attachments.append(build_attachment_markdown(upload_result))
        attachment_md = "\n\n".join(attachments)
    except Exception as e:  # noqa: BLE001
        # Si falla la subida, caemos al enlace HTTP directo.
        links = "\n".join(
            f"- 📥 [Descargar PDF v{i + 1}](http://localhost:8000/output/{Path(p).name})"
            for i, p in enumerate(pdf_paths)
        )
        attachment_md = (
            f"📄 **Examen generado y compilado a PDF.**\n"
            f"- Materia: **{subject_name}**\n"
            f"- Preguntas: {len(result.get('approved_items', []))}\n"
            f"{links}\n\n"
            f"*(No se pudo adjuntar a Open WebUI: {str(e)[:120]})*"
        )

    version_note = f" · Versiones: {num_versions}" if num_versions > 1 else ""
    return (
        f"✅ **Examen generado correctamente.**\n"
        f"- Materia: **{subject_name}**\n"
        f"- Preguntas: {len(result.get('approved_items', []))}{version_note}\n\n"
        f"{attachment_md}"
    )


@tool(args_schema=ExportarPreguntasInput)
def exportar_preguntas(
    subject_id: str,
    formato: str = "csv",
    topic: str | None = None,
    limit: int = 100,
) -> str:
    """EXPORTA las preguntas del banco a un archivo CSV o Excel (xlsx).

    Úsala cuando el profesor pida 'exporta las preguntas', 'descárgamelas en
    Excel/CSV' o quiera llevar el banco a una hoja de cálculo. Genera el
    archivo con TODAS las columnas del banco (enunciado, opciones, respuesta,
    solución, verificación, etc.) y lo sube como adjunto descargable al chat.

    Args:
        subject_id: id de la materia (UUID).
        formato: 'csv' o 'xlsx' (Excel).
        topic: filtro opcional por tema exacto.
        limit: nº máximo de preguntas a exportar (1-500).
    """
    import csv
    import io
    from datetime import datetime

    from exam_app.infrastructure.openwebui_bridge import upload_file_to_openwebui, build_attachment_markdown

    fmt = (formato or "csv").strip().lower()
    if fmt not in ("csv", "xlsx"):
        fmt = "csv"

    questions = bank.list_questions(
        subject_id=subject_id,
        topic=topic,
        limit=limit,
    )
    if not questions:
        return (
            f"No hay preguntas en el banco para exportar "
            f"(tema: '{topic or 'todas'}')."
        )

    # Columnas completas del banco (en orden legible para una hoja de cálculo).
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
        """Valor plano de una celda (sin markdown, para CSV/Excel)."""
        if column == TableColumn.N:
            return ""
        if column == TableColumn.VERIFIED_BY_HUMAN:
            return "sí" if q.verified_by_human else "no"
        if column == TableColumn.VERIFIED_BY_AI:
            return "sí" if q.verified_by_ai else "no"
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

    # Directorio de salida (compartido con latex-compiler y servido por /output).
    from core.infrastructure.files.output_paths import get_output_dir

    OUTPUT_DIR = get_output_dir()

    try:
        if fmt == "xlsx":
            from openpyxl import Workbook

            wb = Workbook()
            ws = wb.active
            ws.title = "Preguntas"
            ws.append(headers)
            for row in rows:
                ws.append(row)
            # Ajustar ancho de columnas (cap a 60 chars para no deformar).
            for col_idx, h in enumerate(headers, start=1):
                width = max((len(str(row[col_idx - 1])) for row in rows), default=10)
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max(width + 2, 10), 60)
            out_path = OUTPUT_DIR / f"preguntas_{stamp}.xlsx"
            wb.save(out_path)
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:
            buf = io.StringIO()
            writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
            writer.writerow(headers)
            writer.writerows(rows)
            out_path = OUTPUT_DIR / f"preguntas_{stamp}.csv"
            out_path.write_text("\ufeff" + buf.getvalue(), encoding="utf-8")  # BOM para Excel
            mime = "text/csv"
    except Exception as e:  # noqa: BLE001
        return f"❌ No pude generar el archivo de exportación: {str(e)[:200]}"

    # Evento tipado: card de artefacto descargable en el frontend.
    emit_event(
        EventType.ARTIFACT_READY,
        {
            "kind": "xlsx" if fmt == "xlsx" else "csv",
            "url": f"/output/{out_path.name}",
            "filename": out_path.name,
        },
    )

    # Subir el archivo a Open WebUI como adjunto nativo del chat.
    try:
        upload = upload_file_to_openwebui(out_path, mime_type=mime)
        titulo = (
            f"📊 **Exportación de {len(questions)} pregunta(s) en Excel adjunta al chat.**"
            if fmt == "xlsx"
            else f"📊 **Exportación de {len(questions)} pregunta(s) en CSV adjunta al chat.**"
        )
        etiqueta = "Descargar el Excel" if fmt == "xlsx" else "Descargar el CSV"
        attachment_md = build_attachment_markdown(upload, titulo=titulo, etiqueta=etiqueta)
    except Exception as e:  # noqa: BLE001
        # Fallback: enlace HTTP directo servido por la API.
        attachment_md = (
            f"📥 [Descargar {out_path.name}](http://localhost:8000/output/{out_path.name})\n\n"
            f"*(No se pudo adjuntar a Open WebUI: {str(e)[:120]})*"
        )

    return (
        f"✅ Exporté **{len(questions)} pregunta(s)** del banco "
        f"(formato: {fmt.upper()}).\n\n{attachment_md}"
    )


@tool(args_schema=AgregarMaterialArchivoInput)
def agregar_material_archivo(
    file_id: str,
    subject_id: str,
    subject_name: str,
    clear_previous: bool = False,
) -> str:
    """Agrega un archivo adjunto del chat como material de estudio de una materia.

    Descarga el archivo subido a Open WebUI (identificado por su `file_id`),
    extrae su texto, lo divide en fragmentos e indexa esos fragmentos en la
    materia indicada (para que el agente pueda usarlos como contexto RAG al
    generar preguntas o examenes).

    Args:
        file_id: id del archivo adjunto en Open WebUI (viene en el tag
            `<file id="..."/>` del mensaje del usuario).
        subject_id: id de la materia a la que agregar el material.
        subject_name: nombre de la materia (para confirmación).
        clear_previous: si True, reemplaza el material previo de ese archivo.
    """
    from core.domain.bank.material import index_text_for_subject
    from exam_app.infrastructure.openwebui_bridge import get_file_metadata, download_file_content

    # 1) Obtener metadata del archivo (nombre, tipo).
    try:
        meta = get_file_metadata(file_id)
    except Exception as e:  # noqa: BLE001
        return (
            f"❌ No pude acceder al archivo adjunto (id: {file_id}). "
            f"Detalle: {str(e)[:200]}"
        )

    filename = meta.get("filename", "archivo")
    data = meta.get("data", {}) or {}
    # Open WebUI ya extrae el contenido textual del archivo en `data.content`.
    content = data.get("content") or ""

    # 2) Descargar el binario del archivo (para guardarlo físicamente en el
    #    repositorio de material de la materia).
    raw_bytes: bytes | None = None
    try:
        raw_bytes = download_file_content(file_id)
    except Exception:  # noqa: BLE001
        raw_bytes = None

    # 3) Si Open WebUI no extrajo el contenido, extraerlo del binario.
    if not content.strip() and raw_bytes:
        try:
            from core.domain.bank.material import extract_file_text

            tmp_path = Path(f"/tmp/owui_{file_id}")
            tmp_path.write_bytes(raw_bytes)
            content = extract_file_text(tmp_path)
            tmp_path.unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            return (
                f"❌ No pude extraer el contenido del archivo '{filename}'. "
                f"Detalle: {str(e)[:200]}"
            )

    if not content.strip():
        return (
            f"⚠️ El archivo '{filename}' no contiene texto extraíble "
            "(puede ser una imagen escaneada o un formato no soportado)."
        )

    # 4) Guardar el archivo físico en el directorio de material de la materia.
    saved_path = None
    if raw_bytes:
        try:
            from core.domain.bank.material import save_file_to_subject_folder

            saved_path = save_file_to_subject_folder(raw_bytes, filename, subject_name)
        except Exception as e:  # noqa: BLE001
            saved_path = None  # no bloquear la indexación si falla el guardado físico

    # 5) Indexar el contenido en la materia (RAG).
    try:
        res = index_text_for_subject(
            text=content,
            subject_id=subject_id,
            source_label=f"chat:{filename}",
            clear_previous=clear_previous,
        )
    except Exception as e:  # noqa: BLE001
        return f"❌ Error al indexar el material: {str(e)[:200]}"

    if res.get("chunks_guardados", 0) == 0:
        return f"⚠️ No se pudo indexar el archivo '{filename}': {res.get('mensaje', '')}"

    # Evento tipado: toast de progreso/confirmación en el frontend.
    emit_event(
        EventType.MATERIAL_INDEXED,
        {
            "chunks": res["chunks_guardados"],
            "filename": filename,
            "subject_name": subject_name,
        },
    )

    # Construir la confirmación (incluye si se guardó el archivo físico).
    lines = [
        f"✅ Material agregado a la materia **{subject_name}**.",
        f"- Archivo: **{filename}**",
        f"- Fragmentos indexados: **{res['chunks_guardados']}**",
    ]
    if saved_path:
        lines.append(f"- 📁 Archivo guardado en el repositorio de material: `{saved_path}`")
    else:
        lines.append("- ⚠️ El archivo físico no se pudo guardar en el repositorio (solo se indexó el texto en la BD).")
    lines.append("\nEl agente ahora puede usar este material como contexto al generar preguntas o exámenes de la materia.")
    return "\n".join(lines)


@tool(args_schema=ListarMaterialInput)
def listar_material_materia(subject_id: str) -> str:
    """Lista el material de estudio indexado de una materia.

    Úsala cuando el profesor pregunte qué material hay disponible para una
    materia, o para verificar si un archivo quedó indexado. Devuelve los
    archivos fuente y el número de fragmentos indexados de cada uno.
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject, MaterialChunk

    db = SessionLocal()
    try:
        subject = db.get(Subject, subject_id)
        if subject is None:
            return f"No existe la materia con id `{subject_id}`."

        rows = (
            db.query(MaterialChunk.source_file, MaterialChunk.subject_id)
            .filter(MaterialChunk.subject_id == subject.id)
            .all()
        )
    finally:
        db.close()

    if not rows:
        return (
            f"La materia **{subject.name}** no tiene material indexado todavía. "
            "Puedes subir un archivo adjunto al chat y pedirme que lo agregue."
        )

    # Agrupar por archivo fuente.
    from collections import Counter

    counts = Counter(source for source, _ in rows)

    lines = [f"📚 Material indexado de **{subject.name}**:\n"]
    for source, count in sorted(counts.items()):
        lines.append(f"- **{source}** · {count} fragmento(s)")
    lines.append("\nEste material se usa como contexto RAG para generar preguntas y exámenes.")
    return "\n".join(lines)


@tool(args_schema=RegistrarMateriaInput)
def registrar_materia(nombre: str) -> str:
    """Registra una NUEVA materia en el sistema.

    Úsala cuando el profesor pida registrar una materia que aún no existe
    (por ejemplo 'registra la materia Economía Aplicada'). Crea la materia
    y devuelve su id para poder trabajar con ella.

    Args:
        nombre: nombre de la materia a registrar (ej. 'Economía Aplicada').
    """
    from core.infrastructure.db.connection import SessionLocal
    from core.domain.models import Subject, MaterialChunk

    nombre = nombre.strip()
    if not nombre:
        return "⚠️ Debes indicar el nombre de la materia a registrar."

    db = SessionLocal()
    try:
        # Evitar duplicados por nombre (case-insensitive).
        existing = db.query(Subject).filter(Subject.name.ilike(nombre)).first()
        if existing is not None:
            return (
                f"ℹ️ La materia **{existing.name}** ya está registrada "
                f"(id: `{existing.id}`). No se creó un duplicado."
            )
        subject = Subject(name=nombre)
        db.add(subject)
        db.commit()
        db.refresh(subject)
        emit_event(
            EventType.SUBJECT_REGISTERED,
            {"subject_id": str(subject.id), "subject_name": subject.name},
        )
        return (
            f"✅ Materia **{subject.name}** registrada correctamente.\n"
            f"- Id: `{subject.id}`\n\n"
            "Ya puedes agregar material de estudio, generar preguntas o armar "
            "exámenes de esta materia."
        )
    finally:
        db.close()



