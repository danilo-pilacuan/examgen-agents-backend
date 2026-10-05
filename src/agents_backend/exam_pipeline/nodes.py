"""Implementación de los nodos del grafo LangGraph.

Cada función recibe el estado completo y devuelve un dict parcial que
LangGraph mezcla con el estado (los campos con reducer `operator.add`
se concatenan; el resto se sobreescribe).
"""
import random
from typing import Any

import requests

from core.config import settings
from core.domain.schemas.exam import (
    BloomLevel,
    Difficulty,
    ExamBlueprint,
    EvaluationResult,
    GeneratedItem,
    ItemSpec,
    QuestionType,
    CompiledExam,
)
from agents_backend.shared.llm import generator_llm, evaluator_llm
from agents_backend.shared.retrieval import retrieve_chunks, format_context
from core.infrastructure.latex.templates.registry import TemplateId
from core.infrastructure.latex.utils import render_exam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weighted_sample(weights: dict, n: int) -> list:
    """Muestrea `n` ítems a partir de un dict {clave: peso} (no-normalizado)."""
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]
    return random.choices(keys, weights=probs, k=n)


def _distribute(weights: dict, total: int) -> dict:
    """Convierte proporciones (suman ~1) en conteos enteros que suman `total`."""
    counts = {k: int(round(w * total)) for k, w in weights.items()}
    # Corrige el error de redondeo para que sumen exactamente `total`.
    diff = total - sum(counts.values())
    if diff != 0:
        keys = sorted(counts, key=lambda k: counts[k], reverse=True)
        for i in range(abs(diff)):
            counts[keys[i % len(keys)]] += 1 if diff > 0 else -1
    return counts


def _extract_error_excerpt(e: Exception) -> str:
    return str(e)[:300]


# ---------------------------------------------------------------------------
# Extracción de texto por tipo de archivo (para el Ingestor)
# ---------------------------------------------------------------------------
def _extract_pdf(path: str) -> str:
    """Extrae texto de un PDF con PyMuPDF."""
    import fitz  # pymupdf

    doc = fitz.open(path)
    parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            parts.append(text.strip())
    doc.close()
    return "\n\n".join(parts)


def _extract_docx(path: str) -> str:
    """Extrae texto de un .docx (párrafos + tablas)."""
    from docx import Document

    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _extract_pptx(path: str) -> str:
    """Extrae texto de un .pptx (texto de cada shape en cada slide)."""
    from pptx import Presentation

    prs = Presentation(path)
    parts = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_parts = [f"--- Slide {i} ---"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        slide_parts.append(text)
            if shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        slide_parts.append(" | ".join(cells))
        if len(slide_parts) > 1:
            parts.append("\n".join(slide_parts))
    return "\n\n".join(parts)


def _extract_xlsx(path: str) -> str:
    """Extrae texto de un .xlsx (todas las hojas, celdas no vacías)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        sheet_parts = [f"--- Hoja: {ws.title} ---"]
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                sheet_parts.append(" | ".join(cells))
        if len(sheet_parts) > 1:
            parts.append("\n".join(sheet_parts))
    wb.close()
    return "\n\n".join(parts)


def _extract_text_file(path: str) -> str:
    """Extrae texto de un archivo de texto plano (.txt, .md, etc.)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# 1. Ingestor
# ---------------------------------------------------------------------------
def ingestor_node(state: dict) -> dict:
    """Lee los materiales crudos y produce un resumen del syllabus.

    Para el MVP no llama al LLM: extrae texto de PDFs, DOCX, PPTX, XLSX y
    archivos de texto, y construye un resumen básico. (El indexado vectorial
    completo es una fase posterior.)
    """
    raw_paths = state.get("raw_materials_paths", [])
    chunks_count = 0
    fragments: list[str] = []
    processed: list[str] = []
    failed: list[str] = []

    for path in raw_paths:
        path = str(path)
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        try:
            if ext == "pdf":
                content = _extract_pdf(path)
            elif ext == "docx":
                content = _extract_docx(path)
            elif ext == "pptx":
                content = _extract_pptx(path)
            elif ext == "xlsx":
                content = _extract_xlsx(path)
            elif ext in ("txt", "md", "csv"):
                content = _extract_text_file(path)
            else:
                # Formato desconocido: intentar como texto plano.
                content = _extract_text_file(path)

            if content.strip():
                fragments.append(content.strip())
                # Heurística de chunking: ~1 chunk por ~2000 caracteres
                chunks_count += max(1, len(content) // 2000)
                processed.append(path)
        except Exception as e:  # noqa: BLE001
            failed.append(path)
            fragments.append(f"[No se pudo leer {path}: {_extract_error_excerpt(e)}]")

    syllabus_summary = (
        "\n\n".join(fragments)[:6000] if fragments else "Sin materiales cargados."
    )
    return {
        "syllabus_summary": syllabus_summary,
        "indexed_chunks_count": chunks_count,
        "processed_files": processed,
        "failed_files": failed,
    }


# ---------------------------------------------------------------------------
# 2. Planificador
# ---------------------------------------------------------------------------
def planner_node(state: dict) -> dict:
    """Descompone el blueprint del examen en una lista explícita de `ItemSpec`.

    Se hace programáticamente (muestreo ponderado de las distribuciones),
    sin gastar una llamada al LLM: más barato, determinista y fácil de
    defender como metodología en la tesis.
    """
    blueprint: ExamBlueprint = state["blueprint"]
    total = blueprint.total_questions

    # Conteos por nivel de Bloom, dificultad y tema.
    bloom_counts = _distribute(blueprint.bloom_distribution, total)
    diff_counts = _distribute(blueprint.difficulty_distribution, total)
    topic_counts = _distribute(blueprint.topics_weight, total)

    topics = [t for t, c in topic_counts.items() for _ in range(c)]

    pending: list[ItemSpec] = []
    spec_index = 0
    for bloom, n_bloom in bloom_counts.items():
        for _ in range(n_bloom):
            topic = topics[spec_index % len(topics)] if topics else "general"
            # Distribución de dificultad dentro del nivel de Bloom actual.
            difficulty = _weighted_sample(diff_counts, 1)[0]
            # Tipo de pregunta: los niveles bajos favorecen opción múltiple.
            if bloom in (BloomLevel.RECORDAR, BloomLevel.COMPRENDER):
                qtype = random.choice(
                    [QuestionType.OPCION_MULTIPLE, QuestionType.VERDADERO_FALSO]
                )
            elif bloom == BloomLevel.CREAR:
                qtype = QuestionType.DESARROLLO
            else:
                qtype = random.choice(
                    [QuestionType.OPCION_MULTIPLE, QuestionType.RESPUESTA_CORTA, QuestionType.DESARROLLO]
                )
            pending.append(
                ItemSpec(
                    topic=topic,
                    bloom_level=bloom,
                    difficulty=difficulty,
                    question_type=qtype,
                    points=1.0,
                )
            )
            spec_index += 1

    return {"pending_specs": pending}


# ---------------------------------------------------------------------------
# 3. Generador
# ---------------------------------------------------------------------------
def generator_node(state: dict) -> dict:
    """Toma el siguiente spec pendiente y genera un `GeneratedItem` con el LLM."""
    if not state.get("pending_specs"):
        return {}

    spec: ItemSpec = state["pending_specs"][0]
    remaining = state["pending_specs"][1:]

    # Configuración RAG desde el blueprint (retrieval config)
    blueprint: ExamBlueprint = state["blueprint"]
    retrieval_cfg = blueprint.retrieval

    # RAG: recuperar chunks relevantes al tema del spec desde la BD (pgvector).
    # Si la materia está indexada, usamos los fragmentos más relevantes en vez
    # del resumen estático. Si no hay chunks, caemos al resumen del syllabus.
    context = ""
    if retrieval_cfg.use_rag:
        try:
            chunks = retrieve_chunks(
                query=f"{spec.topic} {spec.subtopic or ''}",
                subject_id=state["subject_id"],
                top_k=retrieval_cfg.top_k,
                min_score=retrieval_cfg.min_score,
            )
            if chunks:
                context = format_context(chunks, max_chars_per_chunk=retrieval_cfg.max_chars_per_chunk)
        except Exception:  # noqa: BLE001
            context = ""

    if not context:
        context = f"Contexto del syllabus (fragmento):\n{state.get('syllabus_summary', '')[:4000]}"

    prompt = (
        "Eres un generador de ítems de examen universitario. Genera UNA pregunta "
        "que cumpla exactamente esta especificación:\n"
        f"- Tema: {spec.topic}\n"
        f"- Subtema: {spec.subtopic or 'no especificado'}\n"
        f"- Nivel de Bloom: {spec.bloom_level.value}\n"
        f"- Dificultad: {spec.difficulty.value}\n"
        f"- Tipo de pregunta: {spec.question_type.value}\n"
        f"- Puntaje: {spec.points}\n\n"
        f"{context}\n\n"
        "Devuelve solo la estructura solicitada."
    )

    try:
        structured = generator_llm.with_structured_output(GeneratedItem)
        item = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        # Si el LLM falla, registramos el fallo como rechazo y avanzamos.
        return {
            "current_spec": spec,
            "pending_specs": remaining,
            "current_item": None,
            "current_evaluation": EvaluationResult(
                approved=False,
                quality_score=0.0,
                difficulty_match=False,
                rejection_reason=f"generation_error: {_extract_error_excerpt(e)}",
            ),
            "total_llm_calls": state.get("total_llm_calls", 0) + 1,
        }

    return {
        "current_spec": spec,
        "pending_specs": remaining,
        "current_item": item,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 4. Evaluador
# ---------------------------------------------------------------------------
def _semantic_similarity(a: str, b: str) -> float:
    """Similitud aproximada por solapamiento de tokens (placeholder).

    El MVP usa una heurística simple; la similitud por embeddings
    (pgvector) es una mejora posterior para el capítulo de experimentación.
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def evaluator_node(state: dict) -> dict:
    """Evalúa el ítem generado: calidad, coincidencia de dificultad y duplicación."""
    item: GeneratedItem | None = state.get("current_item")
    if item is None:
        return {}

    spec: ItemSpec = state["current_spec"]

    # 1) Chequeo de duplicación contra ítems ya aprobados.
    max_sim = 0.0
    for approved in state.get("approved_items", []):
        sim = _semantic_similarity(item.statement, approved.statement)
        max_sim = max(max_sim, sim)
    is_duplicate = max_sim >= settings.dedup_similarity_threshold

    # 2) Evaluación de calidad con el LLM.
    prompt = (
        "Eres un evaluador riguroso de ítems de examen. Evalúa la pregunta "
        "siguiente y responde con la estructura solicitada.\n\n"
        f"Especificación pedida:\n- Tema: {spec.topic}\n"
        f"- Nivel de Bloom: {spec.bloom_level.value}\n"
        f"- Dificultad: {spec.difficulty.value}\n"
        f"- Tipo: {spec.question_type.value}\n\n"
        f"Ítem generado:\n{item.model_dump_json()}"
    )
    try:
        structured = evaluator_llm.with_structured_output(EvaluationResult)
        result: EvaluationResult = structured.invoke(prompt)
    except Exception as e:  # noqa: BLE001
        result = EvaluationResult(
            approved=False,
            quality_score=0.0,
            difficulty_match=False,
            rejection_reason=f"evaluation_error: {_extract_error_excerpt(e)}",
        )

    # 3) Combinar: la duplicación anula cualquier aprobación.
    if is_duplicate:
        result.approved = False
        result.is_duplicate = True
        result.similarity_score = max_sim
        result.rejection_reason = result.rejection_reason or "duplicate"

    return {
        "current_evaluation": result,
        "total_llm_calls": state.get("total_llm_calls", 0) + 1,
    }


# ---------------------------------------------------------------------------
# 5. Ensamblador
# ---------------------------------------------------------------------------
def assembler_node(state: dict) -> dict:
    """Arma el `CompiledExam` y genera el LaTeX final.

    Si el blueprint pide varias versiones (`metadata` con `num_versions`),
    genera múltiples fuentes LaTeX (preguntas y opciones barajadas, estilo
    AMC anti-copia). Si no, genera una sola versión (comportamiento previo).
    """
    blueprint: ExamBlueprint = state["blueprint"]
    items: list[GeneratedItem] = state.get("approved_items", [])

    exam = CompiledExam(
        subject_id=state["subject_id"],
        blueprint=blueprint,
        items=items,
    )

    # Nº de versiones desde el blueprint (default 1).
    num_versions = getattr(blueprint.metadata, "num_versions", 1) or 1

    rendered = render_exam(
        exam,
        state["template_id"],
        state["subject_name"],
        instructions=blueprint.metadata.instructions or "Responda todas las preguntas en el espacio indicado.",
        duration_minutes=blueprint.estimated_duration_minutes,
        print_answers=True,
        num_versions=num_versions,
    )

    # Normalizar a lista (render_exam devuelve str si num_versions<=1).
    if isinstance(rendered, str):
        latex_sources = [rendered]
    else:
        latex_sources = rendered

    return {
        "compiled_exam": exam,
        "latex_source": latex_sources[0],          # compatibilidad: primera versión
        "latex_sources": latex_sources,            # todas las versiones
    }


# ---------------------------------------------------------------------------
# 6. Compilador LaTeX
# ---------------------------------------------------------------------------
def compiler_node(state: dict) -> dict:
    """Envía el LaTeX al microservicio compilador y guarda el PDF resultante.

    Si hay varias versiones (`latex_sources`), compila cada una y devuelve
    una lista de rutas de PDFs (`pdf_paths`). Si no, compila la única
    versión (comportamiento previo con `pdf_path`).
    """
    latex_sources = state.get("latex_sources") or []
    if not latex_sources:
        single = state.get("latex_source")
        if single:
            latex_sources = [single]
        else:
            return {
                "status": "failed",
                "error_message": "No hay latex_source para compilar",
            }

    compilation_attempts = state.get("compilation_attempts", 0) + 1
    pdf_paths: list[str] = []
    logs = []
    all_success = True

    for idx, latex_source in enumerate(latex_sources):
        try:
            resp = requests.post(
                settings.latex_compiler_url,
                json={"latex_code": latex_source},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()

            if payload.get("status") == "success":
                pdf_paths.append(payload.get("pdf_path"))
                logs.append({"attempt_number": compilation_attempts, "version": idx + 1, "success": True, "error_excerpt": None})
            else:
                all_success = False
                error_excerpt = (payload.get("error_log") or "")[-300:]
                logs.append({"attempt_number": compilation_attempts, "version": idx + 1, "success": False, "error_excerpt": error_excerpt})
        except Exception as e:  # noqa: BLE001
            all_success = False
            logs.append({"attempt_number": compilation_attempts, "version": idx + 1, "success": False, "error_excerpt": _extract_error_excerpt(e)})

    # Actualizar el CompiledExam con las rutas de los PDFs.
    exam = state.get("compiled_exam")
    if exam is not None and pdf_paths:
        exam = exam.model_copy(
            update={"pdf_path": pdf_paths[0], "pdf_paths": pdf_paths, "compilation_attempts": compilation_attempts}
        )

    update: dict = {
        "compilation_attempts": compilation_attempts,
        "compilation_log": logs,
        "pdf_paths": pdf_paths,
    }
    if exam is not None:
        update["compiled_exam"] = exam

    if all_success and pdf_paths:
        update["status"] = "completed"
    elif compilation_attempts >= settings.max_compilation_attempts:
        update["status"] = "failed"
        update["error_message"] = "Alguna versión del examen no compiló."
    else:
        update["status"] = "in_progress"
        update["error_message"] = "Alguna versión del examen no compiló."
    return update
