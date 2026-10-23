"""Implementation of the LangGraph graph nodes.

Each function receives the full state and returns a partial dict that
LangGraph merges into the state (fields with the `operator.add` reducer
are concatenated; the rest are overwritten).
"""
import random
import time
import uuid
from pathlib import Path
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
    """Samples `n` items from a {key: weight} dict (non-normalized)."""
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]
    return random.choices(keys, weights=probs, k=n)


def _distribute(weights: dict, total: int) -> dict:
    """Converts proportions (summing ~1) into integer counts summing `total`."""
    counts = {k: int(round(w * total)) for k, w in weights.items()}
    # Fix the rounding error so they sum exactly `total`.
    diff = total - sum(counts.values())
    if diff != 0:
        keys = sorted(counts, key=lambda k: counts[k], reverse=True)
        for i in range(abs(diff)):
            counts[keys[i % len(keys)]] += 1 if diff > 0 else -1
    return counts


def _extract_error_excerpt(e: Exception) -> str:
    return str(e)[:300]


# ---------------------------------------------------------------------------
# Text extraction per file type (for the Ingestor)
# ---------------------------------------------------------------------------
def _extract_pdf(path: str) -> str:
    """Extracts text from a PDF with PyMuPDF."""
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
    """Extracts text from a .docx (paragraphs + tables)."""
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
    """Extracts text from a .pptx (text of each shape in each slide)."""
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
    """Extracts text from a .xlsx (all sheets, non-empty cells)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        sheet_parts = [f"--- Sheet: {ws.title} ---"]
        for row in ws.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                sheet_parts.append(" | ".join(cells))
        if len(sheet_parts) > 1:
            parts.append("\n".join(sheet_parts))
    wb.close()
    return "\n\n".join(parts)


def _extract_text_file(path: str) -> str:
    """Extracts text from a plain text file (.txt, .md, etc.)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# 1. Ingestor
# ---------------------------------------------------------------------------
def ingestor_node(state: dict) -> dict:
    """Reads the raw materials and produces a syllabus summary.

    For the MVP it doesn't call the LLM: extracts text from PDFs, DOCX, PPTX,
    XLSX and text files, and builds a basic summary. (Full vector indexing is
    a later phase.)
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
                # Unknown format: try as plain text.
                content = _extract_text_file(path)

            if content.strip():
                fragments.append(content.strip())
                # Chunking heuristic: ~1 chunk per ~2000 characters
                chunks_count += max(1, len(content) // 2000)
                processed.append(path)
        except Exception as e:  # noqa: BLE001
            failed.append(path)
            fragments.append(f"[Could not read {path}: {_extract_error_excerpt(e)}]")

    syllabus_summary = (
        "\n\n".join(fragments)[:6000] if fragments else "No materials loaded."
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
    """Breaks the exam blueprint into an explicit list of `ItemSpec`.

    Done programmatically (weighted sampling of the distributions), without
    spending an LLM call: cheaper, deterministic and easier to defend as
    methodology in the thesis.
    """
    blueprint: ExamBlueprint = state["blueprint"]
    total = blueprint.total_questions

    # Counts per Bloom level, difficulty and topic.
    bloom_counts = _distribute(blueprint.bloom_distribution, total)
    diff_counts = _distribute(blueprint.difficulty_distribution, total)
    topic_counts = _distribute(blueprint.topics_weight, total)

    topics = [t for t, c in topic_counts.items() for _ in range(c)]

    pending: list[ItemSpec] = []
    spec_index = 0
    for bloom, n_bloom in bloom_counts.items():
        for _ in range(n_bloom):
            topic = topics[spec_index % len(topics)] if topics else "general"
            # Difficulty distribution within the current Bloom level.
            difficulty = _weighted_sample(diff_counts, 1)[0]
            # Question type: lower Bloom levels favor multiple choice.
            if bloom in (BloomLevel.REMEMBER, BloomLevel.UNDERSTAND):
                qtype = random.choice(
                    [QuestionType.MULTIPLE_CHOICE, QuestionType.TRUE_FALSE]
                )
            elif bloom == BloomLevel.CREATE:
                qtype = QuestionType.ESSAY
            else:
                qtype = random.choice(
                    [QuestionType.MULTIPLE_CHOICE, QuestionType.SHORT_ANSWER, QuestionType.ESSAY]
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
    """Takes the next pending spec and generates a `GeneratedItem` with the LLM."""
    if not state.get("pending_specs"):
        return {}

    spec: ItemSpec = state["pending_specs"][0]
    remaining = state["pending_specs"][1:]

    # RAG config from the blueprint (retrieval config)
    blueprint: ExamBlueprint = state["blueprint"]
    retrieval_cfg = blueprint.retrieval

    # RAG: retrieve chunks relevant to the spec topic from the DB (pgvector).
    # If the subject is indexed, use the most relevant fragments instead of
    # the static summary. If there are no chunks, fall back to the syllabus
    # summary.
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
        context = f"Syllabus context (chunk):\n{state.get('syllabus_summary', '')[:4000]}"

    prompt = (
        "You are a university exam item generator. Generate ONE question "
        "that exactly complies with this specification:\n"
        f"- Topic: {spec.topic}\n"
        f"- Subtopic: {spec.subtopic or 'not specified'}\n"
        f"- Bloom level: {spec.bloom_level.value}\n"
        f"- Difficulty: {spec.difficulty.value}\n"
        f"- Question type: {spec.question_type.value}\n"
        f"- Points: {spec.points}\n\n"
        f"{context}\n\n"
        "Return only the requested structure."
    )

    try:
        structured = generator_llm.with_structured_output(GeneratedItem)
        started_at = time.perf_counter()
        item = structured.invoke(prompt)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
    except Exception as e:  # noqa: BLE001
        # Structured log: silent rejections hide pipeline problems.
        from core.domain.llm_log.service import record_llm_response

        record_llm_response(
            context="pipeline:generator_error",
            model=settings.llm_model,
            prompt_messages=[{"role": "user", "content": prompt}],
            raw_answer=f"generation_error: {str(e)[:300]}",
            latency_ms=None,
        )
        # If the LLM fails, log the failure as a rejection and move on.
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

    # Raw audit (best-effort): prompt + raw answer of the generator call.
    from core.domain.llm_log.service import record_llm_response

    record_llm_response(
        context="pipeline:generator",
        model=settings.llm_model,
        prompt_messages=[{"role": "user", "content": prompt}],
        raw_answer=item.model_dump_json() if item is not None else "",
        latency_ms=latency_ms,
    )

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
    """Approximate similarity by token overlap (placeholder).

    The MVP uses a simple heuristic; embedding similarity (pgvector) is a
    later improvement for the experimentation chapter.
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def evaluator_node(state: dict) -> dict:
    """Evaluates the generated item: quality, difficulty match and duplication."""
    item: GeneratedItem | None = state.get("current_item")
    if item is None:
        return {}

    spec: ItemSpec = state["current_spec"]

    # 1) Duplicate check against already approved items.
    max_sim = 0.0
    for approved in state.get("approved_items", []):
        sim = _semantic_similarity(item.statement, approved.statement)
        max_sim = max(max_sim, sim)
    is_duplicate = max_sim >= settings.dedup_similarity_threshold

    # 2) Quality evaluation with the LLM.
    prompt = (
        "You are a rigorous exam item evaluator. Evaluate the following "
        "question and respond with the requested structure.\n\n"
        f"Requested specification:\n- Topic: {spec.topic}\n"
        f"- Bloom level: {spec.bloom_level.value}\n"
        f"- Difficulty: {spec.difficulty.value}\n"
        f"- Type: {spec.question_type.value}\n\n"
        f"Generated item:\n{item.model_dump_json()}"
    )
    try:
        structured = evaluator_llm.with_structured_output(EvaluationResult)
        started_at = time.perf_counter()
        result: EvaluationResult = structured.invoke(prompt)
        latency_ms = int((time.perf_counter() - started_at) * 1000)
    except Exception as e:  # noqa: BLE001
        # Structured log: silent rejections hide pipeline problems.
        from core.domain.llm_log.service import record_llm_response

        record_llm_response(
            context="pipeline:evaluator_error",
            model=settings.llm_model,
            prompt_messages=[{"role": "user", "content": prompt}],
            raw_answer=f"evaluation_error: {str(e)[:300]}",
            latency_ms=None,
        )
        result = EvaluationResult(
            approved=False,
            quality_score=0.0,
            difficulty_match=False,
            rejection_reason=f"evaluation_error: {_extract_error_excerpt(e)}",
        )

    # Raw audit (best-effort): prompt + raw answer of the evaluator call.
    from core.domain.llm_log.service import record_llm_response

    record_llm_response(
        context="pipeline:evaluator",
        model=settings.llm_model,
        prompt_messages=[{"role": "user", "content": prompt}],
        raw_answer=result.model_dump_json(),
        latency_ms=latency_ms,
    )

    # 3) Combine: duplication overrides any approval.
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
    """Assembles the `CompiledExam` and generates the final LaTeX.

    If the blueprint requests multiple versions (`metadata` with
    `num_versions`), generates multiple LaTeX sources (shuffled questions and
    options, AMC anti-copy style). Otherwise, generates a single version
    (previous behavior).

    Each version is rendered TWICE: with answers (instructor copy) and
    without answers (student copy). The compiler node packages both sets
    into a ZIP artifact.
    """
    blueprint: ExamBlueprint = state["blueprint"]
    items: list[GeneratedItem] = state.get("approved_items", [])

    exam = CompiledExam(
        subject_id=state["subject_id"],
        blueprint=blueprint,
        items=items,
    )

    # Number of versions from the blueprint (default 1).
    num_versions = getattr(blueprint.metadata, "num_versions", 1) or 1

    rendered_with = render_exam(
        exam,
        state["template_id"],
        state["subject_name"],
        instructions=blueprint.metadata.instructions or "Answer all questions in the space provided.",
        duration_minutes=blueprint.estimated_duration_minutes,
        print_answers=True,
        num_versions=num_versions,
    )
    rendered_without = render_exam(
        exam,
        state["template_id"],
        state["subject_name"],
        instructions=blueprint.metadata.instructions or "Answer all questions in the space provided.",
        duration_minutes=blueprint.estimated_duration_minutes,
        print_answers=False,
        num_versions=num_versions,
    )

    # Normalize to lists (render_exam returns str if num_versions<=1).
    with_answers = [rendered_with] if isinstance(rendered_with, str) else rendered_with
    without_answers = [rendered_without] if isinstance(rendered_without, str) else rendered_without

    # Interleave: v1_with, v1_without, v2_with, v2_without, ... so the
    # compiler can name the PDFs by position (2*i = with, 2*i+1 = without).
    latex_sources = []
    for i in range(num_versions):
        latex_sources.append(with_answers[i])
        latex_sources.append(without_answers[i])

    return {
        "compiled_exam": exam,
        "latex_source": latex_sources[0],          # compatibility: first version
        "latex_sources": latex_sources,            # with+without per version
    }


# ---------------------------------------------------------------------------
# 6. LaTeX compiler
# ---------------------------------------------------------------------------
def compiler_node(state: dict) -> dict:
    """Sends the LaTeX to the compiler microservice and saves the resulting PDF.

    If there are multiple versions (`latex_sources`), compiles each one and
    returns a list of PDF paths (`pdf_paths`). Otherwise, compiles the single
    version (previous behavior with `pdf_path`).

    The assembler interleaves with/without answers per version
    (2*i = with answers, 2*i+1 = without). After compiling, this node
    renames the PDFs to `<job_id>_v<N>_with_answers.pdf` /
    `<job_id>_v<N>_no_answers.pdf` and packages everything into a single
    ZIP artifact (`<job_id>.zip`) for download.
    """
    latex_sources = state.get("latex_sources") or []
    if not latex_sources:
        single = state.get("latex_source")
        if single:
            latex_sources = [single]
        else:
            return {
                "status": "failed",
                "error_message": "There is no latex_source to compile",
            }

    # Fail fast: an exam with zero approved items must NOT be compiled or
    # published (it used to produce empty "completed" PDFs with 0 questions).
    if not state.get("approved_items"):
        return {
            "status": "failed",
            "error_message": "No items were approved; nothing to compile.",
            "compilation_log": [
                {
                    "attempt_number": state.get("compilation_attempts", 0) + 1,
                    "version": 0,
                    "success": False,
                    "error_excerpt": "empty_approved_items",
                }
            ],
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

    # Rename the compiled PDFs to meaningful names and build the ZIP.
    # The compiler returns /app/output/<job_id>.pdf (random UUID names);
    # the professor-requested exam name makes the downloads intuitive:
    # <Name>_A.pdf, <Name>_A_respondido.pdf, ..., <Name>.zip
    zip_path: str | None = None
    if all_success and pdf_paths:
        try:
            from core.infrastructure.files.naming import slugify, version_letter
            from core.infrastructure.files.output_paths import get_output_dir

            output_dir = get_output_dir()
            exam = state.get("compiled_exam")
            exam_name = (
                getattr(getattr(exam, "blueprint", None), "metadata", None).name
                if exam is not None and getattr(getattr(exam, "blueprint", None), "metadata", None).name
                else state.get("subject_name") or "exam"
            )
            base = slugify(exam_name) or "exam"
            num_versions = max(1, len(pdf_paths) // 2)
            renamed: list[str] = []
            for i, p in enumerate(pdf_paths):
                src = Path(p)
                if not src.exists():
                    renamed.append(p)
                    continue
                version = version_letter(i // 2)
                suffix = "respondido" if i % 2 == 0 else ""
                stem = f"{base}_{version}" + (f"_{suffix}" if suffix else "")
                dest = output_dir / f"{stem}.pdf"
                # Avoid overwriting an existing artifact from a previous run
                # with the same name: append a numeric suffix.
                if dest.exists():
                    n = 2
                    while (output_dir / f"{stem}_{n}.pdf").exists():
                        n += 1
                    dest = output_dir / f"{stem}_{n}.pdf"
                src.replace(dest)
                renamed.append(str(dest))

            # ZIP with all the PDFs (with and without answers).
            import zipfile

            zip_dest = output_dir / f"{base}.zip"
            if zip_dest.exists():
                n = 2
                while (output_dir / f"{base}_{n}.zip").exists():
                    n += 1
                zip_dest = output_dir / f"{base}_{n}.zip"
            with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in renamed:
                    zf.write(p, arcname=Path(p).name)
            zip_path = str(zip_dest)

            # Persist the exam in the DB (professor request): one row per
            # generation, with the blueprint, items and artifact names.
            try:
                import json as jsonlib
                from datetime import datetime

                from core.domain.models import Exam as ExamModel
                from core.infrastructure.db.connection import SessionLocal

                exam = state.get("compiled_exam")
                db = SessionLocal()
                try:
                    db.add(ExamModel(
                        subject_id=uuid.UUID(str(state.get("subject_id"))),
                        name=exam_name,
                        blueprint=(
                            exam.blueprint.model_dump(mode="json")
                            if exam is not None else {}
                        ),
                        items=[
                            (item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
                            for item in ((exam.items if exam is not None else []) or [])
                        ],
                        num_versions=num_versions,
                        pdf_paths=[Path(p).name for p in renamed],
                        zip_path=zip_dest.name,
                        created_by=state.get("created_by") or "chat",
                    ))
                    db.commit()
                finally:
                    db.close()

                # Raw JSON artifact (blueprint + items): internal record used
                # by the Exams view to enrich the table. Not shown in the chat.
                payload = {
                    "subject_id": state.get("subject_id"),
                    "subject_name": state.get("subject_name"),
                    "exam_name": exam_name,
                    "generated_at": datetime.now().isoformat(),
                    "num_versions": num_versions,
                    "blueprint": (
                        exam.blueprint.model_dump(mode="json")
                        if exam is not None else {}
                    ),
                    "items": [
                        (item.model_dump(mode="json") if hasattr(item, "model_dump") else item)
                        for item in ((exam.items if exam is not None else []) or [])
                    ],
                    # Audit fields: why items were rejected and how many LLM
                    # calls the generation consumed (diagnostics).
                    "rejection_log": state.get("rejection_log") or [],
                    "total_llm_calls": state.get("total_llm_calls", 0),
                }
                (output_dir / f"{base}.json").write_text(
                    jsonlib.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:  # noqa: BLE001
                pass  # persistence/enrichment is optional; never fail the ZIP

            pdf_paths = renamed
        except Exception as e:  # noqa: BLE001
            # The ZIP is a convenience artifact: never fail the compilation.
            logs.append({"attempt_number": compilation_attempts, "version": 0, "success": True, "error_excerpt": f"zip_skipped: {str(e)[:150]}"})

    # Update the CompiledExam with the PDF paths.
    exam = state.get("compiled_exam")
    if exam is not None and pdf_paths:
        exam = exam.model_copy(
            update={"pdf_path": pdf_paths[0], "pdf_paths": pdf_paths, "compilation_attempts": compilation_attempts}
        )

    update: dict = {
        "compilation_attempts": compilation_attempts,
        "compilation_log": logs,
        "pdf_paths": pdf_paths,
        "zip_path": zip_path,
    }
    if exam is not None:
        update["compiled_exam"] = exam

    if all_success and pdf_paths:
        update["status"] = "completed"
    elif compilation_attempts >= settings.max_compilation_attempts:
        update["status"] = "failed"
        update["error_message"] = "Some version of the exam did not compile."
    else:
        update["status"] = "in_progress"
        update["error_message"] = "Some version of the exam did not compile."
    return update
