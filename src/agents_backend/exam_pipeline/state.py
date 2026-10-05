import operator
from typing import Annotated, Optional, TypedDict

from core.domain.schemas.exam import (
    ExamBlueprint,
    ItemSpec,
    GeneratedItem,
    EvaluationResult,
    CompiledExam,
)
from core.infrastructure.latex.templates.registry import TemplateId


class RejectionLogEntry(TypedDict):
    """Audit record: why an item was rejected. Key for the thesis metrics."""
    spec_id: str
    attempt_number: int
    reason: str  # "duplicate" | "low_quality" | "difficulty_mismatch" | "validation_error"
    quality_score: Optional[float]
    similarity_score: Optional[float]


class CompilationLogEntry(TypedDict):
    """Audit record: LaTeX compilation attempts and their results."""
    attempt_number: int
    success: bool
    error_excerpt: Optional[str]


class ExamGenerationState(TypedDict):
    # --- Input context (set at the start, unchanged during the graph) ---
    subject_id: str
    subject_name: str
    template_id: TemplateId
    raw_materials_paths: list[str]  # paths to the documents uploaded by the professor

    # --- Ingestor Agent output ---
    syllabus_summary: Optional[str]
    indexed_chunks_count: int

    # --- Planner Agent output ---
    blueprint: Optional[ExamBlueprint]
    pending_specs: list[ItemSpec]        # queue of "slots" to generate (drained)

    # --- Current item state in the generation loop ---
    current_spec: Optional[ItemSpec]
    current_item: Optional[GeneratedItem]
    current_evaluation: Optional[EvaluationResult]
    item_retry_count: int                # retries of the current item
    max_item_retries: int                # limit before abandoning that spec

    # --- Accumulators (use operator.add reducer: each node APPENDS, never overwrites) ---
    approved_items: Annotated[list[GeneratedItem], operator.add]
    rejection_log: Annotated[list[RejectionLogEntry], operator.add]

    # --- LaTeX assembly and compilation ---
    latex_source: Optional[str]
    latex_sources: Optional[list[str]]   # LaTeX sources of the versions (if num_versions>1)
    compiled_exam: Optional[CompiledExam]
    pdf_paths: Optional[list[str]]       # paths of the version PDFs
    compilation_attempts: int
    max_compilation_attempts: int
    compilation_log: Annotated[list[CompilationLogEntry], operator.add]

    # --- Global safety controls ---
    total_llm_calls: int                 # hard cap to avoid infinite loops / runaway costs
    max_total_llm_calls: int
    status: str                          # "in_progress" | "completed" | "failed"
    error_message: Optional[str]