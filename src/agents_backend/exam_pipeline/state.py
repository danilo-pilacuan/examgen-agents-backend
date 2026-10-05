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
    """Registro de auditoría: por qué se rechazó un ítem. Clave para tus métricas de tesis."""
    spec_id: str
    attempt_number: int
    reason: str  # "duplicate" | "low_quality" | "difficulty_mismatch" | "validation_error"
    quality_score: Optional[float]
    similarity_score: Optional[float]


class CompilationLogEntry(TypedDict):
    """Registro de auditoría: intentos de compilación LaTeX y sus resultados."""
    attempt_number: int
    success: bool
    error_excerpt: Optional[str]


class ExamGenerationState(TypedDict):
    # --- Contexto de entrada (se fija al inicio, no cambia durante el grafo) ---
    subject_id: str
    subject_name: str
    template_id: TemplateId
    raw_materials_paths: list[str]  # rutas a los documentos subidos por el profesor

    # --- Salida del Agente Ingestor ---
    syllabus_summary: Optional[str]
    indexed_chunks_count: int

    # --- Salida del Agente Planificador ---
    blueprint: Optional[ExamBlueprint]
    pending_specs: list[ItemSpec]        # cola de "ranuras" por generar (se va vaciando)

    # --- Estado del ítem actual en el bucle de generación ---
    current_spec: Optional[ItemSpec]
    current_item: Optional[GeneratedItem]
    current_evaluation: Optional[EvaluationResult]
    item_retry_count: int                # reintentos del ítem actual
    max_item_retries: int                # límite antes de abandonar ese spec

    # --- Acumuladores (usan reducer operator.add: cada nodo AGREGA, no sobreescribe) ---
    approved_items: Annotated[list[GeneratedItem], operator.add]
    rejection_log: Annotated[list[RejectionLogEntry], operator.add]

    # --- Ensamblado y compilación LaTeX ---
    latex_source: Optional[str]
    latex_sources: Optional[list[str]]   # fuentes LaTeX de las versiones (si num_versions>1)
    compiled_exam: Optional[CompiledExam]
    pdf_paths: Optional[list[str]]       # rutas de los PDFs de las versiones
    compilation_attempts: int
    max_compilation_attempts: int
    compilation_log: Annotated[list[CompilationLogEntry], operator.add]

    # --- Control global de seguridad ---
    total_llm_calls: int                 # cap duro para evitar loops infinitos / costos descontrolados
    max_total_llm_calls: int
    status: str                          # "in_progress" | "completed" | "failed"
    error_message: Optional[str]