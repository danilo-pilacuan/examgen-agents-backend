"""Estado del agente conversacional (chat con human-in-the-loop).

A diferencia del pipeline batch (`ExamGenerationState`), este estado es una
conversación persistente con `MessagesState` de LangGraph: guarda el historial
de mensajes y el contexto de la sesión (materia activa, preguntas pendientes
de confirmación, etc.).
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ConversationalState(TypedDict):
    """Estado del agente conversacional.

    `messages` usa el reducer `add_messages` de LangGraph y se persiste con
    el checkpointer para mantener el contexto entre turnos.
    """
    messages: Annotated[list, add_messages]

    # --- Contexto de sesión ---
    subject_id: Optional[str]          # materia activa de la conversación
    subject_name: Optional[str]

    # --- Pregunta en curso (pendiente de confirmación humana) ---
    pending_item: Optional[dict]       # GeneratedItem serializado, listo para guardar
    pending_question_id: Optional[str] # id en BD si ya se guardó y falta verificación
