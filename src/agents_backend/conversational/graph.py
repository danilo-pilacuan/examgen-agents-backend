"""Grafo conversacional del agente con human-in-the-loop.

Flujo:
  - El LLM recibe el historial (`messages`) y el system prompt.
  - Puede invocar tools (buscar, generar, guardar, confirmar verificación).
  - Las tools de ESCRITURA (`guardar_pregunta`, `confirmar_verificacion_humana`)
    pasan por un nodo de confirmación humana con `interrupt`: el grafo PAUSA
    y espera la aprobación del humano antes de ejecutarlas.
  - El checkpointer persiste el estado entre turnos.

El grafo se expone como endpoint OpenAI-compatible para conectarlo a OpenWebUI.
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from agents_backend.conversational.prompts import SYSTEM_PROMPT
from agents_backend.conversational.state import ConversationalState
from agents_backend.conversational.tools import (
    agregar_material_archivo,
    buscar_preguntas,
    confirmar_verificacion_humana,
    exportar_preguntas,
    generar_examen_pdf,
    generar_pregunta,
    guardar_pregunta,
    guardar_pregunta_pendiente,
    listar_material_materia,
    listar_materias,
    registrar_materia,
)
from agents_backend.shared.llm import get_llm
from core.config import settings

TOOLS = [
    listar_materias,
    buscar_preguntas,
    exportar_preguntas,
    generar_pregunta,
    guardar_pregunta,
    guardar_pregunta_pendiente,
    confirmar_verificacion_humana,
    generar_examen_pdf,
    agregar_material_archivo,
    listar_material_materia,
    registrar_materia,
]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

# Tools que implican ESCRITURA en la BD. Ya no usan `interrupt` (Open WebUI no
# puede reanudarlo): la confirmación humana se maneja a nivel de prompt/LLM.
WRITE_TOOLS = {"guardar_pregunta", "guardar_pregunta_pendiente", "confirmar_verificacion_humana"}


def _make_llm():
    """LLM del agente conversacional con las tools vinculadas.

    IMPORTANTE: se usa `reasoning_effort="low"` (config `llm_reasoning_chat`).
    GLM-5.3-Flash es un modelo de razonamiento con comportamiento INVERSO a
    DeepSeek: con `none` escribe su razonamiento directamente en `content`
    (respuestas del chat "ensuciadas"); con `low` el contenido sale limpio.
    Verificado con el servidor vLLM local.
    """
    llm = get_llm(
        reasoning_effort=settings.llm_reasoning_chat,
        temperature=0.4,
    )
    return llm.bind_tools(TOOLS)


def _assistant_node(state: ConversationalState) -> dict:
    """Nodo principal: el LLM decide responder o llamar a una tool."""
    llm = _make_llm()
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    )
    return {"messages": [response]}


def _call_tools_node(state: ConversationalState) -> dict:
    """Ejecuta las tool calls del último mensaje.

    Las tools de escritura se ejecutan directamente (sin `interrupt`): la
    confirmación humana se maneja a nivel de prompt/LLM, que es compatible con
    Open WebUI (que no puede reanudar interrupts de LangGraph).
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    if not tool_calls:
        return {"messages": []}

    results: list[ToolMessage] = []
    for tool_call in tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        tool_id = tool_call["id"]

        # Ejecutar la tool (lectura o escritura).
        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            results.append(ToolMessage(content=f"❌ Tool desconocida: {name}", tool_call_id=tool_id))
            continue
        try:
            content = tool.invoke(args)
        except ValidationError as e:
            # Los args del LLM no cumplen el args_schema (Pydantic) de la tool.
            # Se devuelve un mensaje accionable para que el LLM corrija los
            # argumentos y reintente (no crashea el grafo).
            errors = "; ".join(
                f"{'.'.join(str(p) for p in err.get('loc', []))}: {err.get('msg', '')}"
                for err in e.errors()[:5]
            )
            content = (
                f"⚠️ Argumentos inválidos para la tool `{name}`. "
                f"Corrígelos y vuelve a invocarla. Detalle: {errors}"
            )
        except Exception as e:  # noqa: BLE001
            content = f"❌ Error al ejecutar {name}: {str(e)[:200]}"
        results.append(ToolMessage(content=content, tool_call_id=tool_id))

    return {"messages": results}


def _build_postgres_checkpointer():
    """Crea un checkpointer persistente en PostgreSQL.

    Usa la misma BD del proyecto (DATABASE_URL). La tabla `checkpoints` se
    crea con `.setup()` al primer uso.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Connection

    dsn = settings.database_url
    conn = Connection.connect(dsn)
    saver = PostgresSaver(conn)
    saver.setup()  # crea la tabla `checkpoints` si no existe
    return saver


def build_conversational_graph(checkpointer=None):
    """Construye y compila el grafo conversacional.

    Por defecto usa un checkpointer PERSISTENTE en PostgreSQL (para retomar
    conversaciones entre reinicios). Si `checkpointer` se provee, se usa ese.
    """
    if checkpointer is None:
        try:
            checkpointer = _build_postgres_checkpointer()
        except Exception:  # noqa: BLE001
            # Fallback a memoria si no hay BD disponible.
            checkpointer = InMemorySaver()

    graph = StateGraph(ConversationalState)

    graph.add_node("assistant", _assistant_node)
    graph.add_node("tools", _call_tools_node)

    graph.add_edge(START, "assistant")
    # Si el LLM pidió tools → tools; si no, termina.
    graph.add_conditional_edges(
        "assistant",
        lambda state: "tools" if (state["messages"][-1].tool_calls) else END,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)


# Singleton del grafo (reutilizable, con checkpointer en memoria por defecto).
conversational_graph = build_conversational_graph()
