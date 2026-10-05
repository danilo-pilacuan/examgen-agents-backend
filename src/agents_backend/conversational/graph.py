"""Conversational graph of the agent with human-in-the-loop.

Flow:
  - The LLM receives the history (`messages`) and the system prompt.
  - It can invoke tools (search, generate, save, confirm verification).
  - The WRITE tools (`save_question`, `confirm_human_verification`) pass
    through a human-confirmation node with `interrupt`: the graph PAUSES
    and waits for human approval before executing them.
  - The checkpointer persists the state between turns.

The graph is exposed as an OpenAI-compatible endpoint to connect to OpenWebUI.
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
    add_material_file,
    confirm_human_verification,
    export_questions,
    generate_exam_pdf,
    generate_question,
    list_subject_material,
    list_subjects,
    register_subject,
    save_pending_question,
    save_question,
    search_questions,
)
from agents_backend.shared.llm import get_llm
from core.config import settings

TOOLS = [
    list_subjects,
    search_questions,
    export_questions,
    generate_question,
    save_question,
    save_pending_question,
    confirm_human_verification,
    generate_exam_pdf,
    add_material_file,
    list_subject_material,
    register_subject,
]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

# Tools that imply WRITING to the DB. They no longer use `interrupt` (Open WebUI
# cannot resume it): human confirmation is handled at the prompt/LLM level.
WRITE_TOOLS = {"save_question", "save_pending_question", "confirm_human_verification"}


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
    """Executes the tool calls of the last message.

    Write tools run directly (no `interrupt`): human confirmation is handled
    at the prompt/LLM level, which is compatible with Open WebUI (it cannot
    resume LangGraph interrupts).
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
            results.append(ToolMessage(content=f"❌ Unknown tool: {name}", tool_call_id=tool_id))
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
                f"⚠️ Invalid arguments for the tool `{name}`. "
                f"Fix them and invoke it again. Detail: {errors}"
            )
        except Exception as e:  # noqa: BLE001
            content = f"❌ Error executing {name}: {str(e)[:200]}"
        results.append(ToolMessage(content=content, tool_call_id=tool_id))

    # Sincronizar el pendiente del store con el estado del grafo: el
    # checkpointer persiste `pending_item`/`pending_subject_id` por thread
    # (replaces the old module-level global `_PENDING_ITEM`).
    from agents_backend.shared.events import get_current_thread_id
    from agents_backend.shared.pending import pending_store

    tid = get_current_thread_id()
    pending = pending_store.get(tid) if tid else None
    if pending is not None:
        return {
            "messages": results,
            "pending_item": pending.item,
            "pending_subject_id": pending.subject_id,
        }
    return {
        "messages": results,
        "pending_item": None,
        "pending_subject_id": None,
    }


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
    # If the LLM requested tools → tools; otherwise, end.
    graph.add_conditional_edges(
        "assistant",
        lambda state: "tools" if (state["messages"][-1].tool_calls) else END,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "assistant")

    return graph.compile(checkpointer=checkpointer)


# Singleton del grafo (reutilizable, con checkpointer en memoria por defecto).
conversational_graph = build_conversational_graph()
