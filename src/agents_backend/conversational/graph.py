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

import time
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
    delete_material_file_tool,
    delete_question_tool,
    export_questions,
    generate_exam_pdf,
    generate_question,
    list_subject_material,
    list_subjects,
    register_subject,
    save_pending_question,
    save_question,
    search_questions,
    select_subject,
    update_question_tool,
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
    delete_question_tool,
    update_question_tool,
    generate_exam_pdf,
    add_material_file,
    list_subject_material,
    delete_material_file_tool,
    register_subject,
    select_subject,
]
TOOL_BY_NAME = {t.name: t for t in TOOLS}

# Tools that imply WRITING to the DB. They no longer use `interrupt` (Open WebUI
# cannot resume it): human confirmation is handled at the prompt/LLM level.
WRITE_TOOLS = {"save_question", "save_pending_question", "confirm_human_verification"}


def _make_llm():
    """Conversational agent LLM with the tools bound.

    IMPORTANT: `reasoning_effort="low"` is used (config `llm_reasoning_chat`).
    GLM-5.3-Flash is a reasoning model with behavior INVERSE to DeepSeek:
    with `none` it writes its reasoning directly into `content` (chat
    responses get "dirty"); with `low` the content comes out clean.
    Verified against the local vLLM server.
    """
    llm = get_llm(
        reasoning_effort=settings.llm_reasoning_chat,
        temperature=0.4,
    )
    return llm.bind_tools(TOOLS)


def _assistant_node(state: ConversationalState) -> dict:
    """Main node: the LLM decides to answer or call a tool."""
    llm = _make_llm()
    prompt_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    started_at = time.perf_counter()
    response = llm.invoke(prompt_messages)
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    # Raw audit (best-effort): prompt + raw answer (+ thinking if enabled).
    from agents_backend.shared.events import get_current_thread_id
    from core.domain.llm_log.service import (
        extract_thinking,
        record_llm_response,
    )

    record_llm_response(
        context="chat:model",
        model=settings.llm_model,
        prompt_messages=prompt_messages,
        raw_answer=response.content if isinstance(response.content, str) else str(response.content),
        raw_thinking=extract_thinking(response),
        thread_id=get_current_thread_id(),
        latency_ms=latency_ms,
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

        # Run the tool (read or write).
        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            results.append(ToolMessage(content=f"❌ Unknown tool: {name}", tool_call_id=tool_id))
            continue
        try:
            content = tool.invoke(args)
        except ValidationError as e:
            # The LLM args do not satisfy the tool's args_schema (Pydantic).
            # An actionable message is returned so the LLM fixes the
            # arguments and retries (the graph does not crash).
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

    # Sync the pending store with the graph state: the checkpointer
    # persists `pending_item`/`pending_subject_id` per thread
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
    """Creates a persistent PostgreSQL checkpointer.

    Uses the project's own DB (DATABASE_URL). The `checkpoints` table is
    created with `.setup()` on first use.
    """
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg import Connection

    dsn = settings.database_url
    # autocommit: PostgresSaver.setup() runs `CREATE INDEX CONCURRENTLY`,
    # which cannot run inside a transaction block (fails on a fresh DB).
    conn = Connection.connect(dsn, autocommit=True)
    saver = PostgresSaver(conn)
    saver.setup()  # creates the `checkpoints` table if it does not exist
    return saver


def build_conversational_graph(checkpointer=None):
    """Builds and compiles the conversational graph.

    By default it uses a PERSISTENT PostgreSQL checkpointer (to resume
    conversations across restarts). If a `checkpointer` is provided, that
    one is used instead.
    """
    if checkpointer is None:
        try:
            checkpointer = _build_postgres_checkpointer()
        except Exception:  # noqa: BLE001
            # Fall back to memory if no DB is available.
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


# Graph singleton (reusable, with in-memory checkpointer by default).
conversational_graph = build_conversational_graph()
