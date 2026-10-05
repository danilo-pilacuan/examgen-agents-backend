"""State of the conversational agent (chat with human-in-the-loop).

Unlike the batch pipeline (`ExamGenerationState`), this state is a persistent
conversation with LangGraph's `MessagesState`: it stores the message history
and session context (active subject, pending questions awaiting confirmation,
etc.).
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ConversationalState(TypedDict):
    """State of the conversational agent.

    `messages` uses LangGraph's `add_messages` reducer and is persisted with
    the checkpointer to keep context between turns.
    """
    messages: Annotated[list, add_messages]

    # --- Session context ---
    subject_id: Optional[str]          # active subject of the conversation
    subject_name: Optional[str]

    # --- Question in progress (pending human confirmation) ---
    pending_item: Optional[dict]       # serialized GeneratedItem, ready to save
    pending_question_id: Optional[str] # DB id if already saved and awaiting verification
