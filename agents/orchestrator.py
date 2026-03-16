from __future__ import annotations

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import AgentState, TaskStatus


_PROMPT = (Path(__file__).parent.parent / "prompts" / "orchestrator.md").read_text()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("ORCHESTRATOR_MODEL", "claude-opus-4-6"),
        max_tokens=1024,
    )


def orchestrator_node(state: AgentState) -> dict:
    """Entry point: interprets the user request and sets the initial status."""
    llm = get_llm()

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(
            content=(
                f"User request: {state.user_request}\n\n"
                f"Current iteration: {state.iteration}/{state.max_iterations}\n"
                f"Status: {state.status}"
            )
        ),
    ]

    response = llm.invoke(messages)

    return {
        "messages": [response],
        "status": TaskStatus.IN_PROGRESS,
    }


def should_continue(state: AgentState) -> str:
    """Route after review + test: continue, revise, or finish."""
    if state.status == TaskStatus.COMPLETED:
        return "end"

    if state.status == TaskStatus.FAILED:
        return "end"

    if state.iteration >= state.max_iterations:
        return "end"

    review = state.review
    test = state.test_result

    if review and review.approved and test and test.passed:
        return "end"

    if state.iteration < state.max_iterations:
        return "revise"

    return "end"
