from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from models.schemas import AgentState, Plan
from utils import BudgetCallbackHandler, cached_system, with_retries


_PROMPT = (Path(__file__).parent.parent / "prompts" / "planner.md").read_text()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("PLANNER_MODEL", "claude-sonnet-4-6"),
        max_tokens=2048,
        callbacks=[BudgetCallbackHandler()],
    )


def planner_node(state: AgentState) -> dict:
    """Produces a structured implementation plan for the user's request."""
    llm = get_llm().with_structured_output(Plan)

    messages = [
        cached_system(_PROMPT),
        HumanMessage(content=f"Create an implementation plan for: {state.user_request}"),
    ]

    plan: Plan = with_retries(llm.invoke)(messages)  # type: ignore[assignment]

    return {
        "plan": plan,
        "messages": [
            HumanMessage(
                content=f"Plan created:\n{json.dumps(plan.model_dump(), indent=2)}"
            )
        ],
    }
