from __future__ import annotations

import os
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import AgentState, TaskStatus, TestResult


_PROMPT = (Path(__file__).parent.parent / "prompts" / "tester.md").read_text()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=os.getenv("TESTER_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=4096,
    )


def _format_artifacts(state: AgentState) -> str:
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)


def tester_node(state: AgentState) -> dict:
    """Evaluates generated code by writing and simulating test execution."""
    llm = get_llm().with_structured_output(TestResult)

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(content=_format_artifacts(state)),
    ]

    result: TestResult = llm.invoke(messages)  # type: ignore[assignment]

    status = TaskStatus.COMPLETED if result.passed else TaskStatus.NEEDS_REVISION
    summary = HumanMessage(
        content=(
            f"Tests: {result.passed_tests}/{result.total_tests} passed. "
            f"{'All tests passed.' if result.passed else f'Failures: {result.failed_tests}'}"
        )
    )

    return {
        "test_result": result,
        "status": status,
        "messages": [summary],
    }
