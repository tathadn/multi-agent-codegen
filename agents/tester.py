from __future__ import annotations

import json
import os
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from models.schemas import AgentState, CodeArtifact, TaskStatus, TestResult
from sandbox.runner import CodeFile, run_in_sandbox
from utils import BudgetCallbackHandler, cached_system, with_retries

_PROMPT = (Path(__file__).parent.parent / "prompts" / "tester.md").read_text()


class TestFileList(BaseModel):
    artifacts: list[CodeArtifact] = Field(default_factory=list)

    @field_validator("artifacts", mode="before")
    @classmethod
    def parse_if_string(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


DEFAULT_TESTER_MODEL = "claude-sonnet-4-6"


def get_llm(model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(
        model=model or os.getenv("TESTER_MODEL", DEFAULT_TESTER_MODEL),
        max_tokens=8096,
        callbacks=[BudgetCallbackHandler()],
    )


def _format_artifacts(state: AgentState) -> str:
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)


def _parse_pytest_counts(stdout: str) -> tuple[int, int, int]:
    """Extract total/passed/failed counts from pytest summary line."""
    # e.g. "3 passed, 1 failed" or "4 passed"
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    failed_count = int(failed_match.group(1)) if failed_match else 0
    total = passed_count + failed_count
    return total, passed_count, failed_count


def tester_node(state: AgentState) -> dict:
    """Generates test files via LLM, then executes them in the sandbox for real results."""
    # Step 1: LLM generates test files
    llm = get_llm(state.tester_model).with_structured_output(TestFileList)

    messages = [
        cached_system(_PROMPT),
        HumanMessage(content=_format_artifacts(state)),
    ]

    test_file_list: TestFileList = with_retries(llm.invoke)(messages)  # type: ignore[assignment]

    if not test_file_list.artifacts:
        result = TestResult(
            passed=False,
            errors=["Tester agent did not generate any test files."],
            output="",
        )
        return {
            "test_result": result,
            "status": TaskStatus.NEEDS_REVISION,
            "messages": [HumanMessage(content="No test files were generated.")],
        }

    # Step 2: Combine code files + test files and run in sandbox
    code_files = [CodeFile(filename=a.filename, content=a.content) for a in state.artifacts]
    test_files = [
        CodeFile(filename=a.filename, content=a.content)
        for a in test_file_list.artifacts
    ]

    sandbox_result = run_in_sandbox(code_files + test_files)

    # Step 3: Build TestResult from real execution output
    if sandbox_result.success:
        total, passed_count, failed_count = _parse_pytest_counts(sandbox_result.stdout)
        result = TestResult(
            passed=True,
            total_tests=total or len(test_files),
            passed_tests=passed_count or len(test_files),
            failed_tests=0,
            errors=[],
            output=sandbox_result.stdout,
        )
    else:
        total, passed_count, failed_count = _parse_pytest_counts(sandbox_result.stdout)
        errors = [sandbox_result.stderr] if sandbox_result.stderr.strip() else []
        if sandbox_result.stdout.strip():
            errors.append(sandbox_result.stdout)
        result = TestResult(
            passed=False,
            total_tests=total or len(test_files),
            passed_tests=passed_count,
            failed_tests=failed_count or (total - passed_count),
            errors=errors,
            output=sandbox_result.stdout,
        )

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
