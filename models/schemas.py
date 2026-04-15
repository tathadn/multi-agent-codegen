from __future__ import annotations

from enum import Enum
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVISION = "needs_revision"


class CodeArtifact(BaseModel):
    filename: str
    language: str
    content: str
    description: str = ""


class TestResult(BaseModel):
    passed: bool
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    errors: list[str] = Field(default_factory=list)
    output: str = ""


class ReviewFeedback(BaseModel):
    approved: bool
    score: int = Field(ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""


class Plan(BaseModel):
    objective: str
    steps: list[str]
    files_to_create: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    estimated_complexity: str = Field(default="medium", pattern="^(low|medium|high)$")


class AgentState(BaseModel):
    """Shared state passed between all agents in the graph."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    user_request: str = ""
    plan: Plan | None = None
    artifacts: list[CodeArtifact] = Field(default_factory=list)
    review: ReviewFeedback | None = None
    test_result: TestResult | None = None
    status: TaskStatus = TaskStatus.PENDING
    iteration: int = 0
    max_iterations: int = 3
    min_review_score: int = 7
    tester_model: str | None = None
    error: str | None = None
