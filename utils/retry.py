"""Exponential-backoff retry wrapper for agent LLM calls.

Used to retry transient failures from `ChatAnthropic.invoke` — especially the
`with_structured_output` parser raising `OutputParserException` when the model
returns malformed JSON under load. `BudgetExceeded` is deliberately NOT
retried: it's a hard stop from the budget tracker."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.budget import BudgetExceeded

T = TypeVar("T")

DEFAULT_ATTEMPTS = 3
DEFAULT_WAIT_MIN = 1.0
DEFAULT_WAIT_MAX = 10.0


def with_retries(
    func: Callable[..., T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    wait_min: float = DEFAULT_WAIT_MIN,
    wait_max: float = DEFAULT_WAIT_MAX,
) -> Callable[..., T]:
    """Wrap a callable in a retry loop with jittered exponential backoff."""
    decorated = retry(
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(min=wait_min, max=wait_max),
        retry=retry_if_not_exception_type(BudgetExceeded),
        reraise=True,
    )(func)

    def wrapper(*args: Any, **kwargs: Any) -> T:
        return decorated(*args, **kwargs)

    wrapper.__wrapped__ = func  # type: ignore[attr-defined]
    return wrapper
