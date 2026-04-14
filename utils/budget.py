"""Budget tracking for Anthropic API experiments.

Wraps a small callback handler around `ChatAnthropic` that tallies input/output
tokens per call, converts them to USD using a static pricing table, and raises
`BudgetExceeded` once the configured limit is crossed. The tracker is a process
global so the graph can fail fast mid-run when a budget cap is hit."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


# Public pricing in USD per 1M tokens (input, output).
# Update when Anthropic changes list prices.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (15.0, 75.0),
    "claude-opus-4-6[1m]": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

DEFAULT_PRICE: tuple[float, float] = (3.0, 15.0)  # fall back to Sonnet pricing


class BudgetExceeded(RuntimeError):
    """Raised the first time cumulative spend crosses the configured limit."""


@dataclass
class BudgetTracker:
    limit_usd: float
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        in_price, out_price = PRICING.get(model, DEFAULT_PRICE)
        cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price

        self.spent_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1

        bucket = self.by_model.setdefault(
            model, {"input": 0, "output": 0, "cost": 0.0, "calls": 0}
        )
        bucket["input"] += input_tokens
        bucket["output"] += output_tokens
        bucket["cost"] += cost
        bucket["calls"] += 1

        if self.spent_usd > self.limit_usd:
            raise BudgetExceeded(
                f"Budget exceeded: ${self.spent_usd:.4f} > ${self.limit_usd:.2f} "
                f"after {self.calls} calls"
            )

    def remaining(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    def summary(self) -> str:
        lines = [
            f"Budget: ${self.spent_usd:.4f} / ${self.limit_usd:.2f} "
            f"(remaining: ${self.remaining():.4f})",
            f"Calls: {self.calls} | input: {self.input_tokens:,} tok | "
            f"output: {self.output_tokens:,} tok",
        ]
        for model, stats in sorted(self.by_model.items()):
            lines.append(
                f"  {model}: {stats['calls']} calls, "
                f"{int(stats['input']):,} in / {int(stats['output']):,} out, "
                f"${stats['cost']:.4f}"
            )
        return "\n".join(lines)


_global_tracker: BudgetTracker | None = None


def get_tracker() -> BudgetTracker | None:
    return _global_tracker


def set_tracker(tracker: BudgetTracker | None) -> None:
    global _global_tracker
    _global_tracker = tracker


class BudgetCallbackHandler(BaseCallbackHandler):
    """Attaches to every `ChatAnthropic` call and feeds usage into the global tracker."""

    raise_error = True

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        tracker = get_tracker()
        if tracker is None:
            return

        for gen_list in getattr(response, "generations", []):
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                meta = getattr(msg, "response_metadata", {}) or {}
                usage = meta.get("usage") or {}
                model = (
                    meta.get("model_name")
                    or meta.get("model")
                    or "claude-sonnet-4-6"
                )
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                if input_tokens == 0 and output_tokens == 0:
                    continue
                tracker.record(model, input_tokens, output_tokens)
