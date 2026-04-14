"""Headless eval harness — runs every case in evals/cases.jsonl through the
graph and reports pass rate, revision count, latency, and cost. Respects
EXPERIMENT_BUDGET_USD: once the tracker hits the limit, the current run fails
and no further cases are dispatched.

Usage:
    python scripts/run_evals.py                    # run all cases
    python scripts/run_evals.py --cases fizzbuzz   # subset by id (comma-sep)
    python scripts/run_evals.py --budget 5         # override budget (USD)
    python scripts/run_evals.py --max-iter 2       # override revision cap
    python scripts/run_evals.py --out results.json # also write full results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make the project root importable when invoked as `python scripts/run_evals.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph
from models.schemas import AgentState, TaskStatus
from utils import BudgetExceeded, BudgetTracker, set_tracker

CASES_PATH = Path(__file__).resolve().parent.parent / "evals" / "cases.jsonl"


def load_cases(path: Path, ids: list[str] | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    if ids:
        wanted = set(ids)
        cases = [c for c in cases if c["id"] in wanted]
    return cases


def run_case(graph: Any, case: dict[str, Any], max_iter: int) -> dict[str, Any]:
    start = time.perf_counter()
    final_state: dict = {}
    error: str | None = None

    initial_state = AgentState(
        user_request=case["prompt"],
        max_iterations=max_iter,
    )

    try:
        final_state = graph.invoke(
            initial_state,
            config={
                "recursion_limit": int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))
            },
        )
    except BudgetExceeded as exc:
        error = f"BudgetExceeded: {exc}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start

    state = AgentState(**{k: v for k, v in final_state.items() if v is not None}) if final_state else None
    review = state.review if state else None
    test = state.test_result if state else None
    status = state.status if state else TaskStatus.FAILED

    passed = bool(
        state
        and review
        and review.approved
        and review.score >= state.min_review_score
        and test
        and test.passed
    )

    return {
        "id": case["id"],
        "prompt": case["prompt"],
        "passed": passed,
        "status": status.value if hasattr(status, "value") else str(status),
        "review_score": review.score if review else None,
        "review_approved": review.approved if review else None,
        "tests_passed": test.passed_tests if test else 0,
        "tests_total": test.total_tests if test else 0,
        "iterations": state.iteration if state else 0,
        "latency_s": round(elapsed, 2),
        "error": error,
    }


def print_table(results: list[dict[str, Any]], tracker: BudgetTracker) -> None:
    width_id = max(len(r["id"]) for r in results) if results else 4
    header = f"{'id':<{width_id}}  {'pass':<5}  {'score':<5}  {'tests':<7}  {'iter':<4}  {'lat':<6}  status/error"
    print(header)
    print("-" * len(header))
    for r in results:
        passed = "✅" if r["passed"] else "❌"
        score = "-" if r["review_score"] is None else str(r["review_score"])
        tests = f"{r['tests_passed']}/{r['tests_total']}"
        status = r["error"] or r["status"]
        print(
            f"{r['id']:<{width_id}}  {passed:<5}  {score:<5}  {tests:<7}  "
            f"{r['iterations']:<4}  {r['latency_s']:<6.2f}  {status}"
        )

    pass_count = sum(1 for r in results if r["passed"])
    print("-" * len(header))
    print(f"Pass rate: {pass_count}/{len(results)} ({pass_count / max(1, len(results)):.0%})")
    avg_iter = sum(r["iterations"] for r in results) / max(1, len(results))
    avg_lat = sum(r["latency_s"] for r in results) / max(1, len(results))
    print(f"Avg iterations: {avg_iter:.1f} | Avg latency: {avg_lat:.1f}s")
    print(tracker.summary())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=None, help="Comma-separated case ids to run")
    parser.add_argument(
        "--budget",
        type=float,
        default=float(os.getenv("EXPERIMENT_BUDGET_USD", "25")),
    )
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--out", default=None, help="Optional path to dump results JSON")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Populate .env first.", file=sys.stderr)
        return 2

    ids = args.cases.split(",") if args.cases else None
    cases = load_cases(CASES_PATH, ids)
    if not cases:
        print("No cases to run.", file=sys.stderr)
        return 1

    tracker = BudgetTracker(limit_usd=args.budget)
    set_tracker(tracker)

    graph = build_graph()
    results: list[dict[str, Any]] = []

    print(f"Running {len(cases)} case(s) with budget ${args.budget:.2f}\n")
    for case in cases:
        print(f"→ {case['id']} ...")
        result = run_case(graph, case, args.max_iter)
        results.append(result)

        if result["error"] and result["error"].startswith("BudgetExceeded"):
            print(f"  {result['error']} — halting remaining cases.")
            break

    print()
    print_table(results, tracker)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"budget": tracker.__dict__, "results": results},
                indent=2,
                default=str,
            )
        )
        print(f"\nWrote detailed results to {args.out}")

    pass_count = sum(1 for r in results if r["passed"])
    return 0 if pass_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
