from __future__ import annotations

import os

from langgraph.graph import END, START, StateGraph

from agents import (
    coder_node,
    orchestrator_node,
    planner_node,
    reviewer_node,
    should_continue,
    tester_node,
)
from models.schemas import AgentState


def build_graph() -> StateGraph:
    """Constructs and compiles the multi-agent workflow graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tester", tester_node)

    # Linear flow: orchestrator → planner → coder → reviewer → tester
    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "tester")

    # After testing: either end or loop back to coder for revision
    graph.add_conditional_edges(
        "tester",
        should_continue,
        {
            "end": END,
            "revise": "coder",
        },
    )

    return graph.compile()


def run(user_request: str, max_iterations: int = 3) -> AgentState:
    """Run the full multi-agent pipeline for a given user request."""
    app = build_graph()

    initial_state = AgentState(
        user_request=user_request,
        max_iterations=max_iterations,
    )

    recursion_limit = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))
    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": recursion_limit},
    )

    return AgentState(**final_state)
