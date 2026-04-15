import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph  # noqa: E402
from models.schemas import AgentState, TaskStatus  # noqa: E402
from utils import BudgetExceeded, BudgetTracker, set_tracker  # noqa: E402

st.set_page_config(
    page_title="Multi-Agent Code Generator",
    page_icon="🤖",
    layout="wide",
)

st.title("Multi-Agent Code Generator")
st.caption("Powered by LangGraph + Claude")

PIPELINE_ORDER = ["orchestrator", "planner", "coder", "reviewer", "tester"]

AGENT_META = {
    "orchestrator": ("🎯", "Orchestrator", "Parsing your request..."),
    "planner":      ("📋", "Planner",      "Building an implementation plan..."),
    "coder":        ("💻", "Coder",         "Writing code..."),
    "reviewer":     ("🔍", "Reviewer",      "Reviewing code quality..."),
    "tester":       ("🧪", "Tester",        "Running tests..."),
}

CIRCLE = {
    "waiting": "⚪",
    "running": "🟡",
    "done":    "🟢",
    "failed":  "🔴",
}


TESTER_MODELS = {
    "Fast — Haiku (simple projects)":    "claude-haiku-4-5-20251001",
    "Standard — Sonnet (most projects)": "claude-sonnet-4-6",
    "Thorough — Opus (complex projects)": "claude-opus-4-6",
}


def render_sidebar() -> tuple[int, str, dict, dict]:
    """Render sidebar and return (max_iterations, tester_model, placeholders, indicator_states)."""
    placeholders: dict = {}
    indicator_states: dict = {key: "waiting" for key in PIPELINE_ORDER}

    with st.sidebar:
        st.header("Settings")
        max_iterations = st.slider("Max revision iterations", min_value=1, max_value=5, value=3)

        st.markdown("**Tester agent model**")
        tester_choice = st.radio(
            label="tester_model",
            options=list(TESTER_MODELS.keys()),
            index=1,
            label_visibility="collapsed",
            help=(
                "Haiku is fastest but may miss edge cases. "
                "Opus is most thorough but slower and costlier."
            ),
        )
        tester_model = TESTER_MODELS[tester_choice]

        st.divider()
        st.markdown("**Pipeline Status**")
        for key in PIPELINE_ORDER:
            emoji, label, _ = AGENT_META[key]
            circle_col, label_col = st.columns([1, 5])
            with circle_col:
                placeholders[key] = st.empty()
                placeholders[key].markdown(CIRCLE["waiting"])
            with label_col:
                st.markdown(f"{emoji} {label}")

    return max_iterations, tester_model, placeholders, indicator_states


def set_indicator(placeholders: dict, indicator_states: dict, agent: str, state: str) -> None:
    indicator_states[agent] = state
    placeholders[agent].markdown(CIRCLE[state])


def run_with_streaming(
    request: str,
    max_iterations: int,
    tester_model: str,
    placeholders: dict,
    indicator_states: dict,
) -> AgentState:
    """Run the graph with app.stream() and update sidebar indicators in real time."""
    budget_limit = float(os.getenv("EXPERIMENT_BUDGET_USD", "25"))
    tracker = BudgetTracker(limit_usd=budget_limit)
    set_tracker(tracker)
    st.session_state["budget_tracker"] = tracker

    graph = build_graph()
    recursion_limit = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))

    initial_state = AgentState(
        user_request=request,
        max_iterations=max_iterations,
        tester_model=tester_model,
    )

    # Mark the first agent as running before stream starts
    set_indicator(placeholders, indicator_states, "orchestrator", "running")

    final_state: dict = {}
    coder_iteration = 0

    with st.status("Running agent pipeline...", expanded=True) as pipeline_status:
        try:
            for chunk in graph.stream(
                initial_state,
                config={"recursion_limit": recursion_limit},
            ):
                for node_name, state_update in chunk.items():
                    emoji, label, description = AGENT_META.get(
                        node_name, ("⚙️", node_name, "Running...")
                    )

                    # Mark this agent done
                    set_indicator(placeholders, indicator_states, node_name, "done")

                    # Log to the streaming status panel
                    if node_name == "coder":
                        coder_iteration += 1
                        suffix = f" (revision {coder_iteration})" if coder_iteration > 1 else ""
                        st.write(f"{emoji} **{label}{suffix}** — {description}")
                    else:
                        st.write(f"{emoji} **{label}** — {description}")

                    final_state.update(state_update)

                    # Determine which agent runs next and mark it yellow
                    if node_name == "tester":
                        test = state_update.get("test_result")
                        review = final_state.get("review")
                        iteration_count = final_state.get("iteration", 0)
                        will_revise = (
                            (test is not None and not test.passed)
                            or (review is not None and not review.approved)
                        ) and iteration_count < max_iterations

                        if will_revise:
                            # Reset coder → reviewer → tester back to waiting, then coder to running
                            for agent in ["coder", "reviewer", "tester"]:
                                set_indicator(placeholders, indicator_states, agent, "waiting")
                            set_indicator(placeholders, indicator_states, "coder", "running")
                    else:
                        idx = PIPELINE_ORDER.index(node_name)
                        if idx + 1 < len(PIPELINE_ORDER):
                            next_agent = PIPELINE_ORDER[idx + 1]
                            set_indicator(placeholders, indicator_states, next_agent, "running")

        except Exception:
            # Mark any still-running agent as failed
            for agent, state in indicator_states.items():
                if state == "running":
                    set_indicator(placeholders, indicator_states, agent, "failed")
            pipeline_status.update(label="Pipeline failed.", state="error", expanded=True)
            raise

        pipeline_status.update(label="Pipeline complete!", state="complete", expanded=False)

    return AgentState(**{k: v for k, v in final_state.items() if v is not None})


def render_results(state: AgentState) -> None:
    status_color = {
        TaskStatus.COMPLETED: "green",
        TaskStatus.FAILED: "red",
        TaskStatus.NEEDS_REVISION: "orange",
        TaskStatus.IN_PROGRESS: "blue",
        TaskStatus.PENDING: "gray",
    }
    color = status_color.get(state.status, "gray")
    st.markdown(f"**Status:** :{color}[{state.status.value.upper()}]")

    col1, col2 = st.columns(2)

    with col1:
        if state.plan:
            with st.expander("📋 Plan", expanded=False):
                st.markdown(f"**Objective:** {state.plan.objective}")
                st.markdown("**Steps:**")
                for i, step in enumerate(state.plan.steps, 1):
                    st.markdown(f"{i}. {step}")
                if state.plan.dependencies:
                    st.markdown(f"**Dependencies:** `{'`, `'.join(state.plan.dependencies)}`")

        if state.review:
            with st.expander(f"🔍 Review — {state.review.score}/10", expanded=False):
                approved_icon = "✅" if state.review.approved else "❌"
                st.markdown(f"{approved_icon} **{state.review.summary}**")
                if state.review.issues:
                    st.markdown("**Issues:**")
                    for issue in state.review.issues:
                        st.markdown(f"- {issue}")
                if state.review.suggestions:
                    st.markdown("**Suggestions:**")
                    for s in state.review.suggestions:
                        st.markdown(f"- {s}")

    with col2:
        if state.test_result:
            passed_icon = "✅" if state.test_result.passed else "❌"
            tr = state.test_result
            label = f"🧪 Tests — {passed_icon} {tr.passed_tests}/{tr.total_tests}"
            with st.expander(label, expanded=False):
                if state.test_result.errors:
                    st.markdown("**Failures:**")
                    for err in state.test_result.errors:
                        st.code(err)
                if state.test_result.output:
                    st.text(state.test_result.output)

    if state.artifacts:
        st.subheader(f"Generated Code ({len(state.artifacts)} file(s))")

        with st.expander("📁 Export all files to a folder", expanded=False):
            export_path = st.text_input(
                "Destination directory",
                value=str(Path.cwd() / "generated"),
                key="export_path",
                help=(
                    "Absolute or relative path. "
                    "The directory will be created if it doesn't exist."
                ),
            )
            if st.button("Export", key="export_button"):
                try:
                    target = Path(export_path).expanduser().resolve()
                    target.mkdir(parents=True, exist_ok=True)
                    written: list[str] = []
                    for artifact in state.artifacts:
                        dest = target / artifact.filename
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_text(artifact.content)
                        written.append(str(dest))
                    st.success(f"Wrote {len(written)} file(s) to {target}")
                    st.code("\n".join(written))
                except Exception as exc:
                    st.error(f"Export failed: {exc}")

        for artifact in state.artifacts:
            with st.expander(f"`{artifact.filename}` — {artifact.description}", expanded=True):
                st.code(artifact.content, language=artifact.language)
                st.download_button(
                    label=f"Download {artifact.filename}",
                    data=artifact.content,
                    file_name=artifact.filename,
                    mime="text/plain",
                    key=artifact.filename,
                )


def main() -> None:
    max_iterations, tester_model, placeholders, indicator_states = render_sidebar()

    request = st.text_area(
        "Describe what you want to build",
        placeholder=(
            "e.g. A Python FastAPI server with a /health endpoint "
            "and a /echo POST endpoint"
        ),
        height=120,
    )

    if st.button("Generate Code", type="primary", disabled=not request.strip()):
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
            return

        try:
            state = run_with_streaming(
                request.strip(),
                max_iterations,
                tester_model,
                placeholders,
                indicator_states,
            )
            render_results(state)
        except BudgetExceeded as e:
            st.error(f"Budget exceeded: {e}")
        except Exception as e:
            st.error(f"Pipeline error: {e}")

        tracker = st.session_state.get("budget_tracker")
        if tracker is not None:
            st.caption(
                f"💰 Spent ${tracker.spent_usd:.4f} / ${tracker.limit_usd:.2f} "
                f"({tracker.calls} calls, "
                f"{tracker.input_tokens:,} in / {tracker.output_tokens:,} out)"
            )


if __name__ == "__main__":
    main()
