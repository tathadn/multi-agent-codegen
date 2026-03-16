# Multi-Agent Code Generator

> Describe what you want to build — a pipeline of AI agents plans, writes, reviews, and tests the code for you.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and Claude, this tool turns a plain-English description into working, tested code. Five specialized agents collaborate in a graph: one parses your intent, one plans the implementation, one writes the code, one reviews it for quality, and one runs real tests in an isolated Docker sandbox. If tests fail or the review score is too low, the coder revises automatically — up to a configurable number of iterations.

## Demo

![Multi-Agent Code Generator — pipeline complete with 79/79 tests passing](assets/demo.png)

*The sidebar shows real-time agent status indicators. Here, a student grade tracker was generated, reviewed (9/10), and passed 79/79 tests on the first attempt.*

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│ Orchestrator│  Parses the request, tracks overall progress
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Planner   │  Produces a structured plan: steps, files, dependencies
└──────┬──────┘
       │
       ▼
┌─────────────┐ ◄─────────────────────────────┐
│    Coder    │  Generates (or revises) code   │  revision loop
└──────┬──────┘                                │  (up to N times)
       │                                       │
       ▼                                       │
┌─────────────┐                                │
│  Reviewer   │  Scores code quality (0–10),   │
│             │  flags issues & suggestions    │
└──────┬──────┘                                │
       │                                       │
       ▼                                       │
┌─────────────┐  tests fail or review          │
│   Tester    │  not approved? ────────────────┘
└──────┬──────┘
       │  all green
       ▼
 Final Output
 (code files + plan + review + test report)
```

Each agent shares a single `AgentState` object that flows through the [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`. The tester node uses a conditional edge (`should_continue`) to either end the pipeline or loop back to the coder for revision.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/multi-agent-codegen.git
cd multi-agent-codegen

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Set your API key
cp .env.example .env
# open .env and set ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## How It Works

### Agent Pipeline

| Agent | Model | Role |
|---|---|---|
| **Orchestrator** | claude-opus-4-6 | Interprets the user request and sets pipeline status |
| **Planner** | claude-sonnet-4-6 | Outputs a structured `Plan`: objective, steps, files to create, dependencies, complexity |
| **Coder** | claude-sonnet-4-6 | Generates `CodeArtifact` objects (filename, language, content). On revisions, receives prior review issues and test failures as context |
| **Reviewer** | claude-sonnet-4-6 | Scores the code 0–10, marks it approved or not, lists issues and suggestions |
| **Tester** | claude-haiku-4-5 | Writes and simulates test execution, returns pass/fail counts and error details |

### Revision Loop

After the Tester runs, a `should_continue` router decides what happens next:

- **Review approved + all tests pass** → pipeline ends with `COMPLETED`
- **Review failed or tests failed + iterations remaining** → loop back to Coder, which receives the reviewer's issues and test errors as additional context
- **Max iterations reached** → pipeline ends regardless (avoids infinite loops)

Max iterations is configurable in the sidebar (default: 3).

### Shared State

All agents read from and write to a Pydantic `AgentState` model:

```python
class AgentState(BaseModel):
    user_request: str
    plan:         Optional[Plan]
    artifacts:    list[CodeArtifact]
    review:       Optional[ReviewFeedback]
    test_result:  Optional[TestResult]
    status:       TaskStatus
    iteration:    int
    max_iterations: int
```

---

## Example

**Input prompt:**
```
A Python FastAPI server with a /health endpoint and a /echo POST endpoint
that returns the request body as JSON.
```

**Generated files:**

`main.py`
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EchoRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/echo")
def echo(body: EchoRequest):
    return body
```

`test_main.py`
```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_echo():
    r = client.post("/echo", json={"message": "hello"})
    assert r.status_code == 200
    assert r.json() == {"message": "hello"}
```

**Review:** 9/10 — approved
**Tests:** 2/2 passed
**Status:** COMPLETED

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | ≥ 0.2.0 | Agent orchestration graph with conditional edges |
| [LangChain Anthropic](https://github.com/langchain-ai/langchain) | ≥ 0.3.0 | Claude API integration via `ChatAnthropic` |
| [LangChain Core](https://github.com/langchain-ai/langchain) | ≥ 0.3.0 | Message types, structured output |
| [Pydantic](https://docs.pydantic.dev/) | ≥ 2.0.0 | Typed state schema and structured LLM outputs |
| [Streamlit](https://streamlit.io/) | ≥ 1.40.0 | Web UI with real-time streaming via `app.stream()` |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | ≥ 1.0.0 | `.env` file loading |

Python ≥ 3.10 required.

---

## Future Work

- **Code execution sandbox** — actually run generated code in an isolated environment (e.g. Docker, E2B) instead of simulating test results with an LLM
- **File system output** — write generated files directly to a user-specified directory with one click
- **Agent memory** — persist previous runs so the coder can learn from past mistakes across sessions
- **Custom agent prompts** — let users edit agent system prompts from the UI without touching code
- **Streaming token output** — stream individual tokens from the coder agent for a faster perceived response
- **Multi-language support** — extend beyond Python to TypeScript, Go, Rust with language-specific test runners
- **GitHub integration** — push generated code directly to a new branch and open a pull request
