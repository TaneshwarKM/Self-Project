# Tool-Using AI Agent

A ReAct-style autonomous agent that breaks a task into steps, calls
tools (calculator, web search, Python code execution) when it needs
information or computation it can't reliably do in its head, and
verifies its own results before answering — with retry logic so a
single failed tool call doesn't derail the whole task.

## Why this exists

A chatbot that only answers questions from its own knowledge is
limited. The harder, more valuable engineering problem is an agent
that can *take actions* — search, compute, execute code — and chain
them together reliably, including recovering from tool failures.

## Architecture

```
  task (natural language)
         │
         ▼
  ┌─────────────────┐
  │   agent.py       │◀──────────────┐
  │  (ReAct loop)    │                │
  └────────┬─────────┘                │
           │ tool_use                 │ tool_result
           ▼                          │
  ┌─────────────────┐                 │
  │  tools/registry  │                │
  │  ├─ calculator   │                │
  │  ├─ web_search   │────────────────┘
  │  └─ code_executor│   (retried up to MAX_TOOL_RETRIES
  └─────────────────┘    before reporting permanent failure)
           │
           ▼
  memory.py records every step (thought, tool, input, output)
           │
           ▼
  final answer + full reasoning trace
```

## Features

- **ReAct loop** — model alternates between reasoning and tool calls
  using Anthropic's native tool-use API, stopping when it has a final
  answer or hits `MAX_AGENT_STEPS`.
- **Three tools out of the box** — a safe AST-based calculator (no
  `eval()` on model input), a sandboxed Python code executor
  (subprocess + timeout), and web search (live via SerpAPI, or an
  offline stub so the agent runs without a search API key).
- **Retry logic** — a failed tool call is retried up to
  `MAX_TOOL_RETRIES` times before being reported to the model as a
  permanent failure, instead of crashing the task.
- **Full reasoning trace** — every step (thought, tool, input,
  output) is recorded, so a task can be debugged or audited after
  the fact.
- **Benchmark harness** (`benchmark.py`) — runs the agent against a
  labeled task set and reports task completion rate and tool-selection
  accuracy, so quality claims are measured, not asserted.
- **Streamlit demo** — type a task, watch the agent work, inspect the
  full trace.

## Setup

```bash
git clone <this-repo>
cd ai-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Usage

**CLI**
```bash
python -m src.agent --task "Compute compound interest on $1000 at 5% for 10 years"
```

**Streamlit demo**
```bash
streamlit run app.py
```

**Benchmark**
```bash
python -m src.benchmark --tasks data/benchmark_tasks.json
```

## Results

Measured with `benchmark.py` on the 5-task benchmark set in
`data/benchmark_tasks.json` (arithmetic, code execution, web search,
and multi-tool tasks):

| Metric | Value |
|---|---|
| Task completion rate | *run `benchmark.py` and fill in* |
| Tool-selection accuracy | *run `benchmark.py` and fill in* |
| Avg. failed tool calls per task | *run `benchmark.py` and fill in* |

> Run `python -m src.benchmark` with your own `ANTHROPIC_API_KEY` to
> get real numbers — these depend on the live model, so they're left
> as placeholders until you run it.

## Project structure

```
ai-agent/
├── src/
│   ├── agent.py           # ReAct loop, retry logic, entry point
│   ├── memory.py           # per-task scratchpad + trace recording
│   ├── benchmark.py        # task completion / tool-selection eval
│   └── tools/
│       ├── __init__.py     # tool registry
│       ├── calculator.py   # AST-based safe arithmetic
│       ├── web_search.py   # SerpAPI + offline stub
│       └── code_executor.py# sandboxed subprocess Python execution
├── app.py                  # Streamlit chat demo
├── data/
│   └── benchmark_tasks.json
├── tests/
│   └── test_tools.py       # unit tests, no API key required
└── requirements.txt
```

## Design decisions worth noting

- **Why AST-based calculator instead of `eval()`?** The model's
  input to this tool is untrusted from a security standpoint —
  `eval()` on it would be arbitrary code execution wearing a
  calculator costume. Parsing to an AST and whitelisting operators
  closes that hole.
- **Why retry before failing?** Tool calls fail for transient
  reasons (network blip, rate limit) as often as for real errors.
  Retrying a couple of times before surfacing a permanent failure to
  the model avoids the agent giving up on tasks it could actually
  complete.
- **Why an offline web-search stub?** So the project is runnable and
  testable by anyone cloning it, without requiring a paid search API
  key just to see the agent work end-to-end.
- **Sandbox honesty:** the code executor uses a subprocess + timeout,
  which is demo-grade isolation, not production-grade. A real
  deployment would run it in a locked-down container (no network, no
  filesystem access outside a temp dir, CPU/memory limits via
  cgroups) — noted here rather than overclaiming security this
  version doesn't have.

## Possible extensions

- Add long-term memory (persist facts learned across tasks/sessions)
- Add a planning step that decomposes complex tasks before execution
  begins, rather than purely reactive step-by-step tool use
- Harden `code_executor` with container-based sandboxing
- Add more tools (file read/write, a database query tool, a second
  LLM call for self-critique of the final answer)

## License

MIT
