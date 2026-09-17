"""
tools/__init__.py
Central tool registry. Every tool exposes:
  - an Anthropic-format JSON schema (for the model to know how to call it)
  - a run(**kwargs) -> dict function (for the agent loop to execute it)

Adding a new tool means writing one module with those two things and
registering it in TOOLS below — the agent loop itself never changes.
"""

from src.tools.calculator import CALCULATOR_SCHEMA, run_calculator
from src.tools.code_executor import CODE_EXECUTOR_SCHEMA, run_code_executor
from src.tools.web_search import WEB_SEARCH_SCHEMA, run_web_search

TOOLS = {
    "calculator": {"schema": CALCULATOR_SCHEMA, "run": run_calculator},
    "web_search": {"schema": WEB_SEARCH_SCHEMA, "run": run_web_search},
    "code_executor": {"schema": CODE_EXECUTOR_SCHEMA, "run": run_code_executor},
}


def get_tool_schemas() -> list[dict]:
    """Return the list of tool schemas in Anthropic's tools= format."""
    return [t["schema"] for t in TOOLS.values()]


def run_tool(name: str, tool_input: dict) -> dict:
    """Execute a registered tool by name. Raises KeyError if unknown."""
    if name not in TOOLS:
        raise KeyError(f"Unknown tool: {name}")
    return TOOLS[name]["run"](**tool_input)
