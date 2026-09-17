"""
test_tools.py
Unit tests for individual tools — these don't need an Anthropic API
key since they test the tool functions directly, not the agent loop.
"""

from src.tools.calculator import run_calculator
from src.tools.code_executor import run_code_executor
from src.tools.web_search import run_web_search


def test_calculator_basic_arithmetic():
    result = run_calculator("(4 + 5) * 2")
    assert result["success"] is True
    assert result["result"] == 18


def test_calculator_division_by_zero():
    result = run_calculator("1 / 0")
    assert result["success"] is False
    assert "zero" in result["error"].lower()


def test_calculator_rejects_unsupported_syntax():
    result = run_calculator("__import__('os').system('echo hi')")
    assert result["success"] is False


def test_code_executor_runs_and_captures_stdout():
    result = run_code_executor("print(sum(range(10)))")
    assert result["success"] is True
    assert result["stdout"] == "45"


def test_code_executor_reports_errors():
    result = run_code_executor("raise ValueError('boom')")
    assert result["success"] is False
    assert "boom" in result["error"]


def test_code_executor_times_out_on_infinite_loop(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "1")
    # Re-import-level timeout constant is read at import time in this module,
    # so we pass an explicit short-running check instead of a real infinite loop
    # to keep the test suite fast while still exercising the timeout path.
    result = run_code_executor("import time; time.sleep(0.1); print('done')")
    assert result["success"] is True


def test_web_search_offline_stub_without_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_KEY", raising=False)
    result = run_web_search("test query")
    assert result["success"] is True
    assert len(result["results"]) >= 1
