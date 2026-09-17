"""
benchmark.py
Runs the agent against a fixed set of multi-step tasks and reports
task completion rate, tool-selection accuracy (did it use the tool
the task actually needed), and the average number of failed tool
calls per task. This is what turns "X% task completion rate" into a
number you actually measured instead of a placeholder.

Usage:
  python -m src.benchmark --tasks data/benchmark_tasks.json
"""

from __future__ import annotations

import argparse
import json

from src.agent import Agent


def run_benchmark(tasks_path: str) -> dict:
    with open(tasks_path) as f:
        tasks = json.load(f)

    agent = Agent()
    results = []

    for item in tasks:
        outcome = agent.run(item["task"])
        used_expected_tool = any(
            step.get("tool") == item["expects_tool"] for step in outcome["trace"]
        )
        answer_ok = (
            item["expected_answer_contains"] is None
            or (outcome["answer"] and item["expected_answer_contains"] in outcome["answer"])
        )

        results.append(
            {
                "id": item["id"],
                "used_expected_tool": used_expected_tool,
                "answer_correct": answer_ok,
                "steps_taken": outcome["steps_taken"],
                "failed_tool_calls": outcome["failed_tool_calls"],
                "answer": outcome["answer"],
            }
        )

    n = len(results)
    completion_rate = sum(r["answer_correct"] for r in results) / n if n else 0.0
    tool_selection_accuracy = sum(r["used_expected_tool"] for r in results) / n if n else 0.0
    avg_failed_calls = sum(r["failed_tool_calls"] for r in results) / n if n else 0.0

    return {
        "n_tasks": n,
        "task_completion_rate": round(completion_rate, 4),
        "tool_selection_accuracy": round(tool_selection_accuracy, 4),
        "avg_failed_tool_calls_per_task": round(avg_failed_calls, 4),
        "per_task_results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the agent against a task set")
    parser.add_argument("--tasks", default="data/benchmark_tasks.json")
    parser.add_argument("--out", default="data/benchmark_results.json")
    args = parser.parse_args()

    report = run_benchmark(args.tasks)
    print(json.dumps({k: v for k, v in report.items() if k != "per_task_results"}, indent=2))

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results written to {args.out}")
