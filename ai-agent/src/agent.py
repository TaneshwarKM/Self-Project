"""
agent.py
Core ReAct-style agent loop: the model thinks, optionally calls a
tool, observes the result, and repeats until it produces a final
answer or hits MAX_AGENT_STEPS. Failed tool calls are retried up to
MAX_TOOL_RETRIES before being reported back to the model as a
permanent failure for that step, so one flaky call doesn't crash the
whole task.
"""

from __future__ import annotations

import os

import anthropic

from src.memory import TaskMemory
from src.tools import get_tool_schemas, run_tool

SYSTEM_PROMPT = """You are a task-completing agent with access to tools: \
calculator, web_search, and code_executor. Break the task into steps, use \
tools when you need information or computation you can't do reliably in \
your head, and verify your own intermediate results before giving a final \
answer. When you have the final answer, state it clearly and stop calling \
tools."""


class Agent:
    def __init__(
        self,
        model: str | None = None,
        max_steps: int | None = None,
        max_tool_retries: int | None = None,
    ):
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("AGENT_MODEL", "claude-sonnet-4-6")
        self.max_steps = max_steps or int(os.getenv("MAX_AGENT_STEPS", "8"))
        self.max_tool_retries = (
            max_tool_retries if max_tool_retries is not None else int(os.getenv("MAX_TOOL_RETRIES", "2"))
        )

    def _execute_tool_with_retry(self, name: str, tool_input: dict) -> dict:
        last_error = None
        for attempt in range(self.max_tool_retries + 1):
            try:
                result = run_tool(name, tool_input)
                if result.get("success", True):
                    return result
                last_error = result.get("error", "unknown tool error")
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        return {"success": False, "error": f"Failed after {self.max_tool_retries + 1} attempts: {last_error}"}

    def run(self, task: str) -> dict:
        memory = TaskMemory(task=task)
        messages = [{"role": "user", "content": task}]
        final_answer = None

        for step in range(1, self.max_steps + 1):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                tools=get_tool_schemas(),
                messages=messages,
            )

            text_parts = [b.text for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            assistant_text = "\n".join(text_parts).strip() or None

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use" or not tool_use_blocks:
                final_answer = assistant_text
                memory.record(step_number=step, assistant_text=assistant_text)
                break

            tool_result_blocks = []
            for block in tool_use_blocks:
                result = self._execute_tool_with_retry(block.name, block.input)
                memory.record(
                    step_number=step,
                    assistant_text=assistant_text,
                    tool_name=block.name,
                    tool_input=block.input,
                    tool_output=result,
                )
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                        "is_error": not result.get("success", True),
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            final_answer = "[Stopped: reached max_steps without a final answer]"

        return {
            "task": task,
            "answer": final_answer,
            "steps_taken": len(set(s.step_number for s in memory.steps)),
            "tool_calls": memory.tool_call_count(),
            "failed_tool_calls": memory.failed_tool_calls(),
            "trace": memory.to_trace(),
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the agent on a single task")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()

    agent = Agent()
    result = agent.run(args.task)
    print(f"\nAnswer: {result['answer']}\n")
    print(f"Steps: {result['steps_taken']} | Tool calls: {result['tool_calls']} | Failed: {result['failed_tool_calls']}")
    print("\nTrace:")
    print(json.dumps(result["trace"], indent=2, default=str))
