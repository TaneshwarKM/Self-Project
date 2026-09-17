"""
memory.py
Short-term scratchpad memory for a single task run. Tracks every
step the agent takes (thought, tool call, observation) so the agent
can review its own progress, and so the final trace can be logged for
debugging or the benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    step_number: int
    tool_name: str | None
    tool_input: dict | None
    tool_output: dict | None
    assistant_text: str | None


@dataclass
class TaskMemory:
    task: str
    steps: list[Step] = field(default_factory=list)

    def record(
        self,
        step_number: int,
        assistant_text: str | None = None,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_output: dict | None = None,
    ) -> None:
        self.steps.append(
            Step(
                step_number=step_number,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=tool_output,
                assistant_text=assistant_text,
            )
        )

    def tool_call_count(self) -> int:
        return sum(1 for s in self.steps if s.tool_name is not None)

    def failed_tool_calls(self) -> int:
        return sum(
            1
            for s in self.steps
            if s.tool_output is not None and s.tool_output.get("success") is False
        )

    def to_trace(self) -> list[dict]:
        """Human-readable trace for logging/debugging a run."""
        trace = []
        for s in self.steps:
            entry = {"step": s.step_number}
            if s.assistant_text:
                entry["thought"] = s.assistant_text
            if s.tool_name:
                entry["tool"] = s.tool_name
                entry["input"] = s.tool_input
                entry["output"] = s.tool_output
            trace.append(entry)
        return trace
