"""
code_executor.py
Runs short Python snippets in an isolated subprocess with a hard
timeout and no network/file-system-relevant builtins exposed beyond
what a sandboxed subprocess already restricts. This is a demo-grade
sandbox (subprocess + timeout), not a hardened one — see README for
what a production version would add (containers, seccomp, resource
limits).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

CODE_EXECUTOR_SCHEMA = {
    "name": "code_executor",
    "description": (
        "Execute a short Python snippet and return its stdout/stderr. Use this "
        "for computations, data manipulation, or logic too complex for the "
        "calculator tool. The snippet must print() whatever result you need — "
        "return values are not captured."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute. Must use print() for output."}
        },
        "required": ["code"],
    },
}

TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))


def run_code_executor(code: str) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return {"success": False, "error": result.stderr.strip()[-2000:]}
        return {"success": True, "stdout": result.stdout.strip()[-4000:]}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Execution timed out after {TIMEOUT_SECONDS}s"}
    finally:
        os.unlink(script_path)
