"""
tools.py

Defines external tools the agent can use, such as:
- run_python: execute small Python snippets safely
- web_search_stub: placeholder for a real web search API
"""

from typing import Dict, Any
import traceback
import math


class ToolError(Exception):
    """Custom exception raised when a tool fails."""
    pass


def run_python_code(code: str) -> str:
    """
    Execute a small Python snippet in a very restricted environment.

    IMPORTANT: this is for educational demos only and should NOT be exposed
    to arbitrary untrusted users in production.
    """
    try:
        # allow only a tiny safe subset of builtins
        allowed_builtins = {
            "range": range,
            "len": len,
            "print": print,
            "math": math,
        }
        local_env: Dict[str, Any] = {}

        exec(code, {"__builtins__": allowed_builtins, "math": math}, local_env)

        if "_result" in local_env:
            return f"Execution success. _result = {repr(local_env['_result'])}"

        if local_env:
            names = ", ".join(local_env.keys())
            return f"Execution success. Defined variables: {names}"

        return "Execution success. (No variables defined.)"

    except Exception as e:
        tb = traceback.format_exc(limit=2)
        raise ToolError(f"Python execution failed: {e}\n{tb}")


def web_search_stub(query: str) -> str:
    """
    Stub for a web search. In a real system, this would call an external API.

    We keep it simple here to show tool-selection logic without needing
    external credentials.
    """
    return (
        "WEB_SEARCH_RESULT (stub): in a full deployment the agent would call "
        f"a web search API with query: '{query}'. Here we simply return this "
        "placeholder string so the reasoning pipeline can be demonstrated."
    )


# Tool registry the agent can look up by name.
TOOLS = {
    "run_python": {
        "fn": run_python_code,
        "description": (
            "Execute small Python snippets to check code, run examples, "
            "or inspect variable values."
        ),
    },
    "web_search": {
        "fn": web_search_stub,
        "description": (
            "Look up external information about Python concepts or error messages "
            "(stubbed out for this project)."
        ),
    },
}
