"""
agent.py

Defines the LearningAgent class and safety / boundary logic.
The agent uses a ReAct-style controller to decide when to call tools.
"""

import json
import textwrap
from typing import Any, Dict, Optional

from memory import MemorySystem, InteractionRecord  # same folder
from tools import TOOLS, ToolError


# ---------------- Safety & Boundary Logic ---------------- #

DISALLOWED_KEYWORDS = [
    "suicide",
    "kill myself",
    "self-harm",
    "harm others",
    "bomb",
    "explosive",
    "weapon",
    "terrorist",
]

# keywords that strongly suggest a Python-related programming question
PYTHON_KEYWORDS = [
    "python",
    "variable",
    "function",
    "loop",
    "for ",
    "while ",
    "list",
    "tuple",
    "dictionary",
    "dict",
    "class",
    "object",
    "error",
    "traceback",
    "recursion",
    "def ",
    "print(",
    "append(",
    "extend(",
    "len(",
    "index(",
]


def is_safe_input(user_input: str) -> bool:
    """Return False if the input contains disallowed content."""
    text = user_input.lower()
    return not any(k in text for k in DISALLOWED_KEYWORDS)


def is_python_question(user_input: str) -> bool:
    """Heuristic check: does the text look like a Python programming question?"""
    text = user_input.lower()
    return any(k in text for k in PYTHON_KEYWORDS)


def enforce_boundaries(user_input: str) -> Optional[str]:
    """
    Apply safety and scope rules.

    Returns:
        - A refusal/explanation message string if not allowed.
        - None if the question is safe and in scope.
    """
    if not is_safe_input(user_input):
        return (
            "I’m sorry, but I can’t help with harmful or unsafe topics. "
            "If you’re in distress, please reach out to a trusted person "
            "or a professional support line."
        )

    if not is_python_question(user_input):
        return (
            "I’m designed to help with learning Python programming. "
            "Could you rephrase your question so it is about Python code "
            "or concepts?"
        )

    return None


# ---------------- LLM System Prompt ---------------- #

SYSTEM_PROMPT = """
You are an Interactive Learning Companion that tutors students in Python.

You must:
- Adapt explanations to the learner's skill level.
- Decide whether to use tools:
  - run_python(code: str): execute small Python code snippets.
  - web_search(query: str): look up Python concepts (stubbed).
- Follow a ReAct-style pattern:
  1. Think about the question and context.
  2. Decide whether a tool is needed.
  3. Either call a tool or answer directly.

You ALWAYS respond in STRICT JSON with schema:

{
  "thought": "short natural language summary of your reasoning",
  "action": "run_python" | "web_search" | "none",
  "action_input": "string input for the tool, or empty string if none",
  "tutor_reply": "message to the learner BEFORE feedback/reward",
  "suggested_difficulty": 1-5
}

Constraints:
- If you are unsure, prefer using a tool.
- Keep tutor_reply clear, kind, and focused on Python.
- suggested_difficulty: 1 = very easy, 5 = very challenging.
"""


# ---------------- Learning Agent ---------------- #

class LearningAgent:
    """
    Main agent controller. Orchestrates:
    - safety checks
    - LLM reasoning (ReAct style)
    - tool usage
    - simple RL-style reward & policy update
    """

    def __init__(self, llm, memory: MemorySystem):
        """
        Args:
            llm: an object with .generate_content(prompt: str) -> response
            memory: MemorySystem instance
        """
        self.llm = llm
        self.memory = memory

    # ----- internal helpers -----

    def _call_llm_controller(self, user_input: str) -> Dict[str, Any]:
        """Ask the LLM what action to take and what to say."""
        context = self.memory.summary()
        convo = self.memory.short_term_buffer

        prompt = textwrap.dedent(f"""
        SYSTEM INSTRUCTIONS:
        {SYSTEM_PROMPT}

        Learner context:
        {context}

        Recent conversation (last {len(convo)} turns):
        {convo}

        New question from learner:
        {user_input}
        """)

        response = self.llm.generate_content(prompt)
        raw = response.text

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat the whole response as the final answer
            data = {
                "thought": "Failed to parse JSON; using raw LLM output.",
                "action": "none",
                "action_input": "",
                "tutor_reply": raw,
                "suggested_difficulty": 1,
            }
        return data

    def _run_tool(self, action: str, action_input: str) -> str:
        """Execute a tool and handle errors gracefully."""
        if action not in TOOLS:
            return f"Tool '{action}' not found."

        tool = TOOLS[action]
        try:
            output = tool["fn"](action_input)
            return f"Tool '{action}' output:\n{output}"
        except ToolError as e:
            return f"Tool '{action}' failed with error:\n{e}"
        except Exception as e:  # pragma: no cover - unexpected
            return f"Unexpected error in tool '{action}': {e}"

    @staticmethod
    def _compute_reward(
        correctness: Optional[bool],
        user_rating: Optional[int],
    ) -> float:
        """
        Simple reward function combining correctness and subjective rating.

        correctness: True, False, or None
        user_rating: integer 1..5 or None
        """
        reward = 0.0

        if correctness is True:
            reward += 1.0
        elif correctness is False:
            reward -= 0.5

        if user_rating is not None:
            # center around 3 so that 3 -> 0, 5 -> +0.6, 1 -> -0.6
            reward += (user_rating - 3) * 0.3

        return reward

    def _update_policy(self, record: InteractionRecord) -> None:
        """
        Very simple policy improvement:
        - If reward is strongly positive -> increase skill_level (max 5)
        - If strongly negative -> decrease skill_level (min 1)
        """
        profile = self.memory.learner_profile
        profile.history.append(record)

        if record.reward > 0.5 and profile.skill_level < 5:
            profile.skill_level += 1
        elif record.reward < -0.5 and profile.skill_level > 1:
            profile.skill_level -= 1

    # ----- public API -----

    def handle_turn(
        self,
        user_input: str,
        correctness: Optional[bool] = None,
        user_rating: Optional[int] = None,
    ) -> str:
        """
        Process a single learner query end-to-end.

        Returns:
            Final answer string to show to the learner.
        """

        # 1. Safety / boundaries
        boundary_msg = enforce_boundaries(user_input)
        if boundary_msg is not None:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", boundary_msg)
            return boundary_msg

        # 2. Ask controller LLM what to do
        controller_output = self._call_llm_controller(user_input)
        thought = controller_output.get("thought", "")
        action = controller_output.get("action", "none")
        action_input = controller_output.get("action_input", "")
        tutor_reply = controller_output.get("tutor_reply", "")
        suggested_difficulty = int(controller_output.get("suggested_difficulty", 1))

        # 3. Optional tool call + reflection step
        tool_observation = ""
        if action != "none":
            tool_observation = self._run_tool(action, action_input)

            reflection_prompt = textwrap.dedent(f"""
            You are a Python tutor. Explain clearly and kindly.

            Earlier you decided to use the tool '{action}' with input:

            {action_input}

            The tool returned:

            {tool_observation}

            Now, refine your explanation for the learner, making sure it is
            correct, concise, and appropriate for their skill level.
            """)
            reflection_response = self.llm.generate_content(reflection_prompt)
            tutor_reply = reflection_response.text

        # 4. RL-style feedback & policy update
        reward = self._compute_reward(correctness, user_rating)
        record = InteractionRecord(
            user_input=user_input,
            agent_answer=tutor_reply,
            correctness=correctness,
            reward=reward,
            difficulty=suggested_difficulty,
        )
        self._update_policy(record)

        # 5. Update memory buffers
        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", tutor_reply)

        # 6. Transparency note if a tool was used
        meta_note = ""
        if action != "none":
            meta_note = f"\n\n_(I used the **{action}** tool to help answer this.)_"

        return tutor_reply + meta_note
