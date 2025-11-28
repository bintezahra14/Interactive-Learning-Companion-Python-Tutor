"""
memory.py

Defines the data structures for learner memory and interaction history.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import textwrap


@dataclass
class InteractionRecord:
    """Single interaction between learner and agent."""
    user_input: str
    agent_answer: str
    correctness: Optional[bool] = None
    reward: float = 0.0
    difficulty: int = 1


@dataclass
class LearnerProfile:
    """
    Long-term representation of the learner.

    skill_level: 1 (beginner) .. 5 (advanced)
    """
    name: str = "Student"
    skill_level: int = 1
    target_language: str = "python"
    history: List[InteractionRecord] = field(default_factory=list)

    def estimate_skill(self) -> float:
        """Estimate effective skill level based on past rewarded interactions."""
        if not self.history:
            return float(self.skill_level)

        diffs = [h.difficulty for h in self.history if h.reward > 0]
        if not diffs:
            return float(self.skill_level)

        return sum(diffs) / len(diffs)


class MemorySystem:
    """
    Combines long-term learner profile and short-term conversation buffer.
    """

    def __init__(self, max_buffer: int = 10):
        self.learner_profile = LearnerProfile()
        self.short_term_buffer: List[Dict[str, str]] = []
        self.max_buffer = max_buffer

    # --- conversation buffer ---

    def add_message(self, role: str, content: str) -> None:
        """Append a message to short-term memory."""
        self.short_term_buffer.append({"role": role, "content": content})
        if len(self.short_term_buffer) > self.max_buffer:
            self.short_term_buffer.pop(0)

    # --- interaction history ---

    def add_interaction(self, record: InteractionRecord) -> None:
        """Store an interaction in the learner's long-term history."""
        self.learner_profile.history.append(record)

    # --- summary for LLM context ---

    def summary(self) -> str:
        """Return a human-readable summary to include in LLM prompts."""
        skill_est = self.learner_profile.estimate_skill()
        txt = f"""
        Learner name: {self.learner_profile.name}
        Approx skill level (1-5): {skill_est:.1f}
        Target language: {self.learner_profile.target_language}
        Number of past interactions: {len(self.learner_profile.history)}
        """
        return textwrap.dedent(txt).strip()
