"""Methodologist reviewer agent - focuses on study design and statistics."""

from dataclasses import dataclass, field

from .base import BaseReviewerAgent, Message
from src.prompts import METHODOLOGIST_SYSTEM_PROMPT


@dataclass
class MethodologistAgent(BaseReviewerAgent):
    """
    Methodologist reviewer specializing in study design and statistical analysis.

    Focuses on:
    - Appropriateness of study design
    - Statistical methods and interpretation
    - Sample size and power
    - Bias assessment
    - Reproducibility
    """

    name: str = "Methodologist"
    model: str = "gpt-5-nano"
    temperature: float = 0.7
    conversation_history: list[Message] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return METHODOLOGIST_SYSTEM_PROMPT

    @property
    def role_description(self) -> str:
        return "Study design, statistics, and methodology expert"

