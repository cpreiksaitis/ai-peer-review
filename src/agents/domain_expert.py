"""Domain expert reviewer agent - focuses on clinical relevance and novelty."""

from dataclasses import dataclass, field

from .base import BaseReviewerAgent, Message
from src.prompts import DOMAIN_EXPERT_SYSTEM_PROMPT


@dataclass
class DomainExpertAgent(BaseReviewerAgent):
    """
    Domain expert reviewer specializing in clinical relevance and field context.

    Focuses on:
    - Clinical significance of research
    - Novelty and contribution
    - Comparison with existing literature
    - Practical implications
    - Field-specific considerations
    """

    name: str = "Domain Expert"
    model: str = "gemini/gemini-flash-lite-latest"
    temperature: float = 0.7
    conversation_history: list[Message] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return DOMAIN_EXPERT_SYSTEM_PROMPT

    @property
    def role_description(self) -> str:
        return "Clinical relevance, novelty, and field context expert"

