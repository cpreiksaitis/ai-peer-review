"""Communication reviewer agent - focuses on writing clarity and presentation."""

from dataclasses import dataclass, field

from .base import BaseReviewerAgent, Message
from src.prompts import COMMUNICATION_SYSTEM_PROMPT


@dataclass
class CommunicationAgent(BaseReviewerAgent):
    """
    Communication reviewer specializing in writing quality and presentation.

    Focuses on:
    - Writing clarity and precision
    - Logical organization
    - Figure and table quality
    - Abstract completeness
    - Accessibility to audience
    """

    name: str = "Communication Reviewer"
    model: str = "gemini/gemini-flash-lite-latest"
    temperature: float = 0.7
    conversation_history: list[Message] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return COMMUNICATION_SYSTEM_PROMPT

    @property
    def role_description(self) -> str:
        return "Scientific writing, figures, and presentation expert"

