"""Ethics and impact reviewer agent - focuses on ethics, safety, and broader implications."""

from dataclasses import dataclass, field

from .base import BaseReviewerAgent, Message
from src.prompts import ETHICS_SYSTEM_PROMPT


@dataclass
class EthicsAgent(BaseReviewerAgent):
    """
    Ethics and impact reviewer specializing in research ethics and broader implications.

    Focuses on:
    - Research ethics (IRB, consent)
    - Patient safety implications
    - Limitations and generalizability
    - Equity and inclusion
    - Potential for misuse
    - Conflicts of interest
    """

    name: str = "Ethics Reviewer"
    model: str = "gpt-5-nano"
    temperature: float = 0.7
    conversation_history: list[Message] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        return ETHICS_SYSTEM_PROMPT

    @property
    def role_description(self) -> str:
        return "Research ethics, patient safety, and broader impact expert"

