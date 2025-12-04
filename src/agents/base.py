"""Base reviewer agent class using LiteLLM."""

import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm
from litellm.utils import supports_pdf_input
from pydantic import BaseModel

from src.costs import CostTracker, get_global_tracker
from src.document import create_pdf_message_content

# Allow LiteLLM to drop unsupported parameters for different models
litellm.drop_params = True

# Retry configuration
MAX_RETRIES = 5
BASE_DELAY = 1.0  # seconds
MAX_DELAY = 60.0  # seconds


def retry_with_backoff(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    Retry a function with exponential backoff for rate limit errors.
    
    Handles:
    - 429 Too Many Requests
    - RateLimitError
    - APIConnectionError (transient)
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Check if it's a retryable error
            is_rate_limit = (
                "429" in str(e) or
                "rate" in error_str or
                "too many" in error_str or
                "ratelimit" in error_type.lower() or
                "quota" in error_str or
                "overloaded" in error_str
            )
            
            is_transient = (
                "connection" in error_str or
                "timeout" in error_str or
                "503" in str(e) or
                "502" in str(e)
            )
            
            if is_rate_limit or is_transient:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    # print(f"  [Rate limit] Retry {attempt + 1}/{max_retries} in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
            
            # Non-retryable error, raise immediately
            raise
    
    # All retries exhausted
    raise last_exception


class ReviewSection(BaseModel):
    """Structured review section."""

    strengths: list[str] = []
    concerns: list[str] = []
    comments: list[str] = []
    assessment: str = ""


class DebateResponse(BaseModel):
    """Structured debate response."""

    agreements: list[str] = []
    disagreements: list[str] = []
    new_observations: list[str] = []
    responses: list[str] = []


@dataclass
class Message:
    """A message in the conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class BaseReviewerAgent(ABC):
    """
    Base class for all reviewer agents.

    Each specialized agent inherits from this and provides its own system prompt.
    Uses LiteLLM for unified access to different LLM providers.
    """

    name: str
    model: str
    temperature: float = 0.7
    conversation_history: list[Message] = field(default_factory=list)
    cost_tracker: Optional[CostTracker] = None

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent type."""
        pass

    @property
    def role_description(self) -> str:
        """Brief description of this agent's role."""
        return self.name

    def _get_tracker(self) -> CostTracker:
        """Get the cost tracker to use."""
        return self.cost_tracker or get_global_tracker()

    def _build_messages(self, user_content: str, pdf_base64: str | None = None) -> list[dict]:
        """Build the messages list for the LLM call, optionally with PDF."""
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add current user message (with PDF if provided)
        if pdf_base64:
            # Correct LiteLLM format for PDF vision
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content},
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:application/pdf;base64,{pdf_base64}",
                        }
                    },
                ]
            })
        else:
            messages.append({"role": "user", "content": user_content})

        return messages

    def _call_llm(self, messages: list[dict[str, str]], operation: str = "unknown") -> str:
        """Make a call to the LLM via LiteLLM with cost tracking and retry logic."""
        
        # Some models (like gpt-5-nano) don't support custom temperature
        # litellm.drop_params should handle this, but we'll be explicit
        model_lower = self.model.lower()
        supports_temperature = not any(x in model_lower for x in ["nano", "o1", "o3"])
        
        def make_call():
            kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if supports_temperature:
                kwargs["temperature"] = self.temperature
            return litellm.completion(**kwargs)
        
        # Use retry with backoff for rate limits
        response = retry_with_backoff(make_call)

        # Track the cost
        tracker = self._get_tracker()
        tracker.add_entry(
            model=self.model,
            agent_name=self.name,
            operation=operation,
            response=response,
        )

        return response.choices[0].message.content

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(Message(role=role, content=content))

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def generate_initial_review(
        self,
        manuscript_text: str,
        literature_context: str = "",
        review_prompt: str = "",
        pdf_base64: str | None = None,
    ) -> str:
        """
        Generate an initial review of the manuscript.

        Args:
            manuscript_text: The full text of the manuscript
            literature_context: Formatted related literature for grounding
            review_prompt: Custom prompt for the review (uses default if empty)
            pdf_base64: Base64-encoded PDF for vision models (optional)

        Returns:
            The agent's initial review as a string
        """
        from .prompts import INITIAL_REVIEW_PROMPT

        prompt_parts = []

        if literature_context:
            prompt_parts.append(f"## Related Literature for Context\n{literature_context}\n")

        # Check if we should use PDF vision
        use_pdf_vision = (
            pdf_base64 is not None 
            and supports_pdf_input(model=self.model, custom_llm_provider=None)
        )

        if use_pdf_vision:
            # Use PDF directly - include text as backup context
            prompt_parts.append("## Manuscript\nPlease review the attached PDF manuscript. The extracted text is provided below for reference.\n")
            prompt_parts.append(f"### Extracted Text (for reference)\n{manuscript_text[:15000]}...\n")
        else:
            # Use text extraction only
            prompt_parts.append(f"## Manuscript\n{manuscript_text}\n")
        
        prompt_parts.append(review_prompt or INITIAL_REVIEW_PROMPT)

        text_prompt = "\n".join(prompt_parts)
        
        if use_pdf_vision:
            # Build message with PDF attachment
            messages = [{"role": "system", "content": self.system_prompt}]
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
            # User message with PDF
            pdf_content = create_pdf_message_content(pdf_base64, text_prompt)
            messages.append({"role": "user", "content": pdf_content})
        else:
            messages = self._build_messages(text_prompt)

        review = self._call_llm(messages)

        # Store in history for debate rounds (text only for history)
        self.add_to_history("user", text_prompt)
        self.add_to_history("assistant", review)

        return review

    def participate_in_debate(
        self,
        other_reviews: dict[str, str],
        debate_prompt: str = "",
    ) -> str:
        """
        Participate in a debate round by responding to other reviewers.

        Args:
            other_reviews: Dictionary mapping agent names to their reviews
            debate_prompt: Custom prompt for the debate (uses default if empty)

        Returns:
            The agent's debate response as a string
        """
        from .prompts import DEBATE_ROUND_PROMPT

        # Format other reviews for context
        other_reviews_text = "\n\n".join(
            f"## {name}'s Review\n{review}" for name, review in other_reviews.items()
        )

        user_content = f"""## Other Reviewers' Comments
{other_reviews_text}

{debate_prompt or DEBATE_ROUND_PROMPT}"""

        messages = self._build_messages(user_content)
        response = self._call_llm(messages)

        # Store in history
        self.add_to_history("user", user_content)
        self.add_to_history("assistant", response)

        return response

    def get_final_position(self) -> str:
        """
        Get the agent's final position after debate.

        Returns:
            Summary of final position and recommendation
        """
        user_content = """Based on the full discussion, provide your final position:

1. Your final assessment (Strong/Moderate/Weak)
2. Key issues that MUST be addressed (if any)
3. Your recommendation (Accept/Minor Revision/Major Revision/Reject)
4. Your confidence in this recommendation (High/Medium/Low)

Be concise but specific."""

        messages = self._build_messages(user_content)
        response = self._call_llm(messages)

        self.add_to_history("user", user_content)
        self.add_to_history("assistant", response)

        return response


# Import prompts at module level to avoid circular imports
def _get_prompts():
    """Lazy import of prompts module."""
    from src import prompts
    return prompts


# Monkey-patch the generate_initial_review method to use the actual prompts
_original_generate_initial_review = BaseReviewerAgent.generate_initial_review


def _patched_generate_initial_review(self, manuscript_text, literature_context="", review_prompt="", pdf_base64=None):
    from src.prompts import INITIAL_REVIEW_PROMPT

    prompt_parts = []

    if literature_context:
        prompt_parts.append(f"## Related Literature for Context\n{literature_context}\n")

    prompt_parts.append(f"## Manuscript\n{manuscript_text}\n")
    prompt_parts.append(review_prompt or INITIAL_REVIEW_PROMPT)

    user_content = "\n".join(prompt_parts)
    messages = self._build_messages(user_content, pdf_base64=pdf_base64)

    review = self._call_llm(messages, operation="initial_review")

    self.add_to_history("user", user_content)
    self.add_to_history("assistant", review)

    return review


def _patched_participate_in_debate(self, other_reviews, debate_prompt=""):
    from src.prompts import DEBATE_ROUND_PROMPT

    other_reviews_text = "\n\n".join(
        f"## {name}'s Review\n{review}" for name, review in other_reviews.items()
    )

    user_content = f"""## Other Reviewers' Comments
{other_reviews_text}

{debate_prompt or DEBATE_ROUND_PROMPT}"""

    messages = self._build_messages(user_content)
    response = self._call_llm(messages, operation="debate")

    self.add_to_history("user", user_content)
    self.add_to_history("assistant", response)

    return response


def _patched_get_final_position(self):
    user_content = """Based on the full discussion, provide your final position:

1. Your final assessment (Strong/Moderate/Weak)
2. Key issues that MUST be addressed (if any)
3. Your recommendation (Accept/Minor Revision/Major Revision/Reject)
4. Your confidence in this recommendation (High/Medium/Low)

Be concise but specific."""

    messages = self._build_messages(user_content)
    response = self._call_llm(messages, operation="final_position")

    self.add_to_history("user", user_content)
    self.add_to_history("assistant", response)

    return response


# Apply patches
BaseReviewerAgent.generate_initial_review = _patched_generate_initial_review
BaseReviewerAgent.participate_in_debate = _patched_participate_in_debate
BaseReviewerAgent.get_final_position = _patched_get_final_position

