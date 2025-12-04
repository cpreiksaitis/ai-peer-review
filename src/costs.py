"""Cost tracking for LLM API calls."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import litellm


# Model pricing per 1M tokens (from litellm model info)
MODEL_PRICING = {
    # OpenAI - GPT-5 series
    "gpt-5.1-chat-latest": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    # OpenAI - Legacy
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Anthropic - Claude 4.5 series
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Anthropic - Legacy
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # Google - Gemini series (with provider prefix)
    "gemini/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini/gemini-flash-latest": {"input": 0.30, "output": 2.50},
    "gemini/gemini-flash-lite-latest": {"input": 0.10, "output": 0.40},
    # Google - Legacy
    "gemini/gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CostEntry:
    """A single cost tracking entry."""

    model: str
    agent_name: str
    operation: str  # "initial_review", "debate", "synthesis", etc.
    usage: TokenUsage
    cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostTracker:
    """Tracks costs across a review session."""

    entries: list[CostEntry] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(e.usage.total_tokens for e in self.entries)

    @property
    def cost_by_agent(self) -> dict[str, float]:
        """Cost breakdown by agent."""
        costs: dict[str, float] = {}
        for entry in self.entries:
            costs[entry.agent_name] = costs.get(entry.agent_name, 0) + entry.cost_usd
        return costs

    @property
    def cost_by_operation(self) -> dict[str, float]:
        """Cost breakdown by operation type."""
        costs: dict[str, float] = {}
        for entry in self.entries:
            costs[entry.operation] = costs.get(entry.operation, 0) + entry.cost_usd
        return costs

    def add_entry(
        self,
        model: str,
        agent_name: str,
        operation: str,
        response: Any,
    ) -> CostEntry:
        """
        Add a cost entry from an LLM response.

        Args:
            model: The model used
            agent_name: Name of the agent making the call
            operation: Type of operation (e.g., "initial_review")
            response: The LiteLLM response object

        Returns:
            The created CostEntry
        """
        # Extract usage from response
        usage_data = getattr(response, "usage", None)
        if usage_data:
            usage = TokenUsage(
                prompt_tokens=getattr(usage_data, "prompt_tokens", 0),
                completion_tokens=getattr(usage_data, "completion_tokens", 0),
                total_tokens=getattr(usage_data, "total_tokens", 0),
            )
        else:
            usage = TokenUsage()

        # Calculate cost
        cost = calculate_cost(model, usage)

        entry = CostEntry(
            model=model,
            agent_name=agent_name,
            operation=operation,
            usage=usage,
            cost_usd=cost,
        )
        self.entries.append(entry)
        return entry

    def get_summary(self) -> dict:
        """Get a summary of costs."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "num_calls": len(self.entries),
            "by_agent": {k: round(v, 4) for k, v in self.cost_by_agent.items()},
            "by_operation": {k: round(v, 4) for k, v in self.cost_by_operation.items()},
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        summary = self.get_summary()
        lines = [
            "## Cost Summary",
            f"**Total Cost:** ${summary['total_cost_usd']:.4f}",
            f"**Total Tokens:** {summary['total_tokens']:,}",
            f"**API Calls:** {summary['num_calls']}",
            "",
            "### By Agent",
        ]
        for agent, cost in summary["by_agent"].items():
            lines.append(f"- {agent}: ${cost:.4f}")

        lines.append("")
        lines.append("### By Operation")
        for op, cost in summary["by_operation"].items():
            lines.append(f"- {op}: ${cost:.4f}")

        return "\n".join(lines)


def calculate_cost(model: str, usage: TokenUsage) -> float:
    """
    Calculate cost in USD for token usage.

    Args:
        model: The model name
        usage: Token usage

    Returns:
        Cost in USD
    """
    # Normalize model name for lookup
    model_key = model.lower()

    # Try to find pricing
    pricing = MODEL_PRICING.get(model_key)

    if not pricing:
        # Try partial matches
        for key, price in MODEL_PRICING.items():
            if key in model_key or model_key in key:
                pricing = price
                break

    if not pricing:
        # Default to GPT-4o pricing as fallback
        pricing = MODEL_PRICING["gpt-4o"]

    input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def estimate_review_cost(
    manuscript_length: int,
    num_agents: int = 4,
    debate_rounds: int = 2,
    literature_results: int = 10,
) -> dict:
    """
    Estimate the cost of a review before running.

    Args:
        manuscript_length: Approximate length of manuscript in characters
        num_agents: Number of reviewer agents
        debate_rounds: Number of debate rounds
        literature_results: Number of literature results to include

    Returns:
        Cost estimate breakdown
    """
    # Rough token estimates (1 token ≈ 4 characters)
    manuscript_tokens = manuscript_length // 4
    literature_tokens = literature_results * 500  # ~500 tokens per abstract
    base_prompt_tokens = 2000  # System prompts, instructions

    # Estimate per-call tokens
    initial_review_input = manuscript_tokens + literature_tokens + base_prompt_tokens
    initial_review_output = 1500  # ~1500 tokens per review

    debate_input = initial_review_input + (num_agents * initial_review_output)
    debate_output = 800  # Shorter debate responses

    synthesis_input = initial_review_input + (num_agents * initial_review_output * (debate_rounds + 1))
    synthesis_output = 2000

    # Calculate costs for each phase
    # Use average pricing across models
    avg_input_price = 2.50  # $/1M tokens
    avg_output_price = 10.00

    def calc(input_tokens: int, output_tokens: int, multiplier: int = 1) -> float:
        input_cost = (input_tokens / 1_000_000) * avg_input_price * multiplier
        output_cost = (output_tokens / 1_000_000) * avg_output_price * multiplier
        return input_cost + output_cost

    costs = {
        "initial_reviews": calc(initial_review_input, initial_review_output, num_agents),
        "debate_rounds": calc(debate_input, debate_output, num_agents * debate_rounds),
        "final_positions": calc(debate_input, 500, num_agents),
        "synthesis": calc(synthesis_input, synthesis_output, 1),
        "literature_queries": calc(2000, 200, 1),
    }

    costs["total_estimated"] = sum(costs.values())

    return {k: round(v, 4) for k, v in costs.items()}


# Global cost tracker instance
_global_tracker: CostTracker | None = None


def get_global_tracker() -> CostTracker:
    """Get the global cost tracker, creating if needed."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker() -> CostTracker:
    """Reset and return a new global cost tracker."""
    global _global_tracker
    _global_tracker = CostTracker()
    return _global_tracker

