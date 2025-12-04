"""Search providers for agentic literature discovery."""

from .base import SearchProvider, SearchResult, SearchSession
from .openai_provider import OpenAISearchProvider
from .gemini_provider import GeminiSearchProvider
from .claude_provider import ClaudeSearchProvider
from .perplexity_provider import PerplexitySearchProvider
from .pubmed_provider import PubMedSearchProvider

PROVIDERS = {
    "openai": OpenAISearchProvider,
    "gemini": GeminiSearchProvider,
    "claude": ClaudeSearchProvider,
    "perplexity": PerplexitySearchProvider,
    "pubmed": PubMedSearchProvider,  # Our existing non-agentic approach
}


def get_provider(name: str, **kwargs) -> SearchProvider:
    """Get a search provider by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)


def list_providers() -> list[dict]:
    """List available providers with their info."""
    return [
        {
            "name": "openai",
            "display_name": "OpenAI Web Search",
            "description": "Uses GPT with web_search tool, can filter to PubMed",
            "agentic": True,
            "requires_key": "OPENAI_API_KEY",
        },
        {
            "name": "gemini",
            "display_name": "Gemini Grounded Search",
            "description": "Uses Gemini with Google Search grounding",
            "agentic": True,
            "requires_key": "GOOGLE_API_KEY",
        },
        {
            "name": "claude",
            "display_name": "Claude Web Search",
            "description": "Uses Claude with web search tool",
            "agentic": True,
            "requires_key": "ANTHROPIC_API_KEY",
        },
        {
            "name": "perplexity",
            "display_name": "Perplexity Sonar",
            "description": "Native search-augmented LLM with academic focus",
            "agentic": True,
            "requires_key": "PERPLEXITY_API_KEY",
        },
        {
            "name": "pubmed",
            "display_name": "PubMed Direct",
            "description": "Direct PubMed API search with LLM query generation",
            "agentic": False,
            "requires_key": "PUBMED_EMAIL",
        },
    ]


__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearchSession",
    "OpenAISearchProvider",
    "GeminiSearchProvider",
    "ClaudeSearchProvider",
    "PerplexitySearchProvider",
    "PubMedSearchProvider",
    "get_provider",
    "list_providers",
    "PROVIDERS",
]

