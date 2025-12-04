"""Configuration loading utilities."""

import os
from pathlib import Path

import yaml


def load_config(config_path: Path | str | None = None) -> dict:
    """Load configuration from config.yaml."""
    if config_path is None:
        config_path = Path("config.yaml")
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def check_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "perplexity": bool(os.environ.get("PERPLEXITY_API_KEY")),
        "pubmed": bool(os.environ.get("PUBMED_EMAIL")),
    }


def get_default_model() -> str:
    """Get the default model from config or fallback."""
    config = load_config()
    return config.get("models", {}).get("orchestrator", "gpt-4o-mini")

