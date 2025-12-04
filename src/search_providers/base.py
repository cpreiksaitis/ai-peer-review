"""Base class for search providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """A single paper/article result from search."""
    
    title: str
    authors: list[str]
    abstract: str
    source: str  # e.g., "PubMed", "Google Scholar", "Web"
    url: str
    pmid: Optional[str] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    relevance_score: float = 0.0
    relevance_reason: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "source": self.source,
            "url": self.url,
            "pmid": self.pmid,
            "doi": self.doi,
            "journal": self.journal,
            "pub_date": self.pub_date,
            "relevance_score": self.relevance_score,
            "relevance_reason": self.relevance_reason,
        }


@dataclass
class SearchSession:
    """Results from a complete search session."""
    
    provider: str
    query_summary: str  # What the agent searched for
    results: list[SearchResult] = field(default_factory=list)
    reasoning: str = ""  # Agent's reasoning about the search
    queries_used: list[str] = field(default_factory=list)
    total_cost: float = 0.0
    tokens_used: int = 0
    search_steps: list[dict] = field(default_factory=list)  # For transparency
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "query_summary": self.query_summary,
            "results": [r.to_dict() for r in self.results],
            "reasoning": self.reasoning,
            "queries_used": self.queries_used,
            "total_cost": self.total_cost,
            "tokens_used": self.tokens_used,
            "search_steps": self.search_steps,
        }


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    name: str = "base"
    display_name: str = "Base Provider"
    supports_pdf: bool = False  # Whether this provider can process PDFs directly
    
    @abstractmethod
    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """
        Search for papers similar to the manuscript.
        
        Args:
            manuscript_text: Full text of the manuscript
            max_results: Maximum number of results to return
            focus_pubmed: Whether to focus search on PubMed/academic sources
            pdf_base64: Optional base64-encoded PDF for vision models
            
        Returns:
            SearchSession with results and metadata
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available (API key configured, etc.)."""
        pass
    
    def get_status(self) -> dict:
        """Get provider status information."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.is_available(),
            "supports_pdf": self.supports_pdf,
        }

