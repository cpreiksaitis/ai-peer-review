"""Literature search and retrieval using PyPaperRetriever."""

import json
import os
import re
import time
import random
import threading
from dataclasses import dataclass, field

import litellm
from pypaperretriever import PubMedSearcher

# Allow dropping unsupported params
litellm.drop_params = True

# PubMed rate limiting
# NCBI allows 3 requests/second without API key, 10/second with one
# We'll be conservative to avoid issues in parallel execution
_pubmed_lock = threading.Lock()
_last_pubmed_request = 0.0
PUBMED_MIN_INTERVAL = 0.5  # seconds between requests (2 req/sec max)
PUBMED_MAX_RETRIES = 3


def _rate_limit_pubmed():
    """Enforce rate limiting for PubMed API calls."""
    global _last_pubmed_request
    with _pubmed_lock:
        now = time.time()
        elapsed = now - _last_pubmed_request
        if elapsed < PUBMED_MIN_INTERVAL:
            sleep_time = PUBMED_MIN_INTERVAL - elapsed + random.uniform(0.1, 0.3)
            time.sleep(sleep_time)
        _last_pubmed_request = time.time()


@dataclass
class RelatedPaper:
    """Represents a related paper from PubMed."""

    pmid: str
    title: str
    authors: list[str]
    abstract: str
    journal: str
    pub_date: str
    doi: str | None = None
    relevance_score: float = 0.0
    relevance_reason: str = ""


def get_pubmed_email() -> str:
    """Get email for PubMed API from environment or raise error."""
    email = os.environ.get("PUBMED_EMAIL", "")
    if not email:
        raise ValueError(
            "PUBMED_EMAIL environment variable must be set for PubMed API access. "
            "This is required by NCBI for API usage tracking."
        )
    return email


def search_related_literature(
    query: str,
    max_results: int = 10,
    email: str | None = None,
) -> list[RelatedPaper]:
    """
    Search PubMed for papers related to the given query.

    Args:
        query: Search query string (e.g., keywords from manuscript)
        max_results: Maximum number of results to return
        email: Email for PubMed API (uses PUBMED_EMAIL env var if not provided)

    Returns:
        List of RelatedPaper objects with metadata
    """
    if email is None:
        email = get_pubmed_email()

    last_error = None
    for attempt in range(PUBMED_MAX_RETRIES):
        try:
            # Rate limit before making request
            _rate_limit_pubmed()
            
            searcher = PubMedSearcher(search_string=query, email=email)
            searcher.search(count=max_results)
            
            papers = []
            # Results are stored in searcher.df (pandas DataFrame)
            if hasattr(searcher, 'df') and searcher.df is not None and len(searcher.df) > 0:
                for _, row in searcher.df.iterrows():
                    # Parse authors - can be string or list
                    authors = row.get("authors", [])
                    if isinstance(authors, str):
                        authors = [a.strip() for a in authors.split(",") if a.strip()]
                    elif not isinstance(authors, list):
                        authors = []

                    paper = RelatedPaper(
                        pmid=str(row.get("pmid", "")) or "",
                        title=str(row.get("title", "")) or "",
                        authors=authors,
                        abstract=str(row.get("abstract", "")) or "",
                        journal=str(row.get("journal_info", "")) or "",
                        pub_date=str(row.get("publication_date", "")) or "",
                        doi=row.get("doi") if row.get("doi") else None,
                    )
                    papers.append(paper)

            return papers
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check if it's a rate limit or transient error
            is_retryable = (
                "429" in str(e) or
                "too many" in error_str or
                "rate" in error_str or
                "timeout" in error_str or
                "connection" in error_str or
                "503" in str(e) or
                "502" in str(e)
            )
            
            if is_retryable and attempt < PUBMED_MAX_RETRIES - 1:
                # Exponential backoff
                delay = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(delay)
                continue
            
            # Non-retryable or final attempt
            raise
    
    # Should not reach here, but just in case
    if last_error:
        raise last_error
    return []


def extract_search_keywords(manuscript_text: str) -> list[str]:
    """
    Extract key search terms from manuscript text for literature search.

    This is a simple extraction - the orchestrator agent will generate
    more sophisticated queries using LLM.

    Args:
        manuscript_text: Full text of the manuscript

    Returns:
        List of potential search query strings
    """
    # Extract title (usually in first few lines)
    lines = manuscript_text.split("\n")
    title_lines = []
    for line in lines[:20]:
        line = line.strip()
        if line and not line.startswith("---"):
            title_lines.append(line)
            if len(title_lines) >= 2:
                break

    # Use the potential title as a search query
    queries = []
    if title_lines:
        queries.append(" ".join(title_lines[:2]))

    return queries


def score_paper_relevance(
    paper: RelatedPaper,
    manuscript_summary: str,
    model: str = "gpt-5-nano",
) -> tuple[float, str]:
    """
    Score a paper's relevance to the manuscript using an LLM.
    
    DEPRECATED: Use score_papers_batch for efficiency.

    Args:
        paper: The paper to score
        manuscript_summary: Brief summary of the manuscript
        model: LLM model to use for scoring

    Returns:
        Tuple of (score, reason)
    """
    from src.prompts import RELEVANCE_SCORING_PROMPT

    prompt = RELEVANCE_SCORING_PROMPT.format(
        manuscript_summary=manuscript_summary,
        paper_title=paper.title,
        paper_abstract=paper.abstract[:1000] if paper.abstract else "No abstract available",
    )

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content

        # Parse JSON response
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            data = json.loads(json_match.group())
            return float(data.get("score", 5)), data.get("reason", "")
    except Exception:
        pass

    return 5.0, "Unable to score"


def score_papers_batch(
    papers: list[RelatedPaper],
    manuscript_summary: str,
    model: str = "gpt-5-nano",
) -> dict[str, tuple[float, str]]:
    """
    Score multiple papers' relevance in a single LLM call.
    
    Much more efficient than calling score_paper_relevance per paper.

    Args:
        papers: List of papers to score
        manuscript_summary: Brief summary of the manuscript
        model: LLM model to use for scoring

    Returns:
        Dict mapping PMID to (score, reason) tuple
    """
    if not papers:
        return {}
    
    # Build batch prompt with all papers
    papers_text = []
    for i, paper in enumerate(papers):
        abstract = paper.abstract[:500] if paper.abstract else "No abstract"
        papers_text.append(f"""Paper {i+1} (PMID: {paper.pmid}):
Title: {paper.title}
Abstract: {abstract}""")
    
    prompt = f"""You are evaluating the relevance of academic papers to a manuscript under review.

## Manuscript Summary
{manuscript_summary}

## Papers to Score
{chr(10).join(papers_text)}

## Task
Score each paper's relevance to the manuscript on a scale of 1-10:
- 10: Directly relevant, addresses same research question or methodology
- 7-9: Highly relevant, similar topic or important methodological parallel
- 4-6: Moderately relevant, related field or tangential connection
- 1-3: Low relevance, different topic or minimal connection

Return a JSON array with scores for ALL papers:
```json
[
  {{"pmid": "12345678", "score": 8, "reason": "Brief explanation"}},
  {{"pmid": "87654321", "score": 5, "reason": "Brief explanation"}}
]
```

Score all {len(papers)} papers:"""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = response.choices[0].message.content

        # Parse JSON array from response
        json_match = re.search(r'\[[\s\S]*?\]', content)
        if json_match:
            scores_list = json.loads(json_match.group())
            results = {}
            for item in scores_list:
                pmid = str(item.get("pmid", ""))
                score = float(item.get("score", 5))
                reason = item.get("reason", "")
                results[pmid] = (score, reason)
            return results
    except Exception as e:
        # Log but don't fail - return empty and let caller handle
        pass

    return {}


def filter_and_rank_papers(
    papers: list[RelatedPaper],
    manuscript_summary: str,
    top_n: int = 10,
    min_score: float = 5.0,
    model: str = "gpt-5-nano",
) -> list[RelatedPaper]:
    """
    Filter and rank papers by relevance to the manuscript.

    Args:
        papers: List of papers to filter
        manuscript_summary: Brief summary of manuscript
        top_n: Number of top papers to return
        min_score: Minimum relevance score to include
        model: Model for scoring

    Returns:
        Filtered and ranked list of papers
    """
    # Filter to papers with abstracts for scoring
    papers_with_abstracts = [p for p in papers if p.abstract]
    papers_without_abstracts = [p for p in papers if not p.abstract]
    
    # Score papers with abstracts in a single batch call
    if papers_with_abstracts:
        scores = score_papers_batch(papers_with_abstracts, manuscript_summary, model)
        
        for paper in papers_with_abstracts:
            if paper.pmid in scores:
                paper.relevance_score, paper.relevance_reason = scores[paper.pmid]
            else:
                # Fallback if paper wasn't in results
                paper.relevance_score = 5.0
                paper.relevance_reason = "Score not returned"
    
    # Low score for papers without abstracts
    for paper in papers_without_abstracts:
        paper.relevance_score = 3.0
        paper.relevance_reason = "No abstract available"

    # Combine and filter by minimum score, sort by relevance
    all_papers = papers_with_abstracts + papers_without_abstracts
    filtered = [p for p in all_papers if p.relevance_score >= min_score]
    filtered.sort(key=lambda p: p.relevance_score, reverse=True)

    return filtered[:top_n]


def format_literature_context(papers: list[RelatedPaper], include_scores: bool = True) -> str:
    """
    Format related papers into a context string for the review agents.

    Args:
        papers: List of RelatedPaper objects
        include_scores: Whether to include relevance scores

    Returns:
        Formatted string with paper summaries
    """
    if not papers:
        return "No related literature found."

    sections = ["## Related Literature\n"]

    for i, paper in enumerate(papers, start=1):
        authors_str = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += " et al."

        score_info = ""
        if include_scores and paper.relevance_score > 0:
            score_info = f"\n**Relevance:** {paper.relevance_score:.1f}/10 - {paper.relevance_reason}"

        section = f"""### {i}. {paper.title}
**Authors:** {authors_str}
**Journal:** {paper.journal} ({paper.pub_date})
**PMID:** {paper.pmid}{score_info}

**Abstract:** {paper.abstract[:500]}{'...' if len(paper.abstract) > 500 else ''}
"""
        sections.append(section)

    return "\n".join(sections)

