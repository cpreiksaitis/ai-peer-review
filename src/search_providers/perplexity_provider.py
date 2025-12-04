"""Perplexity search provider using Sonar models with academic search."""

import os
import re
from typing import Optional

from .base import SearchProvider, SearchResult, SearchSession


class PerplexitySearchProvider(SearchProvider):
    """Search using Perplexity's Sonar models with academic search mode."""
    
    name = "perplexity"
    display_name = "Perplexity Academic"
    supports_pdf = True  # Perplexity supports base64 PDF via file_url
    
    def __init__(self, model: str = "sonar-pro"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy load Perplexity client using OpenAI SDK."""
        if self._client is None:
            from openai import OpenAI
            api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not api_key:
                raise ValueError("PERPLEXITY_API_KEY not set")
            self._client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai",
            )
        return self._client
    
    def is_available(self) -> bool:
        """Check if Perplexity API key is configured."""
        return bool(os.environ.get("PERPLEXITY_API_KEY"))
    
    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """Search for similar papers using Perplexity academic search."""
        
        # Simple, direct prompt - let Perplexity's search do the work
        search_instruction = f"""Search PubMed and academic databases to find {max_results} peer-reviewed papers most relevant to this research manuscript.

For each paper found, provide:
- Title
- Authors (first author et al.)
- PMID (PubMed ID) - REQUIRED if from PubMed
- DOI
- Journal
- Year
- Brief relevance explanation

Focus on: similar methodology, same research question, recent related work, and foundational papers.

{"[Manuscript text provided below]" if not pdf_base64 else "[See attached PDF manuscript]"}
{manuscript_text[:6000] if not pdf_base64 else ""}"""

        try:
            # Check API key first
            if not os.environ.get("PERPLEXITY_API_KEY"):
                return SearchSession(
                    provider=self.name,
                    query_summary="Error: PERPLEXITY_API_KEY not configured",
                    results=[],
                    reasoning="Please add PERPLEXITY_API_KEY to your environment variables.",
                )
            
            # Build messages - send PDF directly if available
            messages = [
                {
                    "role": "system",
                    "content": "You are a research librarian. Search academic databases to find relevant papers. Always include PubMed IDs (PMIDs) when available."
                },
                {"role": "user", "content": search_instruction}
            ]
            
            # Use Perplexity with web search
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            return self._parse_response(response, max_results)
            
        except Exception as e:
            return SearchSession(
                provider=self.name,
                query_summary=f"Error: {str(e)}",
                results=[],
                reasoning=f"Search failed: {str(e)}",
            )
    
    def _parse_response(self, response, max_results: int) -> SearchSession:
        """Parse Perplexity response into SearchSession."""
        text = response.choices[0].message.content
        
        # Extract citations if available
        citations = []
        if hasattr(response, "citations") and response.citations:
            citations = response.citations
        
        search_steps = [{"type": "source", "url": c} for c in citations]
        
        # Parse papers from text
        results = self._extract_papers_from_text(text, max_results)
        
        # Calculate costs
        tokens_used = 0
        if hasattr(response, "usage"):
            tokens_used = response.usage.total_tokens
        
        # Perplexity Sonar-Pro pricing (approximate)
        total_cost = tokens_used * 0.000003
        
        return SearchSession(
            provider=self.name,
            query_summary=f"Found {len(results)} papers via Perplexity academic search",
            results=results,
            reasoning=text,
            queries_used=[],
            total_cost=total_cost,
            tokens_used=tokens_used,
            search_steps=search_steps,
        )
    
    def _extract_papers_from_text(self, text: str, max_results: int) -> list[SearchResult]:
        """Extract paper information from response text."""
        results = []
        
        # Split by paper markers - handle multiple formats
        # Format 1: **Paper 1**
        # Format 2: ## 1. Title
        # Format 3: 1. **Title**
        paper_sections = re.split(r'\n\s*(?:\*\*Paper \d+\*\*|## \d+\.|^\d+\.\s+\*\*)', text, flags=re.MULTILINE)
        
        for section in paper_sections[1:max_results+1]:
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        # Extract title - handle multiple formats
        title = ""
        # Format 1: - Title: xxx or Title: xxx
        title_match = re.search(r'[-\*]*\s*Title[:\s]+(.+?)(?:\n|$)', section, re.I)
        if title_match:
            title = title_match.group(1).strip().strip('*')
        # Format 2: First line after ## N. (the title itself)
        if not title:
            first_line = section.strip().split('\n')[0]
            # Remove markdown bold
            title = re.sub(r'\*\*(.+?)\*\*', r'\1', first_line).strip()
        
        if not title or len(title) < 10:
            return None
        
        # Extract authors - handle **Authors:** format
        authors_match = re.search(r'\*?\*?Authors?\*?\*?[:\s]+(.+?)(?:\n|$)', section, re.I)
        authors = []
        if authors_match:
            authors_str = authors_match.group(1).strip()
            authors = [a.strip() for a in re.split(r',|;| et al', authors_str) if a.strip()][:5]
        
        # Extract PMID - handle **PMID:** format
        pmid_match = re.search(r'\*?\*?PMID\*?\*?[:\s]*(\d+)', section, re.I)
        pmid = pmid_match.group(1) if pmid_match else None
        
        # Extract DOI - handle **DOI:** format
        doi_match = re.search(r'\*?\*?DOI\*?\*?[:\s]*(10\.[^\s\n]+)', section, re.I)
        doi = doi_match.group(1) if doi_match else None
        
        # Extract journal - handle **Journal:** format
        journal_match = re.search(r'\*?\*?Journal\*?\*?[:\s]*(.+?)(?:\n|$)', section, re.I)
        journal = journal_match.group(1).strip().strip('*') if journal_match else None
        
        # Extract year - handle **Year:** format or standalone year
        year_match = re.search(r'\*?\*?Year\*?\*?[:\s]*(\d{4})', section, re.I)
        if not year_match:
            year_match = re.search(r'\b(20\d{2})\b', section)
        pub_date = year_match.group(1) if year_match else None
        
        # Extract relevance explanation
        relevance_match = re.search(r'\*?\*?Relevance\*?\*?[:\s]*(.+?)(?:\n\n|##|$)', section, re.I | re.DOTALL)
        relevance_reason = relevance_match.group(1).strip() if relevance_match else ""
        
        # Build URL
        url = ""
        if pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif doi:
            url = f"https://doi.org/{doi}"
        
        return SearchResult(
            title=title,
            authors=authors,
            abstract=abstract,
            source="Perplexity Academic",
            url=url,
            pmid=pmid,
            doi=doi,
            journal=journal,
            pub_date=pub_date,
            relevance_reason=relevance_reason,
        )

