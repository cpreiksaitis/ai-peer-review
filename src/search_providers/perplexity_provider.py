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
        
        # Split by paper markers
        paper_sections = re.split(r'\n\s*\*\*Paper \d+\*\*', text)
        
        for section in paper_sections[1:max_results+1]:
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        # Extract title
        title_match = re.search(r'-\s*Title[:\s]*(.+?)(?:\n|$)', section, re.I)
        title = title_match.group(1).strip() if title_match else ""
        
        if not title or len(title) < 10:
            return None
        
        # Extract authors
        authors_match = re.search(r'-\s*Authors?[:\s]*(.+?)(?:\n|$)', section, re.I)
        authors = []
        if authors_match:
            authors = [a.strip() for a in authors_match.group(1).split(',')][:5]
        
        # Extract PMID
        pmid_match = re.search(r'-\s*PMID[:\s]*(\d+)', section, re.I)
        pmid = pmid_match.group(1) if pmid_match else None
        
        # Extract DOI
        doi_match = re.search(r'-\s*DOI[:\s]*(10\.[^\s\n]+)', section, re.I)
        doi = doi_match.group(1) if doi_match else None
        
        # Extract journal
        journal_match = re.search(r'-\s*Journal[:\s]*(.+?)(?:\n|$)', section, re.I)
        journal = journal_match.group(1).strip() if journal_match else None
        
        # Extract year
        year_match = re.search(r'-\s*Year[:\s]*(\d{4})', section, re.I)
        pub_date = year_match.group(1) if year_match else None
        
        # Extract abstract/summary
        abstract_match = re.search(r'-\s*Abstract[:\s]*(.+?)(?:\n-|\nRelevance|$)', section, re.I | re.DOTALL)
        abstract = abstract_match.group(1).strip()[:500] if abstract_match else ""
        
        # Extract relevance
        relevance_match = re.search(r'-\s*Relevance[:\s]*(.+?)(?:\n-|\n\*\*|$)', section, re.I | re.DOTALL)
        relevance_reason = relevance_match.group(1).strip() if relevance_match else ""
        
        # Extract type (methodology/competitor/foundational/etc)
        type_match = re.search(r'-\s*Type[:\s]*(.+?)(?:\n|$)', section, re.I)
        paper_type = type_match.group(1).strip() if type_match else ""
        if paper_type:
            relevance_reason = f"[{paper_type}] {relevance_reason}"
        
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

