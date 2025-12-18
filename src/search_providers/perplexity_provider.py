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
        """Lazy load Perplexity client using official SDK."""
        if self._client is None:
            from perplexity import Perplexity
            api_key = os.environ.get("PERPLEXITY_API_KEY")
            if not api_key:
                raise ValueError("PERPLEXITY_API_KEY not set")
            self._client = Perplexity(api_key=api_key)
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
        
        # Direct, structured prompt to get full details (not just links)
        search_instruction = f"""Search PubMed and academic databases to find {max_results} peer-reviewed papers most relevant to this research manuscript.

Return a numbered list of exactly {max_results} papers.
Format each paper exactly as follows, using markdown headers for the title:

## [Insert Title Here]
**Authors:** [First author et al.]
**Journal:** [Journal Name] ([Year])
**PMID:** [PMID if available]
**DOI:** [DOI if available]
**URL:** [URL]
**Relevance:** [2-3 sentences explaining why this paper is relevant]

Prioritize similar methodology, same research question, recent work, and foundational papers. Do not ask clarifying questions.

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
            
            # Use Perplexity chat completion with academic search
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                search_mode="academic",
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
        try:
            text = response.choices[0].message.content
            
            # Extract citations if available
            citations = []
            if hasattr(response, "citations") and response.citations:
                citations = response.citations
            
            search_steps = [{"type": "source", "url": c} for c in citations]
            
            # Parse papers from text
            results = self._extract_papers_from_text(text, max_results)
            
            # If none parsed, create fallbacks from citations/sources
            if not results and search_steps:
                for step in search_steps[:max_results]:
                    if step.get("type") == "source":
                        results.append(SearchResult(
                            title=step.get("title") or step.get("url") or "Unknown title",
                            authors=[],
                            abstract="",
                            source="Web",
                            url=step.get("url") or "",
                        ))
            
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
        except Exception as e:
             return SearchSession(
                provider=self.name,
                query_summary=f"Error parsing response: {str(e)}",
                results=[],
                reasoning=str(response.choices[0].message.content) if response and response.choices else "No content",
            )
    
    def _extract_papers_from_text(self, text: str, max_results: int) -> list[SearchResult]:
        """Extract paper information from response text."""
        results = []
        
        # Split by paper markers - robust regex for various numbered list formats
        # Matches: "## 1. Title", "## Title", "1. **Title**", "**1. Title**"
        # We split by the start of a new numbered item or header that looks like a paper start
        
        # Regex explanation:
        # \n\s*                 # Start with newline
        # (?:                   # Non-capturing group for alternatives
        #   ##\s*\d+\.?\s+      # ## 1. 
        #   | ##\s+             # ##
        #   | \d+\.\s+\*\*      # 1. **
        #   | \*\*\d+\.\s+      # **1. 
        # )
        paper_sections = re.split(r'\n\s*(?:##\s*\d+\.?\s+|##\s+|\d+\.\s+\*\*|\*\*\d+\.\s+)', text)
        
        # Skip the first section (preamble)
        for section in paper_sections[1:max_results+1]:
            if len(section.strip()) < 50:
                continue
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        lines = section.strip().split('\n')
        
        # 1. Title extraction
        title = ""
        # Try to find a line that looks like a title (bold) or just take the first line
        # Clean up markdown formatting from the first line
        first_line_clean = re.sub(r'\*\*|##|\[|\]', '', lines[0]).strip()
        if first_line_clean:
            title = first_line_clean.split('**')[0].strip() # Handle trailing bold
        
        if not title:
            return None

        # 2. Extract fields using regex
        def extract_field(pattern):
            match = re.search(pattern, section, re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else ""

        authors = extract_field(r'Authors?:?\*?\*?:? (.+)')
        journal = extract_field(r'Journal:?\*?\*?:? (.+)')
        pmid = extract_field(r'PMID:?\*?\*?:? (\d+)')
        doi = extract_field(r'DOI:?\*?\*?:? (10\..+)')
        url = extract_field(r'URL:?\*?\*?:? (https?://.+)')
        
        # 3. Relevance/Abstract
        relevance = extract_field(r'Relevance:?\*?\*?:? (.+)')
        if not relevance:
             # Try to find a block of text at the end
             relevance = lines[-1] if len(lines) > 1 else ""

        # Clean up authors list
        author_list = []
        if authors:
            author_list = [a.strip() for a in re.split(r',|;', authors)]

        # Construct URL if missing
        if not url:
            if pmid:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            elif doi:
                url = f"https://doi.org/{doi}"

        return SearchResult(
            title=title,
            authors=author_list,
            abstract=relevance, # Use relevance as abstract context
            source="Perplexity",
            url=url,
            pmid=pmid if pmid else None,
            doi=doi if doi else None,
            journal=journal,
            relevance_reason=relevance
        )
