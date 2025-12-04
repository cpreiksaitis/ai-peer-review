"""Gemini search provider using Google Search grounding."""

import os
import re
from typing import Optional

from .base import SearchProvider, SearchResult, SearchSession


class GeminiSearchProvider(SearchProvider):
    """Search using Gemini with Google Search grounding."""
    
    name = "gemini"
    display_name = "Gemini Grounded Search"
    supports_pdf = True  # Gemini supports PDF/document vision
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy load Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            self._client = genai
        return self._client
    
    def is_available(self) -> bool:
        """Check if Google/Gemini API key is configured."""
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    
    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """Search for similar papers using Gemini with grounding."""
        
        pubmed_focus = ""
        if focus_pubmed:
            pubmed_focus = """
IMPORTANT: Focus your search on PubMed and academic sources. 
Use search queries like "site:pubmed.ncbi.nlm.nih.gov [topic]" to find papers.
Prioritize peer-reviewed medical/scientific literature."""

        search_instruction = f"""Find {max_results} academic papers that are most similar or relevant to this manuscript.
{pubmed_focus}

For each paper found, provide in a structured format:
- **Title**: [full title]
- **Authors**: [author list]
- **PMID**: [PubMed ID if available]
- **DOI**: [DOI if available]
- **Journal**: [journal name]
- **Year**: [publication year]
- **Abstract**: [brief abstract or summary]
- **Relevance**: [why this paper is relevant]
- **URL**: [link to paper]

Search for papers that:
1. Address the same or similar research question
2. Use similar methodology
3. Study similar populations or settings
4. Are frequently cited in this field

Manuscript to analyze:
---
{manuscript_text[:10000]}
---

Use Google Search to find the most relevant academic papers. Return exactly {max_results} papers."""

        try:
            # Build content parts - use PDF if available
            import base64
            content_parts = []
            
            if pdf_base64:
                # Add PDF as inline data
                pdf_bytes = base64.b64decode(pdf_base64)
                content_parts.append({
                    "inline_data": {
                        "mime_type": "application/pdf",
                        "data": pdf_base64,
                    }
                })
            
            content_parts.append(search_instruction)
            
            # Try with Google Search grounding
            try:
                from google.generativeai.types import Tool
                
                # Create grounding tool
                google_search_tool = Tool(
                    google_search=self.client.protos.GoogleSearch()
                )
                
                model = self.client.GenerativeModel(
                    self.model,
                    tools=[google_search_tool],
                )
                
                response = model.generate_content(
                    content_parts,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 4096,
                    },
                )
            except Exception:
                # Fall back to regular generation without grounding
                model = self.client.GenerativeModel(self.model)
                
                response = model.generate_content(
                    content_parts,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": 4096,
                    },
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
        """Parse Gemini response into SearchSession."""
        text = ""
        search_steps = []
        
        # Extract text and grounding info
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content.parts:
                text = candidate.content.parts[0].text
            
            # Extract grounding metadata if available
            if hasattr(candidate, "grounding_metadata"):
                gm = candidate.grounding_metadata
                if hasattr(gm, "search_entry_point"):
                    search_steps.append({
                        "type": "search_query",
                        "query": gm.search_entry_point.rendered_content if hasattr(gm.search_entry_point, "rendered_content") else "",
                    })
                if hasattr(gm, "grounding_chunks"):
                    for chunk in gm.grounding_chunks:
                        if hasattr(chunk, "web"):
                            search_steps.append({
                                "type": "source",
                                "url": chunk.web.uri if hasattr(chunk.web, "uri") else "",
                                "title": chunk.web.title if hasattr(chunk.web, "title") else "",
                            })
        
        # Parse papers from text
        results = self._extract_papers_from_text(text, max_results)
        
        # Estimate tokens and cost
        tokens_used = len(text.split()) * 1.3  # Rough estimate
        total_cost = tokens_used * 0.0000001  # Gemini Flash pricing
        
        return SearchSession(
            provider=self.name,
            query_summary=f"Found {len(results)} papers via Gemini grounded search",
            results=results,
            reasoning=text,
            queries_used=[],
            total_cost=total_cost,
            tokens_used=int(tokens_used),
            search_steps=search_steps,
        )
    
    def _extract_papers_from_text(self, text: str, max_results: int) -> list[SearchResult]:
        """Extract paper information from response text."""
        results = []
        
        # Try multiple split patterns
        patterns = [
            r'\n\s*\*\*Paper \d+[:\*]*\*\*',  # **Paper 1**
            r'\n\s*(?=\*\*Title\*\*:)',  # **Title**:
            r'\n\s*\d+[\.\)]\s+\*\*',  # 1. ** or 1) **
            r'\n\s*(?=- \*\*Title\*\*:)',  # - **Title**:
        ]
        
        paper_sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in paper_sections:
                split = re.split(pattern, section)
                if len(split) > 1:
                    new_sections.extend(split)
                else:
                    new_sections.append(section)
            paper_sections = new_sections
        
        for section in paper_sections[:max_results + 5]:  # Try a few extra
            if len(section) < 50:
                continue
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
            if len(results) >= max_results:
                break
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        # Extract fields using patterns
        title_match = re.search(r'\*\*Title\*\*:\s*(.+?)(?:\n|$)', section, re.I)
        title = title_match.group(1).strip() if title_match else ""
        
        if not title or len(title) < 10:
            return None
        
        authors_match = re.search(r'\*\*Authors?\*\*:\s*(.+?)(?:\n|$)', section, re.I)
        authors = []
        if authors_match:
            authors = [a.strip() for a in authors_match.group(1).split(',')][:5]
        
        pmid_match = re.search(r'\*\*PMID\*\*:\s*(\d+)', section, re.I)
        pmid = pmid_match.group(1) if pmid_match else None
        
        doi_match = re.search(r'\*\*DOI\*\*:\s*(10\.[^\s\n]+)', section, re.I)
        doi = doi_match.group(1) if doi_match else None
        
        journal_match = re.search(r'\*\*Journal\*\*:\s*(.+?)(?:\n|$)', section, re.I)
        journal = journal_match.group(1).strip() if journal_match else None
        
        year_match = re.search(r'\*\*Year\*\*:\s*(\d{4})', section, re.I)
        pub_date = year_match.group(1) if year_match else None
        
        abstract_match = re.search(r'\*\*Abstract\*\*:\s*(.+?)(?:\n\*\*|$)', section, re.I | re.DOTALL)
        abstract = abstract_match.group(1).strip()[:500] if abstract_match else ""
        
        relevance_match = re.search(r'\*\*Relevance\*\*:\s*(.+?)(?:\n\*\*|$)', section, re.I | re.DOTALL)
        relevance_reason = relevance_match.group(1).strip() if relevance_match else ""
        
        url_match = re.search(r'\*\*URL\*\*:\s*(https?://[^\s\n]+)', section, re.I)
        url = url_match.group(1) if url_match else ""
        
        if not url and pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        elif not url and doi:
            url = f"https://doi.org/{doi}"
        
        return SearchResult(
            title=title,
            authors=authors,
            abstract=abstract,
            source="PubMed" if pmid else "Google Scholar",
            url=url,
            pmid=pmid,
            doi=doi,
            journal=journal,
            pub_date=pub_date,
            relevance_reason=relevance_reason,
        )

