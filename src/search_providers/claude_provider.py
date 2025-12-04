"""Claude search provider using Anthropic's web search tool."""

import os
import re
from typing import Optional

from .base import SearchProvider, SearchResult, SearchSession


class ClaudeSearchProvider(SearchProvider):
    """Search using Claude with web search tool."""
    
    name = "claude"
    display_name = "Claude Web Search"
    supports_pdf = True  # Claude supports PDF vision
    
    def __init__(self, model: str = "claude-3-haiku-20240307"):
        # Use claude-3-haiku as default for wider compatibility
        # Other options: claude-3-5-sonnet-20241022, claude-3-opus-20240229
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client
    
    def is_available(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    
    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """Search for similar papers using Claude with web search."""
        
        pubmed_focus = ""
        if focus_pubmed:
            pubmed_focus = """
IMPORTANT: Focus your searches on PubMed (pubmed.ncbi.nlm.nih.gov) and academic databases.
Use specific medical/scientific terminology in your searches.
Prioritize peer-reviewed literature from reputable journals."""

        search_instruction = f"""Find {max_results} academic papers that are most similar or relevant to this manuscript.
{pubmed_focus}

Use web search to find papers on PubMed. For each paper, provide in this format:

**Paper 1**
- Title: [full title]
- Authors: [author list]
- PMID: [PubMed ID]
- DOI: [DOI if available]
- Journal: [journal name]
- Year: [publication year]
- Abstract: [brief abstract or summary]
- Relevance: [why this paper is relevant]

Search strategy:
1. First, identify the main topics, methods, and populations in the manuscript
2. Search for papers with similar research questions
3. Search for papers using similar methodologies
4. Search for papers studying similar populations

Manuscript to analyze:
---
{manuscript_text[:10000]}
---

Return exactly {max_results} papers with their PubMed links."""

        try:
            # Build message content - use PDF vision if available
            if pdf_base64:
                user_content = [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    },
                    {"type": "text", "text": search_instruction},
                ]
            else:
                user_content = search_instruction
            
            # Try the web search tool first
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    tools=[
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 10,
                        }
                    ],
                    messages=[
                        {"role": "user", "content": user_content}
                    ],
                )
                return self._parse_response(response, max_results)
            except Exception as web_search_error:
                # Fall back to using Claude to generate search recommendations
                # without actual web search capability
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[
                        {
                            "role": "user", 
                            "content": f"""Based on this manuscript, suggest {max_results} specific academic papers 
that would likely be found on PubMed that are relevant to this research.

For each paper, provide realistic details in this format:
**Paper 1**
- Title: [suggested title based on topic]
- Authors: [leave as "Various authors"]
- PMID: [leave blank]
- Journal: [relevant journal name]
- Year: [reasonable year]
- Relevance: [why this type of paper would be relevant]

Note: These are suggested searches, not actual papers. The user should verify on PubMed.

Manuscript:
---
{manuscript_text[:8000]}
---"""
                        }
                    ],
                )
                
                session = self._parse_response(response, max_results)
                session.reasoning = f"Note: Web search unavailable ({str(web_search_error)[:50]}...). Showing search suggestions.\n\n" + session.reasoning
                return session
            
        except Exception as e:
            return SearchSession(
                provider=self.name,
                query_summary=f"Error: {str(e)}",
                results=[],
                reasoning=f"Search failed: {str(e)}",
            )
    
    def _parse_response(self, response, max_results: int) -> SearchSession:
        """Parse Claude response into SearchSession."""
        text = ""
        search_steps = []
        queries_used = []
        
        # Process content blocks
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use" and block.name == "web_search":
                query = block.input.get("query", "") if hasattr(block, "input") else ""
                queries_used.append(query)
                search_steps.append({
                    "type": "search_query",
                    "query": query,
                    "tool_use_id": block.id,
                })
            elif block.type == "tool_result":
                # Tool results contain search results
                if hasattr(block, "content"):
                    search_steps.append({
                        "type": "search_result",
                        "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else "",
                    })
        
        # Parse papers from text
        results = self._extract_papers_from_text(text, max_results)
        
        # Calculate costs
        tokens_used = 0
        if hasattr(response, "usage"):
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        # Claude Sonnet pricing
        total_cost = tokens_used * 0.000003
        
        return SearchSession(
            provider=self.name,
            query_summary=f"Found {len(results)} papers via Claude web search",
            results=results,
            reasoning=text,
            queries_used=queries_used,
            total_cost=total_cost,
            tokens_used=tokens_used,
            search_steps=search_steps,
        )
    
    def _extract_papers_from_text(self, text: str, max_results: int) -> list[SearchResult]:
        """Extract paper information from response text."""
        results = []
        
        # Try multiple split patterns
        patterns = [
            r'\n\s*\*\*Paper \d+[:\*]*\*\*',  # **Paper 1** or **Paper 1:**
            r'\n\s*\d+[\.\)]\s+(?=\*\*|Title)',  # 1. ** or 1) Title
            r'\n\s*##\s+',  # ## heading
            r'\n\s*(?=- Title:)',  # - Title:
        ]
        
        paper_sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in paper_sections:
                split = re.split(pattern, section)
                if len(split) > 1:
                    new_sections.extend(split[1:])  # Skip first (pre-papers content)
                else:
                    new_sections.append(section)
            if len(new_sections) > len(paper_sections):
                paper_sections = new_sections
        
        for section in paper_sections[:max_results + 5]:
            if len(section) < 30:
                continue
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
            if len(results) >= max_results:
                break
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        # Extract title - try multiple patterns
        title = ""
        
        # Pattern 1: - Title: [title]
        title_match = re.search(r'-\s*Title[:\s]+(.+?)(?:\n|$)', section, re.I)
        if title_match:
            title = title_match.group(1).strip()
        
        # Pattern 2: **Title**: [title]
        if not title:
            title_match = re.search(r'\*\*Title\*\*[:\s]*(.+?)(?:\n|$)', section, re.I)
            if title_match:
                title = title_match.group(1).strip()
        
        # Pattern 3: First bold text
        if not title:
            title_match = re.search(r'\*\*([^*]+)\*\*', section)
            if title_match:
                title = title_match.group(1).strip()
        
        # Pattern 4: First line
        if not title:
            lines = section.strip().split('\n')
            for line in lines:
                line = line.strip().lstrip('-').strip()
                if line and len(line) > 15:
                    title = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                    break
        
        if not title or len(title) < 10:
            return None
        
        # Extract PMID
        pmid_match = re.search(r'PMID[:\s]*(\d+)|pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', section, re.I)
        pmid = None
        if pmid_match:
            pmid = pmid_match.group(1) or pmid_match.group(2)
        
        # Extract DOI
        doi_match = re.search(r'DOI[:\s]*(10\.[^\s\n]+)', section, re.I)
        doi = doi_match.group(1) if doi_match else None
        
        # Extract authors - try multiple patterns
        authors = []
        authors_match = re.search(r'-?\s*Authors?[:\s]*(.+?)(?:\n|$)', section, re.I)
        if authors_match:
            authors_str = authors_match.group(1).strip()
            if authors_str and authors_str.lower() != "various authors":
                authors = [a.strip() for a in authors_str.split(',')][:5]
        
        # Extract journal
        journal = None
        journal_match = re.search(r'-?\s*Journal[:\s]*(.+?)(?:\n|$)', section, re.I)
        if journal_match:
            journal = journal_match.group(1).strip()
        
        # Extract year
        year_match = re.search(r'-?\s*Year[:\s]*(\d{4})', section, re.I)
        if not year_match:
            year_match = re.search(r'\b(20\d{2})\b', section)
        pub_date = year_match.group(1) if year_match else None
        
        # Extract abstract
        abstract = ""
        abstract_match = re.search(r'-?\s*Abstract[:\s]*(.+?)(?:\n-|\nRelevance|$)', section, re.I | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:500]
        
        # Extract relevance
        relevance_reason = ""
        relevance_match = re.search(r'-?\s*Relevance[:\s]*(.+?)(?:\n\n|\n\*\*|$)', section, re.I | re.DOTALL)
        if relevance_match:
            relevance_reason = relevance_match.group(1).strip()
        
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
            source="PubMed" if pmid else "Suggested",
            url=url,
            pmid=pmid,
            doi=doi,
            journal=journal,
            pub_date=pub_date,
            relevance_reason=relevance_reason,
        )

