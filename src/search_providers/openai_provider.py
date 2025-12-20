"""OpenAI search provider using web_search tool in Responses API."""

import os
import re
from typing import Optional

from .base import SearchProvider, SearchResult, SearchSession


class OpenAISearchProvider(SearchProvider):
    """Search using OpenAI's Responses API with web_search tool."""
    
    name = "openai"
    display_name = "OpenAI Web Search"
    supports_pdf = True  # OpenAI supports PDF/image vision
    
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(os.environ.get("OPENAI_API_KEY"))
    
    def _build_instruction(self, manuscript_text: str, max_results: int, pdf_base64: Optional[str]) -> str:
        """Build a strict instruction to avoid clarifying questions."""
        return f"""You are a research assistant. Return the {max_results} most relevant papers immediately without asking clarifying questions.

Instructions:
- Use the manuscript context below to infer topic/methods; do not ask the user anything.
- Search PubMed (and Scholar if helpful) and return exactly {max_results} peer-reviewed papers.
- For each paper include: Title; Authors; PMID (if PubMed); DOI; Journal; Year; Why it's relevant (1–2 lines); URL.
- Prefer similar methodology, same research question, recent work. If unsure, still pick the best matches.
- Output only the papers; no preamble or follow-up questions.
- If context is limited, make reasonable assumptions and still return papers.

Context:
{"[See attached PDF]" if pdf_base64 else "[Manuscript below]"}
{'' if pdf_base64 else manuscript_text[:10000]}"""

    def _run_search(self, instruction: str, max_results: int, focus_pubmed: bool) -> tuple[SearchSession, list]:
        """Perform a single search call."""
        try:
            tools = [
                {
                    "type": "web_search",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "medium", # User snippet used medium
                }
            ]
            if focus_pubmed:
                tools[0]["filters"] = {
                    "allowed_domains": [
                        "pubmed.ncbi.nlm.nih.gov",
                        "ncbi.nlm.nih.gov",
                        "scholar.google.com",
                        "pubmed.gov", 
                    ]
                }
            
            # Use explicit input text object structure to ensure compatibility
            content = [{"type": "input_text", "text": instruction}]
            
            print(f"[DEBUG] OpenAI Search: Starting request with model={self.model}")
            
            # Check if responses API is available (defensive check)
            if not hasattr(self.client, "responses"):
                raise AttributeError("Client has no 'responses' attribute")

            response = self.client.responses.create(
                model="gpt-5-nano", # Forced as per user request
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "text"
                    },
                    "verbosity": "medium"
                },
                reasoning={
                    "effort": "medium",
                    "summary": "auto"
                },
                tools=tools,
                store=True,
                include=["reasoning.encrypted_content", "web_search_call.action.sources"],
            )
            session = self._parse_response(response, max_results)
            return session, response
            
        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"[ERROR] OpenAI Search Failed:\n{trace}")
            raise e

    def stream_search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ):
        """
        Stream search results using OpenAI's Responses API.
        Yields events:
        - {"type": "delta", "content": "..."}
        - {"type": "result", "papers": [...], "query_summary": "..."}
        """
        import json
        
        instruction = self._build_instruction(manuscript_text, max_results, pdf_base64)
        
        tools = [
            {
                "type": "web_search",
                "user_location": {"type": "approximate"},
                "search_context_size": "medium",
            }
        ]
        if focus_pubmed:
            tools[0]["filters"] = {
                "allowed_domains": [
                    "pubmed.ncbi.nlm.nih.gov",
                    "ncbi.nlm.nih.gov",
                    "scholar.google.com",
                    "pubmed.gov", 
                ]
            }
        
        content = [{"type": "input_text", "text": instruction}]
        
        try:
            # Check for responses attribute
            if not hasattr(self.client, "responses"):
                raise AttributeError("Client has no 'responses' attribute")

            stream = self.client.responses.create(
                model="gpt-5-nano",
                input=[{"role": "user", "content": content}],
                text={"format": {"type": "text"}, "verbosity": "medium"},
                reasoning={"effort": "medium", "summary": "auto"},
                tools=tools,
                store=True,
                include=["reasoning.encrypted_content", "web_search_call.action.sources"],
                stream=True,
            )
            
            accumulated_text = ""
            
            for chunk in stream:
                # Handle text deltas
                delta = ""
                if hasattr(chunk, "output_text") and hasattr(chunk.output_text, "delta"):
                    delta = chunk.output_text.delta
                elif hasattr(chunk, "delta"): # Fallback for some response shapes
                    delta = chunk.delta
                
                if delta:
                    accumulated_text += delta
                    # Yield delta event
                    yield f"data: {json.dumps({'type': 'delta', 'content': delta})}\n\n"
            
            # Streaming finished, parse results from accumulated text
            # We can use the existing parsing logic if we create a pseudo-response object
            # or just call _extract_papers_from_text directly which is easier.
            results = self._extract_papers_from_text(accumulated_text, max_results)
            
            # Serialize results
            final_data = {
                "type": "result",
                "papers": [
                    {
                        "title": r.title,
                        "authors": r.authors,
                        "journal": r.journal,
                        "pub_date": r.pub_date,
                        "url": r.url,
                        "abstract": r.abstract,
                        "pmid": r.pmid,
                        "relevance_reason": r.relevance_reason,
                        "relevance_score": r.relevance_score
                    } for r in results
                ],
                "query_summary": f"Found {len(results)} papers via OpenAI stream",
                "reasoning": accumulated_text # Return full reasoning text too
            }
            
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Stream Search Failed:\n{traceback.format_exc()}")
            # Yield error event
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """Search for similar papers using OpenAI's web search using Responses API."""
        instruction = self._build_instruction(manuscript_text, max_results, pdf_base64)
        try:
            session, _ = self._run_search(instruction, max_results, focus_pubmed)
            return session
        except Exception as e:
            return SearchSession(
                provider=self.name,
                query_summary=f"Error: {str(e)}",
                results=[],
                reasoning=f"Search failed: {str(e)}",
            )
    
    def _parse_response(self, response, max_results: int) -> SearchSession:
        """Parse OpenAI response into SearchSession."""
        results = []
        reasoning = ""
        queries_used = []
        search_steps = []
        sources = []
        
        # Extract text content and search information
        if response.output:
            for output in response.output:
                if output.type == "message":
                    if output.content:
                        for content in output.content:
                            if hasattr(content, "text"):
                                reasoning = content.text
                            elif content.type == "output_text":
                                reasoning = content.text
                elif output.type == "web_search_call":
                    if hasattr(output, "action") and output.action:
                        action = output.action
                        if hasattr(action, "query") and action.query:
                            queries_used.append(action.query)
                        if hasattr(action, "sources") and action.sources:
                            for source in action.sources:
                                source_info = {
                                    "type": "source",
                                    "url": getattr(source, "url", "") or "",
                                    "title": getattr(source, "title", "") or "",
                                }
                                search_steps.append(source_info)
                                sources.append(source_info)
        
        # Parse papers from the text response
        results = self._extract_papers_from_text(reasoning, max_results)
        
        # If we didn't parse papers from text, create results from sources (even non-PubMed)
        if not results and sources:
            for source in sources[:max_results]:
                pmid = None
                if "pubmed" in source["url"].lower():
                    pmid_match = re.search(r'/(\d+)', source["url"])
                    pmid = pmid_match.group(1) if pmid_match else None
                results.append(SearchResult(
                    title=source["title"] or (f"PubMed Article {pmid}" if pmid else source["url"] or "Unknown title"),
                    authors=[],
                    abstract="",
                    source="PubMed" if pmid else "Web",
                    url=source["url"],
                    pmid=pmid,
                ))
        
        # Calculate costs
        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = getattr(response.usage, "total_tokens", 0) or 0
        
        # Estimate cost (GPT-5-mini pricing)
        total_cost = tokens_used * 0.0000006  # Approximate
        
        return SearchSession(
            provider=self.name,
            query_summary=f"Found {len(results)} papers via OpenAI web search",
            results=results,
            reasoning=reasoning,
            queries_used=queries_used,
            total_cost=total_cost,
            tokens_used=tokens_used,
            search_steps=search_steps,
        )
    
    def _extract_papers_from_text(self, text: str, max_results: int) -> list[SearchResult]:
        """Extract paper information from LLM response text."""
        results = []
        
        # Try to find paper entries in the text
        # Look for patterns like numbered lists, titles in bold/quotes, etc.
        
        # Pattern for PubMed URLs
        pmid_pattern = r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)'
        pmids_found = re.findall(pmid_pattern, text)
        
        # Split text into sections that might be individual papers
        # Look for numbered items or paper titles
        sections = re.split(r'\n\s*\d+[\.\)]\s+|\n\s*[-•]\s+\*\*', text)
        
        for section in sections[1:max_results+1]:  # Skip first split (before first paper)
            if len(section) < 50:  # Skip very short sections
                continue
                
            result = self._parse_paper_section(section)
            if result:
                results.append(result)
        
        # If we found PMIDs but couldn't parse papers, create basic entries
        if not results and pmids_found:
            for pmid in pmids_found[:max_results]:
                results.append(SearchResult(
                    title=f"PubMed Article {pmid}",
                    authors=[],
                    abstract="",
                    source="PubMed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    pmid=pmid,
                ))
        
        return results
    
    def _parse_paper_section(self, section: str) -> Optional[SearchResult]:
        """Parse a text section into a SearchResult."""
        # Extract title (usually first line or in quotes/bold)
        title_match = re.search(r'\*\*([^*]+)\*\*|"([^"]+)"|^([^\n]+)', section)
        title = ""
        if title_match:
            title = title_match.group(1) or title_match.group(2) or title_match.group(3)
            title = title.strip()
        
        if not title or len(title) < 10:
            return None
        
        # Extract PMID
        pmid_match = re.search(r'PMID[:\s]*(\d+)|pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', section, re.I)
        pmid = pmid_match.group(1) or pmid_match.group(2) if pmid_match else None
        
        # Extract DOI
        doi_match = re.search(r'DOI[:\s]*(10\.[^\s]+)|doi\.org/(10\.[^\s]+)', section, re.I)
        doi = doi_match.group(1) or doi_match.group(2) if doi_match else None
        
        # Extract authors
        authors_match = re.search(r'Authors?[:\s]*([^\n]+)', section, re.I)
        authors = []
        if authors_match:
            authors = [a.strip() for a in authors_match.group(1).split(',')][:5]
        
        # Extract abstract
        abstract_match = re.search(r'Abstract[:\s]*([^\n]+(?:\n[^\n]+)*)', section, re.I)
        abstract = abstract_match.group(1).strip()[:500] if abstract_match else ""
        
        # Extract journal
        journal_match = re.search(r'Journal[:\s]*([^\n]+)', section, re.I)
        journal = journal_match.group(1).strip() if journal_match else None
        
        # Build URL
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        if not url and doi:
            url = f"https://doi.org/{doi}"
        
        return SearchResult(
            title=title,
            authors=authors,
            abstract=abstract,
            source="PubMed" if pmid else "Web",
            url=url,
            pmid=pmid,
            doi=doi,
            journal=journal,
        )
