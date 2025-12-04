"""PubMed direct search provider using our existing implementation."""

import os
import json
import re
from typing import Optional

import litellm

from .base import SearchProvider, SearchResult, SearchSession

# Allow dropping unsupported params
litellm.drop_params = True


class PubMedSearchProvider(SearchProvider):
    """Direct PubMed search with LLM-generated queries and relevance scoring."""
    
    name = "pubmed"
    display_name = "PubMed Direct"
    supports_pdf = True  # Uses LLM for query generation, which can use PDF vision
    
    def __init__(self, model: str = "gpt-5-nano"):
        self.model = model
    
    def is_available(self) -> bool:
        """Check if PubMed email is configured."""
        return bool(os.environ.get("PUBMED_EMAIL"))
    
    def search(
        self,
        manuscript_text: str,
        max_results: int = 10,
        focus_pubmed: bool = True,  # Always True for this provider
        pdf_base64: Optional[str] = None,
    ) -> SearchSession:
        """Search PubMed directly using our existing implementation."""
        from src.literature import search_related_literature, filter_and_rank_papers
        from src.prompts import LITERATURE_QUERY_PROMPT
        
        email = os.environ.get("PUBMED_EMAIL")
        if not email:
            return SearchSession(
                provider=self.name,
                query_summary="PUBMED_EMAIL not configured",
                results=[],
                reasoning="Cannot search PubMed without email configured",
            )
        
        search_steps = []
        queries_used = []
        all_papers = {}
        
        try:
            # Step 1: Generate search queries using LLM (with PDF vision if available)
            queries = self._generate_queries(manuscript_text, pdf_base64=pdf_base64)
            queries_used = queries
            
            search_steps.append({
                "type": "query_generation",
                "queries": queries,
            })
            
            # Step 2: Search PubMed with each query
            for query in queries:
                try:
                    papers = search_related_literature(
                        query=query,
                        max_results=max_results,
                        email=email,
                    )
                    search_steps.append({
                        "type": "pubmed_search",
                        "query": query,
                        "results_count": len(papers),
                    })
                    
                    for paper in papers:
                        if paper.pmid and paper.pmid not in all_papers:
                            all_papers[paper.pmid] = paper
                except Exception as e:
                    search_steps.append({
                        "type": "pubmed_search",
                        "query": query,
                        "error": str(e),
                    })
            
            # Step 3: Score and rank papers
            papers_list = list(all_papers.values())
            if papers_list:
                papers_list = filter_and_rank_papers(
                    papers=papers_list,
                    manuscript_summary=manuscript_text[:3000],
                    top_n=max_results,
                    min_score=1.0,
                    model=self.model,
                )
            
            # Convert to SearchResult format
            results = []
            for paper in papers_list:
                results.append(SearchResult(
                    title=paper.title,
                    authors=paper.authors,
                    abstract=paper.abstract,
                    source="PubMed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{paper.pmid}/" if paper.pmid else "",
                    pmid=paper.pmid,
                    doi=paper.doi,
                    journal=paper.journal,
                    pub_date=paper.pub_date,
                    relevance_score=paper.relevance_score,
                    relevance_reason=paper.relevance_reason,
                ))
            
            return SearchSession(
                provider=self.name,
                query_summary=f"Found {len(results)} papers via PubMed API",
                results=results,
                reasoning=f"Generated {len(queries)} search queries, found {len(all_papers)} unique papers, ranked top {len(results)}",
                queries_used=queries_used,
                search_steps=search_steps,
            )
            
        except Exception as e:
            return SearchSession(
                provider=self.name,
                query_summary=f"Error: {str(e)}",
                results=[],
                reasoning=f"Search failed: {str(e)}",
            )
    
    def _generate_queries(self, manuscript_text: str, pdf_base64: Optional[str] = None) -> list[str]:
        """Generate search queries using LLM - text only for reliability."""
        from src.prompts import LITERATURE_QUERY_PROMPT
        
        try:
            # Use text-only for query generation (more reliable)
            prompt_text = f"{LITERATURE_QUERY_PROMPT}\n\n## Manuscript\n{manuscript_text[:8000]}"
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a research assistant helping find related academic papers on PubMed."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            
            print(f"[DEBUG] _generate_queries: Calling {self.model} for query generation")
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=0.3,
            )
            
            content = response.choices[0].message.content
            print(f"[DEBUG] _generate_queries: Got response: {content[:200]}...")
            
            # Parse JSON array
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                queries = json.loads(json_match.group())
                result = [q for q in queries if isinstance(q, str)][:7]
                print(f"[DEBUG] _generate_queries: Parsed {len(result)} queries from JSON")
                return result
            
            # Fallback: parse as lines
            lines = content.strip().split("\n")
            result = [l.strip().strip("-").strip('"').strip("0123456789. ") for l in lines if l.strip() and len(l.strip()) > 10][:5]
            print(f"[DEBUG] _generate_queries: Parsed {len(result)} queries from lines")
            return result
            
        except Exception as e:
            print(f"[ERROR] _generate_queries failed: {e}")
            # Ultimate fallback: extract key terms from title/abstract
            words = manuscript_text[:500].split()
            # Return first ~50 words as a search phrase
            return [" ".join(words[:50])]

