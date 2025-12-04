"""Literature search evaluation using LLM-as-judge."""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import litellm

from .criteria import LITERATURE_CRITERIA, calculate_weighted_score
from .judge_prompts import (
    LITERATURE_JUDGE_SYSTEM,
    MANUSCRIPT_SUMMARY_PROMPT,
    build_literature_judge_prompt,
)

# Allow dropping unsupported params
litellm.drop_params = True


@dataclass
class LiteratureEvalResult:
    """Result of evaluating a literature search."""
    
    provider: str
    scores: dict[str, float] = field(default_factory=dict)
    reasoning: dict[str, str] = field(default_factory=dict)
    overall_assessment: str = ""
    missing_elements: list[str] = field(default_factory=list)
    strongest_papers: list[int] = field(default_factory=list)
    weighted_score: float = 0.0
    papers_count: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "scores": self.scores,
            "reasoning": self.reasoning,
            "overall_assessment": self.overall_assessment,
            "missing_elements": self.missing_elements,
            "strongest_papers": self.strongest_papers,
            "weighted_score": self.weighted_score,
            "papers_count": self.papers_count,
            "error": self.error,
        }
    
    def format_report(self) -> str:
        """Format as human-readable report."""
        if self.error:
            return f"## {self.provider}\n\n**Error:** {self.error}"
        
        lines = [
            f"## {self.provider}",
            f"**Papers Found:** {self.papers_count}",
            f"**Weighted Score:** {self.weighted_score:.1f}/10",
            "",
            "### Scores",
        ]
        
        for key, score in self.scores.items():
            crit = LITERATURE_CRITERIA.get(key, {})
            name = crit.get("name", key)
            reason = self.reasoning.get(key, "")
            lines.append(f"- **{name}:** {score}/10 - {reason}")
        
        lines.extend([
            "",
            "### Overall Assessment",
            self.overall_assessment,
        ])
        
        if self.missing_elements:
            lines.extend([
                "",
                "### Missing Elements",
                *[f"- {elem}" for elem in self.missing_elements],
            ])
        
        if self.strongest_papers:
            lines.extend([
                "",
                f"### Strongest Papers (indices): {self.strongest_papers}",
            ])
        
        return "\n".join(lines)


class LiteratureEvaluator:
    """Evaluates literature search results using LLM-as-judge."""
    
    def __init__(self, model: str = "gpt-5-nano"):
        self.model = model
    
    def generate_manuscript_summary(self, manuscript_text: str) -> str:
        """Generate a concise summary of the manuscript for evaluation."""
        prompt = MANUSCRIPT_SUMMARY_PROMPT.format(
            manuscript_text=manuscript_text[:10000]
        )
        
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        return response.choices[0].message.content
    
    def evaluate(
        self,
        provider_name: str,
        papers: list[dict],
        manuscript_summary: str,
    ) -> LiteratureEvalResult:
        """
        Evaluate a set of literature search results.
        
        Args:
            provider_name: Name of the search provider
            papers: List of paper dictionaries with title, authors, abstract, etc.
            manuscript_summary: Summary of the manuscript being reviewed
            
        Returns:
            LiteratureEvalResult with scores and reasoning
        """
        if not papers:
            return LiteratureEvalResult(
                provider=provider_name,
                error="No papers to evaluate",
                papers_count=0,
            )
        
        # Build evaluation prompt
        prompt = build_literature_judge_prompt(
            manuscript_summary=manuscript_summary,
            papers=papers,
            criteria=LITERATURE_CRITERIA,
        )
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": LITERATURE_JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return LiteratureEvalResult(
                    provider=provider_name,
                    error=f"Could not parse JSON response: {content[:200]}",
                    papers_count=len(papers),
                )
            
            data = json.loads(json_match.group())
            
            scores = data.get("scores", {})
            weighted = calculate_weighted_score(scores, LITERATURE_CRITERIA)
            
            return LiteratureEvalResult(
                provider=provider_name,
                scores=scores,
                reasoning=data.get("reasoning", {}),
                overall_assessment=data.get("overall_assessment", ""),
                missing_elements=data.get("missing_elements", []),
                strongest_papers=data.get("strongest_papers", []),
                weighted_score=weighted,
                papers_count=len(papers),
            )
            
        except json.JSONDecodeError as e:
            return LiteratureEvalResult(
                provider=provider_name,
                error=f"JSON parsing error: {e}",
                papers_count=len(papers),
            )
        except Exception as e:
            return LiteratureEvalResult(
                provider=provider_name,
                error=f"Evaluation error: {e}",
                papers_count=len(papers),
            )
    
    def evaluate_multiple(
        self,
        results: dict[str, list[dict]],
        manuscript_summary: str,
    ) -> dict[str, LiteratureEvalResult]:
        """
        Evaluate results from multiple providers.
        
        Args:
            results: Dict mapping provider name to list of papers
            manuscript_summary: Summary of the manuscript
            
        Returns:
            Dict mapping provider name to evaluation result
        """
        evaluations = {}
        
        for provider_name, papers in results.items():
            evaluations[provider_name] = self.evaluate(
                provider_name=provider_name,
                papers=papers,
                manuscript_summary=manuscript_summary,
            )
        
        return evaluations
    
    def rank_providers(
        self,
        evaluations: dict[str, LiteratureEvalResult],
    ) -> list[tuple[str, float]]:
        """
        Rank providers by weighted score.
        
        Args:
            evaluations: Dict of evaluation results
            
        Returns:
            List of (provider_name, score) tuples, sorted by score descending
        """
        rankings = []
        
        for provider, result in evaluations.items():
            if result.error:
                score = 0.0
            else:
                score = result.weighted_score
            rankings.append((provider, score))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

